//! Recovery through the console: the write log's copy of each committed
//! document, `import`, point-in-time recovery and the reseed.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use super::*;
use crate::lb::MetricsClient;
use crate::routes::short_digest;
use crate::testkit::{V0, log_and_ship, log_records, quiet_fleet};
use crate::writelog::{self, digest_of};

const D3: &str = "3333333333333333333333333333333333333333333333333333333333333333";

/// A document's exact text, as the store serialises one.
fn doc_text(version: u64, budget: u64) -> String {
    json!({"schema": 1, "version": version, "updated_at": "2026-10-11T06:00:00.000Z",
           "clients": [{"name": "acme", "daily_budget": budget, "utilization_limit": null,
                        "models": [], "preferred_endpoints": [], "keys": []}],
           "revoked_digests": []})
    .to_string()
}

/// The JSON lines the binary's own subscriber writes while `f` runs.
async fn captured<F: std::future::Future>(f: F) -> (F::Output, String) {
    let buf = testlog::LogBuf::default();
    let out = {
        let filter = tracing_subscriber::EnvFilter::new("ops_console=info");
        let _guard = testlog::scoped(log_subscriber(filter, buf.clone()));
        f.await
    };
    (out, buf.text())
}

/// A client for `metrics`' routes, as the console builds one.
async fn metrics_client(metrics: &FakeMetrics) -> MetricsClient {
    MetricsClient::new(fake_upstream(metrics.router()).await.parse().unwrap())
}

/// A (version, digest) pair instances installed in the last 30 days.
fn installed(version: u64, digest: &str) -> serde_json::Value {
    series(
        &[("version", &version.to_string()), ("digest", digest)],
        "1",
    )
}

/// A document the console committed, as its label reads back.
fn committed_label(version: u64, digest: &str) -> serde_json::Value {
    series(
        &[("version", &version.to_string()), ("digest", digest)],
        &(V0 / 1000).to_string(),
    )
}

/// The store at `world`, its GET body edited by `edit`.
fn store_with(world: &World, edit: impl FnOnce(&mut serde_json::Value)) -> FakeStore {
    let store = FakeStore::new(world);
    let mut body = world.get_body();
    edit(&mut body);
    *store.gets.lock().unwrap() = vec![body];
    store
}

// ─── The write log ──────────────────────────────────────────────────────────

/// A commit's document goes on the write log whole, in lines that each name
/// its version and digest, and every write-log line stays under the
/// container runtime's 16 KiB line limit, whatever the request and answer
/// hold. The lines read back, through the log store's shape, to the exact
/// bytes.
#[tokio::test]
async fn every_committed_document_is_logged_whole_in_lines_under_16_kib() {
    let models: Vec<String> = (0..64)
        .map(|i| format!("m{i}-{}", "x".repeat(110)))
        .collect();
    let clients: Vec<serde_json::Value> = ["acme", "beta", "gamma", "only-in-the-answer"]
        .iter()
        .map(|n| json!({"name": n, "daily_budget": null, "utilization_limit": null, "models": models, "preferred_endpoints": [], "keys": []}))
        .collect();
    let text = json!({"schema": 1, "version": V0 + 1, "updated_at": "2026-10-11T06:00:00.000Z", "clients": clients, "revoked_digests": []}).to_string();
    let digest = digest_of(&text);
    // An answer whose diff alone is far past a line, every value a quote,
    // which JSON doubles.
    let diff: Vec<serde_json::Value> = (0..400)
        .map(|i| json!({"client": "acme", "change": "policy", "field": "models", "from": "\"".repeat(40), "to": i}))
        .collect();
    let store = FakeStore::new(&World::new(vec![]));
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "version": V0 + 1, "digest": digest, "text": text, "diff": diff, "keys": []}),
    );
    let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let update = doc_form(
        &csrf,
        &[
            ("op", "update"),
            ("name", "acme"),
            ("models", &models.join(",")),
        ],
    );
    let (resp, lines) = captured(post_form(&state, &cookie, "/lb/clients/commit", update)).await;
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
    for line in lines.lines() {
        assert!(
            line.len() < 16 * 1024,
            "a {}-byte line: {}",
            line.len(),
            &line[..200]
        );
    }
    let records = log_records(&lines);
    let parts: Vec<&serde_json::Value> = records
        .iter()
        .filter(|r| r["fields.message"] == "client-store document")
        .collect();
    assert!(parts.len() > 3, "{} parts", parts.len());
    for p in &parts {
        assert_eq!(p["fields.version"], json!((V0 + 1).to_string()));
        assert_eq!(p["fields.digest"], json!(digest));
        assert_eq!(p["target"], "ops_console::write_log");
    }
    // The attempt's own line holds neither the document nor its bytes.
    let attempt = records
        .iter()
        .find(|r| r["fields.message"] == "client-store write committed")
        .expect("the attempt is logged");
    assert!(
        !attempt.to_string().contains("only-in-the-answer"),
        "{attempt}"
    );
    // Read back as the import reads it: the exact bytes.
    let metrics = FakeMetrics::default();
    metrics.ship(&lines);
    let back = writelog::document(&metrics_client(&metrics).await, V0 + 1, &digest)
        .await
        .unwrap();
    assert_eq!(back, text);
}

/// An answer whose bytes do not hash to its digest, or that carries none,
/// puts no document on the write log, and says why.
#[tokio::test]
async fn a_commit_without_matching_bytes_logs_no_document() {
    for (text, why) in [
        (
            Some(json!("not the document")),
            "does not hash to its digest",
        ),
        (None, "the store sent no text"),
    ] {
        let mut answer =
            json!({"status": 200, "version": V0 + 1, "digest": D1, "diff": [], "keys": []});
        if let Some(t) = text {
            answer["text"] = t;
        }
        let store = FakeStore::new(&World::new(vec![]));
        store.answer_post(StatusCode::OK, answer);
        let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
        let (cookie, csrf) = cookie_for(&state, "admin-sub");
        let delete = doc_form(&csrf, &[("op", "delete"), ("name", "acme")]);
        let (_, lines) = captured(post_form(&state, &cookie, "/lb/clients/commit", delete)).await;
        assert!(lines.contains(why), "{lines}");
        assert!(
            !lines.contains("\"message\":\"client-store document\""),
            "{lines}"
        );
        // Still committed: the labels carry it.
        assert!(
            state
                .store
                .as_ref()
                .unwrap()
                .committed
                .render()
                .contains(D1)
        );
    }
}

// ─── Import ─────────────────────────────────────────────────────────────────

fn import_form(csrf: &str, expected: u64, pairs: &[(&str, &str)]) -> String {
    let expected = expected.to_string();
    let mut all = vec![
        ("csrf", csrf),
        ("expected_version", expected.as_str()),
        ("reason", "test"),
        ("op", "import"),
    ];
    all.extend_from_slice(pairs);
    form(&all)
}

/// The commit form an import preview renders, as fields.
fn commit_fields(csrf: &str, expected: u64, text: &str, extra: &[(&str, &str)]) -> String {
    let (b64, digest) = (STANDARD.encode(text.as_bytes()), digest_of(text));
    let mut pairs = vec![
        ("text_b64", b64.as_str()),
        ("expected_digest", digest.as_str()),
    ];
    pairs.extend_from_slice(extra);
    import_form(csrf, expected, &pairs)
}

/// The guard as the design writes it: an empty install record refuses every
/// import, a store at version 0 included, at the preview and at the commit,
/// and so does a metrics store that cannot answer.
#[tokio::test]
async fn an_import_is_refused_on_an_empty_metrics_answer_store_version_0_included() {
    let mut world = World::new(vec![]);
    world.doc = (0, "0".repeat(64));
    let store = FakeStore::new(&world);
    store.answer_post(
        StatusCode::OK,
        dry_run(json!([{"client": "acme", "change": "created"}])),
    );
    let metrics = quiet_fleet();
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let text = doc_text(1, 5);
    let preview = || {
        post_form(
            &state,
            &cookie,
            "/lb/clients/preview",
            import_form(&csrf, 0, &[("source", "paste"), ("document", &text)]),
        )
    };
    let html = body_string(preview().await).await;
    assert!(
        html.contains("The import is refused: the metrics store shows no document installed in the last 30 days, which reads as unknown."),
        "{html}"
    );
    assert!(!html.contains("action=\"/lb/clients/commit\""), "{html}");
    // A commit sent anyway never reaches the store.
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/commit",
        commit_fields(&csrf, 0, &text, &[]),
    )
    .await;
    assert!(flash_of(&state, &resp).msg.contains("reads as unknown"));
    assert_eq!(store.posts().len(), 1, "only the preview's dry run");
    // A metrics store that cannot answer refuses it the same way.
    metrics
        .failing
        .lock()
        .unwrap()
        .push("anthropic_client_store_document".into());
    let html = body_string(preview().await).await;
    assert!(
        html.contains("The import is refused: the metrics store could not be read"),
        "{html}"
    );
    assert!(!html.contains("action=\"/lb/clients/commit\""), "{html}");
}

/// Below the newest version instances installed, the import needs each
/// newer version typed, in any order and nothing else; the very document
/// they installed there needs nothing.
#[tokio::test]
async fn an_import_below_the_high_water_needs_each_newer_version_typed() {
    let store = FakeStore::new(&World::new(vec![]));
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let metrics = quiet_fleet();
    let newest = doc_text(V0 + 3, 5);
    metrics.set_30d(
        "anthropic_client_store_document",
        vec![
            installed(V0 - 5, D1),
            installed(V0, &"0".repeat(64)),
            installed(V0 + 3, &digest_of(&newest)),
        ],
    );
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let text = doc_text(V0 - 5, 5);
    let html = body_string(
        post_form(
            &state,
            &cookie,
            "/lb/clients/preview",
            import_form(&csrf, V0, &[("source", "paste"), ("document", &text)]),
        )
        .await,
    )
    .await;
    assert!(html.contains("name=\"confirm_discard\""), "{html}");
    assert!(
        html.contains(&format!("Version {} is below 2 versions", V0 - 5)),
        "{html}"
    );
    assert!(html.contains(&format!("{V0}, {}", V0 + 3)), "{html}");
    // An empty diff still offers the commit: an import repairs the store.
    assert!(html.contains("action=\"/lb/clients/commit\""), "{html}");
    let commit = |typed: &str| {
        post_form(
            &state,
            &cookie,
            "/lb/clients/commit",
            commit_fields(&csrf, V0, &text, &[("confirm_discard", typed)]),
        )
    };
    for typed in ["", &V0.to_string(), &format!("{V0} {} {}", V0 + 3, V0 + 4)] {
        let resp = commit(typed).await;
        assert!(
            flash_of(&state, &resp).msg.contains("Nothing was sent"),
            "{typed:?}"
        );
    }
    assert_eq!(store.posts().len(), 1, "only the preview's dry run");
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "version": V0 + 9, "digest": D3, "diff": [], "keys": []}),
    );
    let resp = commit(&format!("{}, {V0}", V0 + 3)).await;
    assert_eq!(
        location(&resp).split('?').next(),
        Some("/lb/clients/effect")
    );
    let sent = store.posts().pop().unwrap();
    assert_eq!(sent["document"]["dry_run"], json!(false));
    assert_eq!(
        sent["document"]["ops"],
        json!([{"op": "import", "text": text, "expected_digest": digest_of(&text)}])
    );
    // The document instances installed at the newest version: no override.
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let html = body_string(
        post_form(
            &state,
            &cookie,
            "/lb/clients/preview",
            import_form(&csrf, V0, &[("source", "paste"), ("document", &newest)]),
        )
        .await,
    )
    .await;
    assert!(!html.contains("name=\"confirm_discard\""), "{html}");
    assert!(!html.contains("name=\"confirm_version\""), "{html}");
    assert!(
        html.contains(&format!(
            "Version {} is the newest document instances installed in the last 30 days: they installed it with this digest.",
            V0 + 3
        )),
        "{html}"
    );
}

/// At or above the newest version instances installed, an import goes
/// untyped only as the very document they installed there. Other bytes at
/// that version, or any version above it, which a document written by hand
/// can claim, need the version typed. A pasted document is checked against
/// the digest its log line names.
#[tokio::test]
async fn an_import_at_or_above_the_high_water_is_typed_unless_instances_installed_it() {
    let store = FakeStore::new(&World::new(vec![]));
    let metrics = quiet_fleet();
    let served = doc_text(V0, 5);
    metrics.set_30d(
        "anthropic_client_store_document",
        vec![installed(V0, &digest_of(&served))],
    );
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let preview = |text: String, digest: String| {
        let (state, cookie, csrf) = (state.clone(), cookie.clone(), csrf.clone());
        async move {
            let fields = import_form(
                &csrf,
                V0,
                &[
                    ("source", "paste"),
                    ("document", &text),
                    ("digest", &digest),
                ],
            );
            body_string(post_form(&state, &cookie, "/lb/clients/preview", fields).await).await
        }
    };
    // What instances installed, pasted with the digest its log line names.
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let html = preview(served.clone(), digest_of(&served).to_uppercase()).await;
    assert!(html.contains("matching its logged digest"), "{html}");
    assert!(
        html.contains(&format!(
            "Version {V0} is the newest document instances installed"
        )),
        "{html}"
    );
    assert!(!html.contains("name=\"confirm_version\""), "{html}");
    // Other bytes at that version: not what instances served.
    let other = doc_text(V0, 6);
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let html = preview(other.clone(), String::new()).await;
    assert!(html.contains("with no logged digest"), "{html}");
    assert!(html.contains("name=\"confirm_version\""), "{html}");
    assert!(
        html.contains(&format!(
            "Version {V0} cannot be checked: instances installed version {V0} with digest {}, not this document's {}, so it is not the document they served.",
            short_digest(&digest_of(&served)),
            short_digest(&digest_of(&other))
        )),
        "{html}"
    );
    // Above it: no instance has served it.
    let above = doc_text(V0 + 5, 5);
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let html = preview(above.clone(), String::new()).await;
    assert!(html.contains("name=\"confirm_version\""), "{html}");
    assert!(
        html.contains(&format!(
            "it is above version {V0}, the newest instances installed"
        )),
        "{html}"
    );
    assert_eq!(store.posts().len(), 3, "three dry runs");
    // Its commit goes only with its version typed.
    let commit = |typed: &str| {
        post_form(
            &state,
            &cookie,
            "/lb/clients/commit",
            commit_fields(&csrf, V0, &above, &[("confirm_version", typed)]),
        )
    };
    let resp = commit(&V0.to_string()).await;
    assert!(flash_of(&state, &resp).msg.contains("Nothing was sent"));
    assert_eq!(store.posts().len(), 3, "nothing sent");
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "version": V0 + 9, "digest": D3, "diff": [], "keys": []}),
    );
    commit(&(V0 + 5).to_string()).await;
    let sent = store.posts().pop().unwrap();
    assert_eq!(sent["document"]["dry_run"], json!(false));
    assert_eq!(sent["document"]["ops"][0]["text"], json!(above));
    // A paste that does not hash to the digest its log line names is refused
    // before anything is sent.
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/preview",
        import_form(
            &csrf,
            V0,
            &[("source", "paste"), ("document", &served), ("digest", D1)],
        ),
    )
    .await;
    let msg = flash_of(&state, &resp).msg;
    assert!(msg.contains("does not hash to the digest given"), "{msg}");
    assert_eq!(store.posts().len(), 4, "no dry run for it");
}

/// A document from the write log is imported with its exact logged bytes,
/// named by their logged digest, and its preview diffs against the newest
/// document the console committed. What the import commits goes on the
/// write log in turn.
#[tokio::test]
async fn an_import_from_the_write_log_names_the_logged_digest_and_diffs_against_the_newest() {
    let store = FakeStore::new(&World::new(vec![]));
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let metrics = quiet_fleet();
    let (older, newest) = (doc_text(V0 - 5, 1_000), doc_text(V0, 2_000));
    log_and_ship(&metrics, V0 - 5, &older);
    log_and_ship(&metrics, V0, &newest);
    metrics.set_30d(
        "anthropic_client_store_document",
        vec![
            installed(V0 - 5, &digest_of(&older)),
            installed(V0, &digest_of(&newest)),
        ],
    );
    metrics.set_30d(
        "ops_console_committed_document",
        vec![
            committed_label(V0 - 5, &digest_of(&older)),
            committed_label(V0, &digest_of(&newest)),
        ],
    );
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    // A first line that names no document, as only a fault or a write from
    // outside the console makes, is left out and counted, never allowed to
    // hide the rest.
    metrics.write_log.lock().unwrap().push(json!({
        "_time": "2026-11-08T12:00:00Z", "target": "ops_console::write_log",
        "fields.message": "client-store document", "fields.part": "1", "fields.version": "9",
    }));
    // The import page lists both, newest first, each with a preview form.
    let (status, page) = get_page(&state, &cookie, "/lb/clients/import").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        page.contains("also holds 1 document lines that name no version and digest"),
        "{page}"
    );
    let (at_new, at_old) = (
        page.find(&digest_of(&newest)).unwrap(),
        page.find(&digest_of(&older)).unwrap(),
    );
    assert!(at_new < at_old, "{page}");
    assert_eq!(page.matches("newest committed").count(), 1, "{page}");
    assert!(!page.contains("no committed label"), "{page}");
    assert!(page.contains("name=\"source\" value=\"log\""), "{page}");
    let version = (V0 - 5).to_string();
    let digest = digest_of(&older);
    let html = body_string(
        post_form(
            &state,
            &cookie,
            "/lb/clients/preview",
            import_form(
                &csrf,
                V0,
                &[
                    ("source", "log"),
                    ("version", &version),
                    ("digest", &digest),
                ],
            ),
        )
        .await,
    )
    .await;
    assert_eq!(
        store.posts()[0]["document"]["ops"],
        json!([{"op": "import", "text": older, "expected_digest": digest}])
    );
    assert!(
        html.contains(&format!(
            "Against the newest document the console committed: version {V0}"
        )),
        "{html}"
    );
    assert!(html.contains("acme: daily_budget 2000 → 1000"), "{html}");
    assert!(html.contains("write log."), "{html}");
    assert!(
        html.contains(&format!("name=\"expected_digest\" value=\"{digest}\"")),
        "{html}"
    );
    // The newest one, imported, differs from itself in nothing.
    let html = body_string(
        post_form(
            &state,
            &cookie,
            "/lb/clients/preview",
            import_form(
                &csrf,
                V0,
                &[
                    ("source", "log"),
                    ("version", &V0.to_string()),
                    ("digest", &digest_of(&newest)),
                ],
            ),
        )
        .await,
    )
    .await;
    assert!(
        html.contains("No change from the newest document the console committed"),
        "{html}"
    );
    // Committed, the store's document goes on the write log.
    let committed_text = doc_text(V0 + 7, 1_000);
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "version": V0 + 7, "digest": digest_of(&committed_text), "text": committed_text, "diff": [], "keys": []}),
    );
    let fields = commit_fields(&csrf, V0, &older, &[("confirm_discard", &V0.to_string())]);
    let (_, lines) = captured(post_form(&state, &cookie, "/lb/clients/commit", fields)).await;
    metrics.ship(&lines);
    let logged = writelog::logged(&metrics_client(&metrics).await)
        .await
        .unwrap();
    assert_eq!(logged.docs[0].version, V0 + 7);
    assert_eq!(logged.docs[0].digest, digest_of(&committed_text));
    // Its request is logged without the document's bytes.
    let attempt = log_records(&lines)
        .into_iter()
        .find(|r| r["fields.message"] == "client-store write committed")
        .unwrap();
    let request = attempt["fields.request"].as_str().unwrap();
    assert!(
        request.contains(&format!("\"bytes\":{}", older.len())),
        "{request}"
    );
    assert!(!request.contains("daily_budget"), "{request}");
}

/// A copy on the write log that neither this process's exports nor a
/// committed label confirms is never what an import is diffed against,
/// however new its version or its lines: a forged copy of the newest
/// document with one key added would otherwise be diffed against itself
/// and show nothing. This process's own exports, which nothing outside it
/// can write, come before any label. The page flags an unconfirmed copy,
/// and a version the log names with two digests.
#[tokio::test]
async fn a_logged_copy_nothing_confirms_is_never_the_diff_base() {
    let store = FakeStore::new(&World::new(vec![]));
    let metrics = quiet_fleet();
    let real = doc_text(V0, 1_000);
    let planted = json!({"schema": 1, "version": V0, "updated_at": "2026-10-11T06:00:00.000Z",
        "clients": [{"name": "acme", "daily_budget": 1_000, "utilization_limit": null, "models": [],
                     "preferred_endpoints": [], "keys": [{"id": "k-planted", "label": "planted", "digest": D3}]}],
        "revoked_digests": []})
    .to_string();
    log_and_ship(&metrics, V0, &real);
    // Shipped later, so its lines come first.
    log_and_ship(&metrics, V0, &planted);
    metrics.set_30d(
        "anthropic_client_store_document",
        vec![installed(V0, &digest_of(&real))],
    );
    metrics.set_30d(
        "ops_console_committed_document",
        vec![committed_label(V0, &digest_of(&real))],
    );
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let (_, page) = get_page(&state, &cookie, "/lb/clients/import").await;
    assert_eq!(
        page.matches("another digest at this version").count(),
        2,
        "{page}"
    );
    assert_eq!(page.matches("no committed label").count(), 1, "{page}");
    assert_eq!(page.matches("newest committed").count(), 1, "{page}");
    let (badge, planted_at) = (
        page.find("newest committed").unwrap(),
        page.find(&digest_of(&planted)).unwrap(),
    );
    assert!(
        badge > planted_at,
        "the badge is on the real copy, listed second: {page}"
    );
    // The planted copy's preview diffs against the real one, so its key shows,
    // and the guard asks for its version: instances installed other bytes.
    let from_log = |version: u64, digest: String| {
        let (state, cookie, csrf) = (state.clone(), cookie.clone(), csrf.clone());
        async move {
            let v = version.to_string();
            let fields = import_form(
                &csrf,
                V0,
                &[("source", "log"), ("version", &v), ("digest", &digest)],
            );
            body_string(post_form(&state, &cookie, "/lb/clients/preview", fields).await).await
        }
    };
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let html = from_log(V0, digest_of(&planted)).await;
    assert!(
        html.contains(&format!(
            "Against the newest document the console committed: version {V0}"
        )),
        "{html}"
    );
    assert!(
        html.contains("acme: key added k-planted (planted)"),
        "{html}"
    );
    assert!(html.contains("name=\"confirm_version\""), "{html}");
    // The console commits a document itself; then a forged label and a forged
    // copy name a higher version. This process's own export stays the base.
    let committed_text = doc_text(V0 + 7, 1_000);
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "version": V0 + 7, "digest": digest_of(&committed_text), "text": committed_text, "diff": [], "keys": []}),
    );
    let (_, lines) = captured(post_form(
        &state,
        &cookie,
        "/lb/clients/commit",
        commit_fields(&csrf, V0, &real, &[]),
    ))
    .await;
    metrics.ship(&lines);
    let forged = doc_text(V0 + 50, 1_000);
    log_and_ship(&metrics, V0 + 50, &forged);
    metrics.set_30d(
        "ops_console_committed_document",
        vec![
            committed_label(V0, &digest_of(&real)),
            committed_label(V0 + 50, &digest_of(&forged)),
        ],
    );
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let html = from_log(V0, digest_of(&planted)).await;
    assert!(
        html.contains(&format!(
            "Against the newest document the console committed: version {}",
            V0 + 7
        )),
        "{html}"
    );
}

/// The routes that take a whole document, or a reseed's sets, take a form
/// past axum's 2 MB default, since form encoding can triple a 2 MiB
/// document; every other write keeps the default.
#[tokio::test]
async fn document_forms_take_bodies_past_the_default_limit() {
    let store = FakeStore::new(&World::new(vec![]));
    let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, _) = cookie_for(&state, "admin-sub");
    // 3.3 MB: a 1.1 MB run of quotes, as a form encodes them.
    let body = format!("csrf=x&document={}", "%22".repeat(1_100_000));
    for path in [
        "/lb/clients/preview",
        "/lb/clients/commit",
        "/lb/clients/reseed",
    ] {
        let resp = post_form(&state, &cookie, path, body.clone()).await;
        assert_ne!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE, "{path}");
    }
    let resp = post_form(&state, &cookie, "/lb/clients/restrict", body).await;
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(store.posts().is_empty());
}

/// Pasted text is the document's JSON, or its base64 as an install log
/// carries it; the form's CRLF line breaks read back as LF, JSON loses its
/// surrounding space, base64 every space.
#[tokio::test]
async fn a_pasted_import_reads_json_text_or_base64() {
    let store = FakeStore::new(&World::new(vec![]));
    store.answer_post(StatusCode::OK, dry_run(json!([])));
    let metrics = quiet_fleet();
    metrics.set_30d("anthropic_client_store_document", vec![installed(V0, D1)]);
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let text = "{\n  \"schema\": 1,\n  \"version\": 9\n}";
    let b64 = STANDARD.encode(text);
    let wrapped = format!("{}\r\n{}\r\n", &b64[..10], &b64[10..]);
    for pasted in [format!("\r\n{}\r\n", text.replace('\n', "\r\n")), wrapped] {
        post_form(
            &state,
            &cookie,
            "/lb/clients/preview",
            import_form(&csrf, V0, &[("source", "paste"), ("document", &pasted)]),
        )
        .await;
        let sent = store.posts().pop().unwrap();
        assert_eq!(
            sent["document"]["ops"][0],
            json!({"op": "import", "text": text, "expected_digest": digest_of(text)}),
            "{pasted:?}"
        );
    }
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/preview",
        import_form(
            &csrf,
            V0,
            &[("source", "paste"), ("document", "not a document")],
        ),
    )
    .await;
    assert!(
        flash_of(&state, &resp)
            .msg
            .contains("neither JSON text nor base64")
    );
    // A commit whose bytes no longer hash to the digest its preview named.
    let tampered = import_form(
        &csrf,
        V0,
        &[
            ("text_b64", &STANDARD.encode("{}")),
            ("expected_digest", D1),
        ],
    );
    let resp = post_form(&state, &cookie, "/lb/clients/commit", tampered).await;
    assert!(flash_of(&state, &resp).msg.contains("no longer hashes"));
    assert_eq!(store.posts().len(), 2, "the two dry runs, and nothing else");
}

/// A logged document with a part missing is not offered for import: the
/// preview says what is wrong and sends nothing.
#[tokio::test]
async fn a_damaged_logged_document_is_not_imported() {
    let store = FakeStore::new(&World::new(vec![]));
    let metrics = quiet_fleet();
    let big = json!({"schema": 1, "version": V0, "updated_at": "2026-10-11T06:00:00.000Z",
                     "clients": [{"name": "acme", "models": ["x".repeat(30_000)]}], "revoked_digests": []})
    .to_string();
    log_and_ship(&metrics, V0, &big);
    metrics
        .write_log
        .lock()
        .unwrap()
        .retain(|r| r["fields.part"] != "2");
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/preview",
        import_form(
            &csrf,
            V0,
            &[
                ("source", "log"),
                ("version", &V0.to_string()),
                ("digest", &digest_of(&big)),
            ],
        ),
    )
    .await;
    let msg = flash_of(&state, &resp).msg;
    assert!(
        msg.contains("cannot be imported") && msg.contains("parts"),
        "{msg}"
    );
    assert!(store.posts().is_empty());
}

// ─── Point-in-time recovery ─────────────────────────────────────────────────

/// 2026-11-08T10:00:00Z, in Unix ms.
const T10: u64 = 1_794_132_000_000;
/// Each document's version runs ten minutes ahead of its commit, as the
/// store's clock can: the restore time must ignore it.
const AHEAD: u64 = 600_000;

fn row(version: u64, digest: &str, at: &str) -> serde_json::Value {
    json!({"version": version, "digest": digest, "at": at, "role": "writer", "actor": "ops@example.test",
           "reason": "test", "ops": [], "recovery_id": null, "diff": []})
}

/// Three commits a minute and an hour apart, newest first in history, with
/// a recovery window restoring to `window`, or none open.
fn pitr_store(window: Option<&str>) -> FakeStore {
    store_with(&World::new(vec![]), |b| {
        b["document"]["history"] = json!([
            row(T10 + 3_600_000 + AHEAD, D3, "2026-11-08T11:00:00.000Z"),
            row(T10 + 60_000 + AHEAD, D2, "2026-11-08T10:01:00.000Z"),
            row(T10 + AHEAD, D1, "2026-11-08T10:00:00.000Z"),
        ]);
        b["document"]["pitr_window"] = match window {
            Some(t) => {
                json!({"open": true, "usable": true, "restore_to": t, "closes": "2026-11-08T13:00:00.000Z"})
            }
            None => json!({"open": false, "usable": false, "restore_to": null, "closes": null}),
        };
        b["pitr_window"] = json!(window.is_some());
    })
}

/// The console's committed labels hold the two older commits.
fn labelled(metrics: &FakeMetrics) {
    metrics.set_30d(
        "ops_console_committed_document",
        vec![
            series(
                &[("version", &(T10 + AHEAD).to_string()), ("digest", D1)],
                "1794132000",
            ),
            series(
                &[
                    ("version", &(T10 + 60_000 + AHEAD).to_string()),
                    ("digest", D2),
                ],
                "1794132060",
            ),
        ],
    );
}

/// The targets come from the committed labels and the store's history, the
/// restore time from the commit time and never the version, and the page
/// shows the window's inputs and, once the window is inside the pick's
/// times, the send form.
#[tokio::test]
async fn the_pitr_page_lists_labels_and_history_and_times_the_restore_from_the_commit() {
    let store = pitr_store(Some("2026-11-08T10:00:30.000Z"));
    let metrics = quiet_fleet();
    labelled(&metrics);
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, _) = cookie_for(&state, "admin-sub");
    let target = format!("{}.{D1}", T10 + AHEAD);
    let (status, html) = get_page(
        &state,
        &cookie,
        &format!("/lb/clients/pitr?target={target}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(html.matches("console and history").count(), 2, "{html}");
    assert_eq!(
        html.matches("history only: typed confirmation").count(),
        1,
        "{html}"
    );
    assert!(
        html.contains("The restore time must fall after 2026-11-08T10:00:00.000Z and before the next commit, at 2026-11-08T10:01:00.000Z, and before the wipe or edit being recovered from if that came first."),
        "{html}"
    );
    // Midway between the commits; the version, ten minutes later, plays no part.
    assert!(
        html.contains("-f pitr_at=2026-11-08T10:00:30.000Z -f pitr_until="),
        "{html}"
    );
    assert!(html.contains("action=\"/lb/clients/pitr\""), "{html}");
    assert!(!html.contains("name=\"confirm_version\""), "{html}");
    // A window outside the pick's times offers no send.
    let store = pitr_store(Some("2026-11-08T10:05:00.000Z"));
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, _) = cookie_for(&state, "admin-sub");
    let (_, html) = get_page(
        &state,
        &cookie,
        &format!("/lb/clients/pitr?target={target}"),
    )
    .await;
    assert!(
        html.contains("outside this pick&#x27;s restore times")
            || html.contains("outside this pick's restore times"),
        "{html}"
    );
    assert!(!html.contains("action=\"/lb/clients/pitr\""), "{html}");
}

fn pitr_form(csrf: &str, target: &str, extra: &[(&str, &str)]) -> String {
    let v0 = V0.to_string();
    let mut pairs = vec![
        ("csrf", csrf),
        ("target", target),
        ("expected_version", v0.as_str()),
        ("reason", "recover from the wipe"),
    ];
    pairs.extend_from_slice(extra);
    form(&pairs)
}

/// No metrics value can stop a pitr: with no committed label to read, the
/// store's history is the list, each pick behind its typed version, and the
/// pitr goes out pinned to the page's version, naming the picked digest.
#[tokio::test]
async fn pitr_goes_ahead_on_an_empty_metrics_answer_and_names_the_picked_digest() {
    let store = pitr_store(Some("2026-11-08T11:00:30.000Z"));
    let id = "ab".repeat(16);
    store.answer_post(
        StatusCode::ACCEPTED,
        json!({"status": 202, "message": "restarted", "marker": {"id": id, "state": "pending", "expected_restore_digest": D3}}),
    );
    let metrics = quiet_fleet();
    metrics
        .failing
        .lock()
        .unwrap()
        .push("ops_console_committed_document".into());
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let target = format!("{}.{D3}", T10 + 3_600_000 + AHEAD);
    let (_, html) = get_page(
        &state,
        &cookie,
        &format!("/lb/clients/pitr?target={target}"),
    )
    .await;
    assert!(
        html.contains("The committed labels could not be read"),
        "{html}"
    );
    assert!(html.contains("name=\"confirm_version\""), "{html}");
    // Not typed: nothing sent.
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/pitr",
        pitr_form(&csrf, &target, &[]),
    )
    .await;
    assert!(flash_of(&state, &resp).msg.contains("type"));
    assert!(store.posts().is_empty());
    let version = (T10 + 3_600_000 + AHEAD).to_string();
    let (resp, lines) = captured(post_form(
        &state,
        &cookie,
        "/lb/clients/pitr",
        pitr_form(&csrf, &target, &[("confirm_version", &version)]),
    ))
    .await;
    assert_eq!(
        location(&resp).split('&').next().unwrap(),
        format!("/lb/clients/pitr/marker?id={id}")
    );
    let sent = &store.posts()[0];
    assert_eq!(sent["document"]["expected_version"], json!(V0));
    assert_eq!(sent["document"]["dry_run"], json!(false));
    assert_eq!(
        sent["document"]["ops"],
        json!([{"op": "pitr", "expected_restore_digest": D3}])
    );
    // The attempt is on the write log, with the marker the store left.
    assert!(
        lines.contains("its recovery marker says how it ends"),
        "{lines}"
    );
    assert!(lines.contains(&id), "{lines}");
}

/// pitr is sent only while a usable window restores inside the pick's
/// times; anything else is refused before the store hears of it.
#[tokio::test]
async fn pitr_is_not_sent_while_the_window_cannot_return_to_the_pick() {
    let metrics = quiet_fleet();
    labelled(&metrics);
    let target = format!("{}.{D1}", T10 + AHEAD);
    for (window, why) in [
        (None, "No recovery window is open"),
        (Some("2026-11-08T10:02:00.000Z"), "outside this pick"),
        (Some("2026-11-08T09:59:59.000Z"), "outside this pick"),
    ] {
        let store = pitr_store(window);
        let state = store_state(&store, &metrics, &["admin-sub"]).await;
        let (cookie, csrf) = cookie_for(&state, "admin-sub");
        let resp = post_form(
            &state,
            &cookie,
            "/lb/clients/pitr",
            pitr_form(&csrf, &target, &[]),
        )
        .await;
        let msg = flash_of(&state, &resp).msg;
        assert!(
            msg.contains(why) && msg.contains("Nothing was sent"),
            "{msg}"
        );
        assert!(store.posts().is_empty(), "{window:?}");
    }
    // The 503s that say no pitr ran ask for it again.
    let store = pitr_store(Some("2026-11-08T10:00:30.000Z"));
    store.answer_post(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"status": 503, "errors": [{"code": "pitr.failed", "message": "the pitr failed before it left a recovery marker; nothing was scheduled"}]}),
    );
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/pitr",
        pitr_form(&csrf, &target, &[]),
    )
    .await;
    let msg = flash_of(&state, &resp).msg;
    assert!(
        msg.contains("Nothing was scheduled") && msg.contains("pitr.failed"),
        "{msg}"
    );
    // A target that is no pick, even one no header can carry, is refused
    // back to the page itself.
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/pitr",
        pitr_form(&csrf, "1\nabc", &[]),
    )
    .await;
    assert_eq!(location(&resp), "/lb/clients/pitr");
    assert!(
        flash_of(&state, &resp)
            .msg
            .contains("names no document to return to")
    );
}

fn marker(id: &str, state: &str, reason: Option<&str>) -> serde_json::Value {
    json!({"id": id, "state": state, "expected_restore_digest": D1, "floor": V0,
           "created_at": "2026-11-08T12:00:00.000Z", "updated_at": "2026-11-08T12:00:05.000Z", "reason": reason})
}

/// The marker reads as done, failed or refused, each with what comes next;
/// a pending one refreshes until it is judged.
#[tokio::test]
async fn the_marker_reads_as_done_failed_or_refused() {
    let id = "cd".repeat(16);
    let at = crate::session::now_epoch();
    let page = format!("/lb/clients/pitr/marker?id={id}&at={at}");
    for (state_name, reason, want, refreshes) in [
        (
            "pending",
            None,
            "Pending: the store judges the marker",
            true,
        ),
        (
            "failed",
            Some("no restore ran: the recovery's sentinel survived"),
            "Failed: nothing was restored (no restore ran",
            false,
        ),
        (
            "refused",
            Some("the restore landed on a document other than the one picked"),
            "Refused: the restore landed on a document other than the one picked",
            false,
        ),
    ] {
        let store = store_with(&World::new(vec![]), |b| {
            b["restrictions"]["markers"] = json!([marker(&id, state_name, reason)]);
        });
        let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
        let (cookie, _) = cookie_for(&state, "admin-sub");
        let resp = app(state.clone())
            .oneshot(
                HttpRequest::get(&page)
                    .header(header::COOKIE, &cookie)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::REFRESH).is_some(),
            refreshes,
            "{state_name}"
        );
        let html = body_string(resp).await;
        assert!(html.contains(want), "{state_name}: {html}");
        assert!(!html.contains("Accept the recovery"), "{state_name}");
    }
    let store = store_with(&World::new(vec![]), |b| {
        b["restrictions"]["markers"] = json!([marker(&"ef".repeat(16), "done", None)]);
    });
    let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, _) = cookie_for(&state, "admin-sub");
    let (_, html) = get_page(&state, &cookie, &page).await;
    assert!(
        html.contains("holds no recovery marker with this id"),
        "{html}"
    );
    let (status, _) = get_page(&state, &cookie, "/lb/clients/pitr/marker?id=nope").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// A done recovery shows its commits' diffs, and what the rewind changed
/// against the newest document the console committed, warning on each name
/// it drops that a stack's TOML still holds. It is exported as committed only
/// when the operator accepts it, with the rows they saw, and as of when the
/// store committed each; its document then goes on the write log from the
/// store's view.
#[tokio::test]
async fn a_recovery_is_exported_as_committed_only_once_accepted() {
    let id = "ef".repeat(16);
    let text = doc_text(V0 + 2, 7);
    let metrics = quiet_fleet();
    let before = json!({"schema": 1, "version": V0 - 9, "updated_at": "2026-10-11T06:00:00.000Z",
        "clients": [
            {"name": "acme", "daily_budget": 7, "utilization_limit": null, "models": [], "preferred_endpoints": [], "keys": []},
            {"name": "beta", "daily_budget": null, "utilization_limit": null, "models": [], "preferred_endpoints": [], "keys": []},
        ],
        "revoked_digests": []})
    .to_string();
    log_and_ship(&metrics, V0 - 9, &before);
    metrics.set_30d(
        "ops_console_committed_document",
        vec![committed_label(V0 - 9, &digest_of(&before))],
    );
    metrics.set(
        "anthropic_client_store_toml_client",
        vec![series(
            &[("stack", "k3s"), ("pod", "lb-a"), ("client", "beta")],
            "1",
        )],
    );
    let (newest, first) = (digest_of(&text), D2);
    let store = store_with(&World::new(vec![]), |b| {
        b["restrictions"]["markers"] = json!([marker(&id, "done", None)]);
        b["document"]["version"] = json!(V0 + 2);
        b["document"]["digest"] = json!(newest);
        b["document"]["text"] = json!(text);
        b["document"]["history"] = json!([
            {"version": V0 + 2, "digest": newest, "at": "2026-11-08T12:00:06.000Z", "role": "store", "actor": "client-store",
             "reason": "revoked digests", "ops": [], "recovery_id": id, "diff": [{"change": "digest revoked", "digest": D3}]},
            {"version": V0 + 1, "digest": first, "at": "2026-11-08T12:00:05.000Z", "role": "recovery", "actor": "client-store",
             "reason": "point-in-time recovery", "ops": [], "recovery_id": id, "diff": [{"client": "acme", "change": "created"}]},
            row(V0 - 9, D1, "2026-11-08T09:00:00.000Z"),
        ]);
    });
    let state = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let committed = || state.store.as_ref().unwrap().committed.render();
    let (_, html) = get_page(
        &state,
        &cookie,
        &format!("/lb/clients/pitr/marker?id={id}&at=0"),
    )
    .await;
    assert!(
        html.contains("The restore landed on the picked document"),
        "{html}"
    );
    assert!(
        html.contains("acme: created") && html.contains("digest revoked 3333"),
        "{html}"
    );
    // The rewind itself, against the newest document the console committed.
    assert!(
        html.contains(&format!(
            "Against the newest document the console committed: version {}",
            V0 - 9
        )) && html.contains("beta: deleted (its keys stay unrevoked)"),
        "{html}"
    );
    assert!(
        html.contains("beta is still in the TOML of k3s: on those stacks it goes back to its TOML policy and keys."),
        "{html}"
    );
    assert!(html.contains("Accept the recovery"), "{html}");
    assert!(
        !committed().contains(&newest),
        "exported before it was accepted"
    );
    // Rows other than the ones on the page: refused, nothing exported.
    let stale = form(&[
        ("csrf", &csrf),
        ("id", &id),
        ("row", &format!("{}.{first}", V0 + 1)),
    ]);
    let resp = post_form(&state, &cookie, "/lb/clients/pitr/accept", stale).await;
    assert!(flash_of(&state, &resp).msg.contains("review them again"));
    assert!(!committed().contains(first));
    let rows = [
        format!("{}.{first}", V0 + 1),
        format!("{}.{newest}", V0 + 2),
    ];
    let accept = form(&[
        ("csrf", &csrf),
        ("id", &id),
        ("row", &rows[0]),
        ("row", &rows[1]),
    ]);
    let (resp, lines) = captured(post_form(
        &state,
        &cookie,
        "/lb/clients/pitr/accept",
        accept.clone(),
    ))
    .await;
    assert_eq!(
        location(&resp).split('?').next(),
        Some("/lb/clients/effect")
    );
    // Each as of its commit, from its history row: 12:00:05 and 12:00:06.
    assert!(
        committed().contains(&format!("{first}\"}} 1794139205\n"))
            && committed().contains(&format!("{newest}\"}} 1794139206\n")),
        "{}",
        committed()
    );
    assert!(
        lines.contains("client-store recovery accepted") && lines.contains("admin-sub"),
        "{lines}"
    );
    // Accepted again, it is not logged again: every copy is read back with
    // the rest, and three of the largest run past what a read takes. Not
    // while the log store is still indexing the first copy, which reads as
    // absent meanwhile, and not once that copy reads back whole.
    let documents = |lines: &str| {
        log_records(lines)
            .into_iter()
            .filter(|r| r["fields.message"] == "client-store document")
            .count()
    };
    let accept_again = || {
        captured(post_form(
            &state,
            &cookie,
            "/lb/clients/pitr/accept",
            accept.clone(),
        ))
    };
    let (_, settling) = accept_again().await;
    assert_eq!(documents(&settling), 0, "{settling}");
    assert!(settling.contains("logged it moments ago"), "{settling}");
    metrics.ship(&lines);
    let back = writelog::document(&metrics_client(&metrics).await, V0 + 2, &newest)
        .await
        .unwrap();
    assert_eq!(back, text);
    let (_, held) = accept_again().await;
    assert_eq!(documents(&held), 0, "{held}");
    assert!(held.contains("it is already on the write log"), "{held}");
    // A write log that cannot be read gets no copy: one more could only add
    // to a read that already fails.
    metrics.failing.lock().unwrap().push("write-log".into());
    let unread = store_state(&store, &metrics, &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&unread, "admin-sub");
    let fields = form(&[
        ("csrf", &csrf),
        ("id", &id),
        ("row", &rows[0]),
        ("row", &rows[1]),
    ]);
    let (_, failed) = captured(post_form(
        &unread,
        &cookie,
        "/lb/clients/pitr/accept",
        fields,
    ))
    .await;
    assert_eq!(documents(&failed), 0, "{failed}");
    assert!(
        failed.contains("not logged: the write log could not be read"),
        "{failed}"
    );
    metrics.failing.lock().unwrap().clear();
    // A recovery that is not done has nothing to accept.
    let pending = store_with(&World::new(vec![]), |b| {
        b["restrictions"]["markers"] = json!([marker(&id, "pending", None)]);
    });
    let state = store_state(&pending, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let accept = form(&[("csrf", &csrf), ("id", &id), ("row", &rows[0])]);
    let resp = post_form(&state, &cookie, "/lb/clients/pitr/accept", accept).await;
    assert!(flash_of(&state, &resp).msg.contains("not done"));
    assert!(
        !state
            .store
            .as_ref()
            .unwrap()
            .committed
            .render()
            .contains(first)
    );
    // A row whose time the console cannot read is never exported as of
    // another: nothing is accepted.
    let timeless = store_with(&World::new(vec![]), |b| {
        b["restrictions"]["markers"] = json!([marker(&id, "done", None)]);
        b["document"]["history"] = json!([
            {"version": V0 + 1, "digest": first, "at": "yesterday", "role": "recovery", "actor": "client-store",
             "reason": "point-in-time recovery", "ops": [], "recovery_id": id, "diff": []},
        ]);
    });
    let state = store_state(&timeless, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let accept = form(&[("csrf", &csrf), ("id", &id), ("row", &rows[0])]);
    let resp = post_form(&state, &cookie, "/lb/clients/pitr/accept", accept).await;
    assert!(
        flash_of(&state, &resp)
            .msg
            .contains("names no commit time the console can read")
    );
    assert!(
        !state
            .store
            .as_ref()
            .unwrap()
            .committed
            .render()
            .contains(first)
    );
}

// ─── The reseed ─────────────────────────────────────────────────────────────

/// A reseed form as its page renders one: a fresh one-time token, unless
/// `extra` names one.
fn reseed_form(csrf: &str, sets: &[(&str, &str, bool)], extra: &[(&str, &str)]) -> String {
    let mut pairs: Vec<(String, String)> = vec![
        ("csrf".into(), csrf.into()),
        ("reason".into(), "reseed after the wipe".into()),
    ];
    if !extra.iter().any(|(k, _)| *k == "once") {
        pairs.push(("once".into(), crate::session::random_token()));
    }
    for (i, (text, hash, complete)) in sets.iter().enumerate() {
        pairs.push(("set_text".into(), text.to_string()));
        pairs.push(("set_hash".into(), hash.to_string()));
        if *complete {
            pairs.push(("set_complete".into(), (i + 1).to_string()));
        }
    }
    for (k, v) in extra {
        pairs.push((k.to_string(), v.to_string()));
    }
    let borrowed: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    form(&borrowed)
}

const CF_SET: &str = "r acme disabled 1791700000123000\nr beta enabled 1791700000124000\n";
const K3S_SET: &str = "r acme disabled 1791700000123000\nr gamma disabled 1791700000125000\n";

/// Each reseed form is sent once. Its answer page reloaded, or the form sent
/// twice, is refused, since an override sent again would mark whatever the
/// table holds by then complete; so is a form with no token.
#[tokio::test]
async fn a_reseed_form_is_sent_once() {
    let store = FakeStore::new(&World::new(vec![]));
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "written": [], "digests_added": 0, "digests_ignored": [], "integrity": "override", "cf_set_found": false, "reenable_by_hand": [], "client_store": "answered"}),
    );
    let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let (_, page) = get_page(&state, &cookie, "/lb/clients/reseed").await;
    let once = page
        .split("name=\"once\" value=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("the form carries a one-time token")
        .to_string();
    let phrase = "mark the restrictions complete without the CF set";
    let sent = reseed_form(
        &csrf,
        &[(K3S_SET, &digest_of(K3S_SET), true)],
        &[("override", phrase), ("once", &once)],
    );
    let resp = post_form(&state, &cookie, "/lb/clients/reseed", sent.clone()).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = post_form(&state, &cookie, "/lb/clients/reseed", sent).await;
    assert!(flash_of(&state, &resp).msg.contains("already sent"));
    let tokenless = reseed_form(
        &csrf,
        &[(K3S_SET, &digest_of(K3S_SET), true)],
        &[("once", "")],
    );
    let resp = post_form(&state, &cookie, "/lb/clients/reseed", tokenless).await;
    assert!(flash_of(&state, &resp).msg.contains("already sent"));
    assert_eq!(store.posts().len(), 1, "sent once");
}

/// The form sends the sets as supplied, their line breaks as LF, each
/// marked complete, and no override unless one is typed.
#[tokio::test]
async fn the_reseed_sends_the_supplied_sets() {
    let store = FakeStore::new(&World::new(vec![]));
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "written": [], "digests_added": 0, "digests_ignored": [], "integrity": "cf_set", "cf_set_found": true, "reenable_by_hand": [], "client_store": "answered"}),
    );
    let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let (status, page) = get_page(&state, &cookie, "/lb/clients/reseed").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page.matches("name=\"set_text\"").count(), 8, "{page}");
    let crlf = CF_SET.replace('\n', "\r\n");
    let (cf, k3s) = (digest_of(CF_SET), digest_of(K3S_SET));
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/reseed",
        reseed_form(
            &csrf,
            &[(&crlf, &cf, true), ("", "", false), (K3S_SET, &k3s, true)],
            &[],
        ),
    )
    .await;
    let html = body_string(resp).await;
    assert!(html.contains("The table is complete again"), "{html}");
    let sent = &store.posts()[0];
    assert_eq!(
        sent,
        &json!({"restrictions_import": {"actor": "ops@example.test", "reason": "reseed after the wipe", "sets": [
            {"text": CF_SET, "hash": cf, "complete": true},
            {"text": K3S_SET, "hash": k3s, "complete": true},
        ]}})
    );
}

/// The override goes only when typed exactly, and the write log records
/// that it was sent; the sets themselves stay off the line.
#[tokio::test]
async fn the_reseed_override_is_typed_exactly_and_recorded_on_the_write_log() {
    let store = FakeStore::new(&World::new(vec![]));
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "written": [], "digests_added": 0, "digests_ignored": [], "integrity": "override", "cf_set_found": false, "reenable_by_hand": [], "client_store": "answered"}),
    );
    let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let empty = digest_of("");
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/reseed",
        reseed_form(
            &csrf,
            &[("", &empty, true)],
            &[("override", "mark it complete")],
        ),
    )
    .await;
    assert!(flash_of(&state, &resp).msg.contains("typed exactly"));
    assert!(store.posts().is_empty());
    let phrase = "mark the restrictions complete without the CF set";
    let (resp, lines) = captured(post_form(
        &state,
        &cookie,
        "/lb/clients/reseed",
        reseed_form(
            &csrf,
            &[(K3S_SET, &digest_of(K3S_SET), true)],
            &[("override", phrase)],
        ),
    ))
    .await;
    assert!(
        body_string(resp)
            .await
            .contains("marked complete by the typed override")
    );
    assert_eq!(
        store.posts()[0]["restrictions_import"]["override"],
        json!(phrase)
    );
    let attempt = log_records(&lines)
        .into_iter()
        .find(|r| r["fields.message"] == "client-store write committed")
        .expect("the attempt is logged");
    assert_eq!(attempt["fields.kind"], "restrictions_import");
    let request = attempt["fields.request"].as_str().unwrap();
    assert!(request.contains("\"override\":true"), "{request}");
    assert!(
        !request.contains("gamma"),
        "the sets stay off the line: {request}"
    );
}

/// The answer shows every name to re-enable by hand, and what it wrote is
/// exported as committed; a set that does not hash to its hash never
/// leaves the console.
#[tokio::test]
async fn the_reseed_answer_lists_the_names_to_re_enable() {
    let store = FakeStore::new(&World::new(vec![]));
    store.answer_post(
        StatusCode::OK,
        json!({"status": 200, "written": [{"name": "gamma", "version": T0}], "digests_added": 1,
               "digests_ignored": [D1], "integrity": "unchanged", "cf_set_found": true,
               "reenable_by_hand": ["beta", "delta"], "client_store": "answered"}),
    );
    let state = store_state(&store, &quiet_fleet(), &["admin-sub"]).await;
    let (cookie, csrf) = cookie_for(&state, "admin-sub");
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/reseed",
        reseed_form(&csrf, &[(CF_SET, &digest_of(K3S_SET), true)], &[]),
    )
    .await;
    assert!(flash_of(&state, &resp).msg.contains("does not hash"));
    assert!(store.posts().is_empty());
    let resp = post_form(
        &state,
        &cookie,
        "/lb/clients/reseed",
        reseed_form(&csrf, &[(CF_SET, &digest_of(CF_SET), true)], &[]),
    )
    .await;
    let html = body_string(resp).await;
    assert!(
        html.contains("Re-enable by hand, after checking each should be enabled: beta, delta"),
        "{html}"
    );
    assert!(html.contains("not marked complete"), "{html}");
    assert!(html.contains(&format!("gamma at {T0}")), "{html}");
    assert!(html.contains("ignored 1"), "{html}");
    assert!(
        state
            .store
            .as_ref()
            .unwrap()
            .committed
            .render()
            .contains(&format!("name=\"gamma\",version=\"{T0}\""))
    );
}
