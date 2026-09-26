//! `store`, `patch_memory`, `set_generated_summary` and the access increments
//! against a stateful fake Qdrant collection (`common`).
//!
//! Re-store semantics (alaya#86): the point id is derived from `content_hash`
//! and Qdrant's upsert replaces the payload wholesale, so a re-store carries
//! the record's server-maintained fields over and reports `created: false`.
//!
//! Compare-and-set (alaya#130): every write is conditional on the revision it
//! read — insert-only for new content — so a write that lands in between,
//! from this client or another process, is never rolled back. A lost race is
//! retried from a fresh read, at most 8 rounds, and never turns into an
//! unconditional write.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use alaya_backends::{StoreMode, VectorStorage, qdrant::QdrantClient};
use alaya_types::AlayaError;
use alaya_types::memory::{Memory, MetadataUpdate, PatchMemoryRequest};
use common::{
    FakeQdrant, PAYLOAD_DELETE_PATH, PAYLOAD_PATH, POINTS_DELETE_PATH, POINTS_PATH, REV_LOG_LEN,
    external_write, is_conditional, rev_filter, write_order, writes,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The point id `QdrantClient` derives from `hash()`.
const ID: &str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

fn hash() -> String {
    "a".repeat(64)
}

fn client_for(server: &MockServer) -> QdrantClient {
    QdrantClient::new(server.uri(), "memories".into(), None).unwrap()
}

/// The caller's view of the memory on (re-)store: fresh history, new
/// caller-supplied fields.
fn incoming() -> Memory {
    Memory {
        content: "same content".into(),
        content_hash: hash(),
        tags: vec!["new-tag".into()],
        memory_type: "note".into(),
        metadata: None,
        created_at: 2000.0,
        updated_at: 2000.0,
        embedding: Some(vec![0.1, 0.2]),
        summary: Some("new summary".into()),
        salience_score: 0.5,
        access_count: 0,
        access_timestamps: vec![],
        emotional_valence: None,
        encoding_context: None,
        provenance: None,
        summary_embedding: None,
    }
}

/// What Qdrant already holds under the same point id: written before
/// conditional writes, so it carries no revision yet.
fn existing_payload() -> Value {
    json!({
        "content": "same content",
        "content_hash": hash(),
        "tags": ["old-tag"],
        "memory_type": "note",
        "created_at": 1000.0,
        "updated_at": 1000.0,
        "salience_score": 0.5,
        "access_count": 5,
        "access_timestamps": [1001.0, 1002.0],
    })
}

fn with_rev(mut payload: Value, rev: &str) -> Value {
    payload["rev"] = json!(rev);
    payload["rev_log"] = json!([rev]);
    payload
}

fn payload_with_summary(summary: &str) -> Value {
    let mut payload = existing_payload();
    payload["summary"] = json!(summary);
    payload["summary_embedding"] = json!([0.5, 0.6]);
    payload
}

/// A fake holding `payload` under `ID` (nothing when `None`), mounted on a
/// fresh server.
async fn fake_with(payload: Option<Value>) -> (MockServer, FakeQdrant) {
    let server = MockServer::start().await;
    let fake = FakeQdrant::default();
    if let Some(p) = payload {
        fake.insert(ID, p);
    }
    fake.mount(&server).await;
    (server, fake)
}

/// Bodies of the upserts, in arrival order.
async fn upserts(server: &MockServer) -> Vec<Value> {
    writes(server)
        .await
        .into_iter()
        .filter(|(line, _)| line.starts_with(&format!("PUT {POINTS_PATH}?")))
        .map(|(_, body)| body)
        .collect()
}

fn supersede_update() -> MetadataUpdate {
    let mut extra = HashMap::new();
    extra.insert("supersession_reason".to_string(), json!("merged"));
    MetadataUpdate {
        superseded_by: Some("b".repeat(64)),
        extra: Some(extra),
        ..Default::default()
    }
}

// ─── Re-store semantics (alaya#86) ──────────────────────────────────────────

#[tokio::test]
async fn store_reports_created_true_when_point_absent() {
    let (server, fake) = fake_with(None).await;

    let (created, h) = client_for(&server)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("store succeeds");
    assert!(created, "first store must report created=true");
    assert_eq!(h, hash());

    let puts = upserts(&server).await;
    assert_eq!(puts.len(), 1, "{puts:?}");
    assert_eq!(
        puts[0]["update_mode"],
        json!("insert_only"),
        "new content is insert-only"
    );
    assert!(puts[0].get("update_filter").is_none(), "{}", puts[0]);

    let stored = fake.point(ID).expect("the point was inserted");
    assert_eq!(stored["created_at"], json!(2000.0));
    assert_eq!(stored["access_count"], json!(0));
    assert_eq!(stored["access_timestamps"], json!([]));
    assert_eq!(
        stored["rev_log"],
        json!([stored["rev"]]),
        "an insert starts the log"
    );
}

#[tokio::test]
async fn store_reports_created_false_and_preserves_history_when_point_exists() {
    let (server, fake) = fake_with(Some(existing_payload())).await;

    let (created, _) = client_for(&server)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("store succeeds");
    assert!(
        !created,
        "re-store of existing content must report created=false"
    );

    let puts = upserts(&server).await;
    assert_eq!(puts.len(), 1, "{puts:?}");
    assert_eq!(
        puts[0]["update_mode"],
        json!("update_only"),
        "a re-store never inserts"
    );
    assert_eq!(
        puts[0]["update_filter"],
        rev_filter(None),
        "a point with no revision yet is the zero revision"
    );

    let stored = fake.point(ID).unwrap();
    // Server-maintained history comes from the existing point...
    assert_eq!(stored["created_at"], json!(1000.0));
    assert_eq!(stored["access_count"], json!(5));
    assert_eq!(stored["access_timestamps"], json!([1001.0, 1002.0]));
    // ...while caller-supplied fields keep replace-on-store semantics.
    assert_eq!(stored["updated_at"], json!(2000.0));
    assert_eq!(stored["tags"], json!(["new-tag"]));
    assert_eq!(stored["summary"], json!("new summary"));
}

#[tokio::test]
async fn restore_is_conditional_on_the_revision_it_read() {
    let mut point = existing_payload();
    point["rev"] = json!("r1");
    point["rev_log"] = json!(["r0", "r1"]);
    let (server, fake) = fake_with(Some(point)).await;

    client_for(&server)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("store succeeds");

    assert_eq!(
        upserts(&server).await[0]["update_filter"],
        rev_filter(Some("r1"))
    );
    let stored = fake.point(ID).unwrap();
    let rev = stored["rev"].as_str().unwrap();
    assert_ne!(rev, "r1", "every write stamps a fresh revision");
    assert_eq!(stored["rev_log"], json!(["r0", "r1", rev]));
}

#[tokio::test]
async fn store_fails_closed_when_existence_check_fails() {
    for mode in [StoreMode::Upsert, StoreMode::InsertOnly] {
        let (server, fake) = fake_with(None).await;
        Mock::given(method("POST"))
            .and(path(POINTS_PATH))
            .respond_with(
                ResponseTemplate::new(500).set_body_json(json!({"status": {"error": "boom"}})),
            )
            .with_priority(1)
            .mount(&server)
            .await;

        let result = client_for(&server).store(&incoming(), mode).await;
        assert!(
            result.is_err(),
            "existence-check failure must propagate ({mode:?})"
        );
        assert!(
            write_order(&server).await.is_empty(),
            "no write may follow a failed existence check ({mode:?})"
        );
        assert!(fake.point(ID).is_none());
    }
}

/// The supersession marker is server-written (`mark_superseded`), so a
/// re-store must carry it over even though the caller's `metadata` otherwise
/// replaces the stored one: a re-store must never resurrect a superseded memory.
#[tokio::test]
async fn store_carries_supersession_marker_over_on_restore() {
    let mut superseded = existing_payload();
    superseded["metadata"] = json!({"superseded_by": "b".repeat(64), "other": 1});
    superseded["supersession_reason"] = json!("corrected");
    let (server, fake) = fake_with(Some(superseded)).await;

    let (created, _) = client_for(&server)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("store succeeds");
    assert!(!created);

    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["metadata"]["superseded_by"], json!("b".repeat(64)));
    assert_eq!(stored["supersession_reason"], json!("corrected"));
    assert!(
        stored["metadata"].get("other").is_none(),
        "caller-owned metadata keys are still replace-on-store: {stored}"
    );
}

/// Existence is decided on the raw point, not on whether it parses as a
/// `Memory`: a present-but-malformed point must not be overwritten as new.
#[tokio::test]
async fn store_treats_unparseable_existing_point_as_existing() {
    let (server, fake) = fake_with(Some(json!({ "created_at": 1000.0 }))).await;

    let (created, _) = client_for(&server)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("store succeeds");
    assert!(
        !created,
        "a point Qdrant returned exists, however malformed"
    );
    assert_eq!(
        upserts(&server).await[0]["update_mode"],
        json!("update_only")
    );
    assert_eq!(fake.point(ID).unwrap()["created_at"], json!(1000.0));
}

/// `summary_embedding` is derived from the summary text server-side. It must
/// survive a re-store that keeps the summary unchanged (no enrichment re-runs
/// when the caller supplies a summary) and must NOT survive when the summary
/// changes or is removed, so no stale vector describes the wrong summary.
#[tokio::test]
async fn store_keeps_summary_embedding_only_while_summary_unchanged() {
    // Unchanged: existing summary == incoming "new summary" → embedding kept.
    let (same, fake) = fake_with(Some(payload_with_summary("new summary"))).await;
    client_for(&same)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("store succeeds");
    assert_eq!(
        fake.point(ID).unwrap()["summary_embedding"],
        json!([0.5, 0.6])
    );

    // Changed: existing summary differs from the incoming one → dropped.
    let (changed, fake) = fake_with(Some(payload_with_summary("old summary"))).await;
    client_for(&changed)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("store succeeds");
    assert!(
        fake.point(ID).unwrap().get("summary_embedding").is_none(),
        "a changed summary must not keep the old summary's vector"
    );

    // Removed: incoming carries no summary at all → dropped.
    let (removed, fake) = fake_with(Some(payload_with_summary("new summary"))).await;
    let mut no_summary = incoming();
    no_summary.summary = None;
    client_for(&removed)
        .store(&no_summary, StoreMode::Upsert)
        .await
        .expect("store succeeds");
    let stored = fake.point(ID).unwrap();
    assert!(stored.get("summary").is_none());
    assert!(stored.get("summary_embedding").is_none());
}

/// `result: []` is the only legitimate "absent". A 2xx whose `result` is
/// missing or `null` is a protocol violation and must fail closed: no write,
/// in either mode (an insert-only store must not report `created: true`).
#[tokio::test]
async fn store_fails_closed_when_retrieve_has_no_result() {
    for body in [
        json!({"status": "ok"}),
        json!({"status": "ok", "result": null}),
    ] {
        for mode in [StoreMode::Upsert, StoreMode::InsertOnly] {
            let (server, _fake) = fake_with(Some(existing_payload())).await;
            Mock::given(method("POST"))
                .and(path(POINTS_PATH))
                .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
                .with_priority(1)
                .mount(&server)
                .await;

            let result = client_for(&server).store(&incoming(), mode).await;
            assert!(
                result.is_err(),
                "store must fail closed on {body} ({mode:?})"
            );
            assert!(
                write_order(&server).await.is_empty(),
                "no write may follow a malformed retrieve: {body} ({mode:?})"
            );
        }
    }
}

/// The same rule on the read-back: a malformed answer is a storage error, not
/// "the point is gone" — reading it as absent would report a landed store as
/// deleted (`Conflict`).
#[tokio::test]
async fn store_fails_closed_when_readback_has_no_result() {
    let (server, fake) = fake_with(Some(existing_payload())).await;
    Mock::given(method("POST"))
        .and(path(POINTS_PATH))
        .respond_with(fake.clone())
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(POINTS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .with_priority(2)
        .mount(&server)
        .await;

    let result = client_for(&server)
        .store(&incoming(), StoreMode::Upsert)
        .await;
    assert!(matches!(result, Err(AlayaError::Storage(_))), "{result:?}");
}

// ─── Insert-only stores ─────────────────────────────────────────────────────

/// On an existing point an insert-only store reports `created: false` and
/// writes nothing at all.
#[tokio::test]
async fn insert_only_leaves_an_existing_point_untouched() {
    let before = with_rev(existing_payload(), "r1");
    let (server, fake) = fake_with(Some(before.clone())).await;

    let (created, h) = client_for(&server)
        .store(&incoming(), StoreMode::InsertOnly)
        .await
        .expect("store succeeds");
    assert!(!created);
    assert_eq!(h, hash());
    assert!(
        write_order(&server).await.is_empty(),
        "nothing may be written"
    );
    assert_eq!(fake.point(ID).unwrap(), before);
}

/// Presence is judged exactly as the upsert path judges it: raw point, not
/// parseability.
#[tokio::test]
async fn insert_only_judges_raw_presence_not_parseability() {
    let malformed = json!({ "created_at": 1000.0 });
    let (present, fake) = fake_with(Some(malformed.clone())).await;
    let (created, _) = client_for(&present)
        .store(&incoming(), StoreMode::InsertOnly)
        .await
        .expect("store succeeds");
    assert!(!created, "an unparseable point still exists");
    assert!(write_order(&present).await.is_empty());
    assert_eq!(fake.point(ID).unwrap(), malformed);

    let (absent, fake) = fake_with(None).await;
    let (created, _) = client_for(&absent)
        .store(&incoming(), StoreMode::InsertOnly)
        .await
        .expect("store succeeds");
    assert!(created);
    let puts = upserts(&absent).await;
    assert_eq!(puts.len(), 1);
    assert_eq!(puts[0]["update_mode"], json!("insert_only"));
    assert_eq!(fake.point(ID).unwrap()["tags"], json!(["new-tag"]));
}

// ─── Summary / embedding pair ───────────────────────────────────────────────

/// The summary/embedding pair must stay consistent at EVERY write site. A
/// summary-only patch (the documented REST shape, and enrichment's path when
/// embedding fails) must remove the old vector in the SAME write as the new
/// summary — no separate delete that could leave either half behind — so a
/// later re-store has no stale pair to preserve.
#[tokio::test]
async fn summary_only_patch_invalidates_embedding_so_restore_cannot_carry_it() {
    let (server, fake) = fake_with(Some(payload_with_summary("summary A"))).await;
    let client = client_for(&server);

    let patch = PatchMemoryRequest {
        summary: Some("summary B".into()),
        ..Default::default()
    };
    let patched = client
        .patch_memory(&hash(), &patch)
        .await
        .expect("patch succeeds");
    assert_eq!(patched.summary.as_deref(), Some("summary B"));
    assert!(patched.summary_embedding.is_none());

    let w = writes(&server).await;
    assert_eq!(w.len(), 1, "one write replaces the payload: {w:?}");
    assert_eq!(
        w[0].0,
        format!("PUT {PAYLOAD_PATH}?wait=true"),
        "overwrite, not merge"
    );
    assert!(is_conditional(&w[0].0, &w[0].1), "{w:?}");
    assert_eq!(w[0].1["payload"]["summary"], json!("summary B"));
    assert!(w[0].1["payload"].get("summary_embedding").is_none());

    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["summary"], json!("summary B"));
    assert!(stored.get("summary_embedding").is_none(), "{stored}");
    assert_eq!(stored["tags"], json!(["old-tag"]), "the rest is kept");
    assert_eq!(stored["access_count"], json!(5));

    // Re-store with summary B against the patched record: nothing to carry.
    let mut incoming_b = incoming();
    incoming_b.summary = Some("summary B".into());
    client
        .store(&incoming_b, StoreMode::Upsert)
        .await
        .expect("store succeeds");
    assert!(
        fake.point(ID).unwrap().get("summary_embedding").is_none(),
        "no embedding may reappear on re-store after a summary-only patch"
    );
    assert!(
        write_order(&server)
            .await
            .iter()
            .all(|l| !l.contains(PAYLOAD_DELETE_PATH)),
        "no payload-key delete is ever sent"
    );
}

/// Patching the summary to its current value, or supplying a replacement
/// embedding alongside a new summary, must not discard a valid embedding.
#[tokio::test]
async fn patch_keeps_embedding_when_summary_unchanged_or_replaced() {
    let (server, fake) = fake_with(Some(payload_with_summary("summary A"))).await;
    let client = client_for(&server);

    client
        .patch_memory(
            &hash(),
            &PatchMemoryRequest {
                summary: Some("summary A".into()),
                ..Default::default()
            },
        )
        .await
        .expect("unchanged summary patch succeeds");
    assert_eq!(
        fake.point(ID).unwrap()["summary_embedding"],
        json!([0.5, 0.6])
    );

    client
        .patch_memory(
            &hash(),
            &PatchMemoryRequest {
                summary: Some("summary B".into()),
                summary_embedding: Some(vec![0.75, 0.25]),
                ..Default::default()
            },
        )
        .await
        .expect("summary + embedding patch succeeds");
    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["summary"], json!("summary B"));
    assert_eq!(stored["summary_embedding"], json!([0.75, 0.25]));

    assert!(
        write_order(&server)
            .await
            .iter()
            .all(|l| !l.contains(PAYLOAD_DELETE_PATH)),
        "no payload-key delete is ever sent"
    );
}

/// `set_generated_summary` decides on the copy its write is conditional on
/// whether the record still lacks a summary: a caller summary that landed
/// first always wins.
#[tokio::test]
async fn generated_summary_commits_only_while_summary_is_absent() {
    let (absent, fake) = fake_with(Some(existing_payload())).await;
    let applied = client_for(&absent)
        .set_generated_summary(&hash(), "generated A", Some(vec![0.5, 0.75]))
        .await
        .expect("commit succeeds");
    assert!(applied);
    let w = writes(&absent).await;
    assert_eq!(w.len(), 1, "exactly one write: {w:?}");
    assert!(is_conditional(&w[0].0, &w[0].1), "{w:?}");
    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["summary"], json!("generated A"));
    assert_eq!(stored["summary_embedding"], json!([0.5, 0.75]));

    let caller = payload_with_summary("caller B");
    let (present, fake) = fake_with(Some(caller.clone())).await;
    let applied = client_for(&present)
        .set_generated_summary(&hash(), "generated A", Some(vec![0.5, 0.75]))
        .await
        .expect("no-op succeeds");
    assert!(!applied, "a caller summary that landed first must win");
    assert!(
        write_order(&present).await.is_empty(),
        "nothing may be written over a caller summary"
    );
    assert_eq!(fake.point(ID).unwrap(), caller);
}

// ─── One client's writers are serialised (write_lock) ───────────────────────
//
// Each race parks the FIRST writer on its read: the fake answers that retrieve
// 400 ms late, so the writer holds a snapshot that goes stale while a second
// writer, started once that read has arrived, would land inside its
// read→write window.
// Compare-and-set alone would keep the final state whole even then (the first
// writer loses its write and retries); the client's write lock additionally
// makes the second writer wait, so the write order below is forced and no
// retry is spent on an in-process race.

async fn parked_first_reader(payload: Value) -> (MockServer, FakeQdrant) {
    let (server, fake) = fake_with(Some(payload)).await;
    fake.delay_next(&server, "POST", POINTS_PATH, Duration::from_millis(400))
        .await;
    (server, fake)
}

/// Run `f` once the first writer's read has reached the server, so the
/// second writer never takes the parked read instead, however slow the host.
async fn once_first_read_arrived<T>(
    server: &MockServer,
    f: impl std::future::Future<Output = T>,
) -> T {
    while server.received_requests().await.unwrap().is_empty() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    f.await
}

/// A summary-only patch (A→B) is parked when an enrichment-style patch
/// (A + vector(A)) arrives from another task. Each lands whole, in order: the
/// record never ends up as summary B + vector(A).
#[tokio::test]
async fn concurrent_patch_cannot_interleave_with_another_patch() {
    let (server, fake) = parked_first_reader(payload_with_summary("summary A")).await;
    let client = client_for(&server);
    let hash = hash();

    let to_b = PatchMemoryRequest {
        summary: Some("summary B".into()),
        ..Default::default()
    };
    let enrichment = PatchMemoryRequest {
        summary: Some("summary A".into()),
        summary_embedding: Some(vec![0.5, 0.75]),
        ..Default::default()
    };
    let (a, b) = tokio::join!(
        client.patch_memory(&hash, &to_b),
        once_first_read_arrived(&server, client.patch_memory(&hash, &enrichment))
    );
    a.expect("user patch succeeds");
    b.expect("enrichment patch succeeds");

    let w = writes(&server).await;
    let order: Vec<&str> = w.iter().map(|(l, _)| l.as_str()).collect();
    let put = format!("PUT {PAYLOAD_PATH}?wait=true");
    assert_eq!(
        order,
        [put.as_str(), put.as_str()],
        "one write each, in order"
    );
    assert_eq!(w[0].1["payload"]["summary"], json!("summary B"));
    assert!(w[0].1["payload"].get("summary_embedding").is_none());
    assert_eq!(w[1].1["payload"]["summary_embedding"], json!([0.5, 0.75]));

    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["summary"], json!("summary A"));
    assert_eq!(stored["summary_embedding"], json!([0.5, 0.75]));
    assert_eq!(stored["tags"], json!(["old-tag"]));
}

/// Same window, different second writer: a re-store that read the point with
/// vector(A) must not put that vector back after the patch removed it.
#[tokio::test]
async fn concurrent_store_cannot_interleave_with_a_patch() {
    let (server, fake) = parked_first_reader(payload_with_summary("summary A")).await;
    let client = client_for(&server);
    let hash = hash();

    let to_b = PatchMemoryRequest {
        summary: Some("summary B".into()),
        ..Default::default()
    };
    let mut same_summary = incoming();
    same_summary.summary = Some("summary A".into());
    let (a, b) = tokio::join!(
        client.patch_memory(&hash, &to_b),
        once_first_read_arrived(&server, client.store(&same_summary, StoreMode::Upsert))
    );
    a.expect("user patch succeeds");
    b.expect("store succeeds");

    assert_eq!(
        write_order(&server).await,
        [
            format!("PUT {PAYLOAD_PATH}?wait=true"),
            format!("PUT {POINTS_PATH}?wait=true"),
        ],
        "the store runs after the patch, whole"
    );
    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["summary"], json!("summary A"));
    assert!(
        stored.get("summary_embedding").is_none(),
        "the store read summary B, so it has no vector to keep: {stored}"
    );
    assert_eq!(stored["tags"], json!(["new-tag"]));
    assert_eq!(stored["access_count"], json!(5), "history carried");
}

/// A spawned duplicate-merge supersedes the memory while a re-store of it is
/// parked; the re-store's snapshot must not wipe the marker or the reason.
#[tokio::test]
async fn concurrent_supersession_cannot_land_inside_a_restore() {
    let (server, fake) = parked_first_reader(existing_payload()).await;
    let client = client_for(&server);
    let hash = hash();
    let hashes = [hash.as_str()];
    let mem = incoming();

    let (a, b) = tokio::join!(
        client.store(&mem, StoreMode::Upsert),
        once_first_read_arrived(
            &server,
            client.update_metadata_batch(&hashes, supersede_update())
        )
    );
    a.expect("store succeeds");
    b.expect("supersede succeeds");

    assert_eq!(
        write_order(&server).await,
        [
            format!("PUT {POINTS_PATH}?wait=true"),
            format!("POST {PAYLOAD_PATH}?wait=true"),
        ],
        "the supersession runs after the re-store, as one write"
    );
    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["metadata"]["superseded_by"], json!("b".repeat(64)));
    assert_eq!(stored["supersession_reason"], json!("merged"));
    assert_eq!(
        stored["tags"],
        json!(["new-tag"]),
        "the re-store landed too"
    );
    assert_eq!(stored["created_at"], json!(1000.0));
}

#[tokio::test]
async fn concurrent_access_increment_cannot_land_inside_a_restore() {
    let (server, fake) = parked_first_reader(existing_payload()).await;
    let client = client_for(&server);
    let hash = hash();
    let mem = incoming();

    let (a, b) = tokio::join!(
        client.store(&mem, StoreMode::Upsert),
        once_first_read_arrived(&server, client.increment_access_count(&hash))
    );
    a.expect("store succeeds");
    b.expect("increment succeeds");

    assert_eq!(
        write_order(&server).await,
        [
            format!("PUT {POINTS_PATH}?wait=true"),
            format!("POST {PAYLOAD_PATH}?wait=true"),
        ],
        "an access increment must not be rolled back by a store snapshot"
    );
    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["access_count"], json!(6));
    assert_eq!(stored["access_timestamps"].as_array().unwrap().len(), 3);
    assert_eq!(stored["tags"], json!(["new-tag"]));
}

#[tokio::test]
async fn concurrent_batch_access_increment_cannot_land_inside_a_restore() {
    let (server, fake) = parked_first_reader(existing_payload()).await;
    let client = client_for(&server);
    let hash = hash();
    let hashes = [hash.as_str()];
    let mem = incoming();

    let (a, b) = tokio::join!(
        client.store(&mem, StoreMode::Upsert),
        once_first_read_arrived(&server, client.increment_access_count_batch(&hashes))
    );
    a.expect("store succeeds");
    b.expect("batch increment succeeds");

    assert_eq!(
        write_order(&server).await,
        [
            format!("PUT {POINTS_PATH}?wait=true"),
            format!("POST {PAYLOAD_PATH}?wait=true"),
        ],
        "a batch access increment must not be rolled back by a store snapshot"
    );
    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["access_count"], json!(6));
    assert_eq!(stored["access_timestamps"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn concurrent_delete_cannot_land_inside_a_restore() {
    let (server, fake) = parked_first_reader(existing_payload()).await;
    let client = client_for(&server);
    let hash = hash();
    let mem = incoming();

    let (a, b) = tokio::join!(
        client.store(&mem, StoreMode::Upsert),
        once_first_read_arrived(&server, client.delete(&hash))
    );
    let (created, _) = a.expect("store succeeds");
    assert!(!created);
    b.expect("delete succeeds");

    assert_eq!(
        write_order(&server).await,
        [
            format!("PUT {POINTS_PATH}?wait=true"),
            format!("POST {POINTS_DELETE_PATH}?wait=true"),
        ],
        "the delete runs after the re-store"
    );
    assert!(fake.point(ID).is_none(), "the point stays deleted");
}

// ─── Lost races (another process writes inside the window) ──────────────────

/// Another process supersedes and counts the memory between this re-store's
/// read and its write. The write is rejected, re-read and re-sent on the new
/// revision: both effects survive.
#[tokio::test]
async fn store_retries_a_lost_race_and_keeps_the_other_writers_effect() {
    let (server, fake) = fake_with(Some(with_rev(existing_payload(), "r1"))).await;
    fake.before_writes(1, |points| {
        external_write(
            points,
            ID,
            json!({
                "access_count": 9,
                "supersession_reason": "merged",
                "metadata": {"superseded_by": "b".repeat(64)},
            }),
        );
    });

    let (created, _) = client_for(&server)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("the store lands on its second round");
    assert!(!created);

    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["tags"], json!(["new-tag"]), "this store landed");
    assert_eq!(
        stored["access_count"],
        json!(9),
        "the other writer's count kept"
    );
    assert_eq!(stored["metadata"]["superseded_by"], json!("b".repeat(64)));
    assert_eq!(stored["supersession_reason"], json!("merged"));

    let log = stored["rev_log"].as_array().unwrap();
    assert_eq!(log.len(), 3, "r1, the other writer, this store: {log:?}");
    let puts = upserts(&server).await;
    assert_eq!(puts.len(), 2, "{puts:?}");
    assert_eq!(puts[0]["update_filter"], rev_filter(Some("r1")));
    assert_eq!(puts[1]["update_filter"], rev_filter(log[1].as_str()));
}

/// Another process inserts the same content first: the insert-only write is
/// rejected and the store becomes a re-store over that point.
#[tokio::test]
async fn store_that_loses_the_insert_race_becomes_a_restore() {
    let (server, fake) = fake_with(None).await;
    fake.before_writes(1, |points| {
        external_write(points, ID, existing_payload());
    });

    let (created, _) = client_for(&server)
        .store(&incoming(), StoreMode::Upsert)
        .await
        .expect("store succeeds");
    assert!(!created, "the other process created it");

    let puts = upserts(&server).await;
    assert_eq!(puts.len(), 2, "{puts:?}");
    assert_eq!(puts[0]["update_mode"], json!("insert_only"));
    assert_eq!(puts[1]["update_mode"], json!("update_only"));
    let stored = fake.point(ID).unwrap();
    assert_eq!(
        stored["created_at"],
        json!(1000.0),
        "carried from the winner"
    );
    assert_eq!(stored["access_count"], json!(5));
    assert_eq!(stored["tags"], json!(["new-tag"]));
}

/// An insert-only store that loses the insert reports `created: false` and
/// writes nothing more.
#[tokio::test]
async fn insert_only_store_that_loses_the_insert_race_writes_nothing_more() {
    let (server, fake) = fake_with(None).await;
    fake.before_writes(1, |points| {
        external_write(points, ID, existing_payload());
    });

    let (created, _) = client_for(&server)
        .store(&incoming(), StoreMode::InsertOnly)
        .await
        .expect("store succeeds");
    assert!(!created);
    assert_eq!(upserts(&server).await.len(), 1, "only the rejected insert");
    let stored = fake.point(ID).unwrap();
    assert_eq!(
        stored["tags"],
        json!(["old-tag"]),
        "the winner's point untouched"
    );
}

/// Two processes counting the same hit record two accesses, not one.
#[tokio::test]
async fn increment_retries_a_lost_race_and_counts_both_hits() {
    let (server, fake) = fake_with(Some(existing_payload())).await;
    fake.before_writes(1, |points| {
        external_write(
            points,
            ID,
            json!({"access_count": 6, "access_timestamps": [1001.0, 1002.0, 1003.0]}),
        );
    });

    client_for(&server)
        .increment_access_count(&hash())
        .await
        .expect("increment succeeds");

    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["access_count"], json!(7));
    assert_eq!(stored["access_timestamps"].as_array().unwrap().len(), 4);
    let w = writes(&server).await;
    assert_eq!(w.len(), 2, "{w:?}");
    assert_eq!(
        w[0].1["filter"]["must"][1],
        json!({"is_empty": {"key": "rev"}})
    );
    assert_eq!(
        w[1].1["filter"]["must"][1]["match"]["value"],
        stored["rev_log"][0]
    );
}

/// A patch replaces the whole payload, so a stale copy would roll back
/// anything written since it was read. It loses the race instead and
/// rebuilds from the fresh copy.
#[tokio::test]
async fn patch_retries_a_lost_race_without_rolling_the_other_write_back() {
    let (server, fake) = fake_with(Some(existing_payload())).await;
    fake.before_writes(1, |points| {
        external_write(
            points,
            ID,
            json!({"access_count": 6, "tags": ["their-tag"]}),
        );
    });

    let patched = client_for(&server)
        .patch_memory(
            &hash(),
            &PatchMemoryRequest {
                summary: Some("patched".into()),
                ..Default::default()
            },
        )
        .await
        .expect("patch succeeds");
    assert_eq!(patched.access_count, 6);

    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["summary"], json!("patched"));
    assert_eq!(stored["access_count"], json!(6));
    assert_eq!(stored["tags"], json!(["their-tag"]));
    assert_eq!(writes(&server).await.len(), 2);
}

#[derive(Debug, Clone, Copy)]
enum Writer {
    Store,
    Insert,
    Patch,
    GeneratedSummary,
    Supersede,
    Increment,
    BatchIncrement,
}

const WRITERS: [Writer; 7] = [
    Writer::Store,
    Writer::Insert,
    Writer::Patch,
    Writer::GeneratedSummary,
    Writer::Supersede,
    Writer::Increment,
    Writer::BatchIncrement,
];

/// Run `writer` against the point at `ID`; `Insert` stores new content.
async fn run(client: &QdrantClient, writer: Writer) -> Result<(), AlayaError> {
    let hash = hash();
    let patch = PatchMemoryRequest {
        summary: Some("patched".into()),
        ..Default::default()
    };
    match writer {
        Writer::Store | Writer::Insert => {
            client.store(&incoming(), StoreMode::Upsert).await.map(drop)
        }
        Writer::Patch => client.patch_memory(&hash, &patch).await.map(drop),
        Writer::GeneratedSummary => client
            .set_generated_summary(&hash, "generated", None)
            .await
            .map(drop),
        Writer::Supersede => client.update_metadata(&hash, supersede_update()).await,
        Writer::Increment => client.increment_access_count(&hash).await,
        Writer::BatchIncrement => client.increment_access_count_batch(&[hash.as_str()]).await,
    }
}

/// A writer that loses EVERY round gives up after exactly 8 attempts with a
/// storage error, and never falls back to a write that could overwrite the
/// winner: each one is insert-only or conditional on the revision it read.
/// (The batch increment is fire-and-forget after a search: it gives up the
/// same way but reports success.)
#[tokio::test]
async fn writer_that_loses_every_race_gives_up_after_eight_conditional_attempts() {
    for writer in WRITERS {
        let existing = match writer {
            Writer::Insert => None,
            _ => Some(existing_payload()),
        };
        let (server, fake) = fake_with(existing).await;
        fake.before_writes(usize::MAX, |points| {
            external_write(points, ID, json!({}));
        });

        let result = run(&client_for(&server), writer).await;
        match writer {
            Writer::BatchIncrement => assert!(result.is_ok(), "{writer:?}: {result:?}"),
            _ => assert!(
                matches!(result, Err(AlayaError::Storage(ref m)) if m.contains("lost 8")),
                "{writer:?}: {result:?}"
            ),
        }
        let w = writes(&server).await;
        assert_eq!(w.len(), 8, "{writer:?}: exactly 8 attempts, got {w:?}");
        for (line, body) in &w {
            assert!(is_conditional(line, body), "{writer:?} sent {line} {body}");
        }
    }
}

/// A delete landing between a writer's read and its conditional write: the
/// write is rejected and the point stays deleted. A store says so
/// (`Conflict` — nothing of it survived); a patch or generated summary of a
/// memory that no longer exists is `NotFound`; a supersession skips it (a
/// deleted memory needs no marker, and the rest of a batch has committed, so
/// its caller must go on to write the graph edges); an access count on
/// nothing is a no-op.
#[tokio::test]
async fn delete_inside_a_writers_window_is_never_undone() {
    for writer in WRITERS {
        if matches!(writer, Writer::Insert) {
            continue;
        }
        let (server, fake) = fake_with(Some(existing_payload())).await;
        fake.before_writes(1, |points| {
            points.remove(ID);
        });

        let result = run(&client_for(&server), writer).await;
        match writer {
            Writer::Store => assert!(matches!(result, Err(AlayaError::Conflict(_))), "{result:?}"),
            Writer::Patch | Writer::GeneratedSummary => {
                assert!(
                    matches!(result, Err(AlayaError::NotFound(_))),
                    "{writer:?}: {result:?}"
                )
            }
            _ => assert!(result.is_ok(), "{writer:?}: {result:?}"),
        }
        assert!(fake.point(ID).is_none(), "{writer:?} resurrected the point");
        assert_eq!(writes(&server).await.len(), 1, "{writer:?}: no retry");
    }
}

/// More writes than the revision log holds land inside the window, so the
/// read-back cannot tell whether this write applied: that is an error, not a
/// guess either way — for a point read at a revision and for one read with
/// none.
#[tokio::test]
async fn unconfirmable_write_is_a_storage_error() {
    for (writer, existing) in [
        (Writer::Store, with_rev(existing_payload(), "r1")),
        (Writer::Patch, existing_payload()),
    ] {
        let (server, fake) = fake_with(Some(existing)).await;
        fake.before_writes(1, |points| {
            for _ in 0..REV_LOG_LEN {
                external_write(points, ID, json!({}));
            }
        });

        let result = run(&client_for(&server), writer).await;
        assert!(
            matches!(result, Err(AlayaError::Storage(ref m)) if m.contains("could not confirm")),
            "{writer:?}: {result:?}"
        );
        assert_eq!(writes(&server).await.len(), 1, "{writer:?}: no retry");
    }
}

// ─── Two clients (two processes) racing for real ────────────────────────────
//
// Separate clients hold separate write locks, so nothing in-process orders
// them — the multi-process case alaya#130 is about. Client A is parked on its
// read while client B completes a whole write; only compare-and-set keeps A's
// stale snapshot from rolling B back.

#[tokio::test]
async fn two_clients_racing_a_restore_and_a_supersession_lose_nothing() {
    let (server, fake) = parked_first_reader(existing_payload()).await;
    let (a, b) = (client_for(&server), client_for(&server));
    let hash = hash();
    let mem = incoming();

    let (ra, rb) = tokio::join!(
        a.store(&mem, StoreMode::Upsert),
        once_first_read_arrived(&server, b.update_metadata(&hash, supersede_update()))
    );
    ra.expect("A's store succeeds");
    rb.expect("B's supersession succeeds");

    assert_eq!(
        write_order(&server).await,
        [
            format!("POST {PAYLOAD_PATH}?wait=true"),
            format!("PUT {POINTS_PATH}?wait=true"),
            format!("PUT {POINTS_PATH}?wait=true"),
        ],
        "B lands, A's stale write is rejected, A retries"
    );
    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["metadata"]["superseded_by"], json!("b".repeat(64)));
    assert_eq!(stored["supersession_reason"], json!("merged"));
    assert_eq!(stored["tags"], json!(["new-tag"]));
}

#[tokio::test]
async fn two_clients_counting_the_same_hit_record_two_accesses() {
    let (server, fake) = parked_first_reader(existing_payload()).await;
    let (a, b) = (client_for(&server), client_for(&server));
    let hash = hash();

    let (ra, rb) = tokio::join!(
        a.increment_access_count(&hash),
        once_first_read_arrived(&server, b.increment_access_count(&hash))
    );
    ra.expect("A's increment succeeds");
    rb.expect("B's increment succeeds");

    let stored = fake.point(ID).unwrap();
    assert_eq!(stored["access_count"], json!(7));
    assert_eq!(stored["access_timestamps"].as_array().unwrap().len(), 4);
    assert_eq!(writes(&server).await.len(), 3, "B, A rejected, A again");
}
