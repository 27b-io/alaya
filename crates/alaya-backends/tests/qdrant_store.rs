//! Regression tests for `store` re-store semantics (alaya#86).
//!
//! Qdrant's upsert replaces a point's payload wholesale and the point id is
//! derived from `content_hash`, so storing already-present content used to
//! zero the record's `created_at` / `access_count` / `access_timestamps` and
//! still report `created: true`. `store` now retrieves the point first,
//! carries that history over, and reports `created: false`.

use alaya_backends::{VectorStorage, qdrant::QdrantClient};
use alaya_types::memory::{Memory, PatchMemoryRequest};
use serde_json::{Value, json};
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const POINTS_PATH: &str = "/collections/memories/points";
const PAYLOAD_PATH: &str = "/collections/memories/points/payload";
const PAYLOAD_DELETE_PATH: &str = "/collections/memories/points/payload/delete";

fn client_for(server: &MockServer) -> QdrantClient {
    QdrantClient::new(server.uri(), "memories".into(), None)
}

/// The caller's view of the memory on (re-)store: fresh history, new
/// caller-supplied fields.
fn incoming() -> Memory {
    Memory {
        content: "same content".into(),
        content_hash: "a".repeat(64),
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

/// What Qdrant already holds under the same point id.
fn existing_point() -> Value {
    json!({
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "payload": {
            "content": "same content",
            "content_hash": "a".repeat(64),
            "tags": ["old-tag"],
            "memory_type": "note",
            "created_at": 1000.0,
            "updated_at": 1000.0,
            "salience_score": 0.5,
            "access_count": 5,
            "access_timestamps": [1001.0, 1002.0],
        }
    })
}

async fn mount_retrieve(server: &MockServer, points: Vec<Value>) {
    Mock::given(method("POST"))
        .and(path(POINTS_PATH))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"status": "ok", "result": points})),
        )
        .mount(server)
        .await;
}

async fn mount_upsert_ok(server: &MockServer) {
    Mock::given(method("PUT"))
        .and(path(POINTS_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"status": "ok", "result": {"status": "completed"}})),
        )
        .mount(server)
        .await;
}

async fn upsert_payload(server: &MockServer) -> Value {
    let put = server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .find(|r| r.method == "PUT")
        .expect("an upsert was sent");
    let body: Value = serde_json::from_slice(&put.body).expect("upsert body is JSON");
    body["points"][0]["payload"].clone()
}

#[tokio::test]
async fn store_reports_created_true_when_point_absent() {
    let server = MockServer::start().await;
    mount_retrieve(&server, vec![]).await;
    mount_upsert_ok(&server).await;

    let (created, hash) = client_for(&server)
        .store(&incoming())
        .await
        .expect("store succeeds");
    assert!(created, "first store must report created=true");
    assert_eq!(hash, "a".repeat(64));

    let payload = upsert_payload(&server).await;
    assert_eq!(payload["created_at"], json!(2000.0));
    assert_eq!(payload["access_count"], json!(0));
    assert_eq!(payload["access_timestamps"], json!([]));
}

#[tokio::test]
async fn store_reports_created_false_and_preserves_history_when_point_exists() {
    let server = MockServer::start().await;
    mount_retrieve(&server, vec![existing_point()]).await;
    mount_upsert_ok(&server).await;

    let (created, _) = client_for(&server)
        .store(&incoming())
        .await
        .expect("store succeeds");
    assert!(
        !created,
        "re-store of existing content must report created=false"
    );

    let payload = upsert_payload(&server).await;
    // Server-maintained history comes from the existing point...
    assert_eq!(payload["created_at"], json!(1000.0));
    assert_eq!(payload["access_count"], json!(5));
    assert_eq!(payload["access_timestamps"], json!([1001.0, 1002.0]));
    // ...while caller-supplied fields keep replace-on-store semantics.
    assert_eq!(payload["updated_at"], json!(2000.0));
    assert_eq!(payload["tags"], json!(["new-tag"]));
    assert_eq!(payload["summary"], json!("new summary"));
}

#[tokio::test]
async fn store_fails_closed_when_existence_check_fails() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(POINTS_PATH))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"status": {"error": "boom"}})),
        )
        .mount(&server)
        .await;
    mount_upsert_ok(&server).await;

    let result = client_for(&server).store(&incoming()).await;
    assert!(result.is_err(), "existence-check failure must propagate");

    let requests = server.received_requests().await.unwrap();
    assert!(
        requests.iter().all(|r| r.method != "PUT"),
        "no blind upsert may follow a failed existence check"
    );
}

/// The supersession marker is server-written (`mark_superseded`), so a
/// re-store must carry it over even though the caller's `metadata` otherwise
/// replaces the stored one: a re-store must never resurrect a superseded memory.
#[tokio::test]
async fn store_carries_supersession_marker_over_on_restore() {
    let server = MockServer::start().await;
    let mut superseded = existing_point();
    superseded["payload"]["metadata"] = json!({"superseded_by": "b".repeat(64), "other": 1});
    superseded["payload"]["supersession_reason"] = json!("corrected");
    mount_retrieve(&server, vec![superseded]).await;
    mount_upsert_ok(&server).await;

    let (created, _) = client_for(&server)
        .store(&incoming())
        .await
        .expect("store succeeds");
    assert!(!created);

    let payload = upsert_payload(&server).await;
    assert_eq!(payload["metadata"]["superseded_by"], json!("b".repeat(64)));
    assert_eq!(payload["supersession_reason"], json!("corrected"));
    assert!(
        payload["metadata"].get("other").is_none(),
        "caller-owned metadata keys are still replace-on-store: {payload}"
    );
}

/// Existence is decided on the raw point, not on whether it parses as a
/// `Memory`: a present-but-malformed point must not be blindly overwritten.
#[tokio::test]
async fn store_treats_unparseable_existing_point_as_existing() {
    let server = MockServer::start().await;
    let malformed = json!({
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "payload": { "created_at": 1000.0 }
    });
    mount_retrieve(&server, vec![malformed]).await;
    mount_upsert_ok(&server).await;

    let (created, _) = client_for(&server)
        .store(&incoming())
        .await
        .expect("store succeeds");
    assert!(
        !created,
        "a point Qdrant returned exists, however malformed"
    );
    assert_eq!(upsert_payload(&server).await["created_at"], json!(1000.0));
}

/// `exists` must answer from the same raw-point source `store` reads, so the
/// read-only guard in alaya-core can never disagree with the write.
#[tokio::test]
async fn exists_judges_raw_presence_not_parseability() {
    let present = MockServer::start().await;
    mount_retrieve(
        &present,
        vec![json!({
            "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "payload": { "created_at": 1000.0 }
        })],
    )
    .await;
    assert!(
        client_for(&present).exists(&"a".repeat(64)).await.unwrap(),
        "an unparseable point still exists"
    );

    let absent = MockServer::start().await;
    mount_retrieve(&absent, vec![]).await;
    assert!(!client_for(&absent).exists(&"a".repeat(64)).await.unwrap());
}

/// `summary_embedding` is derived from the summary text server-side. It must
/// survive a re-store that keeps the summary unchanged (no enrichment re-runs
/// when the caller supplies a summary) and must NOT survive when the summary
/// changes or is removed, so no stale vector describes the wrong summary.
#[tokio::test]
async fn store_keeps_summary_embedding_only_while_summary_unchanged() {
    // Unchanged: existing summary == incoming "new summary" → embedding kept.
    let same = MockServer::start().await;
    let mut point = existing_point();
    point["payload"]["summary"] = json!("new summary");
    point["payload"]["summary_embedding"] = json!([0.5, 0.6]);
    mount_retrieve(&same, vec![point.clone()]).await;
    mount_upsert_ok(&same).await;
    client_for(&same)
        .store(&incoming())
        .await
        .expect("store succeeds");
    assert_eq!(
        upsert_payload(&same).await["summary_embedding"],
        json!([0.5, 0.6])
    );

    // Changed: existing summary differs from the incoming one → dropped.
    let changed = MockServer::start().await;
    point["payload"]["summary"] = json!("old summary");
    mount_retrieve(&changed, vec![point.clone()]).await;
    mount_upsert_ok(&changed).await;
    client_for(&changed)
        .store(&incoming())
        .await
        .expect("store succeeds");
    assert!(
        upsert_payload(&changed)
            .await
            .get("summary_embedding")
            .is_none(),
        "a changed summary must not keep the old summary's vector"
    );

    // Removed: incoming carries no summary at all → dropped.
    let removed = MockServer::start().await;
    point["payload"]["summary"] = json!("new summary");
    mount_retrieve(&removed, vec![point]).await;
    mount_upsert_ok(&removed).await;
    let mut no_summary = incoming();
    no_summary.summary = None;
    client_for(&removed)
        .store(&no_summary)
        .await
        .expect("store succeeds");
    let payload = upsert_payload(&removed).await;
    assert!(payload.get("summary").is_none());
    assert!(payload.get("summary_embedding").is_none());
}

async fn mount_ok(server: &MockServer, p: &str) {
    Mock::given(method("POST"))
        .and(path(p))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"status": "ok", "result": {"status": "completed"}})),
        )
        .mount(server)
        .await;
}

fn point_with_summary(summary: &str) -> Value {
    let mut point = existing_point();
    point["payload"]["summary"] = json!(summary);
    point["payload"]["summary_embedding"] = json!([0.5, 0.6]);
    point
}

/// The summary/embedding pair must stay consistent at EVERY write site. A
/// summary-only patch (the documented REST shape, and enrichment's path when
/// embedding fails) must remove the old vector, delete before set, so a later
/// re-store has no stale pair to preserve.
#[tokio::test]
async fn summary_only_patch_invalidates_embedding_so_restore_cannot_carry_it() {
    let server = MockServer::start().await;
    mount_retrieve(&server, vec![point_with_summary("summary A")]).await;
    mount_ok(&server, PAYLOAD_DELETE_PATH).await;
    mount_ok(&server, PAYLOAD_PATH).await;

    let patch = PatchMemoryRequest {
        summary: Some("summary B".into()),
        ..Default::default()
    };
    client_for(&server)
        .patch_memory(&"a".repeat(64), &patch)
        .await
        .expect("patch succeeds");

    let writes: Vec<(String, Value)> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.url.path() != POINTS_PATH)
        .map(|r| {
            (
                r.url.path().to_string(),
                serde_json::from_slice(&r.body).unwrap(),
            )
        })
        .collect();
    assert_eq!(writes.len(), 2, "expected delete + set, got {writes:?}");
    assert_eq!(
        writes[0].0, PAYLOAD_DELETE_PATH,
        "embedding delete must precede the summary write"
    );
    assert_eq!(writes[0].1["keys"], json!(["summary_embedding"]));
    assert_eq!(writes[1].1["payload"]["summary"], json!("summary B"));
    assert!(writes[1].1["payload"].get("summary_embedding").is_none());

    // Re-store with summary B against the patched record: nothing to carry.
    let restore = MockServer::start().await;
    let mut patched = existing_point();
    patched["payload"]["summary"] = json!("summary B");
    mount_retrieve(&restore, vec![patched]).await;
    mount_upsert_ok(&restore).await;
    let mut incoming_b = incoming();
    incoming_b.summary = Some("summary B".into());
    client_for(&restore)
        .store(&incoming_b)
        .await
        .expect("store succeeds");
    assert!(
        upsert_payload(&restore)
            .await
            .get("summary_embedding")
            .is_none(),
        "no embedding may reappear on re-store after a summary-only patch"
    );
}

/// Patching the summary to its current value, or supplying a replacement
/// embedding alongside a new summary, must not discard a valid embedding.
#[tokio::test]
async fn patch_keeps_embedding_when_summary_unchanged_or_replaced() {
    let server = MockServer::start().await;
    mount_retrieve(&server, vec![point_with_summary("summary A")]).await;
    mount_ok(&server, PAYLOAD_DELETE_PATH).await;
    mount_ok(&server, PAYLOAD_PATH).await;
    let client = client_for(&server);

    client
        .patch_memory(
            &"a".repeat(64),
            &PatchMemoryRequest {
                summary: Some("summary A".into()),
                ..Default::default()
            },
        )
        .await
        .expect("unchanged summary patch succeeds");
    client
        .patch_memory(
            &"a".repeat(64),
            &PatchMemoryRequest {
                summary: Some("summary B".into()),
                summary_embedding: Some(vec![0.7, 0.8]),
                ..Default::default()
            },
        )
        .await
        .expect("summary + embedding patch succeeds");

    let requests = server.received_requests().await.unwrap();
    assert!(
        requests.iter().all(|r| r.url.path() != PAYLOAD_DELETE_PATH),
        "no delete for an unchanged summary or a replaced embedding"
    );
}

/// Writes to the payload endpoints, in arrival order, as (path, body).
async fn payload_writes(server: &MockServer) -> Vec<(String, Value)> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.url.path() != POINTS_PATH)
        .map(|r| {
            (
                r.url.path().to_string(),
                serde_json::from_slice(&r.body).unwrap(),
            )
        })
        .collect()
}

/// Stage the race Helly R found: a summary-only patch (A→B) is mid-flight,
/// parked on its `summary_embedding` delete, when an enrichment-style patch
/// (A + vector(A)) arrives from another task. Unserialised, the second patch
/// lands between the first one's delete and set and the record ends up as
/// summary B + vector(A). The client's write lock forces the second patch to
/// wait, so the delete and its set stay adjacent and the last complete
/// writer wins whole. The delete delay only matters for the unlocked mutant;
/// with the lock the order is forced regardless of timing.
async fn race_server() -> MockServer {
    let server = MockServer::start().await;
    mount_retrieve(&server, vec![point_with_summary("summary A")]).await;
    Mock::given(method("POST"))
        .and(path(PAYLOAD_DELETE_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"status": "ok", "result": {"status": "completed"}}))
                .set_delay(Duration::from_millis(400)),
        )
        .mount(&server)
        .await;
    mount_ok(&server, PAYLOAD_PATH).await;
    mount_upsert_ok(&server).await;
    server
}

#[tokio::test]
async fn concurrent_patch_cannot_land_inside_another_patch_invalidation() {
    let server = race_server().await;
    let client = client_for(&server);
    let hash = "a".repeat(64);

    let to_b = PatchMemoryRequest {
        summary: Some("summary B".into()),
        ..Default::default()
    };
    let user_patch = client.patch_memory(&hash, &to_b);
    let enrichment = async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        client
            .patch_memory(
                &hash,
                &PatchMemoryRequest {
                    summary: Some("summary A".into()),
                    summary_embedding: Some(vec![0.5, 0.75]),
                    ..Default::default()
                },
            )
            .await
    };
    let (a, b) = tokio::join!(user_patch, enrichment);
    a.expect("user patch succeeds");
    b.expect("enrichment patch succeeds");

    let writes = payload_writes(&server).await;
    let paths: Vec<&str> = writes.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(
        paths,
        vec![PAYLOAD_DELETE_PATH, PAYLOAD_PATH, PAYLOAD_PATH],
        "the set that completes the invalidation must directly follow its delete"
    );
    assert_eq!(writes[1].1["payload"]["summary"], json!("summary B"));
    assert!(writes[1].1["payload"].get("summary_embedding").is_none());
    assert_eq!(
        writes[2].1["payload"]["summary_embedding"],
        json!([0.5, 0.75])
    );
}

/// Same window, different second writer: a re-store that has already
/// retrieved the point with vector(A) would otherwise PUT that vector back
/// between the patch's delete and set.
#[tokio::test]
async fn concurrent_store_cannot_land_inside_a_patch_invalidation() {
    let server = race_server().await;
    let client = client_for(&server);
    let hash = "a".repeat(64);

    let to_b = PatchMemoryRequest {
        summary: Some("summary B".into()),
        ..Default::default()
    };
    let user_patch = client.patch_memory(&hash, &to_b);
    let restore = async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let mut same_summary = incoming();
        same_summary.summary = Some("summary A".into());
        client.store(&same_summary).await
    };
    let (a, b) = tokio::join!(user_patch, restore);
    a.expect("user patch succeeds");
    b.expect("store succeeds");

    let requests = server.received_requests().await.unwrap();
    let order: Vec<String> = requests
        .iter()
        .filter(|r| r.url.path() != POINTS_PATH || r.method == "PUT")
        .map(|r| format!("{} {}", r.method, r.url.path()))
        .collect();
    assert_eq!(
        order,
        vec![
            format!("POST {PAYLOAD_DELETE_PATH}"),
            format!("POST {PAYLOAD_PATH}"),
            format!("PUT {POINTS_PATH}"),
        ],
        "the store's PUT must not fall between the patch's delete and set"
    );
}
