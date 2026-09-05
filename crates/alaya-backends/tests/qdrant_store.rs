//! Regression tests for `store` re-store semantics (alaya#86).
//!
//! Qdrant's upsert replaces a point's payload wholesale and the point id is
//! derived from `content_hash`, so storing already-present content used to
//! zero the record's `created_at` / `access_count` / `access_timestamps` and
//! still report `created: true`. `store` now retrieves the point first,
//! carries that history over, and reports `created: false`.

use alaya_backends::{VectorStorage, qdrant::QdrantClient};
use alaya_types::memory::Memory;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const POINTS_PATH: &str = "/collections/memories/points";

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
