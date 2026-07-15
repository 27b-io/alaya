//! Regression tests for `update_metadata` write-ordering commit-point semantics.
//!
//! The supersession marker (`metadata.superseded_by`) is the COMMIT POINT of a
//! metadata update: searches hide a memory as soon as it lands. Auxiliary
//! top-level fields must therefore be written first — if that write fails,
//! nothing is marked superseded and a retry is clean (PR #56 review; the
//! reversed order was a critical finding in #55).

use alaya_backends::{VectorStorage, qdrant::QdrantClient};
use alaya_types::memory::MetadataUpdate;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const PAYLOAD_PATH: &str = "/collections/memories/points/payload";

fn client_for(server: &MockServer) -> QdrantClient {
    QdrantClient::new(server.uri(), "memories".into(), None)
}

fn supersede_with_aux() -> MetadataUpdate {
    MetadataUpdate {
        superseded_by: Some("b".repeat(64)),
        access_count: Some(7),
        extra: None,
    }
}

fn body_of(req: &wiremock::Request) -> Value {
    serde_json::from_slice(&req.body).expect("request body is JSON")
}

#[tokio::test]
async fn top_level_fields_written_before_supersession_marker() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PAYLOAD_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .expect(2)
        .mount(&server)
        .await;

    client_for(&server)
        .update_metadata(&"a".repeat(64), supersede_with_aux())
        .await
        .expect("both writes succeed");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);

    let first = body_of(&requests[0]);
    assert_eq!(first["payload"]["access_count"], json!(7));
    assert!(
        first.get("key").is_none(),
        "auxiliary write must be unscoped (top-level), got: {first}"
    );

    let second = body_of(&requests[1]);
    assert_eq!(second["key"], json!("metadata"));
    assert_eq!(second["payload"]["superseded_by"], json!("b".repeat(64)));
}

#[tokio::test]
async fn failed_top_level_write_prevents_supersession_marker() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PAYLOAD_PATH))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"status": {"error": "boom"}})),
        )
        .mount(&server)
        .await;

    let result = client_for(&server)
        .update_metadata(&"a".repeat(64), supersede_with_aux())
        .await;
    assert!(result.is_err(), "failed top-level write must propagate");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.len(),
        1,
        "superseded_by must NOT be written after a failed top-level write"
    );
    let only = body_of(&requests[0]);
    assert!(
        only.get("key").is_none(),
        "the sole attempted write must be the top-level one, got: {only}"
    );
}

#[tokio::test]
async fn supersession_only_update_sends_single_scoped_write() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(PAYLOAD_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .expect(1)
        .mount(&server)
        .await;

    let updates = MetadataUpdate {
        superseded_by: Some("b".repeat(64)),
        ..Default::default()
    };
    client_for(&server)
        .update_metadata(&"a".repeat(64), updates)
        .await
        .expect("scoped write succeeds");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1, "no empty top-level write should be sent");
    let only = body_of(&requests[0]);
    assert_eq!(only["key"], json!("metadata"));
    assert_eq!(only["payload"]["superseded_by"], json!("b".repeat(64)));
}
