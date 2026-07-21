//! Startup collection bootstrap (#31): `ensure_collection` creates the memory
//! collection (and its tag sidecar) with Cosine vectors when absent, and is a
//! no-op when the collection already exists.

use alaya_backends::qdrant::QdrantClient;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn creates_absent_collections_with_cosine_config() {
    let server = MockServer::start().await;
    // Every collection reports absent…
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    // …so both the main collection and its tag sidecar are created exactly once.
    Mock::given(method("PUT"))
        .and(path("/collections/memories"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"result": true})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path("/collections/memories_tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"result": true})))
        .expect(1)
        .mount(&server)
        .await;

    QdrantClient::new(server.uri(), "memories".into(), None)
        .ensure_collection(1024)
        .await
        .expect("ensure succeeds");

    // A create PUT carried the configured vector size and Cosine distance.
    let reqs = server.received_requests().await.unwrap();
    let create = reqs
        .iter()
        .filter(|r| !r.body.is_empty())
        .map(|r| serde_json::from_slice::<Value>(&r.body).expect("body is JSON"))
        .find(|b| b["vectors"]["size"] == json!(1024))
        .expect("a create with the configured vector size was sent");
    assert_eq!(create["vectors"]["distance"], json!("Cosine"));
    // Mount .expect(1) counts are verified on server drop.
}

#[tokio::test]
async fn idempotent_when_collection_exists() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"result": {"status": "green"}, "status": "ok"})),
        )
        .mount(&server)
        .await;
    // A PUT would mean we tried to recreate an existing collection.
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    QdrantClient::new(server.uri(), "memories".into(), None)
        .ensure_collection(1024)
        .await
        .expect("ensure succeeds");
    // Drop verifies the PUT-expected-0 assertion (no recreate).
}

#[tokio::test]
async fn main_collection_create_error_propagates() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path("/collections/memories"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"status": "boom"})))
        .mount(&server)
        .await;

    let result = QdrantClient::new(server.uri(), "memories".into(), None)
        .ensure_collection(1024)
        .await;
    assert!(
        result.is_err(),
        "a failed main-collection create must propagate so the caller retries"
    );
}
