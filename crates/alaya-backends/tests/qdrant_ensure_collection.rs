//! Startup collection bootstrap (#31): `ensure_collection` creates the memory
//! collection (and its tag sidecar) with Cosine vectors when absent, and is a
//! no-op when the collection already exists. The memory collection also
//! carries the BM25 sparse vector (LAB-7716): created with it, or given it in
//! place when it lacks it.

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
        .unwrap()
        .ensure_collection(1024)
        .await
        .expect("ensure succeeds");

    // Each create PUT carried the configured vector size and Cosine distance;
    // only the memory collection's carried the BM25 sparse vector.
    let reqs = server.received_requests().await.unwrap();
    let create = |p: &str| {
        let r = reqs
            .iter()
            .find(|r| r.method.as_str() == "PUT" && r.url.path() == p)
            .expect("a create was sent");
        serde_json::from_slice::<Value>(&r.body).expect("body is JSON")
    };
    for p in ["/collections/memories", "/collections/memories_tags"] {
        assert_eq!(
            create(p)["vectors"],
            json!({"size": 1024, "distance": "Cosine"})
        );
    }
    assert_eq!(
        create("/collections/memories")["sparse_vectors"],
        json!({"bm25": {"modifier": "idf"}})
    );
    assert!(
        create("/collections/memories_tags")
            .get("sparse_vectors")
            .is_none()
    );
    // Mount .expect(1) counts are verified on server drop.
}

/// `GET /collections/{c}` for a present collection, with or without the
/// BM25 vector.
fn present(with_keyword: bool) -> ResponseTemplate {
    let sparse = if with_keyword {
        json!({"bm25": {"modifier": "idf"}})
    } else {
        Value::Null
    };
    ResponseTemplate::new(200).set_body_json(json!({
        "result": {
            "status": "green",
            "config": {"params": {
                "vectors": {"size": 1024, "distance": "Cosine"},
                "sparse_vectors": sparse,
            }},
        },
        "status": "ok",
    }))
}

#[tokio::test]
async fn adds_the_keyword_vector_to_a_collection_that_lacks_it() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(present(false))
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path("/collections/memories/vectors/bm25"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"result": {}})))
        .expect(1)
        .mount(&server)
        .await;

    QdrantClient::new(server.uri(), "memories".into(), None)
        .unwrap()
        .ensure_collection(1024)
        .await
        .expect("ensure succeeds");

    // One PUT in all: the vector, with the lowercase `idf` modifier Qdrant
    // requires. No collection is recreated, and the tag sidecar is not given
    // one.
    let reqs = server.received_requests().await.unwrap();
    let puts: Vec<_> = reqs.iter().filter(|r| r.method.as_str() == "PUT").collect();
    assert_eq!(puts.len(), 1, "{puts:?}");
    let body: Value = serde_json::from_slice(&puts[0].body).unwrap();
    assert_eq!(body, json!({"sparse": {"modifier": "idf"}}));
}

#[tokio::test]
async fn keyword_vector_add_error_propagates() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(present(false))
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"status": "bad"})))
        .mount(&server)
        .await;

    let result = QdrantClient::new(server.uri(), "memories".into(), None)
        .unwrap()
        .ensure_collection(1024)
        .await;
    assert!(
        result.is_err(),
        "a failed keyword-vector add must propagate so the caller retries"
    );
}

#[tokio::test]
async fn idempotent_when_collection_exists() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(present(true))
        .mount(&server)
        .await;
    // A PUT would mean we tried to recreate an existing collection or re-add
    // a vector it already has.
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    QdrantClient::new(server.uri(), "memories".into(), None)
        .unwrap()
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
        .unwrap()
        .ensure_collection(1024)
        .await;
    assert!(
        result.is_err(),
        "a failed main-collection create must propagate so the caller retries"
    );
}

#[tokio::test]
async fn non_404_probe_error_propagates_without_creating() {
    let server = MockServer::start().await;
    // The probe returns a server error (e.g. Qdrant still warming up), not a 404.
    Mock::given(method("GET"))
        .and(path("/collections/memories"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;
    // A create here would mean we misread a transient fault as "collection absent".
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    let result = QdrantClient::new(server.uri(), "memories".into(), None)
        .unwrap()
        .ensure_collection(1024)
        .await;
    assert!(
        result.is_err(),
        "a non-404 probe response must propagate, not fall through to create"
    );
    // Drop verifies zero PUTs were sent.
}
