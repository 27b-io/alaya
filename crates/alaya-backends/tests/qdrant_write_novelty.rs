//! `write_novelty` (the `/stats` write-novelty input): one filtered,
//! payload-restricted scroll over the window, paged to the end, failing
//! rather than reading a malformed page as an empty window.

use alaya_backends::{VectorStorage, qdrant::QdrantClient};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SCROLL: &str = "/collections/memories/points/scroll";

fn client_for(server: &MockServer) -> QdrantClient {
    QdrantClient::new(server.uri(), "memories".into(), None).unwrap()
}

fn page(points: Value, next: Value) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .set_body_json(json!({"result": {"points": points, "next_page_offset": next}}))
}

#[tokio::test]
async fn scrolls_every_page_of_the_window() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(SCROLL))
        .and(body_partial_json(json!({"offset": "p2"})))
        .respond_with(page(
            json!([{"payload": {"created_at": 300.0}}]),
            Value::Null,
        ))
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(SCROLL))
        .respond_with(page(
            json!([
                {"payload": {"created_at": 100.0, "nearest_similarity": 0.91}},
                {"payload": {"created_at": 200.0, "nearest_similarity": null}},
                {"payload": {"nearest_similarity": 0.5}},
            ]),
            json!("p2"),
        ))
        .expect(1)
        .mount(&server)
        .await;

    let rows = client_for(&server).write_novelty(50.0).await.unwrap();
    assert_eq!(
        rows,
        vec![(100.0, Some(0.91)), (200.0, None), (300.0, None)],
        "a point without created_at has no day and is left out"
    );

    let first: Value = server.received_requests().await.unwrap()[0]
        .body_json()
        .unwrap();
    assert_eq!(
        first["filter"],
        json!({"must": [{"key": "created_at", "range": {"gte": 50.0}}]})
    );
    assert_eq!(
        first["with_payload"],
        json!(["created_at", "nearest_similarity"])
    );
    assert_eq!(first["with_vector"], json!(false));
}

#[tokio::test]
async fn fails_closed_on_an_error_or_a_page_without_result() {
    for resp in [
        ResponseTemplate::new(500).set_body_json(json!({"status": {"error": "boom"}})),
        ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(SCROLL))
            .respond_with(resp)
            .mount(&server)
            .await;
        assert!(client_for(&server).write_novelty(0.0).await.is_err());
    }
}
