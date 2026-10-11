//! The BM25 keyword vector against a real Qdrant (LAB-7716).
//!
//! Ignored by default: they need `QDRANT_TEST_URL`, a disposable Qdrant >= 1.18
//! (every test makes and drops its own collection). CI runs them against a
//! Qdrant service container:
//!
//!   docker run -d -p 16417:6333 qdrant/qdrant:v1.19.1
//!   QDRANT_TEST_URL=http://localhost:16417 \
//!     cargo test -p alaya-backends --test qdrant_keyword_real -- --include-ignored --test-threads=1

use alaya_backends::{StoreMode, VectorStorage, qdrant::QdrantClient};
use alaya_types::{memory::Memory, search::PayloadFilter};
use serde_json::{Value, json};

fn qdrant_url() -> String {
    std::env::var("QDRANT_TEST_URL")
        .expect("QDRANT_TEST_URL must point at a disposable Qdrant >= 1.18")
}

fn memory(content: &str, memory_type: &str) -> Memory {
    Memory {
        content: content.into(),
        content_hash: format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(content)),
        tags: vec![],
        memory_type: memory_type.into(),
        metadata: None,
        created_at: 1000.0,
        updated_at: 1000.0,
        embedding: Some(vec![0.5, 0.75]),
        summary: None,
        salience_score: 0.5,
        access_count: 0,
        access_timestamps: vec![],
        emotional_valence: None,
        encoding_context: None,
        provenance: None,
        summary_embedding: None,
        supersession_log: None,
        supersession_reason: None,
        nearest_similarity: None,
    }
}

fn point_id(hash: &str) -> String {
    uuid::Uuid::parse_str(&hash[..32]).unwrap().to_string()
}

#[allow(clippy::disallowed_methods, reason = "test-only client")]
fn http() -> reqwest::Client {
    reqwest::Client::new()
}

/// A collection dropped on `cleanup`, created by the test itself.
struct Collection {
    url: String,
    name: String,
}

impl Collection {
    async fn new(test: &str) -> Self {
        let c = Self {
            url: qdrant_url(),
            name: format!("kw_{test}_{}", std::process::id()),
        };
        c.cleanup().await;
        c
    }

    fn client(&self) -> QdrantClient {
        QdrantClient::new(self.url.clone(), self.name.clone(), None).unwrap()
    }

    /// The collection as a server from before the keyword index made it.
    async fn create_dense_only(&self) {
        let resp = http()
            .put(format!("{}/collections/{}", self.url, self.name))
            .json(&json!({"vectors": {"size": 2, "distance": "Cosine"}}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
    }

    async fn params(&self) -> Value {
        let body: Value = http()
            .get(format!("{}/collections/{}", self.url, self.name))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        body["result"]["config"]["params"].clone()
    }

    /// The raw point (payload and every vector) of `hash`.
    async fn point(&self, hash: &str) -> Value {
        let body: Value = http()
            .post(format!("{}/collections/{}/points", self.url, self.name))
            .json(&json!({"ids": [point_id(hash)], "with_payload": true, "with_vector": true}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        body["result"][0].clone()
    }

    /// Write a point with the dense vector only, the way a server from
    /// before the keyword index stores (and re-stores, mid-rollout).
    async fn put_dense_only(&self, m: &Memory, payload: Value) {
        let resp = http()
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.url, self.name
            ))
            .json(&json!({"points": [{
                "id": point_id(&m.content_hash),
                "vector": m.embedding,
                "payload": payload,
            }]}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
    }

    async fn cleanup(&self) {
        let _ = http()
            .delete(format!("{}/collections/{}", self.url, self.name))
            .send()
            .await;
    }
}

/// A payload with a revision, so a write that changed it would show.
fn payload_of(m: &Memory) -> Value {
    json!({
        "content": m.content, "content_hash": m.content_hash, "tags": [],
        "memory_type": m.memory_type, "created_at": m.created_at,
        "updated_at": m.updated_at, "salience_score": m.salience_score,
        "access_count": 4, "rev": "0000000000000001-1", "rev_log": ["0000000000000001-1"],
    })
}

/// A point's dense vector: Qdrant returns a bare array for a point that
/// carries no other vector, else the unnamed entry of the vector map.
fn dense(point: &Value) -> &Value {
    let v = &point["vector"];
    if v.is_array() { v } else { &v[""] }
}

fn has_keyword_vector(point: &Value) -> bool {
    point["vector"]["bm25"]["indices"]
        .as_array()
        .is_some_and(|i| !i.is_empty())
}

/// AC-1: startup adds the vector to a collection that lacks it, changing no
/// point, and a second start changes nothing.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn startup_adds_the_keyword_vector_once_and_changes_no_point() {
    let coll = Collection::new("startup").await;
    coll.create_dense_only().await;
    let old = memory("written before the keyword index", "note");
    coll.put_dense_only(&old, payload_of(&old)).await;
    let before = coll.point(&old.content_hash).await;
    assert!(coll.params().await.get("sparse_vectors").is_none());

    coll.client()
        .ensure_collection(2)
        .await
        .expect("first start");
    let params = coll.params().await;
    assert_eq!(
        params["sparse_vectors"],
        json!({"bm25": {"modifier": "idf"}})
    );
    assert_eq!(params["vectors"], json!({"size": 2, "distance": "Cosine"}));
    assert_eq!(coll.point(&old.content_hash).await, before);
    assert_eq!(
        coll.client().count_missing_keyword_vectors().await.unwrap(),
        1
    );

    coll.client()
        .ensure_collection(2)
        .await
        .expect("second start");
    assert_eq!(coll.params().await, params);
    assert_eq!(coll.point(&old.content_hash).await, before);
    coll.cleanup().await;
}

/// AC-2 and AC-4: a store and a re-store (both compare-and-set) write the
/// keyword vector, and BM25 search finds content by identifier, best first,
/// with the payload filter applied.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn stores_write_the_keyword_vector_and_bm25_finds_identifiers() {
    let coll = Collection::new("store").await;
    let client = coll.client();
    client.ensure_collection(2).await.unwrap();
    let ticket = memory(
        "LAB-7716 adds BM25 keyword search in crates/alaya-core/src/service.rs",
        "decision",
    );
    let other = memory("a note about lunch and nothing else", "note");
    for m in [&ticket, &other] {
        let (created, _) = client.store(m, StoreMode::Upsert).await.unwrap();
        assert!(created);
        assert!(has_keyword_vector(&coll.point(&m.content_hash).await));
    }
    assert_eq!(client.count_missing_keyword_vectors().await.unwrap(), 0);

    let (created, _) = client.store(&ticket, StoreMode::Upsert).await.unwrap();
    assert!(!created);
    let restored = coll.point(&ticket.content_hash).await;
    assert!(has_keyword_vector(&restored), "{restored}");
    assert_eq!(dense(&restored), &json!([0.5547002, 0.8320503]));

    let hits = client.search_by_text("lab-7716", 10, None).await.unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].memory.content_hash, ticket.content_hash);
    assert!(hits[0].score > 0.0);
    let paths = client.search_by_text("service.rs", 10, None).await.unwrap();
    assert_eq!(paths[0].memory.content_hash, ticket.content_hash);

    let typed = PayloadFilter {
        memory_type: Some("note".into()),
        ..Default::default()
    };
    let none = client
        .search_by_text("lab-7716", 10, Some(typed))
        .await
        .unwrap();
    assert!(none.is_empty(), "{none:?}");
    // Only stop words: no terms, no hits, no error.
    assert!(
        client
            .search_by_text("the of and", 10, None)
            .await
            .unwrap()
            .is_empty()
    );
    coll.cleanup().await;
}

/// AC-3: the backfill writes the keyword vector and nothing else — payload
/// (revision included) and dense vector unchanged — in bounded runs, and a
/// later run repairs a memory a dense-only write re-stored mid-rollout.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn backfill_writes_only_the_keyword_vector_and_repairs_a_rollout() {
    let coll = Collection::new("backfill").await;
    coll.create_dense_only().await;
    let olds: Vec<Memory> = (0..3)
        .map(|i| memory(&format!("legacy memory {i} about qdrant snapshots"), "note"))
        .collect();
    for m in &olds {
        coll.put_dense_only(m, payload_of(m)).await;
    }
    let client = coll.client();
    client.ensure_collection(2).await.unwrap();
    let mut before = Vec::new();
    for m in &olds {
        before.push(coll.point(&m.content_hash).await);
    }
    assert_eq!(client.count_missing_keyword_vectors().await.unwrap(), 3);

    assert_eq!(
        client.fill_keyword_vectors(2).await.unwrap(),
        2,
        "bounded by limit"
    );
    assert_eq!(client.count_missing_keyword_vectors().await.unwrap(), 1);
    assert_eq!(client.fill_keyword_vectors(100).await.unwrap(), 1);
    assert_eq!(client.count_missing_keyword_vectors().await.unwrap(), 0);
    assert_eq!(
        client.fill_keyword_vectors(100).await.unwrap(),
        0,
        "nothing left"
    );

    for (m, was) in olds.iter().zip(&before) {
        let now = coll.point(&m.content_hash).await;
        assert_eq!(now["payload"], was["payload"], "payload untouched");
        assert_eq!(dense(&now), dense(was), "dense vector untouched");
        assert!(has_keyword_vector(&now), "{now}");
    }

    // An older server re-stores one memory dense-only mid-rollout: the
    // keyword vector is gone, and the next run puts it back.
    let mut payload = payload_of(&olds[0]);
    payload["access_count"] = json!(5);
    coll.put_dense_only(&olds[0], payload.clone()).await;
    assert_eq!(client.count_missing_keyword_vectors().await.unwrap(), 1);
    assert_eq!(client.fill_keyword_vectors(100).await.unwrap(), 1);
    let repaired = coll.point(&olds[0].content_hash).await;
    assert!(has_keyword_vector(&repaired));
    assert_eq!(repaired["payload"], payload);
    let hits = client.search_by_text("snapshots", 10, None).await.unwrap();
    assert_eq!(hits.len(), 3, "{hits:?}");
    coll.cleanup().await;
}
