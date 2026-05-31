//! Integration tests against real backends (Qdrant + TEI).
//!
//! Gated on QDRANT_URL + EMBEDDING_URL env vars.
//! Graph operations are optional (need GRAPH_URL pointing at a running bridge).
//!
//! Run:
//!   QDRANT_URL=http://10.43.119.230:6333 \
//!   EMBEDDING_URL=http://10.43.242.167 \
//!   cargo test -p alaya-core --test integration -- --test-threads=1

use alaya_backends::{
    embedding::EmbeddingClient,
    graph::GraphHttpClient,
    graph_ref::{ConsolidationRef, GraphRef, HebbianRef},
    qdrant::QdrantClient,
};
use alaya_core::service::{MemoryService, SearchParams, StoreParams};
use alaya_types::search::SearchMode;

/// Test collection — separate from production to avoid contamination.
const TEST_COLLECTION: &str = "alaya_integration_test";

fn skip_unless_backends() -> Option<(String, String)> {
    let qdrant = std::env::var("QDRANT_URL").ok()?;
    let embedding = std::env::var("EMBEDDING_URL").ok()?;
    Some((qdrant, embedding))
}

fn build_service(qdrant_url: &str, embedding_url: &str) -> MemoryService {
    let qdrant = QdrantClient::new(qdrant_url.into(), TEST_COLLECTION.into(), None);

    let embeddings = EmbeddingClient::new(
        embedding_url.into(),
        "Snowflake/snowflake-arctic-embed-l-v2.0".into(),
        1024,
        32,
        None,
    );

    // Graph client — use a stub that returns errors (non-fatal in MemoryService)
    let graph_url = std::env::var("GRAPH_URL").unwrap_or_else(|_| "http://localhost:9999".into());
    let graph = std::rc::Rc::new(GraphHttpClient::new(graph_url, ""));

    MemoryService::new(
        Box::new(qdrant),
        Box::new(embeddings),
        Box::new(GraphRef(graph.clone())),
        Box::new(HebbianRef(graph.clone())),
        Box::new(ConsolidationRef(graph)),
        None,
    )
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "current_thread")]
async fn store_and_retrieve() {
    let Some((qdrant_url, embedding_url)) = skip_unless_backends() else {
        eprintln!("skipping: QDRANT_URL/EMBEDDING_URL not set");
        return;
    };

    let svc = build_service(&qdrant_url, &embedding_url);

    // Ensure test collection exists
    ensure_test_collection(&qdrant_url).await;

    let result = svc
        .store_memory(StoreParams {
            content: "Integration test: the alaya-core store pipeline works end-to-end".into(),
            tags: Some(vec!["integration-test".into(), "alaya".into()]),
            memory_type: Some("note".into()),
            metadata: None,
            client_hostname: Some("test-runner".into()),
            summary: Some("Test memory".into()),
            dedup_threshold: None,
        })
        .await
        .expect("store_memory should succeed");

    assert_eq!(result["success"], true);
    let hash = result["content_hash"].as_str().unwrap();
    assert_eq!(hash.len(), 64);

    // Retrieve by hash
    let memory = svc
        .vectors
        .get_by_hash(hash)
        .await
        .expect("get_by_hash should succeed")
        .expect("memory should exist");

    assert_eq!(memory.content_hash, hash);
    assert!(memory.content.contains("alaya-core store pipeline"));
    assert_eq!(memory.tags, vec!["integration-test", "alaya"]);

    // Clean up
    svc.delete_memory(hash)
        .await
        .expect("delete should succeed");
}

#[tokio::test(flavor = "current_thread")]
async fn search_hybrid() {
    let Some((qdrant_url, embedding_url)) = skip_unless_backends() else {
        return;
    };

    let svc = build_service(&qdrant_url, &embedding_url);
    ensure_test_collection(&qdrant_url).await;

    // Store a few memories
    let contents = [
        "Rust's ownership model prevents data races at compile time",
        "Python uses garbage collection for memory management",
        "The Qdrant vector database supports HNSW indexing for fast similarity search",
    ];

    let mut stored_hashes = Vec::new();
    for content in &contents {
        let r = svc
            .store_memory(StoreParams {
                content: content.to_string(),
                tags: Some(vec!["search-test".into()]),
                memory_type: Some("note".into()),
                metadata: None,
                client_hostname: None,
                summary: None,
                dedup_threshold: None,
            })
            .await
            .expect("store should succeed");
        stored_hashes.push(r["content_hash"].as_str().unwrap().to_string());
    }

    // Search for Rust-related content
    let result = svc
        .search(SearchParams {
            query: "How does Rust handle memory safety?".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: Default::default(),
            cursor: None,
        })
        .await
        .expect("search should succeed");

    let results = result["results"].as_array().unwrap();
    assert!(!results.is_empty(), "should find at least one result");

    // The Rust memory should be the top result
    let top = &results[0];
    assert!(
        top["content"].as_str().unwrap().contains("Rust"),
        "top result should be about Rust, got: {}",
        top["content"]
    );

    // Clean up
    for hash in &stored_hashes {
        let _ = svc.delete_memory(hash).await;
    }
}

#[tokio::test(flavor = "current_thread")]
async fn dedup_threshold_skips_duplicate() {
    let Some((qdrant_url, embedding_url)) = skip_unless_backends() else {
        return;
    };

    let svc = build_service(&qdrant_url, &embedding_url);
    ensure_test_collection(&qdrant_url).await;

    // Store original
    let r1 = svc
        .store_memory(StoreParams {
            content: "The capital of France is Paris, known for the Eiffel Tower".into(),
            tags: Some(vec!["dedup-test".into()]),
            memory_type: Some("note".into()),
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        })
        .await
        .expect("first store should succeed");

    let hash1 = r1["content_hash"].as_str().unwrap().to_string();

    // Store near-duplicate with dedup threshold
    let r2 = svc
        .store_memory(StoreParams {
            content: "Paris is the capital of France, famous for the Eiffel Tower".into(),
            tags: Some(vec!["dedup-test".into()]),
            memory_type: Some("note".into()),
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: Some(0.7), // Low threshold to catch near-duplicates
        })
        .await
        .expect("second store should succeed");

    // Should be flagged as duplicate
    assert_eq!(
        r2.get("duplicate").and_then(|v| v.as_bool()),
        Some(true),
        "near-duplicate should be detected: {:?}",
        r2
    );

    // Clean up
    let _ = svc.delete_memory(&hash1).await;
    if let Some(hash2) = r2.get("content_hash").and_then(|v| v.as_str()) {
        let _ = svc.delete_memory(hash2).await;
    }
}

#[tokio::test(flavor = "current_thread")]
async fn search_by_tag() {
    let Some((qdrant_url, embedding_url)) = skip_unless_backends() else {
        return;
    };

    let svc = build_service(&qdrant_url, &embedding_url);
    ensure_test_collection(&qdrant_url).await;

    let r = svc
        .store_memory(StoreParams {
            content: "Tag search test: this memory has a unique tag".into(),
            tags: Some(vec!["unique-tag-test-42".into()]),
            memory_type: Some("note".into()),
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        })
        .await
        .expect("store should succeed");

    let hash = r["content_hash"].as_str().unwrap().to_string();

    let result = svc
        .search(SearchParams {
            query: "".into(),
            mode: SearchMode::Tag,
            page: 1,
            page_size: 10,
            tags: Some(vec!["unique-tag-test-42".into()]),
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: Default::default(),
            cursor: None,
        })
        .await
        .expect("tag search should succeed");

    let results = result["results"].as_array().unwrap();
    assert!(
        results
            .iter()
            .any(|r| r["content_hash"].as_str() == Some(hash.as_str())),
        "should find the tagged memory"
    );

    let _ = svc.delete_memory(&hash).await;
}

#[tokio::test(flavor = "current_thread")]
async fn health_check() {
    let Some((qdrant_url, embedding_url)) = skip_unless_backends() else {
        return;
    };

    let svc = build_service(&qdrant_url, &embedding_url);

    let result = svc
        .check_database_health()
        .await
        .expect("health should succeed");
    // Qdrant should be healthy
    assert!(
        result.contains_key("status"),
        "health should have status field"
    );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

async fn ensure_test_collection(qdrant_url: &str) {
    let client = reqwest::Client::new();

    // Check if collection exists
    let resp = client
        .get(format!("{}/collections/{}", qdrant_url, TEST_COLLECTION))
        .send()
        .await;

    if let Ok(r) = resp
        && r.status().is_success()
    {
        return; // Already exists
    }

    // Create collection with Arctic dimensions (1024)
    let body = serde_json::json!({
        "vectors": {
            "size": 1024,
            "distance": "Cosine"
        }
    });

    client
        .put(format!("{}/collections/{}", qdrant_url, TEST_COLLECTION))
        .json(&body)
        .send()
        .await
        .expect("failed to create test collection");
}

// ─── Trait wrappers — see alaya_backends::graph_ref ─────────────────────────
