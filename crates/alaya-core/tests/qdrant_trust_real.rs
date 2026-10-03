//! `min_trust_score` against a real Qdrant: the range filter must read the
//! key the payload actually stores (top-level `provenance.trust_score`).
//!
//! Ignored by default: it needs `QDRANT_TEST_URL`, a disposable Qdrant (the
//! test makes and drops its own collections). Embeddings are a fixed stub and
//! the graph is unreachable (non-fatal), so only Qdrant is real:
//!
//!   docker run -d -p 16417:6333 qdrant/qdrant:v1.17.1
//!   QDRANT_TEST_URL=http://localhost:16417 \
//!     cargo test -p alaya-core --test qdrant_trust_real -- --include-ignored

use std::collections::{HashMap, HashSet};

use alaya_backends::{
    EmbeddingProvider, StoreMode, VectorStorage,
    graph::GraphHttpClient,
    graph_ref::{ConsolidationRef, GraphRef, HebbianRef},
    qdrant::QdrantClient,
};
use alaya_core::service::{MemoryService, SearchParams};
use alaya_types::{
    Result,
    memory::{HealthStatus, Memory},
    search::{PromptName, SearchMode},
};
use async_trait::async_trait;
use serde_json::{Value, json};

const DIMS: usize = 2;

/// Every text embeds to the same vector, so every stored memory is a hit.
struct FixedEmbeddings;

#[async_trait(?Send)]
impl EmbeddingProvider for FixedEmbeddings {
    async fn embed_batch(&self, texts: &[&str], _p: PromptName) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect())
    }
    fn dimensions(&self) -> usize {
        DIMS
    }
    fn model_name(&self) -> &str {
        "fixed"
    }
    async fn health(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            status: "ok".into(),
            backend: "fixed".into(),
            details: None,
        })
    }
}

fn memory(hash: char, trust: f64) -> Memory {
    Memory {
        content: format!("trust {trust}"),
        content_hash: hash.to_string().repeat(64),
        tags: vec![],
        memory_type: "note".into(),
        metadata: None,
        created_at: 1000.0,
        updated_at: 1000.0,
        embedding: Some(vec![0.6, 0.8]),
        summary: None,
        salience_score: 0.5,
        access_count: 0,
        access_timestamps: vec![],
        emotional_valence: None,
        encoding_context: None,
        provenance: Some(HashMap::from([("trust_score".to_string(), json!(trust))])),
        summary_embedding: None,
        supersession_log: None,
        supersession_reason: None,
    }
}

async fn drop_collections(url: &str, name: &str) {
    #[allow(clippy::disallowed_methods, reason = "test-only client")]
    let client = reqwest::Client::new();
    for c in [name.to_string(), format!("{name}_tags")] {
        let _ = client.delete(format!("{url}/collections/{c}")).send().await;
    }
}

fn params(mode: SearchMode, min_trust_score: Option<f64>) -> SearchParams {
    SearchParams {
        query: "trust".into(),
        mode,
        page: 1,
        page_size: 10,
        tags: None,
        match_all: false,
        k: 10,
        min_similarity: None,
        memory_type: None,
        encoding_context: None,
        include_superseded: false,
        min_trust_score,
        output: Default::default(),
        cursor: None,
    }
}

fn returned(result: &Value) -> HashSet<String> {
    result["results"]
        .as_array()
        .expect("results array")
        .iter()
        .filter_map(|r| r["content_hash"].as_str().map(String::from))
        .collect()
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn min_trust_score_keeps_only_memories_that_meet_it() {
    let url = std::env::var("QDRANT_TEST_URL")
        .expect("QDRANT_TEST_URL must point at a disposable Qdrant");
    let name = format!("trust_real_{}", std::process::id());
    drop_collections(&url, &name).await;

    let qdrant = QdrantClient::new(url.clone(), name.clone(), None).unwrap();
    qdrant
        .ensure_collection(DIMS)
        .await
        .expect("create test collection");
    let low = [memory('a', 0.5), memory('b', 0.5)];
    let high = [memory('c', 0.8), memory('d', 0.8)];
    for m in low.iter().chain(&high) {
        qdrant.store(m, StoreMode::Upsert).await.expect("store");
    }

    let graph = std::rc::Rc::new(GraphHttpClient::new("http://127.0.0.1:9".into(), "").unwrap());
    let svc = MemoryService::new(
        Box::new(qdrant),
        Box::new(FixedEmbeddings),
        Box::new(GraphRef(graph.clone())),
        Box::new(HebbianRef(graph.clone())),
        Box::new(ConsolidationRef(graph)),
        None,
    );

    let hashes =
        |ms: &[Memory]| -> HashSet<String> { ms.iter().map(|m| m.content_hash.clone()).collect() };
    let everything: HashSet<String> = hashes(&low).union(&hashes(&high)).cloned().collect();

    let mut outcomes = Vec::new();
    for mode in [SearchMode::Hybrid, SearchMode::Similar] {
        let unfiltered = svc.search(params(mode, None)).await.expect("search");
        let filtered = svc.search(params(mode, Some(0.6))).await.expect("search");
        outcomes.push((mode, returned(&unfiltered), returned(&filtered)));
    }
    drop_collections(&url, &name).await;

    for (mode, unfiltered, filtered) in outcomes {
        assert_eq!(unfiltered, everything, "{mode:?} without min_trust_score");
        assert_eq!(filtered, hashes(&high), "{mode:?} with min_trust_score 0.6");
    }
}
