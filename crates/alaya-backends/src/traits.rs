use std::collections::HashMap;

use async_trait::async_trait;

use alaya_types::{
    Result,
    graph::{
        CoAccessPair, Contradiction, ContradictionRef, Direction, Edge, EdgeMeta, GraphStats,
        Neighbor, SystemRelationType, UserRelationType,
    },
    memory::{
        HealthStatus, Memory, MetadataUpdate, PatchMemoryRequest, ScoredMemory, ScrollResult,
    },
    search::{PayloadFilter, PromptName},
};

/// Vector storage backend (Qdrant REST API).
///
/// `?Send` bound because WASM is single-threaded.
#[async_trait(?Send)]
pub trait VectorStorage {
    // Core CRUD

    /// Upsert a memory keyed by `content_hash`.
    ///
    /// Returns `(created, content_hash)`. `created` is `false` when a point
    /// with the same `content_hash` already existed, whether or not its
    /// stored payload still parses as a `Memory`. On that path the
    /// implementation MUST carry the existing point's server-maintained
    /// fields over the caller's values — `created_at`, `access_count`,
    /// `access_timestamps`, `supersession_reason`, and
    /// `metadata.superseded_by` — so a re-store never zeroes ranking inputs
    /// or resurrects a superseded memory (alaya#86). Every other payload
    /// field is written from `memory` as given and fields absent on `memory`
    /// are removed; `updated_at` is therefore whatever the caller set.
    async fn store(&self, memory: &Memory) -> Result<(bool, String)>;
    async fn get_by_hash(&self, content_hash: &str) -> Result<Option<Memory>>;
    async fn get_batch(&self, hashes: &[&str]) -> Result<Vec<Memory>>;
    async fn delete(&self, content_hash: &str) -> Result<bool>;
    async fn update_metadata(&self, content_hash: &str, updates: MetadataUpdate) -> Result<()>;

    /// Apply the SAME metadata update to many memories.
    ///
    /// Default: sequential fallback via `update_metadata`. Implementations
    /// may override with batched writes (e.g. one Qdrant set-payload call
    /// covering all points) while preserving `update_metadata`'s
    /// aux-fields-first / supersession-marker-last commit ordering.
    async fn update_metadata_batch(
        &self,
        content_hashes: &[&str],
        updates: MetadataUpdate,
    ) -> Result<()> {
        for hash in content_hashes {
            self.update_metadata(hash, updates.clone()).await?;
        }
        Ok(())
    }

    /// Patch mutable fields on an existing memory.
    ///
    /// Updates only the provided fields, sets `updated_at`, and returns the
    /// full updated memory. Metadata merge: incoming keys are merged into
    /// existing metadata; keys with JSON `null` values are deleted.
    async fn patch_memory(&self, content_hash: &str, patch: &PatchMemoryRequest) -> Result<Memory>;

    // Vector search
    async fn search_by_vector(
        &self,
        embedding: &[f32],
        limit: usize,
        filters: Option<PayloadFilter>,
    ) -> Result<Vec<ScoredMemory>>;
    async fn search_by_tags(
        &self,
        tags: &[&str],
        match_all: bool,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>>;
    async fn search_similar_tags(&self, tag_embedding: &[f32], limit: usize)
    -> Result<Vec<String>>;

    /// Upsert tag embeddings into the tag collection.
    /// Each entry is `(tag_name, embedding_vector)`.
    async fn upsert_tags(&self, tags: &[(&str, Vec<f32>)]) -> Result<()>;

    // Scroll / list
    async fn get_all(&self, limit: usize, offset: Option<&str>) -> Result<ScrollResult>;
    /// Fetch memories ordered by `created_at` descending.
    /// `start_from` is a cursor: when `Some(ts)`, only memories with
    /// `created_at < ts` are returned (exclusive, for cursor pagination).
    async fn get_recent(
        &self,
        limit: usize,
        start_from: Option<f64>,
        memory_type: Option<&str>,
    ) -> Result<Vec<Memory>>;

    // Metadata
    async fn count(&self) -> Result<usize>;
    async fn get_all_tags(&self) -> Result<Vec<String>>;
    async fn increment_access_count(&self, content_hash: &str) -> Result<()>;

    /// Batch increment access counts for multiple memories.
    ///
    /// Default: sequential fallback. Implementations may override with
    /// batch GET + concurrent PUTs to reduce round trips from 2N to 1+N.
    async fn increment_access_count_batch(&self, content_hashes: &[&str]) -> Result<()> {
        for hash in content_hashes {
            let _ = self.increment_access_count(hash).await;
        }
        Ok(())
    }

    async fn health(&self) -> Result<HealthStatus>;
}

/// Embedding generation backend (OpenAI-compatible REST API).
#[async_trait(?Send)]
pub trait EmbeddingProvider {
    async fn embed_batch(&self, texts: &[&str], prompt_name: PromptName) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn model_name(&self) -> &str;
}

/// Graph operations backend (calls alaya-bridge typed RPC).
#[async_trait(?Send)]
pub trait GraphService {
    // Node lifecycle
    async fn ensure_node(&self, content_hash: &str, created_at: f64) -> Result<()>;
    async fn delete_node(&self, content_hash: &str) -> Result<()>;

    // Typed edge CRUD (user-created)
    async fn create_typed_edge(
        &self,
        src: &str,
        dst: &str,
        rel: UserRelationType,
        meta: EdgeMeta,
    ) -> Result<bool>;
    async fn get_typed_edges(
        &self,
        hash: &str,
        rel: Option<UserRelationType>,
        dir: Direction,
        limit: usize,
    ) -> Result<Vec<Edge>>;
    async fn delete_typed_edge(&self, src: &str, dst: &str, rel: UserRelationType) -> Result<bool>;

    // System edge (internal only — SUPERSEDES)
    async fn create_system_edge(
        &self,
        src: &str,
        dst: &str,
        rel: SystemRelationType,
        created_at: f64,
    ) -> Result<bool>;

    /// Batch-create system edges in a single round-trip.
    ///
    /// Default: sequential fallback via `create_system_edge`. Implementations
    /// may override with a single HTTP POST to `/edges/create-system-batch`.
    async fn create_system_edges_batch(
        &self,
        edges: &[(String, String, SystemRelationType, f64)],
    ) -> Result<usize> {
        let mut created = 0;
        for (src, dst, rel, created_at) in edges {
            if self.create_system_edge(src, dst, *rel, *created_at).await? {
                created += 1;
            }
        }
        Ok(created)
    }

    // Contradiction queries
    async fn get_all_contradictions(&self, limit: usize) -> Result<Vec<Contradiction>>;
    async fn get_contradictions_for_hashes(
        &self,
        hashes: &[&str],
    ) -> Result<HashMap<String, Vec<ContradictionRef>>>;

    // Hebbian read operations
    async fn get_neighbors(
        &self,
        hash: &str,
        max_hops: u8,
        min_weight: f64,
        limit: usize,
    ) -> Result<Vec<Neighbor>>;
    async fn spreading_activation(
        &self,
        seeds: &[&str],
        max_hops: u8,
        decay: f64,
        min_activation: f64,
        limit: usize,
    ) -> Result<HashMap<String, f64>>;
    async fn hebbian_boosts_within(&self, hashes: &[&str]) -> Result<HashMap<String, f64>>;

    /// Batch-create typed edges in a single round-trip.
    ///
    /// Default: sequential fallback via `create_typed_edge`. Implementations
    /// may override with a single HTTP POST to `/edges/create-batch`.
    async fn create_typed_edges_batch(
        &self,
        edges: &[(String, String, UserRelationType, EdgeMeta)],
    ) -> Result<usize> {
        let mut created = 0;
        for (src, dst, rel, meta) in edges {
            if self.create_typed_edge(src, dst, *rel, meta.clone()).await? {
                created += 1;
            }
        }
        Ok(created)
    }

    // Stats
    async fn get_stats(&self) -> Result<GraphStats>;
}

/// Hebbian co-access write operations (fire-and-forget to bridge write queue).
#[async_trait(?Send)]
pub trait HebbianService {
    async fn enqueue_strengthen(&self, pairs: &[CoAccessPair]) -> Result<()>;
}

/// Consolidation operations (bridge-hosted background maintenance).
#[async_trait(?Send)]
pub trait ConsolidationService {
    async fn decay_all_edges(&self, decay_factor: f64, limit: usize) -> Result<usize>;
    async fn decay_stale_edges(
        &self,
        stale_before: f64,
        decay_factor: f64,
        limit: usize,
    ) -> Result<usize>;
    async fn prune_weak_edges(&self, threshold: f64, limit: usize) -> Result<usize>;
    async fn get_orphan_nodes(&self, limit: usize) -> Result<Vec<String>>;
}

/// Summary generation backend (e.g. Anthropic Messages API).
///
/// Optional — when absent, summaries are only set from client-provided values.
#[async_trait(?Send)]
pub trait SummaryProvider {
    /// Generate a one-line summary (~50 tokens) for the given content.
    async fn summarize(&self, content: &str) -> Result<String>;
}

/// Cross-encoder reranking backend (TEI `/rerank` endpoint).
///
/// Optional — when absent, hybrid search returns RRF-fused results without
/// a second-stage rerank. When present, top-N candidates from RRF are
/// re-scored as (query, document) pairs by a cross-encoder model
/// (e.g. BAAI/bge-reranker-v2-m3) and reordered by that score.
#[async_trait(?Send)]
pub trait RerankingService {
    /// Score each (query, text) pair; returns one score per input text,
    /// in the same order as `texts`. Higher = more relevant.
    /// Scores are sigmoid-normalized to roughly [0, 1] when supported.
    async fn rerank(&self, query: &str, texts: &[&str]) -> Result<Vec<f32>>;

    /// Number of candidates to rerank per query.
    fn top_n(&self) -> usize;
}
