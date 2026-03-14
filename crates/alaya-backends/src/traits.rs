use std::collections::HashMap;

use async_trait::async_trait;

use alaya_types::{
    Result,
    graph::{
        CoAccessPair, Contradiction, ContradictionRef, Direction, Edge, EdgeMeta, GraphStats,
        Neighbor, SystemRelationType, UserRelationType,
    },
    memory::{HealthStatus, Memory, MetadataUpdate, ScoredMemory, ScrollResult},
    search::{PayloadFilter, PromptName},
};

/// Vector storage backend (Qdrant REST API).
///
/// `?Send` bound because WASM is single-threaded.
#[async_trait(?Send)]
pub trait VectorStorage {
    // Core CRUD
    async fn store(&self, memory: &Memory) -> Result<(bool, String)>;
    async fn get_by_hash(&self, content_hash: &str) -> Result<Option<Memory>>;
    async fn get_batch(&self, hashes: &[&str]) -> Result<Vec<Memory>>;
    async fn delete(&self, content_hash: &str) -> Result<bool>;
    async fn update_metadata(&self, content_hash: &str, updates: MetadataUpdate) -> Result<()>;

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

    // Scroll / list
    async fn get_all(&self, limit: usize, offset: Option<&str>) -> Result<ScrollResult>;
    async fn get_recent(
        &self,
        limit: usize,
        offset: usize,
        memory_type: Option<&str>,
    ) -> Result<Vec<Memory>>;

    // Metadata
    async fn count(&self) -> Result<usize>;
    async fn get_all_tags(&self) -> Result<Vec<String>>;
    async fn increment_access_count(&self, content_hash: &str) -> Result<()>;
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
