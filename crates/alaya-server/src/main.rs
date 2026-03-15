//! alaya-server — Native REST API wrapping MemoryService.
//!
//! Uses a channel-based architecture: axum handlers send requests to a
//! MemoryService running on a LocalSet (single-threaded, ?Send compatible).
//! This bridges axum's Send+Sync requirement with the WASM-compat traits.

use axum::{
    Json, Router,
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tower_http::trace::TraceLayer;

use alaya_backends::{embedding::EmbeddingClient, graph::GraphHttpClient, qdrant::QdrantClient};
use alaya_core::deduplication::CanonicalStrategy;
use alaya_core::service::{MemoryService, RelationParams, SearchParams, StoreParams};

// ─── Config ─────────────────────────────────────────────────────────────────

struct Config {
    qdrant_url: String,
    qdrant_collection: String,
    qdrant_api_key: Option<String>,
    embedding_url: String,
    embedding_model: String,
    embedding_dimensions: usize,
    graph_url: String,
    graph_api_key: String,
    listen_addr: String,
}

impl Config {
    fn from_env() -> Self {
        Self {
            qdrant_url: env_required("QDRANT_URL"),
            qdrant_collection: env_or("QDRANT_COLLECTION", "memories_arctic1024"),
            qdrant_api_key: std::env::var("QDRANT_API_KEY").ok(),
            embedding_url: env_required("EMBEDDING_URL"),
            embedding_model: env_or("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-l-v2.0"),
            embedding_dimensions: env_or("EMBEDDING_DIMENSIONS", "1024")
                .parse()
                .expect("EMBEDDING_DIMENSIONS must be a number"),
            graph_url: env_required("GRAPH_URL"),
            graph_api_key: env_or("GRAPH_API_KEY", ""),
            listen_addr: env_or("LISTEN_ADDR", "0.0.0.0:3001"),
        }
    }
}

fn env_required(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} is required"))
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

// ─── Command channel ────────────────────────────────────────────────────────

/// A command sent from axum handlers to the MemoryService worker.
enum Cmd {
    Health(oneshot::Sender<Value>),
    Store(StoreParams, oneshot::Sender<Value>),
    Search(SearchParams, oneshot::Sender<Value>),
    Delete(String, oneshot::Sender<Value>),
    Relation(RelationParams, oneshot::Sender<Value>),
    Supersede(String, String, String, oneshot::Sender<Value>),
    Contradictions(usize, oneshot::Sender<Value>),
    FindDuplicates(f64, usize, CanonicalStrategy, oneshot::Sender<Value>),
    MergeDuplicates(String, Vec<String>, String, bool, oneshot::Sender<Value>),
}

/// Handle for sending commands. Clone + Send + Sync (axum-compatible).
#[derive(Clone)]
struct ServiceHandle {
    tx: mpsc::Sender<Cmd>,
}

impl ServiceHandle {
    async fn call(&self, cmd: Cmd, rx: oneshot::Receiver<Value>) -> Json<Value> {
        if self.tx.send(cmd).await.is_err() {
            return Json(json!({"error": "service unavailable"}));
        }
        match rx.await {
            Ok(v) => Json(v),
            Err(_) => Json(json!({"error": "service dropped response"})),
        }
    }
}

// ─── Service worker ─────────────────────────────────────────────────────────

/// Runs MemoryService on a LocalSet, processing commands from the channel.
async fn service_worker(mut rx: mpsc::Receiver<Cmd>, svc: MemoryService) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            Cmd::Health(reply) => {
                let result = match svc.check_database_health().await {
                    Ok(r) => json!(r),
                    Err(e) => json!({"status": "error", "message": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
            Cmd::Store(params, reply) => {
                let result = match svc.store_memory(params).await {
                    Ok(r) => json!(r),
                    Err(e) => json!({"success": false, "error": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
            Cmd::Search(params, reply) => {
                let result = match svc.search(params).await {
                    Ok(r) => r,
                    Err(e) => json!({"error": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
            Cmd::Delete(hash, reply) => {
                let result = match svc.delete_memory(&hash).await {
                    Ok(r) => json!(r),
                    Err(e) => json!({"success": false, "error": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
            Cmd::Relation(params, reply) => {
                let result = match svc.relation(params).await {
                    Ok(r) => r,
                    Err(e) => json!({"success": false, "error": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
            Cmd::Supersede(old, new, reason, reply) => {
                let result = match svc.memory_supersede(&old, &new, &reason).await {
                    Ok(r) => r,
                    Err(e) => json!({"success": false, "error": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
            Cmd::Contradictions(limit, reply) => {
                let result = match svc.memory_contradictions(limit).await {
                    Ok(r) => r,
                    Err(e) => json!({"success": false, "error": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
            Cmd::FindDuplicates(threshold, limit, strategy, reply) => {
                let result = match svc.find_duplicates(threshold, limit, strategy).await {
                    Ok(r) => r,
                    Err(e) => json!({"success": false, "error": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
            Cmd::MergeDuplicates(canonical, dupes, reason, dry_run, reply) => {
                let refs: Vec<&str> = dupes.iter().map(|s| s.as_str()).collect();
                let result = match svc
                    .merge_duplicates(&canonical, &refs, &reason, dry_run)
                    .await
                {
                    Ok(r) => r,
                    Err(e) => json!({"success": false, "error": e.safe_message()}),
                };
                let _ = reply.send(result);
            }
        }
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "alaya_server=info".into()),
        )
        .init();

    let config = Config::from_env();

    // Multi-threaded runtime for axum; LocalSet thread for MemoryService
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build runtime");

    rt.block_on(async move {
        let (tx, rx) = mpsc::channel::<Cmd>(256);

        // Spawn MemoryService on a dedicated thread with LocalSet
        let cfg_clone = Config {
            qdrant_url: config.qdrant_url.clone(),
            qdrant_collection: config.qdrant_collection.clone(),
            qdrant_api_key: config.qdrant_api_key.clone(),
            embedding_url: config.embedding_url.clone(),
            embedding_model: config.embedding_model.clone(),
            embedding_dimensions: config.embedding_dimensions,
            graph_url: config.graph_url.clone(),
            graph_api_key: config.graph_api_key.clone(),
            listen_addr: config.listen_addr.clone(),
        };

        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build local runtime");

            let local = tokio::task::LocalSet::new();
            local.block_on(&rt, async move {
                let qdrant = QdrantClient::new(
                    cfg_clone.qdrant_url,
                    cfg_clone.qdrant_collection,
                    cfg_clone.qdrant_api_key,
                );
                let embeddings = EmbeddingClient::new(
                    cfg_clone.embedding_url,
                    cfg_clone.embedding_model,
                    cfg_clone.embedding_dimensions,
                    None,
                );
                let graph = std::rc::Rc::new(GraphHttpClient::new(
                    cfg_clone.graph_url,
                    &cfg_clone.graph_api_key,
                ));

                let svc = MemoryService::new(
                    Box::new(qdrant),
                    Box::new(embeddings),
                    Box::new(GraphRef(graph.clone())),
                    Box::new(HebbianRef(graph.clone())),
                    Box::new(ConsolidationRef(graph)),
                );

                service_worker(rx, svc).await;
            });
        });

        // Axum on the main multi-threaded runtime
        let handle = ServiceHandle { tx };

        let app = Router::new()
            .route("/health", get(health))
            .route("/store", post(store))
            .route("/search", post(search))
            .route("/delete", post(delete))
            .route("/relation", post(relation))
            .route("/supersede", post(supersede))
            .route("/contradictions", post(contradictions))
            .route("/duplicates/find", post(find_duplicates))
            .route("/duplicates/merge", post(merge_duplicates))
            .layer(TraceLayer::new_for_http())
            .with_state(handle);

        let listener = tokio::net::TcpListener::bind(&config.listen_addr)
            .await
            .expect("failed to bind");

        tracing::info!("alaya-server listening on {}", config.listen_addr);
        axum::serve(listener, app).await.expect("server error");
    });
}

// ─── Trait wrappers ─────────────────────────────────────────────────────────

struct GraphRef(std::rc::Rc<GraphHttpClient>);
struct HebbianRef(std::rc::Rc<GraphHttpClient>);
struct ConsolidationRef(std::rc::Rc<GraphHttpClient>);

macro_rules! delegate_graph {
    ($wrapper:ident) => {
        #[async_trait::async_trait(?Send)]
        impl alaya_backends::GraphService for $wrapper {
            async fn ensure_node(&self, h: &str, t: f64) -> alaya_types::Result<()> {
                self.0.ensure_node(h, t).await
            }
            async fn delete_node(&self, h: &str) -> alaya_types::Result<()> {
                self.0.delete_node(h).await
            }
            async fn create_typed_edge(
                &self,
                s: &str,
                d: &str,
                r: alaya_types::graph::UserRelationType,
                m: alaya_types::graph::EdgeMeta,
            ) -> alaya_types::Result<bool> {
                self.0.create_typed_edge(s, d, r, m).await
            }
            async fn get_typed_edges(
                &self,
                h: &str,
                r: Option<alaya_types::graph::UserRelationType>,
                d: alaya_types::graph::Direction,
                l: usize,
            ) -> alaya_types::Result<Vec<alaya_types::graph::Edge>> {
                self.0.get_typed_edges(h, r, d, l).await
            }
            async fn delete_typed_edge(
                &self,
                s: &str,
                d: &str,
                r: alaya_types::graph::UserRelationType,
            ) -> alaya_types::Result<bool> {
                self.0.delete_typed_edge(s, d, r).await
            }
            async fn create_system_edge(
                &self,
                s: &str,
                d: &str,
                r: alaya_types::graph::SystemRelationType,
                t: f64,
            ) -> alaya_types::Result<bool> {
                self.0.create_system_edge(s, d, r, t).await
            }
            async fn get_all_contradictions(
                &self,
                l: usize,
            ) -> alaya_types::Result<Vec<alaya_types::graph::Contradiction>> {
                self.0.get_all_contradictions(l).await
            }
            async fn get_contradictions_for_hashes(
                &self,
                h: &[&str],
            ) -> alaya_types::Result<
                std::collections::HashMap<String, Vec<alaya_types::graph::ContradictionRef>>,
            > {
                self.0.get_contradictions_for_hashes(h).await
            }
            async fn get_neighbors(
                &self,
                h: &str,
                hops: u8,
                w: f64,
                l: usize,
            ) -> alaya_types::Result<Vec<alaya_types::graph::Neighbor>> {
                self.0.get_neighbors(h, hops, w, l).await
            }
            async fn spreading_activation(
                &self,
                s: &[&str],
                hops: u8,
                d: f64,
                min: f64,
                l: usize,
            ) -> alaya_types::Result<std::collections::HashMap<String, f64>> {
                self.0.spreading_activation(s, hops, d, min, l).await
            }
            async fn hebbian_boosts_within(
                &self,
                h: &[&str],
            ) -> alaya_types::Result<std::collections::HashMap<String, f64>> {
                self.0.hebbian_boosts_within(h).await
            }
            async fn get_stats(&self) -> alaya_types::Result<alaya_types::graph::GraphStats> {
                self.0.get_stats().await
            }
        }
    };
}

delegate_graph!(GraphRef);

#[async_trait::async_trait(?Send)]
impl alaya_backends::HebbianService for HebbianRef {
    async fn enqueue_strengthen(
        &self,
        p: &[alaya_types::graph::CoAccessPair],
    ) -> alaya_types::Result<()> {
        self.0.enqueue_strengthen(p).await
    }
}

#[async_trait::async_trait(?Send)]
impl alaya_backends::ConsolidationService for ConsolidationRef {
    async fn decay_all_edges(&self, f: f64, l: usize) -> alaya_types::Result<usize> {
        self.0.decay_all_edges(f, l).await
    }
    async fn decay_stale_edges(&self, b: f64, f: f64, l: usize) -> alaya_types::Result<usize> {
        self.0.decay_stale_edges(b, f, l).await
    }
    async fn prune_weak_edges(&self, t: f64, l: usize) -> alaya_types::Result<usize> {
        self.0.prune_weak_edges(t, l).await
    }
    async fn get_orphan_nodes(&self, l: usize) -> alaya_types::Result<Vec<String>> {
        self.0.get_orphan_nodes(l).await
    }
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health(axum::extract::State(h): axum::extract::State<ServiceHandle>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Health(tx), rx).await
}

async fn store(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<StoreParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Store(params, tx), rx).await
}

async fn search(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<SearchParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Search(params, tx), rx).await
}

#[derive(Deserialize)]
struct DeleteReq {
    content_hash: String,
}

async fn delete(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(req): Json<DeleteReq>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Delete(req.content_hash, tx), rx).await
}

async fn relation(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<RelationParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Relation(params, tx), rx).await
}

#[derive(Deserialize)]
struct SupersedeReq {
    old_hash: String,
    new_hash: String,
    #[serde(default)]
    reason: String,
}

async fn supersede(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(req): Json<SupersedeReq>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(
        Cmd::Supersede(req.old_hash, req.new_hash, req.reason, tx),
        rx,
    )
    .await
}

#[derive(Deserialize)]
struct ContradictionsReq {
    #[serde(default = "default_limit")]
    limit: usize,
}
fn default_limit() -> usize {
    20
}

async fn contradictions(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(req): Json<ContradictionsReq>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Contradictions(req.limit, tx), rx).await
}

#[derive(Deserialize)]
struct FindDupReq {
    #[serde(default = "default_threshold")]
    similarity_threshold: f64,
    #[serde(default = "default_dup_limit")]
    limit: usize,
    #[serde(default)]
    strategy: CanonicalStrategy,
}
fn default_threshold() -> f64 {
    0.95
}
fn default_dup_limit() -> usize {
    100
}

async fn find_duplicates(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(req): Json<FindDupReq>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(
        Cmd::FindDuplicates(req.similarity_threshold, req.limit, req.strategy, tx),
        rx,
    )
    .await
}

#[derive(Deserialize)]
struct MergeDupReq {
    canonical_hash: String,
    duplicate_hashes: Vec<String>,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    dry_run: bool,
}

async fn merge_duplicates(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(req): Json<MergeDupReq>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(
        Cmd::MergeDuplicates(
            req.canonical_hash,
            req.duplicate_hashes,
            req.reason,
            req.dry_run,
            tx,
        ),
        rx,
    )
    .await
}
