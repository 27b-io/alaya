//! alaya-server — Native REST + MCP server wrapping MemoryService.
//!
//! Uses a channel-based architecture: axum handlers send requests to a
//! MemoryService running on a LocalSet (single-threaded, ?Send compatible).
//! This bridges axum's Send+Sync requirement with the WASM-compat traits.
//!
//! Endpoints:
//!   POST /mcp          — MCP Streamable HTTP (JSON-RPC 2.0)
//!   POST /store, etc.  — Plain REST API (for Prajna and internal consumers)
//!   GET  /health       — Health check

mod mcp;
mod telemetry;

use axum::{
    Json, Router,
    extract::Request,
    http::{StatusCode, header},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tower_http::trace::TraceLayer;

use alaya_backends::{
    embedding::EmbeddingClient,
    graph::GraphHttpClient,
    graph_ref::{ConsolidationRef, GraphRef, HebbianRef},
    qdrant::QdrantClient,
};
use alaya_core::deduplication::CanonicalStrategy;
use alaya_core::service::{MemoryService, RelationParams, SearchParams, StoreParams};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
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
    api_key: String,
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
            api_key: env_or("ALAYA_API_KEY", ""),
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
pub(crate) enum Cmd {
    Health {
        reply: oneshot::Sender<Value>,
    },
    Store {
        params: StoreParams,
        reply: oneshot::Sender<Value>,
    },
    Search {
        params: SearchParams,
        reply: oneshot::Sender<Value>,
    },
    Delete {
        hash: String,
        reply: oneshot::Sender<Value>,
    },
    Relation {
        params: RelationParams,
        reply: oneshot::Sender<Value>,
    },
    Supersede {
        old_hash: String,
        new_hash: String,
        reason: String,
        reply: oneshot::Sender<Value>,
    },
    Contradictions {
        limit: usize,
        reply: oneshot::Sender<Value>,
    },
    FindDuplicates {
        threshold: f64,
        limit: usize,
        strategy: CanonicalStrategy,
        reply: oneshot::Sender<Value>,
    },
    MergeDuplicates {
        canonical: String,
        duplicates: Vec<String>,
        reason: String,
        dry_run: bool,
        reply: oneshot::Sender<Value>,
    },
}

/// Handle for sending commands. Clone + Send + Sync (axum-compatible).
#[derive(Clone)]
pub(crate) struct ServiceHandle {
    pub(crate) tx: mpsc::Sender<Cmd>,
    api_key: String,
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
            Cmd::Health { reply } => {
                let result = match svc.check_database_health().await {
                    Ok(r) => json!(r),
                    Err(e) => {
                        tracing::error!("check_database_health failed: {e:?}");
                        json!({"status": "error", "message": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            Cmd::Store { params, reply } => {
                let result = match svc.store_memory(params).await {
                    Ok(r) => json!(r),
                    Err(e) => {
                        tracing::error!("store_memory failed: {e:?}");
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            Cmd::Search { params, reply } => {
                let result = match svc.search(params).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("search failed: {e:?}");
                        json!({"error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            Cmd::Delete { hash, reply } => {
                let result = match svc.delete_memory(&hash).await {
                    Ok(r) => json!(r),
                    Err(e) => {
                        tracing::error!("delete_memory failed: {e:?}");
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            Cmd::Relation { params, reply } => {
                let result = match svc.relation(params).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("relation failed: {e:?}");
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            Cmd::Supersede {
                old_hash,
                new_hash,
                reason,
                reply,
            } => {
                let result = match svc.memory_supersede(&old_hash, &new_hash, &reason).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("memory_supersede failed: {e:?}");
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            Cmd::Contradictions { limit, reply } => {
                let result = match svc.memory_contradictions(limit).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("memory_contradictions failed: {e:?}");
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            Cmd::FindDuplicates {
                threshold,
                limit,
                strategy,
                reply,
            } => {
                let result = match svc.find_duplicates(threshold, limit, strategy).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("find_duplicates failed: {e:?}");
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            Cmd::MergeDuplicates {
                canonical,
                duplicates,
                reason,
                dry_run,
                reply,
            } => {
                let refs: Vec<&str> = duplicates.iter().map(|s| s.as_str()).collect();
                let result = match svc
                    .merge_duplicates(&canonical, &refs, &reason, dry_run)
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("merge_duplicates failed: {e:?}");
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
        }
    }
}

// ─── Auth middleware ────────────────────────────────────────────────────────

async fn require_bearer(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if h.api_key.is_empty() {
        return Ok(next.run(req).await);
    }

    let token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match token {
        Some(t) if t == h.api_key => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

fn main() {
    let config = Config::from_env();

    // Multi-threaded runtime for axum; LocalSet thread for MemoryService
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build runtime");

    rt.block_on(async move {
        telemetry::init_tracing();

        let (tx, rx) = mpsc::channel::<Cmd>(256);

        // Spawn MemoryService on a dedicated thread with LocalSet
        let cfg_clone = config.clone();

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
        let handle = ServiceHandle {
            tx,
            api_key: config.api_key.clone(),
        };

        if handle.api_key.is_empty() {
            tracing::warn!("ALAYA_API_KEY not set — all endpoints are unauthenticated");
        }

        let protected = Router::new()
            .route("/mcp", post(mcp::mcp_handler))
            .route("/store", post(store))
            .route("/search", post(search))
            .route("/delete", post(delete))
            .route("/relation", post(relation))
            .route("/supersede", post(supersede))
            .route("/contradictions", post(contradictions))
            .route("/duplicates/find", post(find_duplicates))
            .route("/duplicates/merge", post(merge_duplicates))
            .layer(middleware::from_fn_with_state(
                handle.clone(),
                require_bearer,
            ));

        let app = Router::new()
            .route("/health", get(health))
            .merge(protected)
            .layer(TraceLayer::new_for_http())
            .with_state(handle);

        let listener = tokio::net::TcpListener::bind(&config.listen_addr)
            .await
            .expect("failed to bind");

        tracing::info!("alaya-server listening on {}", config.listen_addr);
        axum::serve(listener, app).await.expect("server error");
    });
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health(axum::extract::State(h): axum::extract::State<ServiceHandle>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Health { reply: tx }, rx).await
}

async fn store(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<StoreParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Store { params, reply: tx }, rx).await
}

async fn search(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<SearchParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Search { params, reply: tx }, rx).await
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
    h.call(
        Cmd::Delete {
            hash: req.content_hash,
            reply: tx,
        },
        rx,
    )
    .await
}

async fn relation(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<RelationParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(Cmd::Relation { params, reply: tx }, rx).await
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
        Cmd::Supersede {
            old_hash: req.old_hash,
            new_hash: req.new_hash,
            reason: req.reason,
            reply: tx,
        },
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
    h.call(
        Cmd::Contradictions {
            limit: req.limit,
            reply: tx,
        },
        rx,
    )
    .await
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
        Cmd::FindDuplicates {
            threshold: req.similarity_threshold,
            limit: req.limit,
            strategy: req.strategy,
            reply: tx,
        },
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
        Cmd::MergeDuplicates {
            canonical: req.canonical_hash,
            duplicates: req.duplicate_hashes,
            reason: req.reason,
            dry_run: req.dry_run,
            reply: tx,
        },
        rx,
    )
    .await
}
