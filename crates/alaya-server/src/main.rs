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

mod cached_embedding;
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
    summary::SummaryClient,
};
use alaya_core::deduplication::CanonicalStrategy;
use alaya_core::service::{MemoryService, OutputMode, RelationParams, SearchParams, StoreParams};
use alaya_types::memory::PatchMemoryRequest;
use alaya_types::search::PromptName;

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
    summary_url: Option<String>,
    summary_api_key: Option<String>,
    summary_model: String,
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
            summary_url: std::env::var("SUMMARY_URL").ok().filter(|s| !s.is_empty()),
            summary_api_key: std::env::var("SUMMARY_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            summary_model: env_or("SUMMARY_MODEL", "claude-haiku-4-5-20251001"),
        }
    }
}

fn env_required(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} is required"))
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

const L2_MAX_ATTEMPTS: u32 = 5;
const L2_BASE_RETRY_MS: u64 = 500;

async fn init_l2_cache(
    url: &str,
    model: &str,
    dims: usize,
) -> std::result::Result<cachekit::CacheKit, Box<dyn std::error::Error>> {
    let redis = cachekit::backend::redis::RedisBackend::builder()
        .url(url)
        .build()?;
    // Retry connection with backoff — at pod startup the CNI/kube-proxy may
    // not have finished installing network rules yet, causing ECONNREFUSED.
    let mut last_err = None;
    for attempt in 0..L2_MAX_ATTEMPTS {
        match redis.connect().await {
            Ok(handle) => {
                drop(handle);
                if attempt > 0 {
                    tracing::info!(attempt, "L2 cache connected after retry");
                }
                let ns = format!("alaya:embed:{model}:{dims}");
                return Ok(cachekit::CacheKit::builder()
                    .backend(std::rc::Rc::new(redis))
                    .namespace(ns)
                    .default_ttl(std::time::Duration::from_secs(86400 * 30))
                    .no_l1()
                    .build()?);
            }
            Err(e) => {
                tracing::debug!(attempt, error = %e, "L2 cache connect attempt failed");
                last_err = Some(e);
                if attempt + 1 < L2_MAX_ATTEMPTS {
                    tokio::time::sleep(std::time::Duration::from_millis(
                        L2_BASE_RETRY_MS * (1 << attempt),
                    ))
                    .await;
                }
            }
        }
    }
    Err(last_err
        .ok_or_else(|| "L2 cache init: no connection attempts made".to_string())?
        .into())
}

// ─── Command channel ────────────────────────────────────────────────────────

const CMD_CHANNEL_CAP: usize = 256;

/// A command sent from axum handlers to the MemoryService worker.
/// Carries the caller's tracing span so service methods become children
/// of the HTTP request span across the mpsc thread boundary.
pub(crate) struct Cmd {
    inner: CmdInner,
    span: tracing::Span,
}

pub(crate) enum CmdInner {
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
    GetMemory {
        hash: String,
        output: OutputMode,
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
    Patch {
        hash: String,
        patch: PatchMemoryRequest,
        reply: oneshot::Sender<Value>,
    },
    BackfillSummaries {
        limit: usize,
        reply: oneshot::Sender<Value>,
    },
}

impl Cmd {
    fn op_name(&self) -> &'static str {
        match &self.inner {
            CmdInner::Health { .. } => "health",
            CmdInner::Store { .. } => "store",
            CmdInner::Search { .. } => "search",
            CmdInner::Delete { .. } => "delete",
            CmdInner::GetMemory { .. } => "get_memory",
            CmdInner::Relation { .. } => "relation",
            CmdInner::Supersede { .. } => "supersede",
            CmdInner::Contradictions { .. } => "contradictions",
            CmdInner::FindDuplicates { .. } => "find_duplicates",
            CmdInner::MergeDuplicates { .. } => "merge_duplicates",
            CmdInner::Patch { .. } => "patch",
            CmdInner::BackfillSummaries { .. } => "backfill_summaries",
        }
    }
}

/// Handle for sending commands. Clone + Send + Sync (axum-compatible).
#[derive(Clone)]
pub(crate) struct ServiceHandle {
    pub(crate) tx: mpsc::Sender<Cmd>,
    api_key: String,
}

impl ServiceHandle {
    /// Non-blocking send. Returns error tuple suitable for both REST and MCP paths.
    fn try_dispatch(&self, cmd: Cmd) -> Result<(), (i32, String)> {
        self.tx.try_send(cmd).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => {
                tracing::warn!(capacity = CMD_CHANNEL_CAP, "command channel full");
                (-32000, "Service overloaded, try again later".to_string())
            }
            mpsc::error::TrySendError::Closed(_) => (-32000, "Service unavailable".to_string()),
        })
    }

    async fn call(&self, inner: CmdInner, rx: oneshot::Receiver<Value>) -> Json<Value> {
        let cmd = Cmd {
            inner,
            span: tracing::Span::current(),
        };
        if let Err((_code, msg)) = self.try_dispatch(cmd) {
            return Json(json!({"error": msg}));
        }
        match rx.await {
            Ok(v) => Json(v),
            Err(_) => Json(json!({"error": "service dropped response"})),
        }
    }
}

/// Direct health checker — bypasses the service worker channel so health
/// probes don't queue behind long-running operations (find_duplicates, etc).
/// Uses its own reqwest::Client (Clone + Send + Sync) on the axum runtime.
#[derive(Clone)]
struct HealthChecker {
    client: reqwest::Client,
    qdrant_url: String,
    collection: String,
    graph_url: String,
    graph_api_key: String,
}

impl HealthChecker {
    fn new(config: &Config) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref key) = config.qdrant_api_key
            && let Ok(val) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
        {
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build health check client");

        Self {
            client,
            qdrant_url: config.qdrant_url.clone(),
            collection: config.qdrant_collection.clone(),
            graph_url: config.graph_url.clone(),
            graph_api_key: config.graph_api_key.clone(),
        }
    }

    async fn check(&self) -> Value {
        let start = std::time::Instant::now();

        // All three checks run concurrently via tokio::join!
        let (qdrant_health, graph_health, count) =
            tokio::join!(self.check_qdrant(), self.check_graph(), self.check_count(),);

        let status = if qdrant_health.is_ok() {
            "healthy"
        } else {
            "degraded"
        };

        let elapsed = start.elapsed().as_millis();
        tracing::debug!(op = "health", elapsed_ms = elapsed, status, "ok (direct)");

        json!({
            "status": status,
            "version": option_env!("ALAYA_GIT_SHA").unwrap_or("dev"),
            "backend": "qdrant",
            "vector_health": match qdrant_health {
                Ok(v) => v,
                Err(e) => json!({"status": "unhealthy", "error": e}),
            },
            "graph_health": match graph_health {
                Ok(v) => v,
                Err(e) => json!({"status": "unhealthy", "error": e}),
            },
            "total_memories": count.unwrap_or(0),
        })
    }

    async fn check_qdrant(&self) -> Result<Value, String> {
        let resp = self
            .client
            .get(format!(
                "{}/collections/{}",
                self.qdrant_url, self.collection
            ))
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!("qdrant returned {}", resp.status()));
        }

        let body: Value = resp.json().await.map_err(|e| e.to_string())?;
        let status = body
            .pointer("/result/status")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");
        let points = body
            .pointer("/result/points_count")
            .and_then(|n| n.as_u64())
            .unwrap_or(0);

        Ok(json!({
            "status": status,
            "backend": "qdrant",
            "details": { "points_count": points },
        }))
    }

    async fn check_graph(&self) -> Result<Value, String> {
        let mut req = self.client.get(format!("{}/stats", self.graph_url));
        if !self.graph_api_key.is_empty() {
            req = req.bearer_auth(&self.graph_api_key);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!("bridge returned {}", resp.status()));
        }

        let body: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(body)
    }

    async fn check_count(&self) -> Result<usize, String> {
        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/count",
                self.qdrant_url, self.collection
            ))
            .json(&json!({}))
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let body: Value = resp.json().await.map_err(|e| e.to_string())?;
        Ok(body
            .pointer("/result/count")
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as usize)
    }
}

// ─── Service worker ─────────────────────────────────────────────────────────

/// Runs MemoryService on a LocalSet, processing commands from the channel.
///
/// Wraps the service in `Rc` so long-running operations (find_duplicates,
/// merge_duplicates) can be spawned as local tasks without blocking the
/// command loop. Other commands continue processing while they run.
async fn service_worker(mut rx: mpsc::Receiver<Cmd>, svc: MemoryService) {
    use tracing::Instrument;

    let svc = std::rc::Rc::new(svc);

    while let Some(cmd) = rx.recv().await {
        let op = cmd.op_name();
        let start = std::time::Instant::now();
        let ps = cmd.span;

        match cmd.inner {
            CmdInner::Health { reply } => {
                let span = tracing::info_span!(parent: &ps, "health");
                let result = match svc.check_database_health().instrument(span).await {
                    Ok(r) => {
                        tracing::debug!(op, elapsed_ms = ms(start), "ok");
                        json!(r)
                    }
                    Err(e) => {
                        log_err(op, &e, start);
                        json!({"status": "error", "message": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            CmdInner::Store { params, reply } => {
                let content_len = params.content.len();
                let mem_type = params.memory_type.as_deref().unwrap_or("note").to_string();
                let tag_count = params.tags.as_ref().map(|t| t.len()).unwrap_or(0);
                let has_dedup = params.dedup_threshold.is_some();
                let client = params.client_hostname.clone().unwrap_or_default();

                // Capture for fire-and-forget summary generation
                let needs_summary = params.summary.is_none() && svc.summary.is_some();
                let content_for_summary = if needs_summary {
                    Some(params.content.clone())
                } else {
                    None
                };

                let span = tracing::info_span!(parent: &ps, "store",
                    content_len, %mem_type);
                let result = match svc.store_memory(params).instrument(span).await {
                    Ok(r) => {
                        let hash = r
                            .get("content_hash")
                            .and_then(|v| v.as_str())
                            .map(|s| &s[..8.min(s.len())])
                            .unwrap_or("-");
                        let skipped = r
                            .get("duplicate")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        tracing::info!(
                            op,
                            hash,
                            mem_type = mem_type.as_str(),
                            content_len,
                            tag_count,
                            has_dedup,
                            skipped,
                            client = client.as_str(),
                            elapsed_ms = ms(start),
                            "ok"
                        );

                        // Fire-and-forget: generate summary in background
                        if let Some(content) = content_for_summary
                            && !skipped
                            && let Some(full_hash) = r.get("content_hash").and_then(|v| v.as_str())
                        {
                            let hash_owned = full_hash.to_string();
                            let svc = svc.clone();
                            tokio::task::spawn_local(async move {
                                enrich_summary(&svc, &hash_owned, &content).await;
                            });
                        }

                        json!(r)
                    }
                    Err(e) => {
                        log_err(op, &e, start);
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            CmdInner::Search { params, reply } => {
                let mode = format!("{:?}", params.mode).to_lowercase();
                let query_preview = truncate(&params.query, 80);
                let page = params.page;
                let page_size = params.page_size;
                let tag_count = params.tags.as_ref().map(|t| t.len()).unwrap_or(0);
                let mem_type = params.memory_type.clone().unwrap_or_default();
                let span = tracing::info_span!(parent: &ps, "search", %mode);
                let result = match svc.search(params).instrument(span).await {
                    Ok(r) => {
                        let n = result_count(&r);
                        let has_more = r.get("has_more").and_then(|v| v.as_bool()).unwrap_or(false);
                        tracing::info!(
                            op,
                            mode = mode.as_str(),
                            query = query_preview.as_str(),
                            results = n,
                            has_more,
                            page,
                            page_size,
                            tag_count,
                            mem_type = mem_type.as_str(),
                            elapsed_ms = ms(start),
                            "ok"
                        );
                        r
                    }
                    Err(e) => {
                        log_err(op, &e, start);
                        json!({"error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            CmdInner::Delete { hash, reply } => {
                let span = tracing::info_span!(parent: &ps, "delete");
                let h = truncate_hash(&hash);
                let result = match svc.delete_memory(&hash).instrument(span).await {
                    Ok(r) => {
                        tracing::info!(op, hash = h.as_str(), elapsed_ms = ms(start), "ok");
                        json!(r)
                    }
                    Err(e) => {
                        log_err(op, &e, start);
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            CmdInner::GetMemory {
                hash,
                output,
                reply,
            } => {
                let span = tracing::info_span!(parent: &ps, "get_memory");
                let h = truncate_hash(&hash);
                let result = match svc.get_memory(&hash, output).instrument(span).await {
                    Ok(r) => {
                        let found = r.get("found").and_then(|f| f.as_bool()).unwrap_or(false);
                        tracing::info!(op, hash = h.as_str(), found, elapsed_ms = ms(start), "ok");
                        r
                    }
                    Err(e) => {
                        log_err(op, &e, start);
                        json!({"error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            CmdInner::Relation { params, reply } => {
                let action = params.action.clone();
                let hash = truncate_hash(&params.content_hash);
                let target = params
                    .target_hash
                    .as_deref()
                    .map(truncate_hash)
                    .unwrap_or_default();
                let rel_type = params.relation_type.clone().unwrap_or_default();
                let span = tracing::info_span!(parent: &ps, "relation", %action);
                let result = match svc.relation(params).instrument(span).await {
                    Ok(r) => {
                        tracing::info!(
                            op,
                            action = action.as_str(),
                            hash = hash.as_str(),
                            target = target.as_str(),
                            rel_type = rel_type.as_str(),
                            elapsed_ms = ms(start),
                            "ok"
                        );
                        r
                    }
                    Err(e) => {
                        log_err(op, &e, start);
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            CmdInner::Supersede {
                old_hash,
                new_hash,
                reason,
                reply,
            } => {
                let span = tracing::info_span!(parent: &ps, "supersede");
                let old_h = truncate_hash(&old_hash);
                let new_h = truncate_hash(&new_hash);
                let reason_preview = truncate(&reason, 60);
                let result = match svc
                    .memory_supersede(&old_hash, &new_hash, &reason)
                    .instrument(span)
                    .await
                {
                    Ok(r) => {
                        tracing::info!(
                            op,
                            old = old_h.as_str(),
                            new = new_h.as_str(),
                            reason = reason_preview.as_str(),
                            elapsed_ms = ms(start),
                            "ok"
                        );
                        r
                    }
                    Err(e) => {
                        tracing::error!(
                            op,
                            error = %e,
                            old = old_h.as_str(),
                            new = new_h.as_str(),
                            old_len = old_hash.len(),
                            new_len = new_hash.len(),
                            elapsed_ms = ms(start),
                            "failed"
                        );
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            CmdInner::Contradictions { limit, reply } => {
                let span = tracing::info_span!(parent: &ps, "contradictions");
                let result = match svc.memory_contradictions(limit).instrument(span).await {
                    Ok(r) => {
                        let pairs = r
                            .get("contradictions")
                            .and_then(|v| v.as_array())
                            .map(|a| a.len())
                            .unwrap_or(0);
                        tracing::info!(op, limit, pairs, elapsed_ms = ms(start), "ok");
                        r
                    }
                    Err(e) => {
                        log_err(op, &e, start);
                        json!({"success": false, "error": e.safe_message()})
                    }
                };
                let _ = reply.send(result);
            }
            CmdInner::Patch { hash, patch, reply } => {
                let span = tracing::info_span!(parent: &ps, "patch");
                let h = truncate_hash(&hash);
                let fields = patch.changed_fields();
                let result = match svc.patch_memory(&hash, &patch).instrument(span).await {
                    Ok(mem) => {
                        tracing::info!(
                            op,
                            hash = h.as_str(),
                            fields = fields.as_str(),
                            elapsed_ms = ms(start),
                            "ok"
                        );
                        json!(mem)
                    }
                    Err(e) => {
                        log_err(op, &e, start);
                        json!({
                            "error": e.safe_message(),
                            "error_kind": match &e {
                                alaya_types::AlayaError::NotFound(_) => "not_found",
                                alaya_types::AlayaError::Validation(_) => "validation",
                                _ => "internal",
                            }
                        })
                    }
                };
                let _ = reply.send(result);
            }

            // ── Long-running ops: spawned as local tasks to avoid blocking ──
            CmdInner::FindDuplicates {
                threshold,
                limit,
                strategy,
                reply,
            } => {
                let strat_name = format!("{strategy:?}").to_lowercase();
                let span = tracing::info_span!(parent: &ps, "find_duplicates");
                let svc = svc.clone();
                tokio::task::spawn_local(
                    async move {
                        let result = match svc.find_duplicates(threshold, limit, strategy).await {
                            Ok(r) => {
                                let n = r
                                    .get("total_duplicates_found")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let groups = r
                                    .get("duplicate_groups")
                                    .and_then(|v| v.as_array())
                                    .map(|a| a.len())
                                    .unwrap_or(0);
                                let scanned =
                                    r.get("raw_scanned").and_then(|v| v.as_u64()).unwrap_or(0);
                                tracing::info!(
                                    op,
                                    duplicates = n,
                                    groups,
                                    scanned,
                                    threshold,
                                    strategy = strat_name.as_str(),
                                    elapsed_ms = ms(start),
                                    "ok"
                                );
                                r
                            }
                            Err(e) => {
                                log_err(op, &e, start);
                                json!({"success": false, "error": e.safe_message()})
                            }
                        };
                        let _ = reply.send(result);
                    }
                    .instrument(span),
                );
            }
            CmdInner::MergeDuplicates {
                canonical,
                duplicates,
                reason,
                dry_run,
                reply,
            } => {
                let span = tracing::info_span!(parent: &ps, "merge_duplicates");
                let svc = svc.clone();
                let dup_count = duplicates.len();
                let canonical_h = truncate_hash(&canonical);
                tokio::task::spawn_local(
                    async move {
                        let refs: Vec<&str> = duplicates.iter().map(|s| s.as_str()).collect();
                        let result = match svc
                            .merge_duplicates(&canonical, &refs, &reason, dry_run)
                            .await
                        {
                            Ok(r) => {
                                tracing::info!(
                                    op,
                                    canonical = canonical_h.as_str(),
                                    dup_count,
                                    dry_run,
                                    elapsed_ms = ms(start),
                                    "ok"
                                );
                                r
                            }
                            Err(e) => {
                                log_err(op, &e, start);
                                json!({"success": false, "error": e.safe_message()})
                            }
                        };
                        let _ = reply.send(result);
                    }
                    .instrument(span),
                );
            }
            CmdInner::BackfillSummaries { limit, reply } => {
                let span = tracing::info_span!(parent: &ps, "backfill_summaries");
                let svc = svc.clone();
                tokio::task::spawn_local(
                    async move {
                        if svc.summary.is_none() {
                            let _ = reply.send(json!({"error": "summary provider not configured"}));
                            return;
                        }

                        // Scroll memories, collect those without summaries
                        let mut offset: Option<String> = None;
                        let mut targets: Vec<(String, String)> = Vec::new();

                        loop {
                            match svc.vectors.get_all(100, offset.as_deref()).await {
                                Ok(scroll) => {
                                    for mem in &scroll.memories {
                                        if mem.summary.is_none() && targets.len() < limit {
                                            targets.push((
                                                mem.content_hash.clone(),
                                                mem.content.clone(),
                                            ));
                                        }
                                    }
                                    if scroll.next_offset.is_none() || targets.len() >= limit {
                                        break;
                                    }
                                    offset = scroll.next_offset;
                                }
                                Err(e) => {
                                    tracing::warn!("backfill scroll failed: {e}");
                                    break;
                                }
                            }
                        }

                        let queued = targets.len();
                        tracing::info!(queued, "backfill: generating summaries");
                        let _ = reply.send(json!({"queued": queued}));

                        // Generate summaries sequentially (avoid API rate limits)
                        for (hash, content) in &targets {
                            enrich_summary(&svc, hash, content).await;
                        }
                        tracing::info!(queued, "backfill summaries complete");
                    }
                    .instrument(span),
                );
            }
        }
    }
}

/// Fire-and-forget summary generation helper.
/// Called from spawn_local — logs errors, never panics.
/// Generates summary text AND its embedding for search boosting.
async fn enrich_summary(svc: &MemoryService, hash: &str, content: &str) {
    let Some(ref summarizer) = svc.summary else {
        return;
    };
    let h = &hash[..8.min(hash.len())];
    match summarizer.summarize(content).await {
        Ok(summary) => {
            // Embed the summary for search boost (non-fatal if it fails)
            let summary_embedding = match svc
                .embeddings
                .embed_batch(&[summary.as_str()], PromptName::Passage)
                .await
            {
                Ok(mut embs) if !embs.is_empty() => Some(embs.remove(0)),
                Ok(_) => {
                    tracing::warn!(hash = h, "summary embedding returned empty (non-fatal)");
                    None
                }
                Err(e) => {
                    tracing::warn!(hash = h, "summary embedding failed (non-fatal): {e}");
                    None
                }
            };

            let patch = PatchMemoryRequest {
                summary: Some(summary),
                summary_embedding,
                ..Default::default()
            };
            if let Err(e) = svc.patch_memory(hash, &patch).await {
                tracing::warn!(hash = h, "summary patch failed (non-fatal): {e}");
            } else {
                tracing::debug!(hash = h, "auto-summary + embedding applied");
            }
        }
        Err(e) => {
            tracing::warn!(hash = h, "summary generation failed (non-fatal): {e}");
        }
    }
}

fn ms(start: std::time::Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

fn result_count(v: &Value) -> u64 {
    v.get("results")
        .and_then(|r| r.as_array())
        .map(|a| a.len() as u64)
        .unwrap_or(0)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let boundary = s.floor_char_boundary(max);
        format!("{}…", &s[..boundary])
    }
}

fn truncate_hash(s: &str) -> String {
    s[..8.min(s.len())].to_string()
}

fn log_err(op: &str, e: &alaya_types::AlayaError, start: std::time::Instant) {
    tracing::error!(op, error = %e, elapsed_ms = ms(start), "failed");
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

        let (tx, rx) = mpsc::channel::<Cmd>(CMD_CHANNEL_CAP);

        // Spawn MemoryService on a dedicated thread with LocalSet
        let cfg_clone = config.clone();

        let worker_handle = std::thread::spawn(move || {
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
                let embed_model = cfg_clone.embedding_model;
                let embed_dims = cfg_clone.embedding_dimensions;
                let embeddings = EmbeddingClient::new(
                    cfg_clone.embedding_url,
                    embed_model.clone(),
                    embed_dims,
                    None,
                );
                // L2 embedding cache: Redis via cachekit-rs (optional)
                let l2_cache = if let Ok(url) = std::env::var("REDIS_CACHE_URL") {
                    match init_l2_cache(&url, &embed_model, embed_dims).await {
                        Ok(ck) => {
                            tracing::info!("L2 embedding cache enabled (Redis)");
                            Some(ck)
                        }
                        Err(e) => {
                            tracing::warn!("L2 cache init failed, running L1-only: {e}");
                            None
                        }
                    }
                } else {
                    tracing::info!("REDIS_CACHE_URL not set — L1-only embedding cache");
                    None
                };
                let cached_embeddings = cached_embedding::CachedEmbedding::new(
                    Box::new(embeddings),
                    10_000, // L1 max cached embeddings (~40 MB at 1024 dims)
                    l2_cache,
                );
                let graph = std::rc::Rc::new(GraphHttpClient::new(
                    cfg_clone.graph_url,
                    &cfg_clone.graph_api_key,
                ));

                let summary: Option<Box<dyn alaya_backends::SummaryProvider>> =
                    if let Some(url) = &cfg_clone.summary_url {
                        tracing::info!(
                            url = url.as_str(),
                            model = cfg_clone.summary_model.as_str(),
                            has_api_key = cfg_clone.summary_api_key.is_some(),
                            "summary provider enabled"
                        );
                        Some(Box::new(SummaryClient::new(
                            url.clone(),
                            cfg_clone.summary_model.clone(),
                            cfg_clone.summary_api_key.clone(),
                        )))
                    } else {
                        tracing::info!("SUMMARY_URL not set — auto-summary disabled");
                        None
                    };

                let svc = MemoryService::new(
                    Box::new(qdrant),
                    Box::new(cached_embeddings),
                    Box::new(GraphRef(graph.clone())),
                    Box::new(HebbianRef(graph.clone())),
                    Box::new(ConsolidationRef(graph)),
                    summary,
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

        // Health checker bypasses the service worker channel entirely —
        // runs directly on the multi-threaded axum runtime with its own
        // reqwest::Client. Prevents health probe timeouts during long ops.
        let checker = HealthChecker::new(&config);

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
            .route(
                "/memories/{content_hash}",
                get(get_memory).patch(patch_memory),
            )
            .route("/backfill/summaries", post(backfill_summaries))
            .layer(middleware::from_fn_with_state(
                handle.clone(),
                require_bearer,
            ))
            .layer(TraceLayer::new_for_http().make_span_with(
                tower_http::trace::DefaultMakeSpan::new().level(tracing::Level::INFO),
            ));

        let health_route = Router::new()
            .route("/health", get(health))
            .with_state(checker);

        let app = health_route.merge(protected.with_state(handle));

        let listener = tokio::net::TcpListener::bind(&config.listen_addr)
            .await
            .expect("failed to bind");

        tracing::info!("alaya-server listening on {}", config.listen_addr);

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .expect("server error");

        // axum returned — all in-flight requests done, router (and its
        // ServiceHandle/Sender clones) dropped. The worker's rx.recv()
        // will return None after processing any remaining queued commands.
        tracing::info!("waiting for service worker to drain…");
        let _ = worker_handle.join();
        tracing::info!("service worker drained");

        // Flush OTLP spans
        telemetry::shutdown_tracing();
        tracing::info!("shutdown complete");
    });
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");

    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT, shutting down…"),
        _ = sigterm.recv() => tracing::info!("received SIGTERM, shutting down…"),
    }
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health(axum::extract::State(checker): axum::extract::State<HealthChecker>) -> Json<Value> {
    Json(checker.check().await)
}

async fn store(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<StoreParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(CmdInner::Store { params, reply: tx }, rx).await
}

async fn search(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<SearchParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(CmdInner::Search { params, reply: tx }, rx).await
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
        CmdInner::Delete {
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
    h.call(CmdInner::Relation { params, reply: tx }, rx).await
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
        CmdInner::Supersede {
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
        CmdInner::Contradictions {
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
    500
}

async fn find_duplicates(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(req): Json<FindDupReq>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(
        CmdInner::FindDuplicates {
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
        CmdInner::MergeDuplicates {
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

async fn patch_memory(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    axum::extract::Path(content_hash): axum::extract::Path<String>,
    Json(patch): Json<PatchMemoryRequest>,
) -> (StatusCode, Json<Value>) {
    if patch.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "at least one field must be provided"})),
        );
    }

    if !alaya_types::memory::validate_content_hash(&content_hash) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid content_hash format"})),
        );
    }

    if let Err(msg) = patch.validate() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": msg})));
    }

    let (tx, rx) = oneshot::channel();
    let cmd = Cmd {
        inner: CmdInner::Patch {
            hash: content_hash,
            patch,
            reply: tx,
        },
        span: tracing::Span::current(),
    };

    if h.tx.send(cmd).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "service unavailable"})),
        );
    }

    match rx.await {
        Ok(v) => match v.get("error_kind").and_then(|k| k.as_str()) {
            Some("not_found") => (StatusCode::NOT_FOUND, Json(v)),
            Some("validation") => (StatusCode::BAD_REQUEST, Json(v)),
            Some(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(v)),
            None => (StatusCode::OK, Json(v)),
        },
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "service dropped response"})),
        ),
    }
}

#[derive(Deserialize)]
struct GetMemoryQuery {
    #[serde(default)]
    output: OutputMode,
}

async fn get_memory(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    axum::extract::Path(content_hash): axum::extract::Path<String>,
    axum::extract::Query(q): axum::extract::Query<GetMemoryQuery>,
) -> (StatusCode, Json<Value>) {
    if !alaya_types::memory::validate_content_hash(&content_hash) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid content_hash format"})),
        );
    }

    let (tx, rx) = oneshot::channel();
    let cmd = Cmd {
        inner: CmdInner::GetMemory {
            hash: content_hash,
            output: q.output,
            reply: tx,
        },
        span: tracing::Span::current(),
    };

    if h.tx.send(cmd).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "service unavailable"})),
        );
    }

    match rx.await {
        Ok(v) => {
            if v.get("error").is_some() {
                (StatusCode::INTERNAL_SERVER_ERROR, Json(v))
            } else if v.get("found").and_then(|f| f.as_bool()).unwrap_or(false) {
                (StatusCode::OK, Json(v))
            } else {
                (StatusCode::NOT_FOUND, Json(v))
            }
        }
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "service dropped response"})),
        ),
    }
}

#[derive(Deserialize)]
struct BackfillParams {
    #[serde(default = "default_backfill_limit")]
    limit: usize,
}
fn default_backfill_limit() -> usize {
    100
}

async fn backfill_summaries(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<BackfillParams>,
) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    h.call(
        CmdInner::BackfillSummaries {
            limit: params.limit,
            reply: tx,
        },
        rx,
    )
    .await
}
