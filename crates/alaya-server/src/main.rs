//! alaya-server — Native REST + MCP server wrapping MemoryService.
//!
//! Uses a channel-based architecture: axum handlers send requests to a
//! MemoryService running on a LocalSet (single-threaded, ?Send compatible).
//! This bridges axum's Send+Sync requirement with the WASM-compat traits.
//!
//! Endpoints:
//!   POST /mcp          — MCP Streamable HTTP (JSON-RPC 2.0)
//!   POST /store, etc.  — Plain REST API (for Prajna and internal consumers)
//!   GET  /health       — Liveness probe (status only, unauthenticated)
//!   GET  /health/detail— Backend health, capacity, build identity (auth)

mod auth;
mod build_info;
mod cached_embedding;
mod mcp;
mod oidc;
mod telemetry;
#[cfg(test)]
mod testkit;
mod wellknown;

use axum::{
    Json, Router,
    extract::DefaultBodyLimit,
    http::{Method, StatusCode, header},
    middleware,
    routing::{get, post},
};
use tower_http::cors::CorsLayer;

use auth::{AuthPrincipal, AuthState, WritePolicy};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tower_http::trace::TraceLayer;

use alaya_backends::{
    embedding::EmbeddingClient,
    graph::GraphHttpClient,
    graph_ref::{ConsolidationRef, GraphRef, HebbianRef},
    qdrant::QdrantClient,
    rerank::RerankClient,
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
    embedding_batch_size: usize,
    graph_url: String,
    graph_api_key: String,
    listen_addr: String,
    api_key: String,
    oidc_issuer: Option<String>,
    public_base_url: String,
    allow_unauthenticated: bool,
    summary_url: Option<String>,
    summary_api_key: Option<String>,
    summary_model: String,
    rerank_url: Option<String>,
    rerank_api_key: Option<String>,
    rerank_top_n: usize,
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
            // Texts per /v1/embeddings request. Default 32 matches TEI's own
            // default max-client-batch-size; raise to match a TEI configured
            // higher (e.g. 256 on the fnord-wsl GPU box). Clamped to [1, 256].
            embedding_batch_size: env_or("EMBEDDING_BATCH_SIZE", "32")
                .parse()
                .expect("EMBEDDING_BATCH_SIZE must be a number"),
            graph_url: env_required("GRAPH_URL"),
            graph_api_key: env_or("GRAPH_API_KEY", ""),
            listen_addr: env_or("LISTEN_ADDR", "0.0.0.0:3001"),
            api_key: env_or("ALAYA_API_KEY", ""),
            oidc_issuer: std::env::var("OIDC_ISSUER").ok().filter(|s| !s.is_empty()),
            public_base_url: normalize_public_base_url(&env_or(
                "PUBLIC_BASE_URL",
                "https://alaya.27b.io",
            )),
            allow_unauthenticated: env_or("DANGEROUSLY_ALLOW_UNAUTHENTICATED", "")
                .eq_ignore_ascii_case("true"),
            summary_url: std::env::var("SUMMARY_URL").ok().filter(|s| !s.is_empty()),
            summary_api_key: std::env::var("SUMMARY_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            summary_model: env_or("SUMMARY_MODEL", "claude-haiku-4-5-20251001"),
            rerank_url: std::env::var("RERANK_URL").ok().filter(|s| !s.is_empty()),
            rerank_api_key: std::env::var("RERANK_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            rerank_top_n: env_or("RERANK_TOP_N", "20")
                .parse()
                .expect("RERANK_TOP_N must be a number"),
        }
    }
}

fn env_required(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} is required"))
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Normalize the single origin source-of-truth: strip a trailing slash and
/// require https (loopback may use http for dev). Aborts on a malformed value
/// since `aud`/`resource` derive from it.
fn normalize_public_base_url(raw: &str) -> String {
    let trimmed = raw.strip_suffix('/').unwrap_or(raw).to_string();
    let is_loopback = host_of(&trimmed)
        .as_deref()
        .map(host_is_loopback)
        .unwrap_or(false);
    if !trimmed.starts_with("https://") && !is_loopback {
        panic!("PUBLIC_BASE_URL must be https:// (except loopback): {raw}");
    }
    trimmed
}

/// True iff `host` is a real loopback target. Parses the host as an IP so
/// `127.0.0.1.evil.com` cannot masquerade as a 127.* literal — a string
/// `starts_with("127.")` check would let it through.
fn host_is_loopback(host: &str) -> bool {
    if host == "localhost" {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => false,
    }
}

/// Host (authority without port) of an absolute URL, lowercased.
/// Handles IPv6 bracketed literals (`[::1]:8443` → `::1`).
fn host_of(url: &str) -> Option<String> {
    let rest = url.split("://").nth(1)?;
    let authority = rest.split('/').next()?;
    let host = if let Some(inner) = authority.strip_prefix('[') {
        // IPv6 literal: take everything up to the closing bracket.
        let end = inner.find(']')?;
        &inner[..end]
    } else {
        authority.split(':').next()?
    };
    Some(host.to_ascii_lowercase())
}

/// True for hosts that are not publicly routable: loopback, RFC1918, or
/// cluster-internal (`.svc`, `.internal`). Used to forbid the dev-only open
/// mode on a public origin. Real IP-literal parsing prevents confusable
/// hostnames like `127.0.0.1.evil.com` from masquerading as loopback.
fn is_private_host(url: &str) -> bool {
    let Some(h) = host_of(url) else {
        return false;
    };
    // DNS-only special names — these can't be IP literals.
    if h == "localhost" || h.ends_with(".svc") || h.ends_with(".internal") {
        return true;
    }
    // Anything else must parse as an actual IP literal to qualify as private.
    match h.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => v4.is_loopback() || v4.is_private(),
        Ok(std::net::IpAddr::V6(v6)) => v6.is_loopback(),
        Err(_) => false,
    }
}

const L2_MAX_ATTEMPTS: u32 = 5;
const L2_BASE_RETRY_MS: u64 = 500;

async fn init_l2_redis(
    url: &str,
) -> std::result::Result<cachekit::CacheKit, Box<dyn std::error::Error>> {
    let redis = cachekit::backend::redis::RedisBackend::builder()
        .url(url)
        .build()?;
    // Retry connection with backoff — at pod startup the CNI/kube-proxy may
    // not have finished installing network rules yet, causing ECONNREFUSED.
    // Each attempt is deadline-bounded: cachekit sets no fred connect
    // timeout, and a blackholed target here would otherwise hang the worker
    // thread before its command loop ever starts (#63).
    let mut last_err: Option<String> = None;
    for attempt in 0..L2_MAX_ATTEMPTS {
        let connected = tokio::time::timeout(std::time::Duration::from_secs(10), redis.connect());
        let err_msg = match connected.await {
            Ok(Ok(handle)) => {
                drop(handle);
                if attempt > 0 {
                    tracing::info!(attempt, "L2 cache connected after retry");
                }
                return Ok(cached_embedding::build_l2_client(std::rc::Rc::new(redis))?);
            }
            Ok(Err(e)) => e.to_string(),
            Err(_) => "connect timed out after 10s".to_string(),
        };
        tracing::debug!(attempt, error = %err_msg, "L2 cache connect attempt failed");
        last_err = Some(err_msg);
        if attempt + 1 < L2_MAX_ATTEMPTS {
            tokio::time::sleep(std::time::Duration::from_millis(
                L2_BASE_RETRY_MS * (1 << attempt),
            ))
            .await;
        }
    }
    Err(last_err
        .unwrap_or_else(|| "L2 cache init: no connection attempts made".to_string())
        .into())
}

/// cachekit.io SaaS backend (`CACHE_BACKEND=saas`). HTTP — no connection to
/// retry; the builder validates the API key and URL (HTTPS-only, host
/// allowlist, private IPs blocked upstream).
fn init_l2_saas() -> std::result::Result<cachekit::CacheKit, Box<dyn std::error::Error>> {
    let api_key =
        env_non_empty("CACHEKIT_API_KEY").ok_or("CACHE_BACKEND=saas requires CACHEKIT_API_KEY")?;
    let mut builder = cachekit::backend::cachekitio::CachekitIO::builder().api_key(api_key);
    if let Some(url) = env_non_empty("CACHEKIT_API_URL") {
        builder = builder.api_url(url);
    }
    let backend = builder.build()?;
    Ok(cached_embedding::build_l2_client(std::rc::Rc::new(
        backend,
    ))?)
}

/// Env var treated as unset when blank, returned trimmed — k8s manifests
/// commonly ship `value: ""` (must route to the explicit "missing" handling,
/// not an opaque downstream builder error) and padded/newline-suffixed values
/// (folded YAML scalars, `echo`-piped secrets) that would fail string matches
/// and downstream builders if passed through raw.
fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// L2 embedding cache init, dispatched on `CACHE_BACKEND` (default `redis`).
/// Never fatal — any failure degrades to L1-only, matching the L2's
/// non-fatal-by-design posture.
async fn init_l2_cache() -> Option<cachekit::CacheKit> {
    let backend = env_non_empty("CACHE_BACKEND").unwrap_or_else(|| "redis".to_string());
    // A CacheKit API key alongside a non-saas backend is a near-certain
    // "forgot the switch" misconfig — say so instead of silently ignoring it.
    if backend != "saas" && env_non_empty("CACHEKIT_API_KEY").is_some() {
        tracing::warn!(
            backend = %backend,
            "CACHEKIT_API_KEY is set but CACHE_BACKEND is not \"saas\" — SaaS cache backend NOT in use"
        );
    }
    let init = match backend.as_str() {
        "redis" => {
            let Some(url) = env_non_empty("REDIS_CACHE_URL") else {
                tracing::info!("REDIS_CACHE_URL not set — L1-only embedding cache");
                return None;
            };
            init_l2_redis(&url).await
        }
        "saas" => init_l2_saas(),
        other => {
            tracing::warn!(
                value = other,
                "unknown CACHE_BACKEND (expected \"redis\" or \"saas\") — L1-only embedding cache"
            );
            return None;
        }
    };
    match init {
        Ok(ck) => {
            // Key-cutover announcement (LAB-372): interop/v1 keys replaced the
            // legacy SHA-256 keys on 2026-08-08, invalidating the warm cache.
            // Legacy entries (namespaced `alaya:embed:<model>:<dims>:<sha256>`)
            // are orphaned and expire via their 30-day TTL; flush them to
            // reclaim memory sooner (command in CLAUDE.md). One-time re-embed
            // cost until the cache re-warms. Log removable after 2026-09-08.
            tracing::info!(
                backend = %backend,
                "L2 embedding cache enabled — keys are cross-SDK interop/v1; legacy \
                 SHA-256 entries are orphaned and expire via TTL (cutover 2026-08-08, LAB-372)"
            );
            Some(ck)
        }
        Err(e) => {
            tracing::warn!(backend = %backend, "L2 cache init failed, running L1-only: {e}");
            None
        }
    }
}

/// Ensure the Qdrant collection exists before the server serves writes, so a
/// fresh deployment needs no manual bootstrap (#31). Retries with backoff: on a
/// cluster cold-start the server pod can come up before Qdrant is ready (same
/// rationale as `init_l2_cache`). Never fatal — after exhausting retries it logs
/// and continues rather than crash-looping the server; the first write would
/// then 404 until Qdrant is reachable, exactly as it did before this fix.
async fn ensure_qdrant_collection(qdrant: &QdrantClient, dimensions: usize) {
    for attempt in 0..L2_MAX_ATTEMPTS {
        match qdrant.ensure_collection(dimensions).await {
            Ok(()) => return,
            Err(e) if attempt + 1 == L2_MAX_ATTEMPTS => {
                tracing::error!(
                    error = %e,
                    "could not ensure Qdrant collection after {L2_MAX_ATTEMPTS} attempts — \
                     writes will 404 until Qdrant is reachable and the collection exists"
                );
            }
            Err(e) => {
                tracing::warn!(
                    attempt = attempt + 1,
                    error = %e,
                    "ensure Qdrant collection failed, retrying"
                );
                tokio::time::sleep(std::time::Duration::from_millis(
                    L2_BASE_RETRY_MS * (1 << attempt),
                ))
                .await;
            }
        }
    }
}

// ─── Command channel ────────────────────────────────────────────────────────

const CMD_CHANNEL_CAP: usize = 256;

// Wedge protection (#63): the worker serializes all ops through one channel,
// so a single await that never resolves used to freeze the whole service —
// invisibly, because /health bypasses the worker. Three layers fix that:
// per-command deadlines (worker drops a stuck handler and keeps draining),
// bounded reply awaits (callers get an error instead of an infinite hang),
// and a progress watchdog (a worker stuck in a way timeouts can't preempt —
// e.g. blocked in sync code — turns /health unhealthy so k8s restarts the pod).

/// Per-command budget for inline ops in the worker. Generous — legit ops
/// finish in seconds; only a stuck backend await ever gets here.
const CMD_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);
/// Budget for spawned long-running scans (find/merge duplicates, backfill).
const LONG_CMD_DEADLINE: std::time::Duration = std::time::Duration::from_secs(600);
/// Caller-side slack over the worker's own deadline (queue wait + scheduling).
const REPLY_MARGIN: std::time::Duration = std::time::Duration::from_secs(30);
/// Worker is considered stalled when no command has completed for this long.
/// Must exceed CMD_DEADLINE — a legit inline op may hold the loop that long.
const WORKER_STALL_THRESHOLD: std::time::Duration = std::time::Duration::from_secs(180);
/// Pinger period — keeps worker progress fresh when the service is idle.
const PING_PERIOD: std::time::Duration = std::time::Duration::from_secs(30);

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// A command sent from axum handlers to the MemoryService worker.
/// Carries the caller's tracing span so service methods become children
/// of the HTTP request span across the mpsc thread boundary.
pub(crate) struct Cmd {
    inner: CmdInner,
    span: tracing::Span,
}

pub(crate) enum CmdInner {
    /// No-op round-trip proving the worker loop is draining. Sent by the
    /// internal pinger (and tests) — not exposed over REST or MCP.
    Ping {
        reply: oneshot::Sender<Value>,
    },
    Health {
        reply: oneshot::Sender<Value>,
    },
    Store {
        params: StoreParams,
        read_only: bool,
        reply: oneshot::Sender<Value>,
    },
    Search {
        params: SearchParams,
        read_only: bool,
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

impl CmdInner {
    /// Worker-side execution budget for this command. Callers add
    /// REPLY_MARGIN on top when bounding their reply await.
    fn deadline(&self) -> std::time::Duration {
        match self {
            CmdInner::FindDuplicates { .. }
            | CmdInner::MergeDuplicates { .. }
            | CmdInner::BackfillSummaries { .. } => LONG_CMD_DEADLINE,
            _ => CMD_DEADLINE,
        }
    }
}

impl Cmd {
    fn op_name(&self) -> &'static str {
        match &self.inner {
            CmdInner::Ping { .. } => "ping",
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

    /// Dispatch a command and await its reply with a deadline. The bound is
    /// the command's worker budget plus queue-wait margin — callers can never
    /// hang unboundedly even if the worker itself is wedged (#63). Error tuple
    /// carries a JSON-RPC code so the MCP path can use it directly.
    pub(crate) async fn call_rpc(
        &self,
        inner: CmdInner,
        rx: oneshot::Receiver<Value>,
    ) -> Result<Value, (i32, String)> {
        let op = Cmd {
            inner,
            span: tracing::Span::current(),
        };
        let op_name = op.op_name();
        let reply_deadline = op.inner.deadline() + REPLY_MARGIN;
        self.try_dispatch(op)?;
        match tokio::time::timeout(reply_deadline, rx).await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(_)) => Err((-32000, "service dropped response".to_string())),
            Err(_) => {
                tracing::error!(
                    op = op_name,
                    deadline_s = reply_deadline.as_secs(),
                    "no reply from service worker within deadline — worker may be wedged"
                );
                Err((
                    -32000,
                    format!(
                        "service did not respond within {}s; the operation may still \
                         complete in the background",
                        reply_deadline.as_secs()
                    ),
                ))
            }
        }
    }

    /// REST wrapper over `call_rpc`. Op-level errors ride inside a 200 body
    /// (unchanged convention), but dispatch/transport failures — full channel,
    /// closed channel, no reply within deadline — are 503, matching
    /// `patch_memory`/`get_memory` so overload/stall semantics are uniform.
    async fn call(
        &self,
        inner: CmdInner,
        rx: oneshot::Receiver<Value>,
    ) -> (StatusCode, Json<Value>) {
        match self.call_rpc(inner, rx).await {
            Ok(v) => (StatusCode::OK, Json(v)),
            Err((_code, msg)) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": msg}))),
        }
    }
}

/// Direct health checker — bypasses the service worker channel so health
/// probes don't queue behind long-running operations (find_duplicates, etc).
/// Uses its own reqwest::Client (Clone + Send + Sync) on the axum runtime.
///
/// Bypassing the worker made a wedged worker invisible to k8s (#63), so the
/// checker also watches `worker_progress` — the epoch-seconds of the last
/// command the worker completed (pings keep it fresh when idle). Stale
/// progress means the loop stopped draining: status goes `unhealthy` and
/// /health returns 503 so a liveness probe restarts the pod. Backend outages
/// stay `degraded` (200) — restarting this pod doesn't fix Qdrant.
#[derive(Clone)]
struct HealthChecker {
    client: reqwest::Client,
    qdrant_url: String,
    collection: String,
    graph_url: String,
    graph_api_key: String,
    worker_progress: std::sync::Arc<std::sync::atomic::AtomicU64>,
    stall_threshold: std::time::Duration,
}

impl HealthChecker {
    fn new(config: &Config, worker_progress: std::sync::Arc<std::sync::atomic::AtomicU64>) -> Self {
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
            worker_progress,
            stall_threshold: WORKER_STALL_THRESHOLD,
        }
    }

    async fn check(&self) -> Value {
        let start = std::time::Instant::now();

        // All three checks run concurrently via tokio::join!
        let (qdrant_health, graph_health, count) =
            tokio::join!(self.check_qdrant(), self.check_graph(), self.check_count(),);

        // 0 = worker loop not entered yet (backend bootstrap in progress).
        // Bootstrap is deadline-bounded but can legitimately exceed the stall
        // threshold on a cluster cold start — report "starting", not a stall,
        // or the liveness probe would restart-loop a pod that's coming up.
        let last_progress = self
            .worker_progress
            .load(std::sync::atomic::Ordering::Relaxed);
        let (worker_state, worker_stalled, progress_age) = if last_progress == 0 {
            ("starting", false, 0)
        } else {
            let age = epoch_secs().saturating_sub(last_progress);
            let stalled = age > self.stall_threshold.as_secs();
            (if stalled { "stalled" } else { "ok" }, stalled, age)
        };

        let status = if worker_stalled {
            "unhealthy"
        } else if qdrant_health.is_ok() {
            "healthy"
        } else {
            "degraded"
        };

        let elapsed = start.elapsed().as_millis();
        if worker_stalled {
            tracing::error!(
                progress_age_s = progress_age,
                threshold_s = self.stall_threshold.as_secs(),
                "service worker stalled — reporting unhealthy so the pod gets restarted"
            );
        } else {
            tracing::debug!(op = "health", elapsed_ms = elapsed, status, "ok (direct)");
        }

        json!({
            "status": status,
            // Build identity so any consumer can answer "is build X live?"
            // without cluster access (#70). null when the build didn't pass it.
            "version": build_info::version(),
            "git_sha": build_info::git_sha(),
            "built_at": build_info::built_at(),
            "backend": "qdrant",
            "worker": {
                "state": worker_state,
                "stalled": worker_stalled,
                "last_progress_age_s": progress_age,
            },
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
        // Probes bridge /health (a single Redis PING) — never /stats.
        // /stats runs 6 full-graph aggregate scans and scales with graph
        // size; readiness probes only need reachability, not workload.
        let mut req = self.client.get(format!("{}/health", self.graph_url));
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

/// Deadlines the worker applies per command. A struct only so tests can
/// shrink them — production always uses `Default` (the consts above).
struct WorkerLimits {
    cmd: std::time::Duration,
    long: std::time::Duration,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            cmd: CMD_DEADLINE,
            long: LONG_CMD_DEADLINE,
        }
    }
}

/// Reply for a command whose handler blew its deadline. The worker drops the
/// in-flight future (cancelling its awaits) and keeps draining the queue —
/// one stuck backend await must never wedge the service (#63).
fn deadline_exceeded(op: &str, deadline: std::time::Duration, start: std::time::Instant) -> Value {
    tracing::error!(
        op,
        deadline_s = deadline.as_secs(),
        elapsed_ms = ms(start),
        "command deadline exceeded — dropping handler, worker continues"
    );
    json!({
        "success": false,
        "error": format!("{op} timed out after {}s", deadline.as_secs()),
        "error_kind": "timeout",
    })
}

/// Runs MemoryService on a LocalSet, processing commands from the channel.
///
/// Wraps the service in `Rc` so long-running operations (find_duplicates,
/// merge_duplicates) can be spawned as local tasks without blocking the
/// command loop. Other commands continue processing while they run.
///
/// Every handler await is bounded by `limits` (#63), and `progress` is
/// stamped after each command so the health watchdog can tell a draining
/// worker from a wedged one.
async fn service_worker(
    mut rx: mpsc::Receiver<Cmd>,
    svc: MemoryService,
    progress: std::sync::Arc<std::sync::atomic::AtomicU64>,
    limits: WorkerLimits,
) {
    use tokio::time::timeout;
    use tracing::Instrument;

    let svc = std::rc::Rc::new(svc);

    // First heartbeat: bootstrap is done and the loop is live. Until this
    // stamp the health checker reports the worker as "starting" (progress
    // sentinel 0), never stalled — see the seed in main().
    progress.store(epoch_secs(), std::sync::atomic::Ordering::Relaxed);

    while let Some(cmd) = rx.recv().await {
        let op = cmd.op_name();
        let start = std::time::Instant::now();
        let ps = cmd.span;

        match cmd.inner {
            CmdInner::Ping { reply } => {
                let _ = reply.send(json!({"ok": true}));
            }
            CmdInner::Health { reply } => {
                let span = tracing::info_span!(parent: &ps, "health");
                let result =
                    match timeout(limits.cmd, svc.check_database_health().instrument(span)).await {
                        Ok(Ok(r)) => {
                            tracing::debug!(op, elapsed_ms = ms(start), "ok");
                            json!(r)
                        }
                        Ok(Err(e)) => {
                            log_err(op, &e, start);
                            json!({"status": "error", "message": e.safe_message()})
                        }
                        Err(_) => deadline_exceeded(op, limits.cmd, start),
                    };
                let _ = reply.send(result);
            }
            CmdInner::Store {
                params,
                read_only,
                reply,
            } => {
                let content_len = params.content.len();
                let mem_type = params.memory_type.as_deref().unwrap_or("note").to_string();
                let tag_count = params.tags.as_ref().map(|t| t.len()).unwrap_or(0);
                let has_dedup = params.dedup_threshold.is_some();
                let client = params.client_hostname.clone().unwrap_or_default();

                // Capture for fire-and-forget summary generation. Suppressed
                // under read_only — the summary path patches the stored record
                // (the gated patch_memory op), so a browser-issued store must
                // not trigger it.
                let needs_summary = !read_only && params.summary.is_none() && svc.summary.is_some();
                let content_for_summary = if needs_summary {
                    Some(params.content.clone())
                } else {
                    None
                };

                let span = tracing::info_span!(parent: &ps, "store",
                    content_len, %mem_type, read_only);
                let result = match timeout(
                    limits.cmd,
                    svc.store_memory_with(params, read_only).instrument(span),
                )
                .await
                {
                    Ok(Ok(r)) => {
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
                    Ok(Err(e)) => {
                        log_err(op, &e, start);
                        json!({"success": false, "error": e.safe_message()})
                    }
                    Err(_) => deadline_exceeded(op, limits.cmd, start),
                };
                let _ = reply.send(result);
            }
            CmdInner::Search {
                params,
                read_only,
                reply,
            } => {
                let mode = format!("{:?}", params.mode).to_lowercase();
                let query_preview = truncate(&params.query, 80);
                let page = params.page;
                let page_size = params.page_size;
                let tag_count = params.tags.as_ref().map(|t| t.len()).unwrap_or(0);
                let mem_type = params.memory_type.clone().unwrap_or_default();
                let span = tracing::info_span!(parent: &ps, "search", %mode, read_only);
                let result = match timeout(
                    limits.cmd,
                    svc.search_with(params, read_only).instrument(span),
                )
                .await
                {
                    Ok(Ok(r)) => {
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
                    Ok(Err(e)) => {
                        log_err(op, &e, start);
                        json!({"error": e.safe_message()})
                    }
                    Err(_) => deadline_exceeded(op, limits.cmd, start),
                };
                let _ = reply.send(result);
            }
            CmdInner::Delete { hash, reply } => {
                let span = tracing::info_span!(parent: &ps, "delete");
                let h = truncate_hash(&hash);
                let result =
                    match timeout(limits.cmd, svc.delete_memory(&hash).instrument(span)).await {
                        Ok(Ok(r)) => {
                            tracing::info!(op, hash = h.as_str(), elapsed_ms = ms(start), "ok");
                            json!(r)
                        }
                        Ok(Err(e)) => {
                            log_err(op, &e, start);
                            json!({"success": false, "error": e.safe_message()})
                        }
                        Err(_) => deadline_exceeded(op, limits.cmd, start),
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
                let result =
                    match timeout(limits.cmd, svc.get_memory(&hash, output).instrument(span)).await
                    {
                        Ok(Ok(r)) => {
                            let found = r.get("found").and_then(|f| f.as_bool()).unwrap_or(false);
                            tracing::info!(
                                op,
                                hash = h.as_str(),
                                found,
                                elapsed_ms = ms(start),
                                "ok"
                            );
                            r
                        }
                        Ok(Err(e)) => {
                            log_err(op, &e, start);
                            json!({"error": e.safe_message()})
                        }
                        Err(_) => deadline_exceeded(op, limits.cmd, start),
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
                let result = match timeout(limits.cmd, svc.relation(params).instrument(span)).await
                {
                    Ok(Ok(r)) => {
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
                    Ok(Err(e)) => {
                        log_err(op, &e, start);
                        json!({"success": false, "error": e.safe_message()})
                    }
                    Err(_) => deadline_exceeded(op, limits.cmd, start),
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
                let result = match timeout(
                    limits.cmd,
                    svc.memory_supersede(&old_hash, &new_hash, &reason)
                        .instrument(span),
                )
                .await
                {
                    Ok(Ok(r)) => {
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
                    Ok(Err(e)) => {
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
                    Err(_) => deadline_exceeded(op, limits.cmd, start),
                };
                let _ = reply.send(result);
            }
            CmdInner::Contradictions { limit, reply } => {
                let span = tracing::info_span!(parent: &ps, "contradictions");
                let result = match timeout(
                    limits.cmd,
                    svc.memory_contradictions(limit).instrument(span),
                )
                .await
                {
                    Ok(Ok(r)) => {
                        let pairs = r
                            .get("contradictions")
                            .and_then(|v| v.as_array())
                            .map(|a| a.len())
                            .unwrap_or(0);
                        tracing::info!(op, limit, pairs, elapsed_ms = ms(start), "ok");
                        r
                    }
                    Ok(Err(e)) => {
                        log_err(op, &e, start);
                        json!({"success": false, "error": e.safe_message()})
                    }
                    Err(_) => deadline_exceeded(op, limits.cmd, start),
                };
                let _ = reply.send(result);
            }
            CmdInner::Patch { hash, patch, reply } => {
                let span = tracing::info_span!(parent: &ps, "patch");
                let h = truncate_hash(&hash);
                let fields = patch.changed_fields();
                let result =
                    match timeout(limits.cmd, svc.patch_memory(&hash, &patch).instrument(span))
                        .await
                    {
                        Ok(Ok(mem)) => {
                            tracing::info!(
                                op,
                                hash = h.as_str(),
                                fields = fields.as_str(),
                                elapsed_ms = ms(start),
                                "ok"
                            );
                            json!(mem)
                        }
                        Ok(Err(e)) => {
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
                        Err(_) => deadline_exceeded(op, limits.cmd, start),
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
                let deadline = limits.long;
                tokio::task::spawn_local(
                    async move {
                        let result = match timeout(
                            deadline,
                            svc.find_duplicates(threshold, limit, strategy),
                        )
                        .await
                        {
                            Ok(Ok(r)) => {
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
                            Ok(Err(e)) => {
                                log_err(op, &e, start);
                                json!({"success": false, "error": e.safe_message()})
                            }
                            Err(_) => deadline_exceeded(op, deadline, start),
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
                let deadline = limits.long;
                tokio::task::spawn_local(
                    async move {
                        let refs: Vec<&str> = duplicates.iter().map(|s| s.as_str()).collect();
                        let result = match timeout(
                            deadline,
                            svc.merge_duplicates(&canonical, &refs, &reason, dry_run),
                        )
                        .await
                        {
                            Ok(Ok(r)) => {
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
                            Ok(Err(e)) => {
                                log_err(op, &e, start);
                                json!({"success": false, "error": e.safe_message()})
                            }
                            Err(_) => deadline_exceeded(op, deadline, start),
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

        // Watchdog heartbeat: the loop just finished (or spawned) a command.
        // Stops advancing exactly when the worker stops draining.
        progress.store(epoch_secs(), std::sync::atomic::Ordering::Relaxed);
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

// ─── Auth ───────────────────────────────────────────────────────────────────
// Dual-mode auth (static bearer + provider-agnostic OIDC) lives in `auth.rs`;
// the JWT verifier in `oidc.rs`. The middleware is `auth::require_auth`.

/// Build `AuthState` from config and enforce fail-closed startup invariants.
fn build_auth_state(config: &Config) -> AuthState {
    let api_key = if config.api_key.is_empty() {
        None
    } else {
        Some(config.api_key.clone())
    };

    let oidc = config.oidc_issuer.as_ref().map(|issuer| {
        let audience = format!("{}/mcp", config.public_base_url);
        tracing::info!(
            issuer = issuer.as_str(),
            audience = audience.as_str(),
            "OIDC enabled"
        );
        oidc::OidcVerifier::new(issuer.clone(), audience)
    });

    // Fail-closed: refuse to start with no auth unless the dev flag is set.
    if api_key.is_none() && oidc.is_none() {
        if !config.allow_unauthenticated {
            panic!(
                "no auth configured: set ALAYA_API_KEY or OIDC_ISSUER, or \
                 DANGEROUSLY_ALLOW_UNAUTHENTICATED=true for dev"
            );
        }
        // The dev-only open mode must never run on a public origin.
        if !is_private_host(&config.public_base_url) {
            panic!(
                "DANGEROUSLY_ALLOW_UNAUTHENTICATED refused on public origin {}",
                config.public_base_url
            );
        }
        tracing::warn!("DANGEROUSLY_ALLOW_UNAUTHENTICATED — all endpoints are UNAUTHENTICATED");
    } else if config.allow_unauthenticated {
        tracing::warn!("DANGEROUSLY_ALLOW_UNAUTHENTICATED ignored — auth is configured");
    }

    AuthState {
        api_key,
        allow_unauthenticated: config.allow_unauthenticated,
        oidc,
        public_base_url: config.public_base_url.clone(),
    }
}

/// CORS for the claude.ai browser connector: exact origin, no credentials.
fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(
            "https://claude.ai"
                .parse::<axum::http::HeaderValue>()
                .expect("valid origin"),
        )
        .allow_methods([Method::GET, Method::POST, Method::PATCH, Method::OPTIONS])
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE])
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

        // Watchdog heartbeat: epoch-seconds of the worker's last completed
        // command. Written by the worker loop, read by the health checker.
        // Seeded 0 = "worker loop not entered yet": backend bootstrap
        // (ensure_qdrant_collection + init_l2_cache retries) can legitimately
        // exceed the stall threshold on a cluster cold start, and /health is
        // already serving — a wall-clock seed here would misreport that as a
        // stall and restart-loop the pod. Every bootstrap await is
        // deadline-bounded, so the loop is always entered in bounded time.
        let worker_progress = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let progress_for_worker = worker_progress.clone();

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
                // Fresh-deploy bootstrap: create the memory collection if it is
                // absent so the first write doesn't 404 (#31).
                ensure_qdrant_collection(&qdrant, cfg_clone.embedding_dimensions).await;
                let embeddings = EmbeddingClient::new(
                    cfg_clone.embedding_url,
                    cfg_clone.embedding_model,
                    cfg_clone.embedding_dimensions,
                    cfg_clone.embedding_batch_size,
                    None,
                );
                // L2 embedding cache via cachekit-rs (optional) — backend
                // selected by CACHE_BACKEND (redis default, saas).
                let l2_cache = init_l2_cache().await;
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

                let mut svc = MemoryService::new(
                    Box::new(qdrant),
                    Box::new(cached_embeddings),
                    Box::new(GraphRef(graph.clone())),
                    Box::new(HebbianRef(graph.clone())),
                    Box::new(ConsolidationRef(graph)),
                    summary,
                );

                if let Some(url) = &cfg_clone.rerank_url {
                    tracing::info!(
                        url = url.as_str(),
                        top_n = cfg_clone.rerank_top_n,
                        has_api_key = cfg_clone.rerank_api_key.is_some(),
                        "cross-encoder reranker enabled"
                    );
                    svc = svc.with_reranker(Box::new(RerankClient::new(
                        url.clone(),
                        cfg_clone.rerank_top_n,
                        cfg_clone.rerank_api_key.clone(),
                    )));
                } else {
                    tracing::info!("RERANK_URL not set — cross-encoder rerank disabled");
                }

                service_worker(rx, svc, progress_for_worker, WorkerLimits::default()).await;
            });
        });

        // Axum on the main multi-threaded runtime
        let handle = ServiceHandle { tx };

        // Dual-mode auth state + fail-closed startup invariants.
        let auth_state = build_auth_state(&config);

        // Health checker bypasses the service worker channel entirely —
        // runs directly on the multi-threaded axum runtime with its own
        // reqwest::Client. Prevents health probe timeouts during long ops.
        // The worker_progress watchdog covers the blind spot that bypass
        // created (#63): a wedged worker now turns /health unhealthy (503).
        let checker = HealthChecker::new(&config, worker_progress);

        // Pinger: sends a no-op Ping through the worker channel so progress
        // stays fresh while idle. try_send on purpose — if the channel is
        // full, real commands are keeping (or failing to keep) progress
        // fresh, which is exactly what the watchdog should observe.
        let pinger = {
            let handle = handle.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(PING_PERIOD);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    tick.tick().await;
                    let (reply, _rx) = oneshot::channel();
                    let _ = handle.tx.try_send(Cmd {
                        inner: CmdInner::Ping { reply },
                        span: tracing::Span::none(),
                    });
                }
            })
        };

        const MAX_BODY: usize = 1_048_576; // 1 MB — covers the /mcp Bytes extractor

        // Protected routes: handlers keep `ServiceHandle` state; `require_auth`
        // is layered with its own `AuthState` (axum allows differing types).
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
                auth_state.clone(),
                auth::require_auth,
            ))
            .layer(DefaultBodyLimit::max(MAX_BODY))
            .layer(TraceLayer::new_for_http().make_span_with(
                tower_http::trace::DefaultMakeSpan::new().level(tracing::Level::INFO),
            ))
            .with_state(handle);

        // Read-only auth-config view (LAB-1684 AC7). Same auth middleware;
        // GET /auth/config is unmapped in rest_route_op → static-bearer only
        // (default-deny), and the payload carries no credential material.
        let auth_config_route = Router::new()
            .route("/auth/config", get(auth_config))
            .layer(middleware::from_fn_with_state(
                auth_state.clone(),
                auth::require_auth,
            ))
            .with_state(auth_state.clone());

        // Health surfaces. Built before `wellknown` takes `auth_state`.
        let health_route = health_routes(checker, auth_state.clone());

        // Unauthenticated protected-resource metadata (404 when OIDC disabled).
        let wellknown = Router::new()
            .route(
                "/.well-known/oauth-protected-resource",
                get(wellknown::protected_resource_metadata),
            )
            .route(
                "/.well-known/oauth-protected-resource/mcp",
                get(wellknown::protected_resource_metadata),
            )
            .with_state(auth_state);

        // CORS is outermost so browser preflight (OPTIONS, no auth header) is
        // answered before `require_auth`.
        let app = health_route
            .merge(wellknown)
            .merge(auth_config_route)
            .merge(protected)
            .layer(cors_layer());

        let listener = tokio::net::TcpListener::bind(&config.listen_addr)
            .await
            .expect("failed to bind");

        tracing::info!("alaya-server listening on {}", config.listen_addr);

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .expect("server error");

        // The pinger holds a ServiceHandle clone — abort it or the worker's
        // rx.recv() never sees the channel close and the drain below hangs.
        pinger.abort();

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

/// Unauthenticated liveness/readiness probe: `status` plus the HTTP code,
/// which is everything an automated prober consumes.
///
/// Nothing else belongs here. This route sits outside `require_auth`, so any
/// field added to the body is world-readable — and the full document carries
/// live capacity (`total_memories`), outage state (`worker.stalled`), the
/// deployed commit, and per-backend error strings that render in-cluster URLs.
/// That moved to `/health/detail` (#77).
///
/// `unhealthy` (wedged worker) returns 503 so an httpGet liveness probe
/// restarts the pod; backend outages stay `degraded`/200 — a restart
/// wouldn't fix those (#63).
async fn health(
    axum::extract::State(checker): axum::extract::State<HealthChecker>,
) -> (StatusCode, Json<Value>) {
    let v = checker.check().await;
    (health_code(&v), Json(json!({ "status": v["status"] })))
}

/// Authenticated operator view: the full health document, including build
/// identity (#70). Every intended consumer — radar, unified-memory, agents —
/// already holds `ALAYA_API_KEY`, so "read the running build with zero cluster
/// access" survives the move behind auth.
async fn health_detail(
    axum::extract::State(checker): axum::extract::State<HealthChecker>,
) -> (StatusCode, Json<Value>) {
    let v = checker.check().await;
    (health_code(&v), Json(v))
}

/// Shared by both surfaces so they can never disagree on liveness.
fn health_code(health: &Value) -> StatusCode {
    if health.get("status").and_then(|s| s.as_str()) == Some("unhealthy") {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}

/// The two health surfaces, composed.
///
/// Assembled here rather than inline in `main` so tests exercise the real
/// composition: a handler-level test would still pass with the auth layer
/// dropped, which is precisely the regression that matters.
fn health_routes(checker: HealthChecker, auth_state: AuthState) -> Router {
    let detail = Router::new()
        .route("/health/detail", get(health_detail))
        .layer(middleware::from_fn_with_state(
            auth_state,
            auth::require_auth,
        ))
        .with_state(checker.clone());

    Router::new()
        .route("/health", get(health))
        .with_state(checker)
        .merge(detail)
}

/// Read-only auth-config view (LAB-1684 AC7): principals, OIDC issuer /
/// audience, and the principal × op matrix. No credential material.
async fn auth_config(
    axum::extract::State(auth): axum::extract::State<auth::AuthState>,
) -> Json<Value> {
    Json(auth::auth_config_view(&auth))
}

async fn store(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    axum::Extension(principal): axum::Extension<AuthPrincipal>,
    Json(params): Json<StoreParams>,
) -> (StatusCode, Json<Value>) {
    let read_only = WritePolicy::for_principal(principal) == WritePolicy::ReadOnly;
    let (tx, rx) = oneshot::channel();
    h.call(
        CmdInner::Store {
            params,
            read_only,
            reply: tx,
        },
        rx,
    )
    .await
}

async fn search(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    axum::Extension(principal): axum::Extension<AuthPrincipal>,
    Json(params): Json<SearchParams>,
) -> (StatusCode, Json<Value>) {
    let read_only = WritePolicy::for_principal(principal) == WritePolicy::ReadOnly;
    let (tx, rx) = oneshot::channel();
    h.call(
        CmdInner::Search {
            params,
            read_only,
            reply: tx,
        },
        rx,
    )
    .await
}

#[derive(Deserialize)]
struct DeleteReq {
    content_hash: String,
}

async fn delete(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(req): Json<DeleteReq>,
) -> (StatusCode, Json<Value>) {
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
) -> (StatusCode, Json<Value>) {
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
) -> (StatusCode, Json<Value>) {
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
) -> (StatusCode, Json<Value>) {
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
) -> (StatusCode, Json<Value>) {
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
) -> (StatusCode, Json<Value>) {
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

    // call_rpc fast-fails on a full channel and bounds the reply await —
    // this handler previously used a blocking send + unbounded await, both
    // of which hang for as long as the worker does (#63).
    let (tx, rx) = oneshot::channel();
    let v = match h
        .call_rpc(
            CmdInner::Patch {
                hash: content_hash,
                patch,
                reply: tx,
            },
            rx,
        )
        .await
    {
        Ok(v) => v,
        Err((_code, msg)) => {
            return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": msg})));
        }
    };

    match v.get("error_kind").and_then(|k| k.as_str()) {
        Some("not_found") => (StatusCode::NOT_FOUND, Json(v)),
        Some("validation") => (StatusCode::BAD_REQUEST, Json(v)),
        Some(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(v)),
        None => (StatusCode::OK, Json(v)),
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

    // Non-blocking dispatch + bounded reply await — a full channel or a
    // wedged worker fast-fails instead of hanging the HTTP request (#63).
    let (tx, rx) = oneshot::channel();
    let v = match h
        .call_rpc(
            CmdInner::GetMemory {
                hash: content_hash,
                output: q.output,
                reply: tx,
            },
            rx,
        )
        .await
    {
        Ok(v) => v,
        Err((_code, msg)) => {
            return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": msg})));
        }
    };

    if v.get("error").is_some() {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(v));
    }
    if v.get("found").and_then(|f| f.as_bool()).unwrap_or(false) {
        return (StatusCode::OK, Json(v));
    }
    (StatusCode::NOT_FOUND, Json(v))
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
) -> (StatusCode, Json<Value>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_of_strips_port_and_unwraps_ipv6_brackets() {
        assert_eq!(host_of("https://id.27b.io"), Some("id.27b.io".into()));
        assert_eq!(host_of("https://id.27b.io:8443"), Some("id.27b.io".into()));
        assert_eq!(host_of("http://[::1]:3001/foo"), Some("::1".into()));
        assert_eq!(host_of("http://localhost:8080"), Some("localhost".into()));
        assert_eq!(host_of("not-a-url"), None);
    }

    #[test]
    fn is_private_host_rejects_dns_confusables() {
        // Real private — loopback / RFC1918 / cluster-internal.
        assert!(is_private_host("http://localhost:8080"));
        assert!(is_private_host("http://127.0.0.1"));
        assert!(is_private_host("http://[::1]:3001"));
        assert!(is_private_host("http://10.0.0.5"));
        assert!(is_private_host("http://192.168.1.1"));
        assert!(is_private_host("http://172.20.0.1"));
        assert!(is_private_host("http://alaya-server.mcp.svc"));
        assert!(is_private_host("http://kube-api.internal"));

        // DNS-name look-alikes must NOT count — the bug fix is this:
        assert!(!is_private_host("http://127.0.0.1.evil.com"));
        assert!(!is_private_host("http://192.168.1.1.attacker.net"));
        assert!(!is_private_host("http://localhost.evil.com"));
        assert!(!is_private_host("http://172.16.0.1.attacker.org"));

        // 172.x outside 16-31 isn't private.
        assert!(!is_private_host("http://172.15.0.1"));
        assert!(!is_private_host("http://172.32.0.1"));

        // Public origins.
        assert!(!is_private_host("https://alaya.27b.io"));
        assert!(!is_private_host("https://example.com"));
    }
}

/// Regression tests for #63: one stuck backend await must never wedge the
/// service, and a wedged worker must be visible to the health check.
#[cfg(test)]
mod wedge_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::time::Duration;

    use async_trait::async_trait;

    use alaya_types::{
        Result,
        graph::{
            CoAccessPair, Contradiction, ContradictionRef, Direction, Edge, EdgeMeta, GraphStats,
            Neighbor, SystemRelationType, UserRelationType,
        },
        memory::{HealthStatus, Memory, MetadataUpdate, PatchMemoryRequest, ScrollResult},
        search::{PayloadFilter, PromptName},
    };

    use super::*;
    use alaya_backends::traits::{
        ConsolidationService, EmbeddingProvider, GraphService, HebbianService, VectorStorage,
    };
    use alaya_types::memory::ScoredMemory;

    /// VectorStorage whose `delete` blackholes — models a backend whose pod
    /// IP vanished without an RST. Every other method panics: the test only
    /// exercises the delete path and the no-op ping.
    struct HangVectors;

    #[async_trait(?Send)]
    impl VectorStorage for HangVectors {
        async fn store(&self, _memory: &Memory) -> Result<(bool, String)> {
            unimplemented!()
        }
        async fn get_by_hash(&self, _content_hash: &str) -> Result<Option<Memory>> {
            unimplemented!()
        }
        async fn get_batch(&self, _hashes: &[&str]) -> Result<Vec<Memory>> {
            unimplemented!()
        }
        async fn delete(&self, _content_hash: &str) -> Result<bool> {
            std::future::pending().await
        }
        async fn update_metadata(
            &self,
            _content_hash: &str,
            _updates: MetadataUpdate,
        ) -> Result<()> {
            unimplemented!()
        }
        async fn patch_memory(
            &self,
            _content_hash: &str,
            _patch: &PatchMemoryRequest,
        ) -> Result<Memory> {
            unimplemented!()
        }
        async fn search_by_vector(
            &self,
            _embedding: &[f32],
            _limit: usize,
            _filters: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            unimplemented!()
        }
        async fn search_by_tags(
            &self,
            _tags: &[&str],
            _match_all: bool,
            _limit: usize,
        ) -> Result<Vec<ScoredMemory>> {
            unimplemented!()
        }
        async fn search_similar_tags(
            &self,
            _tag_embedding: &[f32],
            _limit: usize,
        ) -> Result<Vec<String>> {
            unimplemented!()
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            unimplemented!()
        }
        async fn get_all(&self, _limit: usize, _offset: Option<&str>) -> Result<ScrollResult> {
            unimplemented!()
        }
        async fn get_recent(
            &self,
            _limit: usize,
            _start_from: Option<f64>,
            _memory_type: Option<&str>,
        ) -> Result<Vec<Memory>> {
            unimplemented!()
        }
        async fn count(&self) -> Result<usize> {
            unimplemented!()
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            unimplemented!()
        }
        async fn increment_access_count(&self, _content_hash: &str) -> Result<()> {
            unimplemented!()
        }
        async fn health(&self) -> Result<HealthStatus> {
            unimplemented!()
        }
    }

    struct StubEmbeddings;

    #[async_trait(?Send)]
    impl EmbeddingProvider for StubEmbeddings {
        async fn embed_batch(
            &self,
            _texts: &[&str],
            _prompt_name: PromptName,
        ) -> Result<Vec<Vec<f32>>> {
            unimplemented!()
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn model_name(&self) -> &str {
            "stub"
        }
    }

    struct StubGraph;

    #[async_trait(?Send)]
    impl GraphService for StubGraph {
        async fn ensure_node(&self, _content_hash: &str, _created_at: f64) -> Result<()> {
            unimplemented!()
        }
        async fn delete_node(&self, _content_hash: &str) -> Result<()> {
            unimplemented!()
        }
        async fn create_typed_edge(
            &self,
            _src: &str,
            _dst: &str,
            _rel: UserRelationType,
            _meta: EdgeMeta,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn get_typed_edges(
            &self,
            _hash: &str,
            _rel: Option<UserRelationType>,
            _dir: Direction,
            _limit: usize,
        ) -> Result<Vec<Edge>> {
            unimplemented!()
        }
        async fn delete_typed_edge(
            &self,
            _src: &str,
            _dst: &str,
            _rel: UserRelationType,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn create_system_edge(
            &self,
            _src: &str,
            _dst: &str,
            _rel: SystemRelationType,
            _created_at: f64,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn get_all_contradictions(&self, _limit: usize) -> Result<Vec<Contradiction>> {
            unimplemented!()
        }
        async fn get_contradictions_for_hashes(
            &self,
            _hashes: &[&str],
        ) -> Result<HashMap<String, Vec<ContradictionRef>>> {
            unimplemented!()
        }
        async fn get_neighbors(
            &self,
            _hash: &str,
            _max_hops: u8,
            _min_weight: f64,
            _limit: usize,
        ) -> Result<Vec<Neighbor>> {
            unimplemented!()
        }
        async fn spreading_activation(
            &self,
            _seeds: &[&str],
            _max_hops: u8,
            _decay: f64,
            _min_activation: f64,
            _limit: usize,
        ) -> Result<HashMap<String, f64>> {
            unimplemented!()
        }
        async fn hebbian_boosts_within(&self, _hashes: &[&str]) -> Result<HashMap<String, f64>> {
            unimplemented!()
        }
        async fn get_stats(&self) -> Result<GraphStats> {
            unimplemented!()
        }
    }

    struct StubHebbian;

    #[async_trait(?Send)]
    impl HebbianService for StubHebbian {
        async fn enqueue_strengthen(&self, _pairs: &[CoAccessPair]) -> Result<()> {
            unimplemented!()
        }
    }

    struct StubConsolidation;

    #[async_trait(?Send)]
    impl ConsolidationService for StubConsolidation {
        async fn decay_all_edges(&self, _decay_factor: f64, _limit: usize) -> Result<usize> {
            unimplemented!()
        }
        async fn decay_stale_edges(
            &self,
            _stale_before: f64,
            _decay_factor: f64,
            _limit: usize,
        ) -> Result<usize> {
            unimplemented!()
        }
        async fn prune_weak_edges(&self, _threshold: f64, _limit: usize) -> Result<usize> {
            unimplemented!()
        }
        async fn get_orphan_nodes(&self, _limit: usize) -> Result<Vec<String>> {
            unimplemented!()
        }
    }

    fn hanging_service() -> MemoryService {
        MemoryService::new(
            Box::new(HangVectors),
            Box::new(StubEmbeddings),
            Box::new(StubGraph),
            Box::new(StubHebbian),
            Box::new(StubConsolidation),
            None,
        )
    }

    /// The incident scenario (#63): a backend await that never resolves.
    /// The worker must reply with a timeout error at the command deadline
    /// and keep draining the queue instead of wedging forever.
    #[tokio::test(start_paused = true)]
    async fn stuck_command_errors_at_deadline_and_worker_keeps_draining() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (tx, rx) = mpsc::channel::<Cmd>(8);
                // 0 sentinel, as main() seeds it — the worker's entry stamp
                // must replace it (asserted below).
                let progress = Arc::new(AtomicU64::new(0));
                let limits = WorkerLimits {
                    cmd: Duration::from_millis(100),
                    long: Duration::from_millis(200),
                };
                tokio::task::spawn_local(service_worker(
                    rx,
                    hanging_service(),
                    progress.clone(),
                    limits,
                ));

                // 1. The stuck command errors out at its deadline.
                let (rtx, rrx) = oneshot::channel();
                tx.send(Cmd {
                    inner: CmdInner::Delete {
                        hash: "a".repeat(64),
                        reply: rtx,
                    },
                    span: tracing::Span::none(),
                })
                .await
                .unwrap();
                let reply = tokio::time::timeout(Duration::from_secs(60), rrx)
                    .await
                    .expect("stuck command never replied — worker wedged")
                    .expect("reply sender dropped");
                assert_eq!(reply["error_kind"], "timeout");
                assert!(reply["error"].as_str().unwrap().contains("timed out"));

                // 2. Subsequent commands still complete — the worker drained.
                let (ptx, prx) = oneshot::channel();
                tx.send(Cmd {
                    inner: CmdInner::Ping { reply: ptx },
                    span: tracing::Span::none(),
                })
                .await
                .unwrap();
                let pong = tokio::time::timeout(Duration::from_secs(60), prx)
                    .await
                    .expect("worker did not drain after the stuck command")
                    .expect("ping reply dropped");
                assert_eq!(pong["ok"], true);

                // The worker stamped progress at loop entry and after each
                // command — the 0 "starting" sentinel must be gone.
                assert_ne!(progress.load(std::sync::atomic::Ordering::Relaxed), 0);
            })
            .await;
    }

    fn test_checker(progress_epoch_s: u64) -> HealthChecker {
        HealthChecker {
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_millis(200))
                .timeout(Duration::from_millis(500))
                .build()
                .unwrap(),
            // Port 1 refuses immediately — models an unreachable backend.
            qdrant_url: "http://127.0.0.1:1".into(),
            collection: "test".into(),
            graph_url: "http://127.0.0.1:1".into(),
            graph_api_key: String::new(),
            worker_progress: Arc::new(AtomicU64::new(progress_epoch_s)),
            stall_threshold: WORKER_STALL_THRESHOLD,
        }
    }

    /// Health semantics (#63): a stalled worker is `unhealthy` (503 → k8s
    /// restarts the pod); dead backends alone stay `degraded` (200 — a
    /// restart would not fix Qdrant being down).
    #[tokio::test]
    async fn health_distinguishes_worker_stall_from_backend_outage() {
        // Fresh worker progress + unreachable backends → degraded, not unhealthy.
        let v = test_checker(epoch_secs()).check().await;
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["worker"]["stalled"], false);

        // Stale worker progress → unhealthy, regardless of backend state.
        let v = test_checker(epoch_secs() - 3600).check().await;
        assert_eq!(v["status"], "unhealthy");
        assert_eq!(v["worker"]["stalled"], true);
        assert!(v["worker"]["last_progress_age_s"].as_u64().unwrap() >= 3600);

        // 0 sentinel = worker still bootstrapping backends → "starting",
        // never a stall: a slow cluster cold start must not restart-loop
        // the pod before the command loop has even begun.
        let v = test_checker(0).check().await;
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["worker"]["state"], "starting");
        assert_eq!(v["worker"]["stalled"], false);
    }

    // ─── /health split (#77) ────────────────────────────────────────────────

    const TEST_KEY: &str = "test-api-key";

    fn test_auth_state() -> AuthState {
        AuthState {
            api_key: Some(TEST_KEY.into()),
            allow_unauthenticated: false,
            oidc: None,
            public_base_url: "http://localhost:3001".into(),
        }
    }

    /// Drive the composed router in-process. `token: None` models the k8s
    /// probe and any anonymous caller.
    async fn probe(app: &Router, path: &str, token: Option<&str>) -> (StatusCode, Value) {
        use tower::ServiceExt;

        let mut req = axum::http::Request::builder().uri(path);
        if let Some(t) = token {
            req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        let resp = app
            .clone()
            .oneshot(req.body(axum::body::Body::empty()).expect("build request"))
            .await
            .expect("router call");

        let code = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        (code, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    /// The unauthenticated probe carries `status` and nothing else.
    ///
    /// Asserted as an exact key set, not as spot-checks on the fields that
    /// leak today: anything later added to `HealthChecker::check` is
    /// world-readable the moment it lands, and this is the test that has to
    /// fail when that happens.
    #[tokio::test]
    async fn unauthenticated_health_exposes_only_status() {
        let app = health_routes(test_checker(epoch_secs()), test_auth_state());

        let (code, body) = probe(&app, "/health", None).await;

        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["status"], "degraded");
        let keys: Vec<&str> = body
            .as_object()
            .expect("object body")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["status"], "unauthenticated /health leaked fields");
    }

    /// The operator view is unreachable without a credential, and complete
    /// with one.
    #[tokio::test]
    async fn health_detail_requires_auth() {
        let app = health_routes(test_checker(epoch_secs()), test_auth_state());

        let (code, _) = probe(&app, "/health/detail", None).await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);

        let (code, _) = probe(&app, "/health/detail", Some("wrong-key")).await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);

        let (code, body) = probe(&app, "/health/detail", Some(TEST_KEY)).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["status"], "degraded");
        assert_eq!(body["total_memories"], 0);
        assert!(body["worker"].is_object());
        assert!(body["vector_health"].is_object());
        // Backends are unreachable here, so `check` takes the error arms —
        // the path that renders `reqwest::Error` and with it in-cluster URLs.
        // The detail view keeps that; the bare probe's exact-key-set
        // assertion above is what proves it never reaches an anonymous caller.
        assert!(body["vector_health"]["error"].is_string());
        // Build identity (#70) rides the authenticated surface now.
        assert!(body.get("version").is_some());
        assert!(body.get("git_sha").is_some());
        assert!(body.get("built_at").is_some());
    }

    /// The #63 contract is the HTTP code, not the body: a wedged worker must
    /// still 503 the *unauthenticated* probe, or k8s stops restarting stalled
    /// pods. The failure path must not widen the body either.
    #[tokio::test]
    async fn stalled_worker_still_503s_the_bare_probe() {
        let app = health_routes(test_checker(epoch_secs() - 3600), test_auth_state());

        let (code, body) = probe(&app, "/health", None).await;

        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "unhealthy");
        assert_eq!(body.as_object().expect("object body").len(), 1);
    }
}
