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
    extract::{DefaultBodyLimit, Request},
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
    judge::JudgeClient,
    qdrant::QdrantClient,
    rerank::RerankClient,
    summary::SummaryClient,
};
use alaya_core::deduplication::CanonicalStrategy;
use alaya_core::service::{
    JudgeOutcome, MemoryService, OutputMode, RelationParams, SearchParams, StoreParams,
};
use alaya_types::graph::{Contradiction, ContradictionQuery, Resolution};
use alaya_types::memory::PatchMemoryRequest;

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
    readonly_api_key: String,
    oidc_issuer: Option<String>,
    public_base_url: String,
    allow_unauthenticated: bool,
    summary_url: Option<String>,
    summary_api_key: Option<String>,
    summary_model: String,
    /// Contradiction judge (LAB-3283, LAB-3895). URL and key fall back to the
    /// SUMMARY_* counterpart; with neither set the engine is disabled. The
    /// model has its own default: summaries are priced for volume, verdicts
    /// for precision on the golden set. `judge_daily_cap` bounds store-path
    /// judge spend per UTC day (default 1000).
    judge_url: Option<String>,
    judge_api_key: Option<String>,
    judge_model: String,
    judge_daily_cap: usize,
    rerank_url: Option<String>,
    rerank_api_key: Option<String>,
    rerank_top_n: usize,
    rerank_timeout_ms: std::num::NonZeroU64,
}

impl Config {
    fn from_env() -> Self {
        let cfg = Self {
            qdrant_url: env_required("QDRANT_URL"),
            qdrant_collection: env_or("QDRANT_COLLECTION", "memories_arctic1024"),
            qdrant_api_key: env_non_empty("QDRANT_API_KEY"),
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
            readonly_api_key: env_or("ALAYA_READONLY_API_KEY", ""),
            oidc_issuer: env_non_empty("OIDC_ISSUER"),
            public_base_url: normalize_public_base_url(&env_or(
                "PUBLIC_BASE_URL",
                "https://alaya.27b.io",
            )),
            allow_unauthenticated: env_or("DANGEROUSLY_ALLOW_UNAUTHENTICATED", "")
                .eq_ignore_ascii_case("true"),
            summary_url: env_non_empty("SUMMARY_URL"),
            summary_api_key: env_non_empty("SUMMARY_API_KEY"),
            summary_model: env_or("SUMMARY_MODEL", "claude-haiku-4-5-20251001"),
            judge_url: env_non_empty("JUDGE_URL").or_else(|| env_non_empty("SUMMARY_URL")),
            judge_api_key: env_non_empty("JUDGE_API_KEY")
                .or_else(|| env_non_empty("SUMMARY_API_KEY")),
            judge_model: env_non_empty("JUDGE_MODEL").unwrap_or_else(|| "claude-sonnet-5".into()),
            judge_daily_cap: parse_judge_daily_cap(env_non_empty("JUDGE_DAILY_CAP"))
                .unwrap_or_else(|e| panic!("{e}")),
            rerank_url: env_non_empty("RERANK_URL"),
            rerank_api_key: env_non_empty("RERANK_API_KEY"),
            rerank_top_n: env_or("RERANK_TOP_N", "20")
                .parse()
                .expect("RERANK_TOP_N must be a number"),
            // NonZeroU64: a zero budget would time out every rerank and warn
            // on every search — refuse to start instead.
            rerank_timeout_ms: env_or("RERANK_TIMEOUT_MS", "5000")
                .parse()
                .expect("RERANK_TIMEOUT_MS must be a positive integer (ms)"),
        };
        // Read through the helper `init_l2_cache` uses, so the guard and the
        // cache can never disagree about which string gets dialled.
        let redis_cache_url = env_non_empty("REDIS_CACHE_URL");
        // Every credential-bearing URL this process reads into `Config`,
        // checked on the main thread before the runtime, the worker thread or
        // the listener exist: a refused endpoint means the process never
        // starts. The transport column is the client that dials that var — the
        // wrong one there is a silent downgrade, so it is one column to read
        // rather than seven call sites.
        //
        // Credential-bearing but not in `Config`, so not covered here: the
        // cachekit.io SaaS cache URL (its own builder is HTTPS-only and
        // host-allowlisted) and `OTEL_EXPORTER_OTLP_ENDPOINT`, which
        // `opentelemetry-otlp` reads directly along with the bearer token in
        // `OTEL_EXPORTER_OTLP_HEADERS`.
        for (var, url, has_credential, transport) in [
            (
                "SUMMARY_URL",
                cfg.summary_url.as_deref(),
                cfg.summary_api_key.is_some(),
                Transport::Http,
            ),
            (
                "JUDGE_URL",
                cfg.judge_url.as_deref(),
                cfg.judge_api_key.is_some(),
                Transport::Http,
            ),
            (
                "RERANK_URL",
                cfg.rerank_url.as_deref(),
                cfg.rerank_api_key.is_some(),
                Transport::Http,
            ),
            (
                "QDRANT_URL",
                Some(cfg.qdrant_url.as_str()),
                cfg.qdrant_api_key.is_some(),
                Transport::Http,
            ),
            (
                "GRAPH_URL",
                Some(cfg.graph_url.as_str()),
                // HealthChecker puts QDRANT_API_KEY in the shared client's
                // default_headers; check_graph overrides only when graph_api_key
                // is non-empty, so the Qdrant key leaks to graph probes.
                !cfg.graph_api_key.is_empty() || cfg.qdrant_api_key.is_some(),
                Transport::Http,
            ),
            (
                // `false` holds only because the worker passes `None` to
                // `EmbeddingClient::new`, leaving userinfo as the only
                // credential this can carry. Mirrored at that call site.
                "EMBEDDING_URL",
                Some(cfg.embedding_url.as_str()),
                false,
                Transport::Http,
            ),
            (
                "REDIS_CACHE_URL",
                redis_cache_url.as_deref(),
                false,
                Transport::Redis,
            ),
        ] {
            if let Some(url) = url {
                check_credential_transport(var, url, has_credential, transport)
                    .unwrap_or_else(|e| panic!("{e}"));
            }
        }
        cfg
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
    let parsed = reqwest::Url::parse(url).ok()?;
    let h = parsed.host_str()?;
    Some(
        h.trim_start_matches('[')
            .trim_end_matches(']')
            .to_ascii_lowercase(),
    )
}

/// Scheme + host (+ non-default port) of a provider URL, for startup logs.
/// Env-supplied URLs may carry credentials in the userinfo
/// (`https://user:key@host`) or query (`?api_key=…`); logging the raw value
/// writes them to the log sink (CWE-532). Goes through the WHATWG parser so
/// userinfo, path, query and fragment are dropped by construction, and keeps
/// scheme + port so an operator can tell an in-cluster provider from an
/// external one.
fn log_safe_origin(url: &str) -> String {
    match reqwest::Url::parse(url) {
        Ok(u) if u.origin().is_tuple() => u.origin().ascii_serialization(),
        // Parses, but has no scheme://host — e.g. `tei.mcp.svc:8080` with the
        // scheme forgotten. `null` (the WHATWG opaque-origin serialisation)
        // would read as "no origin configured".
        Ok(_) => "<no host>".to_string(),
        // `url::ParseError` variants are unit-like; Display never echoes input.
        Err(e) => format!("<unparseable: {e}>"),
    }
}

/// True for hosts that are not publicly routable: loopback, RFC1918, or
/// cluster-internal (`.svc`, `.internal`). Used to forbid the dev-only open
/// mode on a public origin. Real IP-literal parsing prevents confusable
/// hostnames like `127.0.0.1.evil.com` from masquerading as loopback.
fn is_private_host(url: &str) -> bool {
    host_of(url).is_some_and(|h| host_is_private(&h))
}

// Duplicated verbatim in ops-console/src/config.rs (no shared crate between
// a Leptos console and the server) — edit both. Drift is safe in one
// direction only: a stale copy is the NARROWER one, so it refuses a boot its
// sibling allows, loudly, in the deploy that introduced it.
fn host_is_private(h: &str) -> bool {
    // DNS-only special names — these can't be IP literals. Both the short
    // Service form and the fully-qualified one: `.svc.cluster.local` is what
    // most k8s docs show, and it ends with `.local`, so `.svc` alone refuses
    // a correct config. Both suffixes stay END-anchored — that is what stops
    // `evil.svc.attacker.com` matching, so neither may become a substring
    // test. A non-default cluster domain needs its literal added here.
    if h == "localhost"
        || h.ends_with(".svc")
        || h.ends_with(".svc.cluster.local")
        || h.ends_with(".internal")
    {
        return true;
    }
    // Anything else must parse as an actual IP literal to qualify as private.
    match h.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => v4.is_loopback() || v4.is_private(),
        // An IPv4-mapped literal (`::ffff:10.0.0.1`) is the v4 address it
        // wraps — `Ipv6Addr::is_loopback` is false for `::ffff:127.0.0.1`,
        // so judge the mapped address or a mapped loopback reads as public.
        // ULA (`fc00::/7`) is v6's private range; without it a v6-native
        // cluster is pushed onto DNS names for no security gain.
        Ok(std::net::IpAddr::V6(v6)) => match v6.to_ipv4_mapped() {
            Some(v4) => v4.is_loopback() || v4.is_private(),
            None => v6.is_loopback() || v6.is_unique_local(),
        },
        Err(_) => false,
    }
}

/// Cluster-local or private: `host_is_private`, plus single-label hostnames
/// (`http://anthropic-lb:8082`) — Kubernetes service DNS that never resolves
/// off the cluster. Decides whether plain HTTP with a credential is refused at
/// boot; the auth gates use `is_private_host` alone. Classifies the host as
/// reqwest parsed it, so userinfo cannot pose as the host and an IPv6 literal
/// (no dots) is never mistaken for a single-label service name.
fn is_cluster_local(url: &reqwest::Url) -> bool {
    let Some(h) = url.host_str() else {
        return false;
    };
    // Lowercase: the url crate only normalises special-scheme hosts (http/https);
    // non-special schemes (redis, rediss) preserve case from the input.
    let h = h
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase();
    let h = h.as_str();
    host_is_private(h) || (h.parse::<std::net::IpAddr>().is_err() && !h.contains('.'))
}

/// Which client dials the URL, and so which schemes it can speak. A scheme the
/// client cannot speak must be refused here, not approved and then discovered
/// at the first request — and the two sets do not overlap, so one merged set
/// is wrong for both.
#[derive(Clone, Copy)]
enum Transport {
    /// `reqwest` — `http` and `https` only; any other scheme is an error out of
    /// `Client::execute`, never a connection.
    Http,
    /// `fred`, via cachekit — `redis` and `rediss`, both of them plaintext:
    /// fred is built without TLS (`enable-rustls`/`enable-native-tls` are not
    /// compiled), so `rediss://` opens plain TCP despite the scheme name. It
    /// also never validates the scheme, so an `https://` cache URL does not
    /// fail — it dials plain TCP on 6379 and sends `AUTH` in the clear.
    Redis,
}

/// A credential sent in the clear to a host that is not cluster-local is a
/// credential on the wire. Fail closed at boot (Ray, 2026-09-11) against the
/// `Transport` that will carry it: `Http` takes `https` anywhere or plain
/// `http` to a cluster-local endpoint such as `http://anthropic-lb:8082`,
/// `Redis` takes only a cluster-local `redis://redis-svc:6379`. The credential
/// is the API key when one is set, or URL-embedded credentials
/// (`http://user:secret@host` sent as Basic auth by reqwest,
/// `redis://user:pw@host` sent as `AUTH` by fred). Classified on the URL as the
/// `url` crate parses it (lowercased
/// scheme, real host), so the check and the transport cannot disagree about
/// where the credential goes. Messages name the host, never the raw value: a
/// URL may carry credentials.
fn check_credential_transport(
    var: &str,
    url: &str,
    has_api_key: bool,
    transport: Transport,
) -> Result<(), String> {
    let parsed = match reqwest::Url::parse(url) {
        Ok(parsed) => parsed,
        // No credential means no question for this guard, so it must not
        // escalate — refusing here would stop the service over a var it was
        // never asked to validate. Say so, though: the client that dials it
        // fails much later with nothing naming the var.
        Err(e) if !has_api_key => {
            tracing::warn!(
                op = "credential_transport_guard",
                var,
                err = %e,
                "not a parseable URL; no credential to protect, so boot continues"
            );
            return Ok(());
        }
        Err(e) => return Err(format!("{var} is not a valid URL ({e})")),
    };
    let has_userinfo = !parsed.username().is_empty() || parsed.password().is_some();
    if !has_api_key && !has_userinfo {
        return Ok(());
    }
    match (transport, parsed.scheme()) {
        (Transport::Http, "https") => Ok(()),
        (Transport::Http, "http") if is_cluster_local(&parsed) => Ok(()),
        // fred dispatches on scheme suffix (`redis-cluster`, `rediss-cluster`,
        // `redis-sentinel`), all plaintext (no TLS compiled).
        (Transport::Redis, scheme) if scheme.starts_with("redis") && is_cluster_local(&parsed) => {
            Ok(())
        }
        // Unusable scheme: reqwest rejects non-http at `Client::execute`, so
        // nothing is ever dialled and "in the clear" would be false — the host
        // in `QDRANT_URL=redis://qdrant:6333` IS cluster-local. Off-cluster
        // `http` is a real cleartext fault and must fall through (hence
        // `!= "http"`).
        (Transport::Http, scheme) if scheme != "http" => Err(format!(
            "{var}: unusable scheme {scheme}:// — this client speaks http and https only",
        )),
        // fred opens plain TCP regardless of scheme, so `https://` buys no
        // encryption — it just makes fred dial port 6379 and send `AUTH` in
        // the clear.
        (Transport::Redis, scheme) if !scheme.starts_with("redis") => Err(format!(
            "{var}: unusable scheme {scheme}:// — this client speaks redis:// \
             variants only (fred opens plain TCP regardless of scheme)",
        )),
        (_, scheme) => Err(format!(
            "{var}: {scheme}://{} is not {}; an API key or URL credential must \
             not travel in the clear",
            parsed.host_str().unwrap_or(""),
            match transport {
                Transport::Http => "https and not a cluster-local http endpoint",
                Transport::Redis =>
                    "a cluster-local redis:// or rediss:// endpoint (this client has no TLS)",
            }
        )),
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
///
/// Reads every optional value, because a whitespace-only one that survives as
/// `Some` reads as configured everywhere downstream — `OIDC_ISSUER="   "` used
/// to satisfy the fail-closed "some auth is configured" check at boot.
/// Deliberately not the bearer vars (`ALAYA_API_KEY`, `ALAYA_READONLY_API_KEY`,
/// `GRAPH_API_KEY`): both ends of those compare the bytes they were given, so
/// trimming one end alone would break the match. They are trimmed in the
/// ExternalSecret template instead, where both ends see it.
fn env_non_empty(key: &str) -> Option<String> {
    non_empty_trimmed(std::env::var(key).ok())
}

/// The transformation above, split from the read so it can be pinned by a
/// test: `set_var` is `unsafe` in edition 2024 and races every other test in
/// the binary. Same shape as `parse_judge_daily_cap`, for the same reason.
fn non_empty_trimmed(raw: Option<String>) -> Option<String> {
    raw.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
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
            tracing::info!(backend = %backend, "L2 embedding cache enabled");
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
/// Pinger period — keeps worker progress fresh when the service is idle and
/// bounds how stale the bare probe's Qdrant verdict can be (#78).
const PING_PERIOD: std::time::Duration = std::time::Duration::from_secs(30);

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Monotonic seconds for the worker heartbeat: elapsed since the first call
/// in this process, `+1` so a live stamp can never read as the `progress` 0
/// sentinel ("loop not entered yet") during the first second of uptime.
///
/// Deliberately not `epoch_secs`: a forward wall-clock step larger than
/// `WORKER_STALL_THRESHOLD` — NTP correcting a drifted node, a VM resume —
/// ages a stamp a healthy worker wrote seconds ago, and the unauthenticated
/// liveness route then 503s a pod that is fine (LAB-3968). `Instant` does not
/// move when the wall clock does. A backward step was not harmless either:
/// `saturating_sub` floored the age at 0, so a wedged worker read healthy
/// until the clock caught back up. Monotonic closes both halves.
fn monotonic_secs() -> u64 {
    static START: std::sync::LazyLock<std::time::Instant> =
        std::sync::LazyLock::new(std::time::Instant::now);
    START.elapsed().as_secs() + 1
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
        offset: usize,
        include_resolved: bool,
        verdicts: Option<Vec<String>>,
        reply: oneshot::Sender<Value>,
    },
    /// Stamp (`Some`) or clear (`None`) the operator's resolution on a
    /// CONTRADICTS pair (LAB-3885). The only write path to `e.resolution*`.
    ResolveContradiction {
        memory_a_hash: String,
        memory_b_hash: String,
        resolution: Option<Resolution>,
        resolved_via: String,
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
    /// Judge up to `limit` CONTRADICTS pairs with no verdict (LAB-3283 AC-5);
    /// `rejudge` also re-annotates pairs judged by a different model.
    BackfillContradictions {
        limit: usize,
        rejudge: bool,
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
            | CmdInner::BackfillSummaries { .. }
            | CmdInner::BackfillContradictions { .. } => LONG_CMD_DEADLINE,
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
            CmdInner::ResolveContradiction { .. } => "resolve_contradiction",
            CmdInner::FindDuplicates { .. } => "find_duplicates",
            CmdInner::MergeDuplicates { .. } => "merge_duplicates",
            CmdInner::Patch { .. } => "patch",
            CmdInner::BackfillSummaries { .. } => "backfill_summaries",
            CmdInner::BackfillContradictions { .. } => "backfill_contradictions",
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
/// checker also watches `worker_progress` — `monotonic_secs` of the last
/// command the worker completed (pings keep it fresh when idle). Stale
/// progress means the loop stopped draining: status goes `unhealthy` and
/// /health returns 503 so a liveness probe restarts the pod. Backend outages
/// stay `degraded` (200) — restarting this pod doesn't fix Qdrant.
#[derive(Clone)]
struct HealthChecker {
    client: reqwest::Client,
    qdrant_url: String,
    qdrant_api_key: Option<String>,
    collection: String,
    embedding_url: String,
    graph_url: String,
    graph_api_key: String,
    worker_progress: std::sync::Arc<std::sync::atomic::AtomicU64>,
    stall_threshold: std::time::Duration,
    /// Time source for the stall age. `monotonic_secs` in production; a fixed
    /// fake in tests, because that clock's origin is its own first call — a
    /// test cannot otherwise hold a stamp that is genuinely 3600s old.
    clock: fn() -> u64,
    /// Last Qdrant verdict, written by the pinger via `refresh_qdrant` and
    /// read by the bare probe — which therefore never touches Qdrant itself.
    qdrant_ok: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl HealthChecker {
    fn new(config: &Config, worker_progress: std::sync::Arc<std::sync::atomic::AtomicU64>) -> Self {
        // One client, three backends: each credential is attached per
        // request (`bearer_auth`), never as a client default header, so
        // Qdrant's bearer is not sent to the embedding endpoint or the bridge.
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build health check client");

        Self {
            client,
            qdrant_url: config.qdrant_url.clone(),
            qdrant_api_key: config.qdrant_api_key.clone(),
            collection: config.qdrant_collection.clone(),
            embedding_url: config.embedding_url.clone(),
            graph_url: config.graph_url.clone(),
            graph_api_key: config.graph_api_key.clone(),
            worker_progress,
            stall_threshold: WORKER_STALL_THRESHOLD,
            clock: monotonic_secs,
            qdrant_ok: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    fn worker_state(&self) -> (&'static str, bool, u64) {
        let last_progress = self
            .worker_progress
            .load(std::sync::atomic::Ordering::Relaxed);
        if last_progress == 0 {
            ("starting", false, 0)
        } else {
            let age = (self.clock)().saturating_sub(last_progress);
            let stalled = age > self.stall_threshold.as_secs();
            (if stalled { "stalled" } else { "ok" }, stalled, age)
        }
    }

    fn status_from(worker_stalled: bool, qdrant_ok: bool) -> &'static str {
        if worker_stalled {
            "unhealthy"
        } else if qdrant_ok {
            "healthy"
        } else {
            "degraded"
        }
    }

    /// Bare-probe path: two atomic reads, zero backend I/O (#78).
    ///
    /// The worker-stall side — the only input to the 503 decision — is read
    /// live. The Qdrant side only picks `healthy` vs `degraded` (both 200) and
    /// comes from the verdict the pinger last published, so an anonymous
    /// caller can neither proxy load into Qdrant nor time an outage off the
    /// probe. Staleness is bounded by `PING_PERIOD` plus the probe client's
    /// 10s timeout — well inside `WORKER_STALL_THRESHOLD`.
    fn check_status(&self) -> Value {
        let (_, worker_stalled, progress_age) = self.worker_state();
        let qdrant_ok = self.qdrant_ok.load(std::sync::atomic::Ordering::Relaxed);
        let status = Self::status_from(worker_stalled, qdrant_ok);

        if worker_stalled {
            tracing::error!(
                progress_age_s = progress_age,
                threshold_s = self.stall_threshold.as_secs(),
                "service worker stalled — reporting unhealthy so the pod gets restarted"
            );
        } else {
            tracing::debug!(op = "health", status, "probe");
        }
        json!({ "status": status })
    }

    /// Probe Qdrant once and publish the verdict `check_status` serves.
    /// Driven by the pinger, off the request path on purpose (#78).
    async fn refresh_qdrant(&self) {
        let ok = self.check_qdrant().await.is_ok();
        self.qdrant_ok
            .store(ok, std::sync::atomic::Ordering::Relaxed);
    }

    async fn check(&self) -> Value {
        let start = std::time::Instant::now();

        let (qdrant_health, graph_health, count) =
            tokio::join!(self.check_qdrant(), self.check_graph(), self.check_count(),);

        let (worker_state, worker_stalled, progress_age) = self.worker_state();
        let status = Self::status_from(worker_stalled, qdrant_health.is_ok());

        let elapsed = start.elapsed().as_millis();
        if worker_stalled {
            tracing::error!(
                progress_age_s = progress_age,
                threshold_s = self.stall_threshold.as_secs(),
                "service worker stalled — reporting unhealthy so the pod gets restarted"
            );
        } else {
            // `check_detail` may still downgrade this to `degraded` (embedding).
            tracing::debug!(op = "health", elapsed_ms = elapsed, status, "ok (direct)");
        }

        json!({
            "status": status,
            // "is build X live?" without cluster access (#70)
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

    fn qdrant_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.qdrant_api_key {
            Some(key) => req.bearer_auth(key),
            None => req,
        }
    }

    async fn check_qdrant(&self) -> Result<Value, String> {
        let resp = self
            .qdrant_auth(self.client.get(format!(
                "{}/collections/{}",
                self.qdrant_url, self.collection
            )))
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
            .qdrant_auth(self.client.post(format!(
                "{}/collections/{}/points/count",
                self.qdrant_url, self.collection
            )))
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

    /// TEI / vLLM readiness endpoint — 200 once the model is loaded. The
    /// same probe `EmbeddingClient::health` makes on the worker side; this
    /// copy exists because the checker bypasses the worker (see struct doc).
    async fn check_embedding(&self) -> Result<Value, String> {
        let resp = self
            .client
            .get(format!("{}/health", self.embedding_url))
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!("embedding returned {}", resp.status()));
        }
        Ok(json!({ "status": "healthy" }))
    }

    /// Operator view: the full document plus the embedding probe (LAB-4025).
    /// Only this path contacts the embedding endpoint — the bare probe must
    /// not grow another anonymous fan-out (#78). A dead endpoint degrades
    /// `status` but never overrides `unhealthy`: that word is the
    /// worker-stall 503 contract (#63), and a restart does not fix TEI.
    async fn check_detail(&self) -> Value {
        let (mut v, embedding) = tokio::join!(self.check(), self.check_embedding());
        if embedding.is_err() && v["status"] == "healthy" {
            v["status"] = json!("degraded");
        }
        v["embedding_health"] = match embedding {
            Ok(e) => e,
            Err(e) => json!({ "status": "unhealthy", "error": e }),
        };
        v
    }
}

// ─── Service worker ─────────────────────────────────────────────────────────

/// Deadlines and limits the worker applies per command. A struct only so tests can
/// shrink them — in production, deadlines use Default and judge_daily_cap is passed from Config.
struct WorkerLimits {
    cmd: std::time::Duration,
    long: std::time::Duration,
    judge_daily_cap: usize,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            cmd: CMD_DEADLINE,
            long: LONG_CMD_DEADLINE,
            judge_daily_cap: JUDGE_DAILY_CAP_DEFAULT,
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

    // LAB-3283: one cap on in-flight judge calls shared by store-path spawns
    // and the backfill (a bulk import must not fan thousands of calls at the
    // LB), and a single-flight guard so an operator retry after the reply
    // deadline cannot run a second pass over the same unjudged pairs.
    let judge_gate = std::rc::Rc::new(tokio::sync::Semaphore::new(JUDGE_CONCURRENCY));
    let backfill_running = std::rc::Rc::new(std::cell::Cell::new(false));
    let judge_limiter = std::rc::Rc::new(std::cell::RefCell::new(JudgeDailyCap::new(
        limits.judge_daily_cap,
    )));

    // First heartbeat: bootstrap is done and the loop is live. Until this
    // stamp the health checker reports the worker as "starting" (progress
    // sentinel 0), never stalled — see the seed in main().
    progress.store(monotonic_secs(), std::sync::atomic::Ordering::Relaxed);

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
                                svc.enrich_summary(&hash_owned, &content).await;
                            });
                        }

                        // Fire-and-forget (LAB-3283 AC-4): judge each new
                        // CONTRADICTS signal off the store path. Suppressed
                        // under read_only — no edges were written — and when
                        // no judge is configured. The edge direction is
                        // (new) -> (existing), matching the batch write above.
                        // Only a genuinely new memory: a re-store re-runs
                        // interference but its edges (and verdicts) already
                        // exist — re-judging would churn and re-bill them.
                        // LAB-3895: store-path judge spend is capped per UTC day.
                        spawn_store_contradiction_judges(
                            &svc,
                            &judge_gate,
                            &judge_limiter,
                            &r,
                            read_only,
                        );

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
            CmdInner::Contradictions {
                limit,
                offset,
                include_resolved,
                verdicts,
                reply,
            } => {
                let span =
                    tracing::info_span!(parent: &ps, "contradictions", offset, include_resolved);
                let result = match timeout(
                    limits.cmd,
                    svc.memory_contradictions(limit, offset, include_resolved, verdicts.as_deref())
                        .instrument(span),
                )
                .await
                {
                    Ok(Ok(r)) => {
                        let pairs = r
                            .get("pairs")
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

            CmdInner::ResolveContradiction {
                memory_a_hash,
                memory_b_hash,
                resolution,
                resolved_via,
                reply,
            } => {
                let span = tracing::info_span!(parent: &ps, "resolve_contradiction");
                let a = truncate_hash(&memory_a_hash);
                let b = truncate_hash(&memory_b_hash);
                let stamp = resolution.map(|r| r.as_str()).unwrap_or("clear");
                let result = match timeout(
                    limits.cmd,
                    svc.resolve_contradiction(
                        &memory_a_hash,
                        &memory_b_hash,
                        resolution,
                        &resolved_via,
                    )
                    .instrument(span),
                )
                .await
                {
                    Ok(Ok(r)) => {
                        tracing::info!(
                            op,
                            a = a.as_str(),
                            b = b.as_str(),
                            resolution = stamp,
                            via = resolved_via.as_str(),
                            elapsed_ms = ms(start),
                            "ok"
                        );
                        r
                    }
                    Ok(Err(e)) => {
                        tracing::error!(
                            op,
                            error = %e,
                            a = a.as_str(),
                            b = b.as_str(),
                            resolution = stamp,
                            elapsed_ms = ms(start),
                            "failed"
                        );
                        json!({"success": false, "error": e.safe_message()})
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
                            svc.enrich_summary(hash, content).await;
                        }
                        tracing::info!(queued, "backfill summaries complete");
                    }
                    .instrument(span),
                );
            }
            CmdInner::BackfillContradictions {
                limit,
                rejudge,
                reply,
            } => {
                if backfill_running.replace(true) {
                    let _ = reply.send(json!({
                        "success": false,
                        "error": "backfill already running"
                    }));
                } else {
                    let span = tracing::info_span!(parent: &ps, "backfill_contradictions");
                    let svc = svc.clone();
                    let gate = judge_gate.clone();
                    let running = backfill_running.clone();
                    tokio::task::spawn_local(
                        async move {
                            run_backfill_contradictions(&svc, &gate, limit, rejudge, reply).await;
                            running.set(false);
                        }
                        .instrument(span),
                    );
                }
            }
        }

        // Watchdog heartbeat: the loop just finished (or spawned) a command.
        // Stops advancing exactly when the worker stops draining.
        progress.store(monotonic_secs(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// `existing_hash` of every contradiction signal in a store result, deduped
/// (negation and temporal cues can both fire on one neighbour).
fn contradicted_hashes(store_result: &std::collections::HashMap<String, Value>) -> Vec<String> {
    let mut hashes: Vec<String> = store_result
        .get("interference")
        .and_then(|i| i.get("contradictions"))
        .and_then(Value::as_array)
        .map(|signals| {
            signals
                .iter()
                .filter_map(|s| s.get("existing_hash").and_then(Value::as_str))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    hashes.sort();
    hashes.dedup();
    hashes
}

/// Default daily cap for store-path judge calls (LAB-3895).
const JUDGE_DAILY_CAP_DEFAULT: usize = 1000;

fn parse_judge_daily_cap(raw: Option<String>) -> Result<usize, String> {
    match raw {
        None => Ok(JUDGE_DAILY_CAP_DEFAULT),
        Some(s) if s.trim().is_empty() => Ok(JUDGE_DAILY_CAP_DEFAULT),
        Some(s) => s.trim().parse::<usize>().map_err(|e| {
            format!("JUDGE_DAILY_CAP must be a non-negative integer (e.g. 1000): {s} ({e})")
        }),
    }
}

/// Returns (year, month, day) in UTC for a given Unix timestamp in seconds.
/// Implements Howard Hinnant's civil calendar algorithm (pure integer math).
fn utc_date(epoch_secs: u64) -> (i32, u32, u32) {
    let days = (epoch_secs / 86400) as i64;
    let z = days + 719468;
    let era = z / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

fn utc_date_str(epoch_secs: u64) -> String {
    let (y, m, d) = utc_date(epoch_secs);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Bounds store-path contradiction judging (LAB-3895) on two independent axes:
/// `cap` judge calls per UTC day (the spend budget, `JUDGE_DAILY_CAP`) and
/// `JUDGE_STORE_BACKLOG_MAX` tasks outstanding (queued on the gate or in
/// flight), so a stalled judge endpoint cannot back up one task per
/// contradicted pair without limit. Separate knobs on purpose: raising the
/// day's budget for a bulk import must not raise the number of live tasks by
/// the same factor. A pair refused by either bound stays unjudged for operator
/// backfill, which is subject to neither.
struct JudgeDailyCap {
    cap: usize,
    current_day: u64,
    count: usize,
    /// One WARN per reason per UTC day. Budget and backlog are latched
    /// separately because they ask the operator for different things — raise
    /// the cap or drain the queue, versus the judge endpoint is not answering —
    /// and a shared latch would hide whichever fired second for the rest of the
    /// day.
    warned_budget: bool,
    warned_backlog: bool,
    /// One permit per outstanding task, held until the task ends. `Arc`, not
    /// `Rc`: `try_acquire_owned` needs it; the worker is single-threaded anyway.
    slots: std::sync::Arc<tokio::sync::Semaphore>,
    /// Clock, so tests can queue calls across a UTC-day boundary.
    clock: fn() -> u64,
}

impl JudgeDailyCap {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            current_day: u64::MAX,
            count: 0,
            warned_budget: false,
            warned_backlog: false,
            slots: std::sync::Arc::new(tokio::sync::Semaphore::new(JUDGE_STORE_BACKLOG_MAX)),
            clock: epoch_secs,
        }
    }

    /// Start a new UTC day when `now_secs` has crossed into one: fresh budget,
    /// fresh warnings. Returns the day key `now_secs` falls in.
    fn roll_day(&mut self, now_secs: u64) -> u64 {
        let day = now_secs / 86400;
        if self.current_day != day {
            self.current_day = day;
            self.count = 0;
            self.warned_budget = false;
            self.warned_backlog = false;
        }
        day
    }

    /// Reserve a slot for one store-path task; `None` once
    /// `JUDGE_STORE_BACKLOG_MAX` tasks are outstanding.
    ///
    /// This refusal warns rather than leaving it to `try_admit`: it happens
    /// *before* a task exists to consult the budget, so against a stalled judge
    /// endpoint it is the only skip an operator would ever see.
    fn try_slot(&mut self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        if let Ok(permit) = self.slots.clone().try_acquire_owned() {
            return Some(permit);
        }
        let now = (self.clock)();
        self.roll_day(now);
        let date = utc_date_str(now);
        const MSG: &str = "store-path judge backlog full; pair left unjudged for backfill";
        if !self.warned_backlog {
            tracing::warn!(reason = "backlog", max = JUDGE_STORE_BACKLOG_MAX, date = %date, MSG);
            self.warned_backlog = true;
        } else {
            tracing::debug!(reason = "backlog", max = JUDGE_STORE_BACKLOG_MAX, date = %date, MSG);
        }
        None
    }

    /// Try to admit one pair to be judged off the store path at the current time.
    fn try_admit(&mut self) -> Option<u64> {
        self.try_admit_at((self.clock)())
    }

    /// Try to admit one pair at the given unix timestamp (seconds). `Some(day)`
    /// has billed the day's budget; hand that day back to `refund` if the call
    /// turns out to have spent nothing.
    fn try_admit_at(&mut self, now_secs: u64) -> Option<u64> {
        let day = self.roll_day(now_secs);
        if self.count < self.cap {
            self.count += 1;
            return Some(day);
        }
        let date = utc_date_str(now_secs);
        const MSG: &str = "judge daily cap reached; skipping store-path contradiction judge call";
        if !self.warned_budget {
            tracing::warn!(reason = "budget", cap = self.cap, date = %date, MSG);
            self.warned_budget = true;
        } else {
            tracing::debug!(reason = "budget", cap = self.cap, date = %date, MSG);
        }
        None
    }

    /// Give back a unit billed by `try_admit` for a call that spent nothing.
    /// Ignored once the UTC day has rolled: that unit was drawn on a budget
    /// that has already reset, and refunding it would credit the wrong day.
    fn refund(&mut self, day: u64) {
        if self.current_day == day && self.count > 0 {
            self.count -= 1;
        }
    }
}

/// Spawns background contradiction judge tasks for new CONTRADICTS signals from a store result
/// (LAB-3895). Returns the number of tasks spawned; a pair is skipped (left unjudged for
/// backfill) once `JUDGE_STORE_BACKLOG_MAX` tasks are outstanding. The daily cap itself is
/// applied inside each task, once it holds a gate permit and just before the call: a pair
/// queued before UTC midnight is billed to the day it actually runs, so a process's calls in
/// any UTC day never exceed the cap.
fn spawn_store_contradiction_judges(
    svc: &std::rc::Rc<MemoryService>,
    judge_gate: &std::rc::Rc<tokio::sync::Semaphore>,
    limiter: &std::rc::Rc<std::cell::RefCell<JudgeDailyCap>>,
    result: &std::collections::HashMap<String, Value>,
    read_only: bool,
) -> usize {
    let skipped = result
        .get("duplicate")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if read_only
        || svc.judge.is_none()
        || skipped
        || result.get("created").and_then(Value::as_bool) != Some(true)
    {
        return 0;
    }
    let Some(new_hash) = result.get("content_hash").and_then(|v| v.as_str()) else {
        return 0;
    };

    let mut spawned = 0;
    for dst in contradicted_hashes(result) {
        // Bound the borrow to this statement: `try_slot` warns, so it needs `&mut`.
        let slot = limiter.borrow_mut().try_slot();
        let Some(slot) = slot else { continue };
        let src = new_hash.to_string();
        let svc = svc.clone();
        let gate = judge_gate.clone();
        let limiter = limiter.clone();
        tokio::task::spawn_local(async move {
            let _slot = slot; // released when this task ends, refused or judged
            let Ok(_permit) = gate.acquire().await else {
                return;
            };
            // Synchronous, so the borrow ends before the await below.
            let admitted = limiter.borrow_mut().try_admit();
            let Some(day) = admitted else {
                return;
            };
            let outcome = svc.judge_contradiction(&src, &dst).await;
            // Give the unit back when nothing was billed: a graph or vector blip
            // mid-import must not spend the day's budget on no-ops. `spent()`
            // owns which paths are free; a judged pair stays billed even if the
            // edge write failed — the tokens went out either way.
            if !outcome.spent() {
                limiter.borrow_mut().refund(day);
            }
        });
        spawned += 1;
    }
    spawned
}

/// In-flight judge calls during a backfill (LAB-3283 AC-5).
const JUDGE_CONCURRENCY: usize = 4;
/// Store-path judge tasks that may be outstanding at once — queued on the gate
/// or in flight (LAB-3895). Bounds how far a stalled judge endpoint can back up
/// behind `JUDGE_CONCURRENCY`; deliberately *not* `JUDGE_DAILY_CAP`, which is a
/// spend budget and would otherwise double as a throughput limit.
const JUDGE_STORE_BACKLOG_MAX: usize = 64;
/// Retries on 429 before a pair is counted unjudged (AC-9).
const JUDGE_MAX_RETRIES: u32 = 5;
const JUDGE_MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(60);

#[derive(Default, Debug, PartialEq)]
struct BackfillTotals {
    judged: usize,
    /// Judged verdicts that actually landed on an edge.
    persisted: usize,
    /// Deterministic failures persisted as an `unjudged` marker (skipped next pass).
    marked: usize,
    /// Transient failures: nothing written, retried next pass.
    unjudged: usize,
    input_tokens: u64,
    output_tokens: u64,
}

/// One backfill pass: fetch unjudged pairs, judge them, reply with totals.
/// Runs detached; every exit path replies exactly once.
async fn run_backfill_contradictions(
    svc: &MemoryService,
    gate: &std::rc::Rc<tokio::sync::Semaphore>,
    limit: usize,
    rejudge: bool,
    reply: oneshot::Sender<Value>,
) {
    let Some(judge) = svc.judge.as_ref() else {
        let _ = reply.send(json!({
            "success": false,
            "error": "contradiction judge not configured"
        }));
        return;
    };
    // Only edges with no verdict at all: a persisted `unjudged` marker is a
    // deterministic failure and is skipped, so re-running is idempotent and
    // never re-spends (AC-5). `rejudge` widens the selection to edges judged
    // by a different model — the recovery path for a model switch. Resolved
    // pairs are hidden from the read surface, so judging them is waste.
    let query = ContradictionQuery {
        limit,
        needs_judging: true,
        rejudge_model: rejudge.then(|| judge.model_name().to_string()),
        exclude_resolved: true,
        ..Default::default()
    };
    let pairs = match svc.graph.get_all_contradictions(&query).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("backfill: fetching unjudged pairs failed: {e}");
            let _ = reply.send(json!({"success": false, "error": e.safe_message()}));
            return;
        }
    };
    let queued = pairs.len();
    tracing::info!(queued, rejudge, "backfill: judging contradictions");
    let t = backfill_judge(svc, gate, pairs).await;
    tracing::info!(
        queued,
        judged = t.judged,
        persisted = t.persisted,
        marked = t.marked,
        unjudged = t.unjudged,
        input_tokens = t.input_tokens,
        output_tokens = t.output_tokens,
        "backfill contradictions complete"
    );
    let _ = reply.send(json!({
        "queued": queued,
        "judged": t.judged,
        "persisted": t.persisted,
        "marked": t.marked,
        "unjudged": t.unjudged,
        "input_tokens": t.input_tokens,
        "output_tokens": t.output_tokens,
    }));
}

/// Judge `pairs` with at most `JUDGE_CONCURRENCY` calls in flight (the
/// semaphore is shared with store-path spawns, so the cap is global).
async fn backfill_judge(
    svc: &MemoryService,
    gate: &std::rc::Rc<tokio::sync::Semaphore>,
    pairs: Vec<Contradiction>,
) -> BackfillTotals {
    use futures::StreamExt;
    let outcomes: Vec<JudgeOutcome> = futures::stream::iter(pairs)
        .map(|p| async move {
            // Permit per attempt, not per pair: a rate-limit sleep inside
            // `with_backoff` must not hold one of the four shared permits
            // (four sleeping pairs would stall the pass and the store path).
            with_backoff(|| async {
                let Ok(_permit) = gate.acquire().await else {
                    return JudgeOutcome::Unjudged {
                        marked: false,
                        spent: false,
                    };
                };
                svc.judge_contradiction(&p.memory_a_hash, &p.memory_b_hash)
                    .await
            })
            .await
        })
        .buffer_unordered(JUDGE_CONCURRENCY)
        .collect()
        .await;
    let mut t = BackfillTotals::default();
    for o in outcomes {
        match o {
            JudgeOutcome::Judged {
                judgement,
                persisted,
            } => {
                t.judged += 1;
                if persisted {
                    t.persisted += 1;
                }
                t.input_tokens += judgement.input_tokens;
                t.output_tokens += judgement.output_tokens;
            }
            JudgeOutcome::Unjudged { marked: true, .. } => t.marked += 1,
            // `with_backoff` never returns RateLimited; treat it as transient.
            JudgeOutcome::Unjudged { marked: false, .. } | JudgeOutcome::RateLimited { .. } => {
                t.unjudged += 1
            }
        }
    }
    t
}

/// Retry one judge attempt on 429, honouring `retry-after` when the LB
/// sends it and doubling from 1s (capped) when it doesn't. Any other
/// outcome is final; exhausted retries count as a transient unjudged.
async fn with_backoff<F, Fut>(mut attempt: F) -> JudgeOutcome
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = JudgeOutcome>,
{
    let mut backoff = std::time::Duration::from_secs(1);
    for n in 0..=JUDGE_MAX_RETRIES {
        match attempt().await {
            outcome @ (JudgeOutcome::Judged { .. } | JudgeOutcome::Unjudged { .. }) => {
                return outcome;
            }
            JudgeOutcome::RateLimited { retry_after_secs } => {
                if n == JUDGE_MAX_RETRIES {
                    tracing::warn!(
                        retries = n,
                        "backfill: still rate limited, giving up on pair"
                    );
                    return JudgeOutcome::Unjudged {
                        marked: false,
                        spent: false,
                    };
                }
                let wait = retry_after_secs
                    .map(std::time::Duration::from_secs)
                    .unwrap_or(backoff)
                    .min(JUDGE_MAX_BACKOFF);
                tracing::info!(
                    attempt = n + 1,
                    wait_s = wait.as_secs(),
                    "backfill: rate limited, backing off"
                );
                tokio::time::sleep(wait).await;
                backoff = (backoff * 2).min(JUDGE_MAX_BACKOFF);
            }
        }
    }
    JudgeOutcome::Unjudged {
        marked: false,
        spent: false,
    }
}

/// Fire-and-forget summary generation helper.
/// Called from spawn_local — logs errors, never panics.
/// Generates summary text AND its embedding for search boosting.
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
    s[..s.floor_char_boundary(8.min(s.len()))].to_string()
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
    let readonly_api_key = if config.readonly_api_key.is_empty() {
        None
    } else {
        Some(config.readonly_api_key.clone())
    };

    // Fail-closed: `authenticate` checks the full key first, so equal keys
    // would silently resolve the "read-only" bearer to Full. Refuse to start.
    if let (Some(full), Some(ro)) = (&api_key, &readonly_api_key)
        && full == ro
    {
        panic!("ALAYA_READONLY_API_KEY must differ from ALAYA_API_KEY");
    }

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
    // A readonly key alone counts as configured auth (reads-only deployment).
    if api_key.is_none() && readonly_api_key.is_none() && oidc.is_none() {
        if !config.allow_unauthenticated {
            panic!(
                "no auth configured: set ALAYA_API_KEY, ALAYA_READONLY_API_KEY, \
                 or OIDC_ISSUER, or DANGEROUSLY_ALLOW_UNAUTHENTICATED=true for dev"
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

    if readonly_api_key.is_some() {
        tracing::info!("read-only static bearer enabled (ALAYA_READONLY_API_KEY)");
    }

    AuthState {
        api_key,
        readonly_api_key,
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

/// Run the service worker under a supervisor that takes the process down
/// if it panics. Left to unwind, a panic ends only that thread: the listener
/// stays bound, every worker-backed request answers 503, and /health reads
/// progress 0 as "starting" forever, so nothing restarts the pod (#97).
/// Exiting non-zero needs no probe to act and turns the outage into
/// CrashLoopBackOff instead of Running 0/1. The shutdown path joins the
/// returned handle, which returns once the worker has drained normally.
fn supervise(worker: std::thread::JoinHandle<()>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        if worker.join().is_err() {
            // The panic hook has already written the reason to stderr.
            tracing::error!("service worker thread panicked — exiting so the pod restarts");
            std::process::exit(101);
        }
    })
}

fn main() {
    // Before `Config::from_env`, so the boot guard's warnings reach a
    // subscriber: `init_tracing` needs no runtime (its OTLP batch processor
    // runs on its own OS thread with a blocking client, which is also happier
    // constructed outside one).
    telemetry::init_tracing();
    let config = Config::from_env();

    // Multi-threaded runtime for axum; LocalSet thread for MemoryService
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build runtime");

    rt.block_on(async move {
        let (tx, rx) = mpsc::channel::<Cmd>(CMD_CHANNEL_CAP);

        // Watchdog heartbeat: `monotonic_secs` of the worker's last completed
        // command. Written by the worker loop, read by the health checker.
        // Seeded 0 = "worker loop not entered yet": backend bootstrap
        // (ensure_qdrant_collection + init_l2_cache retries) can legitimately
        // exceed the stall threshold on a cluster cold start, and /health is
        // already serving — a non-sentinel seed here would misreport that as
        // a stall and restart-loop the pod. Every bootstrap await is
        // deadline-bounded, so the loop is always entered in bounded time.
        let worker_progress = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let progress_for_worker = worker_progress.clone();

        // Built here, on the main thread, so bad bearer material is a
        // fail-closed startup panic (non-zero exit) like the auth invariants
        // below — not a panic inside the worker thread (#97). The value is
        // the secret being rejected: name the env var, never log it.
        let graph = GraphHttpClient::new(config.graph_url.clone(), &config.graph_api_key)
            .expect("GRAPH_API_KEY rejected — must be a single line of visible ASCII");

        let qdrant = QdrantClient::new(
            config.qdrant_url.clone(),
            config.qdrant_collection.clone(),
            config.qdrant_api_key.clone(),
        )
        .expect("QDRANT_API_KEY rejected — must be a single line of visible ASCII");

        let summary: Option<SummaryClient> = if let Some(url) = &config.summary_url {
            tracing::info!(
                origin = log_safe_origin(url).as_str(),
                model = config.summary_model.as_str(),
                has_api_key = config.summary_api_key.is_some(),
                "summary provider enabled"
            );
            Some(
                SummaryClient::new(
                    url.clone(),
                    config.summary_model.clone(),
                    config.summary_api_key.clone(),
                )
                .expect("SUMMARY_API_KEY rejected — must be a single line of visible ASCII"),
            )
        } else {
            tracing::info!("SUMMARY_URL not set — auto-summary disabled");
            None
        };

        let rerank: Option<RerankClient> = if let Some(url) = &config.rerank_url {
            tracing::info!(
                origin = log_safe_origin(url).as_str(),
                top_n = config.rerank_top_n,
                timeout_ms = config.rerank_timeout_ms.get(),
                has_api_key = config.rerank_api_key.is_some(),
                "cross-encoder reranker enabled"
            );
            Some(
                RerankClient::new(
                    url.clone(),
                    config.rerank_top_n,
                    config.rerank_api_key.clone(),
                    std::time::Duration::from_millis(config.rerank_timeout_ms.get()),
                )
                .expect("RERANK_API_KEY rejected — must be a single line of visible ASCII"),
            )
        } else {
            tracing::info!("RERANK_URL not set — cross-encoder rerank disabled");
            None
        };

        let judge: Option<JudgeClient> = if let Some(url) = &config.judge_url {
            tracing::info!(
                origin = log_safe_origin(url).as_str(),
                model = config.judge_model.as_str(),
                has_api_key = config.judge_api_key.is_some(),
                daily_cap = config.judge_daily_cap,
                "contradiction judge enabled (advisory: annotates CONTRADICTS edges, never writes memories)"
            );
            Some(
                JudgeClient::new(
                    url.clone(),
                    config.judge_model.clone(),
                    config.judge_api_key.clone(),
                )
                .expect("JUDGE_API_KEY rejected — must be a single line of visible ASCII"),
            )
        } else {
            tracing::info!("JUDGE_URL/SUMMARY_URL not set — contradiction judge disabled");
            None
        };

        // Spawn MemoryService on a dedicated thread with LocalSet
        let cfg_clone = config.clone();

        let worker_handle = supervise(std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build local runtime");

            let local = tokio::task::LocalSet::new();
            local.block_on(&rt, async move {
                // Fresh-deploy bootstrap: create the memory collection if it is
                // absent so the first write doesn't 404 (#31).
                ensure_qdrant_collection(&qdrant, cfg_clone.embedding_dimensions).await;
                let embeddings = EmbeddingClient::new(
                    cfg_clone.embedding_url,
                    cfg_clone.embedding_model,
                    cfg_clone.embedding_dimensions,
                    cfg_clone.embedding_batch_size,
                    // No API key. `EMBEDDING_URL` is guarded at boot with
                    // `has_credential: false` on the strength of this `None` —
                    // passing a key here means updating that row too.
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
                let graph = std::rc::Rc::new(graph);

                let summary: Option<Box<dyn alaya_backends::SummaryProvider>> =
                    summary.map(|s| Box::new(s) as Box<dyn alaya_backends::SummaryProvider>);

                let mut svc = MemoryService::new(
                    Box::new(qdrant),
                    Box::new(cached_embeddings),
                    Box::new(GraphRef(graph.clone())),
                    Box::new(HebbianRef(graph.clone())),
                    Box::new(ConsolidationRef(graph)),
                    summary,
                );

                if let Some(judge) = judge {
                    svc = svc.with_judge(Box::new(judge));
                }
                if let Some(rerank) = rerank {
                    svc = svc.with_reranker(Box::new(rerank));
                }

                let limits = WorkerLimits {
                    judge_daily_cap: cfg_clone.judge_daily_cap,
                    ..WorkerLimits::default()
                };
                service_worker(rx, svc, progress_for_worker, limits).await;
            });
        }));

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
        // Also refreshes the Qdrant verdict the bare probe serves, so the
        // unauthenticated route does no backend I/O of its own (#78).
        let pinger = {
            let handle = handle.clone();
            let checker = checker.clone();
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
                    checker.refresh_qdrant().await;
                }
            })
        };

        let protected = protected_router(handle, auth_state.clone());

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

/// The authenticated REST + MCP routes, composed.
///
/// Assembled here rather than inline in `main` for the same reason as
/// `health_routes`: tests drive the real layer stack, so dropping the auth
/// layer or reverting the request span fails a test instead of passing CI.
fn protected_router(handle: ServiceHandle, auth_state: AuthState) -> Router {
    const MAX_BODY: usize = 1_048_576; // 1 MB — covers the /mcp Bytes extractor

    // Handlers keep `ServiceHandle` state; `require_auth` is layered with its
    // own `AuthState` (axum allows differing types).
    Router::new()
        .route("/mcp", post(mcp::mcp_handler))
        .route("/store", post(store))
        .route("/search", post(search))
        .route("/delete", post(delete))
        .route("/relation", post(relation))
        .route("/supersede", post(supersede))
        .route("/contradictions", post(contradictions))
        .route("/contradictions/resolution", post(resolve_contradiction))
        .route("/duplicates/find", post(find_duplicates))
        .route("/duplicates/merge", post(merge_duplicates))
        .route(
            "/memories/{content_hash}",
            get(get_memory).patch(patch_memory),
        )
        .route("/backfill/summaries", post(backfill_summaries))
        .route("/backfill/contradictions", post(backfill_contradictions))
        .layer(middleware::from_fn_with_state(
            auth_state,
            auth::require_auth,
        ))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .layer(TraceLayer::new_for_http().make_span_with(request_span))
        .with_state(handle)
}

/// The request span for authenticated REST routes: `method` and `path`
/// only — a query string is never safe to log (mirrors ops-console).
/// `info_span!` targets this module (`alaya_server`), so the directive that
/// keeps it alive is `alaya_server=info` in the default filter
/// (`telemetry.rs`); `tower_http=info` only governs tower-http's own
/// request/response events. `DefaultMakeSpan` can set the span's level but
/// cannot drop its `uri` field, so a custom `MakeSpan` is required.
fn request_span(req: &Request) -> tracing::Span {
    tracing::info_span!("request", method = %req.method(), path = %req.uri().path())
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
    let v = checker.check_status();
    (health_code(&v), Json(v))
}

/// Authenticated operator view: the full health document, including build
/// identity (#70). Every intended consumer — radar, unified-memory, agents —
/// already holds `ALAYA_API_KEY`, so "read the running build with zero cluster
/// access" survives the move behind auth.
async fn health_detail(
    axum::extract::State(checker): axum::extract::State<HealthChecker>,
) -> (StatusCode, Json<Value>) {
    let v = checker.check_detail().await;
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
    let read_only = WritePolicy::read_only_for(principal);
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
    let read_only = WritePolicy::read_only_for(principal);
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
    /// Show pairs where an endpoint is already superseded (default hidden).
    #[serde(default)]
    include_resolved: bool,
    /// Verdict filter; omitted = `contradiction,supersession,unjudged`.
    #[serde(default)]
    verdicts: Option<Vec<String>>,
    /// Pairs to skip (page cursor: pass back the previous `next_offset`).
    #[serde(default)]
    offset: usize,
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
            offset: req.offset,
            include_resolved: req.include_resolved,
            verdicts: req.verdicts,
            reply: tx,
        },
        rx,
    )
    .await
}

/// `resolution` must be present: `"keep_both"` stamps, explicit `null`
/// clears. A missing key is a 4xx, never a silent clear.
#[derive(Deserialize)]
struct ResolveContradictionReq {
    memory_a_hash: String,
    memory_b_hash: String,
    #[serde(deserialize_with = "require_present")]
    resolution: Option<Resolution>,
    /// Who resolved, recorded verbatim (`operator:console`, `engine:<run-id>`).
    resolved_via: String,
}

/// `Option<T>` that rejects an absent key (serde's default reads absent as
/// `None`, which for a set-or-clear field turns a typo into a clear).
pub(crate) fn require_present<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d)
}

async fn resolve_contradiction(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(req): Json<ResolveContradictionReq>,
) -> (StatusCode, Json<Value>) {
    let (tx, rx) = oneshot::channel();
    h.call(
        CmdInner::ResolveContradiction {
            memory_a_hash: req.memory_a_hash,
            memory_b_hash: req.memory_b_hash,
            resolution: req.resolution,
            resolved_via: req.resolved_via,
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

#[derive(Deserialize)]
struct BackfillContradictionsParams {
    #[serde(default = "default_backfill_limit")]
    limit: usize,
    /// Also re-judge edges whose `verdict_model` differs from the configured
    /// judge model (recovery path for a model switch).
    #[serde(default)]
    rejudge: bool,
}

/// Operator-only (same auth class as `/backfill/summaries`, enforced in
/// `auth::rest_route_op`). Blocks until the pass completes and returns
/// `{queued, judged, persisted, marked, unjudged, input_tokens, output_tokens}`.
async fn backfill_contradictions(
    axum::extract::State(h): axum::extract::State<ServiceHandle>,
    Json(params): Json<BackfillContradictionsParams>,
) -> (StatusCode, Json<Value>) {
    let (tx, rx) = oneshot::channel();
    h.call(
        CmdInner::BackfillContradictions {
            limit: params.limit,
            rejudge: params.rejudge,
            reply: tx,
        },
        rx,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── resolve_contradiction wire shape (LAB-3885 AC-4) ─────────────────

    #[test]
    fn resolve_contradiction_req_requires_resolution_key_but_accepts_null() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let set: ResolveContradictionReq = serde_json::from_value(json!({
            "memory_a_hash": a, "memory_b_hash": b,
            "resolution": "keep_both", "resolved_via": "operator:console"
        }))
        .unwrap();
        assert_eq!(set.resolution, Some(Resolution::KeepBoth));

        let clear: ResolveContradictionReq = serde_json::from_value(json!({
            "memory_a_hash": a, "memory_b_hash": b,
            "resolution": null, "resolved_via": "operator:console"
        }))
        .unwrap();
        assert_eq!(clear.resolution, None);

        let missing = serde_json::from_value::<ResolveContradictionReq>(json!({
            "memory_a_hash": a, "memory_b_hash": b, "resolved_via": "operator:console"
        }));
        assert!(missing.is_err(), "an absent key must not read as a clear");
        let bogus = serde_json::from_value::<ResolveContradictionReq>(json!({
            "memory_a_hash": a, "memory_b_hash": b,
            "resolution": "keep-both", "resolved_via": "operator:console"
        }));
        assert!(bogus.is_err());
        let no_via = serde_json::from_value::<ResolveContradictionReq>(json!({
            "memory_a_hash": a, "memory_b_hash": b, "resolution": "keep_both"
        }));
        assert!(no_via.is_err(), "resolved_via is required on REST");
    }

    // ─── Contradiction judge plumbing (LAB-3283 AC-4, AC-5, AC-9) ─────────

    #[test]
    fn contradicted_hashes_dedups_signals_and_tolerates_missing_key() {
        let mut r = std::collections::HashMap::new();
        assert!(contradicted_hashes(&r).is_empty());
        r.insert(
            "interference".to_string(),
            json!({"contradictions": [
                {"existing_hash": "b", "signal_type": "Negation"},
                {"existing_hash": "a", "signal_type": "Temporal"},
                {"existing_hash": "b", "signal_type": "Antonym"},
                {"signal_type": "Bogus"}
            ]}),
        );
        assert_eq!(
            contradicted_hashes(&r),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    fn judged() -> JudgeOutcome {
        JudgeOutcome::Judged {
            persisted: true,
            judgement: alaya_backends::Judgement {
                verdict: alaya_types::graph::Verdict::Coexist,
                survivor: None,
                reason: String::new(),
                confidence: 0.5,
                model: "m".into(),
                input_tokens: 1,
                output_tokens: 1,
            },
        }
    }

    #[tokio::test(start_paused = true)]
    async fn with_backoff_honours_retry_after_then_succeeds() {
        let calls = std::cell::Cell::new(0u32);
        let start = tokio::time::Instant::now();
        let out = with_backoff(|| {
            let n = calls.get();
            calls.set(n + 1);
            async move {
                if n < 2 {
                    JudgeOutcome::RateLimited {
                        retry_after_secs: Some(3),
                    }
                } else {
                    judged()
                }
            }
        })
        .await;
        assert!(matches!(out, JudgeOutcome::Judged { .. }), "{out:?}");
        assert_eq!(calls.get(), 3);
        assert!(
            start.elapsed() >= std::time::Duration::from_secs(6),
            "two 3s waits"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn with_backoff_doubles_without_a_hint_and_gives_up() {
        let calls = std::cell::Cell::new(0u32);
        let start = tokio::time::Instant::now();
        let out = with_backoff(|| {
            calls.set(calls.get() + 1);
            async {
                JudgeOutcome::RateLimited {
                    retry_after_secs: None,
                }
            }
        })
        .await;
        assert!(
            matches!(
                out,
                JudgeOutcome::Unjudged {
                    marked: false,
                    spent: false
                }
            ),
            "{out:?}"
        );
        assert_eq!(calls.get(), JUDGE_MAX_RETRIES + 1);
        // 1+2+4+8+16 s of doubling before the final attempt gives up.
        assert!(start.elapsed() >= std::time::Duration::from_secs(31));
        assert!(
            start.elapsed() < std::time::Duration::from_secs(63),
            "no wait after the last attempt"
        );
    }

    #[tokio::test]
    async fn with_backoff_unjudged_is_final() {
        let calls = std::cell::Cell::new(0u32);
        let out = with_backoff(|| {
            calls.set(calls.get() + 1);
            async {
                JudgeOutcome::Unjudged {
                    marked: true,
                    spent: true,
                }
            }
        })
        .await;
        assert!(
            matches!(
                out,
                JudgeOutcome::Unjudged {
                    marked: true,
                    spent: true
                }
            ),
            "{out:?}"
        );
        assert_eq!(calls.get(), 1);
    }

    // ─── Optional config reads ─────────────────────────────────────────────

    /// `env_non_empty` replaced an untrimmed `env_opt` at every optional call
    /// site, so a whitespace-only value now reads as absent rather than set.
    /// Two live consequences, both silent before: `OIDC_ISSUER="   "` used to
    /// reach the fail-closed boot check as `Some`, satisfying "some auth is
    /// configured" with an issuer that resolves nothing; and a whitespace-only
    /// `JUDGE_URL` used to win its own `or_else` and suppress the `SUMMARY_URL`
    /// fallback, disabling the judge instead of falling back to it.
    ///
    /// Pins the helper, not the wiring: nothing here can catch a call site
    /// that stops using it. A test that rebuilt the `or_else` chain in its own
    /// body was cut for exactly that — it asserted `Option::or_else`.
    #[test]
    fn non_empty_trimmed_treats_blank_as_absent() {
        assert_eq!(non_empty_trimmed(None), None);
        assert_eq!(non_empty_trimmed(Some("".into())), None);
        assert_eq!(non_empty_trimmed(Some("   ".into())), None);
        assert_eq!(non_empty_trimmed(Some("\t\n ".into())), None);
        // Trimmed, not merely accepted — the value reaching a client is clean.
        assert_eq!(
            non_empty_trimmed(Some("  https://api.anthropic.com  ".into())),
            Some("https://api.anthropic.com".into())
        );
    }

    // ─── Daily judge spend cap (LAB-3895) ──────────────────────────────────

    #[test]
    fn parse_judge_daily_cap_defaults_and_validates() {
        assert_eq!(parse_judge_daily_cap(None).unwrap(), 1000);
        assert_eq!(parse_judge_daily_cap(Some("".into())).unwrap(), 1000);
        assert_eq!(parse_judge_daily_cap(Some("  ".into())).unwrap(), 1000);
        assert_eq!(parse_judge_daily_cap(Some("1000".into())).unwrap(), 1000);
        assert_eq!(parse_judge_daily_cap(Some("50".into())).unwrap(), 50);
        assert_eq!(parse_judge_daily_cap(Some("0".into())).unwrap(), 0);

        let err = parse_judge_daily_cap(Some("foo".into())).unwrap_err();
        assert!(err.contains("JUDGE_DAILY_CAP must be a non-negative integer"));
        assert!(err.contains("foo"));

        let err = parse_judge_daily_cap(Some("-10".into())).unwrap_err();
        assert!(err.contains("JUDGE_DAILY_CAP must be a non-negative integer"));

        let err = parse_judge_daily_cap(Some("12.5".into())).unwrap_err();
        assert!(err.contains("JUDGE_DAILY_CAP must be a non-negative integer"));
    }

    const DAY1: u64 = 1789733949; // 2026-09-18

    #[test]
    fn utc_date_str_computes_civil_calendar_correctly() {
        // Unix epoch start
        assert_eq!(utc_date_str(0), "1970-01-01");
        assert_eq!(utc_date_str(86399), "1970-01-01");
        assert_eq!(utc_date_str(86400), "1970-01-02");
        // Leap year 2024-02-29 (1709164800 is 2024-02-29 00:00:00 UTC)
        assert_eq!(utc_date_str(1709164800), "2024-02-29");
        assert_eq!(utc_date_str(1709251199), "2024-02-29");
        assert_eq!(utc_date_str(1709251200), "2024-03-01");
        // Known date: 2026-09-18
        assert_eq!(utc_date_str(DAY1), "2026-09-18");
    }

    #[test]
    fn judge_daily_cap_cap_reached_and_rollover() {
        let mut limiter = JudgeDailyCap::new(2);
        let day1 = DAY1;
        let day2 = day1 + 86400; // 2026-09-19

        // Day 1: admit 2 items, each billed to day 1.
        assert_eq!(limiter.try_admit_at(day1), Some(day1 / 86400));
        assert_eq!(limiter.count, 1);
        assert!(!limiter.warned_budget);

        assert_eq!(limiter.try_admit_at(day1), Some(day1 / 86400));
        assert_eq!(limiter.count, 2);
        assert!(!limiter.warned_budget);

        // Day 1: 3rd item hits cap, sets warned_budget = true
        assert_eq!(limiter.try_admit_at(day1), None);
        assert_eq!(limiter.count, 2);
        assert!(limiter.warned_budget);

        // Day 1: 4th item is silent, still refused
        assert_eq!(limiter.try_admit_at(day1), None);
        assert_eq!(limiter.count, 2);
        assert!(limiter.warned_budget);

        // Day 2 (rollover): counter and warning reset!
        assert_eq!(limiter.try_admit_at(day2), Some(day2 / 86400));
        assert_eq!(limiter.count, 1);
        assert!(!limiter.warned_budget);
        assert_eq!(limiter.current_day, day2 / 86400);

        assert_eq!(limiter.try_admit_at(day2), Some(day2 / 86400));
        assert_eq!(limiter.count, 2);
        assert!(!limiter.warned_budget);

        // Day 2: cap reached again, warned fires once for day 2
        assert_eq!(limiter.try_admit_at(day2), None);
        assert_eq!(limiter.count, 2);
        assert!(limiter.warned_budget);

        assert_eq!(limiter.try_admit_at(day2), None);
        assert_eq!(limiter.count, 2);
        assert!(limiter.warned_budget);
    }

    /// A unit is returned only to the day that was billed for it, and only for
    /// a call that spent nothing. After a UTC rollover the refund is dropped:
    /// crediting it would hand the new day free budget it never used.
    #[test]
    fn judge_daily_cap_refund_is_day_scoped() {
        let mut limiter = JudgeDailyCap::new(2);
        let day1 = DAY1 / 86400;

        let billed = limiter.try_admit_at(DAY1).expect("admitted");
        assert_eq!(limiter.count, 1);
        limiter.refund(billed);
        assert_eq!(limiter.count, 0, "the unit is back for reuse today");

        // A refund for a day that has rolled is ignored, and never underflows.
        assert_eq!(limiter.try_admit_at(DAY1), Some(day1));
        assert_eq!(limiter.try_admit_at(DAY1 + 86400), Some(day1 + 1));
        assert_eq!(limiter.count, 1, "rollover reset the count");
        limiter.refund(day1);
        assert_eq!(limiter.count, 1, "yesterday's refund does not credit today");
        limiter.refund(day1 + 1);
        limiter.refund(day1 + 1);
        assert_eq!(limiter.count, 0, "saturates at zero");
    }

    #[test]
    fn judge_daily_cap_zero_cap_refuses_immediately() {
        let mut limiter = JudgeDailyCap::new(0);
        let now = DAY1;
        assert_eq!(limiter.try_admit_at(now), None);
        assert!(limiter.warned_budget);
        assert_eq!(limiter.try_admit_at(now), None);
    }

    struct StubJudge;

    #[async_trait::async_trait(?Send)]
    impl alaya_backends::ContradictionJudge for StubJudge {
        /// Answers like a model that returned garbage: deterministic, and the
        /// tokens went out. Tests behind a hanging store never get this far.
        async fn judge(
            &self,
            _a: &alaya_types::memory::Memory,
            _b: &alaya_types::memory::Memory,
        ) -> alaya_types::Result<alaya_backends::Judgement> {
            Err(alaya_types::AlayaError::Judge("malformed verdict".into()))
        }
        fn model_name(&self) -> &str {
            "stub-judge"
        }
    }

    thread_local! {
        static FAKE_NOW: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    fn fake_now() -> u64 {
        FAKE_NOW.get()
    }

    /// A limiter on the test clock, shared the way `service_worker` shares it.
    fn capped(cap: usize) -> std::rc::Rc<std::cell::RefCell<JudgeDailyCap>> {
        std::rc::Rc::new(std::cell::RefCell::new(JudgeDailyCap {
            clock: fake_now,
            ..JudgeDailyCap::new(cap)
        }))
    }

    /// A judge-enabled service whose backend never answers: an admitted call
    /// parks holding its gate permit and slot, as against a stalled endpoint.
    fn judged_hanging_service() -> std::rc::Rc<MemoryService> {
        std::rc::Rc::new(wedge_tests::hanging_service().with_judge(Box::new(StubJudge)))
    }

    /// A judge-enabled service whose store answers `get_batch` with `batch`,
    /// so a spawned task runs to completion and the limiter can be read after.
    fn judged_service_with_batch(
        batch: Vec<alaya_types::memory::Memory>,
    ) -> std::rc::Rc<MemoryService> {
        std::rc::Rc::new(wedge_tests::stub_service(Some(batch)).with_judge(Box::new(StubJudge)))
    }

    fn mem(hash: &str) -> alaya_types::memory::Memory {
        serde_json::from_value(json!({
            "content": "c", "content_hash": hash, "tags": [], "memory_type": "note",
            "created_at": 0.0, "updated_at": 0.0,
        }))
        .expect("memory")
    }

    /// A `created` store result whose new memory contradicts `n` existing ones.
    fn store_result_contradicting(n: usize) -> std::collections::HashMap<String, Value> {
        let contradictions: Vec<Value> = (0..n)
            .map(|i| json!({"existing_hash": format!("{i:064x}"), "signal_type": "Negation"}))
            .collect();
        let mut r = std::collections::HashMap::new();
        r.insert("created".to_string(), json!(true));
        r.insert("content_hash".to_string(), json!("a".repeat(64)));
        r.insert(
            "interference".to_string(),
            json!({"contradictions": contradictions}),
        );
        r
    }

    /// Run the spawned local tasks: `run_until` polls every ready local task
    /// each time the inner future returns Pending, so one yield is one tick.
    async fn settle() {
        tokio::task::yield_now().await;
    }

    #[tokio::test]
    async fn store_contradiction_judge_admits_at_call_time_and_bounds_backlog() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let svc = judged_hanging_service();
                let gate = std::rc::Rc::new(tokio::sync::Semaphore::new(4));
                let limiter = capped(2);
                FAKE_NOW.set(DAY1);
                let r = store_result_contradicting(2);

                // 1. Two pairs: queued, then admitted once each holds a permit.
                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    2
                );
                settle().await;
                assert_eq!(limiter.borrow().count, 2);
                assert!(!limiter.borrow().warned_budget);

                // 2. The budget is spent but the backlog is not: the pairs are still
                //    queued, and refused inside the task rather than at the slot gate.
                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    2
                );
                settle().await;
                assert_eq!(limiter.borrow().count, 2, "budget holds at the cap");
                assert!(limiter.borrow().warned_budget, "and the operator is told");

                // 3. Under read_only: nothing is ever queued.
                let fresh = capped(10);
                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &fresh, &r, true),
                    0
                );
                assert_eq!(fresh.borrow().count, 0);
            })
            .await;
    }

    /// With the day's budget spent, a queued pair is refused when its call would
    /// start. The task exits without calling the judge, so its slot frees.
    #[tokio::test]
    async fn store_contradiction_judge_refused_at_call_time_stays_unjudged() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let svc = judged_hanging_service();
                let gate = std::rc::Rc::new(tokio::sync::Semaphore::new(4));
                let limiter = capped(1);
                FAKE_NOW.set(DAY1);
                assert!(
                    limiter.borrow_mut().try_admit().is_some(),
                    "spend today's budget"
                );
                let r = store_result_contradicting(1);

                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    1
                );
                settle().await;
                assert_eq!(limiter.borrow().count, 1);
                assert!(limiter.borrow().warned_budget, "refused at call time");
                // Its slot is free again: the refused task never reached the stalled call.
                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    1
                );
            })
            .await;
    }

    /// A pair queued on the judge gate before UTC midnight is billed to the
    /// day the call runs. Billing at spawn time let the new day's calls exceed
    /// the cap by the size of the overnight backlog.
    #[tokio::test]
    async fn store_contradiction_judge_bills_the_day_the_call_runs() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let svc = judged_hanging_service();
                // No permits: every spawned task queues on the gate.
                let gate = std::rc::Rc::new(tokio::sync::Semaphore::new(0));
                let limiter = capped(2);
                let r = store_result_contradicting(2);

                let midnight = (DAY1 / 86400 + 1) * 86400;
                FAKE_NOW.set(midnight - 1);
                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    2
                );
                settle().await;
                // Queued, not admitted: day 1 has been billed nothing.
                assert_eq!(limiter.borrow().count, 0);
                assert_eq!(limiter.borrow().current_day, u64::MAX);

                // Midnight passes while the pairs are still queued.
                FAKE_NOW.set(midnight);
                gate.add_permits(2);
                settle().await;
                let l = limiter.borrow();
                assert_eq!(l.current_day, midnight / 86400);
                assert_eq!(l.count, 2, "the queued calls draw on day 2's quota");
            })
            .await;
    }

    /// The refusal an operator actually meets first when the judge endpoint
    /// stalls: the backlog fills, so pairs are turned away at the slot gate
    /// before any task exists to consult the day's budget. That path must warn
    /// on its own — `debug!` is invisible under the `alaya_server=info` filter
    /// both the crate default and the compose file pin.
    #[tokio::test]
    async fn store_contradiction_judge_warns_when_the_backlog_refuses() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let svc = judged_hanging_service();
                let gate = std::rc::Rc::new(tokio::sync::Semaphore::new(4));
                // Budget far above the backlog: the backlog is what refuses.
                let limiter = capped(100_000);
                FAKE_NOW.set(DAY1);
                let r = store_result_contradicting(JUDGE_STORE_BACKLOG_MAX + 1);

                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    JUDGE_STORE_BACKLOG_MAX,
                    "one pair over the backlog is refused"
                );
                assert!(
                    limiter.borrow().warned_backlog,
                    "and the operator is told, at WARN"
                );
                assert!(
                    !limiter.borrow().warned_budget,
                    "the budget was never the reason"
                );

                // Subsequent refusals that day are silent, and the backlog stays shut.
                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    0
                );
            })
            .await;
    }

    /// `JUDGE_DAILY_CAP=0` disables store-path judging. It must say so: the
    /// whole point of the cap is that the guard trips loudly.
    #[tokio::test]
    async fn store_contradiction_judge_zero_cap_warns_through_the_spawn_path() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let svc = judged_hanging_service();
                let gate = std::rc::Rc::new(tokio::sync::Semaphore::new(4));
                let limiter = capped(0);
                FAKE_NOW.set(DAY1);
                let r = store_result_contradicting(1);

                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    1,
                    "a zero budget still reserves a slot, so the task can report it"
                );
                settle().await;
                assert_eq!(limiter.borrow().count, 0);
                assert!(limiter.borrow().warned_budget);
            })
            .await;
    }

    /// A call that never reached the judge costs nothing, so it must not cost
    /// budget either — otherwise a vector-store blip mid-import spends the
    /// day's ceiling on no-ops. An invalid pair is rejected inside
    /// `judge_contradiction` before any request goes out.
    #[tokio::test]
    async fn store_contradiction_judge_refunds_a_call_that_spent_nothing() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let svc = judged_hanging_service();
                let gate = std::rc::Rc::new(tokio::sync::Semaphore::new(4));
                let limiter = capped(1);
                FAKE_NOW.set(DAY1);

                let mut r = std::collections::HashMap::new();
                r.insert("created".to_string(), json!(true));
                r.insert("content_hash".to_string(), json!("a".repeat(64)));
                r.insert(
                    "interference".to_string(),
                    json!({"contradictions": [{"existing_hash": "not-a-hash"}]}),
                );

                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    1
                );
                settle().await;
                assert_eq!(
                    limiter.borrow().count,
                    0,
                    "the unit is back: nothing was sent to the judge"
                );
                assert!(!limiter.borrow().warned_budget);
            })
            .await;
    }

    /// The refund keys on `spent`, not `marked` (LAB-3901): a pair whose other
    /// endpoint is missing from the store is marked (so the backfill skips it)
    /// yet sent nothing, so its unit comes back; a malformed verdict is marked
    /// too, but the tokens went out, so it stays billed.
    #[tokio::test]
    async fn store_contradiction_judge_refunds_by_spend_not_by_marker() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let gate = std::rc::Rc::new(tokio::sync::Semaphore::new(4));
                let r = store_result_contradicting(1);
                let (src, dst) = ("a".repeat(64), "0".repeat(64));
                FAKE_NOW.set(DAY1);

                // Endpoint missing: only the new memory is in the store.
                let svc = judged_service_with_batch(vec![mem(&src)]);
                let limiter = capped(1);
                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    1
                );
                settle().await;
                assert_eq!(
                    limiter.borrow().count,
                    0,
                    "marked, but nothing was sent: refunded"
                );

                // Malformed verdict: both endpoints present, the stub judge errs.
                let svc = judged_service_with_batch(vec![mem(&src), mem(&dst)]);
                let limiter = capped(1);
                assert_eq!(
                    spawn_store_contradiction_judges(&svc, &gate, &limiter, &r, false),
                    1
                );
                settle().await;
                assert_eq!(
                    limiter.borrow().count,
                    1,
                    "marked and paid for: stays billed"
                );
            })
            .await;
    }

    /// AC-4: the operator backfill is not subject to the daily cap. The proof
    /// is structural — `backfill_judge` takes no limiter — so the test drives a
    /// pair through it with the budget exhausted and asserts the pair was
    /// actually processed. A vacuous `vec![]` would pass even if it were capped.
    #[tokio::test]
    async fn backfill_ignores_daily_cap() {
        let mut limiter = JudgeDailyCap::new(0);
        assert_eq!(limiter.try_admit(), None, "no budget left today");

        // Invalid hashes: judged without a request going out, so the pass
        // completes without a live judge backend.
        let pair = alaya_types::graph::Contradiction {
            memory_a_hash: "not-a-hash".into(),
            memory_b_hash: "also-not-a-hash".into(),
            confidence: None,
            created_at: None,
            verdict: None,
            resolution: None,
            resolved_at: None,
            resolved_via: None,
        };
        let gate = std::rc::Rc::new(tokio::sync::Semaphore::new(JUDGE_CONCURRENCY));
        let svc = wedge_tests::hanging_service().with_judge(Box::new(StubJudge));
        let totals = backfill_judge(&svc, &gate, vec![pair]).await;
        assert_eq!(
            totals,
            BackfillTotals {
                unjudged: 1,
                ..Default::default()
            },
            "the backfill processed the pair despite the exhausted cap"
        );
    }

    #[test]
    fn cluster_local_accepts_service_dns_and_private_hosts_only() {
        let parse = |u: &str| reqwest::Url::parse(u).unwrap();
        for ok in [
            "http://anthropic-lb:8082",
            "http://alaya-bridge:3000",
            "http://alaya-server.mcp.svc:3001",
            "http://localhost:8082",
            "http://10.43.144.201:8082",
            "http://[::1]:8082",
            // Userinfo bound for a cluster-local proxy is that proxy's business.
            with_userinfo("http", "anthropic-lb:8082").as_str(),
            // Non-special schemes preserve host case; lowercasing fixes this.
            with_userinfo("redis", "redis.mcp.SVC:6379").as_str(),
            with_userinfo("rediss", "Redis-Svc:6379").as_str(),
        ] {
            assert!(is_cluster_local(&parse(ok)), "{ok}");
        }
        for no in [
            "http://api.anthropic.com",
            "http://proxy.example.net:8082",
            "http://1.2.3.4",
            // The host is what reqwest connects to, not what precedes the `@`.
            with_userinfo("http", "api.anthropic.com").as_str(),
            // Spoof probe: the userinfo IS the payload — never rewrite with
            // the builder.
            "http://anthropic-lb:8082@api.anthropic.com",
            // An IPv6 literal has no dots but is not a service name.
            "http://[2606:4700::1111]",
            "http://[::ffff:1.2.3.4]",
        ] {
            assert!(!is_cluster_local(&parse(no)), "{no}");
        }
    }

    /// Userinfo placeholders. This guard exists to refuse credential-shaped
    /// URLs, so the tests must build them — and at fourteen call sites one
    /// builder beats fourteen literals. Only the PRESENCE of userinfo is ever
    /// asserted on, never its value.
    const FAKE_USER: &str = "redacted-user";
    const FAKE_SECRET: &str = "redacted-secret";

    /// `scheme://` + placeholder userinfo + `rest` (authority, and whatever
    /// path/query/fragment the caller is exercising).
    fn with_userinfo(scheme: &str, rest: &str) -> String {
        format!("{scheme}://{FAKE_USER}:{FAKE_SECRET}@{rest}")
    }

    #[test]
    fn credential_transport_fails_closed_off_cluster() {
        // Refused with a key: plain http to a non-cluster host whatever the
        // scheme's case (the transport parses it case-insensitively), a
        // scheme that is not https at all, or a value that does not parse.
        for bad in [
            "http://api.anthropic.com",
            "http://proxy.example.net:8082",
            "HTTP://api.anthropic.com",
            "Http://API.Anthropic.com:80",
            // A mapped PUBLIC v4, and a global v6 — the latter parses as an
            // IP, so it never reaches the single-label fallback.
            "http://[::ffff:93.184.216.34]:8082",
            "http://[2606:4700::1111]:8082",
            // End-anchored: a public domain wearing an `svc` label is not
            // cluster-local, however much it looks like one.
            "http://evil.svc.attacker.com",
            "http://alaya-bridge.mcp.svc.cluster.local.evil.com",
            "htps://api.anthropic.com",
            "api.anthropic.com:443",
            "not a url",
        ] {
            assert!(
                check_credential_transport("JUDGE_URL", bad, true, Transport::Http).is_err(),
                "{bad}"
            );
        }
        // Allowed: https anywhere, cluster-local plaintext, or nothing to
        // protect (no key and no userinfo).
        for ok in [
            "https://api.anthropic.com",
            "HTTPS://api.anthropic.com",
            "http://anthropic-lb:8082",
            "HTTP://Anthropic-LB:8082",
            "http://alaya-bridge.mcp.svc:3000",
            "http://alaya-bridge.mcp.svc.cluster.local:3000",
            "http://localhost:8082",
            // IPv6 literals: loopback, ULA and IPv4-mapped private addresses
            // are as cluster-local as their v4 spellings.
            "http://[::1]:8082",
            "http://[fd00::1]:8082",
            "http://[::ffff:10.0.0.5]:8082",
        ] {
            assert!(
                check_credential_transport("SUMMARY_URL", ok, true, Transport::Http).is_ok(),
                "{ok}"
            );
        }
        for keyless in [
            "http://api.anthropic.com".to_string(),
            "not a url".to_string(),
            with_userinfo("http", "anthropic-lb:8082"),
        ] {
            assert!(
                check_credential_transport("SUMMARY_URL", keyless.as_str(), false, Transport::Http)
                    .is_ok(),
                "{keyless}"
            );
        }
        // Both the keyless warning and the refusal render `url::ParseError`,
        // which is only safe while it keeps the input out of its `Display` —
        // pin it, so a url-crate bump cannot quietly turn either into the leak
        // this guard exists to stop.
        let malformed = with_userinfo("http", "");
        let parse_err = reqwest::Url::parse(&malformed).unwrap_err().to_string();
        assert!(
            !parse_err.contains(FAKE_SECRET) && !parse_err.contains(FAKE_USER),
            "url::ParseError now echoes its input: {parse_err}"
        );

        // Userinfo is a credential too: reqwest sends it as Basic auth on
        // every request, so it faces the same policy with or without a key.
        for (bad, has_key) in [
            (with_userinfo("http", "api.anthropic.com"), true),
            (with_userinfo("http", "api.anthropic.com"), false),
            (format!("http://{FAKE_USER}@api.anthropic.com"), false),
            (format!("http://:{FAKE_SECRET}@api.anthropic.com"), false),
        ] {
            // The refusal goes to pod logs: name the host, never echo a
            // value that carries userinfo.
            let err =
                check_credential_transport("JUDGE_URL", bad.as_str(), has_key, Transport::Http)
                    .unwrap_err();
            assert!(
                err.contains("api.anthropic.com") && !err.contains(FAKE_SECRET),
                "{bad} {has_key}: {err}"
            );
        }
    }

    #[test]
    fn credential_transport_covers_qdrant_graph_and_redis() {
        // Annotated, not inferred: an unannotated closure binds one concrete
        // lifetime from its first call site and rejects every built-at-runtime URL.
        let http =
            |var: &str, url: &str, key| check_credential_transport(var, url, key, Transport::Http);
        let redis =
            |var: &str, url: &str, key| check_credential_transport(var, url, key, Transport::Redis);

        // QDRANT_URL with API key: cluster-local ok, off-cluster refused.
        assert!(http("QDRANT_URL", "http://qdrant:6333", true).is_ok());
        assert!(http("QDRANT_URL", "http://qdrant.cloud.io", true).is_err());

        // GRAPH_URL with API key: same policy.
        assert!(http("GRAPH_URL", "http://alaya-bridge:3000", true).is_ok());
        assert!(http("GRAPH_URL", "http://graph.cloud.io", true).is_err());

        // Cluster-local does not redeem a scheme the client cannot speak, and
        // the refusal must name which fault it is.
        for wrong_scheme in [
            with_userinfo("redis", "qdrant:6333"),
            with_userinfo("rediss", "qdrant:6333"),
            "redis://qdrant:6333".to_string(),
        ] {
            let err = http("QDRANT_URL", wrong_scheme.as_str(), true).unwrap_err();
            assert!(
                err.contains("unusable scheme") && !err.contains("in the clear"),
                "{wrong_scheme}: {err}"
            );
        }

        // redis:// — fred has no TLS, so rediss:// off-cluster is refused too.
        for scheme in ["redis", "rediss"] {
            let off = with_userinfo(scheme, "redis.cloud.io");
            let local = with_userinfo(scheme, "redis-svc:6379");
            assert!(
                redis("REDIS_CACHE_URL", off.as_str(), false).is_err(),
                "{off}"
            );
            // Cluster-local: both redis and rediss are fine.
            assert!(
                redis("REDIS_CACHE_URL", local.as_str(), false).is_ok(),
                "{local}"
            );
        }

        // Non-redis scheme with Redis transport: fred opens plain TCP
        // regardless, so `https` buys no encryption. The refusal must name
        // the scheme as the fault, not the host.
        for wrong_scheme in [
            with_userinfo("https", "redis.cloud.io"),
            with_userinfo("https", "redis-svc:6379"),
        ] {
            let err = redis("REDIS_CACHE_URL", wrong_scheme.as_str(), false).unwrap_err();
            assert!(
                err.contains("unusable scheme") && !err.contains("in the clear"),
                "{wrong_scheme}: {err}"
            );
        }

        // Still a credential guard, not a URL validator: with nothing to keep
        // off the wire a mismatched scheme is left to the client, exactly as an
        // unparseable keyless URL already is.
        assert!(http("QDRANT_URL", "redis://qdrant:6333", false).is_ok());
        assert!(redis("REDIS_CACHE_URL", "https://redis-svc:6379", false).is_ok());
    }

    #[test]
    fn host_of_strips_port_and_unwraps_ipv6_brackets() {
        assert_eq!(host_of("https://id.27b.io"), Some("id.27b.io".into()));
        assert_eq!(host_of("https://id.27b.io:8443"), Some("id.27b.io".into()));
        assert_eq!(host_of("http://[::1]:3001/foo"), Some("::1".into()));
        assert_eq!(host_of("http://localhost:8080"), Some("localhost".into()));
        assert_eq!(host_of("not-a-url"), None);
        assert_eq!(
            host_of("http://127.0.0.1:@evil.com"),
            Some("evil.com".into())
        );
        assert_eq!(
            host_of("http://localhost:@evil.com"),
            Some("evil.com".into())
        );
        assert_eq!(
            host_of(format!("https://{FAKE_USER}@host").as_str()),
            Some("host".into())
        );
    }

    #[test]
    fn log_safe_origin_drops_userinfo_path_and_query() {
        assert_eq!(
            log_safe_origin(
                with_userinfo("https", "tei.mcp.svc:8443/v1/rerank?api_key=k3y#f").as_str()
            ),
            "https://tei.mcp.svc:8443"
        );
        assert_eq!(
            log_safe_origin("http://localhost:8080/v1"),
            "http://localhost:8080"
        );
        assert_eq!(
            log_safe_origin("https://api.openai.com/v1"),
            "https://api.openai.com"
        );
        assert_eq!(log_safe_origin("tei.mcp.svc:8080"), "<no host>");
        let err = log_safe_origin("not-a-url");
        assert!(err.starts_with("<unparseable: "), "{err}");
        assert!(!err.contains("not-a-url"), "{err}");
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
        // Widened with `host_is_private`: this gate decides whether the
        // dev-only open mode may run, so the v6 spellings of a private
        // address need their own fence, not inherited coverage from the
        // credential-transport test.
        assert!(is_private_host("http://[fd00::1]:3001"));
        assert!(is_private_host("http://[::ffff:10.0.0.1]:3001"));
        assert!(!is_private_host("http://[::ffff:93.184.216.34]:3001"));
        assert!(!is_private_host("http://[2606:4700::1111]:3001"));

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
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

    /// The log-prefix helper sees caller-supplied hashes before validation
    /// (get_memory, delete, relation), so it must never byte-slice into a
    /// multibyte character.
    #[test]
    fn truncate_hash_is_char_boundary_safe() {
        assert_eq!(truncate_hash("abcdefgh0123"), "abcdefgh");
        assert_eq!(truncate_hash("abc"), "abc");
        // 'é' spans bytes 7-8: the cut must fall back to the boundary before it.
        assert_eq!(
            truncate_hash(&format!("abcdefgé{}", "a".repeat(55))),
            "abcdefg"
        );
    }

    /// VectorStorage whose `delete` and `get_batch` blackhole — models a
    /// backend whose pod IP vanished without an RST. `get_batch` can instead
    /// answer a fixed batch, so a judge task can run to completion. Every
    /// other method panics: no test exercises them.
    struct HangVectors {
        batch: Option<Vec<Memory>>,
    }

    #[async_trait(?Send)]
    impl VectorStorage for HangVectors {
        async fn store(&self, _memory: &Memory) -> Result<(bool, String)> {
            unimplemented!()
        }
        async fn get_by_hash(&self, _content_hash: &str) -> Result<Option<Memory>> {
            unimplemented!()
        }
        async fn exists(&self, _content_hash: &str) -> Result<bool> {
            unimplemented!()
        }
        async fn set_generated_summary(
            &self,
            _content_hash: &str,
            _summary: &str,
            _summary_embedding: Option<Vec<f32>>,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn get_batch(&self, _hashes: &[&str]) -> Result<Vec<Memory>> {
            match &self.batch {
                Some(b) => Ok(b.clone()),
                None => std::future::pending().await,
            }
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
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "healthy".into(),
                backend: "stub".into(),
                details: None,
            })
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
        async fn get_all_contradictions(
            &self,
            _query: &alaya_types::graph::ContradictionQuery,
        ) -> Result<Vec<Contradiction>> {
            unimplemented!()
        }
        async fn set_contradiction_verdict(
            &self,
            _src: &str,
            _dst: &str,
            _verdict: &alaya_types::graph::EdgeVerdict,
        ) -> Result<bool> {
            // The marker write lands, so a judge task can finish `marked: true`.
            Ok(true)
        }
        async fn set_contradiction_resolution(
            &self,
            _src: &str,
            _dst: &str,
            _resolution: Option<Resolution>,
            _via: &str,
            _at: f64,
        ) -> Result<bool> {
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

    pub(super) fn hanging_service() -> MemoryService {
        stub_service(None)
    }

    /// `hanging_service`, with `get_batch` answering `batch` when given.
    pub(super) fn stub_service(batch: Option<Vec<Memory>>) -> MemoryService {
        MemoryService::new(
            Box::new(HangVectors { batch }),
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
                    ..WorkerLimits::default()
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
                // command, on the same monotonic base the health checker
                // reads. The range carries both halves: 0 is the "starting"
                // sentinel and must be gone, and a stamp the reader cannot
                // outrun (an epoch one, say) saturates every age to 0 —
                // silently disabling the #63 watchdog with every other test
                // still green.
                let stamp = progress.load(std::sync::atomic::Ordering::Relaxed);
                assert!(
                    (1..=monotonic_secs()).contains(&stamp),
                    "worker stamp {stamp} is not on the reader's monotonic base"
                );
            })
            .await;
    }

    /// Stall tests read this fixed monotonic "now", so a stamp of
    /// `TEST_NOW - n` is exactly `n` seconds old however long the test
    /// process has been up — a real `monotonic_secs()` reading is only ever
    /// a few seconds past its origin. Production reads `monotonic_secs`.
    const TEST_NOW: u64 = 1_000_000;

    fn test_checker(progress_s: u64) -> HealthChecker {
        HealthChecker {
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_millis(200))
                .timeout(Duration::from_millis(500))
                .build()
                .unwrap(),
            // Port 1 refuses immediately — models an unreachable backend.
            qdrant_url: "http://127.0.0.1:1".into(),
            qdrant_api_key: None,
            collection: "test".into(),
            embedding_url: "http://127.0.0.1:1".into(),
            graph_url: "http://127.0.0.1:1".into(),
            graph_api_key: String::new(),
            worker_progress: Arc::new(AtomicU64::new(progress_s)),
            stall_threshold: WORKER_STALL_THRESHOLD,
            clock: || TEST_NOW,
            qdrant_ok: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Health semantics (#63): a stalled worker is `unhealthy` (503 → k8s
    /// restarts the pod); dead backends alone stay `degraded` (200 — a
    /// restart would not fix Qdrant being down).
    #[tokio::test]
    async fn health_distinguishes_worker_stall_from_backend_outage() {
        // Fresh worker progress + unreachable backends → degraded, not unhealthy.
        let v = test_checker(TEST_NOW).check().await;
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["worker"]["stalled"], false);

        // Stale worker progress → unhealthy, regardless of backend state.
        let v = test_checker(TEST_NOW - 3600).check().await;
        assert_eq!(v["status"], "unhealthy");
        assert_eq!(v["worker"]["stalled"], true);
        assert_eq!(v["worker"]["last_progress_age_s"], 3600);

        // 0 sentinel = worker still bootstrapping backends → "starting",
        // never a stall: a slow cluster cold start must not restart-loop
        // the pod before the command loop has even begun.
        let v = test_checker(0).check().await;
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["worker"]["state"], "starting");
        assert_eq!(v["worker"]["stalled"], false);
    }

    /// check_status() preserves the #63 tri-state contract from two atomic
    /// reads and carries no field beyond `status` (#78).
    #[test]
    fn check_status_preserves_tri_state_and_carries_only_status() {
        // Fresh worker, Qdrant verdict unpublished → degraded (200).
        let fresh = test_checker(TEST_NOW);
        let v = fresh.check_status();
        assert_eq!(v["status"], "degraded");
        assert_eq!(v.as_object().unwrap().len(), 1, "bare probe leaked fields");

        // Published verdict → healthy.
        fresh.qdrant_ok.store(true, Ordering::Relaxed);
        let v = fresh.check_status();
        assert_eq!(v["status"], "healthy");
        assert_eq!(v.as_object().unwrap().len(), 1);

        // Stale worker → unhealthy (503), whatever Qdrant said.
        let stalled = test_checker(TEST_NOW - 3600);
        stalled.qdrant_ok.store(true, Ordering::Relaxed);
        let v = stalled.check_status();
        assert_eq!(v["status"], "unhealthy");
        assert_eq!(v.as_object().unwrap().len(), 1);

        // Bootstrap sentinel → degraded, not a stall.
        let v = test_checker(0).check_status();
        assert_eq!(v["status"], "degraded");
        assert_eq!(v.as_object().unwrap().len(), 1);
    }

    /// The anonymous probe never reaches Qdrant; only the pinger's refresh
    /// does, and its verdict is what the probe then serves (#78). A counting
    /// loopback Qdrant is the witness — reintroducing any await on the
    /// request path shows up here as hits > 0.
    #[tokio::test]
    async fn bare_probe_does_no_qdrant_io_and_serves_pinger_verdict() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let qdrant_url = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicU64::new(0));
        let counter = hits.clone();
        let app = Router::new().route(
            "/collections/test",
            get(move || {
                counter.fetch_add(1, Ordering::Relaxed);
                std::future::ready(Json(json!({
                    "result": { "status": "green", "points_count": 0 }
                })))
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let checker = HealthChecker {
            qdrant_url,
            ..test_checker(TEST_NOW)
        };
        let routes = health_routes(checker.clone(), test_auth_state());

        for _ in 0..50 {
            let (code, body) = probe(&routes, "/health", None).await;
            assert_eq!(code, StatusCode::OK);
            assert_eq!(body["status"], "degraded");
        }
        assert_eq!(hits.load(Ordering::Relaxed), 0, "bare probe reached Qdrant");

        checker.refresh_qdrant().await;
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        let (_, body) = probe(&routes, "/health", None).await;
        assert_eq!(body["status"], "healthy");

        // Qdrant gone (port 1 refuses) → the next refresh withdraws it.
        let down = HealthChecker {
            qdrant_url: "http://127.0.0.1:1".into(),
            ..checker
        };
        down.refresh_qdrant().await;
        let (_, body) = probe(&routes, "/health", None).await;
        assert_eq!(body["status"], "degraded");
    }

    /// #97: `process::exit` cannot be observed in-process, so the test re-runs
    /// itself as a child. The child exits 0 if the supervisor merely returns,
    /// so a broken supervisor cannot masquerade as libtest's own 101.
    #[test]
    fn supervisor_exits_process_when_worker_panics() {
        const CHILD: &str = "ALAYA_SUPERVISOR_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let worker = std::thread::spawn(|| panic!("bootstrap failed before the command loop"));
            let _ = supervise(worker).join();
            std::process::exit(0);
        }

        // A worker that drains normally must NOT take the process down.
        supervise(std::thread::spawn(|| ())).join().unwrap();

        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("wedge_tests::supervisor_exits_process_when_worker_panics")
            .arg("--exact")
            .env(CHILD, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert_eq!(
            status.code(),
            Some(101),
            "supervisor must exit 101 on worker panic"
        );
    }

    // ─── /health split (#77) ────────────────────────────────────────────────

    const TEST_KEY: &str = "test-api-key";

    fn test_auth_state() -> AuthState {
        AuthState {
            api_key: Some(TEST_KEY.into()),
            readonly_api_key: None,
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
        let app = health_routes(test_checker(TEST_NOW), test_auth_state());

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
        let app = health_routes(test_checker(TEST_NOW), test_auth_state());

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
        // LAB-4025: the embedding probe rides the same authenticated surface.
        assert_eq!(body["embedding_health"]["status"], "unhealthy");
        assert!(body["embedding_health"]["error"].is_string());
        // Build identity (#70) rides the authenticated surface now.
        assert!(body.get("version").is_some());
        assert!(body.get("git_sha").is_some());
        assert!(body.get("built_at").is_some());
    }

    /// LAB-4025: Qdrant and the bridge up, embedding endpoint down. The
    /// operator view degrades and names the probe; the bare probe's verdict —
    /// the k8s contract — is untouched and carries no embedding verdict (#78:
    /// no new anonymous fan-out; `check()` is not wired to `check_embedding`).
    /// A stalled worker still wins: `unhealthy` is the 503 signal and TEI is
    /// not fixed by a restart.
    #[tokio::test]
    async fn embedding_outage_degrades_detail_but_not_bare_probe() {
        // Loopback stand-in for Qdrant *and* the bridge; TEI stays at port 1.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend_url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route(
                "/collections/test",
                get(|| async {
                    Json(json!({ "result": { "status": "green", "points_count": 7 } }))
                }),
            )
            .route(
                "/collections/test/points/count",
                post(|| async { Json(json!({ "result": { "count": 7 } })) }),
            )
            .route(
                "/health",
                get(|| async { Json(json!({ "status": "healthy" })) }),
            );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let checker = HealthChecker {
            qdrant_url: backend_url.clone(),
            graph_url: backend_url,
            ..test_checker(epoch_secs())
        };

        let bare = checker.check().await;
        assert_eq!(bare["status"], "healthy");
        assert!(bare.get("embedding_health").is_none());

        let detail = checker.check_detail().await;
        assert_eq!(detail["status"], "degraded");
        assert_eq!(detail["embedding_health"]["status"], "unhealthy");
        assert!(detail["embedding_health"]["error"].is_string());
        assert_eq!(detail["vector_health"]["status"], "green");
        assert_eq!(detail["total_memories"], 7);

        let stalled = HealthChecker {
            worker_progress: Arc::new(AtomicU64::new(epoch_secs() - 3600)),
            ..checker
        };
        assert_eq!(stalled.check_detail().await["status"], "unhealthy");
    }

    /// The #63 contract is the HTTP code, not the body: a wedged worker must
    /// still 503 the *unauthenticated* probe, or k8s stops restarting stalled
    /// pods. The failure path must not widen the body either.
    #[tokio::test]
    async fn stalled_worker_still_503s_the_bare_probe() {
        let app = health_routes(test_checker(TEST_NOW - 3600), test_auth_state());

        let (code, body) = probe(&app, "/health", None).await;

        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "unhealthy");
        assert_eq!(body.as_object().expect("object body").len(), 1);
    }

    /// Mirrors `request_span_omits_query_string` in `ops-console/src/main.rs`
    /// (LAB-4506), driven through `protected_router` — the router `main`
    /// serves — so reverting production's `make_span_with(request_span)`
    /// fails this test, not just editing `request_span` itself.
    ///
    /// Thread-scoped (`set_default`, not `set_global_default`): this binary
    /// also has `telemetry::tests::installs_without_a_tokio_runtime`, which
    /// installs a real global default — the two would race for the single
    /// process-wide slot. Scoped is normally flaky (tracing-core caches
    /// callsite interest per registering thread), but only for a callsite
    /// another concurrent test hits first. The sole callsite asserted on is
    /// `request_span`'s own close event (`FmtSpan::CLOSE`), and nothing else
    /// in this binary builds `protected_router`. Never assert on tower-http's
    /// `on_request`/`on_response` events: those are shared, and would bring
    /// the flake back.
    #[tokio::test]
    async fn request_span_omits_query_string() {
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;
        use tracing_subscriber::fmt::format::FmtSpan;

        #[derive(Clone, Default)]
        struct LogBuffer(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for LogBuffer {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let sink = LogBuffer::default();
        let writer = sink.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_env_filter("alaya_server=info,tower_http=info")
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        // Anonymous, so `require_auth` answers 401 inside the span and no
        // handler ever reaches the (unserviced) command channel.
        let (tx, _rx) = mpsc::channel(1);
        let app = protected_router(ServiceHandle { tx }, test_auth_state());

        let resp = app
            .oneshot(
                axum::http::Request::post("/store?token=QUERY-SENTINEL")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        // `TraceLayer`'s response body owns the span; it closes on drop.
        drop(resp);

        let log = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        assert!(
            log.contains("path=/store"),
            "positive control: the request span must close with its path:\n{log}"
        );
        assert!(
            !log.contains("QUERY-SENTINEL"),
            "query string reached the log:\n{log}"
        );
    }

    /// The other half of #63's contract: a worker that IS draining must never
    /// 503 because the wall clock moved (LAB-3968). NTP correcting a drifted
    /// node or a VM resume steps `SystemTime` forward; `Instant` does not
    /// follow, so a stamp written seconds ago stays seconds old.
    ///
    /// Simulated at the worst step there is, and with no fake anywhere: the
    /// stamp is a real `monotonic_secs()` — what the production heartbeat
    /// writes — read by the real production clock, and the wall clock sits
    /// ~1.8e9 seconds ahead of that monotonic origin. Age it off `epoch_secs`
    /// and this worker reads ~55 years stale, 503ing a healthy pod on the
    /// unauthenticated route; age it off `monotonic_secs` and it reads ~0.
    #[tokio::test]
    async fn fresh_heartbeat_survives_a_forward_wall_clock_step() {
        // The step is implicit in the two clocks: `epoch_secs()` is ~1.79e9
        // on any host with a post-1970 clock, `monotonic_secs()` is single
        // digits in a test binary.
        let checker = HealthChecker {
            worker_progress: Arc::new(AtomicU64::new(monotonic_secs())),
            clock: monotonic_secs,
            ..test_checker(TEST_NOW)
        };

        let (code, body) = probe(
            &health_routes(checker.clone(), test_auth_state()),
            "/health",
            None,
        )
        .await;
        assert_eq!(code, StatusCode::OK, "healthy worker 503d on a clock step");
        assert_eq!(body["status"], "degraded"); // backends down, worker fine

        // /health/detail agrees, and reports a plausible age rather than an
        // epoch-sized one.
        let v = checker.check().await;
        assert_eq!(v["worker"]["state"], "ok");
        assert_eq!(v["worker"]["stalled"], false);
        // Bounded absolutely, not against the threshold: `stalled == false`
        // already implies the latter, so it would catch nothing on its own.
        assert!(v["worker"]["last_progress_age_s"].as_u64().unwrap() <= 5);
    }
}
