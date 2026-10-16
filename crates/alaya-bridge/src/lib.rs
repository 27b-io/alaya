//! Library interface for alaya-bridge.
//!
//! Exposes internal modules so that integration tests can exercise the
//! Cypher → Redis → RESP pipeline directly without HTTP overhead.

pub mod cypher;
pub mod handlers;
pub mod resp;
pub mod routes;

mod auth;
pub mod queue;

use std::time::Duration;

use redis::aio::{ConnectionManager, ConnectionManagerConfig};

pub use auth::BridgeAuth;
pub use handlers::exec_query;
pub use handlers::value_to_cypher_literal;
pub use resp::FalkorResult;

/// Shared application state injected via axum's `State` extractor.
pub struct AppState {
    pub redis: ConnectionManager,
    pub graph_name: String,
    /// Resolved once at startup; see `auth::BridgeAuth`.
    pub auth: BridgeAuth,
}

// ─── FalkorDB connections (LAB-9869) ─────────────────────────────────────────
//
// A FalkorDB that accepts TCP and never replies used to hang the bridge with
// nothing logged: startup awaited the connection forever, so the liveness probe
// killed it silently, and a wedge mid-flight hung every graph call. Every
// connection attempt is now bounded, and so is every reply.

/// Bounds one connection attempt: TCP connect plus redis-rs's setup commands,
/// which is where a mute peer hangs.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Bounds one query reply. Just under alaya-server's 30 s bridge deadline:
/// FalkorDB bounds reads but not writes, so a slow but legitimate write still
/// lands as before, and a wedge still reaches the server as the bridge's 502,
/// logged here, rather than as the server's own timeout.
pub const RESPONSE_TIMEOUT: Duration = Duration::from_secs(25);

/// Bounds one reply on the Hebbian consumer's connection. Its BRPOP blocks
/// server-side for `queue::BRPOP_TIMEOUT_SECS`, so this sits well above that:
/// the pop is never cut off, but a mute peer cannot stall the consumer silently.
/// A reply that arrives after the deadline is dropped, losing that one item;
/// Hebbian strengthening tolerates loss.
pub const QUEUE_RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);
const _: () = assert!(queue::BRPOP_TIMEOUT_SECS + 2.0 <= QUEUE_RESPONSE_TIMEOUT.as_secs_f64());

/// Retries after a failed attempt, at startup and on reconnect. redis-rs
/// 0.27's default (6, with a backoff that reaches 60 s per step) keeps a mute
/// peer from failing for minutes; this caps startup at
/// 3 × `CONNECT_TIMEOUT` plus at most 4 s of jittered backoff.
const CONNECT_RETRIES: usize = 2;
const MAX_RETRY_DELAY_MS: u64 = 1_000;

/// Config for the multiplexed connection the query handlers share.
pub fn query_config() -> ConnectionManagerConfig {
    connect_config().set_response_timeout(RESPONSE_TIMEOUT)
}

/// Config for the Hebbian consumer's dedicated connection.
pub fn queue_config() -> ConnectionManagerConfig {
    connect_config().set_response_timeout(QUEUE_RESPONSE_TIMEOUT)
}

fn connect_config() -> ConnectionManagerConfig {
    ConnectionManagerConfig::new()
        .set_connection_timeout(CONNECT_TIMEOUT)
        .set_number_of_retries(CONNECT_RETRIES)
        .set_max_delay(MAX_RETRY_DELAY_MS)
}

/// Open a `ConnectionManager`, logging the target before the first attempt and
/// the error if every attempt fails. The log names `host:port` only, never the
/// URL, which may carry a password.
pub async fn connect(
    client: &redis::Client,
    config: ConnectionManagerConfig,
    role: &str,
) -> redis::RedisResult<ConnectionManager> {
    let addr = &client.get_connection_info().addr;
    tracing::info!(%addr, role, "connecting to FalkorDB");
    ConnectionManager::new_with_config(client.clone(), config)
        .await
        .inspect_err(|e| {
            tracing::error!(%addr, role, error = %e, "could not connect to FalkorDB");
        })
}
