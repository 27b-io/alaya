//! Shared test context for alaya-bridge integration tests.
//!
//! Each integration test calls `TestContext::new()`.  If `REDIS_URL` is not
//! set the function returns `None` and the test prints a skip message and
//! returns `Ok(())` immediately.
//!
//! Each context uses a graph name derived from the process-id plus a
//! per-process counter, so parallel tests (threads in one test binary) and
//! separate test binaries never share a graph.  Call `ctx.cleanup()` at the
//! end of every test to delete the ephemeral graph.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::Value;

static NEXT_GRAPH_ID: AtomicU32 = AtomicU32::new(0);

pub struct TestContext {
    pub conn: redis::aio::ConnectionManager,
    pub graph_name: String,
}

impl TestContext {
    /// Build a context connected to the Redis/FalkorDB instance at `REDIS_URL`.
    ///
    /// Returns `None` (and prints a skip notice) when the env var is absent.
    pub async fn new() -> Option<Self> {
        let redis_url = match std::env::var("REDIS_URL") {
            Ok(url) => url,
            Err(_) => {
                eprintln!("REDIS_URL not set — skipping integration test");
                return None;
            }
        };

        // Tests in one binary share the pid and run on parallel threads; a
        // counter (not clock entropy) guarantees each gets its own graph.
        let suffix = NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed);
        let graph_name = format!("test_{}_{}", std::process::id(), suffix);

        let client = redis::Client::open(redis_url.as_str()).expect("Invalid REDIS_URL");
        let conn = alaya_bridge::connect(&client, alaya_bridge::query_config(), "test")
            .await
            .expect("Failed to connect to Redis");

        Some(Self { conn, graph_name })
    }

    /// Execute a Cypher query against this context's graph, returning the
    /// parsed `FalkorResult`.
    pub async fn exec(
        &self,
        cypher: &str,
        params: HashMap<String, Value>,
        readonly: bool,
    ) -> alaya_bridge::FalkorResult {
        let state = build_state(&self.conn, &self.graph_name);
        alaya_bridge::exec_query(&state, cypher, params, readonly)
            .await
            .unwrap_or_else(|status| {
                panic!("exec_query failed with status {status} for query: {cypher}")
            })
    }

    /// Execute a pre-built `CypherQuery` tuple.
    pub async fn exec_tuple(
        &self,
        (cypher, params, readonly): (String, HashMap<String, Value>, bool),
    ) -> alaya_bridge::FalkorResult {
        self.exec(&cypher, params, readonly).await
    }

    /// DELETE the ephemeral graph.  Call at the end of each test.
    pub async fn cleanup(&self) {
        let mut conn = self.conn.clone();
        let _: Result<(), _> = redis::cmd("GRAPH.DELETE")
            .arg(&self.graph_name)
            .query_async(&mut conn)
            .await;
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Build a temporary `AppState` pointing at the given connection + graph.
///
/// The state is stack-allocated and not shared — fine for test-time use.
pub fn build_state(
    conn: &redis::aio::ConnectionManager,
    graph_name: &str,
) -> alaya_bridge::AppState {
    alaya_bridge::AppState {
        redis: conn.clone(),
        graph_name: graph_name.to_string(),
        // Tests call `exec_query` directly; the auth layer is not on this path.
        auth: alaya_bridge::BridgeAuth::Open,
    }
}
