//! Health and stats handlers — GET /health, GET /stats

use std::collections::HashMap;
use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde_json::{Value, json};

use alaya_types::graph::GraphStats;

use crate::{AppState, cypher, handlers::exec_query};

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// GET /health
///
/// Probes Redis/FalkorDB with PING and returns connectivity status.
pub async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let connected = ping_redis(&state).await;
    Json(json!({
        "status": if connected { "ok" } else { "error" },
        "falkordb_connected": connected,
        "write_queue_depth": 0
    }))
}

/// GET /stats
///
/// Executes a single UNION ALL query for all graph statistics.
/// Previously used 6 sequential queries (one per edge type), each a full
/// Redis round-trip. Now does it in one.
pub async fn stats(State(state): State<Arc<AppState>>) -> Result<Json<Value>, StatusCode> {
    let (q, params, readonly) = cypher::get_graph_stats_union();
    let result = exec_query(&state, &q, params, readonly).await?;

    // Parse rows of (kind, cnt) into a lookup
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for row in &result.result_set {
        if let (Some(kind), Some(cnt)) = (
            row.first().and_then(|v| v.as_str()),
            row.get(1).and_then(|v| v.as_i64()),
        ) {
            counts.insert(kind, cnt as usize);
        }
    }

    let node_count = counts.get("nodes").copied().unwrap_or(0);
    let hebbian_edge_count = counts.get("HEBBIAN").copied().unwrap_or(0);
    let relates_to = counts.get("RELATES_TO").copied().unwrap_or(0);
    let precedes = counts.get("PRECEDES").copied().unwrap_or(0);
    let contradicts = counts.get("CONTRADICTS").copied().unwrap_or(0);
    let supersedes = counts.get("SUPERSEDES").copied().unwrap_or(0);

    let edge_count = hebbian_edge_count + relates_to + precedes + contradicts + supersedes;

    let mut typed_edge_counts = HashMap::new();
    typed_edge_counts.insert("RELATES_TO".to_string(), relates_to);
    typed_edge_counts.insert("PRECEDES".to_string(), precedes);
    typed_edge_counts.insert("CONTRADICTS".to_string(), contradicts);
    typed_edge_counts.insert("SUPERSEDES".to_string(), supersedes);

    let graph_stats = GraphStats {
        graph_name: state.graph_name.clone(),
        node_count,
        edge_count,
        hebbian_edge_count,
        typed_edge_counts,
        status: "ok".to_string(),
    };

    Ok(Json(
        serde_json::to_value(graph_stats).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    ))
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async fn ping_redis(state: &AppState) -> bool {
    let mut conn = state.redis.clone();
    let result: redis::RedisResult<String> = redis::cmd("PING").query_async(&mut conn).await;
    matches!(result, Ok(ref s) if s == "PONG")
}
