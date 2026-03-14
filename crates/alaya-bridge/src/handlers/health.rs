//! Health and stats handlers — GET /health, GET /stats

use std::collections::HashMap;
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use serde_json::{json, Value};

use alaya_types::graph::GraphStats;

use crate::{cypher, handlers::exec_query, AppState};

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
/// Executes graph statistics queries and returns a `GraphStats` snapshot.
pub async fn stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, StatusCode> {
    let queries = cypher::get_graph_stats();
    // Queries: [node_count, hebbian_count, RELATES_TO, PRECEDES, CONTRADICTS, SUPERSEDES]
    if queries.len() < 6 {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let mut counts: Vec<i64> = Vec::with_capacity(queries.len());
    for (q, params, readonly) in queries {
        let result = exec_query(&state, &q, params, readonly).await?;
        counts.push(result.count().unwrap_or(0));
    }

    let node_count = counts[0] as usize;
    let hebbian_edge_count = counts[1] as usize;
    let relates_to = counts[2] as usize;
    let precedes = counts[3] as usize;
    let contradicts = counts[4] as usize;
    let supersedes = counts[5] as usize;

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
