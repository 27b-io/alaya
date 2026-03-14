//! Consolidation handlers — decay, prune, and orphan detection.
//!
//! These are maintenance operations executed synchronously against FalkorDB.
//! Each runs a bounded-limit batch to avoid long-running transactions.

use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{AppState, cypher, handlers::exec_query};

// ─── Request types ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DecayAllRequest {
    /// Multiplicative decay factor applied to all HEBBIAN edge weights (e.g. 0.95).
    pub decay_factor: f64,
    /// Max edges to process per call (default 10 000).
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct DecayStaleRequest {
    /// Unix timestamp (seconds).  Edges with `last_co_access < stale_before` are decayed.
    pub stale_before: f64,
    /// Multiplicative decay factor.
    pub decay_factor: f64,
    /// Max edges to process per call.
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct PruneRequest {
    /// Edges with `weight < threshold` are deleted.
    pub threshold: f64,
    /// Max edges to process per call.
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct OrphansRequest {
    /// Max orphan nodes to return (default 100).
    pub limit: Option<i64>,
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// POST /consolidation/decay-all
///
/// Decay all HEBBIAN edge weights by `decay_factor`.
pub async fn decay_all(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DecayAllRequest>,
) -> Result<Json<Value>, StatusCode> {
    let limit = req.limit.unwrap_or(10_000).clamp(1, 100_000) as u32;

    let (cypher, params, readonly) = cypher::decay_all_edges(req.decay_factor, limit);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    let decayed = result.count().unwrap_or(0);
    Ok(Json(json!({ "decayed": decayed })))
}

/// POST /consolidation/decay-stale
///
/// Decay HEBBIAN edges that have not been co-accessed since `stale_before`.
pub async fn decay_stale(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DecayStaleRequest>,
) -> Result<Json<Value>, StatusCode> {
    let limit = req.limit.unwrap_or(10_000).clamp(1, 100_000) as u32;

    let (cypher, params, readonly) =
        cypher::decay_stale_edges(req.stale_before, req.decay_factor, limit);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    let decayed = result.count().unwrap_or(0);
    Ok(Json(json!({ "decayed": decayed })))
}

/// POST /consolidation/prune
///
/// Delete HEBBIAN edges whose weight has fallen below `threshold`.
pub async fn prune(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PruneRequest>,
) -> Result<Json<Value>, StatusCode> {
    let limit = req.limit.unwrap_or(10_000).clamp(1, 100_000) as u32;

    let (cypher, params, readonly) = cypher::prune_weak_edges(req.threshold, limit);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    let pruned = result.count().unwrap_or(0);
    Ok(Json(json!({ "pruned": pruned })))
}

/// POST /consolidation/orphans
///
/// Return content hashes of Memory nodes with no edges of any tracked type.
pub async fn orphans(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OrphansRequest>,
) -> Result<Json<Value>, StatusCode> {
    let limit = req.limit.unwrap_or(100).clamp(1, 10_000) as u32;

    let (cypher, params, readonly) = cypher::get_orphan_nodes(limit);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // Row: [m.content_hash]
    let orphan_hashes: Vec<&str> = result
        .result_set
        .iter()
        .filter_map(|row| row.first()?.as_str())
        .filter(|s| !s.is_empty())
        .collect();

    Ok(Json(json!({ "orphans": orphan_hashes })))
}
