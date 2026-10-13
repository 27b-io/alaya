//! Consolidation handlers — decay, prune, orphan detection, and the sweep of
//! nodes whose memory is gone.
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ListNodesRequest {
    /// Hashes after this one, in hash order (`""` = from the start).
    #[serde(default)]
    pub after: String,
    /// Page size (default 500, at most 5 000).
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepNodesRequest {
    pub hashes: Vec<String>,
    /// The sweep run's mark.
    pub token: String,
}

/// At most this many hashes per mark or delete call.
const MAX_SWEEP_BATCH: usize = 5_000;

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// POST /consolidation/nodes
///
/// One page of Memory node hashes after `after`, in hash order.
pub async fn list_nodes(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ListNodesRequest>,
) -> Result<Json<Value>, StatusCode> {
    let limit = req.limit.unwrap_or(500).clamp(1, MAX_SWEEP_BATCH as i64) as u32;
    let (cypher, params, readonly) = cypher::list_memory_nodes(&req.after, limit);
    let result = exec_query(&state, &cypher, params, readonly).await?;
    let hashes: Vec<&str> = result
        .result_set
        .iter()
        .filter_map(|row| row.first()?.as_str())
        .collect();
    Ok(Json(json!({ "hashes": hashes })))
}

/// The hashes and token of a sweep step, checked: 422 on an empty token or
/// an oversized batch.
fn sweep_args(req: &SweepNodesRequest) -> Result<Vec<&str>, StatusCode> {
    if req.token.trim().is_empty() || req.hashes.len() > MAX_SWEEP_BATCH {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    Ok(req.hashes.iter().map(String::as_str).collect())
}

/// POST /consolidation/sweep-mark
///
/// Mark the given Memory nodes with the sweep's token.
pub async fn sweep_mark(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SweepNodesRequest>,
) -> Result<Json<Value>, StatusCode> {
    let hashes = sweep_args(&req)?;
    let (cypher, params, readonly) = cypher::sweep_mark_memory_nodes(&hashes, &req.token);
    let result = exec_query(&state, &cypher, params, readonly).await?;
    Ok(Json(json!({ "marked": result.count().unwrap_or(0) })))
}

/// POST /consolidation/sweep-delete
///
/// DETACH DELETE the given Memory nodes that still carry the sweep's token.
pub async fn sweep_delete(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SweepNodesRequest>,
) -> Result<Json<Value>, StatusCode> {
    let hashes = sweep_args(&req)?;
    let (cypher, params, readonly) = cypher::sweep_delete_marked_nodes(&hashes, &req.token);
    let result = exec_query(&state, &cypher, params, readonly).await?;
    Ok(Json(json!({ "deleted": result.count().unwrap_or(0) })))
}

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
