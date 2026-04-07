//! Hebbian handlers — read traversals and write-queue enqueue.
//!
//! Read endpoints query FalkorDB directly via `exec_query`.
//! The write endpoint enqueues co-access pairs to a Redis list for
//! async processing by the Hebbian worker (Task 11).

use std::collections::HashMap;
use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};

use alaya_types::graph::{CoAccessPair, Neighbor};

use crate::{AppState, cypher, handlers::exec_query};

// ─── Request types ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct NeighborsRequest {
    pub content_hash: String,
    /// Maximum hop distance (default 1, capped at 3 in cypher layer)
    pub max_hops: Option<u8>,
    /// Minimum edge weight to traverse (default 0.0)
    pub min_weight: Option<f64>,
    /// Max results (default 20)
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct SpreadingRequest {
    pub seed_hashes: Vec<String>,
    /// Maximum hop distance (default 2)
    pub max_hops: Option<u8>,
    /// Decay applied per hop: activation = path_weight * decay^hops (default 0.5)
    pub decay_factor: Option<f64>,
    /// Minimum activation score to include in result (default 0.01)
    pub min_activation: Option<f64>,
    /// Max distinct hashes returned (default 50)
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct BoostsWithinRequest {
    pub hashes: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct StrengthenRequest {
    pub pairs: Vec<CoAccessPair>,
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// POST /hebbian/neighbors
///
/// Walk HEBBIAN edges up to `max_hops` from a source node and return scored neighbors.
pub async fn neighbors(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NeighborsRequest>,
) -> Result<Json<Value>, StatusCode> {
    let max_hops = req.max_hops.unwrap_or(1);
    let min_weight = req.min_weight.unwrap_or(0.0);
    let limit = req.limit.unwrap_or(20).clamp(1, 200) as u32;

    let (cypher, params, readonly) =
        cypher::get_neighbors(&req.content_hash, max_hops, min_weight, limit);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // Row: [hash, path_weight, hops]
    let mut neighbors: Vec<Neighbor> = Vec::with_capacity(result.result_set.len());
    for row in &result.result_set {
        if row.len() < 3 {
            continue;
        }
        let content_hash = row[0].as_str().unwrap_or("").to_string();
        let weight = row[1].as_f64().unwrap_or(0.0);
        let hops = row[2].as_u64().unwrap_or(1) as u32;
        if content_hash.is_empty() {
            continue;
        }
        neighbors.push(Neighbor {
            content_hash,
            weight,
            hops,
        });
    }

    Ok(Json(json!({ "neighbors": neighbors })))
}

/// POST /hebbian/spreading
///
/// Spreading activation from a set of seed hashes.  The Cypher layer returns
/// raw `(hash, path_weight, hops)` tuples; this handler applies the
/// `decay_factor^hops` penalty, filters by `min_activation`, takes the max
/// activation per hash, sorts descending, and applies the limit.
pub async fn spreading(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpreadingRequest>,
) -> Result<Json<Value>, StatusCode> {
    if req.seed_hashes.is_empty() {
        return Ok(Json(json!({ "activations": {} })));
    }

    let max_hops = req.max_hops.unwrap_or(2);
    let decay_factor = req.decay_factor.unwrap_or(0.5);
    let min_activation = req.min_activation.unwrap_or(0.01);
    let limit = req.limit.unwrap_or(50).clamp(1, 500) as usize;

    let seeds_ref: Vec<&str> = req.seed_hashes.iter().map(String::as_str).collect();
    let (cypher, params, readonly) = cypher::spreading_activation(&seeds_ref, max_hops);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // Row: [hash, path_weight, hops]
    // Apply decay and accumulate max activation per hash.
    let mut activation_map: HashMap<String, f64> = HashMap::new();
    for row in &result.result_set {
        if row.len() < 3 {
            continue;
        }
        let hash = match row[0].as_str() {
            Some(h) if !h.is_empty() => h.to_string(),
            _ => continue,
        };
        let path_weight = row[1].as_f64().unwrap_or(0.0);
        let hops = row[2].as_u64().unwrap_or(1) as i32;

        // activation = path_weight * decay_factor^hops
        let activation = path_weight * decay_factor.powi(hops);
        if activation < min_activation {
            continue;
        }

        // Keep highest activation per hash (multiple paths may reach same node)
        let entry = activation_map.entry(hash).or_insert(0.0);
        if activation > *entry {
            *entry = activation;
        }
    }

    // Sort by activation descending, then apply limit
    let mut sorted: Vec<(String, f64)> = activation_map.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(limit);

    let activations: serde_json::Map<String, Value> =
        sorted.into_iter().map(|(k, v)| (k, json!(v))).collect();

    Ok(Json(json!({ "activations": activations })))
}

/// POST /hebbian/boosts-within
///
/// Return the maximum HEBBIAN edge weight for each hash that has at least one
/// edge to another hash in the supplied set.
pub async fn boosts_within(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BoostsWithinRequest>,
) -> Result<Json<Value>, StatusCode> {
    if req.hashes.is_empty() {
        return Ok(Json(json!({ "boosts": {} })));
    }

    let hashes_ref: Vec<&str> = req.hashes.iter().map(String::as_str).collect();
    let (cypher, params, readonly) = cypher::hebbian_boosts_within(&hashes_ref);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // Row: [hash, max_weight]
    let mut boosts: serde_json::Map<String, Value> = serde_json::Map::new();
    for row in &result.result_set {
        if row.len() < 2 {
            continue;
        }
        let hash = match row[0].as_str() {
            Some(h) if !h.is_empty() => h.to_string(),
            _ => continue,
        };
        let max_weight = row[1].as_f64().unwrap_or(0.0);
        boosts.insert(hash, json!(max_weight));
    }

    Ok(Json(json!({ "boosts": boosts })))
}

/// POST /hebbian/strengthen
///
/// Enqueue co-access pairs to Redis list `alaya:hebbian:queue` for async LTP
/// processing by the Hebbian worker.  Returns 202 Accepted immediately.
///
/// Rejects with 422 if more than 50 pairs are submitted in a single call.
pub async fn strengthen(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StrengthenRequest>,
) -> Result<(StatusCode, Json<Value>), StatusCode> {
    const MAX_PAIRS: usize = 50;

    if req.pairs.len() > MAX_PAIRS {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    if req.pairs.is_empty() {
        return Ok((
            StatusCode::ACCEPTED,
            Json(json!({ "accepted": true, "queued": 0 })),
        ));
    }

    let queued = req.pairs.len();
    let mut conn = state.redis.clone();

    // Serialize all pairs first so we fail fast before touching Redis
    let mut serialized: Vec<String> = Vec::with_capacity(req.pairs.len());
    for pair in &req.pairs {
        serialized.push(serde_json::to_string(pair).map_err(|e| {
            tracing::error!("Failed to serialize CoAccessPair: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?);
    }

    // Pipeline all LPUSH commands into a single Redis round-trip
    let mut pipe = redis::pipe();
    for s in &serialized {
        pipe.cmd("LPUSH").arg("alaya:hebbian:queue").arg(s);
    }
    pipe.query_async::<()>(&mut conn).await.map_err(|e| {
        tracing::error!("Redis pipeline LPUSH error: {e}");
        StatusCode::BAD_GATEWAY
    })?;

    Ok((
        StatusCode::ACCEPTED,
        Json(json!({ "accepted": true, "queued": queued })),
    ))
}
