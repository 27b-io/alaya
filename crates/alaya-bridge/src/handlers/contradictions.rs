//! Contradiction handlers — POST /contradictions/all, POST /contradictions/for

use std::collections::HashMap;
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use alaya_types::graph::Contradiction;

use crate::{cypher, handlers::exec_query, AppState};

// ─── Request types ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AllContradictionsRequest {
    /// Max results (default 20)
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct ContradictionsForRequest {
    pub hashes: Vec<String>,
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// POST /contradictions/all
///
/// Return all CONTRADICTS pairs ordered by `created_at DESC`.
pub async fn all(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AllContradictionsRequest>,
) -> Result<Json<Value>, StatusCode> {
    let limit = req.limit.unwrap_or(20).clamp(1, 500) as u32;

    let (cypher, params, readonly) = cypher::get_all_contradictions(limit);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // Row: [a.content_hash, b.content_hash, e.confidence, e.created_at]
    let mut contradictions: Vec<Contradiction> = Vec::with_capacity(result.result_set.len());
    for row in &result.result_set {
        if row.len() < 2 {
            continue;
        }
        let memory_a_hash = row[0].as_str().unwrap_or("").to_string();
        let memory_b_hash = row[1].as_str().unwrap_or("").to_string();
        if memory_a_hash.is_empty() || memory_b_hash.is_empty() {
            continue;
        }
        let confidence = row.get(2).and_then(Value::as_f64);
        let created_at = row.get(3).and_then(Value::as_f64);
        contradictions.push(Contradiction { memory_a_hash, memory_b_hash, confidence, created_at });
    }

    Ok(Json(json!({ "contradictions": contradictions })))
}

/// POST /contradictions/for
///
/// Return CONTRADICTS pairs touching any of the supplied hashes, grouped by hash.
pub async fn for_hashes(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ContradictionsForRequest>,
) -> Result<Json<Value>, StatusCode> {
    if req.hashes.is_empty() {
        return Ok(Json(json!({ "contradictions": {} })));
    }

    let hashes_ref: Vec<&str> = req.hashes.iter().map(String::as_str).collect();
    let (cypher, params, readonly) = cypher::get_contradictions_for_hashes(&hashes_ref);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // Row: [a.content_hash, b.content_hash, e.confidence]
    // Build per-hash map: each hash gets all pairs it participates in.
    let mut map: HashMap<String, Vec<Value>> = HashMap::new();

    // Pre-populate requested hashes so absent ones appear as empty arrays
    for h in &req.hashes {
        map.entry(h.clone()).or_default();
    }

    for row in &result.result_set {
        if row.len() < 2 {
            continue;
        }
        let hash_a = row[0].as_str().unwrap_or("").to_string();
        let hash_b = row[1].as_str().unwrap_or("").to_string();
        if hash_a.is_empty() || hash_b.is_empty() {
            continue;
        }
        let confidence = row.get(2).and_then(Value::as_f64);

        let entry = json!({
            "memory_a_hash": hash_a,
            "memory_b_hash": hash_b,
            "confidence": confidence
        });

        // Attach to both sides if they were in the request set
        if map.contains_key(&hash_a) {
            map.get_mut(&hash_a).unwrap().push(entry.clone());
        }
        if map.contains_key(&hash_b) {
            map.get_mut(&hash_b).unwrap().push(entry);
        }
    }

    let contradictions: serde_json::Map<String, Value> =
        map.into_iter().map(|(k, v)| (k, json!(v))).collect();

    Ok(Json(json!({ "contradictions": contradictions })))
}
