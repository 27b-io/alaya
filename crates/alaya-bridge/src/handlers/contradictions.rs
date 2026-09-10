//! Contradiction handlers — POST /contradictions/{all,for,verdict}

use std::collections::HashMap;
use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};

use alaya_types::{
    graph::{Contradiction, ContradictionQuery, EdgeVerdict, Verdict},
    memory::validate_content_hash,
};

use crate::{AppState, cypher, handlers::exec_query};

// ─── Request types ────────────────────────────────────────────────────────────

/// `POST /contradictions/for` refuses oversized hash lists — an accidental
/// 10k-hash `IN` list is a FalkorDB DoS (LAB-3283 review).
pub const MAX_FOR_HASHES: usize = 500;

#[derive(Debug, Deserialize)]
pub struct ContradictionsForRequest {
    pub hashes: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct SetVerdictRequest {
    pub source: String,
    pub target: String,
    #[serde(flatten)]
    pub verdict: EdgeVerdict,
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// POST /contradictions/all
///
/// One page of CONTRADICTS pairs ordered by `created_at DESC`, every filter
/// in the body applied in Cypher (see `ContradictionQuery`; `limit` is
/// clamped to 1..=500), each carrying its persisted verdict (if any).
pub async fn all(
    State(state): State<Arc<AppState>>,
    Json(query): Json<ContradictionQuery>,
) -> Result<Json<Value>, StatusCode> {
    let (cypher, params, readonly) = cypher::get_all_contradictions(&query);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // Row layout: cypher::CONTRADICTION_COLUMNS
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
        contradictions.push(Contradiction {
            memory_a_hash,
            memory_b_hash,
            confidence,
            created_at,
            verdict: parse_verdict(row),
        });
    }

    Ok(Json(json!({ "contradictions": contradictions })))
}

/// Columns 4.. of a `CONTRADICTION_COLUMNS` row. `None` when the edge has
/// no (recognised) verdict — an unknown verdict string reads as unjudged so
/// the backfill re-judges it rather than trusting a corrupt value.
fn parse_verdict(row: &[Value]) -> Option<EdgeVerdict> {
    let verdict = Verdict::parse(row.get(4)?.as_str()?)?;
    Some(EdgeVerdict {
        verdict,
        verdict_survivor: row.get(5).and_then(Value::as_str).map(str::to_string),
        verdict_reason: row
            .get(6)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        verdict_confidence: row.get(7).and_then(Value::as_f64).unwrap_or_default(),
        verdict_model: row
            .get(8)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        judged_at: row.get(9).and_then(Value::as_f64).unwrap_or_default(),
    })
}

/// POST /contradictions/verdict
///
/// Annotate an existing `source -> target` CONTRADICTS edge with the judge's
/// verdict. MATCH-only (LAB-3283 AC-3): the edge is never created or
/// deleted, and no Memory node is touched. `updated: false` means no such
/// edge exists.
pub async fn set_verdict(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetVerdictRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !validate_content_hash(&req.source) || !validate_content_hash(&req.target) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    if let Some(s) = &req.verdict.verdict_survivor
        && !validate_content_hash(s)
    {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    if !(0.0..=1.0).contains(&req.verdict.verdict_confidence) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let (cypher, params, readonly) =
        cypher::set_contradiction_verdict(&req.source, &req.target, &req.verdict);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    let count = result.count().unwrap_or(0);
    Ok(Json(json!({ "updated": count > 0 })))
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
    if req.hashes.len() > MAX_FOR_HASHES {
        return Err(StatusCode::BAD_REQUEST);
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
