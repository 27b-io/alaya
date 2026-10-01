//! Contradiction handlers — POST /contradictions/{all,for,verdict,resolution,stats}

use std::collections::HashMap;
use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};

use alaya_types::{
    graph::{
        Contradiction, ContradictionQuery, ContradictionStats, ContradictionStatsQuery, DayTally,
        EdgeVerdict, ReasonTally, Resolution, Verdict, VerdictCount, VerdictTally,
    },
    memory::validate_content_hash,
};

use crate::{AppState, cypher, handlers::exec_query, resp::FalkorResult};

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

/// `resolution: null` (or absent) clears the stamp; `resolved_via` and
/// `resolved_at` are then ignored but still sent (the server always has
/// them). Explicit fields, not a flattened struct: a flattened `Option`
/// would read a malformed set as a clear.
#[derive(Debug, Deserialize)]
pub struct SetResolutionRequest {
    pub source: String,
    pub target: String,
    pub resolution: Option<Resolution>,
    pub resolved_via: String,
    pub resolved_at: f64,
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
        let (resolution, resolved_at, resolved_via) = parse_resolution(row);
        contradictions.push(Contradiction {
            memory_a_hash,
            memory_b_hash,
            confidence,
            created_at,
            verdict: parse_verdict(row),
            resolution,
            resolved_at,
            resolved_via,
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

/// Columns 10..=12 of a `CONTRADICTION_COLUMNS` row. An unrecognised
/// `resolution` string (only reachable by a direct graph write — every API
/// path goes through the enum) reads as unresolved here with its companions
/// dropped; the graph-side queue filter still hides that edge, so it shows
/// only under `include_resolved`, as `resolution: null`.
fn parse_resolution(row: &[Value]) -> (Option<Resolution>, Option<f64>, Option<String>) {
    let Some(resolution) = row
        .get(10)
        .and_then(Value::as_str)
        .and_then(Resolution::parse)
    else {
        return (None, None, None);
    };
    (
        Some(resolution),
        row.get(11).and_then(Value::as_f64),
        row.get(12).and_then(Value::as_str).map(str::to_string),
    )
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

/// POST /contradictions/resolution
///
/// Stamp (or clear) the operator's resolution on an existing
/// `source -> target` CONTRADICTS edge (LAB-3885). MATCH-only: the edge is
/// never created or deleted, no Memory node and no verdict property is
/// touched. A set requires a non-empty `resolved_via`. `updated: false`
/// means no such edge exists.
pub async fn set_resolution(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetResolutionRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !validate_content_hash(&req.source) || !validate_content_hash(&req.target) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }
    if req.resolution.is_some() && req.resolved_via.trim().is_empty() {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let (cypher, params, readonly) = cypher::set_contradiction_resolution(
        &req.source,
        &req.target,
        req.resolution,
        req.resolved_via.trim(),
        req.resolved_at,
    );
    let result = exec_query(&state, &cypher, params, readonly).await?;

    let count = result.count().unwrap_or(0);
    Ok(Json(json!({ "updated": count > 0 })))
}

/// POST /contradictions/stats
///
/// Whole-graph aggregates over every CONTRADICTS edge (LAB-6881): counts by
/// verdict and resolved state, the most frequent stored judge failures,
/// judgements per UTC day since `judged_since`, and degenerate reasons per
/// verdict. Read-only (`GRAPH.RO_QUERY` throughout). Any one query failing
/// fails the call — a partial tally would read as a smaller corpus.
///
/// The queries run one after another on purpose. Each is a full pass over
/// the relation, and run together they contend for the same graph: every
/// one slows several-fold for no gain in total time, which pushes each
/// toward FalkorDB's per-query timeout.
pub async fn stats(
    State(state): State<Arc<AppState>>,
    Json(query): Json<ContradictionStatsQuery>,
) -> Result<Json<ContradictionStats>, StatusCode> {
    let run = |(q, p, ro): cypher::CypherQuery| {
        let state = state.clone();
        async move { exec_query(&state, &q, p, ro).await }
    };
    let counts = run(cypher::contradiction_stats_counts()).await?;
    let reverse = run(cypher::contradiction_stats_reverse_stamped()).await?;
    let failures = run(cypher::contradiction_stats_failures(
        ContradictionStatsQuery::FAILURE_REASONS,
    ))
    .await?;
    let per_day = run(cypher::contradiction_stats_judged_per_day(
        query.judged_since,
    ))
    .await?;
    let degenerate = run(cypher::contradiction_stats_degenerate()).await?;

    let mut verdicts = rows(&counts, "counts", |row| {
        Some(VerdictTally {
            verdict: opt_str(row.first())?,
            resolved: row.get(1)?.as_bool()?,
            count: count_cell(row.get(2))?,
        })
    })?;
    // The counts query applies the queue's direct test only; move each
    // reverse-stamped edge from its open row to the resolved one.
    let reverse = rows(&reverse, "reverse_stamped", |row| {
        Some((opt_str(row.first())?, count_cell(row.get(1))?))
    })?;
    for (verdict, n) in reverse {
        move_to_resolved(&mut verdicts, verdict, n);
    }
    verdicts.retain(|t| t.count > 0);

    let failures = rows(&failures, "failures", |row| {
        Some(ReasonTally {
            reason: row.first()?.as_str()?.to_string(),
            count: count_cell(row.get(1))?,
        })
    })?;
    let judged_per_day = rows(&per_day, "judged_per_day", |row| {
        Some(DayTally {
            day: row.first()?.as_i64()?,
            verdict: opt_str(row.get(1))?,
            count: count_cell(row.get(2))?,
        })
    })?;
    let degenerate_reasons = rows(&degenerate, "degenerate", |row| {
        Some(VerdictCount {
            verdict: row.first()?.as_str()?.to_string(),
            count: count_cell(row.get(1))?,
        })
    })?;

    Ok(Json(ContradictionStats {
        verdicts,
        failures,
        judged_per_day,
        degenerate_reasons,
    }))
}

/// Parse every row of an aggregate, or fail the call. A tally that skipped
/// a row it could not read would report a smaller corpus than the graph
/// holds, so a cell of the wrong type is a 500, never a quiet undercount.
fn rows<T>(
    result: &FalkorResult,
    query: &str,
    parse: impl Fn(&[Value]) -> Option<T>,
) -> Result<Vec<T>, StatusCode> {
    result
        .result_set
        .iter()
        .map(|row| {
            parse(row).ok_or_else(|| {
                tracing::error!(query, "contradiction stats: unreadable row");
                StatusCode::INTERNAL_SERVER_ERROR
            })
        })
        .collect()
}

/// A verdict cell: `Some(None)` for NULL (never judged), `None` for a cell
/// that is neither NULL nor a string.
fn opt_str(v: Option<&Value>) -> Option<Option<String>> {
    match v? {
        Value::Null => Some(None),
        Value::String(s) => Some(Some(s.clone())),
        _ => None,
    }
}

fn count_cell(v: Option<&Value>) -> Option<usize> {
    v.and_then(Value::as_u64).map(|n| n as usize)
}

/// Shift `n` edges of `verdict` from the open tally to the resolved one.
/// The reverse query only returns edges the counts query saw as open, so
/// the open row holds at least `n`; `min` keeps a write landing between the
/// two reads from driving it below zero.
fn move_to_resolved(tallies: &mut Vec<VerdictTally>, verdict: Option<String>, n: usize) {
    let Some(open) = tallies
        .iter_mut()
        .find(|t| t.verdict == verdict && !t.resolved)
    else {
        return;
    };
    let n = n.min(open.count);
    open.count -= n;
    match tallies
        .iter_mut()
        .find(|t| t.verdict == verdict && t.resolved)
    {
        Some(resolved) => resolved.count += n,
        None => tallies.push(VerdictTally {
            verdict,
            resolved: true,
            count: n,
        }),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn t(verdict: Option<&str>, resolved: bool, count: usize) -> VerdictTally {
        VerdictTally {
            verdict: verdict.map(str::to_string),
            resolved,
            count,
        }
    }

    #[test]
    fn move_to_resolved_shifts_between_rows_and_creates_the_resolved_row() {
        let mut v = vec![
            t(Some("coexist"), false, 5),
            t(None, false, 3),
            t(None, true, 1),
        ];
        move_to_resolved(&mut v, Some("coexist".into()), 2);
        move_to_resolved(&mut v, None, 1);
        assert_eq!(
            v,
            [
                t(Some("coexist"), false, 3),
                t(None, false, 2),
                t(None, true, 2),
                t(Some("coexist"), true, 2),
            ]
        );
    }

    /// A write between the two reads can leave the reverse count above the
    /// open count; the open row floors at zero and the total is unchanged.
    #[test]
    fn move_to_resolved_never_drives_a_row_below_zero() {
        let mut v = vec![t(Some("coexist"), false, 1)];
        move_to_resolved(&mut v, Some("coexist".into()), 4);
        assert_eq!(
            v,
            [t(Some("coexist"), false, 0), t(Some("coexist"), true, 1)]
        );
        // No open row at all: nothing to move.
        move_to_resolved(&mut v, Some("unrelated".into()), 2);
        assert_eq!(v.len(), 2);
    }
}
