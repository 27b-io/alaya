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
    /// Stamp only an edge with no resolution yet (`settle_contradiction`):
    /// an existing stamp, and its who and when, is kept.
    #[serde(default)]
    pub if_unresolved: bool,
    /// Clear only while the stamp is exactly `resolved_via` at `resolved_at`
    /// (`unsettle_contradiction`): a stamp written since is kept.
    #[serde(default)]
    pub if_stamped: bool,
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

    let via = req.resolved_via.trim();
    let (cypher, params, readonly) = match (req.if_unresolved, req.if_stamped, req.resolution) {
        (false, false, resolution) => cypher::set_contradiction_resolution(
            &req.source,
            &req.target,
            resolution,
            via,
            req.resolved_at,
        ),
        (true, false, Some(Resolution::KeepBoth)) => {
            cypher::settle_contradiction(&req.source, &req.target, via, req.resolved_at)
        }
        (false, true, None) if !via.is_empty() => {
            cypher::unsettle_contradiction(&req.source, &req.target, via, req.resolved_at)
        }
        // "Set only if unresolved" needs a stamp, "clear only if stamped"
        // needs the stamp to match; anything else is malformed.
        _ => return Err(StatusCode::UNPROCESSABLE_ENTITY),
    };
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
/// Every CONTRADICTS edge touching any of the supplied hashes, grouped by
/// requested hash (an edge between two requested hashes is listed under
/// both). Each entry is a `Contradiction` (verdict and resolution included)
/// plus `resolved` (any stamp, the queue's own test) and
/// `memory_a_superseded` / `memory_b_superseded` (an incoming `SUPERSEDES`
/// edge). Read-only.
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

    // Absent hashes still appear, as empty arrays.
    let mut map: HashMap<String, Vec<Value>> =
        req.hashes.iter().map(|h| (h.clone(), Vec::new())).collect();
    for row in &result.result_set {
        let Some(entry) = for_hashes_entry(row) else {
            continue;
        };
        for side in ["memory_a_hash", "memory_b_hash"] {
            if let Some(list) = entry[side].as_str().and_then(|h| map.get_mut(h)) {
                list.push(entry.clone());
            }
        }
    }

    Ok(Json(json!({ "contradictions": map })))
}

/// One `get_contradictions_for_hashes` row as its wire entry; `None` for a
/// row without both endpoint hashes.
fn for_hashes_entry(row: &[Value]) -> Option<Value> {
    let memory_a_hash = row.first()?.as_str().filter(|h| !h.is_empty())?;
    let memory_b_hash = row.get(1)?.as_str().filter(|h| !h.is_empty())?;
    let (resolution, resolved_at, resolved_via) = parse_resolution(row);
    let pair = Contradiction {
        memory_a_hash: memory_a_hash.to_string(),
        memory_b_hash: memory_b_hash.to_string(),
        confidence: row.get(2).and_then(Value::as_f64),
        created_at: row.get(3).and_then(Value::as_f64),
        verdict: parse_verdict(row),
        resolution,
        resolved_at,
        resolved_via,
    };
    let superseded = |i: usize| row.get(i).and_then(Value::as_f64).is_some_and(|n| n > 0.0);
    let mut entry = serde_json::to_value(pair).ok()?;
    let obj = entry.as_object_mut()?;
    // Any stamp settles the pair, a value this build does not parse included.
    obj.insert(
        "resolved".into(),
        json!(row.get(10).is_some_and(|v| !v.is_null())),
    );
    obj.insert("memory_a_superseded".into(), json!(superseded(13)));
    obj.insert("memory_b_superseded".into(), json!(superseded(14)));
    Some(entry)
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

    fn for_row(resolution: Value, a_sup: i64, b_sup: i64) -> Vec<Value> {
        vec![
            json!("a"),
            json!("b"),
            json!(0.7),
            json!(1.0),
            json!("supersession"),
            json!("b"),
            json!("newer"),
            json!(0.95),
            json!("m"),
            json!(2.0),
            resolution,
            Value::Null,
            Value::Null,
            json!(a_sup),
            json!(b_sup),
        ]
    }

    #[test]
    fn for_hashes_entry_carries_verdict_resolution_and_superseded_flags() {
        let e = for_hashes_entry(&for_row(Value::Null, 0, 1)).unwrap();
        assert_eq!(e["memory_a_hash"], "a");
        assert_eq!(e["verdict"], "supersession");
        assert_eq!(e["verdict_survivor"], "b");
        assert_eq!(e["verdict_confidence"], 0.95);
        assert_eq!(e["resolved"], false);
        assert_eq!(e["memory_a_superseded"], false);
        assert_eq!(e["memory_b_superseded"], true);

        let stamped = for_hashes_entry(&for_row(json!("keep_both"), 0, 0)).unwrap();
        assert_eq!(stamped["resolved"], true);
        assert_eq!(stamped["resolution"], "keep_both");
    }

    /// A stamp this build cannot parse still settles the pair, as the
    /// queue's `e.resolution IS NULL` test does.
    #[test]
    fn for_hashes_entry_reads_an_unknown_stamp_as_resolved() {
        let e = for_hashes_entry(&for_row(json!("keep_neither"), 0, 0)).unwrap();
        assert_eq!(e["resolved"], true);
        assert!(e.get("resolution").is_none());
    }

    #[test]
    fn for_hashes_entry_skips_a_row_without_both_hashes() {
        let mut row = for_row(Value::Null, 0, 0);
        row[1] = json!("");
        assert!(for_hashes_entry(&row).is_none());
        assert!(for_hashes_entry(&[json!("a")]).is_none());
    }
}
