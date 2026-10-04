//! Corpus and contradiction-judge statistics (LAB-6881) behind `GET /stats`.
//!
//! Read-only by construction: one vector-store count, one vector-store scroll
//! of the window's writes and two graph aggregates, nothing else — no access
//! bump, no edge write. Each source
//! degrades on its own: a failed backend nulls its sections and adds an
//! error note, and never reads as zero.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use alaya_types::graph::{ContradictionStats, GraphStats, Verdict};

use crate::calendar::utc_date_str;
use crate::service::MemoryService;

const DAY_SECS: f64 = 86_400.0;

/// UTC days in the `judged_per_day` and `writes.novelty` series, today
/// included.
pub const STATS_WINDOW_DAYS: i64 = 14;

/// `writes.novelty` count keys, by the write's `nearest_similarity`:
/// three similarity bands, below the lowest, and none recorded (no live
/// neighbour, no search, or a memory stored before the field existed).
const NOVELTY_BUCKETS: [&str; 5] = ["0.95+", "0.85-0.95", "0.70-0.85", "below_0.70", "none"];

/// Key for an edge never judged (NULL verdict): the backlog.
const NEVER_JUDGED: &str = "never_judged";

impl MemoryService {
    /// The `GET /stats` document, minus the judge daily cap (process-local
    /// state the server adds). See `docs/rest-api.md` for every field.
    pub async fn corpus_stats(&self) -> Value {
        let now = (self.clock)();
        let today = (now / DAY_SECS).floor() as i64;
        let first_day = today - (STATS_WINDOW_DAYS - 1);
        let since = first_day as f64 * DAY_SECS;
        let (total, novelty, graph, contradictions) = futures::join!(
            self.vectors.count(),
            self.vectors.write_novelty(since),
            self.graph.get_stats(),
            self.graph.get_contradiction_stats(since),
        );

        let mut errors: Vec<String> = Vec::new();
        let mut section = |what: &str, r: alaya_types::Result<Value>| match r {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(section = what, error = %e, "stats source unavailable");
                errors.push(format!("{what}: {}", e.safe_message()));
                Value::Null
            }
        };
        let memories = section("vector store", total.map(|n| json!({ "total": n })));
        let writes = section(
            "write novelty",
            novelty.map(|rows| json!({ "novelty": novelty_per_day(&rows, first_day, today) })),
        );
        let graph = section("graph stats", graph.map(|g| graph_section(&g)));
        let contradictions = section(
            "contradiction stats",
            contradictions.map(|c| contradictions_section(&c, first_day, today)),
        );

        json!({
            "memories": memories,
            "graph": graph,
            "contradictions": contradictions,
            "writes": writes,
            "errors": errors,
        })
    }
}

/// Node count and every edge type, Hebbian included.
fn graph_section(g: &GraphStats) -> Value {
    let mut edges: BTreeMap<&str, usize> = g
        .typed_edge_counts
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();
    edges.insert("HEBBIAN", g.hebbian_edge_count);
    json!({ "node_count": g.node_count, "edge_counts": edges })
}

/// Map a raw edge verdict to its report key. A string `Verdict::parse`
/// rejects has none: only a direct graph write can produce one.
fn verdict_key(v: Option<&str>) -> Option<&'static str> {
    match v {
        None => Some(NEVER_JUDGED),
        Some(s) => Verdict::parse(s).map(|v| v.as_str()),
    }
}

/// A per-day series as `[{date, counts}]`, oldest first. A clock within 13
/// days of the epoch starts the window before it; those days have no date and
/// are left out.
fn dated(per_day: BTreeMap<i64, BTreeMap<&str, usize>>) -> Vec<Value> {
    per_day
        .into_iter()
        .filter_map(|(d, counts)| {
            let epoch_secs = u64::try_from(d).ok()?.checked_mul(86_400)?;
            Some(json!({ "date": utc_date_str(epoch_secs), "counts": counts }))
        })
        .collect()
}

fn novelty_bucket(nearest_similarity: Option<f64>) -> &'static str {
    match nearest_similarity {
        None => "none",
        Some(s) if s >= 0.95 => "0.95+",
        Some(s) if s >= 0.85 => "0.85-0.95",
        Some(s) if s >= 0.70 => "0.70-0.85",
        Some(_) => "below_0.70",
    }
}

/// Each UTC creation day's writes by novelty bucket, zero-filled. A re-store
/// recomputes a memory's value but keeps its creation day.
fn novelty_per_day(rows: &[(f64, Option<f64>)], first_day: i64, today: i64) -> Vec<Value> {
    let mut per_day: BTreeMap<i64, BTreeMap<&str, usize>> = (first_day..=today)
        .map(|d| (d, NOVELTY_BUCKETS.iter().map(|b| (*b, 0)).collect()))
        .collect();
    for (created_at, nearest) in rows {
        let day = (created_at / DAY_SECS).floor() as i64;
        if let Some(n) = per_day
            .get_mut(&day)
            .and_then(|c| c.get_mut(novelty_bucket(*nearest)))
        {
            *n += 1;
        }
    }
    dated(per_day)
}

fn contradictions_section(c: &ContradictionStats, first_day: i64, today: i64) -> Value {
    let mut by_verdict: BTreeMap<&str, (usize, usize)> = Verdict::ALL
        .iter()
        .map(|v| v.as_str())
        .chain([NEVER_JUDGED])
        .map(|k| (k, (0, 0)))
        .collect();
    for t in &c.verdicts {
        let Some(slot) = verdict_key(t.verdict.as_deref()).and_then(|k| by_verdict.get_mut(k))
        else {
            continue;
        };
        if t.resolved {
            slot.1 += t.count;
        } else {
            slot.0 += t.count;
        }
    }
    let unjudged = by_verdict[Verdict::UNJUDGED];
    let failures = unjudged.0 + unjudged.1;
    let top_total: usize = c.failures.iter().map(|f| f.count).sum();

    // Zero-filled so a quiet day reads as 0, not as a missing row. A row
    // outside the window (a replica clock a few seconds ahead at midnight)
    // has no slot and is left out.
    let mut per_day: BTreeMap<i64, BTreeMap<&str, usize>> = (first_day..=today)
        .map(|d| (d, Verdict::ALL.iter().map(|v| (v.as_str(), 0)).collect()))
        .collect();
    for t in &c.judged_per_day {
        if let Some(n) = per_day
            .get_mut(&t.day)
            .zip(verdict_key(t.verdict.as_deref()))
            .and_then(|(counts, k)| counts.get_mut(k))
        {
            *n += t.count;
        }
    }

    let mut degenerate: BTreeMap<&str, usize> =
        Verdict::CLASSES.iter().map(|v| (v.as_str(), 0)).collect();
    for d in &c.degenerate_reasons {
        if let Some(n) = degenerate.get_mut(d.verdict.as_str()) {
            *n += d.count;
        }
    }

    json!({
        "by_verdict": by_verdict
            .into_iter()
            .map(|(k, (open, resolved))| {
                (k.to_string(), json!({ "open": open, "resolved": resolved }))
            })
            .collect::<serde_json::Map<String, Value>>(),
        "failures": {
            "top": c.failures,
            // The two reads are not atomic, so a marker written between them
            // can put the top list above the total; floor at zero.
            "other": failures.saturating_sub(top_total),
        },
        "judged_per_day": dated(per_day),
        "degenerate_reasons": degenerate,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use async_trait::async_trait;

    use alaya_backends::traits::{
        ConsolidationService, EmbeddingProvider, GraphService, HebbianService, StoreMode,
        VectorStorage,
    };
    use alaya_types::{
        AlayaError, Result,
        graph::{
            CoAccessPair, Contradiction, ContradictionQuery, ContradictionRef, DayTally, Direction,
            Edge, EdgeMeta, EdgeVerdict, Neighbor, ReasonTally, Resolution, SystemRelationType,
            UserRelationType, VerdictCount, VerdictTally,
        },
        memory::{
            HealthStatus, Memory, MetadataUpdate, PatchMemoryRequest, ScoredMemory, ScrollResult,
        },
        search::{PayloadFilter, PromptName},
    };

    use super::*;

    /// 2026-10-01 12:00 UTC.
    const NOW: f64 = 1_790_856_000.0;
    const TODAY: i64 = 20_727;

    fn now() -> f64 {
        NOW
    }

    /// Every write any backend saw. AC-5: the endpoint must leave it at 0.
    type Writes = Rc<Cell<usize>>;

    fn bump(w: &Writes) {
        w.set(w.get() + 1);
    }

    struct Vectors {
        writes: Writes,
        fail: bool,
    }

    #[async_trait(?Send)]
    impl VectorStorage for Vectors {
        async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
            bump(&self.writes);
            Ok((true, String::new()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
        }
        async fn get_batch(&self, _h: &[&str]) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            bump(&self.writes);
            Ok(true)
        }
        async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
            bump(&self.writes);
            Ok(())
        }
        async fn update_metadata_batch(&self, _h: &[&str], _u: MetadataUpdate) -> Result<()> {
            bump(&self.writes);
            Ok(())
        }
        async fn reverse_supersession(
            &self,
            _h: &str,
            _e: &Value,
            _r: &alaya_backends::ReversalRecord,
        ) -> Result<alaya_backends::ReversalOutcome> {
            bump(&self.writes);
            Err(AlayaError::NotFound("mock".into()))
        }
        async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
            bump(&self.writes);
            Err(AlayaError::NotFound("mock".into()))
        }
        async fn set_generated_summary(
            &self,
            _h: &str,
            _s: &str,
            _e: Option<Vec<f32>>,
        ) -> Result<bool> {
            bump(&self.writes);
            Ok(false)
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            _l: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            _l: usize,
            _mt: Option<&str>,
            _mts: Option<f64>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _t: &[(&str, Vec<f32>)]) -> Result<()> {
            bump(&self.writes);
            Ok(())
        }
        async fn get_all(&self, _l: usize, _o: Option<&str>) -> Result<ScrollResult> {
            Ok(ScrollResult {
                memories: vec![],
                next_offset: None,
            })
        }
        async fn get_recent(
            &self,
            _l: usize,
            _s: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn count(&self) -> Result<usize> {
            if self.fail {
                return Err(AlayaError::Storage("qdrant down".into()));
            }
            Ok(1234)
        }
        async fn write_novelty(&self, since: f64) -> Result<Vec<(f64, Option<f64>)>> {
            if self.fail {
                return Err(AlayaError::Storage("qdrant down".into()));
            }
            assert_eq!(since, (TODAY - 13) as f64 * DAY_SECS);
            let today = TODAY as f64 * DAY_SECS;
            Ok(vec![
                (today + 1.0, Some(0.97)),
                (today + 2.0, Some(0.95)),
                (today + 3.0, Some(0.90)),
                (today + 4.0, Some(0.85)),
                (today + 5.0, Some(0.70)),
                (today + 6.0, Some(0.3)),
                (today + 7.0, None),
                (since, Some(0.5)),
                // A replica clock a few seconds ahead at midnight: no slot.
                (today + DAY_SECS, Some(0.99)),
            ])
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            bump(&self.writes);
            Ok(())
        }
        async fn increment_access_count_batch(&self, _h: &[&str]) -> Result<()> {
            bump(&self.writes);
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "ok".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    struct Embeddings;

    #[async_trait(?Send)]
    impl EmbeddingProvider for Embeddings {
        async fn embed_batch(&self, _t: &[&str], _p: PromptName) -> Result<Vec<Vec<f32>>> {
            unreachable!("stats embed nothing")
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        async fn health(&self) -> Result<HealthStatus> {
            unreachable!("stats probe nothing")
        }
    }

    /// `fail` models the bridge being unavailable: every read errors.
    struct Graph {
        writes: Writes,
        fail: bool,
        stats: ContradictionStats,
        since: Rc<Cell<f64>>,
    }

    impl Graph {
        fn down<T>(&self) -> Result<T> {
            Err(AlayaError::Graph("bridge unreachable".into()))
        }
    }

    #[async_trait(?Send)]
    impl GraphService for Graph {
        async fn ensure_node(&self, _h: &str, _t: f64) -> Result<()> {
            bump(&self.writes);
            Ok(())
        }
        async fn delete_node(&self, _h: &str) -> Result<()> {
            bump(&self.writes);
            Ok(())
        }
        async fn create_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
            _m: EdgeMeta,
        ) -> Result<bool> {
            bump(&self.writes);
            Ok(true)
        }
        async fn get_typed_edges(
            &self,
            _h: &str,
            _r: Option<UserRelationType>,
            _d: Direction,
            _l: usize,
        ) -> Result<Vec<Edge>> {
            Ok(vec![])
        }
        async fn delete_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
        ) -> Result<bool> {
            bump(&self.writes);
            Ok(true)
        }
        async fn create_system_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: SystemRelationType,
            _t: f64,
        ) -> Result<bool> {
            bump(&self.writes);
            Ok(true)
        }
        async fn create_typed_edges_batch(
            &self,
            _e: &[(String, String, UserRelationType, EdgeMeta)],
        ) -> Result<usize> {
            bump(&self.writes);
            Ok(0)
        }
        async fn create_system_edges_batch(
            &self,
            _e: &[(String, String, SystemRelationType, f64)],
        ) -> Result<usize> {
            bump(&self.writes);
            Ok(0)
        }
        async fn get_all_contradictions(
            &self,
            _q: &ContradictionQuery,
        ) -> Result<Vec<Contradiction>> {
            Ok(vec![])
        }
        async fn set_contradiction_verdict(
            &self,
            _s: &str,
            _d: &str,
            _v: &EdgeVerdict,
        ) -> Result<bool> {
            bump(&self.writes);
            Ok(true)
        }
        async fn set_contradiction_resolution(
            &self,
            _s: &str,
            _d: &str,
            _r: Option<Resolution>,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            bump(&self.writes);
            Ok(true)
        }
        async fn delete_incoming_system_edges(
            &self,
            _d: &str,
            _r: SystemRelationType,
        ) -> Result<Vec<String>> {
            bump(&self.writes);
            Ok(vec![])
        }
        async fn settle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            bump(&self.writes);
            Ok(true)
        }
        async fn unsettle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            bump(&self.writes);
            Ok(true)
        }
        async fn get_contradictions_for_hashes(
            &self,
            _h: &[&str],
        ) -> Result<HashMap<String, Vec<ContradictionRef>>> {
            Ok(HashMap::new())
        }
        async fn get_contradiction_stats(&self, since: f64) -> Result<ContradictionStats> {
            self.since.set(since);
            if self.fail {
                return self.down();
            }
            Ok(self.stats.clone())
        }
        async fn get_neighbors(
            &self,
            _h: &str,
            _m: u8,
            _w: f64,
            _l: usize,
        ) -> Result<Vec<Neighbor>> {
            Ok(vec![])
        }
        async fn spreading_activation(
            &self,
            _s: &[&str],
            _m: u8,
            _d: f64,
            _a: f64,
            _l: usize,
        ) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn hebbian_boosts_within(&self, _h: &[&str]) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn get_stats(&self) -> Result<GraphStats> {
            if self.fail {
                return self.down();
            }
            Ok(GraphStats {
                graph_name: "g".into(),
                node_count: 50,
                edge_count: 90,
                hebbian_edge_count: 7,
                typed_edge_counts: HashMap::from([
                    ("CONTRADICTS".into(), 60),
                    ("SUPERSEDES".into(), 20),
                    ("RELATES_TO".into(), 2),
                    ("PRECEDES".into(), 1),
                ]),
                status: "ok".into(),
            })
        }
    }

    struct Hebbian(Writes);

    #[async_trait(?Send)]
    impl HebbianService for Hebbian {
        async fn enqueue_strengthen(&self, _p: &[CoAccessPair]) -> Result<()> {
            bump(&self.0);
            Ok(())
        }
    }

    struct Consolidation(Writes);

    #[async_trait(?Send)]
    impl ConsolidationService for Consolidation {
        async fn decay_all_edges(&self, _d: f64, _l: usize) -> Result<usize> {
            bump(&self.0);
            Ok(0)
        }
        async fn decay_stale_edges(&self, _s: f64, _d: f64, _l: usize) -> Result<usize> {
            bump(&self.0);
            Ok(0)
        }
        async fn prune_weak_edges(&self, _t: f64, _l: usize) -> Result<usize> {
            bump(&self.0);
            Ok(0)
        }
        async fn get_orphan_nodes(&self, _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }

    fn tally(verdict: Option<&str>, resolved: bool, count: usize) -> VerdictTally {
        VerdictTally {
            verdict: verdict.map(str::to_string),
            resolved,
            count,
        }
    }

    fn sample() -> ContradictionStats {
        ContradictionStats {
            verdicts: vec![
                tally(Some("supersession"), false, 10),
                tally(Some("supersession"), true, 2),
                tally(Some("unjudged"), false, 13),
                tally(Some("unjudged"), true, 1),
                tally(None, false, 30),
                tally(Some("bogus"), false, 4),
            ],
            failures: vec![
                ReasonTally {
                    reason: "unjudged: empty response".into(),
                    count: 9,
                },
                ReasonTally {
                    reason: "unjudged: endpoint missing".into(),
                    count: 3,
                },
            ],
            judged_per_day: vec![
                DayTally {
                    day: TODAY,
                    verdict: Some("coexist".into()),
                    count: 5,
                },
                DayTally {
                    day: TODAY - 13,
                    verdict: Some("bogus".into()),
                    count: 1,
                },
                // Outside the window: no slot.
                DayTally {
                    day: TODAY + 1,
                    verdict: Some("coexist".into()),
                    count: 99,
                },
            ],
            degenerate_reasons: vec![VerdictCount {
                verdict: "supersession".into(),
                count: 2,
            }],
        }
    }

    fn service(vectors_fail: bool, graph_fail: bool) -> (MemoryService, Writes, Rc<Cell<f64>>) {
        let writes: Writes = Rc::default();
        let since: Rc<Cell<f64>> = Rc::default();
        let svc = MemoryService::with_clock(
            Box::new(Vectors {
                writes: writes.clone(),
                fail: vectors_fail,
            }),
            Box::new(Embeddings),
            Box::new(Graph {
                writes: writes.clone(),
                fail: graph_fail,
                stats: sample(),
                since: since.clone(),
            }),
            Box::new(Hebbian(writes.clone())),
            Box::new(Consolidation(writes.clone())),
            now,
        );
        (svc, writes, since)
    }

    #[tokio::test]
    async fn full_document_and_no_backend_write() {
        let (svc, writes, since) = service(false, false);
        let v = svc.corpus_stats().await;

        assert_eq!(writes.get(), 0, "GET /stats must write nothing (AC-5)");
        assert_eq!(since.get(), (TODAY - 13) as f64 * DAY_SECS);
        assert_eq!(v["errors"], json!([]));
        assert_eq!(v["memories"], json!({ "total": 1234 }));
        assert_eq!(
            v["graph"],
            json!({ "node_count": 50, "edge_counts": {
                "CONTRADICTS": 60, "SUPERSEDES": 20, "RELATES_TO": 2, "PRECEDES": 1, "HEBBIAN": 7,
            }})
        );

        let c = &v["contradictions"];
        let open_resolved = |o: usize, r: usize| json!({ "open": o, "resolved": r });
        assert_eq!(
            c["by_verdict"],
            json!({
                "contradiction": open_resolved(0, 0),
                "supersession": open_resolved(10, 2),
                "coexist": open_resolved(0, 0),
                "unrelated": open_resolved(0, 0),
                "unjudged": open_resolved(13, 1),
                "never_judged": open_resolved(30, 0),
            })
        );
        assert_eq!(c["failures"]["other"], json!(2));
        assert_eq!(
            c["failures"]["top"][0]["reason"],
            "unjudged: empty response"
        );

        let days = c["judged_per_day"].as_array().unwrap();
        assert_eq!(days.len(), STATS_WINDOW_DAYS as usize);
        assert_eq!(days[0]["date"], "2026-09-18");
        assert!(
            days[0]["counts"].get("bogus").is_none(),
            "an unknown verdict string has no slot"
        );
        assert_eq!(days[13]["date"], "2026-10-01");
        assert_eq!(
            days[13]["counts"]["coexist"], 5,
            "the out-of-window row is not folded in"
        );
        assert_eq!(
            days[6]["counts"]["coexist"], 0,
            "a quiet day is zero-filled"
        );

        assert_eq!(
            c["degenerate_reasons"],
            json!({ "contradiction": 0, "supersession": 2, "coexist": 0, "unrelated": 0 })
        );

        let novelty = v["writes"]["novelty"].as_array().unwrap();
        assert_eq!(novelty.len(), STATS_WINDOW_DAYS as usize);
        assert_eq!(novelty[0]["date"], "2026-09-18");
        let bands = |a: usize, b: usize, c: usize, d: usize, n: usize| json!({ "0.95+": a, "0.85-0.95": b, "0.70-0.85": c, "below_0.70": d, "none": n });
        assert_eq!(
            novelty[0]["counts"],
            bands(0, 0, 0, 1, 0),
            "window start is in"
        );
        assert_eq!(novelty[6]["counts"], bands(0, 0, 0, 0, 0), "zero-filled");
        assert_eq!(
            novelty[13]["counts"],
            bands(2, 2, 1, 1, 1),
            "lower band edges are inclusive; the out-of-window row is left out"
        );
    }

    /// Qdrant down nulls both vector sections with a note, never zeros, and
    /// keeps the graph's.
    #[tokio::test]
    async fn vector_store_unavailable_nulls_memories_and_writes() {
        let (svc, writes, _) = service(true, false);
        let v = svc.corpus_stats().await;

        assert_eq!(writes.get(), 0);
        assert!(v["memories"].is_null());
        assert!(v["writes"].is_null(), "never zeros: {v}");
        assert!(v["graph"].is_object());
        let errors = v["errors"].as_array().unwrap();
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(
            errors
                .iter()
                .any(|e| e.as_str().unwrap().starts_with("write novelty:"))
        );
    }

    /// A clock at the epoch (the wall clock's pre-1970 fallback) starts the
    /// window 13 days before it. Those days have no date and are left out.
    #[test]
    fn pre_epoch_window_days_are_left_out() {
        let c = contradictions_section(&sample(), -(STATS_WINDOW_DAYS - 1), 0);
        let days = c["judged_per_day"].as_array().unwrap();
        assert_eq!(days.len(), 1, "{days:?}");
        assert_eq!(days[0]["date"], "1970-01-01");
    }

    /// AC-4: the bridge down nulls the graph sections with a note, keeps the
    /// vector total, and writes nothing.
    #[tokio::test]
    async fn graph_unavailable_nulls_graph_sections_and_keeps_the_vector_total() {
        let (svc, writes, _) = service(false, true);
        let v = svc.corpus_stats().await;

        assert_eq!(writes.get(), 0);
        assert_eq!(v["memories"], json!({ "total": 1234 }));
        assert!(v["graph"].is_null());
        assert!(v["contradictions"].is_null(), "never zeros: {v}");
        let errors = v["errors"].as_array().unwrap();
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(
            errors
                .iter()
                .all(|e| e.as_str().unwrap().contains("Graph operation failed"))
        );
    }
}
