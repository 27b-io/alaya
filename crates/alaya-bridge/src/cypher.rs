//! Typed Cypher query construction for all FalkorDB graph operations.
//!
//! Each function returns `(query_string, params_map, readonly)`.
//! Relationship labels come from compile-time enums — never from raw user input.

use std::collections::HashMap;

use alaya_types::graph::{
    ContradictionQuery, Direction, EdgeVerdict, SystemRelationType, UserRelationType, Verdict,
};
use serde_json::{Value, json};

/// A fully-constructed Cypher query ready to dispatch.
/// `(cypher, params, readonly)`
pub type CypherQuery = (String, HashMap<String, Value>, bool);

// ─── helpers ─────────────────────────────────────────────────────────────────

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

// ─── node operations ─────────────────────────────────────────────────────────

/// MERGE a Memory node, setting `created_at` only on creation.
pub fn ensure_node(hash: &str, ts: f64) -> CypherQuery {
    let q = "MERGE (m:Memory {content_hash: $hash}) \
             ON CREATE SET m.created_at = $ts"
        .to_string();
    (
        q,
        params(&[("hash", json!(hash)), ("ts", json!(ts))]),
        false,
    )
}

/// DETACH DELETE a Memory node by hash.
pub fn delete_node(hash: &str) -> CypherQuery {
    let q = "MATCH (m:Memory {content_hash: $hash}) DETACH DELETE m".to_string();
    (q, params(&[("hash", json!(hash))]), false)
}

// ─── typed edge operations ────────────────────────────────────────────────────

/// MERGE a typed edge between two memories.
///
/// `confidence` is optional — when `None` the `e.confidence` property is omitted.
pub fn create_typed_edge(
    src: &str,
    dst: &str,
    rel: UserRelationType,
    ts: f64,
    confidence: Option<f64>,
) -> CypherQuery {
    let label = rel.cypher_label();
    let q = if let Some(_conf) = confidence {
        format!(
            "MATCH (a:Memory {{content_hash: $src}}), (b:Memory {{content_hash: $dst}}) \
             MERGE (a)-[e:{label}]->(b) \
             ON CREATE SET e.created_at = $ts, e.confidence = $conf \
             RETURN count(e)"
        )
    } else {
        format!(
            "MATCH (a:Memory {{content_hash: $src}}), (b:Memory {{content_hash: $dst}}) \
             MERGE (a)-[e:{label}]->(b) \
             ON CREATE SET e.created_at = $ts \
             RETURN count(e)"
        )
    };

    let mut p = params(&[("src", json!(src)), ("dst", json!(dst)), ("ts", json!(ts))]);
    if let Some(conf) = confidence {
        p.insert("conf".to_string(), json!(conf));
    }
    (q, p, false)
}

/// Query typed edges for a hash in the given direction.
///
/// Returns one query per `UserRelationType` variant.  Caller unions the results.
pub fn get_typed_edges(
    hash: &str,
    rel: UserRelationType,
    direction: Direction,
    limit: u32,
) -> CypherQuery {
    let label = rel.cypher_label();
    let q = match direction {
        Direction::Outgoing => format!(
            "MATCH (a:Memory {{content_hash: $hash}})-[e:{label}]->(b:Memory) \
             RETURN a.content_hash, b.content_hash, e.created_at \
             LIMIT $lim"
        ),
        Direction::Incoming => format!(
            "MATCH (a:Memory)-[e:{label}]->(b:Memory {{content_hash: $hash}}) \
             RETURN a.content_hash, b.content_hash, e.created_at \
             LIMIT $lim"
        ),
        Direction::Both => format!(
            "MATCH (a:Memory {{content_hash: $hash}})-[e:{label}]-(b:Memory) \
             RETURN a.content_hash, b.content_hash, e.created_at \
             LIMIT $lim"
        ),
    };
    (
        q,
        params(&[("hash", json!(hash)), ("lim", json!(limit))]),
        true,
    )
}

/// MATCH edges of all user relation types in a single round-trip.
pub fn get_all_typed_edges(
    hash: &str,
    types: &[UserRelationType],
    direction: Direction,
    limit: u32,
) -> CypherQuery {
    let clauses: Vec<String> = types
        .iter()
        .map(|rel| {
            let label = rel.cypher_label();
            let pattern = match direction {
                Direction::Outgoing => {
                    format!("(a:Memory {{content_hash: $hash}})-[e:{label}]->(b:Memory)")
                }
                Direction::Incoming => {
                    format!("(a:Memory)-[e:{label}]->(b:Memory {{content_hash: $hash}})")
                }
                Direction::Both => {
                    format!("(a:Memory {{content_hash: $hash}})-[e:{label}]-(b:Memory)")
                }
            };
            format!(
                "MATCH {pattern} \
                 RETURN a.content_hash, b.content_hash, e.created_at, '{label}' AS rel_type \
                 LIMIT $lim"
            )
        })
        .collect();
    let q = clauses.join(" UNION ALL ");
    (
        q,
        params(&[("hash", json!(hash)), ("lim", json!(limit))]),
        true,
    )
}

/// DELETE a single typed edge between two memories.
pub fn delete_typed_edge(src: &str, dst: &str, rel: UserRelationType) -> CypherQuery {
    let label = rel.cypher_label();
    let q = format!(
        "MATCH (a:Memory {{content_hash: $src}})-[e:{label}]->(b:Memory {{content_hash: $dst}}) \
         DELETE e RETURN count(e)"
    );
    (
        q,
        params(&[("src", json!(src)), ("dst", json!(dst))]),
        false,
    )
}

// ─── system edge operations ───────────────────────────────────────────────────

/// MERGE a system-managed edge (e.g. SUPERSEDES).
pub fn create_system_edge(src: &str, dst: &str, rel: SystemRelationType, ts: f64) -> CypherQuery {
    let label = rel.cypher_label();
    let q = format!(
        "MATCH (a:Memory {{content_hash: $src}}), (b:Memory {{content_hash: $dst}}) \
         MERGE (a)-[e:{label}]->(b) \
         ON CREATE SET e.created_at = $ts \
         RETURN count(e)"
    );
    (
        q,
        params(&[("src", json!(src)), ("dst", json!(dst)), ("ts", json!(ts))]),
        false,
    )
}

// ─── contradiction operations ─────────────────────────────────────────────────

/// Row layout of `get_all_contradictions`, in this order.
/// `handlers::contradictions::parse_verdict` indexes columns 4.. by
/// position, so keep the two in lock-step. (`get_contradictions_for_hashes`
/// deliberately returns no verdict columns.)
const CONTRADICTION_COLUMNS: &str = "a.content_hash, b.content_hash, e.confidence, e.created_at, \
     e.verdict, e.verdict_survivor, e.verdict_reason, e.verdict_confidence, \
     e.verdict_model, e.judged_at";

/// One page of CONTRADICTS pairs ordered by `created_at DESC`, every filter
/// applied in Cypher (see `ContradictionQuery`), so `SKIP`/`LIMIT` page over
/// *matching* edges — app-side filtering over a LIMIT-only read starved the
/// queue (LAB-3283 review). WHERE fragments are constant strings; every
/// value travels as a parameter.
pub fn get_all_contradictions(q: &ContradictionQuery) -> CypherQuery {
    let mut clauses: Vec<&str> = Vec::new();
    let mut p = params(&[
        (
            "lim",
            json!(q.limit.clamp(1, ContradictionQuery::MAX_LIMIT)),
        ),
        ("skip", json!(q.skip)),
    ]);
    if let Some(v) = &q.verdicts {
        clauses.push(if v.iter().any(|s| s == Verdict::UNJUDGED) {
            // Absent verdict and the persisted failure marker both read as
            // "unjudged" on the read surface.
            "(e.verdict IS NULL OR e.verdict IN $verdicts)"
        } else {
            "e.verdict IN $verdicts"
        });
        p.insert("verdicts".to_string(), json!(v));
    }
    if q.selects_unjudged() {
        match &q.rejudge_model {
            // A persisted `unjudged` marker has a verdict, so it does NOT
            // match: a deterministic failure is skipped, not retried forever.
            None => clauses.push("e.verdict IS NULL"),
            Some(m) => {
                // Operator opt-in: everything not judged by the current model,
                // markers included (they carry the current model's id, so the
                // model test alone would never retry them).
                clauses.push(
                    "(e.verdict IS NULL OR e.verdict = $marker OR e.verdict_model IS NULL \
                     OR e.verdict_model <> $model)",
                );
                p.insert("model".to_string(), json!(m));
                p.insert("marker".to_string(), json!(Verdict::UNJUDGED));
            }
        }
    }
    if q.exclude_resolved {
        // Graph-side resolved state: `mark_superseded` writes
        // (new)-[:SUPERSEDES]->(old), so an endpoint with an incoming
        // SUPERSEDES edge is superseded. Measured complete against Qdrant
        // on 2026-09-10 (299 superseded memories, 0 without the edge).
        clauses.push("NOT (a)<-[:SUPERSEDES]-() AND NOT (b)<-[:SUPERSEDES]-()");
    }
    let filter = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {} ", clauses.join(" AND "))
    };
    // Tiebreak on the pair: every edge from one `store` shares `created_at`,
    // and FalkorDB orders ties differently per SKIP/LIMIT window, which
    // duplicated and dropped rows across pages (panel, 2026-09-10).
    let cypher = format!(
        "MATCH (a:Memory)-[e:CONTRADICTS]->(b:Memory) \
         {filter}\
         RETURN {CONTRADICTION_COLUMNS} \
         ORDER BY e.created_at DESC, a.content_hash, b.content_hash \
         SKIP $skip LIMIT $lim"
    );
    (cypher, p, true)
}

/// SET the judge's verdict on an existing `src -> dst` CONTRADICTS edge.
/// MATCH-only: never creates or deletes the edge. A `None` survivor is
/// written as `null`, which clears the property. An `unjudged` marker only
/// lands on an edge with no real verdict (NULL or an older marker): a failed
/// re-judge, or a store-path spawn racing a backfill, must never replace a
/// verdict with a failure.
pub fn set_contradiction_verdict(src: &str, dst: &str, v: &EdgeVerdict) -> CypherQuery {
    let guard = if v.verdict == Verdict::Unjudged {
        "WHERE e.verdict IS NULL OR e.verdict = $verdict "
    } else {
        ""
    };
    let q = format!(
        "MATCH (a:Memory {{content_hash: $src}})-[e:CONTRADICTS]->(b:Memory {{content_hash: $dst}}) \
         {guard}\
         SET e.verdict = $verdict, e.verdict_survivor = $survivor, \
             e.verdict_reason = $reason, e.verdict_confidence = $conf, \
             e.verdict_model = $model, e.judged_at = $ts \
         RETURN count(e)"
    );
    (
        q,
        params(&[
            ("src", json!(src)),
            ("dst", json!(dst)),
            ("verdict", json!(v.verdict.as_str())),
            ("survivor", json!(v.verdict_survivor)),
            ("reason", json!(v.verdict_reason)),
            ("conf", json!(v.verdict_confidence)),
            ("model", json!(v.verdict_model)),
            ("ts", json!(v.judged_at)),
        ]),
        false,
    )
}

/// Fetch CONTRADICTS pairs touching any of the supplied hashes.
pub fn get_contradictions_for_hashes(hashes: &[&str]) -> CypherQuery {
    let q = "MATCH (a:Memory)-[e:CONTRADICTS]->(b:Memory) \
             WHERE a.content_hash IN $hashes OR b.content_hash IN $hashes \
             RETURN a.content_hash, b.content_hash, e.confidence"
        .to_string();
    let hash_list: Vec<Value> = hashes.iter().map(|h| json!(h)).collect();
    (q, params(&[("hashes", Value::Array(hash_list))]), true)
}

// ─── Hebbian read operations ──────────────────────────────────────────────────

/// Walk HEBBIAN edges up to `max_hops` (capped at 3) from a source node.
///
/// When `max_hops == 1`, uses a simple single-edge match (FalkorDB binds
/// `e` as an Edge, not a List, for `*1..1` — which breaks `ALL(r IN e ...)`).
pub fn get_neighbors(hash: &str, max_hops: u8, min_weight: f64, limit: u32) -> CypherQuery {
    let hops = max_hops.clamp(1, 3);
    let q = if hops == 1 {
        "MATCH (src:Memory {content_hash: $hash})-[e:HEBBIAN]->(dst:Memory) \
         WHERE e.weight >= $min_w \
         RETURN DISTINCT dst.content_hash AS hash, e.weight AS path_weight, 1 AS hops \
         ORDER BY path_weight DESC \
         LIMIT $lim"
            .to_string()
    } else {
        format!(
            "MATCH (src:Memory {{content_hash: $hash}})-[e:HEBBIAN*1..{hops}]->(dst:Memory) \
             WITH dst, relationships(e) AS edges, length(e) AS hops \
             WHERE ALL(r IN edges WHERE r.weight >= $min_w) \
             RETURN DISTINCT dst.content_hash AS hash, \
                    reduce(w = 1.0, r IN edges | w * r.weight) AS path_weight, \
                    hops \
             ORDER BY path_weight DESC \
             LIMIT $lim"
        )
    };
    (
        q,
        params(&[
            ("hash", json!(hash)),
            ("min_w", json!(min_weight)),
            ("lim", json!(limit)),
        ]),
        true,
    )
}

/// Spreading activation from a set of seed hashes.
pub fn spreading_activation(seeds: &[&str], max_hops: u8) -> CypherQuery {
    let hops = max_hops.clamp(1, 3);
    let q = if hops == 1 {
        "MATCH (src:Memory)-[e:HEBBIAN]->(dst:Memory) \
         WHERE src.content_hash IN $seeds AND NOT dst.content_hash IN $seeds \
         RETURN dst.content_hash AS hash, e.weight AS path_weight, 1 AS hops"
            .to_string()
    } else {
        format!(
            "MATCH (src:Memory)-[e:HEBBIAN*1..{hops}]->(dst:Memory) \
             WHERE src.content_hash IN $seeds AND NOT dst.content_hash IN $seeds \
             WITH dst.content_hash AS hash, \
                  reduce(w = 1.0, r IN relationships(e) | w * r.weight) AS path_weight, \
                  length(e) AS hops \
             RETURN hash, path_weight, hops"
        )
    };
    let seed_list: Vec<Value> = seeds.iter().map(|s| json!(s)).collect();
    (q, params(&[("seeds", Value::Array(seed_list))]), true)
}

/// Maximum HEBBIAN weight for all edges within a set of hashes.
pub fn hebbian_boosts_within(hashes: &[&str]) -> CypherQuery {
    let q = "MATCH (a:Memory)-[e:HEBBIAN]->(b:Memory) \
             WHERE a.content_hash IN $hashes AND b.content_hash IN $hashes \
             RETURN a.content_hash AS hash, max(e.weight) AS max_weight"
        .to_string();
    let hash_list: Vec<Value> = hashes.iter().map(|h| json!(h)).collect();
    (q, params(&[("hashes", Value::Array(hash_list))]), true)
}

// ─── Hebbian write operations ─────────────────────────────────────────────────

/// MERGE a HEBBIAN edge and apply LTP update on match.
#[allow(clippy::too_many_arguments)]
pub fn strengthen_edge(
    src: &str,
    dst: &str,
    init_weight: f64,
    rate: f64,
    max_weight: f64,
    spacing_modifier: f64,
    ts: f64,
) -> CypherQuery {
    let q = "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) \
             MERGE (a)-[e:HEBBIAN]->(b) \
             ON CREATE SET e.weight = $init_w, e.co_access_count = 1, \
                           e.created_at = $ts, e.last_co_access = $ts \
             ON MATCH SET e.weight = toFloat(CASE \
               WHEN e.weight + $rate * (1.0 - e.weight / $max_w) * $sp_mod > $max_w THEN $max_w \
               ELSE e.weight + $rate * (1.0 - e.weight / $max_w) * $sp_mod END), \
               e.co_access_count = e.co_access_count + 1, \
               e.last_co_access = $ts"
        .to_string();
    (
        q,
        params(&[
            ("src", json!(src)),
            ("dst", json!(dst)),
            ("init_w", json!(init_weight)),
            ("rate", json!(rate)),
            ("max_w", json!(max_weight)),
            ("sp_mod", json!(spacing_modifier)),
            ("ts", json!(ts)),
        ]),
        false,
    )
}

// ─── consolidation operations ─────────────────────────────────────────────────

/// Decay all HEBBIAN edge weights by a factor (batch, limited).
pub fn decay_all_edges(decay: f64, limit: u32) -> CypherQuery {
    let q = "MATCH ()-[e:HEBBIAN]->() WITH e LIMIT $lim \
             SET e.weight = toFloat(e.weight * $decay) RETURN count(e)"
        .to_string();
    (
        q,
        params(&[("decay", json!(decay)), ("lim", json!(limit))]),
        false,
    )
}

/// Decay HEBBIAN edges that have not been co-accessed since `before_ts`.
pub fn decay_stale_edges(before_ts: f64, decay: f64, limit: u32) -> CypherQuery {
    let q = "MATCH ()-[e:HEBBIAN]->() WHERE e.last_co_access < $ts WITH e LIMIT $lim \
             SET e.weight = toFloat(e.weight * $decay) RETURN count(e)"
        .to_string();
    (
        q,
        params(&[
            ("ts", json!(before_ts)),
            ("decay", json!(decay)),
            ("lim", json!(limit)),
        ]),
        false,
    )
}

/// DELETE HEBBIAN edges whose weight has dropped below `threshold`.
pub fn prune_weak_edges(threshold: f64, limit: u32) -> CypherQuery {
    let q = "MATCH ()-[e:HEBBIAN]->() WHERE e.weight < $thresh WITH e LIMIT $lim \
             DELETE e RETURN count(e)"
        .to_string();
    (
        q,
        params(&[("thresh", json!(threshold)), ("lim", json!(limit))]),
        false,
    )
}

/// Find Memory nodes with no edges of any tracked type.
pub fn get_orphan_nodes(limit: u32) -> CypherQuery {
    let q = "MATCH (m:Memory) \
             WHERE NOT (m)-[:HEBBIAN]-() \
               AND NOT (m)-[:RELATES_TO]-() \
               AND NOT (m)-[:PRECEDES]-() \
               AND NOT (m)-[:CONTRADICTS]-() \
               AND NOT (m)-[:SUPERSEDES]-() \
             RETURN m.content_hash LIMIT $lim"
        .to_string();
    (q, params(&[("lim", json!(limit))]), true)
}

/// Return a query that fetches graph statistics from FalkorDB's internal
/// `db.meta.stats()` procedure. Returns rows of `(kind, cnt)`.
///
/// Reads from FalkorDB's maintained counts (no MATCH scans):
///   - "nodes" row carries `nodeCount`
///   - One row per non-empty relationship type carries its count
///
/// At 10K nodes / 37K edges this completes in ~2 ms; cost is independent of
/// graph size. Edge types with zero edges are omitted from the result — the
/// caller's `.unwrap_or(0)` for absent keys handles that.
pub fn get_graph_stats_union() -> CypherQuery {
    let q = "CALL db.meta.stats() YIELD nodeCount, relTypes \
             UNWIND ['nodes'] + keys(relTypes) AS kind \
             RETURN kind, \
             CASE WHEN kind = 'nodes' THEN nodeCount ELSE relTypes[kind] END AS cnt"
        .to_string();

    (q, HashMap::new(), true)
}

// ─── schema ───────────────────────────────────────────────────────────────────

// FalkorDB auto-creates range indexes on properties used in MATCH/WHERE filters.
// No explicit CREATE INDEX needed — removed Neo4j-style statements that FalkorDB
// rejects ("Invalid input 'I': expected '=', CREATE INDEX ON or CREATE INDEX FOR").

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // contradiction operations (LAB-3283)

    fn cq(limit: usize) -> ContradictionQuery {
        ContradictionQuery {
            limit,
            ..Default::default()
        }
    }

    #[test]
    fn get_all_contradictions_unfiltered_pages_with_skip_and_limit() {
        let (q, p, ro) = get_all_contradictions(&ContradictionQuery { skip: 40, ..cq(20) });
        assert!(!q.contains("WHERE"));
        assert!(q.contains("e.verdict, e.verdict_survivor"));
        assert!(q.ends_with(
            "ORDER BY e.created_at DESC, a.content_hash, b.content_hash SKIP $skip LIMIT $lim"
        ));
        assert_eq!(p["skip"], json!(40));
        assert_eq!(p["lim"], json!(20));
        assert!(!p.contains_key("verdicts"));
        assert!(ro);
    }

    #[test]
    fn get_all_contradictions_clamps_limit_to_page_cap() {
        let (_, p, _) = get_all_contradictions(&cq(0));
        assert_eq!(p["lim"], json!(1));
        let (_, p, _) = get_all_contradictions(&cq(10_000));
        assert_eq!(p["lim"], json!(ContradictionQuery::MAX_LIMIT));
    }

    #[test]
    fn get_all_contradictions_verdict_filter_is_parameterised() {
        let q = ContradictionQuery {
            verdicts: Some(vec!["contradiction".into(), "supersession".into()]),
            ..cq(20)
        };
        let (c, p, _) = get_all_contradictions(&q);
        assert!(c.contains("WHERE e.verdict IN $verdicts "));
        assert!(!c.contains("IS NULL"));
        assert!(
            !c.contains("contradiction"),
            "verdict text must not be interpolated"
        );
        assert_eq!(p["verdicts"], json!(["contradiction", "supersession"]));
    }

    #[test]
    fn get_all_contradictions_unjudged_sentinel_matches_null_and_marker() {
        let q = ContradictionQuery {
            verdicts: Some(vec!["unjudged".into()]),
            ..cq(20)
        };
        let (c, _, _) = get_all_contradictions(&q);
        assert!(c.contains("WHERE (e.verdict IS NULL OR e.verdict IN $verdicts) "));
    }

    #[test]
    fn get_all_contradictions_needs_judging_selects_null_only_unless_rejudging() {
        let (c, p, _) = get_all_contradictions(&ContradictionQuery {
            needs_judging: true,
            ..cq(20)
        });
        assert!(c.contains("WHERE e.verdict IS NULL "));
        assert!(!p.contains_key("model"));

        let (c, p, _) = get_all_contradictions(&ContradictionQuery {
            rejudge_model: Some("claude-sonnet-5".into()),
            ..cq(20)
        });
        assert!(c.contains(
            "WHERE (e.verdict IS NULL OR e.verdict = $marker OR e.verdict_model IS NULL \
             OR e.verdict_model <> $model) "
        ));
        assert!(!c.contains("sonnet"), "model id must not be interpolated");
        assert_eq!(p["model"], json!("claude-sonnet-5"));
        assert_eq!(p["marker"], json!("unjudged"));
    }

    #[test]
    fn get_all_contradictions_exclude_resolved_uses_supersedes_edges_and_ands_clauses() {
        let (c, _, _) = get_all_contradictions(&ContradictionQuery {
            exclude_resolved: true,
            verdicts: Some(vec!["coexist".into()]),
            ..cq(20)
        });
        assert!(c.contains(
            "WHERE e.verdict IN $verdicts AND NOT (a)<-[:SUPERSEDES]-() AND NOT (b)<-[:SUPERSEDES]-() "
        ));
    }

    #[test]
    fn set_contradiction_verdict_matches_never_merges() {
        let v = EdgeVerdict {
            verdict: Verdict::Supersession,
            verdict_survivor: Some("b".repeat(64)),
            verdict_reason: "it's newer".into(),
            verdict_confidence: 0.9,
            verdict_model: "m".into(),
            judged_at: 1.0,
        };
        let (q, p, ro) = set_contradiction_verdict(&"a".repeat(64), &"b".repeat(64), &v);
        assert!(q.starts_with("MATCH"));
        assert!(!q.contains("MERGE") && !q.contains("CREATE") && !q.contains("DELETE"));
        assert!(
            !q.contains("WHERE"),
            "a real verdict overwrites unconditionally"
        );
        assert!(q.contains("SET e.verdict = $verdict"));
        assert_eq!(p["verdict"], json!("supersession"));
        assert_eq!(p["survivor"], json!("b".repeat(64)));
        assert!(
            !q.contains("it's"),
            "reason must be a parameter, not interpolated"
        );
        assert!(!ro);
    }

    #[test]
    fn set_contradiction_verdict_null_survivor_clears_property() {
        let v = EdgeVerdict {
            verdict: Verdict::Coexist,
            verdict_survivor: None,
            verdict_reason: String::new(),
            verdict_confidence: 0.5,
            verdict_model: "m".into(),
            judged_at: 1.0,
        };
        let (_, p, _) = set_contradiction_verdict("a", "b", &v);
        assert_eq!(p["survivor"], Value::Null);
    }

    #[test]
    fn get_all_contradictions_orders_with_a_pair_tiebreak() {
        let (q, _, _) = get_all_contradictions(&cq(20));
        assert!(q.contains(
            "ORDER BY e.created_at DESC, a.content_hash, b.content_hash SKIP $skip LIMIT $lim"
        ));
    }

    #[test]
    fn set_contradiction_verdict_marker_never_clobbers_a_real_verdict() {
        let v = EdgeVerdict {
            verdict: Verdict::Unjudged,
            verdict_survivor: None,
            verdict_reason: "unjudged: boom".into(),
            verdict_confidence: 0.0,
            verdict_model: "m".into(),
            judged_at: 1.0,
        };
        let (q, p, _) = set_contradiction_verdict("a", "b", &v);
        assert!(q.contains("WHERE e.verdict IS NULL OR e.verdict = $verdict SET e.verdict"));
        assert_eq!(p["verdict"], json!("unjudged"));
    }

    // node operations

    #[test]
    fn ensure_node_shape() {
        let (q, p, ro) = ensure_node("abc123", 1_710_000_000.0);
        assert!(q.contains("MERGE"), "must MERGE");
        assert!(q.contains("Memory"), "must target Memory label");
        assert!(q.contains("ON CREATE SET"), "must set on create");
        assert!(p.contains_key("hash"));
        assert!(p.contains_key("ts"));
        assert!(!ro);
    }

    #[test]
    fn delete_node_shape() {
        let (q, p, ro) = delete_node("abc123");
        assert!(q.contains("DETACH DELETE"));
        assert!(p.contains_key("hash"));
        assert!(!ro);
    }

    // typed edge operations

    #[test]
    fn create_typed_edge_with_confidence() {
        let (q, p, ro) = create_typed_edge("a", "b", UserRelationType::RelatesTo, 1.0, Some(0.9));
        assert!(q.contains("RELATES_TO"));
        assert!(q.contains("e.confidence"));
        assert!(p.contains_key("conf"));
        assert!(p.contains_key("src"));
        assert!(p.contains_key("dst"));
        assert!(p.contains_key("ts"));
        assert!(!ro);
    }

    #[test]
    fn create_typed_edge_without_confidence() {
        let (q, p, ro) = create_typed_edge("a", "b", UserRelationType::Contradicts, 1.0, None);
        assert!(q.contains("CONTRADICTS"));
        assert!(
            !q.contains("e.confidence"),
            "must omit confidence when None"
        );
        assert!(!p.contains_key("conf"));
        assert!(!ro);
    }

    #[test]
    fn get_typed_edges_outgoing() {
        let (q, p, ro) = get_typed_edges("h", UserRelationType::Precedes, Direction::Outgoing, 100);
        assert!(q.contains("PRECEDES"));
        assert!(q.contains("LIMIT"));
        assert!(p.contains_key("hash"));
        assert!(p.contains_key("lim"));
        assert!(ro);
        // outgoing: hash node on left
        let hash_pos = q.find("content_hash: $hash").unwrap();
        let arrow_pos = q.find("->").unwrap();
        assert!(hash_pos < arrow_pos, "source should appear before arrow");
    }

    #[test]
    fn get_typed_edges_incoming() {
        let (q, p, ro) = get_typed_edges("h", UserRelationType::Precedes, Direction::Incoming, 50);
        assert!(ro);
        assert!(p.contains_key("hash"));
        // incoming: hash node on right
        let hash_pos = q.find("content_hash: $hash").unwrap();
        let arrow_pos = q.find("->").unwrap();
        assert!(hash_pos > arrow_pos, "target should appear after arrow");
    }

    #[test]
    fn delete_typed_edge_shape() {
        let (q, p, ro) = delete_typed_edge("s", "d", UserRelationType::RelatesTo);
        assert!(q.contains("DELETE e"));
        assert!(p.contains_key("src"));
        assert!(p.contains_key("dst"));
        assert!(!ro);
    }

    // system edge

    #[test]
    fn create_system_edge_shape() {
        let (q, p, ro) = create_system_edge("s", "d", SystemRelationType::Supersedes, 0.0);
        assert!(q.contains("SUPERSEDES"));
        assert!(p.contains_key("src"));
        assert!(p.contains_key("dst"));
        assert!(p.contains_key("ts"));
        assert!(!ro);
    }

    // contradiction operations

    #[test]
    fn get_all_contradictions_shape() {
        let (q, p, ro) = get_all_contradictions(&cq(20));
        assert!(q.contains("CONTRADICTS"));
        assert!(q.contains("ORDER BY e.created_at DESC"));
        assert!(p.contains_key("lim"));
        assert!(ro);
    }

    #[test]
    fn get_contradictions_for_hashes_shape() {
        let (q, p, ro) = get_contradictions_for_hashes(&["a", "b"]);
        assert!(q.contains("CONTRADICTS"));
        assert!(q.contains("IN $hashes"));
        assert!(p.contains_key("hashes"));
        let arr = p["hashes"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(ro);
    }

    // Hebbian reads

    #[test]
    fn get_neighbors_caps_hops() {
        let (q, p, ro) = get_neighbors("h", 10, 0.1, 25);
        // 10 capped to 3
        assert!(q.contains("HEBBIAN*1..3"), "hops must be capped at 3");
        assert!(p.contains_key("hash"));
        assert!(p.contains_key("min_w"));
        assert!(p.contains_key("lim"));
        assert!(ro);
    }

    #[test]
    fn get_neighbors_hops_not_capped_when_small() {
        let (q, _, _) = get_neighbors("h", 2, 0.1, 10);
        assert!(q.contains("HEBBIAN*1..2"));
    }

    #[test]
    fn spreading_activation_shape() {
        let (q, p, ro) = spreading_activation(&["s1", "s2"], 2);
        assert!(q.contains("HEBBIAN*1..2"));
        assert!(q.contains("NOT dst.content_hash IN $seeds"));
        assert!(p.contains_key("seeds"));
        assert!(ro);
    }

    #[test]
    fn hebbian_boosts_within_shape() {
        let (q, p, ro) = hebbian_boosts_within(&["a", "b", "c"]);
        assert!(q.contains("HEBBIAN"));
        assert!(q.contains("max(e.weight)"));
        assert!(p.contains_key("hashes"));
        assert!(ro);
    }

    // Hebbian writes

    #[test]
    fn strengthen_edge_shape() {
        let (q, p, ro) = strengthen_edge("s", "d", 0.3, 0.1, 1.0, 1.0, 1_710_000_000.0);
        assert!(q.contains("HEBBIAN"));
        assert!(q.contains("ON CREATE SET"));
        assert!(q.contains("ON MATCH SET"));
        assert!(q.contains("co_access_count"));
        assert!(p.contains_key("src"));
        assert!(p.contains_key("dst"));
        assert!(p.contains_key("init_w"));
        assert!(p.contains_key("rate"));
        assert!(p.contains_key("max_w"));
        assert!(p.contains_key("sp_mod"));
        assert!(p.contains_key("ts"));
        assert!(!ro);
    }

    // consolidation

    #[test]
    fn decay_all_edges_shape() {
        let (q, p, ro) = decay_all_edges(0.95, 1000);
        assert!(q.contains("HEBBIAN"));
        assert!(q.contains("e.weight * $decay"));
        assert!(p.contains_key("decay"));
        assert!(p.contains_key("lim"));
        assert!(!ro);
    }

    #[test]
    fn decay_stale_edges_shape() {
        let (q, p, ro) = decay_stale_edges(1_710_000_000.0, 0.9, 500);
        assert!(q.contains("last_co_access < $ts"));
        assert!(p.contains_key("ts"));
        assert!(p.contains_key("decay"));
        assert!(!ro);
    }

    #[test]
    fn prune_weak_edges_shape() {
        let (q, p, ro) = prune_weak_edges(0.05, 500);
        assert!(q.contains("e.weight < $thresh"));
        assert!(q.contains("DELETE e"));
        assert!(p.contains_key("thresh"));
        assert!(p.contains_key("lim"));
        assert!(!ro);
    }

    #[test]
    fn get_orphan_nodes_shape() {
        let (q, p, ro) = get_orphan_nodes(100);
        assert!(q.contains("NOT (m)-[:HEBBIAN]-()"));
        assert!(q.contains("NOT (m)-[:SUPERSEDES]-()"));
        assert!(p.contains_key("lim"));
        assert!(ro);
    }

    // graph stats

    #[test]
    fn get_graph_stats_union_shape() {
        let (q, p, ro) = get_graph_stats_union();
        // No params, readonly
        assert!(p.is_empty());
        assert!(ro);
        // Uses FalkorDB's maintained counts — no MATCH scans
        assert!(q.contains("db.meta.stats()"));
        assert!(!q.contains("MATCH"));
        // Returns kind + cnt columns
        assert!(q.contains("kind"));
        assert!(q.contains("AS cnt"));
    }

    // schema
}
