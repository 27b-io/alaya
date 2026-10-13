//! The `newer_related` pointer on search and `get_memory` results: the newer
//! memories the contradiction judge picked over this one. Read-time and
//! advisory. It names memories to read beside this one; it hides nothing,
//! changes no rank or score, and writes nothing.
//!
//! A survivor is listed only while the verdict can still be trusted to point
//! somewhere useful. So every condition below must hold:
//! - the edge's verdict is `supersession` at `verdict_confidence` >= the
//!   threshold, and names the other endpoint as its survivor;
//! - no edge between the two carries a resolution stamp, in either direction
//!   (an operator, or `memory_unsupersede`, settled the pair as keep-both);
//! - neither endpoint is superseded, by an incoming `SUPERSEDES` edge or by
//!   `metadata.superseded_by`;
//! - the survivor still exists and was created after the result.
//!
//! The first three mirror the contradiction queue's own resolved-state tests.

use std::collections::{HashMap, HashSet};

use alaya_types::graph::{ContradictionRef, Verdict};
use alaya_types::memory::Memory;
use serde_json::Value;

/// The result key. Absent when nothing qualifies, so a response with no
/// pointer is unchanged.
pub const KEY: &str = "newer_related";

/// Default floor on the judge's `verdict_confidence`.
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.90;

/// A formatted search or `get_memory` result, as far as the pointer needs it.
/// Every output mode carries these three fields.
pub(crate) struct Hit<'a> {
    pub hash: &'a str,
    pub created_at: f64,
    pub superseded: bool,
}

impl<'a> Hit<'a> {
    pub fn from_result(v: &'a Value) -> Option<Self> {
        Some(Self {
            hash: v.get("content_hash")?.as_str()?,
            created_at: v.get("created_at")?.as_f64()?,
            // Key presence, as `is_superseded` tests it.
            superseded: v
                .get("metadata")
                .and_then(Value::as_object)
                .is_some_and(|md| md.contains_key("superseded_by")),
        })
    }
}

/// The survivors the graph alone qualifies, each with its highest
/// qualifying `verdict_confidence`. `refs` are every `CONTRADICTS` edge
/// touching the result, from its side.
pub(crate) fn graph_candidates(refs: &[ContradictionRef], min: f64) -> HashMap<String, f64> {
    // One settled edge settles the pair, whichever way it points.
    let settled: HashSet<&str> = refs
        .iter()
        .filter(|r| r.resolved || r.endpoint_superseded)
        .map(|r| r.contradicts_hash.as_str())
        .collect();
    let mut best: HashMap<String, f64> = HashMap::new();
    for r in refs {
        let Some(v) = &r.verdict else { continue };
        let names_other = v.verdict_survivor.as_deref() == Some(r.contradicts_hash.as_str());
        if v.verdict != Verdict::Supersession
            || v.verdict_confidence < min
            || !names_other
            || settled.contains(r.contradicts_hash.as_str())
        {
            continue;
        }
        let c = best.entry(r.contradicts_hash.clone()).or_insert(f64::MIN);
        *c = c.max(v.verdict_confidence);
    }
    best
}

/// The pointer for `hit`: each candidate whose stored memory still exists,
/// is not superseded and is newer than `hit`, highest confidence first
/// (ties by hash, so the order is stable).
pub(crate) fn pointer(
    hit: &Hit,
    candidates: &HashMap<String, f64>,
    survivors: &HashMap<String, Memory>,
) -> Vec<(String, f64)> {
    if hit.superseded {
        return Vec::new();
    }
    let mut out: Vec<(String, f64)> = candidates
        .iter()
        .filter(|(hash, _)| {
            survivors
                .get(hash.as_str())
                .is_some_and(|m| !crate::service::is_superseded(m) && m.created_at > hit.created_at)
        })
        .map(|(hash, c)| (hash.clone(), *c))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// The wire form: `[{content_hash, verdict_confidence}]`.
pub(crate) fn to_value(pointer: &[(String, f64)]) -> Value {
    pointer
        .iter()
        .map(|(hash, c)| serde_json::json!({ "content_hash": hash, "verdict_confidence": c }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alaya_types::graph::EdgeVerdict;

    fn judged(other: &str, verdict: Verdict, survivor: &str, confidence: f64) -> ContradictionRef {
        ContradictionRef {
            contradicts_hash: other.into(),
            verdict: Some(EdgeVerdict {
                verdict,
                verdict_survivor: Some(survivor.into()),
                verdict_reason: "r".into(),
                verdict_confidence: confidence,
                verdict_model: "m".into(),
                judged_at: 1.0,
            }),
            ..Default::default()
        }
    }

    fn survivor(created_at: f64, superseded: bool) -> Memory {
        let mut m: Memory = serde_json::from_value(serde_json::json!({
            "content": "c", "content_hash": "s", "tags": [], "memory_type": "note",
            "created_at": created_at, "updated_at": created_at,
        }))
        .unwrap();
        if superseded {
            m.metadata = Some(HashMap::from([("superseded_by".into(), "x".into())]));
        }
        m
    }

    fn hit(created_at: f64, superseded: bool) -> Hit<'static> {
        Hit {
            hash: "h",
            created_at,
            superseded,
        }
    }

    #[test]
    fn a_confident_supersession_naming_the_other_endpoint_qualifies() {
        let c = graph_candidates(&[judged("s", Verdict::Supersession, "s", 0.95)], 0.9);
        assert_eq!(c, HashMap::from([("s".to_string(), 0.95)]));
        // The survivor's own side: the verdict names itself, not the other.
        assert!(graph_candidates(&[judged("h", Verdict::Supersession, "s", 0.95)], 0.9).is_empty());
    }

    #[test]
    fn below_the_threshold_or_another_verdict_does_not_qualify() {
        assert!(graph_candidates(&[judged("s", Verdict::Supersession, "s", 0.89)], 0.9).is_empty());
        // At the threshold is in.
        assert_eq!(
            graph_candidates(&[judged("s", Verdict::Supersession, "s", 0.9)], 0.9).len(),
            1
        );
        for v in [
            Verdict::Contradiction,
            Verdict::Coexist,
            Verdict::Unrelated,
            Verdict::Unjudged,
        ] {
            assert!(
                graph_candidates(&[judged("s", v, "s", 0.99)], 0.9).is_empty(),
                "{v:?}"
            );
        }
        let unjudged = ContradictionRef {
            contradicts_hash: "s".into(),
            ..Default::default()
        };
        assert!(graph_candidates(&[unjudged], 0.9).is_empty());
    }

    /// A stamp on either edge of the pair settles it: the reverse edge is
    /// unjudged here, but carries the keep-both stamp.
    #[test]
    fn a_resolution_stamp_on_either_edge_suppresses_the_pair() {
        let reverse_stamp = ContradictionRef {
            contradicts_hash: "s".into(),
            resolved: true,
            ..Default::default()
        };
        let refs = [judged("s", Verdict::Supersession, "s", 0.95), reverse_stamp];
        assert!(graph_candidates(&refs, 0.9).is_empty());
        let mut stamped = judged("s", Verdict::Supersession, "s", 0.95);
        stamped.resolved = true;
        assert!(graph_candidates(&[stamped], 0.9).is_empty());
    }

    #[test]
    fn a_superseded_endpoint_in_the_graph_suppresses_the_pair() {
        let mut r = judged("s", Verdict::Supersession, "s", 0.95);
        r.endpoint_superseded = true;
        assert!(graph_candidates(&[r], 0.9).is_empty());
    }

    #[test]
    fn two_edges_to_one_survivor_keep_the_higher_confidence() {
        let refs = [
            judged("s", Verdict::Supersession, "s", 0.91),
            judged("s", Verdict::Supersession, "s", 0.97),
        ];
        assert_eq!(graph_candidates(&refs, 0.9)["s"], 0.97);
    }

    #[test]
    fn pointer_lists_live_newer_survivors_highest_confidence_first() {
        let candidates = HashMap::from([
            ("s1".to_string(), 0.91),
            ("s2".to_string(), 0.99),
            ("s3".to_string(), 0.91),
        ]);
        let survivors = HashMap::from([
            ("s1".to_string(), survivor(20.0, false)),
            ("s2".to_string(), survivor(20.0, false)),
            ("s3".to_string(), survivor(20.0, false)),
        ]);
        assert_eq!(
            pointer(&hit(10.0, false), &candidates, &survivors),
            [
                ("s2".to_string(), 0.99),
                ("s1".to_string(), 0.91),
                ("s3".to_string(), 0.91)
            ]
        );
    }

    #[test]
    fn pointer_drops_deleted_superseded_and_older_survivors() {
        let candidates = HashMap::from([
            ("gone".to_string(), 0.95),
            ("superseded".to_string(), 0.95),
            ("older".to_string(), 0.95),
            ("same_age".to_string(), 0.95),
        ]);
        let survivors = HashMap::from([
            ("superseded".to_string(), survivor(20.0, true)),
            ("older".to_string(), survivor(5.0, false)),
            ("same_age".to_string(), survivor(10.0, false)),
        ]);
        assert!(pointer(&hit(10.0, false), &candidates, &survivors).is_empty());
    }

    #[test]
    fn a_superseded_result_gets_no_pointer() {
        let candidates = HashMap::from([("s".to_string(), 0.95)]);
        let survivors = HashMap::from([("s".to_string(), survivor(20.0, false))]);
        assert!(pointer(&hit(10.0, true), &candidates, &survivors).is_empty());
        assert_eq!(pointer(&hit(10.0, false), &candidates, &survivors).len(), 1);
    }

    #[test]
    fn hit_reads_hash_age_and_marker_from_a_formatted_result() {
        let v = serde_json::json!({
            "content_hash": "h", "created_at": 3.0, "metadata": {"superseded_by": null}
        });
        let h = Hit::from_result(&v).unwrap();
        assert_eq!((h.hash, h.created_at, h.superseded), ("h", 3.0, true));
        let live = serde_json::json!({ "content_hash": "h", "created_at": 3.0, "metadata": null });
        assert!(!Hit::from_result(&live).unwrap().superseded);
        assert!(Hit::from_result(&serde_json::json!({ "content_hash": "h" })).is_none());
    }

    #[test]
    fn wire_form_is_hash_and_confidence() {
        assert_eq!(
            to_value(&[("s".into(), 0.95)]),
            serde_json::json!([{ "content_hash": "s", "verdict_confidence": 0.95 }])
        );
    }
}
