use serde::{Deserialize, Serialize};

/// User-creatable relation types (exposed via MCP `relation` tool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UserRelationType {
    RelatesTo,
    Precedes,
    Contradicts,
}

impl UserRelationType {
    /// Cypher relationship label. Safe for interpolation (compile-time enum).
    pub fn cypher_label(&self) -> &'static str {
        match self {
            Self::RelatesTo => "RELATES_TO",
            Self::Precedes => "PRECEDES",
            Self::Contradicts => "CONTRADICTS",
        }
    }
}

/// System-managed relation type (created internally, not via MCP tool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SystemRelationType {
    Supersedes,
}

impl SystemRelationType {
    pub fn cypher_label(&self) -> &'static str {
        match self {
            Self::Supersedes => "SUPERSEDES",
        }
    }
}

/// Direction for edge queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

/// A typed edge between two memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub direction: Direction,
    pub created_at: Option<f64>,
    pub confidence: Option<f64>,
}

/// Metadata for edge creation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EdgeMeta {
    pub created_at: Option<f64>,
    pub confidence: Option<f64>,
}

/// A neighbor found via Hebbian traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbor {
    pub content_hash: String,
    pub weight: f64,
    pub hops: u32,
}

/// Machine verdict on a `CONTRADICTS` pair, produced by the contradiction
/// judge (LAB-3283). Wire form is lowercase (`"coexist"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Both claim to be current and cannot both be true.
    Contradiction,
    /// The newer memory replaces the older claim; a survivor is named.
    Supersession,
    /// Both are true (e.g. progress snapshots of the same work).
    Coexist,
    /// Shared vocabulary only; the pair is a detector false positive.
    Unrelated,
    /// No usable verdict. Persisted only for a *deterministic* judge
    /// failure (schema / parse / empty answer) so the backfill's NULL
    /// filter stops re-matching the pair; also the read-surface sentinel
    /// for an edge with no verdict at all. Never accepted from the model.
    Unjudged,
}

impl Verdict {
    /// The four classes the model may answer with.
    pub const CLASSES: [Verdict; 4] = [
        Verdict::Contradiction,
        Verdict::Supersession,
        Verdict::Coexist,
        Verdict::Unrelated,
    ];

    /// Every value an edge may carry, including the failure marker.
    pub const ALL: [Verdict; 5] = [
        Verdict::Contradiction,
        Verdict::Supersession,
        Verdict::Coexist,
        Verdict::Unrelated,
        Verdict::Unjudged,
    ];

    /// Wire/sentinel form of `Verdict::Unjudged`.
    pub const UNJUDGED: &'static str = "unjudged";

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Contradiction => "contradiction",
            Self::Supersession => "supersession",
            Self::Coexist => "coexist",
            Self::Unrelated => "unrelated",
            Self::Unjudged => Self::UNJUDGED,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.as_str() == s)
    }
}

/// Verdict fields as persisted on a `CONTRADICTS` edge. Field names are the
/// edge property names, so this flattens 1:1 onto the bridge wire shape.
/// Only `verdict` is required: a partially-written edge still reads as
/// judged (not silently re-queued for backfill).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeVerdict {
    pub verdict: Verdict,
    /// `content_hash` of the recommended survivor, when the verdict names one.
    #[serde(default)]
    pub verdict_survivor: Option<String>,
    #[serde(default)]
    pub verdict_reason: String,
    #[serde(default)]
    pub verdict_confidence: f64,
    #[serde(default)]
    pub verdict_model: String,
    #[serde(default)]
    pub judged_at: f64,
}

/// Operator resolution of a `CONTRADICTS` pair that leaves both memories in
/// place (LAB-3885). Lives as `e.resolution` / `e.resolved_at` /
/// `e.resolved_via` on the edge and is written by the dedicated resolution
/// verb only — never by `relation`, never by the judge — so no open write
/// path can retire a pair from the human queue. Wire form is snake_case
/// (`"keep_both"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resolution {
    /// Both memories are true and stay searchable; the pair leaves the
    /// default queue without a supersession. Reversed by clearing.
    KeepBoth,
}

impl Resolution {
    /// Every value an edge may carry.
    pub const ALL: [Resolution; 1] = [Resolution::KeepBoth];

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::KeepBoth => "keep_both",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.as_str() == s)
    }
}

/// Selection for `GraphService::get_all_contradictions` (bridge
/// `POST /contradictions/all`). Every filter is applied in Cypher, so a
/// page is a page of *matching* edges — the fix for the LAB-3283 review's
/// queue-starvation finding (app-side filtering over a LIMIT-only read).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContradictionQuery {
    /// Page size; every consumer clamps to `1..=MAX_LIMIT`.
    pub limit: usize,
    /// Edges to skip in `created_at DESC` order.
    #[serde(default)]
    pub skip: usize,
    /// Only edges whose verdict is in the list. `"unjudged"` matches both
    /// an absent verdict and the persisted failure marker. `None` = any.
    #[serde(default)]
    pub verdicts: Option<Vec<String>>,
    /// Drop pairs where either endpoint has an incoming `SUPERSEDES` edge —
    /// the graph-side resolved state `mark_superseded` writes.
    #[serde(default)]
    pub exclude_resolved: bool,
    /// Backfill selection: edges with no verdict at all (`NULL`; a persisted
    /// `unjudged` marker does NOT match, so a poison pair is skipped).
    #[serde(default)]
    pub needs_judging: bool,
    /// Re-judge path for a model switch (implies `needs_judging`): also
    /// select edges whose `verdict_model` differs from this one, and every
    /// `unjudged` marker (so an operator can retry deterministic failures).
    #[serde(default)]
    pub rejudge_model: Option<String>,
}

impl ContradictionQuery {
    /// Page cap, shared by the bridge (Cypher `LIMIT`) and every caller.
    /// The two MUST agree: a caller's "page was full" test is
    /// `pairs.len() == limit`.
    pub const MAX_LIMIT: usize = 500;

    /// Whether this query is a backfill selection (see `needs_judging`).
    pub fn selects_unjudged(&self) -> bool {
        self.needs_judging || self.rejudge_model.is_some()
    }
}

/// A contradiction pair from the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub memory_a_hash: String,
    pub memory_b_hash: String,
    pub confidence: Option<f64>,
    pub created_at: Option<f64>,
    /// `None` = unjudged. Flattened so the wire shape stays flat. Serde's
    /// flattened-`Option` semantics swallow ANY `EdgeVerdict` deserialize
    /// error into `None` — so every field but `verdict` must stay
    /// `#[serde(default)]`, or judged pairs silently read as unjudged.
    #[serde(default, flatten, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<EdgeVerdict>,
    /// Operator resolution (LAB-3885); `None` = unresolved. Plain fields,
    /// not a flattened struct: a flattened `Option` swallows every
    /// deserialize error into `None`, which would read a resolved pair as
    /// unresolved and put it back in the queue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<Resolution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_at: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_via: Option<String>,
}

#[cfg(test)]
mod verdict_wire_tests {
    use super::*;

    fn pair(extra: &str) -> Contradiction {
        let json = format!(
            r#"{{"memory_a_hash":"a","memory_b_hash":"b","confidence":0.7,"created_at":1.0{extra}}}"#
        );
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn flattened_verdict_round_trips_and_absent_reads_as_unjudged() {
        assert!(pair("").verdict.is_none());

        let judged = pair(
            r#","verdict":"supersession","verdict_survivor":"b","verdict_reason":"r","verdict_confidence":0.9,"verdict_model":"m","judged_at":2.0"#,
        );
        let v = judged.verdict.clone().expect("judged");
        assert_eq!(v.verdict, Verdict::Supersession);
        assert_eq!(v.verdict_survivor.as_deref(), Some("b"));
        let back: Contradiction =
            serde_json::from_str(&serde_json::to_string(&judged).unwrap()).unwrap();
        assert_eq!(back.verdict, judged.verdict);

        // Only `verdict` is required: a partially written edge still reads judged.
        assert!(pair(r#","verdict":"coexist""#).verdict.is_some());
        // A corrupt verdict string reads as unjudged (re-judged by backfill).
        assert!(
            pair(r#","verdict":"bogus","verdict_reason":"r""#)
                .verdict
                .is_none()
        );
    }

    #[test]
    fn resolution_fields_round_trip_beside_the_flattened_verdict() {
        let bare = pair("");
        assert_eq!(bare.resolution, None);
        assert!(!serde_json::to_string(&bare).unwrap().contains("resolution"));

        let kept = pair(
            r#","verdict":"coexist","resolution":"keep_both","resolved_at":3.0,"resolved_via":"operator:mcp""#,
        );
        assert_eq!(kept.resolution, Some(Resolution::KeepBoth));
        assert_eq!(kept.resolved_at, Some(3.0));
        assert_eq!(kept.resolved_via.as_deref(), Some("operator:mcp"));
        assert_eq!(
            kept.verdict.as_ref().map(|v| v.verdict),
            Some(Verdict::Coexist)
        );
        let back: Contradiction =
            serde_json::from_str(&serde_json::to_string(&kept).unwrap()).unwrap();
        assert_eq!(back.resolution, kept.resolution);
        assert_eq!(back.resolved_via, kept.resolved_via);
        assert_eq!(Resolution::parse("keep_both"), Some(Resolution::KeepBoth));
        assert_eq!(Resolution::parse("keep-both"), None);
    }
}

/// Reference to a contradicting memory (used in search enrichment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionRef {
    pub contradicts_hash: String,
    pub confidence: Option<f64>,
}

/// Hebbian co-access pair with spacing quality for adaptive LTP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoAccessPair {
    pub src: String,
    pub dst: String,
    pub spacing_quality: f64,
    pub timestamp: f64,
}

/// Graph statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub graph_name: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub hebbian_edge_count: usize,
    pub typed_edge_counts: std::collections::HashMap<String, usize>,
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_relation_type_serializes_screaming_snake() {
        let json = serde_json::to_string(&UserRelationType::RelatesTo).unwrap();
        assert_eq!(json, r#""RELATES_TO""#);
    }

    #[test]
    fn user_relation_type_deserializes() {
        let rel: UserRelationType = serde_json::from_str(r#""CONTRADICTS""#).unwrap();
        assert_eq!(rel, UserRelationType::Contradicts);
    }

    #[test]
    fn direction_serializes_lowercase() {
        let json = serde_json::to_string(&Direction::Both).unwrap();
        assert_eq!(json, r#""both""#);
    }

    #[test]
    fn co_access_pair_round_trip() {
        let pair = CoAccessPair {
            src: "abc".into(),
            dst: "def".into(),
            spacing_quality: 0.7,
            timestamp: 1710432000.0,
        };
        let json = serde_json::to_string(&pair).unwrap();
        let back: CoAccessPair = serde_json::from_str(&json).unwrap();
        assert_eq!(back.src, "abc");
        assert!((back.spacing_quality - 0.7).abs() < f64::EPSILON);
    }
}
