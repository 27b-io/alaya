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
}

impl Verdict {
    pub const ALL: [Verdict; 4] = [
        Verdict::Contradiction,
        Verdict::Supersession,
        Verdict::Coexist,
        Verdict::Unrelated,
    ];

    /// Sentinel used on read surfaces for an edge with no verdict yet.
    pub const UNJUDGED: &'static str = "unjudged";

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Contradiction => "contradiction",
            Self::Supersession => "supersession",
            Self::Coexist => "coexist",
            Self::Unrelated => "unrelated",
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

/// A contradiction pair from the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub memory_a_hash: String,
    pub memory_b_hash: String,
    pub confidence: Option<f64>,
    pub created_at: Option<f64>,
    /// `None` = unjudged. Flattened so the wire shape stays flat.
    #[serde(default, flatten, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<EdgeVerdict>,
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
