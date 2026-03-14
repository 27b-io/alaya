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

/// A contradiction pair from the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub memory_a_hash: String,
    pub memory_b_hash: String,
    pub confidence: Option<f64>,
    pub created_at: Option<f64>,
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
