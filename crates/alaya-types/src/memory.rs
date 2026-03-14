use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A stored memory with content, metadata, and optional embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub content: String,
    pub content_hash: String,
    pub tags: Vec<String>,
    pub memory_type: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub created_at: f64,
    pub updated_at: f64,
    pub embedding: Option<Vec<f32>>,
    pub summary: Option<String>,
}

/// A memory with a similarity/relevance score from search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub score: f64,
}

/// Result of a scroll/pagination query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollResult {
    pub memories: Vec<Memory>,
    pub next_offset: Option<String>,
}

/// Metadata fields that can be updated on a memory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

/// Health status from a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<HashMap<String, serde_json::Value>>,
}

/// Content hash validation: must be 64-char lowercase hex (SHA-256).
pub fn validate_content_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_content_hash() {
        let hash = "a".repeat(64);
        assert!(validate_content_hash(&hash));
    }

    #[test]
    fn rejects_short_hash() {
        assert!(!validate_content_hash("abc123"));
    }

    #[test]
    fn rejects_uppercase_hash() {
        let hash = "A".repeat(64);
        assert!(!validate_content_hash(&hash));
    }

    #[test]
    fn rejects_non_hex() {
        let mut hash = "a".repeat(63);
        hash.push('g');
        assert!(!validate_content_hash(&hash));
    }
}
