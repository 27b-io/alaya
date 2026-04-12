use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A stored memory with content, metadata, and optional embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub content: String,
    pub content_hash: String,
    pub tags: Vec<String>,
    pub memory_type: String,
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub created_at: f64,
    pub updated_at: f64,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub salience_score: f64,
    #[serde(default)]
    pub access_count: u64,
    #[serde(default)]
    pub access_timestamps: Vec<f64>,
    #[serde(default)]
    pub emotional_valence: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub encoding_context: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub provenance: Option<HashMap<String, serde_json::Value>>,
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

/// Mutable fields that can be patched on an existing memory.
///
/// All fields are optional — only provided fields are updated.
/// Used by `PATCH /memories/{content_hash}`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchMemoryRequest {
    /// Full replacement of tags array.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Merge into existing metadata. Keys with `null` values are deleted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Full replacement of summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Full replacement of memory_type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,
}

/// Valid memory types matching the MCP tool schema.
const VALID_MEMORY_TYPES: &[&str] = &["note", "decision", "task", "reference"];

/// Maximum number of tags per memory.
const MAX_TAGS: usize = 100;

/// Maximum length of a single tag.
const MAX_TAG_LEN: usize = 200;

/// Maximum number of metadata keys.
const MAX_METADATA_KEYS: usize = 50;

/// Maximum length of summary.
const MAX_SUMMARY_LEN: usize = 2000;

impl PatchMemoryRequest {
    /// Returns true if no fields are set (nothing to patch).
    pub fn is_empty(&self) -> bool {
        self.tags.is_none()
            && self.metadata.is_none()
            && self.summary.is_none()
            && self.memory_type.is_none()
    }

    /// Returns a comma-separated list of fields being patched (for logging).
    pub fn changed_fields(&self) -> String {
        let mut fields = Vec::new();
        if self.tags.is_some() {
            fields.push("tags");
        }
        if self.metadata.is_some() {
            fields.push("metadata");
        }
        if self.summary.is_some() {
            fields.push("summary");
        }
        if self.memory_type.is_some() {
            fields.push("memory_type");
        }
        fields.join(",")
    }

    /// Validate field sizes and values. Returns Err with a description on failure.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if let Some(ref tags) = self.tags {
            if tags.len() > MAX_TAGS {
                return Err(format!("tags: max {MAX_TAGS} tags, got {}", tags.len()));
            }
            for tag in tags {
                if tag.is_empty() {
                    return Err("tag must not be empty".into());
                }
                if tag.len() > MAX_TAG_LEN {
                    return Err(format!("tag too long: max {MAX_TAG_LEN} chars"));
                }
            }
        }
        if let Some(ref metadata) = self.metadata
            && metadata.len() > MAX_METADATA_KEYS
        {
            return Err(format!(
                "metadata: max {MAX_METADATA_KEYS} keys, got {}",
                metadata.len()
            ));
        }
        if let Some(ref summary) = self.summary
            && summary.len() > MAX_SUMMARY_LEN
        {
            return Err(format!(
                "summary: max {MAX_SUMMARY_LEN} chars, got {}",
                summary.len()
            ));
        }
        if let Some(ref mt) = self.memory_type
            && !VALID_MEMORY_TYPES.contains(&mt.as_str())
        {
            return Err(format!(
                "memory_type: must be one of {:?}, got {mt:?}",
                VALID_MEMORY_TYPES
            ));
        }
        Ok(())
    }
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

    // ─── PatchMemoryRequest tests ──────────────────────────────────────

    #[test]
    fn patch_is_empty_when_all_none() {
        let p = PatchMemoryRequest::default();
        assert!(p.is_empty());
    }

    #[test]
    fn patch_not_empty_with_tags() {
        let p = PatchMemoryRequest {
            tags: Some(vec!["a".into()]),
            ..Default::default()
        };
        assert!(!p.is_empty());
    }

    #[test]
    fn patch_not_empty_with_summary() {
        let p = PatchMemoryRequest {
            summary: Some("s".into()),
            ..Default::default()
        };
        assert!(!p.is_empty());
    }

    #[test]
    fn patch_not_empty_with_memory_type() {
        let p = PatchMemoryRequest {
            memory_type: Some("note".into()),
            ..Default::default()
        };
        assert!(!p.is_empty());
    }

    #[test]
    fn patch_not_empty_with_metadata() {
        let mut m = HashMap::new();
        m.insert("k".into(), serde_json::json!("v"));
        let p = PatchMemoryRequest {
            metadata: Some(m),
            ..Default::default()
        };
        assert!(!p.is_empty());
    }

    #[test]
    fn patch_deserialize_empty_object() {
        let p: PatchMemoryRequest = serde_json::from_str("{}").unwrap();
        assert!(p.is_empty());
    }

    #[test]
    fn patch_deserialize_partial() {
        let p: PatchMemoryRequest = serde_json::from_str(r#"{"tags": ["a", "b"]}"#).unwrap();
        assert!(!p.is_empty());
        assert_eq!(p.tags.unwrap().len(), 2);
        assert!(p.metadata.is_none());
        assert!(p.summary.is_none());
        assert!(p.memory_type.is_none());
    }

    #[test]
    fn patch_validate_ok() {
        let p = PatchMemoryRequest {
            tags: Some(vec!["tag1".into()]),
            memory_type: Some("decision".into()),
            summary: Some("short".into()),
            metadata: None,
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn patch_validate_rejects_invalid_memory_type() {
        let p = PatchMemoryRequest {
            memory_type: Some("banana".into()),
            ..Default::default()
        };
        assert!(p.validate().unwrap_err().contains("memory_type"));
    }

    #[test]
    fn patch_validate_rejects_too_many_tags() {
        let p = PatchMemoryRequest {
            tags: Some((0..101).map(|i| format!("tag{i}")).collect()),
            ..Default::default()
        };
        assert!(p.validate().unwrap_err().contains("max 100"));
    }

    #[test]
    fn patch_validate_rejects_long_tag() {
        let p = PatchMemoryRequest {
            tags: Some(vec!["x".repeat(201)]),
            ..Default::default()
        };
        assert!(p.validate().unwrap_err().contains("tag too long"));
    }

    #[test]
    fn patch_validate_rejects_empty_tag() {
        let p = PatchMemoryRequest {
            tags: Some(vec!["".into()]),
            ..Default::default()
        };
        assert!(p.validate().unwrap_err().contains("empty"));
    }

    #[test]
    fn patch_validate_rejects_too_many_metadata_keys() {
        let mut m = HashMap::new();
        for i in 0..51 {
            m.insert(format!("k{i}"), serde_json::json!("v"));
        }
        let p = PatchMemoryRequest {
            metadata: Some(m),
            ..Default::default()
        };
        assert!(p.validate().unwrap_err().contains("max 50"));
    }

    #[test]
    fn patch_validate_rejects_long_summary() {
        let p = PatchMemoryRequest {
            summary: Some("x".repeat(2001)),
            ..Default::default()
        };
        assert!(p.validate().unwrap_err().contains("summary"));
    }

    #[test]
    fn patch_validate_accepts_all_memory_types() {
        for mt in &["note", "decision", "task", "reference"] {
            let p = PatchMemoryRequest {
                memory_type: Some(mt.to_string()),
                ..Default::default()
            };
            assert!(p.validate().is_ok(), "rejected valid memory_type: {mt}");
        }
    }
}
