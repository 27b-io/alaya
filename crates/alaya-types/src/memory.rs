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
    /// Pre-computed embedding of the summary text, used for search boost.
    /// Stored in Qdrant payload, never exposed in API responses.
    #[serde(default, skip_serializing)]
    pub summary_embedding: Option<Vec<f32>>,
    /// The server-maintained `supersession_log` payload key: reversed
    /// supersessions, oldest first, in stored order; a stored value that is
    /// not an array comes back as one entry. Read-only — a store never
    /// writes it — and returned by `get_memory` alone.
    #[serde(default, skip_serializing)]
    pub supersession_log: Option<Vec<serde_json::Value>>,
    /// The `supersession_reason` payload key, as stored: why the memory was
    /// superseded. Written by supersede, removed by unsupersede, never by a
    /// store — and, like the log, returned by `get_memory` alone.
    #[serde(default, skip_serializing)]
    pub supersession_reason: Option<serde_json::Value>,
    /// The `nearest_similarity` payload key: how close this memory's nearest
    /// live neighbour was when it was last stored, `None` when the store-path
    /// search was skipped, failed or found nothing. Write-only: the store
    /// path sets it for its own write and reads never fill it, so a parsed
    /// copy cannot carry a stale value; serde never reads or writes it, so no
    /// caller input can carry one either.
    #[serde(skip)]
    pub nearest_similarity: Option<f64>,
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
    /// Pre-computed embedding of the summary text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_embedding: Option<Vec<f32>>,
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

/// Metadata keys only the server writes, each with why. `superseded_by` is
/// the supersession marker: a caller writing it would hide or un-hide a
/// memory with no SUPERSEDES edge, reason or audit entry, so only supersede
/// and merge set it. `nearest_similarity` is only reserved, never written
/// here: the server records the store's novelty in the root-level
/// `nearest_similarity` payload key, and a caller's metadata copy would pass
/// for it.
pub const RESERVED_METADATA_KEYS: &[(&str, &str)] = &[
    (
        "superseded_by",
        "supersession changes only through supersede or unsupersede",
    ),
    (
        "nearest_similarity",
        "the server records novelty in its own field on every store",
    ),
];

/// Refuse caller metadata carrying a reserved key, whatever its value — null
/// included, since a PATCH null deletes the key. The one check every
/// caller-facing write path (store, PATCH) runs.
pub fn reject_reserved_metadata<V>(
    metadata: &HashMap<String, V>,
) -> std::result::Result<(), String> {
    match RESERVED_METADATA_KEYS
        .iter()
        .find(|(k, _)| metadata.contains_key(*k))
    {
        Some((k, why)) => Err(format!("metadata.{k} is reserved: {why}")),
        None => Ok(()),
    }
}

impl PatchMemoryRequest {
    /// Returns true if no fields are set (nothing to patch).
    pub fn is_empty(&self) -> bool {
        self.tags.is_none()
            && self.metadata.is_none()
            && self.summary.is_none()
            && self.memory_type.is_none()
            && self.summary_embedding.is_none()
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
        if self.summary_embedding.is_some() {
            fields.push("summary_embedding");
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
        if let Some(ref metadata) = self.metadata {
            if metadata.len() > MAX_METADATA_KEYS {
                return Err(format!(
                    "metadata: max {MAX_METADATA_KEYS} keys, got {}",
                    metadata.len()
                ));
            }
            reject_reserved_metadata(metadata)?;
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
            summary_embedding: None,
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn reject_reserved_metadata_refuses_superseded_by_any_value() {
        for v in [serde_json::Value::Null, serde_json::json!("b".repeat(64))] {
            let md = HashMap::from([("superseded_by".to_string(), v)]);
            let err = reject_reserved_metadata(&md).unwrap_err();
            assert!(err.contains("metadata.superseded_by"), "{err}");
            assert!(err.contains("supersede"), "{err}");
        }
        let ok = HashMap::from([("importance".to_string(), serde_json::json!(0.7))]);
        assert!(reject_reserved_metadata(&ok).is_ok());
    }

    #[test]
    fn patch_validate_rejects_reserved_metadata_key() {
        // Null too: a PATCH null deletes the key, which would un-supersede.
        for v in [serde_json::Value::Null, serde_json::json!("b".repeat(64))] {
            let p = PatchMemoryRequest {
                metadata: Some(HashMap::from([("superseded_by".to_string(), v)])),
                ..Default::default()
            };
            assert!(p.validate().unwrap_err().contains("metadata.superseded_by"));
        }
    }

    #[test]
    fn reserved_metadata_refuses_nearest_similarity_on_store_and_patch() {
        for v in [serde_json::Value::Null, serde_json::json!(0.99)] {
            let md = HashMap::from([("nearest_similarity".to_string(), v)]);
            let err = reject_reserved_metadata(&md).unwrap_err();
            assert!(err.contains("metadata.nearest_similarity"), "{err}");
            let p = PatchMemoryRequest {
                metadata: Some(md),
                ..Default::default()
            };
            assert!(p.validate().unwrap_err().contains("nearest_similarity"));
        }
    }

    #[test]
    fn memory_deserialize_never_takes_nearest_similarity() {
        let m: Memory = serde_json::from_value(serde_json::json!({
            "content": "c",
            "content_hash": "a".repeat(64),
            "tags": [],
            "memory_type": "note",
            "created_at": 1.0,
            "updated_at": 1.0,
            "nearest_similarity": 0.99,
        }))
        .unwrap();
        assert_eq!(m.nearest_similarity, None);
        let out = serde_json::to_value(Memory {
            nearest_similarity: Some(0.5),
            ..m
        })
        .unwrap();
        assert!(out.get("nearest_similarity").is_none(), "{out}");
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
