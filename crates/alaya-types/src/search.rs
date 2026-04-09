use serde::{Deserialize, Serialize};

/// Filter criteria for vector storage queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PayloadFilter {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub tags_match_all: bool,
    #[serde(default)]
    pub exclude_superseded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_trust_score: Option<f64>,
}

/// Search mode — dispatches to different code paths in MemoryService.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    #[default]
    Hybrid,
    Scan,
    Similar,
    Tag,
    Recent,
}

impl SearchMode {
    /// Whether this mode requires embedding generation.
    pub fn needs_embedding(&self) -> bool {
        matches!(self, Self::Hybrid | Self::Similar)
    }
}

/// Prompt name for embedding generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptName {
    #[default]
    Query,
    Passage,
}

impl PromptName {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Passage => "passage",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_mode_hybrid_needs_embedding() {
        assert!(SearchMode::Hybrid.needs_embedding());
    }

    #[test]
    fn search_mode_scan_does_not_need_embedding() {
        assert!(!SearchMode::Scan.needs_embedding());
    }

    #[test]
    fn search_mode_tag_does_not_need_embedding() {
        assert!(!SearchMode::Tag.needs_embedding());
    }

    #[test]
    fn search_mode_deserializes_lowercase() {
        let mode: SearchMode = serde_json::from_str(r#""scan""#).unwrap();
        assert_eq!(mode, SearchMode::Scan);
    }

    #[test]
    fn prompt_name_as_str() {
        assert_eq!(PromptName::Passage.as_str(), "passage");
    }
}
