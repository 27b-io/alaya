use serde::{Deserialize, Serialize};

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
        matches!(self, Self::Hybrid | Self::Scan | Self::Similar)
    }
}

/// Prompt name for embedding generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
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
