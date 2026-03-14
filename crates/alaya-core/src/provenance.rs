//! Provenance tracking and trust scoring.

use std::collections::HashMap;

use serde_json::Value;

/// Default trust score when none is specified.
pub const DEFAULT_TRUST_SCORE: f64 = 0.5;

/// Known source trust scores. Higher = more trustworthy.
const SOURCE_TRUST: &[(&str, f64)] = &[
    ("api", 0.8),
    ("mcp", 0.9),
    ("cli", 0.7),
    ("import", 0.6),
    ("migration", 0.5),
    ("unknown", 0.5),
];

/// Look up trust score for a known source.
pub fn compute_trust_score(source: &str) -> f64 {
    SOURCE_TRUST
        .iter()
        .find(|(s, _)| *s == source)
        .map(|(_, score)| *score)
        .unwrap_or(DEFAULT_TRUST_SCORE)
}

/// Build a provenance record for a new memory.
pub fn build_provenance(
    source: Option<&str>,
    creation_method: Option<&str>,
    actor: Option<&str>,
    created_at: f64,
) -> HashMap<String, Value> {
    let src = source.unwrap_or("unknown");
    let mut prov = HashMap::new();
    prov.insert("source".into(), Value::String(src.into()));
    prov.insert(
        "creation_method".into(),
        Value::String(creation_method.unwrap_or("direct").into()),
    );
    if let Some(a) = actor {
        prov.insert("actor".into(), Value::String(a.into()));
    }
    prov.insert(
        "trust_score".into(),
        serde_json::json!(compute_trust_score(src)),
    );
    prov.insert("created_at".into(), serde_json::json!(created_at));
    prov
}

/// Extract trust score from a provenance record.
pub fn resolve_trust_score(provenance: &HashMap<String, Value>) -> f64 {
    provenance
        .get("trust_score")
        .and_then(|v| v.as_f64())
        .unwrap_or(DEFAULT_TRUST_SCORE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_source_scores() {
        assert_eq!(compute_trust_score("api"), 0.8);
        assert_eq!(compute_trust_score("mcp"), 0.9);
    }

    #[test]
    fn unknown_source_default() {
        assert_eq!(compute_trust_score("random"), DEFAULT_TRUST_SCORE);
    }

    #[test]
    fn build_provenance_complete() {
        let prov = build_provenance(Some("api"), Some("direct"), Some("claude"), 1000.0);
        assert_eq!(prov["source"], "api");
        assert_eq!(prov["trust_score"], 0.8);
        assert_eq!(prov["actor"], "claude");
    }

    #[test]
    fn resolve_trust_from_provenance() {
        let prov = build_provenance(Some("mcp"), None, None, 0.0);
        assert_eq!(resolve_trust_score(&prov), 0.9);
    }

    #[test]
    fn resolve_trust_missing() {
        let empty = HashMap::new();
        assert_eq!(resolve_trust_score(&empty), DEFAULT_TRUST_SCORE);
    }
}
