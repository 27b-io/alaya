//! Encoding context capture and similarity.
//!
//! Captures environmental context at storage time and computes similarity
//! between stored and query contexts for relevance boosting.

use std::collections::HashMap;

use serde_json::Value;

/// Capture encoding context from provided parameters.
pub fn capture_encoding_context(
    tags: &[String],
    agent: Option<&str>,
    timestamp: f64,
) -> HashMap<String, Value> {
    let mut ctx = HashMap::new();

    // Time of day classification
    let seconds_in_day = (timestamp % 86400.0) as u64;
    let hour = seconds_in_day / 3600;
    let time_of_day = match hour {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=20 => "evening",
        _ => "night",
    };
    ctx.insert("time_of_day".into(), Value::String(time_of_day.into()));

    // Day type (weekday vs weekend) — approximate from timestamp
    // Unix epoch (1970-01-01) was a Thursday (day 4)
    let days_since_epoch = (timestamp / 86400.0) as u64;
    let day_of_week = (days_since_epoch + 4) % 7; // 0=Sun, 6=Sat
    let day_type = if day_of_week == 0 || day_of_week == 6 {
        "weekend"
    } else {
        "weekday"
    };
    ctx.insert("day_type".into(), Value::String(day_type.into()));

    if let Some(a) = agent {
        ctx.insert("agent".into(), Value::String(a.into()));
    }

    if !tags.is_empty() {
        let tag_values: Vec<Value> = tags.iter().map(|t| Value::String(t.clone())).collect();
        ctx.insert("task_tags".into(), Value::Array(tag_values));
    }

    ctx
}

/// Compute similarity between two encoding contexts.
///
/// Returns 0.0–1.0 based on matching fields.
pub fn compute_context_similarity(
    stored: &HashMap<String, Value>,
    current: &HashMap<String, Value>,
) -> f64 {
    let fields = ["time_of_day", "day_type", "agent"];
    let mut matches = 0.0;
    let mut total = 0.0;

    for field in &fields {
        if let (Some(s), Some(c)) = (stored.get(*field), current.get(*field)) {
            total += 1.0;
            if s == c {
                matches += 1.0;
            }
        }
    }

    // Tag overlap (Jaccard)
    if let (Some(Value::Array(s_tags)), Some(Value::Array(c_tags))) =
        (stored.get("task_tags"), current.get("task_tags"))
    {
        total += 1.0;
        let s_set: std::collections::HashSet<&str> =
            s_tags.iter().filter_map(|v| v.as_str()).collect();
        let c_set: std::collections::HashSet<&str> =
            c_tags.iter().filter_map(|v| v.as_str()).collect();
        let intersection = s_set.intersection(&c_set).count() as f64;
        let union = s_set.union(&c_set).count() as f64;
        if union > 0.0 {
            matches += intersection / union;
        }
    }

    if total == 0.0 { 0.0 } else { matches / total }
}

/// Apply encoding context boost to a search score.
pub fn apply_context_boost(base_score: f64, context_similarity: f64, boost_weight: f64) -> f64 {
    base_score * (1.0 + boost_weight * context_similarity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_basic_context() {
        let ctx = capture_encoding_context(&["tag1".into()], Some("claude"), 1710432000.0);
        assert!(ctx.contains_key("time_of_day"));
        assert!(ctx.contains_key("day_type"));
        assert_eq!(ctx["agent"], "claude");
    }

    #[test]
    fn identical_contexts_score_one() {
        let ctx = capture_encoding_context(&["tag1".into()], Some("claude"), 1710432000.0);
        let sim = compute_context_similarity(&ctx, &ctx);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn empty_contexts_score_zero() {
        let a = HashMap::new();
        let b = HashMap::new();
        assert_eq!(compute_context_similarity(&a, &b), 0.0);
    }

    #[test]
    fn context_boost_identity() {
        let boosted = apply_context_boost(0.8, 0.5, 0.1);
        let expected = 0.8 * (1.0 + 0.1 * 0.5);
        assert!((boosted - expected).abs() < 1e-10);
    }
}
