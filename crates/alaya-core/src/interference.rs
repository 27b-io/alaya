use std::sync::LazyLock;

use regex::RegexSet;

const NEGATION_WORDS: &[&str] = &[
    "not",
    "no",
    "never",
    "without",
    "lack",
    "lacking",
    "false",
    "invalid",
    "incorrect",
    "wrong",
    "fail",
    "failed",
    "failure",
    "unable",
    "cannot",
    "can't",
    "won't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "shouldn't",
    "wouldn't",
    "couldn't",
];

const ANTONYM_PAIRS: &[(&str, &str)] = &[
    ("enable", "disable"),
    ("true", "false"),
    ("add", "remove"),
    ("success", "fail"),
    ("increase", "decrease"),
    ("start", "stop"),
    ("allow", "deny"),
    ("safe", "unsafe"),
    ("required", "optional"),
    ("sync", "async"),
];

const TEMPORAL_PATTERNS: &[&str] = &[
    r"\bno longer\b",
    r"\bstopped\b",
    r"\bswitched from\b",
    r"\breplaced by\b",
    r"\bmoved from\b",
    r"\bmigrated from\b",
    r"\bdeprecated\b",
    r"\bused to\b.*\bnow\b",
    r"\bpreviously\b.*\bnow\b",
    r"\bwas\b.*\bnow\b.*\binstead\b",
];

static TEMPORAL_REGEX_SET: LazyLock<RegexSet> =
    LazyLock::new(|| RegexSet::new(TEMPORAL_PATTERNS).expect("temporal patterns must compile"));

#[derive(Debug, Clone, PartialEq)]
pub enum SignalType {
    Negation,
    Antonym,
    Temporal,
}

#[derive(Debug, Clone)]
pub struct ContradictionSignal {
    pub existing_hash: String,
    pub similarity: f64,
    pub signal_type: SignalType,
    pub confidence: f64,
    pub detail: String,
}

/// Detect potential contradiction signals between new and existing memory content.
///
/// Returns signals with confidence > 0.5, combining negation asymmetry,
/// antonym pair detection, and temporal supersession checks.
pub fn detect_contradiction_signals(
    new_content: &str,
    existing_content: &str,
    existing_hash: &str,
    similarity: f64,
) -> Vec<ContradictionSignal> {
    let mut signals = Vec::new();

    if let Some((confidence, detail)) = check_negation_asymmetry(new_content, existing_content)
        && confidence > 0.5
    {
        signals.push(ContradictionSignal {
            existing_hash: existing_hash.to_string(),
            similarity,
            signal_type: SignalType::Negation,
            confidence,
            detail,
        });
    }

    if let Some((confidence, detail)) = check_antonym_pairs(new_content, existing_content)
        && confidence > 0.5
    {
        signals.push(ContradictionSignal {
            existing_hash: existing_hash.to_string(),
            similarity,
            signal_type: SignalType::Antonym,
            confidence,
            detail,
        });
    }

    if let Some((confidence, detail)) = check_temporal_supersession(new_content)
        && confidence > 0.5
    {
        signals.push(ContradictionSignal {
            existing_hash: existing_hash.to_string(),
            similarity,
            signal_type: SignalType::Temporal,
            confidence,
            detail,
        });
    }

    signals
}

/// Check for negation word asymmetry between new and existing content.
///
/// If one text has significantly more negation words (difference >= 2),
/// this suggests the new content may contradict the existing content.
pub fn check_negation_asymmetry(
    new_content: &str,
    existing_content: &str,
) -> Option<(f64, String)> {
    let new_lower = new_content.to_lowercase();
    let existing_lower = existing_content.to_lowercase();

    let new_count = count_negation_words(&new_lower);
    let existing_count = count_negation_words(&existing_lower);

    let diff = new_count.abs_diff(existing_count);
    if diff >= 2 {
        let detail = format!(
            "negation asymmetry: new has {new_count} negation words, existing has {existing_count} (diff={diff})"
        );
        Some((0.7, detail))
    } else {
        None
    }
}

/// Check if antonym pairs appear across new and existing content.
///
/// If one word of a pair appears in new content and the other in existing
/// content (but not both in both), this suggests potential contradiction.
/// Uses whole-word matching to avoid substring false positives (e.g. "safe" in "unsafe").
pub fn check_antonym_pairs(new_content: &str, existing_content: &str) -> Option<(f64, String)> {
    let new_lower = new_content.to_lowercase();
    let existing_lower = existing_content.to_lowercase();

    for &(word_a, word_b) in ANTONYM_PAIRS {
        let new_has_a = contains_word(&new_lower, word_a);
        let new_has_b = contains_word(&new_lower, word_b);
        let existing_has_a = contains_word(&existing_lower, word_a);
        let existing_has_b = contains_word(&existing_lower, word_b);

        // One word in new, the other in existing (cross-match only)
        if (new_has_a && existing_has_b && !new_has_b && !existing_has_a)
            || (new_has_b && existing_has_a && !new_has_a && !existing_has_b)
        {
            let detail = format!("antonym pair detected: '{word_a}' vs '{word_b}'");
            return Some((0.8, detail));
        }
    }

    None
}

/// Check if new content contains temporal supersession language.
///
/// Patterns like "no longer", "switched from", "used to X now Y" suggest
/// the new memory supersedes an older one.
pub fn check_temporal_supersession(new_content: &str) -> Option<(f64, String)> {
    let lower = new_content.to_lowercase();
    let matches: Vec<usize> = TEMPORAL_REGEX_SET.matches(&lower).into_iter().collect();

    if let Some(&first_idx) = matches.first() {
        let detail = format!(
            "temporal supersession pattern matched: '{}'",
            TEMPORAL_PATTERNS[first_idx]
        );
        Some((0.7, detail))
    } else {
        None
    }
}

/// Check if `text` contains `word` as a whole word (not as a substring of another word).
fn contains_word(text: &str, word: &str) -> bool {
    let bytes = text.as_bytes();
    let word_bytes = word.as_bytes();
    let word_len = word_bytes.len();

    for (i, window) in bytes.windows(word_len).enumerate() {
        if window != word_bytes {
            continue;
        }
        let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        let after_ok = i + word_len >= bytes.len() || !bytes[i + word_len].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

fn count_negation_words(text: &str) -> usize {
    NEGATION_WORDS
        .iter()
        .filter(|&&word| text.contains(word))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negation_asymmetry_detected() {
        let new = "This feature is not working, it failed and cannot be used";
        let existing = "This feature works great and is ready to use";
        let result = check_negation_asymmetry(new, existing);
        assert!(result.is_some());
        let (confidence, detail) = result.unwrap();
        assert!((confidence - 0.7).abs() < f64::EPSILON);
        assert!(detail.contains("negation asymmetry"));
    }

    #[test]
    fn negation_asymmetry_not_triggered_when_similar() {
        let new = "The system is not ready";
        let existing = "The system is not available";
        let result = check_negation_asymmetry(new, existing);
        assert!(result.is_none());
    }

    #[test]
    fn antonym_pair_detected() {
        let new = "We should disable the cache for this service";
        let existing = "We should enable the cache for this service";
        let result = check_antonym_pairs(new, existing);
        assert!(result.is_some());
        let (confidence, detail) = result.unwrap();
        assert!((confidence - 0.8).abs() < f64::EPSILON);
        assert!(detail.contains("enable"));
        assert!(detail.contains("disable"));
    }

    #[test]
    fn antonym_pair_not_triggered_when_both_present() {
        let new = "To enable or disable the feature, use the toggle";
        let existing = "You can enable or disable caching";
        let result = check_antonym_pairs(new, existing);
        assert!(result.is_none());
    }

    #[test]
    fn temporal_supersession_no_longer() {
        let result = check_temporal_supersession("We no longer use Redis for caching");
        assert!(result.is_some());
        let (confidence, _) = result.unwrap();
        assert!((confidence - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn temporal_supersession_used_to_now() {
        let result =
            check_temporal_supersession("We used to deploy on Heroku but now use Cloudflare");
        assert!(result.is_some());
    }

    #[test]
    fn temporal_supersession_previously_now() {
        let result =
            check_temporal_supersession("Previously we ran Python, now we run Rust instead");
        assert!(result.is_some());
    }

    #[test]
    fn temporal_supersession_deprecated() {
        let result = check_temporal_supersession("The old API has been deprecated");
        assert!(result.is_some());
    }

    #[test]
    fn temporal_no_match_on_normal_content() {
        let result = check_temporal_supersession("The service runs on port 8080 and handles JSON");
        assert!(result.is_none());
    }

    #[test]
    fn full_detection_real_contradiction() {
        let new = "Authentication is not required for the API, it failed the security review";
        let existing = "Authentication is required for all API endpoints";
        let signals = detect_contradiction_signals(new, existing, "abc123", 0.92);

        assert!(!signals.is_empty());
        assert!(
            signals
                .iter()
                .any(|s| s.signal_type == SignalType::Negation)
        );
        for signal in &signals {
            assert_eq!(signal.existing_hash, "abc123");
            assert!((signal.similarity - 0.92).abs() < f64::EPSILON);
            assert!(signal.confidence > 0.5);
        }
    }

    #[test]
    fn full_detection_temporal_contradiction() {
        let new = "We switched from PostgreSQL to FalkorDB for graph storage";
        let existing = "We use PostgreSQL for graph storage";
        let signals = detect_contradiction_signals(new, existing, "def456", 0.85);

        assert!(!signals.is_empty());
        assert!(
            signals
                .iter()
                .any(|s| s.signal_type == SignalType::Temporal)
        );
    }

    #[test]
    fn no_false_positive_on_similar_content() {
        let new = "The service handles JSON requests on port 8080";
        let existing = "The API processes JSON payloads on port 8080";
        let signals = detect_contradiction_signals(new, existing, "ghi789", 0.95);

        assert!(signals.is_empty());
    }

    #[test]
    fn antonym_start_stop() {
        let new = "We need to stop the deployment pipeline";
        let existing = "We need to start the deployment pipeline";
        let result = check_antonym_pairs(new, existing);
        assert!(result.is_some());
        let (_, detail) = result.unwrap();
        assert!(detail.contains("start"));
        assert!(detail.contains("stop"));
    }

    #[test]
    fn antonym_safe_unsafe() {
        let new = "This operation is unsafe without validation";
        let existing = "This operation is safe with proper input";
        let result = check_antonym_pairs(new, existing);
        assert!(result.is_some());
    }
}
