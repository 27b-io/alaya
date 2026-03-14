//! Hybrid search — RRF fusion, adaptive alpha, keyword extraction, recency decay.

use std::collections::{HashMap, HashSet};

// ─── Stop words ─────────────────────────────────────────────────────────────

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "if", "while", "about", "up", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "this", "that", "these", "those", "am",
];

/// Tag-only base score (no cosine similarity available).
const TAG_ONLY_BASE_SCORE: f64 = 0.1;

/// Default RRF constant.
pub const RRF_K: usize = 60;

// ─── Keyword extraction ─────────────────────────────────────────────────────

/// Extract keywords from a query, optionally filtering to existing tags.
///
/// Tokenizes on non-alphanumeric, lowercases, removes stop words and short tokens,
/// and generates hyphenated compounds from adjacent tokens.
pub fn extract_query_keywords(query: &str, existing_tags: Option<&HashSet<String>>) -> Vec<String> {
    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    // Tokenize: split on non-alphanumeric
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= 2 && !stop.contains(s.as_str()))
        .collect();

    let mut keywords: Vec<String> = tokens.clone();

    // Generate hyphenated compounds from adjacent pairs
    for pair in tokens.windows(2) {
        keywords.push(format!("{}-{}", pair[0], pair[1]));
    }

    // Filter to existing tags if provided
    if let Some(tags) = existing_tags {
        keywords.retain(|k| tags.contains(k));
    }

    // Deduplicate preserving order
    let mut seen = HashSet::new();
    keywords.retain(|k| seen.insert(k.clone()));

    keywords
}

// ─── Adaptive alpha ─────────────────────────────────────────────────────────

/// Compute adaptive alpha for hybrid search blending.
///
/// Alpha weights semantic (vector) vs tag search:
/// - Higher alpha = more weight on semantic
/// - Lower alpha = more weight on tags
pub fn get_adaptive_alpha(corpus_size: usize, matching_tag_count: usize) -> f64 {
    let base_alpha = if corpus_size < 500 {
        0.5
    } else if corpus_size < 5000 {
        0.7
    } else {
        0.8
    };

    // If many tags match, boost tag weight
    if matching_tag_count >= 3 {
        (1.0_f64 - 1.5 * (1.0 - base_alpha)).clamp(0.0, 1.0)
    } else {
        base_alpha
    }
}

// ─── RRF (Reciprocal Rank Fusion) ───────────────────────────────────────────

/// Compute RRF score for a given rank.
pub fn rrf_score(rank: usize, k: usize) -> f64 {
    1.0 / (k + rank) as f64
}

/// Combine vector search and tag search results via Reciprocal Rank Fusion.
///
/// Each input is `(content_hash, original_score)`.
/// Returns `(content_hash, combined_rrf_score, display_score)` sorted descending.
pub fn combine_results_rrf(
    vector_results: &[(String, f64)],
    tag_results: &[(String, f64)],
    alpha: f64,
    k: usize,
) -> Vec<(String, f64, f64)> {
    let mut vector_ranks: HashMap<&str, (usize, f64)> = HashMap::new();
    for (rank, (hash, score)) in vector_results.iter().enumerate() {
        vector_ranks.insert(hash.as_str(), (rank + 1, *score));
    }

    let mut tag_ranks: HashMap<&str, usize> = HashMap::new();
    for (rank, (hash, _)) in tag_results.iter().enumerate() {
        tag_ranks.insert(hash.as_str(), rank + 1);
    }

    // Collect all unique hashes
    let mut all_hashes: HashSet<&str> = HashSet::new();
    for (hash, _) in vector_results {
        all_hashes.insert(hash.as_str());
    }
    for (hash, _) in tag_results {
        all_hashes.insert(hash.as_str());
    }

    let mut results: Vec<(String, f64, f64)> = all_hashes
        .into_iter()
        .map(|hash| {
            let v_rrf = vector_ranks
                .get(hash)
                .map(|(rank, _)| rrf_score(*rank, k))
                .unwrap_or(0.0);

            let t_rrf = tag_ranks
                .get(hash)
                .map(|rank| rrf_score(*rank, k))
                .unwrap_or(0.0);

            let combined = alpha * v_rrf + (1.0 - alpha) * t_rrf;

            // Display score: cosine from vector search, or base for tag-only
            let display = vector_ranks
                .get(hash)
                .map(|(_, score)| *score)
                .unwrap_or(TAG_ONLY_BASE_SCORE);

            (hash.to_string(), combined, display)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ─── Recency decay ──────────────────────────────────────────────────────────

/// Compute temporal decay factor.
///
/// `exp(-lambda * days_old)` — exponential decay over time.
pub fn temporal_decay_factor(days_old: f64, lambda: f64) -> f64 {
    (-lambda * days_old).exp()
}

/// Apply recency decay to a search score.
pub fn apply_recency_decay(score: f64, created_at: f64, now: f64, decay_rate: f64) -> f64 {
    let days_old = (now - created_at) / 86400.0;
    if days_old <= 0.0 {
        return score;
    }
    score * temporal_decay_factor(days_old, decay_rate)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_keywords_basic() {
        let kw = extract_query_keywords("How do I configure Rust logging?", None);
        assert!(kw.contains(&"configure".to_string()));
        assert!(kw.contains(&"rust".to_string()));
        assert!(kw.contains(&"logging".to_string()));
        // Stop words removed
        assert!(!kw.contains(&"how".to_string()));
        assert!(!kw.contains(&"do".to_string()));
    }

    #[test]
    fn extract_keywords_compounds() {
        let kw = extract_query_keywords("proton bridge setup", None);
        assert!(kw.contains(&"proton-bridge".to_string()));
        assert!(kw.contains(&"bridge-setup".to_string()));
    }

    #[test]
    fn extract_keywords_filter_existing() {
        let tags: HashSet<String> = ["rust", "python"].iter().map(|s| s.to_string()).collect();
        let kw = extract_query_keywords("I love Rust and Python programming", Some(&tags));
        assert!(kw.contains(&"rust".to_string()));
        assert!(kw.contains(&"python".to_string()));
        assert!(!kw.contains(&"programming".to_string()));
        assert!(!kw.contains(&"love".to_string()));
    }

    #[test]
    fn extract_keywords_deduplicated() {
        let kw = extract_query_keywords("rust rust rust", None);
        assert_eq!(kw.iter().filter(|k| *k == "rust").count(), 1);
    }

    #[test]
    fn adaptive_alpha_small_corpus() {
        assert_eq!(get_adaptive_alpha(100, 0), 0.5);
    }

    #[test]
    fn adaptive_alpha_medium_corpus() {
        assert_eq!(get_adaptive_alpha(1000, 0), 0.7);
    }

    #[test]
    fn adaptive_alpha_large_corpus() {
        assert_eq!(get_adaptive_alpha(10000, 0), 0.8);
    }

    #[test]
    fn adaptive_alpha_tag_boost() {
        let alpha = get_adaptive_alpha(100, 5);
        // base=0.5, boosted: 1.0 - 1.5*(1.0-0.5) = 0.25
        assert!((alpha - 0.25).abs() < 1e-10);
    }

    #[test]
    fn rrf_score_values() {
        assert!((rrf_score(1, 60) - 1.0 / 61.0).abs() < 1e-10);
        assert!((rrf_score(10, 60) - 1.0 / 70.0).abs() < 1e-10);
    }

    #[test]
    fn combine_rrf_vector_only() {
        let v = vec![("hash1".into(), 0.9), ("hash2".into(), 0.8)];
        let t: Vec<(String, f64)> = vec![];
        let results = combine_results_rrf(&v, &t, 0.7, 60);
        assert_eq!(results.len(), 2);
        // hash1 should be first (higher vector rank)
        assert_eq!(results[0].0, "hash1");
    }

    #[test]
    fn combine_rrf_overlap() {
        let v = vec![("hash1".into(), 0.9), ("hash2".into(), 0.7)];
        let t = vec![("hash2".into(), 1.0), ("hash3".into(), 1.0)];
        let results = combine_results_rrf(&v, &t, 0.5, 60);
        assert_eq!(results.len(), 3);
        // hash2 appears in both — should get boosted
        let hash2_score = results.iter().find(|r| r.0 == "hash2").unwrap().1;
        let hash1_score = results.iter().find(|r| r.0 == "hash1").unwrap().1;
        assert!(hash2_score > hash1_score);
    }

    #[test]
    fn combine_rrf_tag_only_display_score() {
        let v: Vec<(String, f64)> = vec![];
        let t = vec![("hash1".into(), 1.0)];
        let results = combine_results_rrf(&v, &t, 0.5, 60);
        assert_eq!(results[0].2, TAG_ONLY_BASE_SCORE);
    }

    #[test]
    fn temporal_decay_zero_days() {
        assert!((temporal_decay_factor(0.0, 0.01) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn temporal_decay_30_days() {
        let factor = temporal_decay_factor(30.0, 0.01);
        // exp(-0.3) ≈ 0.7408
        assert!((factor - 0.7408).abs() < 0.001);
    }

    #[test]
    fn recency_decay_preserves_recent() {
        let score = apply_recency_decay(1.0, 1000.0, 1000.0, 0.01);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn recency_decay_reduces_old() {
        let now = 1_000_000.0;
        let old = now - 86400.0 * 30.0; // 30 days ago
        let decayed = apply_recency_decay(1.0, old, now, 0.01);
        assert!(decayed < 1.0);
        assert!(decayed > 0.5);
    }
}
