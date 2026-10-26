//! Hybrid search — RRF fusion, adaptive alpha, recency decay.

use std::collections::{HashMap, HashSet};

/// Display score of a keyword-only (BM25) hit: no cosine similarity available.
/// GEPA-optimized: lowered from 0.1 to reduce keyword-only match influence.
const KEYWORD_ONLY_BASE_SCORE: f64 = 0.02;

/// RRF smoothing constant. Lower = sharper rank differentiation.
/// GEPA-optimized: lowered from 60 to 20 for better top-k separation.
pub const RRF_K: usize = 20;

// ─── Adaptive alpha ─────────────────────────────────────────────────────────

/// Compute adaptive alpha for hybrid search blending.
///
/// Alpha weights semantic (vector) vs keyword (BM25) search:
/// - Higher alpha = more weight on semantic
/// - Lower alpha = more weight on keywords
pub fn get_adaptive_alpha(corpus_size: usize) -> f64 {
    // GEPA-optimized: raised small-corpus alpha from 0.5 to 0.72
    // (more weight on vector similarity for typical memory corpora)
    if corpus_size < 500 {
        0.72
    } else if corpus_size < 5000 {
        0.7
    } else {
        0.8
    }
}

// ─── RRF (Reciprocal Rank Fusion) ───────────────────────────────────────────

/// Compute RRF score for a given rank.
pub fn rrf_score(rank: usize, k: usize) -> f64 {
    1.0 / (k + rank) as f64
}

/// Combine vector search results with keyword pools via Reciprocal Rank
/// Fusion.
///
/// Each input is `(content_hash, original_score)`, best first. The vector
/// pool weighs `alpha` and each keyword pool `1 - alpha` (hybrid search
/// passes one, BM25); an empty pool adds nothing, so passing one is the same
/// as leaving it out.
/// Returns `(content_hash, combined_rrf_score, display_score)` sorted descending.
pub fn combine_results_rrf(
    vector_results: &[(String, f64)],
    keyword_results: &[&[(String, f64)]],
    alpha: f64,
    k: usize,
) -> Vec<(String, f64, f64)> {
    let mut vector_ranks: HashMap<&str, (usize, f64)> = HashMap::new();
    for (rank, (hash, score)) in vector_results.iter().enumerate() {
        vector_ranks.insert(hash.as_str(), (rank + 1, *score));
    }

    // Each hash's summed keyword RRF across the pools it appears in.
    let mut keyword_rrf: HashMap<&str, f64> = HashMap::new();
    for pool in keyword_results {
        for (rank, (hash, _)) in pool.iter().enumerate() {
            *keyword_rrf.entry(hash.as_str()).or_default() += rrf_score(rank + 1, k);
        }
    }

    // Collect all unique hashes
    let mut all_hashes: HashSet<&str> = HashSet::new();
    for (hash, _) in vector_results {
        all_hashes.insert(hash.as_str());
    }
    all_hashes.extend(keyword_rrf.keys());

    let mut results: Vec<(String, f64, f64)> = all_hashes
        .into_iter()
        .map(|hash| {
            let v_rrf = vector_ranks
                .get(hash)
                .map(|(rank, _)| rrf_score(*rank, k))
                .unwrap_or(0.0);

            let kw_rrf = keyword_rrf.get(hash).copied().unwrap_or(0.0);

            let combined = alpha * v_rrf + (1.0 - alpha) * kw_rrf;

            // Display score: cosine from vector search, or base for keyword-only
            let display = vector_ranks
                .get(hash)
                .map(|(_, score)| *score)
                .unwrap_or(KEYWORD_ONLY_BASE_SCORE);

            (hash.to_string(), combined, display)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ─── Cosine similarity ─────────────────────────────────────────────────────

/// Cosine similarity between two vectors. Returns 0.0 on dimension mismatch or zero-norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
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
    fn adaptive_alpha_small_corpus() {
        assert_eq!(get_adaptive_alpha(100), 0.72);
    }

    #[test]
    fn adaptive_alpha_medium_corpus() {
        assert_eq!(get_adaptive_alpha(1000), 0.7);
    }

    #[test]
    fn adaptive_alpha_large_corpus() {
        assert_eq!(get_adaptive_alpha(10000), 0.8);
    }

    #[test]
    fn rrf_score_values() {
        assert!((rrf_score(1, 20) - 1.0 / 21.0).abs() < 1e-10);
        assert!((rrf_score(10, 20) - 1.0 / 30.0).abs() < 1e-10);
    }

    #[test]
    fn combine_rrf_vector_only() {
        let v = vec![("hash1".into(), 0.9), ("hash2".into(), 0.8)];
        let t: Vec<(String, f64)> = vec![];
        let results = combine_results_rrf(&v, &[&t], 0.7, 60);
        assert_eq!(results.len(), 2);
        // hash1 should be first (higher vector rank)
        assert_eq!(results[0].0, "hash1");
    }

    #[test]
    fn combine_rrf_overlap() {
        let v = vec![("hash1".into(), 0.9), ("hash2".into(), 0.7)];
        let t = vec![("hash2".into(), 1.0), ("hash3".into(), 1.0)];
        let results = combine_results_rrf(&v, &[&t], 0.5, 60);
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
        let results = combine_results_rrf(&v, &[&t], 0.5, 60);
        assert_eq!(results[0].2, KEYWORD_ONLY_BASE_SCORE);
    }

    #[test]
    fn combine_rrf_an_empty_keyword_pool_changes_nothing() {
        let v = vec![("hash1".into(), 0.9), ("hash2".into(), 0.7)];
        let t = vec![("hash2".into(), 1.0), ("hash3".into(), 1.0)];
        let none: Vec<(String, f64)> = vec![];
        assert_eq!(
            sorted(combine_results_rrf(&v, &[&t, &none], 0.8, 20)),
            sorted(combine_results_rrf(&v, &[&t], 0.8, 20))
        );
    }

    #[test]
    fn combine_rrf_sums_each_keyword_pool_at_one_minus_alpha() {
        let v = vec![("hash1".into(), 0.9)];
        let tags = vec![("hash2".into(), 1.0)];
        let bm25 = vec![("hash2".into(), 7.5), ("hash3".into(), 3.0)];
        let results = combine_results_rrf(&v, &[&tags, &bm25], 0.8, 20);
        let score = |h: &str| results.iter().find(|r| r.0 == h).unwrap();
        // In both keyword pools at rank 1: twice one pool's weight.
        assert!((score("hash2").1 - 0.2 * 2.0 * rrf_score(1, 20)).abs() < 1e-12);
        assert!((score("hash3").1 - 0.2 * rrf_score(2, 20)).abs() < 1e-12);
        assert!((score("hash1").1 - 0.8 * rrf_score(1, 20)).abs() < 1e-12);
        assert_eq!(score("hash3").2, KEYWORD_ONLY_BASE_SCORE);
        assert_eq!(results[0].0, "hash1");
    }

    fn sorted(mut r: Vec<(String, f64, f64)>) -> Vec<(String, f64, f64)> {
        r.sort_by(|a, b| a.0.cmp(&b.0));
        r
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

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_dimension_mismatch() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn cosine_zero_vector() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    }
}
