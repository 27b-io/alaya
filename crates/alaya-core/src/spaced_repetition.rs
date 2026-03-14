//! Spaced repetition scoring.
//!
//! Tracks access patterns and rewards well-spaced retrieval over cramming.

/// Compute spacing quality from access timestamps.
///
/// Returns 0.0–1.0 where 1.0 = evenly distributed access patterns.
/// Returns 0.0 if fewer than 2 accesses.
pub fn compute_spacing_quality(timestamps: &[f64]) -> f64 {
    if timestamps.len() < 2 {
        return 0.0;
    }

    let mut sorted = timestamps.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let intervals: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();

    if intervals.is_empty() {
        return 0.0;
    }

    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }

    // Coefficient of variation (lower = more uniform)
    let variance =
        intervals.iter().map(|i| (i - mean).powi(2)).sum::<f64>() / intervals.len() as f64;
    let cv = variance.sqrt() / mean;

    // Map CV to quality: CV=0 → quality=1.0, CV≥2 → quality≈0
    (1.0 - cv / 2.0).clamp(0.0, 1.0)
}

/// Apply spacing boost to a search score.
///
/// `boosted = base_score * (1 + boost_weight * spacing_quality)`
pub fn apply_spacing_boost(base_score: f64, spacing_quality: f64, boost_weight: f64) -> f64 {
    base_score * (1.0 + boost_weight * spacing_quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_timestamps() {
        assert_eq!(compute_spacing_quality(&[]), 0.0);
    }

    #[test]
    fn single_timestamp() {
        assert_eq!(compute_spacing_quality(&[100.0]), 0.0);
    }

    #[test]
    fn perfectly_spaced() {
        // Equal intervals → CV=0 → quality=1.0
        let ts = vec![0.0, 100.0, 200.0, 300.0];
        let q = compute_spacing_quality(&ts);
        assert!((q - 1.0).abs() < 1e-10);
    }

    #[test]
    fn clustered_access() {
        // Highly uneven: burst then long gap
        let ts = vec![0.0, 1.0, 2.0, 1000.0];
        let q = compute_spacing_quality(&ts);
        assert!(q < 0.5);
    }

    #[test]
    fn spacing_boost_identity() {
        let boosted = apply_spacing_boost(0.8, 0.7, 0.1);
        let expected = 0.8 * (1.0 + 0.1 * 0.7);
        assert!((boosted - expected).abs() < 1e-10);
    }
}
