//! Salience scoring and boosting.
//!
//! Salience represents how "important" a memory is, combining emotional weight,
//! access frequency, and explicit importance ratings.

/// Compute salience score from components.
///
/// Formula: `0.3 * emotional + 0.3 * log_frequency + 0.4 * importance`, clamped [0, 1].
///
/// - `emotional_magnitude`: 0.0–1.0 (0.0 in v1, deferred)
/// - `access_count`: number of times accessed
/// - `explicit_importance`: 0.0–1.0 from user metadata
pub fn compute_salience(
    emotional_magnitude: f64,
    access_count: u64,
    explicit_importance: f64,
) -> f64 {
    const EMOTIONAL_WEIGHT: f64 = 0.3;
    const FREQUENCY_WEIGHT: f64 = 0.3;
    const IMPORTANCE_WEIGHT: f64 = 0.4;

    let log_frequency = (1.0 + access_count as f64).ln() / (101.0_f64).ln();
    let raw = EMOTIONAL_WEIGHT * emotional_magnitude
        + FREQUENCY_WEIGHT * log_frequency
        + IMPORTANCE_WEIGHT * explicit_importance;

    raw.clamp(0.0, 1.0)
}

/// Apply salience boost to a search score.
///
/// `boosted = base_score * (1 + boost_weight * salience)`
pub fn apply_salience_boost(base_score: f64, salience: f64, boost_weight: f64) -> f64 {
    base_score * (1.0 + boost_weight * salience)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_everything_yields_zero() {
        assert_eq!(compute_salience(0.0, 0, 0.0), 0.0);
    }

    #[test]
    fn max_importance_caps_at_one() {
        let s = compute_salience(1.0, 100, 1.0);
        assert!(s <= 1.0);
    }

    #[test]
    fn importance_dominates() {
        let s = compute_salience(0.0, 0, 1.0);
        assert!((s - 0.4).abs() < 0.001);
    }

    #[test]
    fn access_count_boosts() {
        let s0 = compute_salience(0.0, 0, 0.0);
        let s10 = compute_salience(0.0, 10, 0.0);
        assert!(s10 > s0);
    }

    #[test]
    fn salience_boost_identity() {
        let boosted = apply_salience_boost(0.8, 0.5, 0.15);
        let expected = 0.8 * (1.0 + 0.15 * 0.5);
        assert!((boosted - expected).abs() < 1e-10);
    }

    #[test]
    fn salience_boost_zero_salience() {
        let boosted = apply_salience_boost(0.8, 0.0, 0.15);
        assert!((boosted - 0.8).abs() < 1e-10);
    }
}
