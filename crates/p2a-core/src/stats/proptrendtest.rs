//! Cochran-Armitage test for trend in proportions.
//!
//! Tests for a trend in binomial proportions across ordered groups.

use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of prop.trend.test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropTrendTestResult {
    /// Chi-squared test statistic
    pub statistic: f64,
    /// Degrees of freedom (always 1)
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Score weights used
    pub scores: Vec<f64>,
    /// Number of groups
    pub n_groups: usize,
    /// Method description
    pub method: String,
}

/// Test for trend in proportions (Cochran-Armitage test).
///
/// Tests for a linear trend in binomial proportions across ordered groups.
/// The test is based on the linear-by-linear association test.
///
/// # Arguments
///
/// * `x` - Number of successes in each group
/// * `n` - Number of trials in each group
/// * `scores` - Optional scores for the trend (defaults to 1, 2, 3, ...)
///
/// # Returns
///
/// A `PropTrendTestResult` with chi-squared statistic and p-value.
///
/// # Example
///
/// ```
/// use p2a_core::stats::proptrendtest::prop_trend_test;
///
/// // Test for trend in dose-response
/// let successes = vec![5, 15, 30, 45];
/// let trials = vec![50, 50, 50, 50];
///
/// let result = prop_trend_test(&successes, &trials, None).unwrap();
/// // Should show significant trend
/// ```
pub fn prop_trend_test(
    x: &[usize],
    n: &[usize],
    scores: Option<&[f64]>,
) -> EconResult<PropTrendTestResult> {
    let k = x.len();

    if k < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: k,
            context: "prop.trend.test requires at least 2 groups".to_string(),
        });
    }

    if n.len() != k {
        return Err(EconError::InvalidSpecification {
            message: "x and n must have the same length".to_string(),
        });
    }

    // Check that x <= n for each group
    for i in 0..k {
        if x[i] > n[i] {
            return Err(EconError::InvalidSpecification {
                message: format!("x[{}] = {} > n[{}] = {}", i, x[i], i, n[i]),
            });
        }
    }

    // Default scores: 1, 2, 3, ...
    let default_scores: Vec<f64> = (1..=k).map(|i| i as f64).collect();
    let scores = scores.unwrap_or(&default_scores);

    if scores.len() != k {
        return Err(EconError::InvalidSpecification {
            message: "scores must have same length as x".to_string(),
        });
    }

    // Total successes and trials
    let x_total: usize = x.iter().sum();
    let n_total: usize = n.iter().sum();

    if n_total == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Total sample size must be positive".to_string(),
        });
    }

    // Overall proportion
    let p_hat = x_total as f64 / n_total as f64;
    let q_hat = 1.0 - p_hat;

    // Weighted mean of scores
    let s_bar: f64 = scores
        .iter()
        .zip(n.iter())
        .map(|(s, &ni)| s * ni as f64)
        .sum::<f64>()
        / n_total as f64;

    // Numerator: sum_i x_i * (s_i - s_bar)
    let numerator: f64 = x
        .iter()
        .zip(scores.iter())
        .map(|(&xi, &si)| xi as f64 * (si - s_bar))
        .sum();

    // Denominator: sqrt(p_hat * q_hat * sum_i n_i * (s_i - s_bar)^2)
    let var_s: f64 = n
        .iter()
        .zip(scores.iter())
        .map(|(&ni, &si)| ni as f64 * (si - s_bar).powi(2))
        .sum();

    let denominator = (p_hat * q_hat * var_s).sqrt();

    if denominator < 1e-15 {
        return Err(EconError::Computation(
            "Degenerate case: variance is zero".to_string(),
        ));
    }

    // Chi-squared statistic (z^2)
    let z = numerator / denominator;
    let chi_sq = z * z;

    // P-value from chi-squared distribution with 1 df
    let chi_dist = ChiSquared::new(1.0)
        .map_err(|e| EconError::Computation(format!("Chi-squared distribution error: {}", e)))?;

    let p_value = 1.0 - chi_dist.cdf(chi_sq);

    Ok(PropTrendTestResult {
        statistic: chi_sq,
        df: 1,
        p_value,
        scores: scores.to_vec(),
        n_groups: k,
        method: "Chi-squared Test for Trend in Proportions".to_string(),
    })
}

/// Run prop.trend.test from vectors (MCP wrapper).
pub fn run_prop_trend_test(
    x: &[usize],
    n: &[usize],
    scores: Option<Vec<f64>>,
) -> EconResult<PropTrendTestResult> {
    prop_trend_test(x, n, scores.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prop_trend_basic() {
        // Clear increasing trend
        let x = vec![5, 10, 15, 20];
        let n = vec![50, 50, 50, 50];

        let result = prop_trend_test(&x, &n, None).unwrap();

        // Should be significant
        assert!(result.p_value < 0.05);
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_prop_trend_no_trend() {
        // No trend (flat)
        let x = vec![10, 10, 10, 10];
        let n = vec![50, 50, 50, 50];

        let result = prop_trend_test(&x, &n, None).unwrap();

        // Chi-squared should be 0
        assert!(result.statistic < 1e-10);
        assert!(result.p_value > 0.99);
    }

    #[test]
    fn test_prop_trend_custom_scores() {
        let x = vec![5, 10, 15];
        let n = vec![50, 50, 50];
        let scores = vec![1.0, 2.0, 4.0]; // Non-linear scores

        let result = prop_trend_test(&x, &n, Some(&scores)).unwrap();

        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_prop_trend_decreasing() {
        // Decreasing trend
        let x = vec![20, 15, 10, 5];
        let n = vec![50, 50, 50, 50];

        let result = prop_trend_test(&x, &n, None).unwrap();

        // Should still be significant (test is for any trend)
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_input_validation() {
        // x > n should fail
        let result = prop_trend_test(&[60], &[50], None);
        assert!(result.is_err());

        // Different lengths should fail
        let result = prop_trend_test(&[5, 10], &[50], None);
        assert!(result.is_err());

        // Single group should fail
        let result = prop_trend_test(&[5], &[50], None);
        assert!(result.is_err());
    }
}
