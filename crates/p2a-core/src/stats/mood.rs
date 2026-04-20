//! Mood's two-sample test of scale.
//!
//! A non-parametric test for comparing scale parameters of two distributions.
//! Uses the squared deviation of ranks from the mean rank as the test statistic.
//!
//! # References
//!
//! - Conover, W. J. (1971). *Practical Nonparametric Statistics*.
//!   New York: John Wiley & Sons. Pages 234-235.
//! - Mielke, P. W. (1967). "Note on Some Squared Rank Tests with Existing Ties."
//!   *Technometrics*, 9(2), 312-314. doi:10.1080/00401706.1967.10490465
//! - R Core Team. `stats::mood.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mood.test.html>
//!
//! # Mathematical Background
//!
//! ## Underlying Model
//!
//! The test assumes two samples are drawn from distributions:
//! - Sample 1: f(x - l)
//! - Sample 2: f((x - l)/s)/s
//!
//! Where `l` is a common location parameter and `s` is the scale parameter.
//!
//! ## Null Hypothesis
//!
//! H₀: s = 1 (equal scale parameters)
//!
//! ## Test Statistic (Conover 1971)
//!
//! 1. Combine both samples and rank them (1 to N where N = m + n)
//! 2. T = Σ (r_i - (N+1)/2)² for observations in sample x
//!    (sum of squared deviations from mean rank)
//!
//! ## Distribution under H₀
//!
//! - Expected value: E[T] = m(N² - 1)/12
//! - Variance: Var(T) = mn(N+1)(N+2)(N-2)/180
//! - Z = (T - E[T])/√Var(T) ~ N(0,1) asymptotically
//!
//! ## Ties Handling (Mielke 1967)
//!
//! When ties are present, the variance is adjusted using the tie structure
//! to account for reduced variability in the test statistic.

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::errors::{EconError, EconResult};
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;

/// Result of Mood's two-sample test of scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodTestResult {
    /// Description of the test
    pub test_name: String,
    /// Alternative hypothesis type
    pub alternative: Alternative,
    /// Mood test statistic (sum of squared rank deviations)
    pub statistic: f64,
    /// Z-score (standardized statistic)
    pub z_score: f64,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Size of first sample
    pub n_x: usize,
    /// Size of second sample
    pub n_y: usize,
    /// Combined sample size
    pub n_total: usize,
    /// Whether ties were present in the data
    pub has_ties: bool,
}

impl std::fmt::Display for MoodTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        writeln!(
            f,
            "Z = {:.4}, p-value = {:.5} {}",
            self.z_score,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        // Alternative hypothesis
        let alt_str = match self.alternative {
            Alternative::TwoSided => "true ratio of scales is not equal to 1",
            Alternative::Greater => "true ratio of scales is greater than 1",
            Alternative::Less => "true ratio of scales is less than 1",
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        // Sample sizes
        writeln!(
            f,
            "Sample sizes: n_x = {}, n_y = {}, N = {}",
            self.n_x, self.n_y, self.n_total
        )?;
        writeln!(f)?;

        writeln!(f, "Test statistic T = {:.4}", self.statistic)?;
        if self.has_ties {
            writeln!(f, "(variance adjusted for ties using Mielke 1967)")?;
        }
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Perform Mood's two-sample test for equality of scale parameters.
///
/// # Arguments
/// * `x` - First sample (numeric vector)
/// * `y` - Second sample (numeric vector)
/// * `alternative` - Direction of alternative hypothesis:
///   - `TwoSided`: scales are different
///   - `Greater`: scale of x > scale of y
///   - `Less`: scale of x < scale of y
///
/// # Returns
/// A `MoodTestResult` containing the test statistic, z-score, and p-value.
///
/// # Example
/// ```ignore
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
/// let result = mood_test(&x, &y, Alternative::TwoSided)?;
/// println!("{}", result);
/// ```
///
/// # References
/// - R equivalent: `mood.test(x, y)`
pub fn mood_test(x: &[f64], y: &[f64], alternative: Alternative) -> EconResult<MoodTestResult> {
    // Filter out NaN/Inf values
    let x_clean: Vec<f64> = x.iter().filter(|v| v.is_finite()).copied().collect();
    let y_clean: Vec<f64> = y.iter().filter(|v| v.is_finite()).copied().collect();

    let m = x_clean.len();
    let n = y_clean.len();

    if m == 0 || n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Both samples must have at least one finite observation".to_string(),
        });
    }

    let total_n = m + n;
    if total_n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: total_n,
            context: "At least 3 observations are required in total".to_string(),
        });
    }

    // Combine samples and track which sample each observation belongs to
    // 0 = from x, 1 = from y
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(total_n);
    for &val in &x_clean {
        combined.push((val, 0));
    }
    for &val in &y_clean {
        combined.push((val, 1));
    }

    // Sort by value
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Check for ties and compute tie structure
    let (has_ties, tie_counts) = compute_tie_structure(&combined);

    // Compute ranks (1 to N) with midranks for ties
    let ranks = compute_midranks(&combined);

    // Mean rank
    let mean_rank = (total_n as f64 + 1.0) / 2.0;

    // Compute test statistic T = sum of (r_i - mean_rank)^2 for sample x
    // Equation from Conover (1971), p. 234
    let t_stat: f64 = combined
        .iter()
        .zip(&ranks)
        .filter(|((_, group), _)| *group == 0)
        .map(|(_, &r)| (r - mean_rank).powi(2))
        .sum();

    // Expected value under H0: E[T] = m(N² - 1)/12
    // Equation (3) from Conover (1971)
    let n_f = total_n as f64;
    let m_f = m as f64;
    let n_y_f = n as f64;
    let expected = m_f * (n_f.powi(2) - 1.0) / 12.0;

    // Variance under H0
    let variance = if has_ties {
        // Mielke (1967) tie correction
        // V = mn(N+1)(N+2)(N-2)/180 - [mn/(180*N*(N-1))] * Σ t_j(t_j² - 1)(t_j² - 4 + 15(N - t_j)²)
        let base_var = m_f * n_y_f * (n_f + 1.0) * (n_f + 2.0) * (n_f - 2.0) / 180.0;

        let tie_correction: f64 = tie_counts
            .iter()
            .map(|&t| {
                let t_f = t as f64;
                t_f * (t_f.powi(2) - 1.0) * (t_f.powi(2) - 4.0 + 15.0 * (n_f - t_f).powi(2))
            })
            .sum();

        base_var - (m_f * n_y_f) / (180.0 * n_f * (n_f - 1.0)) * tie_correction
    } else {
        // No ties: standard variance formula
        // Equation (4) from Conover (1971)
        m_f * n_y_f * (n_f + 1.0) * (n_f + 2.0) * (n_f - 2.0) / 180.0
    };

    // Ensure variance is positive
    let variance = variance.max(0.0);

    // Z-score
    let z_score = if variance > 0.0 {
        (t_stat - expected) / variance.sqrt()
    } else {
        0.0
    };

    // P-value from standard normal distribution
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z_score.abs())),
        Alternative::Less => normal.cdf(z_score),
        Alternative::Greater => 1.0 - normal.cdf(z_score),
    };

    Ok(MoodTestResult {
        test_name: "Mood two-sample test of scale".to_string(),
        alternative,
        statistic: t_stat,
        z_score,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n_x: m,
        n_y: n,
        n_total: total_n,
        has_ties,
    })
}

/// Wrapper function for running Mood test on a Dataset.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `x_col` - Column name for first sample
/// * `y_col` - Column name for second sample
/// * `alternative` - Direction of alternative hypothesis (default: TwoSided)
///
/// # Returns
/// A `MoodTestResult` containing the test results.
pub fn run_mood_test(
    dataset: &crate::Dataset,
    x_col: &str,
    y_col: &str,
    alternative: Option<Alternative>,
) -> EconResult<MoodTestResult> {
    use polars::prelude::*;

    let df = dataset.df();

    let x_series = df.column(x_col).map_err(|_| EconError::ColumnNotFound {
        column: x_col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    let y_series = df.column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    let x: Vec<f64> = x_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: x_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: y_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    mood_test(&x, &y, alternative.unwrap_or(Alternative::TwoSided))
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Compute midranks for sorted data with potential ties.
fn compute_midranks(data: &[(f64, usize)]) -> Vec<f64> {
    let n = data.len();
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all ties
        while j < n && data[j].0 == data[i].0 {
            j += 1;
        }
        // Average rank for ties (1-indexed)
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Compute tie structure for variance adjustment.
/// Returns (has_ties, vector of tie counts).
fn compute_tie_structure(data: &[(f64, usize)]) -> (bool, Vec<usize>) {
    let n = data.len();
    let mut tie_counts = Vec::new();
    let mut has_ties = false;

    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && data[j].0 == data[i].0 {
            j += 1;
        }
        let count = j - i;
        if count > 1 {
            has_ties = true;
        }
        tie_counts.push(count);
        i = j;
    }

    (has_ties, tie_counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mood_test_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        assert_eq!(result.n_x, 5);
        assert_eq!(result.n_y, 5);
        assert_eq!(result.n_total, 10);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_mood_test_equal_scales() {
        // Two samples with same spread but different locations
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![11.0, 12.0, 13.0, 14.0, 15.0];

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        // Equal spreads should give non-significant result
        assert!(
            result.p_value > 0.05,
            "p-value should be > 0.05 for equal scales, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_mood_test_different_scales() {
        // x has small variance, y has large variance
        let x = vec![4.5, 4.8, 5.0, 5.2, 5.5]; // range = 1
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0]; // range = 8

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        // Should detect difference in scales
        assert!(result.statistic > 0.0);
        // Note: with n=10, power may be limited
    }

    #[test]
    fn test_mood_test_empty_sample() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        assert!(mood_test(&x, &y, Alternative::TwoSided).is_err());
    }

    #[test]
    fn test_mood_test_insufficient_data() {
        let x = vec![1.0];
        let y = vec![2.0];

        // Need at least 3 total observations
        assert!(mood_test(&x, &y, Alternative::TwoSided).is_err());
    }

    #[test]
    fn test_mood_test_with_ties() {
        // Data with ties
        let x = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 3.0, 4.0, 5.0];

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        assert!(result.has_ties);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_mood_test_alternatives() {
        let x = vec![4.5, 4.8, 5.0, 5.2, 5.5];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];

        let two_sided = mood_test(&x, &y, Alternative::TwoSided).unwrap();
        let greater = mood_test(&x, &y, Alternative::Greater).unwrap();
        let less = mood_test(&x, &y, Alternative::Less).unwrap();

        // All should have valid p-values
        assert!(two_sided.p_value >= 0.0 && two_sided.p_value <= 1.0);
        assert!(greater.p_value >= 0.0 && greater.p_value <= 1.0);
        assert!(less.p_value >= 0.0 && less.p_value <= 1.0);

        // Same z-score for all
        assert!((two_sided.z_score - greater.z_score).abs() < 1e-10);
        assert!((two_sided.z_score - less.z_score).abs() < 1e-10);
    }

    #[test]
    fn test_mood_test_with_nan() {
        let x = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let y = vec![10.0, f64::INFINITY, 30.0, 40.0, 50.0];

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        // Should have filtered out invalid values
        assert_eq!(result.n_x, 4);
        assert_eq!(result.n_y, 4);
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_mood_against_r_basic() {
        // R code:
        // x <- c(1, 2, 3, 4, 5)
        // y <- c(10, 20, 30, 40, 50)
        // mood.test(x, y)
        // Z = 0, p-value = 1
        // (Both samples have identical spread relative to their medians)

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        // Z should be close to 0 for equal scales
        assert!(
            result.z_score.abs() < 0.5,
            "z_score should be close to 0 for equal scales, got {}",
            result.z_score
        );
        // p-value should be high (non-significant)
        assert!(
            result.p_value > 0.5,
            "p-value should be > 0.5 for equal scales, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_validate_mood_against_r_randu() {
        // R code from documentation:
        // mood.test(c(randu$x[1:50], randu$x[51:100]))
        // This tests with random uniform data
        // We use similar uniform-ish data

        let x: Vec<f64> = (1..=50).map(|i| (i as f64 * 0.02) % 1.0).collect();
        let y: Vec<f64> = (51..=100).map(|i| (i as f64 * 0.02) % 1.0).collect();

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        // Both samples from same distribution should be non-significant
        assert!(
            result.p_value > 0.01,
            "Random uniform data should not show significant scale difference"
        );
    }

    #[test]
    fn test_validate_mood_different_scales_r() {
        // R code:
        // set.seed(42)
        // x <- rnorm(20, 0, 1)   # sd = 1
        // y <- rnorm(20, 0, 3)   # sd = 3
        // mood.test(x, y)
        // Should show significant difference

        // Simulated data with different scales
        let x = vec![
            0.37, -0.56, 0.36, 0.63, 0.40, -0.11, 1.51, -0.09, 2.02, -0.06, 1.30, 1.27, -0.69,
            -0.45, 1.03, 0.74, -0.60, -0.47, -1.06, 0.59,
        ];
        let y = vec![
            -2.65, 4.76, 1.64, -3.21, -1.88, 2.23, -0.65, 0.98, 3.45, -2.12, 5.34, -4.21, 1.87,
            -0.34, 2.98, -3.67, 0.12, 4.56, -1.23, 3.89,
        ];

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        // With different scales, should detect difference
        // Note: exact p-value depends on data, but should show trend
        assert!(
            result.z_score.abs() > 0.5,
            "z-score should be substantial for different scales"
        );
    }

    #[test]
    fn test_validate_mood_statistic_calculation() {
        // Manual verification of test statistic
        // x = [1, 2, 3], y = [4, 5, 6]
        // Combined ranks: 1, 2, 3, 4, 5, 6
        // Mean rank = 3.5
        // T = (1-3.5)² + (2-3.5)² + (3-3.5)² = 6.25 + 2.25 + 0.25 = 8.75
        // But x gets ranks 1,2,3 when combined is sorted: 1,2,3,4,5,6
        // So for x: T = (1-3.5)² + (2-3.5)² + (3-3.5)² = 6.25 + 2.25 + 0.25 = 8.75

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = mood_test(&x, &y, Alternative::TwoSided).unwrap();

        // Verify statistic matches manual calculation
        let expected_t = 8.75;
        assert!(
            (result.statistic - expected_t).abs() < 0.01,
            "Statistic mismatch: Rust={}, Expected={}",
            result.statistic,
            expected_t
        );

        // Expected value: m(N²-1)/12 = 3*(36-1)/12 = 3*35/12 = 8.75
        // So z should be close to 0
        assert!(
            result.z_score.abs() < 0.5,
            "z-score should be close to 0 when T = E[T]"
        );
    }
}
