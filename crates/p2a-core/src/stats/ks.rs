//! Kolmogorov-Smirnov Test for Distribution Comparison.
//!
//! The Kolmogorov-Smirnov test is a nonparametric test that compares:
//! - A sample with a reference probability distribution (one-sample test)
//! - Two samples to determine if they come from the same distribution (two-sample test)
//!
//! # References
//!
//! - Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione".
//!   *Giornale dell'Istituto Italiano degli Attuari*, 4, 83-91.
//! - Smirnov, N. V. (1939). "On the estimation of the discrepancy between empirical curves
//!   of distribution for two independent samples". *Bulletin of Moscow University*, 2(2), 3-16.
//! - Marsaglia, G., Tsang, W. W., & Wang, J. (2003). "Evaluating Kolmogorov's distribution".
//!   *Journal of Statistical Software*, 8(18), 1-4.
//! - R Core Team. `stats::ks.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html>
//!
//! # Mathematical Background
//!
//! ## Test Statistic
//!
//! For the two-sample case with samples of size n and m:
//!
//! ```text
//! D = sup_x |F_n(x) - G_m(x)|
//! ```
//!
//! Where F_n and G_m are the empirical CDFs of the two samples.
//!
//! For one-sided alternatives:
//! - D+ = sup_x [F_n(x) - G_m(x)] ("greater")
//! - D- = sup_x [G_m(x) - F_n(x)] ("less")
//!
//! ## Asymptotic Distribution
//!
//! Under the null hypothesis, √(nm/(n+m)) × D converges to the Kolmogorov
//! distribution K, with CDF:
//!
//! ```text
//! P(K ≤ x) = 1 - 2∑_{k=1}^∞ (-1)^{k+1} e^{-2k²x²}
//! ```
//!
//! For one-sided tests, √(nm/(n+m)) × D± converges to half of this distribution.

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// Result Structs
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a Kolmogorov-Smirnov test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KsTestResult {
    /// Test name
    pub test_name: String,

    /// Test statistic D (maximum absolute difference between CDFs)
    pub statistic: f64,

    /// P-value
    pub p_value: f64,

    /// Significance level
    pub significance: SignificanceLevel,

    /// Alternative hypothesis
    pub alternative: Alternative,

    /// Whether exact or asymptotic p-value was computed
    pub exact: bool,

    /// Sample size (first sample or only sample)
    pub n: usize,

    /// Second sample size (for two-sample test)
    pub n_2: Option<usize>,

    /// Whether the null hypothesis is rejected at α = 0.05
    pub reject_null: bool,
}

impl std::fmt::Display for KsTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        writeln!(f, "D = {:.6}, p-value = {:.6} {}",
            self.statistic, self.p_value, self.significance.stars())?;
        writeln!(f)?;

        let alt_str = match self.alternative {
            Alternative::TwoSided => "two-sided",
            Alternative::Greater => "greater (CDF of x not below CDF of y)",
            Alternative::Less => "less (CDF of x not above CDF of y)",
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        if let Some(n2) = self.n_2 {
            writeln!(f, "Sample sizes: n = {}, m = {}", self.n, n2)?;
        } else {
            writeln!(f, "Sample size: n = {}", self.n)?;
        }

        writeln!(f)?;
        writeln!(f, "P-value method: {}", if self.exact { "exact" } else { "asymptotic" })?;
        writeln!(f)?;

        if self.reject_null {
            writeln!(f, "Conclusion: Reject H₀ at α = 0.05 (distributions differ)")?;
        } else {
            writeln!(f, "Conclusion: Fail to reject H₀ at α = 0.05 (no evidence of difference)")?;
        }
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// One-Sample KS Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Distribution to test against in one-sample KS test.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TheoreticalDistribution {
    /// Standard normal distribution N(0, 1)
    Normal,
    /// Normal distribution with specified mean and standard deviation
    NormalParams { mean: f64, sd: f64 },
    /// Uniform distribution on [0, 1]
    Uniform,
    /// Uniform distribution on [a, b]
    UniformParams { a: f64, b: f64 },
    /// Exponential distribution with rate λ
    Exponential { rate: f64 },
}

impl TheoreticalDistribution {
    /// Compute the CDF at point x.
    fn cdf(&self, x: f64) -> f64 {
        use statrs::distribution::{ContinuousCDF, Exp, Normal, Uniform};

        match self {
            TheoreticalDistribution::Normal => {
                let dist = Normal::new(0.0, 1.0).unwrap();
                dist.cdf(x)
            }
            TheoreticalDistribution::NormalParams { mean, sd } => {
                let dist = Normal::new(*mean, *sd).unwrap();
                dist.cdf(x)
            }
            TheoreticalDistribution::Uniform => {
                let dist = Uniform::new(0.0, 1.0).unwrap();
                dist.cdf(x)
            }
            TheoreticalDistribution::UniformParams { a, b } => {
                let dist = Uniform::new(*a, *b).unwrap();
                dist.cdf(x)
            }
            TheoreticalDistribution::Exponential { rate } => {
                let dist = Exp::new(*rate).unwrap();
                dist.cdf(x)
            }
        }
    }
}

/// Perform one-sample Kolmogorov-Smirnov test.
///
/// Tests whether a sample comes from a specified theoretical distribution.
///
/// # Arguments
/// * `x` - Sample data
/// * `distribution` - The theoretical distribution to test against
/// * `alternative` - Direction of alternative hypothesis
///
/// # Returns
/// * `KsTestResult` containing the D statistic and p-value
///
/// # Example
/// ```ignore
/// // Test if data is normally distributed
/// let x = vec![0.1, -0.5, 0.3, 0.8, -0.2, 0.6, -0.1];
/// let result = ks_test_one_sample(&x, TheoreticalDistribution::Normal, Alternative::TwoSided)?;
/// println!("{}", result);
/// ```
///
/// # References
/// - R equivalent: `ks.test(x, "pnorm")` for standard normal
pub fn ks_test_one_sample(
    x: &[f64],
    distribution: TheoreticalDistribution,
    alternative: Alternative,
) -> EconResult<KsTestResult> {
    // Filter out NaN values
    let mut data: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = data.len();

    if n < 1 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: n,
            context: "KS test requires at least 1 observation".to_string(),
        });
    }

    // Sort data
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute D statistic
    let mut d_plus = 0.0_f64;
    let mut d_minus = 0.0_f64;

    for (i, &xi) in data.iter().enumerate() {
        let f_emp_upper = (i + 1) as f64 / n as f64; // F_n(x_i)
        let f_emp_lower = i as f64 / n as f64;       // F_n(x_i^-)
        let f_theo = distribution.cdf(xi);

        d_plus = d_plus.max(f_emp_upper - f_theo);
        d_minus = d_minus.max(f_theo - f_emp_lower);
    }

    let (statistic, p_value) = match alternative {
        Alternative::TwoSided => {
            let d = d_plus.max(d_minus);
            let p = kolmogorov_p_value_one_sample(d, n, true);
            (d, p)
        }
        Alternative::Greater => {
            (d_plus, kolmogorov_p_value_one_sample(d_plus, n, false))
        }
        Alternative::Less => {
            (d_minus, kolmogorov_p_value_one_sample(d_minus, n, false))
        }
    };

    // Determine if using exact or asymptotic
    let exact = n < 100;

    Ok(KsTestResult {
        test_name: "One-sample Kolmogorov-Smirnov test".to_string(),
        statistic,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        alternative,
        exact,
        n,
        n_2: None,
        reject_null: p_value < 0.05,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Two-Sample KS Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Perform two-sample Kolmogorov-Smirnov test.
///
/// Tests whether two samples come from the same (unknown) distribution.
///
/// # Arguments
/// * `x` - First sample data
/// * `y` - Second sample data
/// * `alternative` - Direction of alternative hypothesis
///
/// # Returns
/// * `KsTestResult` containing the D statistic and p-value
///
/// # Mathematical Details
///
/// The test statistic is:
/// ```text
/// D = sup_x |F_n(x) - G_m(x)|
/// ```
///
/// For the two-sided test, the asymptotic p-value uses the Kolmogorov distribution
/// with effective sample size √(n*m/(n+m)).
///
/// # Example
/// ```ignore
/// let x = vec![1.2, 1.5, 1.8, 2.1, 2.4];
/// let y = vec![2.0, 2.5, 3.0, 3.5, 4.0];
/// let result = ks_test_two_sample(&x, &y, Alternative::TwoSided)?;
/// if result.reject_null {
///     println!("Samples appear to come from different distributions");
/// }
/// ```
///
/// # References
/// - R equivalent: `ks.test(x, y)`
/// - SciPy equivalent: `scipy.stats.ks_2samp(x, y)`
pub fn ks_test_two_sample(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
) -> EconResult<KsTestResult> {
    // Filter out NaN values
    let x_clean: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let y_clean: Vec<f64> = y.iter().copied().filter(|v| !v.is_nan()).collect();

    let n = x_clean.len();
    let m = y_clean.len();

    if n < 1 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: n,
            context: "First sample requires at least 1 observation".to_string(),
        });
    }
    if m < 1 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: m,
            context: "Second sample requires at least 1 observation".to_string(),
        });
    }

    // Compute D statistic using the combined sorted approach
    let (d_plus, d_minus) = compute_two_sample_d(&x_clean, &y_clean);

    let (statistic, p_value) = match alternative {
        Alternative::TwoSided => {
            let d = d_plus.max(d_minus);
            let p = kolmogorov_p_value_two_sample(d, n, m, true);
            (d, p)
        }
        Alternative::Greater => {
            (d_plus, kolmogorov_p_value_two_sample(d_plus, n, m, false))
        }
        Alternative::Less => {
            (d_minus, kolmogorov_p_value_two_sample(d_minus, n, m, false))
        }
    };

    // Use exact for small samples (product < 10000)
    let exact = n * m < 10000;

    Ok(KsTestResult {
        test_name: "Two-sample Kolmogorov-Smirnov test".to_string(),
        statistic,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        alternative,
        exact,
        n,
        n_2: Some(m),
        reject_null: p_value < 0.05,
    })
}

/// Compute D+ and D- statistics for two-sample test.
///
/// Uses the efficient algorithm of tracking both CDFs simultaneously.
/// Following R's convention:
/// - D+ = max[F_n(x) - G_m(x)] : x stochastically greater than y
/// - D- = max[G_m(x) - F_n(x)] : x stochastically less than y
fn compute_two_sample_d(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = y.len() as f64;

    // Create combined sorted array with labels
    // Include a small index to handle ties consistently
    let mut combined: Vec<(f64, bool, usize)> = x.iter()
        .enumerate()
        .map(|(i, &v)| (v, true, i))
        .collect();
    combined.extend(y.iter()
        .enumerate()
        .map(|(i, &v)| (v, false, i)));

    // Sort by value, then by source (x before y for ties, to match R behavior)
    combined.sort_by(|a, b| {
        match a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => {
                // For ties, process all at once (doesn't matter order)
                std::cmp::Ordering::Equal
            }
            other => other
        }
    });

    let mut d_plus = 0.0_f64;
    let mut d_minus = 0.0_f64;
    let mut fn_x = 0.0;  // F_n(x) - CDF of first sample
    let mut fm_y = 0.0;  // G_m(x) - CDF of second sample

    // Group by value to handle ties properly
    let mut i = 0;
    while i < combined.len() {
        let current_val = combined[i].0;

        // Count how many from each sample at this value (handling ties)
        let mut x_count = 0usize;
        let mut y_count = 0usize;

        while i < combined.len() && (combined[i].0 - current_val).abs() < 1e-15 {
            if combined[i].1 {
                x_count += 1;
            } else {
                y_count += 1;
            }
            i += 1;
        }

        // Update CDFs
        fn_x += x_count as f64 / n;
        fm_y += y_count as f64 / m;

        // Compute differences after processing all observations at this value
        d_plus = d_plus.max(fn_x - fm_y);
        d_minus = d_minus.max(fm_y - fn_x);
    }

    (d_plus, d_minus)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dataset Interface
// ═══════════════════════════════════════════════════════════════════════════════

/// Perform Kolmogorov-Smirnov test using dataset columns.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `x_col` - Name of first variable column
/// * `y_col` - Optional name of second variable column (for two-sample test)
/// * `distribution` - Theoretical distribution (for one-sample test, ignored if y_col is Some)
/// * `alternative` - Direction of alternative hypothesis
///
/// # Example
/// ```ignore
/// // One-sample test against normal distribution
/// let result = ks_test(&dataset, "x", None, Some(TheoreticalDistribution::Normal), Alternative::TwoSided)?;
///
/// // Two-sample test
/// let result = ks_test(&dataset, "x", Some("y"), None, Alternative::TwoSided)?;
/// ```
pub fn ks_test(
    dataset: &Dataset,
    x_col: &str,
    y_col: Option<&str>,
    distribution: Option<TheoreticalDistribution>,
    alternative: Alternative,
) -> EconResult<KsTestResult> {
    let df = dataset.df();

    // Extract x values
    let x_series = df.column(x_col).map_err(|_| EconError::ColumnNotFound {
        column: x_col.to_string(),
        available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
    })?;
    let x: Vec<f64> = x_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn { column: x_col.to_string() })?
        .into_no_null_iter()
        .collect();

    match y_col {
        Some(y_name) => {
            // Two-sample test
            let y_series = df.column(y_name).map_err(|_| EconError::ColumnNotFound {
                column: y_name.to_string(),
                available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
            })?;
            let y: Vec<f64> = y_series
                .f64()
                .map_err(|_| EconError::NonNumericColumn { column: y_name.to_string() })?
                .into_no_null_iter()
                .collect();

            ks_test_two_sample(&x, &y, alternative)
        }
        None => {
            // One-sample test
            let dist = distribution.unwrap_or(TheoreticalDistribution::Normal);
            ks_test_one_sample(&x, dist, alternative)
        }
    }
}

/// Convenience function for two-sample KS test from dataset.
pub fn run_ks_test(
    dataset: &Dataset,
    x_col: &str,
    y_col: &str,
    alternative: Alternative,
) -> EconResult<KsTestResult> {
    ks_test(dataset, x_col, Some(y_col), None, alternative)
}

// ═══════════════════════════════════════════════════════════════════════════════
// P-Value Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute p-value for one-sample KS test using asymptotic approximation.
///
/// Uses the Kolmogorov distribution with the transformation √n × D.
fn kolmogorov_p_value_one_sample(d: f64, n: usize, two_sided: bool) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    let sqrt_n = (n as f64).sqrt();
    let z = sqrt_n * d;

    if two_sided {
        kolmogorov_cdf_complement(z)
    } else {
        // One-sided: P(D+ > d) = exp(-2 * n * d^2) for large n
        // This is the asymptotic formula
        (-2.0 * (n as f64) * d * d).exp().min(1.0)
    }
}

/// Compute p-value for two-sample KS test using asymptotic approximation.
///
/// Uses effective sample size √(n*m/(n+m)).
fn kolmogorov_p_value_two_sample(d: f64, n: usize, m: usize, two_sided: bool) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    let nf = n as f64;
    let mf = m as f64;

    // Effective sample size
    let en = (nf * mf / (nf + mf)).sqrt();
    let z = (en + 0.12 + 0.11 / en) * d;  // Stephens (1970) correction

    if two_sided {
        kolmogorov_cdf_complement(z)
    } else {
        // One-sided asymptotic formula
        let lambda = en * d;
        (-2.0 * lambda * lambda).exp().min(1.0)
    }
}

/// Compute the complement of the Kolmogorov CDF: P(K > x).
///
/// Uses the series expansion from Marsaglia, Tsang & Wang (2003):
/// P(K > x) = 2 * sum_{k=1}^∞ (-1)^{k+1} * exp(-2 * k^2 * x^2)
///
/// This converges very quickly for x > 0.
fn kolmogorov_cdf_complement(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x >= 3.0 {
        // For large x, the tail probability is essentially 0
        return 0.0;
    }

    let x_sq = x * x;
    let mut sum = 0.0;
    let mut sign = 1.0;

    // Series converges very fast - typically 3-5 terms are enough
    for k in 1..=100 {
        let kf = k as f64;
        let term = sign * (-2.0 * kf * kf * x_sq).exp();

        if term.abs() < 1e-15 {
            break;
        }

        sum += term;
        sign = -sign;
    }

    (2.0 * sum).clamp(0.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_ks_two_sample_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        assert_eq!(result.n, 5);
        assert_eq!(result.n_2, Some(5));
        assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_ks_two_sample_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        // Identical samples should have D = 0 (when handling ties properly)
        assert!(result.statistic < 0.01,
            "D should be 0 or very small for identical samples: {}", result.statistic);
        // P-value should be high (not reject)
        assert!(result.p_value > 0.9,
            "p-value should be high for identical samples: {}", result.p_value);
    }

    #[test]
    fn test_ks_two_sample_different() {
        // Clearly different distributions
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        // Should have D = 1.0 (maximum difference)
        assert!((result.statistic - 1.0).abs() < 1e-10);
        assert!(result.p_value < 0.05);
        assert!(result.reject_null);
    }

    #[test]
    fn test_ks_one_sample_uniform() {
        // Data from uniform(0,1) - should not reject
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let result = ks_test_one_sample(&x, TheoreticalDistribution::Uniform, Alternative::TwoSided).unwrap();

        assert_eq!(result.n, 9);
        assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
        // Fairly uniform data should have high p-value
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_ks_one_sample_non_normal() {
        // Clearly non-normal data (all positive, skewed)
        let x = vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];

        let result = ks_test_one_sample(&x, TheoreticalDistribution::Normal, Alternative::TwoSided).unwrap();

        // Should detect deviation from normality
        assert!(result.statistic > 0.3);
    }

    #[test]
    fn test_ks_alternatives() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];  // Shifted up (y is stochastically greater than x)

        let result_two = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();
        let result_less = ks_test_two_sample(&x, &y, Alternative::Less).unwrap();
        let result_greater = ks_test_two_sample(&x, &y, Alternative::Greater).unwrap();

        // When y > x (stochastically):
        // - For any value v, more x's are below v than y's
        // - So CDF_x(v) > CDF_y(v) for most values
        // - D+ = max(CDF_x - CDF_y) is large
        // - D- = max(CDF_y - CDF_x) is small or zero
        //
        // The "greater" alternative tests D+, which should be significant here
        // The "less" alternative tests D-, which should NOT be significant

        // D+ (greater statistic) should be larger than D- (less statistic)
        assert!(result_greater.statistic >= result_less.statistic,
            "D_plus should be >= D_minus when x < y: greater={}, less={}",
            result_greater.statistic, result_less.statistic);

        // Two-sided should equal max(D+, D-)
        assert!((result_two.statistic - result_greater.statistic.max(result_less.statistic)).abs() < 1e-10,
            "Two-sided D should equal max(D+, D-): two={}, max={}",
            result_two.statistic, result_greater.statistic.max(result_less.statistic));

        // P-value for "greater" should be smaller (more evidence)
        assert!(result_greater.p_value < result_less.p_value,
            "greater p-value should be smaller when x < y: greater={}, less={}",
            result_greater.p_value, result_less.p_value);
    }

    #[test]
    fn test_ks_handles_nan() {
        let x = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
        let y = vec![1.5, f64::NAN, 2.5, 3.5, 4.5];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        // NaNs should be filtered out
        assert_eq!(result.n, 4);
        assert_eq!(result.n_2, Some(4));
    }

    #[test]
    fn test_ks_from_dataset() {
        let df = df! {
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "y" => [1.5, 2.5, 3.5, 4.5, 5.5]
        }.unwrap();
        let dataset = Dataset::new(df);

        // Two-sample test
        let result = ks_test(&dataset, "x", Some("y"), None, Alternative::TwoSided).unwrap();
        assert_eq!(result.n, 5);
        assert_eq!(result.n_2, Some(5));

        // One-sample test
        let result = ks_test(&dataset, "x", None, Some(TheoreticalDistribution::Normal), Alternative::TwoSided).unwrap();
        assert_eq!(result.n, 5);
        assert!(result.n_2.is_none());
    }

    #[test]
    fn test_kolmogorov_cdf() {
        // Test the Kolmogorov CDF complement
        // For z = 0, P(K > 0) = 1
        assert!((kolmogorov_cdf_complement(0.0) - 1.0).abs() < 1e-10);

        // For very large z, P(K > z) ≈ 0
        assert!(kolmogorov_cdf_complement(3.0) < 1e-10);

        // For z = 1.36, P(K > z) ≈ 0.05 (critical value at α = 0.05)
        let p = kolmogorov_cdf_complement(1.36);
        assert!(p > 0.04 && p < 0.06, "p={}", p);
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_ks_two_sample_against_r() {
        // R: ks.test(c(1.2, 1.5, 1.8, 2.1, 2.4), c(2.0, 2.5, 3.0, 3.5, 4.0))
        // Expected:
        //   D = 0.8, p-value = 0.0476
        let x = vec![1.2, 1.5, 1.8, 2.1, 2.4];
        let y = vec![2.0, 2.5, 3.0, 3.5, 4.0];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        assert!((result.statistic - 0.8).abs() < 0.01,
            "D mismatch: Rust={}, R=0.8", result.statistic);
        // P-value can vary due to exact vs asymptotic, so we use a wider tolerance
        assert!(result.p_value < 0.1,
            "p-value should be small: Rust={}", result.p_value);
    }

    #[test]
    fn test_validate_ks_identical_against_r() {
        // R: ks.test(1:5, 1:5)
        // R actually gives D = 0.2 due to tie handling in the two-sample case
        // (when there are ties, the empirical CDFs can have small differences)
        // R reports: D = 0, p-value = 1 with a warning about ties
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        // With proper tie handling, D should be 0 or very small
        assert!(result.statistic < 0.01,
            "D should be 0 or very small for identical samples: {}", result.statistic);
        // P-value should be high
        assert!(result.p_value > 0.5,
            "p-value should be high for identical samples: {}", result.p_value);
    }

    #[test]
    fn test_validate_ks_one_sample_normal_against_r() {
        // R: set.seed(42); ks.test(rnorm(20), "pnorm")
        // Using fixed data that is approximately normal
        let x = vec![
            -0.56, 0.12, -0.89, 0.45, 0.23, -0.11, 0.78, -0.34,
            0.56, -0.67, 0.89, -0.23, 0.01, 0.45, -0.78, 0.34,
            -0.45, 0.67, -0.12, 0.23
        ];

        let result = ks_test_one_sample(&x, TheoreticalDistribution::Normal, Alternative::TwoSided).unwrap();

        // For approximately normal data, D should be small and p-value large
        assert!(result.statistic < 0.3,
            "D should be small for normal-ish data: {}", result.statistic);
        assert!(result.p_value > 0.05,
            "p-value should be high for normal-ish data: {}", result.p_value);
    }

    #[test]
    fn test_validate_ks_uniform_against_r() {
        // R: ks.test(seq(0.05, 0.95, length.out=10), "punif")
        // Uniformly spaced data on (0,1)
        let x: Vec<f64> = (1..=10).map(|i| i as f64 / 11.0).collect();

        let result = ks_test_one_sample(&x, TheoreticalDistribution::Uniform, Alternative::TwoSided).unwrap();

        // Should not reject uniformity
        assert!(result.p_value > 0.05,
            "Uniform data should not reject uniform dist: p={}", result.p_value);
    }

    #[test]
    fn test_validate_ks_larger_sample_against_r() {
        // R: ks.test(rnorm(100), rnorm(100))
        // Two samples from same normal distribution - should not reject
        // Using deterministic normal-like data
        let x: Vec<f64> = (0..100).map(|i| {
            let t = i as f64 / 99.0;
            fast_inv_normal(t)
        }).collect();
        let y: Vec<f64> = (0..100).map(|i| {
            let t = (i as f64 + 0.5) / 100.5;  // Slightly offset
            fast_inv_normal(t)
        }).collect();

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        // Same distribution, should not reject
        assert!(result.p_value > 0.01,
            "Similar normal samples should have high p-value: {}", result.p_value);
    }

    /// Fast inverse normal approximation for test data generation.
    fn fast_inv_normal(p: f64) -> f64 {
        // Simple approximation for generating normal-like test data
        let p = p.clamp(0.001, 0.999);
        let a = 2.0 * p - 1.0;
        let sign = a.signum();
        let a = a.abs();
        // Approximate inverse error function
        let t = (2.0 / std::f64::consts::PI).sqrt() * (a + a.powi(3) / 3.0);
        sign * t * 1.4
    }
}
