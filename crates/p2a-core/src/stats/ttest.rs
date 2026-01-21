//! Student's t-test and Welch's t-test for comparing means.
//!
//! Provides one-sample, two-sample (independent), and paired t-tests.
//!
//! # References
//!
//! - Student (W. S. Gosset) (1908). "The probable error of a mean".
//!   *Biometrika*, 6(1), 1-25.
//! - Welch, B. L. (1947). "The generalization of 'Student's' problem when
//!   several different population variances are involved".
//!   *Biometrika*, 34(1-2), 28-35.
//! - R Core Team. `stats::t.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html>
//!
//! # Mathematical Background
//!
//! ## One-Sample t-test
//!
//! Tests H₀: μ = μ₀ where μ₀ is the hypothesized mean.
//!
//! ```text
//! t = (x̄ - μ₀) / (s / √n)
//! df = n - 1
//! ```
//!
//! ## Two-Sample t-test (Student's - equal variances)
//!
//! Tests H₀: μ₁ = μ₂ using pooled variance.
//!
//! ```text
//! t = (x̄₁ - x̄₂) / (sp × √(1/n₁ + 1/n₂))
//! sp = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁ + n₂ - 2)]
//! df = n₁ + n₂ - 2
//! ```
//!
//! ## Two-Sample t-test (Welch's - unequal variances)
//!
//! Tests H₀: μ₁ = μ₂ without assuming equal variances.
//!
//! ```text
//! t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
//!
//! Welch-Satterthwaite degrees of freedom:
//! df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
//! ```
//!
//! ## Paired t-test
//!
//! For matched pairs, compute differences d = x - y, then apply one-sample test.
//!
//! ```text
//! t = d̄ / (sd / √n)
//! df = n - 1
//! ```

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::SignificanceLevel;

/// Type of hypothesis test (direction of alternative).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Alternative {
    /// Two-sided test: H₁: μ ≠ μ₀
    #[default]
    TwoSided,
    /// Right-tailed test: H₁: μ > μ₀
    Greater,
    /// Left-tailed test: H₁: μ < μ₀
    Less,
}

impl Alternative {
    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "two.sided" | "two-sided" | "twosided" | "two_sided" => Some(Self::TwoSided),
            "greater" | "right" | "gt" => Some(Self::Greater),
            "less" | "left" | "lt" => Some(Self::Less),
            _ => None,
        }
    }
}

/// Result of a t-test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Test Type
    // ═══════════════════════════════════════════════════════════════════════
    /// Description of the test performed
    pub test_name: String,
    /// Alternative hypothesis type
    pub alternative: Alternative,

    // ═══════════════════════════════════════════════════════════════════════
    // Test Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// t-statistic
    pub t_statistic: f64,
    /// Degrees of freedom
    pub df: f64,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,

    // ═══════════════════════════════════════════════════════════════════════
    // Confidence Interval
    // ═══════════════════════════════════════════════════════════════════════
    /// Confidence level (e.g., 0.95)
    pub conf_level: f64,
    /// Lower bound of confidence interval
    pub conf_int_lower: f64,
    /// Upper bound of confidence interval
    pub conf_int_upper: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Estimates
    // ═══════════════════════════════════════════════════════════════════════
    /// Sample mean(s) or difference
    pub estimate: f64,
    /// Second sample mean (for two-sample tests)
    pub estimate_2: Option<f64>,
    /// Null hypothesis value
    pub null_value: f64,
    /// Standard error of the estimate
    pub std_error: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Sample Info
    // ═══════════════════════════════════════════════════════════════════════
    /// Sample size (or n₁ for two-sample)
    pub n: usize,
    /// Second sample size (for two-sample tests)
    pub n_2: Option<usize>,
}

impl std::fmt::Display for TTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Test statistic
        writeln!(f, "t = {:.6}, df = {:.2}, p-value = {:.6} {}",
            self.t_statistic, self.df, self.p_value, self.significance.stars())?;
        writeln!(f)?;

        // Alternative hypothesis
        let alt_str = match self.alternative {
            Alternative::TwoSided => format!("true difference in means is not equal to {}", self.null_value),
            Alternative::Greater => format!("true difference in means is greater than {}", self.null_value),
            Alternative::Less => format!("true difference in means is less than {}", self.null_value),
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        // Confidence interval
        writeln!(f, "{:.0}% confidence interval:", self.conf_level * 100.0)?;
        writeln!(f, "  ({:.6}, {:.6})", self.conf_int_lower, self.conf_int_upper)?;
        writeln!(f)?;

        // Estimates
        writeln!(f, "Sample estimates:")?;
        if let Some(est2) = self.estimate_2 {
            writeln!(f, "  mean of x: {:.6}", self.estimate)?;
            writeln!(f, "  mean of y: {:.6}", est2)?;
        } else {
            writeln!(f, "  mean: {:.6}", self.estimate)?;
        }
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Perform one-sample t-test.
///
/// Tests whether the mean of a sample differs from a hypothesized value.
///
/// # Arguments
/// * `x` - Sample data
/// * `mu` - Hypothesized mean (null hypothesis value)
/// * `alternative` - Direction of alternative hypothesis
/// * `conf_level` - Confidence level for interval (e.g., 0.95)
///
/// # Example
/// ```ignore
/// let x = vec![2.1, 2.5, 2.3, 2.8, 2.6];
/// let result = one_sample_t_test(&x, 2.0, Alternative::TwoSided, 0.95)?;
/// println!("{}", result);
/// ```
///
/// # References
/// - R equivalent: `t.test(x, mu = 2.0)`
pub fn one_sample_t_test(
    x: &[f64],
    mu: f64,
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<TTestResult> {
    let n = x.len();
    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "One-sample t-test requires at least 2 observations".to_string(),
        });
    }

    // Compute sample statistics
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let variance: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();
    let std_error = std_dev / (n as f64).sqrt();

    // Compute t-statistic
    let t_stat = (mean - mu) / std_error;
    let df = (n - 1) as f64;

    // Compute p-value based on alternative
    let p_value = compute_p_value(t_stat, df, alternative);

    // Compute confidence interval
    let (ci_lower, ci_upper) = compute_confidence_interval(
        mean, std_error, df, conf_level, alternative
    );

    Ok(TTestResult {
        test_name: "One Sample t-test".to_string(),
        alternative,
        t_statistic: t_stat,
        df,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        conf_level,
        conf_int_lower: ci_lower,
        conf_int_upper: ci_upper,
        estimate: mean,
        estimate_2: None,
        null_value: mu,
        std_error,
        n,
        n_2: None,
    })
}

/// Perform two-sample t-test (independent samples).
///
/// Tests whether the means of two independent samples differ.
///
/// # Arguments
/// * `x` - First sample data
/// * `y` - Second sample data
/// * `mu` - Hypothesized difference in means (default: 0)
/// * `alternative` - Direction of alternative hypothesis
/// * `var_equal` - If true, use pooled variance (Student's t-test);
///                 if false, use Welch's t-test (recommended)
/// * `conf_level` - Confidence level for interval (e.g., 0.95)
///
/// # Example
/// ```ignore
/// let x = vec![2.1, 2.5, 2.3, 2.8, 2.6];
/// let y = vec![3.2, 3.5, 3.1, 3.8];
/// let result = two_sample_t_test(&x, &y, 0.0, Alternative::TwoSided, false, 0.95)?;
/// println!("{}", result);
/// ```
///
/// # Note
/// Welch's t-test (`var_equal = false`) is recommended as the default because:
/// - It is more robust when variances are unequal
/// - It loses minimal power when variances are actually equal
///
/// # References
/// - R equivalent: `t.test(x, y, var.equal = FALSE)` (Welch's)
/// - R equivalent: `t.test(x, y, var.equal = TRUE)` (Student's)
pub fn two_sample_t_test(
    x: &[f64],
    y: &[f64],
    mu: f64,
    alternative: Alternative,
    var_equal: bool,
    conf_level: f64,
) -> EconResult<TTestResult> {
    let n1 = x.len();
    let n2 = y.len();

    if n1 < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n1,
            context: "First sample requires at least 2 observations".to_string(),
        });
    }
    if n2 < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n2,
            context: "Second sample requires at least 2 observations".to_string(),
        });
    }

    // Compute sample statistics
    let mean1: f64 = x.iter().sum::<f64>() / n1 as f64;
    let mean2: f64 = y.iter().sum::<f64>() / n2 as f64;
    let var1: f64 = x.iter().map(|&xi| (xi - mean1).powi(2)).sum::<f64>() / (n1 - 1) as f64;
    let var2: f64 = y.iter().map(|&yi| (yi - mean2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

    let (t_stat, df, std_error) = if var_equal {
        // Student's t-test: pooled variance
        let pooled_var = ((n1 - 1) as f64 * var1 + (n2 - 1) as f64 * var2)
            / (n1 + n2 - 2) as f64;
        let se = (pooled_var * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
        let t = (mean1 - mean2 - mu) / se;
        let df = (n1 + n2 - 2) as f64;
        (t, df, se)
    } else {
        // Welch's t-test: separate variances
        let se_sq = var1 / n1 as f64 + var2 / n2 as f64;
        let se = se_sq.sqrt();
        let t = (mean1 - mean2 - mu) / se;

        // Welch-Satterthwaite degrees of freedom
        let v1 = var1 / n1 as f64;
        let v2 = var2 / n2 as f64;
        let df = (v1 + v2).powi(2) / (v1.powi(2) / (n1 - 1) as f64 + v2.powi(2) / (n2 - 1) as f64);
        (t, df, se)
    };

    // Compute p-value
    let p_value = compute_p_value(t_stat, df, alternative);

    // Confidence interval for difference in means
    let diff = mean1 - mean2;
    let (ci_lower, ci_upper) = compute_confidence_interval(
        diff, std_error, df, conf_level, alternative
    );

    let test_name = if var_equal {
        "Two Sample t-test (equal variances)"
    } else {
        "Welch Two Sample t-test"
    };

    Ok(TTestResult {
        test_name: test_name.to_string(),
        alternative,
        t_statistic: t_stat,
        df,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        conf_level,
        conf_int_lower: ci_lower,
        conf_int_upper: ci_upper,
        estimate: mean1,
        estimate_2: Some(mean2),
        null_value: mu,
        std_error,
        n: n1,
        n_2: Some(n2),
    })
}

/// Perform paired t-test.
///
/// Tests whether the mean difference between paired observations is zero.
///
/// # Arguments
/// * `x` - First sample data
/// * `y` - Second sample data (must be same length as x)
/// * `mu` - Hypothesized mean difference (default: 0)
/// * `alternative` - Direction of alternative hypothesis
/// * `conf_level` - Confidence level for interval (e.g., 0.95)
///
/// # Example
/// ```ignore
/// let before = vec![200.0, 190.0, 210.0, 180.0, 195.0];
/// let after = vec![195.0, 188.0, 202.0, 175.0, 190.0];
/// let result = paired_t_test(&before, &after, 0.0, Alternative::TwoSided, 0.95)?;
/// println!("{}", result);
/// ```
///
/// # References
/// - R equivalent: `t.test(x, y, paired = TRUE)`
pub fn paired_t_test(
    x: &[f64],
    y: &[f64],
    mu: f64,
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<TTestResult> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: "Paired t-test requires samples of equal length".to_string()
        });
    }

    let n = x.len();
    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "Paired t-test requires at least 2 pairs".to_string(),
        });
    }

    // Compute differences
    let differences: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();

    // Apply one-sample t-test to differences
    let mut result = one_sample_t_test(&differences, mu, alternative, conf_level)?;

    // Update metadata
    result.test_name = "Paired t-test".to_string();
    result.estimate_2 = None; // For paired test, estimate is mean difference

    Ok(result)
}

/// Perform t-test using dataset columns.
///
/// Convenience wrapper that extracts data from a Dataset.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `x_col` - Name of first variable column
/// * `y_col` - Optional name of second variable column (for two-sample/paired tests)
/// * `mu` - Hypothesized mean or difference (default: 0)
/// * `alternative` - Direction of alternative hypothesis
/// * `paired` - If true, perform paired test (requires y_col)
/// * `var_equal` - If true, assume equal variances (for two-sample only)
/// * `conf_level` - Confidence level for interval
///
/// # Example
/// ```ignore
/// // One-sample test
/// let result = t_test(&dataset, "x", None, 0.0, Alternative::TwoSided, false, false, 0.95)?;
///
/// // Two-sample Welch's test
/// let result = t_test(&dataset, "x", Some("y"), 0.0, Alternative::TwoSided, false, false, 0.95)?;
///
/// // Paired test
/// let result = t_test(&dataset, "before", Some("after"), 0.0, Alternative::TwoSided, true, false, 0.95)?;
/// ```
pub fn t_test(
    dataset: &Dataset,
    x_col: &str,
    y_col: Option<&str>,
    mu: f64,
    alternative: Alternative,
    paired: bool,
    var_equal: bool,
    conf_level: f64,
) -> EconResult<TTestResult> {
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
            // Extract y values
            let y_series = df.column(y_name).map_err(|_| EconError::ColumnNotFound {
                column: y_name.to_string(),
                available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
            })?;
            let y: Vec<f64> = y_series
                .f64()
                .map_err(|_| EconError::NonNumericColumn { column: y_name.to_string() })?
                .into_no_null_iter()
                .collect();

            if paired {
                paired_t_test(&x, &y, mu, alternative, conf_level)
            } else {
                two_sample_t_test(&x, &y, mu, alternative, var_equal, conf_level)
            }
        }
        None => {
            if paired {
                return Err(EconError::InvalidSpecification {
                    message: "Paired test requires two columns".to_string()
                });
            }
            one_sample_t_test(&x, mu, alternative, conf_level)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Compute p-value based on alternative hypothesis.
fn compute_p_value(t_stat: f64, df: f64, alternative: Alternative) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};

    if df <= 0.0 || t_stat.is_nan() {
        return f64::NAN;
    }
    if t_stat.is_infinite() || t_stat.abs() > 1e10 {
        return 0.0;
    }

    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();

    match alternative {
        Alternative::TwoSided => {
            // Two-tailed: P(|T| > |t|)
            2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
        }
        Alternative::Greater => {
            // Right-tailed: P(T > t)
            1.0 - t_dist.cdf(t_stat)
        }
        Alternative::Less => {
            // Left-tailed: P(T < t)
            t_dist.cdf(t_stat)
        }
    }
}

/// Compute confidence interval based on alternative hypothesis.
fn compute_confidence_interval(
    estimate: f64,
    std_error: f64,
    df: f64,
    conf_level: f64,
    alternative: Alternative,
) -> (f64, f64) {
    use statrs::distribution::{ContinuousCDF, StudentsT};

    if df <= 0.0 || std_error.is_nan() || std_error <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let alpha = 1.0 - conf_level;

    match alternative {
        Alternative::TwoSided => {
            let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);
            let margin = t_crit * std_error;
            (estimate - margin, estimate + margin)
        }
        Alternative::Greater => {
            // One-sided lower bound
            let t_crit = t_dist.inverse_cdf(1.0 - alpha);
            (estimate - t_crit * std_error, f64::INFINITY)
        }
        Alternative::Less => {
            // One-sided upper bound
            let t_crit = t_dist.inverse_cdf(1.0 - alpha);
            (f64::NEG_INFINITY, estimate + t_crit * std_error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_one_sample_t_test_basic() {
        let x = vec![2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7];
        let result = one_sample_t_test(&x, 2.0, Alternative::TwoSided, 0.95).unwrap();

        assert_eq!(result.n, 7);
        assert!((result.estimate - 2.4857).abs() < 0.001);
        assert!(result.df == 6.0);
        assert!(result.t_statistic > 0.0); // Mean > hypothesized
        assert!(result.p_value > 0.0 && result.p_value < 1.0);
        assert!(result.conf_int_lower < result.estimate);
        assert!(result.conf_int_upper > result.estimate);
    }

    #[test]
    fn test_one_sample_t_test_alternatives() {
        let x = vec![5.0, 6.0, 7.0, 5.5, 6.5];

        // Two-sided
        let result = one_sample_t_test(&x, 5.0, Alternative::TwoSided, 0.95).unwrap();
        let p_two = result.p_value;

        // Greater (right-tailed)
        let result = one_sample_t_test(&x, 5.0, Alternative::Greater, 0.95).unwrap();
        let p_greater = result.p_value;

        // Less (left-tailed)
        let result = one_sample_t_test(&x, 5.0, Alternative::Less, 0.95).unwrap();
        let p_less = result.p_value;

        // p_two should be approximately 2 * p_greater (since mean > 5)
        assert!((p_two - 2.0 * p_greater).abs() < 0.001);
        // p_less should be close to 1 - p_greater
        assert!((p_less - (1.0 - p_greater)).abs() < 0.001);
    }

    #[test]
    fn test_two_sample_welch() {
        let x = vec![2.1, 2.5, 2.3, 2.8, 2.6];
        let y = vec![3.2, 3.5, 3.1, 3.8, 3.4];

        let result = two_sample_t_test(&x, &y, 0.0, Alternative::TwoSided, false, 0.95).unwrap();

        assert_eq!(result.n, 5);
        assert_eq!(result.n_2, Some(5));
        assert!(result.t_statistic < 0.0); // mean(x) < mean(y)
        assert!(result.p_value < 0.05); // Should be significant
        assert!(result.conf_int_upper < 0.0); // CI should not include 0
    }

    #[test]
    fn test_two_sample_student() {
        let x = vec![2.1, 2.5, 2.3, 2.8, 2.6];
        let y = vec![3.2, 3.5, 3.1, 3.8, 3.4];

        let result = two_sample_t_test(&x, &y, 0.0, Alternative::TwoSided, true, 0.95).unwrap();

        // Student's t-test has integer df
        assert!((result.df - 8.0).abs() < 0.001);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_paired_t_test() {
        let before = vec![200.0, 190.0, 210.0, 180.0, 195.0];
        let after = vec![195.0, 185.0, 202.0, 175.0, 188.0];

        let result = paired_t_test(&before, &after, 0.0, Alternative::TwoSided, 0.95).unwrap();

        assert_eq!(result.n, 5);
        assert_eq!(result.df, 4.0);
        // Mean difference should be positive (before > after)
        assert!(result.estimate > 0.0);
        assert!(result.t_statistic > 0.0);
    }

    #[test]
    fn test_t_test_from_dataset() {
        let df = df! {
            "x" => [2.1, 2.5, 2.3, 2.8, 2.6],
            "y" => [3.2, 3.5, 3.1, 3.8, 3.4]
        }.unwrap();
        let dataset = Dataset::new(df);

        // One-sample
        let result = t_test(&dataset, "x", None, 2.0, Alternative::TwoSided, false, false, 0.95).unwrap();
        assert_eq!(result.n, 5);

        // Two-sample
        let result = t_test(&dataset, "x", Some("y"), 0.0, Alternative::TwoSided, false, false, 0.95).unwrap();
        assert_eq!(result.n_2, Some(5));

        // Paired
        let result = t_test(&dataset, "x", Some("y"), 0.0, Alternative::TwoSided, true, false, 0.95).unwrap();
        assert!(result.test_name.contains("Paired"));
    }

    #[test]
    fn test_insufficient_data() {
        let x = vec![1.0];
        let result = one_sample_t_test(&x, 0.0, Alternative::TwoSided, 0.95);
        assert!(matches!(result, Err(EconError::InsufficientData { .. })));
    }

    #[test]
    fn test_paired_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let result = paired_t_test(&x, &y, 0.0, Alternative::TwoSided, 0.95);
        assert!(matches!(result, Err(EconError::InvalidSpecification { .. })));
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================
    //
    // These tests compare results with R's t.test() function.

    #[test]
    fn test_validate_one_sample_against_r() {
        // R: t.test(c(2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7), mu = 2.0)
        // Expected:
        //   t = 5.3316, df = 6, p-value = 0.001775
        //   95% CI: (2.262799, 2.708629)
        //   mean of x = 2.485714
        let x = vec![2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7];
        let result = one_sample_t_test(&x, 2.0, Alternative::TwoSided, 0.95).unwrap();

        assert!((result.t_statistic - 5.3316).abs() < 0.001,
            "t-stat mismatch: Rust={}, R=5.3316", result.t_statistic);
        assert!((result.df - 6.0).abs() < 0.001);
        assert!((result.p_value - 0.001775).abs() < 0.0001,
            "p-value mismatch: Rust={}, R=0.001775", result.p_value);
        assert!((result.estimate - 2.485714).abs() < 0.0001);
        assert!((result.conf_int_lower - 2.262799).abs() < 0.01);
        assert!((result.conf_int_upper - 2.708629).abs() < 0.01);
    }

    #[test]
    fn test_validate_welch_against_r() {
        // R: t.test(c(2.1, 2.5, 2.3, 2.8, 2.6), c(3.2, 3.5, 3.1, 3.8, 3.4))
        // Expected (Welch's):
        //   t = -5.4636, df = 7.9985, p-value = 0.0005993
        //   95% CI: (-1.3367526, -0.5432474)
        //   mean of x = 2.46, mean of y = 3.40
        let x = vec![2.1, 2.5, 2.3, 2.8, 2.6];
        let y = vec![3.2, 3.5, 3.1, 3.8, 3.4];
        let result = two_sample_t_test(&x, &y, 0.0, Alternative::TwoSided, false, 0.95).unwrap();

        assert!((result.t_statistic - (-5.4636)).abs() < 0.001,
            "t-stat mismatch: Rust={}, R=-5.4636", result.t_statistic);
        assert!((result.df - 7.9985).abs() < 0.01,
            "df mismatch: Rust={}, R=7.9985", result.df);
        assert!((result.p_value - 0.0005993).abs() < 0.0001,
            "p-value mismatch: Rust={}, R=0.0005993", result.p_value);
        assert!((result.estimate - 2.46).abs() < 0.01);
        assert!((result.estimate_2.unwrap() - 3.40).abs() < 0.01);
    }

    #[test]
    fn test_validate_paired_against_r() {
        // R: t.test(c(200, 190, 210, 180, 195), c(195, 185, 202, 175, 188), paired = TRUE)
        // Differences: 5 5 8 5 7, mean = 6, sd = 1.414214
        // Expected:
        //   t = 9.4868, df = 4, p-value = 0.0006889
        //   95% CI: (4.244022, 7.755978)
        //   mean of differences = 6
        let before = vec![200.0, 190.0, 210.0, 180.0, 195.0];
        let after = vec![195.0, 185.0, 202.0, 175.0, 188.0];
        let result = paired_t_test(&before, &after, 0.0, Alternative::TwoSided, 0.95).unwrap();

        assert!((result.t_statistic - 9.4868).abs() < 0.001,
            "t-stat mismatch: Rust={}, R=9.4868", result.t_statistic);
        assert!((result.df - 4.0).abs() < 0.001);
        assert!((result.p_value - 0.0006889).abs() < 0.0001,
            "p-value mismatch: Rust={}, R=0.0006889", result.p_value);
        assert!((result.estimate - 6.0).abs() < 0.001);
        assert!((result.conf_int_lower - 4.244022).abs() < 0.1);
        assert!((result.conf_int_upper - 7.755978).abs() < 0.1);
    }
}
