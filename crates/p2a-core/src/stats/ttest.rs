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

use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy, sample_column};
use crate::errors::{EconError, EconResult};
use crate::traits::SignificanceLevel;
use crate::traits::estimator::{t_quantile, t_test_p_value, validate_level};

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

crate::option_enum!(
    Alternative,
    crate::options::Choices {
        option: "alternative",
        values: &[
            (&["two.sided", "twosided"], Alternative::TwoSided),
            (&["greater", "right", "gt"], Alternative::Greater),
            (&["less", "left", "lt"], Alternative::Less),
        ],
    }
);

impl Alternative {
    /// Parse from string, `None` for an unknown spelling. `FromStr` (`s.parse()`) gives the same
    /// answer with an error naming the accepted spellings.
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
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

    /// Rows dropped because of missing values (dataset wrapper `t_test`; nothing for slices).
    /// A two-sample test sums the two columns' summaries, a paired test drops whole pairs.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for TTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Test statistic
        writeln!(
            f,
            "t = {:.6}, df = {:.2}, p-value = {:.6} {}",
            self.t_statistic,
            self.df,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        // Alternative hypothesis
        let alt_str = match self.alternative {
            Alternative::TwoSided => format!(
                "true difference in means is not equal to {}",
                self.null_value
            ),
            Alternative::Greater => format!(
                "true difference in means is greater than {}",
                self.null_value
            ),
            Alternative::Less => {
                format!("true difference in means is less than {}", self.null_value)
            }
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        // Confidence interval
        writeln!(f, "{:.0}% confidence interval:", self.conf_level * 100.0)?;
        writeln!(
            f,
            "  ({:.6}, {:.6})",
            self.conf_int_lower, self.conf_int_upper
        )?;
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
        // Two samples drop their missing values separately; paired rows are dropped jointly.
        let note = if self.estimate_2.is_some() {
            self.missing.two_sample_note()
        } else {
            self.missing.note()
        };
        if let Some(note) = note {
            writeln!(f, "Missing values: {note}")?;
        }

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
/// # Errors
/// `InvalidSpecification` for a level outside (0, 1) or for essentially constant data: a
/// standard error at or below 10 machine epsilons times the absolute mean (R's `t.test` stops with
/// "data are essentially constant"; all-zero data, where R's t is NaN, are refused too);
/// `InsufficientData` for fewer than 2 observations.
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
    validate_level(conf_level, "conf_level")?;
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
    refuse_constant(std_error, mean.abs())?;

    // Compute t-statistic
    let t_stat = (mean - mu) / std_error;
    let df = (n - 1) as f64;

    // Compute p-value based on alternative
    let p_value = compute_p_value(t_stat, df, alternative);

    // Compute confidence interval
    let (ci_lower, ci_upper) =
        compute_confidence_interval(mean, std_error, df, conf_level, alternative)?;

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
        missing: MissingSummary::none(n),
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
/// # Errors
/// `InvalidSpecification` for a level outside (0, 1) or for essentially constant data: a
/// standard error at or below 10 machine epsilons times the larger absolute mean (R's `t.test` stops with
/// "data are essentially constant"; all-zero data, where R's t is NaN, are refused too);
/// `InsufficientData` for fewer than 2 observations in a sample.
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
    validate_level(conf_level, "conf_level")?;
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
        let pooled_var = ((n1 - 1) as f64 * var1 + (n2 - 1) as f64 * var2) / (n1 + n2 - 2) as f64;
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
    refuse_constant(std_error, mean1.abs().max(mean2.abs()))?;

    // Compute p-value
    let p_value = compute_p_value(t_stat, df, alternative);

    // Confidence interval for difference in means
    let diff = mean1 - mean2;
    let (ci_lower, ci_upper) =
        compute_confidence_interval(diff, std_error, df, conf_level, alternative)?;

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
        missing: MissingSummary::none(n1 + n2),
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
/// # Errors
/// `InvalidSpecification` for a level outside (0, 1) or for essentially constant data: a
/// standard error at or below 10 machine epsilons times the absolute mean difference (R's `t.test` stops with
/// "data are essentially constant"; all-zero data, where R's t is NaN, are refused too);
/// `InsufficientData` for fewer than 2 pairs; `InvalidSpecification` for samples of unequal
/// length.
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
    validate_level(conf_level, "conf_level")?;
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: "Paired t-test requires samples of equal length".to_string(),
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
/// Missing values (null or NaN) are dropped as R's `t.test` drops them, and counted in
/// `missing`:
///
/// | Test | Policy |
/// |---|---|
/// | one-sample (`y_col` is `None`) | the column's missing values are dropped (`x[!is.na(x)]`) |
/// | two-sample | each column drops its own missing values (`t.test(x, y)`) |
/// | paired | a pair with a missing value in either column is dropped (`complete.cases(x, y)`) |
///
/// Integer, decimal and boolean columns are accepted; a text column is refused
/// (`EconError::NonNumericColumn`), as are infinite values (`EconError::NonFiniteValues`).
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
    validate_level(conf_level, "conf_level")?;
    let df = dataset.df();

    match y_col {
        Some(y_name) if paired => {
            let cols = ColumnRequest::new(df)
                .numerics(&[x_col, y_name])
                .na(NaPolicy::Omit)
                .extract()?;
            let mut result = paired_t_test(
                cols.numeric(x_col),
                cols.numeric(y_name),
                mu,
                alternative,
                conf_level,
            )?;
            result.missing = cols.missing().clone();
            Ok(result)
        }
        Some(y_name) => {
            let (x, missing_x) = sample_column(df, x_col, "t-test")?;
            let (y, missing_y) = sample_column(df, y_name, "t-test")?;
            let mut result = two_sample_t_test(&x, &y, mu, alternative, var_equal, conf_level)?;
            result.missing = MissingSummary::merge_samples(&missing_x, &missing_y);
            Ok(result)
        }
        None => {
            let (x, missing) = sample_column(df, x_col, "t-test")?;
            if paired {
                return Err(EconError::InvalidSpecification {
                    message: "Paired test requires two columns".to_string(),
                });
            }
            let mut result = one_sample_t_test(&x, mu, alternative, conf_level)?;
            result.missing = missing;
            Ok(result)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// R's `t.test` rule: the data are essentially constant when the standard error is below 10
/// machine epsilons times the (largest absolute) mean. `<=` refuses all-zero data as well, where
/// R's t is NaN.
fn refuse_constant(std_error: f64, scale: f64) -> EconResult<()> {
    if std_error <= 10.0 * f64::EPSILON * scale {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "data are essentially constant: the standard error {std_error:e} is at or below \
                 10 machine epsilons times the mean (R's t.test stops here)"
            ),
        });
    }
    Ok(())
}

/// One-sided p-value from a two-sided one, for a statistic symmetric under the null.
pub(crate) fn one_sided_p(p2: f64, statistic: f64, alternative: Alternative) -> f64 {
    match alternative {
        Alternative::TwoSided => p2,
        Alternative::Greater if statistic > 0.0 => p2 / 2.0,
        Alternative::Greater => 1.0 - p2 / 2.0,
        Alternative::Less if statistic < 0.0 => p2 / 2.0,
        Alternative::Less => 1.0 - p2 / 2.0,
    }
}

/// Compute p-value based on alternative hypothesis.
fn compute_p_value(t_stat: f64, df: f64, alternative: Alternative) -> f64 {
    if df.is_nan() || df <= 0.0 || t_stat.is_nan() {
        return f64::NAN;
    }
    one_sided_p(t_test_p_value(t_stat, df), t_stat, alternative)
}

/// Compute confidence interval based on alternative hypothesis. NaN bounds for a NaN or
/// non-positive df or standard error; a failing t quantile is an error.
fn compute_confidence_interval(
    estimate: f64,
    std_error: f64,
    df: f64,
    conf_level: f64,
    alternative: Alternative,
) -> EconResult<(f64, f64)> {
    if df.is_nan() || df <= 0.0 || std_error.is_nan() || std_error <= 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let p = match alternative {
        Alternative::TwoSided => (1.0 + conf_level) / 2.0,
        Alternative::Greater | Alternative::Less => conf_level,
    };
    let t_crit = t_quantile(p, df)?;
    Ok(match alternative {
        Alternative::TwoSided => {
            let margin = t_crit * std_error;
            (estimate - margin, estimate + margin)
        }
        Alternative::Greater => (estimate - t_crit * std_error, f64::INFINITY),
        Alternative::Less => (f64::NEG_INFINITY, estimate + t_crit * std_error),
    })
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
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // One-sample
        let result = t_test(
            &dataset,
            "x",
            None,
            2.0,
            Alternative::TwoSided,
            false,
            false,
            0.95,
        )
        .unwrap();
        assert_eq!(result.n, 5);

        // Two-sample
        let result = t_test(
            &dataset,
            "x",
            Some("y"),
            0.0,
            Alternative::TwoSided,
            false,
            false,
            0.95,
        )
        .unwrap();
        assert_eq!(result.n_2, Some(5));

        // Paired
        let result = t_test(
            &dataset,
            "x",
            Some("y"),
            0.0,
            Alternative::TwoSided,
            true,
            false,
            0.95,
        )
        .unwrap();
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
        assert!(matches!(
            result,
            Err(EconError::InvalidSpecification { .. })
        ));
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

        assert!(
            (result.t_statistic - 5.3316).abs() < 0.001,
            "t-stat mismatch: Rust={}, R=5.3316",
            result.t_statistic
        );
        assert!((result.df - 6.0).abs() < 0.001);
        assert!(
            (result.p_value - 0.001775).abs() < 0.0001,
            "p-value mismatch: Rust={}, R=0.001775",
            result.p_value
        );
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

        assert!(
            (result.t_statistic - (-5.4636)).abs() < 0.001,
            "t-stat mismatch: Rust={}, R=-5.4636",
            result.t_statistic
        );
        assert!(
            (result.df - 7.9985).abs() < 0.01,
            "df mismatch: Rust={}, R=7.9985",
            result.df
        );
        assert!(
            (result.p_value - 0.0005993).abs() < 0.0001,
            "p-value mismatch: Rust={}, R=0.0005993",
            result.p_value
        );
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

        assert!(
            (result.t_statistic - 9.4868).abs() < 0.001,
            "t-stat mismatch: Rust={}, R=9.4868",
            result.t_statistic
        );
        assert!((result.df - 4.0).abs() < 0.001);
        assert!(
            (result.p_value - 0.0006889).abs() < 0.0001,
            "p-value mismatch: Rust={}, R=0.0006889",
            result.p_value
        );
        assert!((result.estimate - 6.0).abs() < 0.001);
        assert!((result.conf_int_lower - 4.244022).abs() < 0.1);
        assert!((result.conf_int_upper - 7.755978).abs() < 0.1);
    }

    // ========================================================================
    // Missing values and integer columns (case stat_two_sample_na,
    // validation/r/cases/stats_onesample.R)
    // ========================================================================

    /// X-2/X-8/TOOL-9: one-sample, Welch, pooled and paired t-tests with missing cells;
    /// Welch on Int64 columns.
    #[test]
    fn test_validate_t_test_two_sample_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let run = |x: &str, y: Option<&str>, paired: bool, eq: bool| {
            t_test(&ds, x, y, 0.0, Alternative::TwoSided, paired, eq, 0.95)
                .unwrap_or_else(|e| panic!("{x} {y:?}: {e}"))
        };
        let one = run("x", None, false, false);
        let welch = run("x", Some("y"), false, false);
        let pooled = run("x", Some("y"), false, true);
        let paired = run("x", Some("y"), true, false);
        Check::new(&case)
            .scalar("one_t", one.t_statistic, Tol::stat(7))
            .scalar("one_df", one.df, Tol::exact())
            .scalar("one_p", one.p_value, Tol::p_value(7))
            .vector(
                "one_ci",
                &[one.conf_int_lower, one.conf_int_upper],
                Tol::stat(7),
            )
            .scalar("one_mean", one.estimate, Tol::coef(7))
            .scalar("one_se", one.std_error, Tol::se(7))
            .scalar("n_x", one.n as f64, Tol::exact())
            .scalar("welch_t", welch.t_statistic, Tol::stat(14))
            .scalar("welch_df", welch.df, Tol::stat(14))
            .scalar("welch_p", welch.p_value, Tol::p_value(14))
            .vector(
                "welch_ci",
                &[welch.conf_int_lower, welch.conf_int_upper],
                Tol::stat(14),
            )
            .vector(
                "welch_est",
                &[welch.estimate, welch.estimate_2.unwrap()],
                Tol::coef(14),
            )
            .scalar("welch_se", welch.std_error, Tol::se(14))
            .scalar("n_y", welch.n_2.unwrap() as f64, Tol::exact())
            .scalar("pooled_t", pooled.t_statistic, Tol::stat(14))
            .scalar("pooled_df", pooled.df, Tol::exact())
            .scalar("pooled_p", pooled.p_value, Tol::p_value(14))
            .vector(
                "pooled_ci",
                &[pooled.conf_int_lower, pooled.conf_int_upper],
                Tol::stat(14),
            )
            .scalar("pooled_se", pooled.std_error, Tol::se(14))
            .scalar("paired_t", paired.t_statistic, Tol::stat(6))
            .scalar("paired_df", paired.df, Tol::exact())
            .scalar("paired_p", paired.p_value, Tol::p_value(6))
            .vector(
                "paired_ci",
                &[paired.conf_int_lower, paired.conf_int_upper],
                Tol::stat(6),
            )
            .scalar("paired_est", paired.estimate, Tol::coef(6))
            .scalar("paired_se", paired.std_error, Tol::se(6))
            .scalar("n_pairs", paired.n as f64, Tol::exact())
            .finish();
        // Int64 with nulls (pre-fix: NonNumericColumn)
        let iw = run("xi", Some("yi"), false, false);
        Check::new(&case)
            .scalar("int_welch_t", iw.t_statistic, Tol::stat(38))
            .scalar("int_welch_df", iw.df, Tol::stat(38))
            .scalar("int_welch_p", iw.p_value, Tol::p_value(38))
            .vector(
                "int_welch_ci",
                &[iw.conf_int_lower, iw.conf_int_upper],
                Tol::stat(38),
            )
            .vector(
                "int_welch_est",
                &[iw.estimate, iw.estimate_2.unwrap()],
                Tol::coef(38),
            )
            .scalar("int_welch_se", iw.std_error, Tol::se(38))
            .scalar("n_xi", iw.n as f64, Tol::exact())
            .scalar("n_yi", iw.n_2.unwrap() as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_na_t_test_missing_rows() {
        let (_case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let tt = |x, y, paired| {
            t_test(&ds, x, y, 0.0, Alternative::TwoSided, paired, false, 0.95).unwrap()
        };
        let one = tt("x", None, false);
        assert_eq!(
            (
                one.missing.n_total,
                one.missing.n_used,
                one.missing.n_dropped
            ),
            (22, 7, 15)
        );
        assert_eq!(one.missing.by_column, vec![("x".to_string(), 15)]);
        // each sample on its own (t.test(x, y))
        let welch = tt("x", Some("y"), false);
        assert_eq!(
            welch.missing.by_column,
            vec![("x".to_string(), 15), ("y".to_string(), 15)]
        );
        assert_eq!((welch.n, welch.n_2), (7, Some(7)));
        // each sample drops its own missing values: the note counts values
        assert_eq!(
            welch.to_string().lines().last().unwrap(),
            "Missing values: 30 of 44 values dropped because of missing values (two samples of \
             22; x: 15, y: 15)"
        );
        // complete pairs (complete.cases(x, y))
        let paired = tt("x", Some("y"), true);
        assert_eq!(
            (
                paired.missing.n_total,
                paired.missing.n_used,
                paired.missing.n_dropped
            ),
            (22, 6, 16)
        );
        assert_eq!(
            paired.missing.by_column,
            vec![("x".to_string(), 15), ("y".to_string(), 15)]
        );
        assert_eq!(
            paired.to_string().lines().last().unwrap(),
            format!("Missing values: {}", paired.missing.note().unwrap())
        );
        assert!(!tt("id", None, false).to_string().contains("Missing values"));
        // a NaN cell counts as missing (the harness cannot write NaN: the one inline fixture)
        let nan = Dataset::new(df!("v" => [5.1, f64::NAN, 7.3, 5.9, 6.8, 7.7, 6.1, 6.2]).unwrap());
        let r = t_test(
            &nan,
            "v",
            None,
            0.0,
            Alternative::TwoSided,
            false,
            false,
            0.95,
        )
        .unwrap();
        assert_eq!(
            (r.n, r.missing.by_column.clone()),
            (7, vec![("v".to_string(), 1)])
        );
    }

    #[test]
    fn test_validate_na_t_test_errors() {
        let (_case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let tt = |x, y, paired| t_test(&ds, x, y, 0.0, Alternative::TwoSided, paired, false, 0.95);
        assert!(
            matches!(tt("g", None, false), Err(EconError::NonNumericColumn { column }) if column == "g")
        );
        assert!(
            matches!(tt("x", Some("g"), false), Err(EconError::NonNumericColumn { column }) if column == "g")
        );
        assert!(
            matches!(tt("x", Some("g"), true), Err(EconError::NonNumericColumn { column }) if column == "g")
        );
        assert!(
            matches!(tt("nope", None, false), Err(EconError::ColumnNotFound { column, .. }) if column == "nope")
        );
        let inf = Dataset::new(df!("v" => [1.0, f64::INFINITY, 2.0, 3.0]).unwrap());
        assert!(matches!(
            t_test(&inf, "v", None, 0.0, Alternative::TwoSided, false, false, 0.95),
            Err(EconError::NonFiniteValues { column, count: 1 }) if column == "v"
        ));
    }

    /// X-3: a level outside (0, 1) is refused, a percentage among them.
    #[test]
    fn test_t_test_rejects_level_95() {
        let x = [5.1, 4.8, 6.2, 5.5, 5.9, 4.7];
        let y = [4.2, 4.9, 5.0, 3.8, 4.4, 4.6];
        for level in [95.0, 1.5, 1.0, 0.0, -0.5, f64::NAN] {
            for alt in [
                Alternative::TwoSided,
                Alternative::Greater,
                Alternative::Less,
            ] {
                for r in [
                    one_sample_t_test(&x, 5.0, alt, level),
                    two_sample_t_test(&x, &y, 0.0, alt, false, level),
                    two_sample_t_test(&x, &y, 0.0, alt, true, level),
                    paired_t_test(&x, &y, 0.0, alt, level),
                ] {
                    let e = r.unwrap_err();
                    assert!(
                        matches!(&e, EconError::InvalidSpecification { message } if message.contains("conf_level")),
                        "{e}"
                    );
                }
            }
        }
        let ds = Dataset::new(df! { "x" => x.to_vec(), "y" => y.to_vec() }.unwrap());
        let e = t_test(
            &ds,
            "x",
            Some("y"),
            0.0,
            Alternative::TwoSided,
            false,
            false,
            95.0,
        )
        .unwrap_err();
        assert!(
            e.to_string()
                .contains("conf_level must be strictly between 0 and 1, got 95"),
            "{e}"
        );
    }

    /// X-3: CIs at levels other than 0.95 through t_quantile; pinned to t.test.
    #[test]
    fn test_validate_t_test_t3_levels() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_ttest_levels_t3");
        let mut c = Check::new(&case);
        for (tag, alt) in [
            ("two", Alternative::TwoSided),
            ("less", Alternative::Less),
            ("greater", Alternative::Greater),
        ] {
            let one = t_test(&ds, "x", None, 5.0, alt, false, false, 0.90).unwrap();
            let welch = t_test(&ds, "x", Some("y"), 0.0, alt, false, false, 0.99).unwrap();
            let paired = t_test(&ds, "x", Some("y"), 0.0, alt, true, false, 0.90).unwrap();
            for (prefix, r, n) in [
                ("one90", &one, 12),
                ("welch99", &welch, 12),
                ("paired90", &paired, 12),
            ] {
                c.scalar(&format!("{prefix}_{tag}_p"), r.p_value, Tol::p_value(n));
                match alt {
                    Alternative::TwoSided => {
                        c.vector(
                            &format!("{prefix}_two_ci"),
                            &[r.conf_int_lower, r.conf_int_upper],
                            Tol::stat(n),
                        );
                    }
                    Alternative::Less => {
                        assert_eq!(r.conf_int_lower, f64::NEG_INFINITY);
                        c.scalar(
                            &format!("{prefix}_less_upper"),
                            r.conf_int_upper,
                            Tol::stat(n),
                        );
                    }
                    Alternative::Greater => {
                        assert_eq!(r.conf_int_upper, f64::INFINITY);
                        c.scalar(
                            &format!("{prefix}_greater_lower"),
                            r.conf_int_lower,
                            Tol::stat(n),
                        );
                    }
                }
            }
        }
        c.scalar(
            "one90_t",
            t_test(
                &ds,
                "x",
                None,
                5.0,
                Alternative::TwoSided,
                false,
                false,
                0.95,
            )
            .unwrap()
            .t_statistic,
            Tol::stat(12),
        )
        .scalar(
            "welch_df",
            t_test(
                &ds,
                "x",
                Some("y"),
                0.0,
                Alternative::TwoSided,
                false,
                false,
                0.95,
            )
            .unwrap()
            .df,
            Tol::stat(12),
        )
        .finish();
    }

    /// R's t.test stops with "data are essentially constant" when the standard error is below 10
    /// machine epsilons times the mean; all-zero data (t = NaN in R) are refused the same way.
    #[test]
    fn test_t_tests_refuse_essentially_constant_data() {
        let constant = |r: EconResult<TTestResult>, what: &str| match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("essentially constant"),
                    "{what}: {message}"
                )
            }
            other => panic!("{what}: expected InvalidSpecification, got {other:?}"),
        };
        let two = Alternative::TwoSided;
        constant(one_sample_t_test(&[5.0; 4], 3.0, two, 0.95), "one sample");
        for var_equal in [false, true] {
            constant(
                two_sample_t_test(&[5.0; 3], &[6.0; 3], 0.0, two, var_equal, 0.95),
                "two samples",
            );
        }
        constant(
            paired_t_test(&[5.0, 6.0, 7.0], &[4.0, 5.0, 6.0], 0.0, two, 0.95),
            "paired",
        );
        let big = 1e10 + 1.0;
        constant(
            one_sample_t_test(&[big, big, big + 1e-6], 0.0, two, 0.95),
            "near-constant on a high level",
        );
        constant(one_sample_t_test(&[0.0; 3], 0.0, two, 0.95), "all zero");
        // R: t.test(c(5, 5, 5.1), mu = 3)$p.value = 0.000268636673752541
        let r = one_sample_t_test(&[5.0, 5.0, 5.1], 3.0, two, 0.95).unwrap();
        assert!(
            (r.p_value - 0.000268636673752541).abs() <= 1e-8 * 0.000268636673752541,
            "{}",
            r.p_value
        );
    }

    #[test]
    fn test_t_test_interval_propagates_quantile_errors() {
        let (lo, hi) =
            compute_confidence_interval(1.0, 0.5, 10.0, 0.95, Alternative::TwoSided).unwrap();
        assert!(lo.is_finite() && hi.is_finite() && lo < 1.0 && hi > 1.0);
        let (lo, hi) =
            compute_confidence_interval(1.0, 0.5, 10.0, 0.95, Alternative::Greater).unwrap();
        assert!(lo.is_finite() && hi == f64::INFINITY);
    }

    #[test]
    fn test_alternative_parses_aliases_and_rejects_unknown() {
        assert_eq!(Alternative::choices().validate(), Ok(()));
        for s in [
            "two.sided",
            "two-sided",
            "Two Sided",
            "twosided",
            "TWO_SIDED",
        ] {
            assert_eq!(
                s.parse::<Alternative>().unwrap(),
                Alternative::TwoSided,
                "{s}"
            );
        }
        assert_eq!("gt".parse::<Alternative>().unwrap(), Alternative::Greater);
        assert_eq!("Left".parse::<Alternative>().unwrap(), Alternative::Less);
        assert_eq!(Alternative::from_str("right"), Some(Alternative::Greater)); // the inherent Option API stays
        assert_eq!(Alternative::from_str("sideways"), None);
        assert!("sideways".parse::<Alternative>().is_err());
        assert_eq!(Alternative::TwoSided.to_string(), "two.sided");
    }

    #[test]
    fn test_t_test_all_missing_column_is_named() {
        let ds =
            Dataset::new(df! { "x" => [1.2, 0.4, 2.2], "y" => [None::<f64>, None, None] }.unwrap());
        let e = t_test(
            &ds,
            "x",
            Some("y"),
            0.0,
            Alternative::TwoSided,
            false,
            false,
            0.95,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("'y'") && e.contains("t-test"), "{e}");
    }
}
