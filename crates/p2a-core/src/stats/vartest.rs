//! F test for comparing two variances (var.test).
//!
//! Tests the null hypothesis that the ratio of variances of two populations
//! equals a specified value (default 1, meaning equal variances).
//!
//! # References
//!
//! - Snedecor, G. W. and Cochran, W. G. (1989). *Statistical Methods* (8th ed).
//!   Iowa State University Press.
//! - R Core Team. `stats::var.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/var.test.html>
//!
//! # Mathematical Background
//!
//! ## Test Statistic
//!
//! The F statistic is the ratio of sample variances:
//!
//! ```text
//! F = s₁² / s₂²
//! ```
//!
//! Under H₀: σ₁²/σ₂² = ratio, F follows an F distribution with:
//! - df₁ = n₁ - 1 (numerator degrees of freedom)
//! - df₂ = n₂ - 1 (denominator degrees of freedom)
//!
//! ## Confidence Interval for Variance Ratio
//!
//! The (1-α) confidence interval for σ₁²/σ₂² is:
//!
//! ```text
//! [ (s₁²/s₂²) / F_{1-α/2, df₁, df₂} , (s₁²/s₂²) / F_{α/2, df₁, df₂} ]
//! ```
//!
//! For one-sided alternatives, the interval is adjusted accordingly.
//!
//! ## Assumptions
//!
//! - Both samples are from normal populations
//! - The samples are independent
//!
//! **Note:** This test is sensitive to departures from normality. For robust
//! alternatives, consider `bartlett_test` or `fligner_test`.

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{MissingSummary, NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;
use crate::traits::estimator::f_test_p_value;

/// Result of an F test for comparing two variances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarTestResult {
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
    /// F-statistic (ratio of sample variances)
    pub f_statistic: f64,
    /// Numerator degrees of freedom (n₁ - 1)
    pub df_num: f64,
    /// Denominator degrees of freedom (n₂ - 1)
    pub df_denom: f64,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,

    // ═══════════════════════════════════════════════════════════════════════
    // Confidence Interval
    // ═══════════════════════════════════════════════════════════════════════
    /// Confidence level (e.g., 0.95)
    pub conf_level: f64,
    /// Lower bound of confidence interval for variance ratio
    pub conf_int_lower: f64,
    /// Upper bound of confidence interval for variance ratio
    pub conf_int_upper: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Estimates
    // ═══════════════════════════════════════════════════════════════════════
    /// Sample variance of first group
    pub var_x: f64,
    /// Sample variance of second group
    pub var_y: f64,
    /// Ratio of sample variances (estimate)
    pub estimate: f64,
    /// Null hypothesis value for the ratio
    pub null_value: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Sample Info
    // ═══════════════════════════════════════════════════════════════════════
    /// Sample size of first group
    pub n_x: usize,
    /// Sample size of second group
    pub n_y: usize,
    /// Rows dropped because of missing values (dataset wrapper; nothing for slices). Each
    /// sample drops its own missing values; the summary sums the two columns.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for VarTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Test statistic
        writeln!(
            f,
            "F = {:.6}, num df = {:.0}, denom df = {:.0}, p-value = {:.6} {}",
            self.f_statistic,
            self.df_num,
            self.df_denom,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        // Alternative hypothesis
        let alt_str = match self.alternative {
            Alternative::TwoSided => {
                format!(
                    "true ratio of variances is not equal to {}",
                    self.null_value
                )
            }
            Alternative::Greater => {
                format!(
                    "true ratio of variances is greater than {}",
                    self.null_value
                )
            }
            Alternative::Less => {
                format!("true ratio of variances is less than {}", self.null_value)
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
        writeln!(f, "  ratio of variances: {:.6}", self.estimate)?;
        writeln!(f, "  variance of x:      {:.6}", self.var_x)?;
        writeln!(f, "  variance of y:      {:.6}", self.var_y)?;
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        // The two samples drop their missing values separately.
        if let Some(note) = self.missing.two_sample_note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Perform F test for comparing two variances.
///
/// Tests the null hypothesis that the ratio of variances equals a specified value.
///
/// # Arguments
/// * `x` - First sample data
/// * `y` - Second sample data
/// * `ratio` - Hypothesized ratio of variances σ₁²/σ₂² (default: 1.0)
/// * `alternative` - Direction of alternative hypothesis
/// * `conf_level` - Confidence level for interval (e.g., 0.95)
///
/// # Example
/// ```ignore
/// let x = vec![2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7];
/// let y = vec![3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3];
/// let result = var_test(&x, &y, 1.0, Alternative::TwoSided, 0.95)?;
/// println!("{}", result);
/// ```
///
/// # Note
/// This test assumes both samples come from normal populations. It is sensitive
/// to departures from normality. For a more robust test, consider using
/// `bartlett_test` (parametric) or `fligner_test` (non-parametric).
///
/// # References
/// - R equivalent: `var.test(x, y, ratio = 1, alternative = "two.sided")`
pub fn var_test(
    x: &[f64],
    y: &[f64],
    ratio: f64,
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<VarTestResult> {
    let n_x = x.len();
    let n_y = y.len();

    if n_x < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_x,
            context: "First sample requires at least 2 observations".to_string(),
        });
    }
    if n_y < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_y,
            context: "Second sample requires at least 2 observations".to_string(),
        });
    }

    if ratio <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Variance ratio must be positive".to_string(),
        });
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "Confidence level must be between 0 and 1".to_string(),
        });
    }

    // Compute sample variances
    let mean_x: f64 = x.iter().sum::<f64>() / n_x as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n_y as f64;
    let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>() / (n_x - 1) as f64;
    let var_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>() / (n_y - 1) as f64;

    // Handle zero variance edge cases
    if var_x == 0.0 && var_y == 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Both samples have zero variance".to_string(),
        });
    }
    if var_y == 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Second sample has zero variance".to_string(),
        });
    }

    // Degrees of freedom
    let df_num = (n_x - 1) as f64;
    let df_denom = (n_y - 1) as f64;

    // F statistic: ratio of sample variances divided by hypothesized ratio
    let f_stat = (var_x / var_y) / ratio;
    let estimate = var_x / var_y;

    // Compute p-value
    let p_value = compute_f_p_value(f_stat, df_num, df_denom, alternative);

    // Compute confidence interval
    let (ci_lower, ci_upper) =
        compute_variance_ratio_ci(estimate, df_num, df_denom, conf_level, alternative);

    Ok(VarTestResult {
        test_name: "F test to compare two variances".to_string(),
        alternative,
        f_statistic: f_stat,
        df_num,
        df_denom,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        conf_level,
        conf_int_lower: ci_lower,
        conf_int_upper: ci_upper,
        var_x,
        var_y,
        estimate,
        null_value: ratio,
        n_x,
        n_y,
        missing: MissingSummary::none(n_x + n_y),
    })
}

/// Perform F test using dataset columns.
///
/// Convenience wrapper that extracts data from a Dataset. Each column drops its own missing
/// values (null or NaN), as `var.test(x, y)` does, and the drop is reported in `missing`.
/// Integer, decimal and boolean columns are accepted; a text column is refused
/// (`EconError::NonNumericColumn`), as are infinite values (`EconError::NonFiniteValues`).
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `x_col` - Name of first variable column
/// * `y_col` - Name of second variable column
/// * `ratio` - Hypothesized ratio of variances (default: 1.0)
/// * `alternative` - Direction of alternative hypothesis
/// * `conf_level` - Confidence level for interval
///
/// # Example
/// ```ignore
/// let result = run_var_test(&dataset, "group1", "group2", 1.0, Alternative::TwoSided, 0.95)?;
/// ```
pub fn run_var_test(
    dataset: &Dataset,
    x_col: &str,
    y_col: &str,
    ratio: f64,
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<VarTestResult> {
    let df = dataset.df();
    let (x, missing_x) = numeric_column(df, x_col, NaPolicy::Omit)?;
    let (y, missing_y) = numeric_column(df, y_col, NaPolicy::Omit)?;
    let mut result = var_test(&x, &y, ratio, alternative, conf_level)?;
    result.missing = MissingSummary::merge_samples(&missing_x, &missing_y);
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// P-value of the F statistic. R's var.test computes 1 - pf; this computes both tails directly
/// (X-17), the lower one as `P(F <= f) = P(F' >= 1/f)` with `F' ~ F(df2, df1)`, so at an extreme
/// ratio p is the exact tail where R prints 0.
fn compute_f_p_value(f_stat: f64, df1: f64, df2: f64, alternative: Alternative) -> f64 {
    if df1 <= 0.0 || df2 <= 0.0 || f_stat.is_nan() || f_stat < 0.0 {
        return f64::NAN;
    }
    let upper = f_test_p_value(f_stat, df1, df2);
    let lower = f_test_p_value(1.0 / f_stat, df2, df1);
    match alternative {
        Alternative::TwoSided => (2.0 * lower.min(upper)).min(1.0),
        Alternative::Greater => upper,
        Alternative::Less => lower,
    }
}

/// Compute confidence interval for variance ratio.
fn compute_variance_ratio_ci(
    estimate: f64,
    df1: f64,
    df2: f64,
    conf_level: f64,
    alternative: Alternative,
) -> (f64, f64) {
    use statrs::distribution::{ContinuousCDF, FisherSnedecor};

    if df1 <= 0.0 || df2 <= 0.0 || estimate.is_nan() || estimate < 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let f_dist = match FisherSnedecor::new(df1, df2) {
        Ok(d) => d,
        Err(_) => return (f64::NAN, f64::NAN),
    };

    let alpha = 1.0 - conf_level;

    match alternative {
        Alternative::TwoSided => {
            // Two-sided CI: use alpha/2 quantiles
            let f_lower = f_dist.inverse_cdf(alpha / 2.0);
            let f_upper = f_dist.inverse_cdf(1.0 - alpha / 2.0);
            // CI is estimate / F_upper to estimate / F_lower
            (estimate / f_upper, estimate / f_lower)
        }
        Alternative::Greater => {
            // One-sided lower bound
            let f_upper = f_dist.inverse_cdf(1.0 - alpha);
            (estimate / f_upper, f64::INFINITY)
        }
        Alternative::Less => {
            // One-sided upper bound
            let f_lower = f_dist.inverse_cdf(alpha);
            (0.0, estimate / f_lower)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};

    /// Relative-only tolerance for an upper tail: Tol::p_value's absolute floor would accept
    /// 0 for a tail far below 1e-16.
    fn tail(n: usize) -> Tol {
        Tol::new(Tol::p_value(n).rel, 0.0)
    }

    #[test]
    fn test_validate_var_test_ci_quantiles_at_large_df() {
        // At df 2e7 R's qf switches to qchisq(p, df1) / df1, which ignores the denominator's
        // variance and is 2.6e-4 off; var.test's limits are pinned to the exact quantile
        // instead (qbeta in the case: F = x / (1 - x) for x ~ Beta(df / 2, df / 2)).
        let (case, _) = crate::test_support::load_case("tail_var_test");
        let (lo, hi) = compute_variance_ratio_ci(1.0, 2e7, 2e7, 0.95, Alternative::TwoSided);
        Check::new(&case)
            .vector("ci_big_exact_estimate_1", &[lo, hi], Tol::new(1e-10, 0.0))
            .finish();
    }

    #[test]
    fn test_validate_var_test_extreme_ratio() {
        // X-17: R's var.test computes 1 - pf and prints 0 here; the exact tails are pinned to
        // pf(lower.tail = FALSE), and to R's printed values within Tol::p_value.
        let (case, ds) = crate::test_support::load_case("tail_var_test");
        let x = numeric_column(ds.df(), "x", NaPolicy::Omit).unwrap().0;
        let y = numeric_column(ds.df(), "y", NaPolicy::Omit).unwrap().0;
        assert_eq!((x.len(), y.len()), (40, 45));
        let run = |alt| var_test(&x, &y, 1.0, alt, 0.95).unwrap();
        let (two, gt, lt) = (
            run(Alternative::TwoSided),
            run(Alternative::Greater),
            run(Alternative::Less),
        );
        Check::new(&case)
            .scalar("statistic", two.f_statistic, Tol::stat(85))
            .vector("df", &[two.df_num, two.df_denom], Tol::exact())
            .scalar("p_two_sided_accurate", two.p_value, tail(85))
            .scalar("var_test_p_two_sided", two.p_value, Tol::p_value(85))
            .scalar("p_upper", gt.p_value, tail(85))
            .scalar("var_test_p_greater", gt.p_value, Tol::p_value(85))
            .scalar("p_lower", lt.p_value, Tol::p_value(85))
            .scalar("var_test_p_less", lt.p_value, Tol::p_value(85))
            .vector(
                "conf_int_two_sided",
                &[two.conf_int_lower, two.conf_int_upper],
                Tol::stat(85),
            )
            .vector("conf_int_greater", &[gt.conf_int_lower], Tol::stat(85))
            .vector("conf_int_less", &[lt.conf_int_upper], Tol::stat(85))
            .finish();
        assert_eq!((gt.conf_int_upper, lt.conf_int_lower), (f64::INFINITY, 0.0));
    }
    use polars::prelude::*;

    #[test]
    fn test_var_test_basic() {
        let x = vec![2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7];
        let y = vec![3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3];

        let result = var_test(&x, &y, 1.0, Alternative::TwoSided, 0.95).unwrap();

        assert_eq!(result.n_x, 7);
        assert_eq!(result.n_y, 7);
        assert_eq!(result.df_num, 6.0);
        assert_eq!(result.df_denom, 6.0);
        assert!(result.f_statistic > 0.0);
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
        assert!(result.conf_int_lower > 0.0);
        assert!(result.conf_int_upper > result.conf_int_lower);
    }

    #[test]
    fn test_var_test_alternatives() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        // y has higher variance (var(y) = 4 * var(x))
        let result_two = var_test(&x, &y, 1.0, Alternative::TwoSided, 0.95).unwrap();
        let result_less = var_test(&x, &y, 1.0, Alternative::Less, 0.95).unwrap();
        let result_greater = var_test(&x, &y, 1.0, Alternative::Greater, 0.95).unwrap();

        // F should be < 1 since var(x) < var(y)
        assert!(result_two.f_statistic < 1.0);

        // p-value for "less" should be small (ratio < 1)
        assert!(result_less.p_value < result_greater.p_value);
    }

    #[test]
    fn test_var_test_from_dataset() {
        let df = df! {
            "x" => [2.1, 2.5, 2.3, 2.8, 2.6],
            "y" => [3.2, 3.5, 3.1, 3.8, 3.4]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_var_test(&dataset, "x", "y", 1.0, Alternative::TwoSided, 0.95).unwrap();

        assert_eq!(result.n_x, 5);
        assert_eq!(result.n_y, 5);
    }

    #[test]
    fn test_var_test_insufficient_data() {
        let x = vec![1.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = var_test(&x, &y, 1.0, Alternative::TwoSided, 0.95);
        assert!(matches!(result, Err(EconError::InsufficientData { .. })));
    }

    #[test]
    fn test_var_test_invalid_ratio() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = var_test(&x, &y, 0.0, Alternative::TwoSided, 0.95);
        assert!(matches!(
            result,
            Err(EconError::InvalidSpecification { .. })
        ));

        let result = var_test(&x, &y, -1.0, Alternative::TwoSided, 0.95);
        assert!(matches!(
            result,
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================
    //
    // These tests compare results with R's var.test() function.

    #[test]
    fn test_validate_var_test_against_r() {
        // R code:
        // x <- c(2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7)
        // y <- c(3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3)
        // var.test(x, y)
        //
        // R output:
        //   var(x) = 0.05809524, var(y) = 0.05809524
        //   F = 1, num df = 6, denom df = 6, p-value = 1
        //   95% CI: (0.1718285, 5.8197566)

        let x = vec![2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7];
        let y = vec![3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3];

        let result = var_test(&x, &y, 1.0, Alternative::TwoSided, 0.95).unwrap();

        // Check F statistic (both have same variance, so F = 1)
        assert!(
            (result.f_statistic - 1.0).abs() < 0.0001,
            "F-stat mismatch: Rust={}, R=1.0",
            result.f_statistic
        );

        // Check degrees of freedom
        assert!((result.df_num - 6.0).abs() < 0.001);
        assert!((result.df_denom - 6.0).abs() < 0.001);

        // Check p-value (should be 1.0 when F = 1)
        assert!(
            (result.p_value - 1.0).abs() < 0.01,
            "p-value mismatch: Rust={}, R=1.0",
            result.p_value
        );

        // Check estimate (ratio of variances = 1)
        assert!(
            (result.estimate - 1.0).abs() < 0.0001,
            "estimate mismatch: Rust={}, R=1.0",
            result.estimate
        );

        // Check sample variances
        assert!(
            (result.var_x - 0.05809524).abs() < 0.0001,
            "var_x mismatch: Rust={}, R=0.05809524",
            result.var_x
        );
        assert!(
            (result.var_y - 0.05809524).abs() < 0.0001,
            "var_y mismatch: Rust={}, R=0.05809524",
            result.var_y
        );

        // Check confidence interval
        assert!(
            (result.conf_int_lower - 0.1718285).abs() < 0.01,
            "CI lower mismatch: Rust={}, R=0.1718285",
            result.conf_int_lower
        );
        assert!(
            (result.conf_int_upper - 5.8197566).abs() < 0.1,
            "CI upper mismatch: Rust={}, R=5.8197566",
            result.conf_int_upper
        );
    }

    #[test]
    fn test_validate_var_test_unequal_variance() {
        // R code:
        // x <- c(1, 2, 3, 4, 5)
        // y <- c(2, 4, 6, 8, 10)
        // var.test(x, y)
        //
        // R output:
        //   F = 0.25, num df = 4, denom df = 4, p-value = 0.208
        //   var(x) = 2.5, var(y) = 10
        //   95% CI: (0.02602938, 2.401132)

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = var_test(&x, &y, 1.0, Alternative::TwoSided, 0.95).unwrap();

        // F = var(x) / var(y) = 2.5 / 10 = 0.25
        assert!(
            (result.f_statistic - 0.25).abs() < 0.0001,
            "F-stat mismatch: Rust={}, R=0.25",
            result.f_statistic
        );

        // Check p-value (two-sided)
        assert!(
            (result.p_value - 0.208).abs() < 0.01,
            "p-value mismatch: Rust={}, R=0.208",
            result.p_value
        );

        // Check variances
        assert!((result.var_x - 2.5).abs() < 0.0001);
        assert!((result.var_y - 10.0).abs() < 0.0001);

        // Check CI
        assert!(
            (result.conf_int_lower - 0.02602938).abs() < 0.01,
            "CI lower mismatch: Rust={}, R=0.02602938",
            result.conf_int_lower
        );
        assert!(
            (result.conf_int_upper - 2.401132).abs() < 0.1,
            "CI upper mismatch: Rust={}, R=2.401132",
            result.conf_int_upper
        );
    }

    #[test]
    fn test_validate_var_test_one_sided() {
        // R code:
        // x <- c(1, 2, 3, 4, 5)
        // y <- c(2, 4, 6, 8, 10)
        // var.test(x, y, alternative = "less")
        //
        // R output: p-value = 0.104, 95% CI: (0, 1.597058)

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = var_test(&x, &y, 1.0, Alternative::Less, 0.95).unwrap();

        // p-value for "less" = P(F < 0.25)
        assert!(
            (result.p_value - 0.104).abs() < 0.01,
            "p-value mismatch: Rust={}, R=0.104",
            result.p_value
        );

        // One-sided CI should have 0 as lower bound
        assert!(result.conf_int_lower == 0.0);

        // Upper bound
        assert!(
            (result.conf_int_upper - 1.597058).abs() < 0.1,
            "CI upper mismatch: Rust={}, R=1.597058",
            result.conf_int_upper
        );
    }

    // ========================================================================
    // Missing values and integer columns (case stat_two_sample_na,
    // validation/r/cases/stats_onesample.R)
    // ========================================================================

    /// X-2/X-8: var.test on Int64 columns with nulls, each sample dropping its own.
    #[test]
    fn test_validate_var_test_two_sample_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let r = run_var_test(&ds, "xi", "yi", 1.0, Alternative::TwoSided, 0.95).unwrap();
        Check::new(&case)
            .scalar("var_f", r.f_statistic, Tol::stat(38))
            .vector("var_df", &[r.df_num, r.df_denom], Tol::exact())
            .scalar("var_p", r.p_value, Tol::p_value(38))
            .vector(
                "var_ci",
                &[r.conf_int_lower, r.conf_int_upper],
                Tol::stat(38),
            )
            .scalar("var_est", r.estimate, Tol::coef(38))
            .scalar("var_x", r.var_x, Tol::stat(38))
            .scalar("var_y", r.var_y, Tol::stat(38))
            .scalar("n_xi", r.n_x as f64, Tol::exact())
            .scalar("n_yi", r.n_y as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_na_var_test_missing_rows_and_errors() {
        let (_case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let r = run_var_test(&ds, "x", "y", 1.0, Alternative::TwoSided, 0.95).unwrap();
        assert_eq!((r.n_x, r.n_y), (7, 7));
        assert_eq!(
            r.missing.by_column,
            vec![("x".to_string(), 15), ("y".to_string(), 15)]
        );
        assert_eq!(
            r.to_string().lines().last().unwrap(),
            "Missing values: 30 of 44 values dropped because of missing values (two samples of \
             22; x: 15, y: 15)"
        );
        for (x, y) in [("g", "y"), ("x", "g")] {
            assert!(
                matches!(run_var_test(&ds, x, y, 1.0, Alternative::TwoSided, 0.95), Err(EconError::NonNumericColumn { column }) if column == "g"),
                "{x} {y}"
            );
        }
        assert!(matches!(
            run_var_test(&ds, "nope", "y", 1.0, Alternative::TwoSided, 0.95),
            Err(EconError::ColumnNotFound { column, .. }) if column == "nope"
        ));
    }
}
