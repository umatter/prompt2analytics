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

use crate::data::columns::{MissingSummary, NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::resource::try_with_capacity;
use crate::stats::ranks::midranks;
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;
use crate::traits::estimator::{normal_cdf, normal_p_value_two_sided};

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
    /// Rows dropped because of missing values (dataset wrapper; nothing for slices). Each
    /// sample drops its own missing values; the summary sums the two columns.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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
        // The two samples drop their missing values separately.
        if let Some(note) = self.missing.two_sample_note() {
            writeln!(f, "Missing values: {note}")?;
        }

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

    // Pool x then y and rank (exact ties, as R's rank())
    let mut z: Vec<f64> = try_with_capacity(total_n, "Mood pooled sample")?;
    z.extend_from_slice(&x_clean);
    z.extend_from_slice(&y_clean);
    let (ranks, tie_counts) = midranks(&z)?;
    let has_ties = tie_counts.iter().any(|&t| t > 1);

    // Mean rank
    let mean_rank = (total_n as f64 + 1.0) / 2.0;

    // T = sum over sample x of the scores (r_i - (N+1)/2)^2 (Conover 1971, p. 234). Under
    // ties R averages the scores of the positions a tie block occupies (Mielke 1967): for a
    // block of size t with midrank r that average is (r - (N+1)/2)^2 + (t^2 - 1)/12.
    let t_stat: f64 = if has_ties {
        let mut sorted: Vec<f64> = try_with_capacity(total_n, "Mood ranks")?;
        sorted.extend_from_slice(&ranks);
        sorted.sort_by(f64::total_cmp);
        ranks[..m]
            .iter()
            .map(|&r| {
                let lo = sorted.partition_point(|&v| v < r);
                let hi = sorted.partition_point(|&v| v <= r);
                let t = (hi - lo) as f64;
                (r - mean_rank).powi(2) + (t * t - 1.0) / 12.0
            })
            .sum()
    } else {
        ranks[..m].iter().map(|&r| (r - mean_rank).powi(2)).sum()
    };

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
    let p_value = match alternative {
        Alternative::TwoSided => normal_p_value_two_sided(z_score),
        Alternative::Less => normal_cdf(z_score),
        Alternative::Greater => normal_cdf(-z_score),
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
        missing: MissingSummary::none(total_n),
    })
}

/// Wrapper function for running Mood test on a Dataset.
///
/// Each column drops its own missing values (null or NaN), as `mood.test(x, y)` does, and the
/// drop is reported in `missing`. Integer, decimal and boolean columns are accepted; a text
/// column is refused (`EconError::NonNumericColumn`), as are infinite values
/// (`EconError::NonFiniteValues`).
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
    let df = dataset.df();
    let (x, missing_x) = numeric_column(df, x_col, NaPolicy::Omit)?;
    let (y, missing_y) = numeric_column(df, y_col, NaPolicy::Omit)?;
    let mut result = mood_test(&x, &y, alternative.unwrap_or(Alternative::TwoSided))?;
    result.missing = MissingSummary::merge_samples(&missing_x, &missing_y);
    Ok(result)
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
    // Missing values and integer columns (case stat_two_sample_na,
    // validation/r/cases/stats_onesample.R)
    // ========================================================================

    /// X-2/X-8: mood.test on a tie-free Int64 pair with nulls, each sample dropping its own.
    #[test]
    fn test_validate_mood_test_two_sample_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let r = run_mood_test(&ds, "mi", "mj", None).unwrap();
        Check::new(&case)
            .scalar("mood_nt_z", r.z_score, Tol::stat(27))
            .scalar("mood_nt_p", r.p_value, Tol::p_value(27))
            .scalar("mood_nt_T", r.statistic, Tol::stat(27))
            .scalar("n_mi", r.n_x as f64, Tol::exact())
            .scalar("n_mj", r.n_y as f64, Tol::exact())
            .finish();
        assert!(!r.has_ties);
        // the tied pair xi / yi (Mielke's averaged scores)
        let t = run_mood_test(&ds, "xi", "yi", None).unwrap();
        assert!(t.has_ties);
        Check::new(&case)
            .scalar("mood_z", t.z_score, Tol::stat(t.n_total))
            .scalar("mood_p", t.p_value, Tol::p_value(t.n_total))
            .finish();
    }

    #[test]
    fn test_validate_na_mood_test_missing_rows_and_errors() {
        let (_case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let r = run_mood_test(&ds, "x", "y", None).unwrap();
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
                matches!(run_mood_test(&ds, x, y, None), Err(EconError::NonNumericColumn { column }) if column == "g"),
                "{x} {y}"
            );
        }
        assert!(matches!(
            run_mood_test(&ds, "x", "nope", None),
            Err(EconError::ColumnNotFound { column, .. }) if column == "nope"
        ));
    }

    #[test]
    fn test_validate_mood_two_sample() {
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_two_sample");
        let sets = ["basic", "scale", "manual", "sim"];
        let mut v: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
        for set in sets {
            let x = column_where(&ds, "value", &[("set", set), ("group", "x")]);
            let y = column_where(&ds, "value", &[("set", set), ("group", "y")]);
            let two = mood_test(&x, &y, Alternative::TwoSided).unwrap();
            assert!(!two.has_ties, "{set}");
            v.entry("z").or_default().push(two.z_score);
            v.entry("p").or_default().push(two.p_value);
            v.entry("less")
                .or_default()
                .push(mood_test(&x, &y, Alternative::Less).unwrap().p_value);
            v.entry("greater")
                .or_default()
                .push(mood_test(&x, &y, Alternative::Greater).unwrap().p_value);
            v.entry("t").or_default().push(two.statistic);
        }
        let ties = ["ties", "serum"];
        for set in ties {
            let x = column_where(&ds, "value", &[("set", set), ("group", "x")]);
            let y = column_where(&ds, "value", &[("set", set), ("group", "y")]);
            let r = mood_test(&x, &y, Alternative::TwoSided).unwrap();
            assert!(r.has_ties);
            v.entry("z_ties").or_default().push(r.z_score);
            v.entry("p_ties").or_default().push(r.p_value);
        }
        Check::new(&case)
            .named("mood_z_ties", &ties, &v["z_ties"], Tol::stat(30))
            .named("mood_p_ties", &ties, &v["p_ties"], Tol::p_value(30))
            .named("mood_z", &sets, &v["z"], Tol::stat(6))
            .named("mood_p", &sets, &v["p"], Tol::p_value(6))
            .named("mood_p_less", &sets, &v["less"], Tol::p_value(6))
            .named("mood_p_greater", &sets, &v["greater"], Tol::p_value(6))
            .named("mood_T", &sets, &v["t"], Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_mood_small_scale() {
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_small_groups");
        let x = column_where(&ds, "ynt", &[("g", "a")]);
        let y = column_where(&ds, "ynt", &[("g", "b")]);
        let r = mood_test(&x, &y, Alternative::TwoSided).unwrap();
        assert!(!r.has_ties);
        Check::new(&case)
            .scalar("mood_z", r.z_score, Tol::stat(14))
            .scalar("mood_p", r.p_value, Tol::p_value(14))
            .scalar("mood_T", r.statistic, Tol::exact())
            .finish();
    }

    /// Heavy ties: T from R's averaged scores, z, and the three p-values. Fails before: T
    /// summed the squared deviations of the midranks.
    #[test]
    fn test_validate_mood_heavy_ties_t4() {
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_scale_t4");
        let x = column_where(&ds, "value", &[("set", "heavy"), ("group", "x")]);
        let y = column_where(&ds, "value", &[("set", "heavy"), ("group", "y")]);
        let r = mood_test(&x, &y, Alternative::TwoSided).unwrap();
        assert!(r.has_ties);
        Check::new(&case)
            .scalar("mood_heavy_T", r.statistic, Tol::stat(12))
            .scalar("mood_heavy_z", r.z_score, Tol::stat(12))
            .scalar("mood_heavy_p", r.p_value, Tol::p_value(12))
            .scalar(
                "mood_heavy_p_less",
                mood_test(&x, &y, Alternative::Less).unwrap().p_value,
                Tol::p_value(12),
            )
            .scalar(
                "mood_heavy_p_greater",
                mood_test(&x, &y, Alternative::Greater).unwrap().p_value,
                Tol::p_value(12),
            )
            .finish();
    }
}
