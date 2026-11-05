//! Kruskal-Wallis rank sum test
//!
//! A non-parametric test for comparing the medians of two or more independent samples.
//! This is the non-parametric alternative to one-way ANOVA.
//! Ranks as `stats::ranks::midranks` (exact ties, as R), summed per group without storing them.
//!
//! # References
//!
//! - Kruskal, W. H. & Wallis, W. A. (1952). "Use of Ranks in One-Criterion Variance Analysis".
//!   Journal of the American Statistical Association, 47(260), 583-621.
//! - Hollander, M. & Wolfe, D. A. (1973). Nonparametric Statistical Methods.
//!   New York: John Wiley & Sons. Pages 115-120.
//! - R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/kruskal.test.html

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource::try_with_capacity;
use crate::stats::anova::values_by_level;
use crate::stats::ranks::{rank_sums_by_group, tie_cube_sum};
use crate::traits::estimator::chi_squared_p_value;
use serde::{Deserialize, Serialize};

/// Result of the Kruskal-Wallis rank sum test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KruskalWallisResult {
    /// The H statistic (chi-squared approximation)
    pub statistic: f64,
    /// Degrees of freedom (k - 1)
    pub df: usize,
    /// P-value from chi-squared distribution
    pub p_value: f64,
    /// Number of groups
    pub n_groups: usize,
    /// Total sample size
    pub n_total: usize,
    /// Sample sizes per group
    pub group_sizes: Vec<usize>,
    /// Group names/labels
    pub group_names: Vec<String>,
    /// Rank sums per group
    pub rank_sums: Vec<f64>,
    /// Mean ranks per group
    pub mean_ranks: Vec<f64>,
    /// Whether ties were present
    pub has_ties: bool,
    /// Tie correction factor (1.0 if no ties)
    pub tie_correction: f64,
    /// Rows dropped because the value or the group label is missing (R's na.omit);
    /// empty for the array-level function.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

/// Compute the Kruskal-Wallis rank sum test for grouped data.
///
/// # Arguments
///
/// * `groups` - Vector of (group_name, values) tuples
///
/// # Returns
///
/// A `KruskalWallisResult` containing the test statistic, degrees of freedom, and p-value.
///
/// # Example
///
/// ```
/// use p2a_core::stats::kruskal::{kruskal_test, KruskalWallisResult};
///
/// let groups = vec![
///     ("A".to_string(), vec![1.0, 2.0, 3.0]),
///     ("B".to_string(), vec![4.0, 5.0, 6.0]),
///     ("C".to_string(), vec![7.0, 8.0, 9.0]),
/// ];
///
/// let result = kruskal_test(&groups).unwrap();
/// println!("H = {}, p = {}", result.statistic, result.p_value);
/// ```
pub fn kruskal_test(groups: &[(String, Vec<f64>)]) -> EconResult<KruskalWallisResult> {
    // Need at least 2 groups
    if groups.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: groups.len(),
            context: "Kruskal-Wallis test requires at least 2 groups".to_string(),
        });
    }

    // Filter out NaN/Inf from each group
    let clean_groups: Vec<(String, Vec<f64>)> = groups
        .iter()
        .map(|(name, vals)| {
            let clean: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
            (name.clone(), clean)
        })
        .collect();

    // Check each group has at least one observation
    for (name, vals) in &clean_groups {
        if vals.is_empty() {
            return Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: format!("Group '{}' has no valid observations", name),
            });
        }
    }

    let k = clean_groups.len();
    let group_sizes: Vec<usize> = clean_groups.iter().map(|(_, v)| v.len()).collect();
    let group_names: Vec<String> = clean_groups.iter().map(|(n, _)| n.clone()).collect();
    let n_total: usize = group_sizes.iter().sum();

    // Pool the observations in group order; the midrank sums per group (exact ties, as R's
    // rank()).
    let mut values: Vec<f64> = try_with_capacity(n_total, "Kruskal-Wallis values")?;
    let mut group_of: Vec<usize> = try_with_capacity(n_total, "Kruskal-Wallis groups")?;
    for (group_idx, (_, vals)) in clean_groups.iter().enumerate() {
        values.extend_from_slice(vals);
        group_of.extend(std::iter::repeat_n(group_idx, vals.len()));
    }
    let (rank_sums, tie_sizes) = rank_sums_by_group(&values, &group_of, k)?;
    if tie_sizes.len() == 1 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Kruskal-Wallis test: all {n_total} observations are tied, so the statistic is undefined (kruskal.test returns NaN)"
            ),
        });
    }

    // Tie correction C = 1 - Σ(ti³ - ti) / (N³ - N), strictly positive once two values differ
    let n = n_total as f64;
    let tie_correction = 1.0 - tie_cube_sum(&tie_sizes) / (n * n * n - n);
    let has_ties = tie_sizes.iter().any(|&t| t > 1);

    // Compute mean ranks
    let mean_ranks: Vec<f64> = rank_sums
        .iter()
        .zip(group_sizes.iter())
        .map(|(&sum, &size)| sum / size as f64)
        .collect();

    // Compute H statistic
    // H = (12 / (N(N+1))) × Σ(Rj²/nj) - 3(N+1)
    let sum_r2_over_n: f64 = rank_sums
        .iter()
        .zip(group_sizes.iter())
        .map(|(&r, &n)| r * r / n as f64)
        .sum();

    let h_uncorrected = (12.0 / (n * (n + 1.0))) * sum_r2_over_n - 3.0 * (n + 1.0);

    let h = h_uncorrected / tie_correction;

    // Degrees of freedom
    let df = k - 1;

    // P-value from chi-squared distribution
    let p_value = chi_squared_p_value(h, df as f64);

    Ok(KruskalWallisResult {
        statistic: h,
        df,
        p_value,
        n_groups: k,
        n_total,
        group_sizes,
        group_names,
        rank_sums,
        mean_ranks,
        has_ties,
        tie_correction,
        missing: MissingSummary::none(n_total),
    })
}

/// Run Kruskal-Wallis test on a Dataset with grouping variable (R `kruskal.test(value ~ group)`).
///
/// Rows with a missing value or group label are dropped, as R's `na.omit`, and reported in
/// `missing`; a missing label is never a group of its own. Integer values are accepted, and
/// the groups follow R `factor()` order (numeric labels numerically: 2 before 10).
///
/// # Arguments
///
/// * `dataset` - The dataset containing the data
/// * `value_col` - Column name for the values to compare
/// * `group_col` - Column name for the grouping variable
///
/// # Returns
///
/// A `KruskalWallisResult` containing the test results.
pub fn run_kruskal_test(
    dataset: &Dataset,
    value_col: &str,
    group_col: &str,
) -> EconResult<KruskalWallisResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(value_col)
        .factor(group_col)
        .na(NaPolicy::Omit)
        .extract()?;
    let groups = values_by_level(cols.numeric(value_col), cols.factor(group_col));
    let mut result = kruskal_test(&groups)?;
    result.missing = cols.missing().clone();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_kruskal_basic() {
        // Simple test with clearly different groups
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0]),
            ("B".to_string(), vec![4.0, 5.0, 6.0]),
            ("C".to_string(), vec![7.0, 8.0, 9.0]),
        ];

        let result = kruskal_test(&groups).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.n_total, 9);
        assert_eq!(result.df, 2);
        assert!(result.statistic > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        // With perfectly separated groups, H should be high and p should be low
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_kruskal_identical_groups() {
        // Groups with similar distributions should have p > 0.05
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![1.5, 2.5, 3.5, 4.5, 5.5]),
            ("C".to_string(), vec![1.2, 2.2, 3.2, 4.2, 5.2]),
        ];

        let result = kruskal_test(&groups).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.df, 2);
        // Similar groups should have high p-value
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_kruskal_with_ties() {
        // Test with tied values
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 2.0, 3.0]),
            ("B".to_string(), vec![2.0, 3.0, 3.0, 4.0]),
        ];

        let result = kruskal_test(&groups).unwrap();

        assert!(result.has_ties);
        assert!(result.tie_correction < 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_kruskal_two_groups() {
        // Two groups (equivalent to Mann-Whitney U)
        let groups = vec![
            ("Control".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("Treatment".to_string(), vec![6.0, 7.0, 8.0, 9.0, 10.0]),
        ];

        let result = kruskal_test(&groups).unwrap();

        assert_eq!(result.n_groups, 2);
        assert_eq!(result.df, 1);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_kruskal_validates_groups() {
        // Should require at least 2 groups
        let groups = vec![("A".to_string(), vec![1.0, 2.0, 3.0])];

        let result = kruskal_test(&groups);
        assert!(result.is_err());
    }

    #[test]
    fn test_kruskal_statistic_hand_computed() {
        // Manual calculation verification
        // Groups: A = [1, 2], B = [3, 4], C = [5, 6]
        // Combined: 1, 2, 3, 4, 5, 6 with ranks 1, 2, 3, 4, 5, 6
        // R_A = 1 + 2 = 3, R_B = 3 + 4 = 7, R_C = 5 + 6 = 11
        // n = 6, each n_j = 2
        // H = (12/(6*7)) * (9/2 + 49/2 + 121/2) - 3*7
        //   = (12/42) * (89.5) - 21
        //   = 0.2857 * 89.5 - 21
        //   = 25.57 - 21 = 4.57

        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0]),
            ("B".to_string(), vec![3.0, 4.0]),
            ("C".to_string(), vec![5.0, 6.0]),
        ];

        let result = kruskal_test(&groups).unwrap();

        assert_eq!(result.n_total, 6);
        assert_eq!(result.n_groups, 3);
        assert!(!result.has_ties);
        assert!((result.tie_correction - 1.0).abs() < 1e-10);

        // H = (12/42) * 89.5 - 21 = 32/7
        assert!(
            (result.statistic - 32.0 / 7.0).abs() < 1e-12,
            "{}",
            result.statistic
        );
    }

    #[test]
    fn test_kruskal_handles_nan() {
        let groups = vec![
            ("A".to_string(), vec![1.0, f64::NAN, 3.0]),
            ("B".to_string(), vec![4.0, 5.0, f64::INFINITY]),
        ];

        let result = kruskal_test(&groups).unwrap();

        // Should filter out NaN and Inf, leaving 2 values in each group
        assert_eq!(result.n_total, 4);
    }

    #[test]
    fn test_run_kruskal_from_dataset() {
        let df = df! {
            "value" => [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "group" => ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_kruskal_test(&dataset, "value", "group").unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.n_total, 9);
        // Groups are well separated, so p should be low
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_kruskal_tie_correction_hand_computed() {
        // Test with extensive ties to verify correction
        let groups = vec![
            ("A".to_string(), vec![1.0, 1.0, 2.0, 2.0, 3.0]),
            ("B".to_string(), vec![2.0, 2.0, 3.0, 3.0, 4.0]),
            ("C".to_string(), vec![3.0, 3.0, 4.0, 4.0, 5.0]),
        ];

        let result = kruskal_test(&groups).unwrap();

        assert!(result.has_ties);
        // Tie groups 1 (2), 2 (4), 3 (5), 4 (3), 5 (1) of N = 15: 1 - 210/3360
        assert!(
            (result.tie_correction - 0.9375).abs() < 1e-15,
            "{}",
            result.tie_correction
        );
    }

    // ========================================================================
    // R-pinned: missing values and labels (T2-WP07, case stat_oneway_na_labels)
    // ========================================================================
    //
    // kruskal.test(v9 ~ g9): g9 is missing in 3 of 9 rows, which R drops; the labels are
    // "a" and "b" (never a level named null, never quoted).

    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};

    /// The case frame plus `gnum` (Int64): 10 where g9 is "a", 2 where it is "b", else null.
    fn with_numeric_groups(ds: &Dataset) -> Dataset {
        let mut df = ds.df().clone();
        let gnum: Vec<Option<i64>> = df
            .column("g9")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .map(|g| match g {
                Some("a") => Some(10),
                Some("b") => Some(2),
                _ => None,
            })
            .collect();
        df.with_column(Series::new("gnum".into(), gnum)).unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_validate_oneway_na_labels_kruskal() {
        let (case, ds) = load_case("stat_oneway_na_labels");
        let r = run_kruskal_test(&ds, "v9", "g9").unwrap();
        let sizes: Vec<f64> = r.group_sizes.iter().map(|&n| n as f64).collect();
        Check::new(&case)
            .scalar("kw_chisq", r.statistic, Tol::stat(6))
            .scalar("kw_df", r.df as f64, Tol::exact())
            .scalar("kw_p", r.p_value, Tol::p_value(6))
            .named("kw_group_n", &r.group_names, &sizes, Tol::exact())
            .finish();
        assert_eq!(r.group_names, ["a", "b"]);
        assert_eq!(r.missing.n_dropped, 3);
        assert_eq!(r.missing.by_column, vec![("g9".to_string(), 3)]);
    }

    /// Column-type fixture: integer labels in numeric order (2 < 10), a text value column and
    /// an absent column refused with the variant that names the problem.
    #[test]
    fn test_validate_na_kruskal_dtypes_and_errors() {
        let (_case, ds) = load_case("stat_oneway_na_labels");
        let by_label = run_kruskal_test(&ds, "v9", "g9").unwrap();
        let ds = with_numeric_groups(&ds);
        let by_number = run_kruskal_test(&ds, "v9", "gnum").unwrap();
        assert_eq!(by_number.group_names, ["2", "10"]);
        assert_eq!(by_number.group_sizes, [3, 3]);
        assert!((by_number.statistic - by_label.statistic).abs() <= 1e-12 * by_label.statistic);
        assert!(matches!(
            run_kruskal_test(&ds, "g9", "gnum"),
            Err(EconError::NonNumericColumn { ref column }) if column == "g9"
        ));
        assert!(matches!(
            run_kruskal_test(&ds, "v9", "nope"),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
        assert!(matches!(
            run_kruskal_test(&ds, "nope", "g9"),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
    }

    #[test]
    fn test_validate_kruskal_small_scale() {
        // R kruskal.test(y ~ g) on values of order 1e-12 (STAT-53). Before: H = 0, p = 1,
        // tie_correction = 0.
        let (case, ds) = load_case("stat_rank_small_groups");
        let r = run_kruskal_test(&ds, "y", "g").unwrap();
        let unit = run_kruskal_test(&ds, "y_unit", "g").unwrap();
        assert!(r.has_ties);
        assert_eq!(
            r.statistic, unit.statistic,
            "the statistic does not depend on the unit"
        );
        let n = r.n_total; // 21
        Check::new(&case)
            .scalar("kw_chisq", r.statistic, Tol::stat(n))
            .scalar("kw_df", r.df as f64, Tol::exact())
            .scalar("kw_p", r.p_value, Tol::p_value(n))
            .scalar("kw_unit_chisq", unit.statistic, Tol::stat(n))
            .scalar("kw_tie_correction", r.tie_correction, Tol::stat(n))
            .named("kw_rank_sums", &r.group_names, &r.rank_sums, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_kruskal_hw() {
        // R ?kruskal.test example (Hollander & Wolfe 1973, p. 116)
        let (case, ds) = load_case("stat_rank_kruskal_hw");
        let r = run_kruskal_test(&ds, "value", "group").unwrap();
        let n = r.n_total; // 14
        Check::new(&case)
            .scalar("chisq", r.statistic, Tol::stat(n))
            .scalar("df", r.df as f64, Tol::exact())
            .scalar("p", r.p_value, Tol::p_value(n))
            .named("rank_sums", &r.group_names, &r.rank_sums, Tol::exact())
            .finish();
    }

    #[test]
    fn test_kruskal_all_tied_is_error() {
        // R: kruskal.test(list(c(1, 1, 1), c(1, 1))) gives chi-squared NaN, p NA. Before: Ok
        // with H 0, p 1.
        let groups = vec![
            ("A".to_string(), vec![2.5, 2.5, 2.5]),
            ("B".to_string(), vec![2.5, 2.5]),
        ];
        match kruskal_test(&groups) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("all 5 observations are tied"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        // Distinct values far below 1e-10 are not ties: R kruskal.test(list(c(1e-300, 2e-300),
        // 3e-300)) = 1.5
        let tiny = vec![
            ("A".to_string(), vec![1e-300, 2e-300]),
            ("B".to_string(), vec![3e-300]),
        ];
        let r = kruskal_test(&tiny).unwrap();
        assert!(!r.has_ties);
        assert!((r.statistic - 1.5).abs() < 1e-12, "{}", r.statistic);
    }
}
