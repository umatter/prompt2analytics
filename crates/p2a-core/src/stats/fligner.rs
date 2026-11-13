//! Fligner-Killeen test for homogeneity of variances.
//!
//! A robust, non-parametric test for equal variances across groups.
//! Uses median-centered ranks and is more robust to departures from
//! normality than Bartlett's test.
//!
//! # References
//!
//! - Fligner, M. A. and Killeen, T. J. (1976). Distribution-free two-sample
//!   tests for scale. *Journal of the American Statistical Association*, 71(353), 210-213.
//! - Conover, W. J., Johnson, M. E., and Johnson, M. M. (1981). A comparative
//!   study of tests for homogeneity of variances. *Technometrics*, 23(4), 351-361.
//! - R Core Team. `stats::fligner.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fligner.test.html>
//!
//! # Mathematical Background
//!
//! ## Algorithm
//!
//! 1. For each group i with observations x_{ij}:
//!    - Compute group median m_i
//!    - Compute absolute deviations: a_{ij} = |x_{ij} - m_i|
//!
//! 2. Rank all absolute deviations across groups (1 to N)
//!
//! 3. Transform ranks to normal scores:
//!    ```text
//!    z_{ij} = Φ^{-1}((1 + r_{ij}) / (2 * (N + 1)))
//!    ```
//!    where Φ^{-1} is the standard normal quantile function
//!
//! 4. Compute the test statistic (chi-squared):
//!    ```text
//!    χ² = (Σ n_i * (z̄_i - z̄)²) / V
//!    ```
//!    where V is the variance of all z scores
//!
//! Under H₀ (equal variances), χ² ~ χ²(k-1) where k is the number of groups.

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource::try_with_capacity;
use crate::stats::bartlett::values_by_level;
use crate::stats::ranks::midranks;
use crate::traits::SignificanceLevel;
use crate::traits::estimator::chi_squared_p_value;

/// Result of Fligner-Killeen test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlignerResult {
    /// Description of the test
    pub test_name: String,
    /// Chi-squared statistic
    pub chi_squared: f64,
    /// Degrees of freedom (k - 1)
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Number of groups
    pub n_groups: usize,
    /// Group sizes
    pub group_sizes: Vec<usize>,
    /// Group names (if available)
    pub group_names: Vec<String>,
    /// Rows dropped because of a missing value or a missing group label (dataset wrapper;
    /// nothing for slices).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for FlignerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        writeln!(
            f,
            "Fligner-Killeen:med chi-squared = {:.4}, df = {}, p-value = {:.5} {}",
            self.chi_squared,
            self.df,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        if !self.group_names.is_empty() {
            writeln!(
                f,
                "Groups: {} (n = {:?})",
                self.group_names.join(", "),
                self.group_sizes
            )?;
        } else {
            writeln!(f, "Number of groups: {}", self.n_groups)?;
            writeln!(f, "Group sizes: {:?}", self.group_sizes)?;
        }
        writeln!(f)?;

        writeln!(f, "H₀: All group variances are equal")?;
        writeln!(f, "H₁: At least one group has different variance")?;
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Perform Fligner-Killeen test for homogeneity of variances.
///
/// # Arguments
/// * `groups` - Vector of (group_name, group_data) tuples
///
/// # Example
/// ```ignore
/// let groups = vec![
///     ("A".to_string(), vec![2.1, 2.5, 2.3, 2.8, 2.6]),
///     ("B".to_string(), vec![3.2, 3.5, 3.1, 3.8, 3.4]),
/// ];
/// let result = fligner_test(&groups)?;
/// ```
///
/// # References
/// - R equivalent: `fligner.test(value ~ group, data = data)`
pub fn fligner_test(groups: &[(String, Vec<f64>)]) -> EconResult<FlignerResult> {
    let k = groups.len();
    if k < 2 {
        return Err(EconError::InvalidSpecification {
            message: "At least 2 groups required".to_string(),
        });
    }

    // Check group sizes
    let mut group_sizes = Vec::with_capacity(k);
    let mut group_names = Vec::with_capacity(k);
    let mut total_n = 0usize;

    for (name, data) in groups {
        if data.len() < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: data.len(),
                context: format!("Group '{}' requires at least 2 observations", name),
            });
        }
        group_sizes.push(data.len());
        group_names.push(name.clone());
        total_n += data.len();
    }

    // Step 1: Compute absolute deviations from group medians. A non-finite value is refused
    // in the same pass, at the first one of the first group that holds one (the median of
    // such a group is computed but never used).
    let mut all_deviations: Vec<f64> = try_with_capacity(total_n, "Fligner-Killeen deviations")?;
    let mut group_indices: Vec<usize> = try_with_capacity(total_n, "Fligner-Killeen groups")?;

    for (group_idx, (name, data)) in groups.iter().enumerate() {
        let median = compute_median(data);
        for &x in data {
            if !x.is_finite() {
                let what = if x.is_nan() {
                    "NaN"
                } else {
                    "an infinite value"
                };
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Fligner-Killeen test: group '{name}' contains {what}; the test needs finite values (run_fligner_test drops missing values, as fligner.test does; infinite values are refused)"
                    ),
                });
            }
            all_deviations.push((x - median).abs());
            group_indices.push(group_idx);
        }
    }

    // Step 2: Rank all absolute deviations (exact ties, as R's rank())
    let (ranks, _) = midranks(&all_deviations)?;

    // Step 3: Transform ranks to normal scores
    // R uses: a(i) = qnorm((1 + i/(n+1))/2)
    // where i is the rank (1 to n)
    // 1 <= r <= n, so the argument is in (0.5, 1) and needs no validation.
    let n_f = total_n as f64;
    let normal = Normal::standard();
    let mut z_scores: Vec<f64> = try_with_capacity(total_n, "Fligner-Killeen scores")?;
    for &r in &ranks {
        z_scores.push(normal.inverse_cdf((1.0 + r / (n_f + 1.0)) / 2.0));
    }

    // Step 4: Compute test statistic
    // Overall mean of z scores
    let z_mean: f64 = z_scores.iter().sum::<f64>() / n_f;

    // Group means of z scores
    let mut group_z_sums: Vec<f64> = vec![0.0; k];
    for (i, &z) in z_scores.iter().enumerate() {
        group_z_sums[group_indices[i]] += z;
    }
    let group_z_means: Vec<f64> = group_z_sums
        .iter()
        .zip(&group_sizes)
        .map(|(sum, &n)| sum / n as f64)
        .collect();

    // Variance of all z scores
    let variance: f64 = z_scores.iter().map(|&z| (z - z_mean).powi(2)).sum::<f64>() / (n_f - 1.0);

    // Chi-squared statistic
    let chi_squared: f64 = group_sizes
        .iter()
        .zip(&group_z_means)
        .map(|(&n_i, &z_i)| n_i as f64 * (z_i - z_mean).powi(2))
        .sum::<f64>()
        / variance;

    // Degrees of freedom
    let df = k - 1;

    // P-value from chi-squared distribution
    let p_value = chi_squared_p_value(chi_squared, df as f64);

    Ok(FlignerResult {
        test_name: "Fligner-Killeen test of homogeneity of variances".to_string(),
        chi_squared,
        df,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n_groups: k,
        group_sizes,
        group_names,
        missing: MissingSummary::none(total_n),
    })
}

/// Perform Fligner-Killeen test using dataset columns.
///
/// The value and group columns are read in one request: a row with a missing value (null or
/// NaN) or a missing label is dropped from both, as `fligner.test(x ~ g)` does
/// (`model.frame`, `na.omit`), and the drop is reported in `missing`. Groups come in R's
/// factor order (numeric labels numerically, text in byte order) and may have any label
/// type. Integer, decimal and boolean values are accepted; a text value column is refused
/// (`EconError::NonNumericColumn`), as are infinite values (`EconError::NonFiniteValues`).
///
/// # Arguments
/// * `dataset` - The dataset
/// * `value_col` - Name of the column with values
/// * `group_col` - Name of the column with group labels
pub fn run_fligner_test(
    dataset: &Dataset,
    value_col: &str,
    group_col: &str,
) -> EconResult<FlignerResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(value_col)
        .factor(group_col)
        .na(NaPolicy::Omit)
        .extract()?;
    let groups = values_by_level(cols.numeric(value_col), cols.factor(group_col))?;
    let mut result = fligner_test(&groups)?;
    result.missing = cols.missing().clone();
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Compute median of a slice.
fn compute_median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_fligner_test_basic() {
        let groups = vec![
            ("A".to_string(), vec![2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7]),
            ("B".to_string(), vec![3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3]),
        ];

        let result = fligner_test(&groups).unwrap();

        assert_eq!(result.n_groups, 2);
        assert_eq!(result.df, 1);
        assert!(result.chi_squared >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_fligner_test_three_groups() {
        let groups = vec![
            ("A".to_string(), vec![2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7]),
            ("B".to_string(), vec![3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3]),
            ("C".to_string(), vec![4.1, 5.2, 3.9, 4.5, 4.8, 5.0]),
        ];

        let result = fligner_test(&groups).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.df, 2);
    }

    #[test]
    fn test_fligner_test_insufficient_groups() {
        let groups = vec![("A".to_string(), vec![1.0, 2.0, 3.0])];

        assert!(fligner_test(&groups).is_err());
    }

    #[test]
    fn test_fligner_test_small_group() {
        let groups = vec![
            ("A".to_string(), vec![1.0]), // Only 1 observation
            ("B".to_string(), vec![1.0, 2.0, 3.0]),
        ];

        assert!(fligner_test(&groups).is_err());
    }

    #[test]
    fn test_fligner_from_dataset() {
        let df = df! {
            "value" => [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "group" => ["A", "A", "A", "B", "B", "B"]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_fligner_test(&dataset, "value", "group").unwrap();

        assert_eq!(result.n_groups, 2);
        assert_eq!(result.df, 1);
    }

    // ========================================================================
    // Missing values and integer columns (case stat_two_sample_na,
    // validation/r/cases/stats_onesample.R)
    // ========================================================================

    /// X-2/X-8: fligner.test(xi ~ g) on an Int64 response; xi and g are missing in rows
    /// 21-22 (model.frame drops both).
    #[test]
    fn test_validate_fligner_test_two_sample_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let r = run_fligner_test(&ds, "xi", "g").unwrap();
        let sizes: Vec<f64> = r.group_sizes.iter().map(|&n| n as f64).collect();
        Check::new(&case)
            .scalar("fligner_chisq", r.chi_squared, Tol::stat(20))
            .scalar("fligner_df", r.df as f64, Tol::exact())
            .scalar("fligner_p", r.p_value, Tol::p_value(20))
            .named("group_n", &r.group_names, &sizes, Tol::exact())
            .scalar("n_xi_g", sizes.iter().sum::<f64>(), Tol::exact())
            .finish();
        assert_eq!(r.group_names, ["a", "b", "c"]);
        assert_eq!(
            r.missing.by_column,
            vec![("xi".to_string(), 2), ("g".to_string(), 1)]
        );
        assert_eq!(r.missing.n_dropped, 2);
    }

    #[test]
    fn test_validate_na_fligner_test_errors() {
        let (_case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let r = run_fligner_test(&ds, "xi", "g").unwrap();
        assert_eq!(
            r.to_string().lines().last().unwrap(),
            format!("Missing values: {}", r.missing.note().unwrap())
        );
        assert!(matches!(
            run_fligner_test(&ds, "g", "id"),
            Err(EconError::NonNumericColumn { column }) if column == "g"
        ));
        assert!(matches!(
            run_fligner_test(&ds, "nope", "g"),
            Err(EconError::ColumnNotFound { column, .. }) if column == "nope"
        ));
        assert!(matches!(
            run_fligner_test(&ds, "xi", "nope"),
            Err(EconError::ColumnNotFound { column, .. }) if column == "nope"
        ));
    }

    #[test]
    fn test_validate_fligner_small_scale() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_small_groups");
        let r = run_fligner_test(&ds, "y", "g").unwrap();
        Check::new(&case)
            .scalar("fligner_chisq", r.chi_squared, Tol::stat(21))
            .scalar("fligner_df", r.df as f64, Tol::exact())
            .scalar("fligner_p", r.p_value, Tol::p_value(21))
            .finish();
    }

    #[test]
    fn test_validate_fligner_examples() {
        // The two examples the old tests quoted at tolerance 0.1 ("three", "two") and R's
        // ?fligner.test InsectSprays.
        use crate::stats::ranks::test_data::{column_where, levels_where};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_fligner");
        let sets = ["three", "two", "insect"];
        let (mut chi, mut df, mut p) = (Vec::new(), Vec::new(), Vec::new());
        for set in sets {
            let groups: Vec<(String, Vec<f64>)> = levels_where(&ds, "group", &[("set", set)])
                .into_iter()
                .map(|g| {
                    let v = column_where(&ds, "value", &[("set", set), ("group", g.as_str())]);
                    (g, v)
                })
                .collect();
            let r = fligner_test(&groups).unwrap();
            chi.push(r.chi_squared);
            df.push(r.df as f64);
            p.push(r.p_value);
        }
        Check::new(&case)
            .named("chisq", &sets, &chi, Tol::stat(10))
            .named("df", &sets, &df, Tol::exact())
            .named("p", &sets, &p, Tol::p_value(10))
            .finish();
    }

    #[test]
    fn test_fligner_rejects_nan() {
        let groups = vec![
            ("A".to_string(), vec![1.0, f64::NAN, 3.0]),
            ("B".to_string(), vec![2.0, 3.0, 5.0]),
        ];
        match fligner_test(&groups) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("'A'") && message.contains("NaN"),
                    "{message}"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_fligner_rejects_infinite_values() {
        // An infinite group median made Inf - Inf = NaN deviations, refused inside midranks
        // with an internal deviation index instead of the user's group.
        let groups = vec![
            ("A".to_string(), vec![2.0, 3.0, 5.0]),
            ("B".to_string(), vec![1.0, f64::INFINITY, f64::INFINITY]),
        ];
        match fligner_test(&groups) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("'B'") && message.contains("infinite"),
                    "{message}"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }
}
