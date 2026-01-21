//! Welch's one-way ANOVA test (oneway.test)
//!
//! Tests the equality of means in multiple groups without assuming equal variances.
//! This is the generalization of Welch's t-test to more than two groups.
//!
//! # References
//!
//! - Welch, B. L. (1951). "On the Comparison of Several Mean Values: An Alternative
//!   Approach". Biometrika, 38(3/4), 330-336.
//! - R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/oneway.test.html

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Result of Welch's one-way ANOVA test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnewayTestResult {
    /// The F statistic
    pub statistic: f64,
    /// Numerator degrees of freedom (k - 1)
    pub df_num: f64,
    /// Denominator degrees of freedom (Welch-Satterthwaite approximation)
    pub df_denom: f64,
    /// P-value from F distribution
    pub p_value: f64,
    /// Number of groups
    pub n_groups: usize,
    /// Total sample size
    pub n_total: usize,
    /// Sample sizes per group
    pub group_sizes: Vec<usize>,
    /// Group names/labels
    pub group_names: Vec<String>,
    /// Group means
    pub group_means: Vec<f64>,
    /// Group variances
    pub group_variances: Vec<f64>,
    /// Whether equal variances were assumed
    pub var_equal: bool,
}

/// Perform Welch's one-way ANOVA test (or standard ANOVA if var.equal = true).
///
/// # Arguments
///
/// * `groups` - Vector of (group_name, values) tuples
/// * `var_equal` - If true, assume equal variances and use standard F-test
///
/// # Returns
///
/// A `OnewayTestResult` containing the test statistic, degrees of freedom, and p-value.
///
/// # Example
///
/// ```
/// use p2a_core::stats::oneway::{oneway_test, OnewayTestResult};
///
/// let groups = vec![
///     ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
///     ("B".to_string(), vec![6.0, 7.0, 8.0, 9.0, 10.0]),
///     ("C".to_string(), vec![3.0, 4.0, 5.0, 6.0, 7.0]),
/// ];
///
/// let result = oneway_test(&groups, false).unwrap();
/// println!("F = {}, df1 = {}, df2 = {:.2}, p = {}",
///          result.statistic, result.df_num, result.df_denom, result.p_value);
/// ```
pub fn oneway_test(groups: &[(String, Vec<f64>)], var_equal: bool) -> EconResult<OnewayTestResult> {
    // Need at least 2 groups
    if groups.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: groups.len(),
            context: "oneway.test requires at least 2 groups".to_string(),
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

    // Check each group has at least 2 observations
    for (name, vals) in &clean_groups {
        if vals.len() < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: vals.len(),
                context: format!("Group '{}' needs at least 2 observations", name),
            });
        }
    }

    let k = clean_groups.len();
    let group_sizes: Vec<usize> = clean_groups.iter().map(|(_, v)| v.len()).collect();
    let group_names: Vec<String> = clean_groups.iter().map(|(n, _)| n.clone()).collect();
    let n_total: usize = group_sizes.iter().sum();

    // Compute group means and variances
    let group_means: Vec<f64> = clean_groups
        .iter()
        .map(|(_, vals)| vals.iter().sum::<f64>() / vals.len() as f64)
        .collect();

    let group_variances: Vec<f64> = clean_groups
        .iter()
        .zip(group_means.iter())
        .map(|((_, vals), &mean)| {
            let ss: f64 = vals.iter().map(|&x| (x - mean).powi(2)).sum();
            ss / (vals.len() - 1) as f64
        })
        .collect();

    let (statistic, df_num, df_denom, p_value) = if var_equal {
        // Standard one-way ANOVA F-test assuming equal variances
        // Compute grand mean
        let all_values: Vec<f64> = clean_groups.iter().flat_map(|(_, v)| v.clone()).collect();
        let grand_mean = all_values.iter().sum::<f64>() / n_total as f64;

        // Between-group sum of squares
        let ss_between: f64 = group_sizes
            .iter()
            .zip(group_means.iter())
            .map(|(&n, &m)| n as f64 * (m - grand_mean).powi(2))
            .sum();

        // Within-group sum of squares
        let ss_within: f64 = group_sizes
            .iter()
            .zip(group_variances.iter())
            .map(|(&n, &v)| (n - 1) as f64 * v)
            .sum();

        let df1 = (k - 1) as f64;
        let df2 = (n_total - k) as f64;

        let ms_between = ss_between / df1;
        let ms_within = ss_within / df2;

        let f_stat = ms_between / ms_within;
        let p = f_test_p_value(f_stat, df1, df2);

        (f_stat, df1, df2, p)
    } else {
        // Welch's ANOVA (not assuming equal variances)
        // Weights: w_i = n_i / v_i
        let weights: Vec<f64> = group_sizes
            .iter()
            .zip(group_variances.iter())
            .map(|(&n, &v)| {
                if v > 1e-15 {
                    n as f64 / v
                } else {
                    // Very small variance - use large weight
                    n as f64 * 1e15
                }
            })
            .collect();

        let sum_weights: f64 = weights.iter().sum();

        // Weighted grand mean
        let weighted_mean: f64 = weights
            .iter()
            .zip(group_means.iter())
            .map(|(&w, &m)| w * m)
            .sum::<f64>()
            / sum_weights;

        // Compute tmp = Σ((1 - w_i/Σw_j)² / (n_i - 1)) / (k² - 1)
        let tmp: f64 = weights
            .iter()
            .zip(group_sizes.iter())
            .map(|(&w, &n)| {
                let ratio = 1.0 - w / sum_weights;
                ratio.powi(2) / (n - 1) as f64
            })
            .sum::<f64>()
            / ((k * k - 1) as f64);

        // F statistic = Σ(w_i * (m_i - m)²) / ((k-1) * (1 + 2*(k-2)*tmp))
        let numerator: f64 = weights
            .iter()
            .zip(group_means.iter())
            .map(|(&w, &m)| w * (m - weighted_mean).powi(2))
            .sum();

        let df1 = (k - 1) as f64;
        let denominator = df1 * (1.0 + 2.0 * (k as f64 - 2.0) * tmp);

        let f_stat = numerator / denominator;

        // Denominator degrees of freedom
        let df2 = 1.0 / (3.0 * tmp);

        let p = f_test_p_value(f_stat, df1, df2);

        (f_stat, df1, df2, p)
    };

    Ok(OnewayTestResult {
        statistic,
        df_num,
        df_denom,
        p_value,
        n_groups: k,
        n_total,
        group_sizes,
        group_names,
        group_means,
        group_variances,
        var_equal,
    })
}

/// Compute p-value from F distribution
fn f_test_p_value(statistic: f64, df1: f64, df2: f64) -> f64 {
    use statrs::distribution::{FisherSnedecor, ContinuousCDF};

    if df1 <= 0.0 || df2 <= 0.0 || statistic.is_nan() || statistic.is_infinite() {
        return f64::NAN;
    }

    let f_dist = match FisherSnedecor::new(df1, df2) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };

    1.0 - f_dist.cdf(statistic)
}

/// Run Welch's one-way ANOVA test on a Dataset with grouping variable.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the data
/// * `value_col` - Column name for the values to compare
/// * `group_col` - Column name for the grouping variable
/// * `var_equal` - If true, assume equal variances
///
/// # Returns
///
/// A `OnewayTestResult` containing the test results.
pub fn run_oneway_test(
    dataset: &Dataset,
    value_col: &str,
    group_col: &str,
    var_equal: bool,
) -> EconResult<OnewayTestResult> {
    let df = dataset.df();

    // Extract value column
    let values = df
        .column(value_col)
        .map_err(|e| EconError::NonNumericColumn {
            column: format!("{}: {}", value_col, e),
        })?
        .f64()
        .map_err(|e| EconError::NonNumericColumn {
            column: format!("{}: {}", value_col, e),
        })?;

    // Extract group column as strings
    let groups_col = df
        .column(group_col)
        .map_err(|e| EconError::NonNumericColumn {
            column: format!("{}: {}", group_col, e),
        })?;

    // Build group data
    use std::collections::HashMap;
    let mut group_data: HashMap<String, Vec<f64>> = HashMap::new();

    let n = df.height();
    for i in 0..n {
        let group_name = groups_col.get(i).map_err(|e| EconError::NonNumericColumn {
            column: format!("Error accessing group at row {}: {}", i, e),
        })?;
        let group_str = format!("{}", group_name);

        let value = values.get(i);
        if let Some(v) = value {
            group_data.entry(group_str).or_insert_with(Vec::new).push(v);
        }
    }

    // Convert to vector of tuples
    let mut groups: Vec<(String, Vec<f64>)> = group_data.into_iter().collect();
    // Sort by group name for consistent ordering
    groups.sort_by(|a, b| a.0.cmp(&b.0));

    oneway_test(&groups, var_equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_oneway_basic_welch() {
        // Groups with clearly different means
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![6.0, 7.0, 8.0, 9.0, 10.0]),
            ("C".to_string(), vec![11.0, 12.0, 13.0, 14.0, 15.0]),
        ];

        let result = oneway_test(&groups, false).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.n_total, 15);
        assert!(!result.var_equal);
        assert!(result.statistic > 0.0);
        // With clearly different means, should reject H0
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_oneway_basic_standard() {
        // Same data with var.equal = true
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![6.0, 7.0, 8.0, 9.0, 10.0]),
            ("C".to_string(), vec![11.0, 12.0, 13.0, 14.0, 15.0]),
        ];

        let result = oneway_test(&groups, true).unwrap();

        assert_eq!(result.n_groups, 3);
        assert!(result.var_equal);
        assert_eq!(result.df_num, 2.0);
        assert_eq!(result.df_denom, 12.0);  // n - k = 15 - 3
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_oneway_similar_groups() {
        // Groups with similar means should have high p-value
        let groups = vec![
            ("A".to_string(), vec![5.0, 5.1, 5.2, 4.9, 5.0]),
            ("B".to_string(), vec![5.1, 5.0, 5.2, 5.1, 4.9]),
            ("C".to_string(), vec![4.9, 5.0, 5.1, 5.0, 5.2]),
        ];

        let result = oneway_test(&groups, false).unwrap();

        // Similar means should not reject H0
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_validate_oneway_against_r_welch() {
        // R code:
        // x <- c(1, 2, 3, 4, 5)
        // y <- c(10, 11, 12, 13, 14, 15)
        // z <- c(3, 4, 5)
        // g <- factor(c(rep("A", 5), rep("B", 6), rep("C", 3)))
        // df <- data.frame(value = c(x, y, z), group = g)
        // oneway.test(value ~ group, data = df, var.equal = FALSE)
        //
        // Result: F = 46.645, num df = 2, denom df = 6.8877, p-value = 9.905e-05

        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            ("C".to_string(), vec![3.0, 4.0, 5.0]),
        ];

        let result = oneway_test(&groups, false).unwrap();

        println!("F = {}, df1 = {}, df2 = {}, p = {}",
                 result.statistic, result.df_num, result.df_denom, result.p_value);

        // R gives: F = 46.645, df1 = 2, df2 = 6.8877, p = 9.905e-05
        assert!((result.statistic - 46.645).abs() < 0.1, "F mismatch: got {}", result.statistic);
        assert_eq!(result.df_num, 2.0);
        assert!((result.df_denom - 6.8877).abs() < 0.01, "df2 mismatch: got {}", result.df_denom);
        assert!((result.p_value - 9.905e-05).abs() < 0.00005, "p-value mismatch: got {}", result.p_value);
    }

    #[test]
    fn test_validate_oneway_against_r_standard() {
        // R code:
        // x <- c(1, 2, 3, 4, 5)
        // y <- c(10, 11, 12, 13, 14, 15)
        // z <- c(3, 4, 5)
        // g <- factor(c(rep("A", 5), rep("B", 6), rep("C", 3)))
        // df <- data.frame(value = c(x, y, z), group = g)
        // oneway.test(value ~ group, data = df, var.equal = TRUE)
        //
        // Result: F = 53.575, num df = 2, denom df = 11, p-value = 2.134e-06

        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            ("C".to_string(), vec![3.0, 4.0, 5.0]),
        ];

        let result = oneway_test(&groups, true).unwrap();

        println!("Standard ANOVA: F = {}, df1 = {}, df2 = {}, p = {}",
                 result.statistic, result.df_num, result.df_denom, result.p_value);

        // R gives: F = 53.575, df1 = 2, df2 = 11, p = 2.134e-06
        assert!((result.statistic - 53.575).abs() < 0.1, "F mismatch: got {}", result.statistic);
        assert_eq!(result.df_num, 2.0);
        assert_eq!(result.df_denom, 11.0);
        assert!((result.p_value - 2.134e-06).abs() < 0.000001, "p-value mismatch: got {}", result.p_value);
    }

    #[test]
    fn test_oneway_two_groups() {
        // Two groups should work (equivalent to t-test)
        let groups = vec![
            ("Control".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("Treatment".to_string(), vec![6.0, 7.0, 8.0, 9.0, 10.0]),
        ];

        let result = oneway_test(&groups, false).unwrap();

        assert_eq!(result.n_groups, 2);
        assert_eq!(result.df_num, 1.0);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_oneway_validates_groups() {
        // Should require at least 2 groups
        let groups = vec![("A".to_string(), vec![1.0, 2.0, 3.0])];

        let result = oneway_test(&groups, false);
        assert!(result.is_err());

        // Should require at least 2 observations per group
        let groups = vec![
            ("A".to_string(), vec![1.0]),
            ("B".to_string(), vec![2.0, 3.0]),
        ];

        let result = oneway_test(&groups, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_oneway_from_dataset() {
        let df = df! {
            "value" => [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0],
            "group" => ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_oneway_test(&dataset, "value", "group", false).unwrap();

        assert_eq!(result.n_groups, 3);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_oneway_handles_nan() {
        let groups = vec![
            ("A".to_string(), vec![1.0, f64::NAN, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![6.0, 7.0, f64::INFINITY, 9.0]),
        ];

        let result = oneway_test(&groups, false).unwrap();

        // Should filter out NaN and Inf
        assert_eq!(result.group_sizes[0], 4);
        assert_eq!(result.group_sizes[1], 3);
    }

    #[test]
    fn test_oneway_unequal_variances() {
        // Groups with very different variances
        let groups = vec![
            ("A".to_string(), vec![5.0, 5.1, 5.0, 5.1, 5.0]),  // low variance
            ("B".to_string(), vec![1.0, 9.0, 2.0, 8.0, 5.0]),   // high variance
        ];

        let result_welch = oneway_test(&groups, false).unwrap();
        let result_standard = oneway_test(&groups, true).unwrap();

        // Welch should have different df_denom than standard
        assert!(result_welch.df_denom != result_standard.df_denom);
    }
}
