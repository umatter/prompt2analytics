//! Kruskal-Wallis rank sum test
//!
//! A non-parametric test for comparing the medians of two or more independent samples.
//! This is the non-parametric alternative to one-way ANOVA.
//!
//! # References
//!
//! - Kruskal, W. H. & Wallis, W. A. (1952). "Use of Ranks in One-Criterion Variance Analysis".
//!   Journal of the American Statistical Association, 47(260), 583-621.
//! - Hollander, M. & Wolfe, D. A. (1973). Nonparametric Statistical Methods.
//!   New York: John Wiley & Sons. Pages 115-120.
//! - R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/kruskal.test.html

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
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

    // Combine all observations with their group index
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n_total);
    for (group_idx, (_, vals)) in clean_groups.iter().enumerate() {
        for &val in vals {
            combined.push((val, group_idx));
        }
    }

    // Sort by value and compute ranks (with ties handled by averaging)
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Compute ranks with tie handling
    let (ranks, tie_counts) = compute_ranks_with_ties(&combined);

    // Compute tie correction factor
    // C = 1 - Σ(ti³ - ti) / (N³ - N)
    let tie_sum: f64 = tie_counts
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();
    let n = n_total as f64;
    let tie_correction = 1.0 - tie_sum / (n * n * n - n);
    let has_ties = tie_counts.iter().any(|&t| t > 1);

    // Compute rank sums for each group
    let mut rank_sums = vec![0.0; k];
    for (i, &(_, group_idx)) in combined.iter().enumerate() {
        rank_sums[group_idx] += ranks[i];
    }

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

    // Apply tie correction
    let h = if tie_correction > 0.0 {
        h_uncorrected / tie_correction
    } else {
        h_uncorrected
    };

    // Degrees of freedom
    let df = k - 1;

    // P-value from chi-squared distribution
    let p_value = chi_squared_p_value(h, df);

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
    })
}

/// Compute ranks with tie handling (midranks for tied values)
fn compute_ranks_with_ties(sorted_data: &[(f64, usize)]) -> (Vec<f64>, Vec<usize>) {
    let n = sorted_data.len();
    let mut ranks = vec![0.0; n];
    let mut tie_counts = Vec::new();

    let mut i = 0;
    while i < n {
        // Find extent of tie group
        let mut j = i + 1;
        while j < n && (sorted_data[j].0 - sorted_data[i].0).abs() < 1e-10 {
            j += 1;
        }

        // Compute average rank for this tie group
        // Ranks are 1-indexed: positions i+1 through j
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        let tie_size = j - i;

        for idx in i..j {
            ranks[idx] = avg_rank;
        }

        tie_counts.push(tie_size);
        i = j;
    }

    (ranks, tie_counts)
}

/// Compute p-value from chi-squared distribution
fn chi_squared_p_value(statistic: f64, df: usize) -> f64 {
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    if df == 0 || statistic.is_nan() || statistic.is_infinite() {
        return f64::NAN;
    }

    let chi2 = ChiSquared::new(df as f64).unwrap();
    1.0 - chi2.cdf(statistic)
}

/// Run Kruskal-Wallis test on a Dataset with grouping variable.
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

    kruskal_test(&groups)
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
    fn test_validate_kruskal_against_r_basic() {
        // R code:
        // x <- c(2.9, 3.0, 2.5, 2.6, 3.2)
        // y <- c(3.8, 2.7, 4.0, 2.4)
        // z <- c(2.8, 3.4, 3.7, 2.2, 2.0)
        // kruskal.test(list(x, y, z))
        //
        // Expected output:
        // Kruskal-Wallis chi-squared = 0.77143, df = 2, p-value = 0.68
        let groups = vec![
            ("x".to_string(), vec![2.9, 3.0, 2.5, 2.6, 3.2]),
            ("y".to_string(), vec![3.8, 2.7, 4.0, 2.4]),
            ("z".to_string(), vec![2.8, 3.4, 3.7, 2.2, 2.0]),
        ];

        let result = kruskal_test(&groups).unwrap();

        // R gives H = 0.77143, df = 2, p-value = 0.68
        assert!((result.statistic - 0.77143).abs() < 0.01);
        assert_eq!(result.df, 2);
        assert!((result.p_value - 0.68).abs() < 0.05);
    }

    #[test]
    fn test_validate_kruskal_airquality_like() {
        // Based on R's airquality example but simplified
        // kruskal.test(Ozone ~ Month, data = airquality)
        //
        // Using made-up data with similar structure
        let groups = vec![
            ("May".to_string(), vec![41.0, 36.0, 12.0, 18.0, 23.0]),
            ("Jun".to_string(), vec![29.0, 45.0, 71.0, 39.0, 32.0]),
            ("Jul".to_string(), vec![135.0, 49.0, 32.0, 64.0, 40.0]),
        ];

        let result = kruskal_test(&groups).unwrap();

        // Just check reasonable output
        assert_eq!(result.n_groups, 3);
        assert_eq!(result.df, 2);
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_validate_kruskal_statistic_calculation() {
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

        // Manual calculation gives H ≈ 4.571
        assert!((result.statistic - 4.571).abs() < 0.01);
    }

    #[test]
    fn test_rank_computation() {
        let data = vec![
            (1.0, 0),
            (2.0, 0),
            (2.0, 1),
            (3.0, 1),
            (3.0, 2),
            (3.0, 2),
        ];

        let (ranks, tie_counts) = compute_ranks_with_ties(&data);

        // Value 1.0 -> rank 1
        assert!((ranks[0] - 1.0).abs() < 1e-10);
        // Value 2.0 (tied) -> average of ranks 2,3 = 2.5
        assert!((ranks[1] - 2.5).abs() < 1e-10);
        assert!((ranks[2] - 2.5).abs() < 1e-10);
        // Value 3.0 (three tied) -> average of ranks 4,5,6 = 5
        assert!((ranks[3] - 5.0).abs() < 1e-10);
        assert!((ranks[4] - 5.0).abs() < 1e-10);
        assert!((ranks[5] - 5.0).abs() < 1e-10);

        // Tie counts: 1, 2, 3
        assert_eq!(tie_counts, vec![1, 2, 3]);
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
    fn test_validate_kruskal_ties_correction() {
        // Test with extensive ties to verify correction
        let groups = vec![
            ("A".to_string(), vec![1.0, 1.0, 2.0, 2.0, 3.0]),
            ("B".to_string(), vec![2.0, 2.0, 3.0, 3.0, 4.0]),
            ("C".to_string(), vec![3.0, 3.0, 4.0, 4.0, 5.0]),
        ];

        let result = kruskal_test(&groups).unwrap();

        assert!(result.has_ties);
        // With ties, correction should be < 1.0
        assert!(result.tie_correction < 1.0);
        assert!(result.tie_correction > 0.0);
    }
}
