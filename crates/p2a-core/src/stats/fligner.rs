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

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::SignificanceLevel;

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

    // Step 1: Compute absolute deviations from group medians
    let mut all_deviations: Vec<f64> = Vec::with_capacity(total_n);
    let mut group_indices: Vec<usize> = Vec::with_capacity(total_n);

    for (group_idx, (_, data)) in groups.iter().enumerate() {
        let median = compute_median(data);
        for &x in data {
            all_deviations.push((x - median).abs());
            group_indices.push(group_idx);
        }
    }

    // Step 2: Rank all absolute deviations
    let ranks = compute_ranks(&all_deviations);

    // Step 3: Transform ranks to normal scores
    // R uses: a(i) = qnorm((1 + i/(n+1))/2)
    // where i is the rank (1 to n)
    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_f = total_n as f64;

    let z_scores: Vec<f64> = ranks
        .iter()
        .map(|&r| {
            // R formula: qnorm((1 + r/(n+1))/2)
            let p = (1.0 + r / (n_f + 1.0)) / 2.0;
            normal.inverse_cdf(p)
        })
        .collect();

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
    use statrs::distribution::ChiSquared;
    let chi_dist = ChiSquared::new(df as f64).unwrap();
    let p_value = 1.0 - chi_dist.cdf(chi_squared);

    Ok(FlignerResult {
        test_name: "Fligner-Killeen test of homogeneity of variances".to_string(),
        chi_squared,
        df,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n_groups: k,
        group_sizes,
        group_names,
    })
}

/// Perform Fligner-Killeen test using dataset columns.
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
    let df = dataset.df();

    // Extract values
    let value_series = df
        .column(value_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: value_col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;

    // Extract groups
    let group_series = df
        .column(group_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: group_col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;

    // Get unique groups and build data structure
    let mut groups_map: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();

    let values: Vec<f64> = value_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: value_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    let group_labels: Vec<String> = group_series
        .str()
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Column '{}' must be a string/categorical type", group_col),
        })?
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

    if values.len() != group_labels.len() {
        return Err(EconError::InvalidSpecification {
            message: "Value and group columns have different lengths".to_string(),
        });
    }

    for (value, group) in values.iter().zip(group_labels.iter()) {
        groups_map.entry(group.clone()).or_default().push(*value);
    }

    // Convert to vector of tuples
    let mut groups: Vec<(String, Vec<f64>)> = groups_map.into_iter().collect();
    groups.sort_by(|a, b| a.0.cmp(&b.0)); // Sort by group name for consistency

    fligner_test(&groups)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Compute median of a slice.
fn compute_median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute ranks with average tie-breaking.
fn compute_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all ties
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
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
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_fligner_three_groups() {
        // R code:
        // g1 <- c(2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7)
        // g2 <- c(3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3)
        // g3 <- c(4.1, 5.2, 3.9, 4.5, 4.8, 5.0)
        // fligner.test(list(g1, g2, g3))
        // Fligner-Killeen:med chi-squared = 5.1024, df = 2, p-value = 0.07799

        let groups = vec![
            ("A".to_string(), vec![2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7]),
            ("B".to_string(), vec![3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3]),
            ("C".to_string(), vec![4.1, 5.2, 3.9, 4.5, 4.8, 5.0]),
        ];

        let result = fligner_test(&groups).unwrap();

        assert!(
            (result.chi_squared - 5.1024).abs() < 0.1,
            "chi-squared mismatch: Rust={}, R=5.1024",
            result.chi_squared
        );
        assert_eq!(result.df, 2);
        assert!(
            (result.p_value - 0.07799).abs() < 0.01,
            "p-value mismatch: Rust={}, R=0.07799",
            result.p_value
        );
    }

    #[test]
    fn test_validate_fligner_two_groups() {
        // R code:
        // x <- c(1, 2, 3, 4, 5)
        // y <- c(10, 20, 30, 40, 50)
        // fligner.test(list(x, y))
        // Fligner-Killeen:med chi-squared = 3.3306, df = 1, p-value = 0.068

        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        ];

        let result = fligner_test(&groups).unwrap();

        assert!(
            (result.chi_squared - 3.3306).abs() < 0.1,
            "chi-squared mismatch: Rust={}, R=3.3306",
            result.chi_squared
        );
        assert_eq!(result.df, 1);
        assert!(
            (result.p_value - 0.068).abs() < 0.01,
            "p-value mismatch: Rust={}, R=0.068",
            result.p_value
        );
    }
}
