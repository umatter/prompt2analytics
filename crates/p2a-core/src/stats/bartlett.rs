//! Bartlett's Test for Homogeneity of Variances.
//!
//! Tests the null hypothesis that all k population variances are equal
//! against the alternative that at least two are different.
//!
//! # References
//!
//! - Bartlett, M. S. (1937). "Properties of Sufficiency and Statistical Tests".
//!   *Proceedings of the Royal Society of London. Series A, Mathematical and
//!   Physical Sciences*, 160(901), 268-282.
//! - R Core Team. `stats::bartlett.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/bartlett.test.html>
//! - NIST/SEMATECH e-Handbook of Statistical Methods.
//!   <https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.htm>
//!
//! # Mathematical Background
//!
//! The test statistic is:
//! ```text
//! T = [(N-k) ln(s²_p) - Σ(nᵢ-1)ln(s²ᵢ)] / C
//! ```
//!
//! where:
//! - `s²_p` is the pooled variance: `Σ(nᵢ-1)s²ᵢ / (N-k)`
//! - `s²ᵢ` is the sample variance of group i
//! - `nᵢ` is the sample size of group i
//! - `N` is the total sample size
//! - `k` is the number of groups
//! - `C` is the correction factor: `1 + [1/(3(k-1))] × [Σ(1/(nᵢ-1)) - 1/(N-k)]`
//!
//! Under H₀, T ~ χ²(k-1).
//!
//! # Important Notes
//!
//! Bartlett's test is sensitive to departures from normality. If samples are
//! non-normal, consider using Levene's test instead, which is more robust.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::{chi_squared_p_value, SignificanceLevel};

/// Result of Bartlett's test for homogeneity of variances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartlettResult {
    /// Bartlett's K-squared test statistic
    pub statistic: f64,
    /// Degrees of freedom (k - 1)
    pub df: usize,
    /// P-value from chi-squared distribution
    pub p_value: f64,
    /// Significance level based on p-value
    pub significance: SignificanceLevel,
    /// Number of groups
    pub n_groups: usize,
    /// Total number of observations
    pub n_obs: usize,
    /// Pooled variance estimate
    pub pooled_variance: f64,
    /// Group-wise statistics
    pub group_stats: Vec<BartlettGroupStats>,
}

/// Statistics for a single group in Bartlett's test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartlettGroupStats {
    /// Group identifier
    pub group: String,
    /// Number of observations
    pub n: usize,
    /// Sample variance
    pub variance: f64,
    /// Sample standard deviation
    pub std_dev: f64,
}

impl std::fmt::Display for BartlettResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bartlett Test for Homogeneity of Variances")?;
        writeln!(f, "===========================================")?;
        writeln!(f)?;
        writeln!(f, "K-squared = {:.4}, df = {}, p-value = {:.4} {}",
            self.statistic, self.df, self.p_value, self.significance.stars())?;
        writeln!(f)?;
        writeln!(f, "Groups: {}  |  N = {}", self.n_groups, self.n_obs)?;
        writeln!(f, "Pooled variance: {:.4}", self.pooled_variance)?;
        writeln!(f)?;
        writeln!(f, "Group Statistics:")?;
        writeln!(f, "{:>15} {:>8} {:>12} {:>12}", "Group", "n", "Variance", "Std Dev")?;
        writeln!(f, "{}", "-".repeat(50))?;
        for gs in &self.group_stats {
            writeln!(f, "{:>15} {:>8} {:>12.4} {:>12.4}",
                gs.group, gs.n, gs.variance, gs.std_dev)?;
        }
        writeln!(f)?;
        writeln!(f, "H₀: All group variances are equal")?;
        writeln!(f, "H₁: At least two group variances differ")?;
        if self.p_value < 0.05 {
            writeln!(f, "\nConclusion: Reject H₀ - variances are significantly different.")?;
        } else {
            writeln!(f, "\nConclusion: Fail to reject H₀ - no significant difference in variances.")?;
        }
        Ok(())
    }
}

/// Perform Bartlett's test on grouped data provided as slices.
///
/// # Arguments
/// * `groups` - A slice of (group_name, data) tuples
///
/// # Returns
/// `BartlettResult` with the test statistic, p-value, and group statistics.
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::bartlett::bartlett_test;
///
/// let groups = vec![
///     ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0]),
///     ("B".to_string(), vec![2.0, 3.0, 4.0, 5.0]),
///     ("C".to_string(), vec![3.0, 4.0, 5.0, 6.0]),
/// ];
/// let result = bartlett_test(&groups)?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// - Bartlett, M. S. (1937). *Proceedings of the Royal Society of London*, 160(901).
pub fn bartlett_test(groups: &[(String, Vec<f64>)]) -> EconResult<BartlettResult> {
    let k = groups.len();

    if k < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Bartlett test requires at least 2 groups".to_string(),
        });
    }

    // Calculate statistics for each group
    let mut group_stats = Vec::with_capacity(k);
    let mut n_total: usize = 0;
    let mut sum_ni_minus_1_times_var = 0.0;
    let mut sum_ni_minus_1_times_ln_var = 0.0;
    let mut sum_inv_ni_minus_1 = 0.0;

    for (name, data) in groups {
        let n = data.len();

        if n < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: n,
                context: format!("Group '{}' needs at least 2 observations", name),
            });
        }

        // Calculate mean
        let mean: f64 = data.iter().sum::<f64>() / n as f64;

        // Calculate sample variance (using n-1 denominator)
        let variance: f64 = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;

        if variance <= 0.0 {
            return Err(EconError::InvalidSpecification {
                message: format!("Group '{}' has zero or negative variance", name),
            });
        }

        let ni_minus_1 = (n - 1) as f64;
        n_total += n;
        sum_ni_minus_1_times_var += ni_minus_1 * variance;
        sum_ni_minus_1_times_ln_var += ni_minus_1 * variance.ln();
        sum_inv_ni_minus_1 += 1.0 / ni_minus_1;

        group_stats.push(BartlettGroupStats {
            group: name.clone(),
            n,
            variance,
            std_dev: variance.sqrt(),
        });
    }

    let n_minus_k = (n_total - k) as f64;

    // Pooled variance: s²_p = Σ(nᵢ-1)s²ᵢ / (N-k)
    let pooled_variance = sum_ni_minus_1_times_var / n_minus_k;

    // Numerator: (N-k) ln(s²_p) - Σ(nᵢ-1)ln(s²ᵢ)
    let numerator = n_minus_k * pooled_variance.ln() - sum_ni_minus_1_times_ln_var;

    // Correction factor: C = 1 + [1/(3(k-1))] × [Σ(1/(nᵢ-1)) - 1/(N-k)]
    let correction = 1.0 + (1.0 / (3.0 * (k - 1) as f64)) * (sum_inv_ni_minus_1 - 1.0 / n_minus_k);

    // Test statistic: T = numerator / C
    let statistic = numerator / correction;

    // Degrees of freedom: k - 1
    let df = k - 1;

    // P-value from chi-squared distribution
    let p_value = chi_squared_p_value(statistic, df as f64);

    Ok(BartlettResult {
        statistic,
        df,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n_groups: k,
        n_obs: n_total,
        pooled_variance,
        group_stats,
    })
}

/// Perform Bartlett's test from a Dataset with a response and grouping variable.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `response_col` - Name of the response (numeric) variable column
/// * `factor_col` - Name of the factor (grouping) variable column
///
/// # Returns
/// `BartlettResult` with the test statistic, p-value, and group statistics.
///
/// # Example
///
/// ```ignore
/// use p2a_core::stats::bartlett::run_bartlett_test;
///
/// let result = run_bartlett_test(&dataset, "count", "spray")?;
/// println!("{}", result);
/// ```
pub fn run_bartlett_test(
    dataset: &Dataset,
    response_col: &str,
    factor_col: &str,
) -> EconResult<BartlettResult> {
    let df = dataset.df();
    let n_rows = df.height();

    // Get response column
    let response = df.column(response_col).map_err(|_| EconError::ColumnNotFound {
        column: response_col.to_string(),
        available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
    })?;

    let response_values: Vec<Option<f64>> = response.f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: response_col.to_string(),
        })?
        .into_iter()
        .collect();

    // Get factor column
    let factor = df.column(factor_col).map_err(|_| EconError::ColumnNotFound {
        column: factor_col.to_string(),
        available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
    })?;

    // Extract factor values as strings
    let factor_values: Vec<String> = if let Ok(str_col) = factor.str() {
        str_col.into_iter()
            .map(|opt| opt.map(|s| s.to_string()).unwrap_or_default())
            .collect()
    } else if let Ok(int_col) = factor.i64() {
        int_col.into_iter()
            .map(|opt| opt.map(|v| v.to_string()).unwrap_or_default())
            .collect()
    } else if let Ok(float_col) = factor.f64() {
        float_col.into_iter()
            .map(|opt| opt.map(|v| v.to_string()).unwrap_or_default())
            .collect()
    } else {
        return Err(EconError::NonNumericColumn {
            column: format!("{} (cannot extract factor values)", factor_col),
        });
    };

    // Group data by factor
    let mut grouped: HashMap<String, Vec<f64>> = HashMap::new();

    for i in 0..n_rows {
        let group = &factor_values[i];
        if group.is_empty() {
            continue;
        }

        if let Some(value) = response_values[i] {
            grouped.entry(group.clone()).or_default().push(value);
        }
    }

    // Convert to vector and sort by group name for consistent ordering
    let mut groups: Vec<(String, Vec<f64>)> = grouped.into_iter().collect();
    groups.sort_by(|a, b| a.0.cmp(&b.0));

    bartlett_test(&groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_bartlett_basic() {
        // Three groups with similar variances
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![2.0, 3.0, 4.0, 5.0, 6.0]),
            ("C".to_string(), vec![3.0, 4.0, 5.0, 6.0, 7.0]),
        ];

        let result = bartlett_test(&groups).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.n_obs, 15);
        assert_eq!(result.df, 2);  // k - 1 = 3 - 1 = 2
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);

        // Groups have equal variances (all are 2.5), so p-value should be high
        // (fail to reject H0)
        assert!(result.p_value > 0.05,
            "Expected p > 0.05 for equal variances, got {}", result.p_value);
    }

    #[test]
    fn test_bartlett_unequal_variances() {
        // Three groups with clearly different variances
        let groups = vec![
            ("Low".to_string(), vec![1.0, 1.1, 1.2, 0.9, 1.0]),    // small variance
            ("Med".to_string(), vec![1.0, 2.0, 3.0, 0.0, 2.0]),    // medium variance
            ("High".to_string(), vec![1.0, 10.0, 20.0, -5.0, 4.0]), // large variance
        ];

        let result = bartlett_test(&groups).unwrap();

        assert_eq!(result.n_groups, 3);

        // Variances are very different, so p-value should be low (reject H0)
        assert!(result.p_value < 0.05,
            "Expected p < 0.05 for unequal variances, got {}", result.p_value);
    }

    #[test]
    fn test_bartlett_two_groups() {
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0]),
            ("B".to_string(), vec![2.0, 3.0, 4.0, 5.0]),
        ];

        let result = bartlett_test(&groups).unwrap();

        assert_eq!(result.n_groups, 2);
        assert_eq!(result.df, 1);  // k - 1 = 2 - 1 = 1
    }

    #[test]
    fn test_bartlett_insufficient_groups() {
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0]),
        ];

        let result = bartlett_test(&groups);
        assert!(result.is_err());
    }

    #[test]
    fn test_bartlett_insufficient_observations() {
        let groups = vec![
            ("A".to_string(), vec![1.0]),  // Only 1 observation
            ("B".to_string(), vec![2.0, 3.0]),
        ];

        let result = bartlett_test(&groups);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_bartlett_against_r() {
        // Test case from R:
        // > x <- c(1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 5, 10, 15, 20, 25)
        // > g <- factor(c(rep("A", 5), rep("B", 5), rep("C", 5)))
        // > bartlett.test(x ~ g)
        //
        // Actual R output:
        //   Bartlett test of homogeneity of variances
        //
        // data:  x by g
        // Bartlett's K-squared = 12.142, df = 2, p-value = 0.002309
        //
        // Group variances: A=2.5, B=2.5, C=62.5

        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![2.0, 3.0, 4.0, 5.0, 6.0]),
            ("C".to_string(), vec![5.0, 10.0, 15.0, 20.0, 25.0]),
        ];

        let result = bartlett_test(&groups).unwrap();

        // Check statistic matches R (K-squared = 12.142)
        assert!(
            (result.statistic - 12.142).abs() < 0.01,
            "K-squared should be ~12.142, got {}",
            result.statistic
        );

        // Check df
        assert_eq!(result.df, 2);

        // Check p-value matches R (p = 0.002309)
        assert!(
            (result.p_value - 0.002309).abs() < 0.001,
            "p-value should be ~0.002309, got {}",
            result.p_value
        );

        // Verify group variances
        assert!((result.group_stats[0].variance - 2.5).abs() < 0.001);
        assert!((result.group_stats[1].variance - 2.5).abs() < 0.001);
        assert!((result.group_stats[2].variance - 62.5).abs() < 0.001);
    }

    #[test]
    fn test_bartlett_from_dataset() {
        let df = df! {
            "count" => [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "spray" => ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"]
        }.unwrap();

        let dataset = Dataset::new(df);
        let result = run_bartlett_test(&dataset, "count", "spray").unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.n_obs, 15);
        assert_eq!(result.df, 2);
    }

    #[test]
    fn test_bartlett_display() {
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ("B".to_string(), vec![2.0, 3.0, 4.0, 5.0, 6.0]),
        ];

        let result = bartlett_test(&groups).unwrap();
        let display = format!("{}", result);

        assert!(display.contains("Bartlett Test"));
        assert!(display.contains("K-squared"));
        assert!(display.contains("df"));
        assert!(display.contains("p-value"));
        assert!(display.contains("Pooled variance"));
    }

    #[test]
    fn test_bartlett_unequal_sample_sizes() {
        // Test with unequal group sizes
        let groups = vec![
            ("A".to_string(), vec![1.0, 2.0, 3.0]),
            ("B".to_string(), vec![2.0, 3.0, 4.0, 5.0, 6.0]),
            ("C".to_string(), vec![3.0, 4.0, 5.0, 6.0]),
        ];

        let result = bartlett_test(&groups).unwrap();

        assert_eq!(result.n_groups, 3);
        assert_eq!(result.n_obs, 12);
        assert_eq!(result.group_stats[0].n, 3);
        assert_eq!(result.group_stats[1].n, 5);
        assert_eq!(result.group_stats[2].n, 4);
    }
}
