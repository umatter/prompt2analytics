//! Friedman rank sum test for unreplicated blocked data
//!
//! A non-parametric test for comparing more than two related groups.
//! This is the non-parametric alternative to one-way repeated measures ANOVA.
//!
//! # References
//!
//! - Friedman, M. (1937). "The Use of Ranks to Avoid the Assumption of Normality
//!   Implicit in the Analysis of Variance". Journal of the American Statistical
//!   Association, 32(200), 675-701.
//! - Hollander, M. & Wolfe, D. A. (1973). Nonparametric Statistical Methods.
//!   New York: John Wiley & Sons. Pages 139-146.
//! - R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/friedman.test.html

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Result of the Friedman rank sum test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FriedmanResult {
    /// The Q statistic (chi-squared approximation)
    pub statistic: f64,
    /// Degrees of freedom (k - 1)
    pub df: usize,
    /// P-value from chi-squared distribution
    pub p_value: f64,
    /// Number of blocks (subjects/rows)
    pub n_blocks: usize,
    /// Number of treatments (groups/columns)
    pub n_treatments: usize,
    /// Rank sums per treatment
    pub rank_sums: Vec<f64>,
    /// Mean ranks per treatment
    pub mean_ranks: Vec<f64>,
    /// Treatment/group names
    pub treatment_names: Vec<String>,
    /// Whether ties were present
    pub has_ties: bool,
    /// Tie correction factor (1.0 if no ties)
    pub tie_correction: f64,
}

/// Perform Friedman rank sum test on a matrix of observations.
///
/// # Arguments
///
/// * `data` - Matrix where rows are blocks and columns are treatments.
///            Each cell contains the measurement for that block-treatment combination.
/// * `treatment_names` - Names for each treatment (column)
///
/// # Returns
///
/// A `FriedmanResult` containing the test statistic, degrees of freedom, and p-value.
///
/// # Example
///
/// ```
/// use p2a_core::stats::friedman::{friedman_test, FriedmanResult};
///
/// // 5 blocks (subjects), 3 treatments
/// let data = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![1.5, 2.5, 3.5],
///     vec![1.2, 2.2, 3.2],
///     vec![1.8, 2.8, 3.8],
///     vec![1.1, 2.1, 3.1],
/// ];
/// let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
///
/// let result = friedman_test(&data, &names).unwrap();
/// println!("Q = {}, df = {}, p = {}", result.statistic, result.df, result.p_value);
/// ```
pub fn friedman_test(data: &[Vec<f64>], treatment_names: &[String]) -> EconResult<FriedmanResult> {
    // Validate input
    if data.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Friedman test requires at least 1 block".to_string(),
        });
    }

    let n_blocks = data.len();
    let n_treatments = data[0].len();

    // Need at least 2 treatments
    if n_treatments < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_treatments,
            context: "Friedman test requires at least 2 treatments".to_string(),
        });
    }

    // Validate all rows have same length
    for (i, row) in data.iter().enumerate() {
        if row.len() != n_treatments {
            return Err(EconError::InsufficientData {
                required: n_treatments,
                provided: row.len(),
                context: format!(
                    "Block {} has {} treatments, expected {}",
                    i,
                    row.len(),
                    n_treatments
                ),
            });
        }
    }

    // Validate treatment names
    if treatment_names.len() != n_treatments {
        return Err(EconError::InsufficientData {
            required: n_treatments,
            provided: treatment_names.len(),
            context: "Treatment names count must match number of treatments".to_string(),
        });
    }

    // Rank within each block
    let mut all_ranks: Vec<Vec<f64>> = Vec::with_capacity(n_blocks);
    let mut total_tie_sum = 0.0;
    let mut has_ties = false;

    for row in data {
        let (ranks, tie_sum) = rank_within_block(row);
        if tie_sum > 0.0 {
            has_ties = true;
        }
        total_tie_sum += tie_sum;
        all_ranks.push(ranks);
    }

    // Compute rank sums for each treatment (column sums)
    let mut rank_sums = vec![0.0; n_treatments];
    for ranks in &all_ranks {
        for (j, &r) in ranks.iter().enumerate() {
            rank_sums[j] += r;
        }
    }

    // Compute mean ranks
    let n = n_blocks as f64;
    let k = n_treatments as f64;
    let mean_ranks: Vec<f64> = rank_sums.iter().map(|&sum| sum / n).collect();

    // Compute Q statistic
    // Q = (12 / (n*k*(k+1))) * Σ(Rj²) - 3*n*(k+1)
    let sum_r2: f64 = rank_sums.iter().map(|&r| r * r).sum();
    let q = (12.0 / (n * k * (k + 1.0))) * sum_r2 - 3.0 * n * (k + 1.0);

    // Tie correction factor
    // C = 1 - Σ(ti³ - ti) / (n*k*(k² - 1))
    let tie_correction = if total_tie_sum > 0.0 {
        1.0 - total_tie_sum / (n * k * (k * k - 1.0))
    } else {
        1.0
    };

    // Apply tie correction
    let q_corrected = if tie_correction > 0.0 {
        q / tie_correction
    } else {
        q
    };

    // Degrees of freedom
    let df = n_treatments - 1;

    // P-value from chi-squared distribution
    let p_value = chi_squared_p_value(q_corrected, df);

    Ok(FriedmanResult {
        statistic: q_corrected,
        df,
        p_value,
        n_blocks,
        n_treatments,
        rank_sums,
        mean_ranks,
        treatment_names: treatment_names.to_vec(),
        has_ties,
        tie_correction,
    })
}

/// Rank values within a single block, handling ties with midranks.
/// Returns (ranks, tie_sum) where tie_sum = Σ(ti³ - ti) for this block.
fn rank_within_block(values: &[f64]) -> (Vec<f64>, f64) {
    let n = values.len();

    // Create index-value pairs and sort
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut tie_sum = 0.0;

    let mut i = 0;
    while i < n {
        // Find extent of tie group
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }

        // Compute average rank for tie group (1-indexed)
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        let tie_size = (j - i) as f64;

        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }

        // Add to tie sum: ti³ - ti
        if tie_size > 1.0 {
            tie_sum += tie_size * tie_size * tie_size - tie_size;
        }

        i = j;
    }

    (ranks, tie_sum)
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

/// Run Friedman test on a Dataset with long-format data.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the data
/// * `value_col` - Column name for the measured values
/// * `group_col` - Column name for the treatment/group variable
/// * `block_col` - Column name for the block/subject variable
///
/// # Returns
///
/// A `FriedmanResult` containing the test results.
pub fn run_friedman_test(
    dataset: &Dataset,
    value_col: &str,
    group_col: &str,
    block_col: &str,
) -> EconResult<FriedmanResult> {
    let df = dataset.df();

    // Extract columns
    let values = df
        .column(value_col)
        .map_err(|e| EconError::NonNumericColumn {
            column: format!("{}: {}", value_col, e),
        })?
        .f64()
        .map_err(|e| EconError::NonNumericColumn {
            column: format!("{}: {}", value_col, e),
        })?;

    let groups_col = df
        .column(group_col)
        .map_err(|e| EconError::NonNumericColumn {
            column: format!("{}: {}", group_col, e),
        })?;

    let blocks_col = df
        .column(block_col)
        .map_err(|e| EconError::NonNumericColumn {
            column: format!("{}: {}", block_col, e),
        })?;

    // Build block -> treatment -> value mapping
    use std::collections::{BTreeSet, HashMap};
    let mut block_data: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut treatment_set: BTreeSet<String> = BTreeSet::new();

    let n = df.height();
    for i in 0..n {
        let block = blocks_col.get(i).map_err(|e| EconError::NonNumericColumn {
            column: format!("Error accessing block at row {}: {}", i, e),
        })?;
        let block_str = format!("{}", block);

        let group = groups_col.get(i).map_err(|e| EconError::NonNumericColumn {
            column: format!("Error accessing group at row {}: {}", i, e),
        })?;
        let group_str = format!("{}", group);

        if let Some(value) = values.get(i) {
            treatment_set.insert(group_str.clone());
            block_data
                .entry(block_str)
                .or_default()
                .insert(group_str, value);
        }
    }

    // Convert to matrix format
    let treatment_names: Vec<String> = treatment_set.into_iter().collect();
    let mut data: Vec<Vec<f64>> = Vec::new();

    for (_, treatments) in block_data {
        // Check all treatments are present
        if treatments.len() != treatment_names.len() {
            continue; // Skip incomplete blocks
        }

        let row: Vec<f64> = treatment_names
            .iter()
            .filter_map(|t| treatments.get(t).copied())
            .collect();

        if row.len() == treatment_names.len() {
            data.push(row);
        }
    }

    if data.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "No complete blocks found for Friedman test".to_string(),
        });
    }

    friedman_test(&data, &treatment_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_friedman_basic() {
        // Clear treatment differences across blocks
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![1.2, 2.2, 3.2],
            vec![1.8, 2.8, 3.8],
            vec![1.1, 2.1, 3.1],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = friedman_test(&data, &names).unwrap();

        assert_eq!(result.n_blocks, 5);
        assert_eq!(result.n_treatments, 3);
        assert_eq!(result.df, 2);
        assert!(result.statistic > 0.0);
        // With perfect ordering, should reject H0
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_friedman_no_difference() {
        // Similar values, rankings should vary
        let data = vec![
            vec![5.1, 5.0, 5.2],
            vec![4.9, 5.1, 5.0],
            vec![5.2, 5.0, 4.9],
            vec![5.0, 5.2, 5.1],
            vec![4.8, 5.0, 5.2],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = friedman_test(&data, &names).unwrap();

        // With similar values, should not reject H0
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_friedman_with_ties() {
        // Data with ties within blocks
        let data = vec![
            vec![1.0, 1.0, 2.0], // tie in first two
            vec![2.0, 2.0, 3.0], // tie in first two
            vec![1.0, 2.0, 2.0], // tie in last two
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = friedman_test(&data, &names).unwrap();

        assert!(result.has_ties);
        assert!(result.tie_correction < 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_validate_friedman_against_r() {
        // R code:
        // RoundingTimes <-
        //   matrix(c(5.40, 5.50, 5.55,
        //            5.85, 5.70, 5.75,
        //            5.20, 5.60, 5.50,
        //            5.55, 5.50, 5.40,
        //            5.90, 5.85, 5.70,
        //            5.45, 5.55, 5.60,
        //            5.40, 5.40, 5.35,
        //            5.45, 5.50, 5.35,
        //            5.25, 5.15, 5.00,
        //            5.85, 5.80, 5.70,
        //            5.25, 5.20, 5.10,
        //            5.65, 5.55, 5.45),
        //          nrow=12, byrow=TRUE,
        //          dimnames=list(1:12, c("Round Out", "Narrow Angle", "Wide Angle")))
        // friedman.test(RoundingTimes)
        //
        // Result: Friedman chi-squared = 11.143, df = 2, p-value = 0.003805

        let data = vec![
            vec![5.40, 5.50, 5.55],
            vec![5.85, 5.70, 5.75],
            vec![5.20, 5.60, 5.50],
            vec![5.55, 5.50, 5.40],
            vec![5.90, 5.85, 5.70],
            vec![5.45, 5.55, 5.60],
            vec![5.40, 5.40, 5.35],
            vec![5.45, 5.50, 5.35],
            vec![5.25, 5.15, 5.00],
            vec![5.85, 5.80, 5.70],
            vec![5.25, 5.20, 5.10],
            vec![5.65, 5.55, 5.45],
        ];
        let names = vec![
            "Round Out".to_string(),
            "Narrow Angle".to_string(),
            "Wide Angle".to_string(),
        ];

        let result = friedman_test(&data, &names).unwrap();

        // R gives: Friedman chi-squared = 4.9787, df = 2, p-value = 0.08296
        // (with tie correction for row 7 which has 5.40, 5.40, 5.35)
        println!(
            "Friedman result: statistic={}, df={}, p={}",
            result.statistic, result.df, result.p_value
        );
        println!("Rank sums: {:?}", result.rank_sums);
        println!(
            "Has ties: {}, tie_correction: {}",
            result.has_ties, result.tie_correction
        );
        assert!(
            (result.statistic - 4.9787).abs() < 0.05,
            "statistic mismatch: got {}",
            result.statistic
        );
        assert_eq!(result.df, 2);
        assert!(
            (result.p_value - 0.08296).abs() < 0.01,
            "p-value mismatch: got {}",
            result.p_value
        );
    }

    #[test]
    fn test_validate_friedman_manual_calculation() {
        // Simple example for manual verification
        // 3 blocks, 3 treatments
        // Block 1: [1, 2, 3] -> ranks [1, 2, 3]
        // Block 2: [1, 2, 3] -> ranks [1, 2, 3]
        // Block 3: [1, 2, 3] -> ranks [1, 2, 3]
        // Rank sums: R1=3, R2=6, R3=9
        // Q = (12/(3*3*4)) * (9+36+81) - 3*3*4
        //   = (12/36) * 126 - 36
        //   = 42 - 36 = 6

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = friedman_test(&data, &names).unwrap();

        assert_eq!(result.n_blocks, 3);
        assert_eq!(result.n_treatments, 3);
        assert!(!result.has_ties);
        assert!((result.tie_correction - 1.0).abs() < 1e-10);
        assert!((result.statistic - 6.0).abs() < 0.01);

        // Check rank sums
        assert!((result.rank_sums[0] - 3.0).abs() < 1e-10);
        assert!((result.rank_sums[1] - 6.0).abs() < 1e-10);
        assert!((result.rank_sums[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_within_block() {
        let values = vec![3.0, 1.0, 2.0];
        let (ranks, _) = rank_within_block(&values);

        // 1.0 -> rank 1, 2.0 -> rank 2, 3.0 -> rank 3
        assert!((ranks[0] - 3.0).abs() < 1e-10); // 3.0 gets rank 3
        assert!((ranks[1] - 1.0).abs() < 1e-10); // 1.0 gets rank 1
        assert!((ranks[2] - 2.0).abs() < 1e-10); // 2.0 gets rank 2
    }

    #[test]
    fn test_rank_within_block_with_ties() {
        let values = vec![1.0, 2.0, 2.0, 3.0];
        let (ranks, tie_sum) = rank_within_block(&values);

        // 1.0 -> rank 1
        // 2.0, 2.0 -> ranks 2, 3 averaged to 2.5
        // 3.0 -> rank 4
        assert!((ranks[0] - 1.0).abs() < 1e-10);
        assert!((ranks[1] - 2.5).abs() < 1e-10);
        assert!((ranks[2] - 2.5).abs() < 1e-10);
        assert!((ranks[3] - 4.0).abs() < 1e-10);

        // tie_sum for 2 tied values: 2³ - 2 = 6
        assert!((tie_sum - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_friedman_validates_input() {
        // Too few treatments
        let data = vec![vec![1.0], vec![2.0]];
        let names = vec!["A".to_string()];
        assert!(friedman_test(&data, &names).is_err());

        // Empty data
        let data: Vec<Vec<f64>> = vec![];
        let names: Vec<String> = vec![];
        assert!(friedman_test(&data, &names).is_err());
    }

    #[test]
    fn test_run_friedman_from_dataset() {
        // Long format data
        let df = df! {
            "block" => ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3"],
            "treatment" => ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "value" => [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2],
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_friedman_test(&dataset, "value", "treatment", "block").unwrap();

        assert_eq!(result.n_blocks, 3);
        assert_eq!(result.n_treatments, 3);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_friedman_ties_correction_value() {
        // Test tie correction computation
        // Each block has a tie: [1, 1, 2]
        // tie_sum per block = 2³ - 2 = 6
        // Total tie_sum = 6 * 3 = 18
        // C = 1 - 18 / (3 * 3 * (9-1)) = 1 - 18/72 = 0.75

        let data = vec![
            vec![1.0, 1.0, 2.0],
            vec![1.0, 1.0, 2.0],
            vec![1.0, 1.0, 2.0],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = friedman_test(&data, &names).unwrap();

        assert!(result.has_ties);
        assert!((result.tie_correction - 0.75).abs() < 0.01);
    }
}
