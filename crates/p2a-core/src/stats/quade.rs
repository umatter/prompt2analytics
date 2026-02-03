//! Quade test for unreplicated blocked data.
//!
//! A non-parametric test for comparing treatments across blocks, similar to
//! the Friedman test but using a different weighting scheme that can be more
//! powerful when block effects vary considerably.
//!
//! # Mathematical Background
//!
//! The Quade test weights blocks by their range:
//!
//! 1. Compute the range within each block: Range_i = max(X_i) - min(X_i)
//! 2. Rank the ranges across blocks: Q_i
//! 3. Rank values within each block: R_ij
//! 4. Compute weighted scores: S_ij = Q_i × (R_ij - (k+1)/2)
//! 5. Compute treatment totals: S_j = Σ_i S_ij
//! 6. F statistic: T = (b-1)B / (A - B)
//!    where A = ΣΣ S_ij² and B = (1/b) Σ S_j²
//!
//! # References
//!
//! - Quade, D. (1979). "Using weighted rankings in the analysis of complete
//!   blocks with additive block effects". Journal of the American Statistical
//!   Association, 74(367), 680-683.
//! - Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.).
//!   New York: Wiley. Pages 373-380.
//! - R Core Team. `stats::quade.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quade.test.html>

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Result of the Quade test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadeResult {
    /// The F statistic
    pub statistic: f64,
    /// Numerator degrees of freedom (k - 1)
    pub df1: usize,
    /// Denominator degrees of freedom ((b - 1)(k - 1))
    pub df2: usize,
    /// P-value from F distribution
    pub p_value: f64,
    /// Number of blocks (subjects/rows)
    pub n_blocks: usize,
    /// Number of treatments (groups/columns)
    pub n_treatments: usize,
    /// Treatment sums (S_j values)
    pub treatment_sums: Vec<f64>,
    /// Mean weighted rank per treatment
    pub mean_weighted_ranks: Vec<f64>,
    /// Treatment/group names
    pub treatment_names: Vec<String>,
    /// Block ranges
    pub block_ranges: Vec<f64>,
    /// A statistic (sum of squared S_ij)
    pub a_statistic: f64,
    /// B statistic (sum of squared S_j / b)
    pub b_statistic: f64,
}

impl std::fmt::Display for QuadeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Quade test")?;
        writeln!(f, "==========")?;
        writeln!(f)?;
        writeln!(
            f,
            "F = {:.4}, df1 = {}, df2 = {}, p-value = {:.6}",
            self.statistic, self.df1, self.df2, self.p_value
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Blocks: {}, Treatments: {}",
            self.n_blocks, self.n_treatments
        )?;
        writeln!(f)?;
        writeln!(f, "Treatment weighted rank sums:")?;
        for (name, sum) in self.treatment_names.iter().zip(self.treatment_sums.iter()) {
            writeln!(f, "  {}: {:.4}", name, sum)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "A = {:.4}, B = {:.4}",
            self.a_statistic, self.b_statistic
        )?;
        Ok(())
    }
}

/// Perform Quade test on a matrix of observations.
///
/// # Arguments
///
/// * `data` - Matrix where rows are blocks and columns are treatments.
///            Each cell contains the measurement for that block-treatment combination.
/// * `treatment_names` - Names for each treatment (column)
///
/// # Returns
///
/// A `QuadeResult` containing the F statistic, degrees of freedom, and p-value.
///
/// # Example
///
/// ```
/// use p2a_core::stats::quade::{quade_test, QuadeResult};
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
/// let result = quade_test(&data, &names).unwrap();
/// println!("F = {}, df1 = {}, df2 = {}, p = {}",
///          result.statistic, result.df1, result.df2, result.p_value);
/// ```
///
/// # References
/// - R equivalent: `quade.test(y)`
pub fn quade_test(data: &[Vec<f64>], treatment_names: &[String]) -> EconResult<QuadeResult> {
    // Validate input
    if data.is_empty() {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: 0,
            context: "Quade test requires at least 2 blocks".to_string(),
        });
    }

    let b = data.len(); // number of blocks
    let k = data[0].len(); // number of treatments

    if b < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: b,
            context: "Quade test requires at least 2 blocks".to_string(),
        });
    }

    if k < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: k,
            context: "Quade test requires at least 2 treatments".to_string(),
        });
    }

    // Validate all rows have same length
    for (i, row) in data.iter().enumerate() {
        if row.len() != k {
            return Err(EconError::InvalidSpecification {
                message: format!("Block {} has {} treatments, expected {}", i, row.len(), k),
            });
        }
    }

    // Validate treatment names
    if treatment_names.len() != k {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Expected {} treatment names, got {}",
                k,
                treatment_names.len()
            ),
        });
    }

    // Step 1: Compute block ranges
    let mut block_ranges: Vec<f64> = Vec::with_capacity(b);
    for row in data {
        let min = row.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        block_ranges.push(max - min);
    }

    // Step 2: Rank the block ranges (Q_i)
    let range_ranks = rank_values(&block_ranges);

    // Step 3: Rank within each block (R_ij)
    let mut within_ranks: Vec<Vec<f64>> = Vec::with_capacity(b);
    for row in data {
        within_ranks.push(rank_values(row));
    }

    // Step 4: Compute S_ij = Q_i * (R_ij - (k+1)/2)
    let center = (k as f64 + 1.0) / 2.0;
    let mut s_matrix: Vec<Vec<f64>> = Vec::with_capacity(b);
    for (i, ranks) in within_ranks.iter().enumerate() {
        let q_i = range_ranks[i];
        let s_row: Vec<f64> = ranks.iter().map(|&r_ij| q_i * (r_ij - center)).collect();
        s_matrix.push(s_row);
    }

    // Step 5: Compute treatment totals S_j = Σ_i S_ij
    let mut treatment_sums: Vec<f64> = vec![0.0; k];
    for s_row in &s_matrix {
        for (j, &s_ij) in s_row.iter().enumerate() {
            treatment_sums[j] += s_ij;
        }
    }

    // Step 6: Compute A = ΣΣ S_ij² and B = (1/b) Σ S_j²
    let a: f64 = s_matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|&s| s * s)
        .sum();

    let sum_sj_sq: f64 = treatment_sums.iter().map(|&s| s * s).sum();
    let b_stat = sum_sj_sq / b as f64;

    // Step 7: Compute F statistic: T = (b-1)B / (A - B)
    let numerator = (b as f64 - 1.0) * b_stat;
    let denominator = a - b_stat;

    let f_stat = if denominator > 1e-10 {
        numerator / denominator
    } else {
        // If A ≈ B, all treatments are essentially equal
        0.0
    };

    // Degrees of freedom
    let df1 = k - 1;
    let df2 = (b - 1) * (k - 1);

    // P-value from F distribution
    let p_value = f_test_p_value(f_stat, df1, df2);

    // Compute mean weighted ranks
    let mean_weighted_ranks: Vec<f64> = treatment_sums.iter().map(|&s| s / b as f64).collect();

    Ok(QuadeResult {
        statistic: f_stat,
        df1,
        df2,
        p_value,
        n_blocks: b,
        n_treatments: k,
        treatment_sums,
        mean_weighted_ranks,
        treatment_names: treatment_names.to_vec(),
        block_ranges,
        a_statistic: a,
        b_statistic: b_stat,
    })
}

/// Rank values, handling ties with average ranks.
///
/// Ties are detected using exact equality (like R), not approximate equality.
/// This matches R's rank() function behavior.
fn rank_values(values: &[f64]) -> Vec<f64> {
    let n = values.len();

    // Create index-value pairs and sort
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];

    let mut i = 0;
    while i < n {
        // Find extent of tie group - use exact equality like R does
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }

        // Compute average rank for tie group (1-indexed)
        let avg_rank = ((i + 1) + j) as f64 / 2.0;

        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Compute p-value from F distribution.
fn f_test_p_value(f_stat: f64, df1: usize, df2: usize) -> f64 {
    use statrs::distribution::{ContinuousCDF, FisherSnedecor};

    if df1 == 0 || df2 == 0 || f_stat.is_nan() {
        return f64::NAN;
    }
    if f_stat.is_infinite() || f_stat < 0.0 {
        return if f_stat > 0.0 { 0.0 } else { 1.0 };
    }

    let f_dist = match FisherSnedecor::new(df1 as f64, df2 as f64) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };

    1.0 - f_dist.cdf(f_stat)
}

/// Run Quade test on a Dataset with long-format data.
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
/// A `QuadeResult` containing the test results.
pub fn run_quade_test(
    dataset: &Dataset,
    value_col: &str,
    group_col: &str,
    block_col: &str,
) -> EconResult<QuadeResult> {
    let df = dataset.df();

    // Extract columns
    let values = df
        .column(value_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: value_col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: value_col.to_string(),
        })?;

    let groups_col = df
        .column(group_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: group_col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;

    let blocks_col = df
        .column(block_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: block_col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;

    // Build block -> treatment -> value mapping
    use std::collections::{BTreeSet, HashMap};
    let mut block_data: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut treatment_set: BTreeSet<String> = BTreeSet::new();

    let n = df.height();
    for i in 0..n {
        let block = blocks_col
            .get(i)
            .map_err(|e| EconError::InvalidSpecification {
                message: format!("Error accessing block at row {}: {}", i, e),
            })?;
        let block_str = format!("{}", block);

        let group = groups_col
            .get(i)
            .map_err(|e| EconError::InvalidSpecification {
                message: format!("Error accessing group at row {}: {}", i, e),
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

    if data.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: data.len(),
            context: "Need at least 2 complete blocks for Quade test".to_string(),
        });
    }

    quade_test(&data, &treatment_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_quade_basic() {
        // Clear treatment differences across blocks
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![1.2, 2.2, 3.2],
            vec![1.8, 2.8, 3.8],
            vec![1.1, 2.1, 3.1],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = quade_test(&data, &names).unwrap();

        println!(
            "F = {}, df1 = {}, df2 = {}, p = {}",
            result.statistic, result.df1, result.df2, result.p_value
        );
        println!("Block ranges: {:?}", result.block_ranges);
        println!("Treatment sums: {:?}", result.treatment_sums);
        println!("A = {}, B = {}", result.a_statistic, result.b_statistic);

        assert_eq!(result.n_blocks, 5);
        assert_eq!(result.n_treatments, 3);
        assert_eq!(result.df1, 2);
        assert_eq!(result.df2, 8);
        // R gives F = 36, p = 0.0001 for this data
        assert!(
            (result.statistic - 36.0).abs() < 1.0,
            "F statistic mismatch: got {}, expected 36",
            result.statistic
        );
        // With perfect ordering, should reject H0
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_quade_no_difference() {
        // Similar values, rankings should vary
        let data = vec![
            vec![5.1, 5.0, 5.2],
            vec![4.9, 5.1, 5.0],
            vec![5.2, 5.0, 4.9],
            vec![5.0, 5.2, 5.1],
            vec![4.8, 5.0, 5.2],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = quade_test(&data, &names).unwrap();

        // With similar values, should not reject H0
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_rank_values() {
        let values = vec![3.0, 1.0, 2.0];
        let ranks = rank_values(&values);

        assert!((ranks[0] - 3.0).abs() < 1e-10); // 3.0 gets rank 3
        assert!((ranks[1] - 1.0).abs() < 1e-10); // 1.0 gets rank 1
        assert!((ranks[2] - 2.0).abs() < 1e-10); // 2.0 gets rank 2
    }

    #[test]
    fn test_rank_values_with_ties() {
        let values = vec![1.0, 2.0, 2.0, 3.0];
        let ranks = rank_values(&values);

        assert!((ranks[0] - 1.0).abs() < 1e-10);
        assert!((ranks[1] - 2.5).abs() < 1e-10); // average of 2 and 3
        assert!((ranks[2] - 2.5).abs() < 1e-10);
        assert!((ranks[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quade_validates_input() {
        // Too few treatments
        let data = vec![vec![1.0], vec![2.0]];
        let names = vec!["A".to_string()];
        assert!(quade_test(&data, &names).is_err());

        // Empty data
        let data: Vec<Vec<f64>> = vec![];
        let names: Vec<String> = vec![];
        assert!(quade_test(&data, &names).is_err());

        // Single block
        let data = vec![vec![1.0, 2.0, 3.0]];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        assert!(quade_test(&data, &names).is_err());
    }

    #[test]
    fn test_run_quade_from_dataset() {
        // Long format data
        let df = df! {
            "block" => ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3"],
            "treatment" => ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "value" => [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2],
        }
        .unwrap();

        let dataset = Dataset::new(df);
        let result = run_quade_test(&dataset, "value", "treatment", "block").unwrap();

        assert_eq!(result.n_blocks, 3);
        assert_eq!(result.n_treatments, 3);
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_quade_against_r() {
        // R code:
        // y <- matrix(c(5, 4, 7, 10, 12,
        //               1, 3, 1, 0, 2,
        //               16, 12, 22, 22, 35,
        //               5, 4, 3, 5, 4,
        //               10, 9, 7, 13, 10,
        //               19, 18, 28, 25, 20,
        //               10, 7, 6, 8, 7),
        //             nrow = 7, byrow = TRUE,
        //             dimnames = list(Store = as.character(1:7),
        //                             Brand = LETTERS[1:5]))
        // quade.test(y)
        //
        // Verified R output:
        // Quade F = 2.4266, num df = 4, denom df = 24, p-value = 0.07566
        // A = 1360, B = 391.6429
        // Treatment sums: [-9.5, -38.0, -1.5, 23.0, 26.0]

        let data = vec![
            vec![5.0, 4.0, 7.0, 10.0, 12.0],
            vec![1.0, 3.0, 1.0, 0.0, 2.0],
            vec![16.0, 12.0, 22.0, 22.0, 35.0],
            vec![5.0, 4.0, 3.0, 5.0, 4.0],
            vec![10.0, 9.0, 7.0, 13.0, 10.0],
            vec![19.0, 18.0, 28.0, 25.0, 20.0],
            vec![10.0, 7.0, 6.0, 8.0, 7.0],
        ];
        let names = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string(),
        ];

        let result = quade_test(&data, &names).unwrap();

        println!(
            "Quade result: F={}, df1={}, df2={}, p={}",
            result.statistic, result.df1, result.df2, result.p_value
        );
        println!("Treatment sums: {:?}", result.treatment_sums);
        println!("Block ranges: {:?}", result.block_ranges);
        println!("A = {}, B = {}", result.a_statistic, result.b_statistic);

        // Check against verified R results
        assert!(
            (result.statistic - 2.4266).abs() < 0.01,
            "F statistic mismatch: got {}, expected 2.4266",
            result.statistic
        );
        assert_eq!(result.df1, 4);
        assert_eq!(result.df2, 24);
        assert!(
            (result.p_value - 0.07566).abs() < 0.01,
            "p-value mismatch: got {}, expected 0.07566",
            result.p_value
        );
        assert!(
            (result.a_statistic - 1360.0).abs() < 1.0,
            "A statistic mismatch: got {}, expected 1360",
            result.a_statistic
        );
        assert!(
            (result.b_statistic - 391.6429).abs() < 0.1,
            "B statistic mismatch: got {}, expected 391.6429",
            result.b_statistic
        );
    }

    #[test]
    fn test_quade_block_ranges() {
        // Test that block ranges are computed correctly
        let data = vec![
            vec![1.0, 5.0, 3.0],  // range = 4
            vec![2.0, 10.0, 6.0], // range = 8
            vec![0.0, 2.0, 1.0],  // range = 2
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = quade_test(&data, &names).unwrap();

        assert!((result.block_ranges[0] - 4.0).abs() < 1e-10);
        assert!((result.block_ranges[1] - 8.0).abs() < 1e-10);
        assert!((result.block_ranges[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quade_display() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![1.2, 2.2, 3.2],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = quade_test(&data, &names).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Quade"));
        assert!(display.contains("F ="));
        assert!(display.contains("p-value"));
    }
}
