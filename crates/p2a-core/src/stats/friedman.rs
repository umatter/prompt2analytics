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

use std::collections::HashMap;

use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{MissingSummary, factor_options, numeric_options};
use crate::errors::{EconError, EconResult};
use crate::resource;

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
    /// Rows of the blocks dropped because a response is missing (R drops such blocks);
    /// empty for the array-level function.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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
        missing: MissingSummary::none(n_blocks * n_treatments),
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

/// A long-format block design pivoted to one row per complete block, by R's rules.
pub(crate) struct UnreplicatedBlocks {
    /// One row per complete block (block level order), one column per treatment (treatment
    /// level order).
    pub rows: Vec<Vec<f64>>,
    /// Treatment labels, in R `factor()` order.
    pub treatment_names: Vec<String>,
    /// Rows of the blocks dropped because a response is missing.
    pub missing: MissingSummary,
}

/// At most 40 characters of a label, for error messages.
fn short_label(label: &str) -> String {
    let mut shown: String = label.chars().take(40).collect();
    if label.chars().nth(40).is_some() {
        shown.push_str("...");
    }
    shown
}

/// Pivot `value ~ group | block` to a block x treatment table with the checks of R's
/// `friedman.test` and `quade.test`, in R's order:
///
/// 1. The value column must be numeric (`NonNumericColumn`; `ColumnNotFound` for an absent
///    column; `NonFiniteValues` for `±Inf`).
/// 2. A missing group or block label is an error ("NA's are not allowed in 'groups' or
///    'blocks'"), checked for the group column first, whatever the response is.
/// 3. Every (block, treatment) cell must hold exactly one observation: a replicated or an
///    absent cell is "not an unreplicated complete block design".
/// 4. A block with a missing response is dropped (R's `complete.cases` on the table) and
///    reported in `missing`.
///
/// `test` ("Friedman test", "Quade test") names the test in the messages. One O(n) pass
/// builds the cells; the table is allocated only once it is known to be complete, so its
/// size is the number of rows.
pub(crate) fn unreplicated_blocks(
    df: &DataFrame,
    value_col: &str,
    group_col: &str,
    block_col: &str,
    test: &str,
) -> EconResult<UnreplicatedBlocks> {
    let values = numeric_options(df, value_col)?;
    let (groups, group_levels) = factor_options(df, group_col)?;
    let (blocks, block_levels) = factor_options(df, block_col)?;
    for (codes, col) in [(&groups, group_col), (&blocks, block_col)] {
        let k = codes.iter().filter(|c| c.is_none()).count();
        if k > 0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "NA's are not allowed in 'groups' or 'blocks' ({test}, as R): column \
                     '{col}' has {k} missing value(s)"
                ),
            });
        }
    }
    let n = values.len();
    let (n_blocks, n_groups) = (block_levels.len(), group_levels.len());
    let not_unreplicated = |detail: String| EconError::InvalidSpecification {
        message: format!("not an unreplicated complete block design ({test}): {detail}"),
    };

    // (block, treatment) -> row, in one pass.
    let mut cell_row: HashMap<(u32, u32), usize> = HashMap::new();
    for (i, (b, g)) in blocks.iter().zip(&groups).enumerate() {
        let key = (b.unwrap_or(0), g.unwrap_or(0));
        if cell_row.insert(key, i).is_some() {
            return Err(not_unreplicated(format!(
                "block '{}' has more than one observation of treatment '{}'",
                short_label(&block_levels[key.0 as usize]),
                short_label(&group_levels[key.1 as usize])
            )));
        }
    }
    let what = format!("{test} block x treatment table");
    let cells = resource::checked_len(&[n_blocks, n_groups], &what)?;
    if cell_row.len() < cells {
        // The first absent cell in level order: at most n + 1 cells are visited, since every
        // cell before it is present.
        for (b, block) in block_levels.iter().enumerate() {
            for (g, group) in group_levels.iter().enumerate() {
                if !cell_row.contains_key(&(b as u32, g as u32)) {
                    return Err(not_unreplicated(format!(
                        "block '{}' has no observation of treatment '{}'",
                        short_label(block),
                        short_label(group)
                    )));
                }
            }
        }
    }

    // The table (cells == n here), block-major, then the complete blocks.
    let mut table: Vec<Option<f64>> = resource::try_vec(cells, None, &what)?;
    for (&(b, g), &i) in &cell_row {
        table[b as usize * n_groups + g as usize] = values[i];
    }
    let mut rows: Vec<Vec<f64>> = resource::try_with_capacity(n_blocks, &what)?;
    let mut missing_cells = 0usize;
    let mut dropped_blocks = 0usize;
    for block in table.chunks_exact(n_groups.max(1)) {
        let missing_here = block.iter().filter(|v| v.is_none()).count();
        if missing_here > 0 {
            missing_cells += missing_here;
            dropped_blocks += 1;
        } else {
            rows.push(block.iter().map(|v| v.unwrap_or(f64::NAN)).collect());
        }
    }
    let n_dropped = dropped_blocks * n_groups;
    Ok(UnreplicatedBlocks {
        rows,
        treatment_names: group_levels,
        missing: MissingSummary {
            n_total: n,
            n_used: n - n_dropped,
            n_dropped,
            by_column: if missing_cells > 0 {
                vec![(value_col.to_string(), missing_cells)]
            } else {
                Vec::new()
            },
        },
    })
}

/// Run Friedman test on a Dataset with long-format data (R
/// `friedman.test(value, group, block)`).
///
/// The design must be an unreplicated complete block design, as R requires: a missing group
/// or block label is an error, and so is a (block, treatment) cell with more than one
/// observation or with none. A block whose response is missing is dropped (as R does) and
/// reported in `missing`. Integer responses are accepted; treatments follow R `factor()`
/// order.
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
    let blocks = unreplicated_blocks(
        dataset.df(),
        value_col,
        group_col,
        block_col,
        "Friedman test",
    )?;
    if blocks.rows.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "No complete blocks found for Friedman test".to_string(),
        });
    }
    let mut result = friedman_test(&blocks.rows, &blocks.treatment_names)?;
    result.missing = blocks.missing;
    Ok(result)
}

/// Frames derived from the case `stat_friedman_quade` (shared by the Friedman and Quade
/// tests).
#[cfg(test)]
pub(crate) mod fixtures {
    use polars::prelude::*;

    /// The frame with a second (S1, A) row appended: a replicated cell.
    pub(crate) fn with_duplicate_cell(df: &DataFrame) -> DataFrame {
        let extra = df! {
            "y" => [99i64],
            "y_na" => [Some(99i64)],
            "treat" => ["A"],
            "block" => ["S1"],
        }
        .unwrap();
        df.vstack(&extra).unwrap()
    }

    /// The frame with the text column `col` missing in row `row`.
    pub(crate) fn with_null_label(df: &DataFrame, col: &str, row: usize) -> DataFrame {
        let labels: StringChunked = df
            .column(col)
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| if i == row { None } else { v })
            .collect();
        let mut out = df.clone();
        out.with_column(labels.with_name(col.into()).into_series())
            .unwrap();
        out
    }

    /// The frame with the Int64 column `col` missing in row `row`.
    pub(crate) fn with_null_value(df: &DataFrame, col: &str, row: usize) -> DataFrame {
        let values: Int64Chunked = df
            .column(col)
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| if i == row { None } else { v })
            .collect();
        let mut out = df.clone();
        out.with_column(values.with_name(col.into()).into_series())
            .unwrap();
        out
    }

    /// The frame without row `row`: a block without one treatment.
    pub(crate) fn without_row(df: &DataFrame, row: usize) -> DataFrame {
        let mask: BooleanChunked = (0..df.height()).map(|i| Some(i != row)).collect();
        df.filter(&mask).unwrap()
    }

    /// The frame with `col` cast to Float64.
    pub(crate) fn as_float(df: &DataFrame, col: &str) -> DataFrame {
        let cast = df.column(col).unwrap().cast(&DataType::Float64).unwrap();
        let mut out = df.clone();
        out.with_column(cast).unwrap();
        out
    }
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

    // ========================================================================
    // R-pinned: complete blocks, missing responses and refused designs (T2-WP07)
    // ========================================================================
    //
    // Case stat_friedman_quade: 8 blocks x 3 treatments, an Int64 response y, and y_na with
    // block S2 incomplete (R drops the block). R refuses a replicated cell, an absent cell
    // and a missing block or treatment label.

    use super::fixtures::*;
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_validate_friedman_quade_friedman() {
        let (case, ds) = load_case("stat_friedman_quade");
        let f = run_friedman_test(&ds, "y", "treat", "block").unwrap();
        let fna = run_friedman_test(&ds, "y_na", "treat", "block").unwrap();
        let (s, p, x) = (Tol::stat(24), Tol::p_value(24), Tol::exact());
        Check::new(&case)
            .scalar("friedman_q", f.statistic, s)
            .scalar("friedman_df", f.df as f64, x)
            .scalar("friedman_p", f.p_value, p)
            .scalar("na_friedman_q", fna.statistic, s)
            .scalar("na_friedman_p", fna.p_value, p)
            .scalar("na_n_blocks", fna.n_blocks as f64, x)
            .finish();
        assert_eq!(f.treatment_names, ["A", "B", "C"]);
        assert_eq!(f.n_blocks, 8);
        assert!(f.missing.is_empty());
        assert_eq!(fna.missing.n_dropped, 3);
        assert_eq!(fna.missing.n_used, 21);
        assert_eq!(fna.missing.by_column, vec![("y_na".to_string(), 1)]);
    }

    fn friedman_refusal(df: DataFrame) -> String {
        match run_friedman_test(&Dataset::new(df), "y", "treat", "block") {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_friedman_refuses_designs_r_refuses() {
        let (_case, ds) = load_case("stat_friedman_quade");
        let df = ds.df();
        let design = "not an unreplicated complete block design";
        let na = "NA's are not allowed in 'groups' or 'blocks'";

        let m = friedman_refusal(with_duplicate_cell(df));
        assert!(m.contains(design) && m.contains("'S1'"), "{m}");
        let m = friedman_refusal(with_null_label(df, "block", 3));
        assert!(m.contains(na) && m.contains("'block'"), "{m}");
        let m = friedman_refusal(with_null_label(df, "treat", 3));
        assert!(m.contains(na) && m.contains("'treat'"), "{m}");
        let m = friedman_refusal(without_row(df, 4));
        assert!(m.contains(design), "{m}");
        // R checks the labels before the response: still the NA-label error.
        let m = friedman_refusal(with_null_value(&with_null_label(df, "block", 3), "y", 3));
        assert!(m.contains(na) && m.contains("'block'"), "{m}");
    }

    /// Column-type fixture: an Int64 response equals its Float64 cast; a text value column and
    /// an absent column are refused with the variant that names the problem.
    #[test]
    fn test_validate_na_friedman_dtypes_and_errors() {
        let (_case, ds) = load_case("stat_friedman_quade");
        assert_eq!(ds.df().column("y").unwrap().dtype(), &DataType::Int64);
        let int = run_friedman_test(&ds, "y", "treat", "block").unwrap();
        let float = run_friedman_test(&Dataset::new(as_float(ds.df(), "y")), "y", "treat", "block")
            .unwrap();
        assert_eq!(int.statistic, float.statistic);
        assert!(matches!(
            run_friedman_test(&ds, "treat", "treat", "block"),
            Err(EconError::NonNumericColumn { ref column }) if column == "treat"
        ));
        assert!(matches!(
            run_friedman_test(&ds, "nope", "treat", "block"),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
        assert!(matches!(
            run_friedman_test(&ds, "y", "treat", "nope"),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
    }
}
