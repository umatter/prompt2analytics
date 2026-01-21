//! Pearson's Chi-Squared Test for categorical data.
//!
//! Provides chi-squared tests for:
//! - Goodness-of-fit: Tests whether observed frequencies match expected probabilities
//! - Independence: Tests whether two categorical variables are independent
//!
//! # References
//!
//! - Pearson, K. (1900). "On the criterion that a given system of deviations from
//!   the probable in the case of a correlated system of variables is such that it
//!   can be reasonably supposed to have arisen from random sampling".
//!   *Philosophical Magazine*, Series 5, 50(302), 157-175.
//! - Yates, F. (1934). "Contingency tables involving small numbers and the χ² test".
//!   *Supplement to the Journal of the Royal Statistical Society*, 1(2), 217-235.
//! - R Core Team. `stats::chisq.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/chisq.test.html>
//!
//! # Mathematical Background
//!
//! ## Chi-Squared Test Statistic
//!
//! ```text
//! χ² = Σ (O_i - E_i)² / E_i
//! ```
//!
//! ## Goodness-of-Fit Test
//!
//! Tests H₀: The population probabilities equal specified values p_i.
//!
//! ```text
//! E_i = n × p_i
//! df = k - 1  (where k = number of categories)
//! ```
//!
//! ## Test of Independence
//!
//! Tests H₀: Row and column variables are independent.
//!
//! ```text
//! E_ij = (row_i_total × col_j_total) / grand_total
//! df = (r - 1)(c - 1)  (where r = rows, c = columns)
//! ```
//!
//! ## Yates' Continuity Correction (2×2 tables only)
//!
//! ```text
//! χ² = Σ (|O_ij - E_ij| - 0.5)² / E_ij
//! ```
//!
//! Applied when expected counts are small to improve approximation to chi-squared distribution.

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::{chi_squared_p_value, SignificanceLevel};

/// Result of a chi-squared test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquaredResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Test Type
    // ═══════════════════════════════════════════════════════════════════════
    /// Description of the test performed
    pub test_name: String,

    // ═══════════════════════════════════════════════════════════════════════
    // Test Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Chi-squared test statistic
    pub statistic: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,

    // ═══════════════════════════════════════════════════════════════════════
    // Observed and Expected Counts
    // ═══════════════════════════════════════════════════════════════════════
    /// Observed counts (flattened for contingency tables)
    pub observed: Vec<f64>,
    /// Expected counts under the null hypothesis
    pub expected: Vec<f64>,

    // ═══════════════════════════════════════════════════════════════════════
    // Residuals
    // ═══════════════════════════════════════════════════════════════════════
    /// Pearson residuals: (O - E) / sqrt(E)
    pub residuals: Vec<f64>,
    /// Standardized residuals (for independence test)
    pub std_residuals: Option<Vec<f64>>,

    // ═══════════════════════════════════════════════════════════════════════
    // Table Dimensions (for independence test)
    // ═══════════════════════════════════════════════════════════════════════
    /// Number of rows (for contingency tables)
    pub n_rows: Option<usize>,
    /// Number of columns (for contingency tables)
    pub n_cols: Option<usize>,

    // ═══════════════════════════════════════════════════════════════════════
    // Options Used
    // ═══════════════════════════════════════════════════════════════════════
    /// Whether Yates' continuity correction was applied
    pub yates_correction: bool,
}

impl std::fmt::Display for ChiSquaredResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Test statistic
        writeln!(
            f,
            "X-squared = {:.6}, df = {}, p-value = {:.6} {}",
            self.statistic,
            self.df,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        if self.yates_correction {
            writeln!(f, "(with Yates' continuity correction)")?;
            writeln!(f)?;
        }

        // For small tables, show observed vs expected
        if self.observed.len() <= 20 {
            writeln!(f, "Observed vs Expected:")?;

            if let (Some(n_rows), Some(n_cols)) = (self.n_rows, self.n_cols) {
                // Contingency table format
                for i in 0..n_rows {
                    write!(f, "  Row {}: ", i + 1)?;
                    for j in 0..n_cols {
                        let idx = i * n_cols + j;
                        write!(
                            f,
                            " {:>7.1} ({:>7.1})",
                            self.observed[idx], self.expected[idx]
                        )?;
                    }
                    writeln!(f)?;
                }
            } else {
                // One-dimensional (goodness-of-fit)
                for (i, (o, e)) in self.observed.iter().zip(&self.expected).enumerate() {
                    writeln!(f, "  [{}]: {:.1} (expected: {:.1})", i + 1, o, e)?;
                }
            }
        }

        // Warning for small expected counts
        let small_expected = self.expected.iter().filter(|&&e| e < 5.0).count();
        if small_expected > 0 {
            writeln!(f)?;
            writeln!(
                f,
                "Warning: {} cell(s) have expected count < 5. Chi-squared approximation may be unreliable.",
                small_expected
            )?;
        }

        Ok(())
    }
}

/// Chi-squared goodness-of-fit test.
///
/// Tests whether the observed frequency distribution differs from a specified
/// theoretical distribution.
///
/// # Arguments
///
/// * `observed` - Observed counts/frequencies (must be non-negative)
/// * `probs` - Expected probabilities (if None, assumes uniform distribution)
/// * `rescale_p` - If true, rescale probabilities to sum to 1
///
/// # Returns
///
/// `ChiSquaredResult` containing test statistic, p-value, and diagnostics.
///
/// # Mathematical Details
///
/// The test statistic is:
/// ```text
/// χ² = Σᵢ (Oᵢ - Eᵢ)² / Eᵢ
/// ```
///
/// where Eᵢ = n × pᵢ and df = k - 1.
///
/// # Example
///
/// ```
/// use p2a_core::chisq_test_gof;
///
/// // Test if a die is fair
/// let observed = vec![16.0, 18.0, 22.0, 14.0, 15.0, 15.0]; // 100 rolls
/// let result = chisq_test_gof(&observed, None, false).unwrap();
/// println!("{}", result);
/// ```
pub fn chisq_test_gof(
    observed: &[f64],
    probs: Option<&[f64]>,
    rescale_p: bool,
) -> EconResult<ChiSquaredResult> {
    let k = observed.len();

    if k < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: k,
            context: "Chi-squared test requires at least 2 categories".to_string(),
        });
    }

    // Validate observed counts
    if observed.iter().any(|&x| x < 0.0) {
        return Err(EconError::InvalidSpecification {
            message: "Observed counts must be non-negative".to_string(),
        });
    }

    let n: f64 = observed.iter().sum();
    if n <= 0.0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Total count must be positive".to_string(),
        });
    }

    // Determine expected probabilities
    let expected_probs: Vec<f64> = if let Some(p) = probs {
        if p.len() != k {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Probability vector length ({}) must match observed length ({})",
                    p.len(),
                    k
                ),
            });
        }
        if p.iter().any(|&x| x < 0.0) {
            return Err(EconError::InvalidSpecification {
                message: "Probabilities must be non-negative".to_string(),
            });
        }
        let sum: f64 = p.iter().sum();
        if rescale_p {
            p.iter().map(|&x| x / sum).collect()
        } else {
            if (sum - 1.0).abs() > 1e-6 {
                return Err(EconError::InvalidSpecification {
                    message: "Probabilities must sum to 1 (or use rescale_p=true)".to_string(),
                });
            }
            p.to_vec()
        }
    } else {
        // Uniform distribution
        vec![1.0 / k as f64; k]
    };

    // Calculate expected counts
    let expected: Vec<f64> = expected_probs.iter().map(|&p| n * p).collect();

    // Calculate chi-squared statistic
    // χ² = Σ (O - E)² / E
    let statistic: f64 = observed
        .iter()
        .zip(&expected)
        .map(|(&o, &e)| {
            if e > 0.0 {
                (o - e).powi(2) / e
            } else if o > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        })
        .sum();

    // Degrees of freedom
    let df = k - 1;

    // P-value
    let p_value = chi_squared_p_value(statistic, df as f64);
    let significance = SignificanceLevel::from_p_value(p_value);

    // Pearson residuals: (O - E) / sqrt(E)
    let residuals: Vec<f64> = observed
        .iter()
        .zip(&expected)
        .map(|(&o, &e)| if e > 0.0 { (o - e) / e.sqrt() } else { 0.0 })
        .collect();

    Ok(ChiSquaredResult {
        test_name: "Chi-squared test for given probabilities".to_string(),
        statistic,
        df,
        p_value,
        significance,
        observed: observed.to_vec(),
        expected,
        residuals,
        std_residuals: None,
        n_rows: None,
        n_cols: None,
        yates_correction: false,
    })
}

/// Chi-squared test of independence for contingency tables.
///
/// Tests whether the row and column variables of a contingency table are
/// independent.
///
/// # Arguments
///
/// * `table` - 2D contingency table as Vec<Vec<f64>> (rows × columns)
/// * `correct` - If true and table is 2×2, apply Yates' continuity correction
///
/// # Returns
///
/// `ChiSquaredResult` containing test statistic, p-value, and diagnostics.
///
/// # Mathematical Details
///
/// Expected value for cell (i,j):
/// ```text
/// E_ij = (row_i_total × col_j_total) / grand_total
/// ```
///
/// Test statistic:
/// ```text
/// χ² = Σᵢⱼ (O_ij - E_ij)² / E_ij
/// ```
///
/// Degrees of freedom: df = (r - 1)(c - 1)
///
/// With Yates' correction (2×2 only):
/// ```text
/// χ² = Σᵢⱼ (|O_ij - E_ij| - 0.5)² / E_ij
/// ```
///
/// # Example
///
/// ```
/// use p2a_core::chisq_test_independence;
///
/// // Test independence of gender and party preference
/// let table = vec![
///     vec![762.0, 327.0, 468.0], // Female: Democrat, Independent, Republican
///     vec![484.0, 239.0, 477.0], // Male: Democrat, Independent, Republican
/// ];
/// let result = chisq_test_independence(&table, true).unwrap();
/// println!("{}", result);
/// ```
pub fn chisq_test_independence(table: &[Vec<f64>], correct: bool) -> EconResult<ChiSquaredResult> {
    let n_rows = table.len();

    if n_rows < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_rows,
            context: "Independence test requires at least 2 rows".to_string(),
        });
    }

    let n_cols = table[0].len();

    if n_cols < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_cols,
            context: "Independence test requires at least 2 columns".to_string(),
        });
    }

    // Check all rows have same length
    if table.iter().any(|row| row.len() != n_cols) {
        return Err(EconError::InvalidSpecification {
            message: "All rows must have the same number of columns".to_string(),
        });
    }

    // Validate all counts are non-negative
    if table.iter().flatten().any(|&x| x < 0.0) {
        return Err(EconError::InvalidSpecification {
            message: "All counts must be non-negative".to_string(),
        });
    }

    // Calculate marginals
    let row_totals: Vec<f64> = table.iter().map(|row| row.iter().sum()).collect();
    let col_totals: Vec<f64> = (0..n_cols)
        .map(|j| table.iter().map(|row| row[j]).sum())
        .collect();
    let grand_total: f64 = row_totals.iter().sum();

    if grand_total <= 0.0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Total count must be positive".to_string(),
        });
    }

    // Calculate expected values: E_ij = row_i × col_j / n
    let mut expected: Vec<f64> = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            expected.push(row_totals[i] * col_totals[j] / grand_total);
        }
    }

    // Flatten observed
    let observed: Vec<f64> = table.iter().flatten().copied().collect();

    // Determine if Yates' correction should be applied
    let apply_yates = correct && n_rows == 2 && n_cols == 2;

    // Calculate chi-squared statistic
    let statistic: f64 = observed
        .iter()
        .zip(&expected)
        .map(|(&o, &e)| {
            if e > 0.0 {
                if apply_yates {
                    // Yates' correction: (|O - E| - 0.5)² / E
                    let diff = (o - e).abs() - 0.5;
                    let diff = diff.max(0.0); // Don't go negative
                    diff.powi(2) / e
                } else {
                    (o - e).powi(2) / e
                }
            } else if o > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        })
        .sum();

    // Degrees of freedom: (r - 1)(c - 1)
    let df = (n_rows - 1) * (n_cols - 1);

    // P-value
    let p_value = chi_squared_p_value(statistic, df as f64);
    let significance = SignificanceLevel::from_p_value(p_value);

    // Pearson residuals: (O - E) / sqrt(E)
    let residuals: Vec<f64> = observed
        .iter()
        .zip(&expected)
        .map(|(&o, &e)| if e > 0.0 { (o - e) / e.sqrt() } else { 0.0 })
        .collect();

    // Standardized residuals: (O - E) / sqrt(E × (1 - row_prop) × (1 - col_prop))
    let mut std_residuals: Vec<f64> = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            let e = expected[idx];
            if e > 0.0 {
                let row_prop = row_totals[i] / grand_total;
                let col_prop = col_totals[j] / grand_total;
                let variance = e * (1.0 - row_prop) * (1.0 - col_prop);
                if variance > 0.0 {
                    std_residuals.push((observed[idx] - e) / variance.sqrt());
                } else {
                    std_residuals.push(0.0);
                }
            } else {
                std_residuals.push(0.0);
            }
        }
    }

    let test_name = if apply_yates {
        "Pearson's Chi-squared test with Yates' continuity correction".to_string()
    } else {
        "Pearson's Chi-squared test".to_string()
    };

    Ok(ChiSquaredResult {
        test_name,
        statistic,
        df,
        p_value,
        significance,
        observed,
        expected,
        residuals,
        std_residuals: Some(std_residuals),
        n_rows: Some(n_rows),
        n_cols: Some(n_cols),
        yates_correction: apply_yates,
    })
}

/// Run chi-squared goodness-of-fit test on a dataset column.
///
/// Counts occurrences of each unique value and tests against expected probabilities.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `column` - Column name containing categorical values
/// * `probs` - Optional expected probabilities (uniform if None)
///
/// # Example
///
/// ```ignore
/// let result = run_chisq_gof(&dataset, "category", None)?;
/// println!("{}", result);
/// ```
pub fn run_chisq_gof(
    dataset: &Dataset,
    column: &str,
    probs: Option<&[f64]>,
) -> EconResult<ChiSquaredResult> {
    use polars::prelude::*;

    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(EconError::ColumnNotFound {
            column: column.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        });
    }

    // Use lazy frame for value_counts (via Expr, not Series method)
    let lazy = df.clone().lazy();

    let result = lazy
        .select([col(column).value_counts(true, true, "count".into(), false)])
        .collect()
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Failed to get value counts: {}", e),
        })?;

    // The result is a struct column, we need to unnest it
    let unnested = result
        .unnest([column], None)
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Failed to unnest value counts: {}", e),
        })?;

    // Extract counts as f64 vector
    let count_col = unnested.column("count").map_err(|e| {
        EconError::InvalidSpecification {
            message: format!("Failed to get count column: {}", e),
        }
    })?;

    let count_f64 = count_col.cast(&DataType::Float64).map_err(|e| {
        EconError::InvalidSpecification {
            message: format!("Failed to cast count column to f64: {}", e),
        }
    })?;

    let observed: Vec<f64> = count_f64
        .f64()
        .map_err(|_| EconError::InvalidSpecification {
            message: "Failed to convert count column to f64".to_string(),
        })?
        .into_no_null_iter()
        .collect();

    chisq_test_gof(&observed, probs, false)
}

/// Run chi-squared test of independence on two dataset columns.
///
/// Creates a contingency table from two categorical columns and tests independence.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `row_col` - Column name for rows of contingency table
/// * `col_col` - Column name for columns of contingency table
/// * `correct` - If true and table is 2×2, apply Yates' correction
///
/// # Example
///
/// ```ignore
/// let result = run_chisq_independence(&dataset, "gender", "party", true)?;
/// println!("{}", result);
/// ```
pub fn run_chisq_independence(
    dataset: &Dataset,
    row_col: &str,
    col_col: &str,
    correct: bool,
) -> EconResult<ChiSquaredResult> {
    use polars::prelude::*;

    let df = dataset.df();

    let available_cols: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();

    // Get unique values for both columns
    let row_series = df.column(row_col).map_err(|_| EconError::ColumnNotFound {
        column: row_col.to_string(),
        available: available_cols.clone(),
    })?;

    let col_series = df.column(col_col).map_err(|_| EconError::ColumnNotFound {
        column: col_col.to_string(),
        available: available_cols,
    })?;

    // Get unique values
    let row_unique = row_series.unique().map_err(|e| EconError::InvalidSpecification {
        message: format!("Failed to get unique row values: {}", e),
    })?;
    let col_unique = col_series.unique().map_err(|e| EconError::InvalidSpecification {
        message: format!("Failed to get unique column values: {}", e),
    })?;

    let n_rows = row_unique.len();
    let n_cols = col_unique.len();

    if n_rows < 2 || n_cols < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_rows.min(n_cols),
            context: "Need at least 2 unique values in each column".to_string(),
        });
    }

    // Build contingency table using group_by and count
    // Group by both columns
    let grouped = df
        .clone()
        .lazy()
        .group_by([col(row_col), col(col_col)])
        .agg([len().alias("count")])
        .collect()
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Failed to group data: {}", e),
        })?;

    // Convert to contingency table matrix
    // Get sorted unique values for consistent ordering
    let mut row_values: Vec<String> = Vec::new();
    let mut col_values: Vec<String> = Vec::new();

    // Extract unique values as strings for indexing
    for i in 0..row_unique.len() {
        row_values.push(format!("{:?}", row_unique.get(i).unwrap()));
    }
    for i in 0..col_unique.len() {
        col_values.push(format!("{:?}", col_unique.get(i).unwrap()));
    }

    row_values.sort();
    col_values.sort();

    // Initialize table with zeros
    let mut table: Vec<Vec<f64>> = vec![vec![0.0; n_cols]; n_rows];

    // Fill in counts from grouped result
    let row_data = grouped.column(row_col).unwrap();
    let col_data = grouped.column(col_col).unwrap();
    let count_data = grouped.column("count").unwrap();

    for i in 0..grouped.height() {
        let row_val = format!("{:?}", row_data.get(i).unwrap());
        let col_val = format!("{:?}", col_data.get(i).unwrap());
        let count = count_data
            .get(i)
            .unwrap()
            .try_extract::<u64>()
            .unwrap_or(0) as f64;

        if let (Some(row_idx), Some(col_idx)) = (
            row_values.iter().position(|v| v == &row_val),
            col_values.iter().position(|v| v == &col_val),
        ) {
            table[row_idx][col_idx] = count;
        }
    }

    chisq_test_independence(&table, correct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gof_uniform() {
        // Fair die: 100 rolls
        let observed = vec![16.0, 18.0, 22.0, 14.0, 15.0, 15.0];
        let result = chisq_test_gof(&observed, None, false).unwrap();

        assert_eq!(result.df, 5);
        assert!(result.statistic >= 0.0);
        // With these counts, p-value should be high (die seems fair)
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_gof_with_probs() {
        // Observed vs specific expected
        let observed = vec![10.0, 20.0, 30.0, 40.0];
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let result = chisq_test_gof(&observed, Some(&probs), false).unwrap();

        // Expected = 100 * [0.1, 0.2, 0.3, 0.4] = [10, 20, 30, 40]
        // Chi-squared should be 0 (perfect match)
        assert!(result.statistic < 0.001);
        assert_eq!(result.df, 3);
    }

    #[test]
    fn test_independence_2x2() {
        // Classic 2x2 example
        let table = vec![vec![20.0, 30.0], vec![30.0, 20.0]];

        // Without Yates' correction
        let result = chisq_test_independence(&table, false).unwrap();
        assert_eq!(result.df, 1);
        assert_eq!(result.n_rows, Some(2));
        assert_eq!(result.n_cols, Some(2));
        assert!(!result.yates_correction);

        // With Yates' correction
        let result_yates = chisq_test_independence(&table, true).unwrap();
        assert!(result_yates.yates_correction);
        // Yates' correction reduces the statistic
        assert!(result_yates.statistic <= result.statistic);
    }

    #[test]
    fn test_independence_2x3() {
        // Gender x Party preference (R example)
        let table = vec![
            vec![762.0, 327.0, 468.0], // Female
            vec![484.0, 239.0, 477.0], // Male
        ];

        let result = chisq_test_independence(&table, true).unwrap();
        assert_eq!(result.df, 2); // (2-1)(3-1)
        assert_eq!(result.n_rows, Some(2));
        assert_eq!(result.n_cols, Some(3));
        // No Yates' correction for non-2x2 tables
        assert!(!result.yates_correction);
    }

    #[test]
    fn test_validate_gof_against_r() {
        // R code:
        // x <- c(89, 37, 30, 28, 2)
        // chisq.test(x)
        //
        // X-squared = 109.11, df = 4, p-value < 2.2e-16
        let observed = vec![89.0, 37.0, 30.0, 28.0, 2.0];
        let result = chisq_test_gof(&observed, None, false).unwrap();

        // Check against R output
        assert!(
            (result.statistic - 109.11).abs() < 0.01,
            "Expected 109.11, got {}",
            result.statistic
        );
        assert_eq!(result.df, 4);
        assert!(result.p_value < 0.001);
    }

    #[test]
    fn test_validate_independence_against_r() {
        // R code:
        // M <- as.table(rbind(c(762, 327, 468), c(484, 239, 477)))
        // chisq.test(M)
        //
        // X-squared = 30.07, df = 2, p-value = 2.954e-07
        let table = vec![
            vec![762.0, 327.0, 468.0],
            vec![484.0, 239.0, 477.0],
        ];

        let result = chisq_test_independence(&table, false).unwrap();

        // Check against R output
        assert!((result.statistic - 30.07).abs() < 0.1);
        assert_eq!(result.df, 2);
        assert!(result.p_value < 0.001);
    }

    #[test]
    fn test_validate_2x2_yates_against_r() {
        // R code:
        // M <- matrix(c(12, 7, 5, 16), nrow = 2, byrow = FALSE)  # R uses column-major order
        // print(M)
        // #      [,1] [,2]
        // # [1,]   12    5
        // # [2,]    7   16
        // chisq.test(M, correct = TRUE)
        //
        // X-squared = 4.8123, df = 1, p-value = 0.02826
        let table = vec![vec![12.0, 5.0], vec![7.0, 16.0]];

        let result = chisq_test_independence(&table, true).unwrap();

        // Check against R output (with Yates' correction)
        assert!(
            (result.statistic - 4.8123).abs() < 0.01,
            "Expected 4.8123, got {}",
            result.statistic
        );
        assert_eq!(result.df, 1);
        assert!(
            (result.p_value - 0.02826).abs() < 0.001,
            "Expected p=0.02826, got {}",
            result.p_value
        );
        assert!(result.yates_correction);
    }

    #[test]
    fn test_pearson_residuals() {
        let table = vec![vec![10.0, 20.0], vec![20.0, 10.0]];
        let result = chisq_test_independence(&table, false).unwrap();

        // Residuals should be non-zero for this unbalanced table
        assert!(result.residuals.iter().any(|&r| r.abs() > 1.0));

        // Standardized residuals should also be computed
        assert!(result.std_residuals.is_some());
    }

    #[test]
    fn test_error_handling() {
        // Too few categories
        let result = chisq_test_gof(&[10.0], None, false);
        assert!(result.is_err());

        // Negative counts
        let result = chisq_test_gof(&[-1.0, 10.0, 10.0], None, false);
        assert!(result.is_err());

        // Probabilities don't sum to 1
        let result = chisq_test_gof(&[10.0, 20.0], Some(&[0.3, 0.3]), false);
        assert!(result.is_err());

        // Empty table
        let result = chisq_test_independence(&[], false);
        assert!(result.is_err());

        // Single row
        let result = chisq_test_independence(&[vec![10.0, 20.0]], false);
        assert!(result.is_err());
    }
}
