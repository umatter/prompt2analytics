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
use crate::resource;
use crate::traits::{SignificanceLevel, chi_squared_p_value};

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

    // A `Vec<Vec>` passes a per-allocation ceiling row by row, so the total is checked here, by
    // the fallible allocation of the flat table
    let cells = resource::checked_len(&[n_rows, n_cols], "contingency table")?;
    let mut observed: Vec<f64> = resource::try_with_capacity(cells, "contingency table")?;
    for row in table {
        observed.extend_from_slice(row);
    }

    independence_from_flat(observed, n_rows, n_cols, correct)
}

/// The chi-squared test of independence on a row-major `n_rows x n_cols` table of counts
/// (`n_rows, n_cols >= 2`). The expected values and both residual tables are allocated
/// fallibly.
fn independence_from_flat(
    observed: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    correct: bool,
) -> EconResult<ChiSquaredResult> {
    let cells = resource::checked_len(&[n_rows, n_cols], "contingency table")?;

    // Validate all counts are non-negative
    if observed.iter().any(|&x| x < 0.0) {
        return Err(EconError::InvalidSpecification {
            message: "All counts must be non-negative".to_string(),
        });
    }

    // Calculate marginals (each total sums its cells in row-major order)
    let mut row_totals: Vec<f64> = resource::try_with_capacity(n_rows, "row totals")?;
    row_totals.extend(
        observed
            .chunks_exact(n_cols)
            .map(|row| row.iter().sum::<f64>()),
    );
    let mut col_totals: Vec<f64> = resource::try_with_capacity(n_cols, "column totals")?;
    col_totals.extend((0..n_cols).map(|j| {
        observed
            .iter()
            .skip(j)
            .step_by(n_cols)
            .copied()
            .sum::<f64>()
    }));
    let grand_total: f64 = row_totals.iter().sum();

    if grand_total <= 0.0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Total count must be positive".to_string(),
        });
    }

    // Calculate expected values: E_ij = row_i × col_j / n
    let mut expected: Vec<f64> = resource::try_with_capacity(cells, "expected counts")?;
    for i in 0..n_rows {
        for j in 0..n_cols {
            expected.push(row_totals[i] * col_totals[j] / grand_total);
        }
    }

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
    let mut residuals: Vec<f64> = resource::try_with_capacity(cells, "Pearson residuals")?;
    residuals.extend(
        observed
            .iter()
            .zip(&expected)
            .map(|(&o, &e)| if e > 0.0 { (o - e) / e.sqrt() } else { 0.0 }),
    );

    // Standardized residuals: (O - E) / sqrt(E × (1 - row_prop) × (1 - col_prop))
    let mut std_residuals: Vec<f64> = resource::try_with_capacity(cells, "standardized residuals")?;
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
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        });
    }

    // Use lazy frame for value_counts (via Expr, not Series method)
    let lazy = df.clone().lazy();

    let result = lazy
        .select([col(column).value_counts(true, true, "count", false)])
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
    let count_col = unnested
        .column("count")
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Failed to get count column: {}", e),
        })?;

    let count_f64 =
        count_col
            .cast(&DataType::Float64)
            .map_err(|e| EconError::InvalidSpecification {
                message: format!("Failed to cast count column to f64: {}", e),
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
/// # Errors
///
/// - [`EconError::ColumnNotFound`] for a missing column.
/// - [`EconError::InvalidSpecification`] for a struct, list or array column, when a column's
///   distinct values cannot be computed (`Failed to get unique ... values`), or when a value
///   cannot be read (`Failed to read a value`).
/// - [`EconError::InsufficientData`] for a column with fewer than two distinct values (null
///   counts as one).
/// - [`EconError::ResourceLimit`] for a table too large to allocate.
/// - [`EconError::Computation`] when a row matches no level of its column, because the table
///   would be missing those rows.
/// - [`EconError::Cancelled`] when the budget installed for the thread is cancelled or its
///   deadline passes (checked through [`resource::checkpoint`]).
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

    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Get unique values for both columns
    let row_series = df.column(row_col).map_err(|_| EconError::ColumnNotFound {
        column: row_col.to_string(),
        available: available_cols.clone(),
    })?;

    let col_series = df.column(col_col).map_err(|_| EconError::ColumnNotFound {
        column: col_col.to_string(),
        available: available_cols,
    })?;

    // Rows are keyed on the text of their value. A struct value's text prints its whole array
    // (O(n) per row), and struct levels never matched their rows either. A list or array value's
    // text keeps only its first and last elements once it is longer than POLARS_FMT_MAX_ROWS,
    // so two distinct values could share a key and their rows be counted under one level.
    for (name, column) in [(row_col, row_series), (col_col, col_series)] {
        let kind = match column.dtype() {
            DataType::Struct(_) => "structs",
            DataType::List(_) => "lists",
            DataType::Array(_, _) => "arrays",
            _ => continue,
        };
        return Err(EconError::InvalidSpecification {
            message: format!(
                "column `{name}` holds {kind}; the chi-squared test of independence needs \
                 columns of scalar values (strings, numbers, categories or dates)"
            ),
        });
    }

    // Get unique values
    let row_unique = row_series
        .unique()
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Failed to get unique row values: {}", e),
        })?;
    let col_unique = col_series
        .unique()
        .map_err(|e| EconError::InvalidSpecification {
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

    // A flat, fallibly allocated r x c table of counts, refused before the rechunks and the level
    // lists: two ID-like columns would otherwise rechunk both columns and build 4n Strings first
    let cells = resource::checked_len(&[n_rows, n_cols], "contingency table")?;
    let mut observed: Vec<f64> = resource::try_vec(cells, 0.0, "contingency table")?;

    // Random access is O(1) only on a single chunk, and a frame built by stacking (munge_concat)
    // keeps one chunk per part, so rechunk each column once before the per-row reads
    let row_series = row_series.rechunk();
    let col_series = col_series.rechunk();
    let lookup_error = |e: PolarsError| EconError::InvalidSpecification {
        message: format!("Failed to read a value: {}", e),
    };

    // Levels in sorted order of their text (the order the table has always had), each with the
    // key its rows are matched on
    let row_levels = sorted_levels(&row_unique.rechunk(), lookup_error)?;
    let col_levels = sorted_levels(&col_unique.rechunk(), lookup_error)?;

    // Level key -> position in the sorted level order (the first position for equal keys)
    fn index_of(levels: &[(String, String)]) -> std::collections::HashMap<&str, usize> {
        let mut index = std::collections::HashMap::with_capacity(levels.len());
        for (i, (_, key)) in levels.iter().enumerate() {
            index.entry(key.as_str()).or_insert(i);
        }
        index
    }
    let row_index = index_of(&row_levels);
    let col_index = index_of(&col_levels);

    // Count the rows in one pass (no lazy engine, so no nested async runtime), reusing one key
    // buffer per column
    let mut row_key = String::new();
    let mut col_key = String::new();
    let mut unmatched = 0usize;
    for i in 0..df.height() {
        if i % 65_536 == 0 {
            resource::checkpoint()?;
        }
        write_level_key(&mut row_key, &row_series.get(i).map_err(lookup_error)?);
        write_level_key(&mut col_key, &col_series.get(i).map_err(lookup_error)?);
        match (
            row_index.get(row_key.as_str()),
            col_index.get(col_key.as_str()),
        ) {
            (Some(&row_idx), Some(&col_idx)) => observed[row_idx * n_cols + col_idx] += 1.0,
            _ => unmatched += 1,
        }
    }
    // Every value comes from its own column's unique(), so a miss means the keys disagree; a
    // table that silently lost rows would give a wrong statistic
    if unmatched > 0 {
        return Err(EconError::Computation(format!(
            "{unmatched} rows matched no level of `{row_col}` or `{col_col}`, so the \
             contingency table would be incomplete"
        )));
    }

    independence_from_flat(observed, n_rows, n_cols, correct)
}

/// The levels of a column from its single-chunk `unique()` result, as (Debug text, row key)
/// pairs sorted by the text.
fn sorted_levels(
    unique: &polars::prelude::Column,
    lookup_error: impl Fn(polars::prelude::PolarsError) -> EconError,
) -> EconResult<Vec<(String, String)>> {
    let mut levels: Vec<(String, String)> = resource::try_with_capacity(unique.len(), "levels")?;
    for i in 0..unique.len() {
        let v = unique.get(i).map_err(&lookup_error)?;
        let mut key = String::new();
        write_level_key(&mut key, &v);
        levels.push((format!("{v:?}"), key));
    }
    levels.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(levels)
}

/// Writes the key a value is counted under into `buf`: its Debug text, with two exceptions.
///
/// - A float zero is written as `0.0` whatever its sign, because `unique()` merges `0.0` and
///   `-0.0` into one level.
/// - A categorical or enum value is written as its category id alone. Its Debug text also prints
///   the shared mapping's upper bound, which grows whenever a category is inserted into that
///   mapping (for the process-wide one, by any other request), so a key built from it could
///   change between building the levels and counting the rows. The sort text in
///   [`sorted_levels`] keeps the full Debug text: two distinct ids differ before the mapping is
///   printed, so the level order does not depend on it.
fn write_level_key(buf: &mut String, v: &polars::prelude::AnyValue<'_>) {
    use polars::prelude::AnyValue;
    use std::fmt::Write as _;

    buf.clear();
    // Writing into a String cannot fail
    let _ = match *v {
        AnyValue::Float64(x) => write!(buf, "{:?}", AnyValue::Float64(x + 0.0)),
        AnyValue::Float32(x) => write!(buf, "{:?}", AnyValue::Float32(x + 0.0)),
        AnyValue::Categorical(idx, _) | AnyValue::CategoricalOwned(idx, _) => {
            write!(buf, "Categorical({idx})")
        }
        AnyValue::Enum(idx, _) | AnyValue::EnumOwned(idx, _) => write!(buf, "Enum({idx})"),
        ref other => write!(buf, "{other:?}"),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use polars::prelude::*;

    fn gh_dataset() -> Dataset {
        // table(g, h) = [[7, 5, 8], [4, 9, 7]] (rows a, b; columns x, y, z), 40 rows
        let mut h: Vec<&str> = Vec::new();
        for (lvl, k) in [("x", 7), ("y", 5), ("z", 8), ("x", 4), ("y", 9), ("z", 7)] {
            h.extend(std::iter::repeat_n(lvl, k));
        }
        let g: Vec<&str> = (0..40).map(|i| if i < 20 { "a" } else { "b" }).collect();
        Dataset::new(df!("g" => g, "h" => h).unwrap())
    }

    #[test]
    fn test_validate_run_chisq_independence_from_data_against_r() {
        // R: chisq.test(table(g, h)) -> X-squared = 2.0277056277056276, df = 2, p = 0.36281841143824806
        let r = run_chisq_independence(&gh_dataset(), "g", "h", true).unwrap();
        assert!((r.statistic - 2.0277056277056276).abs() < 1e-10);
        assert!((r.p_value - 0.36281841143824806).abs() < 1e-10);
        assert_eq!((r.df, r.n_rows, r.n_cols), (2, Some(2), Some(3)));
        assert_eq!(r.observed, vec![7.0, 5.0, 8.0, 4.0, 9.0, 7.0]);
        assert_eq!(r.expected, vec![5.5, 7.0, 7.5, 5.5, 7.0, 7.5]);
    }

    #[test]
    fn test_run_chisq_independence_inside_async_runtime() {
        // STAT-10: the lazy group_by panicked with "Cannot start a runtime from within a runtime"
        // for >= 20 rows when called from inside a tokio runtime. Counting without the lazy engine fixes it.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let r = rt
            .block_on(async { run_chisq_independence(&gh_dataset(), "g", "h", true) })
            .unwrap();
        assert!((r.statistic - 2.0277056277056276).abs() < 1e-10);
    }

    #[test]
    fn test_chisq_independence_respects_allocation_ceiling() {
        let table = vec![vec![1.0; 12]; 12]; // 144 cells x 8 bytes = 1152 > 1024
        let r = resource::with_alloc_ceiling(Some(1024), || chisq_test_independence(&table, false));
        assert!(matches!(r, Err(EconError::ResourceLimit(_))));
    }

    #[test]
    fn test_validate_run_chisq_independence_on_multi_chunk_frame_against_r() {
        // A frame built by stacking (as munge_concat does) keeps one chunk per part. Pre-fix,
        // each per-row lookup scanned the chunk list; the counts must not depend on the layout.
        let whole = gh_dataset().df().clone();
        let mut df = whole.slice(0, 7);
        for (offset, len) in [(7, 9), (16, 1), (17, 13), (30, 10)] {
            df.vstack_mut(&whole.slice(offset, len)).unwrap();
        }
        assert!(df.column("g").unwrap().n_chunks() > 1);
        // R: chisq.test(table(g, h)) -> X-squared = 2.0277056277056276, df = 2, p = 0.36281841143824806
        let r = run_chisq_independence(&Dataset::new(df), "g", "h", true).unwrap();
        assert!((r.statistic - 2.0277056277056276).abs() < 1e-10);
        assert!((r.p_value - 0.36281841143824806).abs() < 1e-10);
        assert_eq!(r.observed, vec![7.0, 5.0, 8.0, 4.0, 9.0, 7.0]);
    }

    #[test]
    fn test_validate_run_chisq_independence_counts_both_signed_zeros() {
        // polars' unique() merges 0.0 and -0.0 into one level, as R's table() does. Pre-fix, the
        // rows whose zero had the other sign matched no level and were dropped from the table.
        let mut df = gh_dataset().df().clone();
        let x: Vec<f64> = (0..40)
            .map(|i| match (i < 20, i % 2) {
                (true, 0) => 0.0,
                (true, _) => -0.0,
                (false, _) => 1.0,
            })
            .collect();
        df.with_column(Column::new("x".into(), x)).unwrap();
        // R: x <- c(rep(c(0, -0), 10), rep(1, 20)); chisq.test(table(x, h)) -> same as table(g, h)
        let r = run_chisq_independence(&Dataset::new(df), "x", "h", true).unwrap();
        assert_eq!(r.observed.iter().sum::<f64>(), 40.0);
        assert_eq!(r.observed, vec![7.0, 5.0, 8.0, 4.0, 9.0, 7.0]);
        assert!((r.statistic - 2.0277056277056276).abs() < 1e-10);
    }

    #[test]
    fn test_run_chisq_independence_refuses_struct_column() {
        // The text of a struct value prints its whole array, so keying rows on it costs O(n) per row
        let mut df = gh_dataset().df().clone();
        let s = StructChunked::from_columns(
            "s".into(),
            40,
            &[
                df.column("g").unwrap().clone(),
                df.column("h").unwrap().clone(),
            ],
        )
        .unwrap()
        .into_series();
        df.with_column(s).unwrap();
        let e = run_chisq_independence(&Dataset::new(df), "s", "h", true).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("struct")),
            "{e}"
        );
    }

    #[test]
    fn test_run_chisq_independence_refuses_list_and_array_columns() {
        // The text of a list value prints only its first and last five elements once it has more
        // than POLARS_FMT_MAX_ROWS (default 10). Pre-fix, these two distinct lists of 12, which
        // differ only in the middle, got one key: all 40 rows were counted under the first level
        // and the test returned p = 1 with no error
        let a: Vec<i64> = (0..12).collect();
        let mut b = a.clone();
        b[5] = 105;
        b[6] = 106;
        let lists: Vec<Series> = (0..40)
            .map(|i| Series::new("".into(), if i % 3 == 0 { &a } else { &b }))
            .collect();
        let list = Series::new("l".into(), lists);
        assert!(matches!(list.dtype(), DataType::List(_)));
        let array = list
            .cast(&DataType::Array(Box::new(DataType::Int64), 12))
            .unwrap()
            .with_name("arr".into());
        let mut df = gh_dataset().df().clone();
        df.with_column(list).unwrap();
        df.with_column(array).unwrap();
        let dataset = Dataset::new(df);
        for name in ["l", "arr"] {
            let e = run_chisq_independence(&dataset, name, "h", true).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message }
                    if message.contains(&format!("`{name}`")) && message.contains("scalar")),
                "{name}: {e}"
            );
        }
    }

    #[test]
    fn test_categorical_row_keys_survive_inserts_into_the_mapping() {
        // The Debug text of a categorical value prints its mapping's upper bound, which grows
        // whenever a category is inserted into that mapping; for the process-wide mapping, any
        // other request can do that. Pre-fix, rows were keyed on that text, so an insert between
        // building the levels and counting the rows made the rows miss their levels.
        let cats = Categories::random("".into(), CategoricalPhysical::U32);
        let g = gh_dataset()
            .df()
            .column("g")
            .unwrap()
            .cast(&DataType::from_categories(cats))
            .unwrap();
        let levels = sorted_levels(&g.unique().unwrap().rechunk(), |e| {
            EconError::Computation(e.to_string())
        })
        .unwrap();
        g.dtype()
            .cat_mapping()
            .unwrap()
            .insert_cat("inserted by another request")
            .unwrap();
        let mut key = String::new();
        for i in [0, 39] {
            write_level_key(&mut key, &g.get(i).unwrap());
            assert!(
                levels.iter().any(|(_, k)| *k == key),
                "row {i} key {key} matches no level of {levels:?}"
            );
        }
    }

    #[test]
    fn test_validate_run_chisq_independence_on_categorical_and_enum_columns_against_r() {
        // Local categories (not the process-wide mapping), so the category ids, and with them
        // the level order, follow first appearance: a, b and x, y, z
        let categorical =
            || DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
        let enums = |levels: &[&str]| {
            DataType::from_frozen_categories(FrozenCategories::new(levels.iter().copied()).unwrap())
        };
        let base = gh_dataset().df().clone();
        for (g_type, h_type) in [
            (categorical(), categorical()),
            (enums(&["a", "b"]), enums(&["x", "y", "z"])),
        ] {
            let mut df = base.clone();
            df.with_column(base.column("g").unwrap().cast(&g_type).unwrap())
                .unwrap();
            df.with_column(base.column("h").unwrap().cast(&h_type).unwrap())
                .unwrap();
            // R: chisq.test(table(factor(g), factor(h))) -> X-squared = 2.0277056277056276, df = 2,
            // p = 0.36281841143824806
            let r = run_chisq_independence(&Dataset::new(df), "g", "h", true).unwrap();
            assert!(
                (r.statistic - 2.0277056277056276).abs() < 1e-10,
                "{g_type:?}/{h_type:?}"
            );
            assert!(
                (r.p_value - 0.36281841143824806).abs() < 1e-10,
                "{g_type:?}/{h_type:?}"
            );
            assert_eq!(
                (r.df, r.n_rows, r.n_cols),
                (2, Some(2), Some(3)),
                "{g_type:?}/{h_type:?}"
            );
            assert_eq!(
                r.observed,
                vec![7.0, 5.0, 8.0, 4.0, 9.0, 7.0],
                "{g_type:?}/{h_type:?}"
            );
            assert_eq!(
                r.expected,
                vec![5.5, 7.0, 7.5, 5.5, 7.0, 7.5],
                "{g_type:?}/{h_type:?}"
            );
        }
    }

    #[test]
    fn test_run_chisq_independence_refuses_the_table_before_building_levels() {
        // Two ID-like columns give an r x c table the ceiling refuses. Pre-fix, both columns were
        // rechunked and their levels built (two Strings per level) first, and the refusal came
        // from the level list (40 levels x 48 bytes > 1024) instead of the table.
        let a: Vec<i64> = (0..40).collect();
        let b: Vec<i64> = (0..40).map(|i| (i * 7) % 40).collect();
        let dataset = Dataset::new(df!("a" => a, "b" => b).unwrap());
        let r = resource::with_alloc_ceiling(Some(1024), || {
            run_chisq_independence(&dataset, "a", "b", false)
        });
        assert!(
            matches!(&r, Err(EconError::ResourceLimit(m)) if m.contains("contingency table")),
            "{r:?}"
        );
    }

    #[test]
    fn test_run_chisq_independence_honours_cancellation() {
        let t = CancelToken::new();
        t.cancel();
        let r = resource::with_budget(Budget::new(None, t), || {
            run_chisq_independence(&gh_dataset(), "g", "h", true)
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

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
        let table = vec![vec![762.0, 327.0, 468.0], vec![484.0, 239.0, 477.0]];

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
