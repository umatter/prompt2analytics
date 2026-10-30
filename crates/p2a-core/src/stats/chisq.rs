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

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::data::Dataset;
use crate::data::columns::{self, MissingSummary, is_numeric_dtype};
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::contingency::{CategoryIndex, contingency_table, level_counts, level_list};
use crate::traits::SignificanceLevel;

/// R's warning when an expected count is below 5.
const SMALL_EXPECTED_WARNING: &str = "Chi-squared approximation may be incorrect";

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
    /// Observed counts (flattened row-major, in `row_labels` x `col_labels` order, for
    /// contingency tables)
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

    // ═══════════════════════════════════════════════════════════════════════
    // Labels, warnings and missing values
    // ═══════════════════════════════════════════════════════════════════════
    /// Goodness-of-fit categories, or the table's row levels (None for array input).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_labels: Option<Vec<String>>,
    /// Column levels of an independence table (None for goodness of fit and array input).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub col_labels: Option<Vec<String>>,
    /// "Chi-squared approximation may be incorrect (k of K expected counts are below 5)"
    /// when any expected count is below 5 (R's warning condition).
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Rows dropped because of a missing value (dataset input).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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
            let label = |i: usize, fallback: String| -> String {
                self.row_labels
                    .as_ref()
                    .and_then(|labels| labels.get(i))
                    .cloned()
                    .unwrap_or(fallback)
            };

            if let (Some(n_rows), Some(n_cols)) = (self.n_rows, self.n_cols) {
                // Contingency table format
                let rows: Vec<String> = (0..n_rows)
                    .map(|i| label(i, format!("Row {}", i + 1)))
                    .collect();
                let width = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0);
                if let Some(cols) = &self.col_labels {
                    write!(f, "  {:width$} ", "")?;
                    for c in cols {
                        write!(f, " {c:>17}")?;
                    }
                    writeln!(f)?;
                }
                for (i, row) in rows.iter().enumerate() {
                    write!(f, "  {row:width$}:")?;
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
                    let name = label(i, format!("[{}]", i + 1));
                    writeln!(f, "  {name}: {o:.1} (expected: {e:.1})")?;
                }
            }
        }

        // R's warning for small expected counts, and the dropped rows
        for warning in &self.warnings {
            writeln!(f)?;
            writeln!(f, "Warning: {warning}")?;
        }
        if let Some(note) = self.missing.note() {
            writeln!(f)?;
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// The chi-squared upper tail `P(X > stat)` on `df` degrees of freedom (statrs `sf`, so
/// small p-values keep their precision): NaN for NaN, 0 for +inf.
fn chisq_upper_tail(stat: f64, df: usize) -> f64 {
    if stat.is_nan() {
        return f64::NAN;
    }
    if stat == f64::INFINITY {
        return 0.0;
    }
    ChiSquared::new(df as f64)
        .map(|d| d.sf(stat))
        .unwrap_or(f64::NAN)
}

/// R's "Chi-squared approximation may be incorrect" when any expected count is below 5,
/// with how many are.
fn small_expected_warning(expected: &[f64]) -> Option<String> {
    let small = expected.iter().filter(|&&e| e < 5.0).count();
    (small > 0).then(|| {
        format!(
            "{SMALL_EXPECTED_WARNING} ({small} of {} expected counts are below 5)",
            expected.len()
        )
    })
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

    // Validate observed counts (R: "all entries of 'x' must be nonnegative and finite")
    if observed.iter().any(|&x| !(x >= 0.0 && x.is_finite())) {
        return Err(EconError::InvalidSpecification {
            message: "Observed counts must be non-negative and finite".to_string(),
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
        if p.iter().any(|&x| !(x >= 0.0 && x.is_finite())) {
            return Err(EconError::InvalidSpecification {
                message: "Probabilities must be non-negative and finite".to_string(),
            });
        }
        let sum: f64 = p.iter().sum();
        if rescale_p {
            if !(sum > 0.0) {
                return Err(EconError::InvalidSpecification {
                    message: "Probabilities to rescale must have a positive sum".to_string(),
                });
            }
            p.iter().map(|&x| x / sum).collect()
        } else {
            // R's rule: |sum(p) - 1| <= sqrt(.Machine$double.eps)
            if (sum - 1.0).abs() > f64::EPSILON.sqrt() {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Probabilities must sum to 1 (they sum to {sum}); use rescale_p = true \
                         to rescale them"
                    ),
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
    let p_value = chisq_upper_tail(statistic, df);
    let significance = SignificanceLevel::from_p_value(p_value);

    // Pearson residuals: (O - E) / sqrt(E)
    let residuals: Vec<f64> = observed
        .iter()
        .zip(&expected)
        .map(|(&o, &e)| if e > 0.0 { (o - e) / e.sqrt() } else { 0.0 })
        .collect();

    let warnings: Vec<String> = small_expected_warning(&expected).into_iter().collect();

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
        row_labels: None,
        col_labels: None,
        warnings,
        missing: MissingSummary::none(n as usize),
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

    // Validate all counts are non-negative and finite (R: "all entries of 'x' must be
    // nonnegative and finite")
    if observed.iter().any(|&x| !(x >= 0.0 && x.is_finite())) {
        return Err(EconError::InvalidSpecification {
            message: "All counts must be non-negative and finite".to_string(),
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
    let p_value = chisq_upper_tail(statistic, df);
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

    let warnings: Vec<String> = small_expected_warning(&expected).into_iter().collect();

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
        row_labels: None,
        col_labels: None,
        warnings,
        missing: MissingSummary::none(grand_total as usize),
    })
}

/// Expected frequencies of a goodness-of-fit test ([`run_chisq_gof_with`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum GofExpected {
    /// Every observed category equally likely. Default.
    #[default]
    Uniform,
    /// probabilities in sorted level order (the order `ChiSquaredResult::row_labels` reports)
    Probs(Vec<f64>),
    /// category -> probability; categories never observed are counted as 0
    Named(Vec<(String, f64)>),
}

/// Chi-squared goodness-of-fit test on a dataset column against `expected`; `rescale`
/// divides the probabilities by their sum (R `chisq.test(rescale.p = TRUE)`).
///
/// The categories are the column's distinct non-missing values in R `factor()` order
/// (numbers numerically, text in C-locale order), as [`level_counts`] counts them; rows with
/// a missing value are dropped and reported in `missing`. `row_labels` lists the categories.
///
/// - [`GofExpected::Uniform`]: every observed category equally likely.
/// - [`GofExpected::Probs`]: one probability per category in that order.
/// - [`GofExpected::Named`]: a probability per category name; categories that never occur
///   count 0 and are reported after the observed ones, in the order given. Every observed
///   category must be named; on a numeric column a name matches by value ("2.0" is "2").
///
/// # Errors
///
/// [`level_counts`]' errors (`ColumnNotFound`, a column that cannot be categorical, no row
/// left), `InvalidSpecification` for a probs list of the wrong length (listing the
/// categories), an empty map, a category named twice, an observed category without a
/// probability, and probabilities that are negative, not finite or (without `rescale`) do
/// not sum to 1 within `sqrt(f64::EPSILON)`; [`chisq_test_gof`]'s errors.
pub fn run_chisq_gof_with(
    dataset: &Dataset,
    column: &str,
    expected: &GofExpected,
    rescale: bool,
) -> EconResult<ChiSquaredResult> {
    let df = dataset.df();
    let (labels, counts, missing, probs) = match expected {
        GofExpected::Uniform => {
            let (labels, counts, missing) = level_counts(df, column, &[])?;
            (labels, counts, missing, None)
        }
        GofExpected::Probs(p) => {
            let (labels, counts, missing) = level_counts(df, column, &[])?;
            if p.len() != labels.len() {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "probs has {} value(s) but column '{column}' has {} categories after \
                         removing missing values, in this order: [{}]. Give one probability \
                         per category in that order, or name them (expected)",
                        p.len(),
                        labels.len(),
                        level_list(&labels)
                    ),
                });
            }
            (labels, counts, missing, Some(p.clone()))
        }
        GofExpected::Named(pairs) => {
            let (labels, counts, missing, probs) = named_probabilities(df, column, pairs)?;
            (labels, counts, missing, Some(probs))
        }
    };
    let mut result = chisq_test_gof(&counts, probs.as_deref(), rescale)?;
    result.row_labels = Some(labels);
    result.missing = missing;
    Ok(result)
}

/// The categories, counts and missing-value summary of `column` with the probabilities of
/// `pairs` aligned to the categories. Every key is matched by a hash of its canonical form
/// (see `contingency::same_category`), so `e` pairs on `k` categories cost O(k + e).
fn named_probabilities(
    df: &polars::prelude::DataFrame,
    column: &str,
    pairs: &[(String, f64)],
) -> EconResult<(Vec<String>, Vec<f64>, MissingSummary, Vec<f64>)> {
    if pairs.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "expected names no category of column '{column}'; give a probability per \
                 category"
            ),
        });
    }
    resource::check_alloc::<(&str, ())>(
        resource::checked_len(&[pairs.len(), 2], "category names")?,
        "category names",
    )?;
    let mut seen: HashSet<&str> = HashSet::new();
    seen.try_reserve(pairs.len())
        .map_err(|_| EconError::ResourceLimit("category names could not be allocated".into()))?;
    let mut keys: Vec<String> = resource::try_with_capacity(pairs.len(), "category names")?;
    for (i, (key, _)) in pairs.iter().enumerate() {
        if i % 65_536 == 0 {
            resource::checkpoint()?;
        }
        if !seen.insert(key.as_str()) {
            return Err(EconError::InvalidSpecification {
                message: format!("category '{key}' is given twice"),
            });
        }
        keys.push(key.clone());
    }
    let (labels, counts, missing) = level_counts(df, column, &keys)?;
    let numeric = is_numeric_dtype(columns::column(df, column)?.dtype(), false);
    let index = CategoryIndex::new(&labels, numeric)?;
    let mut probs: Vec<Option<f64>> = resource::try_vec(labels.len(), None, "probabilities")?;
    for (i, (key, p)) in pairs.iter().enumerate() {
        if i % 65_536 == 0 {
            resource::checkpoint()?;
        }
        // level_counts appended every key that matches no observed level, so each key
        // names a label
        let Some(at) = index.find(key) else {
            return Err(EconError::Computation(format!(
                "category '{key}' matches no category of column '{column}'"
            )));
        };
        if probs[at].is_some() {
            return Err(EconError::InvalidSpecification {
                message: format!("category '{}' is given twice", labels[at]),
            });
        }
        probs[at] = Some(*p);
    }
    let mut aligned: Vec<f64> = resource::try_with_capacity(labels.len(), "probabilities")?;
    for (i, p) in probs.iter().enumerate() {
        match p {
            Some(p) => aligned.push(*p),
            None => {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "category '{}' occurs {} time(s) in column '{column}' but has no \
                         expected probability; list every observed category",
                        labels[i], counts[i]
                    ),
                });
            }
        }
    }
    Ok((labels, counts, missing, aligned))
}

/// Run chi-squared goodness-of-fit test on a dataset column.
///
/// Counts the column's categories (missing values dropped, R `factor()` order; see
/// [`run_chisq_gof_with`]) and tests them against `probs`, one probability per category
/// in that order, or the uniform distribution.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `column` - Column name containing categorical values
/// * `probs` - Optional expected probabilities in category order (uniform if None)
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
    let expected = probs.map_or(GofExpected::Uniform, |p| GofExpected::Probs(p.to_vec()));
    run_chisq_gof_with(dataset, column, &expected, false)
}

/// Run chi-squared test of independence on two dataset columns.
///
/// Builds the contingency table with [`contingency_table`] and tests independence. Rows with
/// a missing value (null, or NaN in a float column) in either column are dropped and
/// reported in `missing`, as R's `table()` drops them; the levels follow R `factor()` order
/// (numbers numerically, text in C-locale order) and are reported in `row_labels` and
/// `col_labels`.
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
/// - [`EconError::InvalidSpecification`] for a column whose type cannot be categorical
///   (datetime, struct, list, array, ...), naming it.
/// - [`EconError::InsufficientData`] when no row is left after removing missing values, or a
///   column has fewer than two levels.
/// - [`EconError::ResourceLimit`] for a contingency table too large to allocate.
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
    let table = contingency_table(dataset.df(), row_col, col_col)?;
    let (n_rows, n_cols) = (table.n_rows(), table.n_cols());
    if n_rows < 2 || n_cols < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_rows.min(n_cols),
            context: format!(
                "Need at least 2 levels in each column after removing missing values: \
                 '{row_col}' has {n_rows}, '{col_col}' has {n_cols}"
            ),
        });
    }
    let mut result = independence_from_flat(table.counts, n_rows, n_cols, correct)?;
    result.row_labels = Some(table.row_levels);
    result.col_labels = Some(table.col_levels);
    result.missing = table.missing;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    const R_WARNING: &str = "Chi-squared approximation may be incorrect";

    #[test]
    fn test_validate_chisq_na_codes() {
        // R: chisq.test(table(g, h)) after table() drops the 3 NA h cells: 2 x 2 with Yates,
        // X2 0.9506250000000002, df 1, p 0.3295602599606207; correct = FALSE 2.7225,
        // p 0.09894293606729652 (case stat_contingency_na_codes). Fails before: the null is a
        // column "Null": 2 x 3, X2 3.1333333333333333, df 2, no Yates.
        let (case, ds) = crate::test_support::load_case("stat_contingency_na_codes");
        let y = run_chisq_independence(&ds, "g", "h", true).unwrap();
        let u = run_chisq_independence(&ds, "g", "h", false).unwrap();
        Check::new(&case)
            .scalar("gh_chisq_yates", y.statistic, Tol::stat(9))
            .scalar("gh_df", y.df as f64, Tol::exact())
            .scalar("gh_p_yates", y.p_value, Tol::p_value(9))
            .scalar("gh_chisq", u.statistic, Tol::stat(9))
            .scalar("gh_p", u.p_value, Tol::p_value(9))
            .vector("gh_observed", &u.observed, Tol::exact())
            .vector("gh_expected", &u.expected, Tol::stat(9))
            .vector("gh_residuals", &u.residuals, Tol::stat(9))
            .vector(
                "gh_stdres",
                u.std_residuals.as_deref().unwrap(),
                Tol::stat(9),
            )
            .scalar("gh_n", u.missing.n_used as f64, Tol::exact())
            .finish();
        assert!(y.yates_correction);
        assert_eq!(
            (y.row_labels.clone().unwrap(), y.col_labels.clone().unwrap()),
            (
                vec!["a".to_string(), "b".into()],
                vec!["u".to_string(), "v".into()]
            )
        );
        assert_eq!(y.missing.by_column, vec![("h".to_string(), 3)]);
        // expected 1.78 .. 2.78 < 5
        assert!(
            y.warnings.iter().any(|w| w.starts_with(R_WARNING)),
            "{:?}",
            y.warnings
        );
        let shown = y.to_string();
        assert!(
            shown.contains("Warning: Chi-squared approximation may be incorrect")
                && shown.contains("Missing values: 3 of 12 rows dropped"),
            "{shown}"
        );
    }

    #[test]
    fn test_validate_chisq_small_expected() {
        // R: chisq.test(table(x, y)), 3 x 2, X2 1.9345238095238098, df 2, p
        // 0.38012242588687084, warns "Chi-squared approximation may be incorrect" (min expected
        // 1.3846153846153846) (case stat_chisq_small_expected). Fails before: no warnings field
        // (TOOL-33: the warning lived only in Display).
        let (case, ds) = crate::test_support::load_case("stat_chisq_small_expected");
        let r = run_chisq_independence(&ds, "x", "y", true).unwrap();
        Check::new(&case)
            .scalar("chisq", r.statistic, Tol::stat(13))
            .scalar("df", r.df as f64, Tol::exact())
            .scalar("p", r.p_value, Tol::p_value(13))
            .scalar(
                "min_expected",
                r.expected.iter().cloned().fold(f64::INFINITY, f64::min),
                Tol::stat(13),
            )
            .vector("observed", &r.observed, Tol::exact())
            .vector("expected", &r.expected, Tol::stat(13))
            .vector("residuals", &r.residuals, Tol::stat(13))
            .vector("stdres", r.std_residuals.as_deref().unwrap(), Tol::stat(13))
            .scalar(
                "warned",
                r.warnings
                    .iter()
                    .filter(|w| w.starts_with(R_WARNING))
                    .count() as f64,
                Tol::exact(),
            )
            .finish();
        assert!(!r.yates_correction);
        assert_eq!(r.row_labels.unwrap(), ["p", "q", "r"]);
        assert!(r.missing.is_empty());
    }

    #[test]
    fn test_validate_chisq_rows_2000() {
        // R: chisq.test(table(treat, outcome)) with Yates on 2000 rows: 3.042,
        // p 0.08113589702211416.
        let (case, ds) = crate::test_support::load_case("stat_fisher_rows_2000");
        let r = run_chisq_independence(&ds, "treat", "outcome", true).unwrap();
        Check::new(&case)
            .scalar("chisq_yates", r.statistic, Tol::stat(2000))
            .scalar("chisq_yates_p", r.p_value, Tol::p_value(2000))
            .finish();
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn test_chisq_refuses_non_finite_counts() {
        assert!(chisq_test_independence(&[vec![1.0, f64::NAN], vec![2.0, 3.0]], false).is_err());
        assert!(chisq_test_gof(&[1.0, f64::INFINITY], None, false).is_err());
    }

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
                    if message.contains(&format!("'{name}'"))
                        && message.contains("cannot be used as a categorical variable")),
                "{name}: {e}"
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
        // from the level list (40 levels x 48 bytes > 1024) instead of the table. The extractor
        // builds the factor codes and levels first (each allocation below the ceiling), so the
        // 40 x 40 table (12,800 bytes) is still the first allocation refused.
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

    fn named(v: &[(&str, f64)]) -> GofExpected {
        GofExpected::Named(v.iter().map(|(k, p)| (k.to_string(), *p)).collect())
    }

    #[test]
    fn test_validate_chisq_gof() {
        // R (case stat_chisq_gof, cat = a x5, b x30, c x15 plus 2 NA): chisq.test(table(cat))
        // 19.000000000000004, df 2, p 7.485182988770046e-05; p = c(.1, .6, .3) in level order
        // -> 0, p 1, expected [5, 30, 15]; p = c(.2, .5, .3) -> 3.5, p 0.17377394345044514;
        // named map with an unobserved "d" (.1, .5, .3, .1) -> 6, df 3, p 0.11161022509471258.
        // Fails before: value_counts counts the null (uniform: 36.769, df 3) and orders by
        // frequency (probs: "length (3) must match observed length (4)", or 139.17 on
        // complete rows; STAT-3).
        let (case, ds) = crate::test_support::load_case("stat_chisq_gof");
        let uni = run_chisq_gof(&ds, "cat", None).unwrap();
        let p1 = run_chisq_gof(&ds, "cat", Some(&[0.1, 0.6, 0.3][..])).unwrap();
        let p2 = run_chisq_gof_with(&ds, "cat", &GofExpected::Probs(vec![0.2, 0.5, 0.3]), false)
            .unwrap();
        let n1 = run_chisq_gof_with(
            &ds,
            "cat",
            &named(&[("c", 0.3), ("a", 0.1), ("b", 0.6)]),
            false,
        )
        .unwrap();
        let n4 = run_chisq_gof_with(
            &ds,
            "cat",
            &named(&[("a", 0.1), ("b", 0.5), ("c", 0.3), ("d", 0.1)]),
            false,
        )
        .unwrap();
        let r4 = run_chisq_gof_with(
            &ds,
            "cat",
            &named(&[("a", 1.0), ("b", 5.0), ("c", 3.0), ("d", 1.0)]),
            true,
        )
        .unwrap();
        let die = chisq_test_gof(&case.vector("die_observed"), None, false).unwrap();
        Check::new(&case)
            .scalar("uniform_chisq", uni.statistic, Tol::stat(50))
            .scalar("uniform_df", uni.df as f64, Tol::exact())
            .scalar("uniform_p", uni.p_value, Tol::p_value(50))
            .vector("observed", &uni.observed, Tol::exact())
            .scalar("n_used", uni.missing.n_used as f64, Tol::exact())
            .scalar("p1_chisq", p1.statistic, Tol::stat(50))
            .scalar("p1_p", p1.p_value, Tol::p_value(50))
            .vector("p1_expected", &p1.expected, Tol::stat(50))
            .scalar("p1_chisq", n1.statistic, Tol::stat(50))
            .vector("p1_expected", &n1.expected, Tol::stat(50))
            .scalar("p2_chisq", p2.statistic, Tol::stat(50))
            .scalar("p2_p", p2.p_value, Tol::p_value(50))
            .vector("p2_expected", &p2.expected, Tol::stat(50))
            .vector("p2_residuals", &p2.residuals, Tol::stat(50))
            .scalar("named4_chisq", n4.statistic, Tol::stat(50))
            .scalar("named4_df", n4.df as f64, Tol::exact())
            .scalar("named4_p", n4.p_value, Tol::p_value(50))
            .vector("named4_observed", &n4.observed, Tol::exact())
            .vector("named4_expected", &n4.expected, Tol::stat(50))
            .vector("named4_residuals", &n4.residuals, Tol::stat(50))
            .scalar("named4_chisq", r4.statistic, Tol::stat(50))
            .scalar("named4_p", r4.p_value, Tol::p_value(50))
            .scalar("die_chisq", die.statistic, Tol::stat(50))
            .scalar("die_df", die.df as f64, Tol::exact())
            .scalar("die_p", die.p_value, Tol::p_value(50))
            .finish();
        assert_eq!(uni.row_labels.clone().unwrap(), ["a", "b", "c"]);
        assert_eq!(n1.row_labels.clone().unwrap(), ["a", "b", "c"]);
        assert_eq!(n4.row_labels.clone().unwrap(), ["a", "b", "c", "d"]);
        assert_eq!(uni.missing.by_column, vec![("cat".to_string(), 2)]);
        assert!(uni.warnings.is_empty() && n4.warnings.is_empty()); // R warns for neither
    }

    #[test]
    fn test_validate_chisq_gof_na_codes() {
        // R: chisq.test(table(h)) after dropping 3 NA: observed [4, 5], 0.1111111111111111,
        // df 1, p 0.7388826803635273 (with R's warning). Fails before: the null is a third
        // category, 0.5, df 2.
        let (case, ds) = crate::test_support::load_case("stat_contingency_na_codes");
        let r = run_chisq_gof(&ds, "h", None).unwrap();
        Check::new(&case)
            .scalar("h_gof_chisq", r.statistic, Tol::stat(9))
            .scalar("h_gof_df", r.df as f64, Tol::exact())
            .scalar("h_gof_p", r.p_value, Tol::p_value(9))
            .vector("h_gof_observed", &r.observed, Tol::exact())
            .finish();
        assert_eq!(r.row_labels.unwrap(), ["u", "v"]);
        assert!(r.warnings.iter().any(|w| w.starts_with(R_WARNING)));
    }

    #[test]
    fn test_validate_chisq_gof_errors() {
        // The order is part of the message, so a caller can fix probs (STAT-3, TOOL-33); R's
        // sum rule is sqrt(eps).
        let (_case, ds) = crate::test_support::load_case("stat_chisq_gof");
        let e = run_chisq_gof(&ds, "cat", Some(&[0.25, 0.25, 0.25, 0.25][..]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("[a, b, c]") && e.contains("4"), "{e}");
        let e = run_chisq_gof_with(&ds, "cat", &named(&[("a", 0.5), ("c", 0.5)]), false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("'b'") && e.contains("30"), "{e}"); // observed category without p
        assert!(
            run_chisq_gof_with(
                &ds,
                "cat",
                &named(&[("a", 0.1), ("b", 0.6), ("c", 0.3 + 2e-8)]),
                false
            )
            .is_err()
        );
        assert!(
            run_chisq_gof_with(
                &ds,
                "cat",
                &named(&[("a", 0.1), ("b", 0.6), ("c", 0.3), ("a", 0.0)]),
                false
            )
            .is_err()
        );
        assert!(
            run_chisq_gof_with(
                &ds,
                "cat",
                &named(&[("a", -0.1), ("b", 0.8), ("c", 0.3)]),
                false
            )
            .is_err()
        );
        assert!(run_chisq_gof_with(&ds, "cat", &GofExpected::Named(vec![]), false).is_err());
        assert!(matches!(
            run_chisq_gof(&ds, "nope", None),
            Err(EconError::ColumnNotFound { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_run_chisq_gof_inside_tokio_runtime() {
        // STAT-10 (third lazy site): run_chisq_gof called value_counts through the lazy engine.
        // 1000 control and 1000 treated rows: uniform X2 = 0, df 1, p 1 (case
        // stat_fisher_rows_2000's table).
        let (_case, ds) = crate::test_support::load_case("stat_fisher_rows_2000");
        let r = run_chisq_gof(&ds, "treat", None).unwrap();
        assert_eq!((r.statistic, r.df, r.p_value), (0.0, 1, 1.0));
        assert_eq!(r.row_labels.unwrap(), ["control", "treated"]);
    }

    #[test]
    fn test_chisq_gof_named_matches_numeric_categories_by_value() {
        // A numeric (Int64) column with one missing code: "2.0" names the level "2", a key
        // given twice by value is refused, and the declared "33" that never occurs counts 0
        // (R: chisq.test(table(factor(code, levels = c(2, 10, 33))), p = c(.5, .4, .1))).
        let code = [10, 2, 10, 2, 2, 10, 2, 10, 10, 2, -1, 2].map(|c| (c > 0).then_some(c as i64));
        let ds = Dataset::new(df!("code" => code).unwrap());
        let r = run_chisq_gof_with(
            &ds,
            "code",
            &named(&[("2.0", 0.5), ("10", 0.4), ("33", 0.1)]),
            false,
        )
        .unwrap();
        assert_eq!(r.row_labels.clone().unwrap(), ["2", "10", "33"]);
        assert_eq!(r.observed, [6.0, 5.0, 0.0]);
        for (e, want) in r.expected.iter().zip([5.5, 4.4, 1.1]) {
            assert!((e - want).abs() < 1e-12, "{:?}", r.expected);
        }
        let e = run_chisq_gof_with(
            &ds,
            "code",
            &named(&[("2", 0.5), ("2.0", 0.1), ("10", 0.4)]),
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("'2'") && e.contains("twice"), "{e}");
    }
}
