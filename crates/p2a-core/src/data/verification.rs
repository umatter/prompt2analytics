//! Verification and preview system for data cleaning operations.
//!
//! This module provides:
//! - Preview of cleaning operations before applying
//! - Verification reports comparing before/after states
//! - Change tracking with sample examples
//! - Quality delta computation

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use super::Dataset;
use super::columns::is_numeric_dtype;
use super::munging::clean::{
    FillStrategy, ReplaceOptions, deduplicate, drop_na, fill_na_with, replace_with,
};
use super::munging::transform::{drop_columns, filter};
use super::quality::{
    DataIssue, DataQualityProfile, completeness, generate_quality_profile, missing_cells,
};

/// Result of a cleaning operation with verification.
#[derive(Clone, Serialize, Deserialize)]
pub struct CleaningResult {
    /// The cleaned dataset (not serialized, not Debug due to DataFrame)
    #[serde(skip)]
    pub dataset: Option<Dataset>,
    /// Description of the operation performed
    pub operation: String,
    /// Verification report comparing before and after
    pub verification: VerificationReport,
    /// Unique ID for rollback capability
    pub rollback_id: String,
}

/// Verification report comparing before and after states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Number of rows before the operation
    pub rows_before: usize,
    /// Number of rows after the operation
    pub rows_after: usize,
    /// Number of rows with at least one changed cell (rows of the same index; 0 when the row
    /// count changed)
    pub rows_modified: usize,
    /// Number of changed cells
    #[serde(default)]
    pub cells_modified: usize,
    /// Number of rows that were removed
    pub rows_removed: usize,
    /// Number of rows that were added
    pub rows_added: usize,
    /// Sample of changes (before/after pairs)
    pub sample_changes: Vec<ChangeExample>,
    /// Quality metrics comparison
    pub quality_delta: QualityDelta,
    /// Any warnings generated during verification
    pub warnings: Vec<String>,
    /// Columns whose type changed
    #[serde(default)]
    pub dtype_changes: Vec<DtypeChange>,
}

/// A column whose type a cleaning step changed (polars dtype names, e.g. "i64", "f64").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DtypeChange {
    pub column: String,
    pub before: String,
    pub after: String,
}

/// Example of a change made during cleaning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeExample {
    /// Row index in the original dataset
    pub row_index: usize,
    /// Column name that was changed
    pub column: String,
    /// Value before the change
    pub before: String,
    /// Value after the change
    pub after: String,
}

/// Comparison of quality metrics before and after cleaning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDelta {
    /// Completeness score before
    pub completeness_before: f64,
    /// Completeness score after
    pub completeness_after: f64,
    /// Change in completeness (positive = improvement)
    pub completeness_change: f64,
    /// Number of issues before
    pub issues_before: usize,
    /// Number of issues after
    pub issues_after: usize,
    /// Issues that were resolved
    pub issues_resolved: Vec<String>,
    /// New issues introduced (if any)
    pub issues_introduced: Vec<String>,
    /// Issues of the same kind and column on both sides whose count or detail changed, e.g.
    /// "missing values in 'x': 12 -> 3"
    #[serde(default)]
    pub issues_changed: Vec<String>,
}

/// Preview of what a cleaning operation would do.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningPreview {
    /// Description of the operation
    pub operation: String,
    /// Number of distinct rows the operation would change or remove
    pub rows_affected: usize,
    /// Percentage of rows affected (at most 100)
    pub pct_affected: f64,
    /// Number of cells the operation would change (0 for an operation that removes rows or
    /// columns)
    #[serde(default)]
    pub cells_affected: usize,
    /// Sample of changes that would be made
    pub sample_changes: Vec<ChangeExample>,
    /// The change in completeness (the share of present cells, NaN counted as missing) the
    /// operation would make, in percentage points: the operation is run on the frame and the
    /// completeness of the result compared with that of the frame
    pub estimated_quality_change: f64,
    /// Any potential concerns
    pub warnings: Vec<String>,
}

/// Supported cleaning operations for preview.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleaningOperation {
    /// Trim whitespace from string columns
    Trim { columns: Option<Vec<String>> },
    /// Convert to lowercase
    ToLowercase { column: String },
    /// Convert to uppercase
    ToUppercase { column: String },
    /// Fill null values
    FillNa {
        columns: Option<Vec<String>>,
        strategy: String,
        value: Option<String>,
    },
    /// Drop rows with null values
    DropNa {
        columns: Option<Vec<String>>,
        how: String,
    },
    /// Remove duplicate rows
    Deduplicate {
        columns: Option<Vec<String>>,
        keep: String,
    },
    /// Replace values
    Replace {
        column: String,
        old_value: String,
        new_value: String,
    },
    /// Filter rows
    Filter {
        column: String,
        operator: String,
        value: String,
    },
    /// Drop whole columns
    DropColumns { columns: Vec<String> },
}

impl CleaningOperation {
    /// Get a human-readable description of the operation.
    pub fn description(&self) -> String {
        match self {
            CleaningOperation::Trim { columns } => match columns {
                Some(cols) => format!("Trim whitespace from columns: {}", cols.join(", ")),
                None => "Trim whitespace from all string columns".to_string(),
            },
            CleaningOperation::ToLowercase { column } => {
                format!("Convert '{}' to lowercase", column)
            }
            CleaningOperation::ToUppercase { column } => {
                format!("Convert '{}' to uppercase", column)
            }
            CleaningOperation::FillNa {
                columns,
                strategy,
                value,
            } => {
                let cols = match columns {
                    Some(c) => c.join(", "),
                    None => "all columns".to_string(),
                };
                match value {
                    Some(v) => format!("Fill nulls in {} with '{}'", cols, v),
                    None => format!("Fill nulls in {} using {} strategy", cols, strategy),
                }
            }
            CleaningOperation::DropNa { columns, how } => {
                let cols = match columns {
                    Some(c) => c.join(", "),
                    None => "all columns".to_string(),
                };
                format!("Drop rows with {} nulls in {}", how, cols)
            }
            CleaningOperation::Deduplicate { columns, keep } => {
                let cols = match columns {
                    Some(c) => c.join(", "),
                    None => "all columns".to_string(),
                };
                format!("Remove duplicates based on {}, keeping {}", cols, keep)
            }
            CleaningOperation::Replace {
                column,
                old_value,
                new_value,
            } => {
                format!(
                    "Replace '{}' with '{}' in column '{}'",
                    old_value, new_value, column
                )
            }
            CleaningOperation::Filter {
                column,
                operator,
                value,
            } => {
                format!("Filter rows where {} {} {}", column, operator, value)
            }
            CleaningOperation::DropColumns { columns } => {
                format!("Drop columns: {}", columns.join(", "))
            }
        }
    }
}

/// Generate a preview of what a cleaning operation would do.
pub fn preview_cleaning(
    dataset: &Dataset,
    operation: &CleaningOperation,
    sample_size: usize,
) -> CleaningPreview {
    let df = dataset.df();
    let row_count = df.height();

    match operation {
        CleaningOperation::Trim { columns } => {
            preview_trim(df, columns.as_deref(), sample_size, row_count)
        }
        CleaningOperation::ToLowercase { column } => {
            preview_case_change(df, column, true, sample_size, row_count)
        }
        CleaningOperation::ToUppercase { column } => {
            preview_case_change(df, column, false, sample_size, row_count)
        }
        CleaningOperation::FillNa {
            columns,
            strategy,
            value,
        } => preview_fill_na(
            df,
            columns.as_deref(),
            strategy,
            value.as_deref(),
            sample_size,
            row_count,
        ),
        CleaningOperation::DropNa { columns, how } => {
            preview_drop_na(df, columns.as_deref(), how, sample_size, row_count)
        }
        CleaningOperation::Deduplicate { columns, keep } => {
            preview_deduplicate(df, columns.as_deref(), keep, row_count)
        }
        CleaningOperation::Replace {
            column,
            old_value,
            new_value,
        } => preview_replace(df, column, old_value, new_value, sample_size, row_count),
        CleaningOperation::Filter {
            column,
            operator,
            value,
        } => preview_filter(df, column, operator, value, row_count),
        CleaningOperation::DropColumns { columns } => preview_drop_columns(df, columns),
    }
}

/// Preview dropping columns: no row changes, the columns named, and a warning for a name the
/// frame does not have (the apply refuses it).
fn preview_drop_columns(df: &DataFrame, columns: &[String]) -> CleaningPreview {
    let mut warnings = Vec::new();
    let mut quality_change = 0.0;
    let names: Vec<&str> = columns.iter().map(String::as_str).collect();
    if columns.is_empty() {
        warnings.push("No columns named to drop".to_string());
    } else {
        match drop_columns(&Dataset::new(df.clone()), &names) {
            Err(e) => warnings.push(e.to_string()),
            Ok(after) => {
                quality_change = completeness_change(df, after.df());
                if after.df().width() == 0 {
                    warnings.push("Warning: This operation would drop ALL columns!".to_string());
                }
            }
        }
    }
    CleaningPreview {
        operation: CleaningOperation::DropColumns {
            columns: columns.to_vec(),
        }
        .description(),
        rows_affected: 0,
        pct_affected: 0.0,
        cells_affected: 0,
        sample_changes: vec![],
        estimated_quality_change: quality_change,
        warnings,
    }
}

/// `n` as a percentage of `rows`; 0 for a frame with no rows.
fn pct_of(n: usize, rows: usize) -> f64 {
    if rows > 0 {
        n as f64 / rows as f64 * 100.0
    } else {
        0.0
    }
}

/// The completeness change from `before` to `after` in percentage points.
fn completeness_change(before: &DataFrame, after: &DataFrame) -> f64 {
    100.0 * (completeness(after) - completeness(before))
}

/// Calls `f` with each row a fill changed in one column (missing in `before`, present in
/// `after`; missing is null, or NaN in a float column), in order, until `f` returns false.
fn filled_positions(before: &Column, after: &Column, mut f: impl FnMut(usize) -> bool) {
    let float = |c: &Column| matches!(c.dtype(), DataType::Float64 | DataType::Float32);
    if float(before)
        && float(after)
        && let (Ok(b), Ok(a)) = (
            before.cast(&DataType::Float64),
            after.cast(&DataType::Float64),
        )
        && let (Ok(b), Ok(a)) = (b.f64(), a.f64())
    {
        let missing = |v: Option<f64>| v.is_none_or(f64::is_nan);
        for (idx, (was, is)) in b.iter().zip(a.iter()).enumerate() {
            if missing(was) && !missing(is) && !f(idx) {
                return;
            }
        }
        return;
    }
    let (was, is) = (before.is_null(), after.is_null());
    for (idx, (was, is)) in was.iter().zip(is.iter()).enumerate() {
        if was == Some(true) && is == Some(false) && !f(idx) {
            return;
        }
    }
}

/// Preview trim operation.
fn preview_trim(
    df: &DataFrame,
    columns: Option<&[String]>,
    sample_size: usize,
    row_count: usize,
) -> CleaningPreview {
    let mut sample_changes = Vec::new();
    let mut cells = 0usize;
    let mut rows = vec![false; row_count];
    let mut warnings = Vec::new();

    let cols_to_check: Vec<String> = match columns {
        Some(cols) => cols.to_vec(),
        None => df
            .get_column_names()
            .iter()
            .filter(|name| {
                df.column(name.as_str())
                    .map(|c| c.dtype() == &DataType::String)
                    .unwrap_or(false)
            })
            .map(|s| s.to_string())
            .collect(),
    };

    for col_name in &cols_to_check {
        let Ok(col) = df.column(col_name) else {
            warnings.push(format!("Column '{col_name}' not found"));
            continue;
        };
        let Ok(ca) = col.str() else {
            continue;
        };
        for (idx, val) in ca.iter().enumerate() {
            let Some(s) = val else { continue };
            let trimmed = s.trim();
            if trimmed != s {
                cells += 1;
                rows[idx] = true;
                if sample_changes.len() < sample_size {
                    sample_changes.push(ChangeExample {
                        row_index: idx,
                        column: col_name.clone(),
                        before: format!("'{}'", s),
                        after: format!("'{}'", trimmed),
                    });
                }
            }
        }
    }

    if cols_to_check.is_empty() {
        warnings.push("No string columns found to trim".to_string());
    }
    let rows_affected = rows.iter().filter(|r| **r).count();

    CleaningPreview {
        operation: CleaningOperation::Trim {
            columns: columns.map(|c| c.to_vec()),
        }
        .description(),
        rows_affected,
        pct_affected: pct_of(rows_affected, row_count),
        cells_affected: cells,
        sample_changes,
        // Trimming changes no missing value.
        estimated_quality_change: 0.0,
        warnings,
    }
}

/// Preview case change (lowercase/uppercase).
fn preview_case_change(
    df: &DataFrame,
    column: &str,
    to_lower: bool,
    sample_size: usize,
    row_count: usize,
) -> CleaningPreview {
    let mut sample_changes = Vec::new();
    let mut total_affected = 0;
    let mut warnings = Vec::new();

    if let Ok(col) = df.column(column) {
        if col.dtype() != &DataType::String {
            warnings.push(format!("Column '{}' is not a string column", column));
        } else if let Ok(ca) = col.str() {
            for (idx, val) in ca.into_iter().enumerate() {
                if let Some(s) = val {
                    let changed = if to_lower {
                        s.to_lowercase()
                    } else {
                        s.to_uppercase()
                    };
                    if changed != s {
                        total_affected += 1;
                        if sample_changes.len() < sample_size {
                            sample_changes.push(ChangeExample {
                                row_index: idx,
                                column: column.to_string(),
                                before: s.to_string(),
                                after: changed,
                            });
                        }
                    }
                }
            }
        }
    } else {
        warnings.push(format!("Column '{}' not found", column));
    }

    let pct_affected = pct_of(total_affected, row_count);

    let operation = if to_lower {
        CleaningOperation::ToLowercase {
            column: column.to_string(),
        }
    } else {
        CleaningOperation::ToUppercase {
            column: column.to_string(),
        }
    };

    CleaningPreview {
        operation: operation.description(),
        rows_affected: total_affected,
        pct_affected,
        cells_affected: total_affected,
        sample_changes,
        estimated_quality_change: 0.0, // Case changes don't affect quality score
        warnings,
    }
}

/// Cell `idx` of `col` as a preview prints it: floats with `{}` (`3`, `2.5`), everything else as its
/// text (text unquoted, `null` for missing, ISO dates).
fn cell_text(col: &Column, idx: usize) -> String {
    match col.get(idx) {
        Ok(AnyValue::Float64(v)) => v.to_string(),
        Ok(AnyValue::Float32(v)) => v.to_string(),
        Ok(v) => v.str_value().into_owned(),
        Err(_) => "null".to_string(),
    }
}

/// Preview fill_na operation: runs the fill, so the samples show the real new values, and an unknown
/// strategy or one that does not apply is a warning with nothing affected.
fn preview_fill_na(
    df: &DataFrame,
    columns: Option<&[String]>,
    strategy: &str,
    value: Option<&str>,
    sample_size: usize,
    row_count: usize,
) -> CleaningPreview {
    let mut sample_changes = Vec::new();
    let mut cells = 0usize;
    let mut rows = Vec::new();
    let mut single_rows = 0usize;
    let mut quality_change = 0.0;
    let mut warnings = Vec::new();

    let named: Option<Vec<&str>> = columns.map(|c| c.iter().map(String::as_str).collect());
    let outcome = FillStrategy::parse(strategy, value)
        .and_then(|parsed| fill_na_with(&Dataset::new(df.clone()), named.as_deref(), &parsed));

    match outcome {
        Err(e) => warnings.push(e.to_string()),
        Ok(outcome) => {
            let filled_df = outcome.dataset.df();
            if outcome.filled.iter().filter(|(_, n)| *n > 0).count() > 1 {
                rows = vec![false; row_count];
            }
            for (col_name, n_filled) in &outcome.filled {
                if *n_filled == 0 {
                    continue;
                }
                cells += *n_filled;
                let (Ok(before), Ok(after)) = (df.column(col_name), filled_df.column(col_name))
                else {
                    continue;
                };
                // One filled column: its filled cells are the rows, and only the samples need
                // positions. Several: the rows are the union of their filled cells.
                let single = outcome.filled.iter().filter(|(_, n)| *n > 0).count() == 1;
                let mut samples_for_col = 0;
                filled_positions(before, after, |idx| {
                    if !single {
                        rows[idx] = true;
                    } else if samples_for_col == 2 || sample_changes.len() >= sample_size {
                        return false;
                    }
                    // Up to 2 samples per column.
                    if samples_for_col < 2 && sample_changes.len() < sample_size {
                        sample_changes.push(ChangeExample {
                            row_index: idx,
                            column: col_name.clone(),
                            before: cell_text(before, idx),
                            after: cell_text(after, idx),
                        });
                        samples_for_col += 1;
                    }
                    true
                });
                if single {
                    single_rows = *n_filled;
                }
            }
            // A fill turns missing cells into present ones and changes nothing else.
            quality_change = pct_of(cells, row_count.saturating_mul(df.width()));
            for (col_name, reason) in &outcome.skipped {
                warnings.push(format!("Column '{col_name}' would be skipped: {reason}"));
            }
            if cells == 0 {
                warnings.push("No null values found in specified columns".to_string());
            }
        }
    }
    let rows_affected = single_rows.max(rows.iter().filter(|r| **r).count());

    CleaningPreview {
        operation: CleaningOperation::FillNa {
            columns: columns.map(|c| c.to_vec()),
            strategy: strategy.to_string(),
            value: value.map(|v| v.to_string()),
        }
        .description(),
        rows_affected,
        pct_affected: pct_of(rows_affected, row_count),
        cells_affected: cells,
        sample_changes,
        estimated_quality_change: quality_change,
        warnings,
    }
}

/// Preview drop_na operation.
fn preview_drop_na(
    df: &DataFrame,
    columns: Option<&[String]>,
    how: &str,
    _sample_size: usize,
    row_count: usize,
) -> CleaningPreview {
    let mut warnings = Vec::new();
    let mut rows_to_drop = 0;
    let mut quality_change = 0.0;

    // The apply's own drop: NaN is missing, a missing column or an unknown `how` is reported.
    let named: Option<Vec<&str>> = columns.map(|c| c.iter().map(String::as_str).collect());
    match drop_na(&Dataset::new(df.clone()), named.as_deref(), how) {
        Err(e) => warnings.push(e.to_string()),
        Ok(after) => {
            rows_to_drop = row_count.saturating_sub(after.nrows());
            quality_change = completeness_change(df, after.df());
        }
    }

    if row_count > 0 && rows_to_drop == row_count {
        warnings.push("Warning: This operation would drop ALL rows!".to_string());
    } else if rows_to_drop * 2 > row_count {
        warnings.push(format!(
            "Warning: This operation would drop more than 50% of rows ({} of {})",
            rows_to_drop, row_count
        ));
    }

    CleaningPreview {
        operation: CleaningOperation::DropNa {
            columns: columns.map(|c| c.to_vec()),
            how: how.to_string(),
        }
        .description(),
        rows_affected: rows_to_drop,
        pct_affected: pct_of(rows_to_drop, row_count),
        cells_affected: 0,
        sample_changes: vec![], // No specific changes to show for row deletion
        estimated_quality_change: quality_change,
        warnings,
    }
}

/// Preview deduplicate operation.
fn preview_deduplicate(
    df: &DataFrame,
    columns: Option<&[String]>,
    keep: &str,
    row_count: usize,
) -> CleaningPreview {
    let mut warnings = Vec::new();
    let mut duplicates = 0;
    let mut quality_change = 0.0;

    let named: Option<Vec<&str>> = columns.map(|c| c.iter().map(String::as_str).collect());
    match deduplicate(&Dataset::new(df.clone()), named.as_deref(), keep) {
        Err(e) => warnings.push(e.to_string()),
        Ok(after) => {
            duplicates = row_count.saturating_sub(after.nrows());
            quality_change = completeness_change(df, after.df());
            if duplicates == 0 {
                warnings.push("No duplicate rows found".to_string());
            }
        }
    }

    CleaningPreview {
        operation: CleaningOperation::Deduplicate {
            columns: columns.map(|c| c.to_vec()),
            keep: keep.to_string(),
        }
        .description(),
        rows_affected: duplicates,
        pct_affected: pct_of(duplicates, row_count),
        cells_affected: 0,
        sample_changes: vec![],
        estimated_quality_change: quality_change,
        warnings,
    }
}

/// Preview replace operation.
fn preview_replace(
    df: &DataFrame,
    column: &str,
    old_value: &str,
    new_value: &str,
    sample_size: usize,
    row_count: usize,
) -> CleaningPreview {
    let mut sample_changes = Vec::new();
    let mut total_affected = 0;
    let mut quality_change = 0.0;
    let mut warnings = Vec::new();

    match replace_with(
        &Dataset::new(df.clone()),
        column,
        old_value,
        Some(new_value),
        &ReplaceOptions::default(),
    ) {
        Err(e) => warnings.push(e.to_string()),
        Ok(outcome) => {
            total_affected = outcome.n_replaced;
            // Only `column` changes, and only where a value was replaced.
            if total_affected > 0
                && let (Ok(before), Ok(after)) =
                    (df.column(column), outcome.dataset.df().column(column))
            {
                let cells = row_count.saturating_mul(df.width());
                quality_change =
                    pct_of(missing_cells(before), cells) - pct_of(missing_cells(after), cells);
            }
            if total_affected == 0 {
                warnings.push(format!(
                    "No values matching '{}' found in column '{}'",
                    old_value, column
                ));
            } else if let (Ok(before), Ok(after)) =
                (df.column(column), outcome.dataset.df().column(column))
            {
                match first_changes(before, after, sample_size) {
                    Ok(rows) => {
                        for idx in rows {
                            sample_changes.push(ChangeExample {
                                row_index: idx,
                                column: column.to_string(),
                                before: cell_text(before, idx),
                                after: cell_text(after, idx),
                            });
                        }
                    }
                    Err(reason) => warnings.push(format!(
                        "Column '{column}': values were not compared ({reason})"
                    )),
                }
            }
        }
    }

    let pct_affected = pct_of(total_affected, row_count);

    CleaningPreview {
        operation: CleaningOperation::Replace {
            column: column.to_string(),
            old_value: old_value.to_string(),
            new_value: new_value.to_string(),
        }
        .description(),
        rows_affected: total_affected,
        pct_affected,
        cells_affected: total_affected,
        sample_changes,
        estimated_quality_change: quality_change,
        warnings,
    }
}

/// Preview filter operation.
fn preview_filter(
    df: &DataFrame,
    column: &str,
    operator: &str,
    value: &str,
    row_count: usize,
) -> CleaningPreview {
    let mut warnings = Vec::new();
    let mut rows_to_remove = 0;
    let mut quality_change = 0.0;

    // The apply's own filter: its operators, literal `contains`, its typing of `value` and its
    // treatment of missing values.
    match filter(&Dataset::new(df.clone()), column, operator, value) {
        Err(e) => warnings.push(e.to_string()),
        Ok(after) => {
            rows_to_remove = row_count.saturating_sub(after.nrows());
            quality_change = completeness_change(df, after.df());
            if row_count > 0 && rows_to_remove == row_count {
                warnings.push("Warning: This filter would remove ALL rows!".to_string());
            }
        }
    }

    CleaningPreview {
        operation: CleaningOperation::Filter {
            column: column.to_string(),
            operator: operator.to_string(),
            value: value.to_string(),
        }
        .description(),
        rows_affected: rows_to_remove,
        pct_affected: pct_of(rows_to_remove, row_count),
        cells_affected: 0,
        sample_changes: vec![],
        estimated_quality_change: quality_change,
        warnings,
    }
}

/// Generate a verification report comparing before and after datasets.
pub fn verify_cleaning(before: &Dataset, after: &Dataset, _operation: &str) -> VerificationReport {
    let rows_before = before.nrows();
    let rows_after = after.nrows();

    // Generate quality profiles
    let profile_before = generate_quality_profile(before);
    let profile_after = generate_quality_profile(after);

    let quality_delta = quality_delta(&profile_before, &profile_after);

    // Detect changes by comparing dataframes
    let scan = detect_changes(before, after, 5);

    let mut warnings = Vec::new();
    if rows_after == 0 && rows_before > 0 {
        warnings.push("Warning: All rows were removed!".to_string());
    }
    if !quality_delta.issues_introduced.is_empty() {
        warnings.push(format!(
            "Warning: {} new issues were introduced",
            quality_delta.issues_introduced.len()
        ));
    }
    warnings.extend(scan.warnings);

    VerificationReport {
        rows_before,
        rows_after,
        rows_modified: scan.rows_modified,
        cells_modified: scan.cells_modified,
        rows_removed: rows_before.saturating_sub(rows_after),
        rows_added: rows_after.saturating_sub(rows_before),
        sample_changes: scan.sample_changes,
        quality_delta,
        warnings,
        dtype_changes: scan.dtype_changes,
    }
}

/// An issue's identity across two profiles: its kind and its column (`None` for a dataset-level
/// issue; the columns joined for possible duplicates).
type IssueKey = (std::mem::Discriminant<DataIssue>, Option<String>);

fn issue_key(issue: &DataIssue) -> IssueKey {
    let column = match issue {
        DataIssue::DuplicateRows { .. } => None,
        DataIssue::PossibleDuplicates { columns, .. } => Some(columns.join(", ")),
        DataIssue::HighNullRate { column, .. }
        | DataIssue::MixedTypes { column, .. }
        | DataIssue::OutlierValues { column, .. }
        | DataIssue::InconsistentFormat { column, .. }
        | DataIssue::WhitespaceIssues { column, .. }
        | DataIssue::ConstantColumn { column, .. }
        | DataIssue::LowCardinality { column, .. }
        | DataIssue::HighCardinality { column, .. }
        | DataIssue::EmptyStrings { column, .. } => Some(column.clone()),
    };
    (std::mem::discriminant(issue), column)
}

/// What an issue counts, and how many, for the "changed" line; `None` for a kind without a count.
fn issue_count(issue: &DataIssue, profile: &DataQualityProfile) -> Option<(&'static str, usize)> {
    match issue {
        DataIssue::HighNullRate { column, .. } => profile
            .columns
            .iter()
            .find(|c| &c.name == column)
            .map(|c| ("missing values", c.null_count)),
        DataIssue::DuplicateRows { count } => Some(("duplicate rows", *count)),
        DataIssue::PossibleDuplicates { count, .. } => Some(("possible duplicate rows", *count)),
        DataIssue::OutlierValues { count, .. } => Some(("outliers", *count)),
        DataIssue::WhitespaceIssues { count, .. } => {
            Some(("values with leading/trailing whitespace", *count))
        }
        DataIssue::EmptyStrings { count, .. } => Some(("empty strings", *count)),
        DataIssue::LowCardinality { unique_count, .. } => Some(("unique values", *unique_count)),
        _ => None,
    }
}

/// The quality delta of two profiles, issues matched by kind and column: an issue on one side
/// only is resolved or introduced, one on both sides whose count or detail differs is changed.
fn quality_delta(before: &DataQualityProfile, after: &DataQualityProfile) -> QualityDelta {
    let (issues_before, issues_after) = (before.all_issues(), after.all_issues());
    let find = |issues: &[&DataIssue], key: &IssueKey| -> Option<DataIssue> {
        issues
            .iter()
            .find(|i| &issue_key(i) == key)
            .map(|i| (*i).clone())
    };
    let mut issues_resolved = Vec::new();
    let mut issues_changed = Vec::new();
    for issue in &issues_before {
        let key = issue_key(issue);
        match find(&issues_after, &key) {
            None => issues_resolved.push(issue.description()),
            Some(now) => {
                let (was_text, now_text) = (issue.description(), now.description());
                if was_text == now_text {
                    continue;
                }
                let line = match (issue_count(issue, before), issue_count(&now, after)) {
                    (Some((what, was)), Some((_, is))) if was != is => match &key.1 {
                        Some(column) => format!("{what} in '{column}': {was} -> {is}"),
                        None => format!("{what}: {was} -> {is}"),
                    },
                    _ => format!("{was_text} -> {now_text}"),
                };
                issues_changed.push(line);
            }
        }
    }
    let issues_introduced = issues_after
        .iter()
        .filter(|i| find(&issues_before, &issue_key(i)).is_none())
        .map(|i| i.description())
        .collect();
    QualityDelta {
        completeness_before: before.completeness_score,
        completeness_after: after.completeness_score,
        completeness_change: after.completeness_score - before.completeness_score,
        issues_before: issues_before.len(),
        issues_after: issues_after.len(),
        issues_resolved,
        issues_introduced,
        issues_changed,
    }
}

/// What [`detect_changes`] found.
struct ChangeScan {
    rows_modified: usize,
    cells_modified: usize,
    sample_changes: Vec<ChangeExample>,
    dtype_changes: Vec<DtypeChange>,
    warnings: Vec<String>,
}

/// Do `before` and `after` compare cell by cell as values (one type, not categorical), rather
/// than through a cast?
fn compares_by_value(before: &Column, after: &Column) -> bool {
    let categorical = |dt: &DataType| matches!(dt, DataType::Categorical(..) | DataType::Enum(..));
    before.dtype() == after.dtype() && !categorical(before.dtype())
}

/// Does cell `idx` differ between two columns that [`compares_by_value`]? NaN equals NaN.
fn cell_changed(before: &Column, after: &Column, idx: usize) -> bool {
    match (before.get(idx), after.get(idx)) {
        (Ok(AnyValue::Float64(a)), Ok(AnyValue::Float64(b))) => {
            !(a == b || (a.is_nan() && b.is_nan()))
        }
        (Ok(AnyValue::Float32(a)), Ok(AnyValue::Float32(b))) => {
            !(a == b || (a.is_nan() && b.is_nan()))
        }
        (a, b) => a.ok() != b.ok(),
    }
}

/// The first `limit` rows where [`changed_mask`] would be true, without comparing the rest of the
/// column when the two columns share a type.
fn first_changes(before: &Column, after: &Column, limit: usize) -> Result<Vec<usize>, String> {
    if compares_by_value(before, after) {
        return Ok((0..before.len())
            .filter(|&idx| cell_changed(before, after, idx))
            .take(limit)
            .collect());
    }
    let mask = changed_mask(before, after)?;
    Ok(mask
        .iter()
        .enumerate()
        .filter(|(_, changed)| **changed)
        .map(|(idx, _)| idx)
        .take(limit)
        .collect())
}

/// One flag per row: does the cell differ between `before` and `after` (of the same height)?
///
/// Columns of one type compare by value. polars' `AnyValue` comparison has no rule for two types, so
/// a type change compares as doubles when both sides are numeric or boolean (R: `1L == 1.0` is TRUE)
/// and as text otherwise (labels for categoricals, ISO dates, decimal numbers). The error names a
/// pair that cannot be cast.
fn changed_mask(before: &Column, after: &Column) -> Result<Vec<bool>, String> {
    let (dt_before, dt_after) = (before.dtype(), after.dtype());
    let incomparable = || format!("{dt_before} -> {dt_after}");
    if compares_by_value(before, after) {
        return Ok((0..before.len())
            .map(|idx| cell_changed(before, after, idx))
            .collect());
    }
    if is_numeric_dtype(dt_before, true) && is_numeric_dtype(dt_after, true) {
        let as_doubles = |c: &Column| c.cast(&DataType::Float64).map_err(|_| incomparable());
        let (a, b) = (as_doubles(before)?, as_doubles(after)?);
        // na-ok: cast to Float64 above; both sides are compared with their nulls
        let (a, b) = (
            a.f64().map_err(|_| incomparable())?,
            b.f64().map_err(|_| incomparable())?,
        );
        return Ok(a
            .iter()
            .zip(b.iter())
            .map(|cells| match cells {
                (None, None) => false,
                (Some(x), Some(y)) => !(x == y || (x.is_nan() && y.is_nan())),
                _ => true,
            })
            .collect());
    }
    let as_text = |c: &Column| c.cast(&DataType::String).map_err(|_| incomparable());
    let (a, b) = (as_text(before)?, as_text(after)?);
    let (a, b) = (
        a.str().map_err(|_| incomparable())?,
        b.str().map_err(|_| incomparable())?,
    );
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x != y).collect())
}

/// Detect changes between two datasets.
fn detect_changes(before: &Dataset, after: &Dataset, sample_size: usize) -> ChangeScan {
    let df_before = before.df();
    let df_after = after.df();

    let mut scan = ChangeScan {
        rows_modified: 0,
        cells_modified: 0,
        sample_changes: Vec::new(),
        dtype_changes: Vec::new(),
        warnings: Vec::new(),
    };

    let col_names: Vec<String> = df_before
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let same_height = df_before.height() == df_after.height();
    let mut rows = vec![false; if same_height { df_before.height() } else { 0 }];

    for col_name in &col_names {
        let col_before = match df_before.column(col_name) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let col_after = match df_after.column(col_name) {
            Ok(c) => c,
            Err(_) => continue,
        };

        if col_before.dtype() != col_after.dtype() {
            let change = DtypeChange {
                column: col_name.clone(),
                before: col_before.dtype().to_string(),
                after: col_after.dtype().to_string(),
            };
            scan.warnings.push(format!(
                "Column '{}' changed type from {} to {}",
                change.column, change.before, change.after
            ));
            scan.dtype_changes.push(change);
        }

        // Cells are only compared when the row counts are the same
        if !same_height {
            continue;
        }

        let mask = match changed_mask(col_before, col_after) {
            Ok(mask) => mask,
            Err(reason) => {
                scan.warnings.push(format!(
                    "Column '{col_name}': values were not compared ({reason})"
                ));
                continue;
            }
        };
        for (idx, _) in mask.iter().enumerate().filter(|(_, changed)| **changed) {
            scan.cells_modified += 1;
            rows[idx] = true;
            if scan.sample_changes.len() < sample_size {
                scan.sample_changes.push(ChangeExample {
                    row_index: idx,
                    column: col_name.clone(),
                    before: format!("{:?}", col_before.get(idx).ok()),
                    after: format!("{:?}", col_after.get(idx).ok()),
                });
            }
        }
    }

    scan.rows_modified = rows.iter().filter(|r| **r).count();
    scan
}

impl VerificationReport {
    /// Get a human-readable summary of the verification.
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("Verification Report\n");
        summary.push_str("===================\n\n");

        summary.push_str(&format!(
            "Rows: {} → {} ",
            self.rows_before, self.rows_after
        ));
        if self.rows_removed > 0 {
            summary.push_str(&format!("(-{} removed) ", self.rows_removed));
        }
        if self.rows_added > 0 {
            summary.push_str(&format!("(+{} added) ", self.rows_added));
        }
        if self.rows_modified > 0 {
            summary.push_str(&format!(
                "({} modified, {} cells)",
                self.rows_modified, self.cells_modified
            ));
        }
        summary.push('\n');

        summary.push_str(&format!(
            "\nCompleteness: {:.1}% → {:.1}% ({:+.1}%)\n",
            self.quality_delta.completeness_before * 100.0,
            self.quality_delta.completeness_after * 100.0,
            self.quality_delta.completeness_change * 100.0
        ));

        summary.push_str(&format!(
            "Issues: {} → {}\n",
            self.quality_delta.issues_before, self.quality_delta.issues_after
        ));

        if !self.dtype_changes.is_empty() {
            summary.push_str("\nType changes:\n");
            for change in &self.dtype_changes {
                summary.push_str(&format!(
                    "  {}: {} → {}\n",
                    change.column, change.before, change.after
                ));
            }
        }

        if !self.quality_delta.issues_resolved.is_empty() {
            summary.push_str(&format!(
                "\n✓ Issues Resolved ({}):\n",
                self.quality_delta.issues_resolved.len()
            ));
            for issue in &self.quality_delta.issues_resolved {
                summary.push_str(&format!("  - {}\n", issue));
            }
        }

        if !self.quality_delta.issues_changed.is_empty() {
            summary.push_str(&format!(
                "\nChanged issues ({}):\n",
                self.quality_delta.issues_changed.len()
            ));
            for issue in &self.quality_delta.issues_changed {
                summary.push_str(&format!("  - {}\n", issue));
            }
        }

        if !self.quality_delta.issues_introduced.is_empty() {
            summary.push_str(&format!(
                "\n⚠ Issues Introduced ({}):\n",
                self.quality_delta.issues_introduced.len()
            ));
            for issue in &self.quality_delta.issues_introduced {
                summary.push_str(&format!("  - {}\n", issue));
            }
        }

        if !self.sample_changes.is_empty() {
            summary.push_str("\nSample Changes:\n");
            for change in &self.sample_changes {
                summary.push_str(&format!(
                    "  Row {}, '{}': {} → {}\n",
                    change.row_index, change.column, change.before, change.after
                ));
            }
        }

        if !self.warnings.is_empty() {
            summary.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                summary.push_str(&format!("  ⚠ {}\n", warning));
            }
        }

        summary
    }
}

impl CleaningPreview {
    /// Get a human-readable summary of the preview.
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("Cleaning Preview\n");
        summary.push_str("================\n\n");

        summary.push_str(&format!("Operation: {}\n", self.operation));
        summary.push_str(&format!(
            "Rows affected: {} ({:.1}%)\n",
            self.rows_affected, self.pct_affected
        ));

        if self.cells_affected > 0 {
            summary.push_str(&format!("Cells affected: {}\n", self.cells_affected));
        }
        if self.estimated_quality_change != 0.0 {
            summary.push_str(&format!(
                "Completeness change: {:+.1} percentage points\n",
                self.estimated_quality_change
            ));
        }

        if !self.sample_changes.is_empty() {
            summary.push_str("\nSample Changes:\n");
            summary.push_str("| Row | Column | Before | After |\n");
            summary.push_str("|-----|--------|--------|-------|\n");
            for change in &self.sample_changes {
                summary.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    change.row_index, change.column, change.before, change.after
                ));
            }
        }

        if !self.warnings.is_empty() {
            summary.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                summary.push_str(&format!("  ⚠ {}\n", warning));
            }
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    /// x = [1, null, NaN, 4], y = ["a", null, "c", "d"]: completeness 5/8.
    fn half_missing() -> Dataset {
        Dataset::new(
            df! {
                "x" => [Some(1.0), None, Some(f64::NAN), Some(4.0)],
                "y" => [Some("a"), None, Some("c"), Some("d")],
            }
            .unwrap(),
        )
    }

    #[test]
    fn trim_preview_counts_rows_not_cells() {
        // Pre-fix the 3 cells were 3 rows (75%), and above 100% when every cell changed.
        let ds = Dataset::new(
            df! {
                "a" => [" p", "q", "r ", "s"],
                "b" => ["t ", "u", "v", "w"],
            }
            .unwrap(),
        );
        let p = preview_cleaning(&ds, &CleaningOperation::Trim { columns: None }, 5);
        assert_eq!(p.rows_affected, 2);
        assert_eq!(p.cells_affected, 3);
        assert_eq!(p.pct_affected, 50.0);
        assert!(p.summary().contains("Cells affected: 3"));
        let all = Dataset::new(df! { "a" => [" p", " q"], "b" => ["r ", "s "] }.unwrap());
        let p = preview_cleaning(&all, &CleaningOperation::Trim { columns: None }, 5);
        assert_eq!(
            (p.rows_affected, p.cells_affected, p.pct_affected),
            (2, 4, 100.0)
        );
    }

    #[test]
    fn fill_na_preview_counts_nan_and_reports_completeness() {
        // Pre-fix NaN was not counted and the quality change was pct_affected * 0.5.
        let p = preview_cleaning(
            &half_missing(),
            &CleaningOperation::FillNa {
                columns: Some(vec!["x".into()]),
                strategy: "median".into(),
                value: None,
            },
            5,
        );
        assert_eq!(p.rows_affected, 2);
        assert_eq!(p.cells_affected, 2);
        assert_eq!(p.pct_affected, 50.0);
        assert!(
            (p.estimated_quality_change - 25.0).abs() < 1e-12,
            "{}",
            p.estimated_quality_change
        );
        assert!(
            p.summary().contains("Completeness change: +25.0"),
            "{}",
            p.summary()
        );
    }

    #[test]
    fn filter_preview_agrees_with_filter() {
        let ds = Dataset::new(
            df! {
                "v" => [Some(1.5), Some(2.5), None, Some(4.25), Some(f64::NAN)],
                "s" => ["abc", "a.c", "xyz", "a.cd", "q"],
            }
            .unwrap(),
        );
        let preview = |column: &str, operator: &str, value: &str| {
            preview_cleaning(
                &ds,
                &CleaningOperation::Filter {
                    column: column.into(),
                    operator: operator.into(),
                    value: value.into(),
                },
                5,
            )
        };
        // Pre-fix an unknown operator was silently 0 rows.
        let p = preview("v", "~", "2");
        assert_eq!(p.rows_affected, 0);
        assert!(
            p.warnings.iter().any(|w| w.contains("Valid operators")),
            "{:?}",
            p.warnings
        );
        // Pre-fix contains was a regex, so "a.c" also matched "abc".
        let p = preview("s", "contains", "a.c");
        let kept = filter(&ds, "s", "contains", "a.c").unwrap().nrows();
        assert_eq!(kept, 2);
        assert_eq!(p.rows_affected, 5 - kept);
        let p = preview("nope", ">", "1");
        assert!(
            p.warnings.iter().any(|w| w.contains("'nope'")),
            "{:?}",
            p.warnings
        );
        for (op, value) in [(">", "2"), ("<=", "4.25"), ("!=", "1.5")] {
            let p = preview("v", op, value);
            let kept = filter(&ds, "v", op, value).unwrap().nrows();
            assert_eq!(p.rows_affected, 5 - kept, "{op} {value}");
        }
        // drop_na: NaN is missing, as in the apply; a missing column is reported.
        let drop = |columns: Option<Vec<String>>| {
            preview_cleaning(
                &ds,
                &CleaningOperation::DropNa {
                    columns,
                    how: "any".into(),
                },
                5,
            )
        };
        let p = drop(Some(vec!["v".into()]));
        let kept = drop_na(&ds, Some(&["v"]), "any").unwrap().nrows();
        assert_eq!(kept, 3);
        assert_eq!(p.rows_affected, 2);
        let p = drop(Some(vec!["nope".into()]));
        assert_eq!(p.rows_affected, 0);
        assert!(
            p.warnings.iter().any(|w| w.contains("'nope'")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn replace_preview_reports_the_completeness_change() {
        // The change is taken from the replaced column only; it equals the whole-frame measure.
        let ds = Dataset::new(
            df! {
                "x" => [Some(1.5), None, Some(f64::NAN), Some(4.25)],
                "y" => ["a", "", "c", ""],
            }
            .unwrap(),
        );
        let op = CleaningOperation::Replace {
            column: "y".into(),
            old_value: "".into(),
            new_value: "null".into(),
        };
        let p = preview_cleaning(&ds, &op, 5);
        let after = replace_with(&ds, "y", "", Some("null"), &ReplaceOptions::default()).unwrap();
        let whole = completeness_change(ds.df(), after.dataset.df());
        assert_eq!(p.rows_affected, 2);
        assert!((p.estimated_quality_change - whole).abs() < 1e-12);
        assert!(
            (p.estimated_quality_change + 25.0).abs() < 1e-12,
            "{}",
            p.estimated_quality_change
        );
    }

    #[test]
    fn empty_frame_preview() {
        let ds = Dataset::new(df! { "v" => Vec::<f64>::new() }.unwrap());
        for op in [
            CleaningOperation::Filter {
                column: "v".into(),
                operator: ">".into(),
                value: "1".into(),
            },
            CleaningOperation::DropNa {
                columns: None,
                how: "any".into(),
            },
            CleaningOperation::Trim { columns: None },
        ] {
            let p = preview_cleaning(&ds, &op, 5);
            assert_eq!(p.pct_affected, 0.0);
            assert!(
                !p.warnings.iter().any(|w| w.contains("ALL")),
                "{:?}",
                p.warnings
            );
        }
    }

    #[test]
    fn fill_and_drop_previews_measure_the_completeness_change() {
        // Three columns, one complete: filling x's 2 missing cells moves completeness from 9/12
        // to 11/12, +16.67 points; the old pct_affected * 0.5 rule said 25.
        let mut wide = half_missing().df().clone();
        wide.with_column(Column::new("z".into(), [1.5, 2.5, 3.5, 4.5]))
            .unwrap();
        let p = preview_cleaning(
            &Dataset::new(wide),
            &CleaningOperation::FillNa {
                columns: Some(vec!["x".into()]),
                strategy: "median".into(),
                value: None,
            },
            5,
        );
        assert_eq!(p.cells_affected, 2);
        let want = 100.0 * 2.0 / 12.0;
        assert!(
            (p.estimated_quality_change - want).abs() < 1e-12,
            "{}",
            p.estimated_quality_change
        );
        // Dropping the rows with any missing cell keeps rows 1 and 4: 5/8 -> 1, +37.5 points.
        let p = preview_cleaning(
            &half_missing(),
            &CleaningOperation::DropNa {
                columns: None,
                how: "any".into(),
            },
            5,
        );
        assert_eq!(p.rows_affected, 2);
        assert!(
            (p.estimated_quality_change - 37.5).abs() < 1e-12,
            "{}",
            p.estimated_quality_change
        );
    }

    #[test]
    fn nan_constant_fill_preview_warns() {
        let p = preview_cleaning(
            &half_missing(),
            &CleaningOperation::FillNa {
                columns: Some(vec!["x".into()]),
                strategy: "constant".into(),
                value: Some("NaN".into()),
            },
            5,
        );
        assert_eq!(p.rows_affected, 0);
        assert!(
            p.warnings
                .iter()
                .any(|w| w.contains("NaN is a missing value")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn drop_columns_preview_reports_completeness() {
        let p = preview_cleaning(
            &half_missing(),
            &CleaningOperation::DropColumns {
                columns: vec!["x".into()],
            },
            5,
        );
        assert_eq!((p.rows_affected, p.cells_affected), (0, 0));
        // 5/8 -> 3/4
        assert!((p.estimated_quality_change - 12.5).abs() < 1e-12);
        let p = preview_cleaning(
            &half_missing(),
            &CleaningOperation::DropColumns {
                columns: vec!["nope".into()],
            },
            5,
        );
        assert!(
            p.warnings.iter().any(|w| w.contains("'nope'")),
            "{:?}",
            p.warnings
        );
    }

    #[test]
    fn first_changes_and_changed_mask_agree() {
        let cat = |v: &[&str]| {
            Column::new("c".into(), v)
                .cast(&DataType::from_categories(Categories::global()))
                .unwrap()
        };
        let pairs = [
            (
                Column::new("s".into(), ["a", "b", "c", "d"]),
                Column::new("s".into(), ["a", "B", "c", "D"]),
            ),
            (
                Column::new("n".into(), [1.0, f64::NAN, 3.0, 4.5]),
                Column::new("n".into(), [1.0, f64::NAN, 3.5, 4.0]),
            ),
            (cat(&["u", "v", "w", "u"]), cat(&["u", "w", "w", "v"])),
            (
                Column::new("i".into(), [1i64, 2, 3, 4]),
                Column::new("i".into(), [1.0, 2.5, 3.0, 4.0]),
            ),
        ];
        for (before, after) in &pairs {
            let mask = changed_mask(before, after).unwrap();
            let expected: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter(|(_, c)| **c)
                .map(|(i, _)| i)
                .collect();
            assert!(!expected.is_empty());
            for limit in 0..=4 {
                let first = first_changes(before, after, limit).unwrap();
                assert_eq!(
                    first,
                    expected.iter().copied().take(limit).collect::<Vec<_>>()
                );
            }
        }
        // NaN to NaN is no change.
        assert_eq!(
            changed_mask(&pairs[1].0, &pairs[1].1).unwrap(),
            [false, false, true, true]
        );
    }

    #[test]
    fn verify_reports_a_count_change_once() {
        // Pre-fix: one issue resolved ("30.0% missing") and one introduced ("7.5% missing").
        let n = 40;
        let x = |missing: usize| -> Vec<Option<f64>> {
            (0..n)
                .map(|i| (i >= missing).then_some(i as f64 * 1.37 + 0.2))
                .collect()
        };
        // `id` keeps the rows with a missing x apart (no duplicate-row issue).
        let id: Vec<i64> = (0..n as i64).collect();
        let before = Dataset::new(df! { "id" => id.clone(), "x" => x(12) }.unwrap());
        let after = Dataset::new(df! { "id" => id, "x" => x(3) }.unwrap());
        let report = verify_cleaning(&before, &after, "fill");
        let d = &report.quality_delta;
        assert!(d.issues_resolved.is_empty(), "{:?}", d.issues_resolved);
        assert!(d.issues_introduced.is_empty(), "{:?}", d.issues_introduced);
        assert_eq!(d.issues_changed, ["missing values in 'x': 12 -> 3"]);
        assert!(report.summary().contains("Changed issues (1):"));
    }

    #[test]
    fn verify_counts_modified_rows() {
        // Pre-fix rows_modified counted the 3 cells.
        let before =
            Dataset::new(df! { "a" => ["p", "q", "r", "s"], "b" => [1.5, 2.5, 3.5, 4.5] }.unwrap());
        let after =
            Dataset::new(df! { "a" => ["P", "q", "R", "s"], "b" => [1.5, 2.5, 9.5, 4.5] }.unwrap());
        let report = verify_cleaning(&before, &after, "edit");
        assert_eq!(report.rows_modified, 2);
        assert_eq!(report.cells_modified, 3);
    }

    #[test]
    fn test_preview_trim() {
        let test_df = df! {
            "name" => ["  Alice  ", "Bob", " Charlie "],
            "value" => [1, 2, 3],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let preview = preview_cleaning(
            &dataset,
            &CleaningOperation::Trim {
                columns: Some(vec!["name".to_string()]),
            },
            5,
        );

        assert_eq!(preview.rows_affected, 2); // Alice and Charlie have whitespace
        assert!(!preview.sample_changes.is_empty());
    }

    #[test]
    fn test_preview_to_lowercase() {
        let test_df = df! {
            "name" => ["ALICE", "bob", "CHARLIE"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let preview = preview_cleaning(
            &dataset,
            &CleaningOperation::ToLowercase {
                column: "name".to_string(),
            },
            5,
        );

        assert_eq!(preview.rows_affected, 2); // ALICE and CHARLIE will change, bob is already lowercase
    }

    #[test]
    fn test_preview_fill_na() {
        let test_df = df! {
            "value" => [Some(1), None, Some(3), None, Some(5)],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let preview = preview_cleaning(
            &dataset,
            &CleaningOperation::FillNa {
                columns: Some(vec!["value".to_string()]),
                strategy: "constant".to_string(),
                value: Some("0".to_string()),
            },
            5,
        );

        assert_eq!(preview.rows_affected, 2); // Two null values
    }

    #[test]
    fn test_preview_drop_na() {
        let test_df = df! {
            "a" => [Some(1), None, Some(3)],
            "b" => [Some(10), Some(20), None],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let preview = preview_cleaning(
            &dataset,
            &CleaningOperation::DropNa {
                columns: None,
                how: "any".to_string(),
            },
            5,
        );

        assert_eq!(preview.rows_affected, 2); // Two rows have nulls
    }

    #[test]
    fn test_preview_deduplicate() {
        let test_df = df! {
            "id" => [1, 1, 2, 2, 3],
            "name" => ["A", "A", "B", "B", "C"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let preview = preview_cleaning(
            &dataset,
            &CleaningOperation::Deduplicate {
                columns: None,
                keep: "first".to_string(),
            },
            5,
        );

        assert_eq!(preview.rows_affected, 2); // Two duplicate rows
    }

    #[test]
    fn test_preview_replace() {
        let test_df = df! {
            "status" => ["active", "inactive", "active", "pending"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let preview = preview_cleaning(
            &dataset,
            &CleaningOperation::Replace {
                column: "status".to_string(),
                old_value: "active".to_string(),
                new_value: "enabled".to_string(),
            },
            5,
        );

        assert_eq!(preview.rows_affected, 2); // Two "active" values
    }

    #[test]
    fn test_verify_cleaning() {
        let before_df = df! {
            "name" => ["  Alice  ", "Bob"],
            "value" => [1, 2],
        }
        .unwrap();

        let after_df = df! {
            "name" => ["Alice", "Bob"],
            "value" => [1, 2],
        }
        .unwrap();

        let before = Dataset::new(before_df);
        let after = Dataset::new(after_df);

        let report = verify_cleaning(&before, &after, "trim");

        assert_eq!(report.rows_before, 2);
        assert_eq!(report.rows_after, 2);
        assert_eq!(report.rows_removed, 0);
    }

    #[test]
    fn test_cleaning_preview_summary() {
        let preview = CleaningPreview {
            operation: "Trim whitespace".to_string(),
            rows_affected: 10,
            pct_affected: 5.0,
            cells_affected: 12,
            sample_changes: vec![ChangeExample {
                row_index: 0,
                column: "name".to_string(),
                before: "' Alice '".to_string(),
                after: "'Alice'".to_string(),
            }],
            estimated_quality_change: 0.5,
            warnings: vec![],
        };

        let summary = preview.summary();
        assert!(summary.contains("Trim whitespace"));
        assert!(summary.contains("10"));
        assert!(summary.contains("Alice"));
    }

    #[test]
    fn test_verify_cleaning_int_to_float_fill() {
        // DATA-23: a mean fill turns Int64 into Float64; verify_cleaning reports the type change and
        // compares the cells by value across the two numeric types.
        let before =
            Dataset::new(df!("x" => [Some(1i64), None, Some(3)], "y" => ["a", "b", "c"]).unwrap());
        let after =
            crate::data::munging::fill_na(&before, None, crate::data::munging::FillStrategy::Mean)
                .unwrap();
        assert_eq!(after.df().column("x").unwrap().dtype(), &DataType::Float64);
        let report = verify_cleaning(&before, &after, "fill_na mean");
        assert_eq!(report.rows_modified, 1); // only the filled cell (1 == 1.0, as R's ==)
        assert_eq!(
            report.dtype_changes,
            [DtypeChange {
                column: "x".into(),
                before: "i64".into(),
                after: "f64".into()
            }]
        );
        assert_eq!(
            (
                report.sample_changes[0].row_index,
                report.sample_changes[0].column.as_str()
            ),
            (1, "x")
        );
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w == "Column 'x' changed type from i64 to f64"),
            "{:?}",
            report.warnings
        );
        assert!(
            report.summary().contains("Type changes:\n  x: i64 → f64\n"),
            "{}",
            report.summary()
        );
    }

    #[test]
    fn test_verify_cleaning_cast_text_to_int() {
        // String vs Int64 cells reached the same unimplemented! arm
        let before = Dataset::new(df!("v" => ["1", "N/A", "3"]).unwrap());
        let after = crate::data::munging::cast(&before, "v", "int").unwrap();
        let report = verify_cleaning(&before, &after, "cast");
        assert_eq!(report.rows_modified, 1); // "1" == 1 and "3" == 3 as text; "N/A" -> missing
        assert_eq!(report.dtype_changes.len(), 1);
        assert_eq!(report.sample_changes[0].row_index, 1);
    }

    #[test]
    fn test_preview_fill_na_shows_real_fill_values() {
        // LLM-12 (core): the preview showed "<strategy value>" for every strategy and ignored the strategy
        let ds = Dataset::new(df!("value" => [Some(1i64), None, Some(3), None, Some(5)]).unwrap());
        let after = |strategy: &str, value: Option<&str>| -> Vec<String> {
            let op = CleaningOperation::FillNa {
                columns: Some(vec!["value".into()]),
                strategy: strategy.into(),
                value: value.map(str::to_string),
            };
            preview_cleaning(&ds, &op, 5)
                .sample_changes
                .iter()
                .map(|c| c.after.clone())
                .collect()
        };
        assert_eq!(after("mean", None), ["3", "3"]); // mean(c(1, 3, 5)) = 3
        assert_eq!(after("Median", None), ["3", "3"]);
        assert_eq!(after("mode", None), ["1", "1"]); // all tie; R's which.max(table()) takes level 1
        assert_eq!(after("forward", None), ["1", "3"]);
        assert_eq!(after("bfill", None), ["3", "5"]);
        assert_eq!(after("constant", Some("7")), ["7", "7"]);
        let bad = preview_cleaning(
            &ds,
            &CleaningOperation::FillNa {
                columns: None,
                strategy: "interpolate".into(),
                value: None,
            },
            5,
        );
        assert_eq!((bad.rows_affected, bad.sample_changes.len()), (0, 0));
        assert!(
            bad.warnings
                .iter()
                .any(|w| w.contains("unknown fill strategy 'interpolate'")),
            "{:?}",
            bad.warnings
        );
    }

    #[test]
    fn test_preview_replace_counts_typed_matches() {
        // DATA-38: a numeric column previewed 0 matches with "not a string column", although apply replaced them
        let ds = Dataset::new(df!("v" => [-99i64, 1, -99, 3]).unwrap());
        let op = CleaningOperation::Replace {
            column: "v".into(),
            old_value: "-99".into(),
            new_value: "null".into(),
        };
        let p = preview_cleaning(&ds, &op, 5);
        assert_eq!(p.rows_affected, 2);
        let s: Vec<(usize, &str, &str)> = p
            .sample_changes
            .iter()
            .map(|c| (c.row_index, c.before.as_str(), c.after.as_str()))
            .collect();
        assert_eq!(s, [(0, "-99", "null"), (2, "-99", "null")]);
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn test_preview_replace_samples_the_first_changes() {
        let ds = Dataset::new(df!("v" => [1i64, -99, 2, -99, -99, 3]).unwrap());
        let op = CleaningOperation::Replace {
            column: "v".into(),
            old_value: "-99".into(),
            new_value: "0".into(),
        };
        let p = preview_cleaning(&ds, &op, 2);
        assert_eq!(p.rows_affected, 3);
        let rows: Vec<usize> = p.sample_changes.iter().map(|c| c.row_index).collect();
        assert_eq!(rows, [1, 3]);

        // a label a categorical column did not hold goes through the type-changing comparison
        let labels = Column::new("c".into(), ["a", "b", "a"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let ds = Dataset::new(DataFrame::new(vec![labels]).unwrap());
        let op = CleaningOperation::Replace {
            column: "c".into(),
            old_value: "a".into(),
            new_value: "z".into(),
        };
        let p = preview_cleaning(&ds, &op, 1);
        assert_eq!(p.rows_affected, 2);
        assert_eq!(p.sample_changes.len(), 1);
        assert_eq!(p.sample_changes[0].row_index, 0);
        assert_eq!(p.sample_changes[0].after, "z");
    }

    #[test]
    fn test_preview_with_nothing_to_change_says_so() {
        let ds = Dataset::new(df!("v" => [1i64, 2, 3], "s" => ["a", "b", "c"]).unwrap());
        let p = preview_cleaning(
            &ds,
            &CleaningOperation::FillNa {
                columns: None,
                strategy: "forward".into(),
                value: None,
            },
            5,
        );
        assert_eq!(p.rows_affected, 0);
        assert!(p.sample_changes.is_empty());
        assert_eq!(p.warnings, ["No null values found in specified columns"]);
        let p = preview_cleaning(
            &ds,
            &CleaningOperation::Replace {
                column: "s".into(),
                old_value: "absent".into(),
                new_value: "x".into(),
            },
            5,
        );
        assert_eq!(p.rows_affected, 0);
        assert!(p.sample_changes.is_empty());
        assert_eq!(
            p.warnings,
            ["No values matching 'absent' found in column 's'"]
        );
    }
}
