//! Verification and preview system for data cleaning operations.
//!
//! This module provides:
//! - Preview of cleaning operations before applying
//! - Verification reports comparing before/after states
//! - Change tracking with sample examples
//! - Quality delta computation

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use super::quality::generate_quality_profile;
use super::Dataset;

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
    /// Number of rows that were modified
    pub rows_modified: usize,
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
}

/// Preview of what a cleaning operation would do.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningPreview {
    /// Description of the operation
    pub operation: String,
    /// Number of rows that would be affected
    pub rows_affected: usize,
    /// Percentage of rows affected
    pub pct_affected: f64,
    /// Sample of changes that would be made
    pub sample_changes: Vec<ChangeExample>,
    /// Estimated quality improvement
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
    FillNa { columns: Option<Vec<String>>, strategy: String, value: Option<String> },
    /// Drop rows with null values
    DropNa { columns: Option<Vec<String>>, how: String },
    /// Remove duplicate rows
    Deduplicate { columns: Option<Vec<String>>, keep: String },
    /// Replace values
    Replace { column: String, old_value: String, new_value: String },
    /// Filter rows
    Filter { column: String, operator: String, value: String },
}

impl CleaningOperation {
    /// Get a human-readable description of the operation.
    pub fn description(&self) -> String {
        match self {
            CleaningOperation::Trim { columns } => {
                match columns {
                    Some(cols) => format!("Trim whitespace from columns: {}", cols.join(", ")),
                    None => "Trim whitespace from all string columns".to_string(),
                }
            }
            CleaningOperation::ToLowercase { column } => {
                format!("Convert '{}' to lowercase", column)
            }
            CleaningOperation::ToUppercase { column } => {
                format!("Convert '{}' to uppercase", column)
            }
            CleaningOperation::FillNa { columns, strategy, value } => {
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
            CleaningOperation::Replace { column, old_value, new_value } => {
                format!("Replace '{}' with '{}' in column '{}'", old_value, new_value, column)
            }
            CleaningOperation::Filter { column, operator, value } => {
                format!("Filter rows where {} {} {}", column, operator, value)
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
        CleaningOperation::FillNa { columns, strategy, value } => {
            preview_fill_na(df, columns.as_deref(), strategy, value.as_deref(), sample_size, row_count)
        }
        CleaningOperation::DropNa { columns, how } => {
            preview_drop_na(df, columns.as_deref(), how, sample_size, row_count)
        }
        CleaningOperation::Deduplicate { columns, keep } => {
            preview_deduplicate(df, columns.as_deref(), keep, row_count)
        }
        CleaningOperation::Replace { column, old_value, new_value } => {
            preview_replace(df, column, old_value, new_value, sample_size, row_count)
        }
        CleaningOperation::Filter { column, operator, value } => {
            preview_filter(df, column, operator, value, row_count)
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
    let mut total_affected = 0;
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
        if let Ok(col) = df.column(col_name) {
            if let Ok(ca) = col.str() {
                for (idx, val) in ca.into_iter().enumerate() {
                    if let Some(s) = val {
                        let trimmed = s.trim();
                        if trimmed != s {
                            total_affected += 1;
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
            }
        }
    }

    if cols_to_check.is_empty() {
        warnings.push("No string columns found to trim".to_string());
    }

    let pct_affected = if row_count > 0 {
        (total_affected as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    CleaningPreview {
        operation: CleaningOperation::Trim { columns: columns.map(|c| c.to_vec()) }.description(),
        rows_affected: total_affected,
        pct_affected,
        sample_changes,
        estimated_quality_change: pct_affected * 0.1, // Simple heuristic
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

    let pct_affected = if row_count > 0 {
        (total_affected as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    let operation = if to_lower {
        CleaningOperation::ToLowercase { column: column.to_string() }
    } else {
        CleaningOperation::ToUppercase { column: column.to_string() }
    };

    CleaningPreview {
        operation: operation.description(),
        rows_affected: total_affected,
        pct_affected,
        sample_changes,
        estimated_quality_change: 0.0, // Case changes don't affect quality score
        warnings,
    }
}

/// Preview fill_na operation.
fn preview_fill_na(
    df: &DataFrame,
    columns: Option<&[String]>,
    _strategy: &str,
    value: Option<&str>,
    sample_size: usize,
    row_count: usize,
) -> CleaningPreview {
    let mut sample_changes = Vec::new();
    let mut total_affected = 0;
    let mut warnings = Vec::new();

    let cols_to_check: Vec<String> = match columns {
        Some(cols) => cols.to_vec(),
        None => df.get_column_names().iter().map(|s| s.to_string()).collect(),
    };

    let fill_value = value.unwrap_or("<strategy value>");

    for col_name in &cols_to_check {
        if let Ok(col) = df.column(col_name) {
            let null_count = col.null_count();
            if null_count > 0 {
                total_affected += null_count;
                // Find sample null positions
                let mut samples_for_col = 0;
                for idx in 0..col.len() {
                    if col.get(idx).map(|v| v.is_null()).unwrap_or(false) {
                        if sample_changes.len() < sample_size && samples_for_col < 2 {
                            sample_changes.push(ChangeExample {
                                row_index: idx,
                                column: col_name.clone(),
                                before: "null".to_string(),
                                after: fill_value.to_string(),
                            });
                            samples_for_col += 1;
                        }
                    }
                }
            }
        }
    }

    if total_affected == 0 {
        warnings.push("No null values found in specified columns".to_string());
    }

    let pct_affected = if row_count > 0 {
        (total_affected as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    CleaningPreview {
        operation: CleaningOperation::FillNa {
            columns: columns.map(|c| c.to_vec()),
            strategy: _strategy.to_string(),
            value: value.map(|v| v.to_string()),
        }.description(),
        rows_affected: total_affected,
        pct_affected,
        sample_changes,
        estimated_quality_change: pct_affected * 0.5, // Filling nulls significantly improves completeness
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

    let cols_to_check: Vec<&str> = match columns {
        Some(cols) => cols.iter().map(|s| s.as_str()).collect(),
        None => df.get_column_names().into_iter().map(|s| s.as_str()).collect(),
    };

    // Count rows that would be dropped
    let mut rows_to_drop = 0;

    for row_idx in 0..df.height() {
        let mut has_null = false;
        let mut all_null = true;

        for col_name in &cols_to_check {
            if let Ok(col) = df.column(*col_name) {
                let is_null = col.get(row_idx).map(|v| v.is_null()).unwrap_or(false);
                if is_null {
                    has_null = true;
                } else {
                    all_null = false;
                }
            }
        }

        let should_drop = match how.to_lowercase().as_str() {
            "any" => has_null,
            "all" => all_null,
            _ => has_null,
        };

        if should_drop {
            rows_to_drop += 1;
        }
    }

    if rows_to_drop == row_count {
        warnings.push("Warning: This operation would drop ALL rows!".to_string());
    } else if rows_to_drop as f64 / row_count as f64 > 0.5 {
        warnings.push(format!(
            "Warning: This operation would drop more than 50% of rows ({} of {})",
            rows_to_drop, row_count
        ));
    }

    let pct_affected = if row_count > 0 {
        (rows_to_drop as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    CleaningPreview {
        operation: CleaningOperation::DropNa {
            columns: columns.map(|c| c.to_vec()),
            how: how.to_string(),
        }.description(),
        rows_affected: rows_to_drop,
        pct_affected,
        sample_changes: vec![], // No specific changes to show for row deletion
        estimated_quality_change: 10.0, // Dropping nulls improves completeness to 100%
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

    // Count duplicates
    let subset: Option<Vec<String>> = columns.map(|cols| cols.to_vec());

    let keep_strategy = match keep.to_lowercase().as_str() {
        "first" => UniqueKeepStrategy::First,
        "last" => UniqueKeepStrategy::Last,
        "none" => UniqueKeepStrategy::None,
        _ => UniqueKeepStrategy::First,
    };

    let unique_count = df
        .unique::<String, String>(subset.as_deref(), keep_strategy, None)
        .map(|u| u.height())
        .unwrap_or(row_count);

    let duplicates = row_count.saturating_sub(unique_count);

    if duplicates == 0 {
        warnings.push("No duplicate rows found".to_string());
    }

    let pct_affected = if row_count > 0 {
        (duplicates as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    CleaningPreview {
        operation: CleaningOperation::Deduplicate {
            columns: columns.map(|c| c.to_vec()),
            keep: keep.to_string(),
        }.description(),
        rows_affected: duplicates,
        pct_affected,
        sample_changes: vec![],
        estimated_quality_change: pct_affected * 0.3,
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
    let mut warnings = Vec::new();

    if let Ok(col) = df.column(column) {
        match col.dtype() {
            DataType::String => {
                if let Ok(ca) = col.str() {
                    for (idx, val) in ca.into_iter().enumerate() {
                        if let Some(s) = val {
                            if s == old_value {
                                total_affected += 1;
                                if sample_changes.len() < sample_size {
                                    sample_changes.push(ChangeExample {
                                        row_index: idx,
                                        column: column.to_string(),
                                        before: s.to_string(),
                                        after: new_value.to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // For numeric types, try to parse
                warnings.push(format!(
                    "Column '{}' is not a string column. Comparison will be exact.",
                    column
                ));
            }
        }
    } else {
        warnings.push(format!("Column '{}' not found", column));
    }

    if total_affected == 0 {
        warnings.push(format!("No values matching '{}' found in column '{}'", old_value, column));
    }

    let pct_affected = if row_count > 0 {
        (total_affected as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    CleaningPreview {
        operation: CleaningOperation::Replace {
            column: column.to_string(),
            old_value: old_value.to_string(),
            new_value: new_value.to_string(),
        }.description(),
        rows_affected: total_affected,
        pct_affected,
        sample_changes,
        estimated_quality_change: 0.0,
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

    if let Ok(col) = df.column(column) {
        // Try to build a filter mask
        let mask = match col.dtype() {
            DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32 => {
                if let Ok(val) = value.parse::<f64>() {
                    let float_col = col.cast(&DataType::Float64).ok();
                    float_col.and_then(|fc| {
                        let ca = fc.f64().ok()?;
                        Some(match operator {
                            ">" | "gt" => ca.gt(val),
                            ">=" | "gte" | "ge" => ca.gt_eq(val),
                            "<" | "lt" => ca.lt(val),
                            "<=" | "lte" | "le" => ca.lt_eq(val),
                            "==" | "eq" | "=" => ca.equal(val),
                            "!=" | "ne" | "<>" => ca.not_equal(val),
                            _ => return None,
                        })
                    })
                } else {
                    warnings.push(format!("Cannot parse '{}' as a number", value));
                    None
                }
            }
            DataType::String => {
                if let Ok(ca) = col.str() {
                    Some(match operator {
                        "==" | "eq" | "=" => ca.equal(value),
                        "!=" | "ne" | "<>" => ca.not_equal(value),
                        "contains" => ca.contains(value, false).unwrap_or_else(|_| BooleanChunked::new(PlSmallStr::from("mask"), &[] as &[bool])),
                        _ => {
                            warnings.push(format!("Operator '{}' not supported for string columns", operator));
                            return CleaningPreview {
                                operation: format!("Filter {} {} {}", column, operator, value),
                                rows_affected: 0,
                                pct_affected: 0.0,
                                sample_changes: vec![],
                                estimated_quality_change: 0.0,
                                warnings,
                            };
                        }
                    })
                } else {
                    None
                }
            }
            _ => {
                warnings.push(format!("Unsupported column type for filtering: {:?}", col.dtype()));
                None
            }
        };

        if let Some(m) = mask {
            // Count rows that DON'T match (will be removed)
            rows_to_remove = m.into_iter().filter(|v| !v.unwrap_or(false)).count();
        }
    } else {
        warnings.push(format!("Column '{}' not found", column));
    }

    let pct_affected = if row_count > 0 {
        (rows_to_remove as f64 / row_count as f64) * 100.0
    } else {
        0.0
    };

    if rows_to_remove == row_count {
        warnings.push("Warning: This filter would remove ALL rows!".to_string());
    }

    CleaningPreview {
        operation: CleaningOperation::Filter {
            column: column.to_string(),
            operator: operator.to_string(),
            value: value.to_string(),
        }.description(),
        rows_affected: rows_to_remove,
        pct_affected,
        sample_changes: vec![],
        estimated_quality_change: 0.0,
        warnings,
    }
}

/// Generate a verification report comparing before and after datasets.
pub fn verify_cleaning(
    before: &Dataset,
    after: &Dataset,
    _operation: &str,
) -> VerificationReport {
    let rows_before = before.nrows();
    let rows_after = after.nrows();

    // Generate quality profiles
    let profile_before = generate_quality_profile(before);
    let profile_after = generate_quality_profile(after);

    // Calculate quality delta
    let issues_before: Vec<String> = profile_before.all_issues()
        .iter()
        .map(|i| i.description())
        .collect();
    let issues_after: Vec<String> = profile_after.all_issues()
        .iter()
        .map(|i| i.description())
        .collect();

    let issues_resolved: Vec<String> = issues_before
        .iter()
        .filter(|i| !issues_after.contains(i))
        .cloned()
        .collect();
    let issues_introduced: Vec<String> = issues_after
        .iter()
        .filter(|i| !issues_before.contains(i))
        .cloned()
        .collect();

    let quality_delta = QualityDelta {
        completeness_before: profile_before.completeness_score,
        completeness_after: profile_after.completeness_score,
        completeness_change: profile_after.completeness_score - profile_before.completeness_score,
        issues_before: issues_before.len(),
        issues_after: issues_after.len(),
        issues_resolved,
        issues_introduced,
    };

    // Detect changes by comparing dataframes
    let (rows_modified, sample_changes) = detect_changes(before, after, 5);

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

    VerificationReport {
        rows_before,
        rows_after,
        rows_modified,
        rows_removed: rows_before.saturating_sub(rows_after),
        rows_added: rows_after.saturating_sub(rows_before),
        sample_changes,
        quality_delta,
        warnings,
    }
}

/// Detect changes between two datasets.
fn detect_changes(before: &Dataset, after: &Dataset, sample_size: usize) -> (usize, Vec<ChangeExample>) {
    let df_before = before.df();
    let df_after = after.df();

    let mut changes = Vec::new();
    let mut modified_count = 0;

    // Only compare if row counts are the same
    if df_before.height() != df_after.height() {
        return (0, changes);
    }

    let col_names: Vec<String> = df_before.get_column_names().iter().map(|s| s.to_string()).collect();

    for col_name in &col_names {
        let col_before = match df_before.column(col_name) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let col_after = match df_after.column(col_name) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Compare values
        for idx in 0..df_before.height() {
            let val_before = col_before.get(idx).ok();
            let val_after = col_after.get(idx).ok();

            if val_before != val_after {
                modified_count += 1;
                if changes.len() < sample_size {
                    changes.push(ChangeExample {
                        row_index: idx,
                        column: col_name.clone(),
                        before: format!("{:?}", val_before),
                        after: format!("{:?}", val_after),
                    });
                }
            }
        }
    }

    (modified_count, changes)
}

impl VerificationReport {
    /// Get a human-readable summary of the verification.
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("Verification Report\n");
        summary.push_str("===================\n\n");

        summary.push_str(&format!("Rows: {} → {} ", self.rows_before, self.rows_after));
        if self.rows_removed > 0 {
            summary.push_str(&format!("(-{} removed) ", self.rows_removed));
        }
        if self.rows_added > 0 {
            summary.push_str(&format!("(+{} added) ", self.rows_added));
        }
        if self.rows_modified > 0 {
            summary.push_str(&format!("({} modified)", self.rows_modified));
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
            self.quality_delta.issues_before,
            self.quality_delta.issues_after
        ));

        if !self.quality_delta.issues_resolved.is_empty() {
            summary.push_str(&format!(
                "\n✓ Issues Resolved ({}):\n",
                self.quality_delta.issues_resolved.len()
            ));
            for issue in &self.quality_delta.issues_resolved {
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

        if self.estimated_quality_change != 0.0 {
            summary.push_str(&format!(
                "Estimated quality change: {:+.1}%\n",
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

    #[test]
    fn test_preview_trim() {
        let test_df = df! {
            "name" => ["  Alice  ", "Bob", " Charlie "],
            "value" => [1, 2, 3],
        }.unwrap();

        let dataset = Dataset::new(test_df);
        let preview = preview_cleaning(
            &dataset,
            &CleaningOperation::Trim { columns: Some(vec!["name".to_string()]) },
            5,
        );

        assert_eq!(preview.rows_affected, 2); // Alice and Charlie have whitespace
        assert!(!preview.sample_changes.is_empty());
    }

    #[test]
    fn test_preview_to_lowercase() {
        let test_df = df! {
            "name" => ["ALICE", "bob", "CHARLIE"],
        }.unwrap();

        let dataset = Dataset::new(test_df);
        let preview = preview_cleaning(
            &dataset,
            &CleaningOperation::ToLowercase { column: "name".to_string() },
            5,
        );

        assert_eq!(preview.rows_affected, 2); // ALICE and CHARLIE will change, bob is already lowercase
    }

    #[test]
    fn test_preview_fill_na() {
        let test_df = df! {
            "value" => [Some(1), None, Some(3), None, Some(5)],
        }.unwrap();

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
        }.unwrap();

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
        }.unwrap();

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
        }.unwrap();

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
        }.unwrap();

        let after_df = df! {
            "name" => ["Alice", "Bob"],
            "value" => [1, 2],
        }.unwrap();

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
            sample_changes: vec![
                ChangeExample {
                    row_index: 0,
                    column: "name".to_string(),
                    before: "' Alice '".to_string(),
                    after: "'Alice'".to_string(),
                },
            ],
            estimated_quality_change: 0.5,
            warnings: vec![],
        };

        let summary = preview.summary();
        assert!(summary.contains("Trim whitespace"));
        assert!(summary.contains("10"));
        assert!(summary.contains("Alice"));
    }
}
