//! Data quality profiling for LLM-assisted data cleaning.
//!
//! This module provides comprehensive data quality analysis including:
//! - Column-level statistics (nulls, uniques, types)
//! - Numeric column analysis (outliers, bounds, distribution)
//! - String column analysis (patterns, whitespace, encoding)
//! - Automated issue detection

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::Dataset;

/// Threshold for flagging high null rate (default: 5%)
const HIGH_NULL_THRESHOLD: f64 = 0.05;

/// Threshold for flagging potential duplicates (based on unique ratio)
const LOW_UNIQUE_THRESHOLD: f64 = 0.01;

/// IQR multiplier for outlier detection
const OUTLIER_IQR_MULTIPLIER: f64 = 1.5;

/// Comprehensive data quality profile for a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityProfile {
    /// Per-column profiles
    pub columns: Vec<ColumnProfile>,
    /// Total number of rows
    pub row_count: usize,
    /// Number of completely duplicate rows
    pub duplicate_rows: usize,
    /// Overall completeness score (% non-null cells)
    pub completeness_score: f64,
    /// Dataset-level issues detected
    pub dataset_issues: Vec<DataIssue>,
}

/// Profile for a single column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnProfile {
    /// Column name
    pub name: String,
    /// Data type as string
    pub dtype: String,
    /// Number of null/missing values
    pub null_count: usize,
    /// Percentage of null values (0.0 to 1.0)
    pub null_pct: f64,
    /// Number of unique values
    pub unique_count: usize,
    /// Percentage of unique values (0.0 to 1.0)
    pub unique_pct: f64,
    /// Statistics for numeric columns
    pub numeric_stats: Option<NumericStats>,
    /// Statistics for string columns
    pub string_stats: Option<StringStats>,
    /// Issues detected in this column
    pub issues: Vec<DataIssue>,
}

/// Statistics for numeric columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericStats {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std: f64,
    /// 25th percentile (Q1)
    pub q1: f64,
    /// 75th percentile (Q3)
    pub q3: f64,
    /// Number of outliers (outside 1.5*IQR)
    pub outlier_count: usize,
    /// Lower bound for non-outliers
    pub outlier_lower_bound: f64,
    /// Upper bound for non-outliers
    pub outlier_upper_bound: f64,
    /// Number of zero values
    pub zero_count: usize,
    /// Number of negative values
    pub negative_count: usize,
}

/// Statistics for string columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringStats {
    /// Minimum string length
    pub min_length: usize,
    /// Maximum string length
    pub max_length: usize,
    /// Mean string length
    pub mean_length: f64,
    /// Number of empty strings
    pub empty_count: usize,
    /// Number of strings with leading/trailing whitespace
    pub whitespace_issues_count: usize,
    /// Most common values (up to 10)
    pub top_values: Vec<(String, usize)>,
    /// Detected patterns (e.g., email, date formats)
    pub detected_patterns: Vec<String>,
}

/// Types of data quality issues that can be detected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataIssue {
    /// Column has high percentage of null values
    HighNullRate { column: String, pct: f64 },
    /// Dataset has duplicate rows
    DuplicateRows { count: usize },
    /// Potential duplicate rows based on subset of columns
    PossibleDuplicates { columns: Vec<String>, count: usize },
    /// Column appears to have mixed types (detected via string patterns)
    MixedTypes {
        column: String,
        examples: Vec<String>,
    },
    /// Numeric column has outlier values
    OutlierValues {
        column: String,
        count: usize,
        lower_bound: f64,
        upper_bound: f64,
    },
    /// String column has inconsistent formats
    InconsistentFormat {
        column: String,
        patterns: Vec<String>,
    },
    /// String column has whitespace issues
    WhitespaceIssues { column: String, count: usize },
    /// Constant column (all same value)
    ConstantColumn { column: String, value: String },
    /// Column has very low cardinality relative to row count
    LowCardinality {
        column: String,
        unique_count: usize,
        unique_pct: f64,
    },
    /// Column has very high cardinality (possibly an ID)
    HighCardinality { column: String, unique_pct: f64 },
    /// Empty strings detected
    EmptyStrings { column: String, count: usize },
}

impl DataIssue {
    /// Get a human-readable description of the issue.
    pub fn description(&self) -> String {
        match self {
            DataIssue::HighNullRate { column, pct } => {
                format!("Column '{}' has {:.1}% null values", column, pct * 100.0)
            }
            DataIssue::DuplicateRows { count } => {
                format!("Dataset has {} duplicate rows", count)
            }
            DataIssue::PossibleDuplicates { columns, count } => {
                format!(
                    "{} possible duplicate rows based on columns: {}",
                    count,
                    columns.join(", ")
                )
            }
            DataIssue::MixedTypes { column, examples } => {
                format!(
                    "Column '{}' appears to have mixed types: {}",
                    column,
                    examples.join(", ")
                )
            }
            DataIssue::OutlierValues {
                column,
                count,
                lower_bound,
                upper_bound,
            } => {
                format!(
                    "Column '{}' has {} outliers (expected range: {:.2} to {:.2})",
                    column, count, lower_bound, upper_bound
                )
            }
            DataIssue::InconsistentFormat { column, patterns } => {
                format!(
                    "Column '{}' has inconsistent formats: {}",
                    column,
                    patterns.join(", ")
                )
            }
            DataIssue::WhitespaceIssues { column, count } => {
                format!(
                    "Column '{}' has {} values with leading/trailing whitespace",
                    column, count
                )
            }
            DataIssue::ConstantColumn { column, value } => {
                format!("Column '{}' is constant (all values = '{}')", column, value)
            }
            DataIssue::LowCardinality {
                column,
                unique_count,
                unique_pct,
            } => {
                format!(
                    "Column '{}' has low cardinality: {} unique values ({:.1}%)",
                    column,
                    unique_count,
                    unique_pct * 100.0
                )
            }
            DataIssue::HighCardinality { column, unique_pct } => {
                format!(
                    "Column '{}' has very high cardinality ({:.1}% unique) - possibly an ID column",
                    column,
                    unique_pct * 100.0
                )
            }
            DataIssue::EmptyStrings { column, count } => {
                format!("Column '{}' has {} empty strings", column, count)
            }
        }
    }

    /// Get the severity of the issue (1-3, higher = more severe).
    pub fn severity(&self) -> u8 {
        match self {
            DataIssue::HighNullRate { pct, .. } => {
                if *pct > 0.5 {
                    3
                } else if *pct > 0.2 {
                    2
                } else {
                    1
                }
            }
            DataIssue::DuplicateRows { .. } => 2,
            DataIssue::PossibleDuplicates { .. } => 1,
            DataIssue::MixedTypes { .. } => 3,
            DataIssue::OutlierValues { count, .. } => {
                if *count > 100 {
                    2
                } else {
                    1
                }
            }
            DataIssue::InconsistentFormat { .. } => 2,
            DataIssue::WhitespaceIssues { .. } => 1,
            DataIssue::ConstantColumn { .. } => 1,
            DataIssue::LowCardinality { .. } => 1,
            DataIssue::HighCardinality { .. } => 1,
            DataIssue::EmptyStrings { .. } => 1,
        }
    }
}

/// Generate a comprehensive data quality profile for a dataset.
pub fn generate_quality_profile(dataset: &Dataset) -> DataQualityProfile {
    let df = dataset.df();
    let row_count = df.height();
    let col_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Calculate duplicate rows
    let duplicate_rows = calculate_duplicate_rows(df);

    // Profile each column
    let mut columns = Vec::new();
    let mut total_null_count = 0usize;
    let total_cells = row_count * col_names.len();

    for col_name in &col_names {
        let col = df.column(col_name).unwrap();
        let profile = profile_column(col, row_count);
        total_null_count += profile.null_count;
        columns.push(profile);
    }

    // Calculate completeness score
    let completeness_score = if total_cells > 0 {
        1.0 - (total_null_count as f64 / total_cells as f64)
    } else {
        1.0
    };

    // Collect dataset-level issues
    let mut dataset_issues = Vec::new();
    if duplicate_rows > 0 {
        dataset_issues.push(DataIssue::DuplicateRows {
            count: duplicate_rows,
        });
    }

    DataQualityProfile {
        columns,
        row_count,
        duplicate_rows,
        completeness_score,
        dataset_issues,
    }
}

/// Profile a single column.
fn profile_column(col: &Column, row_count: usize) -> ColumnProfile {
    let name = col.name().to_string();
    let dtype = format!("{:?}", col.dtype());
    let null_count = col.null_count();
    let null_pct = if row_count > 0 {
        null_count as f64 / row_count as f64
    } else {
        0.0
    };

    // Calculate unique count
    let unique_count = col.n_unique().unwrap_or(0);
    let unique_pct = if row_count > 0 {
        unique_count as f64 / row_count as f64
    } else {
        0.0
    };

    let mut issues = Vec::new();

    // Check for high null rate
    if null_pct > HIGH_NULL_THRESHOLD {
        issues.push(DataIssue::HighNullRate {
            column: name.clone(),
            pct: null_pct,
        });
    }

    // Check for constant column
    if unique_count == 1 && row_count > 1 {
        let value = get_first_non_null_value(col);
        issues.push(DataIssue::ConstantColumn {
            column: name.clone(),
            value,
        });
    }

    // Check for high cardinality (potential ID column)
    if unique_pct > 0.95 && row_count > 100 {
        issues.push(DataIssue::HighCardinality {
            column: name.clone(),
            unique_pct,
        });
    }

    // Check for low cardinality
    if unique_pct < LOW_UNIQUE_THRESHOLD && unique_count > 1 && row_count > 100 {
        issues.push(DataIssue::LowCardinality {
            column: name.clone(),
            unique_count,
            unique_pct,
        });
    }

    // Type-specific statistics
    let (numeric_stats, string_stats) = match col.dtype() {
        DataType::Float64
        | DataType::Float32
        | DataType::Int64
        | DataType::Int32
        | DataType::Int16
        | DataType::Int8 => {
            let stats = compute_numeric_stats(col, &name);
            if let Some(ref s) = stats {
                if s.outlier_count > 0 {
                    issues.push(DataIssue::OutlierValues {
                        column: name.clone(),
                        count: s.outlier_count,
                        lower_bound: s.outlier_lower_bound,
                        upper_bound: s.outlier_upper_bound,
                    });
                }
            }
            (stats, None)
        }
        DataType::String => {
            let stats = compute_string_stats(col, &name);
            if let Some(ref s) = stats {
                if s.whitespace_issues_count > 0 {
                    issues.push(DataIssue::WhitespaceIssues {
                        column: name.clone(),
                        count: s.whitespace_issues_count,
                    });
                }
                if s.empty_count > 0 {
                    issues.push(DataIssue::EmptyStrings {
                        column: name.clone(),
                        count: s.empty_count,
                    });
                }
                // Check for inconsistent formats
                if s.detected_patterns.len() > 1 {
                    issues.push(DataIssue::InconsistentFormat {
                        column: name.clone(),
                        patterns: s.detected_patterns.clone(),
                    });
                }
            }
            (None, stats)
        }
        _ => (None, None),
    };

    ColumnProfile {
        name,
        dtype,
        null_count,
        null_pct,
        unique_count,
        unique_pct,
        numeric_stats,
        string_stats,
        issues,
    }
}

/// Compute statistics for a numeric column.
fn compute_numeric_stats(col: &Column, _name: &str) -> Option<NumericStats> {
    let float_col = col.cast(&DataType::Float64).ok()?;
    let ca = float_col.f64().ok()?;

    // Basic statistics
    let min = ca.min()?;
    let max = ca.max()?;
    let mean = ca.mean()?;
    let std = ca.std(1).unwrap_or(0.0); // ddof=1 for sample std

    // Quantiles for outlier detection
    let sorted = ca.sort(false);
    let n = sorted.len();
    if n == 0 {
        return None;
    }

    let q1_idx = n / 4;
    let median_idx = n / 2;
    let q3_idx = (3 * n) / 4;

    let q1 = sorted.get(q1_idx).unwrap_or(min);
    let median = sorted.get(median_idx).unwrap_or(mean);
    let q3 = sorted.get(q3_idx).unwrap_or(max);

    // IQR-based outlier bounds
    let iqr = q3 - q1;
    let outlier_lower_bound = q1 - OUTLIER_IQR_MULTIPLIER * iqr;
    let outlier_upper_bound = q3 + OUTLIER_IQR_MULTIPLIER * iqr;

    // Count outliers, zeros, negatives
    let mut outlier_count = 0;
    let mut zero_count = 0;
    let mut negative_count = 0;

    for val in ca.into_iter().flatten() {
        if val < outlier_lower_bound || val > outlier_upper_bound {
            outlier_count += 1;
        }
        if val == 0.0 {
            zero_count += 1;
        }
        if val < 0.0 {
            negative_count += 1;
        }
    }

    Some(NumericStats {
        min,
        max,
        mean,
        median,
        std,
        q1,
        q3,
        outlier_count,
        outlier_lower_bound,
        outlier_upper_bound,
        zero_count,
        negative_count,
    })
}

/// Compute statistics for a string column.
fn compute_string_stats(col: &Column, _name: &str) -> Option<StringStats> {
    let ca = col.str().ok()?;

    let mut min_length = usize::MAX;
    let mut max_length = 0;
    let mut total_length = 0usize;
    let mut empty_count = 0;
    let mut whitespace_issues_count = 0;
    let mut value_counts: HashMap<String, usize> = HashMap::new();
    let mut non_null_count = 0;

    // Pattern detection
    let mut has_email_pattern = false;
    let mut has_date_pattern = false;
    let mut has_numeric_pattern = false;
    let mut has_phone_pattern = false;

    for val in ca.into_iter().flatten() {
        non_null_count += 1;
        let len = val.chars().count();

        if len < min_length {
            min_length = len;
        }
        if len > max_length {
            max_length = len;
        }
        total_length += len;

        if val.is_empty() {
            empty_count += 1;
        }

        // Check for whitespace issues
        let trimmed = val.trim();
        if trimmed != val {
            whitespace_issues_count += 1;
        }

        // Count values for top values
        *value_counts.entry(val.to_string()).or_insert(0) += 1;

        // Pattern detection (simple heuristics)
        if !has_email_pattern && val.contains('@') && val.contains('.') {
            has_email_pattern = true;
        }
        if !has_date_pattern && detect_date_pattern(val) {
            has_date_pattern = true;
        }
        if !has_numeric_pattern
            && val
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
            && !val.is_empty()
        {
            has_numeric_pattern = true;
        }
        if !has_phone_pattern && detect_phone_pattern(val) {
            has_phone_pattern = true;
        }
    }

    if min_length == usize::MAX {
        min_length = 0;
    }

    let mean_length = if non_null_count > 0 {
        total_length as f64 / non_null_count as f64
    } else {
        0.0
    };

    // Get top 10 values
    let mut top_values: Vec<(String, usize)> = value_counts.into_iter().collect();
    top_values.sort_by(|a, b| b.1.cmp(&a.1));
    top_values.truncate(10);

    // Collect detected patterns
    let mut detected_patterns = Vec::new();
    if has_email_pattern {
        detected_patterns.push("email".to_string());
    }
    if has_date_pattern {
        detected_patterns.push("date".to_string());
    }
    if has_numeric_pattern {
        detected_patterns.push("numeric_string".to_string());
    }
    if has_phone_pattern {
        detected_patterns.push("phone".to_string());
    }

    Some(StringStats {
        min_length,
        max_length,
        mean_length,
        empty_count,
        whitespace_issues_count,
        top_values,
        detected_patterns,
    })
}

/// Detect if a string matches a date pattern.
fn detect_date_pattern(s: &str) -> bool {
    // Common date patterns
    let patterns = [
        r"^\d{4}-\d{2}-\d{2}$", // YYYY-MM-DD
        r"^\d{2}/\d{2}/\d{4}$", // MM/DD/YYYY or DD/MM/YYYY
        r"^\d{2}-\d{2}-\d{4}$", // MM-DD-YYYY or DD-MM-YYYY
        r"^\d{4}/\d{2}/\d{2}$", // YYYY/MM/DD
    ];

    for pattern in &patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if re.is_match(s) {
                return true;
            }
        }
    }
    false
}

/// Detect if a string matches a phone pattern.
fn detect_phone_pattern(s: &str) -> bool {
    // Simple phone pattern detection
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.len() >= 10
        && digits.len() <= 15
        && (s.contains('-') || s.contains('.') || s.contains(' ') || s.contains('('))
}

/// Calculate the number of duplicate rows in a DataFrame.
fn calculate_duplicate_rows(df: &DataFrame) -> usize {
    let total_rows = df.height();
    if total_rows == 0 {
        return 0;
    }

    // Use unique to count distinct rows
    match df.unique::<String, &str>(None, UniqueKeepStrategy::First, None) {
        Ok(unique_df) => total_rows - unique_df.height(),
        Err(_) => 0,
    }
}

/// Get the first non-null value from a column as a string.
fn get_first_non_null_value(col: &Column) -> String {
    match col.dtype() {
        DataType::String => {
            if let Ok(ca) = col.str() {
                if let Some(val) = ca.into_iter().flatten().next() {
                    return val.to_string();
                }
            }
        }
        DataType::Int64 => {
            if let Ok(ca) = col.i64() {
                if let Some(val) = ca.into_iter().flatten().next() {
                    return val.to_string();
                }
            }
        }
        DataType::Float64 => {
            if let Ok(ca) = col.f64() {
                if let Some(val) = ca.into_iter().flatten().next() {
                    return val.to_string();
                }
            }
        }
        _ => {}
    }
    "N/A".to_string()
}

impl DataQualityProfile {
    /// Get all issues across the entire dataset, sorted by severity.
    pub fn all_issues(&self) -> Vec<&DataIssue> {
        let mut issues: Vec<&DataIssue> = self.dataset_issues.iter().collect();
        for col in &self.columns {
            issues.extend(col.issues.iter());
        }
        issues.sort_by(|a, b| b.severity().cmp(&a.severity()));
        issues
    }

    /// Get a summary of the profile suitable for LLM consumption.
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Dataset Quality Profile\n\
             =======================\n\
             Rows: {}\n\
             Columns: {}\n\
             Duplicate rows: {}\n\
             Completeness: {:.1}%\n\n",
            self.row_count,
            self.columns.len(),
            self.duplicate_rows,
            self.completeness_score * 100.0
        );

        let issues = self.all_issues();
        if issues.is_empty() {
            summary.push_str("No data quality issues detected.\n");
        } else {
            summary.push_str(&format!("Issues Found ({}):\n", issues.len()));
            for (i, issue) in issues.iter().enumerate() {
                summary.push_str(&format!(
                    "  {}. [Severity {}] {}\n",
                    i + 1,
                    issue.severity(),
                    issue.description()
                ));
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
    fn test_generate_quality_profile_basic() {
        let test_df = df! {
            "id" => [1, 2, 3, 4, 5],
            "name" => ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score" => [85.5, 90.0, 78.5, 92.0, 88.0],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);

        assert_eq!(profile.row_count, 5);
        assert_eq!(profile.columns.len(), 3);
        assert_eq!(profile.duplicate_rows, 0);
        assert!((profile.completeness_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_detect_high_null_rate() {
        let test_df = df! {
            "id" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "value" => [Some(1.0), Some(2.0), None, None, None, None, Some(7.0), None, None, None],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);

        let value_profile = profile.columns.iter().find(|c| c.name == "value").unwrap();
        assert!(value_profile.null_pct > 0.5);
        assert!(
            value_profile
                .issues
                .iter()
                .any(|i| matches!(i, DataIssue::HighNullRate { .. }))
        );
    }

    #[test]
    fn test_detect_whitespace_issues() {
        let test_df = df! {
            "email" => ["  alice@test.com", "bob@test.com  ", " charlie@test.com ", "diana@test.com"],
        }.unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);

        let email_profile = profile.columns.iter().find(|c| c.name == "email").unwrap();
        assert!(
            email_profile
                .issues
                .iter()
                .any(|i| matches!(i, DataIssue::WhitespaceIssues { count: 3, .. }))
        );
    }

    #[test]
    fn test_detect_outliers() {
        // Values tightly clustered around 10-15, with 1000 as clear outlier
        let test_df = df! {
            "value" => [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 1000.0],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);

        let value_profile = profile.columns.iter().find(|c| c.name == "value").unwrap();
        assert!(value_profile.numeric_stats.is_some());
        let stats = value_profile.numeric_stats.as_ref().unwrap();
        // 1000 should be detected as an outlier
        assert!(
            stats.outlier_count > 0,
            "Expected outliers but found: outlier_count={}, bounds=({}, {})",
            stats.outlier_count,
            stats.outlier_lower_bound,
            stats.outlier_upper_bound
        );
    }

    #[test]
    fn test_detect_duplicate_rows() {
        let test_df = df! {
            "id" => [1, 2, 2, 3, 3, 3],
            "name" => ["A", "B", "B", "C", "C", "C"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);

        assert_eq!(profile.duplicate_rows, 3); // 3 duplicate rows
    }

    #[test]
    fn test_constant_column_detection() {
        let test_df = df! {
            "id" => [1, 2, 3, 4, 5],
            "constant" => ["same", "same", "same", "same", "same"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);

        let const_profile = profile
            .columns
            .iter()
            .find(|c| c.name == "constant")
            .unwrap();
        assert!(
            const_profile
                .issues
                .iter()
                .any(|i| matches!(i, DataIssue::ConstantColumn { .. }))
        );
    }

    #[test]
    fn test_string_stats_patterns() {
        let test_df = df! {
            "email" => ["alice@test.com", "bob@example.org", "charlie@mail.net"],
            "date" => ["2024-01-15", "2024-02-20", "2024-03-25"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);

        let email_profile = profile.columns.iter().find(|c| c.name == "email").unwrap();
        let email_stats = email_profile.string_stats.as_ref().unwrap();
        assert!(email_stats.detected_patterns.contains(&"email".to_string()));

        let date_profile = profile.columns.iter().find(|c| c.name == "date").unwrap();
        let date_stats = date_profile.string_stats.as_ref().unwrap();
        assert!(date_stats.detected_patterns.contains(&"date".to_string()));
    }

    #[test]
    fn test_profile_summary() {
        let test_df = df! {
            "id" => [1, 2, 3],
            "value" => [Some(1.0), None, Some(3.0)],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let summary = profile.summary();

        assert!(summary.contains("Rows: 3"));
        assert!(summary.contains("Columns: 2"));
    }
}
