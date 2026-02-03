//! Data cleaning operations: drop_na, fill_na, deduplicate, cast, replace, regex.

use polars::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::error::{MungeError, MungeResult};
use crate::data::Dataset;

/// Strategy for handling missing values when filling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillStrategy {
    /// Fill with a constant value (parsed based on column type)
    Constant(String),
    /// Fill with column mean (numeric only)
    Mean,
    /// Fill with column median (numeric only)
    Median,
    /// Forward fill (use previous non-null value)
    Forward,
    /// Backward fill (use next non-null value)
    Backward,
    /// Fill with zero (numeric) or empty string (string)
    Zero,
}

/// Drop rows containing null values.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Columns to check for nulls (None = all columns)
/// * `how` - "any" (drop if any null) or "all" (drop only if all specified columns are null)
///
/// # Example
/// ```ignore
/// // Drop rows with any null
/// let clean = drop_na(&dataset, None, "any")?;
///
/// // Drop rows with null in specific columns
/// let clean = drop_na(&dataset, Some(&["age", "income"]), "any")?;
/// ```
pub fn drop_na(dataset: &Dataset, columns: Option<&[&str]>, how: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns if specified
    if let Some(cols) = columns {
        for col in cols {
            if df.column(col).is_err() {
                return Err(MungeError::ColumnNotFound(col.to_string()));
            }
        }
    }

    let cols_to_check: Vec<&str> = match columns {
        Some(cols) => cols.to_vec(),
        None => df
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect(),
    };

    // Build null mask
    let mut null_mask: Option<BooleanChunked> = None;

    for col_name in &cols_to_check {
        let col = df.column(col_name)?;
        let is_null = col.is_null();

        null_mask = Some(match null_mask {
            None => is_null,
            Some(existing) => match how.to_lowercase().as_str() {
                "any" => existing | is_null,
                "all" => existing & is_null,
                _ => {
                    return Err(MungeError::InvalidExpression(format!(
                        "Invalid 'how' parameter: {}. Use 'any' or 'all'",
                        how
                    )));
                }
            },
        });
    }

    // Invert mask to keep non-null rows
    let keep_mask = match null_mask {
        Some(mask) => !mask,
        None => return Ok(dataset.clone()),
    };

    let filtered = df.filter(&keep_mask)?;
    Ok(Dataset::new(filtered))
}

/// Fill null values in specified columns.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Columns to fill (None = all columns)
/// * `strategy` - How to fill null values
///
/// # Example
/// ```ignore
/// // Fill all numeric nulls with mean
/// let filled = fill_na(&dataset, None, FillStrategy::Mean)?;
///
/// // Fill specific column with constant
/// let filled = fill_na(&dataset, Some(&["age"]), FillStrategy::Constant("0".to_string()))?;
/// ```
pub fn fill_na(
    dataset: &Dataset,
    columns: Option<&[&str]>,
    strategy: FillStrategy,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let cols_to_fill: Vec<String> = match columns {
        Some(cols) => {
            for col in cols {
                if df.column(col).is_err() {
                    return Err(MungeError::ColumnNotFound(col.to_string()));
                }
            }
            cols.iter().map(|s| s.to_string()).collect()
        }
        None => df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };

    let mut result_df = df.clone();

    for col_name in &cols_to_fill {
        let col = result_df.column(col_name)?;
        let dtype = col.dtype();

        let filled_col = match &strategy {
            FillStrategy::Constant(value) => fill_with_constant(col, value)?,
            FillStrategy::Mean => {
                if !dtype.is_primitive_numeric() {
                    continue; // Skip non-numeric columns for mean
                }
                fill_with_mean(col)?
            }
            FillStrategy::Median => {
                if !dtype.is_primitive_numeric() {
                    continue;
                }
                fill_with_median(col)?
            }
            FillStrategy::Forward => fill_forward(col)?,
            FillStrategy::Backward => fill_backward(col)?,
            FillStrategy::Zero => fill_with_zero(col)?,
        };

        result_df = result_df.with_column(filled_col)?.clone();
    }

    Ok(Dataset::new(result_df))
}

fn fill_with_constant(col: &Column, value: &str) -> MungeResult<Column> {
    let dtype = col.dtype();
    let name = col.name().clone();

    match dtype {
        DataType::Int64 => {
            let val: i64 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: name.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as integer".to_string(),
            })?;
            let ca = col.i64()?.fill_null_with_values(val)?;
            Ok(ca.into_column())
        }
        DataType::Int32 => {
            let val: i32 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: name.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as integer".to_string(),
            })?;
            let ca = col.i32()?.fill_null_with_values(val)?;
            Ok(ca.into_column())
        }
        DataType::Float64 => {
            let val: f64 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: name.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as float".to_string(),
            })?;
            let ca = col.f64()?.fill_null_with_values(val)?;
            Ok(ca.into_column())
        }
        DataType::Float32 => {
            let val: f32 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: name.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as float".to_string(),
            })?;
            let ca = col.f32()?.fill_null_with_values(val)?;
            Ok(ca.into_column())
        }
        DataType::String => {
            let ca = col.str()?;
            let filled: StringChunked = ca
                .into_iter()
                .map(|opt| {
                    opt.map(|s| s.to_string())
                        .unwrap_or_else(|| value.to_string())
                })
                .collect();
            Ok(filled.with_name(name).into_column())
        }
        DataType::Boolean => {
            let val: bool = value.parse().map_err(|_| MungeError::InvalidValue {
                column: name.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as boolean".to_string(),
            })?;
            let ca = col.bool()?.fill_null_with_values(val)?;
            Ok(ca.into_column())
        }
        _ => {
            // For other types, try to fill as-is or skip
            Ok(col.clone())
        }
    }
}

fn fill_with_mean(col: &Column) -> MungeResult<Column> {
    let name = col.name().clone();
    let float_col = col.cast(&DataType::Float64)?;
    let ca = float_col.f64()?;

    let mean_val = ca.mean().unwrap_or(0.0);
    let filled = ca.fill_null_with_values(mean_val)?;
    Ok(filled.with_name(name).into_column())
}

fn fill_with_median(col: &Column) -> MungeResult<Column> {
    let name = col.name().clone();
    let float_col = col.cast(&DataType::Float64)?;
    let ca = float_col.f64()?;

    let median_val = ca.median().unwrap_or(0.0);
    let filled = ca.fill_null_with_values(median_val)?;
    Ok(filled.with_name(name).into_column())
}

fn fill_forward(col: &Column) -> MungeResult<Column> {
    let filled = col.fill_null(FillNullStrategy::Forward(None))?;
    Ok(filled)
}

fn fill_backward(col: &Column) -> MungeResult<Column> {
    let filled = col.fill_null(FillNullStrategy::Backward(None))?;
    Ok(filled)
}

fn fill_with_zero(col: &Column) -> MungeResult<Column> {
    let dtype = col.dtype();
    let name = col.name().clone();

    match dtype {
        DataType::Int64 => {
            let ca = col.i64()?.fill_null_with_values(0)?;
            Ok(ca.into_column())
        }
        DataType::Int32 => {
            let ca = col.i32()?.fill_null_with_values(0)?;
            Ok(ca.into_column())
        }
        DataType::Float64 => {
            let ca = col.f64()?.fill_null_with_values(0.0)?;
            Ok(ca.into_column())
        }
        DataType::Float32 => {
            let ca = col.f32()?.fill_null_with_values(0.0)?;
            Ok(ca.into_column())
        }
        DataType::String => {
            let ca = col.str()?;
            let filled: StringChunked = ca
                .into_iter()
                .map(|opt| opt.map(|s| s.to_string()).unwrap_or_default())
                .collect();
            Ok(filled.with_name(name).into_column())
        }
        _ => Ok(col.clone()),
    }
}

/// Remove duplicate rows.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `subset` - Columns to consider for duplicates (None = all columns)
/// * `keep` - Which duplicate to keep: "first", "last", or "none"
///
/// # Example
/// ```ignore
/// // Remove duplicates based on id column, keep first
/// let deduped = deduplicate(&dataset, Some(&["id"]), "first")?;
/// ```
pub fn deduplicate(dataset: &Dataset, subset: Option<&[&str]>, keep: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns if specified
    if let Some(cols) = subset {
        for col in cols {
            if df.column(col).is_err() {
                return Err(MungeError::ColumnNotFound(col.to_string()));
            }
        }
    }

    let keep_strategy = match keep.to_lowercase().as_str() {
        "first" => UniqueKeepStrategy::First,
        "last" => UniqueKeepStrategy::Last,
        "none" => UniqueKeepStrategy::None,
        "any" => UniqueKeepStrategy::Any,
        _ => {
            return Err(MungeError::InvalidExpression(format!(
                "Invalid 'keep' parameter: {}. Use 'first', 'last', 'none', or 'any'",
                keep
            )));
        }
    };

    let subset_cols: Option<Vec<String>> =
        subset.map(|cols| cols.iter().map(|&s| s.to_string()).collect());
    let unique_df = df.unique::<String, &str>(subset_cols.as_deref(), keep_strategy, None)?;

    Ok(Dataset::new(unique_df))
}

/// Cast a column to a different data type.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to cast
/// * `dtype` - Target type: "int", "float", "string", "bool"
///
/// # Example
/// ```ignore
/// let casted = cast(&dataset, "age", "float")?;
/// ```
pub fn cast(dataset: &Dataset, column: &str, dtype: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let target_dtype = parse_dtype(dtype)?;
    let col = df.column(column)?;
    let casted = col.cast(&target_dtype)?;

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(casted)?.clone();

    Ok(Dataset::new(result_df))
}

/// Cast multiple columns.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `casts` - Vec of (column, dtype) pairs
pub fn cast_columns(dataset: &Dataset, casts: &[(&str, &str)]) -> MungeResult<Dataset> {
    let mut result = dataset.clone();
    for (column, dtype) in casts {
        result = cast(&result, column, dtype)?;
    }
    Ok(result)
}

fn parse_dtype(dtype: &str) -> MungeResult<DataType> {
    match dtype.to_lowercase().as_str() {
        "int" | "int64" | "integer" => Ok(DataType::Int64),
        "int32" => Ok(DataType::Int32),
        "float" | "float64" | "double" => Ok(DataType::Float64),
        "float32" => Ok(DataType::Float32),
        "string" | "str" | "text" => Ok(DataType::String),
        "bool" | "boolean" => Ok(DataType::Boolean),
        _ => Err(MungeError::InvalidExpression(format!(
            "Unknown dtype: {}. Use int, float, string, or bool",
            dtype
        ))),
    }
}

/// Clip numeric values to a range.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to clip
/// * `min` - Minimum value (None = no lower bound)
/// * `max` - Maximum value (None = no upper bound)
pub fn clip(
    dataset: &Dataset,
    column: &str,
    min: Option<f64>,
    max: Option<f64>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let col = df.column(column)?;
    let float_col = col.cast(&DataType::Float64)?;
    let ca = float_col.f64()?;

    let clipped: Float64Chunked = ca
        .into_iter()
        .map(|opt| {
            opt.map(|v| {
                let mut val = v;
                if let Some(min_val) = min {
                    val = val.max(min_val);
                }
                if let Some(max_val) = max {
                    val = val.min(max_val);
                }
                val
            })
        })
        .collect();

    let clipped_col = clipped.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(clipped_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Replace specific values in a column.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to modify
/// * `old_value` - Value to replace
/// * `new_value` - Replacement value
pub fn replace(
    dataset: &Dataset,
    column: &str,
    old_value: &str,
    new_value: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let col = df.column(column)?;
    let dtype = col.dtype();
    let name = col.name().clone();

    let replaced_col = match dtype {
        DataType::String => {
            let ca = col.str()?;
            let replaced: StringChunked = ca
                .into_iter()
                .map(|opt| {
                    opt.map(|s| {
                        if s == old_value {
                            new_value.to_string()
                        } else {
                            s.to_string()
                        }
                    })
                })
                .collect();
            replaced.with_name(name).into_column()
        }
        DataType::Int64 => {
            let old_val: i64 = old_value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: old_value.to_string(),
                reason: "Cannot parse as integer".to_string(),
            })?;
            let new_val: i64 = new_value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: new_value.to_string(),
                reason: "Cannot parse as integer".to_string(),
            })?;
            let ca = col.i64()?;
            let replaced: Int64Chunked = ca
                .into_iter()
                .map(|opt| opt.map(|v| if v == old_val { new_val } else { v }))
                .collect();
            replaced.with_name(name).into_column()
        }
        DataType::Float64 => {
            let old_val: f64 = old_value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: old_value.to_string(),
                reason: "Cannot parse as float".to_string(),
            })?;
            let new_val: f64 = new_value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: new_value.to_string(),
                reason: "Cannot parse as float".to_string(),
            })?;
            let ca = col.f64()?;
            let replaced: Float64Chunked = ca
                .into_iter()
                .map(|opt| {
                    opt.map(|v| {
                        if (v - old_val).abs() < f64::EPSILON {
                            new_val
                        } else {
                            v
                        }
                    })
                })
                .collect();
            replaced.with_name(name).into_column()
        }
        _ => col.clone(),
    };

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(replaced_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Trim whitespace from string columns.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Columns to trim (None = all string columns)
pub fn trim(dataset: &Dataset, columns: Option<&[&str]>) -> MungeResult<Dataset> {
    let df = dataset.df();

    let cols_to_trim: Vec<String> = match columns {
        Some(cols) => {
            for col in cols {
                if df.column(col).is_err() {
                    return Err(MungeError::ColumnNotFound(col.to_string()));
                }
            }
            cols.iter().map(|s| s.to_string()).collect()
        }
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

    let mut result_df = df.clone();

    // Create a null pattern column for whitespace trimming
    let null_pattern = Column::full_null("pattern".into(), 1, &DataType::String);

    for col_name in &cols_to_trim {
        let col = result_df.column(col_name)?;
        if col.dtype() != &DataType::String {
            continue;
        }

        let ca = col.str()?;
        // Use vectorized strip_chars with null pattern (strips whitespace) - much faster
        let trimmed = ca.strip_chars(&null_pattern)?;

        let trimmed_col = trimmed.with_name(col.name().clone()).into_column();
        result_df = result_df.drop(col_name)?;
        result_df = result_df.with_column(trimmed_col)?.clone();
    }

    Ok(Dataset::new(result_df))
}

/// Convert string column to lowercase.
pub fn to_lowercase(dataset: &Dataset, column: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let ca = col.str()?;
    // Use vectorized to_lowercase() - much faster than row-by-row iteration
    let lowered: StringChunked = ca.to_lowercase();

    let lowered_col = lowered.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(lowered_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Convert string column to uppercase.
pub fn to_uppercase(dataset: &Dataset, column: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let ca = col.str()?;
    // Use vectorized to_uppercase() - much faster than row-by-row iteration
    let uppered: StringChunked = ca.to_uppercase();

    let uppered_col = uppered.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(uppered_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Replace substrings matching a regex pattern in a string column.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to modify
/// * `pattern` - Regex pattern to match
/// * `replacement` - Replacement string (can use capture groups like $1, $2)
///
/// # Example
/// ```ignore
/// // Remove all digits from a column
/// let cleaned = regex_replace(&dataset, "text", r"\d+", "")?;
///
/// // Standardize phone numbers
/// let cleaned = regex_replace(&dataset, "phone", r"(\d{3})(\d{3})(\d{4})", "($1) $2-$3")?;
/// ```
pub fn regex_replace(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    replacement: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let ca = col.str()?;
    let replaced: StringChunked = ca
        .into_iter()
        .map(|opt| opt.map(|s| re.replace_all(s, replacement).to_string()))
        .collect();

    let replaced_col = replaced.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(replaced_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Replace all occurrences of a regex pattern (same as regex_replace but with clearer naming).
pub fn regex_replace_all(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    replacement: &str,
) -> MungeResult<Dataset> {
    regex_replace(dataset, column, pattern, replacement)
}

/// Extract substrings matching a regex pattern into a new column.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to extract from
/// * `pattern` - Regex pattern with capture groups
/// * `new_column` - Name for the new column containing extracted values
/// * `group` - Which capture group to extract (0 = entire match, 1 = first group, etc.)
///
/// # Example
/// ```ignore
/// // Extract email domain
/// let result = regex_extract(&dataset, "email", r"@([\w.-]+)", "domain", 1)?;
///
/// // Extract first number from text
/// let result = regex_extract(&dataset, "text", r"(\d+)", "first_number", 1)?;
/// ```
pub fn regex_extract(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    new_column: &str,
    group: usize,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let ca = col.str()?;
    let extracted: StringChunked = ca
        .into_iter()
        .map(|opt| {
            opt.and_then(|s| {
                re.captures(s)
                    .and_then(|caps| caps.get(group).map(|m| m.as_str().to_string()))
            })
        })
        .collect();

    let extracted_col = extracted.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(extracted_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Extract all matches of a regex pattern into a new column (as comma-separated string).
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to extract from
/// * `pattern` - Regex pattern to match
/// * `new_column` - Name for the new column containing all matches
/// * `separator` - Separator for multiple matches (default ",")
///
/// # Example
/// ```ignore
/// // Extract all numbers from text
/// let result = regex_extract_all(&dataset, "text", r"\d+", "numbers", Some(","))?;
/// ```
pub fn regex_extract_all(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    new_column: &str,
    separator: Option<&str>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let sep = separator.unwrap_or(",");

    let ca = col.str()?;
    let extracted: StringChunked = ca
        .into_iter()
        .map(|opt| {
            opt.map(|s| {
                let matches: Vec<&str> = re.find_iter(s).map(|m| m.as_str()).collect();
                if matches.is_empty() {
                    String::new()
                } else {
                    matches.join(sep)
                }
            })
        })
        .collect();

    let extracted_col = extracted.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(extracted_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Count the number of matches of a regex pattern in each row.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to search in
/// * `pattern` - Regex pattern to match
/// * `new_column` - Name for the new column containing counts
///
/// # Example
/// ```ignore
/// // Count digits in each string
/// let result = regex_count(&dataset, "text", r"\d", "digit_count")?;
/// ```
pub fn regex_count(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    new_column: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let ca = col.str()?;
    let counts: Int64Chunked = ca
        .into_iter()
        .map(|opt| opt.map(|s| re.find_iter(s).count() as i64))
        .collect();

    let counts_col = counts.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(counts_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Split a string column by a pattern into multiple new columns.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to split
/// * `pattern` - Pattern to split on (regex or literal)
/// * `max_splits` - Maximum number of splits (creates max_splits + 1 columns)
/// * `prefix` - Prefix for new column names (columns named prefix_0, prefix_1, etc.)
///
/// # Example
/// ```ignore
/// // Split full name into parts
/// let result = str_split(&dataset, "full_name", r"\s+", Some(2), "name_part")?;
/// ```
pub fn str_split(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    max_splits: Option<usize>,
    prefix: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let ca = col.str()?;
    let n_splits = max_splits.unwrap_or(usize::MAX);

    // First pass: determine how many columns we need
    let mut max_parts = 0;
    let splits: Vec<Vec<String>> = ca
        .into_iter()
        .map(|opt| {
            let parts: Vec<String> = match opt {
                Some(s) => {
                    let mut parts: Vec<String> =
                        re.splitn(s, n_splits + 1).map(|p| p.to_string()).collect();
                    if max_splits.is_some() && parts.len() > n_splits + 1 {
                        parts.truncate(n_splits + 1);
                    }
                    parts
                }
                None => vec![],
            };
            if parts.len() > max_parts {
                max_parts = parts.len();
            }
            parts
        })
        .collect();

    // Cap max_parts if max_splits was specified
    if let Some(max) = max_splits {
        max_parts = max_parts.min(max + 1);
    }

    // Create columns for each split part
    let mut result_df = df.clone();
    for i in 0..max_parts {
        let col_name = format!("{}_{}", prefix, i);
        let values: StringChunked = splits
            .iter()
            .map(|parts| parts.get(i).map(|s| s.as_str()))
            .collect();
        let new_col = values.with_name(col_name.into()).into_column();
        result_df = result_df.with_column(new_col)?.clone();
    }

    Ok(Dataset::new(result_df))
}

/// Concatenate multiple string columns into one.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Columns to concatenate
/// * `new_column` - Name for the concatenated column
/// * `separator` - Separator between values (default "")
///
/// # Example
/// ```ignore
/// let result = str_concat(&dataset, &["first_name", "last_name"], "full_name", Some(" "))?;
/// ```
pub fn str_concat(
    dataset: &Dataset,
    columns: &[&str],
    new_column: &str,
    separator: Option<&str>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate all columns exist and are strings
    for col_name in columns {
        let col = df.column(col_name)?;
        if col.dtype() != &DataType::String {
            return Err(MungeError::TypeMismatch {
                column: col_name.to_string(),
                expected: "String".to_string(),
                found: format!("{:?}", col.dtype()),
            });
        }
    }

    let sep = separator.unwrap_or("");

    // Collect string chunked arrays for all columns
    let string_cols: Vec<&StringChunked> = columns
        .iter()
        .map(|name| df.column(name).unwrap().str().unwrap())
        .collect();

    // Use vectorized hor_str_concat - much faster than row-by-row iteration
    let concatenated = hor_str_concat(&string_cols, sep, true)?;

    let concat_col = concatenated.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(concat_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Pad a string column to a fixed width.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to pad
/// * `width` - Target width
/// * `pad_char` - Character to pad with (default space)
/// * `side` - "left" or "right" (default "left")
pub fn str_pad(
    dataset: &Dataset,
    column: &str,
    width: usize,
    pad_char: Option<char>,
    side: Option<&str>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let pad = pad_char.unwrap_or(' ');
    let pad_side = side.unwrap_or("left");

    let ca = col.str()?;
    let padded: StringChunked = ca
        .into_iter()
        .map(|opt| {
            opt.map(|s| {
                let len = s.chars().count();
                if len >= width {
                    s.to_string()
                } else {
                    let padding: String = std::iter::repeat_n(pad, width - len).collect();
                    match pad_side {
                        "left" | "start" => format!("{}{}", padding, s),
                        _ => format!("{}{}", s, padding),
                    }
                }
            })
        })
        .collect();

    let padded_col = padded.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(padded_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Get substring from a string column.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to extract from
/// * `start` - Start index (0-based, supports negative for from-end)
/// * `length` - Optional length to extract
pub fn str_substring(
    dataset: &Dataset,
    column: &str,
    start: i64,
    length: Option<usize>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let ca = col.str()?;

    // Use vectorized str_slice() - much faster than row-by-row iteration
    // Create scalar columns for offset and length
    let offset_col = Column::new_scalar("offset".into(), Scalar::from(start), 1);
    let length_col = match length {
        Some(l) => Column::new_scalar("length".into(), Scalar::from(l as u64), 1),
        None => Column::full_null("length".into(), 1, &DataType::UInt64),
    };

    let substr = ca.str_slice(&offset_col, &length_col)?;

    let substr_col = substr.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(substr_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Get the length of strings in a column (counts Unicode characters).
///
/// Uses SIMD-optimized `bytecount::num_chars` for fast UTF-8 character counting.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to measure
/// * `new_column` - Name for the length column
pub fn str_length(dataset: &Dataset, column: &str, new_column: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let ca = col.str()?;
    // Use SIMD-optimized bytecount::num_chars for fast UTF-8 character counting
    // This is significantly faster than std chars().count() or Polars str_len_chars()
    let lengths: Int64Chunked = ca
        .into_iter()
        .map(|opt| opt.map(|s| bytecount::num_chars(s.as_bytes()) as i64))
        .collect();

    let length_col = lengths.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(length_col)?.clone();

    Ok(Dataset::new(result_df))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    fn test_dataset_with_nulls() -> Dataset {
        let df = df! {
            "id" => [Some(1), Some(2), Some(3), Some(4), Some(5)],
            "name" => [Some("Alice"), None, Some("Charlie"), Some("Diana"), None],
            "age" => [Some(25), Some(30), None, Some(28), Some(22)],
            "score" => [Some(85.5), None, Some(78.5), None, Some(95.5)],
        }
        .unwrap();
        Dataset::new(df)
    }

    fn test_dataset_with_duplicates() -> Dataset {
        let df = df! {
            "id" => [1, 1, 2, 2, 3],
            "name" => ["Alice", "Alice", "Bob", "Bob", "Charlie"],
            "value" => [10, 10, 20, 25, 30],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_drop_na_any() {
        let ds = test_dataset_with_nulls();
        let clean = drop_na(&ds, None, "any").unwrap();
        assert_eq!(clean.nrows(), 1); // Only row 1 (id=1) has no nulls in any column
    }

    #[test]
    fn test_drop_na_specific_columns() {
        let ds = test_dataset_with_nulls();
        let clean = drop_na(&ds, Some(&["age"]), "any").unwrap();
        assert_eq!(clean.nrows(), 4); // Only row 3 has null age
    }

    #[test]
    fn test_fill_na_constant() {
        let ds = test_dataset_with_nulls();
        let filled = fill_na(&ds, Some(&["age"]), FillStrategy::Constant("0".to_string())).unwrap();
        let null_count = filled.df().column("age").unwrap().null_count();
        assert_eq!(null_count, 0);
    }

    #[test]
    fn test_fill_na_mean() {
        let ds = test_dataset_with_nulls();
        let filled = fill_na(&ds, Some(&["score"]), FillStrategy::Mean).unwrap();
        let null_count = filled.df().column("score").unwrap().null_count();
        assert_eq!(null_count, 0);
    }

    #[test]
    fn test_fill_na_forward() {
        let ds = test_dataset_with_nulls();
        let filled = fill_na(&ds, Some(&["name"]), FillStrategy::Forward).unwrap();
        // Row 2's name should be filled with "Alice" (forward fill)
        let names: Vec<Option<&str>> = filled
            .df()
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(names[1], Some("Alice"));
    }

    #[test]
    fn test_deduplicate_first() {
        let ds = test_dataset_with_duplicates();
        let deduped = deduplicate(&ds, Some(&["id"]), "first").unwrap();
        assert_eq!(deduped.nrows(), 3);
    }

    #[test]
    fn test_deduplicate_all_columns() {
        let ds = test_dataset_with_duplicates();
        let deduped = deduplicate(&ds, None, "first").unwrap();
        assert_eq!(deduped.nrows(), 4); // Two identical rows removed
    }

    #[test]
    fn test_cast() {
        let df = df! {
            "value" => ["1", "2", "3"],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let casted = cast(&ds, "value", "int").unwrap();
        assert_eq!(
            casted.df().column("value").unwrap().dtype(),
            &DataType::Int64
        );
    }

    #[test]
    fn test_clip() {
        let df = df! {
            "value" => [1.0, 5.0, 10.0, 15.0, 20.0],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let clipped = clip(&ds, "value", Some(5.0), Some(15.0)).unwrap();
        let values: Vec<f64> = clipped
            .df()
            .column("value")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![5.0, 5.0, 10.0, 15.0, 15.0]);
    }

    #[test]
    fn test_replace() {
        let df = df! {
            "status" => ["active", "inactive", "active", "pending"],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let replaced = replace(&ds, "status", "inactive", "disabled").unwrap();
        let statuses: Vec<&str> = replaced
            .df()
            .column("status")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert!(statuses.contains(&"disabled"));
        assert!(!statuses.contains(&"inactive"));
    }

    #[test]
    fn test_trim() {
        let df = df! {
            "name" => ["  Alice  ", "Bob", "  Charlie"],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let trimmed = trim(&ds, Some(&["name"])).unwrap();
        let names: Vec<&str> = trimmed
            .df()
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
    }

    #[test]
    fn test_to_lowercase() {
        let df = df! {
            "name" => ["ALICE", "Bob", "CHARLIE"],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let lowered = to_lowercase(&ds, "name").unwrap();
        let names: Vec<&str> = lowered
            .df()
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(names, vec!["alice", "bob", "charlie"]);
    }
}
