//! Core data transformation operations: filter, select, rename, mutate, sort.

use polars::prelude::PlSmallStr;
use polars::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::error::{MungeError, MungeResult};
use crate::data::Dataset;

/// Comparison operators for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOp {
    /// Equal to
    Eq,
    /// Not equal to
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal to
    Ge,
    /// Less than
    Lt,
    /// Less than or equal to
    Le,
    /// String contains substring
    Contains,
    /// String does not contain substring
    NotContains,
    /// String starts with prefix
    StartsWith,
    /// String ends with suffix
    EndsWith,
    /// String matches regex pattern
    Regex,
    /// String does not match regex pattern
    NotRegex,
}

impl FilterOp {
    /// Parse operator from string.
    pub fn from_str(s: &str) -> MungeResult<Self> {
        match s.to_lowercase().as_str() {
            "eq" | "==" | "=" => Ok(FilterOp::Eq),
            "ne" | "!=" | "<>" => Ok(FilterOp::Ne),
            "gt" | ">" => Ok(FilterOp::Gt),
            "ge" | ">=" => Ok(FilterOp::Ge),
            "lt" | "<" => Ok(FilterOp::Lt),
            "le" | "<=" => Ok(FilterOp::Le),
            "contains" | "like" => Ok(FilterOp::Contains),
            "not_contains" | "notcontains" | "not contains" => Ok(FilterOp::NotContains),
            "starts_with" | "startswith" | "starts" => Ok(FilterOp::StartsWith),
            "ends_with" | "endswith" | "ends" => Ok(FilterOp::EndsWith),
            "regex" | "matches" | "regexp" => Ok(FilterOp::Regex),
            "not_regex" | "notregex" | "not_matches" => Ok(FilterOp::NotRegex),
            _ => Err(MungeError::InvalidOperator(s.to_string())),
        }
    }

    /// Check if this operator is a string-only operator
    pub fn is_string_only(&self) -> bool {
        matches!(
            self,
            FilterOp::Contains
                | FilterOp::NotContains
                | FilterOp::StartsWith
                | FilterOp::EndsWith
                | FilterOp::Regex
                | FilterOp::NotRegex
        )
    }
}

/// Filter rows based on a condition.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to filter on
/// * `op` - Comparison operator. Supported operators:
///   - Comparison: eq, ne, gt, ge, lt, le
///   - String: contains, not_contains, starts_with, ends_with, regex, not_regex
/// * `value` - Value to compare against (parsed based on column type)
///
/// # Example
/// ```ignore
/// // Numeric filter
/// let filtered = filter(&dataset, "age", "ge", "18")?;
///
/// // String contains
/// let filtered = filter(&dataset, "name", "contains", "Smith")?;
///
/// // Regex filter
/// let filtered = filter(&dataset, "email", "regex", r"^[\w.-]+@[\w.-]+\.\w+$")?;
/// ```
pub fn filter(dataset: &Dataset, column: &str, op: &str, value: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Check column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let col_series = df.column(column)?;
    let dtype = col_series.dtype();
    let op = FilterOp::from_str(op)?;

    // String-only operators require string columns
    if op.is_string_only() && !matches!(dtype, DataType::String) {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "string".to_string(),
            found: format!("{:?}", dtype),
        });
    }

    // Build the filter expression based on column type
    // Cast to common types for comparison
    let mask = match dtype {
        DataType::Int64 => {
            let val: i64 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as integer".to_string(),
            })?;
            apply_numeric_filter_i64(col_series, op, val)?
        }
        DataType::Int32 => {
            let val: i32 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as integer".to_string(),
            })?;
            apply_numeric_filter_i32(col_series, op, val)?
        }
        DataType::Int16 | DataType::Int8 => {
            // Cast to i32 for smaller types
            let casted = col_series.cast(&DataType::Int32)?;
            let val: i32 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as integer".to_string(),
            })?;
            apply_numeric_filter_i32(&casted, op, val)?
        }
        DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8 => {
            let val: u64 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as unsigned integer".to_string(),
            })?;
            apply_numeric_filter_u64(col_series, op, val)?
        }
        DataType::Float64 | DataType::Float32 => {
            let val: f64 = value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as float".to_string(),
            })?;
            apply_numeric_filter_f64(col_series, op, val)?
        }
        DataType::String => apply_string_filter(col_series, op, value)?,
        DataType::Boolean => {
            let val: bool = value.parse().map_err(|_| MungeError::InvalidValue {
                column: column.to_string(),
                value: value.to_string(),
                reason: "Cannot parse as boolean".to_string(),
            })?;
            apply_bool_filter(col_series, op, val)?
        }
        _ => {
            return Err(MungeError::TypeMismatch {
                column: column.to_string(),
                expected: "numeric, string, or boolean".to_string(),
                found: format!("{:?}", dtype),
            });
        }
    };

    let filtered_df = df.filter(&mask)?;
    Ok(Dataset::new(filtered_df))
}

fn apply_numeric_filter_i64(
    series: &Column,
    op: FilterOp,
    value: i64,
) -> MungeResult<BooleanChunked> {
    let ca = series.i64()?;
    Ok(match op {
        FilterOp::Eq => ca.equal(value),
        FilterOp::Ne => ca.not_equal(value),
        FilterOp::Gt => ca.gt(value),
        FilterOp::Ge => ca.gt_eq(value),
        FilterOp::Lt => ca.lt(value),
        FilterOp::Le => ca.lt_eq(value),
        // String-only operators should not reach here (caught earlier)
        _ => {
            return Err(MungeError::InvalidOperator(
                "String operators not valid for numeric columns".to_string(),
            ));
        }
    })
}

fn apply_numeric_filter_i32(
    series: &Column,
    op: FilterOp,
    value: i32,
) -> MungeResult<BooleanChunked> {
    let ca = series.i32()?;
    Ok(match op {
        FilterOp::Eq => ca.equal(value),
        FilterOp::Ne => ca.not_equal(value),
        FilterOp::Gt => ca.gt(value),
        FilterOp::Ge => ca.gt_eq(value),
        FilterOp::Lt => ca.lt(value),
        FilterOp::Le => ca.lt_eq(value),
        _ => {
            return Err(MungeError::InvalidOperator(
                "String operators not valid for numeric columns".to_string(),
            ));
        }
    })
}

fn apply_numeric_filter_u64(
    series: &Column,
    op: FilterOp,
    value: u64,
) -> MungeResult<BooleanChunked> {
    let ca = series.u64()?;
    Ok(match op {
        FilterOp::Eq => ca.equal(value),
        FilterOp::Ne => ca.not_equal(value),
        FilterOp::Gt => ca.gt(value),
        FilterOp::Ge => ca.gt_eq(value),
        FilterOp::Lt => ca.lt(value),
        FilterOp::Le => ca.lt_eq(value),
        _ => {
            return Err(MungeError::InvalidOperator(
                "String operators not valid for numeric columns".to_string(),
            ));
        }
    })
}

fn apply_numeric_filter_f64(
    series: &Column,
    op: FilterOp,
    value: f64,
) -> MungeResult<BooleanChunked> {
    let ca = series.f64()?;
    Ok(match op {
        FilterOp::Eq => ca.equal(value),
        FilterOp::Ne => ca.not_equal(value),
        FilterOp::Gt => ca.gt(value),
        FilterOp::Ge => ca.gt_eq(value),
        FilterOp::Lt => ca.lt(value),
        FilterOp::Le => ca.lt_eq(value),
        _ => {
            return Err(MungeError::InvalidOperator(
                "String operators not valid for numeric columns".to_string(),
            ));
        }
    })
}

fn apply_string_filter(series: &Column, op: FilterOp, value: &str) -> MungeResult<BooleanChunked> {
    let ca = series.str()?;
    Ok(match op {
        FilterOp::Eq => ca.equal(value),
        FilterOp::Ne => ca.not_equal(value),
        FilterOp::Gt => ca.gt(value),
        FilterOp::Ge => ca.gt_eq(value),
        FilterOp::Lt => ca.lt(value),
        FilterOp::Le => ca.lt_eq(value),
        FilterOp::Contains => ca
            .into_iter()
            .map(|opt| opt.map(|s| s.contains(value)))
            .collect(),
        FilterOp::NotContains => ca
            .into_iter()
            .map(|opt| opt.map(|s| !s.contains(value)))
            .collect(),
        FilterOp::StartsWith => ca
            .into_iter()
            .map(|opt| opt.map(|s| s.starts_with(value)))
            .collect(),
        FilterOp::EndsWith => ca
            .into_iter()
            .map(|opt| opt.map(|s| s.ends_with(value)))
            .collect(),
        FilterOp::Regex => {
            let re = Regex::new(value).map_err(|e| {
                MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", value, e))
            })?;
            ca.into_iter()
                .map(|opt| opt.map(|s| re.is_match(s)))
                .collect()
        }
        FilterOp::NotRegex => {
            let re = Regex::new(value).map_err(|e| {
                MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", value, e))
            })?;
            ca.into_iter()
                .map(|opt| opt.map(|s| !re.is_match(s)))
                .collect()
        }
    })
}

fn apply_bool_filter(series: &Column, op: FilterOp, value: bool) -> MungeResult<BooleanChunked> {
    let ca = series.bool()?;
    // For boolean comparison, we compare each element manually
    let result: BooleanChunked = match op {
        FilterOp::Eq => ca.into_iter().map(|opt| opt.map(|v| v == value)).collect(),
        FilterOp::Ne => ca.into_iter().map(|opt| opt.map(|v| v != value)).collect(),
        _ => {
            return Err(MungeError::InvalidOperator(
                "Only eq/ne valid for boolean columns".to_string(),
            ));
        }
    };
    Ok(result)
}

/// Filter with multiple conditions (AND logic).
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `conditions` - Vec of (column, operator, value) tuples
///
/// # Example
/// ```ignore
/// let filtered = filter_and(&dataset, &[
///     ("age", "ge", "18"),
///     ("status", "eq", "active"),
/// ])?;
/// ```
pub fn filter_and(dataset: &Dataset, conditions: &[(&str, &str, &str)]) -> MungeResult<Dataset> {
    let mut result = dataset.clone();
    for (column, op, value) in conditions {
        result = filter(&result, column, op, value)?;
    }
    Ok(result)
}

/// Select specific columns from a dataset.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Column names to keep
///
/// # Example
/// ```ignore
/// let subset = select(&dataset, &["id", "name", "value"])?;
/// ```
pub fn select(dataset: &Dataset, columns: &[&str]) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate all columns exist
    for col in columns {
        if df.column(col).is_err() {
            return Err(MungeError::ColumnNotFound(col.to_string()));
        }
    }

    // Convert to owned strings for Polars API
    let col_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
    let selected = df.select(&col_names)?;
    Ok(Dataset::new(selected))
}

/// Drop specific columns from a dataset.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Column names to drop
///
/// # Example
/// ```ignore
/// let reduced = drop_columns(&dataset, &["temp_col", "debug_info"])?;
/// ```
pub fn drop_columns(dataset: &Dataset, columns: &[&str]) -> MungeResult<Dataset> {
    let mut df = dataset.df().clone();

    for col in columns {
        if df.column(col).is_err() {
            return Err(MungeError::ColumnNotFound(col.to_string()));
        }
        df = df.drop(col)?;
    }

    Ok(Dataset::new(df))
}

/// Rename columns in a dataset.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `renames` - Vec of (old_name, new_name) pairs
///
/// # Example
/// ```ignore
/// let renamed = rename(&dataset, &[("old_col", "new_col")])?;
/// ```
pub fn rename(dataset: &Dataset, renames: &[(&str, &str)]) -> MungeResult<Dataset> {
    let mut df = dataset.df().clone();

    for (old_name, new_name) in renames {
        if df.column(old_name).is_err() {
            return Err(MungeError::ColumnNotFound(old_name.to_string()));
        }
        // Polars rename returns &mut DataFrame, so we need to work around it
        let new_name_str: PlSmallStr = (*new_name).into();
        df.rename(old_name, new_name_str)?;
    }

    Ok(Dataset::new(df))
}

/// Arithmetic operations for mutate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ArithOp {
    pub fn from_str(s: &str) -> MungeResult<Self> {
        match s {
            "+" | "add" => Ok(ArithOp::Add),
            "-" | "sub" => Ok(ArithOp::Sub),
            "*" | "mul" => Ok(ArithOp::Mul),
            "/" | "div" => Ok(ArithOp::Div),
            _ => Err(MungeError::InvalidOperator(s.to_string())),
        }
    }
}

/// Expression types for mutate operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutateExpr {
    /// Constant value
    Constant(String),
    /// Copy from another column
    Copy(String),
    /// Arithmetic between two columns: (col1, op, col2)
    Arithmetic(String, ArithOp, String),
    /// Arithmetic with constant: (col, op, constant)
    ArithmeticConst(String, ArithOp, f64),
    /// Apply function to column: (function_name, column)
    Function(String, String),
}

/// Create or update a column with computed values.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `new_col` - Name for the new/updated column
/// * `expr` - Expression defining the computation
///
/// # Example
/// ```ignore
/// // Copy a column
/// let ds = mutate(&dataset, "x_copy", MutateExpr::Copy("x".to_string()))?;
///
/// // Arithmetic between columns
/// let ds = mutate(&dataset, "total", MutateExpr::Arithmetic(
///     "price".to_string(), ArithOp::Mul, "quantity".to_string()
/// ))?;
///
/// // Apply function
/// let ds = mutate(&dataset, "log_x", MutateExpr::Function("log".to_string(), "x".to_string()))?;
/// ```
pub fn mutate(dataset: &Dataset, new_col: &str, expr: MutateExpr) -> MungeResult<Dataset> {
    let df = dataset.df();

    let new_series = match expr {
        MutateExpr::Constant(value) => {
            // Try to parse as f64, otherwise use string
            if let Ok(val) = value.parse::<f64>() {
                Column::new(new_col.into(), vec![val; df.height()])
            } else {
                Column::new(new_col.into(), vec![value.as_str(); df.height()])
            }
        }
        MutateExpr::Copy(col_name) => {
            let col = df.column(&col_name)?;
            col.clone().with_name(new_col.into())
        }
        MutateExpr::Arithmetic(col1, op, col2) => {
            let s1 = df.column(&col1)?.cast(&DataType::Float64)?;
            let s2 = df.column(&col2)?.cast(&DataType::Float64)?;
            let ca1 = s1.f64()?;
            let ca2 = s2.f64()?;

            let result: Float64Chunked = match op {
                ArithOp::Add => ca1 + ca2,
                ArithOp::Sub => ca1 - ca2,
                ArithOp::Mul => ca1 * ca2,
                ArithOp::Div => ca1 / ca2,
            };
            result.into_column().with_name(new_col.into())
        }
        MutateExpr::ArithmeticConst(col_name, op, constant) => {
            let s = df.column(&col_name)?.cast(&DataType::Float64)?;
            let ca = s.f64()?;

            let result: Float64Chunked = match op {
                ArithOp::Add => ca + constant,
                ArithOp::Sub => ca - constant,
                ArithOp::Mul => ca * constant,
                ArithOp::Div => ca / constant,
            };
            result.into_column().with_name(new_col.into())
        }
        MutateExpr::Function(func_name, col_name) => {
            let s = df.column(&col_name)?.cast(&DataType::Float64)?;
            let ca = s.f64()?;

            let result: Float64Chunked = match func_name.to_lowercase().as_str() {
                "log" | "ln" => ca.apply(|v| v.map(|x| x.ln())),
                "log10" => ca.apply(|v| v.map(|x| x.log10())),
                "log2" => ca.apply(|v| v.map(|x| x.log2())),
                "exp" => ca.apply(|v| v.map(|x| x.exp())),
                "sqrt" => ca.apply(|v| v.map(|x| x.sqrt())),
                "abs" => ca.apply(|v| v.map(|x| x.abs())),
                "square" => ca.apply(|v| v.map(|x| x * x)),
                "floor" => ca.apply(|v| v.map(|x| x.floor())),
                "ceil" => ca.apply(|v| v.map(|x| x.ceil())),
                "round" => ca.apply(|v| v.map(|x| x.round())),
                "sin" => ca.apply(|v| v.map(|x| x.sin())),
                "cos" => ca.apply(|v| v.map(|x| x.cos())),
                "tan" => ca.apply(|v| v.map(|x| x.tan())),
                _ => {
                    return Err(MungeError::InvalidExpression(format!(
                        "Unknown function: {}",
                        func_name
                    )));
                }
            };
            result.into_column().with_name(new_col.into())
        }
    };

    let mut new_df = df.clone();
    // Remove existing column if present, then add new one
    if new_df.column(new_col).is_ok() {
        new_df = new_df.drop(new_col)?;
    }
    new_df = new_df.with_column(new_series)?.clone();

    Ok(Dataset::new(new_df))
}

/// Sort dataset by one or more columns.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Column names to sort by
/// * `descending` - Whether each column should be sorted descending (must match length of columns)
///
/// # Example
/// ```ignore
/// let sorted = sort(&dataset, &["category", "value"], &[false, true])?;
/// ```
pub fn sort(dataset: &Dataset, columns: &[&str], descending: &[bool]) -> MungeResult<Dataset> {
    let df = dataset.df();

    if columns.len() != descending.len() {
        return Err(MungeError::ColumnCountMismatch {
            expected: columns.len(),
            found: descending.len(),
        });
    }

    // Validate columns exist
    for col in columns {
        if df.column(col).is_err() {
            return Err(MungeError::ColumnNotFound(col.to_string()));
        }
    }

    // Convert column names to Vec<String> for Polars API
    let col_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();

    let sorted = df.sort(
        &col_names,
        SortMultipleOptions::new().with_order_descending_multi(descending.to_vec()),
    )?;
    Ok(Dataset::new(sorted))
}

/// Take the first n rows.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `n` - Number of rows to take
pub fn head(dataset: &Dataset, n: usize) -> MungeResult<Dataset> {
    let df = dataset.df().head(Some(n));
    Ok(Dataset::new(df))
}

/// Take the last n rows.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `n` - Number of rows to take
pub fn tail(dataset: &Dataset, n: usize) -> MungeResult<Dataset> {
    let df = dataset.df().tail(Some(n));
    Ok(Dataset::new(df))
}

/// Slice rows by index range.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `offset` - Starting row index
/// * `length` - Number of rows to take
pub fn slice(dataset: &Dataset, offset: i64, length: usize) -> MungeResult<Dataset> {
    let df = dataset.df().slice(offset, length);
    Ok(Dataset::new(df))
}

/// Sample random rows from the dataset.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `n` - Number of rows to sample (mutually exclusive with fraction)
/// * `fraction` - Fraction of rows to sample (mutually exclusive with n)
/// * `with_replacement` - Whether to sample with replacement
/// * `seed` - Random seed for reproducibility
pub fn sample(
    dataset: &Dataset,
    n: Option<usize>,
    fraction: Option<f64>,
    with_replacement: bool,
    seed: Option<u64>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let sampled = if let Some(n_rows) = n {
        df.sample_n_literal(n_rows, with_replacement, false, seed)?
    } else if let Some(frac) = fraction {
        df.sample_frac(
            &Series::new("".into(), &[frac]),
            with_replacement,
            false,
            seed,
        )?
    } else {
        return Err(MungeError::InvalidExpression(
            "Must specify either n or fraction for sampling".to_string(),
        ));
    };

    Ok(Dataset::new(sampled))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    fn test_dataset() -> Dataset {
        let df = df! {
            "id" => [1, 2, 3, 4, 5],
            "name" => ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age" => [25, 30, 35, 28, 22],
            "score" => [85.5, 92.0, 78.5, 88.0, 95.5],
            "active" => [true, true, false, true, false],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_filter_numeric() {
        let ds = test_dataset();
        let filtered = filter(&ds, "age", "ge", "30").unwrap();
        assert_eq!(filtered.nrows(), 2);

        let filtered = filter(&ds, "age", "lt", "25").unwrap();
        assert_eq!(filtered.nrows(), 1);
    }

    #[test]
    fn test_filter_string() {
        let ds = test_dataset();
        let filtered = filter(&ds, "name", "eq", "Alice").unwrap();
        assert_eq!(filtered.nrows(), 1);
    }

    #[test]
    fn test_filter_float() {
        let ds = test_dataset();
        let filtered = filter(&ds, "score", "gt", "90.0").unwrap();
        assert_eq!(filtered.nrows(), 2);
    }

    #[test]
    fn test_filter_and() {
        let ds = test_dataset();
        let filtered = filter_and(&ds, &[("age", "ge", "25"), ("active", "eq", "true")]).unwrap();
        assert_eq!(filtered.nrows(), 3);
    }

    #[test]
    fn test_select() {
        let ds = test_dataset();
        let selected = select(&ds, &["id", "name"]).unwrap();
        assert_eq!(selected.ncols(), 2);
        assert!(selected.column_names().contains(&"id".to_string()));
        assert!(selected.column_names().contains(&"name".to_string()));
    }

    #[test]
    fn test_drop_columns() {
        let ds = test_dataset();
        let reduced = drop_columns(&ds, &["active", "score"]).unwrap();
        assert_eq!(reduced.ncols(), 3);
        assert!(!reduced.column_names().contains(&"active".to_string()));
    }

    #[test]
    fn test_rename() {
        let ds = test_dataset();
        let renamed = rename(&ds, &[("id", "user_id"), ("name", "user_name")]).unwrap();
        assert!(renamed.column_names().contains(&"user_id".to_string()));
        assert!(renamed.column_names().contains(&"user_name".to_string()));
        assert!(!renamed.column_names().contains(&"id".to_string()));
    }

    #[test]
    fn test_mutate_constant() {
        let ds = test_dataset();
        let mutated = mutate(&ds, "flag", MutateExpr::Constant("1".to_string())).unwrap();
        assert!(mutated.column_names().contains(&"flag".to_string()));
    }

    #[test]
    fn test_mutate_copy() {
        let ds = test_dataset();
        let mutated = mutate(&ds, "age_copy", MutateExpr::Copy("age".to_string())).unwrap();
        assert!(mutated.column_names().contains(&"age_copy".to_string()));
    }

    #[test]
    fn test_mutate_arithmetic() {
        let ds = test_dataset();
        let mutated = mutate(
            &ds,
            "age_plus_score",
            MutateExpr::Arithmetic("age".to_string(), ArithOp::Add, "score".to_string()),
        )
        .unwrap();
        assert!(
            mutated
                .column_names()
                .contains(&"age_plus_score".to_string())
        );
    }

    #[test]
    fn test_mutate_function() {
        let ds = test_dataset();
        let mutated = mutate(
            &ds,
            "log_score",
            MutateExpr::Function("log".to_string(), "score".to_string()),
        )
        .unwrap();
        assert!(mutated.column_names().contains(&"log_score".to_string()));
    }

    #[test]
    fn test_sort() {
        let ds = test_dataset();
        let sorted = sort(&ds, &["age"], &[true]).unwrap();
        let ages: Vec<i32> = sorted
            .df()
            .column("age")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(ages, vec![35, 30, 28, 25, 22]);
    }

    #[test]
    fn test_head_tail() {
        let ds = test_dataset();
        let h = head(&ds, 2).unwrap();
        assert_eq!(h.nrows(), 2);

        let t = tail(&ds, 2).unwrap();
        assert_eq!(t.nrows(), 2);
    }

    #[test]
    fn test_slice() {
        let ds = test_dataset();
        let sliced = slice(&ds, 1, 2).unwrap();
        assert_eq!(sliced.nrows(), 2);
    }

    #[test]
    fn test_sample() {
        let ds = test_dataset();
        let sampled = sample(&ds, Some(2), None, false, Some(42)).unwrap();
        assert_eq!(sampled.nrows(), 2);

        let sampled = sample(&ds, None, Some(0.4), false, Some(42)).unwrap();
        assert_eq!(sampled.nrows(), 2);
    }
}
