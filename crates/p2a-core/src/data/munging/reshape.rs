//! Reshape operations for transforming dataset structure.
//!
//! This module provides operations for reshaping datasets including
//! pivot (long to wide), melt (wide to long), and transpose.
//!
//! # Operations
//!
//! - **Pivot**: Transform from long format to wide format
//! - **Melt**: Transform from wide format to long format
//! - **Transpose**: Swap rows and columns
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::data::munging::*;
//!
//! // Pivot: long to wide
//! let wide = pivot(&long_data, &["id"], "variable", "value")?;
//!
//! // Melt: wide to long
//! let long = melt(&wide_data, &["id"], &["var1", "var2", "var3"])?;
//!
//! // Transpose
//! let transposed = transpose(&data)?;
//! ```

use super::error::{MungeError, MungeResult};
use crate::data::Dataset;
use polars::prelude::*;
use polars_ops::frame::pivot::pivot as polars_pivot;

/// Pivot a dataset from long to wide format.
///
/// Takes a dataset in long format (where each row represents a single observation)
/// and transforms it to wide format (where observations are spread across columns).
///
/// # Arguments
///
/// * `dataset` - Dataset to pivot
/// * `index` - Columns to use as index (will remain as rows)
/// * `on` - Column whose values become new column names
/// * `values` - Column containing values to fill the new columns
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::pivot;
///
/// // Input (long format):
/// // id | year | value
/// // 1  | 2020 | 100
/// // 1  | 2021 | 110
/// // 2  | 2020 | 200
/// // 2  | 2021 | 220
///
/// let wide = pivot(&data, &["id"], "year", "value")?;
///
/// // Output (wide format):
/// // id | 2020 | 2021
/// // 1  | 100  | 110
/// // 2  | 200  | 220
/// ```
pub fn pivot(dataset: &Dataset, index: &[&str], on: &str, values: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns exist
    for col_name in index {
        if df.column(col_name).is_err() {
            return Err(MungeError::ColumnNotFound(col_name.to_string()));
        }
    }
    if df.column(on).is_err() {
        return Err(MungeError::ColumnNotFound(on.to_string()));
    }
    if df.column(values).is_err() {
        return Err(MungeError::ColumnNotFound(values.to_string()));
    }

    // Use polars_pivot free function (not a method on DataFrame)
    let result = polars_pivot(
        df,
        [on],
        Some(index.iter().copied()),
        Some([values]),
        false, // sort_columns
        None,  // aggregate_expr
        None,  // separator
    )
    .map_err(|e: PolarsError| MungeError::ReshapeError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Melt a dataset from wide to long format.
///
/// Takes a dataset in wide format (where observations are spread across columns)
/// and transforms it to long format (where each row represents a single observation).
///
/// # Arguments
///
/// * `dataset` - Dataset to melt
/// * `id_vars` - Columns to use as identifier variables (kept as-is)
/// * `value_vars` - Columns to unpivot into rows
/// * `variable_name` - Name for the new column containing variable names
/// * `value_name` - Name for the new column containing values
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::melt;
///
/// // Input (wide format):
/// // id | 2020 | 2021
/// // 1  | 100  | 110
/// // 2  | 200  | 220
///
/// let long = melt(&data, &["id"], &["2020", "2021"], "year", "value")?;
///
/// // Output (long format):
/// // id | year | value
/// // 1  | 2020 | 100
/// // 1  | 2021 | 110
/// // 2  | 2020 | 200
/// // 2  | 2021 | 220
/// ```
pub fn melt(
    dataset: &Dataset,
    id_vars: &[&str],
    value_vars: &[&str],
    variable_name: &str,
    value_name: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns exist
    for col_name in id_vars {
        if df.column(col_name).is_err() {
            return Err(MungeError::ColumnNotFound(col_name.to_string()));
        }
    }
    for col_name in value_vars {
        if df.column(col_name).is_err() {
            return Err(MungeError::ColumnNotFound(col_name.to_string()));
        }
    }

    let n_rows = df.height();
    let n_value_cols = value_vars.len();

    if n_value_cols == 0 {
        return Err(MungeError::ReshapeError(
            "At least one value variable is required".to_string(),
        ));
    }

    // Build the melted dataframe manually
    let mut columns: Vec<Column> = Vec::new();

    // For each id column, repeat values for each value_var
    for id_col in id_vars {
        let id_column = df
            .column(id_col)
            .map_err(|e| MungeError::ReshapeError(e.to_string()))?;
        // Build index array to repeat each row n_value_cols times
        let mut idx = Vec::with_capacity(n_rows * n_value_cols);
        for i in 0..n_rows {
            for _ in 0..n_value_cols {
                idx.push(i as IdxSize);
            }
        }
        let idx_ca = IdxCa::new("".into(), idx);
        let repeated = id_column
            .take(&idx_ca)
            .map_err(|e| MungeError::ReshapeError(e.to_string()))?;
        columns.push(repeated);
    }

    // Create the variable name column
    let mut var_names: Vec<&str> = Vec::with_capacity(n_rows * n_value_cols);
    for _ in 0..n_rows {
        for var in value_vars {
            var_names.push(var);
        }
    }
    let var_col = Column::new(variable_name.into(), var_names);
    columns.push(var_col);

    // Create the value column by stacking all value columns
    let mut values: Vec<f64> = Vec::with_capacity(n_rows * n_value_cols);
    for row_idx in 0..n_rows {
        for var in value_vars {
            let val_column = df
                .column(var)
                .map_err(|e| MungeError::ReshapeError(e.to_string()))?;
            // Try to get as f64, handling different types
            let val = if let Ok(f64_col) = val_column.f64() {
                f64_col.get(row_idx).unwrap_or(f64::NAN)
            } else if let Ok(i64_col) = val_column.i64() {
                i64_col.get(row_idx).map(|v| v as f64).unwrap_or(f64::NAN)
            } else if let Ok(i32_col) = val_column.i32() {
                i32_col.get(row_idx).map(|v| v as f64).unwrap_or(f64::NAN)
            } else {
                f64::NAN
            };
            values.push(val);
        }
    }
    let value_col = Column::new(value_name.into(), values);
    columns.push(value_col);

    let result = DataFrame::new(columns).map_err(|e| MungeError::ReshapeError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Transpose a dataset (swap rows and columns).
///
/// The column names become the first column of the result, and each row
/// becomes a new column.
///
/// # Arguments
///
/// * `dataset` - Dataset to transpose
/// * `header_name` - Name for the new header column containing original column names
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::transpose;
///
/// // Input:
/// // a | b | c
/// // 1 | 2 | 3
/// // 4 | 5 | 6
///
/// let transposed = transpose(&data, "column")?;
///
/// // Output:
/// // column | column_0 | column_1
/// // a      | 1        | 4
/// // b      | 2        | 5
/// // c      | 3        | 6
/// ```
pub fn transpose(dataset: &Dataset, header_name: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    if df.height() == 0 || df.width() == 0 {
        return Err(MungeError::EmptyDataset);
    }

    // Get column names (will become first column of result)
    let col_names: Vec<String> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();

    // Clone and transpose the data using Polars
    let mut transposed = df
        .clone()
        .transpose(None, None)
        .map_err(|e| MungeError::ReshapeError(e.to_string()))?;

    // Add the original column names as the first column
    let header_col = Column::new(header_name.into(), col_names);

    let result = transposed
        .with_column(header_col)
        .map_err(|e| MungeError::ReshapeError(e.to_string()))?;

    // Move header column to first position
    let mut col_order: Vec<String> = vec![header_name.to_string()];
    for name in result.get_column_names() {
        if name != header_name {
            col_order.push(name.to_string());
        }
    }

    let result = result
        .select(col_order.iter().map(|s| s.as_str()))
        .map_err(|e| MungeError::ReshapeError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Explode a column containing lists into multiple rows.
///
/// # Arguments
///
/// * `dataset` - Dataset containing a list column
/// * `column` - Name of the column to explode
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::explode;
///
/// // Input:
/// // id | values
/// // 1  | [1, 2, 3]
/// // 2  | [4, 5]
///
/// let exploded = explode(&data, "values")?;
///
/// // Output:
/// // id | values
/// // 1  | 1
/// // 1  | 2
/// // 1  | 3
/// // 2  | 4
/// // 2  | 5
/// ```
pub fn explode(dataset: &Dataset, column: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate column exists
    if df.column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    let result = df
        .clone()
        .lazy()
        .explode(cols([column]))
        .collect()
        .map_err(|e| MungeError::ReshapeError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Stack multiple columns into one, keeping track of the source column.
///
/// This is similar to melt but allows specifying the output more directly.
///
/// # Arguments
///
/// * `dataset` - Dataset to stack
/// * `columns` - Columns to stack into one
/// * `stacked_name` - Name for the stacked values column
/// * `source_name` - Name for the source column name
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::stack;
///
/// // Stack columns q1, q2, q3, q4 into a single column
/// let stacked = stack(&data, &["q1", "q2", "q3", "q4"], "value", "quarter")?;
/// ```
pub fn stack(
    dataset: &Dataset,
    columns: &[&str],
    stacked_name: &str,
    source_name: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns exist
    for col_name in columns {
        if df.column(col_name).is_err() {
            return Err(MungeError::ColumnNotFound(col_name.to_string()));
        }
    }

    // Get non-stacked columns
    let all_cols: std::collections::HashSet<_> = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    let stack_cols: std::collections::HashSet<_> = columns.iter().map(|s| s.to_string()).collect();
    let id_vars: Vec<String> = all_cols.difference(&stack_cols).cloned().collect();
    let id_vars_refs: Vec<&str> = id_vars.iter().map(|s| s.as_str()).collect();

    // Use melt to stack
    melt(dataset, &id_vars_refs, columns, source_name, stacked_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    #[test]
    fn test_pivot() {
        let df = df! {
            "id" => [1, 1, 2, 2],
            "year" => ["2020", "2021", "2020", "2021"],
            "value" => [100.0, 110.0, 200.0, 220.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = pivot(&ds, &["id"], "year", "value").unwrap();

        assert_eq!(result.nrows(), 2);
        assert!(result.df().column("id").is_ok());
        assert!(result.df().column("2020").is_ok());
        assert!(result.df().column("2021").is_ok());
    }

    #[test]
    fn test_melt() {
        let df = df! {
            "id" => [1, 2],
            "val_2020" => [100.0, 200.0],
            "val_2021" => [110.0, 220.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = melt(&ds, &["id"], &["val_2020", "val_2021"], "year", "value").unwrap();

        assert_eq!(result.nrows(), 4);
        assert!(result.df().column("id").is_ok());
        assert!(result.df().column("year").is_ok());
        assert!(result.df().column("value").is_ok());
    }

    #[test]
    fn test_transpose() {
        let df = df! {
            "a" => [1, 2],
            "b" => [3, 4],
            "c" => [5, 6],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = transpose(&ds, "column").unwrap();

        // Original: 2 rows, 3 cols -> Transposed: 3 rows, 3 cols (original cols + data cols)
        assert_eq!(result.nrows(), 3);
        assert!(result.df().column("column").is_ok());
    }

    #[test]
    fn test_melt_column_not_found() {
        let df = df! {
            "id" => [1, 2],
            "a" => [100.0, 200.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = melt(&ds, &["id"], &["nonexistent"], "var", "val");
        assert!(matches!(result, Err(MungeError::ColumnNotFound(_))));
    }

    #[test]
    fn test_pivot_column_not_found() {
        let df = df! {
            "id" => [1, 2],
            "value" => [100.0, 200.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = pivot(&ds, &["id"], "nonexistent", "value");
        assert!(matches!(result, Err(MungeError::ColumnNotFound(_))));
    }

    #[test]
    fn test_stack() {
        let df = df! {
            "id" => [1, 2],
            "q1" => [10.0, 20.0],
            "q2" => [11.0, 21.0],
            "q3" => [12.0, 22.0],
        }
        .unwrap();
        let ds = Dataset::new(df);

        let result = stack(&ds, &["q1", "q2", "q3"], "value", "quarter").unwrap();

        assert_eq!(result.nrows(), 6); // 2 ids x 3 quarters
        assert!(result.df().column("id").is_ok());
        assert!(result.df().column("quarter").is_ok());
        assert!(result.df().column("value").is_ok());
    }
}
