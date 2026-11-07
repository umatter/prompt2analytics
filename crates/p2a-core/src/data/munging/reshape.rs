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

use super::CHECKPOINT_EVERY;
use super::error::{MungeError, MungeResult};
use crate::data::Dataset;
use polars::prelude::*;
// polars-ops 0.52's `pivot` runs exactly the code `pivot_stable` runs, and
// also prints a notice to stdout on every call. Stdout is the MCP stdio
// transport's JSON-RPC stream and the CLI's JSON output, so call the stable
// entry point, which prints nothing.
use polars_ops::frame::pivot::pivot_stable;

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

    // polars' free function (not a method on DataFrame); see the import.
    let result = pivot_stable(
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
///
/// As `tidyr::pivot_longer`, the long order runs row by row, nulls stay null, and the value column
/// takes the columns' common type (integer + double gives double; two integer columns stay
/// integer).
///
/// # Errors
///
/// [`MungeError::ColumnNotFound`] for a column that does not exist; [`MungeError::ReshapeError`] for
/// no value variable, duplicate output names, text melted together with non-text columns (tidyr
/// refuses them too), a date, datetime, duration or time column melted together with another kind of
/// column (dates combine with datetimes, as in tidyr, and durations of different units with each
/// other), and types without a common type.
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

    let value_type = melt_value_type(df, value_vars)?;

    let stopped = |e: crate::errors::EconError| MungeError::ReshapeError(e.to_string());
    let n_out =
        crate::resource::checked_len(&[n_rows, n_value_cols], "melt output").map_err(stopped)?;
    if IdxSize::try_from(n_out).is_err() {
        return Err(MungeError::ReshapeError(format!(
            "melt would give {n_out} rows, more than the {} a dataset holds",
            IdxSize::MAX
        )));
    }
    crate::resource::check_alloc::<IdxSize>(n_out, "melt row index").map_err(stopped)?;

    let mut columns: Vec<Column> = Vec::with_capacity(id_vars.len() + 2);

    // Each id cell, repeated once per value variable. The size check above covers `as IdxSize`.
    if !id_vars.is_empty() {
        let mut idx: Vec<IdxSize> =
            crate::resource::try_with_capacity(n_out, "melt row index").map_err(stopped)?;
        for i in 0..n_rows {
            if i.is_multiple_of(CHECKPOINT_EVERY) {
                crate::resource::checkpoint().map_err(stopped)?;
            }
            for _ in 0..n_value_cols {
                idx.push(i as IdxSize);
            }
        }
        let idx_ca = IdxCa::from_vec("".into(), idx);
        for id_col in id_vars {
            let repeated = df
                .column(id_col)
                .and_then(|c| c.take(&idx_ca))
                .map_err(|e| MungeError::ReshapeError(e.to_string()))?;
            columns.push(repeated);
        }
    }

    // The variable names, row by row.
    let mut var_names: Vec<&str> =
        crate::resource::try_with_capacity(n_out, "melt variable names").map_err(stopped)?;
    for i in 0..n_rows {
        if i.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(stopped)?;
        }
        var_names.extend_from_slice(value_vars);
    }
    columns.push(Column::new(variable_name.into(), var_names));

    // The values: every column in the common type, end to end, then gathered row by row. Nulls stay
    // null and NaN stays NaN.
    let mut stacked: Option<Series> = None;
    for var in value_vars {
        let cast = df
            .column(var)?
            .strict_cast(&value_type)
            .map_err(|e| MungeError::ReshapeError(format!("cannot melt column '{var}': {e}")))?
            .take_materialized_series();
        match &mut stacked {
            None => stacked = Some(cast),
            Some(all) => {
                all.append(&cast)?;
            }
        }
    }
    let stacked = stacked.ok_or_else(|| {
        MungeError::ReshapeError("At least one value variable is required".to_string())
    })?;
    let mut order: Vec<IdxSize> =
        crate::resource::try_with_capacity(n_out, "melt row order").map_err(stopped)?;
    for i in 0..n_rows {
        if i.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(stopped)?;
        }
        for j in 0..n_value_cols {
            order.push((j * n_rows + i) as IdxSize);
        }
    }
    let values = stacked.take(&IdxCa::from_vec("".into(), order))?;
    columns.push(values.with_name(value_name.into()).into_column());

    let result = DataFrame::new(columns).map_err(|e| MungeError::ReshapeError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// The type of [`melt`]'s value column: the common type of the value columns, or an error when they
/// do not combine without changing what the values mean.
fn melt_value_type(df: &DataFrame, value_vars: &[&str]) -> MungeResult<DataType> {
    let mut distinct: Vec<&str> = Vec::with_capacity(value_vars.len());
    for v in value_vars {
        if !distinct.contains(v) {
            distinct.push(v);
        }
    }
    let selected = df.select(distinct.iter().copied())?;
    let dtypes = selected.dtypes();
    let refuse = |why: &str| {
        let list: Vec<String> = distinct
            .iter()
            .zip(&dtypes)
            .map(|(name, dt)| format!("'{name}' ({dt})"))
            .collect();
        MungeError::ReshapeError(format!(
            "cannot melt {} into one value column: {why}; cast them to one type first with \
             munge_cast",
            list.join(", ")
        ))
    };
    // Dates and datetimes are points in time and combine. The polars supertype would also take a
    // duration with a date, and read the duration's count as a date.
    let temporal_family = |dt: &DataType| match dt {
        DataType::Date | DataType::Datetime(..) => Some(0),
        DataType::Duration(..) => Some(1),
        DataType::Time => Some(2),
        _ => None,
    };
    if dtypes.iter().any(DataType::is_temporal)
        && dtypes
            .iter()
            .any(|dt| temporal_family(dt) != temporal_family(&dtypes[0]))
    {
        return Err(refuse(
            "dates, durations, times and other columns do not combine (R's tidyr::pivot_longer \
             refuses a date with anything but a datetime too)",
        ));
    }
    let common = match selected.get_supertype() {
        Some(Ok(dt)) => dt,
        _ => return Err(refuse("the types have no common type")),
    };
    let is_label = |dt: &DataType| {
        matches!(
            dt,
            DataType::String | DataType::Categorical(..) | DataType::Enum(..)
        )
    };
    if matches!(common, DataType::String) && !dtypes.iter().all(is_label) {
        return Err(refuse(
            "text and non-text columns do not combine (R's tidyr::pivot_longer refuses them too)",
        ));
    }
    if matches!(common, DataType::Categorical(..) | DataType::Enum(..))
        && dtypes.iter().any(|dt| dt != &dtypes[0])
    {
        return Ok(DataType::String);
    }
    Ok(common)
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

    /// `pivot` calls polars' `pivot_stable`, and must give exactly what the
    /// `pivot` entry point it replaced gives (the two differ only in a notice
    /// `pivot` prints to stdout): same column names, dtypes, order and cells,
    /// on shapes with several index columns, a null in `on`, a missing
    /// combination, integer and string `on` columns, and the same error for
    /// duplicated keys.
    #[test]
    fn test_pivot_matches_polars_pivot() {
        let shapes: Vec<(DataFrame, Vec<&str>, &str, &str)> = vec![
            (
                df! {
                    "id" => [1, 1, 2, 2],
                    "year" => ["2020", "2021", "2020", "2021"],
                    "value" => [100.3, 110.9, 201.7, 219.4],
                }
                .unwrap(),
                vec!["id"],
                "year",
                "value",
            ),
            // Several index columns, an integer `on` column and a missing
            // (id, grp, year) combination, which leaves a null cell.
            (
                df! {
                    "id" => [1, 1, 1, 2, 2, 2, 3],
                    "grp" => ["a", "a", "b", "a", "b", "b", "a"],
                    "year" => [2020, 2021, 2020, 2021, 2020, 2021, 2020],
                    "value" => [1.1, 2.3, 0.7, 4.2, 3.9, 5.6, 2.8],
                }
                .unwrap(),
                vec!["id", "grp"],
                "year",
                "value",
            ),
            // A null in `on`, a null value, and rows whose `on` values first
            // appear out of sorted order.
            (
                df! {
                    "id" => [2, 1, 2, 1, 3, 3],
                    "key" => [Some("z"), None, Some("a"), Some("z"), None, Some("a")],
                    "value" => [Some(5.2), Some(1.4), None, Some(3.3), Some(6.1), Some(2.9)],
                }
                .unwrap(),
                vec!["id"],
                "key",
                "value",
            ),
            // Three index columns and integer values.
            (
                df! {
                    "a" => [1, 1, 1, 1, 2, 2],
                    "b" => ["x", "x", "y", "y", "x", "x"],
                    "c" => [true, false, true, false, true, true],
                    "key" => ["p", "q", "p", "q", "p", "q"],
                    "value" => [11, 23, 7, 42, 39, 56],
                }
                .unwrap(),
                vec!["a", "b", "c"],
                "key",
                "value",
            ),
            // A duplicated (index, on) pair: both refuse it the same way.
            (
                df! {
                    "id" => [1, 1, 2],
                    "key" => ["p", "p", "q"],
                    "value" => [1.2, 3.4, 5.6],
                }
                .unwrap(),
                vec!["id"],
                "key",
                "value",
            ),
        ];

        let mut pivoted = 0;
        for (i, (df, index, on, values)) in shapes.into_iter().enumerate() {
            let polars = polars_ops::frame::pivot::pivot(
                &df,
                [on],
                Some(index.iter().copied()),
                Some([values]),
                false,
                None,
                None,
            );
            let ours = pivot(&Dataset::new(df.clone()), &index, on, values);
            match (polars, ours) {
                (Ok(expected), Ok(got)) => {
                    let got = got.df();
                    assert!(
                        got.equals_missing(&expected),
                        "shape {i}: got {got}, polars gives {expected}"
                    );
                    got.schema_equal(&expected)
                        .unwrap_or_else(|e| panic!("shape {i}: {e}"));
                    pivoted += 1;
                }
                (Err(expected), Err(MungeError::ReshapeError(got))) => {
                    assert_eq!(got, expected.to_string(), "shape {i}");
                }
                (expected, got) => panic!(
                    "shape {i}: polars gives {:?}, p2a gives {:?}",
                    expected.map(|d| d.shape()),
                    got.map(|d| d.df().shape())
                ),
            }
        }
        // Every shape but the duplicated key pivots.
        assert_eq!(pivoted, 4);
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
        assert_eq!(
            text_cells(&result, "year"),
            ["val_2020", "val_2021", "val_2020", "val_2021"].map(|v| Some(v.to_string()))
        );
        assert_eq!(
            num_cells(&result, "value"),
            [Some(100.0), Some(110.0), Some(200.0), Some(220.0)]
        );
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
        assert_eq!(
            num_cells(&result, "value"),
            [10.0, 11.0, 12.0, 20.0, 21.0, 22.0].map(Some)
        );
        assert_eq!(
            text_cells(&result, "quarter"),
            ["q1", "q2", "q3", "q1", "q2", "q3"].map(|v| Some(v.to_string()))
        );
    }

    /// The error text of a call that must fail (`Dataset` is not `Debug`).
    fn err_text(r: MungeResult<Dataset>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }
    /// Cells of any numeric/boolean column as f64 (null -> None).
    fn num_cells(ds: &Dataset, c: &str) -> Vec<Option<f64>> {
        ds.df()
            .column(c)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect()
    }
    /// Cells as text (categoricals by label; null -> None).
    fn text_cells(ds: &Dataset, c: &str) -> Vec<Option<String>> {
        ds.df()
            .column(c)
            .unwrap()
            .cast(&DataType::String)
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|v| v.map(str::to_string))
            .collect()
    }
    /// R's NA as NaN, to compare with a case vector (`Check` treats NaN as equal to NaN). Every test that
    /// uses it also asserts the null count, because NaN would pass the comparison.
    fn as_r(v: &[Option<f64>]) -> Vec<f64> {
        v.iter().map(|x| x.unwrap_or(f64::NAN)).collect()
    }
    fn dtype(ds: &Dataset, c: &str) -> DataType {
        ds.df().column(c).unwrap().dtype().clone()
    }

    #[test]
    fn test_validate_melt_pivot_longer() {
        use p2a_validation::{Check, Tol};
        // validation/r/cases/data_munging_t3.R, case data_munging_pivot_longer (tidyr::pivot_longer):
        // row-major long format; integer + double -> double; integer + integer -> integer; NA stays NA
        let (case, ds) = crate::test_support::load_case("data_munging_pivot_longer");
        let long = melt(&ds, &["id"], &["a", "b", "c"], "var", "val").unwrap();
        let int_long = melt(&ds, &["id"], &["a", "c"], "var", "val").unwrap();
        assert_eq!(dtype(&long, "val"), DataType::Float64);
        assert_eq!(dtype(&int_long, "val"), DataType::Int64);
        let val = long.df().column("val").unwrap();
        assert_eq!(val.null_count(), 2); // R: 2 NA (was 0: NaN)
        assert!(
            num_cells(&long, "val")
                .iter()
                .flatten()
                .all(|v| !v.is_nan())
        );
        assert_eq!(int_long.df().column("val").unwrap().null_count(), 1);
        let vars = |d: &Dataset| -> Vec<String> {
            text_cells(d, "var")
                .into_iter()
                .map(Option::unwrap)
                .collect()
        };
        assert_eq!(vars(&long), case.names("long_val"));
        assert_eq!(vars(&int_long), case.names("long_int_val"));
        Check::new(&case)
            .vector("long_id", &as_r(&num_cells(&long, "id")), Tol::exact())
            .vector("long_val", &as_r(&num_cells(&long, "val")), Tol::exact())
            .vector(
                "long_int_id",
                &as_r(&num_cells(&int_long, "id")),
                Tol::exact(),
            )
            .vector(
                "long_int_val",
                &as_r(&num_cells(&int_long, "val")),
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_melt_keeps_dtypes_and_nulls() {
        // DATA-15 repro: values were rebuilt as f64, so nulls became NaN and text, u32, f32 and bool columns
        // became all-NaN (val = [1, NaN, NaN, ..], null_count 0, group means NaN).
        let ds = Dataset::new(
            DataFrame::new(vec![
                Column::new("id".into(), [1i64, 2]),
                Column::new("s1".into(), [Some("x"), None]),
                Column::new("s2".into(), [None, Some("y")]),
                Column::new("u".into(), [5u32, 6]),
                Column::new("f".into(), [Some(1.5f32), None]),
                Column::new("b".into(), [Some(true), None]),
                Column::new("a".into(), [Some(1.0f64), None]),
            ])
            .unwrap(),
        );
        let m = |vars: &[&str]| melt(&ds, &["id"], vars, "var", "val").unwrap();
        // tidyr::pivot_longer(d, c(s1, s2)): "x" NA NA "y" (character; R 4.3.3, tidyr 1.3.2)
        let s = m(&["s1", "s2"]);
        assert_eq!(
            (dtype(&s, "val"), text_cells(&s, "val")),
            (
                DataType::String,
                vec![Some("x".into()), None, None, Some("y".into())]
            )
        );
        for (c, dt) in [
            ("u", DataType::UInt32),
            ("f", DataType::Float32),
            ("b", DataType::Boolean),
        ] {
            assert_eq!(dtype(&m(&[c]), "val"), dt, "{c}");
        }
        assert_eq!(m(&["b"]).df().column("val").unwrap().null_count(), 1);
        // u32 + f32 -> f64 (polars supertype; vctrs: integer + double -> double), nulls stay null
        let uf = m(&["u", "f"]);
        assert_eq!(
            (dtype(&uf, "val"), num_cells(&uf, "val")),
            (
                DataType::Float64,
                vec![Some(5.0), Some(1.5), Some(6.0), None]
            )
        );
        // the mean of 'a' over its present values is 1 (was NaN for every group)
        let au = m(&["a", "u"]);
        let mask = au.df().column("var").unwrap().str().unwrap().equal("a");
        assert_eq!(
            au.df()
                .filter(&mask)
                .unwrap()
                .column("val")
                .unwrap()
                .f64()
                .unwrap()
                .mean(),
            Some(1.0)
        );
    }

    #[test]
    fn test_melt_refuses_text_with_numbers() {
        // tidyr: pivot_longer(d, c(a, s)) -> "Can't combine `a` <integer> and `s` <character>."
        let ds = Dataset::new(
            df!("id" => [1i64, 2], "a" => [Some(1i64), None], "s" => [Some("x"), None]).unwrap(),
        );
        let e = err_text(melt(&ds, &["id"], &["a", "s"], "var", "val"));
        assert!(
            e.contains("'a' (i64)") && e.contains("'s' (str)") && e.contains("munge_cast"),
            "{e}"
        );
        // vctrs refuses a date with a number too; a polars supertype would give day counts
        let dated = Dataset::new(
            DataFrame::new(vec![
                Column::new("id".into(), [1i64, 2]),
                Int32Chunked::from_slice("d".into(), &[18262, 18263])
                    .into_date()
                    .into_column(),
                Column::new("a".into(), [1i64, 2]),
            ])
            .unwrap(),
        );
        let e = err_text(melt(&dated, &["id"], &["d", "a"], "var", "val"));
        assert!(e.contains("'d' (date)") && e.contains("munge_cast"), "{e}");
        // a duration's count is not a date, whatever the polars supertype says; nor a time of day
        let date = || {
            Int32Chunked::from_slice("d".into(), &[18262, 18263])
                .into_date()
                .into_column()
        };
        let span = |unit| {
            Int64Chunked::from_slice("span".into(), &[1_000, 2_000])
                .into_duration(unit)
                .into_column()
        };
        let moment = Int64Chunked::from_slice("at".into(), &[86_400_000, 172_800_000])
            .into_datetime(TimeUnit::Milliseconds, None)
            .into_column();
        let clock = Int64Chunked::from_slice("clock".into(), &[1_000, 2_000])
            .into_time()
            .into_column();
        let frame = Dataset::new(
            DataFrame::new(vec![
                Column::new("id".into(), [1i64, 2]),
                date(),
                moment.clone(),
                span(TimeUnit::Milliseconds),
                clock.clone(),
            ])
            .unwrap(),
        );
        for pair in [
            ["d", "span"],
            ["at", "span"],
            ["span", "clock"],
            ["d", "clock"],
        ] {
            let e = err_text(melt(&frame, &["id"], &pair, "var", "val"));
            assert!(e.contains(pair[0]) && e.contains("munge_cast"), "{e}");
        }
    }

    #[test]
    fn test_melt_combines_dates_with_datetimes() {
        let ds = Dataset::new(
            DataFrame::new(vec![
                Column::new("id".into(), [1i64, 2]),
                Int32Chunked::from_slice("d".into(), &[1, 2])
                    .into_date()
                    .into_column(),
                Int64Chunked::from_slice("at".into(), &[3_600_000, 90_000_000])
                    .into_datetime(TimeUnit::Milliseconds, None)
                    .into_column(),
            ])
            .unwrap(),
        );
        let out = melt(&ds, &["id"], &["d", "at"], "var", "val").unwrap();
        let val = out.df().column("val").unwrap();
        assert!(matches!(val.dtype(), DataType::Datetime(..)), "{val:?}");
        assert_eq!(val.len(), 4);
        assert_eq!(val.null_count(), 0);
    }
}
