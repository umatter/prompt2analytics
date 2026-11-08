//! Aggregation operations for summarizing datasets.
//!
//! This module provides grouping and aggregation operations including
//! group_by with various aggregation functions and value_counts.
//!
//! # Aggregation Functions
//!
//! - **Count**: Number of non-null values
//! - **Sum**: Sum of values
//! - **Mean**: Arithmetic mean
//! - **Median**: Median value
//! - **Min**: Minimum value
//! - **Max**: Maximum value
//! - **Std**: Standard deviation
//! - **Var**: Variance
//! - **First**: First value in group
//! - **Last**: Last value in group
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::data::munging::*;
//!
//! // Group by region and compute aggregations
//! let summary = group_by(&sales, &["region"], &[
//!     AggSpec::new("revenue", AggFn::Sum),
//!     AggSpec::new("orders", AggFn::Count),
//!     AggSpec::new("avg_order", AggFn::Mean),
//! ])?;
//!
//! // Get value counts for a column
//! let counts = value_counts(&data, "category")?;
//! ```

use super::error::{MungeError, MungeResult};
use crate::data::Dataset;
use polars::prelude::*;

/// Aggregation function to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFn {
    /// Count non-null values
    Count,
    /// Sum of values
    Sum,
    /// Arithmetic mean
    Mean,
    /// Median value
    Median,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Standard deviation (sample)
    Std,
    /// Variance (sample)
    Var,
    /// First value in group
    First,
    /// Last value in group
    Last,
    /// Number of unique values
    NUnique,
}

/// Specification for an aggregation operation.
#[derive(Debug, Clone)]
pub struct AggSpec {
    /// Column to aggregate
    pub column: String,
    /// Aggregation function to apply
    pub agg_fn: AggFn,
    /// Optional alias for the result column
    pub alias: Option<String>,
}

impl AggSpec {
    /// Create a new aggregation specification.
    pub fn new(column: impl Into<String>, agg_fn: AggFn) -> Self {
        Self {
            column: column.into(),
            agg_fn,
            alias: None,
        }
    }

    /// Set an alias for the result column.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// Get the default alias for this aggregation.
    fn default_alias(&self) -> String {
        let fn_name = match self.agg_fn {
            AggFn::Count => "count",
            AggFn::Sum => "sum",
            AggFn::Mean => "mean",
            AggFn::Median => "median",
            AggFn::Min => "min",
            AggFn::Max => "max",
            AggFn::Std => "std",
            AggFn::Var => "var",
            AggFn::First => "first",
            AggFn::Last => "last",
            AggFn::NUnique => "nunique",
        };
        format!("{}_{}", self.column, fn_name)
    }

    /// Get the result column name.
    pub fn result_name(&self) -> String {
        self.alias.clone().unwrap_or_else(|| self.default_alias())
    }
}

/// Group by columns and apply aggregations.
///
/// Groups the dataset by the specified columns and applies aggregation
/// functions to compute summary statistics for each group.
///
/// # Arguments
///
/// * `dataset` - Dataset to aggregate
/// * `by` - Columns to group by
/// * `aggs` - Aggregation specifications to apply
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::{group_by, AggSpec, AggFn};
///
/// let summary = group_by(&sales, &["region", "year"], &[
///     AggSpec::new("revenue", AggFn::Sum).with_alias("total_revenue"),
///     AggSpec::new("revenue", AggFn::Mean).with_alias("avg_revenue"),
///     AggSpec::new("orders", AggFn::Count),
/// ])?;
/// ```
pub fn group_by(dataset: &Dataset, by: &[&str], aggs: &[AggSpec]) -> MungeResult<Dataset> {
    // Validate group by columns exist
    for col in by {
        if dataset.df().column(col).is_err() {
            return Err(MungeError::ColumnNotFound(col.to_string()));
        }
    }

    // Validate aggregation columns exist
    for agg in aggs {
        if dataset.df().column(&agg.column).is_err() {
            return Err(MungeError::ColumnNotFound(agg.column.clone()));
        }
    }

    if aggs.is_empty() {
        return Err(MungeError::AggregationError(
            "At least one aggregation must be specified".to_string(),
        ));
    }

    // Convert to lazy frame for aggregation
    let lazy = dataset.df().clone().lazy();

    // Build group by expression
    let by_exprs: Vec<Expr> = by.iter().map(|c| col(*c)).collect();

    // Build aggregation expressions
    let agg_exprs: Vec<Expr> = aggs
        .iter()
        .map(|spec| {
            let base_expr = col(&spec.column);
            let agg_expr = match spec.agg_fn {
                AggFn::Count => base_expr.count(),
                AggFn::Sum => base_expr.sum(),
                AggFn::Mean => base_expr.mean(),
                AggFn::Median => base_expr.median(),
                AggFn::Min => base_expr.min(),
                AggFn::Max => base_expr.max(),
                AggFn::Std => base_expr.std(1), // ddof=1 for sample std
                AggFn::Var => base_expr.var(1), // ddof=1 for sample var
                AggFn::First => base_expr.first(),
                AggFn::Last => base_expr.last(),
                AggFn::NUnique => base_expr.n_unique(),
            };
            agg_expr.alias(spec.result_name())
        })
        .collect();

    // Execute groupby
    let result = lazy
        .group_by(by_exprs)
        .agg(agg_exprs)
        .collect()
        .map_err(|e| MungeError::AggregationError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Compute value counts for a column.
///
/// Returns a dataset with each unique value and its count, sorted by
/// count in descending order.
///
/// # Arguments
///
/// * `dataset` - Dataset to analyze
/// * `column` - Column to count values for
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::value_counts;
///
/// let counts = value_counts(&data, "category")?;
/// // Returns: category | count
/// //          A        | 150
/// //          B        | 100
/// //          C        | 50
/// ```
pub fn value_counts(dataset: &Dataset, column: &str) -> MungeResult<Dataset> {
    value_counts_with(dataset, column, false)
}

/// [`value_counts`], with a Float64 column "proportion" (the count over all rows: missing values are
/// a group of their own and count in the denominator) instead of "count" when `normalize`, as
/// `dplyr::count(d, x, sort = TRUE) |> mutate(proportion = n / sum(n))`.
///
/// Rows run from the largest count down; tied counts in the order their values first appear in the
/// column, so the order does not change between calls. `dplyr::count(sort = TRUE)` orders tied counts
/// by value instead: sorting every group by value costs more than the count itself on columns where
/// most values are distinct, so tie order is a recorded deviation from dplyr.
pub fn value_counts_with(dataset: &Dataset, column: &str, normalize: bool) -> MungeResult<Dataset> {
    // Validate column exists
    if dataset.df().column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }
    let value_column = if normalize { "proportion" } else { "count" };

    let aggregation = |e: PolarsError| MungeError::AggregationError(e.to_string());
    let source = dataset.df().column(column).map_err(aggregation)?;
    let source = source.as_materialized_series();

    // Sorted group tuples come in order of first appearance; the stable sort by count keeps that
    // order among tied counts. Missing values are a group of their own.
    let groups = source.group_tuples(true, true).map_err(aggregation)?;
    // SAFETY: the groups were computed from `source` itself, so every index is in bounds.
    let values: Column = unsafe { source.agg_first(&groups) }.into();
    let counts = groups.group_count();
    let order = counts.arg_sort(SortOptions {
        descending: true,
        nulls_last: true,
        maintain_order: true,
        multithreaded: true,
        ..Default::default()
    });
    let counts = counts.take(&order).map_err(aggregation)?;
    let tallies = if normalize {
        let rows = source.len() as f64;
        let shares: Float64Chunked =
            counts.apply_nonnull_values_generic(DataType::Float64, |n| n as f64 / rows);
        shares.with_name(value_column.into()).into_column()
    } else {
        counts.with_name(value_column.into()).into_column()
    };
    let sorted = DataFrame::new(vec![values.take(&order).map_err(aggregation)?, tallies])
        .map_err(aggregation)?;

    Ok(Dataset::new(sorted))
}

/// Compute multiple aggregations without grouping (overall summary).
///
/// # Arguments
///
/// * `dataset` - Dataset to summarize
/// * `aggs` - Aggregation specifications to apply
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::{summarize, AggSpec, AggFn};
///
/// let summary = summarize(&sales, &[
///     AggSpec::new("revenue", AggFn::Sum),
///     AggSpec::new("revenue", AggFn::Mean),
///     AggSpec::new("orders", AggFn::Count),
/// ])?;
/// ```
pub fn summarize(dataset: &Dataset, aggs: &[AggSpec]) -> MungeResult<Dataset> {
    // Validate aggregation columns exist
    for agg in aggs {
        if dataset.df().column(&agg.column).is_err() {
            return Err(MungeError::ColumnNotFound(agg.column.clone()));
        }
    }

    if aggs.is_empty() {
        return Err(MungeError::AggregationError(
            "At least one aggregation must be specified".to_string(),
        ));
    }

    // Convert to lazy frame for aggregation
    let lazy = dataset.df().clone().lazy();

    // Build aggregation expressions
    let agg_exprs: Vec<Expr> = aggs
        .iter()
        .map(|spec| {
            let base_expr = col(&spec.column);
            let agg_expr = match spec.agg_fn {
                AggFn::Count => base_expr.count(),
                AggFn::Sum => base_expr.sum(),
                AggFn::Mean => base_expr.mean(),
                AggFn::Median => base_expr.median(),
                AggFn::Min => base_expr.min(),
                AggFn::Max => base_expr.max(),
                AggFn::Std => base_expr.std(1),
                AggFn::Var => base_expr.var(1),
                AggFn::First => base_expr.first(),
                AggFn::Last => base_expr.last(),
                AggFn::NUnique => base_expr.n_unique(),
            };
            agg_expr.alias(spec.result_name())
        })
        .collect();

    // Execute aggregation
    let result = lazy
        .select(agg_exprs)
        .collect()
        .map_err(|e| MungeError::AggregationError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Compute descriptive statistics for numeric columns.
///
/// Returns count, mean, std, min, max for each column.
///
/// # Arguments
///
/// * `dataset` - Dataset to describe
/// * `columns` - Optional specific columns to describe (defaults to all numeric)
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::describe;
///
/// let stats = describe(&data, None)?;
/// ```
pub fn describe(dataset: &Dataset, columns: Option<&[&str]>) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Determine which columns to describe
    let cols_to_describe: Vec<String> = match columns {
        Some(cols) => {
            // Validate all specified columns exist and are numeric
            for col_name in cols {
                let column = df
                    .column(col_name)
                    .map_err(|_| MungeError::ColumnNotFound(col_name.to_string()))?;
                if !column.dtype().is_primitive_numeric() {
                    return Err(MungeError::TypeMismatch {
                        column: col_name.to_string(),
                        expected: "numeric".to_string(),
                        found: column.dtype().to_string(),
                    });
                }
            }
            cols.iter().map(|s| s.to_string()).collect()
        }
        None => {
            // Find all numeric columns
            df.get_columns()
                .iter()
                .filter(|c| c.dtype().is_primitive_numeric())
                .map(|c| c.name().to_string())
                .collect()
        }
    };

    if cols_to_describe.is_empty() {
        return Err(MungeError::AggregationError(
            "No numeric columns to describe".to_string(),
        ));
    }

    // Build aggregation expressions for each column
    let lazy = df.clone().lazy();

    // Statistics: count, mean, std, min, max for each column
    let mut agg_exprs: Vec<Expr> = Vec::new();
    for col_name in &cols_to_describe {
        agg_exprs.push(col(col_name).count().alias(format!("{}_count", col_name)));
        agg_exprs.push(col(col_name).mean().alias(format!("{}_mean", col_name)));
        agg_exprs.push(col(col_name).std(1).alias(format!("{}_std", col_name)));
        agg_exprs.push(col(col_name).min().alias(format!("{}_min", col_name)));
        agg_exprs.push(col(col_name).max().alias(format!("{}_max", col_name)));
    }

    let result = lazy
        .select(agg_exprs)
        .collect()
        .map_err(|e| MungeError::AggregationError(e.to_string()))?;

    Ok(Dataset::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    fn make_sales_data() -> Dataset {
        let df = df! {
            "region" => ["North", "North", "South", "South", "East"],
            "product" => ["A", "B", "A", "B", "A"],
            "revenue" => [100.0, 200.0, 150.0, 250.0, 175.0],
            "orders" => [10, 20, 15, 25, 18],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_group_by_single_column() {
        let data = make_sales_data();

        let result = group_by(
            &data,
            &["region"],
            &[
                AggSpec::new("revenue", AggFn::Sum),
                AggSpec::new("orders", AggFn::Count),
            ],
        )
        .unwrap();

        // 3 unique regions
        assert_eq!(result.nrows(), 3);
        assert!(result.df().column("region").is_ok());
        assert!(result.df().column("revenue_sum").is_ok());
        assert!(result.df().column("orders_count").is_ok());
    }

    #[test]
    fn test_group_by_multiple_columns() {
        let data = make_sales_data();

        let result = group_by(
            &data,
            &["region", "product"],
            &[AggSpec::new("revenue", AggFn::Mean)],
        )
        .unwrap();

        // North-A, North-B, South-A, South-B, East-A = 5 groups
        assert_eq!(result.nrows(), 5);
    }

    #[test]
    fn test_group_by_with_alias() {
        let data = make_sales_data();

        let result = group_by(
            &data,
            &["region"],
            &[AggSpec::new("revenue", AggFn::Sum).with_alias("total_revenue")],
        )
        .unwrap();

        assert!(result.df().column("total_revenue").is_ok());
    }

    #[test]
    fn test_group_by_all_agg_functions() {
        let data = make_sales_data();

        let result = group_by(
            &data,
            &["region"],
            &[
                AggSpec::new("revenue", AggFn::Count),
                AggSpec::new("revenue", AggFn::Sum),
                AggSpec::new("revenue", AggFn::Mean),
                AggSpec::new("revenue", AggFn::Min),
                AggSpec::new("revenue", AggFn::Max),
            ],
        )
        .unwrap();

        assert!(result.df().column("revenue_count").is_ok());
        assert!(result.df().column("revenue_sum").is_ok());
        assert!(result.df().column("revenue_mean").is_ok());
        assert!(result.df().column("revenue_min").is_ok());
        assert!(result.df().column("revenue_max").is_ok());
    }

    #[test]
    fn test_group_by_column_not_found() {
        let data = make_sales_data();

        let result = group_by(
            &data,
            &["nonexistent"],
            &[AggSpec::new("revenue", AggFn::Sum)],
        );
        assert!(matches!(result, Err(MungeError::ColumnNotFound(_))));
    }

    #[test]
    fn test_value_counts() {
        let data = make_sales_data();

        let result = value_counts(&data, "region").unwrap();

        // 3 unique regions
        assert_eq!(result.nrows(), 3);
        assert!(result.df().column("region").is_ok());
        assert!(result.df().column("count").is_ok());
    }

    #[test]
    fn test_value_counts_column_not_found() {
        let data = make_sales_data();

        let result = value_counts(&data, "nonexistent");
        assert!(matches!(result, Err(MungeError::ColumnNotFound(_))));
    }

    #[test]
    fn test_summarize() {
        let data = make_sales_data();

        let result = summarize(
            &data,
            &[
                AggSpec::new("revenue", AggFn::Sum),
                AggSpec::new("revenue", AggFn::Mean),
                AggSpec::new("orders", AggFn::Max),
            ],
        )
        .unwrap();

        // Single row with aggregated values
        assert_eq!(result.nrows(), 1);

        // Check sum
        let sum = result
            .df()
            .column("revenue_sum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        assert!((sum - 875.0).abs() < 0.01); // 100 + 200 + 150 + 250 + 175 = 875
    }

    #[test]
    fn test_describe_all_numeric() {
        let data = make_sales_data();

        let result = describe(&data, None).unwrap();

        // Should describe revenue and orders (the numeric columns)
        assert!(result.nrows() >= 1);
    }

    #[test]
    fn test_describe_non_numeric_error() {
        let data = make_sales_data();

        let result = describe(&data, Some(&["region"]));
        assert!(matches!(result, Err(MungeError::TypeMismatch { .. })));
    }

    #[test]
    fn test_value_counts_normalize() {
        // DATA-65 (core): normalize did not exist; the MCP field was ignored.
        // R: prop.table(table(c("a","b","a","c","a","b"))) = 0.5, 0.3333333, 0.1666667
        let ds = Dataset::new(df!("g" => ["a", "b", "a", "c", "a", "b"]).unwrap());
        let p = value_counts_with(&ds, "g", true).unwrap();
        let names: Vec<String> = p
            .df()
            .get_column_names()
            .iter()
            .map(|n| n.to_string())
            .collect();
        assert_eq!(names, ["g", "proportion"]);
        assert_eq!(
            p.df()
                .column("g")
                .unwrap()
                .str()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            [Some("a"), Some("b"), Some("c")]
        );
        assert_eq!(
            p.df()
                .column("proportion")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            [0.5, 2.0 / 6.0, 1.0 / 6.0]
        );
        // Tied counts come in the order their values first appear ("b", "a", then "c" and the missing
        // value), the same on every call. dplyr::count(x, sort = TRUE) gives "a", "b", "c", NA.
        let tied = Dataset::new(
            df!("g" => [Some("b"), Some("a"), Some("b"), Some("a"), Some("c"), None]).unwrap(),
        );
        for _ in 0..20 {
            let c = value_counts_with(&tied, "g", false).unwrap();
            assert_eq!(
                c.df()
                    .column("g")
                    .unwrap()
                    .str()
                    .unwrap()
                    .into_iter()
                    .collect::<Vec<_>>(),
                [Some("b"), Some("a"), Some("c"), None]
            );
            assert_eq!(
                c.df()
                    .column("count")
                    .unwrap()
                    .cast(&DataType::Int64)
                    .unwrap()
                    .i64()
                    .unwrap()
                    .into_no_null_iter()
                    .collect::<Vec<_>>(),
                [2, 2, 1, 1]
            );
            let t = value_counts_with(&tied, "g", true).unwrap();
            assert_eq!(
                t.df()
                    .column("g")
                    .unwrap()
                    .str()
                    .unwrap()
                    .into_iter()
                    .collect::<Vec<_>>(),
                [Some("b"), Some("a"), Some("c"), None]
            );
            assert_eq!(
                t.df()
                    .column("proportion")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_no_null_iter()
                    .collect::<Vec<_>>(),
                [2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
            );
        }
        assert!(
            value_counts_with(&ds, "g", false)
                .unwrap()
                .df()
                .equals_missing(value_counts(&ds, "g").unwrap().df())
        );
    }

    #[test]
    fn test_value_counts_order_matches_a_naive_tally_on_many_groups() {
        // Enough groups for the grouping and the sort to run on the thread pool, with many tied counts
        // and a missing group; the expected order is a plain tally in order of first appearance,
        // stable-sorted by count.
        let mut state = 7u64;
        let keys: Vec<Option<i64>> = (0..20_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let draw = (state >> 40) % 4_001;
                (draw != 4_000).then_some(draw as i64)
            })
            .collect();
        let labels: Vec<Option<String>> = keys
            .iter()
            .map(|k| k.map(|k| format!("user_{k}")))
            .collect();

        fn expected<K: Ord + Clone>(cells: &[Option<K>]) -> Vec<(Option<K>, u32)> {
            let mut slot = std::collections::BTreeMap::new();
            let mut rows: Vec<(Option<K>, u32)> = Vec::new();
            for cell in cells {
                let at = *slot.entry(cell.clone()).or_insert_with(|| {
                    rows.push((cell.clone(), 0));
                    rows.len() - 1
                });
                rows[at].1 += 1;
            }
            rows.sort_by_key(|row| std::cmp::Reverse(row.1)); // stable: first appearance among ties
            rows
        }
        let tallies = |c: &Dataset| -> Vec<u32> {
            c.df()
                .column("count")
                .unwrap()
                .cast(&DataType::UInt32)
                .unwrap()
                .u32()
                .unwrap()
                .into_no_null_iter()
                .collect()
        };

        let ints = Dataset::new(df!("g" => &keys).unwrap());
        let (want_keys, want_counts): (Vec<_>, Vec<_>) = expected(&keys).into_iter().unzip();
        let got = value_counts(&ints, "g").unwrap();
        let got_keys: Vec<Option<i64>> = got
            .df()
            .column("g")
            .unwrap()
            .i64()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(got_keys, want_keys);
        assert_eq!(tallies(&got), want_counts);

        let text = Dataset::new(df!("g" => &labels).unwrap());
        let (want_labels, want_counts): (Vec<_>, Vec<_>) = expected(&labels).into_iter().unzip();
        let got = value_counts(&text, "g").unwrap();
        let got_labels: Vec<Option<String>> = got
            .df()
            .column("g")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|cell| cell.map(str::to_string))
            .collect();
        assert_eq!(got_labels, want_labels);
        assert_eq!(tallies(&got), want_counts);
    }

    #[test]
    fn test_value_counts_empty_and_all_missing_columns() {
        let empty = Dataset::new(df!("g" => Vec::<i64>::new()).unwrap());
        for normalize in [false, true] {
            assert_eq!(
                value_counts_with(&empty, "g", normalize).unwrap().nrows(),
                0
            );
        }
        let missing = Dataset::new(df!("g" => [None::<&str>, None, None]).unwrap());
        let c = value_counts_with(&missing, "g", false).unwrap();
        assert_eq!(c.nrows(), 1);
        assert_eq!(c.df().column("g").unwrap().null_count(), 1);
        let n = c
            .df()
            .column("count")
            .unwrap()
            .cast(&DataType::Int64)
            .unwrap();
        assert_eq!(n.i64().unwrap().get(0), Some(3));
        let p = value_counts_with(&missing, "g", true).unwrap();
        assert_eq!(
            p.df().column("proportion").unwrap().f64().unwrap().get(0),
            Some(1.0)
        );
    }

    #[test]
    fn test_value_counts_orders_tied_categories_by_first_appearance() {
        // "a" and "b" tie; "a" comes first in the data, as does its row of the result. The global
        // mapping may already hold "b" with a smaller code (another test), which must not decide.
        let c = Column::new("c".into(), ["a", "b", "b", "a", "z"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let ds = Dataset::new(DataFrame::new(vec![c]).unwrap());
        let out = value_counts_with(&ds, "c", false).unwrap();
        let labels = out
            .df()
            .column("c")
            .unwrap()
            .cast(&DataType::String)
            .unwrap();
        assert_eq!(
            labels.str().unwrap().into_iter().collect::<Vec<_>>(),
            [Some("a"), Some("b"), Some("z")]
        );
        let later = Column::new("c".into(), ["b", "a", "b", "a", "z"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let out = value_counts_with(
            &Dataset::new(DataFrame::new(vec![later]).unwrap()),
            "c",
            false,
        )
        .unwrap();
        let labels = out
            .df()
            .column("c")
            .unwrap()
            .cast(&DataType::String)
            .unwrap();
        assert_eq!(
            labels.str().unwrap().into_iter().collect::<Vec<_>>(),
            [Some("b"), Some("a"), Some("z")]
        );
    }

    #[test]
    fn test_validate_value_counts() {
        use p2a_validation::{Check, Tol};
        // validation/r/cases/data_munging_t3.R, case data_munging_value_counts:
        // dplyr::count(d, g, sort = TRUE) (NA is a group) and n / sum(n); counts 4, 3, 2, 1 are distinct
        // deviation DATA-65: tied counts follow first appearance, not dplyr's value order; the case has
        // no tie (the R case stops on one), so tie order stays out of the Check.
        let (case, ds) = crate::test_support::load_case("data_munging_value_counts");
        let f = |d: &Dataset, c: &str| -> Vec<f64> {
            d.df()
                .column(c)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect()
        };
        let c = value_counts_with(&ds, "g", false).unwrap();
        let p = value_counts_with(&ds, "g", true).unwrap();
        Check::new(&case)
            .vector("vc_g", &f(&c, "g"), Tol::exact())
            .vector("vc_n", &f(&c, "count"), Tol::exact())
            .vector("vc_proportion", &f(&p, "proportion"), Tol::exact())
            .finish();
        assert_eq!(f(&p, "g")[..2], f(&c, "g")[..2]);
        assert_eq!(
            (
                c.df().column("g").unwrap().null_count(),
                p.df().column("g").unwrap().null_count()
            ),
            (1, 1)
        );
    }

    #[test]
    fn test_validate_describe() {
        use p2a_validation::{Check, Tol};
        // case data_munging_describe: describe returns ONE row with <col>_count/_mean/_std/_min/_max;
        // R: sum(!is.na(x)), mean/sd/min/max(na.rm = TRUE) on an Int64 and a Float64 column with NA
        let (case, ds) = crate::test_support::load_case("data_munging_describe");
        let d = describe(&ds, Some(&["k", "f"])).unwrap();
        assert_eq!(d.nrows(), 1);
        let names = ["count", "mean", "std", "min", "max"];
        let stats = |c: &str| -> Vec<f64> {
            names
                .iter()
                .map(|s| {
                    d.df()
                        .column(&format!("{c}_{s}"))
                        .unwrap()
                        .cast(&DataType::Float64)
                        .unwrap()
                        .f64()
                        .unwrap()
                        .get(0)
                        .unwrap()
                })
                .collect()
        };
        // 8 non-missing values each; mean and sd at the README tolerances, the rest exact
        let m = 8;
        let (k, f) = (stats("k"), stats("f"));
        Check::new(&case)
            .named_subset(
                "describe_k",
                &["count", "min", "max"],
                &names,
                &k,
                Tol::exact(),
            )
            .named_subset("describe_k", &["mean"], &names, &k, Tol::coef(m))
            .named_subset("describe_k", &["std"], &names, &k, Tol::stat(m))
            .named_subset(
                "describe_f",
                &["count", "min", "max"],
                &names,
                &f,
                Tol::exact(),
            )
            .named_subset("describe_f", &["mean"], &names, &f, Tol::coef(m))
            .named_subset("describe_f", &["std"], &names, &f, Tol::stat(m))
            .finish();
    }
}
