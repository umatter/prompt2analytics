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
    // Validate column exists
    if dataset.df().column(column).is_err() {
        return Err(MungeError::ColumnNotFound(column.to_string()));
    }

    // Use lazy frame for value_counts
    let lazy = dataset.df().clone().lazy();

    let result = lazy
        .select([col(column).value_counts(true, true, "count", false)])
        .collect()
        .map_err(|e| MungeError::AggregationError(e.to_string()))?;

    // The result is a struct column, we need to unnest it
    let unnested = result
        .unnest([column], None)
        .map_err(|e| MungeError::AggregationError(e.to_string()))?;

    Ok(Dataset::new(unnested))
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
    fn test_describe() {
        let data = make_sales_data();

        let result = describe(&data, Some(&["revenue", "orders"])).unwrap();

        // describe returns multiple rows (count, mean, std, min, 25%, 50%, 75%, max)
        assert!(result.nrows() >= 1);
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
}
