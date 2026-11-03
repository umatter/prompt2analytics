//! Data munging operations: filtering, joining, reshaping, cleaning, aggregation, and feature engineering.
//!
//! This module provides a comprehensive set of data manipulation functions
//! for preparing datasets before analysis. All functions follow the immutable
//! pattern, returning new datasets rather than modifying in place.
//!
//! # Modules
//!
//! - [`transform`] - Core transforms: filter, select, rename, mutate, sort
//! - [`clean`] - Data cleaning: drop_na, fill_na, deduplicate, cast (reports values that fail to convert)
//! - [`join`](mod@join) - Join operations: left/right/inner/full joins, concat
//! - [`aggregate`] - Aggregation: group_by, value_counts, summarize, describe
//! - [`reshape`] - Reshape: pivot, melt, transpose, explode, stack
//! - [`features`] - Feature engineering: lag, lead, diff, standardize, bin, one_hot
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::data::munging::*;
//!
//! // Filter and select
//! let filtered = filter(&dataset, "age", "ge", "18")?;
//! let selected = select(&filtered, &["id", "name", "age"])?;
//!
//! // Clean data
//! let clean = drop_na(&selected, None, "any")?;
//! let filled = fill_na(&dataset, Some(&["income"]), FillStrategy::Mean)?;
//!
//! // Transform
//! let renamed = rename(&clean, &[("id", "user_id")])?;
//! let computed = mutate(&renamed, "age_squared", MutateExpr::Function("square".into(), "age".into()))?;
//! ```

pub mod aggregate;
pub mod clean;
pub mod error;
pub mod features;
pub mod join;
pub mod reshape;
pub mod transform;

/// Loop iterations between two cooperative-cancellation checkpoints
/// ([`crate::resource::checkpoint`]) in the munging loops.
const CHECKPOINT_EVERY: usize = 65_536;

/// The numeric column `column` as Float64 (nulls and NaN as they are), by the
/// strict rule of [`crate::data::columns`]: integers, floats and decimals,
/// booleans as 0/1 when `allow_bool`; a text column is refused, never cast to
/// nulls. A column whose cells are all null is accepted whatever its dtype (an
/// all-empty CSV column is text).
fn numeric_f64(
    df: &polars::prelude::DataFrame,
    column: &str,
    allow_bool: bool,
) -> MungeResult<polars::prelude::Float64Chunked> {
    let col = crate::data::columns::column(df, column)
        .map_err(|_| MungeError::ColumnNotFound(column.to_string()))?;
    crate::data::columns::float64_strict(col, allow_bool).ok_or_else(|| MungeError::TypeMismatch {
        column: column.to_string(),
        expected: "numeric (convert text with munge_cast first)".to_string(),
        found: col.dtype().to_string(),
    })
}

/// The values of the numeric column `column` ([`numeric_f64`]'s rule), `None`
/// where a value is missing (null, or NaN in a float column; R's `is.na`).
/// Infinite values are kept.
fn numeric_cells(
    df: &polars::prelude::DataFrame,
    column: &str,
    allow_bool: bool,
) -> MungeResult<Vec<Option<f64>>> {
    let ca = numeric_f64(df, column, allow_bool)?;
    let stopped = |e: crate::errors::EconError| MungeError::FeatureError(e.to_string());
    let mut cells =
        crate::resource::try_with_capacity(ca.len(), "column values").map_err(stopped)?;
    for (i, v) in ca.iter().enumerate() {
        if i.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(stopped)?;
        }
        cells.push(v.filter(|x| !x.is_nan()));
    }
    Ok(cells)
}

// Re-export error types
pub use error::{MungeError, MungeResult};

// Re-export transform operations
pub use transform::{
    ArithOp, FilterOp, MutateExpr, drop_columns, filter, filter_and, head, mutate, rename, sample,
    sample_rows, select, slice, sort, tail, take_rows,
};

// Re-export clean operations
pub use clean::{
    CastOptions,
    CastReport,
    FillStrategy,
    cast,
    cast_columns,
    cast_columns_with,
    cast_with,
    clip,
    deduplicate,
    drop_na,
    fill_na,
    parse_cast_dtype,
    regex_count,
    regex_extract,
    regex_extract_all,
    // Regex and advanced string operations
    regex_replace,
    regex_replace_all,
    regex_replace_bytes,
    replace,
    str_concat,
    str_length,
    str_pad,
    str_split,
    str_split_part_count,
    str_substring,
    to_lowercase,
    to_uppercase,
    trim,
};

// Re-export join operations
pub use join::{
    JoinOptions, JoinShape, JoinType, anti_join, concat, cross_join, cross_join_with,
    estimate_join_rows, estimate_join_shape, full_join, hconcat, inner_join, join, join_with,
    left_join, right_join, semi_join,
};

// Re-export aggregate operations
pub use aggregate::{AggFn, AggSpec, describe, group_by, summarize, value_counts};

// Re-export reshape operations
pub use reshape::{explode, melt, pivot, stack, transpose};

// Re-export feature engineering operations
pub use features::{
    BinOptions, BinOutcome, BinStrategy, bin, bin_with, cumprod, cumsum, diff, lag, lead,
    normalize, one_hot_encode, pct_change, rank, standardize,
};
