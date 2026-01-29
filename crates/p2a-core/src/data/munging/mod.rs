//! Data munging operations: filtering, joining, reshaping, cleaning, aggregation, and feature engineering.
//!
//! This module provides a comprehensive set of data manipulation functions
//! for preparing datasets before analysis. All functions follow the immutable
//! pattern, returning new datasets rather than modifying in place.
//!
//! # Modules
//!
//! - [`transform`] - Core transforms: filter, select, rename, mutate, sort
//! - [`clean`] - Data cleaning: drop_na, fill_na, deduplicate, cast
//! - [`join`] - Join operations: left/right/inner/full joins, concat
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

mod aggregate;
mod clean;
mod error;
mod features;
mod join;
mod reshape;
mod transform;

// Re-export error types
pub use error::{MungeError, MungeResult};

// Re-export transform operations
pub use transform::{
    ArithOp, FilterOp, MutateExpr, drop_columns, filter, filter_and, head, mutate, rename, sample,
    select, slice, sort, tail,
};

// Re-export clean operations
pub use clean::{
    FillStrategy,
    cast,
    cast_columns,
    clip,
    deduplicate,
    drop_na,
    fill_na,
    regex_count,
    regex_extract,
    regex_extract_all,
    // Regex and advanced string operations
    regex_replace,
    regex_replace_all,
    replace,
    str_concat,
    str_length,
    str_pad,
    str_split,
    str_substring,
    to_lowercase,
    to_uppercase,
    trim,
};

// Re-export join operations
pub use join::{
    JoinType, anti_join, concat, cross_join, full_join, hconcat, inner_join, join, left_join,
    right_join, semi_join,
};

// Re-export aggregate operations
pub use aggregate::{AggFn, AggSpec, describe, group_by, summarize, value_counts};

// Re-export reshape operations
pub use reshape::{explode, melt, pivot, stack, transpose};

// Re-export feature engineering operations
pub use features::{
    BinStrategy, bin, cumprod, cumsum, diff, lag, lead, normalize, one_hot_encode, pct_change,
    rank, standardize,
};
