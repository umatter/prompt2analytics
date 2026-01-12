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

mod error;
mod transform;
mod clean;
mod join;
mod aggregate;
mod reshape;
mod features;

// Re-export error types
pub use error::{MungeError, MungeResult};

// Re-export transform operations
pub use transform::{
    filter, filter_and, select, drop_columns, rename, mutate,
    sort, head, tail, slice, sample,
    FilterOp, MutateExpr, ArithOp,
};

// Re-export clean operations
pub use clean::{
    drop_na, fill_na, deduplicate, cast, cast_columns,
    clip, replace, trim, to_lowercase, to_uppercase,
    // Regex and advanced string operations
    regex_replace, regex_replace_all, regex_extract, regex_extract_all, regex_count,
    str_split, str_concat, str_pad, str_substring, str_length,
    FillStrategy,
};

// Re-export join operations
pub use join::{
    join, left_join, right_join, inner_join, full_join, anti_join, semi_join,
    concat, hconcat, cross_join,
    JoinType,
};

// Re-export aggregate operations
pub use aggregate::{
    group_by, value_counts, summarize, describe,
    AggFn, AggSpec,
};

// Re-export reshape operations
pub use reshape::{
    pivot, melt, transpose, explode, stack,
};

// Re-export feature engineering operations
pub use features::{
    lag, lead, diff, pct_change,
    standardize, normalize,
    bin, one_hot_encode,
    cumsum, cumprod, rank,
    BinStrategy,
};
