//! Error types for data munging operations.

use polars::prelude::PolarsError;
use thiserror::Error;

/// Error type for data munging operations.
#[derive(Debug, Error)]
pub enum MungeError {
    #[error("Column '{0}' not found in dataset")]
    ColumnNotFound(String),

    #[error("Column '{column}' has incompatible type. Expected: {expected}, found: {found}")]
    TypeMismatch {
        column: String,
        expected: String,
        found: String,
    },

    #[error("Invalid operator '{0}'. Valid operators: {ops}", ops = filter_operators())]
    InvalidOperator(String),

    #[error("Invalid expression: {0}")]
    InvalidExpression(String),

    #[error("Invalid value '{value}' for column '{column}': {reason}")]
    InvalidValue {
        column: String,
        value: String,
        reason: String,
    },

    #[error("Join failed: {0}")]
    JoinError(String),

    #[error("Reshape failed: {0}")]
    ReshapeError(String),

    #[error("Aggregation failed: {0}")]
    AggregationError(String),

    #[error("Feature engineering failed: {0}")]
    FeatureError(String),

    #[error("Empty result after operation")]
    EmptyResult,

    #[error("Dataset is empty")]
    EmptyDataset,

    #[error("Column count mismatch: expected {expected}, found {found}")]
    ColumnCountMismatch { expected: usize, found: usize },

    #[error("Row count mismatch: expected {expected}, found {found}")]
    RowCountMismatch { expected: usize, found: usize },

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),
}

/// The operators [`super::transform::filter`] accepts, for the [`MungeError::InvalidOperator`]
/// message.
fn filter_operators() -> String {
    super::transform::FilterOp::NAMES.join(", ")
}

/// Result type for data munging operations.
pub type MungeResult<T> = Result<T, MungeError>;
