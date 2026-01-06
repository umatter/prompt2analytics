//! Statistical analysis module.
//!
//! Provides descriptive statistics, correlation analysis, and hypothesis tests.

mod descriptive;
mod correlation;

pub use descriptive::{DescriptiveStats, ColumnStats};
pub use correlation::{correlation_matrix, CorrelationMatrix};
