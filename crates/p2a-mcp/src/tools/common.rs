//! Common utilities and helpers shared across all MCP tool handlers.
//!
//! This module provides:
//! - Helper functions for creating tool results
//! - Common re-exports for tool handlers
//! - Shared request type patterns

use rmcp::model::*;

/// MCP error type alias for convenience.
pub type McpError = rmcp::ErrorData;

/// Create a successful tool result with text content.
pub fn success_text(msg: impl ToString) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![Content::text(msg.to_string())]))
}

/// Create an error tool result with text content.
pub fn error_text(msg: impl ToString) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![Content::text(msg.to_string())]))
}

/// Create a successful tool result with an image (base64-encoded PNG).
pub fn success_image(base64_data: impl ToString) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![Content::image(
        base64_data.to_string(),
        "image/png",
    )]))
}

/// Create a successful tool result with HTML content.
pub fn success_html(html: impl ToString) -> Result<CallToolResult, McpError> {
    // Return HTML as text content for now (MCP doesn't have native HTML type)
    Ok(CallToolResult::success(vec![Content::text(html.to_string())]))
}

/// Helper macro for getting a dataset from the datasets map.
/// Returns an error result if the dataset is not found.
#[macro_export]
macro_rules! get_dataset {
    ($datasets:expr, $name:expr) => {
        match $datasets.get($name) {
            Some(ds) => ds,
            None => {
                return $crate::tools::common::error_text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    $name
                ));
            }
        }
    };
}

/// Helper macro for getting a mutable dataset from the datasets map.
#[macro_export]
macro_rules! get_dataset_mut {
    ($datasets:expr, $name:expr) => {
        match $datasets.get_mut($name) {
            Some(ds) => ds,
            None => {
                return $crate::tools::common::error_text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    $name
                ));
            }
        }
    };
}

/// Helper macro for getting spatial weights from the spatial_weights map.
#[macro_export]
macro_rules! get_spatial_weights {
    ($weights:expr, $name:expr) => {
        match $weights.get($name) {
            Some(w) => w,
            None => {
                return $crate::tools::common::error_text(format!(
                    "Spatial weights '{}' not found. Use 'spatial_neighbors' first to create weights.",
                    $name
                ));
            }
        }
    };
}

/// Helper macro for handling p2a_core function results.
/// Converts EconError to an MCP error response.
#[macro_export]
macro_rules! handle_result {
    ($result:expr) => {
        match $result {
            Ok(v) => v,
            Err(e) => {
                return $crate::tools::common::error_text(format!("Error: {}", e));
            }
        }
    };
    ($result:expr, $context:expr) => {
        match $result {
            Ok(v) => v,
            Err(e) => {
                return $crate::tools::common::error_text(format!("{}: {}", $context, e));
            }
        }
    };
}

// Re-export commonly used types for handlers
pub use rmcp::{
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool,
};
pub use schemars::JsonSchema;
pub use serde::Deserialize;

/// Helper function to extract a single numeric column as a Vec<f64>.
///
/// Used by stats tools that operate on single columns.
pub fn extract_column_f64(
    dataset: &p2a_core::data::Dataset,
    column: &str,
) -> Result<Vec<f64>, String> {
    use p2a_core::polars::prelude::*;

    let df = dataset.df();
    let col = df
        .column(column)
        .map_err(|e| format!("Column '{}' not found: {}", column, e))?;

    let values: Vec<f64> = col
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot convert column '{}' to numeric: {}", column, e))?
        .f64()
        .map_err(|e| format!("Column '{}' is not numeric: {}", column, e))?
        .into_iter()
        .map(|v: Option<f64>| v.unwrap_or(f64::NAN))
        .collect();

    Ok(values)
}

/// Helper function to extract numeric columns into an ndarray matrix.
///
/// Used by ML tools that require numeric matrix input.
pub fn extract_numeric_matrix(
    dataset: &p2a_core::data::Dataset,
    columns: &[String],
) -> Result<ndarray::Array2<f64>, String> {
    use p2a_core::polars::prelude::*;

    let df = dataset.df();
    let n_rows = df.height();
    let n_cols = columns.len();

    if columns.is_empty() {
        return Err("At least one column must be specified".to_string());
    }

    let mut data = ndarray::Array2::zeros((n_rows, n_cols));

    for (j, col_name) in columns.iter().enumerate() {
        let col = df
            .column(col_name)
            .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;

        let values: Vec<f64> = col
            .cast(&DataType::Float64)
            .map_err(|e| format!("Cannot convert column '{}' to numeric: {}", col_name, e))?
            .f64()
            .map_err(|e| format!("Column '{}' is not numeric: {}", col_name, e))?
            .into_iter()
            .map(|v: Option<f64>| v.unwrap_or(f64::NAN))
            .collect();

        for (i, &val) in values.iter().enumerate() {
            data[[i, j]] = val;
        }
    }

    Ok(data)
}
