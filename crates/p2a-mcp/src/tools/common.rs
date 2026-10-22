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
    Ok(CallToolResult::success(vec![Content::text(
        msg.to_string(),
    )]))
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
    Ok(CallToolResult::success(vec![Content::text(
        html.to_string(),
    )]))
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

    // Fallible, so an n x p request above the allocation ceiling is an error, not an abort
    let mut data = p2a_core::resource::try_zeros2(n_rows, n_cols, "numeric matrix")
        .map_err(|e| e.to_string())?;

    for (j, col_name) in columns.iter().enumerate() {
        let col = df
            .column(col_name)
            .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;

        let cast = col
            .cast(&DataType::Float64)
            .map_err(|e| format!("Cannot convert column '{}' to numeric: {}", col_name, e))?;
        let values = cast
            .f64()
            .map_err(|e| format!("Column '{}' is not numeric: {}", col_name, e))?;

        for (cell, v) in data.column_mut(j).iter_mut().zip(values) {
            *cell = v.unwrap_or(f64::NAN);
        }
    }

    Ok(data)
}

/// Check a size derived from the data (not a request field) against a named budget of the
/// installed request limits.
///
/// `limit` is the budget value the caller read from [`crate::tools::limits::current`] (for
/// example `limits.table_cells`); `None` passes. On refusal this returns the LLM-readable message
/// and logs the override key an operator would raise.
pub fn check_named_budget(
    tool: &str,
    knob: &'static str,
    requested: u64,
    limit: Option<u64>,
) -> Result<(), String> {
    crate::tools::limits::current()
        .check_budget(tool, knob, requested, limit)
        .map_err(|v| {
            tracing::info!(
                tool,
                knob = v.override_key,
                requested,
                limit = %v.limit,
                "request refused by a named budget"
            );
            v.message
        })
}

/// Shared helpers for handler unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::server::AnalyticsServer;
    use crate::tools::limits::Limits;
    use p2a_core::polars::prelude::DataFrame;
    use rmcp::model::CallToolResult;

    /// A current-thread runtime and a server holding each (name, frame) as a dataset.
    pub(crate) fn server_with(
        datasets: Vec<(&str, DataFrame)>,
    ) -> (tokio::runtime::Runtime, AnalyticsServer) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let server = AnalyticsServer::new();
        rt.block_on(async {
            let mut map = server.datasets.write().await;
            for (name, df) in datasets {
                map.insert(name.to_string(), p2a_core::data::Dataset::new(df));
            }
        });
        (rt, server)
    }

    /// The concatenated text content of a tool result.
    pub(crate) fn text(r: &CallToolResult) -> String {
        r.content
            .iter()
            .filter_map(|c| c.raw.as_text())
            .map(|t| t.text.as_str())
            .collect()
    }

    /// `Limits::hosted()` with `key=value` overrides applied.
    pub(crate) fn hosted_with(overrides: &[&str]) -> Limits {
        Limits::hosted()
            .with_overrides(&overrides.iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .expect("valid limit overrides")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_named_budget_refuses_over_and_passes_unlimited() {
        let e = check_named_budget("stats_loglin", "table_cells", 2_000_000, Some(1_000_000))
            .unwrap_err();
        assert!(
            e.contains("table_cells") && e.contains("2000000") && e.contains("1000000"),
            "{e}"
        );
        assert!(check_named_budget("stats_loglin", "table_cells", u64::MAX, None).is_ok());
    }

    #[test]
    fn extract_numeric_matrix_allocation_is_fallible() {
        use p2a_core::polars::prelude::*;
        let ds = p2a_core::data::Dataset::new(
            df!("x" => (0..200).map(|i| i as f64 * 0.5).collect::<Vec<_>>()).unwrap(),
        );
        let e = p2a_core::resource::with_alloc_ceiling(Some(1024), || {
            extract_numeric_matrix(&ds, &["x".to_string()])
        })
        .unwrap_err();
        assert!(e.contains("limit"), "{e}"); // 200 x 1 x 8 = 1600 bytes > 1024
        // repeated names are still allowed
        assert_eq!(
            extract_numeric_matrix(&ds, &["x".to_string(), "x".to_string()])
                .unwrap()
                .dim(),
            (200, 2)
        );
    }
}
