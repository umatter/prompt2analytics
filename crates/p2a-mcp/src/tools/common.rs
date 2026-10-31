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

pub use p2a_core::data::columns::{Columns, MissingSummary, NaPolicy};

use p2a_core::data::columns::ColumnRequest;

/// One numeric column under `policy`: the kept values and what was dropped. The error is
/// the core rule's LLM-readable text ("Column 'x' contains 3 missing value(s) (null or
/// NaN). Drop or impute them first.").
pub fn numeric_column(
    dataset: &p2a_core::data::Dataset,
    column: &str,
    policy: NaPolicy,
) -> Result<(Vec<f64>, MissingSummary), String> {
    p2a_core::data::columns::numeric_column(dataset.df(), column, policy).map_err(|e| e.to_string())
}

/// Row-aligned numeric columns under `policy` (rows with a missing value in any of them
/// are dropped from all with [`NaPolicy::Omit`]).
pub fn numeric_columns(
    dataset: &p2a_core::data::Dataset,
    columns: &[&str],
    policy: NaPolicy,
) -> Result<Columns, String> {
    ColumnRequest::new(dataset.df())
        .numerics(columns)
        .na(policy)
        .extract()
        .map_err(|e| e.to_string())
}

/// Numeric columns as an n x k matrix (column order of `columns`, repeated names
/// allowed) under `policy`, allocated fallibly.
pub fn numeric_matrix(
    dataset: &p2a_core::data::Dataset,
    columns: &[String],
    policy: NaPolicy,
) -> Result<(ndarray::Array2<f64>, MissingSummary), String> {
    if columns.is_empty() {
        return Err("At least one column must be specified".to_string());
    }
    let names: Vec<&str> = columns.iter().map(String::as_str).collect();
    let cols = numeric_columns(dataset, &names, policy)?;
    let mut data = p2a_core::resource::try_zeros2(cols.n(), names.len(), "numeric matrix")
        .map_err(|e| e.to_string())?;
    for (j, name) in names.iter().enumerate() {
        for (cell, &v) in data.column_mut(j).iter_mut().zip(cols.numeric(name)) {
            *cell = v;
        }
    }
    Ok((data, cols.missing().clone()))
}

/// Append "\nMissing values: <note>" to a text result when rows were dropped.
pub fn append_missing_note(text: &mut String, missing: &MissingSummary) {
    if let Some(note) = missing.note() {
        text.push_str("\nMissing values: ");
        text.push_str(&note);
    }
}

/// `{"n_total", "n_used", "n_dropped", "by_column": {column: count}}` when rows were
/// dropped, for a JSON result's `"missing"` key; `None` when nothing was dropped. The
/// shape is `MissingSummary`'s own `Serialize`; as a `serde_json::Value`, the keys of
/// `by_column` are sorted.
pub fn missing_json(missing: &MissingSummary) -> Option<serde_json::Value> {
    // `to_value` cannot fail for a MissingSummary (string keys, integer values), so `None`
    // only means that no row was dropped.
    (!missing.is_empty())
        .then(|| serde_json::to_value(missing))
        .and_then(Result::ok)
}

/// Adds `"missing"` ([`missing_json`]) to a JSON tool output when rows were dropped; leaves
/// it as it is otherwise.
pub fn insert_missing(output: &mut serde_json::Value, missing: &MissingSummary) {
    if let Some(m) = missing_json(missing) {
        output["missing"] = m;
    }
}

/// The distinct values of column `name` for a size budget, null counted as one: an upper bound
/// on the levels the core keeps after dropping missing values. `Ok(None)` when the column is
/// missing, which the core reports itself.
pub fn budget_level_count(
    df: &p2a_core::polars::prelude::DataFrame,
    name: &str,
) -> Result<Option<usize>, String> {
    let Ok(col) = df.column(name) else {
        return Ok(None);
    };
    level_count_or_refusal(name, col.n_unique()).map(Some)
}

/// A distinct-value count a budget depends on, or the refusal when it could not be taken (the
/// budget cannot be checked, so the request must not run unbounded).
pub fn level_count_or_refusal(
    name: &str,
    count: p2a_core::polars::prelude::PolarsResult<usize>,
) -> Result<usize, String> {
    count.map_err(|e| {
        format!(
            "the distinct values of `{}` could not be counted to check this server's limits \
             ({e}).",
            crate::tools::limits::echo(name)
        )
    })
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

/// A request refused by a resource limit, as a handler returns it: logged at `info` (tool,
/// override key, requested size and limit), and returned as an in-band tool error whose text
/// names the parameter, the requested size and the limit.
pub(crate) fn limit_refusal(v: crate::tools::limits::LimitViolation) -> CallToolResult {
    tracing::info!(
        tool = %v.tool,
        knob = v.override_key,
        requested = %v.requested,
        limit = %v.limit,
        "tool request refused by a resource limit"
    );
    CallToolResult::error(vec![Content::text(v.message)])
}

/// How many caller-supplied names a refusal lists before "(and N more)".
pub(crate) const REFUSAL_NAMES: usize = 5;

/// Caller-supplied names for a refusal, joined by `sep`: each cut by
/// [`crate::tools::limits::echo`], at most `max` of them, then "(and N more)".
pub(crate) fn echo_list<S: AsRef<str>>(items: &[S], sep: &str, max: usize) -> String {
    let mut out = items
        .iter()
        .take(max)
        .map(|s| crate::tools::limits::echo(s.as_ref()))
        .collect::<Vec<_>>()
        .join(sep);
    if items.len() > max {
        out.push_str(&format!(" (and {} more)", items.len() - max));
    }
    out
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
    fn echo_list_cuts_each_name_and_the_list() {
        let long = "n".repeat(200);
        let names: Vec<String> = (0..7).map(|i| format!("{long}{i}")).collect();
        let s = echo_list(&names, ", ", 3);
        assert!(s.ends_with(" (and 4 more)"), "{s}");
        assert_eq!(s.matches(&"n".repeat(64)).count(), 3, "{s}");
        assert!(!s.contains(&"n".repeat(65)), "{s}");
        assert_eq!(echo_list(&["a", "b"], " x ", 3), "a x b");
    }

    #[test]
    fn test_validate_mcp_numeric_column_omit_and_fail() {
        use p2a_core::data::{DataLoader, Dataset};
        let case = p2a_validation::Case::load("columns_conventions");
        let ds = Dataset::new(DataLoader::load_csv(case.dataset_path()).unwrap());
        let (y, missing) = numeric_column(&ds, "y", NaPolicy::Omit).unwrap();
        assert_eq!(y.len(), 8);
        assert_eq!(missing.n_dropped as f64, case.scalar("missing_y"));
        assert_eq!(
            numeric_column(&ds, "y", NaPolicy::Fail).unwrap_err(),
            "Column 'y' contains 2 missing value(s) (null or NaN). Drop or impute them first."
        );
    }

    #[test]
    fn test_validate_mcp_numeric_columns_int64_and_text() {
        use p2a_core::data::{DataLoader, Dataset};
        let case = p2a_validation::Case::load("columns_conventions");
        let ds = Dataset::new(DataLoader::load_csv(case.dataset_path()).unwrap());
        let cols = numeric_columns(&ds, &["y", "n"], NaPolicy::Omit).unwrap();
        assert_eq!(cols.rows(), [0, 2, 4, 5, 7, 8, 9]);
        // Derived from the fixture, not case values (the case's complete rows are those of
        // complete.cases(y, n, s)): over these 7 rows mean y is 44/7, sum n is 94.
        let y = cols.numeric("y");
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        assert!(
            (mean - 6.285_714_285_714_286).abs() <= 1e-15 * 6.3,
            "{mean}"
        );
        assert_eq!(cols.numeric("n").iter().sum::<f64>(), 94.0);
        let e = numeric_columns(&ds, &["s"], NaPolicy::Omit).unwrap_err();
        assert!(e.contains("Column 's' contains non-numeric values"), "{e}");
        // The per-column missing counts are the case's.
        let by = &cols.missing().by_column;
        assert_eq!((by[0].0.as_str(), by[1].0.as_str()), ("y", "n"));
        p2a_validation::Check::new(&case)
            .scalar("missing_y", by[0].1 as f64, p2a_validation::Tol::exact())
            .scalar("missing_n", by[1].1 as f64, p2a_validation::Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_mcp_numeric_matrix_dims_repeats_and_ceiling() {
        use p2a_core::data::{DataLoader, Dataset};
        let case = p2a_validation::Case::load("columns_conventions");
        let ds = Dataset::new(DataLoader::load_csv(case.dataset_path()).unwrap());
        let (m, missing) = numeric_matrix(
            &ds,
            &["y".into(), "flag".into(), "y".into()],
            NaPolicy::Omit,
        )
        .unwrap();
        // Derived from the fixture, not case values: y and flag are complete in rows
        // 0 2 3 4 7 8 9, and flag sums to 4 there.
        assert_eq!(m.dim(), (7, 3));
        assert_eq!(m.column(0), m.column(2));
        assert_eq!(m.column(1).sum(), 4.0);
        assert_eq!(
            missing.by_column,
            [("y".to_string(), 2), ("flag".to_string(), 1)]
        );
        assert_eq!(
            numeric_matrix(&ds, &[], NaPolicy::Omit).unwrap_err(),
            "At least one column must be specified"
        );
        let e = p2a_core::resource::with_alloc_ceiling(Some(64), || {
            numeric_matrix(&ds, &["y".into()], NaPolicy::Omit)
        })
        .unwrap_err();
        assert!(e.contains("Resource limit"), "{e}");
        // y's missing count is the case's (flag's is not a case value).
        p2a_validation::Check::new(&case)
            .scalar(
                "missing_y",
                missing.by_column[0].1 as f64,
                p2a_validation::Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_mcp_insert_missing_adds_the_key_only_when_rows_were_dropped() {
        let dropped = MissingSummary {
            n_total: 10,
            n_used: 7,
            n_dropped: 3,
            by_column: vec![("y".to_string(), 2), ("n".to_string(), 1)],
        };
        let mut output = serde_json::json!({"statistic": 1.5});
        insert_missing(&mut output, &dropped);
        assert_eq!(output["missing"], missing_json(&dropped).unwrap());
        assert_eq!(output["statistic"], 1.5);
        let mut untouched = serde_json::json!({"statistic": 1.5});
        insert_missing(&mut untouched, &MissingSummary::none(10));
        assert_eq!(untouched, serde_json::json!({"statistic": 1.5}));
    }

    #[test]
    fn test_validate_mcp_missing_note_and_json() {
        use p2a_core::data::{DataLoader, Dataset};
        let case = p2a_validation::Case::load("columns_conventions");
        let ds = Dataset::new(DataLoader::load_csv(case.dataset_path()).unwrap());
        let cols = numeric_columns(&ds, &["y", "n"], NaPolicy::Omit).unwrap();
        let mut text = "t".to_string();
        append_missing_note(&mut text, cols.missing());
        assert_eq!(
            text,
            "t\nMissing values: 3 of 10 rows dropped because of missing values (y: 2, n: 1)"
        );
        assert_eq!(
            missing_json(cols.missing()),
            Some(serde_json::json!({
                "n_total": 10, "n_used": 7, "n_dropped": 3, "by_column": {"y": 2, "n": 1}
            }))
        );
        let none = MissingSummary::none(10);
        let mut text = "t".to_string();
        append_missing_note(&mut text, &none);
        assert_eq!(text, "t");
        assert_eq!(missing_json(&none), None);
        // The per-column counts in the note and the JSON are the case's.
        let by = &cols.missing().by_column;
        assert_eq!((by[0].0.as_str(), by[1].0.as_str()), ("y", "n"));
        p2a_validation::Check::new(&case)
            .scalar("missing_y", by[0].1 as f64, p2a_validation::Tol::exact())
            .scalar("missing_n", by[1].1 as f64, p2a_validation::Tol::exact())
            .finish();
    }
}
