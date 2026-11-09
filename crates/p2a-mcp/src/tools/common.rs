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

/// The distinct non-null values of column `name`, the group count a `max_groups` budget holds:
/// an upper bound on the groups the core compares (it may drop more rows). `Ok(None)` when the
/// column is missing, which the core reports itself; a count that fails is the refusal message.
pub fn budget_group_count(
    df: &p2a_core::polars::prelude::DataFrame,
    name: &str,
) -> Result<Option<u64>, String> {
    let Ok(col) = df.column(name) else {
        return Ok(None);
    };
    let n_unique = level_count_or_refusal(name, col.n_unique())?;
    Ok(Some(
        n_unique.saturating_sub(usize::from(col.null_count() > 0)) as u64,
    ))
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

// ============================================================================
// Options, declared file types (T3 design 2.5)
// ============================================================================

/// Parses an optional option string: `None` gives `default`, `Some(s)` gives `s.parse()`.
///
/// An [`p2a_core::EconError::InvalidSpecification`] error returns its bare `message` (the
/// option and the accepted values); any other error its Display.
pub fn parse_option<T: std::str::FromStr<Err = p2a_core::EconError>>(
    value: Option<&str>,
    default: T,
) -> Result<T, String> {
    match value {
        None => Ok(default),
        Some(s) => s.parse::<T>().map_err(|e| match e {
            p2a_core::EconError::InvalidSpecification { message } => message,
            other => other.to_string(),
        }),
    }
}

/// The `file_type` of `load_dataset` / `upload_dataset` (the web UI's File Type selector).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclaredFileType {
    /// No declaration: the extension decides.
    Auto,
    /// CSV text.
    Csv,
    /// Apache Parquet.
    Parquet,
    /// Excel or OpenDocument spreadsheet.
    Excel,
    /// Stata `.dta`.
    Stata,
    /// SAS `.sas7bdat`.
    Sas,
}

impl DeclaredFileType {
    /// The canonical name: "auto", "csv", "parquet", "excel", "stata" or "sas".
    pub fn canonical(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Csv => "csv",
            Self::Parquet => "parquet",
            Self::Excel => "excel",
            Self::Stata => "stata",
            Self::Sas => "sas",
        }
    }

    /// The file extensions of the type, as `DataLoader::load` dispatches on them; none for
    /// `Auto`.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            Self::Auto => &[],
            Self::Csv => &["csv"],
            Self::Parquet => &["parquet", "pq"],
            Self::Excel => &["xlsx", "xls", "xlsb", "ods"],
            Self::Stata => &["dta"],
            Self::Sas => &["sas7bdat"],
        }
    }
}

impl std::str::FromStr for DeclaredFileType {
    type Err = p2a_core::EconError;

    /// A canonical name or one of the type's extensions, ignoring case and surrounding
    /// whitespace.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t = s.trim().to_ascii_lowercase();
        if t == "auto" {
            return Ok(Self::Auto);
        }
        for ty in [
            Self::Csv,
            Self::Parquet,
            Self::Excel,
            Self::Stata,
            Self::Sas,
        ] {
            if t == ty.canonical() || ty.extensions().contains(&t.as_str()) {
                return Ok(ty);
            }
        }
        Err(p2a_core::EconError::InvalidSpecification {
            message: format!(
                "unknown file_type '{}'; expected one of: auto, csv, parquet, excel, stata, sas",
                s.chars().take(64).collect::<String>()
            ),
        })
    }
}

/// Checks a declared `file_type` against the extension of `file_name`: `Ok` with the parsed
/// type when `declared` is `None` (auto) or auto, or names the type of the extension; else the
/// parse error, or a message naming the declared type, the file name (its last path component,
/// at most 64 characters, never its directory) and the extensions the type expects.
pub fn check_declared_file_type(
    declared: Option<&str>,
    file_name: &str,
) -> Result<DeclaredFileType, String> {
    let declared = parse_option(declared, DeclaredFileType::Auto)?;
    if declared == DeclaredFileType::Auto {
        return Ok(declared);
    }
    let path = std::path::Path::new(file_name);
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if declared.extensions().contains(&ext.as_str()) {
        return Ok(declared);
    }
    let shown: String = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
        .chars()
        .take(64)
        .collect();
    let exts: Vec<String> = declared
        .extensions()
        .iter()
        .map(|e| format!(".{e}"))
        .collect();
    let expected = match exts.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        Some((last, _)) => last.clone(),
        None => String::new(),
    };
    let name = declared.canonical();
    Err(format!(
        "file_type '{name}' does not match the file '{shown}': the file is read by its \
         extension, and {name} files end in {expected}. Rename the file or use file_type 'auto'."
    ))
}

// ============================================================================
// Convergence and warning notes (T3 design 2.5)
// ============================================================================

/// A warning's text without a leading "Warning:" and surrounding whitespace; `None` when
/// nothing is left.
fn warning_text(w: &str) -> Option<&str> {
    let t = w.trim();
    let t = t.strip_prefix("Warning:").map(str::trim_start).unwrap_or(t);
    (!t.is_empty()).then_some(t)
}

/// Appends a tool's convergence and warnings to its text: "Converged: yes (N iterations)" or
/// "Converged: no (N iterations)" ("1 iteration" for N = 1, as p2a-core's convergence line),
/// then one "Warning: <text>" line per non-empty warning (a leading "Warning:" in the text is
/// not repeated), each on its own line. Nothing is appended when there is nothing to report.
pub fn append_notes(text: &mut String, converged: Option<(bool, usize)>, warnings: &[String]) {
    let mut lines: Vec<String> = Vec::new();
    if let Some((ok, n)) = converged {
        let unit = if n == 1 { "iteration" } else { "iterations" };
        let yes = if ok { "yes" } else { "no" };
        lines.push(format!("Converged: {yes} ({n} {unit})"));
    }
    lines.extend(
        warnings
            .iter()
            .filter_map(|w| warning_text(w))
            .map(|w| format!("Warning: {w}")),
    );
    if lines.is_empty() {
        return;
    }
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&lines.join("\n"));
}

/// The JSON form of [`append_notes`]: `{"converged": bool, "iterations": n, "warnings": [..]}`,
/// with `converged` and `iterations` only when `converged` is `Some`; `warnings` is always
/// present (trimmed, a leading "Warning:" removed, empty entries dropped).
pub fn notes_json(converged: Option<(bool, usize)>, warnings: &[String]) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    if let Some((ok, n)) = converged {
        out.insert("converged".into(), ok.into());
        out.insert("iterations".into(), n.into());
    }
    let w: Vec<serde_json::Value> = warnings
        .iter()
        .filter_map(|w| warning_text(w))
        .map(|w| w.into())
        .collect();
    out.insert("warnings".into(), w.into());
    serde_json::Value::Object(out)
}

/// Shared helpers for handler unit tests.
// ============================================================================
// Option tables, result notes, JSON replies and seeds shared by the handlers
// ============================================================================
use p2a_core::options::Choices;
use p2a_core::stats::CovWtMethod;
use p2a_core::traits::convergence::ResultNotes;

/// The bare message of an option error (the option and the accepted values), else the error's
/// Display.
pub(crate) fn option_message(e: p2a_core::EconError) -> String {
    match e {
        p2a_core::EconError::InvalidSpecification { message } => message,
        other => other.to_string(),
    }
}

/// Parses an optional option string with a local [`Choices`] table (for p2a-core enums that
/// have no `FromStr` and for handler-local choices; enums with one use [`parse_option`]):
/// `None` gives `default`; any given string, blank included, is parsed; an unknown value is
/// the table's message ("unknown method 'biased'; expected one of: unbiased, ML").
pub(crate) fn parse_choice<T: Copy + PartialEq + 'static>(
    choices: &Choices<T>,
    value: Option<&str>,
    default: T,
) -> Result<T, String> {
    match value {
        None => Ok(default),
        Some(s) => parse_choice_value(choices, s),
    }
}

/// [`parse_choice`] for a value that is always given: the canonical value, or the bare message
/// of an unknown spelling ("unknown how 'some'; expected one of: any, all").
pub(crate) fn parse_choice_value<T: Copy + PartialEq + 'static>(
    choices: &Choices<T>,
    value: &str,
) -> Result<T, String> {
    choices.parse(value).map_err(option_message)
}

/// `stats_cov_wt` method (R's cov.wt `method`).
pub(crate) static COV_WT_METHOD: Choices<CovWtMethod> = Choices {
    option: "method",
    values: &[
        (&["unbiased"], CovWtMethod::Unbiased),
        (&["ML", "mle", "maximum_likelihood"], CovWtMethod::ML),
    ],
};

/// Merges `common::notes_json(converged, warnings)` into a JSON tool output object; a
/// non-object is left alone.
pub(crate) fn insert_notes(
    output: &mut serde_json::Value,
    converged: Option<(bool, usize)>,
    warnings: &[String],
) {
    if let (Some(o), serde_json::Value::Object(n)) =
        (output.as_object_mut(), notes_json(converged, warnings))
    {
        o.extend(n);
    }
}

/// Appends the convergence line and the warnings of `result` that `text` does not show yet.
pub(crate) fn append_unshown_notes(text: &mut String, result: &dyn ResultNotes) {
    let converged = result
        .notes_convergence()
        .filter(|_| !text.contains("Converged:"))
        .map(|c| (c.converged, c.iterations));
    let warnings: Vec<String> = result
        .notes_warnings()
        .into_iter()
        .map(|w| {
            let w = w.trim();
            w.strip_prefix("Warning:").map_or(w, str::trim).to_string()
        })
        .filter(|w| !w.is_empty() && !text.contains(w.as_str()))
        .collect();
    append_notes(text, converged, &warnings);
}

/// Pretty JSON of a tool result; never panics.
pub(crate) fn json_text(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|e| format!("{{\"error\": \"JSON serialisation failed: {e}\"}}"))
}

/// Compact JSON of a tool result (one line, for replies the chat loop may cut); never panics.
pub(crate) fn json_compact(value: &serde_json::Value) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| format!("{{\"error\": \"JSON serialisation failed: {e}\"}}"))
}

/// Seed of the randomized handlers whose core would otherwise fall back to entropy, when
/// neither the request nor `set_seed` gives one, so every run is reproducible. The tool
/// descriptions name it.
pub(crate) const MCP_DEFAULT_SEED: u64 = 42;

impl crate::server::AnalyticsServer {
    /// The request seed, else the session's `set_seed` value; the read guard is released
    /// before returning, so no lock is held while a tool computes.
    pub(crate) async fn seed_or_global(&self, request: Option<u64>) -> Option<u64> {
        match request {
            Some(s) => Some(s),
            None => *self.global_seed.read().await,
        }
    }

    /// [`Self::seed_or_global`], else [`MCP_DEFAULT_SEED`].
    pub(crate) async fn seed_for(&self, request: Option<u64>) -> u64 {
        self.seed_or_global(request)
            .await
            .unwrap_or(MCP_DEFAULT_SEED)
    }

    /// Writes `columns` into dataset `dataset` under a short write lock. `snapshot` is the
    /// dataset the tool read and `inputs` the columns it read from it. Refuses a dataset that
    /// is gone, whose height is no longer the snapshot's, or in which an input column is
    /// missing or no longer equal to the snapshot's (missing values compared as equal): a
    /// concurrent call in the same session replaced, sorted or filtered it while the tool
    /// ran, and the new column would not line up with the rows it was computed from. Returns
    /// the names that replaced an existing column.
    pub(crate) async fn write_output_columns(
        &self,
        dataset: &str,
        snapshot: &p2a_core::data::Dataset,
        inputs: &[&str],
        columns: Vec<p2a_core::polars::prelude::Series>,
    ) -> Result<Vec<String>, String> {
        let n_rows = snapshot.nrows();
        let mut map = self.datasets.write().await;
        let current = map
            .get(dataset)
            .ok_or_else(|| format!("Dataset '{dataset}' not found; the column was not written."))?;
        if current.nrows() != n_rows {
            return Err(format!(
                "Dataset '{dataset}' changed while the tool ran (it has {} rows, the fit used \
                 {n_rows}); the column was not written. Run the tool again.",
                current.nrows()
            ));
        }
        for &name in inputs {
            let unchanged = match (current.df().column(name), snapshot.df().column(name)) {
                (Ok(now), Ok(then)) => now.equals_missing(then),
                _ => false,
            };
            if !unchanged {
                return Err(format!(
                    "Dataset '{dataset}' changed while the tool ran (column '{name}' is no \
                     longer the one the fit used); the column was not written. Run the tool \
                     again."
                ));
            }
        }
        let mut updated = current.clone();
        let mut replaced = Vec::new();
        for s in columns {
            let name = s.name().to_string();
            if updated.df().column(&name).is_ok() {
                replaced.push(name.clone());
            }
            updated
                .df_mut()
                .with_column(s)
                .map_err(|e| format!("could not write column '{name}': {e}"))?;
        }
        map.insert(dataset.to_string(), updated);
        Ok(replaced)
    }
}

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
    fn test_parse_choice_none_default_and_blank_refused() {
        static CRIT: Choices<u8> = Choices {
            option: "criterion",
            values: &[(&["AIC"], 1), (&["BIC", "schwarz"], 2)],
        };
        assert_eq!(parse_choice(&CRIT, None, 1), Ok(1));
        assert_eq!(parse_choice(&CRIT, Some(" Schwarz "), 1), Ok(2));
        for bad in ["  ", "cp"] {
            let e = parse_choice(&CRIT, Some(bad), 1).unwrap_err();
            assert!(
                e.starts_with("unknown criterion '") && e.contains("expected one of: AIC, BIC"),
                "{e}"
            );
        }
        assert_eq!(parse_choice_value(&CRIT, "bic"), Ok(2));
    }

    #[test]
    fn test_append_notes_lines() {
        let mut s = String::from("Coefficients: 1.0");
        append_notes(
            &mut s,
            Some((false, 100)),
            &[
                "Warning: x did not converge".into(),
                "  ".into(),
                "rank 2 < 3".into(),
            ],
        );
        assert_eq!(
            s,
            "Coefficients: 1.0\nConverged: no (100 iterations)\nWarning: x did not converge\nWarning: rank 2 < 3"
        );
        let mut t = String::from("ok\n");
        append_notes(&mut t, Some((true, 12)), &[]);
        assert_eq!(t, "ok\nConverged: yes (12 iterations)");
        let mut u = String::from("unchanged");
        append_notes(&mut u, None, &[]);
        assert_eq!(u, "unchanged");
        let mut one = String::from("fit");
        append_notes(&mut one, Some((true, 1)), &[]);
        assert_eq!(one, "fit\nConverged: yes (1 iteration)"); // singular, as Convergence::line()
    }

    #[test]
    fn test_notes_json_keys() {
        assert_eq!(
            notes_json(Some((true, 7)), &["Warning: a".into(), "".into()]),
            serde_json::json!({"converged": true, "iterations": 7, "warnings": ["a"]})
        );
        assert_eq!(notes_json(None, &[]), serde_json::json!({"warnings": []}));
    }

    #[test]
    fn test_declared_file_type_parses_aliases_and_rejects_unknown() {
        for (s, t) in [
            ("auto", DeclaredFileType::Auto),
            ("CSV", DeclaredFileType::Csv),
            (" parquet ", DeclaredFileType::Parquet),
            ("pq", DeclaredFileType::Parquet),
            ("excel", DeclaredFileType::Excel),
            ("xlsx", DeclaredFileType::Excel),
            ("ods", DeclaredFileType::Excel),
            ("stata", DeclaredFileType::Stata),
            ("dta", DeclaredFileType::Stata),
            ("sas", DeclaredFileType::Sas),
            ("sas7bdat", DeclaredFileType::Sas),
        ] {
            assert_eq!(s.parse::<DeclaredFileType>().unwrap(), t, "{s}");
        }
        let e = parse_option::<DeclaredFileType>(Some("tsv"), DeclaredFileType::Auto).unwrap_err();
        assert_eq!(
            e,
            "unknown file_type 'tsv'; expected one of: auto, csv, parquet, excel, stata, sas"
        );
    }

    #[test]
    fn test_check_declared_file_type() {
        // The parsed type is returned, so callers never parse file_type a second time.
        assert_eq!(
            check_declared_file_type(None, "/data/x.csv"),
            Ok(DeclaredFileType::Auto)
        );
        assert_eq!(
            check_declared_file_type(Some("auto"), "x.txt"),
            Ok(DeclaredFileType::Auto)
        );
        assert_eq!(
            check_declared_file_type(Some("csv"), "/data/X.CSV"),
            Ok(DeclaredFileType::Csv)
        );
        assert_eq!(
            check_declared_file_type(Some("excel"), "book.ods"),
            Ok(DeclaredFileType::Excel)
        );
        assert_eq!(
            check_declared_file_type(Some("tsv"), "x.tsv"),
            Err(
                "unknown file_type 'tsv'; expected one of: auto, csv, parquet, excel, stata, sas"
                    .to_string()
            )
        );
        let e = check_declared_file_type(Some("parquet"), "/home/u/private/sales.csv").unwrap_err();
        assert_eq!(
            e,
            "file_type 'parquet' does not match the file 'sales.csv': the file is read by its \
             extension, and parquet files end in .parquet or .pq. Rename the file or use \
             file_type 'auto'."
        );
        assert!(!e.contains("/home"));
        let e = check_declared_file_type(Some("excel"), "a.csv").unwrap_err();
        assert!(
            e.contains("excel files end in .xlsx, .xls, .xlsb or .ods"),
            "{e}"
        );
    }

    #[test]
    fn test_parse_option_parses_and_reports_the_core_message() {
        use p2a_core::regression::CovarianceType;
        assert_eq!(
            parse_option::<CovarianceType>(None, CovarianceType::HC1).unwrap(),
            CovarianceType::HC1
        );
        assert_eq!(
            parse_option::<CovarianceType>(Some("HC3"), CovarianceType::HC1).unwrap(),
            CovarianceType::HC3
        );
        let e = parse_option::<CovarianceType>(Some("hc9"), CovarianceType::HC1).unwrap_err();
        assert!(
            e.starts_with("unknown covariance type 'hc9'")
                && !e.contains("Invalid model specification"),
            "{e}"
        );
    }

    #[test]
    fn budget_group_count_leaves_out_null_and_an_absent_column() {
        let df = p2a_core::polars::prelude::df! {
            "g" => [Some("a"), Some("b"), None, Some("a"), Some("c")],
            "h" => [Some("x"), Some("y"), Some("x"), Some("y"), Some("y")],
        }
        .unwrap();
        assert_eq!(budget_group_count(&df, "g"), Ok(Some(3)));
        assert_eq!(budget_group_count(&df, "h"), Ok(Some(2)));
        assert_eq!(budget_group_count(&df, "absent"), Ok(None));
        // the level count for a budget keeps null as a value
        assert_eq!(budget_level_count(&df, "g"), Ok(Some(4)));
    }

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
