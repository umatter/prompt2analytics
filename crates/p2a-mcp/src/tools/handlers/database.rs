//! Database tool handlers (SQLite, DuckDB queries).
//!
//! This module defines database tools using the `#[tool_router(router = database_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::path_jail;
use crate::server::AnalyticsServer;
use crate::tools::requests::database::*;

use p2a_core::{
    Dataset,
    data::{
        QueryOptions, QueryOutcome, duckdb_table_schema, list_duckdb_tables, list_sqlite_tables,
        query_duckdb_with, query_file_with_duckdb_with, query_sqlite_with, sqlite_table_schema,
    },
};

/// Validate a database file path, allowing the special `:memory:` DuckDB path.
fn jail_db_path(requested: &str) -> Result<PathBuf, String> {
    if requested == ":memory:" {
        return Ok(PathBuf::from(":memory:"));
    }
    path_jail::validate_data_path(requested)
}

/// The refusal `db_duckdb_query` and `db_query_file` give when the
/// `sql_freeform` budget is false, which is the hosted profile's default (user
/// decision of 2026-10-21). DuckDB builds per-row values and materialised
/// results outside its `memory_limit`, and no engine setting caps them, so on
/// the public demo the two free-form DuckDB tools stay off. Operators can turn
/// them back on with `P2A_LIMIT_OVERRIDES=sql_freeform=true`.
pub(crate) const SQL_FREEFORM_REFUSAL: &str =
    "free-form SQL is not available on the public demo; load the file with load_dataset instead";

/// The tool error for `db_duckdb_query`/`db_query_file` when the active limits
/// disable free-form SQL (`sql_freeform = false`); `None` when they may run.
fn freeform_sql_refusal() -> Option<CallToolResult> {
    if crate::tools::limits::current().sql_freeform {
        None
    } else {
        Some(CallToolResult::error(vec![Content::text(
            SQL_FREEFORM_REFUSAL,
        )]))
    }
}

const MIB: u64 = 1024 * 1024;

/// The SQL bounds of the active request limits (the `db_*` budgets): rows,
/// estimated result bytes, run time, DuckDB memory and threads. All `None`
/// when limits are off; the engine lockdown applies either way.
fn sql_query_options() -> QueryOptions {
    let limits = crate::tools::limits::current();
    QueryOptions {
        max_rows: limits.db_result_rows,
        max_bytes: limits.db_result_bytes,
        timeout: limits.db_timeout_secs.map(Duration::from_secs),
        memory_limit_mb: limits.db_memory_mb,
        threads: limits.db_threads,
    }
}

/// `message` with the resolved server path replaced by the path the caller
/// sent, so error text never reveals where the server keeps its files. The
/// SQL-literal form (`'` doubled, as substituted for `{file}`) is replaced too.
/// A no-op for `:memory:`.
fn redact_path(message: &str, resolved: &Path, requested: &str) -> String {
    let resolved = resolved.to_string_lossy();
    if resolved.is_empty() || resolved == requested {
        return message.to_string();
    }
    let quoted = resolved.replace('\'', "''");
    let mut out = message.to_string();
    if quoted != resolved {
        out = out.replace(&quoted, &requested.replace('\'', "''"));
    }
    out.replace(resolved.as_ref(), requested)
}

/// `bytes` as MiB for the truncation note (`bytes` below 1 MiB stay bytes).
fn format_mib(bytes: u64) -> String {
    if bytes < MIB {
        format!("{bytes} bytes")
    } else if bytes.is_multiple_of(MIB) {
        format!("{} MiB", bytes / MIB)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    }
}

/// The line that follows "Rows returned" when the stored result was cut at the
/// row or byte budget; empty otherwise.
fn truncation_note(outcome: &QueryOutcome, opts: &QueryOptions) -> String {
    if !outcome.truncated {
        return String::new();
    }
    let mut caps = Vec::new();
    if let Some(rows) = opts.max_rows {
        caps.push(format!("{rows} rows"));
    }
    if let Some(bytes) = opts.max_bytes {
        caps.push(format_mib(bytes));
    }
    let caps = if caps.is_empty() {
        "a limited result".to_string()
    } else {
        caps.join(" / ")
    };
    format!(
        "Note: the result was truncated to the first {} rows because this server returns at \
         most {} per query; the stored dataset contains only these rows. Use WHERE, GROUP BY \
         or LIMIT to select what you need.\n",
        outcome.rows_returned, caps
    )
}

/// File-reading SQL table functions (chiefly DuckDB's) that can open an
/// arbitrary path from *inside* a query string, bypassing the path-jail check
/// applied to the top-level `db_path`/`file_path` argument. Names are matched
/// case-insensitively as identifier prefixes, so `read_csv` also covers
/// `read_csv_auto`, and `read_json` covers `read_json_auto`.
const FILE_READING_SQL_FUNCTIONS: &[&str] = &[
    "read_csv",
    "read_parquet",
    "read_json",
    "read_ndjson",
    "read_text",
    "read_blob",
    "parquet_scan",
    "csv_scan",
    "glob",
];

/// Reject a query that calls a file-reading SQL function with anything other
/// than the `{file}` placeholder (which the handler replaces with a jailed,
/// pre-validated path), with guidance to use the placeholder instead.
///
/// This is defence in depth that only gives a clearer message. The engine is
/// the enforcement: every query connection has DuckDB external access disabled
/// and its configuration locked, and `db_query_file` can read exactly its own
/// jailed file, so a literal path is refused by DuckDB even when it gets past
/// this lexical check.
fn guard_sql_file_access(query: &str) -> Result<(), String> {
    let bytes = query.as_bytes();
    let lower = query.to_ascii_lowercase();

    let is_ident_byte = |b: u8| b == b'_' || b.is_ascii_alphanumeric();

    for func in FILE_READING_SQL_FUNCTIONS {
        let mut from = 0;
        while let Some(rel) = lower[from..].find(func) {
            let start = from + rel;
            from = start + 1;

            // Left boundary: the match must begin a fresh identifier, not sit
            // inside a longer one (e.g. avoid matching "glob" in "my_global").
            if start > 0 && is_ident_byte(bytes[start - 1]) {
                continue;
            }

            // Consume the rest of the identifier (e.g. the `_auto` suffix).
            let mut j = start + func.len();
            while j < bytes.len() && is_ident_byte(bytes[j]) {
                j += 1;
            }
            // Skip whitespace before the call parenthesis.
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            // Not a function call — a bare identifier/column, ignore it.
            if j >= bytes.len() || bytes[j] != b'(' {
                continue;
            }
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }

            // The only permitted argument is the `{file}` placeholder, which the
            // handler substitutes with a jail-validated path.
            if query[j..].starts_with("{file}") {
                continue;
            }

            return Err(format!(
                "query calls the file-reading function `{func}(...)` with a literal argument; \
                 this is not allowed because it bypasses the data-root jail. Use the `{{file}}` \
                 placeholder (via db_query_file) so the path is validated against the data root."
            ));
        }
    }
    Ok(())
}

/// Short-circuit a handler when a query references an unjailed file read.
macro_rules! guard_sql_or_return {
    ($query:expr, $action:expr) => {
        if let Err(e) = guard_sql_file_access($query) {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Refused to {}: {}",
                $action, e
            ))]));
        }
    };
}

/// Helper to short-circuit a handler when the path fails validation.
macro_rules! jail_or_return {
    ($input:expr, $action:expr) => {
        match jail_db_path($input) {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Refused to {}: {}",
                    $action, e
                ))]));
            }
        }
    };
}

#[tool_router(router = database_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // SQLite Tools
    // ========================================================================

    /// Query a SQLite database and load results as a dataset.
    #[tool(
        description = "Execute exactly one read-only SQL statement (SELECT, WITH, EXPLAIN or a schema PRAGMA: table_info, table_list, index_list, ...) against a SQLite database and load the results as a dataset. The resulting dataset can then be analyzed using other tools. Statements that would modify the database, ATTACH, other PRAGMAs and any second statement are refused. The server may cap the number of rows, the result size and the run time; a truncated result is reported."
    )]
    pub async fn db_sqlite_query(
        &self,
        Parameters(request): Parameters<SqliteQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let db_path = jail_or_return!(&request.db_path, "query SQLite database");
        guard_sql_or_return!(&request.query, "query SQLite database");
        let opts = sql_query_options();
        let outcome = match query_sqlite_with(&db_path, &request.query, &opts) {
            Ok(o) => o,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SQLite query failed: {}",
                    redact_path(&e.to_string(), &db_path, &request.db_path)
                ))]));
            }
        };
        let note = truncation_note(&outcome, &opts);
        let result = outcome.result;

        // Get preview before moving dataframe
        let preview = result.dataframe.head(Some(5));

        // Create dataset from result
        let dataset = Dataset::new(result.dataframe);

        // Generate name
        let name = request.name.unwrap_or_else(|| {
            format!(
                "sqlite_query_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            )
        });

        // Store dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(name.clone(), dataset);

        let output = format!(
            "SQLite Query Results\n\
             ====================\n\
             Rows returned: {}\n\
             {}\
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            result.rows,
            note,
            result.columns.join(", "),
            name,
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// List tables in a SQLite database.
    #[tool(description = "List all tables in a SQLite database.")]
    pub async fn db_sqlite_tables(
        &self,
        Parameters(request): Parameters<SqliteListTablesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let db_path = jail_or_return!(&request.db_path, "list SQLite tables");
        let tables = match list_sqlite_tables(&db_path) {
            Ok(t) => t,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to list tables: {}",
                    redact_path(&e.to_string(), &db_path, &request.db_path)
                ))]));
            }
        };

        if tables.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No tables found in database.",
            )]));
        }

        let output = format!(
            "Tables in SQLite database:\n\n{}",
            tables
                .iter()
                .map(|t| format!("  - {}", t))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Get schema for a SQLite table.
    #[tool(
        description = "Get the schema (column names and types) for a table in a SQLite database."
    )]
    pub async fn db_sqlite_schema(
        &self,
        Parameters(request): Parameters<SqliteSchemaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let db_path = jail_or_return!(&request.db_path, "inspect SQLite schema");
        let schema = match sqlite_table_schema(&db_path, &request.table_name) {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to get schema: {}",
                    redact_path(&e.to_string(), &db_path, &request.db_path)
                ))]));
            }
        };

        if schema.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Table '{}' not found or has no columns.",
                request.table_name
            ))]));
        }

        let output = format!(
            "Schema for table '{}':\n\n{}",
            request.table_name,
            schema
                .iter()
                .map(|(name, dtype)| format!("  - {} ({})", name, dtype))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // DuckDB Tools
    // ========================================================================

    /// Query a DuckDB database and load results as a dataset.
    #[tool(
        description = "Execute exactly one read-only SQL statement (SELECT, WITH, EXPLAIN without ANALYZE, PRAGMA, DESCRIBE or SHOW) against a DuckDB database and load the results as a dataset. DuckDB supports advanced analytics SQL including window functions, CTEs, and more. EXPLAIN ANALYZE, ATTACH, COPY, INSTALL/LOAD, SET and any second statement are refused, and the query cannot read files: ':memory:' starts empty, so to query a CSV or Parquet file use db_query_file with {file}. The server may cap the number of rows, the result size and the run time; a truncated result is reported. Not available when the server disables free-form SQL (the public demo); use load_dataset there."
    )]
    pub async fn db_duckdb_query(
        &self,
        Parameters(request): Parameters<DuckDBQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Before any path is validated or any engine is opened.
        if let Some(refusal) = freeform_sql_refusal() {
            return Ok(refusal);
        }
        let db_path = jail_or_return!(&request.db_path, "query DuckDB database");
        guard_sql_or_return!(&request.query, "query DuckDB database");
        let opts = sql_query_options();
        let outcome = match query_duckdb_with(&db_path, &request.query, &opts) {
            Ok(o) => o,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DuckDB query failed: {}",
                    redact_path(&e.to_string(), &db_path, &request.db_path)
                ))]));
            }
        };
        let note = truncation_note(&outcome, &opts);
        let result = outcome.result;

        // Get preview before moving dataframe
        let preview = result.dataframe.head(Some(5));

        // Create dataset from result
        let dataset = Dataset::new(result.dataframe);

        // Generate name
        let name = request.name.unwrap_or_else(|| {
            format!(
                "duckdb_query_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            )
        });

        // Store dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(name.clone(), dataset);

        let output = format!(
            "DuckDB Query Results\n\
             ====================\n\
             Rows returned: {}\n\
             {}\
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            result.rows,
            note,
            result.columns.join(", "),
            name,
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// List tables in a DuckDB database.
    #[tool(description = "List all tables in a DuckDB database.")]
    pub async fn db_duckdb_tables(
        &self,
        Parameters(request): Parameters<DuckDBListTablesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let db_path = jail_or_return!(&request.db_path, "list DuckDB tables");
        let tables = match list_duckdb_tables(&db_path) {
            Ok(t) => t,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to list tables: {}",
                    redact_path(&e.to_string(), &db_path, &request.db_path)
                ))]));
            }
        };

        if tables.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No tables found in database.",
            )]));
        }

        let output = format!(
            "Tables in DuckDB database:\n\n{}",
            tables
                .iter()
                .map(|t| format!("  - {}", t))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Get schema for a DuckDB table.
    #[tool(
        description = "Get the schema (column names and types) for a table in a DuckDB database."
    )]
    pub async fn db_duckdb_schema(
        &self,
        Parameters(request): Parameters<DuckDBSchemaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let db_path = jail_or_return!(&request.db_path, "inspect DuckDB schema");
        let schema = match duckdb_table_schema(&db_path, &request.table_name) {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to get schema: {}",
                    redact_path(&e.to_string(), &db_path, &request.db_path)
                ))]));
            }
        };

        if schema.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Table '{}' not found or has no columns.",
                request.table_name
            ))]));
        }

        let output = format!(
            "Schema for table '{}':\n\n{}",
            request.table_name,
            schema
                .iter()
                .map(|(name, dtype)| format!("  - {} ({})", name, dtype))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Query a Parquet or CSV file directly using DuckDB SQL.
    #[tool(
        description = "Execute exactly one read-only SQL statement with DuckDB on one Parquet or CSV file, file_path, and load the result as a dataset, for example to filter or aggregate the file before loading it. Refer to the file as {file}, also inside reader functions, for example SELECT * FROM {file} or read_csv_auto({file}, header = true). The query can read no other file, so it cannot join files. The rules and caps of db_duckdb_query apply: EXPLAIN ANALYZE, ATTACH, COPY, INSTALL/LOAD, SET and any second statement are refused, and the server may cap the number of rows, the result size and the run time; a truncated result is reported. Not available when the server disables free-form SQL (the public demo); use load_dataset there."
    )]
    pub async fn db_query_file(
        &self,
        Parameters(request): Parameters<DuckDBFileQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Before any path is validated or any engine is opened.
        if let Some(refusal) = freeform_sql_refusal() {
            return Ok(refusal);
        }
        let file_path = jail_or_return!(&request.file_path, "query data file");
        guard_sql_or_return!(&request.query, "query data file");
        let opts = sql_query_options();
        let outcome = match query_file_with_duckdb_with(&file_path, &request.query, &opts) {
            Ok(o) => o,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DuckDB file query failed: {}",
                    redact_path(&e.to_string(), &file_path, &request.file_path)
                ))]));
            }
        };
        let note = truncation_note(&outcome, &opts);
        let result = outcome.result;

        // Get preview before moving dataframe
        let preview = result.dataframe.head(Some(5));

        // Convert to Dataset
        let dataset = Dataset::new(result.dataframe);

        // Generate name
        let name = request.name.unwrap_or_else(|| {
            format!(
                "file_query_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            )
        });

        // Store in datasets
        let mut datasets = self.datasets.write().await;
        datasets.insert(name.clone(), dataset);

        let output = format!(
            "DuckDB File Query Results\n\
             =========================\n\
             File: {}\n\
             Rows returned: {}\n\
             {}\
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            // The path as the caller sent it, never the resolved server path.
            request.file_path,
            result.rows,
            note,
            result.columns.join(", "),
            name,
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }
}

#[cfg(test)]
mod tests {
    use super::{SQL_FREEFORM_REFUSAL, guard_sql_file_access, redact_path, sql_query_options};
    use crate::server::AnalyticsServer;
    use crate::tools::limits::{self, Limits};
    use crate::tools::requests::database::{
        DuckDBFileQueryRequest, DuckDBQueryRequest, SqliteQueryRequest,
    };
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::model::{CallToolResult, RawContent};
    use std::path::Path;
    use std::time::Duration;

    fn text_of(r: &CallToolResult) -> String {
        r.content
            .iter()
            .filter_map(|c| match &c.raw {
                RawContent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn duckdb_query_refuses_multiple_statements_and_writes_nothing() {
        let outside = tempfile::tempdir().unwrap(); // outside the jail (cf. utils.rs canary)
        let target = outside.path().join("evil.duckdb");
        let server = AnalyticsServer::new();
        let result = server
            .db_duckdb_query(Parameters(DuckDBQueryRequest {
                db_path: ":memory:".into(),
                query: format!(
                    "SELECT 1; ATTACH '{}' AS e; CREATE TABLE e.s AS SELECT 'x' AS v",
                    target.display()
                ),
                name: Some("multi".into()),
            }))
            .await
            .unwrap();
        assert_eq!(result.is_error, Some(true));
        assert!(
            text_of(&result).contains("only one SQL statement"),
            "{}",
            text_of(&result)
        );
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn duckdb_query_cannot_read_files_by_literal_path() {
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.csv");
        std::fs::write(&secret, "k,v\nsecret,42\n").unwrap();
        let server = AnalyticsServer::new();
        let result = server
            .db_duckdb_query(Parameters(DuckDBQueryRequest {
                db_path: ":memory:".into(),
                query: format!("SELECT * FROM '{}'", secret.display()),
                name: Some("leak".into()),
            }))
            .await
            .unwrap();
        assert_eq!(result.is_error, Some(true));
        assert!(!server.datasets.read().await.contains_key("leak"));
    }

    #[test]
    fn duckdb_query_truncates_at_the_row_budget_and_says_so() {
        // sql_freeform=true: the hosted profile refuses free-form DuckDB SQL otherwise.
        let limits = Limits::hosted()
            .with_overrides(&[
                "db_result_rows=10".to_string(),
                "sql_freeform=true".to_string(),
            ])
            .unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = AnalyticsServer::new();
        let result = limits::with_limits(limits, || {
            rt.block_on(server.db_duckdb_query(Parameters(DuckDBQueryRequest {
                db_path: ":memory:".into(),
                query: "SELECT range AS x FROM range(1000)".into(),
                name: Some("t".into()),
            })))
        })
        .unwrap();
        let text = text_of(&result);
        assert_ne!(result.is_error, Some(true), "{text}");
        assert!(
            text.contains("Rows returned: 10") && text.contains("truncated"),
            "{text}"
        );
        assert_eq!(
            rt.block_on(server.datasets.read())
                .get("t")
                .unwrap()
                .nrows(),
            10
        );
    }

    #[test]
    fn sql_query_options_follow_installed_limits() {
        let o = limits::with_limits(Limits::hosted(), sql_query_options);
        assert_eq!(o.max_rows, Some(100_000));
        assert_eq!(o.max_bytes, Some(64 * 1024 * 1024));
        assert_eq!(o.timeout, Some(Duration::from_secs(20)));
        assert_eq!(o.memory_limit_mb, Some(256));
        assert_eq!(o.threads, Some(1));
        let o = limits::with_limits(Limits::off(), sql_query_options);
        assert!(
            o.max_rows.is_none()
                && o.max_bytes.is_none()
                && o.timeout.is_none()
                && o.memory_limit_mb.is_none()
                && o.threads.is_none()
        );
    }

    /// `db_query_file` reads only its own file (the core test
    /// `file_query_reads_only_the_named_file`), so its description offers one file and
    /// `db_duckdb_query`'s rules, not the joins of several files it offered before the lockdown.
    #[test]
    fn db_query_file_description_matches_the_lockdown() {
        let tool = AnalyticsServer::db_query_file_tool_attr();
        let d = tool.description.as_deref().unwrap_or_default();
        for needle in [
            "on one Parquet or CSV file, file_path,",
            "The query can read no other file, so it cannot join files.",
            "The rules and caps of db_duckdb_query apply",
            "Not available when the server disables free-form SQL",
        ] {
            assert!(d.contains(needle), "{needle:?} missing from: {d}");
        }
        assert!(!d.contains("joining large files"), "{d}");
    }

    #[test]
    fn redact_path_hides_resolved_server_paths() {
        let msg =
            "Permission Error: Cannot access file \"/srv/data/sessions/abc/in.csv\" - disabled";
        let out = redact_path(msg, Path::new("/srv/data/sessions/abc/in.csv"), "in.csv");
        assert!(
            out.contains("\"in.csv\"") && !out.contains("/srv/data"),
            "{out}"
        );
    }

    #[test]
    fn freeform_duckdb_sql_is_refused_when_sql_freeform_is_false() {
        // User decision 2026-10-21 (3). The file lies outside the jail on purpose: the
        // refusal must come before path validation, so the text is the refusal and never
        // a jail error or a server path.
        assert_eq!(
            SQL_FREEFORM_REFUSAL,
            "free-form SQL is not available on the public demo; load the file with load_dataset instead"
        );
        let outside = tempfile::tempdir().unwrap();
        let csv = outside.path().join("in.csv");
        std::fs::write(&csv, "a,b\n1,2\n").unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = AnalyticsServer::new();
        let (q, f) = limits::with_limits(Limits::hosted(), || {
            let q = rt
                .block_on(server.db_duckdb_query(Parameters(DuckDBQueryRequest {
                    db_path: ":memory:".into(),
                    query: "SELECT 1 AS x".into(),
                    name: Some("q".into()),
                })))
                .unwrap();
            let f = rt
                .block_on(server.db_query_file(Parameters(DuckDBFileQueryRequest {
                    file_path: csv.to_string_lossy().into_owned(),
                    query: "SELECT * FROM {file}".into(),
                    name: Some("f".into()),
                })))
                .unwrap();
            (q, f)
        });
        for r in [&q, &f] {
            assert_eq!(r.is_error, Some(true));
            assert_eq!(text_of(r), SQL_FREEFORM_REFUSAL);
        }
        let ds = rt.block_on(server.datasets.read());
        assert!(!ds.contains_key("q") && !ds.contains_key("f"));
    }

    #[test]
    fn freeform_duckdb_sql_runs_when_sql_freeform_is_true_and_sqlite_is_untouched() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = AnalyticsServer::new();
        let run = |l: Limits, name: &str| {
            limits::with_limits(l, || {
                rt.block_on(server.db_duckdb_query(Parameters(DuckDBQueryRequest {
                    db_path: ":memory:".into(),
                    query: "SELECT 1 AS x".into(),
                    name: Some(name.into()),
                })))
            })
            .unwrap()
        };
        let on = Limits::hosted()
            .with_overrides(&["sql_freeform=true".to_string()])
            .unwrap();
        for (l, name) in [(on, "override"), (Limits::off(), "off")] {
            let r = run(l, name);
            assert_ne!(r.is_error, Some(true), "{name}: {}", text_of(&r));
            assert!(rt.block_on(server.datasets.read()).contains_key(name));
        }
        // db_sqlite_query is not free-form DuckDB: whatever it answers here (no such
        // file), it is not the refusal.
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.db_sqlite_query(Parameters(SqliteQueryRequest {
                db_path: "p2a-wp10-missing.sqlite".into(),
                query: "SELECT 1".into(),
                name: None,
            })))
        })
        .unwrap();
        assert_ne!(text_of(&r), SQL_FREEFORM_REFUSAL);
    }

    #[test]
    fn allows_ordinary_queries() {
        assert!(guard_sql_file_access("SELECT * FROM my_table WHERE x > 1").is_ok());
        assert!(guard_sql_file_access("WITH t AS (SELECT 1) SELECT * FROM t").is_ok());
        // A column literally named similarly must not trip the identifier match.
        assert!(guard_sql_file_access("SELECT my_glob_col FROM t").is_ok());
    }

    #[test]
    fn allows_file_placeholder() {
        // The {file} placeholder is substituted with a jail-validated path.
        assert!(guard_sql_file_access("SELECT * FROM read_csv_auto({file})").is_ok());
        assert!(guard_sql_file_access("SELECT * FROM read_parquet( {file} )").is_ok());
    }

    #[test]
    fn rejects_literal_file_paths() {
        assert!(guard_sql_file_access("SELECT * FROM read_csv_auto('/etc/passwd')").is_err());
        assert!(guard_sql_file_access("SELECT * FROM read_parquet('/tmp/x.parquet')").is_err());
        assert!(guard_sql_file_access("select * from READ_JSON('/secret')").is_err());
        assert!(guard_sql_file_access("SELECT * FROM glob('/**')").is_err());
    }
}
