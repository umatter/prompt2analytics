//! Database tool handlers (SQLite, DuckDB queries).
//!
//! This module defines database tools using the `#[tool_router(router = database_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use std::path::PathBuf;

use crate::path_jail;
use crate::server::AnalyticsServer;
use crate::tools::requests::database::*;

use p2a_core::{
    Dataset,
    data::{
        duckdb_table_schema, list_duckdb_tables, list_sqlite_tables, query_duckdb,
        query_file_with_duckdb, query_sqlite, sqlite_table_schema,
    },
};

/// Validate a database file path, allowing the special `:memory:` DuckDB path.
fn jail_db_path(requested: &str) -> Result<PathBuf, String> {
    if requested == ":memory:" {
        return Ok(PathBuf::from(":memory:"));
    }
    path_jail::validate_data_path(requested)
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
/// pre-validated path). A literal path argument such as
/// `read_csv_auto('/etc/passwd')` would otherwise escape the data-root jail,
/// so it is refused with guidance to use the `{file}` placeholder instead.
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
        description = "Execute a SQL query against a SQLite database and load the results as a dataset. The resulting dataset can then be analyzed using other tools."
    )]
    pub async fn db_sqlite_query(
        &self,
        Parameters(request): Parameters<SqliteQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let db_path = jail_or_return!(&request.db_path, "query SQLite database");
        guard_sql_or_return!(&request.query, "query SQLite database");
        let result = match query_sqlite(&db_path, &request.query) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SQLite query failed: {}",
                    e
                ))]));
            }
        };

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
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            result.rows,
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
                    e
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
                    e
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
        description = "Execute a SQL query against a DuckDB database and load the results as a dataset. DuckDB supports advanced analytics SQL including window functions, CTEs, and more."
    )]
    pub async fn db_duckdb_query(
        &self,
        Parameters(request): Parameters<DuckDBQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let db_path = jail_or_return!(&request.db_path, "query DuckDB database");
        guard_sql_or_return!(&request.query, "query DuckDB database");
        let result = match query_duckdb(&db_path, &request.query) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DuckDB query failed: {}",
                    e
                ))]));
            }
        };

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
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            result.rows,
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
                    e
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
                    e
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
        description = "Execute a SQL query directly on a Parquet or CSV file using DuckDB. This is powerful for filtering, aggregating, or joining large files before loading them as datasets. Use {file} as a placeholder for the file path in your query."
    )]
    pub async fn db_query_file(
        &self,
        Parameters(request): Parameters<DuckDBFileQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let file_path = jail_or_return!(&request.file_path, "query data file");
        guard_sql_or_return!(&request.query, "query data file");
        let result = match query_file_with_duckdb(&file_path, &request.query) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DuckDB file query failed: {}",
                    e
                ))]));
            }
        };

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
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            file_path.display(),
            result.rows,
            result.columns.join(", "),
            name,
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }
}

#[cfg(test)]
mod tests {
    use super::guard_sql_file_access;

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
