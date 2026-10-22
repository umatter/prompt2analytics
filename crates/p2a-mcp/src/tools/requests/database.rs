//! Request types for database tools (SQLite, DuckDB queries).

use schemars::JsonSchema;
use serde::Deserialize;

// ============================================================================
// SQLite Requests
// ============================================================================

/// Request to query a SQLite database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteQueryRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file (.db, .sqlite, .sqlite3).")]
    pub db_path: String,

    /// SQL query to execute
    #[schemars(
        description = "Exactly one read-only SQL statement (SELECT, WITH, EXPLAIN, or a schema PRAGMA: table_info, table_list, index_list, ...). Statements that would modify the database, ATTACH, other PRAGMAs and any second statement are refused. The server may cap the rows returned."
    )]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the resulting dataset. If not provided, a default name will be generated."
    )]
    pub name: Option<String>,
}

/// Request to list tables in a SQLite database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteListTablesRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file.")]
    pub db_path: String,
}

/// Request to get schema for a SQLite table.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteSchemaRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file.")]
    pub db_path: String,

    /// Table name
    #[schemars(description = "Name of the table to get schema for.")]
    pub table_name: String,
}

// ============================================================================
// DuckDB Requests
// ============================================================================

/// Request to query a DuckDB database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBQueryRequest {
    /// Path to the DuckDB database file
    #[schemars(
        description = "Path to the DuckDB database file (.duckdb, .db), opened read-only. Use ':memory:' for an empty in-memory database; it cannot read files (use db_query_file with {file} for CSV or Parquet)."
    )]
    pub db_path: String,

    /// SQL query to execute
    #[schemars(
        description = "Exactly one read-only SQL statement (SELECT, WITH, EXPLAIN, PRAGMA, DESCRIBE or SHOW). DuckDB supports advanced analytics SQL. ATTACH, COPY, INSTALL/LOAD, SET and any second statement are refused. The server may cap the rows returned."
    )]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the resulting dataset.")]
    pub name: Option<String>,
}

/// Request to list tables in a DuckDB database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBListTablesRequest {
    /// Path to the DuckDB database file
    #[schemars(description = "Path to the DuckDB database file.")]
    pub db_path: String,
}

/// Request to get schema for a DuckDB table.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBSchemaRequest {
    /// Path to the DuckDB database file
    #[schemars(description = "Path to the DuckDB database file.")]
    pub db_path: String,

    /// Table name
    #[schemars(description = "Name of the table to get schema for.")]
    pub table_name: String,
}

/// Request to query a file (Parquet, CSV) using DuckDB SQL.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBFileQueryRequest {
    /// Path to the data file (Parquet or CSV)
    #[schemars(
        description = "Path to the data file (.parquet, .csv). DuckDB can query these files directly with SQL; the query can read this file and no other."
    )]
    pub file_path: String,

    /// SQL query to execute
    #[schemars(
        description = "Exactly one read-only SQL statement. Use {file} as the placeholder for the file path, also inside reader functions. Example: 'SELECT * FROM {file} WHERE amount > 100' or 'SELECT * FROM read_csv_auto({file}, header = true)'. The server may cap the rows returned."
    )]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(
        description = "Optional name for the resulting dataset. If not provided, one will be generated."
    )]
    pub name: Option<String>,
}
