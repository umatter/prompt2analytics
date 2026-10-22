//! Database connectivity for SQLite and DuckDB.
//!
//! Provides functions to execute SQL queries and load results as Polars DataFrames.

use polars::prelude::*;
use std::path::Path;
use thiserror::Error;

/// Validate that a string is a safe SQL identifier (table name, column name, etc.).
/// Only allows alphanumeric characters and underscores.
fn validate_identifier(name: &str) -> Result<(), DatabaseError> {
    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Ok(())
    } else {
        Err(DatabaseError::UnsupportedType(format!(
            "Invalid SQL identifier: '{}'. Only alphanumeric characters and underscores are allowed.",
            name
        )))
    }
}

/// Database-related errors.
#[derive(Debug, Error)]
pub enum DatabaseError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("DuckDB error: {0}")]
    DuckDB(#[from] duckdb::Error),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("Unsupported column type: {0}")]
    UnsupportedType(String),

    #[error("Database file not found: {0}")]
    FileNotFound(String),

    #[error("Query rejected: {0}")]
    Forbidden(String),
}

/// Validate that a SQL statement is a read-only query.
///
/// Accepts `SELECT`, `WITH` (CTE), `EXPLAIN`, `PRAGMA` (metadata), `DESCRIBE`,
/// and `SHOW`. Anything else — `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`,
/// `CREATE`, `ATTACH`, `COPY`, `INSTALL`, `LOAD`, … — is rejected.
///
/// This is a keyword-level filter, not a full SQL parser. It strips SQL block
/// comments (`/* ... */`) and line comments (`-- ...`) from the leading whitespace
/// before inspecting the first statement keyword.
pub fn validate_read_only_query(query: &str) -> Result<(), DatabaseError> {
    const ALLOWED: &[&str] = &["select", "with", "explain", "pragma", "describe", "show"];

    let cleaned = strip_leading_comments(query);
    let first_token: String = cleaned
        .chars()
        .take_while(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();

    if first_token.is_empty() {
        return Err(DatabaseError::Forbidden("empty SQL statement".to_string()));
    }
    if !ALLOWED.contains(&first_token.as_str()) {
        return Err(DatabaseError::Forbidden(format!(
            "statement starting with `{}` is not allowed; only read-only queries \
             (SELECT, WITH, EXPLAIN, PRAGMA, DESCRIBE, SHOW) may be executed",
            first_token
        )));
    }
    Ok(())
}

/// Refuse any input that is not exactly one SQL statement.
///
/// The text may end in `;`, further `;`, whitespace and comments, but nothing
/// else may follow the first top-level `;`. Strings (`'..'`, `E'..'`), quoted
/// identifiers, dollar quotes (`$$..$$`, `$tag$..$tag$`) and comments (`--`,
/// nested `/* */`) are skipped with DuckDB's rules, so a `;` inside them does
/// not count. Unterminated strings or comments, empty input and NUL bytes are
/// refused as well.
pub fn ensure_single_statement(query: &str) -> Result<(), DatabaseError> {
    single_statement_body(query).map(|_| ())
}

const MULTIPLE_STATEMENTS: &str =
    "only one SQL statement may be run per call; remove everything after the first ';'";
const UNTERMINATED: &str = "unterminated string literal, quoted identifier or comment";

/// Bytes that may continue an identifier (DuckDB: letters, digits, `_`, `$`,
/// and any non-ASCII byte).
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$' || b >= 0x80
}

/// The single statement in `query`, without its terminating `;` and anything
/// after it. The text is not trimmed: a trailing line comment stays part of
/// the body (callers that wrap the body put a newline after it).
fn single_statement_body(query: &str) -> Result<&str, DatabaseError> {
    let bytes = query.as_bytes();
    if bytes.contains(&0) {
        return Err(DatabaseError::Forbidden(
            "the query contains a NUL byte".to_string(),
        ));
    }
    let unterminated = || DatabaseError::Forbidden(UNTERMINATED.to_string());

    let n = bytes.len();
    let mut i = 0;
    let mut has_code = false;
    let mut terminator = None;
    while i < n {
        let b = bytes[i];
        let next = bytes.get(i + 1).copied();
        match b {
            b'\'' => {
                let escape = i >= 1
                    && matches!(bytes[i - 1], b'E' | b'e')
                    && (i < 2 || !is_ident_byte(bytes[i - 2]));
                i = skip_string(bytes, i + 1, escape).ok_or_else(unterminated)?;
                has_code = true;
            }
            b'"' => {
                i = skip_quoted_identifier(bytes, i + 1).ok_or_else(unterminated)?;
                has_code = true;
            }
            b'$' if i == 0 || !is_ident_byte(bytes[i - 1]) => {
                match dollar_tag_len(&bytes[i..]) {
                    Some(len) => {
                        let tag = &bytes[i..i + len];
                        let start = i + len;
                        let close = bytes[start..]
                            .windows(len)
                            .position(|w| w == tag)
                            .ok_or_else(unterminated)?;
                        i = start + close + len;
                    }
                    None => i += 1,
                }
                has_code = true;
            }
            b'-' if next == Some(b'-') => i = skip_line_comment(bytes, i + 2),
            b'/' if next == Some(b'*') => {
                i = skip_block_comment(bytes, i + 2).ok_or_else(unterminated)?;
            }
            b';' => {
                terminator = Some(i);
                break;
            }
            b if b.is_ascii_whitespace() => i += 1,
            _ => {
                has_code = true;
                i += 1;
            }
        }
    }

    let Some(end) = terminator else {
        return if has_code {
            Ok(query)
        } else {
            Err(DatabaseError::Forbidden("empty SQL statement".to_string()))
        };
    };

    // After the first top-level `;`: only blanks, `;` and terminated comments.
    let mut j = end + 1;
    while j < n {
        let b = bytes[j];
        let next = bytes.get(j + 1).copied();
        match b {
            b';' => j += 1,
            b if b.is_ascii_whitespace() => j += 1,
            b'-' if next == Some(b'-') => j = skip_line_comment(bytes, j + 2),
            b'/' if next == Some(b'*') => {
                j = skip_block_comment(bytes, j + 2).ok_or_else(unterminated)?;
            }
            _ => return Err(DatabaseError::Forbidden(MULTIPLE_STATEMENTS.to_string())),
        }
    }
    if !has_code {
        return Err(DatabaseError::Forbidden("empty SQL statement".to_string()));
    }
    Ok(&query[..end])
}

/// Index just past the closing `'` of a string whose body starts at `i`.
/// `''` stays inside the string; in an escape string (`E'..'`) a backslash
/// also skips the next byte. `None` if the string is unterminated.
fn skip_string(bytes: &[u8], mut i: usize, escape: bool) -> Option<usize> {
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if escape => i += 2,
            b'\'' if bytes.get(i + 1) == Some(&b'\'') => i += 2,
            b'\'' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// Index just past the closing `"` of a quoted identifier whose body starts at `i`.
fn skip_quoted_identifier(bytes: &[u8], mut i: usize) -> Option<usize> {
    while i < bytes.len() {
        match bytes[i] {
            b'"' if bytes.get(i + 1) == Some(&b'"') => i += 2,
            b'"' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// Length of the dollar-quote delimiter (`$$` or `$tag$`) at the start of `s`,
/// which begins with `$`; `None` if `s` does not start a dollar quote (`$1`).
fn dollar_tag_len(s: &[u8]) -> Option<usize> {
    let tag_start = |b: u8| b.is_ascii_alphabetic() || b == b'_' || b >= 0x80;
    let tag_byte = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80;
    match s.get(1) {
        Some(b'$') => Some(2),
        Some(&b) if tag_start(b) => {
            let mut k = 2;
            while k < s.len() && tag_byte(s[k]) {
                k += 1;
            }
            (s.get(k) == Some(&b'$')).then_some(k + 1)
        }
        _ => None,
    }
}

/// Index of the byte after the newline ending a line comment whose body
/// starts at `i`, or the end of the input.
fn skip_line_comment(bytes: &[u8], i: usize) -> usize {
    let i = i.min(bytes.len());
    match bytes[i..].iter().position(|&b| b == b'\n') {
        Some(p) => i + p + 1,
        None => bytes.len(),
    }
}

/// Index just past the `*/` closing a (nested) block comment whose body starts
/// at `i`; `None` if it is unterminated.
fn skip_block_comment(bytes: &[u8], mut i: usize) -> Option<usize> {
    let mut depth = 1usize;
    while i < bytes.len() {
        match (bytes[i], bytes.get(i + 1)) {
            (b'/', Some(b'*')) => {
                depth += 1;
                i += 2;
            }
            (b'*', Some(b'/')) => {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => i += 1,
        }
    }
    None
}

fn strip_leading_comments(query: &str) -> &str {
    let mut rest = query.trim_start();
    loop {
        if let Some(after) = rest.strip_prefix("--") {
            // Line comment: drop through end of line (or end of string).
            let end = after.find('\n').map(|i| i + 1).unwrap_or(after.len());
            rest = after[end..].trim_start();
            continue;
        }
        if let Some(after) = rest.strip_prefix("/*") {
            // Block comment: drop through matching `*/`.
            match after.find("*/") {
                Some(i) => {
                    rest = after[i + 2..].trim_start();
                    continue;
                }
                None => {
                    // Unterminated comment; let the SQL parser report it.
                    return rest;
                }
            }
        }
        return rest;
    }
}

/// Result of a database query.
#[derive(Debug)]
pub struct QueryResult {
    /// The resulting DataFrame
    pub dataframe: DataFrame,
    /// Number of rows returned
    pub rows: usize,
    /// Column names
    pub columns: Vec<String>,
}

impl std::fmt::Display for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Query Result")?;
        writeln!(f, "============")?;
        writeln!(f, "Rows returned: {}", self.rows)?;
        writeln!(f, "Columns: {}", self.columns.join(", "))?;
        writeln!(f)?;
        writeln!(f, "Preview (first 5 rows):")?;
        write!(f, "{}", self.dataframe.head(Some(5)))
    }
}

// ============================================================================
// SQLite Support
// ============================================================================

/// Execute a SQL query against a SQLite database and return results as a DataFrame.
///
/// # Arguments
/// * `db_path` - Path to the SQLite database file
/// * `query` - SQL query to execute
///
/// # Returns
/// QueryResult containing the DataFrame and metadata
pub fn query_sqlite(db_path: impl AsRef<Path>, query: &str) -> Result<QueryResult, DatabaseError> {
    let query = single_statement_body(query)?;
    validate_read_only_query(query)?;
    let path = db_path.as_ref();

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    // Open read-only; combined with validate_read_only_query this blocks DDL/DML
    // even if the keyword filter is ever bypassed.
    let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
        | rusqlite::OpenFlags::SQLITE_OPEN_URI
        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = rusqlite::Connection::open_with_flags(path, flags)?;
    let mut stmt = conn.prepare(query)?;

    // Get column information
    let column_count = stmt.column_count();
    let column_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

    // Collect all rows first to determine types
    let mut rows_data: Vec<Vec<rusqlite::types::Value>> = Vec::new();
    let mut rows = stmt.query([])?;

    while let Some(row) = rows.next()? {
        let mut row_values = Vec::with_capacity(column_count);
        for i in 0..column_count {
            let value: rusqlite::types::Value = row.get_ref(i)?.into();
            row_values.push(value);
        }
        rows_data.push(row_values);
    }

    let n_rows = rows_data.len();

    // Build columns for DataFrame
    let mut columns_vec: Vec<Column> = Vec::with_capacity(column_count);

    for col_idx in 0..column_count {
        let col_name = &column_names[col_idx];

        // Determine column type from first non-null value
        let col_type = rows_data
            .iter()
            .find_map(|row| match &row[col_idx] {
                rusqlite::types::Value::Null => None,
                v => Some(v.data_type()),
            })
            .unwrap_or(rusqlite::types::Type::Text);

        let series = match col_type {
            rusqlite::types::Type::Integer => {
                let values: Vec<Option<i64>> = rows_data
                    .iter()
                    .map(|row| match &row[col_idx] {
                        rusqlite::types::Value::Integer(i) => Some(*i),
                        rusqlite::types::Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            rusqlite::types::Type::Real => {
                let values: Vec<Option<f64>> = rows_data
                    .iter()
                    .map(|row| match &row[col_idx] {
                        rusqlite::types::Value::Real(f) => Some(*f),
                        rusqlite::types::Value::Integer(i) => Some(*i as f64),
                        rusqlite::types::Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            rusqlite::types::Type::Text | rusqlite::types::Type::Blob => {
                let values: Vec<Option<String>> = rows_data
                    .iter()
                    .map(|row| match &row[col_idx] {
                        rusqlite::types::Value::Text(s) => Some(s.clone()),
                        rusqlite::types::Value::Blob(b) => {
                            Some(String::from_utf8_lossy(b).to_string())
                        }
                        rusqlite::types::Value::Null => None,
                        rusqlite::types::Value::Integer(i) => Some(i.to_string()),
                        rusqlite::types::Value::Real(f) => Some(f.to_string()),
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            rusqlite::types::Type::Null => {
                // All nulls - default to string
                let values: Vec<Option<String>> = vec![None; n_rows];
                Series::new(col_name.as_str().into(), values)
            }
        };

        columns_vec.push(series.into());
    }

    let df = DataFrame::new(columns_vec)?;

    Ok(QueryResult {
        dataframe: df,
        rows: n_rows,
        columns: column_names,
    })
}

/// List all tables in a SQLite database.
pub fn list_sqlite_tables(db_path: impl AsRef<Path>) -> Result<Vec<String>, DatabaseError> {
    let path = db_path.as_ref();

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
        | rusqlite::OpenFlags::SQLITE_OPEN_URI
        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = rusqlite::Connection::open_with_flags(path, flags)?;
    let mut stmt =
        conn.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")?;

    let tables: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(tables)
}

/// Get schema information for a SQLite table.
pub fn sqlite_table_schema(
    db_path: impl AsRef<Path>,
    table_name: &str,
) -> Result<Vec<(String, String)>, DatabaseError> {
    let path = db_path.as_ref();

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
        | rusqlite::OpenFlags::SQLITE_OPEN_URI
        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = rusqlite::Connection::open_with_flags(path, flags)?;
    validate_identifier(table_name)?;
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table_name))?;

    let schema: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(schema)
}

// ============================================================================
// DuckDB Support
// ============================================================================

/// Execute a SQL query against a DuckDB database and return results as a DataFrame.
///
/// # Arguments
/// * `db_path` - Path to the DuckDB database file (use ":memory:" for in-memory)
/// * `query` - SQL query to execute
///
/// # Returns
/// QueryResult containing the DataFrame and metadata
pub fn query_duckdb(db_path: impl AsRef<Path>, query: &str) -> Result<QueryResult, DatabaseError> {
    let query = single_statement_body(query)?;
    validate_read_only_query(query)?;
    let path = db_path.as_ref();
    let path_str = path.to_string_lossy();

    // Allow :memory: for in-memory databases
    if path_str != ":memory:" && !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    // Open read-only. In-memory databases don't support ReadOnly mode so fall back
    // to the default configuration for those; the keyword filter already rejects DDL/DML.
    let conn = if path_str == ":memory:" {
        duckdb::Connection::open(path)?
    } else {
        let config = duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?;
        duckdb::Connection::open_with_flags(path, config)?
    };
    let mut stmt = conn.prepare(query)?;

    // Execute query first - DuckDB requires this before accessing column info
    let mut rows = stmt.query([])?;

    // Get column information after execution
    let column_count = rows.as_ref().map(|r| r.column_count()).unwrap_or(0);
    let column_names: Vec<String> = rows
        .as_ref()
        .map(|r| r.column_names().iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();

    // DuckDB Arrow integration would be ideal, but for simplicity use row-by-row
    let mut all_rows: Vec<Vec<duckdb::types::Value>> = Vec::new();

    while let Some(row) = rows.next()? {
        let mut row_values = Vec::with_capacity(column_count);
        for i in 0..column_count {
            let value: duckdb::types::Value = row.get(i)?;
            row_values.push(value);
        }
        all_rows.push(row_values);
    }

    let n_rows = all_rows.len();

    // Build columns for DataFrame
    let mut columns_vec: Vec<Column> = Vec::with_capacity(column_count);

    for col_idx in 0..column_count {
        let col_name = &column_names[col_idx];

        // Determine column type from first non-null value
        let first_non_null = all_rows.iter().find_map(|row| match &row[col_idx] {
            duckdb::types::Value::Null => None,
            v => Some(v.clone()),
        });

        let series = match first_non_null {
            Some(duckdb::types::Value::Boolean(_)) => {
                let values: Vec<Option<bool>> = all_rows
                    .iter()
                    .map(|row| match &row[col_idx] {
                        duckdb::types::Value::Boolean(b) => Some(*b),
                        duckdb::types::Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            Some(duckdb::types::Value::TinyInt(_))
            | Some(duckdb::types::Value::SmallInt(_))
            | Some(duckdb::types::Value::Int(_))
            | Some(duckdb::types::Value::BigInt(_)) => {
                let values: Vec<Option<i64>> = all_rows
                    .iter()
                    .map(|row| match &row[col_idx] {
                        duckdb::types::Value::TinyInt(i) => Some(*i as i64),
                        duckdb::types::Value::SmallInt(i) => Some(*i as i64),
                        duckdb::types::Value::Int(i) => Some(*i as i64),
                        duckdb::types::Value::BigInt(i) => Some(*i),
                        duckdb::types::Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            Some(duckdb::types::Value::Float(_)) | Some(duckdb::types::Value::Double(_)) => {
                let values: Vec<Option<f64>> = all_rows
                    .iter()
                    .map(|row| match &row[col_idx] {
                        duckdb::types::Value::Float(f) => Some(*f as f64),
                        duckdb::types::Value::Double(f) => Some(*f),
                        duckdb::types::Value::TinyInt(i) => Some(*i as f64),
                        duckdb::types::Value::SmallInt(i) => Some(*i as f64),
                        duckdb::types::Value::Int(i) => Some(*i as f64),
                        duckdb::types::Value::BigInt(i) => Some(*i as f64),
                        duckdb::types::Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            _ => {
                // Default to string for text, blob, and unknown types
                let values: Vec<Option<String>> = all_rows
                    .iter()
                    .map(|row| match &row[col_idx] {
                        duckdb::types::Value::Text(s) => Some(s.clone()),
                        duckdb::types::Value::Blob(b) => {
                            Some(String::from_utf8_lossy(b).to_string())
                        }
                        duckdb::types::Value::Null => None,
                        duckdb::types::Value::Boolean(b) => Some(b.to_string()),
                        duckdb::types::Value::TinyInt(i) => Some(i.to_string()),
                        duckdb::types::Value::SmallInt(i) => Some(i.to_string()),
                        duckdb::types::Value::Int(i) => Some(i.to_string()),
                        duckdb::types::Value::BigInt(i) => Some(i.to_string()),
                        duckdb::types::Value::Float(f) => Some(f.to_string()),
                        duckdb::types::Value::Double(f) => Some(f.to_string()),
                        _ => Some("".to_string()),
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
        };

        columns_vec.push(series.into());
    }

    let df = DataFrame::new(columns_vec)?;

    Ok(QueryResult {
        dataframe: df,
        rows: n_rows,
        columns: column_names,
    })
}

/// List all tables in a DuckDB database.
pub fn list_duckdb_tables(db_path: impl AsRef<Path>) -> Result<Vec<String>, DatabaseError> {
    let path = db_path.as_ref();
    let path_str = path.to_string_lossy();

    if path_str != ":memory:" && !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let conn = if path_str == ":memory:" {
        duckdb::Connection::open(path)?
    } else {
        let config = duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?;
        duckdb::Connection::open_with_flags(path, config)?
    };
    let mut stmt = conn
        .prepare("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'")?;

    let mut tables = Vec::new();
    let mut rows = stmt.query([])?;

    while let Some(row) = rows.next()? {
        let name: String = row.get(0)?;
        tables.push(name);
    }

    Ok(tables)
}

/// Get schema information for a DuckDB table.
pub fn duckdb_table_schema(
    db_path: impl AsRef<Path>,
    table_name: &str,
) -> Result<Vec<(String, String)>, DatabaseError> {
    let path = db_path.as_ref();
    let path_str = path.to_string_lossy();

    if path_str != ":memory:" && !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let conn = if path_str == ":memory:" {
        duckdb::Connection::open(path)?
    } else {
        let config = duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?;
        duckdb::Connection::open_with_flags(path, config)?
    };
    let query = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position";
    let mut stmt = conn.prepare(query)?;

    let mut schema = Vec::new();
    let mut rows = stmt.query([table_name])?;

    while let Some(row) = rows.next()? {
        let col_name: String = row.get(0)?;
        let col_type: String = row.get(1)?;
        schema.push((col_name, col_type));
    }

    Ok(schema)
}

/// Execute a DuckDB query directly on a Parquet/CSV file without loading into database.
/// DuckDB can query files directly, which is very powerful for analytics.
pub fn query_file_with_duckdb(
    file_path: impl AsRef<Path>,
    query: &str,
) -> Result<QueryResult, DatabaseError> {
    // External-access SQL functions (read_csv_auto, read_parquet, …) are the whole
    // point of this entry point, so we cannot disable external access wholesale;
    // but the statement itself must still be a read-only query.
    let path = file_path.as_ref();

    // Replace placeholder with actual file path, escaping single quotes to prevent
    // injection. The path sits inside a quoted literal, so it cannot open a second
    // statement; the lexer checks the substituted text.
    let sanitized_path = path.display().to_string().replace('\'', "''");
    let full_query = query.replace("{file}", &format!("'{}'", sanitized_path));
    let body = single_statement_body(&full_query)?;
    validate_read_only_query(body)?;

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    // Use in-memory DuckDB
    let conn = duckdb::Connection::open_in_memory()?;

    query_duckdb_connection(&conn, body)
}

/// Internal helper to query an existing DuckDB connection.
fn query_duckdb_connection(
    conn: &duckdb::Connection,
    query: &str,
) -> Result<QueryResult, DatabaseError> {
    let mut stmt = conn.prepare(query)?;

    // Execute query first - DuckDB requires this before accessing column info
    let mut rows = stmt.query([])?;

    // Get column information after execution
    let column_count = rows.as_ref().map(|r| r.column_count()).unwrap_or(0);
    let column_names: Vec<String> = rows
        .as_ref()
        .map(|r| r.column_names().iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();

    let mut all_rows: Vec<Vec<duckdb::types::Value>> = Vec::new();

    while let Some(row) = rows.next()? {
        let mut row_values = Vec::with_capacity(column_count);
        for i in 0..column_count {
            let value: duckdb::types::Value = row.get(i)?;
            row_values.push(value);
        }
        all_rows.push(row_values);
    }

    let n_rows = all_rows.len();
    let mut columns_vec: Vec<Column> = Vec::with_capacity(column_count);

    for col_idx in 0..column_count {
        let col_name = &column_names[col_idx];

        let first_non_null = all_rows.iter().find_map(|row| match &row[col_idx] {
            duckdb::types::Value::Null => None,
            v => Some(v.clone()),
        });

        let series = match first_non_null {
            Some(duckdb::types::Value::Boolean(_)) => {
                let values: Vec<Option<bool>> = all_rows
                    .iter()
                    .map(|row| match &row[col_idx] {
                        duckdb::types::Value::Boolean(b) => Some(*b),
                        _ => None,
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            Some(duckdb::types::Value::TinyInt(_))
            | Some(duckdb::types::Value::SmallInt(_))
            | Some(duckdb::types::Value::Int(_))
            | Some(duckdb::types::Value::BigInt(_)) => {
                let values: Vec<Option<i64>> = all_rows
                    .iter()
                    .map(|row| match &row[col_idx] {
                        duckdb::types::Value::TinyInt(i) => Some(*i as i64),
                        duckdb::types::Value::SmallInt(i) => Some(*i as i64),
                        duckdb::types::Value::Int(i) => Some(*i as i64),
                        duckdb::types::Value::BigInt(i) => Some(*i),
                        _ => None,
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            Some(duckdb::types::Value::Float(_)) | Some(duckdb::types::Value::Double(_)) => {
                let values: Vec<Option<f64>> = all_rows
                    .iter()
                    .map(|row| match &row[col_idx] {
                        duckdb::types::Value::Float(f) => Some(*f as f64),
                        duckdb::types::Value::Double(f) => Some(*f),
                        duckdb::types::Value::TinyInt(i) => Some(*i as f64),
                        duckdb::types::Value::SmallInt(i) => Some(*i as f64),
                        duckdb::types::Value::Int(i) => Some(*i as f64),
                        duckdb::types::Value::BigInt(i) => Some(*i as f64),
                        _ => None,
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
            _ => {
                let values: Vec<Option<String>> = all_rows
                    .iter()
                    .map(|row| match &row[col_idx] {
                        duckdb::types::Value::Text(s) => Some(s.clone()),
                        duckdb::types::Value::Null => None,
                        duckdb::types::Value::Boolean(b) => Some(b.to_string()),
                        duckdb::types::Value::TinyInt(i) => Some(i.to_string()),
                        duckdb::types::Value::SmallInt(i) => Some(i.to_string()),
                        duckdb::types::Value::Int(i) => Some(i.to_string()),
                        duckdb::types::Value::BigInt(i) => Some(i.to_string()),
                        duckdb::types::Value::Float(f) => Some(f.to_string()),
                        duckdb::types::Value::Double(f) => Some(f.to_string()),
                        _ => Some("".to_string()),
                    })
                    .collect();
                Series::new(col_name.as_str().into(), values)
            }
        };

        columns_vec.push(series.into());
    }

    let df = DataFrame::new(columns_vec)?;

    Ok(QueryResult {
        dataframe: df,
        rows: n_rows,
        columns: column_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_file(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn dir_names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    fn i64_col(df: &DataFrame, name: &str) -> Vec<Option<i64>> {
        df.column(name)
            .unwrap()
            .as_materialized_series()
            .i64()
            .unwrap()
            .into_iter()
            .collect()
    }

    /// `t(x INTEGER)` holding 0..=999.
    fn sqlite_fixture(dir: &Path) -> PathBuf {
        let db = dir.join("t.sqlite");
        rusqlite::Connection::open(&db)
            .unwrap()
            .execute_batch(
                "CREATE TABLE t(x INTEGER); \
                 WITH RECURSIVE c(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM c WHERE n < 999) \
                 INSERT INTO t SELECT n FROM c;",
            )
            .unwrap();
        db
    }

    // ------------------------------------------------------------------
    // Single statement (LLM-1)
    // ------------------------------------------------------------------

    #[test]
    fn single_statement_accepts_one_statement() {
        for q in [
            "SELECT 1",
            "SELECT 1;",
            "SELECT 1; -- done",
            "SELECT 1;;",
            "SELECT 1 /* c */ ;  \n",
            r"SELECT 'a;b'",
            r"SELECT 'it''s; fine'",
            r"SELECT E'it\'s; fine'",
            "SELECT $$a;b$$",
            "SELECT $tag$a;$$b$tag$",
            r#"SELECT "a;b" FROM t"#,
            r#"SELECT "a""b;" FROM t"#,
            "SELECT /* ; */ 1",
            "SELECT /* outer /* inner ; */ still ; */ 2",
            "SELECT 1 -- ; trailing",
            "SELECT $1",
            "WITH x AS (SELECT 1) SELECT * FROM x",
        ] {
            assert!(
                ensure_single_statement(q).is_ok(),
                "expected {q:?} to be accepted"
            );
        }
        assert_eq!(
            single_statement_body("SELECT 1; -- done").unwrap(),
            "SELECT 1"
        );
        assert_eq!(
            single_statement_body("SELECT 'a;b';").unwrap(),
            "SELECT 'a;b'"
        );
        assert_eq!(
            single_statement_body("SELECT 1 -- c").unwrap(),
            "SELECT 1 -- c"
        );
    }

    #[test]
    fn single_statement_rejects_a_second_statement() {
        for q in [
            "SELECT 1; SELECT 2",
            "SELECT 1;\nATTACH 'x.duckdb' AS e",
            "SELECT 1 -- c\n; SELECT 2",
            // E-string: DuckDB sees two statements
            r"SELECT E'\''; ATTACH 'x' AS e; --'",
            // nested comment: DuckDB sees two statements
            r"SELECT 1 /* /* */ ' */ ; ATTACH 'x' AS e; -- '",
            "SELECT $$ x $$; SELECT 2",
            // `$` inside an identifier is not a quote
            "SELECT a$b$ FROM t; SELECT 2",
        ] {
            let err = ensure_single_statement(q).unwrap_err();
            assert!(matches!(err, DatabaseError::Forbidden(_)), "{q:?}: {err}");
            assert!(
                err.to_string().contains("only one SQL statement"),
                "{q:?}: {err}"
            );
        }
    }

    #[test]
    fn single_statement_rejects_unterminated_empty_and_nul() {
        for (q, needle) in [
            ("", "empty"),
            ("   ", "empty"),
            (";", "empty"),
            ("-- only a comment", "empty"),
            ("/* c */ ;", "empty"),
            ("SELECT 'abc", "unterminated"),
            ("SELECT /* x", "unterminated"),
            ("SELECT $$x", "unterminated"),
            (r#"SELECT "abc"#, "unterminated"),
            ("SELECT 1\0", "NUL"),
        ] {
            let err = ensure_single_statement(q).unwrap_err();
            assert!(
                matches!(err, DatabaseError::Forbidden(_)) && err.to_string().contains(needle),
                "{q:?}: {err}"
            );
        }
    }

    #[test]
    fn query_duckdb_refuses_multi_statement_and_creates_no_file() {
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("evil.duckdb");
        let q = format!(
            "SELECT 1; ATTACH '{}' AS e; CREATE TABLE e.s AS SELECT 'x' AS v",
            target.display()
        );
        let err = query_duckdb(":memory:", &q).unwrap_err();
        assert!(matches!(err, DatabaseError::Forbidden(_)), "{err}");
        assert!(!target.exists(), "a second statement reached the engine");
    }

    #[test]
    fn query_sqlite_refuses_trailing_statement() {
        let dir = tempfile::tempdir().unwrap();
        let db = sqlite_fixture(dir.path());
        assert!(matches!(
            query_sqlite(&db, "SELECT x FROM t; SELECT 2").unwrap_err(),
            DatabaseError::Forbidden(_)
        ));
    }

    #[test]
    fn query_file_refuses_trailing_statement() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_file(dir.path(), "in.csv", "a,b\n1,2\n3,4\n");
        let out = dir.path().join("copied.csv");
        let q = format!(
            "SELECT * FROM {{file}}; COPY (SELECT 1) TO '{}'",
            out.display()
        );
        assert!(matches!(
            query_file_with_duckdb(&csv, &q).unwrap_err(),
            DatabaseError::Forbidden(_)
        ));
        assert!(!out.exists());
    }

    #[test]
    fn test_validate_read_only_allows_select() {
        assert!(validate_read_only_query("SELECT * FROM t").is_ok());
        assert!(validate_read_only_query("  select 1").is_ok());
        assert!(validate_read_only_query("WITH x AS (SELECT 1) SELECT * FROM x").is_ok());
        assert!(validate_read_only_query("EXPLAIN SELECT 1").is_ok());
        assert!(validate_read_only_query("PRAGMA table_info('t')").is_ok());
        assert!(validate_read_only_query("DESCRIBE t").is_ok());
        assert!(validate_read_only_query("SHOW TABLES").is_ok());
    }

    #[test]
    fn test_validate_read_only_rejects_mutations() {
        for q in [
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET x=1",
            "DELETE FROM t",
            "DROP TABLE t",
            "ALTER TABLE t ADD COLUMN x INT",
            "CREATE TABLE t (x INT)",
            "ATTACH 'evil.db' AS e",
            "COPY t TO 'out.csv'",
            "INSTALL 'httpfs'",
            "LOAD 'httpfs'",
            "",
            "   ",
        ] {
            assert!(
                validate_read_only_query(q).is_err(),
                "expected {:?} to be rejected",
                q
            );
        }
    }

    #[test]
    fn test_validate_read_only_strips_comments() {
        assert!(validate_read_only_query("-- a comment\nSELECT 1").is_ok());
        assert!(validate_read_only_query("/* block */ SELECT 1").is_ok());
        assert!(validate_read_only_query("-- hide\nDROP TABLE t").is_err());
    }

    #[test]
    fn test_sqlite_query() {
        // A fresh temporary directory: a fixed path would race between parallel runs.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_p2a.db");

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT, value REAL)", [])
            .unwrap();
        conn.execute("INSERT INTO test VALUES (1, 'Alice', 10.5)", [])
            .unwrap();
        conn.execute("INSERT INTO test VALUES (2, 'Bob', 20.3)", [])
            .unwrap();
        drop(conn);

        let result = query_sqlite(&db_path, "SELECT * FROM test").unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.columns.len(), 3);
    }

    #[test]
    fn test_duckdb_query() {
        // Use in-memory DuckDB
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE test (id INTEGER, name VARCHAR, value DOUBLE)",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO test VALUES (1, 'Alice', 10.5)", [])
            .unwrap();
        conn.execute("INSERT INTO test VALUES (2, 'Bob', 20.3)", [])
            .unwrap();

        let result = query_duckdb_connection(&conn, "SELECT * FROM test").unwrap();
        assert_eq!(result.rows, 2);
        assert_eq!(result.columns.len(), 3);
    }
}
