//! Database connectivity for SQLite and DuckDB.
//!
//! Provides functions to execute SQL queries and load results as Polars DataFrames.

use polars::prelude::*;
use std::path::Path;
use thiserror::Error;

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
}

/// Result of a database query.
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
    let path = db_path.as_ref();

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let conn = rusqlite::Connection::open(path)?;
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

    let conn = rusqlite::Connection::open(path)?;
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

    let conn = rusqlite::Connection::open(path)?;
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
    let path = db_path.as_ref();
    let path_str = path.to_string_lossy();

    // Allow :memory: for in-memory databases
    if path_str != ":memory:" && !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    let conn = duckdb::Connection::open(path)?;
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

    let conn = duckdb::Connection::open(path)?;
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

    let conn = duckdb::Connection::open(path)?;
    let query = format!(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{}' ORDER BY ordinal_position",
        table_name
    );
    let mut stmt = conn.prepare(&query)?;

    let mut schema = Vec::new();
    let mut rows = stmt.query([])?;

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
    let path = file_path.as_ref();

    if !path.exists() {
        return Err(DatabaseError::FileNotFound(path.display().to_string()));
    }

    // Use in-memory DuckDB
    let conn = duckdb::Connection::open_in_memory()?;

    // Replace placeholder with actual file path
    let full_query = query.replace("{file}", &format!("'{}'", path.display()));

    query_duckdb_connection(&conn, &full_query)
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
    use std::fs;

    #[test]
    fn test_sqlite_query() {
        // Create a temporary SQLite database
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_p2a.db");

        // Clean up if exists
        let _ = fs::remove_file(&db_path);

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

        // Clean up
        let _ = fs::remove_file(&db_path);
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
