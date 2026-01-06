//! Data loading and manipulation module.
//!
//! Provides functionality for loading datasets from various file formats
//! and basic data manipulation operations.

mod loader;
mod dataset;
mod stata;
mod sas;
mod database;

pub use loader::DataLoader;
pub use dataset::{Dataset, DatasetInfo};
pub use stata::{load_stata, StataError};
pub use sas::{load_sas, SasError};
pub use database::{
    DatabaseError, QueryResult,
    query_sqlite, list_sqlite_tables, sqlite_table_schema,
    query_duckdb, list_duckdb_tables, duckdb_table_schema,
    query_file_with_duckdb,
};
