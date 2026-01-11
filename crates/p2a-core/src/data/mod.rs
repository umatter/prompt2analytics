//! Data loading and manipulation module.
//!
//! Provides functionality for loading datasets from various file formats
//! and comprehensive data manipulation operations.
//!
//! # Submodules
//!
//! - [`munging`] - Data transformation, cleaning, joining, reshaping, and aggregation

mod loader;
mod dataset;
mod stata;
mod sas;
mod database;
pub mod munging;

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

// Re-export munging operations at the data module level
pub use munging::*;
