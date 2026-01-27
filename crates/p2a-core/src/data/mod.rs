//! Data loading and manipulation module.
//!
//! This module provides comprehensive data handling functionality including file loading,
//! data transformation, quality profiling, and LLM-assisted cleaning workflows.
//!
//! ## Data Loading
//!
//! | Format | Function | Description |
//! |--------|----------|-------------|
//! | **CSV/TSV** | [`DataLoader`] | Comma/tab-separated values |
//! | **Parquet** | [`DataLoader`] | Apache Parquet columnar format |
//! | **JSON** | [`DataLoader`] | JSON Lines format |
//! | **Excel** | [`DataLoader`] | XLSX files (feature: `excel`) |
//! | **Stata** | [`load_stata`] | Stata .dta files |
//! | **SAS** | [`load_sas`] | SAS .sas7bdat files |
//! | **SQLite** | [`query_sqlite`] | SQLite queries (feature: `database`) |
//! | **DuckDB** | [`query_duckdb`] | DuckDB queries (feature: `database`) |
//!
//! ## Data Manipulation ([`munging`])
//!
//! | Operation | Function | Description |
//! |-----------|----------|-------------|
//! | **Select** | `select_columns` | Choose columns |
//! | **Filter** | `filter_rows` | Row filtering with predicates |
//! | **Mutate** | `mutate`, `transform_column` | Create/transform columns |
//! | **Join** | `left_join`, `inner_join`, `full_join` | Merge datasets |
//! | **Reshape** | `pivot_wider`, `pivot_longer` | Wide/long transformations |
//! | **Aggregate** | `group_by`, `summarize` | Group-wise operations |
//! | **Sort** | `arrange`, `order_by` | Row ordering |
//! | **Distinct** | `distinct` | Remove duplicates |
//! | **Missing** | `drop_na`, `fill_na`, `replace_na` | Handle missing values |
//!
//! ## Data Quality ([`quality`])
//!
//! Generate comprehensive quality profiles for LLM-assisted cleaning:
//!
//! ```rust,no_run
//! use p2a_core::data::{generate_quality_profile, Dataset};
//!
//! # fn example(dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
//! let profile = generate_quality_profile(dataset, None)?;
//! println!("Issues found: {:?}", profile.issues);
//! # Ok(())
//! # }
//! ```
//!
//! ## Cleaning Sessions ([`cleaning_session`])
//!
//! Multi-step cleaning workflows with rollback support:
//!
//! ```rust,no_run
//! use p2a_core::data::{CleaningSession, Dataset};
//!
//! # fn example(dataset: Dataset) -> Result<(), Box<dyn std::error::Error>> {
//! let mut session = CleaningSession::new(dataset);
//! session.checkpoint("Initial state")?;
//! // ... apply cleaning operations ...
//! session.rollback()?;  // Undo last operation
//! # Ok(())
//! # }
//! ```
//!
//! ## Cleaning Suggestions ([`suggestion`])
//!
//! AI-powered suggestions with priority ranking:
//!
//! ```rust,no_run
//! use p2a_core::data::{generate_suggestions, Dataset};
//!
//! # fn example(dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
//! let suggestions = generate_suggestions(dataset)?;
//! for s in suggestions.suggestions {
//!     println!("{}: {}", s.priority, s.description);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Submodules
//!
//! - [`munging`] - Data transformation, cleaning, joining, reshaping, and aggregation
//! - [`quality`] - Data quality profiling for LLM-assisted cleaning
//! - [`verification`] - Verification and preview for cleaning operations
//! - [`cleaning_session`] - Session management for multi-step cleaning workflows
//! - [`suggestion`] - Smart cleaning suggestions with priority ranking

mod loader;
mod dataset;
mod stata;
mod sas;
#[cfg(feature = "database")]
mod database;
pub mod munging;
pub mod quality;
pub mod verification;
pub mod cleaning_session;
pub mod suggestion;

pub use loader::DataLoader;
pub use dataset::{Dataset, DatasetInfo};
pub use stata::{load_stata, StataError};
pub use sas::{load_sas, SasError};

#[cfg(feature = "database")]
pub use database::{
    DatabaseError, QueryResult,
    query_sqlite, list_sqlite_tables, sqlite_table_schema,
    query_duckdb, list_duckdb_tables, duckdb_table_schema,
    query_file_with_duckdb,
};
pub use quality::{
    DataQualityProfile, ColumnProfile, NumericStats, StringStats, DataIssue,
    generate_quality_profile,
};
pub use verification::{
    CleaningResult, VerificationReport, ChangeExample, QualityDelta,
    CleaningPreview, CleaningOperation, preview_cleaning, verify_cleaning,
};
pub use cleaning_session::{
    CleaningSession, SessionCheckpoint, OperationRecord, SessionStatus,
    CheckpointInfo, VerificationReportSummary,
};
pub use suggestion::{
    CleaningSuggestion, SuggestionPriority, CleaningCategory,
    SuggestionParameters, EstimatedImpact, SuggestionReport, DatasetSummary,
    generate_suggestions,
};

// Re-export munging operations at the data module level
pub use munging::*;
