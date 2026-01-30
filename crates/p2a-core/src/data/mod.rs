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
//! ```rust,ignore
//! use p2a_core::data::{generate_quality_profile, Dataset};
//!
//! let profile = generate_quality_profile(&dataset, None)?;
//! println!("Issues found: {:?}", profile.issues);
//! ```
//!
//! ## Cleaning Sessions ([`cleaning_session`])
//!
//! Multi-step cleaning workflows with rollback support:
//!
//! ```rust,ignore
//! use p2a_core::data::{CleaningSession, Dataset};
//!
//! let mut session = CleaningSession::new(dataset);
//! session.checkpoint("Initial state")?;
//! // ... apply cleaning operations ...
//! session.rollback()?;  // Undo last operation
//! ```
//!
//! ## Cleaning Suggestions ([`suggestion`])
//!
//! AI-powered suggestions with priority ranking:
//!
//! ```rust,ignore
//! use p2a_core::data::{generate_suggestions, Dataset};
//!
//! let suggestions = generate_suggestions(&dataset)?;
//! for s in suggestions.suggestions {
//!     println!("{}: {}", s.priority, s.description);
//! }
//! ```
//!
//! ## Submodules
//!
//! - [`munging`] - Data transformation, cleaning, joining, reshaping, and aggregation
//! - [`quality`] - Data quality profiling for LLM-assisted cleaning
//! - [`verification`] - Verification and preview for cleaning operations
//! - [`cleaning_session`] - Session management for multi-step cleaning workflows
//! - [`suggestion`] - Smart cleaning suggestions with priority ranking

pub mod cleaning_session;
#[cfg(feature = "database")]
mod database;
mod dataset;
mod loader;
pub mod munging;
pub mod quality;
mod sas;
mod stata;
pub mod suggestion;
pub mod verification;

pub use dataset::{Dataset, DatasetInfo};
pub use loader::DataLoader;
pub use sas::{SasError, load_sas};
pub use stata::{StataError, load_stata};

pub use cleaning_session::{
    CheckpointInfo, CleaningSession, OperationRecord, SessionCheckpoint, SessionStatus,
    VerificationReportSummary,
};
#[cfg(feature = "database")]
pub use database::{
    DatabaseError, QueryResult, duckdb_table_schema, list_duckdb_tables, list_sqlite_tables,
    query_duckdb, query_file_with_duckdb, query_sqlite, sqlite_table_schema,
};
pub use quality::{
    ColumnProfile, DataIssue, DataQualityProfile, NumericStats, StringStats,
    generate_quality_profile,
};
pub use suggestion::{
    CleaningCategory, CleaningSuggestion, DatasetSummary, EstimatedImpact, SuggestionParameters,
    SuggestionPriority, SuggestionReport, generate_suggestions,
};
pub use verification::{
    ChangeExample, CleaningOperation, CleaningPreview, CleaningResult, QualityDelta,
    VerificationReport, preview_cleaning, verify_cleaning,
};

// Re-export munging operations at the data module level
pub use munging::*;
