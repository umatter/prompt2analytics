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
//! The SQL functions run exactly one read-only statement on a connection the
//! engine itself locks (no file access beyond the named database or file, no
//! extensions, no reconfiguration); the `*_with` variants also take
//! `QueryOptions` bounds on rows, bytes, time, DuckDB memory and threads.
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
//! ```
//! use p2a_core::data::{generate_quality_profile, DataIssue, Dataset};
//! use polars::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let df = df! {
//!     "x" => [Some(1.0), Some(2.0), None, Some(4.0)],
//!     "y" => [Some("a"), Some("b"), None, Some("d")]
//! }?;
//! let dataset = Dataset::new(df);
//! let profile = generate_quality_profile(&dataset);
//!
//! // 2 of the 8 cells are missing, so completeness is 6/8.
//! assert_eq!(profile.row_count, 4);
//! assert_eq!(profile.duplicate_rows, 0);
//! assert!((profile.completeness_score - 0.75).abs() < 1e-12);
//!
//! // Detected problems hang off the column that raised them; only
//! // whole-dataset findings (e.g. duplicate rows) go in `dataset_issues`.
//! let x = profile.columns.iter().find(|c| c.name == "x").unwrap();
//! assert_eq!(x.null_count, 1);
//! assert!(x.issues.iter().any(|i| matches!(i, DataIssue::HighNullRate { .. })));
//! assert!(profile.dataset_issues.is_empty());
//!
//! // `all_issues` flattens both levels, most severe first.
//! for issue in profile.all_issues() {
//!     println!("[severity {}] {}", issue.severity(), issue.description());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Cleaning Sessions ([`cleaning_session`])
//!
//! Multi-step cleaning workflows with rollback support:
//!
//! ```
//! use p2a_core::data::{CleaningSession, Dataset};
//! use polars::prelude::*;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let df = df! { "x" => [Some(1), Some(2), None, Some(4)] }?;
//! let dataset = Dataset::new(df);
//! // Checkpoint 0 is the untouched input; each applied operation appends one.
//! let mut session = CleaningSession::new(dataset, "mydata");
//! assert_eq!(session.current_checkpoint, 0);
//!
//! // Cleaning operations are closures over the current dataset. The session
//! // snapshots the result and hands back a verification report.
//! let report = session.apply_operation(
//!     "filter",
//!     "Drop rows with missing x",
//!     HashMap::new(),
//!     |ds| {
//!         let filtered = ds
//!             .df()
//!             .clone()
//!             .lazy()
//!             .filter(col("x").is_not_null())
//!             .collect()
//!             .map_err(|e: PolarsError| e.to_string())?;
//!         Ok(Dataset::new(filtered))
//!     },
//! )?;
//! assert_eq!(report.rows_before, 4);
//! assert_eq!(report.rows_after, 3);
//! assert_eq!(report.quality_delta.completeness_after, 1.0);
//! assert_eq!(session.current_checkpoint, 1);
//!
//! session.rollback()?; // Undo last operation
//! assert_eq!(session.current_checkpoint, 0);
//! assert_eq!(session.current_dataset().nrows(), 4);
//! // Rolling back moves the cursor; the checkpoint history is kept.
//! assert_eq!(session.list_checkpoints().len(), 2);
//! # Ok(())
//! # }
//! ```
//!
//! ## Cleaning Suggestions ([`suggestion`])
//!
//! AI-powered suggestions with priority ranking:
//!
//! ```
//! use p2a_core::data::{generate_quality_profile, generate_suggestions, Dataset};
//! use polars::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // NB: use a null, not f64::NAN, to represent a missing number. polars counts
//! // NaN as a present value, so a NaN never raises a missing-data issue here.
//! let df = df! {
//!     "x" => [Some(1.0), Some(2.0), None, Some(4.0)],
//!     "y" => [Some("a"), Some("b"), None, Some("d")]
//! }?;
//! let dataset = Dataset::new(df);
//! let profile = generate_quality_profile(&dataset);
//! let suggestions = generate_suggestions(&profile);
//! assert!(!suggestions.suggestions.is_empty()); // both columns are 25% null
//! for s in suggestions.suggestions {
//!     println!("{:?}: {}", s.priority, s.description);
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
pub use loader::{CsvChunkIterator, DataLoader, FileInfo, LoadError};
pub use sas::{SasError, load_sas};
pub use stata::{StataError, load_stata};

pub use cleaning_session::{
    CheckpointInfo, CleaningSession, OperationRecord, SessionCheckpoint, SessionStatus,
    VerificationReportSummary,
};
#[cfg(feature = "database")]
pub use database::{
    DatabaseError, QueryOptions, QueryOutcome, QueryResult, duckdb_table_schema,
    ensure_single_statement, list_duckdb_tables, list_sqlite_tables, query_duckdb,
    query_duckdb_with, query_file_with_duckdb, query_file_with_duckdb_with, query_sqlite,
    query_sqlite_with, sqlite_table_schema,
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
