//! Data loading and manipulation module.
//!
//! Provides functionality for loading datasets from various file formats
//! and comprehensive data manipulation operations.
//!
//! # Submodules
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
