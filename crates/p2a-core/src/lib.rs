//! # p2a-core
//!
//! Core analytics engine for prompt2analytics.
//!
//! This crate provides the data loading, statistical analysis, and machine learning
//! functionality that powers the MCP server.

pub mod data;
pub mod stats;
pub mod regression;
pub mod econometrics;
pub mod ml;

pub use data::{Dataset, DataLoader, DatasetInfo};
pub use stats::{DescriptiveStats, CorrelationMatrix, correlation_matrix};
pub use regression::{OlsResult, run_ols};
pub use econometrics::{
    PanelResult, run_fixed_effects, run_random_effects,
    IVResult, run_iv2sls,
    DiDResult, run_did,
};

/// Re-export polars for downstream use
pub use polars;
