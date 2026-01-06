//! Regression analysis module.
//!
//! Provides OLS regression, diagnostics, and related analyses.

mod ols;
mod diagnostics;

pub use ols::{OlsResult, OlsCoefficient, OlsClusteredResult, run_ols, run_ols_clustered};
pub use diagnostics::{DiagnosticsResult, run_diagnostics};
