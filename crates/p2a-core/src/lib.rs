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
pub mod forecasting;
pub mod ml;

pub use data::{Dataset, DataLoader, DatasetInfo};
pub use stats::{DescriptiveStats, CorrelationMatrix, correlation_matrix};
pub use regression::{OlsResult, run_ols, run_ols_clustered, DiagnosticsResult, run_diagnostics};
pub use econometrics::{
    PanelResult, HausmanResult, run_fixed_effects, run_random_effects, run_hausman_test,
    IVResult, run_iv2sls, FirstStageDiagnostics, run_first_stage_diagnostics,
    DiDResult, run_did,
    DiscreteResult, run_logit, run_probit,
    VarResult, VarmaResult, VecmResult, VarIrfResult, run_var, run_varma, run_vecm, run_var_irf,
};
pub use forecasting::{
    ArimaResult, ArimaForecastResult, run_arima, forecast_arima,
    MstlResult, run_mstl,
};

/// Re-export polars for downstream use
pub use polars;
