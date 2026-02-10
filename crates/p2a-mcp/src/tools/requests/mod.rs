//! Request types for MCP tools, organized by category.
//!
//! Each module contains the request structs for a category of tools.
//! Request structs define the input parameters for each MCP tool.
//!
//! # Migration Status
//!
//! Request types are being migrated incrementally from server.rs.
//! Modules marked with "(migrated)" have their types defined here.
//! Other modules are placeholders awaiting migration.
//!
//! - `data`: (migrated) Data management requests
//! - `utils`: (migrated) Utility requests (seed, reports)
//! - `database`: (migrated) Database requests (SQLite, DuckDB)
//! - `viz`: (migrated) Visualization requests
//! - `ml`: (migrated) Machine learning requests
//! - `stats`: (migrated) Statistics requests (loglin, contrasts, weighted stats, robust stats, splines)
//! - `regression`: (migrated) Regression requests (OLS, diagnostics, HAC, bootstrap, quantile, NLS, LOESS, GLS)
//! - `panel`: (migrated) Panel data requests (FE, RE, Hausman, PVCM, PMG, GMM, panel GLS, unit root, HDFE)
//! - `discrete`: (migrated) Discrete choice requests (logit, probit, multinomial, mlogit, mixed logit, ordered, count models, FEGLM)
//! - `causal`: (migrated) Causal inference requests (IV, DiD, treatment effects, matching, TMLE, synth, RD, mediation)
//! - `timeseries`: (migrated) Time series requests (ACF, VAR, ARIMA, GARCH, decomposition, filtering)
//! - `spatial`: (migrated) Spatial econometrics requests (neighbors, Moran, SAR, SEM, spatial probit, panel)
//! - `munging`: (migrated) Data munging requests (filter, select, join, concat, string ops, transformations)

// Migrated modules
pub mod causal;
pub mod cleaning;
pub mod data;
pub mod database;
pub mod discrete;
pub mod hypothesis;
pub mod ml;
pub mod munging;
pub mod panel;
pub mod regression;
pub mod search;
pub mod spatial;
pub mod stats;
pub mod survival;
pub mod timeseries;
pub mod utils;
pub mod viz;

// Re-export migrated types
pub use causal::*;
pub use cleaning::*;
pub use data::*;
pub use database::*;
pub use discrete::*;
pub use hypothesis::*;
pub use ml::*;
pub use munging::*;
pub use panel::*;
pub use regression::*;
pub use search::*;
pub use spatial::*;
pub use stats::*;
pub use survival::*;
pub use timeseries::*;
pub use utils::*;
pub use viz::*;
