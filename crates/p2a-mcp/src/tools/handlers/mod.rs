//! Tool handler implementations, organized by category.
//!
//! Each module contains the tool handler functions for a category of tools.
//! Handlers use the `#[tool_router]` macro for router composition.
//!
//! # Router Composition Pattern
//!
//! Each handler module defines a `*_router()` function via `#[tool_router(router = name)]`.
//! These are composed in server.rs using the `+` operator:
//!
//! ```ignore
//! let tool_router = Self::tool_router()  // main router
//!     + Self::utils_router()
//!     + Self::database_router()
//!     + Self::viz_router()
//!     + Self::data_router()
//!     + Self::ml_router()
//!     // ... other category routers
//!     ;
//! ```
//!
//! # Migration Status
//!
//! - `utils`: Migrated (seed management, reports, session export/import)
//! - `database`: Migrated (SQLite, DuckDB queries)
//! - `viz`: Migrated (visualization tools)
//! - `data`: Migrated (data loading, export, inspection, cleaning)
//! - `ml`: Migrated (clustering, dimensionality reduction, supervised learning, causal ML)
//! - `stats`: Migrated (loglin, model tables, contrasts, weighted stats, robust stats, splines)
//! - `regression`: Migrated (OLS, diagnostics, HAC, bootstrap, quantile, NLS, LOESS, GLS, etc.)
//! - `panel`: Migrated (FE, RE, Hausman, PVCM, PMG, GMM, panel GLS, unit root, HDFE)
//! - `discrete`: Migrated (logit, probit, multinomial, mlogit, mixed logit, ordered, count models, FEGLM)
//! - `causal`: Migrated (IV, DiD, treatment effects, matching, TMLE, synth, RD, mediation)
//! - `timeseries`: Migrated (ACF, VAR, ARIMA, GARCH, decomposition, filtering, etc.)
//! - `spatial`: Migrated (neighbors, Moran, SAR, SEM, spatial probit, panel)
//! - `munging`: Migrated (filter, select, join, concat, string ops, transformations)

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
pub mod spatial;
pub mod stats;
pub mod survival;
pub mod timeseries;
pub mod utils;
pub mod viz;
pub mod search;
