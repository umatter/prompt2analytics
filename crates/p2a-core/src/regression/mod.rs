//! Regression analysis module.
//!
//! Provides OLS regression and related analyses.

mod ols;

pub use ols::{OlsResult, OlsCoefficient, run_ols};
