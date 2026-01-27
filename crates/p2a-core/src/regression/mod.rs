//! Regression analysis module.
//!
//! This module provides linear and nonlinear regression methods with robust inference.
//!
//! ## Ordinary Least Squares
//!
//! - [`run_ols`] - OLS with heteroskedasticity-robust standard errors (HC0-HC3)
//! - [`run_ols_clustered`] - Clustered standard errors
//! - [`run_ols_raw`] - Low-level OLS from raw arrays
//!
//! ## Robust Standard Errors
//!
//! - [`vcov_hac`] - HAC (Newey-West) for time series
//! - [`vcov_bootstrap`] - Bootstrap covariance estimation
//! - [`vcov_driscoll_kraay`] - Panel-robust SEs (cross-sectional + serial)
//!
//! ## Generalized & Specialized Regression
//!
//! - [`run_gls`] - Generalized least squares (AR1, MA1, ARMA correlation)
//! - [`run_quantreg`] - Quantile/median regression
//! - [`run_loess`] - Local polynomial regression (LOESS/LOWESS)
//! - [`smooth_spline`] - Smoothing splines
//! - [`run_nls`] - Nonlinear least squares (Gauss-Newton/L-M)
//!
//! ## Model Selection
//!
//! - [`run_step`] - Stepwise selection (forward, backward, both)
//! - [`add1`], [`drop1`] - Single-term addition/deletion
//!
//! ## Diagnostics
//!
//! - [`run_diagnostics`] - Comprehensive diagnostics (JB, BP, DW, VIF)
//! - [`bg_test`] - Breusch-Godfrey serial correlation test
//! - [`reset_test`] - Ramsey's RESET for functional form
//! - [`wald_test`] - Linear hypothesis testing
//! - [`harvey_collier_test`] - Linearity test
//!
//! ## Marginal Effects & Sensitivity
//!
//! - [`marginal_effects`] - Average marginal effects (AME)
//! - [`contrasts`] - Effect contrasts
//! - [`run_sensemakr`] - Sensitivity analysis for unmeasured confounding
//! - [`evalue_rr`], [`evalue_or`] - E-values for causal inference
//!
//! ## Other
//!
//! - [`line`] - Tukey's resistant line
//! - [`supsmu`] - Friedman's SuperSmoother
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::{Dataset, run_ols};
//! use p2a_core::regression::CovarianceType;
//!
//! # fn example(dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
//! // OLS with HC1 (Stata default) robust standard errors
//! let result = run_ols(dataset, "price", &["sqft", "bedrooms"], true, CovarianceType::HC1)?;
//!
//! println!("R² = {:.4}", result.r_squared);
//! for coef in &result.coefficients {
//!     println!("{}: {:.4} (SE: {:.4})", coef.name, coef.estimate, coef.std_error);
//! }
//! # Ok(())
//! # }
//! ```

mod ols;
mod diagnostics;
mod nls;
mod loess;
mod gls;
mod smooth_spline;
mod step;
mod quantreg;
mod marginal_effects;
mod sensemakr;
mod evalue;
pub mod line;
pub mod supsmu;

pub use ols::{
    OlsResult, OlsCoefficient, OlsClusteredResult, CovarianceType,
    run_ols, run_ols_raw, run_ols_clustered,
    // HAC (Newey-West) standard errors
    HacResult, HacKernel, vcov_hac, run_vcov_hac,
    // Bootstrap covariance estimation
    BootstrapResult, BootstrapType, vcov_bootstrap, run_vcov_bootstrap,
    // Driscoll-Kraay panel-robust standard errors
    DriscollKraayResult, vcov_driscoll_kraay, run_vcov_driscoll_kraay,
};
pub use diagnostics::{
    DiagnosticsResult, TestResult, DurbinWatsonResult, VifResult, AutocorrelationType,
    run_diagnostics,
    // Breusch-Godfrey test for serial correlation
    BgTestResult, BgTestType, bg_test, run_bg_test, bg_test_from_ols,
    // Ramsey's RESET test for functional form
    ResetTestResult, ResetType, reset_test, run_reset_test, reset_test_from_ols,
    // Wald test for comparing nested models
    WaldTestResult, wald_test, run_wald_test, wald_test_from_ols,
    // Harvey-Collier test for linearity
    HarveyCollierResult, harvey_collier_test, run_harvey_collier, recursive_residuals,
};
pub use nls::{
    NlsResult, NlsConfig, NlsAlgorithm,
    nls, nls_multi, run_nls, run_nls_with_config,
    model_exponential_decay, model_exponential_growth,
    model_michaelis_menten, model_logistic_growth,
    model_power, model_asymptotic,
    ModelFn,
};
pub use loess::{
    LoessResult, LoessConfig, LoessModel,
    loess, loess_predict, run_loess,
};
pub use gls::{
    GlsResult, CorrelationStructure,
    gls, gls_ar1_auto, run_gls,
};
pub use smooth_spline::{
    SmoothSplineResult, SmoothSplineConfig,
    smooth_spline, smooth_spline_predict, run_smooth_spline,
};
pub use step::{
    StepResult, StepRecord, StepConfig, StepDirection,
    Add1Result, Drop1Result, TermEvaluation,
    step, run_step, add1, drop1,
};
pub use line::{
    LineResult, line, run_line,
};
pub use supsmu::{
    SupsmuResult, SupsmuConfig, supsmu, run_supsmu,
};
pub use quantreg::{
    QuantRegResult, QuantRegCoefficient, QuantRegConfig, QuantRegAlgorithm,
    quantreg, run_quantreg, quantreg_multi,
};
pub use marginal_effects::{
    MarginalEffectsResult, MarginalEffect, ModelType, ContrastsResult, ContrastEffect,
    marginal_effects, marginal_effects_ols, marginal_effects_discrete, contrasts,
};
pub use sensemakr::{
    SensemakrResult, SensitivityBound, ContourData,
    sensemakr, run_sensemakr, generate_contour_data,
    // Core sensitivity functions
    partial_r2, robustness_value, robustness_value_alpha,
    confounding_bias, adjusted_estimate, adjusted_se,
};
pub use evalue::{
    EValueResult, EffectType,
    // E-value functions for different effect measures
    evalue_rr, evalue_rr_ci, evalue_or, evalue_hr, evalue_smd, evalue_rd,
    // Bias factor functions
    bias_factor, bounding_factor, could_explain_away,
};
