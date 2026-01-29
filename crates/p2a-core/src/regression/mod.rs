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

mod diagnostics;
mod evalue;
mod gls;
pub mod line;
mod loess;
mod marginal_effects;
mod nls;
mod ols;
mod quantreg;
mod sensemakr;
mod smooth_spline;
mod step;
pub mod supsmu;

pub use diagnostics::{
    AutocorrelationType,
    // Breusch-Godfrey test for serial correlation
    BgTestResult,
    BgTestType,
    DiagnosticsResult,
    DurbinWatsonResult,
    // Harvey-Collier test for linearity
    HarveyCollierResult,
    // Ramsey's RESET test for functional form
    ResetTestResult,
    ResetType,
    TestResult,
    VifResult,
    // Wald test for comparing nested models
    WaldTestResult,
    bg_test,
    bg_test_from_ols,
    harvey_collier_test,
    recursive_residuals,
    reset_test,
    reset_test_from_ols,
    run_bg_test,
    run_diagnostics,
    run_harvey_collier,
    run_reset_test,
    run_wald_test,
    wald_test,
    wald_test_from_ols,
};
pub use evalue::{
    EValueResult,
    EffectType,
    // Bias factor functions
    bias_factor,
    bounding_factor,
    could_explain_away,
    evalue_hr,
    evalue_or,
    evalue_rd,
    // E-value functions for different effect measures
    evalue_rr,
    evalue_rr_ci,
    evalue_smd,
};
pub use gls::{CorrelationStructure, GlsResult, gls, gls_ar1_auto, run_gls};
pub use line::{LineResult, line, run_line};
pub use loess::{LoessConfig, LoessModel, LoessResult, loess, loess_predict, run_loess};
pub use marginal_effects::{
    ContrastEffect, ContrastsResult, MarginalEffect, MarginalEffectsResult, ModelType, contrasts,
    marginal_effects, marginal_effects_discrete, marginal_effects_ols,
};
pub use nls::{
    ModelFn, NlsAlgorithm, NlsConfig, NlsResult, model_asymptotic, model_exponential_decay,
    model_exponential_growth, model_logistic_growth, model_michaelis_menten, model_power, nls,
    nls_multi, run_nls, run_nls_with_config,
};
pub use ols::{
    // Bootstrap covariance estimation
    BootstrapResult,
    BootstrapType,
    CovarianceType,
    // Driscoll-Kraay panel-robust standard errors
    DriscollKraayResult,
    HacKernel,
    // HAC (Newey-West) standard errors
    HacResult,
    OlsClusteredResult,
    OlsCoefficient,
    OlsResult,
    run_ols,
    run_ols_clustered,
    run_ols_raw,
    run_vcov_bootstrap,
    run_vcov_driscoll_kraay,
    run_vcov_hac,
    vcov_bootstrap,
    vcov_driscoll_kraay,
    vcov_hac,
};
pub use quantreg::{
    QuantRegAlgorithm, QuantRegCoefficient, QuantRegConfig, QuantRegResult, quantreg,
    quantreg_multi, run_quantreg,
};
pub use sensemakr::{
    ContourData,
    SensemakrResult,
    SensitivityBound,
    adjusted_estimate,
    adjusted_se,
    confounding_bias,
    generate_contour_data,
    // Core sensitivity functions
    partial_r2,
    robustness_value,
    robustness_value_alpha,
    run_sensemakr,
    sensemakr,
};
pub use smooth_spline::{
    SmoothSplineConfig, SmoothSplineResult, run_smooth_spline, smooth_spline, smooth_spline_predict,
};
pub use step::{
    Add1Result, Drop1Result, StepConfig, StepDirection, StepRecord, StepResult, TermEvaluation,
    add1, drop1, run_step, step,
};
pub use supsmu::{SupsmuConfig, SupsmuResult, run_supsmu, supsmu};
