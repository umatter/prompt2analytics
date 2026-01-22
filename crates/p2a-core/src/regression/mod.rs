//! Regression analysis module.
//!
//! Provides OLS regression, nonlinear least squares, local polynomial regression (LOESS),
//! diagnostics, and related analyses.

mod ols;
mod diagnostics;
mod nls;
mod loess;
mod gls;
mod smooth_spline;

pub use ols::{
    OlsResult, OlsCoefficient, OlsClusteredResult, CovarianceType,
    run_ols, run_ols_raw, run_ols_clustered,
};
pub use diagnostics::{
    DiagnosticsResult, TestResult, DurbinWatsonResult, VifResult, AutocorrelationType,
    run_diagnostics,
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
