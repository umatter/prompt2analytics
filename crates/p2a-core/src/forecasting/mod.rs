//! Time series forecasting functionality.
//!
//! This module provides 20+ forecasting and time series analysis methods including
//! ARIMA modeling, exponential smoothing, state-space models, and changepoint detection.
//!
//! ## ARIMA & Autoregressive Models
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **ARIMA** | [`run_arima`] | Auto-regressive integrated moving average |
//! | **AR** | [`ar`], [`run_ar`] | Autoregressive models (Yule-Walker, OLS, Burg) |
//! | **ARIMA Simulation** | [`arima_sim`] | Simulate ARIMA processes |
//! | **ARMA ACF** | [`arma_acf`] | Theoretical ACF of ARMA models |
//! | **ARMA to MA** | [`arma_to_ma`] | Convert ARMA to MA representation |
//! | **ACF to AR** | [`acf_to_ar`] | Convert ACF to AR coefficients |
//!
//! ## Exponential Smoothing
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Holt-Winters** | [`holt_winters`], [`run_holt_winters`] | Additive/multiplicative seasonality |
//! | **Structural TS** | [`struct_ts`], [`run_struct_ts`] | Local level, trend, seasonal models |
//!
//! ## Decomposition
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **STL** | [`stl`], [`run_stl`] | Seasonal-trend decomposition (LOESS) |
//! | **MSTL** | [`run_mstl`] | Multiple seasonal decomposition |
//! | **Classical** | [`decompose`], [`run_decompose`] | Additive/multiplicative decomposition |
//!
//! ## Volatility Models
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **GARCH** | [`garch`], [`run_garch`] | Generalized autoregressive conditional heteroskedasticity |
//!
//! ## State-Space Models
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Kalman Filter** | [`kalman_filter`] | Optimal state estimation |
//! | **Kalman Smoother** | [`kalman_smoother`] | Fixed-interval smoothing |
//! | **Kalman Forecast** | [`kalman_forecast`] | State-space forecasting |
//!
//! ## Changepoint Detection
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **PELT** | [`detect_changepoints`] | Optimal changepoint detection |
//! | **Binary Segmentation** | [`binary_segmentation`] | Recursive partitioning |
//!
//! ## Causal Analysis
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Causal Impact** | [`run_causal_impact`] | Bayesian time series causal inference |
//!
//! ## Utilities
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **Lag** | [`lag`], [`lag_padded`] | Create lagged series |
//! | **Embed** | [`embed`] | Time-delay embedding |
//! | **Diffinv** | [`diffinv`] | Inverse differencing |
//! | **Filter** | [`filter`] | Convolution/recursive filtering |
//! | **Window** | [`window`] | Extract time series windows |
//! | **Running Median** | [`runmed`] | Smoothing via running medians |
//! | **Cumulative Periodogram** | [`cpgram()`] | Test for white noise |
//!
//! ## Example
//!
//! ```rust,ignore
//! use p2a_core::forecasting::{run_arima, forecast_arima, run_holt_winters};
//! use p2a_core::Dataset;
//!
//! // Fit ARIMA(1,1,1)
//! let arima = run_arima(&dataset, "sales", Some(1), Some(1), Some(1))?;
//! println!("AIC: {:.2}", arima.aic);
//!
//! // Forecast 12 periods ahead
//! let forecast = forecast_arima(&arima, 12)?;
//! println!("Forecast: {:?}", forecast.point_forecast);
//!
//! // Holt-Winters seasonal model
//! let hw = run_holt_winters(&dataset, "sales", 12, "additive", None, None, None)?;
//! println!("Fitted values: {:?}", hw.fitted);
//! ```
//!
//! ## R Package Equivalents
//!
//! | R Function | p2a-core Function |
//! |------------|-------------------|
//! | `arima()` | [`run_arima`] |
//! | `ar()` | [`ar`], [`run_ar`] |
//! | `HoltWinters()` | [`holt_winters`], [`run_holt_winters`] |
//! | `stl()` | [`stl`], [`run_stl`] |
//! | `decompose()` | [`decompose`], [`run_decompose`] |
//! | `StructTS()` | [`struct_ts`], [`run_struct_ts`] |
//! | `KalmanFilter()` | [`kalman_filter`] |
//! | `CausalImpact` | [`causal_impact()`], [`run_causal_impact`] |

pub mod ar;
mod arima_model;
pub mod causal_impact;
mod changepoint;
pub mod cpgram;
pub mod decompose;
pub mod garch;
mod holtwinters;
pub mod kalman;
mod mstl;
pub mod stl;
pub mod structts;
pub mod tsutils;

pub use ar::{ArConfig, ArMethod, ArResult, ar, run_ar, run_ar_with_order};
pub use arima_model::{ArimaForecastResult, ArimaResult, forecast_arima, run_arima};
pub use causal_impact::{
    CausalImpactConfig, CausalImpactModel, CausalImpactResult, CausalImpactSeries,
    CausalImpactSummary, CausalInference, causal_impact, run_causal_impact,
};
pub use changepoint::{
    ChangepointResult, CostFunction, SegmentStats, binary_segmentation, detect_changepoints,
    run_binary_segmentation, run_changepoint,
};
pub use cpgram::{CpgramResult, cpgram, run_cpgram, white_noise_test};
pub use decompose::{
    DecomposeConfig, DecomposeResult, DecomposeType, decompose, run_decompose,
    run_decompose_with_filter,
};
pub use garch::{GarchConfig, GarchResult, garch, garch_forecast, run_garch};
pub use holtwinters::{
    HoltWintersCoefficients, HoltWintersConfig, HoltWintersResult, SeasonalType, holt_winters,
    holt_winters_forecast, run_holt_winters,
};
pub use kalman::{
    KalmanFilterResult, KalmanForecastResult, KalmanSmootherResult, StateSpaceModel, kalman_filter,
    kalman_forecast, kalman_loglik, kalman_smoother,
};
pub use mstl::{MstlResult, run_mstl};
pub use stl::{StlConfig, StlResult, run_stl, run_stl_with_config, stl};
pub use structts::{
    StructTsCoefficients, StructTsConfig, StructTsResult, StructTsType, run_struct_ts, struct_ts,
};
pub use tsutils::{
    // ACF to AR
    Acf2ArResult,
    // ARIMA simulation
    ArimaSimResult,
    // ARMA ACF
    ArmaAcfResult,
    // ARMA to MA
    ArmaToMaResult,
    // Diffinv function
    DiffinvResult,
    // Embed function
    EmbedResult,
    // Running median
    EndRule,
    // Filter function
    FilterMethod,
    FilterResult,
    FilterSides,
    // Lag function
    LagResult,
    RunmedResult,
    // Window function
    WindowResult,
    acf_to_ar,
    arima_sim,
    arma_acf,
    arma_to_ma,
    diffinv,
    embed,
    embed_array,
    filter,
    lag,
    lag_padded,
    runmed,
    window,
};
