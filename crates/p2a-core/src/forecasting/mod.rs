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
//! ```rust,no_run
//! use p2a_core::forecasting::{run_arima, forecast_arima, run_holt_winters};
//! use p2a_core::Dataset;
//!
//! # fn example(dataset: &Dataset) -> Result<(), Box<dyn std::error::Error>> {
//! // Fit ARIMA(1,1,1)
//! let arima = run_arima(dataset, "sales", Some(1), Some(1), Some(1))?;
//! println!("AIC: {:.2}", arima.aic);
//!
//! // Forecast 12 periods ahead
//! let forecast = forecast_arima(&arima, 12)?;
//! println!("Forecast: {:?}", forecast.point_forecast);
//!
//! // Holt-Winters seasonal model
//! let hw = run_holt_winters(dataset, "sales", 12, "additive", None, None, None)?;
//! println!("Fitted values: {:?}", hw.fitted);
//! # Ok(())
//! # }
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

mod arima_model;
mod mstl;
mod changepoint;
mod holtwinters;
pub mod ar;
pub mod decompose;
pub mod kalman;
pub mod structts;
pub mod stl;
pub mod tsutils;
pub mod cpgram;
pub mod garch;
pub mod causal_impact;

pub use arima_model::{ArimaResult, ArimaForecastResult, run_arima, forecast_arima};
pub use mstl::{MstlResult, run_mstl};
pub use changepoint::{
    ChangepointResult, SegmentStats, CostFunction,
    detect_changepoints, binary_segmentation,
    run_changepoint, run_binary_segmentation,
};
pub use holtwinters::{
    HoltWintersResult, HoltWintersConfig, HoltWintersCoefficients, SeasonalType,
    holt_winters, holt_winters_forecast, run_holt_winters,
};
pub use ar::{
    ArResult, ArConfig, ArMethod, ar, run_ar, run_ar_with_order,
};
pub use decompose::{
    DecomposeResult, DecomposeConfig, DecomposeType,
    decompose, run_decompose, run_decompose_with_filter,
};
pub use kalman::{
    StateSpaceModel, KalmanFilterResult, KalmanSmootherResult, KalmanForecastResult,
    kalman_filter, kalman_smoother, kalman_forecast, kalman_loglik,
};
pub use structts::{
    StructTsType, StructTsConfig, StructTsResult, StructTsCoefficients,
    struct_ts, run_struct_ts,
};
pub use stl::{
    StlResult, StlConfig, stl, run_stl, run_stl_with_config,
};
pub use tsutils::{
    // Lag function
    LagResult, lag, lag_padded,
    // Embed function
    EmbedResult, embed, embed_array,
    // Diffinv function
    DiffinvResult, diffinv,
    // Filter function
    FilterMethod, FilterSides, FilterResult, filter,
    // Window function
    WindowResult, window,
    // ARMA ACF
    ArmaAcfResult, arma_acf,
    // ARMA to MA
    ArmaToMaResult, arma_to_ma,
    // ACF to AR
    Acf2ArResult, acf_to_ar,
    // ARIMA simulation
    ArimaSimResult, arima_sim,
    // Running median
    EndRule, RunmedResult, runmed,
};
pub use cpgram::{
    CpgramResult, cpgram, run_cpgram, white_noise_test,
};
pub use garch::{
    GarchConfig, GarchResult, garch, garch_forecast, run_garch,
};
pub use causal_impact::{
    CausalImpactConfig, CausalImpactSummary, CausalImpactSeries,
    CausalImpactModel, CausalInference, CausalImpactResult,
    causal_impact, run_causal_impact,
};
