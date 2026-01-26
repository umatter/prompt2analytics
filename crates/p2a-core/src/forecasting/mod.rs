//! Time series forecasting functionality.

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
