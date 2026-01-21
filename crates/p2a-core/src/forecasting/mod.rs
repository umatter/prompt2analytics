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
