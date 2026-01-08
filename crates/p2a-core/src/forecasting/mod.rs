//! Time series forecasting functionality.

mod arima_model;
mod mstl;
mod changepoint;

pub use arima_model::{ArimaResult, ArimaForecastResult, run_arima, forecast_arima};
pub use mstl::{MstlResult, run_mstl};
pub use changepoint::{
    ChangepointResult, SegmentStats, CostFunction,
    detect_changepoints, binary_segmentation,
    run_changepoint, run_binary_segmentation,
};
