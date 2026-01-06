//! Time series forecasting functionality.

mod arima_model;
mod mstl;

pub use arima_model::{ArimaResult, ArimaForecastResult, run_arima, forecast_arima};
pub use mstl::{MstlResult, run_mstl};
