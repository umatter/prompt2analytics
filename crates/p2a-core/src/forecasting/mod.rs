//! Time series forecasting functionality.
//!
//! This module provides 20+ forecasting and time series analysis methods including
//! ARIMA modeling, exponential smoothing, state-space models, and changepoint detection.
//!
//! ## ARIMA & Autoregressive Models
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **ARIMA** | [`run_arima`], [`run_arima_with`], [`forecast_arima`], [`forecast_arima_with`], [`ArimaSpec`] | ARIMA(p,d,q) by conditional sum of squares (mean and drift as R), forecasts with 80/95% intervals |
//! | **AR** | [`ar()`], [`run_ar`] | Autoregressive models (Yule-Walker, OLS, Burg) |
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
//! | **STL** | [`stl()`], [`run_stl`] | Seasonal-trend decomposition (LOESS) |
//! | **MSTL** | [`run_mstl`] | Multiple seasonal decomposition |
//! | **Classical** | [`decompose()`], [`run_decompose`] | Additive/multiplicative decomposition |
//!
//! ## Volatility Models
//!
//! | Method | Function | Description |
//! |--------|----------|-------------|
//! | **GARCH** | [`garch()`], [`run_garch`] | Generalized autoregressive conditional heteroskedasticity |
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
//! ```ignore
//! use p2a_core::forecasting::{
//!     ArimaSpec, forecast_arima, forecast_arima_with, run_arima, run_arima_with,
//!     run_holt_winters, SeasonalType,
//! };
//! use p2a_core::data::Dataset;
//!
//! // Assuming dataset is loaded with a "sales" time series column
//!
//! // Fit ARIMA(1,1,1) to sales column (R's defaults: no mean or drift for d = 1)
//! let arima = run_arima(&dataset, "sales", 1, 1, 1)?;
//! println!("AIC (CSS likelihood): {:.2}", arima.aic);
//!
//! // The same order with a drift, as forecast::Arima(include.drift = TRUE)
//! let spec = ArimaSpec { include_drift: true, ..ArimaSpec::new(1, 1, 1) };
//! let with_drift = run_arima_with(&dataset, "sales", &spec)?;
//! println!("drift: {:?}", with_drift.drift);
//!
//! // Forecast 12 periods ahead, with standard errors and 80/95% limits
//! let forecast = forecast_arima(&dataset, "sales", 1, 1, 1, 12)?;
//! println!("Forecast: {:?}", forecast.forecast);
//! let forecast = forecast_arima_with(&dataset, "sales", &spec, 12)?;
//! println!("95% lower limits: {:?}", forecast.lower[1]);
//!
//! // Holt-Winters seasonal model (period=12 for monthly data)
//! let hw = run_holt_winters(&dataset, "sales", 12, SeasonalType::Additive, None, None, None)?;
//! println!("Fitted values: {:?}", hw.fitted);
//! ```
//!
//! ## R Package Equivalents
//!
//! | R Function | p2a-core Function |
//! |------------|-------------------|
//! | `arima(method = "CSS")`, `forecast::Arima()` | [`run_arima`], [`run_arima_with`] |
//! | `forecast::forecast()` of an `Arima` fit | [`forecast_arima`], [`forecast_arima_with`] |
//! | `ar()` | [`ar()`], [`run_ar`] |
//! | `HoltWinters()` | [`holt_winters`], [`run_holt_winters`] |
//! | `stl()` | [`stl()`], [`run_stl`] |
//! | `decompose()` | [`decompose()`], [`run_decompose`] |
//! | `StructTS()` | [`struct_ts`], [`run_struct_ts`] |
//! | `KalmanFilter()` | [`kalman_filter`] |
//! | `CausalImpact` | [`causal_impact()`], [`run_causal_impact`] |

pub mod ar;
pub mod arima_model;
pub mod causal_impact;
pub mod changepoint;
pub mod cpgram;
pub mod decompose;
pub mod garch;
pub mod holtwinters;
pub mod kalman;
pub mod mstl;
pub mod stl;
pub mod structts;
pub mod tsutils;

pub use ar::{ArConfig, ArMethod, ArResult, ar, run_ar, run_ar_with_order};
pub use arima_model::{
    ArimaForecastResult, ArimaResult, ArimaSpec, forecast_arima, forecast_arima_with, run_arima,
    run_arima_with,
};
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
pub use garch::{GarchConfig, GarchDistribution, GarchResult, garch, garch_forecast, run_garch};
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

#[cfg(test)]
pub(crate) mod test_fixtures {
    //! Case data and assertions shared by the time-series tests (T2-WP16). Tests that
    //! compare no case value read the data through `arima_series()`, so they are not
    //! registered R pins.

    use polars::prelude::*;

    use crate::data::Dataset;
    use crate::errors::EconError;

    /// validation/datasets/generated/arima_series.csv (case arima_css_forecast): a, c, m,
    /// b, dr, a_na (Float64), t, ai, ai_na (Int64), lab (String); a_na / ai_na are null at
    /// rows 149 and 150.
    pub(crate) fn arima_series() -> Dataset {
        crate::test_support::load_case("arima_css_forecast").1
    }

    /// `ds` plus "<column>_f64" = `column` cast to Float64 (the same values).
    pub(crate) fn with_float_copy(ds: &Dataset, column: &str) -> Dataset {
        let copy = ds
            .df()
            .column(column)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .with_name(format!("{column}_f64").into());
        let mut df = ds.df().clone();
        df.with_column(copy).unwrap();
        Dataset::new(df)
    }

    /// `ds` with `column`'s value at `row` replaced by NaN (Float64 columns only).
    pub(crate) fn with_nan_at(ds: &Dataset, column: &str, row: usize) -> Dataset {
        let values: Vec<Option<f64>> = ds
            .df()
            .column(column)
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| if i == row { Some(f64::NAN) } else { v })
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Float64Chunked::from_slice_options(column.into(), &values).into_series())
            .unwrap();
        Dataset::new(df)
    }

    /// Equal length and bitwise-equal values (NaN equals NaN).
    pub(crate) fn assert_same(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "lengths differ: {a:?} vs {b:?}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "element {i} differs: {x:?} vs {y:?}"
            );
        }
    }

    /// The EconError inside an anyhow::Error (panics with the message otherwise).
    pub(crate) fn econ(err: &anyhow::Error) -> &EconError {
        err.downcast_ref::<EconError>()
            .unwrap_or_else(|| panic!("not an EconError: {err:#}"))
    }

    pub(crate) fn assert_null_values(err: &EconError, column: &str, count: usize) {
        assert!(
            matches!(err, EconError::NullValues { column: c, count: k } if c == column && *k == count),
            "expected NullValues {{ {column}, {count} }}, got {err:?}"
        );
    }

    pub(crate) fn assert_non_numeric(err: &EconError, column: &str) {
        assert!(
            matches!(err, EconError::NonNumericColumn { column: c } if c == column),
            "expected NonNumericColumn {{ {column} }}, got {err:?}"
        );
    }

    pub(crate) fn assert_not_found(err: &EconError, column: &str) {
        assert!(
            matches!(err, EconError::ColumnNotFound { column: c, .. } if c == column),
            "expected ColumnNotFound {{ {column} }}, got {err:?}"
        );
    }
}
