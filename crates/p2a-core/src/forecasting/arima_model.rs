//! ARIMA (AutoRegressive Integrated Moving Average) modeling.
//!
//! Provides ARIMA(p,d,q) model fitting and forecasting for univariate time series.
//!
//! # Mathematical Background
//!
//! An ARIMA(p,d,q) model combines:
//! - **AR(p)**: p autoregressive terms: φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ
//! - **I(d)**: d differences to achieve stationarity
//! - **MA(q)**: q moving average terms: θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q
//!
//! The general form (after differencing) is:
//!
//! φ(B)(1-B)ᵈ yₜ = θ(B) εₜ
//!
//! where B is the backshift operator, φ(B) is the AR polynomial,
//! and θ(B) is the MA polynomial.
//!
//! ## Stationarity and Invertibility
//!
//! For a valid ARIMA model:
//! - AR polynomial roots must lie outside the unit circle (stationarity)
//! - MA polynomial roots must lie outside the unit circle (invertibility)
//!
//! # References
//!
//! - Box, G.E.P., & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and
//!   Control*. Holden-Day. The foundational work on ARIMA modeling.
//!
//! - Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series
//!   Analysis: Forecasting and Control* (5th ed.). Wiley. ISBN: 978-1118675021.
//!
//! - Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
//!   Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
//!
//! - Brockwell, P.J., & Davis, R.A. (1991). *Time Series: Theory and Methods*
//!   (2nd ed.). Springer. ISBN: 978-0387974293.
//!
//! - Shumway, R.H., & Stoffer, D.S. (2017). *Time Series Analysis and Its
//!   Applications* (4th ed.). Springer. ISBN: 978-3319524511.
//!
//! R equivalent: `stats::arima()`, `forecast::auto.arima()`

use anyhow::{Result, anyhow};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;

/// Result from ARIMA model fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArimaResult {
    /// Column used for fitting
    pub column: String,
    /// AR order (p)
    pub p: usize,
    /// Differencing order (d)
    pub d: usize,
    /// MA order (q)
    pub q: usize,
    /// AR coefficients (phi)
    pub ar_coeffs: Vec<f64>,
    /// MA coefficients (theta)
    pub ma_coeffs: Vec<f64>,
    /// Intercept (mean)
    pub intercept: f64,
    /// Residuals
    pub residuals: Vec<f64>,
    /// Number of observations used
    pub n_obs: usize,
    /// Sum of squared residuals
    pub ssr: f64,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
}

/// Result from ARIMA forecasting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArimaForecastResult {
    /// Column being forecasted
    pub column: String,
    /// Number of forecast steps
    pub horizon: usize,
    /// Forecasted values
    pub forecast: Vec<f64>,
    /// Standard errors of forecast (if available)
    pub std_errors: Option<Vec<f64>>,
}

/// Fit an ARIMA model to a time series column.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series
/// * `column` - The column name with time series values
/// * `p` - AR order (number of autoregressive terms)
/// * `d` - Differencing order
/// * `q` - MA order (number of moving average terms)
///
/// # Returns
/// `ArimaResult` containing fitted model parameters.
pub fn run_arima(
    dataset: &Dataset,
    column: &str,
    p: usize,
    d: usize,
    q: usize,
) -> Result<ArimaResult> {
    use arima::estimate;

    // Extract time series data
    let df = dataset.df();
    let col = df
        .column(column)
        .map_err(|e| anyhow!("Column '{}' not found: {}", column, e))?;

    let values: Vec<f64> = col
        .f64()
        .map_err(|e| anyhow!("Column must be numeric: {}", e))?
        .into_no_null_iter()
        .collect();

    if values.len() < p + d + q + 10 {
        return Err(anyhow!(
            "Not enough observations ({}) for ARIMA({},{},{})",
            values.len(),
            p,
            d,
            q
        ));
    }

    // Fit ARIMA model using conditional sum of squares
    // The fit function handles differencing internally via the d parameter
    let coeffs =
        estimate::fit(&values, p, d, q).map_err(|e| anyhow!("ARIMA fitting failed: {:?}", e))?;

    // Split coefficients into AR and MA parts
    // The coefficients are returned as [phi_1, ..., phi_p, theta_1, ..., theta_q]
    let (ar_coeffs, ma_coeffs) = if coeffs.len() == p + q {
        (coeffs[..p].to_vec(), coeffs[p..].to_vec())
    } else {
        // Fallback: all coefficients are AR if split doesn't match expected
        (coeffs.clone(), vec![])
    };

    // Apply differencing for residual calculation
    let diff_series = difference(&values, d);

    // Calculate intercept (mean of differenced series)
    let intercept = diff_series.iter().sum::<f64>() / diff_series.len() as f64;

    // Calculate residuals
    let phi_opt = if ar_coeffs.is_empty() {
        None
    } else {
        Some(ar_coeffs.as_slice())
    };
    let theta_opt = if ma_coeffs.is_empty() {
        None
    } else {
        Some(ma_coeffs.as_slice())
    };
    let residuals = estimate::residuals(&diff_series, intercept, phi_opt, theta_opt)
        .map_err(|e| anyhow!("Residual calculation failed: {:?}", e))?;

    // Calculate sum of squared residuals
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();

    // Calculate AIC: n*log(SSR/n) + 2*(p+q+1)
    let n = residuals.len() as f64;
    let k = (p + q + 1) as f64;
    let aic = n * (ssr / n).ln() + 2.0 * k;

    Ok(ArimaResult {
        column: column.to_string(),
        p,
        d,
        q,
        ar_coeffs,
        ma_coeffs,
        intercept,
        residuals,
        n_obs: values.len(),
        ssr,
        aic,
    })
}

/// Forecast future values using a fitted ARIMA model.
///
/// # Arguments
/// * `dataset` - The dataset containing the time series
/// * `column` - The column name with time series values
/// * `p` - AR order
/// * `d` - Differencing order
/// * `q` - MA order
/// * `horizon` - Number of steps to forecast
///
/// # Returns
/// `ArimaForecastResult` containing forecasted values.
pub fn forecast_arima(
    dataset: &Dataset,
    column: &str,
    p: usize,
    d: usize,
    q: usize,
    horizon: usize,
) -> Result<ArimaForecastResult> {
    use arima::sim;

    // First fit the model
    let model = run_arima(dataset, column, p, d, q)?;

    // Extract original time series
    let df = dataset.df();
    let col = df
        .column(column)
        .map_err(|e| anyhow!("Column '{}' not found: {}", column, e))?;

    let values: Vec<f64> = col
        .f64()
        .map_err(|e| anyhow!("Column must be numeric: {}", e))?
        .into_no_null_iter()
        .collect();

    // Prepare AR and MA coefficients as Option slices
    let ar_opt = if model.ar_coeffs.is_empty() {
        None
    } else {
        Some(model.ar_coeffs.as_slice())
    };
    let ma_opt = if model.ma_coeffs.is_empty() {
        None
    } else {
        Some(model.ma_coeffs.as_slice())
    };

    // Create a zero-noise function for deterministic forecast
    let noise_fn = |_idx: usize, _rng: &mut StdRng| 0.0;
    let mut rng = StdRng::seed_from_u64(42);

    // Forecast using arima_forecast
    // The function handles differencing internally via the d parameter
    let forecast = sim::arima_forecast(&values, horizon, ar_opt, ma_opt, d, &noise_fn, &mut rng)
        .map_err(|e| anyhow!("Forecasting failed: {:?}", e))?;

    Ok(ArimaForecastResult {
        column: column.to_string(),
        horizon,
        forecast,
        std_errors: None,
    })
}

/// Apply differencing to a time series.
fn difference(series: &[f64], d: usize) -> Vec<f64> {
    let mut result = series.to_vec();
    for _ in 0..d {
        result = result.windows(2).map(|w| w[1] - w[0]).collect();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difference() {
        let series = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let diff1 = difference(&series, 1);
        assert_eq!(diff1, vec![1.0, 2.0, 3.0, 4.0]);

        let diff2 = difference(&series, 2);
        assert_eq!(diff2, vec![1.0, 1.0, 1.0]);
    }
}
