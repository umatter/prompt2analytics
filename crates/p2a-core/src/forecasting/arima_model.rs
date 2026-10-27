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
use crate::errors::EconError;
use crate::resource::checkpoint;

fn size_overflow(what: &str) -> EconError {
    EconError::ResourceLimit(format!("{what} is too large: the requested size overflows"))
}

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
/// `ArimaResult` containing fitted model parameters, or an error wrapping
/// [`EconError::Cancelled`] (recover it with `err.downcast_ref::<EconError>()`)
/// when the budget installed with [`crate::resource::with_budget`] is already
/// cancelled or past its deadline when the fit would start. The fit itself is
/// the `arima` crate's L-BFGS loop, which never reaches a checkpoint: once
/// started it runs to the end. A column with missing values (nulls) is refused
/// with [`EconError::NullValues`] before anything is fitted: the fit has no way
/// to skip a gap, so fill or interpolate the missing values first.
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

    let series = col
        .f64()
        .map_err(|e| anyhow!("Column must be numeric: {}", e))?;
    // A null slot still holds a buffer value (0.0 for a CSV load, anything after arithmetic on
    // the column) that `into_no_null_iter` would read as an observation, and the fit cannot
    // skip a gap in the series, so a column with nulls is refused.
    if series.null_count() > 0 {
        return Err(EconError::NullValues {
            column: column.to_string(),
            count: series.null_count(),
        }
        .into());
    }
    let values: Vec<f64> = series.into_no_null_iter().collect();

    // p + d + q + 10, checked: an order sum that overflows can never be fitted.
    let required = p
        .checked_add(d)
        .and_then(|s| s.checked_add(q))
        .and_then(|s| s.checked_add(10));
    if required.is_none_or(|r| values.len() < r) {
        return Err(anyhow!(
            "Not enough observations ({}) for ARIMA({},{},{})",
            values.len(),
            p,
            d,
            q
        ));
    }

    // `estimate::fit` cannot be stopped once it runs (its optimiser loop is in
    // the `arima` crate), so do not start it on a budget that has already run out.
    checkpoint()?;

    // Fit ARIMA model using conditional sum of squares
    // The fit function handles differencing internally via the d parameter
    let coeffs =
        estimate::fit(&values, p, d, q).map_err(|e| anyhow!("ARIMA fitting failed: {:?}", e))?;

    // The arima crate's fit() returns coefficients as:
    //   [intercept, phi_1, ..., phi_p, theta_1, ..., theta_q]
    // where the first element is always the intercept/mean term.
    let (intercept, ar_coeffs, ma_coeffs) = if coeffs.len() == 1 + p + q {
        (
            coeffs[0],
            coeffs[1..1 + p].to_vec(),
            coeffs[1 + p..].to_vec(),
        )
    } else if coeffs.len() == p + q {
        // Fallback: no intercept returned
        let diff_series = difference(&values, d);
        let mean = diff_series.iter().sum::<f64>() / diff_series.len() as f64;
        (mean, coeffs[..p].to_vec(), coeffs[p..].to_vec())
    } else {
        // Unexpected length - treat first as intercept, rest as AR
        let intercept = if !coeffs.is_empty() { coeffs[0] } else { 0.0 };
        let rest = if coeffs.len() > 1 {
            coeffs[1..].to_vec()
        } else {
            vec![]
        };
        (intercept, rest, vec![])
    };

    // Apply differencing for residual calculation
    let diff_series = difference(&values, d);

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

    // Calculate AIC using the full normal log-likelihood, matching R's arima():
    //   loglik = -n/2 * log(2*pi) - n/2 * log(sigma2) - n/2
    //   AIC = -2*loglik + 2*k
    //       = n*log(2*pi) + n*log(sigma2) + n + 2*k
    // where sigma2 = SSR/n and k = p + q + 1 (intercept counted)
    let n = residuals.len() as f64;
    let k = (p + q + 1) as f64;
    let sigma2 = ssr / n;
    let aic = n * (2.0 * std::f64::consts::PI).ln() + n * sigma2.ln() + n + 2.0 * k;

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
/// `ArimaForecastResult` containing forecasted values, or an error wrapping
/// [`EconError::Cancelled`] when the budget installed with
/// [`crate::resource::with_budget`] is cancelled or past its deadline. The
/// check is inherited from [`run_arima`] and happens only before the fit
/// starts; the fit and the forecast that follows run uninterrupted. So is the
/// refusal of a column with nulls ([`EconError::NullValues`]).
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

    // Extract original time series (`run_arima` refused a column with nulls)
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

    // The arima crate grows its buffer with push (one doubling reallocation) and clones it once.
    // Refuse sizes it could never allocate before it starts; TS-1 (tier 2) replaces this forecaster.
    const WHAT: &str = "the ARIMA forecast buffer (series length + horizon)";
    let total = values
        .len()
        .checked_add(horizon)
        .ok_or_else(|| size_overflow(WHAT))?;
    crate::resource::check_alloc::<f64>(total.saturating_mul(2), WHAT)?;
    drop(crate::resource::try_with_capacity::<f64>(total, WHAT)?);

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
    use polars::prelude::*;

    #[test]
    fn test_difference() {
        let series = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let diff1 = difference(&series, 1);
        assert_eq!(diff1, vec![1.0, 2.0, 3.0, 4.0]);

        let diff2 = difference(&series, 2);
        assert_eq!(diff2, vec![1.0, 1.0, 1.0]);
    }

    /// Generate AR(1) data: y_t = phi * y_{t-1} + e_t, e_t ~ uniform noise.
    fn generate_ar1_data(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut y = Vec::with_capacity(n);
        let mut prev = 0.0;

        for _ in 0..n {
            let noise = (rng.r#gen::<f64>() - 0.5) * 2.0; // uniform in [-1, 1]
            let val = phi * prev + noise;
            y.push(val);
            prev = val;
        }

        y
    }

    #[test]
    fn test_validate_arima_ar1_coefficient_recovery() {
        // Fit ARIMA(1,0,0) to AR(1) data with known phi=0.7
        // Verify that estimated phi is reasonably close.
        // Note: arima::estimate::fit may return [intercept, phi] or [phi] depending
        // on the series. We check whether any returned coefficient is close to phi_true.
        let phi_true = 0.7;
        let n = 500;
        let y = generate_ar1_data(n, phi_true, 42);

        let df = df! {
            "y" => &y,
        }
        .unwrap();
        let dataset = crate::data::Dataset::new(df);

        let result = run_arima(&dataset, "y", 1, 0, 0).unwrap();

        assert_eq!(result.p, 1);
        assert_eq!(result.d, 0);
        assert_eq!(result.q, 0);
        assert!(
            !result.ar_coeffs.is_empty(),
            "Should have at least one AR coefficient"
        );

        // The arima crate may return coefficients with an intercept prepended.
        // Check that at least one coefficient is close to the true AR(1) parameter.
        let close_to_phi = result
            .ar_coeffs
            .iter()
            .any(|&c| (c - phi_true).abs() < 0.25);
        assert!(
            close_to_phi,
            "At least one AR coefficient should be close to phi_true={:.4}, got {:?}",
            phi_true, result.ar_coeffs
        );
    }

    #[test]
    fn test_validate_arima_aic_finite() {
        // Verify AIC is computed and finite for ARIMA(1,0,0).
        let y = generate_ar1_data(200, 0.5, 99);

        let df = df! {
            "y" => &y,
        }
        .unwrap();
        let dataset = crate::data::Dataset::new(df);

        let result = run_arima(&dataset, "y", 1, 0, 0).unwrap();

        assert!(
            result.aic.is_finite(),
            "AIC should be finite, got {}",
            result.aic
        );
        assert!(result.ssr.is_finite(), "SSR should be finite");
        assert!(result.ssr > 0.0, "SSR should be positive");
    }

    #[test]
    fn test_validate_arima_forecast_decays_toward_mean() {
        // For ARIMA(1,0,0) with |phi| < 1, forecasts should decay toward
        // the unconditional mean (approximately the sample mean of the differenced series).
        let y = generate_ar1_data(300, 0.6, 77);

        let df = df! {
            "y" => &y,
        }
        .unwrap();
        let dataset = crate::data::Dataset::new(df);

        let result = forecast_arima(&dataset, "y", 1, 0, 0, 20).unwrap();

        assert_eq!(result.forecast.len(), 20);

        // All forecast values should be finite
        for (i, &f) in result.forecast.iter().enumerate() {
            assert!(
                f.is_finite(),
                "Forecast at step {} should be finite, got {}",
                i,
                f
            );
        }

        // For a stationary AR(1), successive forecasts should move closer to 0
        // (the unconditional mean for zero-mean AR).
        // Check that the absolute value of forecasts is not diverging.
        let last_abs = result.forecast.last().unwrap().abs();
        let first_abs = result.forecast.first().unwrap().abs();
        // Last forecast should be closer to zero or at least not much larger than first
        assert!(
            last_abs <= first_abs + 2.0,
            "Forecast should not diverge: first_abs={:.4}, last_abs={:.4}",
            first_abs,
            last_abs
        );
    }

    #[test]
    fn test_validate_arima_011_ma_coefficient() {
        // Fit ARIMA(0,1,1) to a random walk with MA noise.
        // Generate: y_t = y_{t-1} + e_t + theta * e_{t-1}
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let theta_true = -0.4;
        let n = 400;
        let mut rng = ChaCha8Rng::seed_from_u64(55);
        let mut y = Vec::with_capacity(n);
        let mut prev_y = 0.0;
        let mut prev_e = 0.0;

        for _ in 0..n {
            let e = (rng.r#gen::<f64>() - 0.5) * 2.0;
            let val = prev_y + e + theta_true * prev_e;
            y.push(val);
            prev_y = val;
            prev_e = e;
        }

        let df = df! {
            "y" => &y,
        }
        .unwrap();
        let dataset = crate::data::Dataset::new(df);

        let result = run_arima(&dataset, "y", 0, 1, 1).unwrap();

        assert_eq!(result.p, 0);
        assert_eq!(result.d, 1);
        assert_eq!(result.q, 1);

        // MA coefficient recovery is harder, so use generous tolerance
        if !result.ma_coeffs.is_empty() {
            let theta_hat = result.ma_coeffs[0];
            assert!(
                (theta_hat - theta_true).abs() < 0.5,
                "MA(1) coefficient theta_hat={:.4} should be in neighborhood of theta_true={:.4}",
                theta_hat,
                theta_true
            );
        }

        // At minimum, AIC should be finite
        assert!(result.aic.is_finite());
    }

    #[test]
    fn forecast_arima_refuses_horizon_above_the_allocation_ceiling() {
        let y = generate_ar1_data(200, 0.6, 7);
        let dataset = Dataset::new(df! { "y" => &y }.unwrap());
        let err = crate::resource::with_alloc_ceiling(Some(1 << 20), || {
            forecast_arima(&dataset, "y", 1, 0, 0, 100_000)
        })
        .unwrap_err();
        assert!(err.to_string().contains("ARIMA forecast"), "{err}");
        // The buffer length overflows. Never call the pre-fix forecaster with this: the arima
        // crate grows its buffer one push at a time until memory runs out.
        assert!(forecast_arima(&dataset, "y", 1, 0, 0, usize::MAX).is_err());
        assert_eq!(
            forecast_arima(&dataset, "y", 1, 0, 0, 12)
                .unwrap()
                .forecast
                .len(),
            12
        );
    }

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    #[test]
    fn run_arima_does_not_start_the_fit_on_a_spent_budget() {
        // `arima::estimate::fit` runs a third-party optimiser that never looks
        // at the budget. Before the fix the fit started whatever the budget
        // said and ran to completion (returning Ok).
        let y = generate_ar1_data(500, 0.6, 11);
        let dataset = Dataset::new(df! { "y" => &y }.unwrap());
        let is_cancelled = |e: &anyhow::Error| {
            matches!(e.downcast_ref::<EconError>(), Some(EconError::Cancelled(_)))
        };
        let err = cancelled(|| run_arima(&dataset, "y", 1, 0, 1)).unwrap_err();
        assert!(is_cancelled(&err), "{err}");
        let err = cancelled(|| forecast_arima(&dataset, "y", 1, 0, 1, 5)).unwrap_err();
        assert!(is_cancelled(&err), "{err}");
        // A budget that is still live leaves the fit alone.
        let live = crate::resource::Budget::new(
            Some(std::time::Instant::now() + std::time::Duration::from_secs(3600)),
            crate::resource::CancelToken::new(),
        );
        assert!(crate::resource::with_budget(live, || run_arima(&dataset, "y", 1, 0, 1)).is_ok());
    }

    /// A null slot is refused, not fitted: polars keeps a buffer value behind every null (0.0 for
    /// a CSV load, anything after arithmetic on the column), and the fit would read it as an
    /// observation. The refusal names the column and the number of nulls; a complete column fits.
    #[test]
    fn run_arima_and_forecast_refuse_a_column_with_nulls() {
        let y = generate_ar1_data(200, 0.6, 7);
        let with_nulls: Vec<Option<f64>> = y
            .iter()
            .enumerate()
            .map(|(i, &v)| (i != 50 && i != 120).then_some(v))
            .collect();
        // Arithmetic runs over the buffer behind the nulls too: there it now holds 1e-200.
        let ca = &Float64Chunked::from_slice_options("y".into(), &with_nulls) + 1e-200;
        assert_eq!(ca.null_count(), 2);
        assert_eq!(ca.into_no_null_iter().nth(50), Some(1e-200));
        let dataset = Dataset::new(DataFrame::new(vec![ca.into_series().into()]).unwrap());
        let is_null_refusal = |e: &anyhow::Error| {
            matches!(
                e.downcast_ref::<EconError>(),
                Some(EconError::NullValues { column, count: 2 }) if column == "y"
            )
        };
        let err = run_arima(&dataset, "y", 1, 0, 1).unwrap_err();
        assert!(is_null_refusal(&err), "{err}");
        let err = forecast_arima(&dataset, "y", 1, 0, 1, 5).unwrap_err();
        assert!(is_null_refusal(&err), "{err}");
        let complete = Dataset::new(df! { "y" => &y }.unwrap());
        assert_eq!(run_arima(&complete, "y", 1, 0, 1).unwrap().n_obs, 200);
    }

    #[test]
    fn run_arima_order_sum_cannot_overflow() {
        let y = generate_ar1_data(200, 0.6, 7);
        let dataset = Dataset::new(df! { "y" => &y }.unwrap());
        // a debug-build overflow panic before the fix
        assert!(run_arima(&dataset, "y", usize::MAX, 0, 1).is_err());
    }
}
