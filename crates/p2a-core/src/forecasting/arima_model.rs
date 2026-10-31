//! ARIMA(p, d, q) models fitted by conditional sum of squares, with R's parameterisation,
//! and their forecasts with prediction intervals.
//!
//! # Model
//!
//! With the backshift operator B, the AR polynomial φ(B) = 1 - φ₁B - ... - φₚBᵖ and the MA
//! polynomial θ(B) = 1 + θ₁B + ... + θ_qB^q (R's sign convention),
//!
//! φ(B) (1 - B)ᵈ (xₜ - μ - δ t) = θ(B) εₜ,   t = 1, ..., n,
//!
//! where μ is the series mean (R's "intercept", estimated by default when d = 0) and δ an
//! optional drift (forecast::Arima's `include.drift`, for d = 0 or 1). [`ArimaSpec`] selects
//! the orders and the two options.
//!
//! # Estimation
//!
//! As `stats::arima(method = "CSS")`: on w = (1 - B)ᵈ (x - μ - δ t) the residuals follow
//! eₜ = wₜ - Σᵢ φᵢ wₜ₋ᵢ - Σⱼ θⱼ eₜ₋ⱼ, conditioning on the first d + p observations (their
//! residuals are 0, and so are the residuals the MA terms would reach before them). The
//! coefficients minimise the sum of squared residuals; p2a solves that least-squares problem
//! by Levenberg-Marquardt with the analytic Jacobian of the recursion and Nielsen's
//! gain-ratio damping, starting where R starts (AR and MA coefficients 0, μ and δ by least
//! squares). The fit stops when the largest Gauss-Newton step, scaled by 1 + |coefficient|,
//! is at most 1e-8, or at most 1e-6 once the sum of squares no longer resolves a smaller step
//! (the rounding floor); `converged` says whether it did. That bounds the last step, not the
//! distance to the optimum, which is of the same order where the Gauss-Newton model of the
//! sum of squares is adequate. R's default method, CSS-ML, is not implemented.
//!
//! # Forecasts
//!
//! As `stats::predict.Arima` does for a CSS fit: the state-space form of
//! `stats::makeARIMA`, a Kalman filter over the mean- and drift-adjusted series, and
//! `KalmanForecast`. Standard errors use forecast::Arima's σ² = SSR / (n - d - k); the 80%
//! and 95% limits are those of `forecast::forecast`.
//!
//! # References
//!
//! - Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series
//!   Analysis: Forecasting and Control* (5th ed.). Wiley. ISBN: 978-1118675021.
//!
//! - Gardner, G., Harvey, A.C., & Phillips, G.D.A. (1980). Algorithm AS 154: An algorithm
//!   for exact maximum likelihood estimation of autoregressive-moving average models by
//!   means of Kalman filtering. *Applied Statistics*, 29(3), 311-322.
//!
//! - Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
//!   Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
//!
//! - Brockwell, P.J., & Davis, R.A. (1991). *Time Series: Theory and Methods*
//!   (2nd ed.). Springer. ISBN: 978-0387974293.
//!
//! - Nielsen, H.B. (1999). *Damping Parameter in Marquardt's Method*. Report
//!   IMM-REP-1999-05, Technical University of Denmark.
//!
//! R equivalent: `stats::arima(method = "CSS")`, `forecast::Arima(method = "CSS")`,
//! `forecast::forecast()`, `stats::predict.Arima()`.

use std::fmt;

use anyhow::Result;
use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::data::Dataset;
use crate::data::columns::{NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::solve;
use crate::resource::{check_alloc, checked_len, checkpoint, try_vec, try_zeros1, try_zeros2};

/// Outer iterations of the Levenberg-Marquardt loop.
const MAX_ITERATIONS: usize = 200;
/// Converged when the largest |Gauss-Newton step| / (1 + |coefficient|) is at most this.
const STEP_TOLERANCE: f64 = 1e-8;
/// When damping finds no better point, the fit counts as converged up to this scaled step.
const FLOOR_TOLERANCE: f64 = 1e-6;
const LAMBDA_START: f64 = 1e-3;
/// The factor that raises the damping after a rejected step; it doubles on every further
/// rejection and returns to this value after an accepted step (Nielsen 1999).
const NU_START: f64 = 2.0;
/// Above this damping no step improves the sum of squares: the rounding floor.
const LAMBDA_MAX: f64 = 1e16;
/// Keeps the damping positive after many accepted steps.
const LAMBDA_MIN: f64 = 1e-20;
/// `stats::makeARIMA`'s prior variance of the differencing states.
const KAPPA: f64 = 1e6;
/// Rows between budget checks in a pass over the series.
const ROW_STRIDE: usize = 4096;
/// Kalman filter steps between budget checks.
const FILTER_STRIDE: usize = 1024;
/// A matrix power below this (largest absolute entry) has decayed.
const DECAYED: f64 = 1e-12;
/// Squarings before a matrix that has not decayed counts as not stable.
const MAX_SQUARINGS: usize = 64;

/// The warning R's `predict.Arima` gives for an MA polynomial with a root inside the unit
/// circle.
const MA_NOT_INVERTIBLE: &str = "MA part of model is not invertible";
const AR_NOT_STATIONARY: &str = "AR part is not stationary";
const DIFFUSE_ARMA_PRIOR: &str = "AR part is not stationary: the Kalman filter starts the ARMA \
     state from a diffuse prior (variance 1e6) instead of the stationary covariance";

/// ARIMA(p, d, q) with forecast::Arima's mean and drift options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArimaSpec {
    /// AR order.
    pub p: usize,
    /// Order of differencing.
    pub d: usize,
    /// MA order.
    pub q: usize,
    /// Estimate the series mean (R's "intercept"). `None` = R's default: true when d = 0.
    /// `Some(true)` with d > 0 is an error.
    pub include_mean: Option<bool>,
    /// Estimate a drift: a linear trend `drift * t`, t = 1..n, in the levels
    /// (forecast::Arima `include.drift`). Only for d = 0 or 1.
    pub include_drift: bool,
}

impl ArimaSpec {
    /// ARIMA(p, d, q) with R's defaults: a mean when d = 0, no drift.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        ArimaSpec {
            p,
            d,
            q,
            include_mean: None,
            include_drift: false,
        }
    }

    /// Whether the mean is estimated: `include_mean`, or d = 0 when it is `None`.
    pub fn mean_included(&self) -> bool {
        self.include_mean.unwrap_or(self.d == 0)
    }

    /// forecast::Arima's label: "ARIMA(1,0,1) with non-zero mean", "ARIMA(1,0,0) with zero
    /// mean", "ARIMA(0,1,1) with drift" or "ARIMA(0,1,1)".
    pub fn label(&self) -> String {
        let base = format!("ARIMA({},{},{})", self.p, self.d, self.q);
        if self.include_drift {
            format!("{base} with drift")
        } else if self.mean_included() {
            format!("{base} with non-zero mean")
        } else if self.d == 0 {
            format!("{base} with zero mean")
        } else {
            base
        }
    }
}

impl fmt::Display for ArimaSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

/// Result of an ARIMA fit by conditional sum of squares.
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
    /// AR coefficients (phi), R's `ar1, ar2, ...`
    pub ar_coeffs: Vec<f64>,
    /// MA coefficients (theta, R's sign: `+ theta_j e_{t-j}`), R's `ma1, ma2, ...`
    pub ma_coeffs: Vec<f64>,
    /// The series mean (R's "intercept"); 0.0 when no mean is estimated (d > 0 or
    /// `include_mean = false`).
    pub intercept: f64,
    /// Constant of the differenced equation: mean x (1 - sum ar) (d = 0), drift x
    /// (1 - sum ar) (d = 1 with drift), 0.0 without a mean or drift; `None` when it varies
    /// with t (d = 0 with drift).
    #[serde(default)]
    pub constant: Option<f64>,
    /// Drift coefficient (forecast::Arima's "drift"); `None` when not estimated.
    #[serde(default)]
    pub drift: Option<f64>,
    /// Whether the mean was estimated.
    #[serde(default)]
    pub include_mean: bool,
    /// Every estimated coefficient in R's order and naming: ar1.., ma1.., intercept, drift.
    #[serde(default)]
    pub coef_names: Vec<String>,
    /// The values of `coef_names`.
    #[serde(default)]
    pub coefficients: Vec<f64>,
    /// CSS residuals, one per observation; the first `n_cond` are 0 (as R's
    /// `residuals()` of a CSS fit).
    pub residuals: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// d + p: observations that condition the CSS fit (stats::arima's `ncond`).
    #[serde(default)]
    pub n_cond: usize,
    /// Sum of squared residuals
    pub ssr: f64,
    /// SSR / (n - d - k), k = number of estimated coefficients (forecast::Arima's sigma2;
    /// used by the prediction intervals).
    #[serde(default)]
    pub sigma2: f64,
    /// CSS log-likelihood of stats::arima:
    /// -0.5 [(n - d) ln(SSR / (n - d - p)) + (n - d)(1 + ln 2 pi)].
    #[serde(default)]
    pub loglik: f64,
    /// -2 loglik + 2 (k + 1) on the CSS log-likelihood. R reports NA for CSS fits.
    pub aic: f64,
    /// Estimation method: "CSS".
    #[serde(default)]
    pub method: String,
    /// Whether the optimiser reached the CSS optimum.
    #[serde(default)]
    pub converged: bool,
    /// Levenberg-Marquardt iterations.
    #[serde(default)]
    pub iterations: usize,
    /// Non-convergence, an MA part that is not invertible, an AR part that is not
    /// stationary.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ArimaResult {
    /// The specification this fit was made with.
    pub fn spec(&self) -> ArimaSpec {
        ArimaSpec {
            p: self.p,
            d: self.d,
            q: self.q,
            include_mean: Some(self.include_mean),
            include_drift: self.drift.is_some(),
        }
    }

    /// The line that explains the mean, drift and constant, if any.
    fn note(&self) -> Option<String> {
        let constant = self
            .constant
            .map_or_else(|| "NA".to_string(), |c| format!("{c:.6}"));
        match (self.drift.is_some(), self.include_mean) {
            (true, _) if self.d == 0 => {
                Some("intercept + drift x t is the trend of the series mean".to_string())
            }
            (true, _) => Some(format!(
                "drift is the mean change per period; constant drift x (1 - sum(ar)) = {constant}"
            )),
            (false, true) => Some(format!(
                "intercept is the series mean (R's \"intercept\"); constant mean x (1 - sum(ar)) = {constant}"
            )),
            (false, false) => None,
        }
    }
}

impl fmt::Display for ArimaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} fitted by conditional sum of squares (CSS, as R arima(method = \"CSS\"))",
            self.spec().label()
        )?;
        writeln!(f, "Column: {}", self.column)?;
        writeln!(
            f,
            "Observations: {} (conditioning on the first {})",
            self.n_obs, self.n_cond
        )?;
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        if self.coefficients.is_empty() {
            writeln!(f, "  (none)")?;
        }
        for (name, value) in self.coef_names.iter().zip(&self.coefficients) {
            writeln!(f, "  {name:<10} {value:>14.6}")?;
        }
        if let Some(note) = self.note() {
            writeln!(f, "{note}")?;
        }
        writeln!(
            f,
            "sigma^2: {:.6} (SSR / (n - d - k), as forecast::Arima)",
            self.sigma2
        )?;
        writeln!(f, "Log-likelihood (CSS): {:.4}", self.loglik)?;
        writeln!(
            f,
            "AIC (CSS likelihood; R reports NA for CSS fits): {:.4}",
            self.aic
        )?;
        writeln!(f, "Sum of squared residuals: {:.4}", self.ssr)?;
        write!(
            f,
            "Converged: {} ({} iterations)",
            if self.converged { "yes" } else { "no" },
            self.iterations
        )?;
        for w in &self.warnings {
            write!(f, "\nWarning: {w}")?;
        }
        Ok(())
    }
}

/// Result of an ARIMA forecast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArimaForecastResult {
    /// Column being forecasted
    pub column: String,
    /// Number of forecast steps
    pub horizon: usize,
    /// Point forecasts, h = 1..horizon
    pub forecast: Vec<f64>,
    /// Standard errors of the forecasts (stats::predict.Arima); always `Some` from
    /// [`forecast_arima_with`], an `Option` for compatibility with stored results.
    pub std_errors: Option<Vec<f64>>,
    /// Coverage of the prediction intervals in percent: [80.0, 95.0].
    #[serde(default)]
    pub levels: Vec<f64>,
    /// `lower[i][h]`: lower limit for `levels[i]` (forecast::forecast's `lower`).
    #[serde(default)]
    pub lower: Vec<Vec<f64>>,
    /// `upper[i][h]`: upper limit for `levels[i]` (forecast::forecast's `upper`).
    #[serde(default)]
    pub upper: Vec<Vec<f64>>,
    /// The fit's warnings plus those of the forecaster.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The fit the forecasts come from (always `Some` from [`forecast_arima_with`]).
    #[serde(default)]
    pub fit: Option<ArimaResult>,
}

impl fmt::Display for ArimaForecastResult {
    /// The forecast table. Warnings the fit reported are left to the fit's own `Display`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self
            .fit
            .as_ref()
            .map_or_else(|| "ARIMA".to_string(), |fit| fit.spec().label());
        write!(
            f,
            "Forecasts from {label} for '{}' (80% and 95% prediction intervals, as R forecast())",
            self.column
        )?;
        let width = format!("  t+{}: ", self.forecast.len()).len();
        write!(
            f,
            "\n{:width$}{:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Forecast", "Std.Err", "Lo 80", "Hi 80", "Lo 95", "Hi 95"
        )?;
        // A value or "NA" where the vectors are shorter than the forecast (a stored result).
        let cell = |v: Option<&f64>| v.map_or_else(|| "NA".to_string(), |x| format!("{x:.4}"));
        let at = |rows: &[Vec<f64>], i: usize, h: usize| cell(rows.get(i).and_then(|r| r.get(h)));
        for (h, mean) in self.forecast.iter().enumerate() {
            let se = cell(self.std_errors.as_ref().and_then(|s| s.get(h)));
            let step = format!("  t+{}: ", h + 1);
            write!(
                f,
                "\n{step:width$}{mean:>11.4} {se:>11} {:>11} {:>11} {:>11} {:>11}",
                at(&self.lower, 0, h),
                at(&self.upper, 0, h),
                at(&self.lower, 1, h),
                at(&self.upper, 1, h)
            )?;
        }
        let fit_warnings = self.fit.as_ref().map(|fit| fit.warnings.as_slice());
        for w in &self.warnings {
            if !fit_warnings.is_some_and(|fw| fw.contains(w)) {
                write!(f, "\nWarning: {w}")?;
            }
        }
        Ok(())
    }
}

/// Fit an ARIMA(p, d, q) model by conditional sum of squares, as
/// `stats::arima(method = "CSS")` with forecast::Arima's mean and drift rules.
///
/// The column is read with [`NaPolicy::Fail`]: integer and Boolean columns are accepted,
/// and a missing value (null or NaN) is [`EconError::NullValues`] naming the column, since
/// the residual recursion cannot skip a gap; text is [`EconError::NonNumericColumn`].
///
/// # Errors
/// - `include_mean = Some(true)` with d > 0, `include_drift` with d >= 2, and fewer than
///   p + d + q + 10 observations: [`EconError::InvalidSpecification`];
/// - a coefficient that does not change the residuals (for instance an AR term on a series
///   that is constant after differencing): `InvalidSpecification` naming it;
/// - [`EconError::Cancelled`] when the budget installed with
///   [`crate::resource::with_budget`] is spent: the fit checks it before it starts, on
///   every iteration and every 4096 rows of a pass over the series.
///
/// Non-convergence is not an error: `converged` is false and `warnings` says why.
pub fn run_arima_with(
    dataset: &Dataset,
    column: &str,
    spec: &ArimaSpec,
) -> EconResult<ArimaResult> {
    let (x, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;
    fit_css(&x, spec, column)
}

/// Fit an ARIMA model to a time series column with R's defaults ([`ArimaSpec::new`]: a
/// mean when d = 0, no drift); see [`run_arima_with`].
///
/// # Arguments
/// * `dataset` - The dataset containing the time series
/// * `column` - The column name with time series values
/// * `p` - AR order (number of autoregressive terms)
/// * `d` - Differencing order
/// * `q` - MA order (number of moving average terms)
///
/// # Returns
/// `ArimaResult` with the fitted model, or the [`EconError`] of [`run_arima_with`] inside
/// the `anyhow::Error` (recover it with `err.downcast_ref::<EconError>()`). The fit checks
/// the budget installed with [`crate::resource::with_budget`] before it starts, on every
/// iteration and every 4096 rows of a pass ([`EconError::Cancelled`]). A column with
/// missing values (null or NaN) is [`EconError::NullValues`]; an integer column is
/// accepted.
pub fn run_arima(
    dataset: &Dataset,
    column: &str,
    p: usize,
    d: usize,
    q: usize,
) -> Result<ArimaResult> {
    Ok(run_arima_with(dataset, column, &ArimaSpec::new(p, d, q))?)
}

/// Forecast `horizon` steps ahead with an ARIMA model fitted by conditional sum of squares,
/// with standard errors and 80% / 95% prediction intervals, as
/// `forecast::forecast(forecast::Arima(x, order, method = "CSS"), h, level = c(80, 95))`.
///
/// The forecasts come from the state-space form of `stats::makeARIMA` and a Kalman filter
/// over the series minus its mean (and drift), as `stats::predict.Arima` computes them for
/// a CSS fit. The standard errors use forecast::Arima's sigma2 = SSR / (n - d - k).
///
/// # Errors
/// Those of [`run_arima_with`]; a horizon of 0 is `InvalidSpecification`, and a horizon
/// whose output vectors cannot be allocated is [`EconError::ResourceLimit`] (checked before
/// the fit). The filter checks the budget every 1024 steps and once per forecast step.
///
/// # Example
/// ```
/// use p2a_core::data::Dataset;
/// use p2a_core::forecasting::{ArimaSpec, forecast_arima_with};
/// use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // An AR(1) series around 10 with deterministic noise.
/// let mut y = vec![10.0];
/// for t in 1..60 {
///     let noise = ((t * 37 % 17) as f64 - 8.0) / 8.0;
///     y.push(10.0 + 0.6 * (y[t - 1] - 10.0) + noise);
/// }
/// let ds = Dataset::new(df! { "y" => &y }?);
/// let fc = forecast_arima_with(&ds, "y", &ArimaSpec::new(1, 0, 0), 4)?;
/// assert_eq!(fc.forecast.len(), 4);
/// assert!(fc.lower[1][0] < fc.forecast[0]); // the 95% lower limit
/// # Ok(())
/// # }
/// ```
pub fn forecast_arima_with(
    dataset: &Dataset,
    column: &str,
    spec: &ArimaSpec,
    horizon: usize,
) -> EconResult<ArimaForecastResult> {
    const WHAT: &str = "the ARIMA forecast (horizon x 6 output vectors)";
    let cells = checked_len(&[horizon, 6], WHAT)?;
    check_alloc::<f64>(cells, WHAT)?;
    if horizon == 0 {
        return Err(invalid("horizon must be at least 1".to_string()));
    }
    let (x, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;
    let fit = fit_css(&x, spec, column)?;
    let pred = arima_predict(&x, spec, &fit.coefficients, fit.sigma2, horizon)?;

    let normal = Normal::new(0.0, 1.0).map_err(|e| EconError::Internal(e.to_string()))?;
    let levels = vec![80.0, 95.0];
    let mut lower = Vec::with_capacity(levels.len());
    let mut upper = Vec::with_capacity(levels.len());
    for level in &levels {
        let z = normal.inverse_cdf(0.5 * (1.0 + level / 100.0));
        let mut lo = try_vec(horizon, 0.0, WHAT)?;
        let mut hi = try_vec(horizon, 0.0, WHAT)?;
        for h in 0..horizon {
            lo[h] = pred.mean[h] - z * pred.se[h];
            hi[h] = pred.mean[h] + z * pred.se[h];
        }
        lower.push(lo);
        upper.push(hi);
    }
    let mut warnings = fit.warnings.clone();
    for w in pred.warnings {
        if !warnings.contains(&w) {
            warnings.push(w);
        }
    }
    Ok(ArimaForecastResult {
        column: column.to_string(),
        horizon,
        forecast: pred.mean,
        std_errors: Some(pred.se),
        levels,
        lower,
        upper,
        warnings,
        fit: Some(fit),
    })
}

/// Forecast future values with an ARIMA model with R's defaults ([`ArimaSpec::new`]); see
/// [`forecast_arima_with`].
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
/// `ArimaForecastResult` with forecasts, standard errors and 80/95% limits, or the
/// [`EconError`] of [`forecast_arima_with`] inside the `anyhow::Error`, including
/// [`EconError::Cancelled`] (the fit checks the budget before it starts and on every
/// iteration, the forecast every 1024 filter steps) and [`EconError::NullValues`] for a
/// column with missing values.
pub fn forecast_arima(
    dataset: &Dataset,
    column: &str,
    p: usize,
    d: usize,
    q: usize,
    horizon: usize,
) -> Result<ArimaForecastResult> {
    Ok(forecast_arima_with(
        dataset,
        column,
        &ArimaSpec::new(p, d, q),
        horizon,
    )?)
}

fn invalid(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// A validated specification.
#[derive(Debug, Clone, Copy)]
struct Model {
    p: usize,
    d: usize,
    q: usize,
    mean: bool,
    drift: bool,
    /// Number of estimated coefficients.
    k: usize,
}

impl Model {
    /// Checks the options and that `n` observations can fit the model.
    fn new(spec: &ArimaSpec, n: usize) -> EconResult<Model> {
        let ArimaSpec { p, d, q, .. } = *spec;
        // p + d + q + 10, checked: an order sum that overflows can never be fitted.
        let required = p
            .checked_add(d)
            .and_then(|s| s.checked_add(q))
            .and_then(|s| s.checked_add(10));
        if required.is_none_or(|r| n < r) {
            return Err(invalid(format!(
                "Not enough observations ({n}) for ARIMA({p},{d},{q}): need at least p + d + q + 10"
            )));
        }
        if spec.include_mean == Some(true) && d > 0 {
            return Err(invalid(
                "include_mean = true needs d = 0: after differencing a mean has no effect on \
                 the fit or the forecasts (R's arima and forecast::Arima drop it); use \
                 include_drift for a trend with d = 1"
                    .to_string(),
            ));
        }
        if spec.include_drift && d >= 2 {
            return Err(invalid(
                "include_drift needs d = 0 or 1: (1 - B)^d removes a linear trend for d >= 2 \
                 (forecast::Arima drops the drift with a warning)"
                    .to_string(),
            ));
        }
        let mean = spec.mean_included();
        let drift = spec.include_drift;
        // p + q + 2 <= n here, so this cannot overflow.
        let k = p + q + usize::from(mean) + usize::from(drift);
        Ok(Model {
            p,
            d,
            q,
            mean,
            drift,
            k,
        })
    }

    /// Number of regression coefficients (mean, drift).
    fn n_beta(&self) -> usize {
        usize::from(self.mean) + usize::from(self.drift)
    }

    /// R's coefficient names: ar1.., ma1.., intercept, drift.
    fn names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.k);
        names.extend((1..=self.p).map(|i| format!("ar{i}")));
        names.extend((1..=self.q).map(|j| format!("ma{j}")));
        if self.mean {
            names.push("intercept".to_string());
        }
        if self.drift {
            names.push("drift".to_string());
        }
        names
    }

    /// Regressor `b` (mean, then drift) of the differenced equation at index `s` of the
    /// differenced series: 1 for the mean (d = 0) and for the drift with d = 1, the time
    /// t = s + 1 for the drift with d = 0.
    fn regressor(&self, b: usize, s: usize) -> f64 {
        let is_drift = self.drift && b + 1 == self.n_beta();
        if is_drift && self.d == 0 {
            (s + 1) as f64
        } else {
            1.0
        }
    }

    /// (ar, ma, [mean, drift]) of a coefficient vector in R order.
    fn split<'a>(&self, coef: &'a [f64]) -> (&'a [f64], &'a [f64], &'a [f64]) {
        let (ar, rest) = coef.split_at(self.p);
        let (ma, beta) = rest.split_at(self.q);
        (ar, ma, beta)
    }

    /// (mean, drift) of the regression coefficients, 0.0 when not estimated.
    fn mean_drift(&self, beta: &[f64]) -> (f64, f64) {
        let mean = if self.mean { beta[0] } else { 0.0 };
        let drift = if self.drift {
            beta[beta.len() - 1]
        } else {
            0.0
        };
        (mean, drift)
    }

    fn check_coefficients(&self, coefficients: &[f64]) -> EconResult<()> {
        if coefficients.len() == self.k {
            Ok(())
        } else {
            Err(invalid(format!(
                "expected {} coefficients ({}), got {}",
                self.k,
                self.names().join(", "),
                coefficients.len()
            )))
        }
    }
}

/// (1 - B)^d applied `d` times in place, as R's ARIMA_CSS does; returns the m = n - d
/// differenced values.
fn difference(x: &[f64], d: usize) -> EconResult<Vec<f64>> {
    let mut w = try_vec(x.len(), 0.0, "the differenced series")?;
    w.copy_from_slice(x);
    for _ in 0..d {
        checkpoint()?;
        for l in (1..w.len()).rev() {
            w[l] -= w[l - 1];
        }
    }
    let d = d.min(w.len());
    w.copy_within(d.., 0);
    w.truncate(w.len() - d);
    Ok(w)
}

/// Buffers of one CSS pass: adjusted series, residuals, the last q derivative rows, the
/// current row, J'J and J'e.
struct Workspace {
    w: Vec<f64>,
    e: Vec<f64>,
    ring: Vec<f64>,
    row: Vec<f64>,
    jtj: Array2<f64>,
    jte: Array1<f64>,
}

impl Workspace {
    fn new(m: usize, model: &Model) -> EconResult<Workspace> {
        const WHAT: &str = "the ARIMA CSS workspace";
        Ok(Workspace {
            w: try_vec(m, 0.0, WHAT)?,
            e: try_vec(m, 0.0, WHAT)?,
            ring: try_vec(checked_len(&[model.q, model.k], WHAT)?, 0.0, WHAT)?,
            row: try_vec(model.k, 0.0, WHAT)?,
            jtj: try_zeros2(model.k, model.k, WHAT)?,
            jte: try_zeros1(model.k, WHAT)?,
        })
    }
}

/// One pass of R's ARIMA_CSS recursion over the differenced series `dx` for `coef` (R
/// order): the residuals in `ws.e` (the first p are 0) and the sum of squares; with
/// `jacobian`, also J'J and J'e of the residuals' derivatives, accumulated row by row.
fn css_pass(
    dx: &[f64],
    model: &Model,
    coef: &[f64],
    ws: &mut Workspace,
    jacobian: bool,
) -> EconResult<f64> {
    let Model { p, q, k, .. } = *model;
    let (phi, theta, beta) = model.split(coef);
    let m = dx.len();
    let Workspace {
        w,
        e,
        ring,
        row,
        jtj,
        jte,
    } = ws;
    for s in 0..m {
        if s % ROW_STRIDE == ROW_STRIDE - 1 {
            checkpoint()?;
        }
        let mut v = dx[s];
        for (b, &coef_b) in beta.iter().enumerate() {
            v -= coef_b * model.regressor(b, s);
        }
        w[s] = v;
    }
    e[..p.min(m)].fill(0.0);
    if jacobian {
        jtj.fill(0.0);
        jte.fill(0.0);
    }
    let mut ssr = 0.0;
    for s in p..m {
        if (s - p) % ROW_STRIDE == ROW_STRIDE - 1 {
            checkpoint()?;
        }
        // The MA terms reach back only to residuals after the conditioning ones.
        let lags = q.min(s - p);
        let mut v = w[s];
        for (i, &phi_i) in phi.iter().enumerate() {
            v -= phi_i * w[s - i - 1];
        }
        for (j, &theta_j) in theta.iter().enumerate().take(lags) {
            v -= theta_j * e[s - j - 1];
        }
        e[s] = v;
        ssr += v * v;
        if !jacobian {
            continue;
        }
        for i in 0..p {
            row[i] = -w[s - i - 1];
        }
        for j in 0..q {
            row[p + j] = if j < lags { -e[s - j - 1] } else { 0.0 };
        }
        for b in 0..beta.len() {
            let mut g = -model.regressor(b, s);
            for (i, &phi_i) in phi.iter().enumerate() {
                g += phi_i * model.regressor(b, s - i - 1);
            }
            row[p + q + b] = g;
        }
        for (j, &theta_j) in theta.iter().enumerate().take(lags) {
            let slot = (s - j - 1) % q;
            let prev = &ring[slot * k..(slot + 1) * k];
            for (r, &pv) in row.iter_mut().zip(prev) {
                *r -= theta_j * pv;
            }
        }
        if q > 0 {
            let slot = s % q;
            ring[slot * k..(slot + 1) * k].copy_from_slice(row);
        }
        for a in 0..k {
            jte[a] += row[a] * v;
            for b in a..k {
                jtj[[a, b]] += row[a] * row[b];
            }
        }
    }
    if jacobian {
        for a in 0..k {
            for b in 0..a {
                jtj[[a, b]] = jtj[[b, a]];
            }
        }
    }
    Ok(ssr)
}

/// Residuals (length n, the first d + p are 0) and SSR for given coefficients in R order.
fn css_residuals(x: &[f64], spec: &ArimaSpec, coefficients: &[f64]) -> EconResult<(Vec<f64>, f64)> {
    let model = Model::new(spec, x.len())?;
    model.check_coefficients(coefficients)?;
    let dx = difference(x, model.d)?;
    let mut ws = Workspace::new(dx.len(), &model)?;
    let ssr = css_pass(&dx, &model, coefficients, &mut ws, false)?;
    Ok((full_residuals(x.len(), &model, &ws.e)?, ssr))
}

/// The residuals of the differenced series at length n: d leading zeros, then `e`.
fn full_residuals(n: usize, model: &Model, e: &[f64]) -> EconResult<Vec<f64>> {
    let mut residuals = try_vec(n, 0.0, "the ARIMA residuals")?;
    residuals[model.d..].copy_from_slice(e);
    Ok(residuals)
}

/// R's start: AR and MA coefficients 0, mean and drift by least squares of the differenced
/// series on their regressors.
fn start_values(dx: &[f64], model: &Model) -> EconResult<Vec<f64>> {
    let mut coef = try_vec(model.k, 0.0, "the ARIMA coefficients")?;
    let m = dx.len() as f64;
    let at = model.p + model.q;
    match (model.mean, model.drift) {
        (false, false) => {}
        (true, false) => coef[at] = dx.iter().sum::<f64>() / m,
        (false, true) if model.d == 1 => coef[at] = dx.iter().sum::<f64>() / m,
        (false, true) => {
            // drift alone with d = 0: regression through the origin on t = 1..m
            let (sxt, stt) = dx
                .iter()
                .enumerate()
                .fold((0.0, 0.0), |(sxt, stt), (s, &v)| {
                    let t = (s + 1) as f64;
                    (sxt + v * t, stt + t * t)
                });
            coef[at] = sxt / stt;
        }
        (true, true) => {
            // mean and drift with d = 0: a least-squares line on t = 1..m
            let t_mean = (m + 1.0) / 2.0;
            let x_mean = dx.iter().sum::<f64>() / m;
            let (sxt, stt) = dx
                .iter()
                .enumerate()
                .fold((0.0, 0.0), |(sxt, stt), (s, &v)| {
                    let dt = (s + 1) as f64 - t_mean;
                    (sxt + (v - x_mean) * dt, stt + dt * dt)
                });
            let slope = sxt / stt;
            coef[at] = x_mean - slope * t_mean;
            coef[at + 1] = slope;
        }
    }
    Ok(coef)
}

/// How the optimiser stopped.
struct Convergence {
    converged: bool,
    iterations: usize,
    /// The last scaled Gauss-Newton step (infinite when J'J could not be solved).
    step: f64,
}

/// max_k |delta_k| / (1 + |coef_k|).
fn scaled_step(delta: &Array1<f64>, coef: &[f64]) -> f64 {
    delta
        .iter()
        .zip(coef)
        .map(|(d, c)| d.abs() / (1.0 + c.abs()))
        .fold(0.0, f64::max)
}

/// The final Gauss-Newton step `delta` from `coef` (whose sum of squares is `ssr`), kept when
/// it does not raise the sum of squares beyond rounding: without it small coefficients stop
/// short.
fn final_step(
    dx: &[f64],
    model: &Model,
    coef: &mut [f64],
    trial: &mut [f64],
    delta: &Array1<f64>,
    ssr: f64,
    ws: &mut Workspace,
) -> EconResult<()> {
    for ((t, c), d) in trial.iter_mut().zip(coef.iter()).zip(delta) {
        *t = c + d;
    }
    let ssr_trial = css_pass(dx, model, trial, ws, false)?;
    if ssr_trial.is_finite() && ssr_trial <= ssr * (1.0 + 1e-15) {
        coef.copy_from_slice(trial);
    }
    Ok(())
}

/// Levenberg-Marquardt on the CSS residuals from `coef` (updated in place), damped by
/// Nielsen's (1999) gain ratio: rho = the actual decrease of the sum of squares over the
/// decrease the linearised residuals predict. A step with rho > 0 is accepted and lambda
/// scaled by max(1/3, 1 - (2 rho - 1)^3); otherwise lambda is multiplied by nu, which then
/// doubles. Accepting every step that lowers the sum of squares is not enough: where the
/// model leaves autocorrelated residuals, J'J underestimates the curvature and the
/// Gauss-Newton map can have an eigenvalue near -1, so the MA coefficients alternate around
/// the optimum while each step still lowers the sum of squares. Such steps achieve little of
/// their predicted decrease, and a small rho raises lambda until the oscillation is damped.
fn optimise(
    dx: &[f64],
    model: &Model,
    coef: &mut [f64],
    ws: &mut Workspace,
) -> EconResult<Convergence> {
    let k = model.k;
    if k == 0 {
        return Ok(Convergence {
            converged: true,
            iterations: 0,
            step: 0.0,
        });
    }
    let mut trial = try_vec(k, 0.0, "the ARIMA coefficients")?;
    let mut lambda = LAMBDA_START;
    let mut nu = NU_START;
    let mut step = f64::INFINITY;
    for iteration in 1..=MAX_ITERATIONS {
        checkpoint()?;
        let ssr = css_pass(dx, model, coef, ws, true)?;
        if !ssr.is_finite() {
            return Err(EconError::Computation(
                "ARIMA CSS: the sum of squared residuals is not finite at the starting values"
                    .to_string(),
            ));
        }
        for c in 0..k {
            if ws.jtj[[c, c]] == 0.0 {
                return Err(invalid(format!(
                    "coefficient '{}' does not change the CSS residuals, so it is not \
                     identified (is the series constant after differencing?)",
                    model.names()[c]
                )));
            }
        }
        let rhs = -&ws.jte;
        let gauss_newton = solve(&ws.jtj.view(), &rhs).ok();
        step = gauss_newton
            .as_ref()
            .map_or(f64::INFINITY, |delta| scaled_step(delta, coef));
        if let Some(delta) = gauss_newton.as_ref().filter(|_| step <= STEP_TOLERANCE) {
            final_step(dx, model, coef, &mut trial, delta, ssr, ws)?;
            return Ok(Convergence {
                converged: true,
                iterations: iteration,
                step,
            });
        }
        loop {
            let mut damped = ws.jtj.clone();
            for c in 0..k {
                damped[[c, c]] += lambda * ws.jtj[[c, c]];
            }
            let gain = match solve(&damped.view(), &rhs) {
                Ok(delta) => {
                    checkpoint()?;
                    for c in 0..k {
                        trial[c] = coef[c] + delta[c];
                    }
                    let ssr_trial = css_pass(dx, model, &trial, ws, false)?;
                    // ||e||^2 - ||e + J delta||^2 = delta' (lambda diag(J'J) delta - J'e),
                    // positive for a nonzero step.
                    let predicted: f64 = (0..k)
                        .map(|c| delta[c] * (lambda * ws.jtj[[c, c]] * delta[c] + rhs[c]))
                        .sum();
                    if ssr_trial.is_finite() {
                        (ssr - ssr_trial) / predicted
                    } else {
                        f64::NAN
                    }
                }
                // A singular or non-finite damped system: the step is rejected.
                Err(_) => f64::NAN,
            };
            // Accepted only when rho > 0; NaN (a failed solve, a non-finite sum of squares,
            // 0 / 0) is not.
            if gain > 0.0 {
                coef.copy_from_slice(&trial);
                let scale = (1.0 - (2.0 * gain - 1.0).powi(3)).max(1.0 / 3.0);
                lambda = (lambda * scale).max(LAMBDA_MIN);
                nu = NU_START;
                break;
            }
            lambda *= nu;
            nu *= 2.0;
            if lambda > LAMBDA_MAX {
                // No damped step lowers the sum of squares: the rounding floor is reached,
                // possibly a little short of the step test (a step that leaves the sum of
                // squares unchanged has rho = 0 and is rejected). The Gauss-Newton step is
                // still tried there.
                let converged = step <= FLOOR_TOLERANCE;
                if let Some(delta) = gauss_newton.as_ref().filter(|_| converged) {
                    final_step(dx, model, coef, &mut trial, delta, ssr, ws)?;
                }
                return Ok(Convergence {
                    converged,
                    iterations: iteration,
                    step,
                });
            }
        }
    }
    Ok(Convergence {
        converged: false,
        iterations: MAX_ITERATIONS,
        step,
    })
}

/// The companion matrix with first row `first_row` and ones on the subdiagonal.
fn companion(first_row: &[f64]) -> EconResult<Array2<f64>> {
    let r = first_row.len();
    let mut a = try_zeros2(r, r, "the ARIMA companion matrix")?;
    for (j, &c) in first_row.iter().enumerate() {
        a[[0, j]] = c;
    }
    for i in 1..r {
        a[[i, i - 1]] = 1.0;
    }
    Ok(a)
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}

fn all_finite(a: &Array2<f64>) -> bool {
    a.iter().all(|v| v.is_finite())
}

/// Whether the powers of `a` decay to 0 (repeated squaring): its eigenvalues lie inside the
/// unit circle.
fn decays(a: &Array2<f64>) -> EconResult<bool> {
    let mut power = a.clone();
    for _ in 0..MAX_SQUARINGS {
        if max_abs(&power) <= DECAYED {
            return Ok(true);
        }
        checkpoint()?;
        power = power.dot(&power);
        if !all_finite(&power) {
            return Ok(false);
        }
    }
    Ok(max_abs(&power) <= DECAYED)
}

/// Whether the polynomial 1 - c_1 B - ... - c_r B^r has all roots outside the unit circle
/// (its companion matrix is stable). AR part: c = phi; MA part: c = -theta.
fn polynomial_stable(c: &[f64]) -> EconResult<bool> {
    if c.iter().all(|&v| v == 0.0) {
        return Ok(true);
    }
    decays(&companion(c)?)
}

/// Stationary covariance of x_t = A x_{t-1} + R e_t (doubling: X <- X + A X A',
/// A <- A A from X = V = R R', until A has decayed); `None` when A is not stable.
fn stationary_covariance(a: &Array2<f64>, v: &Array2<f64>) -> EconResult<Option<Array2<f64>>> {
    let mut x = v.clone();
    let mut power = a.clone();
    for _ in 0..MAX_SQUARINGS {
        if max_abs(&power) <= DECAYED {
            return Ok(Some(x));
        }
        checkpoint()?;
        x = &x + &power.dot(&x).dot(&power.t());
        power = power.dot(&power);
        if !all_finite(&x) || !all_finite(&power) {
            return Ok(None);
        }
    }
    Ok((max_abs(&power) <= DECAYED).then_some(x))
}

/// The ARIMA state space of `stats::makeARIMA`: r = max(p, q + 1) ARMA states followed by
/// d differencing states.
struct StateSpace<'a> {
    phi: &'a [f64],
    /// (1 - B)^d = 1 - delta_1 B - ... - delta_d B^d
    delta: Vec<f64>,
    /// (1, theta_1, ..., theta_q, 0, ...), length r
    r_vec: Vec<f64>,
    r: usize,
    rd: usize,
}

impl StateSpace<'_> {
    /// out = T v.
    fn apply_t(&self, v: &[f64], out: &mut [f64]) {
        let (r, p) = (self.r, self.phi.len());
        for i in 0..r {
            let mut t = if i + 1 < r { v[i + 1] } else { 0.0 };
            if i < p {
                t += self.phi[i] * v[0];
            }
            out[i] = t;
        }
        let d = self.delta.len();
        if d > 0 {
            let mut t = v[0];
            for (k, &dk) in self.delta.iter().enumerate() {
                t += dk * v[r + k];
            }
            out[r] = t;
            for i in 1..d {
                out[r + i] = v[r + i - 1];
            }
        }
    }

    /// Z'v, Z = (1, 0, ..., 0, delta_1, ..., delta_d).
    fn z_dot(&self, v: &[f64]) -> f64 {
        let mut t = v[0];
        for (k, &dk) in self.delta.iter().enumerate() {
            t += dk * v[self.r + k];
        }
        t
    }

    /// m = P Z.
    fn p_z(&self, pm: &Array2<f64>, m: &mut [f64]) {
        for (i, mi) in m.iter_mut().enumerate() {
            let mut t = pm[[i, 0]];
            for (k, &dk) in self.delta.iter().enumerate() {
                t += dk * pm[[i, self.r + k]];
            }
            *mi = t;
        }
    }

    /// a <- T a; P <- T P T' + V, with the companion structure (O(rd^2)).
    fn time_update(&self, a: &mut [f64], pm: &mut Array2<f64>, buf: &mut UpdateBuffers) {
        let rd = self.rd;
        self.apply_t(a, &mut buf.out);
        a.copy_from_slice(&buf.out);
        for j in 0..rd {
            for i in 0..rd {
                buf.col[i] = pm[[i, j]];
            }
            self.apply_t(&buf.col, &mut buf.out);
            for i in 0..rd {
                buf.tp[[i, j]] = buf.out[i];
            }
        }
        for i in 0..rd {
            for j in 0..rd {
                buf.col[j] = buf.tp[[i, j]];
            }
            self.apply_t(&buf.col, &mut buf.out);
            for j in 0..rd {
                pm[[i, j]] = buf.out[j];
            }
        }
        let nz = self
            .r_vec
            .iter()
            .rposition(|&v| v != 0.0)
            .map_or(0, |i| i + 1);
        for i in 0..nz {
            for j in 0..nz {
                pm[[i, j]] += self.r_vec[i] * self.r_vec[j];
            }
        }
    }
}

/// Scratch space of [`StateSpace::time_update`].
struct UpdateBuffers {
    tp: Array2<f64>,
    col: Vec<f64>,
    out: Vec<f64>,
}

/// stats::predict.Arima's output for given coefficients.
struct ArimaPrediction {
    mean: Vec<f64>,
    se: Vec<f64>,
    warnings: Vec<String>,
}

/// The coefficients of (1 - B)^d as R's `Delta`: (1 - B)^d = 1 - delta_1 B - ... -
/// delta_d B^d.
fn differencing_weights(d: usize) -> EconResult<Vec<f64>> {
    let mut poly = try_vec(d + 1, 0.0, "the differencing polynomial")?;
    poly[0] = 1.0;
    for len in 1..=d {
        for i in (1..=len).rev() {
            poly[i] -= poly[i - 1];
        }
    }
    Ok(poly[1..].iter().map(|c| -c).collect())
}

/// predict.Arima for given coefficients (R order) and sigma2: `stats::makeARIMA`, the
/// Kalman filter of `ARIMA_Like` over x - mean - drift t, then `KalmanForecast`.
fn arima_predict(
    x: &[f64],
    spec: &ArimaSpec,
    coefficients: &[f64],
    sigma2: f64,
    horizon: usize,
) -> EconResult<ArimaPrediction> {
    const WHAT: &str = "the ARIMA state space";
    let n = x.len();
    let model = Model::new(spec, n)?;
    model.check_coefficients(coefficients)?;
    let (phi, theta, beta) = model.split(coefficients);
    let (mean, drift) = model.mean_drift(beta);
    let r = model.p.max(model.q + 1);
    let d = model.d;
    let rd = r + d;
    let mut warnings = Vec::new();

    let mut r_vec = try_vec(r, 0.0, WHAT)?;
    r_vec[0] = 1.0;
    r_vec[1..=model.q].copy_from_slice(theta);

    // P0: the stationary covariance of the ARMA states; kappa for the differencing states.
    let mut pm = try_zeros2(rd, rd, WHAT)?;
    let mut t_r = try_zeros2(r, r, WHAT)?;
    for (i, &phi_i) in phi.iter().enumerate() {
        t_r[[i, 0]] = phi_i;
    }
    for i in 0..r - 1 {
        t_r[[i, i + 1]] = 1.0;
    }
    let mut v_r = try_zeros2(r, r, WHAT)?;
    for i in 0..r {
        for j in 0..r {
            v_r[[i, j]] = r_vec[i] * r_vec[j];
        }
    }
    match stationary_covariance(&t_r, &v_r)? {
        Some(q0) => pm.slice_mut(s![0..r, 0..r]).assign(&q0),
        None => {
            // CSS does not constrain the AR part. R's Gardner Q0 is undefined here.
            for i in 0..r {
                pm[[i, i]] = KAPPA;
            }
            warnings.push(DIFFUSE_ARMA_PRIOR.to_string());
        }
    }
    drop(t_r);
    drop(v_r);
    for i in 0..d {
        pm[[r + i, r + i]] = KAPPA;
    }

    let ss = StateSpace {
        phi,
        delta: differencing_weights(d)?,
        r_vec,
        r,
        rd,
    };
    let mut buf = UpdateBuffers {
        tp: try_zeros2(rd, rd, WHAT)?,
        col: try_vec(rd, 0.0, WHAT)?,
        out: try_vec(rd, 0.0, WHAT)?,
    };
    let mut a = try_vec(rd, 0.0, WHAT)?;
    let mut m = try_vec(rd, 0.0, WHAT)?;
    for (t, &xt) in x.iter().enumerate() {
        if t % FILTER_STRIDE == FILTER_STRIDE - 1 {
            checkpoint()?;
        }
        let zt = xt - mean - drift * (t + 1) as f64;
        if t > 0 {
            ss.time_update(&mut a, &mut pm, &mut buf);
        }
        ss.p_z(&pm, &mut m);
        let f = ss.z_dot(&m);
        if !(f > 0.0 && f.is_finite()) {
            return Err(EconError::Computation(format!(
                "ARIMA Kalman filter: innovation variance {f} at t = {}",
                t + 1
            )));
        }
        let v = zt - ss.z_dot(&a);
        for (ai, &mi) in a.iter_mut().zip(m.iter()) {
            *ai += mi * v / f;
        }
        for i in 0..rd {
            for j in 0..rd {
                pm[[i, j]] -= m[i] * m[j] / f;
            }
        }
    }

    let mut forecast = try_vec(horizon, 0.0, "the ARIMA forecasts")?;
    let mut se = try_vec(horizon, 0.0, "the ARIMA forecasts")?;
    for h in 0..horizon {
        checkpoint()?;
        ss.time_update(&mut a, &mut pm, &mut buf);
        ss.p_z(&pm, &mut m);
        forecast[h] = ss.z_dot(&a) + mean + drift * (n + h + 1) as f64;
        se[h] = (sigma2 * ss.z_dot(&m)).sqrt();
    }
    let neg_theta: Vec<f64> = theta.iter().map(|t| -t).collect();
    if !polynomial_stable(&neg_theta)? {
        warnings.push(MA_NOT_INVERTIBLE.to_string());
    }
    Ok(ArimaPrediction {
        mean: forecast,
        se,
        warnings,
    })
}

/// The CSS fit of `x` (see the module docs).
fn fit_css(x: &[f64], spec: &ArimaSpec, column: &str) -> EconResult<ArimaResult> {
    let n = x.len();
    let model = Model::new(spec, n)?;
    // A spent budget never starts the fit (Tier 0); with k = 0 no iteration checks it.
    checkpoint()?;
    let dx = difference(x, model.d)?;
    let mut ws = Workspace::new(dx.len(), &model)?;
    let mut coef = start_values(&dx, &model)?;
    let convergence = optimise(&dx, &model, &mut coef, &mut ws)?;
    let ssr = css_pass(&dx, &model, &coef, &mut ws, false)?;
    if !ssr.is_finite() {
        return Err(EconError::Computation(
            "ARIMA CSS: the sum of squared residuals is not finite".to_string(),
        ));
    }
    let residuals = full_residuals(n, &model, &ws.e)?;

    let Model { p, d, q, k, .. } = model;
    let (ar, ma, beta) = model.split(&coef);
    let (mean, drift) = model.mean_drift(beta);
    let sum_ar: f64 = ar.iter().sum();
    let constant = if model.drift && d == 0 {
        None
    } else if model.mean {
        Some(mean * (1.0 - sum_ar))
    } else if model.drift {
        Some(drift * (1.0 - sum_ar))
    } else {
        Some(0.0)
    };
    let n_used = (n - d) as f64;
    let sigma2 = ssr / (n - d - k) as f64;
    let loglik = -0.5
        * (n_used * (ssr / (n - d - p) as f64).ln()
            + n_used * (1.0 + (2.0 * std::f64::consts::PI).ln()));
    let aic = -2.0 * loglik + 2.0 * (k + 1) as f64;

    let mut warnings = Vec::new();
    if !convergence.converged {
        warnings.push(format!(
            "CSS optimisation did not converge after {} iterations (largest scaled \
             Gauss-Newton step {:.1e}); the coefficients may be inaccurate",
            convergence.iterations, convergence.step
        ));
    }
    let neg_ma: Vec<f64> = ma.iter().map(|t| -t).collect();
    if !polynomial_stable(&neg_ma)? {
        warnings.push(MA_NOT_INVERTIBLE.to_string());
    }
    if !polynomial_stable(ar)? {
        warnings.push(AR_NOT_STATIONARY.to_string());
    }

    Ok(ArimaResult {
        column: column.to_string(),
        p,
        d,
        q,
        ar_coeffs: ar.to_vec(),
        ma_coeffs: ma.to_vec(),
        intercept: mean,
        constant,
        drift: model.drift.then_some(drift),
        include_mean: model.mean,
        coef_names: model.names(),
        coefficients: coef,
        residuals,
        n_obs: n,
        n_cond: d + p,
        ssr,
        sigma2,
        loglik,
        aic,
        method: "CSS".to_string(),
        converged: convergence.converged,
        iterations: convergence.iterations,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_difference() {
        let series = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let diff1 = difference(&series, 1).unwrap();
        assert_eq!(diff1, vec![1.0, 2.0, 3.0, 4.0]);

        let diff2 = difference(&series, 2).unwrap();
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

        // The AR(1) coefficient is close to the true parameter.
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
        // The output size overflows: refused before the fit (the arima crate's forecaster,
        // replaced by the state-space one, grew its buffer one push at a time).
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
        // The CSS fit checks the budget before it starts and on every
        // iteration; a spent budget never starts it.
        let y = generate_ar1_data(500, 0.6, 11);
        let dataset = Dataset::new(df! { "y" => &y }.unwrap());
        let is_cancelled = |e: &anyhow::Error| {
            matches!(e.downcast_ref::<EconError>(), Some(EconError::Cancelled(_)))
        };
        let err = cancelled(|| run_arima(&dataset, "y", 1, 0, 1)).unwrap_err();
        assert!(is_cancelled(&err), "{err}");
        let err = cancelled(|| forecast_arima(&dataset, "y", 1, 0, 1, 5)).unwrap_err();
        assert!(is_cancelled(&err), "{err}");
        let err =
            cancelled(|| run_arima_with(&dataset, "y", &ArimaSpec::new(1, 0, 1))).unwrap_err();
        assert!(matches!(err, EconError::Cancelled(_)), "{err}");
        let err = cancelled(|| forecast_arima_with(&dataset, "y", &ArimaSpec::new(1, 0, 1), 5))
            .unwrap_err();
        assert!(matches!(err, EconError::Cancelled(_)), "{err}");
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

    /// TS-17 on the old API: `intercept` is the mean, d > 0 fits no drift, AIC = -2 loglik + 2 (k + 1).
    /// R: forecast::Arima(x, order, method = "CSS"), re-solved to the exact CSS optimum (case
    /// arima_css_forecast, *_tight).
    #[test]
    fn test_validate_arima_mean_parameterisation() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("arima_css_forecast");
        let a = run_arima(&ds, "a", 1, 0, 0).unwrap();
        let c = run_arima(&ds, "c", 0, 1, 1).unwrap();
        Check::new(&case)
            .named(
                "a_100_coef_tight",
                &["ar1", "intercept"],
                &[a.ar_coeffs[0], a.intercept],
                Tol::iterative(1e-6),
            )
            .scalar("a_100_aic_css_tight", a.aic, Tol::stat(300))
            .named(
                "c_011_coef_tight",
                &["ma1"],
                &c.ma_coeffs,
                Tol::iterative(1e-6),
            )
            .scalar("c_011_aic_css_tight", c.aic, Tol::stat(300))
            .finish();
        assert_eq!(
            c.intercept, 0.0,
            "d = 1 without drift has no mean or drift term"
        );
    }

    /// X-2 (ARIMA sites) on the old API: Int64 accepted, null / NaN / text / absent columns
    /// refused by name.
    #[test]
    fn test_na_arima_int64_and_missing() {
        use crate::forecasting::test_fixtures::*;
        let ds = with_float_copy(&arima_series(), "ai");
        let int = run_arima(&ds, "ai", 1, 0, 1).unwrap();
        let flt = run_arima(&ds, "ai_f64", 1, 0, 1).unwrap();
        let coefs = |r: &ArimaResult| {
            [r.ar_coeffs.clone(), r.ma_coeffs.clone(), vec![r.intercept]].concat()
        };
        assert_same(&coefs(&int), &coefs(&flt));
        assert_eq!(int.ssr.to_bits(), flt.ssr.to_bits());
        assert_null_values(
            econ(&run_arima(&ds, "a_na", 1, 0, 0).unwrap_err()),
            "a_na",
            2,
        );
        assert_null_values(
            econ(&run_arima(&ds, "ai_na", 1, 0, 0).unwrap_err()),
            "ai_na",
            2,
        );
        assert_non_numeric(econ(&run_arima(&ds, "lab", 1, 0, 0).unwrap_err()), "lab");
        assert_not_found(econ(&run_arima(&ds, "nope", 1, 0, 0).unwrap_err()), "nope");
        // the one Rust-built fixture (design 2.2): NaN is missing
        let nan = with_nan_at(&arima_series(), "a", 17);
        assert_null_values(econ(&run_arima(&nan, "a", 1, 0, 0).unwrap_err()), "a", 1);
    }

    /// R: forecast::Arima(method = "CSS") at the exact optimum; 13 fits, n = 300.
    #[test]
    fn test_validate_arima_css_fit() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("arima_css_forecast");
        let n = 300;
        let mut check = Check::new(&case);
        for (key, column, spec) in css_fits() {
            let fit = run_arima_with(&ds, column, &spec).unwrap();
            assert!(fit.converged, "{key}: {:?}", fit.warnings);
            assert_eq!(fit.method, "CSS");
            if !fit.coefficients.is_empty() {
                check
                    .named(
                        &format!("{key}_coef_tight"),
                        &fit.coef_names,
                        &fit.coefficients,
                        Tol::iterative(1e-6),
                    )
                    // forecast::Arima's own optimum (BFGS, numerical gradients): up to 2.3e-4
                    // away (dr_121)
                    .named(
                        &format!("{key}_coef"),
                        &fit.coef_names,
                        &fit.coefficients,
                        Tol::iterative(1e-3),
                    );
            }
            let head = case.vector(&format!("{key}_resid_head_tight")).len();
            check
                .scalar(&format!("{key}_ssr_tight"), fit.ssr, Tol::stat(n))
                .scalar(&format!("{key}_sigma2_tight"), fit.sigma2, Tol::stat(n))
                .scalar(&format!("{key}_loglik_tight"), fit.loglik, Tol::stat(n))
                .scalar(&format!("{key}_aic_css_tight"), fit.aic, Tol::stat(n))
                .scalar(&format!("{key}_n_cond"), fit.n_cond as f64, Tol::exact())
                .scalar(
                    &format!("{key}_n_coef"),
                    fit.coefficients.len() as f64,
                    Tol::exact(),
                )
                // residuals: coefficients converge to 1e-8, so residuals agree to about 2e-8
                // absolute (scale 1-10)
                .vector(
                    &format!("{key}_resid_head_tight"),
                    &fit.residuals[..head],
                    Tol::new(1e-6, 1e-7),
                )
                .vector(
                    &format!("{key}_resid_tail_tight"),
                    &fit.residuals[n - 3..],
                    Tol::new(1e-6, 1e-7),
                );
        }
        check.finish();
    }

    /// TS-35. R: forecast::Arima(order = c(1, 1, 1), include.drift = TRUE, method = "CSS") on a
    /// seasonal series the model misspecifies (n = 1000 and its first 100 values), and the exact
    /// CSS optimum. There the Gauss-Newton map has an eigenvalue near -1, so the MA coefficient
    /// oscillates while every step lowers the sum of squares; the fit must still converge.
    #[test]
    fn test_validate_arima_misspecified_seasonal() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("arima_misspecified_seasonal");
        let spec = ArimaSpec {
            include_drift: true,
            ..ArimaSpec::new(1, 1, 1)
        };
        let mut check = Check::new(&case);
        for n in [100, 1000] {
            let key = format!("n{n}");
            let series = Dataset::new(ds.head(Some(n)));
            let fit = run_arima_with(&series, "y", &spec).unwrap();
            assert!(
                fit.converged,
                "{key}: {} iterations, {:?}",
                fit.iterations, fit.warnings
            );
            assert!(fit.warnings.is_empty(), "{key}: {:?}", fit.warnings);
            check
                .named(
                    &format!("{key}_coef_tight"),
                    &fit.coef_names,
                    &fit.coefficients,
                    Tol::iterative(1e-6),
                )
                // forecast::Arima's own optimum (optim's default reltol): up to 2.8e-5
                // (relative) away, the drift at n = 100
                .named(
                    &format!("{key}_coef"),
                    &fit.coef_names,
                    &fit.coefficients,
                    Tol::iterative(1e-3),
                )
                .scalar(&format!("{key}_ssr_tight"), fit.ssr, Tol::stat(n))
                .scalar(&format!("{key}_sigma2_tight"), fit.sigma2, Tol::stat(n))
                .scalar(&format!("{key}_loglik_tight"), fit.loglik, Tol::stat(n))
                .scalar(&format!("{key}_aic_css_tight"), fit.aic, Tol::stat(n))
                .scalar(&format!("{key}_n"), fit.n_obs as f64, Tol::exact());
        }
        check.finish();
    }

    /// (case prefix, column, spec): the 13 fits of arima_css_forecast.
    fn css_fits() -> Vec<(&'static str, &'static str, ArimaSpec)> {
        let drift = |p, d, q| ArimaSpec {
            include_drift: true,
            ..ArimaSpec::new(p, d, q)
        };
        vec![
            ("a_100", "a", ArimaSpec::new(1, 0, 0)),
            ("a_101", "a", ArimaSpec::new(1, 0, 1)),
            ("b_200", "b", ArimaSpec::new(2, 0, 0)),
            ("m_101", "m", ArimaSpec::new(1, 0, 1)),
            ("m_002", "m", ArimaSpec::new(0, 0, 2)),
            ("c_011", "c", ArimaSpec::new(0, 1, 1)),
            ("c_110", "c", ArimaSpec::new(1, 1, 0)),
            ("c_010", "c", ArimaSpec::new(0, 1, 0)),
            ("dr_011_drift", "dr", drift(0, 1, 1)),
            ("dr_121", "dr", ArimaSpec::new(1, 2, 1)),
            ("dr_100_drift", "dr", drift(1, 0, 0)),
            (
                "m_100_nomean",
                "m",
                ArimaSpec {
                    include_mean: Some(false),
                    ..ArimaSpec::new(1, 0, 0)
                },
            ),
            ("ai_101", "ai", ArimaSpec::new(1, 0, 1)),
        ]
    }

    #[test]
    fn test_arima_spec_validation() {
        let ds = crate::forecasting::test_fixtures::arima_series();
        let msg = |r: EconResult<ArimaResult>| r.unwrap_err().to_string();
        let mean_d1 = ArimaSpec {
            include_mean: Some(true),
            ..ArimaSpec::new(0, 1, 1)
        };
        assert!(msg(run_arima_with(&ds, "c", &mean_d1)).contains("include_mean"));
        let drift_d2 = ArimaSpec {
            include_drift: true,
            ..ArimaSpec::new(1, 2, 1)
        };
        assert!(msg(run_arima_with(&ds, "dr", &drift_d2)).contains("include_drift"));
        let short = Dataset::new(df! { "y" => [1.0, 2.5, 1.7, 3.1, 2.2, 2.9] }.unwrap());
        assert!(
            msg(run_arima_with(&short, "y", &ArimaSpec::new(1, 0, 1)))
                .contains("Not enough observations")
        );
        let flat = Dataset::new(df! { "y" => vec![5.0; 40] }.unwrap());
        assert!(
            msg(run_arima_with(&flat, "y", &ArimaSpec::new(1, 0, 0))).contains("not identified")
        );
        assert_eq!(
            ArimaSpec::new(1, 0, 1).label(),
            "ARIMA(1,0,1) with non-zero mean"
        );
        let drift = ArimaSpec {
            include_drift: true,
            ..ArimaSpec::new(0, 1, 1)
        };
        assert_eq!(drift.label(), "ARIMA(0,1,1) with drift");
    }

    /// TS-1 on the old API. R: forecast::forecast(Arima(x, order, method = "CSS"), h = 5) at
    /// the exact optimum.
    #[test]
    fn test_validate_arima_forecast_default_spec() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("arima_css_forecast");
        let n = 300;
        let mut check = Check::new(&case);
        for (key, column, (p, d, q)) in [
            ("a_100", "a", (1, 0, 0)),
            ("c_011", "c", (0, 1, 1)),
            ("dr_121", "dr", (1, 2, 1)),
        ] {
            let fc = forecast_arima(&ds, column, p, d, q, 5).unwrap();
            check
                .vector(&format!("{key}_mean_tight"), &fc.forecast, Tol::stat(n))
                .vector(
                    &format!("{key}_se_tight"),
                    &fc.std_errors.clone().unwrap_or_default(),
                    Tol::stat(n),
                );
        }
        check.finish();
    }

    /// The forecaster alone: stats::predict.Arima for fixed coefficients (stats::arima's
    /// sigma2 = SSR / (n - d - p)).
    #[test]
    fn test_validate_arima_fixed_forecast() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("arima_fixed_forecast");
        let mut check = Check::new(&case);
        for (key, column, spec) in [
            ("a_100", "a", ArimaSpec::new(1, 0, 0)),
            ("m_101", "m", ArimaSpec::new(1, 0, 1)),
            ("c_011", "c", ArimaSpec::new(0, 1, 1)),
            ("b_200", "b", ArimaSpec::new(2, 0, 0)),
            ("m40_101", "m", ArimaSpec::new(1, 0, 1)),
        ] {
            let n = case.scalar(&format!("{key}_n")) as usize;
            let (x, _) = numeric_column(ds.df(), column, NaPolicy::Fail).unwrap();
            let coefs = case.vector(&format!("{key}_fixed"));
            let sigma2 = case.scalar(&format!("{key}_sigma2"));
            let (resid, _) = css_residuals(&x[..n], &spec, &coefs).unwrap();
            let pred = arima_predict(&x[..n], &spec, &coefs, sigma2, 8).unwrap();
            check
                .vector(&format!("{key}_pred"), &pred.mean, Tol::coef(n))
                .vector(&format!("{key}_se"), &pred.se, Tol::se(n))
                .vector(&format!("{key}_resid_tail"), &resid[n - 3..], Tol::coef(n));
        }
        check.finish();
    }

    /// R: forecast::forecast(h = 5, level = c(80, 95)) at the exact CSS optimum; 13 fits.
    #[test]
    fn test_validate_arima_css_forecast() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("arima_css_forecast");
        let n = 300;
        let mut check = Check::new(&case);
        for (key, column, spec) in css_fits() {
            let fc = forecast_arima_with(&ds, column, &spec, 5).unwrap();
            assert_eq!(fc.levels, vec![80.0, 95.0]);
            assert!(fc.fit.as_ref().unwrap().converged, "{key}");
            check
                .vector(&format!("{key}_mean_tight"), &fc.forecast, Tol::stat(n))
                .vector(
                    &format!("{key}_se_tight"),
                    fc.std_errors.as_ref().unwrap(),
                    Tol::stat(n),
                )
                .vector(&format!("{key}_lower80_tight"), &fc.lower[0], Tol::stat(n))
                .vector(&format!("{key}_upper80_tight"), &fc.upper[0], Tol::stat(n))
                .vector(&format!("{key}_lower95_tight"), &fc.lower[1], Tol::stat(n))
                .vector(&format!("{key}_upper95_tight"), &fc.upper[1], Tol::stat(n))
                // forecast::Arima's own optimum: its means sit up to 3.0e-6 away (dr_121)
                .vector(&format!("{key}_mean"), &fc.forecast, Tol::iterative(1e-5));
        }
        check.finish();
    }

    #[test]
    fn run_arima_order_sum_cannot_overflow() {
        let y = generate_ar1_data(200, 0.6, 7);
        let dataset = Dataset::new(df! { "y" => &y }.unwrap());
        // a debug-build overflow panic before the fix
        assert!(run_arima(&dataset, "y", usize::MAX, 0, 1).is_err());
    }
}
