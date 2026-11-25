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
//! A seasonal model ARIMA(p, d, q)(P, D, Q)[s] ([`SeasonalOrder`]) multiplies the polynomials
//! by Φ(Bˢ) = 1 - Φ₁Bˢ - ... and Θ(Bˢ) = 1 + Θ₁Bˢ + ..., and differences by (1 - Bˢ)ᴰ as well:
//! the fit and the forecasts work on the expanded polynomials (orders p + sP and q + sQ), with
//! the Jacobian in the original coefficients. Below, d and p then stand for d + sD and p + sP.
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
//!   Practice* (3rd ed.). OTexts. <https://otexts.com/fpp3/>
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
/// When no damped step up to [`LAMBDA_MAX`] has a positive gain ratio (the rounding floor of
/// the sum of squares), the fit counts as converged if the scaled Gauss-Newton step is at
/// most this.
const FLOOR_TOLERANCE: f64 = 1e-6;
const LAMBDA_START: f64 = 1e-3;
/// The factor that raises the damping after a rejected step; it doubles on every further
/// rejection and returns to this value after an accepted step (Nielsen 1999).
const NU_START: f64 = 2.0;
/// The damping at which the search for a step with a positive gain ratio stops: above it
/// the damped steps are too short to lower the sum of squares beyond rounding.
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
/// A fit whose root-mean-square CSS residual is at most this times the largest absolute
/// value of the differenced series reproduces the series up to rounding.
const ZERO_RESIDUAL: f64 = 1e-12;

/// The warning R's `predict.Arima` gives for an MA polynomial with a root inside the unit
/// circle.
const MA_NOT_INVERTIBLE: &str = "MA part of model is not invertible";
const AR_NOT_STATIONARY: &str = "AR part is not stationary";
const DIFFUSE_ARMA_PRIOR: &str = "AR part is not stationary: the Kalman filter starts the ARMA \
     state from a diffuse prior (variance 1e6) instead of the stationary covariance";

/// Largest lag a seasonal model may reach in p + s P, q + s Q or d + s D: the forecasts' state
/// space has up to their sum of states, and its covariance the square of that.
const MAX_SEASONAL_LAG: usize = 400;

/// The seasonal part (P, D, Q)[period] of an ARIMA model; all zero (the default) for none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeasonalOrder {
    /// Seasonal AR order.
    pub p: usize,
    /// Order of seasonal differencing.
    pub d: usize,
    /// Seasonal MA order.
    pub q: usize,
    /// Observations per season (12 for monthly data with a yearly pattern); at least 2 when
    /// an order is above 0.
    pub period: usize,
}

impl SeasonalOrder {
    /// Whether the model has a seasonal term.
    pub fn is_seasonal(&self) -> bool {
        self.p > 0 || self.d > 0 || self.q > 0
    }
}

/// ARIMA(p, d, q)(P, D, Q)[s] with forecast::Arima's mean and drift options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArimaSpec {
    /// AR order.
    pub p: usize,
    /// Order of differencing.
    pub d: usize,
    /// MA order.
    pub q: usize,
    /// Seasonal part.
    #[serde(default)]
    pub seasonal: SeasonalOrder,
    /// Estimate the series mean (R's "intercept"). `None` = R's default: true when the series
    /// is not differenced (d = 0 and no seasonal difference). `Some(true)` otherwise is an error.
    pub include_mean: Option<bool>,
    /// Estimate a drift: a linear trend `drift * t`, t = 1..n, in the levels
    /// (forecast::Arima `include.drift`). Only with at most one difference in all.
    pub include_drift: bool,
}

impl ArimaSpec {
    /// ARIMA(p, d, q) with R's defaults: a mean when d = 0, no drift.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        ArimaSpec {
            p,
            d,
            q,
            seasonal: SeasonalOrder::default(),
            include_mean: None,
            include_drift: false,
        }
    }

    /// Whether the series is differenced, regularly or seasonally.
    fn differenced(&self) -> bool {
        self.d > 0 || self.seasonal.d > 0
    }

    /// Whether the mean is estimated: `include_mean`, or no differencing when it is `None`.
    pub fn mean_included(&self) -> bool {
        self.include_mean.unwrap_or(!self.differenced())
    }

    /// "ARIMA(1,1,1)" or "ARIMA(0,1,1)(0,1,1)[12]".
    pub fn orders(&self) -> String {
        let base = format!("ARIMA({},{},{})", self.p, self.d, self.q);
        let o = &self.seasonal;
        if o.is_seasonal() {
            format!("{base}({},{},{})[{}]", o.p, o.d, o.q, o.period)
        } else {
            base
        }
    }

    /// forecast::Arima's label: "ARIMA(1,0,1) with non-zero mean", "ARIMA(1,0,0) with zero
    /// mean", "ARIMA(0,1,1) with drift" or "ARIMA(0,1,1)".
    pub fn label(&self) -> String {
        let base = self.orders();
        if self.include_drift {
            format!("{base} with drift")
        } else if self.mean_included() {
            format!("{base} with non-zero mean")
        } else if !self.differenced() {
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
    /// Seasonal part (P, D, Q)[s]; all zero without one.
    #[serde(default)]
    pub seasonal: SeasonalOrder,
    /// AR coefficients (phi), R's `ar1, ar2, ...`
    pub ar_coeffs: Vec<f64>,
    /// MA coefficients (theta, R's sign: `+ theta_j e_{t-j}`), R's `ma1, ma2, ...`
    pub ma_coeffs: Vec<f64>,
    /// Seasonal AR coefficients, R's `sar1, ...`
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sar_coeffs: Vec<f64>,
    /// Seasonal MA coefficients, R's `sma1, ...`
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sma_coeffs: Vec<f64>,
    /// The series mean (R's "intercept"); 0.0 when no mean is estimated (d > 0 or
    /// `include_mean = false`).
    pub intercept: f64,
    /// Constant of the differenced equation: mean x (1 - sum ar) (d = 0), drift x
    /// (1 - sum ar) (d = 1 with drift; times s with a seasonal difference instead), 0.0 without
    /// a mean or drift; `None` when it varies with t (drift, no difference). "sum ar" runs over
    /// the expanded AR polynomial of a seasonal model.
    #[serde(default)]
    pub constant: Option<f64>,
    /// Drift coefficient (forecast::Arima's "drift"); `None` when not estimated.
    #[serde(default)]
    pub drift: Option<f64>,
    /// Whether the mean was estimated.
    #[serde(default)]
    pub include_mean: bool,
    /// Every estimated coefficient in R's order and naming: ar1.., ma1.., sar1.., sma1.., intercept,
    /// drift.
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
    /// Whether the Levenberg-Marquardt fit met its step test: the largest Gauss-Newton step,
    /// scaled by 1 + |coefficient|, was at most 1e-8, or at most 1e-6 where no damped step
    /// lowered the sum of squares further. It is a local criterion: the CSS objective can
    /// have more than one minimum, and `converged` does not say this one is the lowest.
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
            seasonal: self.seasonal,
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
            (true, _) if self.constant.is_none() => {
                Some("intercept + drift x t is the trend of the series mean".to_string())
            }
            (true, _) if self.seasonal.is_seasonal() => {
                Some("drift is the mean change per period".to_string())
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
/// - a fit that reproduces the series up to rounding (zero residuals, for instance an MA
///   term or the mean alone on such a series), whose variance is 0 and log-likelihood
///   infinite: `InvalidSpecification`. R's arima stops on such a fit when the model has a
///   coefficient (mean, drift, AR or MA), on its singular Hessian or on an objective that
///   is not finite at the start; for ARIMA(0,d,0) without mean or drift it reports
///   sigma^2 = 0 and an infinite log-likelihood, which p2a refuses as well;
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
    seasonal: SeasonalOrder,
    mean: bool,
    drift: bool,
    /// Number of estimated coefficients.
    k: usize,
    /// Orders of the expanded polynomials: p + s P, d + s D, q + s Q.
    ar_lags: usize,
    diff_lags: usize,
    ma_lags: usize,
}

/// A coefficient vector in R order, by part.
struct Parts<'a> {
    ar: &'a [f64],
    ma: &'a [f64],
    sar: &'a [f64],
    sma: &'a [f64],
    beta: &'a [f64],
}

impl Model {
    /// Checks the options and that `n` observations can fit the model.
    fn new(spec: &ArimaSpec, n: usize) -> EconResult<Model> {
        let ArimaSpec { p, d, q, .. } = *spec;
        let mut seasonal = spec.seasonal;
        let label = spec.orders();
        if !seasonal.is_seasonal() {
            seasonal = SeasonalOrder::default();
        } else if seasonal.period < 2 {
            return Err(invalid(format!(
                "a seasonal ARIMA needs a period of at least 2 observations per season, got {}",
                seasonal.period
            )));
        }
        let reach = |order: usize, seasonal_order: usize| {
            seasonal_order
                .checked_mul(seasonal.period)
                .and_then(|l| l.checked_add(order))
        };
        let lags = (
            reach(p, seasonal.p),
            reach(d, seasonal.d),
            reach(q, seasonal.q),
        );
        // The lags + 10, checked: an order sum that overflows can never be fitted.
        let required = match lags {
            (Some(a), Some(b), Some(c)) => a
                .checked_add(b)
                .and_then(|s| s.checked_add(c))
                .and_then(|s| s.checked_add(10)),
            _ => None,
        };
        let (Some(required), (Some(ar_lags), Some(diff_lags), Some(ma_lags))) = (required, lags)
        else {
            return Err(invalid(format!(
                "Not enough observations ({n}) for {label}: orders too large to fit any series"
            )));
        };
        if n < required {
            let rule = if seasonal.is_seasonal() {
                "p + d + q + s (P + D + Q) + 10"
            } else {
                "p + d + q + 10"
            };
            return Err(invalid(format!(
                "Not enough observations ({n}) for {label}: need at least {required} ({rule})"
            )));
        }
        if seasonal.is_seasonal() && ar_lags.max(diff_lags).max(ma_lags) > MAX_SEASONAL_LAG {
            return Err(invalid(format!(
                "{label} reaches back too far: p + s P, d + s D and q + s Q may each be at most \
                 {MAX_SEASONAL_LAG}"
            )));
        }
        if spec.include_mean == Some(true) && diff_lags > 0 {
            return Err(invalid(
                "include_mean = true needs d = 0: after differencing a mean has no effect on \
                 the fit or the forecasts (R's arima and forecast::Arima drop it); use \
                 include_drift for a trend with d = 1"
                    .to_string(),
            ));
        }
        if spec.include_drift && d.saturating_add(seasonal.d) >= 2 {
            return Err(invalid(
                "include_drift needs d = 0 or 1: (1 - B)^d removes a linear trend for d >= 2, \
                 seasonal differences included (forecast::Arima drops the drift with a warning)"
                    .to_string(),
            ));
        }
        let mean = spec.mean_included();
        let drift = spec.include_drift;
        // The sum is at most n here, so this cannot overflow.
        let k = p + q + seasonal.p + seasonal.q + usize::from(mean) + usize::from(drift);
        Ok(Model {
            p,
            d,
            q,
            seasonal,
            mean,
            drift,
            k,
            ar_lags,
            diff_lags,
            ma_lags,
        })
    }

    /// Number of regression coefficients (mean, drift).
    fn n_beta(&self) -> usize {
        usize::from(self.mean) + usize::from(self.drift)
    }

    /// Number of ARMA coefficients.
    fn n_arma(&self) -> usize {
        self.p + self.q + self.seasonal.p + self.seasonal.q
    }

    /// R's coefficient names: ar1.., ma1.., sar1.., sma1.., intercept, drift.
    fn names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.k);
        names.extend((1..=self.p).map(|i| format!("ar{i}")));
        names.extend((1..=self.q).map(|j| format!("ma{j}")));
        names.extend((1..=self.seasonal.p).map(|i| format!("sar{i}")));
        names.extend((1..=self.seasonal.q).map(|j| format!("sma{j}")));
        if self.mean {
            names.push("intercept".to_string());
        }
        if self.drift {
            names.push("drift".to_string());
        }
        names
    }

    /// Regressor `b` (mean, then drift) of the differenced equation at index `s` of the
    /// differenced series: 1 for the mean and for the drift with d = 1, the period for the
    /// drift with one seasonal difference, the time t = s + 1 for the drift without any.
    fn regressor(&self, b: usize, s: usize) -> f64 {
        let is_drift = self.drift && b + 1 == self.n_beta();
        if is_drift && self.diff_lags == 0 {
            (s + 1) as f64
        } else if is_drift {
            self.diff_lags as f64
        } else {
            1.0
        }
    }

    /// The parts of a coefficient vector in R order.
    fn split<'a>(&self, coef: &'a [f64]) -> Parts<'a> {
        let (ar, rest) = coef.split_at(self.p);
        let (ma, rest) = rest.split_at(self.q);
        let (sar, rest) = rest.split_at(self.seasonal.p);
        let (sma, beta) = rest.split_at(self.seasonal.q);
        Parts {
            ar,
            ma,
            sar,
            sma,
            beta,
        }
    }

    /// The expanded AR and MA polynomials of `parts` as (lag, coefficient), R's signs
    /// (`ARIMA_transPars`): phi(B) Phi(B^s) = 1 - sum c B^lag, theta(B) Theta(B^s) =
    /// 1 + sum c B^lag.
    fn expand(&self, parts: &Parts<'_>, ar: &mut Vec<(usize, f64)>, ma: &mut Vec<(usize, f64)>) {
        let expand = |low: &[f64], high: &[f64], sign: f64, out: &mut Vec<(usize, f64)>| {
            out.clear();
            out.extend(low.iter().enumerate().map(|(i, &c)| (i + 1, c)));
            for (j, &big) in high.iter().enumerate() {
                let base = (j + 1) * self.seasonal.period;
                out.push((base, big));
                out.extend(
                    low.iter()
                        .enumerate()
                        .map(|(i, &c)| (base + i + 1, sign * c * big)),
                );
            }
        };
        expand(parts.ar, parts.sar, -1.0, ar);
        expand(parts.ma, parts.sma, 1.0, ma);
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

    /// Whether the AR part is stationary and the MA part invertible: each factor has its roots
    /// outside the unit circle (a seasonal factor as a polynomial in B^s).
    fn stable(&self, parts: &Parts<'_>) -> EconResult<(bool, bool)> {
        let negated = |c: &[f64]| c.iter().map(|t| -t).collect::<Vec<f64>>();
        Ok((
            polynomial_stable(parts.ar)? && polynomial_stable(parts.sar)?,
            polynomial_stable(&negated(parts.ma))? && polynomial_stable(&negated(parts.sma))?,
        ))
    }
}

/// The series of a model differenced as R's ARIMA_CSS does, regularly and then seasonally; the
/// m = n - d - s D differenced values.
fn difference_model(x: &[f64], model: &Model) -> EconResult<Vec<f64>> {
    let w = difference(x, model.d, 1)?;
    if model.seasonal.d == 0 {
        return Ok(w);
    }
    difference(&w, model.seasonal.d, model.seasonal.period)
}

/// (1 - B^lag)^d of `x`, without the first d lag values.
fn difference(x: &[f64], d: usize, lag: usize) -> EconResult<Vec<f64>> {
    let mut w = try_vec(x.len(), 0.0, "the differenced series")?;
    w.copy_from_slice(x);
    for _ in 0..d {
        checkpoint()?;
        for l in (lag..w.len()).rev() {
            w[l] -= w[l - lag];
        }
    }
    let lost = (d * lag).min(w.len());
    w.copy_within(lost.., 0);
    w.truncate(w.len() - lost);
    Ok(w)
}

/// Buffers of one CSS pass: adjusted series, residuals, the last q + s Q derivative rows, the
/// current row, J'J and J'e.
struct Workspace {
    w: Vec<f64>,
    e: Vec<f64>,
    ring: Vec<f64>,
    row: Vec<f64>,
    jtj: Array2<f64>,
    jte: Array1<f64>,
    ar: Vec<(usize, f64)>,
    ma: Vec<(usize, f64)>,
}

impl Workspace {
    fn new(m: usize, model: &Model) -> EconResult<Workspace> {
        const WHAT: &str = "the ARIMA CSS workspace";
        Ok(Workspace {
            w: try_vec(m, 0.0, WHAT)?,
            e: try_vec(m, 0.0, WHAT)?,
            ring: try_vec(checked_len(&[model.ma_lags, model.k], WHAT)?, 0.0, WHAT)?,
            row: try_vec(model.k, 0.0, WHAT)?,
            jtj: try_zeros2(model.k, model.k, WHAT)?,
            jte: try_zeros1(model.k, WHAT)?,
            ar: Vec::new(),
            ma: Vec::new(),
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
    let parts = model.split(coef);
    let beta = parts.beta;
    let m = dx.len();
    let Workspace {
        w,
        e,
        ring,
        row,
        jtj,
        jte,
        ar,
        ma,
    } = ws;
    model.expand(&parts, ar, ma);
    let (start, ring_len, period) = (model.ar_lags, model.ma_lags, model.seasonal.period);
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
    e[..start.min(m)].fill(0.0);
    if jacobian {
        jtj.fill(0.0);
        jte.fill(0.0);
    }
    let mut ssr = 0.0;
    for s in start..m {
        if (s - start) % ROW_STRIDE == ROW_STRIDE - 1 {
            checkpoint()?;
        }
        // The MA terms reach back only to residuals after the conditioning ones.
        let lags = ring_len.min(s - start);
        let mut v = w[s];
        for &(lag, c) in ar.iter() {
            v -= c * w[s - lag];
        }
        for &(lag, c) in ma.iter() {
            if lag <= lags {
                v -= c * e[s - lag];
            }
        }
        e[s] = v;
        ssr += v * v;
        if !jacobian {
            continue;
        }
        // A coefficient's derivative is the other factor's filter at its own lag.
        let residual = |lag: usize| if lag <= lags { e[s - lag] } else { 0.0 };
        for i in 0..p {
            let mut g = w[s - i - 1];
            for (j, &c) in parts.sar.iter().enumerate() {
                g -= c * w[s - i - 1 - (j + 1) * period];
            }
            row[i] = -g;
        }
        for j in 0..q {
            let mut g = residual(j + 1);
            for (i, &c) in parts.sma.iter().enumerate() {
                g += c * residual(j + 1 + (i + 1) * period);
            }
            row[p + j] = -g;
        }
        for j in 0..parts.sar.len() {
            let base = (j + 1) * period;
            let mut g = w[s - base];
            for (i, &c) in parts.ar.iter().enumerate() {
                g -= c * w[s - base - i - 1];
            }
            row[p + q + j] = -g;
        }
        for j in 0..parts.sma.len() {
            let base = (j + 1) * period;
            let mut g = residual(base);
            for (i, &c) in parts.ma.iter().enumerate() {
                g += c * residual(base + i + 1);
            }
            row[p + q + parts.sar.len() + j] = -g;
        }
        for b in 0..beta.len() {
            let mut g = -model.regressor(b, s);
            for &(lag, c) in ar.iter() {
                g += c * model.regressor(b, s - lag);
            }
            row[model.n_arma() + b] = g;
        }
        for &(lag, c) in ma.iter() {
            if lag > lags {
                continue;
            }
            let slot = (s - lag) % ring_len;
            let prev = &ring[slot * k..(slot + 1) * k];
            for (r, &pv) in row.iter_mut().zip(prev) {
                *r -= c * pv;
            }
        }
        if ring_len > 0 {
            let slot = s % ring_len;
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
#[cfg(test)]
fn css_residuals(x: &[f64], spec: &ArimaSpec, coefficients: &[f64]) -> EconResult<(Vec<f64>, f64)> {
    let model = Model::new(spec, x.len())?;
    model.check_coefficients(coefficients)?;
    let dx = difference_model(x, &model)?;
    let mut ws = Workspace::new(dx.len(), &model)?;
    let ssr = css_pass(&dx, &model, coefficients, &mut ws, false)?;
    Ok((full_residuals(x.len(), &model, &ws.e)?, ssr))
}

/// The residuals of the differenced series at length n: d leading zeros, then `e`.
fn full_residuals(n: usize, model: &Model, e: &[f64]) -> EconResult<Vec<f64>> {
    let mut residuals = try_vec(n, 0.0, "the ARIMA residuals")?;
    residuals[model.diff_lags..].copy_from_slice(e);
    Ok(residuals)
}

/// R's start: AR and MA coefficients 0, mean and drift by least squares of the differenced
/// series on their regressors.
fn start_values(dx: &[f64], model: &Model) -> EconResult<Vec<f64>> {
    let mut coef = try_vec(model.k, 0.0, "the ARIMA coefficients")?;
    let m = dx.len() as f64;
    let at = model.n_arma();
    match (model.mean, model.drift) {
        (false, false) => {}
        (true, false) => coef[at] = dx.iter().sum::<f64>() / m,
        (false, true) if model.diff_lags > 0 => {
            coef[at] = dx.iter().sum::<f64>() / m / model.diff_lags as f64;
        }
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

/// Nielsen's gain ratio of a trial step: the decrease of the sum of squares from `ssr` to
/// `ssr_trial` over the decrease `predicted` by the linearised residuals. NaN, which rejects
/// the step, when the trial sum of squares is not finite or the predicted decrease is not
/// positive: the predicted decrease of a nonzero damped step is positive in exact
/// arithmetic, so a value at or below 0 is rounding in a nearly singular system, and its
/// sign would turn a rise of the sum of squares into a positive ratio.
fn gain_ratio(ssr: f64, ssr_trial: f64, predicted: f64) -> f64 {
    if ssr_trial.is_finite() && predicted > 0.0 {
        (ssr - ssr_trial) / predicted
    } else {
        f64::NAN
    }
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
                    gain_ratio(ssr, ssr_trial, predicted)
                }
                // A singular or non-finite damped system: the step is rejected.
                Err(_) => f64::NAN,
            };
            // Accepted only when rho > 0; NaN (a failed solve, a non-finite sum of squares,
            // a predicted decrease that is not positive) is not.
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
/// d differencing states (orders of the expanded polynomials).
struct StateSpace {
    phi: Vec<f64>,
    /// (1 - B)^d = 1 - delta_1 B - ... - delta_d B^d
    delta: Vec<f64>,
    /// (1, theta_1, ..., theta_q, 0, ...), length r
    r_vec: Vec<f64>,
    r: usize,
    rd: usize,
}

impl StateSpace {
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

/// The coefficients of (1 - B)^d (1 - B^s)^D as R's `Delta`: 1 - delta_1 B - delta_2 B^2 - ...
fn differencing_weights(model: &Model) -> EconResult<Vec<f64>> {
    let mut poly = try_vec(model.diff_lags + 1, 0.0, "the differencing polynomial")?;
    poly[0] = 1.0;
    let seasonal = std::iter::repeat_n(model.seasonal.period, model.seasonal.d);
    for lag in std::iter::repeat_n(1, model.d).chain(seasonal) {
        for i in (lag..poly.len()).rev() {
            poly[i] -= poly[i - lag];
        }
    }
    Ok(poly[1..].iter().map(|c| -c).collect())
}

/// Expanded terms as a dense vector over lags 1..=len.
fn dense(terms: &[(usize, f64)], len: usize) -> EconResult<Vec<f64>> {
    let mut out = try_vec(len, 0.0, "the expanded ARIMA polynomial")?;
    for &(lag, c) in terms {
        out[lag - 1] += c;
    }
    Ok(out)
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
    let parts = model.split(coefficients);
    let (mean, drift) = model.mean_drift(parts.beta);
    let (mut ar, mut ma) = (Vec::new(), Vec::new());
    model.expand(&parts, &mut ar, &mut ma);
    let phi = dense(&ar, model.ar_lags)?;
    let theta = dense(&ma, model.ma_lags)?;
    let r = model.ar_lags.max(model.ma_lags + 1);
    let d = model.diff_lags;
    let rd = r + d;
    let mut warnings = Vec::new();

    let mut r_vec = try_vec(r, 0.0, WHAT)?;
    r_vec[0] = 1.0;
    r_vec[1..=model.ma_lags].copy_from_slice(&theta);

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
        delta: differencing_weights(&model)?,
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
    // A step costs rd^2: large state spaces check the budget more often.
    let stride = (FILTER_STRIDE / (1 + rd * rd / 1024)).max(1);
    for (t, &xt) in x.iter().enumerate() {
        if t % stride == stride - 1 {
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
    if !model.stable(&parts)?.1 {
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
    let dx = difference_model(x, &model)?;
    let mut ws = Workspace::new(dx.len(), &model)?;
    let mut coef = start_values(&dx, &model)?;
    let convergence = optimise(&dx, &model, &mut coef, &mut ws)?;
    let ssr = css_pass(&dx, &model, &coef, &mut ws, false)?;
    if !ssr.is_finite() {
        return Err(EconError::Computation(
            "ARIMA CSS: the sum of squared residuals is not finite".to_string(),
        ));
    }
    // Residuals at the rounding level of the data (all zero for a series that is constant
    // after differencing, or after its mean and drift): the model reproduces the series, so
    // sigma2 is 0, the log-likelihood is infinite and an MA or ARMA coefficient can take any
    // value. R's arima stops there when the model has a coefficient (mean, drift, AR or MA),
    // on the singular Hessian or on an objective that is not finite at the start; for
    // ARIMA(0,d,0) without mean or drift it reports sigma^2 = 0 and an infinite
    // log-likelihood, which is refused here as well.
    let scale = dx.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if ssr <= dx.len() as f64 * (ZERO_RESIDUAL * scale).powi(2) {
        return Err(invalid(
            "the CSS residuals are zero up to rounding: the model reproduces the series \
             exactly, so its residual variance is 0 and the fit is not identified (is the \
             series constant after differencing?)"
                .to_string(),
        ));
    }
    let residuals = full_residuals(n, &model, &ws.e)?;

    let Model { p, d, q, k, .. } = model;
    let parts = model.split(&coef);
    let (mean, drift) = model.mean_drift(parts.beta);
    let sum_ar: f64 = ws.ar.iter().map(|(_, c)| c).sum();
    let constant = if model.drift && model.diff_lags == 0 {
        None
    } else if model.mean {
        Some(mean * (1.0 - sum_ar))
    } else if model.drift {
        Some(drift * model.diff_lags as f64 * (1.0 - sum_ar))
    } else {
        Some(0.0)
    };
    let n_used = (n - model.diff_lags) as f64;
    let sigma2 = ssr / (n - model.diff_lags - k) as f64;
    let loglik = -0.5
        * (n_used * (ssr / (n - model.diff_lags - model.ar_lags) as f64).ln()
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
    let (stationary, invertible) = model.stable(&parts)?;
    if !invertible {
        warnings.push(MA_NOT_INVERTIBLE.to_string());
    }
    if !stationary {
        warnings.push(AR_NOT_STATIONARY.to_string());
    }

    Ok(ArimaResult {
        column: column.to_string(),
        p,
        d,
        q,
        seasonal: model.seasonal,
        ar_coeffs: parts.ar.to_vec(),
        ma_coeffs: parts.ma.to_vec(),
        sar_coeffs: parts.sar.to_vec(),
        sma_coeffs: parts.sma.to_vec(),
        intercept: mean,
        constant,
        drift: model.drift.then_some(drift),
        include_mean: model.mean,
        coef_names: model.names(),
        coefficients: coef,
        residuals,
        n_obs: n,
        n_cond: model.diff_lags + model.ar_lags,
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
        let diff1 = difference(&series, 1, 1).unwrap();
        assert_eq!(diff1, vec![1.0, 2.0, 3.0, 4.0]);

        let diff2 = difference(&series, 2, 1).unwrap();
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
    fn test_arima_ar1_recovers_dgp() {
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
    fn test_arima_aic_is_finite() {
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
    fn test_arima_011_recovers_ma_coefficient() {
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
        // the one Rust-built fixture (the R harness cannot write a NaN): NaN is missing
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

    /// R: forecast::Arima(seasonal = list(order, period), method = "CSS") at the exact optimum and
    /// forecast::forecast there: seven seasonal models, the airline model among them.
    #[test]
    fn test_validate_arima_seasonal() {
        use p2a_validation::{Check, Tol};
        let case = crate::test_support::load_case("arima_seasonal").0;
        let load = |name: &str| {
            crate::data::DataLoader::load(&format!(
                "{}/../../validation/datasets/generated/{name}.csv",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap()
        };
        let (sim, air) = (load("arima_seasonal"), load("arima_seasonal_air"));
        let spec = |(p, d, q), (sp, sd, sq), period| ArimaSpec {
            seasonal: SeasonalOrder {
                p: sp,
                d: sd,
                q: sq,
                period,
            },
            ..ArimaSpec::new(p, d, q)
        };
        let drift = |s: ArimaSpec| ArimaSpec {
            include_drift: true,
            ..s
        };
        let fits = [
            ("air_011_011", &air, "lair", spec((0, 1, 1), (0, 1, 1), 12)),
            ("air_110_110", &air, "lair", spec((1, 1, 0), (1, 1, 0), 12)),
            ("q4_101_101", &sim, "q4", spec((1, 0, 1), (1, 0, 1), 4)),
            (
                "q0_002_001_nomean",
                &sim,
                "q0",
                ArimaSpec {
                    include_mean: Some(false),
                    ..spec((0, 0, 2), (0, 0, 1), 4)
                },
            ),
            ("q4_000_200", &sim, "q4", spec((0, 0, 0), (2, 0, 0), 4)),
            (
                "sea_100_011_drift",
                &sim,
                "sea",
                drift(spec((1, 0, 0), (0, 1, 1), 12)),
            ),
            (
                "sea_100_100_drift",
                &sim,
                "sea",
                drift(spec((1, 0, 0), (1, 0, 0), 12)),
            ),
        ];
        let mut check = Check::new(&case);
        for (key, ds, column, spec) in fits {
            let fc = forecast_arima_with(ds, column, &spec, 15).unwrap();
            let fit = fc.fit.as_ref().unwrap();
            let n = fit.n_obs;
            assert!(fit.converged, "{key}: {:?}", fit.warnings);
            assert_eq!(fit.spec().label(), spec.label());
            check
                .named(
                    &format!("{key}_coef_tight"),
                    &fit.coef_names,
                    &fit.coefficients,
                    Tol::iterative(1e-6),
                )
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
                .scalar(&format!("{key}_n_cond"), fit.n_cond as f64, Tol::exact())
                .vector(
                    &format!("{key}_resid_tail_tight"),
                    &fit.residuals[n - 3..],
                    Tol::new(1e-6, 1e-7),
                )
                // Forecasts follow the coefficients' 1e-8, so those near 0 have an absolute floor
                // like the residuals (series scale 1-10).
                .vector(
                    &format!("{key}_mean_tight"),
                    &fc.forecast,
                    Tol::new(1e-6, 1e-7),
                )
                .vector(
                    &format!("{key}_se_tight"),
                    fc.std_errors.as_ref().unwrap(),
                    Tol::stat(n),
                )
                .vector(
                    &format!("{key}_lower80_tight"),
                    &fc.lower[0],
                    Tol::new(1e-6, 1e-7),
                )
                .vector(
                    &format!("{key}_upper80_tight"),
                    &fc.upper[0],
                    Tol::new(1e-6, 1e-7),
                )
                .vector(
                    &format!("{key}_lower95_tight"),
                    &fc.lower[1],
                    Tol::new(1e-6, 1e-7),
                )
                .vector(
                    &format!("{key}_upper95_tight"),
                    &fc.upper[1],
                    Tol::new(1e-6, 1e-7),
                );
        }
        check.finish();
    }

    #[test]
    fn seasonal_specifications_are_checked() {
        let y = generate_ar1_data(200, 0.6, 7);
        let ds = Dataset::new(df! { "y" => &y }.unwrap());
        let fit = |p, d, q, period, drift| {
            let spec = ArimaSpec {
                seasonal: SeasonalOrder { p, d, q, period },
                include_drift: drift,
                ..ArimaSpec::new(0, 1, 1)
            };
            run_arima_with(&ds, "y", &spec)
        };
        let message = |r: EconResult<ArimaResult>| r.unwrap_err().to_string();
        assert!(message(fit(0, 1, 1, 0, false)).contains("period of at least 2"));
        assert!(message(fit(0, 1, 1, 100, false)).contains("Not enough observations (200)"));
        assert!(message(fit(usize::MAX, 0, 0, 12, false)).contains("orders too large"));
        assert!(message(fit(0, 1, 0, 12, true)).contains("seasonal differences included"));
        assert_eq!(
            fit(0, 1, 1, 12, false).unwrap().spec().orders(),
            "ARIMA(0,1,1)(0,1,1)[12]"
        );
        // A period without seasonal orders changes nothing.
        let plain = run_arima_with(&ds, "y", &ArimaSpec::new(0, 1, 1)).unwrap();
        let period_only = fit(0, 0, 0, 12, false).unwrap();
        assert_eq!(plain.coefficients, period_only.coefficients);
        assert_eq!(plain.spec(), period_only.spec());
        let long = generate_ar1_data(2000, 0.5, 3);
        let long = Dataset::new(df! { "y" => &long }.unwrap());
        let spec = ArimaSpec {
            seasonal: SeasonalOrder {
                p: 2,
                d: 0,
                q: 0,
                period: 365,
            },
            ..ArimaSpec::new(0, 0, 0)
        };
        let e = run_arima_with(&long, "y", &spec).unwrap_err().to_string();
        assert!(e.contains("reaches back too far"), "{e}");
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

    /// A non-stationary AR part starts the ARMA state from the diffuse prior, with the
    /// warning; the forecast is the AR recursion from the last value.
    #[test]
    fn arima_predict_warns_on_a_non_stationary_ar() {
        let ds = crate::forecasting::test_fixtures::arima_series();
        let (x, _) = numeric_column(ds.df(), "c", NaPolicy::Fail).unwrap();
        let spec = ArimaSpec::new(1, 0, 0);
        let pred = arima_predict(&x, &spec, &[1.02, 50.0], 1.0, 6).unwrap();
        assert!(
            pred.warnings.iter().any(|w| w == DIFFUSE_ARMA_PRIOR),
            "{:?}",
            pred.warnings
        );
        let last = x[x.len() - 1];
        for (h, (m, se)) in pred.mean.iter().zip(&pred.se).enumerate() {
            assert!(m.is_finite() && se.is_finite(), "h {h}: {m} {se}");
            let expected = 50.0 + 1.02f64.powi(h as i32 + 1) * (last - 50.0);
            assert!(
                (m - expected).abs() <= 1e-8 * expected.abs().max(1.0),
                "h {h}: {m} vs {expected}"
            );
        }
    }

    #[test]
    fn arima_predict_warns_on_a_non_invertible_ma() {
        let ds = crate::forecasting::test_fixtures::arima_series();
        let (x, _) = numeric_column(ds.df(), "a", NaPolicy::Fail).unwrap();
        let pred = arima_predict(&x, &ArimaSpec::new(0, 0, 1), &[1.5, 0.0], 1.0, 3).unwrap();
        assert!(
            pred.warnings.iter().any(|w| w == MA_NOT_INVERTIBLE),
            "{:?}",
            pred.warnings
        );
    }

    /// A short explosive series: the CSS optimum has |phi| > 1, which CSS does not
    /// constrain, and the fit says so. (The "did not converge" warning is reached only when
    /// Levenberg-Marquardt exhausts its fixed iteration limit, which no option exposes.)
    #[test]
    fn arima_fit_warns_on_a_non_stationary_ar() {
        let mut y = vec![1.0];
        for t in 1..30 {
            let noise = (((t * 37) % 11) as f64 - 5.0) * 0.05;
            y.push(1.15 * y[t - 1] + noise);
        }
        let ds = Dataset::new(df! { "y" => y }.unwrap());
        let spec = ArimaSpec {
            include_mean: Some(false),
            ..ArimaSpec::new(1, 0, 0)
        };
        let fit = run_arima_with(&ds, "y", &spec).unwrap();
        assert!(fit.ar_coeffs[0] > 1.0, "{:?}", fit.ar_coeffs);
        assert!(
            fit.warnings.iter().any(|w| w == AR_NOT_STATIONARY),
            "{:?}",
            fit.warnings
        );
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
        let big = Dataset::new(
            df! { "y" => (0..40).map(|i| ((i * 7) % 5) as f64).collect::<Vec<f64>>() }.unwrap(),
        );
        assert!(
            msg(run_arima_with(&big, "y", &ArimaSpec::new(20, 5, 10)))
                .contains("need at least 45 (p + d + q + 10)")
        );
        assert!(
            msg(run_arima_with(&big, "y", &ArimaSpec::new(usize::MAX, 0, 1)))
                .contains("orders too large to fit any series")
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

    /// A step is accepted only when its gain ratio is positive. A predicted decrease that is
    /// not positive (rounding in a nearly singular damped system) accepts no step, also one
    /// that raises the sum of squares, which the plain ratio would score as a gain.
    #[test]
    fn test_gain_ratio_needs_a_positive_predicted_decrease() {
        assert_eq!(gain_ratio(2.0, 1.0, 2.0), 0.5);
        assert!(gain_ratio(2.0, 3.0, 2.0) < 0.0);
        for (ssr_trial, predicted) in [
            (3.0, -1.0),
            (1.0, -1.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (f64::NAN, 1.0),
        ] {
            let gain = gain_ratio(2.0, ssr_trial, predicted);
            assert!(
                gain.is_nan() || gain <= 0.0,
                "trial {ssr_trial}, predicted {predicted}: gain {gain} accepts the step"
            );
        }
    }

    /// A series the model reproduces exactly (constant, or constant after differencing or
    /// after its mean and drift) leaves zero CSS residuals. R's arima stops on such a fit
    /// when the model has a coefficient (mean, drift, AR or MA), on the singular Hessian or
    /// on an objective that is not finite at the start; for ARIMA(0,d,0) without mean or
    /// drift (`five` with (0, 1, 0) below) it reports sigma^2 = 0 and an infinite
    /// log-likelihood. Every such fit is refused as not identified, never reported converged
    /// with an infinite log-likelihood.
    #[test]
    fn test_arima_refuses_a_fit_with_zero_residuals() {
        let trend: Vec<f64> = (0..40).map(|i| 1.0 + 0.3 * i as f64).collect();
        let ds = Dataset::new(
            df! { "five" => vec![5.0; 40], "tenth" => vec![0.1; 40], "trend" => trend }.unwrap(),
        );
        let drift = |p, d, q| ArimaSpec {
            include_drift: true,
            ..ArimaSpec::new(p, d, q)
        };
        for (column, spec) in [
            ("five", ArimaSpec::new(0, 0, 0)),
            ("five", ArimaSpec::new(0, 0, 1)),
            ("tenth", ArimaSpec::new(0, 0, 1)),
            ("tenth", ArimaSpec::new(0, 0, 2)),
            ("tenth", ArimaSpec::new(1, 0, 1)),
            ("five", ArimaSpec::new(0, 1, 0)),
            ("five", ArimaSpec::new(0, 1, 1)),
            ("trend", drift(0, 1, 1)),
            ("trend", drift(0, 0, 1)),
        ] {
            match run_arima_with(&ds, column, &spec) {
                Err(EconError::InvalidSpecification { message }) => assert!(
                    message.contains("not identified"),
                    "{column} {}: {message}",
                    spec.label()
                ),
                Err(e) => panic!("{column} {}: {e:?}", spec.label()),
                Ok(r) => panic!(
                    "{column} {}: fitted, SSR {} loglik {} converged {}",
                    spec.label(),
                    r.ssr,
                    r.loglik,
                    r.converged
                ),
            }
        }
        // a noisy series still fits
        let noisy: Vec<f64> = (0..40)
            .map(|i| 5.0 + ((i * 37 % 17) as f64 - 8.0) / 8.0)
            .collect();
        let ds = Dataset::new(df! { "y" => noisy }.unwrap());
        let r = run_arima_with(&ds, "y", &ArimaSpec::new(0, 0, 1)).unwrap();
        assert!(r.loglik.is_finite() && r.ssr > 0.0, "{r:?}");
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
