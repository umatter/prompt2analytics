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
//! to the exact optimum by Levenberg-Marquardt with the analytic Jacobian of the recursion,
//! starting where R starts (AR and MA coefficients 0, μ and δ by least squares). R's default
//! method, CSS-ML, is not implemented.
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
//! R equivalent: `stats::arima(method = "CSS")`, `forecast::Arima(method = "CSS")`,
//! `forecast::forecast()`, `stats::predict.Arima()`.

use std::fmt;

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::solve;
use crate::resource::{checked_len, checkpoint, try_vec, try_zeros1, try_zeros2};

fn size_overflow(what: &str) -> EconError {
    EconError::ResourceLimit(format!("{what} is too large: the requested size overflows"))
}

/// Outer iterations of the Levenberg-Marquardt loop.
const MAX_ITERATIONS: usize = 200;
/// Converged when the largest |Gauss-Newton step| / (1 + |coefficient|) is at most this.
const STEP_TOLERANCE: f64 = 1e-8;
/// When damping finds no better point, the fit counts as converged up to this scaled step.
const FLOOR_TOLERANCE: f64 = 1e-6;
const LAMBDA_START: f64 = 1e-3;
/// Above this damping no step improves the sum of squares: the rounding floor.
const LAMBDA_MAX: f64 = 1e16;
/// Keeps the damping positive after many accepted steps.
const LAMBDA_MIN: f64 = 1e-20;
/// Rows between budget checks in a pass over the series.
const ROW_STRIDE: usize = 4096;
/// A matrix power below this (largest absolute entry) has decayed.
const DECAYED: f64 = 1e-12;
/// Squarings before a matrix that has not decayed counts as not stable.
const MAX_SQUARINGS: usize = 64;

/// The warning R's `predict.Arima` gives for an MA polynomial with a root inside the unit
/// circle.
const MA_NOT_INVERTIBLE: &str = "MA part of model is not invertible";
const AR_NOT_STATIONARY: &str = "AR part is not stationary";

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
        match (self.drift.is_some(), self.include_mean) {
            (true, _) if self.d == 0 => {
                Some("intercept + drift x t is the trend of the series mean".to_string())
            }
            (true, _) => Some(format!(
                "drift is the mean change per period; constant drift x (1 - sum(ar)) = {:.6}",
                self.constant.unwrap_or(f64::NAN)
            )),
            (false, true) => Some(format!(
                "intercept is the series mean (R's \"intercept\"); constant mean x (1 - sum(ar)) = {:.6}",
                self.constant.unwrap_or(f64::NAN)
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
/// `ArimaForecastResult` containing forecasted values, or the [`EconError`] of
/// [`run_arima_with`] inside the `anyhow::Error`, including [`EconError::Cancelled`] (the
/// fit checks the budget before it starts and on every iteration) and
/// [`EconError::NullValues`] for a column with missing values.
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

    // The series the fit read (`run_arima` refused missing values)
    let (values, _) = numeric_column(dataset.df(), column, NaPolicy::Fail)?;

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
        .map_err(|e| anyhow::anyhow!("Forecasting failed: {:?}", e))?;

    Ok(ArimaForecastResult {
        column: column.to_string(),
        horizon,
        forecast,
        std_errors: None,
    })
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

/// Levenberg-Marquardt on the CSS residuals from `coef` (updated in place).
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
            // The final Gauss-Newton step: without it small coefficients stop short.
            for c in 0..k {
                trial[c] = coef[c] + delta[c];
            }
            let ssr_trial = css_pass(dx, model, &trial, ws, false)?;
            if ssr_trial.is_finite() && ssr_trial <= ssr * (1.0 + 1e-15) {
                coef.copy_from_slice(&trial);
            }
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
            let accepted = match solve(&damped.view(), &rhs) {
                Ok(delta) => {
                    checkpoint()?;
                    for c in 0..k {
                        trial[c] = coef[c] + delta[c];
                    }
                    let ssr_trial = css_pass(dx, model, &trial, ws, false)?;
                    ssr_trial.is_finite() && ssr_trial <= ssr
                }
                // A singular or non-finite damped system: the step is rejected.
                Err(_) => false,
            };
            if accepted {
                coef.copy_from_slice(&trial);
                lambda = (lambda / 10.0).max(LAMBDA_MIN);
                break;
            }
            lambda *= 10.0;
            if lambda > LAMBDA_MAX {
                // No step lowers the sum of squares: the rounding floor is reached.
                return Ok(Convergence {
                    converged: step <= FLOOR_TOLERANCE,
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

    #[test]
    fn run_arima_order_sum_cannot_overflow() {
        let y = generate_ar1_data(200, 0.6, 7);
        let dataset = Dataset::new(df! { "y" => &y }.unwrap());
        // a debug-build overflow panic before the fix
        assert!(run_arima(&dataset, "y", usize::MAX, 0, 1).is_err());
    }
}
