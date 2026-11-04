//! VARMA(p, q) by the Hannan-Rissanen two-step estimator ([`run_varma()`], [`run_varma_with`]).
//!
//! **Model.** VARMA(p, q) in standard form with every coefficient free:
//! `y_t = c + A_1 y_{t-1} + .. + A_p y_{t-p} + u_t + M_1 u_{t-1} + .. + M_q u_{t-q}`.
//!
//! **Estimator** (Hannan & Rissanen 1982; Luetkepohl 2005, ch. 12):
//! 1. *Step 1* (q > 0): a long VAR(m) with intercept by OLS on t = m + 1..n; its residuals
//!    `ê_t` estimate the innovations `u_t`. Only the residuals are needed, so the long VAR is
//!    solved by the (equilibrated) normal equations, with a fallback to the pivoted QR of
//!    `stats::lm.fit` when the Cholesky factor is ill-conditioned or fails.
//! 2. *Step 2*: OLS of `y_t` on `[1, y_{t-1..p}, ê_{t-1..q}]`, equation by equation, on
//!    t = m + max(p, q) + 1..n (`n_obs = n - m - max(p, q)`), by the pivoted QR of `lm.fit`
//!    (an aliased regressor is `NaN` with R's warning).
//!
//! **Likelihood.** `Sigma_u = U'U / n_obs` (ML), `logLik = -n_obs/2 (K ln 2 pi + ln det Sigma_u
//! + K)`, AIC/BIC over the identified coefficients (K x rank of the step-2 design).
//!
//! **Default long-VAR order** ([`VarmaConfig::long_var_order`] `None`): 0 when q = 0 (the fit is
//! the OLS VAR(p), equal to `vars::VAR(type = "const")`); otherwise `max(floor(ln(n)^2), 2 max(p,
//! q))` (Gómez & Maravall 2001), reduced to `floor((n - 2) / (2K + 1))` with a warning, so that
//! the long VAR keeps at least as many residual degrees of freedom as it has regressors.
//!
//! **Standard errors.** Reported only for q = 0 (an OLS VAR). For q > 0 the step-2 OLS standard
//! errors treat the estimated innovations as data: the step-1 estimation error enters the
//! limiting distribution of the AR and constant estimates (a generated-regressor problem), so
//! they are not valid for inference and are not reported (`std_errors` is empty).
//!
//! **Limitations.** No identification restrictions (echelon form), so a VARMA with common
//! factors is not detected; no third (bias-correction) step and no ML refinement.
//!
//! **References.** Hannan, E. J. & Rissanen, J. (1982), Recursive estimation of mixed
//! autoregressive-moving average order, *Biometrika* 69(1), 81-94, doi:10.1093/biomet/69.1.81.
//! Lütkepohl, H. (2005), *New Introduction to Multiple Time Series Analysis*, ch. 12. Gómez, V.
//! & Maravall, A. (2001), Automatic modeling methods for univariate series, in Peña, Tiao &
//! Tsay (eds.), *A Course in Time Series Analysis*, Wiley, 171-201.
//!
//! **R.** No VARMA package is installed (MTS, marima, dse); the estimator is written out in
//! `validation/r/cases/ts_varma.R` (case `varma_hr`); q = 0 is checked against `vars::VAR`.

use faer::Mat;
use faer::linalg::solvers::Solve;
use ndarray::{Array2, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::common::df_to_array2;
use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{cholesky, xtx};
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, aliasing_message, sym_inverse};
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::t_test_p_value;

/// Options of [`run_varma_with`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VarmaConfig {
    /// AR order p.
    pub p: usize,
    /// MA order q.
    pub q: usize,
    /// Order m of the step-1 long VAR. `None`: 0 when q = 0; otherwise max(floor(ln(n)^2),
    /// 2 max(p, q)) (Gómez & Maravall 2001), reduced to floor((n - 2) / (2K + 1)) with a
    /// warning. Must be at least 1 when q > 0. With q = 0 a given m only moves the sample
    /// start to m + p + 1, so the criteria compare with VARMA(p, q) fits at that m.
    pub long_var_order: Option<usize>,
}

impl Default for VarmaConfig {
    fn default() -> Self {
        VarmaConfig {
            p: 1,
            q: 1,
            long_var_order: None,
        }
    }
}

/// Result from a VARMA model estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarmaResult {
    /// AR lags (p)
    pub p_lags: usize,
    /// MA lags (q)
    pub q_lags: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Observations of the step-2 regression: n - long_var_order - max(p, q)
    pub n_obs: usize,
    /// AIC: -2 logLik + 2 n_params
    pub aic: f64,
    /// BIC: -2 logLik + ln(n_obs) n_params
    pub bic: f64,
    /// Per equation: the constant, then the rows of A_1..A_p (`[const, A_1 row, .., A_p row]`);
    /// NaN where aliased
    pub ar_params: Vec<Vec<f64>>,
    /// Per equation: the rows of M_1..M_q (`[M_1 row, .., M_q row]`); NaN where aliased
    pub ma_params: Vec<Vec<f64>>,
    /// Residual covariance matrix Sigma_u (ML, divisor n_obs)
    pub sigma_u: Vec<Vec<f64>>,
    /// Gaussian log-likelihood with ln det Sigma_u
    pub log_likelihood: f64,
    /// Names of the series, in equation order
    #[serde(default)]
    pub var_names: Vec<String>,
    /// The estimator that was run
    #[serde(default)]
    pub method: String,
    /// Order m of the step-1 long VAR (0 when no long VAR was fitted)
    #[serde(default)]
    pub long_var_order: usize,
    /// Step-2 regressors: "const", "<v>.l<j>" (vars' names), "e.<v>.l<j>" (lagged step-1
    /// residuals)
    #[serde(default)]
    pub regressor_names: Vec<String>,
    /// K rows of step-2 coefficients in `regressor_names` order (NaN where aliased)
    #[serde(default)]
    pub coefficients: Vec<Vec<f64>>,
    /// K rows of OLS standard errors for q = 0 (NaN where aliased); empty for q > 0, where
    /// the step-2 OLS standard errors are not valid (see the module documentation)
    #[serde(default)]
    pub std_errors: Vec<Vec<f64>>,
    /// n_obs - rank of the step-2 design (equal for every equation)
    #[serde(default)]
    pub df_residual: usize,
    /// Identified coefficients over all equations: K x rank (the AIC/BIC parameter count)
    #[serde(default)]
    pub n_params: usize,
    /// Aliased step-2 regressors (coefficients not defined because of singularities)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliased: Vec<String>,
    /// Warnings (long-VAR order reduced, aliased regressors)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl VarmaResult {
    /// Coefficient / standard error per equation (NaN where aliased); empty when no
    /// standard errors are reported (q > 0).
    pub fn t_values(&self) -> Vec<Vec<f64>> {
        self.coefficients
            .iter()
            .zip(&self.std_errors)
            .map(|(b, se)| b.iter().zip(se).map(|(b, s)| b / s).collect())
            .collect()
    }

    /// Two-sided t p-values with `df_residual` degrees of freedom (summary.lm); empty when
    /// no standard errors are reported (q > 0).
    pub fn p_values(&self) -> Vec<Vec<f64>> {
        let df = self.df_residual as f64;
        self.t_values()
            .into_iter()
            .map(|row| row.into_iter().map(|t| t_test_p_value(t, df)).collect())
            .collect()
    }

    /// Observations used as initial values: long_var_order + max(p, q).
    pub fn n_initial(&self) -> usize {
        self.long_var_order
            .saturating_add(self.p_lags.max(self.q_lags))
    }
}

impl ResultNotes for VarmaResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

fn fmt_num(x: f64, decimals: usize) -> String {
    if x.is_nan() {
        "NA".to_string()
    } else {
        format!("{x:.decimals$}")
    }
}

fn fmt_p(p: f64) -> String {
    if p.is_nan() {
        "NA".to_string()
    } else if p < 1e-4 {
        format!("{p:.2e}")
    } else {
        format!("{p:.4}")
    }
}

/// Entry `(j, c)` of a per-equation table, NaN when absent (deserialised older results).
fn cell(m: &[Vec<f64>], j: usize, c: usize) -> f64 {
    m.get(j).and_then(|r| r.get(c)).copied().unwrap_or(f64::NAN)
}

impl fmt::Display for VarmaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VARMA({}, {}) Results", self.p_lags, self.q_lags)?;
        writeln!(f, "===========================================")?;
        if !self.method.is_empty() {
            writeln!(f, "Method: {}", self.method)?;
        }
        if !self.var_names.is_empty() {
            writeln!(f, "Variables: {}", self.var_names.join(", "))?;
        }
        writeln!(f, "No. Variables: {}", self.n_vars)?;
        let initial = self.n_initial();
        writeln!(
            f,
            "Observations: {} (of {}; {} initial values)",
            self.n_obs,
            self.n_obs.saturating_add(initial),
            initial
        )?;
        writeln!(f, "Long VAR order (step 1): {}", self.long_var_order)?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "AIC: {:.4}", self.aic)?;
        writeln!(f, "BIC: {:.4}", self.bic)?;
        writeln!(
            f,
            "Parameters: {} (residual df per equation: {})",
            self.n_params, self.df_residual
        )?;
        let with_se = !self.std_errors.is_empty();
        let (t, pv) = (self.t_values(), self.p_values());
        for (j, row) in self.coefficients.iter().enumerate() {
            let eq = self
                .var_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("Var{}", j + 1));
            writeln!(f)?;
            writeln!(f, "Equation {eq}:")?;
            if with_se {
                writeln!(
                    f,
                    "  {:<14} {:>12} {:>12} {:>9} {:>10}",
                    "Regressor", "Estimate", "Std. Error", "t value", "Pr(>|t|)"
                )?;
            } else {
                writeln!(f, "  {:<14} {:>12}", "Regressor", "Estimate")?;
            }
            for (c, name) in self.regressor_names.iter().enumerate() {
                let est = fmt_num(row.get(c).copied().unwrap_or(f64::NAN), 6);
                if with_se {
                    writeln!(
                        f,
                        "  {:<14} {:>12} {:>12} {:>9} {:>10}",
                        name,
                        est,
                        fmt_num(cell(&self.std_errors, j, c), 6),
                        fmt_num(cell(&t, j, c), 3),
                        fmt_p(cell(&pv, j, c))
                    )?;
                } else {
                    writeln!(f, "  {name:<14} {est:>12}")?;
                }
            }
        }
        if !with_se && self.q_lags > 0 {
            writeln!(f)?;
            writeln!(
                f,
                "Standard errors are not reported for q > 0: the step-2 OLS standard errors \
                 treat the estimated innovations as data and are not valid for inference \
                 (see varma.md)"
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Residual Covariance Matrix (Sigma_u, ML):")?;
        for (i, row) in self.sigma_u.iter().enumerate() {
            let name = self
                .var_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("Var{}", i + 1));
            let cells: Vec<String> = row.iter().map(|v| format!("{v:.4}")).collect();
            writeln!(f, "  {name}: [{}]", cells.join(", "))?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// `1 + k * lags` regressors per equation; None on overflow.
fn regressor_count(k: usize, lags: usize) -> Option<usize> {
    k.checked_mul(lags)?.checked_add(1)
}

/// InsufficientData for a regression after `initial` leading values with `n_reg` regressors
/// per equation (None = overflow) that must keep `df_min` residual degrees of freedom: it
/// needs initial + n_reg + df_min rows (rule R4).
fn too_short(
    n: usize,
    initial: Option<usize>,
    n_reg: Option<usize>,
    df_min: usize,
    context: String,
) -> EconError {
    let required = initial
        .zip(n_reg)
        .and_then(|(a, b)| a.checked_add(b)?.checked_add(df_min))
        .unwrap_or(usize::MAX);
    EconError::InsufficientData {
        required,
        provided: n,
        context,
    }
}

fn show(v: Option<usize>) -> String {
    v.map_or_else(|| "more than usize::MAX".to_string(), |x| x.to_string())
}

/// Default order of the step-1 long VAR (see [`VarmaConfig::long_var_order`]) and the
/// warning when it was reduced.
fn default_long_var_order(
    n: usize,
    k: usize,
    p: usize,
    q: usize,
) -> EconResult<(usize, Option<String>)> {
    if q == 0 {
        return Ok((0, None));
    }
    let denom = k
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .unwrap_or(usize::MAX);
    let cap = n.saturating_sub(2) / denom;
    if cap == 0 {
        let required = denom.saturating_add(2);
        return Err(EconError::InsufficientData {
            required,
            provided: n,
            context: format!(
                "VARMA default long VAR order: a long VAR(1) in {k} variables with as many \
                 residual degrees of freedom as regressors needs {required} observations; set \
                 long_var_order"
            ),
        });
    }
    let rule = ((n as f64).ln().powi(2).floor() as usize).max(p.max(q).saturating_mul(2));
    if rule > cap {
        let warning = format!(
            "long VAR order reduced from {rule} to {cap} (default max(floor(ln(n)^2), 2 max(p, \
             q)) for n = {n}) so that the step-1 regression keeps at least as many residual \
             degrees of freedom as regressors; set long_var_order to choose it"
        );
        Ok((cap, Some(warning)))
    } else {
        Ok((rule, None))
    }
}

/// "const", "<v>.l<j>" for j = 1..p, then "e.<v>.l<j>" for j = 1..q.
fn regressor_names(vars: &[String], p: usize, q: usize) -> Vec<String> {
    let mut names = vec!["const".to_string()];
    for l in 1..=p {
        for v in vars {
            names.push(format!("{v}.l{l}"));
        }
    }
    for l in 1..=q {
        for v in vars {
            names.push(format!("e.{v}.l{l}"));
        }
    }
    names
}

/// Rows t = start..n of [1, y_{t-1..p}, e_{t-1..q}] (the caller checked start < n, that
/// `cols` = 1 + k (p + q) does not overflow and that start >= max(p, q)).
fn lag_design(
    y: ArrayView2<f64>,
    e: ArrayView2<f64>,
    start: usize,
    p: usize,
    q: usize,
    cols: usize,
) -> EconResult<Array2<f64>> {
    let (n, k) = y.dim();
    let mut x = resource::try_zeros2(n - start, cols, "VARMA design matrix")?;
    for (i, t) in (start..n).enumerate() {
        x[[i, 0]] = 1.0;
        for l in 1..=p {
            for v in 0..k {
                x[[i, 1 + (l - 1) * k + v]] = y[[t - l, v]];
            }
        }
        for l in 1..=q {
            for v in 0..k {
                x[[i, 1 + (p + l - 1) * k + v]] = e[[t - l, v]];
            }
        }
    }
    Ok(x)
}

/// Largest ratio of the Cholesky diagonal of the equilibrated long-VAR Z'Z accepted by the
/// normal-equation path (about a condition number of 1e8); above it the residuals come from
/// the pivoted QR.
const LONG_VAR_CHOL_RATIO: f64 = 1e4;

/// Residuals of the step-1 long VAR(m) on rows m..n (`(n - m) x k`). Only residuals are
/// needed, so the equilibrated normal equations are solved by Cholesky; an ill-conditioned or
/// failed factorisation falls back to the pivoted QR of `lm.fit` (with R's aliasing warning).
/// The caller checked that n - m > 1 + k m.
fn long_var_residuals(
    y: ArrayView2<f64>,
    m: usize,
    var_names: &[String],
    warnings: &mut Vec<String>,
) -> EconResult<Array2<f64>> {
    let c = 1 + y.ncols() * m; // checked by the caller
    let z = lag_design(y, y, m, m, 0, c)?;
    let target = y.slice(s![m.., ..]);
    if let Some(b) = normal_equation_coefficients(z.view(), target)? {
        return Ok(&target - &z.dot(&b));
    }
    // Ill-conditioned or singular: lm.fit's pivoted QR (aliased lags get zero weight).
    let (b, aliased) = qr_coefficients(z.view(), target)?;
    if let Some(w) = aliasing_message(&aliased, &regressor_names(var_names, m, 0)) {
        warnings.push(format!("long VAR (step 1): {w}"));
    }
    Ok(&target - &z.dot(&b))
}

/// OLS coefficients (`c x k`) of every column of `target` on `z` by Cholesky of the
/// diagonally equilibrated Z'Z, or `None` when a column of `z` is zero, the factorisation
/// fails, or its diagonal ratio exceeds [`LONG_VAR_CHOL_RATIO`].
fn normal_equation_coefficients(
    z: ArrayView2<f64>,
    target: ArrayView2<f64>,
) -> EconResult<Option<Array2<f64>>> {
    let (c, k) = (z.ncols(), target.ncols());
    resource::check_alloc::<f64>(
        resource::checked_len(&[c, c], "VARMA long VAR Z'Z")?,
        "VARMA long VAR Z'Z",
    )?;
    resource::checkpoint()?;
    let g = xtx(&z);
    let d: Vec<f64> = (0..c).map(|j| 1.0 / g[[j, j]].sqrt()).collect();
    if !d.iter().all(|v| v.is_finite()) {
        return Ok(None);
    }
    let geq = Mat::<f64>::from_fn(c, c, |i, j| g[[i, j]] * d[i] * d[j]);
    let Ok(chol) = geq.llt(faer::Side::Lower) else {
        return Ok(None);
    };
    let l = chol.L();
    let (lo, hi) = (0..c).fold((f64::INFINITY, 0.0_f64), |(lo, hi), i| {
        let v = l[(i, i)];
        (lo.min(v), hi.max(v))
    });
    if !(lo > 0.0 && hi / lo <= LONG_VAR_CHOL_RATIO) {
        return Ok(None);
    }
    resource::checkpoint()?;
    let zy = z.t().dot(&target);
    let rhs = Mat::<f64>::from_fn(c, k, |i, j| zy[[i, j]] * d[i]);
    let sol = chol.solve(&rhs);
    Ok(Some(Array2::from_shape_fn((c, k), |(i, j)| {
        sol[(i, j)] * d[i]
    })))
}

/// OLS coefficients (`c x k`, 0 for aliased columns) of every column of `target` on `z` by
/// one pivoted QR (`lm.fit`), and the aliased columns.
fn qr_coefficients(
    z: ArrayView2<f64>,
    target: ArrayView2<f64>,
) -> EconResult<(Array2<f64>, Vec<usize>)> {
    let qr = Qr::decompose(z, LM_QR_TOL)?;
    let mut b = resource::try_zeros2(z.ncols(), target.ncols(), "VARMA long VAR coefficients")?;
    for j in 0..target.ncols() {
        resource::checkpoint()?;
        b.column_mut(j)
            .assign(&qr.coefficients_zeroed(target.column(j))?);
    }
    Ok((b, qr.aliased()))
}

/// Equation-by-equation OLS of every column of y on x with one pivoted QR (stats::lm.fit).
struct OlsSystem {
    /// n_reg x k, NaN where aliased
    coefficients: Array2<f64>,
    /// rows x k
    residuals: Array2<f64>,
    rank: usize,
    aliased: Vec<usize>,
    /// n_reg x n_reg, NaN rows/columns where aliased (only when requested)
    unscaled_cov: Option<Array2<f64>>,
}

impl OlsSystem {
    fn fit(x: ArrayView2<f64>, y: ArrayView2<f64>, want_cov: bool) -> EconResult<Self> {
        let n_reg = x.ncols();
        let k = y.ncols();
        let qr = Qr::decompose(x, LM_QR_TOL)?;
        let mut coefficients = resource::try_zeros2(n_reg, k, "VARMA coefficients")?;
        let mut zeroed = resource::try_zeros2(n_reg, k, "VARMA coefficients")?;
        for j in 0..k {
            resource::checkpoint()?;
            let b = qr.coefficients(y.column(j))?;
            zeroed
                .column_mut(j)
                .assign(&b.mapv(|v| if v.is_nan() { 0.0 } else { v }));
            coefficients.column_mut(j).assign(&b);
        }
        // U = Y - X B in one matrix product
        let residuals = &y - &x.dot(&zeroed);
        let unscaled_cov = if want_cov {
            Some(qr.unscaled_covariance()?)
        } else {
            None
        };
        Ok(Self {
            coefficients,
            residuals,
            rank: qr.rank(),
            aliased: qr.aliased(),
            unscaled_cov,
        })
    }
}

/// Run a VARMA(p, q) model by the Hannan-Rissanen two-step estimator with the default
/// long-VAR order (see [`run_varma_with`] and [`VarmaConfig::long_var_order`]).
///
/// # Arguments
/// * `dataset` - The dataset containing the time series data
/// * `columns` - Names of the columns to include in the VARMA
/// * `p` - Number of AR lags
/// * `q` - Number of MA lags
pub fn run_varma(
    dataset: &Dataset,
    columns: &[&str],
    p: usize,
    q: usize,
) -> EconResult<VarmaResult> {
    let mut config = VarmaConfig::default();
    config.p = p;
    config.q = q;
    run_varma_with(dataset, columns, &config)
}

/// Run a VARMA(p, q) model by the Hannan-Rissanen two-step estimator.
///
/// Step 1 (q > 0) fits a long VAR(m) with intercept by OLS; its residuals estimate the
/// innovations. Step 2 regresses `y_t` on `[1, y_{t-1..p}, ê_{t-1..q}]` equation by equation
/// on t = m + max(p, q) + 1..n. VARMA(p, 0) is the OLS VAR(p) (with the default m = 0 it
/// equals `vars::VAR(type = "const")`); VARMA(0, q) is a vector MA model. Standard errors, t
/// and p values are reported for q = 0 only (see the module documentation).
///
/// The default long-VAR order is max(floor(ln(n)^2), 2 max(p, q)) for q > 0, reduced to
/// floor((n - 2) / (2K + 1)) with a warning.
///
/// # Errors
/// * `InvalidSpecification`: p = q = 0, or `long_var_order = Some(0)` with q > 0.
/// * `InsufficientData`: the long VAR needs more rows than regressors after its m initial
///   values; the step-2 regression needs at least K residual degrees of freedom (the K x K
///   Sigma_u), i.e. n >= m + max(p, q) + 1 + K (p + q) + K.
/// * `SingularMatrix`: Sigma_u is singular (linearly dependent series).
/// * `NullValues` / `NonNumericColumn`: missing or non-numeric values (as `vars::VAR`).
///
/// ```
/// use p2a_core::Dataset;
/// use p2a_core::econometrics::timeseries::varma::{VarmaConfig, run_varma_with};
/// use polars::prelude::*;
///
/// let n = 120;
/// let (mut y1, mut y2) = (vec![0.0; n], vec![0.0; n]);
/// let mut prev = (0.0, 0.0);
/// for t in 1..n {
///     let e = (((t * 7919) % 101) as f64 / 101.0 - 0.5, ((t * 104_729) % 97) as f64 / 97.0 - 0.5);
///     y1[t] = 0.5 * y1[t - 1] + e.0 + 0.3 * prev.0;
///     y2[t] = 0.1 * y1[t - 1] + 0.4 * y2[t - 1] + e.1 + 0.2 * prev.1;
///     prev = e;
/// }
/// let ds = Dataset::new(df!("y1" => y1, "y2" => y2).unwrap());
/// let mut config = VarmaConfig::default(); // VARMA(1, 1)
/// config.long_var_order = Some(6);
/// let fit = run_varma_with(&ds, &["y1", "y2"], &config).unwrap();
/// assert_eq!(fit.n_obs, n - 6 - 1);
/// assert_eq!(fit.regressor_names, ["const", "y1.l1", "y2.l1", "e.y1.l1", "e.y2.l1"]);
/// assert!(fit.std_errors.is_empty()); // not reported for q > 0
/// ```
pub fn run_varma_with(
    dataset: &Dataset,
    columns: &[&str],
    config: &VarmaConfig,
) -> EconResult<VarmaResult> {
    let (p, q) = (config.p, config.q);
    if p == 0 && q == 0 {
        return Err(EconError::InvalidSpecification {
            message: "VARMA requires p >= 1 or q >= 1 (p = q = 0 is a constant-mean model)"
                .to_string(),
        });
    }
    if q > 0 && config.long_var_order == Some(0) {
        return Err(EconError::InvalidSpecification {
            message: "long_var_order must be at least 1 when q > 0: the long VAR supplies the \
                      innovation estimates (got 0)"
                .to_string(),
        });
    }
    let (data, var_names) = df_to_array2(dataset.df(), columns)?;
    let (n, k) = data.dim();
    let mut warnings = Vec::new();
    let m = match config.long_var_order {
        Some(m) => m,
        None => {
            let (m, w) = default_long_var_order(n, k, p, q)?;
            warnings.extend(w);
            m
        }
    };
    let s = p.max(q);
    // Size checks before anything is fitted (ECON-79, X-7; rule R4).
    if q > 0 {
        let reg1 = regressor_count(k, m);
        if !matches!((reg1, n.checked_sub(m)), (Some(r), Some(rows)) if rows > r) {
            return Err(too_short(
                n,
                Some(m),
                reg1,
                1,
                format!(
                    "VARMA long VAR({m}) (step 1): {} regressors per equation after {m} initial \
                     values need more rows than regressors",
                    show(reg1)
                ),
            ));
        }
    }
    let n_reg = p.checked_add(q).and_then(|l| regressor_count(k, l));
    let start = m.checked_add(s);
    let rows = start.and_then(|st| n.checked_sub(st));
    let (n_reg, start, rows) = match (n_reg, start, rows) {
        (Some(r), Some(st), Some(rows)) if rows.checked_sub(r).is_some_and(|d| d >= k) => {
            (r, st, rows)
        }
        _ => {
            return Err(too_short(
                n,
                start,
                n_reg,
                k,
                format!(
                    "VARMA({p}, {q}) step-2 regression: {} regressors per equation after {} \
                     initial values (long VAR order {m} + max(p, q) = {s}), and the {k} x {k} \
                     residual covariance needs at least K = {k} residual degrees of freedom",
                    show(n_reg),
                    show(start)
                ),
            ));
        }
    };
    // Step 1: long VAR(m) residuals; rows before m stay 0 and are never read (t - l >= m in
    // step 2, because t >= m + max(p, q) and l <= q).
    let mut e = resource::try_zeros2(n, k, "VARMA innovation estimates")?;
    if q > 0 {
        let u1 = long_var_residuals(data.view(), m, &var_names, &mut warnings)?;
        e.slice_mut(s![m.., ..]).assign(&u1);
    }
    // Step 2.
    let names = regressor_names(&var_names, p, q);
    let x = lag_design(data.view(), e.view(), start, p, q, n_reg)?;
    let fit = OlsSystem::fit(x.view(), data.slice(s![start.., ..]), q == 0)?;
    if let Some(w) = aliasing_message(&fit.aliased, &names) {
        warnings.push(w);
    }
    // At least K by the size check; kept for safety.
    let df_residual = rows
        .checked_sub(fit.rank)
        .filter(|&d| d >= k)
        .ok_or_else(|| {
            too_short(
                n,
                Some(start),
                Some(n_reg),
                k,
                format!(
                    "VARMA({p}, {q}) step-2 regression has fewer than K residual degrees of freedom"
                ),
            )
        })?;
    // std_errors (k x n_reg) and sigma (k x k) are smaller than the step-2 design that
    // try_zeros2 already allocated (rows > n_reg >= k), so plain allocations are safe here.
    let cross = fit.residuals.t().dot(&fit.residuals);
    let std_errors: Vec<Vec<f64>> = match &fit.unscaled_cov {
        Some(unscaled) => (0..k)
            .map(|j| {
                let s2 = cross[[j, j]] / df_residual as f64;
                (0..n_reg)
                    .map(|c| (unscaled[[c, c]] * s2).sqrt()) // NaN for aliased columns
                    .collect()
            })
            .collect(),
        None => Vec::new(),
    };
    let sigma = &cross / rows as f64;
    // A rank-deficient Sigma_u (linearly dependent series) has no Gaussian likelihood (R9).
    sym_inverse(
        sigma.view(),
        SingularPolicy::Error,
        "VARMA residual covariance Sigma_u (are the series linearly dependent?)",
    )?;
    let l = cholesky(&sigma.view()).map_err(|err| EconError::SingularMatrix {
        context: "VARMA residual covariance Sigma_u".to_string(),
        suggestion: format!(
            "Cholesky factorisation failed ({err:?}); check for linearly dependent series"
        ),
    })?;
    let log_det = 2.0 * (0..k).map(|i| l[[i, i]].ln()).sum::<f64>();
    let (tf, kf) = (rows as f64, k as f64);
    let log_likelihood = -0.5 * tf * (kf * (2.0 * std::f64::consts::PI).ln() + log_det + kf);
    let n_params = k * fit.rank;
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + tf.ln() * n_params as f64;
    let coefficients: Vec<Vec<f64>> = (0..k)
        .map(|j| fit.coefficients.column(j).to_vec())
        .collect();
    let n_ar = 1 + k * p;
    let method = if q > 0 {
        format!(
            "Hannan-Rissanen two-step: long VAR({m}) by OLS, then OLS on lagged y and lagged \
             long-VAR residuals"
        )
    } else if m == 0 {
        format!("OLS VAR({p}) (q = 0: no long VAR needed)")
    } else {
        format!(
            "OLS VAR({p}) on observations {}..{n} (q = 0; sample aligned with long VAR order {m})",
            start + 1
        )
    };
    Ok(VarmaResult {
        p_lags: p,
        q_lags: q,
        n_vars: k,
        n_obs: rows,
        aic,
        bic,
        ar_params: coefficients.iter().map(|r| r[..n_ar].to_vec()).collect(),
        ma_params: coefficients.iter().map(|r| r[n_ar..].to_vec()).collect(),
        sigma_u: (0..k).map(|i| sigma.row(i).to_vec()).collect(),
        log_likelihood,
        aliased: fit.aliased.iter().map(|&c| names[c].clone()).collect(),
        var_names,
        method,
        long_var_order: m,
        regressor_names: names,
        coefficients,
        std_errors,
        df_residual,
        n_params,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::EconError;
    use polars::prelude::*;

    const COLS: [&str; 2] = ["y1", "y2"];

    /// Column-major flattening of a k x k matrix (R's vec()).
    fn col_major(m: &[Vec<f64>]) -> Vec<f64> {
        let k = m.len();
        (0..k).flat_map(|c| (0..k).map(move |i| m[i][c])).collect()
    }

    /// Ten rows of k deterministic, non-collinear noisy series named a, b, c.
    fn ten_rows(k: usize) -> Dataset {
        let s = |f: fn(f64) -> f64, w: f64, m: usize, sc: f64| -> Vec<f64> {
            (0..10)
                .map(|t| f(w * t as f64) + sc * ((t * m) % 11) as f64 / 11.0)
                .collect()
        };
        let cols = [
            s(f64::sin, 0.9, 37, 0.3),
            s(f64::cos, 1.7, 53, 0.25),
            s(f64::sin, 2.3, 71, 0.2),
        ];
        let series: Vec<Column> = ["a", "b", "c"]
            .iter()
            .zip(cols)
            .take(k)
            .map(|(n, v)| Column::new((*n).into(), v))
            .collect();
        Dataset::new(DataFrame::new(series).unwrap())
    }

    fn expect_insufficient(
        r: EconResult<VarmaResult>,
        required: usize,
        provided: usize,
        needle: &str,
    ) {
        match r {
            Err(EconError::InsufficientData {
                required: req,
                provided: prov,
                context,
            }) => {
                assert_eq!((req, prov), (required, provided), "{context}");
                assert!(
                    context.contains("VARMA") && context.contains(needle),
                    "{context}"
                );
            }
            other => panic!("expected InsufficientData ({needle}), got {other:?}"),
        }
    }

    /// R (varma_hr): HR VARMA(1, 1) at the default long-VAR order m = 35, n_eff = 364. Fails before:
    /// run_varma fitted a VAR(1) on t = 2..n with zero MA coefficients (y1.l1 0.6142 instead of
    /// 0.4737, e.y1.l1 0 instead of 0.3045, n 399 instead of 364, logLik from the product of the
    /// |Sigma| entries).
    #[test]
    fn test_validate_varma_hr_default_order() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("varma_hr");
        let r = run_varma(&ds, &COLS, 1, 1).unwrap();
        let n = 364;
        let ar = ["const", "y1.l1", "y2.l1"];
        let ma = ["e.y1.l1", "e.y2.l1"];
        let mut check = Check::new(&case);
        for (j, eq) in COLS.iter().enumerate() {
            let key = format!("default_coef_{eq}");
            check
                .named_subset(&key, &ar, &ar, &r.ar_params[j], Tol::coef(n))
                .named_subset(&key, &ma, &ma, &r.ma_params[j], Tol::coef(n));
        }
        check
            .vector("default_sigma_ml", &col_major(&r.sigma_u), Tol::stat(n))
            .scalar("default_loglik", r.log_likelihood, Tol::stat(n))
            .scalar("default_aic", r.aic, Tol::stat(n))
            .scalar("default_bic", r.bic, Tol::stat(n))
            .scalar("default_n_eff", r.n_obs as f64, Tol::exact())
            .finish();
    }

    /// R: vars::VAR(varma_sim, p = 2, type = "const"); VARMA(2, 0) at the default order is that VAR.
    /// Fails before: the coefficients agree (same OLS sample) but logLik -592.0442052 (product of the
    /// |Sigma| entries) instead of -1065.4496896401483, and AIC/BIC with it.
    #[test]
    fn test_validate_varma_q0_matches_vars_var() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("varma_hr");
        let r = run_varma(&ds, &COLS, 2, 0).unwrap();
        let n = 398;
        let names = ["const", "y1.l1", "y2.l1", "y1.l2", "y2.l2"];
        let mut check = Check::new(&case);
        for (j, eq) in COLS.iter().enumerate() {
            check.named(
                &format!("var_p2_coef_{eq}"),
                &names,
                &r.ar_params[j],
                Tol::coef(n),
            );
            assert!(r.ma_params[j].is_empty());
        }
        check
            .scalar("var_p2_loglik", r.log_likelihood, Tol::stat(n))
            .scalar("var_p2_aic", r.aic, Tol::stat(n))
            .scalar("var_p2_bic", r.bic, Tol::stat(n))
            .scalar("var_p2_n_eff", r.n_obs as f64, Tol::exact())
            .finish();
    }

    /// Phantom option (design 3.2): q changes the estimates. Fails before: VARMA(1, 1) was VARMA(1, 0)'s
    /// VAR(1) with zero MA coefficients, and VARMA(0, 1) was intercept-only.
    #[test]
    fn test_validate_varma_ma_terms_change_fit() {
        let (_case, ds) = crate::test_support::load_case("varma_hr");
        let r11 = run_varma(&ds, &COLS, 1, 1).unwrap();
        let r10 = run_varma(&ds, &COLS, 1, 0).unwrap();
        let r01 = run_varma(&ds, &COLS, 0, 1).unwrap();
        let max_abs = |m: &[Vec<f64>]| m.iter().flatten().fold(0.0_f64, |a, x| a.max(x.abs()));
        // R: (1,1) e.y1.l1 = 0.3044730844267806; (0,1) e.y1.l1 = 0.776823637785237 (both at m = 35)
        assert!(max_abs(&r11.ma_params) > 0.25, "{:?}", r11.ma_params);
        assert!(max_abs(&r01.ma_params) > 0.7, "{:?}", r01.ma_params);
        // y1.l1 in equation y1: 0.47369331597875647 (1,1) vs 0.61423754050749235 (1,0) = vars::VAR(p = 1)
        assert!((r11.ar_params[0][1] - r10.ar_params[0][1]).abs() > 0.1);
        assert_eq!((r11.n_obs, r10.n_obs, r01.n_obs), (364, 399, 364));
    }

    /// ECON-79 / X-7 (VARMA sites): the step-2 regression needs at least K residual degrees of
    /// freedom (the K x K Sigma_u) and the default long VAR must fit. Fails before: p >= T gave
    /// InsufficientTimePeriods (a panic before T0-WP13), and p = 9 on 10 rows or K = 3, p = 3
    /// returned Ok with a perfect fit (Sigma ~ 1e-31).
    #[test]
    fn test_varma_rejects_p_ge_t() {
        let ds2 = ten_rows(2);
        let ds3 = ten_rows(3);
        let ab = ["a", "b"];
        // required = initial values + regressors per equation + K
        expect_insufficient(run_varma(&ds2, &ab, 20, 0), 63, 10, "step-2"); // 20 + 41 + 2
        expect_insufficient(run_varma(&ds2, &ab, 10, 0), 33, 10, "step-2"); // 10 + 21 + 2
        expect_insufficient(run_varma(&ds2, &ab, 9, 0), 30, 10, "step-2"); // one row, 19 regressors
        expect_insufficient(run_varma(&ds3, &["a", "b", "c"], 3, 0), 16, 10, "step-2"); // 7 rows, 10 regressors
        // q > T: default m = 1 (reduced from 40), start 21 > 10 rows
        expect_insufficient(run_varma(&ds2, &ab, 1, 20), 66, 10, "step-2"); // 21 + 43 + 2
        expect_insufficient(
            run_varma(&ds2, &ab, usize::MAX, 0),
            usize::MAX,
            10,
            "step-2",
        );
        expect_insufficient(
            run_varma(&ds2, &ab, 1, usize::MAX),
            usize::MAX,
            10,
            "step-2",
        );
        // the default long VAR needs n >= 2K + 3 = 7 rows
        let ds6 = Dataset::new(ds2.df().slice(0, 6));
        expect_insufficient(run_varma(&ds6, &ab, 1, 1), 7, 6, "long VAR");
    }

    /// Linearly dependent series make Sigma_u singular: an error, not a likelihood. Fails before:
    /// Ok (pseudo-inverse of the singular Z'Z, logLik from the product of the |Sigma| entries).
    #[test]
    fn test_validate_varma_linearly_dependent_series_is_error() {
        let (_case, ds) = crate::test_support::load_case("varma_hr");
        let y1 = ds
            .df()
            .column("y1")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<f64>>();
        let z: Vec<f64> = y1.iter().map(|v| 2.0 * v).collect();
        let dep = Dataset::new(df!("y1" => y1, "z" => z).unwrap());
        match run_varma(&dep, &["y1", "z"], 1, 1) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(
                    context.contains("Sigma_u") && context.contains("rank 1 of 2"),
                    "{context}"
                );
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
    }

    /// Display prints the estimator, the long-VAR order and one coefficient table per equation;
    /// for q > 0 without standard errors (they are not valid there), for q = 0 with them.
    /// Fails before: only the logLik/AIC/BIC and the parameter shapes were printed.
    #[test]
    fn test_validate_varma_display_prints_coefficient_table() {
        let (case, ds) = crate::test_support::load_case("varma_hr");
        let s = run_varma(&ds, &COLS, 1, 1).unwrap().to_string();
        for needle in [
            "Method: Hannan-Rissanen two-step",
            "Long VAR order (step 1): 35",
            "Observations: 364 (of 400; 36 initial values)",
            "Equation y1:",
            "Equation y2:",
            "e.y2.l1",
            "Standard errors are not reported for q > 0",
        ] {
            assert!(s.contains(needle), "missing {needle:?} in\n{s}");
        }
        assert!(!s.contains("Pr(>|t|)"), "{s}");
        let e11 = case.named("default_coef_y1", "e.y1.l1"); // 0.3044730844267806
        assert!(s.contains(&format!("{e11:.6}")), "{s}");
        assert!(
            s.contains(&format!(
                "Log-Likelihood: {:.4}",
                case.scalar("default_loglik")
            )),
            "{s}"
        ); // -976.9647
        let short = Dataset::new(ds.df().slice(0, 60));
        let s60 = run_varma(&short, &COLS, 1, 1).unwrap().to_string();
        assert!(
            s60.contains("Warning: long VAR order reduced from 16 to 11"),
            "{s60}"
        );
        let s20 = run_varma(&ds, &COLS, 2, 0).unwrap().to_string();
        for needle in ["Equation y1:", "y2.l2", "Std. Error", "Pr(>|t|)"] {
            assert!(s20.contains(needle), "missing {needle:?} in\n{s20}");
        }
        assert!(!s20.contains("Standard errors are not reported"), "{s20}");
    }

    fn fit(ds: &Dataset, cols: &[&str], p: usize, q: usize, m: Option<usize>) -> VarmaResult {
        let mut config = VarmaConfig::default();
        config.p = p;
        config.q = q;
        config.long_var_order = m;
        run_varma_with(ds, cols, &config).unwrap()
    }

    /// Pins `<pre>coef_<eq>`, `<pre>sigma_ml`, `<pre>loglik`, `<pre>aic`, `<pre>bic`,
    /// `<pre>n_eff`, `<pre>df_residual` and `<pre>n_params` of one fit, and for q = 0 also
    /// `<pre>se_<eq>` and `<pre>p_<eq>` (for q > 0 no standard errors are reported).
    fn check_fit(check: &mut p2a_validation::Check<'_>, pre: &str, r: &VarmaResult) {
        use p2a_validation::Tol;
        let n = r.n_obs;
        let pv = r.p_values();
        for (j, eq) in r.var_names.iter().enumerate() {
            check.named(
                &format!("{pre}coef_{eq}"),
                &r.regressor_names,
                &r.coefficients[j],
                Tol::coef(n),
            );
            if r.q_lags == 0 {
                check
                    .named(
                        &format!("{pre}se_{eq}"),
                        &r.regressor_names,
                        &r.std_errors[j],
                        Tol::se(n),
                    )
                    .named(
                        &format!("{pre}p_{eq}"),
                        &r.regressor_names,
                        &pv[j],
                        Tol::p_value(n),
                    );
            }
        }
        if r.q_lags > 0 {
            assert!(r.std_errors.is_empty() && pv.is_empty(), "{pre}");
        }
        check
            .vector(
                &format!("{pre}sigma_ml"),
                &col_major(&r.sigma_u),
                Tol::stat(n),
            )
            .scalar(&format!("{pre}loglik"), r.log_likelihood, Tol::stat(n))
            .scalar(&format!("{pre}aic"), r.aic, Tol::stat(n))
            .scalar(&format!("{pre}bic"), r.bic, Tol::stat(n))
            .scalar(&format!("{pre}n_eff"), r.n_obs as f64, Tol::exact())
            .scalar(
                &format!("{pre}df_residual"),
                r.df_residual as f64,
                Tol::exact(),
            )
            .scalar(&format!("{pre}n_params"), r.n_params as f64, Tol::exact());
    }

    /// R (varma_hr, Hannan-Rissanen written out; MTS/marima not installed): VARMA(1,1) with a
    /// long VAR(8): step-2 coefficients, Sigma_ML, logLik -1046.1905002990459, AIC, BIC, n_eff
    /// 391; VARMA(1,0) (with SEs and p-values), (0,1), (2,1) at m = 8 and (2,2) at m = 10; the
    /// default order (35) and its reduction at n = 60 (16 -> 11); q = 0 against vars::VAR(p = 2)
    /// including SEs.
    #[test]
    fn test_validate_varma_hr_two_step() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("varma_hr");
        let r11 = fit(&ds, &COLS, 1, 1, Some(8));
        let flat = |m: &[Vec<f64>]| m.iter().flatten().copied().collect::<Vec<f64>>();
        let mut check = Check::new(&case);
        // design key: R's vec() of the 5 x 2 coefficient matrix (equation y1, then y2; rows
        // const, y1.l1, y2.l1, e.y1.l1, e.y2.l1)
        check
            .scalar("long_var_order", r11.long_var_order as f64, Tol::exact())
            .vector("step2_coef", &flat(&r11.coefficients), Tol::coef(391));
        assert_eq!(
            r11.regressor_names,
            ["const", "y1.l1", "y2.l1", "e.y1.l1", "e.y2.l1"]
        );
        check_fit(&mut check, "", &r11);
        check_fit(&mut check, "p1q0_lv8_", &fit(&ds, &COLS, 1, 0, Some(8)));
        check_fit(&mut check, "p0q1_lv8_", &fit(&ds, &COLS, 0, 1, Some(8)));
        check_fit(&mut check, "p2q1_lv8_", &fit(&ds, &COLS, 2, 1, Some(8)));
        check_fit(&mut check, "p2q2_lv10_", &fit(&ds, &COLS, 2, 2, Some(10)));
        let rdef = fit(&ds, &COLS, 1, 1, None);
        check.scalar(
            "default_long_var_order",
            rdef.long_var_order as f64,
            Tol::exact(),
        );
        check_fit(&mut check, "default_", &rdef);
        assert!(rdef.warnings.is_empty(), "{:?}", rdef.warnings);
        let short = Dataset::new(ds.df().slice(0, 60));
        let rs = fit(&short, &COLS, 1, 1, None);
        check.scalar(
            "short60_long_var_order",
            rs.long_var_order as f64,
            Tol::exact(),
        );
        check_fit(&mut check, "short60_", &rs);
        let rv = fit(&ds, &COLS, 2, 0, None);
        assert_eq!(rv.long_var_order, 0);
        for (j, eq) in COLS.iter().enumerate() {
            check
                .named(
                    &format!("var_p2_coef_{eq}"),
                    &rv.regressor_names,
                    &rv.coefficients[j],
                    Tol::coef(398),
                )
                .named(
                    &format!("var_p2_se_{eq}"),
                    &rv.regressor_names,
                    &rv.std_errors[j],
                    Tol::se(398),
                );
        }
        check
            .scalar("var_p2_loglik", rv.log_likelihood, Tol::stat(398))
            .scalar("var_p2_aic", rv.aic, Tol::stat(398))
            .scalar("var_p2_bic", rv.bic, Tol::stat(398))
            .scalar("var_p2_n_eff", rv.n_obs as f64, Tol::exact())
            .finish();
    }

    /// R9: an aliased step-2 regressor gets NaN (R: NA) and R's warning; SEs and the parameter
    /// count use the identified columns (R: lm on [1, y1.l1, ya.l1] with ya.l1 constant).
    #[test]
    fn test_validate_varma_aliased_regressor_is_nan() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("varma_hr");
        let y1 = ds
            .df()
            .column("y1")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<f64>>();
        let n = y1.len();
        let ya: Vec<f64> = (0..n).map(|i| if i + 1 == n { 2.0 } else { 1.0 }).collect();
        let alias = Dataset::new(df!("y1" => y1, "ya" => ya).unwrap());
        let r = fit(&alias, &["y1", "ya"], 1, 0, None);
        let m = r.n_obs; // 399
        let mut check = Check::new(&case);
        for (j, eq) in ["y1", "ya"].iter().enumerate() {
            check
                .named(
                    &format!("alias_coef_{eq}"),
                    &r.regressor_names,
                    &r.coefficients[j],
                    Tol::coef(m),
                )
                .named(
                    &format!("alias_se_{eq}"),
                    &r.regressor_names,
                    &r.std_errors[j],
                    Tol::se(m),
                );
        }
        check
            .scalar("alias_loglik", r.log_likelihood, Tol::stat(m))
            .scalar("alias_aic", r.aic, Tol::stat(m))
            .scalar("alias_n_params", r.n_params as f64, Tol::exact())
            .scalar("alias_df_residual", r.df_residual as f64, Tol::exact())
            .finish();
        assert_eq!(r.aliased, ["ya.l1"]);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("singularities") && w.contains("ya.l1")),
            "{:?}",
            r.warnings
        );
        assert!(r.ar_params[0][2].is_nan());
    }

    /// The default long-VAR order rule, pinned to the case: 35 at n = 400 (floor(ln(400)^2)),
    /// 16 reduced to 11 = floor(58 / 5) at n = 60 with a warning, 0 for q = 0, 2 max(p, q)
    /// when larger, and an error when no long VAR fits.
    #[test]
    fn test_validate_varma_long_var_order_rule() {
        use p2a_validation::{Check, Tol};
        let (case, _ds) = crate::test_support::load_case("varma_hr");
        let (m400, w400) = default_long_var_order(400, 2, 1, 1).unwrap();
        let (m60, w60) = default_long_var_order(60, 2, 1, 1).unwrap();
        Check::new(&case)
            .scalar("default_long_var_order", m400 as f64, Tol::exact())
            .scalar("short60_long_var_order", m60 as f64, Tol::exact())
            .finish();
        assert!(w400.is_none());
        let unreduced = case.scalar("short60_unreduced_order") as usize; // 16
        let w60 = w60.expect("the n = 60 default is reduced");
        assert!(
            w60.contains(&format!("reduced from {unreduced} to {m60}")),
            "{w60}"
        );
        assert_eq!(default_long_var_order(400, 2, 2, 0).unwrap(), (0, None));
        assert_eq!(default_long_var_order(400, 2, 20, 20).unwrap(), (40, None));
        assert_eq!(default_long_var_order(1000, 10, 1, 1).unwrap(), (47, None)); // 47 = floor(998 / 21)
        match default_long_var_order(6, 2, 1, 1) {
            Err(EconError::InsufficientData {
                required: 7,
                provided: 6,
                context,
            }) => {
                assert!(context.contains("long VAR"), "{context}")
            }
            other => panic!("{other:?}"),
        }
    }

    /// An explicit long_var_order: 0 is refused when q > 0, and a too-long VAR is
    /// InsufficientData.
    #[test]
    fn test_varma_rejects_bad_long_var_order() {
        let ds2 = ten_rows(2);
        let ab = ["a", "b"];
        let mut config = VarmaConfig::default(); // p = 1, q = 1
        config.long_var_order = Some(0);
        match run_varma_with(&ds2, &ab, &config) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("long_var_order"), "{message}")
            }
            other => panic!("{other:?}"),
        }
        config.long_var_order = Some(4); // 6 rows for 9 regressors
        expect_insufficient(run_varma_with(&ds2, &ab, &config), 14, 10, "long VAR(4)");
        config.q = 0;
        config.long_var_order = Some(0); // q = 0: m = 0 is the default
        let r = run_varma_with(&ds2, &ab, &config).unwrap();
        // 9 rows - rank 3 ([1, a.l1, b.l1]; R: lm.fit rank 3)
        assert_eq!((r.n_obs, r.long_var_order, r.df_residual), (9, 0, 6));
    }

    /// A bivariate VARMA(1, 1) of n rows (the varma_hr DGP: A = [[.5, -.2], [.1, .4]],
    /// M = [[.3, .2], [0, -.25]]) with the shared test fixtures' seeded normal draws.
    fn varma11_series(n: usize) -> Array2<f64> {
        use crate::econometrics::timeseries::common::test_fixtures::box_muller_ts;
        let mut seed = 20261114_u64;
        let mut y = Array2::<f64>::zeros((n + 100, 2));
        let mut prev = [0.0, 0.0];
        for t in 1..n + 100 {
            let e = [box_muller_ts(&mut seed), 0.9 * box_muller_ts(&mut seed)];
            let (a, b) = (y[[t - 1, 0]], y[[t - 1, 1]]);
            y[[t, 0]] = 0.5 * a - 0.2 * b + e[0] + 0.3 * prev[0] + 0.2 * prev[1];
            y[[t, 1]] = 0.1 * a + 0.4 * b + e[1] - 0.25 * prev[1];
            prev = e;
        }
        y.slice(s![100.., ..]).to_owned()
    }

    /// The step-1 normal-equation path gives the QR (lm.fit) coefficients on well-conditioned
    /// long VARs (m = 8 and m = 35 = the default at n = 400), and declines a design with
    /// duplicated lags, which then goes to the QR with R's aliasing warning.
    #[test]
    fn test_varma_long_var_normal_equations_match_qr() {
        let y = varma11_series(400);
        let names = vec!["y1".to_string(), "y2".to_string()];
        for m in [8, 35] {
            let z = lag_design(y.view(), y.view(), m, m, 0, 1 + 2 * m).unwrap();
            let target = y.slice(s![m.., ..]);
            let ne = normal_equation_coefficients(z.view(), target)
                .unwrap()
                .expect("well-conditioned long VAR takes the normal equations");
            let (qr, aliased) = qr_coefficients(z.view(), target).unwrap();
            assert!(aliased.is_empty());
            let worst = ne
                .iter()
                .zip(qr.iter())
                .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                .fold(0.0_f64, f64::max);
            assert!(worst < 1e-10, "m = {m}: {worst:e}");
        }
        let dup = ndarray::concatenate![
            ndarray::Axis(1),
            y.column(0).insert_axis(ndarray::Axis(1)),
            y.column(0).insert_axis(ndarray::Axis(1))
        ];
        let z = lag_design(dup.view(), dup.view(), 4, 4, 0, 9).unwrap();
        assert!(
            normal_equation_coefficients(z.view(), dup.slice(s![4.., ..]))
                .unwrap()
                .is_none()
        );
        let mut warnings = Vec::new();
        let dup_names = vec![names[0].clone(), "copy".to_string()];
        long_var_residuals(dup.view(), 4, &dup_names, &mut warnings).unwrap();
        assert!(
            warnings.len() == 1
                && warnings[0].contains("long VAR (step 1)")
                && warnings[0].contains("copy.l1"),
            "{warnings:?}"
        );
    }
}
