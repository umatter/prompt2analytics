//! GARCH(p, q) models of conditional heteroskedasticity, estimated by conditional maximum
//! likelihood with normal or standardised Student-t errors.
//!
//! # Model
//!
//! r_t = mu + e_t (mu = 0 without a mean term), e_t = sqrt(h_t) z_t and
//!
//! h_t = omega + sum_{i=1..p} alpha_i e_{t-i}^2 + sum_{j=1..q} beta_j h_{t-j},
//!
//! with omega > 0, alpha_i, beta_j >= 0 and sum alpha + sum beta < 1. z_t is N(0, 1)
//! ([`GarchDistribution::Normal`], fGarch "norm") or fGarch's standardised Student-t with shape
//! nu > 2 ([`GarchDistribution::StudentT`], fGarch "std"). The likelihood is conditional with
//! fGarch's "mci" start: h_t = mean(e^2) for the first max(p, q) periods, and the sum runs over
//! all n observations.
//!
//! # Estimation
//!
//! * **Parameters.** The optimiser works on unconstrained theta: mu = xbar + s theta_mu,
//!   omega = s^2 exp(theta_omega) (s^2 the sample variance), (alpha, beta) by a positive softmax,
//!   so that every coefficient is positive and their sum below 1, and
//!   nu = 2 + (GARCH_SHAPE_MAX - 2) logistic(theta_nu). The problem is scale free: data times c
//!   converge in the same iterations to (mu c, omega c^2, alpha, beta).
//! * **Optimiser.** BFGS (Nocedal & Wright 2006, Alg. 6.1) with a strong-Wolfe line search
//!   (Alg. 3.5/3.6) on the analytic gradient, which comes from the recursions for
//!   dh_t / dparameter in one O(n (p + q)) pass. The start inverse Hessian is the inverse
//!   finite-difference Hessian. Three starting values, (sum alpha, sum beta) = (0.1, 0.8),
//!   (0.05, 0.05) and (0.3, 0.6) with omega = s^2 (1 - sum), are tried and the highest
//!   log-likelihood wins.
//! * **Convergence.** max_i |d(loglik / n) / d theta_i| <= `tolerance`. Up to three Newton steps
//!   in the natural parameters then polish the winner; each is kept only when it is feasible,
//!   does not lower the log-likelihood and lowers the criterion. Stopping at the iteration
//!   limit is reported with a warning.
//! * **Inference.** Standard errors are sqrt(diag(I^-1)), I the observed information (the
//!   negative Hessian of the log-likelihood in the natural parameters, from central differences
//!   of the analytic gradient); `t_stats` are Wald z values with normal p-values 2 Phi(-|z|), as
//!   in fGarch and tseries. Where I is not positive definite the standard errors are NaN with a
//!   warning. A coefficient at or below 1e-4 of alpha + beta, or a shape within 1e-3 of the
//!   range of its upper bound [`GARCH_SHAPE_MAX`], is on its bound: it is held fixed for the
//!   standard errors (its own are NaN) and reported in `at_bound` and the warnings.
//! * **Forecasts.** [`garch_forecast`] runs the GARCH(p, q) recursion with E[e^2_s] = sigma^2_s
//!   beyond the sample.
//!
//! # References
//!
//! - Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
//!   *Journal of Econometrics*, 31(3), 307-327.
//! - Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates
//!   of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
//! - Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., ch. 3 and 6.
//! - Würtz, D., Chalabi, Y. and Luksan, L. (2006). Parameter estimation of ARMA models with
//!   GARCH/APARCH errors: an R and SPlus software implementation (fGarch).
//!
//! R equivalent: `fGarch::garchFit(~ garch(p, q), cond.dist = "norm" | "std")`.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::cholesky;
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::resource::{checkpoint, checkpoint_stride, try_with_capacity, try_zeros2};
use crate::traits::SignificanceLevel;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::normal_p_value_two_sided;

/// Upper bound of the Student-t shape nu (degrees of freedom): above it the standardised t cannot be told
/// from the normal and lgamma((nu+1)/2) - lgamma(nu/2) loses digits to cancellation.
pub const GARCH_SHAPE_MAX: f64 = 200.0;
/// Shape at every starting value.
const GARCH_SHAPE_START: f64 = 8.0;
/// A variance coefficient at or below this fraction of alpha + beta is on its lower bound 0.
const BOUND_REL: f64 = 1e-4;
/// Newton steps after BFGS (each kept only if feasible, logLik does not fall, the criterion falls).
const NEWTON_STEPS: usize = 3;
const MAX_GARCH_ORDER: usize = 20;
const MIN_OBS: usize = 20;
const WOLFE_C1: f64 = 1e-4;
const WOLFE_C2: f64 = 0.9;
const LINE_SEARCH_TRIALS: usize = 40;
const START_HESSIAN_STEP: f64 = 1e-4;
/// A later start replaces the best one only above this relative margin.
const TIE_REL: f64 = 1e-9;
const CRITERION: &str = "scaled gradient";
const NOT_PD: &str = "the observed information (negative Hessian of the log-likelihood) is not positive definite at the estimate: standard errors, z values and p-values are not reported (NaN)";
const NO_HESSIAN: &str = "the Hessian of the log-likelihood could not be evaluated at the estimate: standard errors are not reported (NaN)";

/// Conditional distribution of the standardised errors z_t = e_t / sqrt(h_t) (fGarch's `cond.dist`).
///
/// ```
/// use p2a_core::forecasting::garch::GarchDistribution;
///
/// assert_eq!("t".parse::<GarchDistribution>()?, GarchDistribution::StudentT);
/// assert_eq!(GarchDistribution::Normal.to_string(), "norm");
/// # Ok::<(), p2a_core::errors::EconError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GarchDistribution {
    /// z_t ~ N(0, 1) (fGarch "norm").
    #[default]
    #[serde(rename = "norm", alias = "normal", alias = "gaussian")]
    Normal,
    /// z_t standardised Student-t with shape nu > 2 (fGarch "std").
    #[serde(rename = "std", alias = "t", alias = "student_t", alias = "student")]
    StudentT,
}

crate::option_enum!(
    GarchDistribution,
    crate::options::Choices {
        option: "distribution",
        values: &[
            (&["norm", "normal", "gaussian"], GarchDistribution::Normal),
            (
                &["std", "t", "student_t", "student", "studentt"],
                GarchDistribution::StudentT
            ),
        ],
    }
);

/// Configuration for GARCH estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarchConfig {
    /// ARCH order p, 1..=20 (default 1).
    pub p: usize,
    /// GARCH order q, 0..=20 (default 1).
    pub q: usize,
    /// Estimate a mean mu (default true); without it mu = 0.
    pub include_mean: bool,
    /// BFGS iterations per starting value (default 500), at least 1; up to 3 Newton steps follow.
    pub max_iter: usize,
    /// Convergence tolerance on max_i |d(loglik / n) / d theta_i| over the transformed parameters
    /// (default 1e-8); finite and positive.
    pub tolerance: f64,
    /// Conditional distribution of the standardised errors (default normal).
    #[serde(default)]
    pub distribution: GarchDistribution,
}

impl Default for GarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            include_mean: true,
            max_iter: 500,
            tolerance: 1e-8,
            distribution: GarchDistribution::Normal,
        }
    }
}

/// Result from GARCH estimation.
///
/// `std_errors`, `t_stats`, `p_values` and `significance` follow `param_names`. They are NaN for a
/// parameter on its bound (`at_bound`) and for every parameter when the observed information is not
/// positive definite (see `warnings`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarchResult {
    /// Model specification string (e.g., "GARCH(1,1)")
    pub model: String,
    /// Estimated mean (mu); 0 without a mean term
    pub mu: f64,
    /// Estimated intercept in variance equation (omega)
    pub omega: f64,
    /// ARCH coefficients (alpha)
    pub alpha: Vec<f64>,
    /// GARCH coefficients (beta)
    pub beta: Vec<f64>,
    /// Standard errors, sqrt of the diagonal of the inverse observed information (NaN where
    /// unavailable)
    pub std_errors: Vec<f64>,
    /// Wald z values, estimate / standard error
    pub t_stats: Vec<f64>,
    /// Two-sided normal p-values 2 Phi(-|z|)
    pub p_values: Vec<f64>,
    /// Significance levels for all parameters
    pub significance: Vec<SignificanceLevel>,
    /// Conditional variances (h_t)
    #[serde(skip)]
    pub conditional_variance: Vec<f64>,
    /// Standardized residuals (z_t = e_t / sqrt(h_t))
    #[serde(skip)]
    pub std_residuals: Vec<f64>,
    /// Log-likelihood value
    pub log_likelihood: f64,
    /// AIC, -2 logLik + 2 k
    pub aic: f64,
    /// BIC, -2 logLik + k ln n
    pub bic: f64,
    /// Persistence (sum of alpha and beta)
    pub persistence: f64,
    /// Unconditional variance omega / (1 - persistence)
    pub unconditional_variance: f64,
    /// Half-life of volatility shock (in periods)
    pub half_life: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters, including the shape of the t
    pub n_params: usize,
    /// Convergence achieved (mirrors `convergence.converged`)
    pub converged: bool,
    /// BFGS iterations of the winning start plus its kept Newton steps
    pub iterations: usize,
    /// Parameter names: [mu], omega, alpha1..p, beta1..q, [shape]
    #[serde(default)]
    pub param_names: Vec<String>,
    /// Conditional distribution of the standardised errors
    #[serde(default)]
    pub distribution: GarchDistribution,
    /// Student-t shape nu (t errors only)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<f64>,
    /// Covariance of the estimates in `param_names` order (NaN rows and columns where unavailable)
    #[serde(default)]
    pub vcov: Vec<Vec<f64>>,
    /// Parameters held fixed on a bound for the standard errors
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub at_bound: Vec<String>,
    /// Convergence record of the optimiser
    #[serde(default)]
    pub convergence: Convergence,
    /// Warnings (bounds, inference, convergence)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl GarchResult {
    /// Estimates in `param_names` order.
    pub fn estimates(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.param_names.len());
        if self.param_names.first().is_some_and(|n| n == "mu") {
            v.push(self.mu);
        }
        v.push(self.omega);
        v.extend(&self.alpha);
        v.extend(&self.beta);
        if let Some(nu) = self.shape {
            v.push(nu);
        }
        v
    }
}

impl ResultNotes for GarchResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

/// `{x:.4e}` for small or large magnitudes, `{x:.6}` otherwise; non-finite as Rust prints it.
fn fmt_num(x: f64) -> String {
    if !x.is_finite() {
        format!("{x}")
    } else if x != 0.0 && (x.abs() < 1e-3 || x.abs() >= 1e5) {
        format!("{x:.4e}")
    } else {
        format!("{x:.6}")
    }
}

impl fmt::Display for GarchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rule = "=".repeat(64);
        let dash = "-".repeat(64);
        writeln!(f, "\n{rule}")?;
        writeln!(f, "{:^64}", self.model)?;
        writeln!(f, "{rule}")?;
        match (self.distribution, self.shape) {
            (GarchDistribution::StudentT, Some(nu)) => writeln!(
                f,
                "Conditional distribution: std (standardised Student-t errors, shape {nu:.4})"
            )?,
            (GarchDistribution::StudentT, None) => writeln!(
                f,
                "Conditional distribution: std (standardised Student-t errors)"
            )?,
            (GarchDistribution::Normal, _) => {
                writeln!(f, "Conditional distribution: norm (normal errors)")?
            }
        }

        writeln!(f, "\nCoefficients:")?;
        writeln!(f, "{dash}")?;
        writeln!(
            f,
            "{:<10} {:>12} {:>12} {:>10} {:>10}",
            "Parameter", "Estimate", "Std.Error", "z value", "Pr(>|z|)"
        )?;
        writeln!(f, "{dash}")?;
        let est = self.estimates();
        for (i, name) in self.param_names.iter().enumerate() {
            let get = |v: &[f64]| v.get(i).copied().unwrap_or(f64::NAN);
            let stars = self.significance.get(i).map_or("", |s| s.stars());
            writeln!(
                f,
                "{:<10} {:>12} {:>12} {:>10.4} {:>10.4} {}",
                name,
                fmt_num(get(&est)),
                fmt_num(get(&self.std_errors)),
                get(&self.t_stats),
                get(&self.p_values),
                stars
            )?;
        }
        writeln!(f, "{dash}")?;

        writeln!(f, "\nModel Diagnostics:")?;
        writeln!(f, "  Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "  AIC:            {:.4}", self.aic)?;
        writeln!(f, "  BIC:            {:.4}", self.bic)?;

        writeln!(f, "\nVolatility Characteristics:")?;
        writeln!(f, "  Persistence (α + β):   {:.6}", self.persistence)?;
        writeln!(
            f,
            "  Unconditional Var:     {}",
            fmt_num(self.unconditional_variance)
        )?;
        writeln!(
            f,
            "  Unconditional Vol:     {:.4}%",
            self.unconditional_variance.sqrt() * 100.0
        )?;
        writeln!(f, "  Half-life (periods):   {:.2}", self.half_life)?;

        writeln!(f, "\nEstimation:")?;
        writeln!(f, "  Observations:   {}", self.n_obs)?;
        writeln!(f, "  Parameters:     {}", self.n_params)?;
        write!(f, "{}", notes_text(self))
    }
}

/// Layout of the natural parameter vector [mu], omega, alpha_1..p, beta_1..q, [shape] and the data scale.
#[derive(Debug, Clone)]
struct Spec {
    p: usize,
    q: usize,
    mean: bool,
    student: bool,
    n: usize,
    xbar: f64,
    s2: f64,
    s: f64,
}

impl Spec {
    fn new(x: &[f64], p: usize, q: usize, mean: bool, dist: GarchDistribution) -> Spec {
        let n = x.len();
        let xbar = x.iter().sum::<f64>() / n as f64;
        let s2 = x.iter().map(|v| (v - xbar) * (v - xbar)).sum::<f64>() / n as f64;
        Spec {
            p,
            q,
            mean,
            student: dist == GarchDistribution::StudentT,
            n,
            xbar,
            s2,
            s: s2.sqrt(),
        }
    }
    fn m(&self) -> usize {
        self.p.max(self.q)
    }
    /// Index of omega.
    fn io(&self) -> usize {
        usize::from(self.mean)
    }
    /// Indices of the alphas and betas.
    fn coef(&self) -> std::ops::Range<usize> {
        self.io() + 1..self.io() + 1 + self.p + self.q
    }
    /// Number of parameters.
    fn k(&self) -> usize {
        self.coef().end + usize::from(self.student)
    }
    fn names(&self) -> Vec<String> {
        let mut v = Vec::with_capacity(self.k());
        if self.mean {
            v.push("mu".to_string());
        }
        v.push("omega".to_string());
        v.extend((1..=self.p).map(|i| format!("alpha{i}")));
        v.extend((1..=self.q).map(|j| format!("beta{j}")));
        if self.student {
            v.push("shape".to_string());
        }
        v
    }
}

fn is_budget_error(e: &EconError) -> bool {
    matches!(e, EconError::Cancelled(_) | EconError::ResourceLimit(_))
}

/// Log-likelihood and its analytic gradient in the natural parameters, in one pass.
///
/// `Ok(None)` when the point is infeasible (some h_t not positive and finite, a shape at or below
/// 2, or a non-finite result); `Err` only for a cancelled budget. The derivatives d_t = dh_t /
/// d(mu, omega, alpha, beta) follow the beta recursion; the last q of them sit in a ring buffer.
fn loglik_grad(spec: &Spec, x: &[f64], nat: &[f64]) -> EconResult<Option<(f64, Vec<f64>)>> {
    let (p, q, n, io) = (spec.p, spec.q, spec.n, spec.io());
    let k = spec.k();
    let kv = spec.coef().end;
    let m = spec.m();
    let mu = if spec.mean { nat[0] } else { 0.0 };
    let omega = nat[io];
    let alpha = &nat[io + 1..io + 1 + p];
    let beta = &nat[io + 1 + p..io + 1 + p + q];
    let nu = if spec.student { nat[k - 1] } else { f64::NAN };
    if spec.student && !(nu > 2.0 && nu.is_finite()) {
        return Ok(None);
    }
    let (mut se, mut se2) = (0.0, 0.0);
    for &v in x {
        let e = v - mu;
        se += e;
        se2 += e * e;
    }
    let ebar = se / n as f64;
    let hbar = se2 / n as f64;

    let mut g = vec![0.0; k];
    let mut hring = vec![0.0; q.max(1)];
    let mut ring = vec![0.0; q.max(1) * kv];
    let mut d = vec![0.0; kv];
    let (mut sum_norm, mut ll_t, mut g_nu) = (0.0, 0.0, 0.0);
    let stride = checkpoint_stride(kv * (q + 1) + p + 1);
    for t in 0..n {
        if t % stride == 0 {
            checkpoint()?;
        }
        let e = x[t] - mu;
        let h;
        if t < m {
            h = hbar;
            d.iter_mut().for_each(|v| *v = 0.0);
            if spec.mean {
                d[0] = -2.0 * ebar;
            }
        } else {
            let mut ht = omega;
            let mut dmu = 0.0;
            for i in 1..=p {
                let el = x[t - i] - mu;
                ht += alpha[i - 1] * el * el;
                dmu += alpha[i - 1] * el;
            }
            d.iter_mut().for_each(|v| *v = 0.0);
            if spec.mean {
                d[0] = -2.0 * dmu;
            }
            d[io] = 1.0;
            for i in 1..=p {
                let el = x[t - i] - mu;
                d[io + i] = el * el;
            }
            for j in 1..=q {
                let slot = (t - j) % q;
                let hl = hring[slot];
                ht += beta[j - 1] * hl;
                d[io + p + j] += hl;
                let b = beta[j - 1];
                let prev = &ring[slot * kv..(slot + 1) * kv];
                for (dc, pc) in d.iter_mut().zip(prev) {
                    *dc += b * pc;
                }
            }
            h = ht;
        }
        if !(h > 0.0 && h.is_finite()) {
            return Ok(None);
        }
        if q > 0 {
            let slot = t % q;
            hring[slot] = h;
            ring[slot * kv..(slot + 1) * kv].copy_from_slice(&d);
        }
        let (dl_dh, dl_de);
        if spec.student {
            let km = nu - 2.0;
            let r = e * e / (h * km);
            let l1p = r.ln_1p();
            ll_t += -0.5 * h.ln() - (nu + 1.0) / 2.0 * l1p;
            dl_dh = -0.5 / h + (nu + 1.0) / 2.0 * r / (h * (1.0 + r));
            dl_de = -(nu + 1.0) * e / (h * km * (1.0 + r));
            g_nu += -0.5 * l1p + (nu + 1.0) / 2.0 * r / (km * (1.0 + r));
        } else {
            sum_norm += h.ln() + e * e / h;
            dl_dh = -0.5 * (1.0 / h - e * e / (h * h));
            dl_de = -e / h;
        }
        for (gc, dc) in g.iter_mut().zip(&d) {
            *gc += dl_dh * dc;
        }
        if spec.mean {
            g[0] -= dl_de;
        }
    }
    let nf = n as f64;
    let ll = if spec.student {
        g[k - 1] = nf
            * (0.5 * digamma((nu + 1.0) / 2.0) - 0.5 * digamma(nu / 2.0) - 0.5 / (nu - 2.0))
            + g_nu;
        ll_t + nf
            * (ln_gamma((nu + 1.0) / 2.0)
                - ln_gamma(nu / 2.0)
                - 0.5 * (std::f64::consts::PI * (nu - 2.0)).ln())
    } else {
        -0.5 * (nf * (2.0 * std::f64::consts::PI).ln() + sum_norm)
    };
    if !ll.is_finite() || g.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    Ok(Some((ll, g)))
}

/// Conditional variances h_t and residuals e_t at `nat` by the same recursion, without the gradient.
fn conditional_variances(spec: &Spec, x: &[f64], nat: &[f64]) -> EconResult<(Vec<f64>, Vec<f64>)> {
    let (p, q, n, io) = (spec.p, spec.q, spec.n, spec.io());
    let mu = if spec.mean { nat[0] } else { 0.0 };
    let omega = nat[io];
    let alpha = &nat[io + 1..io + 1 + p];
    let beta = &nat[io + 1 + p..io + 1 + p + q];
    let mut e = try_with_capacity(n, "the GARCH residuals")?;
    e.extend(x.iter().map(|v| v - mu));
    let hbar = e.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let mut h: Vec<f64> = try_with_capacity(n, "the GARCH conditional variances")?;
    let stride = checkpoint_stride(p + q + 1);
    for t in 0..n {
        if t % stride == 0 {
            checkpoint()?;
        }
        if t < spec.m() {
            h.push(hbar);
            continue;
        }
        let mut ht = omega;
        for i in 1..=p {
            ht += alpha[i - 1] * e[t - i] * e[t - i];
        }
        for j in 1..=q {
            ht += beta[j - 1] * h[t - j];
        }
        h.push(ht);
    }
    Ok((h, e))
}

fn logistic(t: f64) -> f64 {
    if t >= 0.0 {
        1.0 / (1.0 + (-t).exp())
    } else {
        let z = t.exp();
        z / (1.0 + z)
    }
}

/// Natural parameters of the transformed vector theta.
fn to_natural(spec: &Spec, th: &[f64]) -> Vec<f64> {
    let k = spec.k();
    let io = spec.io();
    let mut nat = vec![0.0; k];
    if spec.mean {
        nat[0] = spec.xbar + spec.s * th[0];
    }
    nat[io] = spec.s2 * th[io].exp();
    let c = spec.coef();
    let top = th[c.clone()].iter().fold(0.0_f64, |a, &b| a.max(b));
    let mut denom = (-top).exp();
    for i in c.clone() {
        nat[i] = (th[i] - top).exp();
        denom += nat[i];
    }
    for v in &mut nat[c] {
        *v /= denom;
    }
    if spec.student {
        nat[k - 1] = 2.0 + (GARCH_SHAPE_MAX - 2.0) * logistic(th[k - 1]);
    }
    nat
}

/// Inverse of [`to_natural`].
fn to_theta(spec: &Spec, nat: &[f64]) -> Vec<f64> {
    let k = spec.k();
    let io = spec.io();
    let mut th = vec![0.0; k];
    if spec.mean {
        th[0] = (nat[0] - spec.xbar) / spec.s;
    }
    th[io] = (nat[io] / spec.s2).ln();
    let c = spec.coef();
    let rest = 1.0 - nat[c.clone()].iter().sum::<f64>();
    for i in c {
        th[i] = (nat[i] / rest).ln();
    }
    if spec.student {
        let nu = nat[k - 1];
        th[k - 1] = ((nu - 2.0) / (GARCH_SHAPE_MAX - nu)).ln();
    }
    th
}

/// Gradient with respect to theta from the gradient `g` in the natural parameters at `nat`.
fn theta_gradient(spec: &Spec, nat: &[f64], g: &[f64]) -> Vec<f64> {
    let k = spec.k();
    let io = spec.io();
    let mut gt = vec![0.0; k];
    if spec.mean {
        gt[0] = spec.s * g[0];
    }
    gt[io] = nat[io] * g[io];
    let c = spec.coef();
    let sum: f64 = c.clone().map(|i| nat[i] * g[i]).sum();
    for i in c {
        gt[i] = nat[i] * (g[i] - sum);
    }
    if spec.student {
        let nu = nat[k - 1];
        gt[k - 1] = (nu - 2.0) * (GARCH_SHAPE_MAX - nu) / (GARCH_SHAPE_MAX - 2.0) * g[k - 1];
    }
    gt
}

/// The convergence criterion max_i |d(loglik / n) / d theta_i|.
fn criterion(spec: &Spec, nat: &[f64], g: &[f64]) -> f64 {
    theta_gradient(spec, nat, g)
        .iter()
        .fold(0.0_f64, |a, v| a.max(v.abs()))
        / spec.n as f64
}

/// f = -loglik / n and its gradient in theta; `Ok(None)` where infeasible.
fn objective(spec: &Spec, x: &[f64], th: &[f64]) -> EconResult<Option<(f64, Vec<f64>)>> {
    let nat = to_natural(spec, th);
    let n = spec.n as f64;
    Ok(loglik_grad(spec, x, &nat)?.map(|(ll, g)| {
        let gt = theta_gradient(spec, &nat, &g);
        (-ll / n, gt.iter().map(|v| -v / n).collect())
    }))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
}

/// One accepted line-search point: step length, theta, f and gradient.
type Trial = (f64, Vec<f64>, f64, Vec<f64>);

/// Strong-Wolfe line search (Nocedal & Wright 2006, Alg. 3.5 with bisection zoom, Alg. 3.6).
/// Returns the best Armijo point when the curvature condition is never met; `None` when there
/// is no Armijo point.
fn line_search<F>(
    obj: &mut F,
    th: &[f64],
    f0: f64,
    g0: &[f64],
    d: &[f64],
) -> EconResult<Option<Trial>>
where
    F: FnMut(&[f64]) -> EconResult<Option<(f64, Vec<f64>)>>,
{
    let dphi0 = dot(g0, d);
    let mut best: Option<Trial> = None;
    let mut eval =
        |a: f64, best: &mut Option<Trial>| -> EconResult<Option<(f64, Vec<f64>, Vec<f64>)>> {
            let ta: Vec<f64> = th.iter().zip(d).map(|(t, di)| t + a * di).collect();
            match obj(&ta)? {
                Some((fa, ga)) if fa.is_finite() => {
                    if fa <= f0 + WOLFE_C1 * a * dphi0 && best.as_ref().is_none_or(|b| fa < b.2) {
                        *best = Some((a, ta.clone(), fa, ga.clone()));
                    }
                    Ok(Some((fa, ga, ta)))
                }
                _ => Ok(None),
            }
        };
    let (mut a_prev, mut f_prev) = (0.0, f0);
    let mut a = 1.0;
    let mut bracket: Option<(f64, f64, f64)> = None; // (lo, f_lo, hi)
    for i in 0..LINE_SEARCH_TRIALS {
        let r = eval(a, &mut best)?;
        match r {
            Some((fa, ga, ta)) if fa <= f0 + WOLFE_C1 * a * dphi0 && (i == 0 || fa < f_prev) => {
                let dphi = dot(&ga, d);
                if dphi.abs() <= -WOLFE_C2 * dphi0 {
                    return Ok(Some((a, ta, fa, ga)));
                }
                if dphi >= 0.0 {
                    bracket = Some((a, fa, a_prev));
                    break;
                }
                a_prev = a;
                f_prev = fa;
                a *= 2.0;
            }
            _ => {
                bracket = Some((a_prev, f_prev, a));
                break;
            }
        }
    }
    if let Some((mut lo, mut f_lo, mut hi)) = bracket {
        for _ in 0..LINE_SEARCH_TRIALS {
            let a = 0.5 * (lo + hi);
            match eval(a, &mut best)? {
                Some((fa, ga, ta)) if fa <= f0 + WOLFE_C1 * a * dphi0 && fa < f_lo => {
                    let dphi = dot(&ga, d);
                    if dphi.abs() <= -WOLFE_C2 * dphi0 {
                        return Ok(Some((a, ta, fa, ga)));
                    }
                    if dphi * (hi - lo) >= 0.0 {
                        hi = lo;
                    }
                    lo = a;
                    f_lo = fa;
                }
                _ => hi = a,
            }
        }
    }
    Ok(best)
}

/// Inverse of the central-difference Hessian of f in theta at `th` (step 1e-4, symmetrised) when
/// that Hessian is positive definite; the identity otherwise. The flag says which.
fn start_inverse_hessian<F>(obj: &mut F, th: &[f64]) -> EconResult<(Array2<f64>, bool)>
where
    F: FnMut(&[f64]) -> EconResult<Option<(f64, Vec<f64>)>>,
{
    let k = th.len();
    let identity = Array2::eye(k);
    let mut hess = try_zeros2(k, k, "the GARCH start Hessian")?;
    for j in 0..k {
        let mut tp = th.to_vec();
        tp[j] += START_HESSIAN_STEP;
        let mut tm = th.to_vec();
        tm[j] -= START_HESSIAN_STEP;
        let (Some((_, gp)), Some((_, gm))) = (obj(&tp)?, obj(&tm)?) else {
            return Ok((identity, false));
        };
        for i in 0..k {
            hess[[i, j]] = (gp[i] - gm[i]) / (2.0 * START_HESSIAN_STEP);
        }
    }
    let sym = (&hess + &hess.t()) * 0.5;
    if cholesky(&sym.view()).is_err() {
        return Ok((identity, false));
    }
    match sym_inverse(sym.view(), SingularPolicy::Error, "the GARCH start Hessian") {
        Ok(inv) => Ok((inv.inverse, true)),
        Err(e) if is_budget_error(&e) => Err(e),
        Err(_) => Ok((identity, false)),
    }
}

/// End point of a BFGS run.
struct BfgsOut {
    th: Vec<f64>,
    f: f64,
    iterations: usize,
}

/// BFGS in theta (Nocedal & Wright 2006, Alg. 6.1) until max |grad f| <= `tol`, `max_iter`
/// iterations or a failed line search. `None` when f is not finite at `th0`.
fn bfgs<F>(obj: &mut F, th0: Vec<f64>, max_iter: usize, tol: f64) -> EconResult<Option<BfgsOut>>
where
    F: FnMut(&[f64]) -> EconResult<Option<(f64, Vec<f64>)>>,
{
    let Some((mut f, mut g)) = obj(&th0)? else {
        return Ok(None);
    };
    let k = th0.len();
    let mut th = th0;
    let (mut b, mut scaled) = start_inverse_hessian(obj, &th)?;
    let mut iterations = 0;
    while max_abs(&g) > tol && iterations < max_iter {
        checkpoint()?;
        let mut d: Vec<f64> = (0..k)
            .map(|i| -(0..k).map(|j| b[[i, j]] * g[j]).sum::<f64>())
            .collect();
        if dot(&d, &g) >= 0.0 {
            b = Array2::eye(k);
            scaled = false;
            d = g.iter().map(|v| -v).collect();
        }
        let Some((a, th_new, f_new, g_new)) = line_search(obj, &th, f, &g, &d)? else {
            break;
        };
        let s: Vec<f64> = d.iter().map(|v| a * v).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(n, o)| n - o).collect();
        let sy = dot(&s, &y);
        if sy > 0.0 {
            if !scaled {
                b = Array2::eye(k) * (sy / dot(&y, &y));
                scaled = true;
            }
            let by: Vec<f64> = (0..k)
                .map(|i| (0..k).map(|j| b[[i, j]] * y[j]).sum())
                .collect();
            let rho = 1.0 / sy;
            let c = rho + rho * rho * dot(&y, &by);
            for i in 0..k {
                for j in 0..k {
                    b[[i, j]] += c * s[i] * s[j] - rho * (by[i] * s[j] + s[i] * by[j]);
                }
            }
        }
        th = th_new;
        f = f_new;
        g = g_new;
        iterations += 1;
    }
    Ok(Some(BfgsOut { th, f, iterations }))
}

/// The three starting values: (sum alpha, sum beta) = (0.1, 0.8), (0.05, 0.05), (0.3, 0.6)
/// (sum beta = 0 without GARCH terms), split evenly over the lags, with omega = s^2 (1 - sum),
/// mu = xbar and shape 8.
fn starts(spec: &Spec) -> [Vec<f64>; 3] {
    [(0.1, 0.8), (0.05, 0.05), (0.3, 0.6)].map(|(sa, sb)| {
        let sb = if spec.q == 0 { 0.0 } else { sb };
        let mut v = Vec::with_capacity(spec.k());
        if spec.mean {
            v.push(spec.xbar);
        }
        v.push(spec.s2 * (1.0 - sa - sb));
        v.extend(std::iter::repeat_n(sa / spec.p as f64, spec.p));
        v.extend(std::iter::repeat_n(sb / spec.q.max(1) as f64, spec.q));
        if spec.student {
            v.push(GARCH_SHAPE_START);
        }
        v
    })
}

/// Parameters on a bound: a coefficient at or below BOUND_REL of alpha + beta, or a shape above
/// GARCH_SHAPE_MAX - 1e-3 (GARCH_SHAPE_MAX - 2).
fn at_bounds(spec: &Spec, nat: &[f64]) -> Vec<bool> {
    let k = spec.k();
    let mut fixed = vec![false; k];
    let c = spec.coef();
    let total: f64 = nat[c.clone()].iter().sum();
    for i in c {
        fixed[i] = nat[i] <= BOUND_REL * total;
    }
    if spec.student {
        fixed[k - 1] = nat[k - 1] > GARCH_SHAPE_MAX - 1e-3 * (GARCH_SHAPE_MAX - 2.0);
    }
    fixed
}

/// Hessian of the log-likelihood on the free parameters by central differences of the analytic
/// gradient (steps: mu 1e-5 s; omega, alpha, beta 1e-6 |value|; shape 1e-6 of its distance to the
/// nearer bound), symmetrised. `None` when a step is not positive or a gradient is infeasible.
fn hessian_free(
    spec: &Spec,
    x: &[f64],
    nat: &[f64],
    fixed: &[bool],
) -> EconResult<Option<Array2<f64>>> {
    let k = spec.k();
    let free: Vec<usize> = (0..k).filter(|&i| !fixed[i]).collect();
    let kf = free.len();
    let mut hess = try_zeros2(kf, kf, "the GARCH Hessian")?;
    for (cj, &j) in free.iter().enumerate() {
        checkpoint()?;
        let step = if spec.mean && j == 0 {
            1e-5 * spec.s
        } else if spec.student && j == k - 1 {
            1e-6 * (nat[j] - 2.0).min(GARCH_SHAPE_MAX - nat[j])
        } else {
            1e-6 * nat[j].abs()
        };
        if !(step > 0.0 && step.is_finite()) {
            return Ok(None);
        }
        let mut up = nat.to_vec();
        up[j] += step;
        let mut down = nat.to_vec();
        down[j] -= step;
        let (Some((_, gp)), Some((_, gm))) =
            (loglik_grad(spec, x, &up)?, loglik_grad(spec, x, &down)?)
        else {
            return Ok(None);
        };
        for (ci, &i) in free.iter().enumerate() {
            hess[[ci, cj]] = (gp[i] - gm[i]) / (2.0 * step);
        }
    }
    Ok(Some((&hess + &hess.t()) * 0.5))
}

/// Whether `nat` lies in the parameter space: omega > 0, coefficients >= 0 with sum < 1, 2 < nu < max.
fn feasible(spec: &Spec, nat: &[f64]) -> bool {
    let c = spec.coef();
    let ok = nat[spec.io()] > 0.0
        && nat[c.clone()].iter().all(|v| *v >= 0.0)
        && nat[c].iter().sum::<f64>() < 1.0
        && nat.iter().all(|v| v.is_finite());
    let k = spec.k();
    ok && (!spec.student || (nat[k - 1] > 2.0 && nat[k - 1] < GARCH_SHAPE_MAX))
}

/// Up to NEWTON_STEPS Newton steps on the free parameters; each kept only when the candidate is
/// feasible, its log-likelihood does not fall (1e-12 relative) and its criterion falls. Returns
/// the polished point and the number of kept steps.
fn newton_polish(
    spec: &Spec,
    x: &[f64],
    nat: &[f64],
    fixed: &[bool],
) -> EconResult<(Vec<f64>, usize)> {
    let mut cur = nat.to_vec();
    let Some((mut ll, mut g)) = loglik_grad(spec, x, &cur)? else {
        return Ok((cur, 0));
    };
    let mut crit = criterion(spec, &cur, &g);
    let free: Vec<usize> = (0..spec.k()).filter(|&i| !fixed[i]).collect();
    let mut kept = 0;
    for _ in 0..NEWTON_STEPS {
        checkpoint()?;
        let Some(h) = hessian_free(spec, x, &cur, fixed)? else {
            break;
        };
        let info = -h;
        let inv = match sym_inverse(info.view(), SingularPolicy::Error, "the GARCH Hessian") {
            Ok(inv) => inv.inverse,
            Err(e) if is_budget_error(&e) => return Err(e),
            Err(_) => break,
        };
        let mut cand = cur.clone();
        for (ci, &i) in free.iter().enumerate() {
            cand[i] += free
                .iter()
                .enumerate()
                .map(|(cj, &j)| inv[[ci, cj]] * g[j])
                .sum::<f64>();
        }
        if !feasible(spec, &cand) {
            break;
        }
        let Some((ll_c, g_c)) = loglik_grad(spec, x, &cand)? else {
            break;
        };
        let crit_c = criterion(spec, &cand, &g_c);
        if ll_c < ll - 1e-12 * ll.abs() || crit_c >= crit {
            break;
        }
        cur = cand;
        ll = ll_c;
        g = g_c;
        crit = crit_c;
        kept += 1;
    }
    Ok((cur, kept))
}

/// Standard errors and covariance from the inverse observed information on the free parameters.
struct Inference {
    std_errors: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    warning: Option<String>,
}

fn inference(spec: &Spec, x: &[f64], nat: &[f64], fixed: &[bool]) -> EconResult<Inference> {
    let k = spec.k();
    let mut out = Inference {
        std_errors: vec![f64::NAN; k],
        vcov: vec![vec![f64::NAN; k]; k],
        warning: None,
    };
    let Some(h) = hessian_free(spec, x, nat, fixed)? else {
        out.warning = Some(NO_HESSIAN.to_string());
        return Ok(out);
    };
    let info = -h;
    let kf = info.nrows();
    let diag: Vec<f64> = (0..kf).map(|i| info[[i, i]]).collect();
    if diag.iter().any(|v| !(*v > 0.0 && v.is_finite())) {
        out.warning = Some(NOT_PD.to_string());
        return Ok(out);
    }
    // Equilibrated so that the check is scale free (omega ~1e-6 next to beta ~0.9).
    let mut eq = info.clone();
    for i in 0..kf {
        for j in 0..kf {
            eq[[i, j]] /= (diag[i] * diag[j]).sqrt();
        }
    }
    if cholesky(&eq.view()).is_err() {
        out.warning = Some(NOT_PD.to_string());
        return Ok(out);
    }
    let inv = match sym_inverse(
        info.view(),
        SingularPolicy::Error,
        "the GARCH observed information",
    ) {
        Ok(inv) => inv.inverse,
        Err(e) if is_budget_error(&e) => return Err(e),
        Err(e) => {
            out.warning = Some(format!("standard errors are not reported (NaN): {e}"));
            return Ok(out);
        }
    };
    let free: Vec<usize> = (0..k).filter(|&i| !fixed[i]).collect();
    for (ci, &i) in free.iter().enumerate() {
        for (cj, &j) in free.iter().enumerate() {
            out.vcov[i][j] = inv[[ci, cj]];
        }
        if inv[[ci, ci]] > 0.0 {
            out.std_errors[i] = inv[[ci, ci]].sqrt();
        }
    }
    Ok(out)
}

/// Fit a GARCH(p, q) model by conditional maximum likelihood (see the module docs).
///
/// # Errors
///
/// * [`EconError::InsufficientData`]: fewer than 20 observations, or no more observations than
///   parameters.
/// * [`EconError::InvalidSpecification`]: p = 0, an order above 20 or at least n, `max_iter` = 0,
///   a tolerance that is not finite and positive, or a constant series.
/// * [`EconError::NullValues`] / [`EconError::NonFiniteValues`]: NaN or infinite data.
/// * [`EconError::DivergenceError`]: the log-likelihood is not finite at any starting value.
/// * [`EconError::Cancelled`] when the budget installed with [`crate::resource::with_budget`] is
///   cancelled or past its deadline: it is checked once per optimiser iteration, once per
///   Hessian column and about every 65,536 operations of each likelihood pass.
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::garch::{GarchConfig, garch, garch_forecast};
///
/// // volatility clustering: h_t = 2e-6 + 0.1 e_{t-1}^2 + 0.85 h_{t-1}, z_t a fixed sequence in (-1.7, 1.7)
/// let (mut h, mut e) = (2e-5_f64, 0.0_f64);
/// let mut x = Vec::new();
/// for t in 0..400_u64 {
///     h = 2e-6 + 0.1 * e * e + 0.85 * h;
///     let z = ((t * 7919 + 13) % 1000) as f64 / 1000.0 * 3.4 - 1.7;
///     e = h.sqrt() * z;
///     x.push(5e-4 + e);
/// }
/// let fit = garch(&x, Some(GarchConfig::default()))?;
/// assert_eq!(fit.param_names, ["mu", "omega", "alpha1", "beta1"]);
/// let (variance, volatility, _mean) = garch_forecast(&fit, 5)?;
/// assert_eq!(variance.len(), 5);
/// assert!((volatility[0] - variance[0].sqrt()).abs() < 1e-15);
/// # Ok::<(), p2a_core::errors::EconError>(())
/// ```
pub fn garch(data: &[f64], config: Option<GarchConfig>) -> EconResult<GarchResult> {
    let config = config.unwrap_or_default();
    validate_inputs(data, &config)?;
    let n = data.len();
    let spec = Spec::new(
        data,
        config.p,
        config.q,
        config.include_mean,
        config.distribution,
    );
    let k = spec.k();

    // BFGS from each start; the highest log-likelihood wins (ties go to the earlier start).
    let mut best: Option<(f64, Vec<f64>, usize)> = None;
    let mut obj = |th: &[f64]| objective(&spec, data, th);
    for start in starts(&spec) {
        let Some(out) = bfgs(
            &mut obj,
            to_theta(&spec, &start),
            config.max_iter,
            config.tolerance,
        )?
        else {
            continue;
        };
        let ll = -out.f * n as f64;
        if best
            .as_ref()
            .is_none_or(|(b, _, _)| ll > b + TIE_REL * b.abs().max(1.0))
        {
            best = Some((ll, to_natural(&spec, &out.th), out.iterations));
        }
    }
    let Some((_, nat0, bfgs_iterations)) = best else {
        return Err(EconError::DivergenceError {
            iteration: 0,
            suggestion: "the log-likelihood is not finite at any starting value; check the data for extreme values".to_string(),
        });
    };

    // Newton polish of the winner, then the bounds and inference at the final estimate.
    let (nat, kept) = newton_polish(&spec, data, &nat0, &at_bounds(&spec, &nat0))?;
    let Some((log_lik, grad)) = loglik_grad(&spec, data, &nat)? else {
        return Err(EconError::Computation(
            "the GARCH log-likelihood is not finite at the estimate".to_string(),
        ));
    };
    let crit = criterion(&spec, &nat, &grad);
    let fixed = at_bounds(&spec, &nat);
    let inf = inference(&spec, data, &nat, &fixed)?;
    let (h, e) = conditional_variances(&spec, data, &nat)?;
    let std_residuals: Vec<f64> = e.iter().zip(&h).map(|(e, h)| e / h.sqrt()).collect();

    let names = spec.names();
    let t_stats: Vec<f64> = nat
        .iter()
        .zip(&inf.std_errors)
        .map(|(v, se)| v / se)
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|z| normal_p_value_two_sided(*z))
        .collect();
    let significance = p_values
        .iter()
        .map(|p| SignificanceLevel::from_p_value(*p))
        .collect();

    let io = spec.io();
    let mu = if spec.mean { nat[0] } else { 0.0 };
    let omega = nat[io];
    let alpha = nat[io + 1..io + 1 + spec.p].to_vec();
    let beta = nat[io + 1 + spec.p..spec.coef().end].to_vec();
    let shape = spec.student.then(|| nat[k - 1]);
    let persistence: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
    let unconditional_variance = if persistence < 1.0 {
        omega / (1.0 - persistence)
    } else {
        f64::INFINITY
    };
    let half_life = if persistence > 0.0 && persistence < 1.0 {
        0.5_f64.ln() / persistence.ln()
    } else {
        f64::INFINITY
    };

    let iterations = bfgs_iterations + kept;
    let convergence = Convergence::new(
        crit <= config.tolerance,
        iterations,
        config.max_iter.saturating_add(NEWTON_STEPS),
        CRITERION,
        crit,
        config.tolerance,
    );
    let mut warnings = Vec::new();
    let mut at_bound = Vec::new();
    for (i, name) in names.iter().enumerate().filter(|(i, _)| fixed[*i]) {
        let v = nat[i];
        warnings.push(if spec.student && i == k - 1 {
            format!(
                "shape = {v:.4} is at its upper bound {GARCH_SHAPE_MAX}: the standardised residuals show no excess kurtosis over the normal; it is held fixed for the standard errors (distribution = norm is the limit)"
            )
        } else {
            format!(
                "{name} = {v:.3e} is at its lower bound 0 (at most 1e-4 of alpha + beta): it is held fixed for the standard errors, which are conditional on it; consider a model without this term"
            )
        });
        at_bound.push(name.clone());
    }
    warnings.extend(inf.warning);
    warnings.extend(convergence.note("GARCH"));

    Ok(GarchResult {
        model: format!("GARCH({},{})", config.p, config.q),
        mu,
        omega,
        alpha,
        beta,
        std_errors: inf.std_errors,
        t_stats,
        p_values,
        significance,
        conditional_variance: h,
        std_residuals,
        log_likelihood: log_lik,
        aic: -2.0 * log_lik + 2.0 * k as f64,
        bic: -2.0 * log_lik + k as f64 * (n as f64).ln(),
        persistence,
        unconditional_variance,
        half_life,
        n_obs: n,
        n_params: k,
        converged: convergence.converged,
        iterations,
        param_names: names,
        distribution: config.distribution,
        shape,
        vcov: inf.vcov,
        at_bound,
        convergence,
        warnings,
    })
}

/// Refuse settings and data the likelihood cannot use, before any work.
fn validate_inputs(data: &[f64], config: &GarchConfig) -> EconResult<()> {
    let n = data.len();
    if n < MIN_OBS {
        return Err(EconError::InsufficientData {
            required: MIN_OBS,
            provided: n,
            context: "GARCH estimation requires at least 20 observations".to_string(),
        });
    }
    let nan = data.iter().filter(|v| v.is_nan()).count();
    if nan > 0 {
        return Err(EconError::NullValues {
            column: "data".into(),
            count: nan,
        });
    }
    let inf = data.iter().filter(|v| v.is_infinite()).count();
    if inf > 0 {
        return Err(EconError::NonFiniteValues {
            column: "data".into(),
            count: inf,
        });
    }
    let (p, q) = (config.p, config.q);
    // `p`/`q` come from the caller (and, via the MCP/CLI tools, from untrusted input).
    if p > MAX_GARCH_ORDER || q > MAX_GARCH_ORDER {
        return Err(EconError::InvalidSpecification {
            message: format!("GARCH orders p={p} and q={q} must not exceed {MAX_GARCH_ORDER}"),
        });
    }
    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message:
                "GARCH needs at least one ARCH term: the ARCH order p must be at least 1, got 0"
                    .to_string(),
        });
    }
    if p.max(q) >= n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "GARCH orders p={p} and q={q} must be smaller than the number of observations ({n})"
            ),
        });
    }
    let student = config.distribution == GarchDistribution::StudentT;
    let n_params = usize::from(config.include_mean) + 1 + p + q + usize::from(student);
    if n <= n_params {
        return Err(EconError::InsufficientData {
            required: n_params + 1,
            provided: n,
            context: format!(
                "GARCH({p},{q}) {} a mean term{} has {n_params} parameters; the conditional likelihood needs more observations than parameters",
                if config.include_mean {
                    "with"
                } else {
                    "without"
                },
                if student { " and Student-t errors" } else { "" }
            ),
        });
    }
    if config.max_iter < 1 {
        return Err(EconError::InvalidSpecification {
            message: format!("max_iter must be at least 1, got {}", config.max_iter),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be finite and positive, got {}",
                config.tolerance
            ),
        });
    }
    let xbar = data.iter().sum::<f64>() / n as f64;
    let s2 = data.iter().map(|x| (x - xbar) * (x - xbar)).sum::<f64>() / n as f64;
    // all-equal first: the two-pass s2 of a constant series can be a rounding residue above 0
    if data.iter().all(|&v| v == data[0]) || s2 <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "GARCH needs a series with positive variance; the data are constant ({})",
                data[0]
            ),
        });
    }
    Ok(())
}

/// Forecast conditional variance and returns using a fitted GARCH model.
///
/// # Arguments
///
/// * `result` - Fitted GARCH result
/// * `h` - Forecast horizon
///
/// # Returns
///
/// Tuple of (forecast variances, forecast volatilities, forecast returns), or
/// [`EconError::ResourceLimit`] when the horizon is too large to allocate.
pub fn garch_forecast(
    result: &GarchResult,
    h: usize,
) -> EconResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    const WHAT: &str = "the GARCH forecast horizon";

    let alpha_sum: f64 = result.alpha.iter().sum();
    let beta_sum: f64 = result.beta.iter().sum();
    let persistence = alpha_sum + beta_sum;

    // Get last conditional variance and residual
    let last_var = *result
        .conditional_variance
        .last()
        .unwrap_or(&result.unconditional_variance);
    let last_resid_sq = result
        .std_residuals
        .last()
        .map(|z| z.powi(2) * last_var)
        .unwrap_or(result.unconditional_variance);

    let mut forecast_var = try_with_capacity(h, WHAT)?;
    let mut forecast_vol = try_with_capacity(h, WHAT)?;
    let mut forecast_ret = try_with_capacity(h, WHAT)?;

    let uncond_var = result.unconditional_variance;

    for i in 0..h {
        // Multi-step ahead variance forecast
        // For GARCH(1,1): E[σ²_{t+h}] = σ² + (α+β)^(h-1) * (σ²_{t+1} - σ²)
        let var_h = if i == 0 {
            result.omega
                + result.alpha.iter().sum::<f64>() * last_resid_sq
                + result.beta.iter().sum::<f64>() * last_var
        } else {
            uncond_var + persistence.powi(i as i32) * (forecast_var[i - 1] - uncond_var)
        };

        forecast_var.push(var_h);
        forecast_vol.push(var_h.sqrt());
        forecast_ret.push(result.mu);
    }

    Ok((forecast_var, forecast_vol, forecast_ret))
}

/// Run GARCH from a Dataset (MCP entry point).
///
/// Errors as [`garch`] does, including [`EconError::Cancelled`] on a stopped
/// budget.
pub fn run_garch(
    data: &[f64],
    p: Option<usize>,
    q: Option<usize>,
    include_mean: Option<bool>,
) -> EconResult<GarchResult> {
    let config = GarchConfig {
        p: p.unwrap_or(1),
        q: q.unwrap_or(1),
        include_mean: include_mean.unwrap_or(true),
        ..Default::default()
    };

    garch(data, Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Check, Tol};

    /// Parameters, h_t and forecasts at an interior optimum. The R emulation of this algorithm reaches
    /// the analytic-gradient root to <= 5e-14 (/dev/shm/p2a-plan/T3/WP35/compare.R). The tolerance is
    /// relative only, because omega ~1e-6 and the covariances ~1e-13 sit at Tol::iterative's 1e-12
    /// absolute floor.
    const PAR: Tol = Tol::new(1e-8, 0.0);
    /// SEs, z and covariances from the central-difference Hessian of the analytic gradient (emulation <= 2e-8).
    const SE: Tol = Tol::new(1e-6, 0.0);
    const N: usize = 1500;

    /// Column `name` of the committed garch_returns dataset (case garch_normal_t).
    /// Called as `returns("garch_normal_t", "r_norm")`: the case id is a literal in each test body (sync S6).
    fn returns(case_id: &str, name: &str) -> (p2a_validation::Case, Vec<f64>) {
        let (case, ds) = crate::test_support::load_case(case_id);
        let x = numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0;
        (case, x)
    }

    fn fit(x: &[f64], p: usize, q: usize, include_mean: bool, dist: &str) -> GarchResult {
        let mut c = GarchConfig::default();
        c.p = p;
        c.q = q;
        c.include_mean = include_mean;
        c.distribution = dist.parse().unwrap(); // String before the fix, GarchDistribution after
        garch(x, Some(c)).unwrap()
    }

    /// R's names [mu], omega, alpha1..p, beta1..q.
    fn names(p: usize, q: usize, include_mean: bool) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        if include_mean {
            v.push("mu".into());
        }
        v.push("omega".into());
        v.extend((1..=p).map(|i| format!("alpha{i}")));
        v.extend((1..=q).map(|j| format!("beta{j}")));
        v
    }

    /// Estimates in R's order from the fields that exist before and after the fix.
    fn estimates(r: &GarchResult, include_mean: bool) -> Vec<f64> {
        let mut v = Vec::new();
        if include_mean {
            v.push(r.mu);
        }
        v.push(r.omega);
        v.extend(&r.alpha);
        v.extend(&r.beta);
        v
    }

    /// R (garch_normal_t): GARCH(1,1) with mean and normal errors on r_norm (n = 1500, variance ~1e-4).
    /// Fails before: the R port of the old estimator (old_emul.R) gives mu 0.4340, omega 8.886e5,
    /// alpha = beta = 0 and logLik -11651.47 after 500 fixed-step iterations, with the warm-up
    /// h = omega / (1 - persistence) instead of mean(e^2) (TS-11).
    #[test]
    fn test_validate_garch_normal() {
        let (case, x) = returns("garch_normal_t", "r_norm");
        let r = fit(&x, 1, 1, true, "norm");
        let est = estimates(&r, true);
        Check::new(&case)
            .named("norm_par_tight", &names(1, 1, true), &est, PAR)
            .scalar("norm_loglik_tight", r.log_likelihood, Tol::stat(N))
            .scalar("norm_aic", r.aic, Tol::stat(N))
            .scalar("norm_bic", r.bic, Tol::stat(N))
            .vector("norm_h_first", &r.conditional_variance[..3], PAR)
            .scalar(
                "norm_h_last_tight",
                *r.conditional_variance.last().unwrap(),
                PAR,
            )
            .scalar("norm_uncond_var_tight", r.unconditional_variance, PAR)
            // Appendix A.31 (optim, finite-difference gradients): 1.7e-6 short of the optimum
            .vector("norm_par", &est, Tol::new(1e-5, 0.0))
            .scalar("norm_loglik", r.log_likelihood, Tol::stat(N))
            .finish();
        assert!(r.converged, "GARCH(1,1) on r_norm must converge");
        assert_eq!(r.n_params, 4);
        assert!((r.persistence - (r.alpha[0] + r.beta[0])).abs() <= 1e-15);
        // R: half-life 47.693054313217765 periods
        assert!(
            (r.half_life - 47.693054313217765).abs() <= 1e-6 * 47.7,
            "{}",
            r.half_life
        );
    }

    /// R: GARCH(1,1) with fGarch's standardised-t errors on r_t, and the normal fit of the same series.
    /// Fails before: `distribution = "std"` was ignored (TS-32), so the t fit was the old Gaussian
    /// gradient ascent (R port: omega 1.662e7, alpha = beta = 0, logLik -13847.97, "converged" after 3
    /// iterations) and identical to the normal one.
    #[test]
    fn test_validate_garch_student_t() {
        let (case, x) = returns("garch_normal_t", "r_t");
        let t = fit(&x, 1, 1, true, "std");
        let g = fit(&x, 1, 1, true, "norm");
        let core = ["mu", "omega", "alpha1", "beta1"];
        Check::new(&case)
            .named_subset(
                "std_par_tight",
                &core,
                &names(1, 1, true),
                &estimates(&t, true),
                PAR,
            )
            .scalar("std_loglik_tight", t.log_likelihood, Tol::stat(N))
            .scalar("std_aic", t.aic, Tol::stat(N))
            .scalar("std_bic", t.bic, Tol::stat(N))
            .named(
                "norm_on_t_par_tight",
                &names(1, 1, true),
                &estimates(&g, true),
                PAR,
            )
            .scalar("norm_on_t_loglik_tight", g.log_likelihood, Tol::stat(N))
            .scalar("norm_on_t_aic", g.aic, Tol::stat(N))
            .scalar("norm_on_t_bic", g.bic, Tol::stat(N))
            .finish();
        assert_eq!((t.n_params, g.n_params), (5, 4));
        assert!(t.converged && g.converged);
        // phantom-option pair (design section 3.2): R distinguishes the two distributions, so must p2a
        assert!(
            (t.omega - g.omega).abs() > 0.05 * g.omega
                && (t.log_likelihood - g.log_likelihood) > 40.0
        );
    }

    /// R: SEs sqrt(diag((-H)^-1)), H the Hessian of the log-likelihood in the natural parameters
    /// (numDeriv::jacobian of the analytic gradient); z = estimate / SE; p = 2 Phi(-|z|) as fGarch and
    /// tseries. Fails before: diagonal-only second differences at the degenerate estimate, with the
    /// fabricated SE = 0.1 |param| wherever the curvature was not negative (R port: mu and omega t =
    /// 10.000, '***') (TS-12). The diagonal rule is 2.6x-5.8x too small even at R's optimum (finding TS-12).
    #[test]
    fn test_validate_garch_se() {
        let (case, x) = returns("garch_normal_t", "r_norm");
        let r = fit(&x, 1, 1, true, "norm");
        let (_, xt) = returns("garch_normal_t", "r_t");
        let g = fit(&xt, 1, 1, true, "norm");
        let nm = names(1, 1, true);
        Check::new(&case)
            .named("norm_se_tight", &nm, &r.std_errors, SE)
            .named("norm_z_tight", &nm, &r.t_stats, SE)
            .named("norm_p_tight", &nm, &r.p_values, Tol::new(1e-5, 1e-12))
            .named("norm_on_t_se_tight", &nm, &g.std_errors, SE)
            .finish();
        let stars: Vec<&str> = r.significance.iter().map(|s| s.stars()).collect();
        assert_eq!(stars, ["", "*", "***", "***"]);
    }

    /// Convention 5: the transformed problem is scale free, so data times 1e-6 and 1e6 converge in the
    /// same number of iterations to the scaled solution (omega c^2, mu c, logLik - n ln c). Fails before:
    /// the fixed 0.01 step and the absolute 1e-8 likelihood change diverged at every scale.
    #[test]
    fn test_validate_garch_scale_invariance() {
        let (case, x) = returns("garch_normal_t", "r_norm");
        let base = fit(&x, 1, 1, true, "norm");
        for c in [1e-6_f64, 1e6] {
            let xc: Vec<f64> = x.iter().map(|v| v * c).collect();
            let r = fit(&xc, 1, 1, true, "norm");
            assert!(r.converged, "scale {c}");
            assert!(
                r.iterations.abs_diff(base.iterations) <= 2,
                "scale {c}: {} vs {} iterations",
                r.iterations,
                base.iterations
            );
            let back = [r.mu / c, r.omega / (c * c), r.alpha[0], r.beta[0]];
            Check::new(&case)
                .named("norm_par_tight", &names(1, 1, true), &back, PAR)
                .scalar(
                    "norm_loglik_tight",
                    r.log_likelihood + N as f64 * c.ln(),
                    Tol::stat(N),
                )
                .finish();
        }
    }

    /// TS-19 (with the estimator's parameter vector): one row per estimated parameter, named from
    /// include_mean and the orders, each with its own SE. Fails before: with include_mean = false the
    /// table started with a "mu" row carrying omega's SE and lost the beta1 row; GARCH(3,2) was labelled
    /// alpha1, alpha2, alpha2.
    #[test]
    fn test_validate_garch_display_rows() {
        let (case, y) = returns("garch_normal_t", "r_norm_dm");
        let r = fit(&y, 1, 1, false, "norm");
        let rows = table_rows(&r.to_string());
        let row_names: Vec<&str> = rows.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(row_names, ["omega", "alpha1", "beta1"]);
        for (name, _, se) in &rows {
            let want = case.named("zm_se_tight", name);
            assert!(
                (se - want).abs() <= 5e-4 * want,
                "{name}: printed SE {se}, R {want}"
            );
        }
        let (_, x) = returns("garch_normal_t", "r_g21");
        let r = fit(&x, 3, 2, true, "norm");
        let row_names: Vec<String> = table_rows(&r.to_string())
            .into_iter()
            .map(|(n, _, _)| n)
            .collect();
        assert_eq!(
            row_names,
            [
                "mu", "omega", "alpha1", "alpha2", "alpha3", "beta1", "beta2"
            ]
        );
    }

    /// Rows of the coefficient table as printed: (name, estimate, SE).
    fn table_rows(text: &str) -> Vec<(String, f64, f64)> {
        let mut rows = Vec::new();
        let mut inside = false;
        for line in text.lines() {
            if line.starts_with("Parameter") {
                inside = true;
                continue;
            }
            if !inside {
                continue;
            }
            if line.starts_with('-') {
                if rows.is_empty() {
                    continue;
                }
                break;
            }
            let tok: Vec<&str> = line.split_whitespace().collect();
            rows.push((
                tok[0].to_string(),
                tok[1].parse().unwrap(),
                tok[2].parse().unwrap(),
            ));
        }
        rows
    }

    fn generate_garch_data(n: usize, omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut data = Vec::with_capacity(n);
        let mut sigma2: f64 = omega / (1.0 - alpha - beta);
        let mut epsilon: f64 = 0.0;

        // Simple deterministic pseudo-random for reproducibility
        let mut hasher = DefaultHasher::new();
        for i in 0..n {
            i.hash(&mut hasher);
            let seed = hasher.finish();
            let z: f64 = ((seed % 10000) as f64 / 5000.0 - 1.0) * 1.5; // pseudo-normal

            sigma2 = omega + alpha * epsilon.powi(2) + beta * sigma2;
            epsilon = sigma2.sqrt() * z;
            data.push(epsilon);
        }

        data
    }

    #[test]
    fn test_garch_rejects_orders_at_or_above_n() {
        // Orders >= n must error rather than panic in the warm-up indexing.
        let data = generate_garch_data(25, 0.0001, 0.1, 0.8);
        let config = GarchConfig {
            p: 30,
            q: 1,
            ..Default::default()
        };
        assert!(matches!(
            garch(&data, Some(config)),
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    #[test]
    fn test_garch_rejects_excessive_orders() {
        let data = generate_garch_data(200, 0.0001, 0.1, 0.8);
        let config = GarchConfig {
            p: 25,
            q: 25,
            ..Default::default()
        };
        assert!(matches!(
            garch(&data, Some(config)),
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    #[test]
    fn test_garch_insufficient_data() {
        let data = vec![0.01; 10];
        let result = garch(&data, None);

        assert!(result.is_err());
        match result {
            Err(EconError::InsufficientData { .. }) => {}
            Err(e) => panic!("Expected InsufficientData error, got {:?}", e),
            Ok(_) => panic!("Expected error for short data"),
        }
    }

    /// R: the t fit's shape and all five SEs; the result reports the distribution.
    #[test]
    fn test_validate_garch_student_t_shape() {
        let (case, x) = returns("garch_normal_t", "r_t");
        let t = fit(&x, 1, 1, true, "std");
        Check::new(&case)
            .named("std_par_tight", &t.param_names, &t.estimates(), PAR)
            .named("std_se_tight", &t.param_names, &t.std_errors, SE)
            .vector("std_par", &t.estimates(), Tol::new(1e-5, 0.0)) // Appendix optim value, 4.9e-7 off
            .finish();
        assert_eq!(t.distribution, GarchDistribution::StudentT);
        assert_eq!(t.param_names, ["mu", "omega", "alpha1", "beta1", "shape"]);
        assert_eq!(t.shape, t.estimates().last().copied());
        let text = t.to_string();
        assert!(
            text.contains("Conditional distribution: std") && text.contains("\nshape "),
            "{text}"
        );
        // the shape adds a parameter to the n > n_params rule: 23 parameters need 24 observations
        let x23 = generate_garch_data(23, 0.0001, 0.1, 0.8);
        let mut c = GarchConfig::default();
        c.p = 10;
        c.q = 10;
        c.distribution = GarchDistribution::StudentT;
        assert!(matches!(
            garch(&x23, Some(c)),
            Err(EconError::InsufficientData {
                required: 24,
                provided: 23,
                ..
            })
        ));
    }

    /// R: the covariance of the estimates (inverse observed information); omega and beta are strongly
    /// correlated, which the diagonal-only SE of TS-12 ignored.
    #[test]
    fn test_validate_garch_vcov() {
        let (case, x) = returns("garch_normal_t", "r_norm");
        let r = fit(&x, 1, 1, true, "norm");
        let k = r.vcov.len();
        assert_eq!(k, 4);
        let col_major: Vec<f64> = (0..k)
            .flat_map(|c| (0..k).map(move |i| (i, c)))
            .map(|(i, c)| r.vcov[i][c])
            .collect();
        Check::new(&case)
            .vector("norm_vcov_tight", &col_major, SE)
            .finish();
        for i in 0..k {
            assert!((r.vcov[i][i].sqrt() - r.std_errors[i]).abs() <= 1e-14 * r.std_errors[i]);
        }
        let corr = r.vcov[1][3] / (r.vcov[1][1] * r.vcov[3][3]).sqrt();
        assert!(
            (corr - -0.7115468641459799).abs() <= 1e-6,
            "corr(omega, beta1) = {corr}"
        ); // R
    }

    /// R: GARCH(1,2) on r_norm, GARCH(2,1) on r_g21 (a GARCH(2,1) series), ARCH(1) on r_norm and the
    /// zero-mean GARCH(1,1) on the demeaned r_norm_dm; h_1..h_3 pin the mean(e^2) start of the first
    /// max(p, q) periods (equal h_1 = h_2 for m = 2).
    #[test]
    fn test_validate_garch_orders() {
        let (case, rn) = returns("garch_normal_t", "r_norm");
        let (_, r21) = returns("garch_normal_t", "r_g21");
        let (_, dm) = returns("garch_normal_t", "r_norm_dm");
        let fits = [
            ("g12", fit(&rn, 1, 2, true, "norm"), 3),
            ("g21", fit(&r21, 2, 1, true, "norm"), 3),
            ("arch1", fit(&rn, 1, 0, true, "norm"), 0),
            ("zm", fit(&dm, 1, 1, false, "norm"), 0),
        ];
        let mut check = Check::new(&case);
        for (key, r, nh) in &fits {
            check
                .named(
                    &format!("{key}_par_tight"),
                    &r.param_names,
                    &r.estimates(),
                    PAR,
                )
                .named(
                    &format!("{key}_se_tight"),
                    &r.param_names,
                    &r.std_errors,
                    SE,
                )
                .scalar(
                    &format!("{key}_loglik_tight"),
                    r.log_likelihood,
                    Tol::stat(N),
                )
                .scalar(&format!("{key}_aic"), r.aic, Tol::stat(N))
                .scalar(&format!("{key}_bic"), r.bic, Tol::stat(N));
            if *nh > 0 {
                check.vector(
                    &format!("{key}_h_first"),
                    &r.conditional_variance[..*nh],
                    PAR,
                );
            }
            assert!(
                r.converged && r.at_bound.is_empty() && r.warnings.is_empty(),
                "{key}: {:?}",
                r.warnings
            );
        }
        check.finish();
        assert_eq!(fits[3].1.param_names, ["omega", "alpha1", "beta1"]);
        assert_eq!(fits[3].1.mu, 0.0);
    }

    /// R: GARCH(2,1) on the GARCH(1,1) series has its maximum on alpha2 = 0 (dl/dalpha2 = -30.496 there);
    /// p2a stops at a tiny alpha2, flags it, and the other parameters and SEs are R's restricted ones.
    #[test]
    fn test_validate_garch_bound() {
        let (case, x) = returns("garch_normal_t", "r_norm");
        let r = fit(&x, 2, 1, true, "norm");
        let free = ["mu", "omega", "alpha1", "beta1"];
        Check::new(&case)
            .named_subset(
                "g21b_par_tight",
                &free,
                &r.param_names,
                &r.estimates(),
                Tol::new(1e-6, 0.0),
            )
            .named_subset("g21b_se_tight", &free, &r.param_names, &r.std_errors, SE)
            .scalar("g21b_loglik_tight", r.log_likelihood, Tol::stat(N))
            .finish();
        assert!(r.alpha[1] < 1e-6, "alpha2 = {}", r.alpha[1]);
        assert_eq!(r.at_bound, ["alpha2"]);
        assert!(r.std_errors[3].is_nan() && r.t_stats[3].is_nan() && r.p_values[3].is_nan());
        assert!(r.vcov[3].iter().all(|v| v.is_nan()));
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("alpha2 = ") && w.contains("at its lower bound 0")),
            "{:?}",
            r.warnings
        );
        assert!(r.converged);
    }

    /// The bound rule: a variance coefficient at or below 1e-4 of alpha + beta, or a shape above
    /// GARCH_SHAPE_MAX - 1e-3 (GARCH_SHAPE_MAX - 2) = 199.802.
    #[test]
    fn test_garch_bound_rule() {
        let x = generate_garch_data(200, 0.0001, 0.1, 0.8);
        let s = Spec::new(&x, 2, 1, true, GarchDistribution::Normal);
        assert_eq!(
            at_bounds(&s, &[3e-4, 1.5e-6, 0.073, 9.0e-10, 0.9126]),
            [false, false, false, true, false]
        );
        assert_eq!(
            at_bounds(&s, &[3e-4, 1.5e-6, 0.073, 2.0e-4, 0.9126]),
            [false; 5]
        ); // 2e-4 > 9.858e-5
        let s = Spec::new(&x, 1, 1, true, GarchDistribution::StudentT);
        assert_eq!(
            at_bounds(&s, &[3e-4, 1.5e-6, 0.07, 0.9, 199.9]),
            [false, false, false, false, true]
        );
        assert_eq!(at_bounds(&s, &[3e-4, 1.5e-6, 0.07, 0.9, 29.0]), [false; 5]);
    }

    /// TS-12: where the observed information is not positive definite the SEs are NaN with a warning.
    /// At omega = 10 s^2 the log-likelihood is convex in omega (R: eigen(-H) = 1463693.09, 838.47, -9.615,
    /// -5.7589e8); the old rule printed SE(omega) = 0.1 omega = 1.0232e-4 and t = 10, '***'.
    #[test]
    fn test_garch_hessian_not_pd_gives_nan_se() {
        let (_case, x) = returns("garch_normal_t", "r_norm");
        let s = Spec::new(&x, 1, 1, true, GarchDistribution::Normal);
        let nat = [s.xbar, 10.0 * s.s2, 0.05, 0.05];
        let inf = inference(&s, &x, &nat, &[false; 4]).unwrap();
        assert!(
            inf.std_errors.iter().all(|v| v.is_nan()),
            "{:?}",
            inf.std_errors
        );
        assert!(inf.vcov.iter().flatten().all(|v| v.is_nan()));
        assert!(
            inf.warning
                .as_deref()
                .is_some_and(|w| w.contains("not positive definite")),
            "{:?}",
            inf.warning
        );
    }

    /// R10: stopping at the iteration limit is reported, never "converged". With max_iter = 1 the t fit
    /// stops at scaled gradient 1.4e-2 (R emulation, maxit.R).
    #[test]
    fn test_garch_reports_non_convergence() {
        let (_case, x) = returns("garch_normal_t", "r_t");
        let mut c = GarchConfig::default();
        c.distribution = GarchDistribution::StudentT;
        c.max_iter = 1;
        let r = garch(&x, Some(c)).unwrap();
        assert!(!r.converged && !r.convergence.converged);
        assert_eq!(r.convergence.max_iterations, 1 + 3);
        assert!(r.convergence.final_change > r.convergence.tolerance);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("GARCH did not converge after")),
            "{:?}",
            r.warnings
        );
        let text = r.to_string();
        assert!(
            text.contains("Converged: no (") && text.contains("Warning: GARCH did not converge"),
            "{text}"
        );
    }

    #[test]
    fn test_garch_distribution_parses_aliases_and_rejects_unknown() {
        GarchDistribution::choices().validate().unwrap();
        for s in ["norm", "Normal", "GAUSSIAN", " normal "] {
            assert_eq!(
                s.parse::<GarchDistribution>().unwrap(),
                GarchDistribution::Normal,
                "{s}"
            );
        }
        for s in [
            "std",
            "t",
            "T",
            "Student-t",
            "student t",
            "student_t",
            "studentt",
        ] {
            assert_eq!(
                s.parse::<GarchDistribution>().unwrap(),
                GarchDistribution::StudentT,
                "{s}"
            );
        }
        match "sstd".parse::<GarchDistribution>() {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("distribution")
                        && message.contains("norm")
                        && message.contains("std"),
                    "{message}"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        assert_eq!(GarchDistribution::StudentT.to_string(), "std");
        assert_eq!(
            serde_json::to_string(&GarchDistribution::StudentT).unwrap(),
            "\"std\""
        );
        let c: GarchConfig = serde_json::from_str(
            r#"{"p":1,"q":1,"include_mean":true,"max_iter":500,"tolerance":1e-8,"distribution":"normal"}"#,
        )
        .unwrap();
        assert_eq!(c.distribution, GarchDistribution::Normal);
    }

    /// run_garch is garch() with p, q, include_mean and the defaults.
    #[test]
    fn test_run_garch_delegates() {
        let (_case, x) = returns("garch_normal_t", "r_norm");
        let a = run_garch(&x, Some(1), Some(2), Some(false)).unwrap();
        let mut c = GarchConfig::default();
        c.q = 2;
        c.include_mean = false;
        let b = garch(&x, Some(c)).unwrap();
        assert_eq!(a.estimates(), b.estimates());
        assert_eq!(a.log_likelihood.to_bits(), b.log_likelihood.to_bits());
        assert_eq!(a.distribution, GarchDistribution::Normal);
    }

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        let token = crate::resource::CancelToken::new();
        token.cancel();
        crate::resource::with_budget(crate::resource::Budget::new(None, token), f)
    }

    #[test]
    fn garch_fit_stops_when_the_budget_is_cancelled() {
        // Before the fix the optimiser never looked at the budget: under a
        // cancelled budget it ran all its iterations and returned Ok, so the
        // server's deadline could not stop a high-order fit on a long series.
        let data = generate_garch_data(500, 0.0001, 0.1, 0.8);
        let config = GarchConfig {
            p: 2,
            q: 2,
            ..Default::default()
        };
        assert!(matches!(
            cancelled(|| garch(&data, Some(config.clone()))),
            Err(EconError::Cancelled(_))
        ));
        // Without a budget nothing changes.
        assert!(garch(&data, Some(config)).is_ok());
    }

    #[test]
    fn garch_likelihood_and_inference_passes_honour_cancellation() {
        // At p = q = 20 one likelihood pass over a long series takes seconds, so each pass
        // checks the budget as it goes, and the Hessian checks once per column.
        let data = generate_garch_data(200, 0.0001, 0.1, 0.8);
        let s = Spec::new(&data, 1, 1, true, GarchDistribution::Normal);
        let nat = [0.0, 1e-4, 0.1, 0.8];
        let fixed = [false; 4];
        assert!(matches!(
            cancelled(|| loglik_grad(&s, &data, &nat)),
            Err(EconError::Cancelled(_))
        ));
        assert!(matches!(
            cancelled(|| hessian_free(&s, &data, &nat, &fixed)),
            Err(EconError::Cancelled(_))
        ));
        assert!(matches!(
            cancelled(|| inference(&s, &data, &nat, &fixed)),
            Err(EconError::Cancelled(_))
        ));
        let (ll, g) = loglik_grad(&s, &data, &nat).unwrap().unwrap();
        assert!(ll.is_finite() && g.len() == 4);
        assert!(hessian_free(&s, &data, &nat, &fixed).unwrap().is_some());
        assert!(inference(&s, &data, &nat, &fixed).is_ok());
    }

    /// The optimiser checks the budget every iteration and never takes a budget error for an
    /// infeasible trial: a token cancelled during the 12th objective evaluation stops the run there,
    /// while the free run needs more evaluations than that.
    #[test]
    fn garch_bfgs_checkpoints_each_iteration() {
        // 1 evaluation at the start and 8 for its Hessian, so the 12th is inside the BFGS loop
        const CANCEL_AT: usize = 12;
        let (_case, x) = returns("garch_normal_t", "r_norm");
        let s = Spec::new(&x, 1, 1, true, GarchDistribution::Normal);
        let th0 = to_theta(&s, &starts(&s)[0]);
        let mut free_evals = 0;
        let mut obj = |th: &[f64]| {
            free_evals += 1;
            objective(&s, &x, th)
        };
        assert!(bfgs(&mut obj, th0.clone(), 500, 1e-8).unwrap().is_some());
        assert!(
            free_evals > CANCEL_AT,
            "the free run needs {free_evals} evaluations"
        );

        let token = crate::resource::CancelToken::new();
        let mut evals = 0;
        let r =
            crate::resource::with_budget(crate::resource::Budget::new(None, token.clone()), || {
                let mut obj = |th: &[f64]| {
                    evals += 1;
                    if evals == CANCEL_AT {
                        token.cancel();
                    }
                    objective(&s, &x, th)
                };
                bfgs(&mut obj, th0, 500, 1e-8).map(|o| o.map(|o| o.iterations))
            });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{r:?}");
        assert!(
            evals <= CANCEL_AT,
            "{evals} evaluations after the cancellation"
        );
    }

    /// TS-20: the conditional likelihood needs more observations than parameters. Fails before: with
    /// include_mean (22 parameters, 21 rows) `n - n_params` underflowed (debug: "attempt to subtract with
    /// overflow"; release: df ~1.8e19 and p-values 1); without the mean (21 parameters) df was 0 and every
    /// p-value NaN, returned as Ok.
    #[test]
    fn test_garch_rejects_n_le_params() {
        let x = generate_garch_data(21, 0.0001, 0.1, 0.8);
        for (include_mean, required) in [(true, 23), (false, 22)] {
            let mut c = GarchConfig::default();
            c.p = 10;
            c.q = 10;
            c.include_mean = include_mean;
            match garch(&x, Some(c)) {
                Err(EconError::InsufficientData {
                    required: r,
                    provided,
                    context,
                }) => {
                    assert_eq!((r, provided), (required, 21), "{context}");
                    assert!(
                        context.contains("GARCH(10,10)") && context.contains("parameters"),
                        "{context}"
                    );
                }
                other => panic!(
                    "include_mean = {include_mean}: expected InsufficientData, got {other:?}"
                ),
            }
        }
        // one observation more than parameters is accepted
        let x22 = generate_garch_data(22, 0.0001, 0.1, 0.8);
        let mut c = GarchConfig::default();
        c.p = 10;
        c.q = 10;
        c.include_mean = false;
        assert!(garch(&x22, Some(c)).is_ok());
    }

    /// R3/G1: degenerate settings and data are errors that name the problem. Fails before: each returned Ok
    /// (p = 0 fitted a GARCH(0,1); NaN data propagated into the likelihood; a constant series gave log(0)).
    #[test]
    fn test_garch_rejects_degenerate_inputs() {
        let x = generate_garch_data(200, 0.0001, 0.1, 0.8);
        let expect_spec = |c: GarchConfig, data: &[f64], needle: &str| match garch(data, Some(c)) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains(needle), "{message}")
            }
            other => panic!("expected InvalidSpecification naming '{needle}', got {other:?}"),
        };
        let mut c = GarchConfig::default();
        c.p = 0;
        expect_spec(c, &x, "ARCH order p");
        let mut c = GarchConfig::default();
        c.max_iter = 0;
        expect_spec(c, &x, "max_iter");
        for tol in [0.0, -1e-8, f64::NAN, f64::INFINITY] {
            let mut c = GarchConfig::default();
            c.tolerance = tol;
            expect_spec(c, &x, "tolerance");
        }
        expect_spec(GarchConfig::default(), &[0.01; 50], "constant");
        let mut y = x.clone();
        y[7] = f64::NAN;
        y[9] = f64::NAN;
        assert!(matches!(
            garch(&y, None),
            Err(EconError::NullValues { count: 2, .. })
        ));
        let mut y = x.clone();
        y[3] = f64::INFINITY;
        assert!(matches!(
            garch(&y, None),
            Err(EconError::NonFiniteValues { count: 1, .. })
        ));
    }

    #[test]
    fn garch_forecast_refuses_absurd_horizon() {
        let result = garch(&generate_garch_data(200, 0.0001, 0.1, 0.8), None).unwrap();
        assert!(matches!(
            garch_forecast(&result, 1usize << 50),
            Err(EconError::ResourceLimit(_))
        ));
        assert_eq!(garch_forecast(&result, 10).unwrap().0.len(), 10);
    }
}
