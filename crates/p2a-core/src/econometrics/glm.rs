//! Generalized linear models by iteratively reweighted least squares: the one GLM that
//! Tier 3 estimators share (propensity scores, TMLE fluctuations, Poisson and binomial
//! models).
//!
//! [`fit_glm`] fits the Gaussian, binomial, Poisson, quasi-binomial and quasi-Poisson
//! families with the identity, logit, probit, complementary log-log and log links, prior
//! weights, offsets and start values, by Fisher scoring as R's `stats::glm.fit` (R 4.3.3):
//! the same family and link functions (including R's clamps of the logit, probit,
//! cloglog and log inverse links), `mustart`, step halving on a non-finite deviance or
//! invalid fitted values, and R's warning texts. Each IRLS step is solved by the pivoted
//! QR of [`crate::linalg::solve::Qr`]; aliased columns get `NaN` coefficients and SEs.
//! Standard errors and dispersion follow `summary.glm`, AIC and the log-likelihood
//! `family$aic` and `logLik.glm`, and the null deviance `glm()` (refitted when the model
//! has an offset and an intercept).
//!
//! Checked against `stats::glm` at the solution on the R case `linalg_qr_glm`
//! (`validation/linalg/solve.md`); agreement with `glm.fit`'s path (iteration counts,
//! intermediate deviances) is not a goal.
//!
//! Where this differs from `glm.fit` / `glm.control`:
//! * Defaults: `epsilon` 1e-10 (R 1e-8) and `max_iter` 100 (R 25): about one more IRLS
//!   iteration, closer to the MLE.
//! * Convergence: `|dev - devold| / (|dev| + 0.1 * D0 / n_ok) < epsilon`, where `D0` is
//!   the deviance of the weighted mean of `y` and `n_ok` the number of positive weights
//!   (R's 0.1 when `D0 / n_ok` is 0 or not finite). `glm.fit` uses an absolute 0.1, which
//!   makes the rule depend on the scale of `y` and of the weights: a Gaussian log-link fit
//!   of a small-scale response stops after one step, away from the MLE.
//! * The aliasing tolerance of the IRLS QR is fixed at 1e-11, R's value at the default
//!   `glm.control` (`min(1e-7, epsilon / 1000)`), independent of `epsilon`.
//! * The Gaussian AIC counts the rows with positive weight (`logLik.lm`'s convention);
//!   `glm` gives `+Inf` when a weight is 0.
//! * The model has an intercept when some column of `x` is identically 1 (R reads it from
//!   the formula). When `x` is that single column, the null model is the fitted model and
//!   no refit is made.
//! * Errors where `glm.fit` warns and returns an undefined fit: non-finite IRLS
//!   coefficients ([`EconError::DivergenceError`]) and no positive weight
//!   ([`EconError::InsufficientData`]).
//! * The separation flag and its warning also cover the quasi-binomial family, and the
//!   "fitted rates numerically 0" warning the quasi-Poisson family.
//! * [`GlmSpec::coef_tol`] optionally keeps iterating after the deviance rule until the
//!   coefficients settle; [`fit_propensity`] uses it, so probit (and cloglog) propensity
//!   scores reach the MLE to about 1e-10 instead of the deviance rule's 2e-6.
//!
//! ```
//! use ndarray::array;
//! use p2a_core::econometrics::glm::{fit_glm, GlmSpec};
//! let x = array![[1.0, -1.2], [1.0, -0.3], [1.0, 0.4], [1.0, 1.1], [1.0, 1.9], [1.0, -0.7], [1.0, 0.2], [1.0, 0.9]];
//! let y = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
//! let fit = fit_glm(x.view(), y.view(), &GlmSpec::logit()).unwrap();
//! assert!(fit.converged && fit.aliased.is_empty());
//! assert_eq!(fit.coefficients.len(), 2);
//! ```

use crate::errors::{EconError, EconResult};
use crate::linalg::solve::Qr;
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, relative_change_vec};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;

/// GLM family (`stats::family`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmFamily {
    /// `gaussian()`: variance 1, dispersion estimated.
    Gaussian,
    /// `binomial()`: `y` a proportion in [0, 1] (with prior weights as trials), dispersion 1.
    Binomial,
    /// `poisson()`: `y >= 0`, dispersion 1.
    Poisson,
    /// `quasibinomial()`: binomial variance, dispersion estimated, no AIC.
    QuasiBinomial,
    /// `quasipoisson()`: Poisson variance, dispersion estimated, no AIC.
    QuasiPoisson,
}

/// GLM link function (`stats::make.link`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmLink {
    /// Identity, `mu`.
    Identity,
    /// Logit, `log(mu / (1 - mu))`.
    Logit,
    /// Probit, `qnorm(mu)`.
    Probit,
    /// Complementary log-log, `log(-log(1 - mu))`.
    Cloglog,
    /// Log, `log(mu)`.
    Log,
}

/// What to fit: family, link, prior weights, offset, start values and IRLS control.
#[derive(Debug, Clone)]
pub struct GlmSpec<'a> {
    /// Error distribution (variance function and dispersion).
    pub family: GlmFamily,
    /// Link from the mean to the linear predictor.
    pub link: GlmLink,
    /// Prior weights (`>= 0`); rows with weight 0 are dropped from the fit.
    pub weights: Option<ArrayView1<'a, f64>>,
    /// Offset added to the linear predictor.
    pub offset: Option<ArrayView1<'a, f64>>,
    /// Start values of the coefficients (one per column of `x`).
    pub start: Option<ArrayView1<'a, f64>>,
    /// Convergence tolerance on the relative deviance change; default 1e-10 (R's
    /// `glm.control` default is 1e-8).
    pub epsilon: f64,
    /// Maximum IRLS iterations; default 100 (R's default `maxit` is 25).
    pub max_iter: usize,
    /// Optional refinement after the deviance rule is met: keep iterating (within
    /// `max_iter`) until the relative coefficient change
    /// ([`crate::traits::convergence::relative_change_vec`]) falls below this tolerance, or
    /// until a binomial linear predictor reaches its link's clamp (separation: the
    /// coefficients diverge and further steps no longer change the fitted values). The fit
    /// counts as converged once the deviance rule is met. The deviance rule alone leaves an
    /// `O(sqrt(epsilon))` coefficient error when the link is not canonical (Fisher scoring
    /// converges linearly), about 2e-6 relative for the probit at `epsilon` 1e-10. Default
    /// `None`; [`fit_propensity`] uses 1e-10.
    pub coef_tol: Option<f64>,
}

impl<'a> GlmSpec<'a> {
    /// `family` with `link`, no weights, offset or start, `epsilon` 1e-10, `max_iter` 100.
    pub fn new(family: GlmFamily, link: GlmLink) -> Self {
        GlmSpec {
            family,
            link,
            weights: None,
            offset: None,
            start: None,
            epsilon: 1e-10,
            max_iter: 100,
            coef_tol: None,
        }
    }

    /// Binomial with the logit link.
    pub fn logit() -> Self {
        Self::new(GlmFamily::Binomial, GlmLink::Logit)
    }

    /// Binomial with the probit link.
    pub fn probit() -> Self {
        Self::new(GlmFamily::Binomial, GlmLink::Probit)
    }

    /// Poisson with the log link.
    pub fn poisson() -> Self {
        Self::new(GlmFamily::Poisson, GlmLink::Log)
    }

    /// With prior weights.
    pub fn weights(mut self, w: ArrayView1<'a, f64>) -> Self {
        self.weights = Some(w);
        self
    }

    /// With an offset.
    pub fn offset(mut self, o: ArrayView1<'a, f64>) -> Self {
        self.offset = Some(o);
        self
    }
}

/// A fitted GLM. Coefficient-indexed arrays are in the column order of `x`; aliased
/// columns have `NaN` coefficients, SEs and `vcov` rows and columns. `NaN` is written as
/// JSON `null` and read back as `NaN`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmFit {
    /// Coefficients in the column order of `x`; `NaN` for aliased columns.
    #[serde(with = "nan_vec")]
    pub coefficients: Array1<f64>,
    /// `sqrt(diag(vcov))` (`summary.glm`).
    #[serde(with = "nan_vec")]
    pub std_errors: Array1<f64>,
    /// `dispersion * (R'R)^{-1}` from the QR of the last IRLS step.
    #[serde(with = "nan_mat")]
    pub vcov: Array2<f64>,
    /// Fitted means `mu`.
    #[serde(with = "nan_vec")]
    pub fitted: Array1<f64>,
    /// Linear predictor `eta` (including the offset).
    #[serde(with = "nan_vec")]
    pub linear_predictor: Array1<f64>,
    /// IRLS working weights of the last iteration (0 for rows not used).
    #[serde(with = "nan_vec")]
    pub working_weights: Array1<f64>,
    /// Residual deviance at the last iterate.
    pub deviance: f64,
    /// Deviance of the intercept-only model (plus the offset), or of the offset alone
    /// when `x` has no column of ones (`glm.fit`).
    pub null_deviance: f64,
    /// `logLik.glm`; `NaN` for the quasi families.
    #[serde(with = "nan_f64")]
    pub log_likelihood: f64,
    /// `family$aic + 2 rank`; `NaN` for the quasi families.
    #[serde(with = "nan_f64")]
    pub aic: f64,
    /// 1 for binomial and Poisson; Pearson estimate otherwise (`NaN` when no residual df).
    #[serde(with = "nan_f64")]
    pub dispersion: f64,
    /// Rows with positive weight minus rank.
    pub df_residual: usize,
    /// Numerical rank of the weighted design (pivoted QR, `tol` 1e-11 as `glm.fit`).
    pub rank: usize,
    /// Original indices of the aliased columns, ascending.
    pub aliased: Vec<usize>,
    /// The relative deviance change fell below `epsilon` within `max_iter` iterations.
    pub converged: bool,
    /// IRLS iterations of the main fit.
    pub iterations: usize,
    /// `max_iter` of the spec (0 when read from JSON written before this field existed).
    #[serde(default)]
    pub max_iter: usize,
    /// `epsilon` of the spec (0 when read from JSON written before this field existed).
    #[serde(default)]
    pub epsilon: f64,
    /// The last iteration had to halve its step.
    pub boundary: bool,
    /// Binomial families: some fitted probability is within `10 eps` of 0 or 1.
    pub separation: bool,
    /// R's warning texts, deduplicated.
    pub warnings: Vec<String>,
}

/// Propensity scores from a binary-response GLM.
#[derive(Debug, Clone)]
pub struct PropensityFit {
    /// Fitted probabilities (`fit.fitted`).
    pub scores: Array1<f64>,
    /// The underlying binomial GLM fit (coefficients, standard errors, convergence).
    pub fit: GlmFit,
}

const EPS: f64 = f64::EPSILON;
/// R's logit-link clamp (`family.c`, THRESH).
const LOGIT_THRESH: f64 = 30.0;
/// R's probit clamp: `-qnorm(.Machine$double.eps)`.
const PROBIT_THRESH: f64 = 8.125890664701906;
/// QR aliasing tolerance of each IRLS step: R's `min(1e-7, epsilon / 1000)` at the
/// default `glm.control`.
const GLM_QR_TOL: f64 = 1e-11;

fn spec_err(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

fn std_normal() -> Normal {
    Normal::standard()
}

impl GlmFamily {
    fn is_binomial(self) -> bool {
        matches!(self, GlmFamily::Binomial | GlmFamily::QuasiBinomial)
    }

    fn is_poisson(self) -> bool {
        matches!(self, GlmFamily::Poisson | GlmFamily::QuasiPoisson)
    }

    fn is_quasi(self) -> bool {
        matches!(self, GlmFamily::QuasiBinomial | GlmFamily::QuasiPoisson)
    }

    fn links(self) -> &'static [GlmLink] {
        match self {
            GlmFamily::Gaussian => &[GlmLink::Identity, GlmLink::Log],
            GlmFamily::Binomial | GlmFamily::QuasiBinomial => {
                &[GlmLink::Logit, GlmLink::Probit, GlmLink::Cloglog]
            }
            GlmFamily::Poisson | GlmFamily::QuasiPoisson => &[GlmLink::Log, GlmLink::Identity],
        }
    }

    fn variance(self, mu: f64) -> f64 {
        match self {
            GlmFamily::Gaussian => 1.0,
            GlmFamily::Binomial | GlmFamily::QuasiBinomial => mu * (1.0 - mu),
            GlmFamily::Poisson | GlmFamily::QuasiPoisson => mu,
        }
    }

    fn dev_resid(self, y: f64, mu: f64, w: f64) -> f64 {
        match self {
            GlmFamily::Gaussian => w * (y - mu) * (y - mu),
            GlmFamily::Binomial | GlmFamily::QuasiBinomial => {
                let ylogy = |y: f64, mu: f64| if y != 0.0 { y * (y / mu).ln() } else { 0.0 };
                2.0 * w * (ylogy(y, mu) + ylogy(1.0 - y, 1.0 - mu))
            }
            GlmFamily::Poisson | GlmFamily::QuasiPoisson => {
                if y > 0.0 {
                    2.0 * w * (y * (y / mu).ln() - (y - mu))
                } else {
                    2.0 * mu * w
                }
            }
        }
    }

    fn deviance(self, y: &[f64], mu: &Array1<f64>, w: &[f64]) -> f64 {
        (0..y.len())
            .map(|i| self.dev_resid(y[i], mu[i], w[i]))
            .sum()
    }

    fn valid_mu(self, mu: &Array1<f64>) -> bool {
        match self {
            GlmFamily::Gaussian => true,
            GlmFamily::Binomial | GlmFamily::QuasiBinomial => {
                mu.iter().all(|&m| m.is_finite() && m > 0.0 && m < 1.0)
            }
            GlmFamily::Poisson | GlmFamily::QuasiPoisson => {
                mu.iter().all(|&m| m.is_finite() && m > 0.0)
            }
        }
    }
}

impl GlmLink {
    fn linkfun(self, mu: f64) -> f64 {
        match self {
            GlmLink::Identity => mu,
            GlmLink::Logit => (mu / (1.0 - mu)).ln(),
            GlmLink::Probit => std_normal().inverse_cdf(mu),
            GlmLink::Cloglog => (-(1.0 - mu).ln()).ln(),
            GlmLink::Log => mu.ln(),
        }
    }

    fn linkinv(self, eta: f64) -> f64 {
        match self {
            GlmLink::Identity => eta,
            GlmLink::Logit => {
                let t = if eta < -LOGIT_THRESH {
                    EPS
                } else if eta > LOGIT_THRESH {
                    1.0 / EPS
                } else {
                    eta.exp()
                };
                t / (1.0 + t)
            }
            GlmLink::Probit => std_normal().cdf(eta.clamp(-PROBIT_THRESH, PROBIT_THRESH)),
            GlmLink::Cloglog => (-(-eta.exp()).exp_m1()).min(1.0 - EPS).max(EPS),
            GlmLink::Log => eta.exp().max(EPS),
        }
    }

    fn mu_eta(self, eta: f64) -> f64 {
        match self {
            GlmLink::Identity => 1.0,
            GlmLink::Logit => {
                if eta.abs() > LOGIT_THRESH {
                    EPS
                } else {
                    let e = eta.exp();
                    e / ((1.0 + e) * (1.0 + e))
                }
            }
            GlmLink::Probit => std_normal().pdf(eta).max(EPS),
            GlmLink::Cloglog => {
                let e = eta.min(700.0);
                (e.exp() * (-e.exp()).exp()).max(EPS)
            }
            GlmLink::Log => eta.exp().max(EPS),
        }
    }
}

/// How IRLS starts: from `start` coefficients, or from `mustart`.
enum Init<'a> {
    Start(ArrayView1<'a, f64>),
    Mustart(Array1<f64>),
}

/// The state IRLS leaves behind.
struct Irls {
    coef: Array1<f64>,
    eta: Array1<f64>,
    mu: Array1<f64>,
    dev: f64,
    /// QR of the last step; `None` for the empty model.
    qr: Option<Qr>,
    /// Working weights `w^2` of the last step (0 for rows not used).
    working_weights: Array1<f64>,
    converged: bool,
    boundary: bool,
    iterations: usize,
}

/// Family, link, data and control shared by the fit and the null-deviance refit.
struct Problem<'a> {
    family: GlmFamily,
    link: GlmLink,
    y: &'a [f64],
    w: &'a [f64],
    offset: &'a [f64],
    epsilon: f64,
    max_iter: usize,
    /// [`GlmSpec::coef_tol`].
    coef_tol: Option<f64>,
    /// Floor of the convergence denominator: 0.1 * D0 / n_ok (0.1 when that is 0 or not
    /// finite).
    floor: f64,
}

impl Problem<'_> {
    fn linkinv(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| self.link.linkinv(e))
    }

    fn deviance(&self, mu: &Array1<f64>) -> f64 {
        self.family.deviance(self.y, mu, self.w)
    }

    fn valid(&self, mu: &Array1<f64>) -> bool {
        self.family.valid_mu(mu)
    }

    /// A binomial linear predictor at or beyond its link's clamp: the fitted probability no
    /// longer moves (separation), so coefficient refinement stops.
    fn saturated(&self, eta: &Array1<f64>) -> bool {
        if !self.family.is_binomial() {
            return false;
        }
        let clamp = match self.link {
            GlmLink::Logit => LOGIT_THRESH,
            GlmLink::Probit => PROBIT_THRESH,
            _ => {
                let eps10 = 10.0 * EPS;
                return eta.iter().any(|&e| {
                    let m = self.link.linkinv(e);
                    m > 1.0 - eps10 || m < eps10
                });
            }
        };
        eta.iter().any(|e| e.abs() >= clamp)
    }

    /// `x b + offset`.
    fn eta(&self, x: &ArrayView2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let mut eta = x.dot(b);
        for (e, o) in eta.iter_mut().zip(self.offset) {
            *e += o;
        }
        eta
    }

    fn no_valid_coefficients(iteration: usize) -> EconError {
        EconError::DivergenceError {
            iteration,
            suggestion: "no valid set of coefficients has been found: please supply starting \
                         values"
                .into(),
        }
    }

    /// Fisher scoring as `glm.fit`, with this crate's convergence criterion.
    fn irls(
        &self,
        x: ArrayView2<f64>,
        init: Init<'_>,
        warnings: &mut Vec<String>,
    ) -> EconResult<Irls> {
        let (n, p) = x.dim();
        if p == 0 {
            // R's EMPTY model: eta = offset.
            let eta = Array1::from_vec(self.offset.to_vec());
            let mu = self.linkinv(&eta);
            if !self.valid(&mu) {
                return Err(spec_err("invalid fitted means in empty model".into()));
            }
            let working_weights = Array1::from_iter((0..n).map(|i| {
                let d = self.link.mu_eta(eta[i]);
                self.w[i] * d * d / self.family.variance(mu[i])
            }));
            let dev = self.deviance(&mu);
            return Ok(Irls {
                coef: Array1::zeros(0),
                eta,
                mu,
                dev,
                qr: None,
                working_weights,
                converged: true,
                boundary: false,
                iterations: 0,
            });
        }
        let (mut eta, mut coefold) = match init {
            Init::Start(s) => {
                let s = s.to_owned();
                (self.eta(&x, &s), Some(s))
            }
            Init::Mustart(m) => (m.mapv(|v| self.link.linkfun(v)), None),
        };
        let mut mu = self.linkinv(&eta);
        if !self.valid(&mu) {
            return Err(spec_err(
                "cannot find valid starting values: please specify some".into(),
            ));
        }
        let mut devold = self.deviance(&mu);
        let mut start = coefold.clone().unwrap_or_else(|| Array1::zeros(p));
        let mut converged = false;
        let mut boundary = false;
        let mut prev_change = f64::INFINITY;
        let mut last_qr = None;
        let mut working_weights = Array1::zeros(n);
        let mut iterations = 0;
        for iter in 1..=self.max_iter {
            resource::checkpoint()?;
            iterations = iter;
            let mut mu_eta = Array1::zeros(n);
            for i in 0..n {
                if self.w[i] > 0.0 {
                    let v = self.family.variance(mu[i]);
                    if !v.is_finite() {
                        return Err(EconError::Computation("glm.fit: NAs in V(mu)".into()));
                    }
                    if v == 0.0 {
                        return Err(EconError::Computation("glm.fit: 0s in V(mu)".into()));
                    }
                }
                mu_eta[i] = self.link.mu_eta(eta[i]);
            }
            let good: Vec<usize> = (0..n)
                .filter(|&i| self.w[i] > 0.0 && mu_eta[i] != 0.0)
                .collect();
            let ng = good.len();
            if ng == 0 {
                return Err(EconError::InsufficientData {
                    required: 1,
                    provided: 0,
                    context: format!("glm: no observations informative at iteration {iter}"),
                });
            }
            let len = resource::checked_len(&[p, ng], "glm IRLS workspace")?;
            let mut cols = resource::try_vec(len, 0.0, "glm IRLS workspace")?;
            let mut zw = resource::try_zeros1(ng, "glm IRLS workspace")?;
            let mut wt = resource::try_zeros1(ng, "glm IRLS workspace")?;
            for (k, &i) in good.iter().enumerate() {
                let d = mu_eta[i];
                let z = eta[i] - self.offset[i] + (self.y[i] - mu[i]) / d;
                let wk = (self.w[i] * d * d / self.family.variance(mu[i])).sqrt();
                wt[k] = wk;
                zw[k] = z * wk;
                for j in 0..p {
                    cols[j * ng + k] = x[[i, j]] * wk;
                }
            }
            let qr = Qr::from_columns(cols, ng, p, GLM_QR_TOL).map_err(|e| match e {
                EconError::InvalidSpecification { .. } => EconError::DivergenceError {
                    iteration: iter,
                    suggestion: "glm.fit: non-finite working weights or responses; supply \
                                 starting values or check for separation"
                        .into(),
                },
                other => other,
            })?;
            let b = qr.coefficients_zeroed(zw.view())?;
            if !b.iter().all(|v| v.is_finite()) {
                return Err(EconError::DivergenceError {
                    iteration: iter,
                    suggestion: "glm.fit: non-finite coefficients; supply starting values or \
                                 check for separation"
                        .into(),
                });
            }
            working_weights.fill(0.0);
            for (k, &i) in good.iter().enumerate() {
                working_weights[i] = wt[k] * wt[k];
            }
            last_qr = Some(qr);
            start = b;
            eta = self.eta(&x, &start);
            mu = self.linkinv(&eta);
            let mut dev = self.deviance(&mu);
            boundary = false;
            if !dev.is_finite() {
                let Some(old) = &coefold else {
                    return Err(Self::no_valid_coefficients(iter));
                };
                warnings.push("step size truncated due to divergence".into());
                let mut ii = 1;
                while !dev.is_finite() {
                    if ii > self.max_iter {
                        return Err(EconError::DivergenceError {
                            iteration: iter,
                            suggestion: "inner loop 1; cannot correct step size".into(),
                        });
                    }
                    ii += 1;
                    start = (&start + old) / 2.0;
                    eta = self.eta(&x, &start);
                    mu = self.linkinv(&eta);
                    dev = self.deviance(&mu);
                }
                boundary = true;
            }
            if !self.valid(&mu) {
                let Some(old) = &coefold else {
                    return Err(Self::no_valid_coefficients(iter));
                };
                warnings.push("step size truncated: out of bounds".into());
                let mut ii = 1;
                while !self.valid(&mu) {
                    if ii > self.max_iter {
                        return Err(EconError::DivergenceError {
                            iteration: iter,
                            suggestion: "inner loop 2; cannot correct step size".into(),
                        });
                    }
                    ii += 1;
                    start = (&start + old) / 2.0;
                    eta = self.eta(&x, &start);
                    mu = self.linkinv(&eta);
                }
                boundary = true;
                dev = self.deviance(&mu);
            }
            if (dev - devold).abs() / (dev.abs() + self.floor) < self.epsilon {
                converged = true;
            }
            if converged {
                let Some(tol) = self.coef_tol else {
                    devold = dev;
                    break;
                };
                let change = coefold.as_ref().map_or(f64::INFINITY, |old| {
                    relative_change_vec(
                        start.as_slice().unwrap_or(&[]),
                        old.as_slice().unwrap_or(&[]),
                    )
                });
                // Stop when the coefficients settle, at a link clamp (separation), or once
                // the change stops shrinking (rounding level).
                if !(change >= tol) || self.saturated(&eta) || change >= prev_change {
                    devold = dev;
                    break;
                }
                prev_change = change;
            }
            devold = dev;
            coefold = Some(start.clone());
        }
        Ok(Irls {
            coef: start,
            eta,
            mu,
            dev: devold,
            qr: last_qr,
            working_weights,
            converged,
            boundary,
            iterations,
        })
    }
}

fn check_len(name: &str, len: usize, n: usize) -> EconResult<()> {
    if len == n {
        Ok(())
    } else {
        Err(spec_err(format!("{name} has {len} rows but x has {n}")))
    }
}

fn check_finite(name: &str, v: &ArrayView1<f64>) -> EconResult<()> {
    match v.iter().position(|x| !x.is_finite()) {
        None => Ok(()),
        Some(i) => Err(spec_err(format!(
            "{name} contains a non-finite value (row {i})"
        ))),
    }
}

/// `log dbinom(k, s, p)` with `0 log 0 = 0`.
fn ldbinom(k: f64, s: f64, p: f64) -> f64 {
    let lchoose = if s <= 1.0 {
        0.0
    } else {
        ln_gamma(s + 1.0) - ln_gamma(k + 1.0) - ln_gamma(s - k + 1.0)
    };
    let a = if k > 0.0 { k * p.ln() } else { 0.0 };
    let b = if s - k > 0.0 {
        (s - k) * (1.0 - p).ln()
    } else {
        0.0
    };
    lchoose + a + b
}

/// Fit a GLM to `y` on `x` (`n x p`; include a column of ones for an intercept).
///
/// Refused ([`EconError::InvalidSpecification`], naming the parameter): length
/// mismatches, non-finite `x`, `y` or offset, negative or non-finite weights, a
/// non-positive `epsilon`, `max_iter` 0, a family-link pair outside R's table, a binomial
/// `y` outside [0, 1], a negative Poisson `y`, and starting values that give invalid
/// fitted values. No positive weight is [`EconError::InsufficientData`]; non-finite IRLS
/// coefficients, or a divergence that step halving cannot correct, are
/// [`EconError::DivergenceError`]. See the module documentation for the conventions.
pub fn fit_glm(x: ArrayView2<f64>, y: ArrayView1<f64>, spec: &GlmSpec<'_>) -> EconResult<GlmFit> {
    let (n, p) = x.dim();
    check_len("y", y.len(), n)?;
    if let Some(o) = &spec.offset {
        check_len("offset", o.len(), n)?;
    }
    if let Some(w) = &spec.weights {
        check_len("weights", w.len(), n)?;
    }
    if let Some(s) = &spec.start
        && s.len() != p
    {
        return Err(spec_err(format!(
            "start has {} values but x has {p} columns",
            s.len()
        )));
    }
    for (j, col) in x.columns().into_iter().enumerate() {
        if let Some(i) = col.iter().position(|v| !v.is_finite()) {
            return Err(spec_err(format!(
                "x contains a non-finite value (row {i}, column {j})"
            )));
        }
    }
    check_finite("y", &y)?;
    if let Some(o) = &spec.offset {
        check_finite("offset", o)?;
    }
    if let Some(s) = &spec.start {
        check_finite("start", s)?;
    }
    if let Some(w) = &spec.weights
        && let Some((i, v)) = w
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite() || **v < 0.0)
    {
        return Err(spec_err(format!(
            "weights must be finite and >= 0 (row {i} is {v})"
        )));
    }
    if !(spec.epsilon.is_finite() && spec.epsilon > 0.0) {
        return Err(spec_err(format!(
            "epsilon must be finite and > 0, got {}",
            spec.epsilon
        )));
    }
    if spec.max_iter == 0 {
        return Err(spec_err("max_iter must be at least 1".into()));
    }
    let (family, link) = (spec.family, spec.link);
    if !family.links().contains(&link) {
        let allowed: Vec<String> = family.links().iter().map(|l| format!("{l:?}")).collect();
        return Err(spec_err(format!(
            "family {family:?} does not support link {link:?}; use one of: {}",
            allowed.join(", ")
        )));
    }
    let w: Vec<f64> = match &spec.weights {
        Some(w) => w.to_vec(),
        None => vec![1.0; n],
    };
    let offset: Vec<f64> = match &spec.offset {
        Some(o) => o.to_vec(),
        None => vec![0.0; n],
    };
    let n_ok = w.iter().filter(|&&v| v > 0.0).count();
    if n_ok == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "glm: no observation has a positive weight".into(),
        });
    }

    // family$initialize
    let mut warnings: Vec<String> = Vec::new();
    let mut yv = y.to_vec();
    let mustart: Array1<f64> = if family.is_binomial() {
        for i in 0..n {
            if w[i] == 0.0 {
                yv[i] = 0.0;
            }
        }
        if yv.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
            return Err(spec_err("y values must be 0 <= y <= 1".into()));
        }
        if family == GlmFamily::Binomial
            && (0..n).any(|i| {
                let m = w[i] * yv[i];
                (m - m.round_ties_even()).abs() > 1e-3
            })
        {
            warnings.push("non-integer #successes in a binomial glm!".into());
        }
        Array1::from_iter((0..n).map(|i| (w[i] * yv[i] + 0.5) / (w[i] + 1.0)))
    } else if family.is_poisson() {
        if yv.iter().any(|&v| v < 0.0) {
            return Err(spec_err(
                "negative values not allowed for the Poisson family".into(),
            ));
        }
        Array1::from_iter(yv.iter().map(|v| v + 0.1))
    } else {
        if link == GlmLink::Log && spec.start.is_none() && yv.iter().any(|&v| v <= 0.0) {
            return Err(spec_err(
                "cannot find valid starting values: please specify some".into(),
            ));
        }
        Array1::from_vec(yv.clone())
    };

    // Convergence floor: 0.1 * (deviance of the weighted mean) per informative row.
    let sw: f64 = w.iter().sum();
    let ybar = (0..n).map(|i| w[i] * yv[i]).sum::<f64>() / sw;
    let d0: f64 = (0..n)
        .filter(|&i| w[i] > 0.0)
        .map(|i| family.dev_resid(yv[i], ybar, w[i]))
        .sum();
    let per = d0 / n_ok as f64;
    let floor = if per.is_finite() && per > 0.0 {
        0.1 * per
    } else {
        0.1
    };

    let problem = Problem {
        family,
        link,
        y: &yv,
        w: &w,
        offset: &offset,
        epsilon: spec.epsilon,
        max_iter: spec.max_iter,
        coef_tol: spec.coef_tol,
        floor,
    };
    let init = match &spec.start {
        Some(s) => Init::Start(*s),
        None => Init::Mustart(mustart),
    };
    let fit = problem.irls(x, init, &mut warnings)?;

    if !fit.converged {
        warnings.push("glm.fit: algorithm did not converge".into());
    }
    if fit.boundary {
        warnings.push("glm.fit: algorithm stopped at boundary value".into());
    }
    let eps10 = 10.0 * EPS;
    let mut separation = false;
    if family.is_binomial() && fit.mu.iter().any(|&m| m > 1.0 - eps10 || m < eps10) {
        separation = true;
        warnings.push("glm.fit: fitted probabilities numerically 0 or 1 occurred".into());
    }
    if family.is_poisson() && fit.mu.iter().any(|&m| m < eps10) {
        warnings.push("glm.fit: fitted rates numerically 0 occurred".into());
    }

    let (rank, aliased, unscaled) = match &fit.qr {
        Some(qr) => (qr.rank(), qr.aliased(), qr.unscaled_covariance()?),
        None => (0, Vec::new(), Array2::zeros((0, 0))),
    };
    let mut coefficients = fit.coef.clone();
    for &j in &aliased {
        coefficients[j] = f64::NAN;
    }
    let df_residual = n_ok.saturating_sub(rank);

    // summary.glm dispersion
    let dispersion = match family {
        GlmFamily::Binomial | GlmFamily::Poisson => 1.0,
        _ if df_residual > 0 => {
            if fit.working_weights.iter().any(|&v| v == 0.0) {
                warnings.push(
                    "observations with zero weight not used for calculating dispersion".into(),
                );
            }
            let s: f64 = (0..n)
                .filter(|&i| fit.working_weights[i] > 0.0)
                .map(|i| {
                    let r = (yv[i] - fit.mu[i]) / link.mu_eta(fit.eta[i]);
                    fit.working_weights[i] * r * r
                })
                .sum();
            s / df_residual as f64
        }
        _ => f64::NAN,
    };
    let vcov = unscaled.mapv(|v| v * dispersion);
    let std_errors = Array1::from_iter(vcov.diag().iter().map(|v| v.sqrt()));

    // Null deviance (glm.fit, and glm()'s refit with an offset and an intercept).
    let ones_col = (0..p).find(|&j| x.column(j).iter().all(|&v| v == 1.0));
    let null_deviance = match ones_col {
        Some(_) if spec.offset.is_some() && p == 1 => fit.dev,
        Some(j) if spec.offset.is_some() => {
            let xi = x.slice(ndarray::s![.., j..j + 1]);
            let mut w2 = Vec::new();
            let refit = problem.irls(xi, Init::Mustart(fit.mu.clone()), &mut w2)?;
            if !refit.converged {
                warnings.push(
                    "fitting to calculate the null deviance did not converge -- increase \
                     'maxit'?"
                        .into(),
                );
            }
            refit.dev
        }
        Some(_) => {
            let wtdmu = (0..n).map(|i| w[i] * yv[i]).sum::<f64>() / sw;
            (0..n).map(|i| family.dev_resid(yv[i], wtdmu, w[i])).sum()
        }
        None => (0..n)
            .map(|i| family.dev_resid(yv[i], link.linkinv(offset[i]), w[i]))
            .sum(),
    };

    // AIC (family$aic + 2 rank) and logLik.glm
    let aic_family = match family {
        GlmFamily::Gaussian => {
            let nk = n_ok as f64;
            let slw: f64 = w.iter().filter(|&&v| v > 0.0).map(|v| v.ln()).sum();
            nk * ((2.0 * std::f64::consts::PI * fit.dev / nk).ln() + 1.0) + 2.0 - slw
        }
        GlmFamily::Binomial => {
            -2.0 * (0..n)
                .filter(|&i| w[i] > 0.0)
                .map(|i| {
                    let m = w[i];
                    ldbinom(
                        (m * yv[i]).round_ties_even(),
                        m.round_ties_even(),
                        fit.mu[i],
                    )
                })
                .sum::<f64>()
        }
        GlmFamily::Poisson => {
            let mut s = 0.0;
            let mut warned = false;
            for i in 0..n {
                let yi = yv[i];
                let ld = if yi.fract() != 0.0 {
                    if !warned {
                        warnings.push(format!("non-integer x = {yi:.6}"));
                        warned = true;
                    }
                    f64::NEG_INFINITY
                } else if yi == 0.0 {
                    -fit.mu[i]
                } else {
                    yi * fit.mu[i].ln() - fit.mu[i] - ln_gamma(yi + 1.0)
                };
                if w[i] != 0.0 {
                    s += ld * w[i];
                }
            }
            -2.0 * s
        }
        GlmFamily::QuasiBinomial | GlmFamily::QuasiPoisson => f64::NAN,
    };
    let aic = aic_family + 2.0 * rank as f64;
    let log_likelihood = if family.is_quasi() {
        f64::NAN
    } else {
        let p_ll = if family == GlmFamily::Gaussian {
            rank as f64 + 1.0
        } else {
            rank as f64
        };
        p_ll - aic / 2.0
    };

    let mut seen = std::collections::HashSet::new();
    warnings.retain(|m| seen.insert(m.clone()));
    Ok(GlmFit {
        coefficients,
        std_errors,
        vcov,
        fitted: fit.mu,
        linear_predictor: fit.eta,
        working_weights: fit.working_weights,
        deviance: fit.dev,
        null_deviance,
        log_likelihood,
        aic,
        dispersion,
        df_residual,
        rank,
        aliased,
        converged: fit.converged,
        iterations: fit.iterations,
        max_iter: spec.max_iter,
        epsilon: spec.epsilon,
        boundary: fit.boundary,
        separation,
        warnings,
    })
}

impl ResultNotes for GlmFit {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn notes_convergence(&self) -> Option<Convergence> {
        Some(Convergence::new(
            self.converged,
            self.iterations,
            self.max_iter,
            "relative change in deviance",
            f64::NAN,
            self.epsilon,
        ))
    }
}

/// Relative coefficient change at which [`fit_propensity`] stops refining after the
/// deviance rule ([`GlmSpec::coef_tol`]).
pub const PROPENSITY_COEF_TOL: f64 = 1e-10;

/// Propensity scores: the binomial MLE of `treat` (coded 0/1) on `x` with the logit,
/// probit or complementary log-log link. The caller includes the intercept column.
///
/// After the deviance rule the fit keeps iterating until the relative coefficient change
/// is below [`PROPENSITY_COEF_TOL`] (or a fitted probability reaches the link's clamp), so
/// the non-canonical links reach the MLE as closely as the canonical logit.
pub fn fit_propensity(
    x: ArrayView2<f64>,
    treat: ArrayView1<f64>,
    link: GlmLink,
) -> EconResult<PropensityFit> {
    if !matches!(link, GlmLink::Logit | GlmLink::Probit | GlmLink::Cloglog) {
        return Err(spec_err(format!(
            "propensity link must be logit, probit or cloglog, got {link:?}"
        )));
    }
    if let Some((i, v)) = treat
        .iter()
        .enumerate()
        .find(|(_, v)| **v != 0.0 && **v != 1.0)
    {
        return Err(spec_err(format!(
            "propensity treatment must be coded 0/1 (row {i} is {v})"
        )));
    }
    let mut spec = GlmSpec::new(GlmFamily::Binomial, link);
    spec.coef_tol = Some(PROPENSITY_COEF_TOL);
    let fit = fit_glm(x, treat, &spec)?;
    Ok(PropensityFit {
        scores: fit.fitted.clone(),
        fit,
    })
}

/// The logistic fluctuation of TMLE: a quasi-binomial logit fit of `y` (in [0, 1]) on
/// `covariates` with `offset` (the initial fit on the logit scale), optionally weighted.
///
/// Two forms are common: the clever covariate `H` as the single column with no
/// intercept (`glm(y ~ -1 + H, offset, quasibinomial)`), or a column of ones with the
/// clever covariate as weights (`glm(y ~ 1, offset, weights = H, quasibinomial)`).
pub fn fit_logistic_fluctuation<'a>(
    y: ArrayView1<f64>,
    offset: ArrayView1<'a, f64>,
    covariates: ArrayView2<f64>,
    weights: Option<ArrayView1<'a, f64>>,
) -> EconResult<GlmFit> {
    let mut spec = GlmSpec::new(GlmFamily::QuasiBinomial, GlmLink::Logit).offset(offset);
    if let Some(w) = weights {
        spec = spec.weights(w);
    }
    fit_glm(covariates, y, &spec)
}

/// R's `qlogis(p)`: `ln(p / (1 - p))`.
pub fn qlogis(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// R's `plogis(x)`: `1 / (1 + exp(-x))`.
pub fn plogis(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `p` truncated to `[lo, hi]` (the `bound()` of the TMLE packages); `NaN` stays `NaN`.
pub fn bound(p: f64, lo: f64, hi: f64) -> f64 {
    if p.is_nan() { p } else { p.max(lo).min(hi) }
}

/// `f64` with `NaN` as JSON `null`.
mod nan_f64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_nan() {
            s.serialize_none()
        } else {
            s.serialize_some(v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
}

/// `Array1<f64>` as a JSON array with `NaN` as `null`.
mod nan_vec {
    use ndarray::Array1;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Array1<f64>, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(v.iter().map(|x| if x.is_nan() { None } else { Some(*x) }))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Array1<f64>, D::Error> {
        let v = Vec::<Option<f64>>::deserialize(d)?;
        Ok(v.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect())
    }
}

/// `Array2<f64>` as a JSON array of rows with `NaN` as `null`.
mod nan_mat {
    use ndarray::Array2;
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(m: &Array2<f64>, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(m.rows().into_iter().map(|row| {
            row.iter()
                .map(|x| if x.is_nan() { None } else { Some(*x) })
                .collect::<Vec<Option<f64>>>()
        }))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Array2<f64>, D::Error> {
        let rows = Vec::<Vec<Option<f64>>>::deserialize(d)?;
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, |r| r.len());
        if rows.iter().any(|r| r.len() != ncols) {
            return Err(D::Error::custom("matrix rows have different lengths"));
        }
        let data: Vec<f64> = rows
            .into_iter()
            .flatten()
            .map(|x| x.unwrap_or(f64::NAN))
            .collect();
        Array2::from_shape_vec((nrows, ncols), data).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::solve::case_data::{columns, design};
    use ndarray::Array1;
    use p2a_validation::{Check, Tol};

    /// Fits at the default epsilon (1e-10) against R at 1e-14, for quantities that
    /// converge only linearly under the deviance-change rule (non-canonical links; SEs and
    /// dispersion from the working weights of the last step): measured gaps up to 2.1e-6
    /// relative, the gaps R's own glm shows at epsilon 1e-10.
    const DEFAULT_EPS_TOL: Tol = Tol::new(1e-5, 1e-12);

    /// (key, y, design columns, family, link, offset column, weights column, has AIC)
    type Spec = (
        &'static str,
        &'static str,
        &'static [&'static str],
        GlmFamily,
        GlmLink,
        Option<&'static str>,
        Option<&'static str>,
        bool,
    );
    const FAMILIES: &[Spec] = &[
        (
            "logit",
            "yb",
            &["1", "x1", "x2"],
            GlmFamily::Binomial,
            GlmLink::Logit,
            None,
            None,
            true,
        ),
        (
            "probit",
            "yb",
            &["1", "x1", "x2"],
            GlmFamily::Binomial,
            GlmLink::Probit,
            None,
            None,
            true,
        ),
        (
            "cloglog",
            "yb",
            &["1", "x1", "x2"],
            GlmFamily::Binomial,
            GlmLink::Cloglog,
            None,
            None,
            true,
        ),
        (
            "poisson",
            "yp",
            &["1", "x1"],
            GlmFamily::Poisson,
            GlmLink::Log,
            Some("off"),
            None,
            true,
        ),
        (
            "poisid",
            "yp",
            &["1", "x1"],
            GlmFamily::Poisson,
            GlmLink::Identity,
            None,
            None,
            true,
        ),
        (
            "qpois",
            "yp",
            &["1", "x1"],
            GlmFamily::QuasiPoisson,
            GlmLink::Log,
            Some("off"),
            None,
            false,
        ),
        (
            "quasibin",
            "yq",
            &["1", "x1", "x4"],
            GlmFamily::QuasiBinomial,
            GlmLink::Logit,
            None,
            Some("wq"),
            false,
        ),
        (
            "gaussian",
            "yc",
            &["1", "x1", "x2"],
            GlmFamily::Gaussian,
            GlmLink::Identity,
            None,
            None,
            true,
        ),
        (
            "gausslog",
            "yp+0.5",
            &["1", "x1"],
            GlmFamily::Gaussian,
            GlmLink::Log,
            None,
            None,
            true,
        ),
    ];

    fn response(c: &crate::data::columns::Columns, name: &str) -> Array1<f64> {
        if name == "yp+0.5" {
            c.numeric_array("yp").mapv(|v| v + 0.5)
        } else {
            c.numeric_array(name)
        }
    }

    /// max_j |sum_i x_ij W_i r_i| relative to max_j sum_i |x_ij W_i r_i|: the score
    /// equations at the returned fit (W the working weights, r the working residuals).
    fn score_ratio(x: &ndarray::Array2<f64>, y: &Array1<f64>, fit: &GlmFit, link: GlmLink) -> f64 {
        let (n, p) = x.dim();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for j in 0..p {
            let (mut s, mut a) = (0.0, 0.0);
            for i in 0..n {
                let r = (y[i] - fit.fitted[i]) / link.mu_eta(fit.linear_predictor[i]);
                let t = x[[i, j]] * fit.working_weights[i] * r;
                s += t;
                a += t.abs();
            }
            num = num.max(s.abs());
            den = den.max(a);
        }
        num / den
    }

    #[test]
    fn test_validate_glm_families() {
        let (case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let n = c.n();
        let mut check = Check::new(&case);
        for &(key, yname, cols, family, link, off, wts, has_aic) in FAMILIES {
            let x = design(&c, cols);
            let y = response(&c, yname);
            let o = off.map(|o| c.numeric_array(o));
            let w = wts.map(|w| c.numeric_array(w));
            let mut spec = GlmSpec::new(family, link);
            if let Some(o) = &o {
                spec = spec.offset(o.view());
            }
            if let Some(w) = &w {
                spec = spec.weights(w.view());
            }
            // Both sides converged: R and p2a at epsilon 1e-14.
            spec.epsilon = 1e-14;
            let fit = fit_glm(x.view(), y.view(), &spec).unwrap();
            assert!(fit.converged, "{key}");
            check
                .vector(
                    &format!("{key}_coef"),
                    &fit.coefficients.to_vec(),
                    Tol::coef(n),
                )
                .vector(&format!("{key}_se"), &fit.std_errors.to_vec(), Tol::se(n))
                .scalar(&format!("{key}_deviance"), fit.deviance, Tol::stat(n))
                .scalar(
                    &format!("{key}_null_deviance"),
                    fit.null_deviance,
                    Tol::stat(n),
                )
                .scalar(&format!("{key}_dispersion"), fit.dispersion, Tol::stat(n))
                .scalar(
                    &format!("{key}_df_residual"),
                    fit.df_residual as f64,
                    Tol::exact(),
                );
            if has_aic {
                check
                    .scalar(&format!("{key}_aic"), fit.aic, Tol::stat(n))
                    .scalar(&format!("{key}_loglik"), fit.log_likelihood, Tol::stat(n));
            } else {
                assert!(
                    fit.aic.is_nan() && fit.log_likelihood.is_nan(),
                    "{key}: quasi families have no AIC"
                );
            }
            // (b) the score equations hold at the solution
            let ratio = score_ratio(&x, &y, &fit, link);
            assert!(ratio <= 1e-8, "{key}: score ratio {ratio}");
            // (c) the default spec converges too, near the same solution: the deviance-change
            // rule leaves an O(sqrt(epsilon)) coefficient error when the link is not
            // canonical (Fisher scoring then converges linearly). R's glm at epsilon 1e-10
            // stops at the same point (probit intercept -0.0722050664731 against
            // -0.0722049136445 at 1e-14, R 4.3.3).
            let mut d = GlmSpec::new(family, link);
            d.offset = spec.offset;
            d.weights = spec.weights;
            let fd = fit_glm(x.view(), y.view(), &d).unwrap();
            assert!(fd.converged, "{key}: default spec");
            for (a, b) in fd.coefficients.iter().zip(fit.coefficients.iter()) {
                assert!(DEFAULT_EPS_TOL.allows(*a, *b), "{key}: default {a} vs {b}");
            }
        }
        // aliased logit: x2 = (x1 - x3) / 2 is aliased (exactly collinear)
        let xa = design(&c, &["1", "x1", "x3", "x2"]);
        let yb = c.numeric_array("yb");
        let mut sa = GlmSpec::logit();
        sa.epsilon = 1e-10;
        let fa = fit_glm(xa.view(), yb.view(), &sa).unwrap();
        assert_eq!((fa.rank, fa.aliased.clone()), (3, vec![3]));
        assert!(fa.coefficients[3].is_nan() && fa.std_errors[3].is_nan());
        assert!(fa.vcov.row(3).iter().all(|v| v.is_nan()));
        check
            .vector("aliased_coef", &fa.coefficients.to_vec(), Tol::coef(n))
            .vector("aliased_se", &fa.std_errors.to_vec()[..3], Tol::se(n)) // summary.glm drops the NA row
            .scalar("aliased_deviance", fa.deviance, Tol::stat(n))
            .scalar("aliased_null_deviance", fa.null_deviance, Tol::stat(n))
            .scalar("aliased_dispersion", fa.dispersion, Tol::stat(n))
            .scalar("aliased_df_residual", fa.df_residual as f64, Tol::exact())
            .scalar("aliased_aic", fa.aic, Tol::stat(n))
            .scalar("aliased_loglik", fa.log_likelihood, Tol::stat(n));
        // fit_propensity: the logit and probit MLE (CAUS-20), against R at 1e-14, both at
        // 1e-8: the coefficient refinement (PROPENSITY_COEF_TOL) removes the deviance rule's
        // linear-convergence gap of the probit
        let x = design(&c, &["1", "x1", "x2"]);
        let pl = fit_propensity(x.view(), yb.view(), GlmLink::Logit).unwrap();
        let pp = fit_propensity(x.view(), yb.view(), GlmLink::Probit).unwrap();
        check
            .vector(
                "logit_fitted_head",
                &pl.scores.to_vec()[..5],
                Tol::iterative(1e-8),
            )
            .vector(
                "probit_fitted_head",
                &pp.scores.to_vec()[..5],
                Tol::iterative(1e-8),
            );
        for (got, want) in pp.fit.coefficients.iter().zip(case.vector("probit_coef")) {
            assert!(
                Tol::iterative(1e-8).allows(*got, want),
                "propensity probit {got} vs R {want}"
            );
        }
        // the probit's default-spec fit is the deviance rule alone (about 2e-6 away)
        let pd = fit_glm(x.view(), yb.view(), &GlmSpec::probit()).unwrap();
        assert!(pd.converged && pp.fit.converged && pp.fit.iterations > pd.iterations);
        assert_eq!(pl.scores, pl.fit.fitted);
        // (a) start values work: start = 0 converges to the default-start solution
        let zero = Array1::<f64>::zeros(3);
        let mut s0 = GlmSpec::logit();
        s0.start = Some(zero.view());
        let f0 = fit_glm(x.view(), yb.view(), &s0).unwrap();
        assert!(f0.converged);
        for (a, b) in f0.coefficients.iter().zip(pl.fit.coefficients.iter()) {
            assert!(Tol::iterative(1e-8).allows(*a, *b), "start 0: {a} vs {b}");
        }
        check.finish();
    }

    #[test]
    fn test_validate_glm_fluctuation() {
        let (case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let y = c.numeric_array("yq");
        let off = c.numeric_array("x1").mapv(|v| 0.2 + 0.5 * v); // R: 0.2 + 0.5 * x1
        let h = design(&c, &["x4"]); // clever covariate, no intercept
        let ones = design(&c, &["1"]);
        let w = c.numeric_array("wq");
        let fl = fit_logistic_fluctuation(y.view(), off.view(), h.view(), None).unwrap();
        let fi =
            fit_logistic_fluctuation(y.view(), off.view(), ones.view(), Some(w.view())).unwrap();
        // Default epsilon 1e-10 against R at 1e-14. Coefficients and deviances at
        // Tol::iterative(1e-8); SE and dispersion at DEFAULT_EPS_TOL: summary.glm uses the
        // working weights of the last IRLS step, one step behind the solution (R's glm at
        // epsilon 1e-10 gives SE 0.039090634610887, against 0.039090643870346 at 1e-14).
        let t = Tol::iterative(1e-8);
        let mut check = Check::new(&case);
        for (key, f) in [("fluct", &fl), ("fluct_int", &fi)] {
            assert!(f.converged, "{key}");
            check
                .vector(&format!("{key}_coef"), &f.coefficients.to_vec(), t)
                .vector(
                    &format!("{key}_se"),
                    &f.std_errors.to_vec(),
                    DEFAULT_EPS_TOL,
                )
                .scalar(&format!("{key}_deviance"), f.deviance, t)
                // no intercept: linkinv(offset); intercept only: the fitted model
                .scalar(&format!("{key}_null_deviance"), f.null_deviance, t)
                .scalar(&format!("{key}_dispersion"), f.dispersion, DEFAULT_EPS_TOL)
                .scalar(
                    &format!("{key}_df_residual"),
                    f.df_residual as f64,
                    Tol::exact(),
                );
            assert!(f.aic.is_nan());
        }
        check.finish();
    }

    /// Separation (no finite MLE): flagged, warned, and every observation classified
    /// correctly; neither convergence nor an iteration count is asserted.
    #[test]
    fn test_validate_glm_separation_flag() {
        let (_case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let x = design(&c, &["1", "x4"]);
        let y = c.numeric_array("ysep"); // ysep = 1{x4 > 0.1}
        let mut short = GlmSpec::logit();
        short.epsilon = 1e-8;
        short.max_iter = 25; // R's glm.control() defaults
        for spec in [short, GlmSpec::logit()] {
            let fit = fit_glm(x.view(), y.view(), &spec).unwrap();
            assert!(fit.separation && !fit.boundary);
            assert!(
                fit.warnings
                    .iter()
                    .any(|w| w == "glm.fit: fitted probabilities numerically 0 or 1 occurred")
            );
            assert!(fit.deviance < 2.0 * 2f64.ln(), "deviance {}", fit.deviance);
            for i in 0..y.len() {
                assert_eq!(fit.fitted[i] > 0.5, y[i] == 1.0, "row {i}");
            }
        }
    }

    /// The coefficient refinement of fit_propensity stops at the link clamp under perfect
    /// separation instead of running to max_iter; the fit's notes report the deviance rule
    /// and R's separation warning.
    #[test]
    fn test_propensity_refinement_stops_under_separation() {
        let x = ndarray::array![
            [1.0, 10.0],
            [1.0, 11.0],
            [1.0, 12.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.5]
        ];
        let d = ndarray::array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        for link in [GlmLink::Logit, GlmLink::Probit, GlmLink::Cloglog] {
            let p = fit_propensity(x.view(), d.view(), link).unwrap();
            assert!(p.fit.converged && p.fit.separation, "{link:?}");
            assert!(
                p.fit.iterations < p.fit.max_iter,
                "{link:?}: {}",
                p.fit.iterations
            );
            let text = crate::traits::convergence::notes_text(&p.fit);
            assert!(text.starts_with("Converged: yes ("), "{text}");
            assert!(
                text.contains("Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred"),
                "{text}"
            );
        }
        let f = fit_glm(x.view(), d.view(), &GlmSpec::logit()).unwrap();
        let c = f.notes_convergence().unwrap();
        assert_eq!((c.max_iterations, c.tolerance), (100, 1e-10));
    }

    #[test]
    fn test_validate_glm_binomial_noninteger() {
        let (case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let n = c.n();
        let x = design(&c, &["1", "x1", "x4"]);
        let (y, w) = (c.numeric_array("yq"), c.numeric_array("wq"));
        let mut spec = GlmSpec::new(GlmFamily::Binomial, GlmLink::Logit).weights(w.view());
        spec.epsilon = 1e-14;
        let fit = fit_glm(x.view(), y.view(), &spec).unwrap();
        Check::new(&case)
            .vector("binfrac_coef", &fit.coefficients.to_vec(), Tol::coef(n)) // = quasibinomial coefficients
            .scalar("binfrac_aic", fit.aic, Tol::stat(n)) // round(w y), round(w): ties to even
            .finish();
        assert!(
            fit.warnings
                .iter()
                .any(|m| m == "non-integer #successes in a binomial glm!")
        );
        assert_eq!(fit.dispersion, 1.0);
    }

    /// Rescaling a regressor or the response rescales the solution and leaves the
    /// iteration count alone (properties; not R-pinned). The Gaussian log-link fit of a
    /// small-scale response is where glm.fit's absolute 0.1 stops after one step.
    #[test]
    fn test_validate_glm_scale_equivariance() {
        let (_case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let yb = c.numeric_array("yb");
        let yc = c.numeric_array("yc");
        let t = Tol::iterative(1e-8);
        let x = design(&c, &["1", "x1", "x2"]);
        let base = fit_glm(x.view(), yb.view(), &GlmSpec::logit()).unwrap();
        for scale in [1e6, 1e-6] {
            let mut xs = x.clone();
            xs.column_mut(1).mapv_inplace(|v| v * scale);
            let f = fit_glm(xs.view(), yb.view(), &GlmSpec::logit()).unwrap();
            assert!(f.converged);
            assert!(t.allows(f.coefficients[1] * scale, base.coefficients[1]));
            assert!(t.allows(f.coefficients[0], base.coefficients[0]));
            assert!(t.allows(f.coefficients[2], base.coefficients[2]));
            assert_eq!(f.iterations, base.iterations, "scale {scale}");
        }
        let gspec = GlmSpec::new(GlmFamily::Gaussian, GlmLink::Identity);
        let g = fit_glm(x.view(), yc.view(), &gspec).unwrap();
        for scale in [1e6, 1e-6] {
            let y = yc.mapv(|v| v * scale);
            let f = fit_glm(x.view(), y.view(), &gspec).unwrap();
            for (a, b) in f.coefficients.iter().zip(g.coefficients.iter()) {
                assert!(
                    t.allows(*a, b * scale),
                    "scale {scale}: {a} vs {}",
                    b * scale
                );
            }
        }
        // Gaussian log link on 1e-6 * (yp + 0.5): the unscaled slope, intercept + ln 1e-6
        let xl = design(&c, &["1", "x1"]);
        let yl = c.numeric_array("yp").mapv(|v| v + 0.5);
        let lspec = GlmSpec::new(GlmFamily::Gaussian, GlmLink::Log);
        let fl = fit_glm(xl.view(), yl.view(), &lspec).unwrap();
        let ys = yl.mapv(|v| 1e-6 * v);
        let fs = fit_glm(xl.view(), ys.view(), &lspec).unwrap();
        assert!(
            fs.converged && fs.iterations >= 2,
            "{} iterations",
            fs.iterations
        );
        assert!(t.allows(fs.coefficients[1], fl.coefficients[1]));
        assert!(t.allows(fs.coefficients[0], fl.coefficients[0] + 1e-6f64.ln()));
    }

    /// Zero prior weights drop their rows: the same coefficients, deviance and AIC as the
    /// fit on the other rows (Gaussian AIC over the positive weights: a deviation from
    /// glm's +Inf).
    #[test]
    fn test_validate_glm_zero_weights_match_subset() {
        let (_case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let x = design(&c, &["1", "x1", "x2"]);
        let w = c.numeric_array("w0"); // w0[0..5] == 0
        let keep: Vec<usize> = (0..c.n()).filter(|&i| w[i] > 0.0).collect();
        assert_eq!(keep.len(), 115);
        let xs = x.select(ndarray::Axis(0), &keep);
        let ws = w.select(ndarray::Axis(0), &keep);
        for (yname, family) in [("yc", GlmFamily::Gaussian), ("yb", GlmFamily::Binomial)] {
            let link = if family == GlmFamily::Gaussian {
                GlmLink::Identity
            } else {
                GlmLink::Logit
            };
            let y = c.numeric_array(yname);
            let ysub = y.select(ndarray::Axis(0), &keep);
            let full = fit_glm(
                x.view(),
                y.view(),
                &GlmSpec::new(family, link).weights(w.view()),
            )
            .unwrap();
            let sub = fit_glm(
                xs.view(),
                ysub.view(),
                &GlmSpec::new(family, link).weights(ws.view()),
            )
            .unwrap();
            let t = Tol::iterative(1e-10);
            for (a, b) in full.coefficients.iter().zip(sub.coefficients.iter()) {
                assert!(t.allows(*a, *b), "{yname}: {a} vs {b}");
            }
            assert!(t.allows(full.deviance, sub.deviance), "{yname} deviance");
            assert!(
                full.aic.is_finite() && t.allows(full.aic, sub.aic),
                "{yname} AIC {} vs {}",
                full.aic,
                sub.aic
            );
            assert_eq!(full.df_residual, sub.df_residual);
        }
    }

    fn spec_err<T: std::fmt::Debug>(r: EconResult<T>, needle: &str) {
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains(needle), "{message}")
            }
            other => panic!("expected InvalidSpecification containing {needle:?}, got {other:?}"),
        }
    }

    #[test]
    fn test_glm_rejects_invalid_input() {
        let x = ndarray::array![
            [1.0, -1.2],
            [1.0, -0.3],
            [1.0, 0.4],
            [1.0, 1.1],
            [1.0, 1.9],
            [1.0, -0.7]
        ];
        let yb = ndarray::array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        spec_err(
            fit_glm(
                x.view(),
                ndarray::array![0.0, 1.0, 0.0, 1.5, 1.0, 0.0].view(),
                &GlmSpec::logit(),
            ),
            "0 <= y <= 1",
        );
        spec_err(
            fit_glm(
                x.view(),
                ndarray::array![0.0, 1.0, -1.0, 2.0, 1.0, 0.0].view(),
                &GlmSpec::poisson(),
            ),
            "negative",
        );
        spec_err(
            fit_glm(
                x.view(),
                yb.view(),
                &GlmSpec::new(GlmFamily::Poisson, GlmLink::Probit),
            ),
            "link",
        );
        spec_err(
            fit_glm(
                x.view(),
                yb.view(),
                &GlmSpec::new(GlmFamily::Gaussian, GlmLink::Logit),
            ),
            "link",
        );
        let wneg = ndarray::array![1.0, 1.0, -1.0, 1.0, 1.0, 1.0];
        spec_err(
            fit_glm(x.view(), yb.view(), &GlmSpec::logit().weights(wneg.view())),
            "weights",
        );
        let o3 = ndarray::array![0.0, 0.0, 0.0];
        spec_err(
            fit_glm(x.view(), yb.view(), &GlmSpec::logit().offset(o3.view())),
            "offset",
        );
        let mut xn = x.clone();
        xn[[3, 1]] = f64::NAN;
        spec_err(fit_glm(xn.view(), yb.view(), &GlmSpec::logit()), "x");
        spec_err(
            fit_glm(x.view(), yb.slice(ndarray::s![..5]), &GlmSpec::logit()),
            "rows",
        );
        let st = ndarray::array![0.0];
        let mut s = GlmSpec::logit();
        s.start = Some(st.view());
        spec_err(fit_glm(x.view(), yb.view(), &s), "start");
        let mut s = GlmSpec::logit();
        s.epsilon = 0.0;
        spec_err(fit_glm(x.view(), yb.view(), &s), "epsilon");
        let mut s = GlmSpec::logit();
        s.max_iter = 0;
        spec_err(fit_glm(x.view(), yb.view(), &s), "max_iter");
        // gaussian log without start needs y > 0 (R: "cannot find valid starting values")
        spec_err(
            fit_glm(
                x.view(),
                yb.view(),
                &GlmSpec::new(GlmFamily::Gaussian, GlmLink::Log),
            ),
            "starting values",
        );
        spec_err(
            fit_propensity(
                x.view(),
                ndarray::array![0.0, 1.0, 2.0, 1.0, 1.0, 0.0].view(),
                GlmLink::Logit,
            ),
            "0/1",
        );
        spec_err(fit_propensity(x.view(), yb.view(), GlmLink::Log), "link");
        assert!(matches!(
            fit_glm(
                x.view(),
                yb.view(),
                &GlmSpec::logit().weights(Array1::<f64>::zeros(6).view())
            ),
            Err(EconError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_logistic_helpers() {
        assert_eq!(plogis(0.0), 0.5);
        assert!((qlogis(0.25) - (-1.0986122886681098)).abs() <= 1e-15); // R qlogis(0.25)
        assert!((plogis(0.3) - 0.574442516811659).abs() <= 1e-15); // R plogis(0.3)
        assert_eq!((plogis(-800.0), plogis(800.0)), (0.0, 1.0));
        assert!((plogis(qlogis(0.2)) - 0.2).abs() <= 1e-15);
        assert_eq!(
            (qlogis(0.0), qlogis(1.0)),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(bound(0.001, 0.01, 0.99), 0.01);
        assert_eq!(bound(0.995, 0.01, 0.99), 0.99);
        assert_eq!(bound(0.5, 0.01, 0.99), 0.5);
        assert!(bound(f64::NAN, 0.01, 0.99).is_nan());
        // probit clamp: R -qnorm(.Machine$double.eps); statrs agrees
        let q = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        assert!(
            (-statrs::distribution::ContinuousCDF::inverse_cdf(&q, f64::EPSILON) - PROBIT_THRESH)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn test_validate_glm_fit_serde_round_trip() {
        let (_case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let x = design(&c, &["1", "x1", "x3", "x2"]);
        let y = c.numeric_array("yq");
        let w = c.numeric_array("wq");
        let fit = fit_glm(
            x.view(),
            y.view(),
            &GlmSpec::new(GlmFamily::QuasiBinomial, GlmLink::Logit).weights(w.view()),
        )
        .unwrap();
        assert!(fit.coefficients[3].is_nan() && fit.aic.is_nan()); // aliased column, quasi family
        let json = serde_json::to_string(&fit).unwrap();
        let back: GlmFit = serde_json::from_str(&json).unwrap();
        assert!(back.coefficients[3].is_nan() && back.aic.is_nan());
        assert_eq!(
            back.coefficients.slice(ndarray::s![..3]),
            fit.coefficients.slice(ndarray::s![..3])
        );
        assert_eq!(back.vcov.dim(), fit.vcov.dim());
        assert_eq!(
            (back.rank, back.aliased.clone(), back.iterations),
            (fit.rank, fit.aliased.clone(), fit.iterations)
        );
    }
}
