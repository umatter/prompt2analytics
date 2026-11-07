//! Generalized linear models by iteratively reweighted least squares: the one GLM that
//! Tier 3 estimators share (propensity scores, TMLE fluctuations, Poisson and binomial
//! models).
//!
//! [`fit_glm`] fits the Gaussian, binomial, Poisson, quasi-binomial and quasi-Poisson
//! families with the identity, logit, probit, complementary log-log and log links, prior
//! weights, offsets and start values, by Fisher scoring as R's `stats::glm.fit` (R 4.3.3):
//! the same family and link functions (including R's clamps of the logit, probit,
//! cloglog and log inverse links), `mustart`, step halving on a non-finite deviance or
//! invalid fitted values, and R's warning texts. Each IRLS step takes OLS's solve decision
//! on the weighted design: the normal equations when it is of full rank and well
//! conditioned (reciprocal condition of the equilibrated `X'WX` at least 1e-8, refined
//! once below 1e-2), otherwise the pivoted QR of [`crate::linalg::solve::Qr`], which
//! decides the rank; aliased columns get `NaN` coefficients and SEs.
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
//!   coefficients settle; [`fit_propensity`] uses it for the non-canonical links only, so
//!   probit (and cloglog) propensity scores reach the MLE to about 1e-10 instead of the
//!   deviance rule's 2e-6. The canonical logit keeps the deviance rule alone: its IRLS is
//!   Newton's method, which converges quadratically and already lands at about 1e-8.
//!
//! Refits that read only the coefficients, fitted values, covariance and warnings (the
//! propensity and outcome models of IPW, AIPW, WeightIt, mediation and the g-formula, and
//! the TMLE nuisance fits) go through `lean_logit`, the same IRLS for a unit-weight logit
//! on its common path (the normal equations, with the refinement of an uncentred or correlated
//! design in `lean_logit_refined`), which skips the AIC, the row-wise deviance logarithms and the
//! result assembly; a fit that leaves that path (separation, an ill-conditioned or rank-deficient
//! step, no convergence) is `fit_glm`'s.
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
use crate::regression::ols::normal_inverse_from_gram;
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, relative_change_vec};
use faer::linalg::matmul::triangular;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;
use std::borrow::Cow;

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
    /// `glm.control` default is 1e-8). At a tolerance of rounding size the rule is decided by
    /// rounding: at 1e-14 the lean refits ([`fit_logit_lean`]) and [`fit_glm`] can stop one or two
    /// iterations apart (23 of 61,704 fits measured; coefficient gap at most 7.6e-13).
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
    /// `None`; [`fit_propensity`] uses 1e-10 for the probit and cloglog links.
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
        mu.iter().all(|&m| self.valid_mu_one(m))
    }

    /// `family$validmu` for one fitted value.
    fn valid_mu_one(self, m: f64) -> bool {
        match self {
            GlmFamily::Gaussian => true,
            GlmFamily::Binomial | GlmFamily::QuasiBinomial => m.is_finite() && m > 0.0 && m < 1.0,
            GlmFamily::Poisson | GlmFamily::QuasiPoisson => m.is_finite() && m > 0.0,
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

    /// `(linkinv(eta), mu_eta(eta))`, sharing `exp(eta)` for the logit and log links.
    fn inv_and_deriv(self, eta: f64) -> (f64, f64) {
        match self {
            GlmLink::Logit => {
                if eta < -LOGIT_THRESH {
                    (EPS / (1.0 + EPS), EPS)
                } else if eta > LOGIT_THRESH {
                    let t = 1.0 / EPS;
                    (t / (1.0 + t), EPS)
                } else {
                    let t = eta.exp();
                    (t / (1.0 + t), t / ((1.0 + t) * (1.0 + t)))
                }
            }
            GlmLink::Log => {
                let m = eta.exp().max(EPS);
                (m, m)
            }
            link => (link.linkinv(eta), link.mu_eta(eta)),
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

/// Values of a pure function of a few `f64` arguments (keyed by their bits), for loops
/// over rows that take few distinct argument values: up to [`MEMO_SLOTS`] are kept, the
/// rest are computed each time.
struct Memo<V: Copy> {
    keys: [[u64; 3]; MEMO_SLOTS],
    values: [Option<V>; MEMO_SLOTS],
    len: usize,
}

/// Entries of a [`Memo`].
const MEMO_SLOTS: usize = 8;

impl<V: Copy> Default for Memo<V> {
    fn default() -> Self {
        Self {
            keys: [[0; 3]; MEMO_SLOTS],
            values: [None; MEMO_SLOTS],
            len: 0,
        }
    }
}

impl<V: Copy> Memo<V> {
    fn get_or(&mut self, key: [u64; 3], f: impl FnOnce() -> V) -> V {
        for k in 0..self.len {
            if self.keys[k] == key
                && let Some(v) = self.values[k]
            {
                return v;
            }
        }
        let v = f();
        if self.len < MEMO_SLOTS {
            self.keys[self.len] = key;
            self.values[self.len] = Some(v);
            self.len += 1;
        }
        v
    }
}

/// How IRLS starts: from `start` coefficients, or from `mustart`.
enum Init<'a> {
    Start(ArrayView1<'a, f64>),
    Mustart(Array1<f64>),
}

/// How one IRLS step was solved: OLS's decision
/// ([`crate::regression::ols::normal_equations_inverse`], from the `X'WX` the step
/// accumulates) on the weighted design.
enum StepSolve {
    /// `(X'WX)^{-1}` of a full-rank, well-conditioned weighted design (the normal
    /// equations, refined once when the conditioning asks for it).
    Normal(Array2<f64>),
    /// The pivoted QR (`lm.fit`'s `dqrdc2` at [`GLM_QR_TOL`]): rank-deficient or
    /// ill-conditioned weighted designs.
    Qr(Qr),
}

/// The state IRLS leaves behind.
struct Irls {
    coef: Array1<f64>,
    eta: Array1<f64>,
    mu: Array1<f64>,
    dev: f64,
    /// Solve of the last step; `None` for the empty model.
    solve: Option<StepSolve>,
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
    /// Solve every step by the pivoted QR (the equivalence tests of the two paths).
    force_qr: bool,
}

impl Problem<'_> {
    fn linkinv(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| self.link.linkinv(e))
    }

    /// The fit at coefficients `b`, in place: `eta = X b + offset`, `mu = linkinv(eta)`,
    /// `mu.eta(eta)`, and the deviance and whether every `mu` is valid. The values are, bit
    /// for bit, those of `x.dot(b) + offset` (the same matrix-vector product, written into
    /// `eta`), `linkinv`, `mu_eta`, [`Self::deviance`] and [`Self::valid`], without
    /// allocating.
    fn update(
        &self,
        x: &ArrayView2<f64>,
        b: &Array1<f64>,
        eta: &mut Array1<f64>,
        mu: &mut Array1<f64>,
        mu_eta: &mut Array1<f64>,
    ) -> (f64, bool) {
        ndarray::linalg::general_mat_vec_mul(1.0, x, b, 0.0, eta);
        for (e, o) in eta.iter_mut().zip(self.offset) {
            *e += o;
        }
        let link = self.link;
        for ((m, d), &e) in mu.iter_mut().zip(mu_eta.iter_mut()).zip(eta.iter()) {
            (*m, *d) = link.inv_and_deriv(e);
        }
        let dev = self.deviance(mu);
        (dev, self.valid(mu))
    }

    /// The IRLS start from `mustart`: `eta = linkfun(mustart)`, `mu`, `mu.eta` and the
    /// deviance. Each row's values depend only on its (`mustart`, `y`, weight); a binary
    /// or count response takes few such triples, so they are computed once per triple
    /// (a small cache; the values and the deviance sum are the ones the row-by-row
    /// computation gives, bit for bit).
    fn start_from_mustart(
        &self,
        mustart: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>, f64) {
        let n = mustart.len();
        let mut eta = Array1::zeros(n);
        let mut mu = Array1::zeros(n);
        let mut mu_eta = Array1::zeros(n);
        let mut memo: Memo<[f64; 4]> = Memo::default();
        let mut dev = Vec::with_capacity(n);
        for i in 0..n {
            let (m, y, w) = (mustart[i], self.y[i], self.w[i]);
            let [e, mi, d, r] = memo.get_or([m.to_bits(), y.to_bits(), w.to_bits()], || {
                let e = self.link.linkfun(m);
                let (mi, d) = self.link.inv_and_deriv(e);
                [e, mi, d, self.family.dev_resid(y, mi, w)]
            });
            eta[i] = e;
            mu[i] = mi;
            mu_eta[i] = d;
            dev.push(r);
        }
        (eta, mu, mu_eta, dev.into_iter().sum())
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
                solve: None,
                working_weights,
                converged: true,
                boundary: false,
                iterations: 0,
            });
        }
        let (mut eta, mut mu, mut mu_eta, mut coefold, start_dev, start_valid) = match init {
            Init::Start(s) => {
                let s = s.to_owned();
                let (mut eta, mut mu, mut mu_eta) =
                    (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
                let (dev, valid) = self.update(&x, &s, &mut eta, &mut mu, &mut mu_eta);
                (eta, mu, mu_eta, Some(s), dev, valid)
            }
            Init::Mustart(m) => {
                let (eta, mu, mu_eta, dev) = self.start_from_mustart(&m);
                let valid = self.valid(&mu);
                (eta, mu, mu_eta, None, dev, valid)
            }
        };
        if !start_valid {
            return Err(spec_err(
                "cannot find valid starting values: please specify some".into(),
            ));
        }
        let mut devold = start_dev;
        let mut start = coefold.clone().unwrap_or_else(|| Array1::zeros(p));
        let mut converged = false;
        let mut boundary = false;
        let mut prev_change = f64::INFINITY;
        let mut last_solve = None;
        let mut working_weights = Array1::zeros(n);
        let mut iterations = 0;
        // Workspaces of every iteration, sized for all rows: the informative rows, their
        // square-root working weights and their weighted working response. Below
        // LEAN_BLOCKED_MIN_COLUMNS the weighted design itself is formed only when a step needs
        // its rows (the refinement or the QR), from these (the same products, bit for bit).
        let mut zw = resource::try_zeros1(n, "glm IRLS workspace")?;
        let mut wt = resource::try_vec(n, 0.0, "glm IRLS workspace")?;
        let mut good = resource::try_vec(n, 0usize, "glm IRLS workspace")?;
        let gram_len = resource::checked_len(&[p, p], "glm IRLS workspace")?;
        let mut gram = resource::try_vec(gram_len, 0.0, "glm IRLS workspace")?;
        let mut xtwz = Array1::<f64>::zeros(p);
        let mut row = vec![0.0; p];
        // From LEAN_BLOCKED_MIN_COLUMNS columns on, X'WX is the blocked symmetric product of the
        // weighted rows, which stay here for the refinement and the QR.
        let blocked = p >= LEAN_BLOCKED_MIN_COLUMNS;
        let mut xw_rows = if blocked {
            let len = resource::checked_len(&[n, p], "glm IRLS workspace")?;
            resource::try_vec(len, 0.0, "glm IRLS workspace")?
        } else {
            Vec::new()
        };
        // Informative rows of the last step (its working weights are those of the fit).
        let mut ng_last = 0;
        for iter in 1..=self.max_iter {
            resource::checkpoint()?;
            iterations = iter;
            // V(mu) of the rows with positive weight, checked in row order, and the working
            // response and weights of the informative rows (positive weight, mu.eta != 0).
            let mut ng = 0;
            // X'WX (upper triangle, row-major) and X'Wz of the informative rows,
            // accumulated in the same pass.
            gram.fill(0.0);
            xtwz.fill(0.0);
            for i in 0..n {
                let wi = self.w[i];
                if wi <= 0.0 {
                    continue;
                }
                let v = self.family.variance(mu[i]);
                if !v.is_finite() {
                    return Err(EconError::Computation("glm.fit: NAs in V(mu)".into()));
                }
                if v == 0.0 {
                    return Err(EconError::Computation("glm.fit: 0s in V(mu)".into()));
                }
                let d = mu_eta[i];
                if d == 0.0 {
                    continue;
                }
                let z = eta[i] - self.offset[i] + (self.y[i] - mu[i]) / d;
                let wk = (wi * d * d / v).sqrt();
                good[ng] = i;
                wt[ng] = wk;
                let zk = z * wk;
                zw[ng] = zk;
                if blocked {
                    let out = &mut xw_rows[ng * p..(ng + 1) * p];
                    for ((c, &xv), xz) in out.iter_mut().zip(x.row(i)).zip(xtwz.iter_mut()) {
                        *c = xv * wk;
                        *xz += *c * zk;
                    }
                } else {
                    for (j, c) in row.iter_mut().enumerate() {
                        *c = x[[i, j]] * wk;
                    }
                    for (a, &ra) in row.iter().enumerate() {
                        xtwz[a] += ra * zk;
                        for (g, &rb) in gram[a * p + a..(a + 1) * p].iter_mut().zip(&row[a..]) {
                            *g += ra * rb;
                        }
                    }
                }
                ng += 1;
            }
            if ng == 0 {
                return Err(EconError::InsufficientData {
                    required: 1,
                    provided: 0,
                    context: format!("glm: no observations informative at iteration {iter}"),
                });
            }
            ng_last = ng;
            if blocked {
                let a = faer::MatRef::from_row_major_slice(&xw_rows[..ng * p], ng, p);
                // The upper triangle of the row-major `gram` is the lower triangle of its
                // column-major view.
                triangular::matmul(
                    faer::MatMut::from_column_major_slice_mut(&mut gram, p, p),
                    triangular::BlockStructure::TriangularLower,
                    faer::Accum::Replace,
                    a.transpose(),
                    triangular::BlockStructure::Rectangular,
                    a,
                    triangular::BlockStructure::Rectangular,
                    1.0,
                    faer::Par::Seq,
                );
            }
            // The weighted design of the informative rows (`ng x p`, row-major).
            let weighted_rows = || -> EconResult<Cow<'_, [f64]>> {
                if blocked {
                    return Ok(Cow::Borrowed(&xw_rows[..ng * p]));
                }
                let len = resource::checked_len(&[p, ng], "glm IRLS workspace")?;
                let mut xw = resource::try_vec(len, 0.0, "glm IRLS workspace")?;
                for ((r, &i), &wk) in xw.chunks_exact_mut(p).zip(&good[..ng]).zip(&wt[..ng]) {
                    for (c, &xv) in r.iter_mut().zip(x.row(i)) {
                        *c = xv * wk;
                    }
                }
                Ok(Cow::Owned(xw))
            };
            let zw_good = zw.slice(ndarray::s![..ng]);
            // OLS's solve decision on the weighted design: the normal equations when it is
            // of full rank and well conditioned, the pivoted QR otherwise.
            for a in 0..p {
                for b in (a + 1)..p {
                    gram[b * p + a] = gram[a * p + b];
                }
            }
            let normal = if self.force_qr || ng <= p || gram.iter().any(|v| !v.is_finite()) {
                None
            } else {
                let g = ArrayView2::from_shape((p, p), &gram[..])
                    .map_err(|e| EconError::Computation(format!("glm IRLS workspace: {e}")))?;
                match normal_inverse_from_gram(g)? {
                    Some(ni) => {
                        let mut b = ni.inverse.dot(&xtwz);
                        if ni.refine {
                            // OLS's refinement step: b += (X'WX)^{-1} X'W (z - X b)
                            let xw_rows = weighted_rows()?;
                            let xw = ArrayView2::from_shape((ng, p), &xw_rows).map_err(|e| {
                                EconError::Computation(format!("glm IRLS workspace: {e}"))
                            })?;
                            let r = &zw_good - &xw.dot(&b);
                            b += &ni.inverse.dot(&xw.t().dot(&r));
                        }
                        Some((b, ni.inverse))
                    }
                    None => None,
                }
            };
            let (b, solve) = match normal {
                Some((b, inverse)) => (b, StepSolve::Normal(inverse)),
                None => {
                    let len = resource::checked_len(&[p, ng], "glm IRLS workspace")?;
                    let mut by_column = resource::try_vec(len, 0.0, "glm IRLS workspace")?;
                    for (k, r) in weighted_rows()?.chunks_exact(p).enumerate() {
                        for (j, &v) in r.iter().enumerate() {
                            by_column[j * ng + k] = v;
                        }
                    }
                    let qr =
                        Qr::from_columns(by_column, ng, p, GLM_QR_TOL).map_err(|e| match e {
                            EconError::InvalidSpecification { .. } => EconError::DivergenceError {
                                iteration: iter,
                                suggestion: "glm.fit: non-finite working weights or \
                                             responses; supply starting values or check for \
                                             separation"
                                    .into(),
                            },
                            other => other,
                        })?;
                    (qr.coefficients_zeroed(zw_good)?, StepSolve::Qr(qr))
                }
            };
            if !b.iter().all(|v| v.is_finite()) {
                return Err(EconError::DivergenceError {
                    iteration: iter,
                    suggestion: "glm.fit: non-finite coefficients; supply starting values or \
                                 check for separation"
                        .into(),
                });
            }
            last_solve = Some(solve);
            start = b;
            let (mut dev, mut valid) = self.update(&x, &start, &mut eta, &mut mu, &mut mu_eta);
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
                    (dev, valid) = self.update(&x, &start, &mut eta, &mut mu, &mut mu_eta);
                }
                boundary = true;
            }
            if !valid {
                let Some(old) = &coefold else {
                    return Err(Self::no_valid_coefficients(iter));
                };
                warnings.push("step size truncated: out of bounds".into());
                let mut ii = 1;
                while !valid {
                    if ii > self.max_iter {
                        return Err(EconError::DivergenceError {
                            iteration: iter,
                            suggestion: "inner loop 2; cannot correct step size".into(),
                        });
                    }
                    ii += 1;
                    start = (&start + old) / 2.0;
                    (dev, valid) = self.update(&x, &start, &mut eta, &mut mu, &mut mu_eta);
                }
                boundary = true;
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
                if change.is_nan() || change < tol || self.saturated(&eta) || change >= prev_change
                {
                    devold = dev;
                    break;
                }
                prev_change = change;
            }
            devold = dev;
            coefold = Some(start.clone());
        }
        for (&i, &wk) in good[..ng_last].iter().zip(&wt[..ng_last]) {
            working_weights[i] = wk * wk;
        }
        Ok(Irls {
            coef: start,
            eta,
            mu,
            dev: devold,
            solve: last_solve,
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
    fit_glm_solved(x, y, spec, false).map(|(fit, _)| fit)
}

/// [`fit_glm`], and whether its last IRLS step took the normal equations (`true`) or the
/// pivoted QR; `force_qr` solves every step by the QR.
fn fit_glm_solved(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    spec: &GlmSpec<'_>,
    force_qr: bool,
) -> EconResult<(GlmFit, bool)> {
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
    // One pass in memory order; the column scan only names the first offending cell.
    if !x.iter().all(|v| v.is_finite()) {
        for (j, col) in x.columns().into_iter().enumerate() {
            if let Some(i) = col.iter().position(|v| !v.is_finite()) {
                return Err(spec_err(format!(
                    "x contains a non-finite value (row {i}, column {j})"
                )));
            }
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
    // dev_resid(y, ybar, w) depends on (y, w) only: computed once per distinct pair.
    let mut memo: Memo<f64> = Memo::default();
    let d0: f64 = (0..n)
        .filter(|&i| w[i] > 0.0)
        .map(|i| {
            memo.get_or([yv[i].to_bits(), w[i].to_bits(), 0], || {
                family.dev_resid(yv[i], ybar, w[i])
            })
        })
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
        force_qr,
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

    let (rank, aliased, unscaled) = match &fit.solve {
        Some(StepSolve::Qr(qr)) => (qr.rank(), qr.aliased(), qr.unscaled_covariance()?),
        Some(StepSolve::Normal(inverse)) => (p, Vec::new(), inverse.clone()),
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
        // Every weight positive: the deviance of the weighted mean is `d0`, the same sum
        // over the same rows.
        Some(_) if n_ok == n => d0,
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
    let normal_path = matches!(fit.solve, Some(StepSolve::Normal(_)));
    let fit_out = GlmFit {
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
    };
    Ok((fit_out, normal_path))
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

// ═══ The lean logit IRLS (refits that read only the coefficients and fitted values) ═══

/// `glm.control` of the nuisance fits (R's `epsilon` and `maxit`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct GlmControl {
    pub epsilon: f64,
    pub max_iter: usize,
}

impl GlmControl {
    /// `family`/`link` with this control and optional 0/1 training weights.
    pub(crate) fn spec<'a>(
        self,
        family: GlmFamily,
        link: GlmLink,
        train: Option<ArrayView1<'a, f64>>,
    ) -> GlmSpec<'a> {
        let mut spec = GlmSpec::new(family, link);
        spec.epsilon = self.epsilon;
        spec.max_iter = self.max_iter;
        if let Some(t) = train {
            spec = spec.weights(t);
        }
        spec
    }
}

/// A logit fit that stayed on the fast path of [`lean_logit`].
pub(crate) struct LeanLogit {
    pub coefficients: Array1<f64>,
    pub iterations: usize,
    /// Fitted probabilities at `coefficients`.
    pub fitted: Vec<f64>,
    /// Linear predictor at `coefficients` (the offset included).
    pub eta: Vec<f64>,
    /// `(X'WX)^{-1}` of the last step: the unscaled covariance the shared GLM reports on its
    /// normal-equations path.
    pub inverse: Array2<f64>,
    /// Working weights `mu (1 - mu)` of the last step.
    pub working_weights: Vec<f64>,
    /// Deviance at `coefficients`; `NaN` for [`LeanStart::Zero`], which computes none.
    pub deviance: f64,
    /// Deviance of the mean of `y` (no offset): the shared GLM's null deviance of a model
    /// with an intercept and no offset.
    pub null_deviance: f64,
}

/// Where [`lean_logit`] starts.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LeanStart<'a> {
    /// glm.fit's `mustart = (y + 1/2) / 2`; glm.fit's deviance rule.
    Mustart,
    /// These coefficients (as glm's `start`); glm.fit's deviance rule.
    Coef(ArrayView1<'a, f64>),
    /// Zero coefficients (`eta = offset`); the Newton-step rule.
    Zero,
}

/// R's logit-link clamp: a linear predictor beyond it leaves [`lean_logit`]'s fast path.
const LEAN_ETA_LIMIT: f64 = 30.0;

/// Terms per logarithm in [`lean_logit`]'s deviance: `sum ln(1 + exp(eta))` is taken as the sum
/// of the logarithms of products of this many factors. Every factor is below `exp(30) + 1`,
/// so a product stays below `1e209` (finite) and loses at most a few ulps.
const LEAN_LOG_CHUNK: usize = 16;

/// The IRLS of `glm.fit` for a logit fit of `y` (in [0, 1], unit weights) on every row of `x`,
/// with an optional offset, on its common path only: Fisher scoring (Newton's method for the
/// canonical link) solved through the normal equations, by a Cholesky factorisation of
/// `X'WX` when it is well conditioned ([`well_conditioned_inverse`]; the shared GLM takes the
/// normal equations without refinement there).
///
/// - [`LeanStart::Mustart`]: glm.fit's start (`mustart = (y + 1/2) / 2`) and its deviance rule
///   (`|dev - devold| / (|dev| + floor) < epsilon`, the floor as in [`fit_glm`]): the same
///   iterates and iteration count as the shared GLM, up to rounding.
/// - [`LeanStart::Coef`]: the deviance rule from the given coefficients (glm's `start`; a
///   warm start from a fit on overlapping rows).
/// - [`LeanStart::Zero`]: Newton's method from zero coefficients (`eta = offset`), stopped when
///   the largest step is below `epsilon * max(1, |b|)`, or is below `1e-8 * max(1, |b|)` and
///   no longer shrinks (rounding level, for an `epsilon` below what rounding allows). No
///   deviance is computed. For the targeting step, whose estimate starts near 0.
///
/// It computes only the coefficients: no covariance, null deviance, AIC or warnings. The
/// shared GLM computes those, and the hundreds of cross-validation fits of C-TMLE do not use
/// them. `None` whenever the fit leaves the path on which the shared GLM would report nothing:
/// a linear predictor at the link's clamp (separation), a design that is not of full rank or
/// is ill conditioned for the normal equations, a non-finite step, no convergence within
/// `max_iter`, a constant `y`, or a converged fit with a row beyond [`LEAN_MISFIT_ETA`]. The caller
/// then refits with [`fit_glm`], which reports the case. A converged fit is the maximum-likelihood estimate the shared GLM reaches.
pub(crate) fn lean_logit(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    offset: Option<ArrayView1<'_, f64>>,
    start: LeanStart<'_>,
    ctl: GlmControl,
) -> EconResult<Option<LeanLogit>> {
    lean_logit_within(x, y, offset, start, ctl, LEAN_MAX_CONDITION)
}

/// [`lean_logit`] for a design with uncentred covariates or correlated columns (natural scales,
/// long treatment histories): when the condition number of the equilibrated `X'WX` is above
/// [`LEAN_MAX_CONDITION`] and at most [`LEAN_REFINED_MAX_CONDITION`], the steps near the estimate
/// ([`LEAN_REFINE_WITHIN`]) and the last step take the refinement the shared GLM's normal equations
/// take there (`b += (X'WX)^{-1} X'W (z - X b)`) instead of leaving the path. Further out it leaves
/// the path, and the shared GLM takes OLS's decision.
pub(crate) fn lean_logit_refined(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    offset: Option<ArrayView1<'_, f64>>,
    start: LeanStart<'_>,
    ctl: GlmControl,
) -> EconResult<Option<LeanLogit>> {
    lean_logit_within(x, y, offset, start, ctl, LEAN_REFINED_MAX_CONDITION)
}

fn lean_logit_within(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    offset: Option<ArrayView1<'_, f64>>,
    start: LeanStart<'_>,
    ctl: GlmControl,
    max_condition: f64,
) -> EconResult<Option<LeanLogit>> {
    let (n, k) = x.dim();
    if n <= k || k == 0 || ctl.max_iter == 0 {
        return Ok(None);
    }
    if let LeanStart::Coef(c) = start
        && (c.len() != k || !c.iter().all(|v| v.is_finite()))
    {
        return Ok(None);
    }
    let from_zero = matches!(start, LeanStart::Zero);
    let xs = x.as_standard_layout();
    let Some(xrows) = xs.as_slice() else {
        return Ok(None);
    };
    let y = y.as_standard_layout();
    let Some(y) = y.as_slice() else {
        return Ok(None);
    };
    let off_layout = offset.as_ref().map(|o| o.as_standard_layout());
    let off: Option<&[f64]> = match &off_layout {
        Some(o) => match o.as_slice() {
            Some(v) => Some(v),
            None => return Ok(None),
        },
        None => None,
    };
    let xlogx = |v: f64| if v > 0.0 { v * v.ln() } else { 0.0 };
    // dev = 2 (sum_i c_i - sum_i (y_i eta_i - ln(1 + exp(eta_i)))), c_i = y ln y + (1-y) ln(1-y)
    let (mut sum_y, mut sum_c) = (0.0, 0.0);
    let mut binary = true;
    for &v in y {
        if !(0.0..=1.0).contains(&v) {
            return Ok(None);
        }
        sum_y += v;
        if v != 0.0 && v != 1.0 {
            binary = false;
            sum_c += xlogx(v) + xlogx(1.0 - v);
        }
    }
    let nf = n as f64;
    let ybar = sum_y / nf;
    if !(ybar > 0.0 && ybar < 1.0) {
        return Ok(None);
    }
    let d0 = 2.0 * (sum_c - sum_y * ybar.ln() - (nf - sum_y) * (1.0 - ybar).ln());
    let per = d0 / nf;
    let floor = if per.is_finite() && per > 0.0 {
        0.1 * per
    } else {
        0.1
    };
    let mut eta = resource::try_vec(n, 0.0, "logit IRLS workspace")?;
    let mut mu = resource::try_vec(n, 0.0, "logit IRLS workspace")?;
    // the fit each step starts from: its working weights, and a late refinement
    let mut eta_step = resource::try_vec(n, 0.0, "logit IRLS workspace")?;
    let mut mu_step = resource::try_vec(n, 0.0, "logit IRLS workspace")?;
    // mu at `eta`, and the deviance when `deviance`; `None` at the clamp
    let eval = |eta: &[f64], mu: &mut [f64], deviance: bool| -> Option<f64> {
        let (mut ye, mut logs, mut prod, mut m) = (0.0, 0.0, 1.0, 0);
        for ((&e, mi), &yi) in eta.iter().zip(mu.iter_mut()).zip(y) {
            if e.is_nan() || e.abs() >= LEAN_ETA_LIMIT {
                return None;
            }
            let t = e.exp();
            *mi = t / (1.0 + t);
            if deviance {
                ye += yi * e;
                prod *= 1.0 + t;
                m += 1;
                if m == LEAN_LOG_CHUNK {
                    logs += prod.ln();
                    prod = 1.0;
                    m = 0;
                }
            }
        }
        if !deviance {
            return Some(0.0);
        }
        let dev = 2.0 * (sum_c - ye + logs + prod.ln());
        dev.is_finite().then_some(dev)
    };
    match start {
        LeanStart::Zero => {
            if let Some(o) = off {
                eta.copy_from_slice(o);
            }
        }
        LeanStart::Mustart => {
            if !binary {
                for (e, &v) in eta.iter_mut().zip(y) {
                    let m = (v + 0.5) / 2.0;
                    *e = (m / (1.0 - m)).ln();
                }
            }
        }
        LeanStart::Coef(c) => {
            for (i, (row, e)) in xrows.chunks_exact(k).zip(eta.iter_mut()).enumerate() {
                let mut v = off.map_or(0.0, |o| o[i]);
                for (xa, ba) in row.iter().zip(c.iter()) {
                    v += xa * ba;
                }
                *e = v;
            }
        }
    }
    // glm.fit's start on a 0/1 response needs no logarithm per row: mu is 1/4 or 3/4, eta is
    // -+ln(3), and each unit's deviance is 2 ln(4/3).
    let mustart_binary = binary && matches!(start, LeanStart::Mustart);
    if mustart_binary {
        let ln3 = 3.0_f64.ln();
        for ((e, m), &v) in eta.iter_mut().zip(mu.iter_mut()).zip(y) {
            (*e, *m) = if v == 1.0 { (ln3, 0.75) } else { (-ln3, 0.25) };
        }
    }
    let first_dev = if mustart_binary {
        Some(2.0 * nf * (4.0_f64 / 3.0).ln())
    } else {
        eval(&eta, &mut mu, !from_zero)
    };
    let Some(mut devold) = first_dev else {
        return Ok(None);
    };
    let mut g = vec![0.0; k * k];
    let mut inverse = Array2::<f64>::zeros((k, k));
    let mut chol = vec![0.0; k * k];
    let mut rhs = vec![0.0; k];
    let (mut solve_scratch, mut scale_scratch) = (vec![0.0; k], vec![0.0; k]);
    let mut b = Array1::<f64>::zeros(k);
    let mut b_old = Array1::<f64>::zeros(k);
    let mut prev_step = f64::INFINITY;
    // Whether the next step is likely to be the last, and takes its refinement as it goes.
    // glm.fit's start is far from the estimate; a warm start, or the Newton-step rule (no
    // deviance), may stop at any step.
    let mut near_estimate = !matches!(start, LeanStart::Mustart);
    // the weighted rows of the blocked cross-product
    let blocked = k >= LEAN_BLOCKED_MIN_COLUMNS;
    let mut sqrt_w_rows = if blocked {
        let len = resource::checked_len(&[n, k], "logit IRLS workspace")?;
        resource::try_vec(len, 0.0, "logit IRLS workspace")?
    } else {
        Vec::new()
    };
    for iter in 1..=ctl.max_iter {
        resource::checkpoint()?;
        // X'WX (upper triangle) and X'Wz, z = eta - offset + (y - mu) / W, W = mu (1 - mu)
        rhs.fill(0.0);
        if blocked {
            for (i, ((row, out), (&m, (&e, &yi)))) in xrows
                .chunks_exact(k)
                .zip(sqrt_w_rows.chunks_exact_mut(k))
                .zip(mu.iter().zip(eta.iter().zip(y)))
                .enumerate()
            {
                let wi = m * (1.0 - m);
                let wz = wi * (e - off.map_or(0.0, |o| o[i])) + (yi - m);
                let sw = wi.sqrt();
                for ((rh, o), &xa) in rhs.iter_mut().zip(out.iter_mut()).zip(row) {
                    *rh += xa * wz;
                    *o = sw * xa;
                }
            }
            let a = faer::MatRef::from_row_major_slice(&sqrt_w_rows, n, k);
            // The upper triangle of the row-major `g` is the lower triangle of its column-major
            // view.
            triangular::matmul(
                faer::MatMut::from_column_major_slice_mut(&mut g, k, k),
                triangular::BlockStructure::TriangularLower,
                faer::Accum::Replace,
                a.transpose(),
                triangular::BlockStructure::Rectangular,
                a,
                triangular::BlockStructure::Rectangular,
                1.0,
                faer::Par::Seq,
            );
        } else {
            g.fill(0.0);
            let cols = WeightedColumns {
                xrows,
                mu: &mu,
                eta: &eta,
                y,
                off,
            };
            match k {
                1 => cols.cross_fixed::<1>(&mut g, &mut rhs),
                2 => cols.cross_fixed::<2>(&mut g, &mut rhs),
                3 => cols.cross_fixed::<3>(&mut g, &mut rhs),
                4 => cols.cross_fixed::<4>(&mut g, &mut rhs),
                5 => cols.cross_fixed::<5>(&mut g, &mut rhs),
                6 => cols.cross_fixed::<6>(&mut g, &mut rhs),
                _ => cols.cross(k, &mut g, &mut rhs),
            }
        }
        let Some(kappa) = inverse_and_condition(
            &g,
            k,
            &mut chol,
            &mut solve_scratch,
            &mut scale_scratch,
            &mut inverse,
        ) else {
            return Ok(None);
        };
        if kappa > max_condition {
            return Ok(None);
        }
        b_old.assign(&b);
        ndarray::linalg::general_mat_vec_mul(
            1.0,
            &inverse,
            &ArrayView1::from(&rhs[..]),
            0.0,
            &mut b,
        );
        let needs_refinement = kappa > LEAN_MAX_CONDITION;
        if needs_refinement && near_estimate {
            let cols = WeightedColumns {
                xrows,
                mu: &mu,
                eta: &eta,
                y,
                off,
            };
            cols.refine(k, &inverse, &mut rhs, &mut b);
        }
        if !b.iter().all(|v| v.is_finite()) {
            return Ok(None);
        }
        let Some(bs) = b.as_slice() else {
            return Ok(None);
        };
        std::mem::swap(&mut eta, &mut eta_step);
        lean_linear_predictor(xrows, off, bs, &mut eta);
        std::mem::swap(&mut mu, &mut mu_step);
        let Some(mut dev) = eval(&eta, &mut mu, !from_zero) else {
            return Ok(None);
        };
        let change = (dev - devold).abs() / (dev.abs() + floor);
        let done = if from_zero {
            let step = b
                .iter()
                .zip(b_old.iter())
                .fold(0.0_f64, |m, (u, v)| m.max((u - v).abs()));
            let scale = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            // rounding level: a small step that no longer shrinks
            let done = step <= ctl.epsilon * scale
                || (iter > 2 && step >= prev_step && step <= 1e-8 * scale);
            prev_step = step;
            done
        } else {
            change < ctl.epsilon
        };
        if done {
            if needs_refinement && !near_estimate {
                // The rule stopped at a step that took no refinement: it takes it now, from the
                // fit it started at. Only the deviance rule gets here (`near_estimate` is always
                // set under the Newton-step rule).
                let cols = WeightedColumns {
                    xrows,
                    mu: &mu_step,
                    eta: &eta_step,
                    y,
                    off,
                };
                cols.refine(k, &inverse, &mut rhs, &mut b);
                if !b.iter().all(|v| v.is_finite()) {
                    return Ok(None);
                }
                let Some(bs) = b.as_slice() else {
                    return Ok(None);
                };
                lean_linear_predictor(xrows, off, bs, &mut eta);
                let Some(refined) = eval(&eta, &mut mu, true) else {
                    return Ok(None);
                };
                if (refined - devold).abs() / (refined.abs() + floor) >= ctl.epsilon {
                    return Ok(None);
                }
                dev = refined;
            }
            if eta.iter().zip(y).any(|(&e, &v)| {
                (v < 1.0 && e > LEAN_MISFIT_ETA) || (v > 0.0 && e < -LEAN_MISFIT_ETA)
            }) {
                return Ok(None);
            }
            for m in mu_step.iter_mut() {
                *m *= 1.0 - *m;
            }
            return Ok(Some(LeanLogit {
                coefficients: b,
                iterations: iter,
                fitted: mu,
                eta,
                inverse,
                working_weights: mu_step,
                deviance: if from_zero { f64::NAN } else { dev },
                null_deviance: d0,
            }));
        }
        near_estimate = from_zero || change < LEAN_REFINE_WITHIN;
        devold = dev;
    }
    Ok(None)
}

/// `eta = offset + X b`, with `eta.len()` rows of `b.len()` columns in `xrows`.
fn lean_linear_predictor(xrows: &[f64], off: Option<&[f64]>, b: &[f64], eta: &mut [f64]) {
    for (i, (row, e)) in xrows.chunks_exact(b.len()).zip(eta.iter_mut()).enumerate() {
        let mut v = off.map_or(0.0, |o| o[i]);
        for (xa, ba) in row.iter().zip(b) {
            v += xa * ba;
        }
        *e = v;
    }
}

/// The rows and the current fit of a [`lean_logit_within`] step.
struct WeightedColumns<'a> {
    xrows: &'a [f64],
    mu: &'a [f64],
    eta: &'a [f64],
    y: &'a [f64],
    off: Option<&'a [f64]>,
}

impl WeightedColumns<'_> {
    /// OLS's refinement step on the solve `b` of this fit's normal equations:
    /// `b += (X'WX)^{-1} X'W (z - X b)`. `rhs` is scratch.
    fn refine(&self, k: usize, inverse: &Array2<f64>, rhs: &mut [f64], b: &mut Array1<f64>) {
        rhs.fill(0.0);
        for (i, (row, (&m, (&e, &yi)))) in self
            .xrows
            .chunks_exact(k)
            .zip(self.mu.iter().zip(self.eta.iter().zip(self.y)))
            .enumerate()
        {
            let wi = m * (1.0 - m);
            let wz = wi * (e - self.off.map_or(0.0, |o| o[i])) + (yi - m);
            let r = wz - wi * ArrayView1::from(row).dot(&*b);
            for (rh, &xa) in rhs.iter_mut().zip(row) {
                *rh += xa * r;
            }
        }
        ndarray::linalg::general_mat_vec_mul(1.0, inverse, &ArrayView1::from(&*rhs), 1.0, b);
    }

    /// Adds the upper triangle of `X'WX` to `g` (`k x k`, row-major) and `X'Wz` to `rhs`, row by
    /// row, with `W = mu (1 - mu)` and `z = eta - offset + (y - mu) / W`.
    fn cross(&self, k: usize, g: &mut [f64], rhs: &mut [f64]) {
        for (i, ((row, &m), (&e, &yi))) in self
            .xrows
            .chunks_exact(k)
            .zip(self.mu)
            .zip(self.eta.iter().zip(self.y))
            .enumerate()
        {
            let wi = m * (1.0 - m);
            let eo = e - self.off.map_or(0.0, |o| o[i]);
            let wz = wi * eo + (yi - m);
            for (a, (ga, rh)) in g.chunks_exact_mut(k).zip(rhs.iter_mut()).enumerate() {
                let xa = row[a];
                *rh += xa * wz;
                let wxa = wi * xa;
                for (gc, xc) in ga[a..].iter_mut().zip(&row[a..]) {
                    *gc += wxa * xc;
                }
            }
        }
    }

    /// [`Self::cross`] at a width known when compiling, which keeps the sums in registers: the
    /// same products, added in the same order, so the same bits.
    fn cross_fixed<const K: usize>(&self, g: &mut [f64], rhs: &mut [f64]) {
        let mut acc = [[0.0_f64; K]; K];
        let mut xz = [0.0_f64; K];
        let (rows, _) = self.xrows.as_chunks::<K>();
        for (i, ((row, &m), (&e, &yi))) in rows
            .iter()
            .zip(self.mu)
            .zip(self.eta.iter().zip(self.y))
            .enumerate()
        {
            let wi = m * (1.0 - m);
            let eo = e - self.off.map_or(0.0, |o| o[i]);
            let wz = wi * eo + (yi - m);
            add_weighted_row(&mut acc, &mut xz, row, wi, wz);
        }
        g.as_chunks_mut::<K>().0.copy_from_slice(&acc);
        rhs.copy_from_slice(&xz);
    }
}

/// `acc += w x x'` (the upper triangle) and `xz += v x`, for a row `x` of a fixed width.
#[inline(always)]
pub(crate) fn add_weighted_row<const K: usize>(
    acc: &mut [[f64; K]; K],
    xz: &mut [f64; K],
    row: &[f64; K],
    w: f64,
    v: f64,
) {
    for a in 0..K {
        let xa = row[a];
        xz[a] += xa * v;
        let wxa = w * xa;
        for c in a..K {
            acc[a][c] += wxa * row[c];
        }
    }
}

/// Largest 1-norm condition number of the equilibrated `X'WX` (`E = S X'WX S`, `S` the inverse
/// square roots of its diagonal) that [`lean_logit`] solves by its own Cholesky: OLS's normal
/// equations take a matrix with `1 / kappa_1(E)` at or above `OLS_REFINE_BELOW_RCOND` (1e-2)
/// without refinement (`regression::ols::normal_inverse_from_gram`, the decision of the shared
/// GLM's IRLS); anything else leaves the lean path.
const LEAN_MAX_CONDITION: f64 = 100.0;

/// Largest such condition number on the path of [`lean_logit_refined`], well inside what OLS's
/// normal equations take with a refinement step (`1 / kappa_1(E)` at or above
/// `OLS_NORMAL_EQUATIONS_MIN_RCOND`, 1e-8).
const LEAN_REFINED_MAX_CONDITION: f64 = 1e4;

/// Relative deviance change of a step below which the next step of [`lean_logit_refined`] takes
/// the refinement as it goes. The steps before need none: Newton's method corrects their rounding
/// (of order `eps * kappa`, far from the estimate). A step the deviance rule stops at without one,
/// which happens when the deviance change falls from above this to below the rule's `epsilon` in
/// one step, takes it afterwards from the fit it started at, so every fit ends on a refined step.
const LEAN_REFINE_WITHIN: f64 = 1e-3;

/// Linear predictor beyond which a row that its fitted probability does not explain (`y < 1` at a
/// large `eta`, `y > 0` at a very negative one) sends a converged fit from [`lean_logit_within`] to
/// [`fit_glm`]. On such a row `fit_glm`'s `1 - mu` has lost most of its digits, and they reach its
/// working weights and deviance, so its iterations, convergence status and coefficients are no
/// longer those of the lean path, which works from `ln(1 + exp(eta))`. The lean numbers are the more
/// accurate there (score 6e-15 against 2.3e-5 on the data of
/// `test_fit_logit_lean_keeps_fit_glm_past_the_misfit_bound`), but `fit_glm`, like R's `glm`, reports
/// such a fit as not converged, and this path does not change what is reported. Measured on 1,535
/// heavy-tailed designs at kappa between 100 and 1e4: 15 keeps 1,491 on the path with no iteration
/// or status difference (12: 1,462, 17: 1,507, 20: 1,522, with the coefficient gap growing from
/// 7.6e-12 through 8.2e-11 at 15 to 8.1e-9 at 20).
const LEAN_MISFIT_ETA: f64 = 15.0;

/// From this many columns both IRLS loops, [`lean_logit`]'s and [`fit_glm`]'s, form `X'WX` by the
/// blocked symmetric product of the square-root-weighted rows, which beats the row-by-row
/// accumulation once the rows outgrow the vector registers; below it the row pass stays. So every
/// [`fit_glm`] caller at or above this width gets it. [`fit_glm`] keeps the weighted rows of the
/// step for its refinement and QR solves.
const LEAN_BLOCKED_MIN_COLUMNS: usize = 12;

/// The inverse of the symmetric `k x k` matrix whose upper triangle is `g` (row-major), into
/// `inverse`, when its Cholesky factorisation exists and the 1-norm condition number of its
/// equilibrated form is at most [`LEAN_MAX_CONDITION`]; `false` otherwise. `chol` (`k x k`),
/// `y` and `r` (length `k`) are scratch, allocated once by the caller.
pub(crate) fn well_conditioned_inverse(
    g: &[f64],
    k: usize,
    chol: &mut [f64],
    y: &mut [f64],
    r: &mut [f64],
    inverse: &mut Array2<f64>,
) -> bool {
    inverse_and_condition(g, k, chol, y, r, inverse).is_some_and(|c| c <= LEAN_MAX_CONDITION)
}

/// [`well_conditioned_inverse`]'s inverse and the condition number it decides on: `None` when
/// the Cholesky factorisation does not exist or the condition number is not finite.
fn inverse_and_condition(
    g: &[f64],
    k: usize,
    chol: &mut [f64],
    y: &mut [f64],
    r: &mut [f64],
    inverse: &mut Array2<f64>,
) -> Option<f64> {
    let at = |a: usize, c: usize| if c >= a { g[a * k + c] } else { g[c * k + a] };
    // lower-triangular L with L L' = G, row-major in `chol`
    chol.fill(0.0);
    for i in 0..k {
        for j in 0..=i {
            let mut v = at(i, j);
            for m in 0..j {
                v -= chol[i * k + m] * chol[j * k + m];
            }
            if i == j {
                if !(v > 0.0 && v.is_finite()) {
                    return None;
                }
                chol[i * k + i] = v.sqrt();
            } else {
                chol[i * k + j] = v / chol[j * k + j];
            }
        }
    }
    // G^-1 = L^-T L^-1, column by column: solve L y = e_c, then L' x = y
    for c in 0..k {
        for i in 0..k {
            let mut v = if i == c { 1.0 } else { 0.0 };
            for m in 0..i {
                v -= chol[i * k + m] * y[m];
            }
            y[i] = v / chol[i * k + i];
        }
        for i in (0..k).rev() {
            let mut v = y[i];
            for m in (i + 1)..k {
                v -= chol[m * k + i] * inverse[[m, c]];
            }
            inverse[[i, c]] = v / chol[i * k + i];
        }
    }
    // kappa_1(E) = ||E||_1 ||E^-1||_1, E = S G S, E^-1 = S^-1 G^-1 S^-1
    for (i, ri) in r.iter_mut().enumerate().take(k) {
        *ri = at(i, i).sqrt();
    }
    let (mut e_norm, mut einv_norm) = (0.0f64, 0.0f64);
    for c in 0..k {
        let (mut ce, mut ci) = (0.0, 0.0);
        for i in 0..k {
            ce += (at(i, c) / (r[i] * r[c])).abs();
            ci += (inverse[[i, c]] * r[i] * r[c]).abs();
        }
        e_norm = e_norm.max(ce);
        einv_norm = einv_norm.max(ci);
    }
    let kappa = e_norm * einv_norm;
    kappa.is_finite().then_some(kappa)
}

/// [`fit_glm`] for a binomial logit fit of a 0/1 response with no prior weights, offset or
/// coefficient refinement, by [`lean_logit_refined`] when the fit stays on its path: the refits of
/// propensity and outcome models (IPW, AIPW, WeightIt, mediation, the g-formula) that read the
/// coefficients, fitted values, covariance and warnings but not the AIC or a refitted null
/// deviance. The IRLS is the same (glm.fit's start or `spec.start`, Fisher scoring through the
/// normal equations, the deviance rule at `spec.epsilon` within `spec.max_iter`), so the fit
/// is the shared GLM's up to rounding. A lean fit is converged, of full rank, with no warning. This
/// is what [`fit_glm`] reports on the designs measured, not a proof. On the default spec over
/// 3,276 designs of 13 kinds (1,790 fits on the path) no iteration, status or warning differs.
/// The largest gaps are 4.5e-9 on a coefficient, 9.1e-8 on an SE and 2.1e-6 on a covariance entry,
/// on quasi-separated fits of n = 15 to 80 with a rare binary covariate; on 1,491 heavy-tailed fits
/// that stay on the path they are 8.2e-11 and 2.7e-11.
/// Any other case (separation, a rank-deficient or ill-conditioned weighted design, no convergence,
/// a row beyond [`LEAN_MISFIT_ETA`], invalid input, another
/// family, link or spec) is [`fit_glm`]'s own fit, errors and warnings included.
///
/// The fields of a lean fit: `vcov` is the inverse of the last step's `X'WX` (dispersion 1),
/// `deviance` and `null_deviance` are computed with fewer logarithms (a few ulps), and the
/// log-likelihood and AIC follow from the deviance, which for a 0/1 response is `-2` times the
/// binomial log-likelihood.
pub(crate) fn fit_logit_lean(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    spec: &GlmSpec<'_>,
) -> EconResult<GlmFit> {
    match logit_lean_path(x, y, spec)? {
        Some(fit) => Ok(fit),
        None => fit_glm(x, y, spec),
    }
}

/// The fit of [`fit_logit_lean`] when it stays on the lean path, `None` when it needs [`fit_glm`].
fn logit_lean_path(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    spec: &GlmSpec<'_>,
) -> EconResult<Option<GlmFit>> {
    let plain = spec.family == GlmFamily::Binomial
        && spec.link == GlmLink::Logit
        && spec.weights.is_none()
        && spec.offset.is_none()
        && spec.coef_tol.is_none()
        && y.len() == x.nrows()
        && y.iter().all(|&v| v == 0.0 || v == 1.0);
    // an intercept column: lean_logit's null deviance is the intercept-only model's
    let has_ones = (0..x.ncols()).any(|j| x.column(j).iter().all(|&v| v == 1.0));
    if plain && has_ones && spec.epsilon.is_finite() && spec.epsilon > 0.0 {
        let start = spec.start.map_or(LeanStart::Mustart, LeanStart::Coef);
        let ctl = GlmControl {
            epsilon: spec.epsilon,
            max_iter: spec.max_iter,
        };
        if let Some(fit) = lean_logit_refined(x, y, None, start, ctl)? {
            let (n, p) = x.dim();
            let std_errors = Array1::from_iter(fit.inverse.diag().iter().map(|v| v.sqrt()));
            let aic = fit.deviance + 2.0 * p as f64;
            return Ok(Some(GlmFit {
                coefficients: fit.coefficients,
                std_errors,
                vcov: fit.inverse,
                fitted: Array1::from_vec(fit.fitted),
                linear_predictor: Array1::from_vec(fit.eta),
                working_weights: Array1::from_vec(fit.working_weights),
                deviance: fit.deviance,
                null_deviance: fit.null_deviance,
                log_likelihood: p as f64 - aic / 2.0,
                aic,
                dispersion: 1.0,
                df_residual: n - p,
                rank: p,
                aliased: Vec::new(),
                converged: true,
                iterations: fit.iterations,
                max_iter: spec.max_iter,
                epsilon: spec.epsilon,
                boundary: false,
                separation: false,
                warnings: Vec::new(),
            }));
        }
    }
    Ok(None)
}

/// [`fit_propensity`] with the logit link through [`fit_logit_lean`]: the same checks and
/// the same fit up to rounding, without the AIC and null-deviance work a propensity refit
/// does not read.
pub(crate) fn fit_propensity_lean(
    x: ArrayView2<f64>,
    treat: ArrayView1<f64>,
) -> EconResult<PropensityFit> {
    if let Some((i, v)) = treat
        .iter()
        .enumerate()
        .find(|(_, v)| **v != 0.0 && **v != 1.0)
    {
        return Err(spec_err(format!(
            "propensity treatment must be coded 0/1 (row {i} is {v})"
        )));
    }
    let fit = fit_logit_lean(x, treat, &GlmSpec::logit())?;
    Ok(PropensityFit {
        scores: fit.fitted.clone(),
        fit,
    })
}

/// Relative coefficient change at which [`fit_propensity`] stops refining after the
/// deviance rule ([`GlmSpec::coef_tol`]); applied to the non-canonical links (probit,
/// cloglog) only.
pub const PROPENSITY_COEF_TOL: f64 = 1e-10;

/// Propensity scores: the binomial MLE of `treat` (coded 0/1) on `x` with the logit,
/// probit or complementary log-log link. The caller includes the intercept column.
///
/// For the probit and cloglog links the fit keeps iterating after the deviance rule until
/// the relative coefficient change is below [`PROPENSITY_COEF_TOL`] (or a fitted
/// probability reaches the link's clamp), so the non-canonical links, whose Fisher scoring
/// converges only linearly, reach the MLE as closely as the canonical logit. The logit
/// stops at the deviance rule (`glm.fit`'s): Newton's method converges quadratically, and
/// extra passes would cost `O(n p^2)` each for nothing.
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
    if link != GlmLink::Logit {
        spec.coef_tol = Some(PROPENSITY_COEF_TOL);
    }
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

    /// Intercept, `binary` columns of mean 1/2 and noise columns: the binary columns correlate
    /// with the intercept, so `X'WX` is less well conditioned the more of them there are.
    fn history_design(
        n: usize,
        binary: usize,
        noise: usize,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(17);
        let x = Array2::from_shape_fn((n, 1 + binary + noise), |(_, j)| {
            if j == 0 {
                1.0
            } else if j <= binary {
                f64::from(u8::from(rng.gen_range(0.0..1.0) < 0.5))
            } else {
                rng.gen_range(-1.0..1.0)
            }
        });
        let eta = x.dot(&Array1::from_shape_fn(x.ncols(), |j| {
            if j == 0 { -0.8 } else { 0.25 - 0.03 * j as f64 }
        }));
        let yb = eta.mapv(|e| f64::from(u8::from(rng.gen_range(0.0..1.0) < plogis(e))));
        let yf =
            eta.mapv(|e| (0.8 * plogis(e) + 0.2 * rng.gen_range(0.0..1.0_f64)).clamp(0.0, 1.0));
        (x, yb, yf)
    }

    #[test]
    fn test_lean_logit_refined_matches_the_shared_glm_on_long_histories() {
        let ctl = GlmControl {
            epsilon: 1e-12,
            max_iter: 100,
        };
        // 3 and 11 columns take the row pass, 16 and 31 the blocked cross-product
        for (binary, noise, refined) in
            [(1, 1, false), (3, 7, false), (5, 10, false), (10, 20, true)]
        {
            let (x, yb, yf) = history_design(3000, binary, noise);
            for (y, family) in [(&yb, GlmFamily::Binomial), (&yf, GlmFamily::QuasiBinomial)] {
                let glm =
                    fit_glm(x.view(), y.view(), &ctl.spec(family, GlmLink::Logit, None)).unwrap();
                let plain = lean_logit(x.view(), y.view(), None, LeanStart::Mustart, ctl).unwrap();
                assert_eq!(plain.is_none(), refined, "{binary} binary columns");
                let lean = lean_logit_refined(x.view(), y.view(), None, LeanStart::Mustart, ctl)
                    .unwrap()
                    .expect("lean path");
                assert_eq!(lean.iterations, glm.iterations);
                for (a, b) in lean.coefficients.iter().zip(&glm.coefficients) {
                    assert!((a - b).abs() <= 1e-10 * b.abs().max(1e-2), "{a} vs {b}");
                }
                for (a, b) in lean.fitted.iter().zip(&glm.fitted) {
                    assert!((a - b).abs() <= 1e-10, "{a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn test_lean_logit_refined_leaves_the_path_on_near_collinear_designs() {
        use rand::{Rng, SeedableRng};
        let (mut x, yb, _) = history_design(2000, 4, 12);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let ctl = GlmControl {
            epsilon: 1e-10,
            max_iter: 100,
        };
        for i in 0..2000 {
            x[[i, 16]] = x[[i, 15]] + 1e-5 * rng.gen_range(-1.0..1.0);
        }
        let lean = lean_logit_refined(x.view(), yb.view(), None, LeanStart::Mustart, ctl).unwrap();
        assert!(lean.is_none());
    }

    /// `glm.fit` for a logit with X'WX accumulated row by row: every step takes the same solve
    /// decision as the shared GLM (the refinement), on the gram of this accumulation. Returns
    /// the coefficients, `(X'WX)^{-1}` of the last step, the step count and whether a step
    /// refined.
    fn naive_row_by_row_logit(
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: &Array1<f64>,
        eps: f64,
    ) -> (Array1<f64>, Array2<f64>, usize, bool) {
        let (n, p) = x.dim();
        let xlogy = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
        let deviance = |mu: &Array1<f64>| -> f64 {
            (0..n)
                .map(|i| 2.0 * w[i] * (xlogy(y[i], mu[i]) + xlogy(1.0 - y[i], 1.0 - mu[i])))
                .sum()
        };
        let mut mu = Array1::from_shape_fn(n, |i| (w[i] * y[i] + 0.5) / (w[i] + 1.0));
        let mut eta = mu.mapv(|m: f64| (m / (1.0 - m)).ln());
        let mut devold = deviance(&mu);
        let mut refined = false;
        for iter in 1..=100 {
            let mut gram = Array2::<f64>::zeros((p, p));
            let mut rhs = Array1::<f64>::zeros(p);
            let mut rows = Vec::new();
            for i in 0..n {
                if w[i] == 0.0 {
                    continue;
                }
                let wk = w[i] * mu[i] * (1.0 - mu[i]);
                let z = eta[i] + (y[i] - mu[i]) / (mu[i] * (1.0 - mu[i]));
                for a in 0..p {
                    rhs[a] += x[[i, a]] * wk * z;
                    for c in 0..p {
                        gram[[a, c]] += x[[i, a]] * wk * x[[i, c]];
                    }
                }
                rows.push((i, wk, z));
            }
            let ni = normal_inverse_from_gram(gram.view())
                .unwrap()
                .expect("a design the normal equations take");
            let mut b = ni.inverse.dot(&rhs);
            if ni.refine {
                refined = true;
                let mut r = Array1::<f64>::zeros(p);
                for &(i, wk, z) in &rows {
                    let res = wk * (z - x.row(i).dot(&b));
                    for a in 0..p {
                        r[a] += x[[i, a]] * res;
                    }
                }
                b = b + ni.inverse.dot(&r);
            }
            eta = x.dot(&b);
            mu = eta.mapv(plogis);
            let dev = deviance(&mu);
            if (dev - devold).abs() / (dev.abs() + 0.1) < eps {
                return (b, ni.inverse, iter, refined);
            }
            devold = dev;
        }
        panic!("the naive reference did not converge");
    }

    #[test]
    fn test_fit_glm_blocked_gram_matches_a_row_by_row_accumulation() {
        use rand::{Rng, SeedableRng};
        const EPS: f64 = 1e-12;
        // 16 columns; a second design whose near-duplicate column makes the normal equations
        // refine each step.
        let (plain, y, _) = history_design(1500, 5, 10);
        let mut near_duplicate = plain.clone();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        for i in 0..1500 {
            near_duplicate[[i, 15]] = near_duplicate[[i, 14]] + 1e-3 * rng.gen_range(-1.0..1.0);
        }
        assert!(plain.ncols() >= LEAN_BLOCKED_MIN_COLUMNS);
        // every prior weight 1 (all rows informative), and every seventh row with weight 0
        let all = Array1::<f64>::ones(1500);
        let some = Array1::from_shape_fn(1500, |i| if i % 7 == 0 { 0.0 } else { 1.0 });
        let close = |what: &str, a: &[f64], b: &[f64], tol: f64| {
            let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (u, v) in a.iter().zip(b) {
                assert!((u - v).abs() <= tol * scale, "{what}: {u} vs {v}");
            }
        };
        for (design, must_refine) in [(&plain, false), (&near_duplicate, true)] {
            for w in [&all, &some] {
                let wv: ArrayView1<'_, f64> = w.view();
                let mut s = GlmSpec::logit();
                s.epsilon = EPS;
                s.weights = Some(wv);
                let fit = fit_glm(design.view(), y.view(), &s).unwrap();
                let (b, inverse, iterations, refined) = naive_row_by_row_logit(design, &y, w, EPS);
                assert_eq!(refined, must_refine);
                assert_eq!(fit.iterations, iterations);
                close(
                    "coefficients",
                    fit.coefficients.as_slice().unwrap(),
                    b.as_slice().unwrap(),
                    1e-10,
                );
                // (X'WX)^{-1} of the last step, the dispersion of a binomial being 1. An inverse
                // holds the rounding of the gram times its condition number, about 1e7 with the
                // near-duplicate.
                close(
                    "vcov",
                    fit.vcov.as_slice().unwrap(),
                    inverse.as_standard_layout().as_slice().unwrap(),
                    if must_refine { 1e-8 } else { 1e-10 },
                );
                // The QR, from the weighted rows the blocked product kept.
                let (qr, _) = fit_glm_solved(design.view(), y.view(), &s, true).unwrap();
                assert_eq!(qr.iterations, iterations);
                close(
                    "QR coefficients",
                    qr.coefficients.as_slice().unwrap(),
                    b.as_slice().unwrap(),
                    1e-10,
                );
            }
        }
    }

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
        // 1e-8: the logit stops at the deviance rule (Newton, quadratic convergence), and the
        // probit's coefficient refinement (PROPENSITY_COEF_TOL) removes the deviance rule's
        // linear-convergence gap
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
        // the logit is not refined: Newton converges quadratically, so fit_propensity runs
        // exactly the deviance rule's iterations (no extra O(n p^2) passes)
        let ld = fit_glm(x.view(), yb.view(), &GlmSpec::logit()).unwrap();
        assert_eq!(pl.fit.iterations, ld.iterations);
        assert_eq!(pl.fit.coefficients, ld.coefficients);
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

    /// The fused inverse link and derivative of the logit and log links are, bit for bit,
    /// `linkinv` and `mu_eta`, at the clamps and between them.
    #[test]
    fn test_glm_inv_and_deriv_match_separate_link_functions() {
        let mut etas = vec![-LOGIT_THRESH, LOGIT_THRESH, -800.0, 800.0, 0.0, -0.0];
        etas.extend((0..4001).map(|i| -40.0 + 0.02 * i as f64 + 1e-3 * ((i * 37 % 11) as f64)));
        for link in [
            GlmLink::Logit,
            GlmLink::Log,
            GlmLink::Probit,
            GlmLink::Cloglog,
        ] {
            for &e in &etas {
                let (m, d) = link.inv_and_deriv(e);
                assert_eq!(
                    (m.to_bits(), d.to_bits()),
                    (link.linkinv(e).to_bits(), link.mu_eta(e).to_bits()),
                    "{link:?} at {e}"
                );
            }
        }
    }

    /// The relative gap of `a` from `b` (absolute below 1).
    fn rel_gap(a: f64, b: f64) -> f64 {
        if a == b {
            return 0.0;
        }
        (a - b).abs() / b.abs().max(1.0)
    }

    fn assert_fits_agree(key: &str, a: &GlmFit, b: &GlmFit, tol: f64) {
        assert_eq!(
            (a.rank, &a.aliased, a.iterations, a.converged),
            (b.rank, &b.aliased, b.iterations, b.converged),
            "{key}"
        );
        let pairs = a
            .coefficients
            .iter()
            .zip(&b.coefficients)
            .chain(a.std_errors.iter().zip(&b.std_errors))
            .chain(a.vcov.iter().zip(&b.vcov))
            .chain(a.fitted.iter().zip(&b.fitted))
            .chain(a.working_weights.iter().zip(&b.working_weights));
        for (x, y) in pairs {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert!(rel_gap(*x, *y) <= tol, "{key}: {x} vs {y}");
        }
        for (x, y) in [
            (a.deviance, b.deviance),
            (a.null_deviance, b.null_deviance),
            (a.dispersion, b.dispersion),
            (a.aic, b.aic),
            (a.log_likelihood, b.log_likelihood),
        ] {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert!(rel_gap(x, y) <= tol, "{key}: {x} vs {y}");
        }
    }

    /// A deterministic noisy design (intercept and three regressors) with binary, count
    /// and positive responses, an offset and weights (every 13th weight 0).
    struct SimData {
        x: Array2<f64>,
        yb: Array1<f64>,
        yc: Array1<f64>,
        yg: Array1<f64>,
        offset: Array1<f64>,
        weights: Array1<f64>,
    }

    fn sim_data(n: usize, seed: u64) -> SimData {
        let mut state = seed;
        let mut u = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            for j in 1..4 {
                x[[i, j]] = 2.0 * u() - 1.0 + 0.3 * j as f64;
            }
        }
        let eta: Vec<f64> = (0..n)
            .map(|i| 0.2 + 0.8 * x[[i, 1]] - 0.5 * x[[i, 2]] + 0.3 * x[[i, 3]])
            .collect();
        let yb = eta
            .iter()
            .map(|&e| f64::from(u() < 1.0 / (1.0 + (-e).exp())))
            .collect();
        let yc = eta.iter().map(|&e| (e.exp() * 2.0 * u()).floor()).collect();
        let yg = eta.iter().map(|&e| e.exp() + 0.3 * u()).collect();
        let offset = (0..n).map(|_| 0.1 * u()).collect();
        let weights = (0..n)
            .map(|i| if i % 13 == 0 { 0.0 } else { 0.5 + u() })
            .collect();
        SimData {
            x,
            yb,
            yc,
            yg,
            offset,
            weights,
        }
    }

    /// The IRLS step takes OLS's solve decision: on every family and link (noisy data,
    /// offsets and weights included) the last step is a normal-equations solve, and the
    /// fit equals the one whose every step is the pivoted QR to 1e-10 relative.
    #[test]
    fn test_glm_normal_equations_match_qr() {
        use GlmFamily as F;
        use GlmLink as L;
        for (n, seed) in [(60, 3), (500, 11), (3000, 29)] {
            let d = sim_data(n, seed);
            let cases: [(&str, F, L, &Array1<f64>, bool, bool); 12] = [
                ("logit", F::Binomial, L::Logit, &d.yb, false, false),
                ("probit", F::Binomial, L::Probit, &d.yb, false, false),
                ("cloglog", F::Binomial, L::Cloglog, &d.yb, false, false),
                (
                    "logit offset weights",
                    F::Binomial,
                    L::Logit,
                    &d.yb,
                    true,
                    true,
                ),
                (
                    "quasibinomial",
                    F::QuasiBinomial,
                    L::Logit,
                    &d.yb,
                    false,
                    true,
                ),
                ("poisson", F::Poisson, L::Log, &d.yc, false, false),
                ("poisson offset", F::Poisson, L::Log, &d.yc, true, false),
                ("poisson weights", F::Poisson, L::Log, &d.yc, false, true),
                ("quasipoisson", F::QuasiPoisson, L::Log, &d.yc, false, false),
                ("gaussian", F::Gaussian, L::Identity, &d.yg, false, false),
                ("gaussian log", F::Gaussian, L::Log, &d.yg, true, false),
                (
                    "gaussian weights",
                    F::Gaussian,
                    L::Identity,
                    &d.yg,
                    false,
                    true,
                ),
            ];
            for (key, family, link, y, off, wts) in cases {
                let mut spec = GlmSpec::new(family, link);
                if off {
                    spec = spec.offset(d.offset.view());
                }
                if wts {
                    spec = spec.weights(d.weights.view());
                }
                let key = format!("{key}, n {n}");
                let (normal, used_normal) =
                    fit_glm_solved(d.x.view(), y.view(), &spec, false).unwrap();
                let (qr, used_normal_qr) =
                    fit_glm_solved(d.x.view(), y.view(), &spec, true).unwrap();
                assert!(used_normal, "{key}: the normal equations were not used");
                assert!(!used_normal_qr, "{key}");
                assert_fits_agree(&key, &normal, &qr, 1e-10);
            }
        }
    }

    /// A full-rank but nearly collinear design (reciprocal condition of the equilibrated
    /// X'WX far below 1e-8) still takes the pivoted QR, bit for bit the all-QR fit; an
    /// exactly duplicated column is aliased as before.
    #[test]
    fn test_glm_near_collinear_design_takes_qr() {
        let d = sim_data(400, 7);
        let mut x = d.x.slice(ndarray::s![.., ..3]).to_owned();
        let n = x.nrows();
        let mut near = x.column(1).to_owned();
        for (i, v) in near.iter_mut().enumerate() {
            // x1 plus a deterministic perturbation of relative size 1e-7
            *v += 1e-7 * ((i * 7919 % 101) as f64 / 101.0 - 0.5);
        }
        let mut xn = Array2::zeros((n, 4));
        xn.slice_mut(ndarray::s![.., ..3]).assign(&x);
        xn.column_mut(3).assign(&near);
        for spec in [GlmSpec::logit(), GlmSpec::poisson()] {
            let y = if spec.family == GlmFamily::Poisson {
                &d.yc
            } else {
                &d.yb
            };
            let (fit, used_normal) = fit_glm_solved(xn.view(), y.view(), &spec, false).unwrap();
            let (qr, _) = fit_glm_solved(xn.view(), y.view(), &spec, true).unwrap();
            assert!(!used_normal, "a near-collinear design must take the QR");
            assert_eq!(fit.rank, 4);
            assert_fits_agree("near-collinear", &fit, &qr, 0.0);
        }

        let x1 = x.column(1).to_owned();
        x.column_mut(2).assign(&x1);
        let spec = GlmSpec::logit();
        let (dup, used_normal) = fit_glm_solved(x.view(), d.yb.view(), &spec, false).unwrap();
        assert!(!used_normal);
        assert_eq!((dup.rank, dup.aliased.clone()), (2, vec![2]));
        assert!(dup.coefficients[2].is_nan() && dup.std_errors[2].is_nan());
    }

    /// A noisy logit design whose equilibrated `X'WX` is off [`LEAN_MAX_CONDITION`] from the first
    /// step. `natural`: two covariates on natural scales (mean 40, sd 10, uncentred). Otherwise
    /// a mediator that trimming to the overlap of its propensity leaves nearly a function of the
    /// covariate (the rows IPW mediation's bootstrap resamples).
    fn off_lean_design(n: usize, seed: u64, natural: bool) -> (Array2<f64>, Array1<f64>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut x = Array2::<f64>::ones((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let (a, b): (f64, f64) = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
            let eta = if natural {
                x[[i, 1]] = 40.0 + 17.3 * a;
                x[[i, 2]] = 40.0 + 17.3 * b;
                -5.0 + 0.09 * x[[i, 1]] + 0.05 * x[[i, 2]]
            } else {
                x[[i, 1]] = a;
                x[[i, 2]] = 0.25 + 0.3 * a + 0.04 * b;
                -8.0 - 9.0 * a + 32.0 * x[[i, 2]]
            };
            y[i] = f64::from(u8::from(rng.gen_range(0.0..1.0_f64) < plogis(eta)));
        }
        (x, y)
    }

    /// Largest `|a - b| / |b|` over the numbers the fit reports (`inf` when only one is zero).
    fn largest_relative_gap(a: &GlmFit, b: &GlmFit) -> f64 {
        let mut worst = 0.0_f64;
        let pairs = a
            .coefficients
            .iter()
            .zip(&b.coefficients)
            .chain(a.std_errors.iter().zip(&b.std_errors))
            .chain(a.vcov.iter().zip(&b.vcov))
            .chain(a.fitted.iter().zip(&b.fitted))
            .chain(a.linear_predictor.iter().zip(&b.linear_predictor))
            .chain(a.working_weights.iter().zip(&b.working_weights))
            .chain([
                (&a.deviance, &b.deviance),
                (&a.null_deviance, &b.null_deviance),
            ]);
        for (u, v) in pairs {
            if u != v {
                let gap = (u - v).abs() / v.abs();
                worst = worst.max(if gap.is_nan() { f64::INFINITY } else { gap });
            }
        }
        worst
    }

    /// The lean path (not `fit_glm` under its name: `logit_lean_path` returns `None` off it) against
    /// the shared GLM on designs past the unrefined path's bound. Largest relative gap measured
    /// over these fits: 1.3e-11, in a linear predictor near 0; coefficients and
    /// covariances are closer.
    #[test]
    fn test_logit_lean_refined_path_matches_fit_glm_on_natural_scales() {
        for (n, seed, natural) in [
            (300, 3, true),
            (2500, 5, true),
            (20000, 7, true),
            (400, 11, false),
            (3000, 13, false),
        ] {
            let (x, y) = off_lean_design(n, seed, natural);
            let ctl = GlmControl {
                epsilon: 1e-8,
                max_iter: 100,
            };
            // past the unrefined path, so the refinement is what is compared
            let unrefined = lean_logit(x.view(), y.view(), None, LeanStart::Mustart, ctl).unwrap();
            assert!(
                unrefined.is_none(),
                "n {n}: the design stayed well conditioned"
            );
            let glm = fit_glm(x.view(), y.view(), &GlmSpec::logit()).unwrap();
            let lean = logit_lean_path(x.view(), y.view(), &GlmSpec::logit())
                .unwrap()
                .unwrap_or_else(|| panic!("n {n}: left the lean path"));
            assert!(glm.warnings.is_empty(), "{:?}", glm.warnings);
            assert!(lean.warnings.is_empty() && lean.converged && lean.aliased.is_empty());
            assert_eq!(lean.iterations, glm.iterations, "n {n}");
            assert_eq!((lean.rank, lean.df_residual), (glm.rank, glm.df_residual));
            let gap = largest_relative_gap(&lean, &glm);
            assert!(gap <= 1e-10, "n {n}: relative gap {gap:e}");
            // from the estimate's neighbourhood (glm's `start`): the first step may be the last
            let start = glm.coefficients.mapv(|b| b * (1.0 + 1e-7));
            let mut spec = GlmSpec::logit();
            spec.start = Some(start.view());
            let glm_s = fit_glm(x.view(), y.view(), &spec).unwrap();
            let lean_s = logit_lean_path(x.view(), y.view(), &spec)
                .unwrap()
                .unwrap_or_else(|| panic!("n {n}: warm start left the lean path"));
            assert_eq!(lean_s.iterations, glm_s.iterations, "n {n}");
            let gap = largest_relative_gap(&lean_s, &glm_s);
            assert!(gap <= 1e-10, "n {n}: warm start, relative gap {gap:e}");
        }
    }

    /// A heavy-tailed logit design: `x1 = shift + exp(hk a)`, `x2 = shift + exp(1.5 b)`, `d` drawn
    /// from `plogis(alpha + sig (a + b) / 1.414)`, and a noisy outcome for IPW. The covariates reach
    /// far beyond the index that generated `d`, so a few rows with `d = 0` get a large `eta`.
    fn heavy_tailed_design(
        n: usize,
        seed: u64,
        (shift, hk, sig, alpha): (f64, f64, f64, f64),
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        use rand::{Rng, SeedableRng};
        fn normal(r: &mut rand_chacha::ChaCha8Rng) -> f64 {
            let (u, v): (f64, f64) = (r.gen_range(1e-12..1.0), r.gen_range(0.0..1.0));
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
        let mut r = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut x = Array2::<f64>::ones((n, 3));
        let (mut d, mut y) = (Array1::<f64>::zeros(n), Array1::<f64>::zeros(n));
        for i in 0..n {
            let (a, b) = (normal(&mut r), normal(&mut r));
            x[[i, 1]] = shift + (hk * a).exp();
            x[[i, 2]] = shift + (1.5 * b).exp();
            let eta = alpha + sig * (a + b) / 1.414;
            d[i] = f64::from(u8::from(r.gen_range(0.0..1.0_f64) < plogis(eta)));
            y[i] = a + 0.5 * b + ((i + 1) as f64 * 0.73).sin();
        }
        (x, d, y)
    }

    /// Rows with `d = 0` at `eta` around 20-30 make `fit_glm` run out its 100 steps and warn on the
    /// default spec, as R's `glm` does at this epsilon, where the lean IRLS converges in 6. The lean
    /// path keeps what `fit_glm` reports. Without the bound the fit differs in iterations, status
    /// and warning, and by 1.1e-4 in a coefficient and 1.3e-5 in the IPW effect.
    #[test]
    fn test_fit_logit_lean_keeps_fit_glm_past_the_misfit_bound() {
        let spec = GlmSpec::logit();
        let (x, d, y) = heavy_tailed_design(3000, 183 * 7 + 3000, (100.0, 2.5, 2.0, -3.0));
        let glm = fit_glm(x.view(), d.view(), &spec).unwrap();
        assert_eq!((glm.iterations, glm.converged), (100, false));
        assert_eq!(glm.warnings, ["glm.fit: algorithm did not converge"]);
        assert!(
            logit_lean_path(x.view(), d.view(), &spec)
                .unwrap()
                .is_none()
        );
        let lean = fit_logit_lean(x.view(), d.view(), &spec).unwrap();
        assert_eq!(
            (lean.iterations, lean.converged, &lean.warnings),
            (glm.iterations, glm.converged, &glm.warnings)
        );
        assert_eq!(largest_relative_gap(&lean, &glm), 0.0);

        // The method that reads it: IPW keeps the warning and fit_glm's numbers.
        use crate::data::Dataset;
        use crate::econometrics::treatment::{Estimand, IpwConfig, run_ipw_treatment};
        use polars::prelude::*;
        let ds = Dataset::new(
            df! {
                "y" => y.to_vec(),
                "d" => d.to_vec(),
                "x1" => x.column(1).to_vec(),
                "x2" => x.column(2).to_vec(),
            }
            .unwrap(),
        );
        let config = IpwConfig {
            estimand: Estimand::ATE,
            trim: 0.01,
            ..Default::default()
        };
        let ipw = run_ipw_treatment(&ds, "y", "d", &["x1", "x2"], config).unwrap();
        assert!(
            ipw.warnings
                .iter()
                .any(|w| w == "propensity model: glm.fit: algorithm did not converge"),
            "{:?}",
            ipw.warnings
        );
        assert!(
            (ipw.effect - 1.294919769371).abs() <= 1e-10,
            "{}",
            ipw.effect
        );
        assert!(
            (ipw.std_error - 0.130460621285).abs() <= 1e-10,
            "{}",
            ipw.std_error
        );

        // The same kind of data at kappa below 100, where the unrefined path takes 7 steps to
        // fit_glm's 9 without the bound.
        let (x, d, _) = heavy_tailed_design(600, 2600, (0.0, 2.0, 4.0, -3.0));
        let glm = fit_glm(x.view(), d.view(), &spec).unwrap();
        assert!(
            logit_lean_path(x.view(), d.view(), &spec)
                .unwrap()
                .is_none()
        );
        let lean = fit_logit_lean(x.view(), d.view(), &spec).unwrap();
        assert_eq!(lean.iterations, glm.iterations);
        assert_eq!(largest_relative_gap(&lean, &glm), 0.0);
    }

    /// The bound takes only misfit rows: a row beyond it with the outcome its probability predicts
    /// stays on the path. This is the largest coefficient gap of the 1,535-design search at the
    /// bound, 8.2e-11 (SE 2.7e-11).
    #[test]
    fn test_logit_lean_stays_on_the_path_when_far_rows_fit() {
        let spec = GlmSpec::logit();
        let (x, d, _) = heavy_tailed_design(3000, 56 * 7 + 3000, (100.0, 2.5, 2.0, -3.0));
        let glm = fit_glm(x.view(), d.view(), &spec).unwrap();
        let lean = logit_lean_path(x.view(), d.view(), &spec)
            .unwrap()
            .expect("on the lean path");
        assert!(
            lean.linear_predictor
                .iter()
                .any(|e| e.abs() > LEAN_MISFIT_ETA)
        );
        assert_eq!(
            (lean.iterations, lean.converged, &lean.warnings),
            (glm.iterations, glm.converged, &glm.warnings)
        );
        for (a, b) in lean
            .coefficients
            .iter()
            .zip(&glm.coefficients)
            .chain(lean.std_errors.iter().zip(&glm.std_errors))
        {
            assert!((a - b).abs() <= 1e-10 * b.abs(), "{a} vs {b}");
        }
    }

    /// Further out than the refined path's bound the shared GLM decides and reports.
    #[test]
    fn test_logit_lean_leaves_the_path_for_fit_glm() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(19);
        let (x, y) = off_lean_design(1500, 17, true);
        let same = |x: &Array2<f64>, spec: &GlmSpec<'_>| {
            let lean = fit_logit_lean(x.view(), y.view(), spec).unwrap();
            let glm = fit_glm(x.view(), y.view(), spec).unwrap();
            let bits = |v: &[f64]| v.iter().map(|a| a.to_bits()).collect::<Vec<_>>();
            let of = |f: &GlmFit| {
                let vcov: Vec<f64> = f.vcov.iter().copied().collect();
                (bits(&f.coefficients.to_vec()), bits(&vcov))
            };
            assert_eq!(of(&lean), of(&glm));
            assert_eq!(lean.warnings, glm.warnings);
            assert_eq!(lean.aliased, glm.aliased);
            assert_eq!(lean.iterations, glm.iterations);
            glm
        };
        // near collinear, past the refined bound: the normal equations with a refinement
        let mut near = x.clone();
        for i in 0..1500 {
            near[[i, 2]] = near[[i, 1]] + 1e-3 * rng.gen_range(-1.0..1.0);
        }
        assert!(
            logit_lean_path(near.view(), y.view(), &GlmSpec::logit())
                .unwrap()
                .is_none()
        );
        assert_eq!(same(&near, &GlmSpec::logit()).rank, 3);
        // an exact alias reaches the QR and its report
        let mut alias = x.clone();
        for i in 0..1500 {
            alias[[i, 2]] = 2.0 * alias[[i, 1]];
        }
        assert!(
            logit_lean_path(alias.view(), y.view(), &GlmSpec::logit())
                .unwrap()
                .is_none()
        );
        let g = same(&alias, &GlmSpec::logit());
        assert_eq!((g.rank, g.aliased.clone()), (2, vec![2]));
    }

    /// A loose rule stops at a step that took no refinement while it ran. The step takes it
    /// afterwards and the fit stays on the path, as close to the shared GLM's as a fit refined at
    /// every step.
    #[test]
    fn test_logit_lean_refines_a_last_step_after_the_rule_stops() {
        let (x, y) = off_lean_design(1500, 17, true);
        let mut loose = GlmSpec::logit();
        loose.epsilon = 1e-2;
        let glm = fit_glm(x.view(), y.view(), &loose).unwrap();
        let lean = logit_lean_path(x.view(), y.view(), &loose)
            .unwrap()
            .expect("on the lean path");
        assert_eq!(
            (lean.iterations, lean.converged, &lean.warnings),
            (glm.iterations, glm.converged, &glm.warnings)
        );
        for (a, b) in lean
            .coefficients
            .iter()
            .zip(&glm.coefficients)
            .chain(lean.std_errors.iter().zip(&glm.std_errors))
        {
            assert!((a - b).abs() <= 1e-10 * b.abs(), "{a} vs {b}");
        }
    }

    /// The fixed-width cross-products are the row pass, bit for bit.
    #[test]
    fn test_lean_cross_fixed_is_the_row_pass() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(23);
        let n = 257;
        for k in 1..=6 {
            let xrows: Vec<f64> = (0..n * k).map(|_| rng.gen_range(-50.0..50.0)).collect();
            let mu: Vec<f64> = (0..n).map(|_| rng.gen_range(0.01..0.99)).collect();
            let eta: Vec<f64> = mu.iter().map(|m: &f64| (m / (1.0 - m)).ln()).collect();
            let y: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 3 == 0))).collect();
            let offset: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            for off in [None, Some(&offset[..])] {
                let cols = WeightedColumns {
                    xrows: &xrows,
                    mu: &mu,
                    eta: &eta,
                    y: &y,
                    off,
                };
                let (mut g, mut rhs) = (vec![0.0; k * k], vec![0.0; k]);
                cols.cross(k, &mut g, &mut rhs);
                let (mut gf, mut rf) = (vec![0.0; k * k], vec![0.0; k]);
                match k {
                    1 => cols.cross_fixed::<1>(&mut gf, &mut rf),
                    2 => cols.cross_fixed::<2>(&mut gf, &mut rf),
                    3 => cols.cross_fixed::<3>(&mut gf, &mut rf),
                    4 => cols.cross_fixed::<4>(&mut gf, &mut rf),
                    5 => cols.cross_fixed::<5>(&mut gf, &mut rf),
                    _ => cols.cross_fixed::<6>(&mut gf, &mut rf),
                }
                let bits = |v: &[f64]| v.iter().map(|a| a.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&g), bits(&gf), "k {k}");
                assert_eq!(bits(&rhs), bits(&rf), "k {k}");
            }
        }
    }

    /// The lean logit refit (`fit_logit_lean`) is the shared GLM's fit: on noisy designs, from
    /// glm.fit's start and from a warm start, at several sizes, every reported number agrees
    /// with `fit_glm` to 1e-12 relative (the coefficients to 1e-12, the covariance and SEs, the
    /// deviance, AIC and log-likelihood to 1e-12), with the same iterations, convergence and
    /// (no) warnings. A near-separated design that stays below the logit clamp agrees too;
    /// separated, rank-deficient, weighted, offset and non-logit fits are fit_glm's own.
    #[test]
    fn test_fit_logit_lean_matches_fit_glm() {
        for (n, seed) in [(40, 5), (300, 17), (2500, 41)] {
            let d = sim_data(n, seed);
            let glm = fit_glm(d.x.view(), d.yb.view(), &GlmSpec::logit()).unwrap();
            let lean = fit_logit_lean(d.x.view(), d.yb.view(), &GlmSpec::logit()).unwrap();
            assert!(glm.warnings.is_empty() && lean.warnings.is_empty());
            assert_fits_agree(&format!("logit n {n}"), &lean, &glm, 1e-12);
            assert_eq!(lean.df_residual, glm.df_residual);
            assert_eq!(lean.linear_predictor.len(), n);
            for (a, b) in lean.linear_predictor.iter().zip(&glm.linear_predictor) {
                assert!(rel_gap(*a, *b) <= 1e-12, "eta {a} vs {b}");
            }
            // warm start from slightly perturbed coefficients (a bootstrap refit's start)
            let start = glm.coefficients.mapv(|b| b * 0.9 + 0.01);
            let mut spec = GlmSpec::logit();
            spec.start = Some(start.view());
            let glm_s = fit_glm(d.x.view(), d.yb.view(), &spec).unwrap();
            let lean_s = fit_logit_lean(d.x.view(), d.yb.view(), &spec).unwrap();
            assert_fits_agree(&format!("logit start n {n}"), &lean_s, &glm_s, 1e-12);
        }
        // near separation: a strong predictor, fitted probabilities close to 0 and 1 but the
        // linear predictor below the clamp
        let d = sim_data(400, 23);
        let mut x = d.x.clone();
        let y = Array1::from_iter((0..400).map(|i| {
            let e = 6.0 * x[[i, 1]] + 0.3 * x[[i, 2]] - 0.8;
            f64::from(d.offset[i] * 10.0 < 1.0 / (1.0 + (-e).exp()))
        }));
        let glm = fit_glm(x.view(), y.view(), &GlmSpec::logit()).unwrap();
        assert!(
            glm.fitted
                .iter()
                .any(|&m| !(1e-3..=1.0 - 1e-3).contains(&m))
        );
        let lean = fit_logit_lean(x.view(), y.view(), &GlmSpec::logit()).unwrap();
        assert_fits_agree("near separation", &lean, &glm, 1e-12);
        // complete separation: fit_glm's fit, warnings and flags, bit for bit
        let ysep = x.column(1).mapv(|v| f64::from(v > 0.3));
        let a = fit_glm(x.view(), ysep.view(), &GlmSpec::logit()).unwrap();
        let b = fit_logit_lean(x.view(), ysep.view(), &GlmSpec::logit()).unwrap();
        assert!(a.separation && b.separation);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        // aliased column, weights, offset, probit, non-binary response, refinement:
        // fit_glm's own fit
        let x1 = x.column(1).to_owned();
        x.column_mut(3).assign(&x1);
        let w = d.weights.clone();
        let o = d.offset.clone();
        let yq = d.yb.mapv(|v| 0.25 + 0.5 * v);
        let mut refine = GlmSpec::logit();
        refine.coef_tol = Some(1e-10);
        let cases: Vec<(&str, ArrayView2<f64>, ArrayView1<f64>, GlmSpec<'_>)> = vec![
            ("aliased", x.view(), d.yb.view(), GlmSpec::logit()),
            (
                "weights",
                d.x.view(),
                d.yb.view(),
                GlmSpec::logit().weights(w.view()),
            ),
            (
                "offset",
                d.x.view(),
                d.yb.view(),
                GlmSpec::logit().offset(o.view()),
            ),
            ("probit", d.x.view(), d.yb.view(), GlmSpec::probit()),
            ("non-binary", d.x.view(), yq.view(), GlmSpec::logit()),
            ("refinement", d.x.view(), d.yb.view(), refine),
        ];
        for (key, xv, yv, spec) in cases {
            let a = fit_glm(xv, yv, &spec).unwrap();
            let b = fit_logit_lean(xv, yv, &spec).unwrap();
            assert_eq!(
                serde_json::to_string(&a).unwrap(),
                serde_json::to_string(&b).unwrap(),
                "{key}"
            );
        }
        // invalid input: fit_glm's error
        let mut bad = d.x.clone();
        bad[[3, 2]] = f64::NAN;
        let ea = fit_glm(bad.view(), d.yb.view(), &GlmSpec::logit()).unwrap_err();
        let eb = fit_logit_lean(bad.view(), d.yb.view(), &GlmSpec::logit()).unwrap_err();
        assert_eq!(ea.to_string(), eb.to_string());
    }
}
