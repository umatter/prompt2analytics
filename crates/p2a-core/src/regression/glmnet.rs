//! Regularized linear and logistic regression with the elastic net penalty (glmnet).
//!
//! The gaussian objective, in the standardised x units `z` (see below), is glmnet's:
//!
//! min (1/2n) ||y - b0 - Z b||² + λ [α ||b||₁ + (1 - α) / (2 ys) ||b||²]
//!
//! - α = 1 gives the lasso, α = 0 ridge, 0 < α < 1 the elastic net;
//! - `ys` is the population sd of y with an intercept and `sqrt(mean(y²))` without one: glmnet
//!   solves on `y / ys`, so a textbook ridge with penalty (λ'/2)||β||² is glmnet's λ = λ' ys.
//!
//! The binomial objective is `-(1/n) sum(y η - log(1 + e^η)) + λ [α ||b||₁ + (1 - α)/2 ||b||²]`
//! with y not standardised.
//!
//! # Standardisation
//!
//! x is centred by its column means if and only if an intercept is fitted, and divided by its
//! population sd when `standardize` (otherwise by 1). Coefficients are reported on the original
//! scale. A constant column gets coefficient 0 (glmnet's `ju`).
//!
//! # Algorithm
//!
//! Cyclical coordinate descent with warm starts (Friedman, Hastie and Tibshirani 2010): the
//! update of coordinate j is `b_j = S(g_j, l1) / (v_j + l2)` with S the soft-threshold. A lambda
//! has converged when a full pass over every variable moves no coefficient by more than
//! `max_j v_j Δ_j² < thresh` (passes over the active set in between), so no excluded variable
//! violates its KKT condition. `max_iter` counts passes over the whole path. Binomial fits use
//! IRLS with the intercept as an unpenalised coordinate of the inner weighted loop.
//!
//! The lambda path, its stopping rules, `cv.glmnet` and `predict.glmnet(exact = FALSE)` follow
//! glmnet 5.1; `validation/regression/glmnet.md` lists the conventions and the known
//! differences from R: `alpha` outside [0, 1], an empty user lambda and non-convergence at the
//! first lambda are errors; `nlambda = 1` reports `lambda_max`; `nfolds = 2` is accepted; no
//! covariance updates or strong rules (pass counts differ); no weights, offsets, penalty
//! factors, limits or other families.
//!
//! # References
//!
//! - Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
//!   generalized linear models via coordinate descent. *Journal of Statistical Software*,
//!   33(1), 1-22. <https://doi.org/10.18637/jss.v033.i01>
//!
//! - Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic
//!   net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
//!   <https://doi.org/10.1111/j.1467-9868.2005.00503.x>
//!
//! R equivalent: `glmnet::glmnet()`, `glmnet::cv.glmnet()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, ShapeBuilder, s};
use polars::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{NaPolicy, econ_error_from_design, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::linalg::DesignMatrix;
use crate::resource;

/// Family type for GLM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GlmnetFamily {
    /// Gaussian (linear regression)
    #[default]
    Gaussian,
    /// Binomial (logistic regression)
    Binomial,
}

impl std::str::FromStr for GlmnetFamily {
    type Err = EconError;

    /// `gaussian` (also `linear`, `normal`) or `binomial` (also `logistic`, `logit`); case,
    /// surrounding space and the separators ' ', '-', '.', '_' are ignored.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let key: String = s
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|c| !matches!(c, ' ' | '-' | '.' | '_'))
            .collect();
        match key.as_str() {
            "gaussian" | "linear" | "normal" => Ok(GlmnetFamily::Gaussian),
            "binomial" | "logistic" | "logit" => Ok(GlmnetFamily::Binomial),
            _ => Err(EconError::InvalidSpecification {
                message: format!(
                    "unknown family '{}'; expected one of: gaussian, binomial",
                    s.trim()
                ),
            }),
        }
    }
}

impl std::fmt::Display for GlmnetFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            GlmnetFamily::Gaussian => "gaussian",
            GlmnetFamily::Binomial => "binomial",
        })
    }
}

/// Configuration for glmnet fitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetConfig {
    /// Elastic net mixing parameter (0 = ridge, 1 = lasso). Finite in [0, 1], otherwise
    /// `InvalidSpecification` (R clamps a value above 1 to 1 with a warning).
    pub alpha: f64,
    /// A user lambda path (None: the generated path). Non-empty, every value finite and >= 0;
    /// sorted decreasingly as R does, fitted in full (never cut short by the stopping rules,
    /// `dfmax` does not apply).
    pub lambda: Option<Vec<f64>>,
    /// Number of lambda values in the generated path (default: 100). At least 1 when `lambda`
    /// is `None` (otherwise `InvalidSpecification`); `nlambda = 1` is the single value
    /// `lambda_max`; a path above the allocation ceiling is `ResourceLimit`.
    pub nlambda: usize,
    /// `lambda_min / lambda_max` of the generated path, finite and strictly between 0 and 1.
    /// `None` is glmnet's default: 0.01 when n < p, else 1e-4.
    pub lambda_min_ratio: Option<f64>,
    /// Standardize predictors before fitting (default: true)
    pub standardize: bool,
    /// Include intercept (default: true)
    pub intercept: bool,
    /// glmnet's `maxit`: the total number of coordinate passes over the whole path (default
    /// 100 000). When it runs out the path is cut at the last converged lambda with a warning;
    /// if not even the first lambda converged the fit is a `ConvergenceFailure`. At least 1
    /// (otherwise `InvalidSpecification`).
    pub max_iter: usize,
    /// glmnet's `thresh` (default 1e-7): coordinate descent has converged when a full pass over
    /// every variable moves no coefficient by more than `max_j v_j delta_j^2 < thresh`, in
    /// standardised units (`v_j` the weighted variance of column j, y divided by its sd for
    /// gaussian), so the criterion does not depend on the scale of x or y. The default leaves
    /// coefficients within about 1e-4 sd(y)/sd(x_j) of the minimiser, as R's do; `thresh = 1e-14`
    /// gives about 1e-7. Must be finite and > 0 (otherwise `InvalidSpecification`).
    pub thresh: f64,
    /// Model family
    pub family: GlmnetFamily,
    /// glmnet's `dfmax`: a generated path stops after the first fit (from the 5th) with more
    /// than `dfmax` nonzero coefficients; that fit is kept.
    pub dfmax: Option<usize>,
    /// glmnet's `devmax` (default 0.999), in (0, 1]: a generated path stops after the first fit
    /// (from the 5th) whose dev.ratio exceeds it.
    pub deviance_thresh: f64,
    /// glmnet's `fdev` (default 1e-5), finite and >= 0: a generated path stops after the first
    /// fit (from the 5th) whose dev.ratio gain over the previous fit is below `fdev * dev.ratio`
    /// (gaussian) or `fdev` (binomial). 0 disables the rule.
    #[serde(default = "default_fdev")]
    pub fdev: f64,
}

fn default_fdev() -> f64 {
    1e-5
}

impl Default for GlmnetConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0, // lasso by default
            lambda: None,
            nlambda: 100,
            lambda_min_ratio: None,
            standardize: true,
            intercept: true,
            max_iter: 100_000,
            thresh: 1e-7,
            family: GlmnetFamily::Gaussian,
            dfmax: None,
            deviance_thresh: 0.999,
            fdev: 1e-5,
        }
    }
}

/// Result for a single lambda value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetCoefficients {
    /// Lambda value
    pub lambda: f64,
    /// Intercept term
    pub intercept: f64,
    /// Coefficients (excluding intercept)
    pub coefficients: Vec<f64>,
    /// Number of non-zero coefficients
    pub df: usize,
    /// Percent deviance explained
    pub dev_ratio: f64,
}

/// Result of glmnet fitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetResult {
    /// Variable names (excluding intercept)
    pub variable_names: Vec<String>,
    /// Dependent variable name
    pub dependent_var: String,
    /// Alpha parameter used
    pub alpha: f64,
    /// Lambda sequence
    pub lambda: Vec<f64>,
    /// Results for each lambda
    pub fits: Vec<GlmnetCoefficients>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of features
    pub n_features: usize,
    /// Null deviance
    pub null_deviance: f64,
    /// Family used
    pub family: GlmnetFamily,
    /// The centring of x: column means when an intercept is fitted, else `None`.
    pub x_mean: Option<Vec<f64>>,
    /// The scales of x: population sd (1 for a constant column) when standardised, else `None`.
    pub x_scale: Option<Vec<f64>>,
    /// Mean of y.
    pub y_mean: f64,
    /// Whether x was standardised (`x_scale` holds the scales).
    #[serde(default = "default_true")]
    pub standardize: bool,
    /// Whether an intercept was fitted (`x_mean` holds the centring).
    #[serde(default = "default_true")]
    pub intercept: bool,
    /// Coordinate passes over the whole path (glmnet's `npasses`).
    #[serde(default)]
    pub npasses: usize,
    /// Notes on the fit (a path truncated by `max_iter`, ...).
    #[serde(default)]
    pub warnings: Vec<String>,
}

fn default_true() -> bool {
    true
}

/// Cross-validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetCvResult {
    /// Full glmnet result
    pub fit: GlmnetResult,
    /// Lambda values tested
    pub lambda: Vec<f64>,
    /// Mean cross-validated error for each lambda
    pub cvm: Vec<f64>,
    /// Standard error of cross-validated error
    pub cvsd: Vec<f64>,
    /// Upper confidence bound (cvm + cvsd)
    pub cvup: Vec<f64>,
    /// Lower confidence bound (cvm - cvsd)
    pub cvlo: Vec<f64>,
    /// Lambda that gives minimum CV error
    pub lambda_min: f64,
    /// Index of lambda_min in lambda sequence
    pub lambda_min_index: usize,
    /// Largest lambda within 1 SE of minimum
    pub lambda_1se: f64,
    /// Index of lambda_1se
    pub lambda_1se_index: usize,
    /// Number of folds used
    pub nfolds: usize,
    /// Fold label (1..=nfolds, R's `foldid`) of each observation.
    #[serde(default)]
    pub foldid: Vec<usize>,
    /// Length of each fold's own lambda path.
    #[serde(default)]
    pub fold_path_lengths: Vec<usize>,
    /// Notes of the fold fits, prefixed `fold k: `.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl GlmnetCvResult {
    /// The full-data fit at `lambda.min`.
    pub fn fit_min(&self) -> &GlmnetCoefficients {
        &self.fit.fits[self.lambda_min_index]
    }

    /// The full-data fit at `lambda.1se`.
    pub fn fit_1se(&self) -> &GlmnetCoefficients {
        &self.fit.fits[self.lambda_1se_index]
    }
}

/// Seed of [`cv_glmnet`]'s fold assignment when `seed` is `None` (so a call is reproducible).
pub const CV_GLMNET_DEFAULT_SEED: u64 = 42;

/// cv.glmnet's bound on predicted probabilities in the binomial deviance (`cv.lognet`).
const CV_PROB_MIN: f64 = 1e-5;

/// Relative slack of the soft-threshold: a gradient within summation rounding of the threshold
/// counts as at the threshold. A fit at `lambda_max` computed in other arithmetic (R's path, a
/// path passed back in as user lambdas) is then exactly zero instead of carrying a 1e-17
/// coefficient; the effect on any coefficient is below `1e-13 * gamma / v_j`.
const SOFT_RTOL: f64 = 1e-13;

/// Soft-thresholding operator: S(z, γ) = sign(z)(|z| - γ)₊
#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z.abs() <= gamma * (1.0 + SOFT_RTOL) {
        0.0
    } else if z > 0.0 {
        z - gamma
    } else {
        z + gamma
    }
}

/// The standardised design glmnet works on.
///
/// `z = (x - center) / scale`, column-major (`z[j * n + i]`), with `center_j = mean(x_j)` when
/// an intercept is fitted (else 0) and `scale_j` the population sd of `x_j` when `standardize`
/// (else 1). A constant column is not `usable`: its coefficient stays 0 (glmnet's `ju`).
/// `xv_j = mean(z_j^2)`, the unit-weight curvature of coordinate j.
struct Design {
    n: usize,
    p: usize,
    z: Vec<f64>,
    center: Vec<f64>,
    scale: Vec<f64>,
    xv: Vec<f64>,
    usable: Vec<bool>,
}

impl Design {
    fn col(&self, j: usize) -> &[f64] {
        &self.z[j * self.n..(j + 1) * self.n]
    }

    /// Indices of the usable (non-constant) columns.
    fn usable_columns(&self) -> Vec<usize> {
        (0..self.p).filter(|&j| self.usable[j]).collect()
    }
}

/// glmnet's standardisation of x (glmnetpp `standard1` with unit weights).
fn standardize_design(
    x: ArrayView2<f64>,
    standardize: bool,
    intercept: bool,
) -> EconResult<Design> {
    let (n, p) = x.dim();
    let what = "glmnet standardised design";
    let mut z = resource::try_vec(resource::checked_len(&[n, p], what)?, 0.0, what)?;
    let mut center = Vec::with_capacity(p);
    let mut scale = Vec::with_capacity(p);
    let mut xv = Vec::with_capacity(p);
    let mut usable = Vec::with_capacity(p);
    let nf = n as f64;
    for j in 0..p {
        let col = x.column(j);
        let mean = col.sum() / nf;
        let (lo, hi) = col
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let ok = hi > lo;
        let sd = (col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / nf).sqrt();
        let c = if intercept { mean } else { 0.0 };
        let s = if standardize && ok { sd } else { 1.0 };
        let zj = &mut z[j * n..(j + 1) * n];
        for (zi, &v) in zj.iter_mut().zip(col.iter()) {
            *zi = (v - c) / s;
        }
        let v = zj.iter().map(|v| v * v).sum::<f64>() / nf;
        if !(mean.is_finite() && sd.is_finite() && v.is_finite()) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "column {} of x is too large in magnitude: its mean or variance is not \
                     finite in double precision; rescale it",
                    j + 1
                ),
            });
        }
        xv.push(v);
        center.push(c);
        scale.push(s);
        usable.push(ok);
    }
    if !usable.iter().any(|&u| u) {
        return Err(EconError::InvalidSpecification {
            message: "every column of x is constant".to_string(),
        });
    }
    Ok(Design {
        n,
        p,
        z,
        center,
        scale,
        xv,
        usable,
    })
}

/// The coordinate-pass budget of one path: glmnet's `maxit` counts passes over the whole path.
struct PassBudget {
    used: usize,
    max: usize,
}

impl PassBudget {
    /// Claims one pass; `false` when the budget is spent. Checks the call's budget first.
    fn take(&mut self) -> EconResult<bool> {
        resource::checkpoint()?;
        if self.used >= self.max {
            return Ok(false);
        }
        self.used += 1;
        Ok(true)
    }
}

/// How a coordinate-descent solve ended.
enum Solve {
    /// A full pass over every usable column moved no coordinate by more than `thresh`.
    Converged,
    /// The path's pass budget ran out; the largest last weighted change.
    Budget(f64),
}

/// A weighted least-squares problem `(1/2n) sum_i w_i (r_i - z_i b)^2 + l1 |b|_1 + l2/2 |b|^2`
/// solved by coordinate descent on the working residual `r` (glmnet's `elnet` / `lognet` inner
/// loop). `w = None` means unit weights.
struct Wls<'a> {
    design: &'a Design,
    w: Option<&'a [f64]>,
    /// Weighted curvature `v_j = sum_i w_i z_ij^2 / n`.
    v: Vec<f64>,
    /// `sum_i w_i`.
    sum_w: f64,
    all: Vec<usize>,
}

impl<'a> Wls<'a> {
    /// [`Wls::new`] for the weights `w`, reusing the column list and the curvature buffer of
    /// an earlier solve (from [`Wls::into_buffers`]) instead of allocating them again.
    fn reweighted(design: &'a Design, w: &'a [f64], all: Vec<usize>, mut v: Vec<f64>) -> Self {
        let nf = design.n as f64;
        v.clear();
        v.extend((0..design.p).map(|j| {
            design
                .col(j)
                .iter()
                .zip(w)
                .map(|(z, w)| w * z * z)
                .sum::<f64>()
                / nf
        }));
        Wls {
            design,
            w: Some(w),
            v,
            sum_w: w.iter().sum(),
            all,
        }
    }

    /// The column list and the curvature buffer, for [`Wls::reweighted`].
    fn into_buffers(self) -> (Vec<usize>, Vec<f64>) {
        (self.all, self.v)
    }

    fn new(design: &'a Design, w: Option<&'a [f64]>) -> Self {
        let nf = design.n as f64;
        let (v, sum_w) = match w {
            None => (design.xv.clone(), nf),
            Some(w) => (
                (0..design.p)
                    .map(|j| {
                        design
                            .col(j)
                            .iter()
                            .zip(w)
                            .map(|(z, w)| w * z * z)
                            .sum::<f64>()
                            / nf
                    })
                    .collect(),
                w.iter().sum(),
            ),
        };
        Wls {
            design,
            w,
            v,
            sum_w,
            all: design.usable_columns(),
        }
    }

    /// One pass over `set` (and the intercept when `b0` is given); returns
    /// `max(v_j delta_j^2)` over the moves made.
    fn pass(
        &self,
        set: &[usize],
        b: &mut [f64],
        b0: Option<&mut f64>,
        r: &mut [f64],
        l1: f64,
        l2: f64,
    ) -> f64 {
        let nf = self.design.n as f64;
        let mut dlx = 0.0f64;
        for &j in set {
            let zj = self.design.col(j);
            let old = b[j];
            let dot: f64 = match self.w {
                None => zj.iter().zip(r.iter()).map(|(z, r)| z * r).sum(),
                Some(w) => zj
                    .iter()
                    .zip(r.iter())
                    .zip(w)
                    .map(|((z, r), w)| w * z * r)
                    .sum(),
            };
            let g = dot / nf + self.v[j] * old;
            let new = soft_threshold(g, l1) / (self.v[j] + l2);
            if new != old {
                let d = new - old;
                for (ri, zi) in r.iter_mut().zip(zj) {
                    *ri -= d * zi;
                }
                b[j] = new;
                dlx = dlx.max(self.v[j] * d * d);
            }
        }
        if let Some(b0) = b0 {
            let d = match self.w {
                None => r.iter().sum::<f64>() / nf,
                Some(w) => r.iter().zip(w).map(|(r, w)| w * r).sum::<f64>() / self.sum_w,
            };
            if d != 0.0 {
                *b0 += d;
                for ri in r.iter_mut() {
                    *ri -= d;
                }
                dlx = dlx.max(self.sum_w / nf * d * d);
            }
        }
        dlx
    }

    /// Coordinate descent to convergence: converged only on a full pass over every usable column
    /// (so no excluded variable violates its KKT condition), with passes over the active set in
    /// between (glmnet's scheme).
    fn solve(
        &self,
        b: &mut [f64],
        mut b0: Option<&mut f64>,
        r: &mut [f64],
        l1: f64,
        l2: f64,
        thresh: f64,
        budget: &mut PassBudget,
    ) -> EconResult<Solve> {
        let mut active = Vec::with_capacity(self.all.len());
        let mut last = f64::INFINITY;
        loop {
            if !budget.take()? {
                return Ok(Solve::Budget(last));
            }
            last = self.pass(&self.all, b, b0.as_deref_mut(), r, l1, l2);
            if last < thresh {
                return Ok(Solve::Converged);
            }
            active.clear();
            active.extend(self.all.iter().copied().filter(|&j| b[j] != 0.0));
            if active.is_empty() && b0.is_none() {
                continue;
            }
            loop {
                if !budget.take()? {
                    return Ok(Solve::Budget(last));
                }
                last = self.pass(&active, b, b0.as_deref_mut(), r, l1, l2);
                if last < thresh {
                    break;
                }
            }
        }
    }
}

/// The truncation note when the pass budget runs out at the `k`-th (1-based) of `len` lambdas.
fn truncation_warning(k: usize, len: usize, lambda: f64, max_iter: usize) -> String {
    format!(
        "Convergence not reached at lambda {k} of {len} (lambda = {lambda:.6e}) after max_iter = \
         {max_iter} coordinate passes over the path; the fits for the {} larger lambdas are returned",
        k - 1
    )
}

/// The truncation note when the IRLS iterations of the `k`-th (1-based) of `len` lambdas hit
/// their cap ([`MXIT_IRLS`]), which is not the coordinate-pass budget `max_iter`.
fn irls_truncation_warning(k: usize, len: usize, lambda: f64) -> String {
    format!(
        "IRLS did not converge in {MXIT_IRLS} iterations at lambda[{k}] of {len} (lambda = \
         {lambda:.6e}); the fits for the {} larger lambdas are returned",
        k - 1
    )
}

/// `ConvergenceFailure` when the IRLS iterations of the first lambda hit their cap.
fn irls_failure(k: usize, last_change: f64) -> EconError {
    EconError::ConvergenceFailure {
        iterations: MXIT_IRLS,
        last_change,
        suggestion: format!(
            "IRLS did not converge in {MXIT_IRLS} iterations at lambda[{k}]; the binomial \
             fit may be (quasi-)separated: try a larger lambda or alpha < 1"
        ),
    }
}

/// `ConvergenceFailure` when not even the first lambda converged.
fn first_lambda_failure(max_iter: usize, last_change: f64) -> EconError {
    EconError::ConvergenceFailure {
        iterations: max_iter,
        last_change,
        suggestion: "increase max_iter or thresh".to_string(),
    }
}

/// Check the controls of a glmnet configuration (rule R3: every error is an
/// `InvalidSpecification` naming the parameter):
///
/// - `thresh` finite and > 0; `max_iter` >= 1; `nlambda` >= 1 when no user lambda is given;
/// - `alpha` finite in [0, 1];
/// - a user `lambda`: non-empty, every value finite and >= 0;
/// - `lambda_min_ratio`, when given: finite, strictly between 0 and 1;
/// - `fdev` finite and >= 0; `deviance_thresh` finite in (0, 1].
///
/// The data checks (columns, rows, finite values) are in [`glmnet`].
fn validate_glmnet_config(config: &GlmnetConfig) -> EconResult<()> {
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    if !(config.thresh.is_finite() && config.thresh > 0.0) {
        return spec(format!(
            "thresh must be a positive finite number (got {})",
            config.thresh
        ));
    }
    if config.max_iter == 0 {
        return spec("max_iter must be at least 1".to_string());
    }
    if config.lambda.is_none() && config.nlambda == 0 {
        return spec("nlambda must be at least 1 when no lambda sequence is given".to_string());
    }
    if !(config.alpha.is_finite() && (0.0..=1.0).contains(&config.alpha)) {
        return spec(format!(
            "alpha must be a finite number in [0, 1] (got {})",
            config.alpha
        ));
    }
    if let Some(lambda) = &config.lambda {
        if lambda.is_empty() {
            return spec("lambda must contain at least one value".to_string());
        }
        if let Some(bad) = lambda.iter().find(|l| !(l.is_finite() && **l >= 0.0)) {
            return spec(format!(
                "lambda: lambdas should be non-negative and finite (got {bad})"
            ));
        }
    }
    if let Some(r) = config.lambda_min_ratio {
        if !(r.is_finite() && r > 0.0 && r < 1.0) {
            return spec(format!(
                "lambda_min_ratio must be finite and strictly between 0 and 1 (got {r})"
            ));
        }
    }
    if !(config.fdev.is_finite() && config.fdev >= 0.0) {
        return spec(format!(
            "fdev must be finite and non-negative (got {})",
            config.fdev
        ));
    }
    let d = config.deviance_thresh;
    if !(d.is_finite() && d > 0.0 && d <= 1.0) {
        return spec(format!(
            "deviance_thresh must be finite, above 0 and at most 1 (got {d})"
        ));
    }
    Ok(())
}

/// Check the data glmnet is given: at least one column, as many rows as y, at least two rows,
/// every value finite.
fn validate_glmnet_data(x: ArrayView2<f64>, y: ArrayView1<f64>) -> EconResult<()> {
    let (n, p) = x.dim();
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    if p == 0 {
        return spec("x must have at least one column".to_string());
    }
    if n != y.len() {
        return spec(format!("x has {n} rows but y has {} elements", y.len()));
    }
    if n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "glmnet".to_string(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return spec("x contains a missing or non-finite value".to_string());
    }
    if y.iter().any(|v| !v.is_finite()) {
        return spec("y contains a missing or non-finite value".to_string());
    }
    Ok(())
}

/// The smallest `alpha` the generated path's `lambda_max` divides by (glmnet: `max(alpha, 1e-3)`),
/// so a ridge path starts at `max|x'y| / n / 1e-3`.
const ALPHA_FLOOR: f64 = 1e-3;

/// glmnet's `mnlam`: the stopping rules of a generated path apply from this fit on.
const MNLAM: usize = 5;

/// The lambda path: the user's, sorted decreasingly, or the generated one from `lambda_max`
/// down to `ratio * lambda_max` on a log scale. Returns the path, whether it was generated, and
/// a note for `nlambda = 1`.
fn lambda_path(
    config: &GlmnetConfig,
    lambda_max: f64,
    n: usize,
    p: usize,
) -> EconResult<(Vec<f64>, bool, Option<String>)> {
    if let Some(user) = &config.lambda {
        let mut seq = user.clone();
        seq.sort_by(|a, b| b.total_cmp(a));
        return Ok((seq, false, None));
    }
    if config.nlambda == 1 {
        let tail = if config.alpha >= ALPHA_FLOOR {
            ", at which every coefficient is zero"
        } else {
            ""
        };
        let note = format!(
            "nlambda = 1: the path is the single value lambda_max = {lambda_max:.6e}{tail}"
        );
        return Ok((vec![lambda_max], true, Some(note)));
    }
    let ratio = config
        .lambda_min_ratio
        .unwrap_or(if n < p { 0.01 } else { 1e-4 });
    let mut seq = resource::try_with_capacity::<f64>(config.nlambda, "glmnet lambda path")?;
    let last = (config.nlambda - 1) as f64;
    seq.extend((0..config.nlambda).map(|k| lambda_max * ratio.powf(k as f64 / last)));
    Ok((seq, true, None))
}

/// glmnet's stopping rules for a generated gaussian path, after the fit just pushed.
fn gaussian_path_stops(fits: &[GlmnetCoefficients], config: &GlmnetConfig) -> bool {
    let m = fits.len();
    if m < MNLAM {
        return false;
    }
    let (cur, prev) = (&fits[m - 1], &fits[m - 2]);
    config.dfmax.is_some_and(|d| cur.df > d)
        || cur.dev_ratio - prev.dev_ratio < config.fdev * cur.dev_ratio
        || cur.dev_ratio > config.deviance_thresh
}

/// The fits of one path, the coordinate passes spent and the notes.
struct PathFit {
    fits: Vec<GlmnetCoefficients>,
    npasses: usize,
    warnings: Vec<String>,
}

/// Gaussian path: glmnet's `elnet` on `yt = (y - ym) / ys`, with the penalty in those units
/// (`l1 = lambda alpha / ys`, `l2 = lambda (1 - alpha) / ys`), warm starts along the path.
fn gaussian_path(
    design: &Design,
    yc: &[f64],
    ym: f64,
    ys: f64,
    lambdas: &[f64],
    generated: bool,
    config: &GlmnetConfig,
) -> EconResult<PathFit> {
    let p = design.p;
    let wls = Wls::new(design, None);
    let mut r: Vec<f64> = yc.iter().map(|v| v / ys).collect();
    let tss: f64 = r.iter().map(|v| v * v).sum();
    let mut b = vec![0.0; p];
    let mut budget = PassBudget {
        used: 0,
        max: config.max_iter,
    };
    let what = "glmnet coefficient path";
    let mut fits = resource::try_with_capacity(lambdas.len(), what)?;
    let mut warnings = Vec::new();
    for (k, &lambda) in lambdas.iter().enumerate() {
        let l1 = lambda * config.alpha / ys;
        let l2 = lambda * (1.0 - config.alpha) / ys;
        if let Solve::Budget(dlx) =
            wls.solve(&mut b, None, &mut r, l1, l2, config.thresh, &mut budget)?
        {
            if fits.is_empty() {
                return Err(first_lambda_failure(config.max_iter, dlx));
            }
            warnings.push(truncation_warning(
                k + 1,
                lambdas.len(),
                lambda,
                config.max_iter,
            ));
            break;
        }
        let coefficients: Vec<f64> = (0..p).map(|j| ys * b[j] / design.scale[j]).collect();
        let intercept = if config.intercept {
            ym - (0..p)
                .map(|j| design.center[j] * coefficients[j])
                .sum::<f64>()
        } else {
            0.0
        };
        let rss: f64 = r.iter().map(|v| v * v).sum();
        let df = coefficients.iter().filter(|&&c| c != 0.0).count();
        fits.push(GlmnetCoefficients {
            lambda,
            intercept,
            coefficients,
            df,
            dev_ratio: 1.0 - rss / tss,
        });
        if generated && gaussian_path_stops(&fits, config) {
            break;
        }
    }
    Ok(PathFit {
        fits,
        npasses: budget.used,
        warnings,
    })
}

/// glmnet's stopping rules for a generated binomial path (lognet: the dev.ratio gain is compared
/// with `fdev` itself, not `fdev * dev.ratio`), after the fit just pushed.
fn binomial_path_stops(fits: &[GlmnetCoefficients], config: &GlmnetConfig) -> bool {
    let m = fits.len();
    if m < MNLAM {
        return false;
    }
    let (cur, prev) = (&fits[m - 1], &fits[m - 2]);
    config.dfmax.is_some_and(|d| cur.df > d)
        || cur.dev_ratio - prev.dev_ratio < config.fdev
        || cur.dev_ratio > config.deviance_thresh
}

/// glmnet's bound on fitted probabilities inside IRLS (`glmnet.control(pmin)`).
const PMIN: f64 = 1e-9;

/// glmnet's `mxitnr`: IRLS (outer) iterations per lambda.
const MXIT_IRLS: usize = 100;

/// Codes a binomial response by its two levels as R's `factor()` (the larger value is 1) and
/// applies R's class-size checks. Returns the 0/1 response and a warning for a small class.
fn binomial_response(y: ArrayView1<f64>) -> EconResult<(Vec<f64>, Option<String>)> {
    let mut levels: Vec<f64> = y.to_vec();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    match levels.len() {
        1 => return spec("binomial glmnet needs two classes; y has one value".to_string()),
        2 => {}
        k => {
            return spec(format!(
                "More than two classes (found {k}); use a two-class response"
            ));
        }
    }
    let hi = levels[1];
    let y01: Vec<f64> = y.iter().map(|&v| if v == hi { 1.0 } else { 0.0 }).collect();
    let ones = y01.iter().filter(|&&v| v == 1.0).count();
    let smallest = ones.min(y01.len() - ones);
    if smallest <= 1 {
        return spec("one binomial class has 1 or 0 observations; not allowed".to_string());
    }
    let note = (smallest < 8)
        .then(|| "one binomial class has fewer than 8 observations; dangerous ground".to_string());
    Ok((y01, note))
}

/// Binomial deviance `-2 sum(y ln p + (1 - y) ln(1 - p))` of `eta`, probabilities bounded at
/// 1e-10.
fn binomial_deviance_eta(y: &[f64], eta: &[f64]) -> f64 {
    y.iter()
        .zip(eta)
        .map(|(&yi, &e)| {
            let p = (1.0 / (1.0 + (-e).exp())).clamp(1e-10, 1.0 - 1e-10);
            -2.0 * (yi * p.ln() + (1.0 - yi) * (1.0 - p).ln())
        })
        .sum()
}

/// Binomial path: glmnet's `lognet`, IRLS with an inner weighted coordinate descent in which
/// the intercept is an unpenalised coordinate (so `sum(y - p) = 0` at convergence). y is not
/// standardised; the penalty is `lambda alpha |b|_1 + lambda (1 - alpha) / 2 |b|^2` in the
/// standardised x units.
fn binomial_path(
    design: &Design,
    y01: &[f64],
    p0: f64,
    null_deviance: f64,
    lambdas: &[f64],
    generated: bool,
    config: &GlmnetConfig,
) -> EconResult<PathFit> {
    let (n, p) = (design.n, design.p);
    let nf = n as f64;
    let what = "glmnet IRLS working vectors";
    let mut eta = resource::try_vec(n, 0.0, what)?;
    let mut w = resource::try_vec(n, 0.0, what)?;
    let mut r = resource::try_vec(n, 0.0, what)?;
    let mut b = vec![0.0; p];
    let mut b0 = if config.intercept {
        (p0 / (1.0 - p0)).ln()
    } else {
        0.0
    };
    let mut budget = PassBudget {
        used: 0,
        max: config.max_iter,
    };
    let mut fits = resource::try_with_capacity(lambdas.len(), "glmnet coefficient path")?;
    let mut warnings = Vec::new();
    // buffers reused across IRLS iterations and lambdas
    let mut b_old = vec![0.0; p];
    let (mut all, mut v_buf) = (design.usable_columns(), Vec::with_capacity(p));
    for (k, &lambda) in lambdas.iter().enumerate() {
        let l1 = lambda * config.alpha;
        let l2 = lambda * (1.0 - config.alpha);
        // Some((last change, whether the IRLS iteration cap was hit))
        let mut failed: Option<(f64, bool)> = None;
        for outer in 0..MXIT_IRLS {
            // eta = b0 + z b
            eta.fill(b0);
            for j in (0..p).filter(|&j| b[j] != 0.0) {
                for (e, z) in eta.iter_mut().zip(design.col(j)) {
                    *e += b[j] * z;
                }
            }
            for i in 0..n {
                let pr = (1.0 / (1.0 + (-eta[i]).exp())).clamp(PMIN, 1.0 - PMIN);
                w[i] = pr * (1.0 - pr);
                r[i] = (y01[i] - pr) / w[i];
            }
            let wls = Wls::reweighted(
                design,
                &w,
                std::mem::take(&mut all),
                std::mem::take(&mut v_buf),
            );
            b_old.copy_from_slice(&b);
            let b0_old = b0;
            let intercept = config.intercept.then_some(&mut b0);
            let solved = wls.solve(
                &mut b,
                intercept,
                &mut r,
                l1,
                l2,
                config.thresh,
                &mut budget,
            );
            let change = (0..p)
                .map(|j| wls.v[j] * (b[j] - b_old[j]).powi(2))
                .fold(wls.sum_w / nf * (b0 - b0_old).powi(2), f64::max);
            (all, v_buf) = wls.into_buffers();
            if let Solve::Budget(dlx) = solved? {
                failed = Some((dlx, false));
                break;
            }
            if change < config.thresh {
                break;
            }
            if outer + 1 == MXIT_IRLS {
                failed = Some((change, true));
            }
        }
        if let Some((dlx, irls_cap)) = failed {
            if fits.is_empty() {
                return Err(if irls_cap {
                    irls_failure(k + 1, dlx)
                } else {
                    first_lambda_failure(config.max_iter, dlx)
                });
            }
            warnings.push(if irls_cap {
                irls_truncation_warning(k + 1, lambdas.len(), lambda)
            } else {
                truncation_warning(k + 1, lambdas.len(), lambda, config.max_iter)
            });
            break;
        }
        // deviance at the converged coefficients
        eta.fill(b0);
        for j in (0..p).filter(|&j| b[j] != 0.0) {
            for (e, z) in eta.iter_mut().zip(design.col(j)) {
                *e += b[j] * z;
            }
        }
        let dev_ratio = 1.0 - binomial_deviance_eta(y01, &eta) / null_deviance;
        let coefficients: Vec<f64> = (0..p).map(|j| b[j] / design.scale[j]).collect();
        let intercept = if config.intercept {
            b0 - (0..p)
                .map(|j| design.center[j] * coefficients[j])
                .sum::<f64>()
        } else {
            0.0
        };
        let df = coefficients.iter().filter(|&&c| c != 0.0).count();
        fits.push(GlmnetCoefficients {
            lambda,
            intercept,
            coefficients,
            df,
            dev_ratio,
        });
        if generated && binomial_path_stops(&fits, config) {
            break;
        }
    }
    Ok(PathFit {
        fits,
        npasses: budget.used,
        warnings,
    })
}

/// Fit glmnet model
pub fn glmnet(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &GlmnetConfig,
) -> EconResult<GlmnetResult> {
    validate_glmnet_config(config)?;
    validate_glmnet_data(x, y)?;
    let (n, p) = x.dim();

    let design = standardize_design(x, config.standardize, config.intercept)?;
    let nf = n as f64;
    // The response as fitted (binomial: coded 0/1), the residual at the null fit that
    // lambda_max is computed from, and the null deviance.
    let (yfit, class_note) = match config.family {
        GlmnetFamily::Gaussian => (y.to_vec(), None),
        GlmnetFamily::Binomial => binomial_response(y)?,
    };
    let y_mean = yfit.iter().sum::<f64>() / nf;
    // gaussian: ym; binomial: p0 (glmnet: 0.5 without an intercept)
    let ym = match (config.family, config.intercept) {
        (_, true) => y_mean,
        (GlmnetFamily::Gaussian, false) => 0.0,
        (GlmnetFamily::Binomial, false) => 0.5,
    };
    let yc: Vec<f64> = yfit.iter().map(|v| v - ym).collect();
    let null_deviance = match config.family {
        GlmnetFamily::Gaussian => yc.iter().map(|v| v * v).sum(),
        GlmnetFamily::Binomial => {
            -2.0 * yfit
                .iter()
                .map(|&v| v * ym.ln() + (1.0 - v) * (1.0 - ym).ln())
                .sum::<f64>()
        }
    };

    if !null_deviance.is_finite() {
        return Err(EconError::InvalidSpecification {
            message: "the response is too large in magnitude: its null deviance is not finite \
                      in double precision; rescale it"
                .to_string(),
        });
    }

    // lambda_max: the smallest lambda at which every coefficient is zero (glmnet), from the
    // gradient at the null fit; divided by max(alpha, 1e-3) so a ridge path starts where
    // glmnet's does.
    let grad_max = design
        .usable_columns()
        .into_iter()
        .map(|j| {
            design
                .col(j)
                .iter()
                .zip(&yc)
                .map(|(z, y)| z * y)
                .sum::<f64>()
                .abs()
                / nf
        })
        .fold(0.0f64, f64::max);
    let lambda_max = grad_max / config.alpha.max(ALPHA_FLOOR);
    let (lambda_seq, generated, note) = lambda_path(config, lambda_max, n, p)?;

    // Each fit holds p coefficients plus seven words (lambda, intercept, the coefficient Vec
    // header, df, dev_ratio).
    let what = "glmnet coefficient path";
    resource::check_alloc::<f64>(
        resource::checked_len(&[lambda_seq.len(), p.saturating_add(7)], what)?,
        what,
    )?;

    let path = match config.family {
        GlmnetFamily::Gaussian => {
            let ys = (null_deviance / nf).sqrt();
            if ys == 0.0 {
                return Err(EconError::InvalidSpecification {
                    message: "y is constant; gaussian glmnet fails at standardization step"
                        .to_string(),
                });
            }
            gaussian_path(&design, &yc, ym, ys, &lambda_seq, generated, config)?
        }
        GlmnetFamily::Binomial => binomial_path(
            &design,
            &yfit,
            ym,
            null_deviance,
            &lambda_seq,
            generated,
            config,
        )?,
    };

    let mut warnings: Vec<String> = class_note.into_iter().chain(note).collect();
    warnings.extend(path.warnings);
    Ok(GlmnetResult {
        variable_names: (0..p).map(|i| format!("V{}", i + 1)).collect(),
        dependent_var: "y".to_string(),
        alpha: config.alpha,
        lambda: path.fits.iter().map(|f| f.lambda).collect(),
        fits: path.fits,
        n_obs: n,
        n_features: p,
        null_deviance,
        family: config.family,
        x_mean: config.intercept.then(|| design.center.clone()),
        x_scale: config.standardize.then(|| design.scale.clone()),
        y_mean,
        standardize: config.standardize,
        intercept: config.intercept,
        npasses: path.npasses,
        warnings,
    })
}

/// A numeric column as `Array1<f64>`; a missing value is an error, as in R's glmnet.
fn extract_column(df: &DataFrame, col_name: &str) -> EconResult<Array1<f64>> {
    let (values, _) = numeric_column(df, col_name, NaPolicy::Fail)?;
    Ok(Array1::from_vec(values))
}

/// Fit glmnet model from Dataset. A missing value in y or X is an `EconError::NullValues`
/// naming the column (R's glmnet stops); integer columns are numbers.
pub fn run_glmnet(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &GlmnetConfig,
) -> EconResult<GlmnetResult> {
    // Build design matrix (without intercept - handled internally)
    let df = dataset.df();
    let dm = DesignMatrix::from_dataframe(df, x_cols, false)
        .map_err(|e| econ_error_from_design(e, df))?;
    let x = dm.data.view();
    let y = extract_column(df, y_col)?;

    let mut result = glmnet(x, y.view(), config)?;

    // Update variable names
    result.variable_names = x_cols.iter().map(|&s| s.to_string()).collect();
    result.dependent_var = y_col.to_string();

    Ok(result)
}

/// Cross-validated glmnet with random folds: `nfolds` in 2..=n (R requires at least 3), labels
/// `(i mod nfolds) + 1` shuffled with ChaCha8 seeded by `seed` (default
/// [`CV_GLMNET_DEFAULT_SEED`]), then [`cv_glmnet_foldid`].
pub fn cv_glmnet(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &GlmnetConfig,
    nfolds: usize,
    seed: Option<u64>,
) -> EconResult<GlmnetCvResult> {
    let n = x.nrows();
    if nfolds < 2 || nfolds > n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "nfolds must be at least 2 and at most the number of observations ({n}), got {nfolds}"
            ),
        });
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed.unwrap_or(CV_GLMNET_DEFAULT_SEED));
    let mut foldid = resource::try_with_capacity::<usize>(n, "glmnet fold labels")?;
    foldid.extend((0..n).map(|i| i % nfolds + 1));
    foldid.shuffle(&mut rng);
    cv_glmnet_foldid(x, y, config, &foldid)
}

/// `cv.glmnet(foldid = ...)`: labels 1..=K (K >= 2), one per row, every label used.
///
/// The full data are fitted first; its lambdas are the CV grid. Each fold is fitted on its own
/// path (the user's `lambda` when given, as cv.glmnet passes it through) and its held-out rows
/// are predicted at every grid lambda by [`glmnet_predict`]'s interpolation, clamped at the
/// fold path's ends, so every lambda has an error from every fold. The error is the squared
/// residual (gaussian) or the binomial deviance with probabilities bounded at 1e-5;
/// `cvm` and `cvsd` are cv.glmnet's fold-size-weighted mean and standard error of the fold means.
pub fn cv_glmnet_foldid(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &GlmnetConfig,
    foldid: &[usize],
) -> EconResult<GlmnetCvResult> {
    let n = x.nrows();
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    if foldid.len() != n {
        return spec(format!(
            "foldid must have one label per row: {} labels for {n} rows",
            foldid.len()
        ));
    }
    if foldid.contains(&0) {
        return spec("foldid labels must be 1, 2, ..., K (got 0)".to_string());
    }
    let k = foldid.iter().copied().max().unwrap_or(0);
    if k < 2 {
        return spec("foldid must use at least 2 folds".to_string());
    }
    if k > n {
        return spec(format!(
            "foldid labels must be 1, 2, ..., K with K at most the {n} rows (got label {k})"
        ));
    }
    let mut sizes = resource::try_vec(k, 0usize, "glmnet fold sizes")?;
    for &f in foldid {
        sizes[f - 1] += 1;
    }
    if let Some(unused) = sizes.iter().position(|&c| c == 0) {
        return spec(format!(
            "foldid: label {} of 1..={k} has no observation",
            unused + 1
        ));
    }

    let full = glmnet(x, y, config)?;
    let lambda = full.lambda.clone();
    let nl = lambda.len();
    // the response the errors are computed on (binomial: coded 0/1 as the fit codes it)
    let yerr: Vec<f64> = match config.family {
        GlmnetFamily::Gaussian => y.to_vec(),
        GlmnetFamily::Binomial => binomial_response(y)?.0,
    };
    let what = "glmnet cross-validation errors";
    let mut fold_means = resource::try_vec(resource::checked_len(&[k, nl], what)?, 0.0, what)?;
    let mut fold_path_lengths = Vec::with_capacity(k);
    let mut warnings = Vec::new();
    for fold in 1..=k {
        resource::checkpoint()?;
        let train: Vec<usize> = (0..n).filter(|&i| foldid[i] != fold).collect();
        let test: Vec<usize> = (0..n).filter(|&i| foldid[i] == fold).collect();
        let x_train = x.select(Axis(0), &train);
        let y_train = y.select(Axis(0), &train);
        let x_test = x.select(Axis(0), &test);
        let f = glmnet(x_train.view(), y_train.view(), config)?;
        fold_path_lengths.push(f.lambda.len());
        warnings.extend(f.warnings.iter().map(|w| format!("fold {fold}: {w}")));
        for (l, &lam) in lambda.iter().enumerate() {
            let (b0, beta) = interpolate(&f, lam)?;
            let eta = x_test.dot(&Array1::from_vec(beta)) + b0;
            let err: f64 = test
                .iter()
                .zip(eta.iter())
                .map(|(&i, &e)| match config.family {
                    GlmnetFamily::Gaussian => (yerr[i] - e).powi(2),
                    GlmnetFamily::Binomial => {
                        let p = (1.0 / (1.0 + (-e).exp())).clamp(CV_PROB_MIN, 1.0 - CV_PROB_MIN);
                        -2.0 * (yerr[i] * p.ln() + (1.0 - yerr[i]) * (1.0 - p).ln())
                    }
                })
                .sum();
            fold_means[(fold - 1) * nl + l] = err / test.len() as f64;
        }
    }

    let total = n as f64;
    let mut cvm = Vec::with_capacity(nl);
    let mut cvsd = Vec::with_capacity(nl);
    for l in 0..nl {
        let m = (0..k)
            .map(|f| sizes[f] as f64 * fold_means[f * nl + l])
            .sum::<f64>()
            / total;
        let var = (0..k)
            .map(|f| sizes[f] as f64 * (fold_means[f * nl + l] - m).powi(2))
            .sum::<f64>()
            / total
            / (k - 1) as f64;
        cvm.push(m);
        cvsd.push(var.sqrt());
    }
    let cvup: Vec<f64> = cvm.iter().zip(&cvsd).map(|(&m, &s)| m + s).collect();
    let cvlo: Vec<f64> = cvm.iter().zip(&cvsd).map(|(&m, &s)| m - s).collect();

    // lambda.min: the largest lambda with the smallest finite cvm (the first index)
    let lambda_min_index = cvm
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .min_by(|(i, a), (j, b)| a.total_cmp(b).then(i.cmp(j)))
        .map(|(i, _)| i)
        .ok_or_else(|| {
            EconError::Computation("cross-validation produced no finite error".to_string())
        })?;
    let threshold = cvm[lambda_min_index] + cvsd[lambda_min_index];
    let lambda_1se_index = (0..=lambda_min_index)
        .find(|&i| cvm[i] <= threshold)
        .unwrap_or(lambda_min_index);

    Ok(GlmnetCvResult {
        lambda_min: lambda[lambda_min_index],
        lambda_1se: lambda[lambda_1se_index],
        fit: full,
        lambda,
        cvm,
        cvsd,
        cvup,
        cvlo,
        lambda_min_index,
        lambda_1se_index,
        nfolds: k,
        foldid: foldid.to_vec(),
        fold_path_lengths,
        warnings,
    })
}

/// Run cross-validated glmnet from Dataset (missing values refused as in [`run_glmnet`]).
pub fn run_cv_glmnet(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &GlmnetConfig,
    nfolds: usize,
    seed: Option<u64>,
) -> EconResult<GlmnetCvResult> {
    let df = dataset.df();
    let dm = DesignMatrix::from_dataframe(df, x_cols, false)
        .map_err(|e| econ_error_from_design(e, df))?;
    let x = dm.data.view();
    let y = extract_column(df, y_col)?;

    let mut result = cv_glmnet(x, y.view(), config, nfolds, seed)?;

    // Update variable names
    result.fit.variable_names = x_cols.iter().map(|&s| s.to_string()).collect();
    result.fit.dependent_var = y_col.to_string();

    Ok(result)
}

/// Convenience function for ridge regression (alpha = 0)
pub fn ridge(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    lambda: Option<Vec<f64>>,
) -> EconResult<GlmnetResult> {
    let config = GlmnetConfig {
        alpha: 0.0,
        lambda,
        ..Default::default()
    };
    glmnet(x, y, &config)
}

/// Convenience function for lasso regression (alpha = 1)
pub fn lasso(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    lambda: Option<Vec<f64>>,
) -> EconResult<GlmnetResult> {
    let config = GlmnetConfig {
        alpha: 1.0,
        lambda,
        ..Default::default()
    };
    glmnet(x, y, &config)
}

/// Run ridge regression from Dataset
pub fn run_ridge(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    lambda: Option<Vec<f64>>,
) -> EconResult<GlmnetResult> {
    let config = GlmnetConfig {
        alpha: 0.0,
        lambda,
        ..Default::default()
    };
    run_glmnet(dataset, y_col, x_cols, &config)
}

/// Run lasso regression from Dataset
pub fn run_lasso(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    lambda: Option<Vec<f64>>,
) -> EconResult<GlmnetResult> {
    let config = GlmnetConfig {
        alpha: 1.0,
        lambda,
        ..Default::default()
    };
    run_glmnet(dataset, y_col, x_cols, &config)
}

/// The fit at `s` as `predict.glmnet(s, exact = FALSE)` (glmnet's `lambda.interp`): linear
/// interpolation in lambda between the two bracketing fits, clamped to the first / last fit
/// outside the path. Returns the intercept and the coefficients.
fn interpolate(result: &GlmnetResult, s: f64) -> EconResult<(f64, Vec<f64>)> {
    if !(s.is_finite() && s >= 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("lambda must be finite and non-negative (got {s})"),
        });
    }
    let fits = &result.fits;
    let (Some(first), Some(last)) = (fits.first(), fits.last()) else {
        return Err(EconError::Computation("empty lambda path".to_string()));
    };
    let pick = |f: &GlmnetCoefficients| Ok((f.intercept, f.coefficients.clone()));
    if fits.len() == 1 || s >= first.lambda {
        return pick(first);
    }
    if s <= last.lambda {
        return pick(last);
    }
    // l: the last fit with lambda >= s (lambdas decrease along the path)
    let l = fits.iter().rposition(|f| f.lambda >= s).unwrap_or(0);
    if fits[l].lambda == s {
        return pick(&fits[l]);
    }
    let (left, right) = (&fits[l], &fits[l + 1]);
    let frac = (s - right.lambda) / (left.lambda - right.lambda);
    let mix = |a: f64, b: f64| frac * a + (1.0 - frac) * b;
    Ok((
        mix(left.intercept, right.intercept),
        left.coefficients
            .iter()
            .zip(&right.coefficients)
            .map(|(&a, &b)| mix(a, b))
            .collect(),
    ))
}

/// Predict at `lambda` as `predict.glmnet(s = lambda, exact = FALSE)`: the coefficients are
/// linearly interpolated between the bracketing fits (clamped to the first / last fit outside
/// the path); probabilities for binomial. `lambda` must be finite and >= 0 and `x` must have the
/// fit's number of columns (otherwise `InvalidSpecification`).
pub fn glmnet_predict(
    result: &GlmnetResult,
    x: ArrayView2<f64>,
    lambda: f64,
) -> EconResult<Array1<f64>> {
    if x.ncols() != result.n_features {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "x has {} columns but the fit has {}",
                x.ncols(),
                result.n_features
            ),
        });
    }
    let (b0, beta) = interpolate(result, lambda)?;
    let eta = x.dot(&Array1::from_vec(beta)) + b0;
    match result.family {
        GlmnetFamily::Gaussian => Ok(eta),
        GlmnetFamily::Binomial => Ok(eta.mapv(|e| 1.0 / (1.0 + (-e).exp()))),
    }
}

/// `coef.glmnet(s = lambda)`: the intercept and the nonzero coefficients, by name, of the fit
/// interpolated at `lambda` (as [`glmnet_predict`]).
pub fn glmnet_coef(result: &GlmnetResult, lambda: f64) -> EconResult<(f64, Vec<(String, f64)>)> {
    let (b0, beta) = interpolate(result, lambda)?;
    let coefs = result
        .variable_names
        .iter()
        .zip(beta)
        .filter(|(_, c)| *c != 0.0)
        .map(|(name, c)| (name.clone(), c))
        .collect();
    Ok((b0, coefs))
}

/// At most this many nonzero coefficients are printed per fit (the largest by |coef|).
const DISPLAY_MAX_COEFS: usize = 20;

/// `(Intercept) b0, name coef, ...` for the nonzero coefficients; more than
/// [`DISPLAY_MAX_COEFS`]: the largest by |coef| (in column order) and a count of the rest.
fn coefficient_terms(names: &[String], fit: &GlmnetCoefficients) -> String {
    let mut nonzero: Vec<usize> = (0..fit.coefficients.len())
        .filter(|&j| fit.coefficients[j] != 0.0)
        .collect();
    let hidden = nonzero.len().saturating_sub(DISPLAY_MAX_COEFS);
    if hidden > 0 {
        nonzero.sort_by(|&a, &b| {
            fit.coefficients[b]
                .abs()
                .total_cmp(&fit.coefficients[a].abs())
                .then(a.cmp(&b))
        });
        nonzero.truncate(DISPLAY_MAX_COEFS);
        nonzero.sort_unstable();
    }
    let mut out = format!("(Intercept) {:.6}", fit.intercept);
    for j in nonzero {
        let name = names
            .get(j)
            .map_or_else(|| format!("V{}", j + 1), Clone::clone);
        out.push_str(&format!(", {name} {:.6}", fit.coefficients[j]));
    }
    if hidden > 0 {
        out.push_str(&format!(", (+{hidden} more nonzero)"));
    }
    out
}

impl std::fmt::Display for GlmnetResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Glmnet Result: {} ~ {} variables",
            self.dependent_var, self.n_features
        )?;
        writeln!(f, "  Family:      {}", self.family)?;
        writeln!(
            f,
            "  Alpha:       {:.2} ({})",
            self.alpha,
            if self.alpha == 1.0 {
                "lasso"
            } else if self.alpha == 0.0 {
                "ridge"
            } else {
                "elastic net"
            }
        )?;
        writeln!(f, "  Standardize: {}", self.standardize)?;
        writeln!(f, "  Intercept:   {}", self.intercept)?;
        writeln!(f, "  N:           {}", self.n_obs)?;
        writeln!(
            f,
            "  Lambda path: {} values ({} passes)",
            self.lambda.len(),
            self.npasses
        )?;
        writeln!(f)?;
        writeln!(f, "  Lambda      Df    %Dev")?;
        for fit in &self.fits {
            writeln!(
                f,
                "  {:.4e}    {:3}   {:.3}",
                fit.lambda, fit.df, fit.dev_ratio
            )?;
        }
        writeln!(f)?;
        writeln!(f, "  Coefficients (nonzero; intercept first):")?;
        for (i, fit) in self.fits.iter().enumerate() {
            writeln!(
                f,
                "  [{:>3}] lambda {:.4e}: {}",
                i + 1,
                fit.lambda,
                coefficient_terms(&self.variable_names, fit)
            )?;
        }
        for w in &self.warnings {
            writeln!(f, "Warning: {w}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for GlmnetCvResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cross-Validated Glmnet ({}-fold)", self.nfolds)?;
        writeln!(f)?;
        writeln!(
            f,
            "  lambda.min:  {:.4e} (index {})",
            self.lambda_min,
            self.lambda_min_index + 1
        )?;
        writeln!(
            f,
            "  lambda.1se:  {:.4e} (index {})",
            self.lambda_1se,
            self.lambda_1se_index + 1
        )?;
        writeln!(f)?;
        let (min, se1) = (self.fit_min(), self.fit_1se());
        writeln!(
            f,
            "  At lambda.min: {} non-zero coefs, {:.1}% dev explained",
            min.df,
            min.dev_ratio * 100.0
        )?;
        writeln!(
            f,
            "  At lambda.1se: {} non-zero coefs, {:.1}% dev explained",
            se1.df,
            se1.dev_ratio * 100.0
        )?;
        writeln!(f)?;
        let names = &self.fit.variable_names;
        writeln!(
            f,
            "  Coefficients at lambda.min (nonzero; intercept first):"
        )?;
        writeln!(f, "    {}", coefficient_terms(names, min))?;
        writeln!(
            f,
            "  Coefficients at lambda.1se (nonzero; intercept first):"
        )?;
        writeln!(f, "    {}", coefficient_terms(names, se1))?;
        for w in self.fit.warnings.iter().chain(&self.warnings) {
            writeln!(f, "Warning: {w}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use approx::assert_relative_eq;
    use ndarray::array;

    /// Family names ignore case and the separators ' ', '-', '.', '_' (dropped, as documented).
    #[test]
    fn test_glmnet_family_ignores_separators() {
        for g in ["gaus-sian", "gaus sian", "Gaus.Sian", "gaus_sian"] {
            assert_eq!(
                g.parse::<GlmnetFamily>().unwrap(),
                GlmnetFamily::Gaussian,
                "{g}"
            );
        }
        for b in ["bi nomial", "bi-nomial", "LOG_IT"] {
            assert_eq!(
                b.parse::<GlmnetFamily>().unwrap(),
                GlmnetFamily::Binomial,
                "{b}"
            );
        }
    }

    /// Inputs whose mean, variance or null deviance overflow are refused, naming the column
    /// or the response.
    #[test]
    fn test_glmnet_refuses_overflowing_inputs() {
        let n = 12;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            let v = ((i * 7 + j * 3) % 5) as f64 + 0.1 * i as f64;
            if j == 1 { v * 1e300 } else { v }
        });
        let y = Array1::from_shape_fn(n, |i| 1.0 + 0.3 * i as f64 + ((i * 5) % 3) as f64 * 0.2);
        match glmnet(x.view(), y.view(), &GlmnetConfig::default()) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("column 2"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        let x = x.mapv(|v| if v > 1e200 { v / 1e300 } else { v });
        let y = y.mapv(|v| v * 1e300);
        match glmnet(x.view(), y.view(), &GlmnetConfig::default()) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("response"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// A fold label above the number of rows is refused before any allocation sized by it.
    #[test]
    fn test_cv_glmnet_fold_label_above_n() {
        let x = Array2::from_shape_fn((6, 1), |(i, _)| i as f64 + 0.3 * ((i * 3) % 2) as f64);
        let y = Array1::from(vec![1.1, 1.9, 3.2, 3.8, 5.1, 6.3]);
        let foldid = [1usize, 2, 1, 2, 1, usize::MAX];
        match cv_glmnet_foldid(x.view(), y.view(), &GlmnetConfig::default(), &foldid) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("at most the 6 rows"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// The IRLS cap has its own message, not the coordinate-pass budget's.
    #[test]
    fn test_glmnet_irls_cap_message() {
        let w = irls_truncation_warning(3, 10, 0.25);
        assert!(
            w.starts_with("IRLS did not converge in 100 iterations at lambda[3]"),
            "{w}"
        );
        assert!(!w.contains("max_iter"), "{w}");
        match irls_failure(1, 0.5) {
            EconError::ConvergenceFailure {
                iterations,
                suggestion,
                ..
            } => {
                assert_eq!(iterations, MXIT_IRLS);
                assert!(suggestion.contains("lambda[1]"), "{suggestion}");
            }
            e => panic!("{e:?}"),
        }
    }

    /// The CV Display numbers lambdas from 1, as the fit list does.
    #[test]
    fn test_cv_glmnet_display_index_is_one_based() {
        let x = Array2::from_shape_fn((30, 2), |(i, j)| {
            ((i * (j + 3)) % 7) as f64 + 0.1 * i as f64 * (j as f64 + 1.0)
        });
        let y = Array1::from_shape_fn(30, |i| {
            1.0 + 0.5 * x[[i, 0]] - 0.2 * x[[i, 1]] + (((i * 13) % 5) as f64 - 2.0) * 0.3
        });
        let foldid: Vec<usize> = (0..30).map(|i| i % 3 + 1).collect();
        let cv = cv_glmnet_foldid(x.view(), y.view(), &GlmnetConfig::default(), &foldid).unwrap();
        let t = cv.to_string();
        assert!(
            t.contains(&format!("(index {})", cv.lambda_min_index + 1)),
            "{t}"
        );
        assert!(
            t.contains(&format!("(index {})", cv.lambda_1se_index + 1)),
            "{t}"
        );
    }

    #[test]
    fn test_soft_threshold() {
        assert_relative_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_relative_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_relative_eq!(soft_threshold(1.0, 2.0), 0.0);
        assert_relative_eq!(soft_threshold(-1.0, 2.0), 0.0);
    }

    #[test]
    fn test_ridge_closed_form() {
        // Ridge has closed form: β = (X'X + λI)⁻¹ X'y
        // For simple case, verify it produces non-zero coefficients for all variables
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0], [5.0, 4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = GlmnetConfig {
            alpha: 0.0,
            lambda: Some(vec![0.1]),
            nlambda: 1,
            standardize: false,
            intercept: false,
            ..Default::default()
        };

        let result = glmnet(x.view(), y.view(), &config).unwrap();

        // Both coefficients should be non-zero (ridge doesn't zero out coefficients)
        assert_eq!(result.fits.len(), 1);
        assert!(result.fits[0].coefficients[0].abs() > 0.01);
        assert!(result.fits[0].coefficients[1].abs() > 0.01);
    }

    #[test]
    fn test_lasso_sparsity() {
        // Lasso should produce sparse solutions with high lambda
        let x = array![
            [1.0, 0.1, 0.2],
            [2.0, 0.2, 0.1],
            [3.0, 0.1, 0.3],
            [4.0, 0.3, 0.1],
            [5.0, 0.2, 0.2]
        ];
        // y is strongly correlated with first column, weakly with others
        let y = array![1.1, 2.0, 3.1, 4.0, 5.1];

        let config = GlmnetConfig {
            alpha: 1.0, // lasso
            nlambda: 50,
            standardize: true,
            intercept: true,
            ..Default::default()
        };

        let result = glmnet(x.view(), y.view(), &config).unwrap();

        // At high lambda, should have fewer non-zero coefficients
        let high_lambda_fit = &result.fits[0];
        let low_lambda_fit = result.fits.last().unwrap();

        assert!(high_lambda_fit.df <= low_lambda_fit.df);
    }

    #[test]
    fn test_lambda_path() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y = array![2.0, 4.0, 5.0, 7.0, 9.0];

        let config = GlmnetConfig {
            alpha: 0.5,
            nlambda: 20,
            ..Default::default()
        };

        let result = glmnet(x.view(), y.view(), &config).unwrap();

        // Lambda should be decreasing
        for i in 1..result.lambda.len() {
            assert!(result.lambda[i] < result.lambda[i - 1]);
        }

        // Deviance ratio should generally increase
        assert!(result.fits.last().unwrap().dev_ratio >= result.fits[0].dev_ratio);
    }

    #[test]
    fn test_cv_glmnet() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0]
        ];
        let y = array![2.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];

        let config = GlmnetConfig {
            alpha: 1.0,
            nlambda: 20,
            ..Default::default()
        };

        let cv_result = cv_glmnet(x.view(), y.view(), &config, 5, Some(42)).unwrap();

        // lambda.1se should be >= lambda.min
        assert!(cv_result.lambda_1se >= cv_result.lambda_min);

        // CV errors should be computed
        assert_eq!(cv_result.cvm.len(), cv_result.lambda.len());
        assert_eq!(cv_result.cvsd.len(), cv_result.lambda.len());
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// The `x`/`y` data of `test_lambda_path`.
    fn lambda_path_data() -> (Array2<f64>, Array1<f64>) {
        (
            array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
            array![2.0, 4.0, 5.0, 7.0, 9.0],
        )
    }

    #[test]
    fn test_glmnet_rejects_invalid_controls() {
        let (x, y) = lambda_path_data();
        for (cfg, field) in [
            (
                GlmnetConfig {
                    nlambda: 0,
                    ..Default::default()
                },
                "nlambda",
            ),
            (
                GlmnetConfig {
                    thresh: 0.0,
                    ..Default::default()
                },
                "thresh",
            ),
            (
                GlmnetConfig {
                    thresh: f64::NAN,
                    ..Default::default()
                },
                "thresh",
            ),
            (
                GlmnetConfig {
                    max_iter: 0,
                    ..Default::default()
                },
                "max_iter",
            ),
        ] {
            match glmnet(x.view(), y.view(), &cfg).unwrap_err() {
                EconError::InvalidSpecification { message } => {
                    assert!(message.contains(field), "{message}")
                }
                e => panic!("{field}: {e:?}"),
            }
        }
        // a user lambda sequence makes nlambda irrelevant (test_ridge_closed_form relies on this)
        let cfg = GlmnetConfig {
            lambda: Some(vec![0.1]),
            nlambda: 0,
            ..Default::default()
        };
        assert!(glmnet(x.view(), y.view(), &cfg).is_ok());
    }

    #[test]
    fn test_glmnet_impossible_nlambda_is_an_error() {
        let (x, y) = lambda_path_data();
        let cfg = GlmnetConfig {
            nlambda: 1usize << 61,
            ..Default::default()
        };
        let e = glmnet(x.view(), y.view(), &cfg).unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_glmnet_path_respects_alloc_ceiling() {
        let (x, y) = lambda_path_data();
        let cfg = GlmnetConfig {
            nlambda: 100_000,
            ..Default::default()
        };
        // 100_000 lambdas * 8 bytes = 800 KB
        let e = resource::with_alloc_ceiling(Some(64 << 10), || glmnet(x.view(), y.view(), &cfg))
            .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_glmnet_and_cv_stop_when_cancelled() {
        let (x, y) = lambda_path_data();
        let cfg = GlmnetConfig {
            nlambda: 20,
            ..Default::default()
        };
        let e =
            resource::with_budget(cancelled(), || glmnet(x.view(), y.view(), &cfg)).unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
        let e = resource::with_budget(cancelled(), || {
            cv_glmnet(x.view(), y.view(), &cfg, 2, Some(1))
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    /// REG-25: the glmnet wrappers refuse a missing y as R's glmnet does (it stops), and
    /// a text y is not numeric (it used to become zeros); an Int64 y is a number. A
    /// property test: no value is compared with R, so it loads the datasets without a
    /// case.
    #[test]
    fn test_na_glmnet_columns() {
        use crate::regression::ols::na_fixtures::{as_float, mtcars, smoothers_na};
        let cfg = GlmnetConfig::default();
        let sm = smoothers_na();
        let null_y = |r: EconResult<()>| match r {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!((column.as_str(), count), ("y", 2))
            }
            other => panic!("{other:?}"),
        };
        null_y(run_glmnet(&sm, "y", &["id"], &cfg).map(|_| ()));
        null_y(run_cv_glmnet(&sm, "y", &["id"], &cfg, 5, Some(1)).map(|_| ()));
        null_y(run_ridge(&sm, "y", &["id"], None).map(|_| ()));
        null_y(run_lasso(&sm, "y", &["id"], None).map(|_| ()));
        let mt = mtcars();
        match run_glmnet(&mt, "model", &["wt", "disp"], &cfg) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "model"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            run_glmnet(&mt, "nope", &["wt", "disp"], &cfg),
            Err(EconError::ColumnNotFound { .. })
        ));
        assert!(matches!(
            run_glmnet(&mt, "hp", &["wt", "nope"], &cfg),
            Err(EconError::ColumnNotFound { .. })
        ));
        let a = run_glmnet(&mt, "hp", &["wt", "disp"], &cfg).unwrap();
        let b = run_glmnet(&as_float(&mt, "hp"), "hp", &["wt", "disp"], &cfg).unwrap();
        let (fa, fb) = (a.fits.last().unwrap(), b.fits.last().unwrap());
        assert_eq!(
            (&a.lambda, fa.intercept, &fa.coefficients),
            (&b.lambda, fb.intercept, &fb.coefficients)
        );
    }

    // ---- R reference case glmnet_paths (T3-WP07) ----

    use p2a_validation::{Check, Tol};

    const X8: [&str; 8] = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"];
    const LAMS: [f64; 4] = [1.0, 0.3, 0.1, 0.03];
    const BLAMS: [f64; 3] = [0.1, 0.03, 0.01];
    /// Coordinate descent converges linearly. At thresh 1e-24 an R port of this algorithm
    /// lands within 1e-12 of the KKT point (`*_tight`); abs 1e-11 leaves a 10x margin for
    /// summation order.
    const EXACT: Tol = Tol::new(1e-9, 1e-11);
    /// glmnet's own output stops at thresh 1e-14 (max_j xv_j delta_j^2): up to 1.0e-7 from
    /// the minimiser here.
    const R_GLMNET: Tol = Tol::new(1e-6, 2e-7);
    /// glmnet's own dev.ratio: up to 2.0e-9 from the minimiser's.
    const R_DEV: Tol = Tol::new(1e-8, 5e-9);

    fn k30() -> Vec<String> {
        (1..=30).map(|j| format!("K{j}")).collect()
    }
    fn refs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }
    fn tight(
        alpha: f64,
        standardize: bool,
        intercept: bool,
        lambda: Option<&[f64]>,
    ) -> GlmnetConfig {
        let mut c = GlmnetConfig::default();
        c.alpha = alpha;
        c.standardize = standardize;
        c.intercept = intercept;
        c.lambda = lambda.map(<[f64]>::to_vec);
        c.thresh = 1e-24;
        c
    }
    /// R's as.numeric(as.matrix(coef(fit))): (p + 1) x L, column-major, intercept first.
    fn coef_path(r: &GlmnetResult) -> Vec<f64> {
        r.fits
            .iter()
            .flat_map(|f| std::iter::once(f.intercept).chain(f.coefficients.iter().copied()))
            .collect()
    }
    fn coef_vec(f: &GlmnetCoefficients) -> Vec<f64> {
        std::iter::once(f.intercept)
            .chain(f.coefficients.iter().copied())
            .collect()
    }
    fn fold_ids(ds: &Dataset, col: &str) -> Vec<usize> {
        numeric_column(ds.df(), col, NaPolicy::Fail)
            .unwrap()
            .0
            .iter()
            .map(|&v| v as usize)
            .collect()
    }
    fn dev_ratios(r: &GlmnetResult) -> Vec<f64> {
        r.fits.iter().map(|f| f.dev_ratio).collect()
    }
    fn dfs(r: &GlmnetResult) -> Vec<f64> {
        r.fits.iter().map(|f| f.df as f64).collect()
    }
    fn xy(ds: &Dataset, y: &str, cols: &[&str]) -> (Array2<f64>, Array1<f64>) {
        let x = DesignMatrix::from_dataframe(ds.df(), cols, false)
            .unwrap()
            .data;
        (
            x,
            Array1::from_vec(numeric_column(ds.df(), y, NaPolicy::Fail).unwrap().0),
        )
    }

    fn check_gaussian(
        case: &p2a_validation::Case,
        ds: &Dataset,
        alpha: f64,
        standardize: bool,
        tag: &str,
    ) {
        let mut chk = Check::new(case);
        for (intercept, t) in [(true, tag.to_string()), (false, format!("{tag}_noint"))] {
            let r = run_glmnet(
                ds,
                "y",
                &X8,
                &tight(alpha, standardize, intercept, Some(&LAMS[..])),
            )
            .unwrap();
            let (b, d) = (coef_path(&r), dev_ratios(&r));
            chk.vector(&format!("{t}_coef_tight"), &b, EXACT)
                .vector(&format!("{t}_coef"), &b, R_GLMNET)
                .vector(&format!("{t}_dev_ratio_tight"), &d, Tol::iterative(1e-9))
                .vector(&format!("{t}_dev_ratio"), &d, R_DEV);
        }
        chk.finish();
    }

    #[test]
    fn test_validate_glmnet_gaussian_lasso_std() {
        // REG-7 (reverse): intercept = false scales x without centring it. The intercept fit
        // is a guard.
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        check_gaussian(&case, &ds, 1.0, true, "a10_std");
    }

    #[test]
    fn test_validate_glmnet_gaussian_lasso_raw() {
        // REG-7: standardize = false still centres x for the intercept.
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        check_gaussian(&case, &ds, 1.0, false, "a10_raw");
    }

    #[test]
    fn test_validate_glmnet_gaussian_enet_std() {
        // REG-40: the L2 penalty is lambda (1 - alpha) / sd(y) in glmnet's units.
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        check_gaussian(&case, &ds, 0.5, true, "a5_std");
    }

    #[test]
    fn test_validate_glmnet_gaussian_enet_raw() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        check_gaussian(&case, &ds, 0.5, false, "a5_raw");
    }

    #[test]
    fn test_validate_glmnet_gaussian_ridge_std() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        check_gaussian(&case, &ds, 0.0, true, "a0_std");
    }

    #[test]
    fn test_validate_glmnet_gaussian_ridge_raw() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        check_gaussian(&case, &ds, 0.0, false, "a0_raw");
    }

    #[test]
    fn test_validate_glmnet_kkt_excluded_variable() {
        // Pre-WP07: K1, K16, K23 stay at 0 at path indices 54, 61, 80 with
        // |x~_j'r|/n = 1.00995, 1.00386, 1.05708 x lambda, df 26/28/29 where the solution
        // has 27/29/30 (REG-41).
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let cols = k30();
        let lam = case.vector("kkt_lambda");
        let r = run_glmnet(
            &ds,
            "yk",
            &refs(&cols),
            &tight(1.0, true, true, Some(lam.as_slice())),
        )
        .unwrap();
        let (x, y) = xy(&ds, "yk", &refs(&cols));
        let n = x.nrows() as f64;
        for (k, f) in r.fits.iter().enumerate() {
            let fitted = x.dot(&Array1::from_vec(f.coefficients.clone())) + f.intercept;
            let resid = &y - &fitted;
            for j in 0..30 {
                if f.coefficients[j] != 0.0 {
                    continue;
                }
                let col = x.column(j);
                let m = col.sum() / n;
                let sd = (col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt();
                let g = col
                    .iter()
                    .zip(resid.iter())
                    .map(|(v, e)| (v - m) * e)
                    .sum::<f64>()
                    .abs()
                    / (n * sd);
                assert!(
                    g <= f.lambda * (1.0 + 1e-9),
                    "index {k}: K{} excluded with |x~'r|/n = {g:.9} > lambda {:.9}",
                    j + 1,
                    f.lambda
                );
            }
        }
        Check::new(&case)
            .vector("kkt_lambda", &r.lambda, Tol::exact())
            .vector("kkt_df", &dfs(&r), Tol::exact())
            // AR(0.85) design: linear convergence; the largest gap at thresh 1e-24 is 2.35e-11
            // in the R port of this algorithm
            .vector("kkt_coef_tight", &coef_path(&r), Tol::new(1e-9, 1e-10))
            .finish();
    }

    #[test]
    fn test_validate_glmnet_convergence_is_scale_free() {
        // glmnet's criterion is in standardised-y units: y * 1e-6 and y * 1e6 take the same
        // passes. Pre-WP07: absolute |delta beta| < thresh, 3.07 relative error at 1e-6.
        let (_case, ds) = crate::test_support::load_case("glmnet_paths");
        let (x, y) = xy(&ds, "y", &X8);
        let mut c = GlmnetConfig::default();
        c.thresh = 1e-14;
        let base = glmnet(x.view(), y.view(), &c).unwrap();
        for s in [1e-6, 1e6] {
            let ys = &y * s;
            let r = glmnet(x.view(), ys.view(), &c).unwrap();
            assert_eq!(r.fits.len(), base.fits.len(), "scale {s}");
            assert!(
                r.npasses.abs_diff(base.npasses) <= 2,
                "scale {s}: {} vs {} passes",
                r.npasses,
                base.npasses
            );
            for (a, b) in coef_path(&r).iter().zip(coef_path(&base)) {
                assert!(
                    (a / s - b).abs() <= 1e-12 + 1e-6 * b.abs(),
                    "scale {s}: {} vs {b}",
                    a / s
                );
            }
        }
    }

    #[test]
    fn test_validate_glmnet_maxit_truncates_path_with_warning() {
        let (_case, ds) = crate::test_support::load_case("glmnet_paths");
        let full = run_glmnet(&ds, "y", &X8, &GlmnetConfig::default()).unwrap();
        let mut c = GlmnetConfig::default();
        c.max_iter = 60;
        let short = run_glmnet(&ds, "y", &X8, &c).unwrap();
        let k = short.fits.len();
        assert!(k >= 1 && k < full.fits.len(), "{k} fits");
        assert!(short.npasses <= 60, "{} passes", short.npasses);
        assert_eq!(
            short.warnings,
            vec![format!(
                "Convergence not reached at lambda {} of {} (lambda = {:.6e}) after max_iter = 60 \
                 coordinate passes over the path; the fits for the {k} larger lambdas are returned",
                k + 1,
                c.nlambda,
                full.lambda[k]
            )]
        );
        // same deterministic iterations until the budget ends
        for (a, b) in short.fits.iter().zip(&full.fits) {
            assert_eq!(
                (a.intercept, &a.coefficients),
                (b.intercept, &b.coefficients)
            );
        }
        // no lambda converged at all: an error (R returns an empty model)
        let mut c1 = GlmnetConfig::default();
        c1.max_iter = 1;
        c1.lambda = Some(vec![0.01]);
        assert!(matches!(
            run_glmnet(&ds, "y", &X8, &c1).unwrap_err(),
            EconError::ConvergenceFailure { .. }
        ));
    }

    #[test]
    fn test_validate_glmnet_constant_column_is_excluded() {
        // glmnet's ju: a constant column gets coefficient 0 and leaves the others unchanged.
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let (x, y) = xy(&ds, "y", &X8);
        let x9 = ndarray::concatenate![Axis(1), x, Array2::from_elem((x.nrows(), 1), 3.0)];
        let mut chk = Check::new(&case);
        for (standardize, key) in [(true, "a10_std_coef_tight"), (false, "a10_raw_coef_tight")] {
            let r = glmnet(
                x9.view(),
                y.view(),
                &tight(1.0, standardize, true, Some(&LAMS[..])),
            )
            .unwrap();
            assert!(r.fits.iter().all(|f| f.coefficients[8] == 0.0));
            let b: Vec<f64> = r
                .fits
                .iter()
                .flat_map(|f| {
                    std::iter::once(f.intercept).chain(f.coefficients[..8].iter().copied())
                })
                .collect();
            chk.vector(key, &b, EXACT);
        }
        chk.finish();
    }

    #[test]
    fn test_validate_glmnet_default_path() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let lasso = run_glmnet(&ds, "y", &X8, &tight(1.0, true, true, None)).unwrap();
        let ridge = run_glmnet(&ds, "y", &X8, &tight(0.0, true, true, None)).unwrap();
        let enet = run_glmnet(&ds, "y", &X8, &tight(0.5, true, true, None)).unwrap();
        Check::new(&case)
            .vector("path_lambda", &lasso.lambda, Tol::coef(150))
            .vector("path_df", &dfs(&lasso), Tol::exact())
            .vector("path_coef_tight", &coef_path(&lasso), EXACT)
            .vector(
                "path_dev_ratio_tight",
                &dev_ratios(&lasso),
                Tol::iterative(1e-9),
            )
            .vector("path_dev_ratio", &dev_ratios(&lasso), R_DEV)
            .scalar("ridge_path_lambda_max", ridge.lambda[0], Tol::coef(150))
            .scalar("ridge_path_length", ridge.lambda.len() as f64, Tol::exact())
            .scalar("enet_path_lambda_max", enet.lambda[0], Tol::coef(150))
            .finish();
    }

    #[test]
    fn test_validate_glmnet_path_rules() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let cols = k30();
        let (k, yk) = xy(&ds, "yk", &refs(&cols));
        // n = 20 < p = 30
        let (kw, yw) = (
            k.slice(s![..20, ..]).to_owned(),
            yk.slice(s![..20]).to_owned(),
        );
        let mut cw = GlmnetConfig::default();
        cw.thresh = 1e-20;
        let wide = glmnet(kw.view(), yw.view(), &cw).unwrap();
        let mut cd = tight(1.0, true, true, None);
        cd.dfmax = Some(2);
        let dfm = run_glmnet(&ds, "y", &X8, &cd).unwrap();
        let mut cf = tight(1.0, true, true, None);
        cf.fdev = 0.0;
        let full = run_glmnet(&ds, "y", &X8, &cf).unwrap();
        Check::new(&case)
            .vector("wide_path_lambda", &wide.lambda, Tol::coef(20))
            .vector("wide_path_df", &dfs(&wide), Tol::exact())
            .vector("dfmax2_path_df", &dfs(&dfm), Tol::exact())
            .scalar("fdev0_path_length", full.lambda.len() as f64, Tol::exact())
            .finish();
        // user lambdas: sorted decreasing (R), never early-stopped, dfmax not applied
        let mut cu = tight(1.0, true, true, Some(&[0.03, 1.0, 0.1, 0.3][..]));
        cu.dfmax = Some(0);
        let u = run_glmnet(&ds, "y", &X8, &cu).unwrap();
        assert_eq!(u.lambda, vec![1.0, 0.3, 0.1, 0.03]);
        assert_eq!(coef_path(&u).len(), 36);
    }

    #[test]
    fn test_validate_glmnet_nlambda_one_is_lambda_max() {
        // REG-42 (b): nlambda = 1 divided by nlambda - 1 = 0 (a NaN lambda).
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let mut c = GlmnetConfig::default();
        c.nlambda = 1;
        let r = run_glmnet(&ds, "y", &X8, &c).unwrap();
        let lmax = case.vector("path_lambda")[0];
        assert_eq!(r.lambda.len(), 1);
        assert!(
            Tol::coef(150).allows(r.lambda[0], lmax),
            "{} vs {lmax}",
            r.lambda[0]
        );
        assert!(
            r.warnings.iter().any(|w| w.contains("nlambda = 1")),
            "{:?}",
            r.warnings
        );
        Check::new(&case)
            .vector("nlambda1_coef", &coef_path(&r), EXACT) // mean(y), zeros
            .scalar("nlambda1_dev_ratio", r.fits[0].dev_ratio, Tol::exact())
            .finish();
        let (x, y) = xy(&ds, "y", &X8);
        let cv = cv_glmnet(x.view(), y.view(), &c, 5, Some(1)).unwrap();
        assert_eq!(
            (cv.cvm.len(), cv.lambda_min_index, cv.lambda_1se_index),
            (1, 0, 0)
        );
        assert!(cv.cvm[0].is_finite() && cv.cvsd[0].is_finite());
    }

    #[test]
    fn test_glmnet_rejects_empty_lambda() {
        // REG-42 (c): an empty user lambda returned Ok with no fit at all.
        let (x, y) = lambda_path_data();
        let mut c = GlmnetConfig::default();
        c.lambda = Some(vec![]);
        match glmnet(x.view(), y.view(), &c).unwrap_err() {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains("lambda"), "{message}")
            }
            e => panic!("{e:?}"),
        }
        assert!(matches!(
            cv_glmnet(x.view(), y.view(), &c, 2, Some(1)).unwrap_err(),
            EconError::InvalidSpecification { .. }
        ));
    }

    /// `r` is an `InvalidSpecification` whose message contains `needle`.
    fn assert_spec<T: std::fmt::Debug>(r: EconResult<T>, needle: &str) {
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains(needle), "{needle}: {message}")
            }
            other => panic!("{needle}: {other:?}"),
        }
    }

    #[test]
    fn test_glmnet_rejects_invalid_inputs() {
        // R3: every invalid input is an InvalidSpecification naming the parameter.
        let (x, y) = lambda_path_data();
        let fit = |c: &GlmnetConfig| glmnet(x.view(), y.view(), c);
        let with = |f: &dyn Fn(&mut GlmnetConfig)| {
            let mut c = GlmnetConfig::default();
            f(&mut c);
            c
        };
        for a in [1.5, -0.1, f64::NAN] {
            assert_spec(fit(&with(&|c| c.alpha = a)), "alpha");
        }
        for l in [vec![1.0, -0.1], vec![f64::NAN]] {
            assert_spec(fit(&with(&|c| c.lambda = Some(l.clone()))), "lambda");
        }
        for r in [1.0, 0.0, f64::NAN] {
            assert_spec(
                fit(&with(&|c| c.lambda_min_ratio = Some(r))),
                "lambda_min_ratio",
            );
        }
        for v in [-1.0, f64::NAN] {
            assert_spec(fit(&with(&|c| c.fdev = v)), "fdev");
        }
        for v in [0.0, 1.5] {
            assert_spec(fit(&with(&|c| c.deviance_thresh = v)), "deviance_thresh");
        }
        let c = GlmnetConfig::default();
        assert_spec(
            glmnet(x.view(), Array1::from_elem(5, 2.0).view(), &c),
            "y is constant",
        );
        let mut xn = x.clone();
        xn[[2, 1]] = f64::NAN;
        assert_spec(glmnet(xn.view(), y.view(), &c), "x contains");
        let mut yi = y.clone();
        yi[3] = f64::INFINITY;
        assert_spec(glmnet(x.view(), yi.view(), &c), "y contains");
        assert_spec(
            glmnet(Array2::<f64>::zeros((5, 0)).view(), y.view(), &c),
            "at least one column",
        );
        assert_spec(glmnet(x.view(), y.slice(s![..4]), &c), "rows");
        assert_spec(cv_glmnet(x.view(), y.view(), &c, 1, Some(1)), "nfolds");
        let mut cb = GlmnetConfig::default();
        cb.family = GlmnetFamily::Binomial;
        assert_spec(
            glmnet(x.view(), array![0.0, 1.0, 2.0, 0.0, 1.0].view(), &cb),
            "More than two classes",
        );
        assert_spec(
            glmnet(x.view(), array![1.0, 0.0, 0.0, 0.0, 0.0].view(), &cb),
            "1 or 0 observations",
        );
        for f in [
            &[1, 1, 2, 2][..],
            &[0, 1, 1, 2, 2],
            &[1, 1, 1, 1, 1],
            &[1, 1, 3, 3, 3],
        ] {
            assert_spec(cv_glmnet_foldid(x.view(), y.view(), &c, f), "foldid");
        }
    }

    #[test]
    fn test_validate_glmnet_binomial() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let (x, y) = xy(&ds, "yb", &X8);
        let mut chk = Check::new(&case);
        for (tag, alpha, standardize, intercept) in [
            ("binom", 1.0, true, true),
            ("binom_raw", 1.0, false, true),
            ("binom_noint", 1.0, true, false),
            ("binom_a5", 0.5, true, true),
        ] {
            let mut c = tight(alpha, standardize, intercept, Some(&BLAMS[..]));
            c.family = GlmnetFamily::Binomial;
            let r = run_glmnet(&ds, "yb", &X8, &c).unwrap();
            let (b, d) = (coef_path(&r), dev_ratios(&r));
            chk.vector(&format!("{tag}_coef_tight"), &b, EXACT)
                .vector(&format!("{tag}_coef"), &b, R_GLMNET)
                .vector(&format!("{tag}_dev_ratio_tight"), &d, Tol::iterative(1e-9))
                .vector(&format!("{tag}_dev_ratio"), &d, R_DEV);
            if intercept {
                // REG-6: the intercept score sum(y - p) is zero at convergence
                for f in &r.fits {
                    let eta = x.dot(&Array1::from_vec(f.coefficients.clone())) + f.intercept;
                    let score: f64 = y
                        .iter()
                        .zip(eta.iter())
                        .map(|(yi, e)| yi - 1.0 / (1.0 + (-e).exp()))
                        .sum();
                    assert!(
                        score.abs() < 1e-9,
                        "{tag} lambda {}: sum(y - p) = {score}",
                        f.lambda
                    );
                }
            }
        }
        chk.finish();
    }

    #[test]
    fn test_validate_glmnet_binomial_path() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let mut c = tight(1.0, true, true, None);
        c.family = GlmnetFamily::Binomial;
        let r = run_glmnet(&ds, "yb", &X8, &c).unwrap();
        let (x, y) = xy(&ds, "yb", &X8);
        let mut cn = c.clone();
        cn.intercept = false;
        let sub = glmnet(x.slice(s![..120, ..]), y.slice(s![..120]), &cn).unwrap(); // ybar = 0.45
        Check::new(&case)
            .vector("binom_path_lambda", &r.lambda, Tol::coef(150))
            .vector("binom_path_df", &dfs(&r), Tol::exact())
            .vector("binom_path_coef_tight", &coef_path(&r), EXACT)
            .vector(
                "binom_path_dev_ratio_tight",
                &dev_ratios(&r),
                Tol::iterative(1e-9),
            )
            .vector("binom_path_dev_ratio", &dev_ratios(&r), R_DEV)
            .scalar("binom_noint_sub_lambda_max", sub.lambda[0], Tol::coef(120))
            .finish();
    }

    #[test]
    fn test_validate_glmnet_binomial_two_level_coding() {
        // R codes the response by its two factor levels.
        let (_case, ds) = crate::test_support::load_case("glmnet_paths");
        let (x, y) = xy(&ds, "yb", &X8);
        let mut c = tight(1.0, true, true, Some(&BLAMS[..]));
        c.family = GlmnetFamily::Binomial;
        let a = glmnet(x.view(), y.view(), &c).unwrap();
        let pm = y.mapv(|v| 2.0 * v - 1.0); // {-1, 1}: identical fit in R
        let b = glmnet(x.view(), pm.view(), &c).unwrap();
        assert_eq!(coef_path(&a), coef_path(&b));
        let few = Array1::from_iter((0..150).map(|i| if i < 5 { 1.0 } else { 0.0 }));
        let w = glmnet(x.view(), few.view(), &c).unwrap();
        assert!(
            w.warnings
                .iter()
                .any(|m| m.contains("fewer than 8 observations")),
            "{:?}",
            w.warnings
        );
    }

    #[test]
    fn test_validate_cv_glmnet_foldid() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let (x, y) = xy(&ds, "y", &X8);
        let cv = cv_glmnet_foldid(
            x.view(),
            y.view(),
            &tight(1.0, true, true, None),
            &fold_ids(&ds, "foldid"),
        )
        .unwrap();
        let (xb, yb) = xy(&ds, "yb", &X8);
        let mut cb = tight(1.0, true, true, None);
        cb.family = GlmnetFamily::Binomial;
        let cvb = cv_glmnet_foldid(xb.view(), yb.view(), &cb, &fold_ids(&ds, "foldid7")).unwrap();
        let idx = |c: &GlmnetCvResult| {
            vec![
                (c.lambda_min_index + 1) as f64,
                (c.lambda_1se_index + 1) as f64,
            ]
        };
        let lens = |c: &GlmnetCvResult| {
            c.fold_path_lengths
                .iter()
                .map(|&l| l as f64)
                .collect::<Vec<_>>()
        };
        Check::new(&case)
            .vector("path_lambda", &cv.lambda, Tol::coef(150))
            .vector("cv_cvm_tight", &cv.cvm, Tol::iterative(1e-9))
            .vector("cv_cvsd_tight", &cv.cvsd, Tol::iterative(1e-9))
            .vector("cv_cvm", &cv.cvm, Tol::new(1e-7, 0.0)) // cv.glmnet's folds stop at thresh 1e-14: 6.7e-9 rel
            .vector("cv_cvsd", &cv.cvsd, Tol::new(1e-6, 0.0)) // 3.04e-8 rel
            .vector("cv_index", &idx(&cv), Tol::exact())
            .vector("cv_fold_path_length", &lens(&cv), Tol::exact())
            .scalar("cv_lambda_min", cv.lambda_min, Tol::coef(150))
            .scalar("cv_lambda_1se", cv.lambda_1se, Tol::coef(150))
            .vector("cv_coef_min", &coef_vec(cv.fit_min()), R_GLMNET)
            .vector("cv_coef_1se", &coef_vec(cv.fit_1se()), R_GLMNET)
            .vector("cvb_lambda", &cvb.lambda, Tol::coef(150))
            .vector("cvb_cvm_tight", &cvb.cvm, Tol::iterative(1e-9))
            .vector("cvb_cvsd_tight", &cvb.cvsd, Tol::iterative(1e-9))
            .vector("cvb_cvm", &cvb.cvm, Tol::new(1e-7, 0.0))
            .vector("cvb_cvsd", &cvb.cvsd, Tol::new(1e-6, 0.0))
            .vector("cvb_index", &idx(&cvb), Tol::exact())
            .vector("cvb_fold_path_length", &lens(&cvb), Tol::exact())
            .finish();
        // the full-data fit's coefficients at lambda.min / lambda.1se are path columns 38 / 22
        let pt = case.vector("path_coef_tight");
        for (f, col) in [(cv.fit_min(), 37), (cv.fit_1se(), 21)] {
            for (a, e) in coef_vec(f).iter().zip(&pt[col * 9..col * 9 + 9]) {
                assert!(EXACT.allows(*a, *e), "{a} vs {e}");
            }
        }
    }

    #[test]
    fn test_cv_glmnet_exact_fit_has_finite_cvsd() {
        // REG-42 (d); degenerate input on purpose (no noise). Pre-WP07 every fold ran the
        // full-data lambdas and stopped at dev.ratio > 0.999 at its own index, so lambdas past a
        // fold's stop had one fold error (cvsd 0/0 = NaN) or none (`errors.len() - 1` underflow).
        let x = Array2::from_shape_fn((20, 2), |(i, j)| {
            if j == 0 {
                (i + 1) as f64
            } else {
                ((7 * (i + 1)) % 11) as f64
            }
        });
        let y = x.column(0).mapv(|v| 3.0 + 2.0 * v);
        for seed in 0..20u64 {
            let cv =
                cv_glmnet(x.view(), y.view(), &GlmnetConfig::default(), 2, Some(seed)).unwrap();
            assert_eq!(cv.cvm.len(), cv.lambda.len());
            assert!(
                cv.cvm.iter().chain(&cv.cvsd).all(|v| v.is_finite()),
                "seed {seed}: {:?}",
                cv.cvsd
            );
        }
    }

    #[test]
    fn test_validate_glmnet_predict_interpolates() {
        let (case, ds) = crate::test_support::load_case("glmnet_paths");
        let fit = run_glmnet(&ds, "y", &X8, &tight(1.0, true, true, None)).unwrap();
        let mut got = Vec::new();
        for s in case.vector("interp_s") {
            // 2.5 (above lambda_max), 0.5, 0.05, 1e-4 (below the path)
            let (b0, nz) = glmnet_coef(&fit, s).unwrap();
            got.push(b0);
            got.extend(
                X8.iter()
                    .map(|n| nz.iter().find(|(m, _)| m == n).map_or(0.0, |(_, c)| *c)),
            );
        }
        let (x, _) = xy(&ds, "y", &X8);
        let pred = glmnet_predict(&fit, x.slice(s![..3, ..]), 0.5).unwrap();
        let mut cb = tight(1.0, true, true, None);
        cb.family = GlmnetFamily::Binomial;
        let fb = run_glmnet(&ds, "yb", &X8, &cb).unwrap();
        let prob = glmnet_predict(&fb, x.slice(s![..3, ..]), 0.05).unwrap();
        Check::new(&case)
            .vector("interp_coef_tight", &got, EXACT)
            .vector("interp_coef", &got, R_GLMNET)
            .vector(
                "interp_pred",
                pred.as_slice().unwrap(),
                Tol::new(1e-6, 1e-6),
            )
            .vector("binom_interp_prob_tight", prob.as_slice().unwrap(), EXACT)
            .vector(
                "binom_interp_prob",
                prob.as_slice().unwrap(),
                Tol::new(1e-6, 1e-7),
            )
            .finish();
        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            // R5: a NaN lambda was a partial_cmp(..).unwrap() panic
            assert!(matches!(
                glmnet_predict(&fit, x.view(), bad).unwrap_err(),
                EconError::InvalidSpecification { .. }
            ));
            assert!(matches!(
                glmnet_coef(&fit, bad).unwrap_err(),
                EconError::InvalidSpecification { .. }
            ));
        }
        assert!(matches!(
            glmnet_predict(&fit, x.slice(s![.., ..3]), 0.5).unwrap_err(),
            EconError::InvalidSpecification { .. }
        ));
    }

    #[test]
    fn test_validate_glmnet_display_prints_coefficients() {
        let (_case, ds) = crate::test_support::load_case("glmnet_paths");
        let r = run_glmnet(&ds, "y", &X8, &tight(1.0, true, true, Some(&LAMS[..]))).unwrap();
        let text = r.to_string();
        for f in &r.fits {
            assert!(
                text.contains(&format!("(Intercept) {:.6}", f.intercept)),
                "{text}"
            );
            for (name, c) in X8.iter().zip(&f.coefficients) {
                if *c != 0.0 {
                    assert!(
                        text.contains(&format!("{name} {c:.6}")),
                        "{name} at lambda {}: {text}",
                        f.lambda
                    );
                }
            }
        }
        // one line per fit
        assert_eq!(
            text.lines()
                .filter(|l| l.trim_start().starts_with('['))
                .count(),
            4
        );
        // p > 20: the 20 largest |coef| and a count of the rest (30 nonzero at this lambda)
        let cols = k30();
        let k = run_glmnet(
            &ds,
            "yk",
            &refs(&cols),
            &tight(1.0, true, true, Some(&[0.0009602538655657498][..])),
        )
        .unwrap();
        assert!(k.to_string().contains("(+10 more nonzero)"), "{k}");
        let (x, y) = xy(&ds, "y", &X8);
        let mut cv = cv_glmnet_foldid(
            x.view(),
            y.view(),
            &tight(1.0, true, true, None),
            &fold_ids(&ds, "foldid"),
        )
        .unwrap();
        // cv_glmnet_foldid on arrays names the columns V1..V8, as glmnet does
        cv.fit.variable_names = X8.iter().map(|n| n.to_string()).collect();
        let ct = cv.to_string();
        assert!(
            ct.contains("Coefficients at lambda.min (nonzero; intercept first):"),
            "{ct}"
        );
        assert!(
            ct.contains("Coefficients at lambda.1se (nonzero; intercept first):"),
            "{ct}"
        );
        assert!(
            ct.contains(&format!("X1 {:.6}", cv.fit_min().coefficients[0])),
            "{ct}"
        );
        assert!(
            ct.contains(&format!("X1 {:.6}", cv.fit_1se().coefficients[0])),
            "{ct}"
        );
        let mut c = GlmnetConfig::default();
        c.max_iter = 60;
        assert!(
            run_glmnet(&ds, "y", &X8, &c)
                .unwrap()
                .to_string()
                .contains("\nWarning: Convergence not reached at lambda ")
        );
    }

    #[test]
    fn test_glmnetfamily_parses_aliases_and_rejects_unknown() {
        for g in ["gaussian", "Gaussian", " GAUSSIAN ", "linear", "normal"] {
            assert_eq!(
                g.parse::<GlmnetFamily>().unwrap(),
                GlmnetFamily::Gaussian,
                "{g}"
            );
        }
        for b in ["binomial", "logistic", "Logit"] {
            assert_eq!(
                b.parse::<GlmnetFamily>().unwrap(),
                GlmnetFamily::Binomial,
                "{b}"
            );
        }
        assert_spec(
            "poisson".parse::<GlmnetFamily>(),
            "unknown family 'poisson'; expected one of: gaussian, binomial",
        );
        assert_eq!(GlmnetFamily::Binomial.to_string(), "binomial");
        assert_eq!(GlmnetFamily::Gaussian.to_string(), "gaussian");
    }
}
