//! Regularized linear regression with elastic net penalty (glmnet).
//!
//! Implements the elastic net estimator combining L1 (lasso) and L2 (ridge) penalties:
//!
//! min_β (1/2n) ||y - Xβ||² + λ[(1-α)||β||₂²/2 + α||β||₁]
//!
//! where:
//! - α = 1 gives the lasso
//! - α = 0 gives ridge regression
//! - 0 < α < 1 gives elastic net
//!
//! # Algorithm
//!
//! Uses cyclical coordinate descent with warm starts and active set convergence.
//! The coordinate update for coefficient j is:
//!
//! β̃_j = S(∂L/∂β_j, λα) / (1 + λ(1-α))
//!
//! where S is the soft-thresholding operator: S(z, γ) = sign(z)(|z| - γ)₊
//!
//! # Features
//!
//! - Pathwise solution along sequence of λ values
//! - Cross-validation for λ selection
//! - Supports Gaussian and binomial families
//! - Standardization of predictors (optional)
//!
//! # References
//!
//! - Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
//!   generalized linear models via coordinate descent. *Journal of Statistical Software*,
//!   33(1), 1-22. https://doi.org/10.18637/jss.v033.i01
//!
//! - Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic
//!   net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
//!   https://doi.org/10.1111/j.1467-9868.2005.00503.x
//!
//! R equivalent: `glmnet::glmnet()`, `glmnet::cv.glmnet()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, ShapeBuilder, s};
use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
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

/// Configuration for glmnet fitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmnetConfig {
    /// Elastic net mixing parameter (0 = ridge, 1 = lasso)
    pub alpha: f64,
    /// Sequence of lambda values (if None, automatically generated)
    pub lambda: Option<Vec<f64>>,
    /// Number of lambda values in path (default: 100). At least 1 when `lambda` is `None`
    /// (otherwise `InvalidSpecification`); a path above the allocation ceiling is `ResourceLimit`.
    pub nlambda: usize,
    /// Minimum lambda ratio (lambda_min / lambda_max)
    pub lambda_min_ratio: f64,
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
    /// Maximum number of non-zero coefficients
    pub dfmax: Option<usize>,
    /// Minimum deviance ratio for early stopping
    pub deviance_thresh: f64,
}

impl Default for GlmnetConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0, // lasso by default
            lambda: None,
            nlambda: 100,
            lambda_min_ratio: 0.0001,
            standardize: true,
            intercept: true,
            max_iter: 100_000,
            thresh: 1e-7,
            family: GlmnetFamily::Gaussian,
            dfmax: None,
            deviance_thresh: 0.999,
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
}

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
        xv.push(zj.iter().map(|v| v * v).sum::<f64>() / nf);
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

/// `ConvergenceFailure` when not even the first lambda converged.
fn first_lambda_failure(max_iter: usize, last_change: f64) -> EconError {
    EconError::ConvergenceFailure {
        iterations: max_iter,
        last_change,
        suggestion: "increase max_iter or thresh".to_string(),
    }
}

/// Coordinate descent for Binomial family (logistic regression)
fn coordinate_descent_binomial(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    lambda: f64,
    alpha: f64,
    beta: &mut Array1<f64>,
    intercept: &mut f64,
    max_iter: usize,
    thresh: f64,
) -> EconResult<usize> {
    let n = x.nrows() as f64;
    let p = x.ncols();

    for iter in 0..max_iter {
        resource::checkpoint()?;
        let mut max_change = 0.0f64;

        // Compute probabilities: p = 1 / (1 + exp(-eta))
        let eta: Array1<f64> = x.dot(beta) + *intercept;
        let prob: Array1<f64> = eta.mapv(|e| 1.0 / (1.0 + (-e).exp()));

        // Working weights: w = p * (1 - p)
        let w: Array1<f64> = prob.mapv(|p| (p * (1.0 - p)).max(1e-5));

        // Working response: z = eta + (y - p) / w
        let z: Array1<f64> = &eta + &((&y.to_owned() - &prob) / &w);

        // Update intercept (unpenalized)
        let old_intercept = *intercept;
        let w_sum: f64 = w.sum();
        let w_z_sum: f64 = (&w * &z).sum();
        *intercept = w_z_sum / w_sum;
        max_change = max_change.max((*intercept - old_intercept).abs());

        // Update each coefficient
        for j in 0..p {
            let old_beta = beta[j];

            // Weighted partial residual
            let r: Array1<f64> = &z - &x.dot(beta) - *intercept + &(&x.column(j) * old_beta);

            // Weighted gradient
            let xj = x.column(j);
            let w_xj_r: f64 = (&w * &xj.to_owned() * &r).sum();
            let w_xj_sq: f64 = (&w * &xj.to_owned() * &xj.to_owned()).sum();

            // Coordinate update
            let z_val = soft_threshold(w_xj_r / n, lambda * alpha);
            let new_beta = z_val / (w_xj_sq / n + lambda * (1.0 - alpha));

            beta[j] = new_beta;
            max_change = max_change.max((new_beta - old_beta).abs());
        }

        if max_change < thresh {
            return Ok(iter + 1);
        }
    }

    Ok(max_iter)
}

/// Compute deviance for Binomial family
fn deviance_binomial(y: &ArrayView1<f64>, prob: &Array1<f64>) -> f64 {
    let mut dev = 0.0;
    for i in 0..y.len() {
        let p = prob[i].max(1e-10).min(1.0 - 1e-10);
        dev -= 2.0 * (y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln());
    }
    dev
}

/// Check the path and iteration controls of a glmnet configuration.
///
/// `nlambda = 1` (which generates a NaN lambda) and an empty user lambda
/// sequence are not handled here (REG-42).
fn validate_glmnet_config(config: &GlmnetConfig) -> EconResult<()> {
    if !(config.thresh.is_finite() && config.thresh > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "thresh must be a positive finite number (got {})",
                config.thresh
            ),
        });
    }
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1".to_string(),
        });
    }
    if config.lambda.is_none() && config.nlambda == 0 {
        return Err(EconError::InvalidSpecification {
            message: "nlambda must be at least 1 when no lambda sequence is given".to_string(),
        });
    }
    Ok(())
}

/// The fits of one path, the coordinate passes spent and the notes.
struct PathFit {
    fits: Vec<GlmnetCoefficients>,
    npasses: usize,
    warnings: Vec<String>,
}

/// Whether a path stops after the fit just pushed (pre-WP07 rule, replaced in Task 3).
fn path_stops(fit: &GlmnetCoefficients, config: &GlmnetConfig) -> bool {
    fit.dev_ratio > config.deviance_thresh || config.dfmax.is_some_and(|d| fit.df >= d)
}

/// Gaussian path: glmnet's `elnet` on `yt = (y - ym) / ys`, with the penalty in those units
/// (`l1 = lambda alpha / ys`, `l2 = lambda (1 - alpha) / ys`), warm starts along the path.
fn gaussian_path(
    design: &Design,
    yc: &[f64],
    ym: f64,
    ys: f64,
    lambdas: &[f64],
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
        if fits.last().is_some_and(|f| path_stops(f, config)) {
            break;
        }
    }
    Ok(PathFit {
        fits,
        npasses: budget.used,
        warnings,
    })
}

/// Binomial path with the pre-WP07 loop (replaced in Task 4).
fn binomial_path(
    design: &Design,
    y: ArrayView1<f64>,
    lambdas: &[f64],
    config: &GlmnetConfig,
    null_deviance: f64,
) -> EconResult<PathFit> {
    let (n, p) = (design.n, design.p);
    let z = ArrayView2::from_shape((n, p).f(), &design.z)
        .map_err(|e| EconError::Computation(format!("glmnet design: {e}")))?;
    let mut beta = Array1::zeros(p);
    let mut b0 = 0.0;
    let mut fits = resource::try_with_capacity(lambdas.len(), "glmnet coefficient path")?;
    for &lambda in lambdas {
        resource::checkpoint()?;
        coordinate_descent_binomial(
            &z,
            &y,
            lambda,
            config.alpha,
            &mut beta,
            &mut b0,
            config.max_iter,
            config.thresh,
        )?;
        let prob = (z.dot(&beta) + b0).mapv(|e| 1.0 / (1.0 + (-e).exp()));
        let dev_ratio = 1.0 - deviance_binomial(&y, &prob) / null_deviance;
        let coefficients: Vec<f64> = (0..p).map(|j| beta[j] / design.scale[j]).collect();
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
        if fits.last().is_some_and(|f| path_stops(f, config)) {
            break;
        }
    }
    Ok(PathFit {
        fits,
        npasses: 0,
        warnings: Vec::new(),
    })
}

/// Fit glmnet model
pub fn glmnet(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &GlmnetConfig,
) -> EconResult<GlmnetResult> {
    validate_glmnet_config(config)?;
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(EconError::Computation(format!(
            "X has {} rows but y has {} elements",
            n,
            y.len()
        )));
    }

    if config.alpha < 0.0 || config.alpha > 1.0 {
        return Err(EconError::Computation(
            "alpha must be between 0 and 1".to_string(),
        ));
    }

    let design = standardize_design(x, config.standardize, config.intercept)?;
    let nf = n as f64;
    let y_mean = y.sum() / nf;
    let ym = if config.intercept { y_mean } else { 0.0 };
    let yc: Vec<f64> = y.iter().map(|v| v - ym).collect();

    let null_deviance = match config.family {
        GlmnetFamily::Gaussian => yc.iter().map(|v| v * v).sum(),
        GlmnetFamily::Binomial => {
            let prob_null = Array1::from_elem(n, y_mean);
            deviance_binomial(&y, &prob_null)
        }
    };

    // Generate lambda sequence
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
    let lambda_max = if config.alpha > 0.0 {
        grad_max / config.alpha.max(1e-3)
    } else {
        grad_max * 100.0
    };
    let lambda_min = lambda_max * config.lambda_min_ratio;

    let lambda_seq: Vec<f64> = if let Some(ref user_lambda) = config.lambda {
        user_lambda.clone()
    } else {
        let log_max = lambda_max.ln();
        let log_min = lambda_min.ln();
        let mut seq = resource::try_with_capacity::<f64>(config.nlambda, "glmnet lambda path")?;
        seq.extend((0..config.nlambda).map(|i| {
            let t = i as f64 / (config.nlambda - 1) as f64;
            (log_max + t * (log_min - log_max)).exp()
        }));
        seq
    };

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
            gaussian_path(&design, &yc, ym, ys, &lambda_seq, config)?
        }
        GlmnetFamily::Binomial => binomial_path(&design, y, &lambda_seq, config, null_deviance)?,
    };

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
        warnings: path.warnings,
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

/// Cross-validation for glmnet
pub fn cv_glmnet(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &GlmnetConfig,
    nfolds: usize,
    seed: Option<u64>,
) -> EconResult<GlmnetCvResult> {
    let n = x.nrows();

    if nfolds < 2 {
        return Err(EconError::Computation(
            "nfolds must be at least 2".to_string(),
        ));
    }
    if nfolds > n {
        return Err(EconError::Computation(format!(
            "nfolds ({}) cannot exceed number of observations ({})",
            nfolds, n
        )));
    }

    // First fit on full data to get lambda sequence
    let full_fit = glmnet(x, y, config)?;
    let lambda_seq = full_fit.lambda.clone();
    let nlambda = lambda_seq.len();

    // Create fold assignments
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut fold_ids: Vec<usize> = (0..n).map(|i| i % nfolds).collect();
    fold_ids.shuffle(&mut rng);

    // Cross-validation errors
    let mut cv_errors: Vec<Vec<f64>> = vec![Vec::with_capacity(nlambda); nfolds];

    for fold in 0..nfolds {
        resource::checkpoint()?;
        // Create train/test split
        let train_idx: Vec<usize> = (0..n).filter(|&i| fold_ids[i] != fold).collect();
        let test_idx: Vec<usize> = (0..n).filter(|&i| fold_ids[i] == fold).collect();

        let n_train = train_idx.len();
        let n_test = test_idx.len();

        let mut x_train = Array2::zeros((n_train, x.ncols()));
        let mut y_train = Array1::zeros(n_train);
        let mut x_test = Array2::zeros((n_test, x.ncols()));
        let mut y_test = Array1::zeros(n_test);

        for (new_i, &orig_i) in train_idx.iter().enumerate() {
            x_train.row_mut(new_i).assign(&x.row(orig_i));
            y_train[new_i] = y[orig_i];
        }
        for (new_i, &orig_i) in test_idx.iter().enumerate() {
            x_test.row_mut(new_i).assign(&x.row(orig_i));
            y_test[new_i] = y[orig_i];
        }

        // Fit on training data with fixed lambda sequence
        let mut fold_config = config.clone();
        fold_config.lambda = Some(lambda_seq.clone());
        let fold_fit = glmnet(x_train.view(), y_train.view(), &fold_config)?;

        // Compute test error for each lambda
        for fit in &fold_fit.fits {
            let coefs = Array1::from_vec(fit.coefficients.clone());
            let fitted = x_test.dot(&coefs) + fit.intercept;

            let error = match config.family {
                GlmnetFamily::Gaussian => {
                    let mse: f64 = (&y_test - &fitted).mapv(|r| r * r).sum() / n_test as f64;
                    mse
                }
                GlmnetFamily::Binomial => {
                    let prob = fitted.mapv(|e| 1.0 / (1.0 + (-e).exp()));
                    deviance_binomial(&y_test.view(), &prob) / n_test as f64
                }
            };
            cv_errors[fold].push(error);
        }
    }

    // Compute mean and standard error across folds
    let cvm: Vec<f64> = (0..nlambda)
        .map(|l| {
            let errors: Vec<f64> = cv_errors.iter().filter_map(|f| f.get(l).copied()).collect();
            errors.iter().sum::<f64>() / errors.len() as f64
        })
        .collect();

    let cvsd: Vec<f64> = (0..nlambda)
        .map(|l| {
            let errors: Vec<f64> = cv_errors.iter().filter_map(|f| f.get(l).copied()).collect();
            let mean = cvm[l];
            let var: f64 =
                errors.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / (errors.len() - 1) as f64;
            (var / errors.len() as f64).sqrt()
        })
        .collect();

    let cvup: Vec<f64> = cvm.iter().zip(&cvsd).map(|(&m, &s)| m + s).collect();
    let cvlo: Vec<f64> = cvm.iter().zip(&cvsd).map(|(&m, &s)| m - s).collect();

    // Find lambda.min and lambda.1se
    let (lambda_min_index, &lambda_min_cvm) = cvm
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    let lambda_min = lambda_seq[lambda_min_index];

    // lambda.1se: largest lambda within 1 SE of minimum
    let threshold = lambda_min_cvm + cvsd[lambda_min_index];
    let lambda_1se_index = (0..=lambda_min_index)
        .find(|&i| cvm[i] <= threshold)
        .unwrap_or(lambda_min_index);
    let lambda_1se = lambda_seq[lambda_1se_index];

    Ok(GlmnetCvResult {
        fit: full_fit,
        lambda: lambda_seq,
        cvm,
        cvsd,
        cvup,
        cvlo,
        lambda_min,
        lambda_min_index,
        lambda_1se,
        lambda_1se_index,
        nfolds,
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

/// Predict from glmnet result at a specific lambda
pub fn glmnet_predict(
    result: &GlmnetResult,
    x: ArrayView2<f64>,
    lambda: f64,
) -> EconResult<Array1<f64>> {
    // Find the closest lambda
    let idx = result
        .lambda
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - lambda).abs())
                .partial_cmp(&((*b - lambda).abs()))
                .unwrap()
        })
        .map(|(i, _)| i)
        .ok_or_else(|| EconError::Computation("Empty lambda sequence".to_string()))?;

    let fit = &result.fits[idx];
    let coefs = Array1::from_vec(fit.coefficients.clone());

    let eta = x.dot(&coefs) + fit.intercept;

    match result.family {
        GlmnetFamily::Gaussian => Ok(eta),
        GlmnetFamily::Binomial => Ok(eta.mapv(|e| 1.0 / (1.0 + (-e).exp()))),
    }
}

/// Get coefficients at a specific lambda
pub fn glmnet_coef(result: &GlmnetResult, lambda: f64) -> EconResult<(f64, Vec<(String, f64)>)> {
    let idx = result
        .lambda
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - lambda).abs())
                .partial_cmp(&((*b - lambda).abs()))
                .unwrap()
        })
        .map(|(i, _)| i)
        .ok_or_else(|| EconError::Computation("Empty lambda sequence".to_string()))?;

    let fit = &result.fits[idx];
    let coefs: Vec<(String, f64)> = result
        .variable_names
        .iter()
        .zip(&fit.coefficients)
        .filter(|(_, c)| c.abs() > 1e-10)
        .map(|(name, &c)| (name.clone(), c))
        .collect();

    Ok((fit.intercept, coefs))
}

impl std::fmt::Display for GlmnetResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Glmnet Result: {} ~ {} variables",
            self.dependent_var, self.n_features
        )?;
        writeln!(f, "  Family:     {:?}", self.family)?;
        writeln!(
            f,
            "  Alpha:      {:.2} ({})",
            self.alpha,
            if self.alpha == 1.0 {
                "lasso"
            } else if self.alpha == 0.0 {
                "ridge"
            } else {
                "elastic net"
            }
        )?;
        writeln!(f, "  N:          {}", self.n_obs)?;
        writeln!(f, "  Lambda path: {} values", self.lambda.len())?;
        writeln!(f)?;
        writeln!(f, "  Lambda      Df    %Dev")?;
        for fit in self.fits.iter().take(10) {
            writeln!(
                f,
                "  {:.4e}    {:3}   {:.3}",
                fit.lambda, fit.df, fit.dev_ratio
            )?;
        }
        if self.fits.len() > 10 {
            writeln!(f, "  ... ({} more)", self.fits.len() - 10)?;
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
            self.lambda_min, self.lambda_min_index
        )?;
        writeln!(
            f,
            "  lambda.1se:  {:.4e} (index {})",
            self.lambda_1se, self.lambda_1se_index
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  At lambda.min: {} non-zero coefs, {:.1}% dev explained",
            self.fit.fits[self.lambda_min_index].df,
            self.fit.fits[self.lambda_min_index].dev_ratio * 100.0
        )?;
        writeln!(
            f,
            "  At lambda.1se: {} non-zero coefs, {:.1}% dev explained",
            self.fit.fits[self.lambda_1se_index].df,
            self.fit.fits[self.lambda_1se_index].dev_ratio * 100.0
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use approx::assert_relative_eq;
    use ndarray::array;

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
    fn test_glmnet_kkt_excluded_variable() {
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
    fn test_glmnet_convergence_is_scale_free() {
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
    fn test_glmnet_maxit_truncates_path_with_warning() {
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
    fn test_glmnet_constant_column_is_excluded() {
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
}
