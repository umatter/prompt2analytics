//! Finite mixtures: normal mixtures by EM ([`normal_mix_em`], [`mvnorm_mix_em`]; R `mixtools`) and mixtures of regressions ([`flexmix`]; R `flexmix`).
//!
//! References and differences from R: `validation/ml/mixtures.md`.

use super::gmm::{self, Component, CovarianceType};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::xtx;
use crate::linalg::solve::weighted_least_squares;
use crate::regression::ols::normal_inverse_from_gram;
use crate::resource::{checkpoint, try_zeros2};
use crate::traits::convergence::{Convergence, ResultNotes, deviance_change, notes_text};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::prelude::*;
use rand_distr::{Exp1, Normal};
use serde::{Deserialize, Serialize};

/// Seed of the random restarts and of mvnorm's start when `seed` is `None`.
pub const MIXTURE_DEFAULT_SEED: u64 = 42;
/// `normalmixEM`'s `maxrestarts`.
pub const MIXTOOLS_MAX_RESTARTS: usize = 20;
/// `mvnormalmixEM`'s limit on restarts.
pub const MVMIXTOOLS_MAX_RESTARTS: usize = 15;
/// A component sd below this times sd(x) counts as a collapse (mixtools: sigma < 1e-8, absolute).
pub const MIXTOOLS_SIGMA_COLLAPSE_REL: f64 = 1e-8;
/// Seed of flexmix's random start when `seed` is `None`.
pub const FLEXMIX_DEFAULT_SEED: u64 = 42;
/// flexmix's `FLXcontrol` default for `minprior`.
pub const FLEXMIX_DEFAULT_MINPRIOR: f64 = 0.05;
/// A component residual sd below this times sd(y) is floored there and flagged.
pub const FLEXMIX_SIGMA_FLOOR_REL: f64 = 1e-6;
/// The mixture weight below which `mvnormalmixEM` restarts.
const MV_WEIGHT_COLLAPSE: f64 = 1e-8;
/// `ln(2 pi) / 2`.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// The shared checks of the EM fits.
fn check_em_limits(k: usize, n: usize, max_iterations: usize, tolerance: f64) -> EconResult<()> {
    if k == 0 || k > n {
        return Err(spec(format!(
            "k must be between 1 and {n} (the number of observations), got {k}"
        )));
    }
    if max_iterations == 0 {
        return Err(spec("max_iterations must be at least 1, got 0".into()));
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(spec(format!(
            "tolerance must be finite and positive, got {tolerance}"
        )));
    }
    Ok(())
}

/// Positive proportions summing to 1 (within 1e-8), renormalised.
fn checked_lambda(lambda: &[f64], k: usize) -> EconResult<Vec<f64>> {
    if lambda.len() != k {
        return Err(spec(format!(
            "start lambda must have {k} values, got {}",
            lambda.len()
        )));
    }
    if lambda.iter().any(|l| !l.is_finite() || *l <= 0.0) {
        return Err(spec("start lambda must be finite and positive".into()));
    }
    let total: f64 = lambda.iter().sum();
    if (total - 1.0).abs() > 1e-8 {
        return Err(spec(format!("start lambda must sum to 1, got {total}")));
    }
    Ok(lambda.iter().map(|l| l / total).collect())
}

/// First maximum of each row.
fn hard_labels(posterior: &Array2<f64>) -> Vec<usize> {
    posterior
        .rows()
        .into_iter()
        .map(|row| {
            let mut best = 0;
            for (j, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = j;
                }
            }
            best
        })
        .collect()
}

// =============================================================================
// FlexMix - Finite Mixture Regression
// =============================================================================

/// One removal by flexmix's `minprior` rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlexMixRemoval {
    /// The iteration at whose start the component was removed (1-based).
    pub iteration: usize,
    /// The component, 0-based, in the initial numbering.
    pub component: usize,
    /// Its prior at that point, below `minprior`.
    pub prior: f64,
}

/// Result of FlexMix finite mixture regression.
///
/// # References
///
/// - Leisch, F. (2004). "FlexMix: A General Framework for Finite Mixture Models
///   and Latent Class Regression in R". Journal of Statistical Software, 11(8).
/// - R flexmix package documentation
///   Source: <https://www.rdocumentation.org/packages/flexmix/versions/2.3-20/topics/flexmix>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexMixResult {
    /// Component membership (posterior) probabilities for each observation
    #[serde(skip)]
    pub posterior: Array2<f64>,
    /// Hard cluster assignments (0-indexed)
    pub cluster: Vec<usize>,
    /// Regression coefficients for each component
    pub coefficients: Vec<Vec<f64>>,
    /// Residual standard deviations for each component (flexmix's formula)
    pub sigma: Vec<f64>,
    /// Component weights (mixing proportions)
    pub prior: Vec<f64>,
    /// Log-likelihood
    pub loglik: f64,
    /// BIC (`-2 logLik + df ln n`)
    pub bic: f64,
    /// AIC (`-2 logLik + 2 df`)
    pub aic: f64,
    /// Number of iterations
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components left after `minprior` removals
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Number of predictors (including intercept)
    pub p: usize,
    /// Number of components at the start
    #[serde(default)]
    pub k_initial: usize,
    /// The `minprior` the fit ran with
    #[serde(default)]
    pub minprior: f64,
    /// Number of free parameters: `(k - 1) + k (p + 1)` for the final `k`
    #[serde(default)]
    pub n_parameters: usize,
    /// Every removal by the `minprior` rule, in order
    #[serde(default)]
    pub removed_components: Vec<FlexMixRemoval>,
    /// Components whose residual sd was floored (0-based, initial numbering)
    #[serde(default)]
    pub degenerate_components: Vec<usize>,
    /// What the fit had to do or could not do
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The stopping rule's final state
    #[serde(default)]
    pub convergence: Convergence,
}

impl ResultNotes for FlexMixResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl std::fmt::Display for FlexMixResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FlexMix Finite Mixture Regression")?;
        writeln!(f, "==================================")?;
        writeln!(f, "Components: {}", self.k)?;
        writeln!(f, "Initial components: {}", self.k_initial)?;
        writeln!(f, "Observations: {}", self.n)?;
        writeln!(f, "Predictors: {}", self.p)?;
        for r in &self.removed_components {
            writeln!(
                f,
                "Removed component {} at iteration {} (prior {:.4})",
                r.component + 1,
                r.iteration,
                r.prior
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Model fit:")?;
        writeln!(f, "  Log-likelihood: {:.4}", self.loglik)?;
        writeln!(f, "  AIC: {:.4}", self.aic)?;
        writeln!(f, "  BIC: {:.4}", self.bic)?;
        writeln!(f)?;
        writeln!(f, "Component weights:")?;
        for (i, w) in self.prior.iter().enumerate() {
            writeln!(f, "  Component {}: {:.4}", i + 1, w)?;
        }
        writeln!(f)?;
        writeln!(f, "Component parameters:")?;
        for (i, (coef, sig)) in self.coefficients.iter().zip(&self.sigma).enumerate() {
            writeln!(f, "  Component {}:", i + 1)?;
            writeln!(f, "    Coefficients: {:?}", coef)?;
            writeln!(f, "    Sigma: {:.4}", sig)?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Where flexmix starts (R `flexmix(cluster = )`).
#[derive(Debug, Clone, Default, PartialEq)]
pub enum FlexMixStart {
    /// Random hard clusters drawn uniformly from `0..k` (`seed` or [`FLEXMIX_DEFAULT_SEED`]),
    /// posteriors 0.9/0.1.
    #[default]
    Random,
    /// Hard clusters (0-based, below `k`): posteriors 0.9 on the cluster and 0.1 elsewhere,
    /// normalised.
    Clusters(Vec<usize>),
    /// An n x k matrix of non-negative weights with positive row sums, normalised by row.
    Posterior(Array2<f64>),
}

/// Configuration of [`flexmix_with`] (flexmix's `FLXcontrol` defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct FlexMixConfig {
    /// Number of components at the start.
    pub k: usize,
    /// The start.
    pub start: FlexMixStart,
    /// Components whose prior falls below this are removed (default
    /// [`FLEXMIX_DEFAULT_MINPRIOR`]).
    pub minprior: f64,
    /// Maximum number of iterations (default 200, `iter.max`).
    pub max_iterations: usize,
    /// Stopping rule `deviance_change(l, l_prev) < tolerance` (default 1e-6, `tolerance`).
    pub tolerance: f64,
    /// Seed of the random start; `None` means [`FLEXMIX_DEFAULT_SEED`].
    pub seed: Option<u64>,
}

impl FlexMixConfig {
    /// The defaults for `k` components.
    pub fn new(k: usize) -> Self {
        FlexMixConfig {
            k,
            start: FlexMixStart::default(),
            minprior: FLEXMIX_DEFAULT_MINPRIOR,
            max_iterations: 200,
            tolerance: 1e-6,
            seed: None,
        }
    }
}

/// The weighted least-squares fit of one component: coefficients, residuals of every row on the
/// original scale, and rank. The normal equations of the sqrt(w)-scaled design first, as OLS
/// decides; the weighted pivoted QR when they are singular or ill conditioned, for the rank.
fn component_fit(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    scaled_x: &mut Array2<f64>,
    scaled_y: &mut Array1<f64>,
) -> EconResult<(Array1<f64>, Array1<f64>, usize)> {
    let p = x.ncols();
    for ((mut sx, row), wi) in scaled_x.rows_mut().into_iter().zip(x.rows()).zip(w.iter()) {
        let root = wi.sqrt();
        sx.iter_mut()
            .zip(row.iter())
            .for_each(|(s, v)| *s = root * v);
    }
    for ((s, v), wi) in scaled_y.iter_mut().zip(y.iter()).zip(w.iter()) {
        *s = wi.sqrt() * v;
    }
    let gram = xtx(&scaled_x.view());
    if let Some(normal) = normal_inverse_from_gram(gram.view())? {
        let beta = normal.coefficients(scaled_x.view(), scaled_y);
        let residuals = &y - &x.dot(&beta);
        return Ok((beta, residuals, p));
    }
    let ls = weighted_least_squares(x, y, w)?;
    Ok((ls.coefficients, ls.residuals, ls.rank))
}

/// Fit a finite mixture of Gaussian regressions as `flexmix::FLXfit` does
/// (`FLXMRglm()` components, `FLXPconstant()` prior, `classify = "weighted"`).
///
/// Each iteration: the prior is the mean posterior; a component whose prior is below `minprior`
/// is removed and recorded, and the posteriors are renormalised over the rest; each component gets
/// a weighted least-squares fit (`lm.wfit`'s solution) and
/// `sigma_j = sqrt(sum_i w_i r_i^2 / (mean(w) (n - rank)))`; the E-step gives the new posteriors and
/// the log-likelihood; the fit stops when `deviance_change(l, l_prev) < tolerance`.
///
/// Where R stops, p2a differs: a residual sd below [`FLEXMIX_SIGMA_FLOOR_REL`]` x sd(y)` (R:
/// "Log-likelihood: NaN") is floored there and flagged in `warnings` and `degenerate_components`,
/// because the likelihood is unbounded at an exact fit. R's random start uses R's generator; give
/// [`FlexMixStart::Clusters`] for the same start on both sides. `validation/ml/mixtures.md`.
pub fn flexmix_with(
    y: ArrayView1<'_, f64>,
    x: ArrayView2<'_, f64>,
    config: &FlexMixConfig,
) -> EconResult<FlexMixResult> {
    let (n, p) = x.dim();
    if y.len() != n {
        return Err(spec(format!(
            "y ({}) and x ({n}) must have the same number of rows",
            y.len()
        )));
    }
    if p == 0 {
        return Err(spec("x must have at least one column".into()));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(spec("flexmix: y must contain only finite values".into()));
    }
    crate::ml::extract::require_finite(&x, "flexmix").map_err(spec)?;
    let k0 = config.k;
    if k0 == 0 || k0 > n {
        return Err(spec(format!(
            "k must be between 1 and {n} (the number of observations), got {k0}"
        )));
    }
    if config.max_iterations == 0 {
        return Err(spec("max_iterations must be at least 1, got 0".into()));
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(spec(format!(
            "tolerance must be finite and positive, got {}",
            config.tolerance
        )));
    }
    if !config.minprior.is_finite() || !(0.0..1.0).contains(&config.minprior) {
        return Err(spec(format!(
            "minprior must be finite, at least 0 and below 1, got {}",
            config.minprior
        )));
    }
    if n <= p {
        return Err(EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: "flexmix: flexmix's sigma divides by n - rank".into(),
        });
    }
    let y_mean = y.sum() / n as f64;
    let sd_y = (y.iter().map(|v| (v - y_mean) * (v - y_mean)).sum::<f64>() / (n - 1) as f64).sqrt();
    if !sd_y.is_finite() || sd_y <= 0.0 {
        return Err(spec("flexmix: y is constant".into()));
    }
    let floor = FLEXMIX_SIGMA_FLOOR_REL * sd_y;

    // flexmix's initPosteriors.
    let mut post = try_zeros2(n, k0, "flexmix posteriors")?;
    match &config.start {
        FlexMixStart::Posterior(start) => {
            if start.dim() != (n, k0) {
                return Err(spec(format!(
                    "start posterior must be {n} x {k0}, got {} x {}",
                    start.nrows(),
                    start.ncols()
                )));
            }
            for (i, (row, mut out)) in start.rows().into_iter().zip(post.rows_mut()).enumerate() {
                if row.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(spec(format!(
                        "start posterior row {i} must be finite and non-negative"
                    )));
                }
                let total = row.sum();
                if total <= 0.0 {
                    return Err(spec(format!(
                        "start posterior row {i} must have a positive sum"
                    )));
                }
                out.iter_mut()
                    .zip(row.iter())
                    .for_each(|(o, v)| *o = v / total);
            }
        }
        start => {
            let random: Vec<usize>;
            let labels: &[usize] = match start {
                FlexMixStart::Clusters(c) => {
                    if c.len() != n {
                        return Err(spec(format!(
                            "start clusters have length {}, the data have {n} rows",
                            c.len()
                        )));
                    }
                    if let Some(bad) = c.iter().find(|l| **l >= k0) {
                        return Err(spec(format!("start cluster {bad} must be below k = {k0}")));
                    }
                    c
                }
                _ => {
                    let mut rng =
                        StdRng::seed_from_u64(config.seed.unwrap_or(FLEXMIX_DEFAULT_SEED));
                    random = (0..n).map(|_| rng.gen_range(0..k0)).collect();
                    &random
                }
            };
            let total = 0.9 + 0.1 * (k0 - 1) as f64;
            post.fill(0.1 / total);
            for (i, &l) in labels.iter().enumerate() {
                post[[i, l]] = 0.9 / total;
            }
        }
    }

    let mut scaled_x = try_zeros2(n, p, "flexmix weighted design")?;
    let mut scaled_y = Array1::zeros(n);
    let mut orig: Vec<usize> = (0..k0).collect();
    let mut removed: Vec<FlexMixRemoval> = Vec::new();
    let mut degenerate: Vec<usize> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut coefficients: Vec<Vec<f64>> = Vec::new();
    let mut sigma: Vec<f64> = Vec::new();
    let mut llh = f64::NEG_INFINITY;
    let mut change = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 1..=config.max_iterations {
        checkpoint()?;
        iterations = iter;
        let mut prior: Vec<f64> = post
            .columns()
            .into_iter()
            .map(|c| c.sum() / n as f64)
            .collect();

        // minprior: flexmix removes at the start of an iteration.
        if prior.iter().any(|p| *p < config.minprior) {
            let keep: Vec<usize> = (0..prior.len())
                .filter(|&j| prior[j] >= config.minprior)
                .collect();
            for (j, &pj) in prior.iter().enumerate() {
                if pj < config.minprior {
                    removed.push(FlexMixRemoval {
                        iteration: iter,
                        component: orig[j],
                        prior: pj,
                    });
                    warnings.push(format!(
                        "removed component {} at iteration {iter} (prior {pj:.4} < minprior {})",
                        orig[j] + 1,
                        config.minprior
                    ));
                }
            }
            if keep.is_empty() {
                return Err(spec(format!(
                    "flexmix: all components removed (minprior {})",
                    config.minprior
                )));
            }
            orig = keep.iter().map(|&j| orig[j]).collect();
            prior = keep.iter().map(|&j| prior[j]).collect();
            let total: f64 = prior.iter().sum();
            prior.iter_mut().for_each(|p| *p /= total);
            post = post.select(Axis(1), &keep);
            for mut row in post.rows_mut() {
                let row_sum = row.sum();
                if row_sum > 0.0 {
                    row.iter_mut().for_each(|v| *v /= row_sum);
                } else {
                    row.iter_mut().zip(&prior).for_each(|(v, p)| *v = *p);
                }
            }
        }

        // M-step, then the component log-densities (written over the posteriors).
        let k = prior.len();
        coefficients.clear();
        sigma.clear();
        let mut resid_rows: Vec<Array1<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let w = post.column(j);
            let (beta, residuals, rank) = component_fit(x, y, w, &mut scaled_x, &mut scaled_y)?;
            if rank < p {
                return Err(spec(format!(
                    "flexmix: the weighted design of component {} is rank deficient (rank {rank} \
                     of {p})",
                    orig[j] + 1
                )));
            }
            let mean_w = w.sum() / n as f64;
            let wrss: f64 = w
                .iter()
                .zip(residuals.iter())
                .map(|(wi, r)| wi * r * r)
                .sum();
            let mut s = (wrss / (mean_w * (n - rank) as f64)).sqrt();
            if !(s >= floor) {
                s = floor;
                if !degenerate.contains(&orig[j]) {
                    degenerate.push(orig[j]);
                    warnings.push(format!(
                        "component {} is degenerate: its residual standard deviation fell below \
                         1e-6 x sd(y) and was set to {floor:.6e}; the likelihood is unbounded \
                         there, so logLik, AIC and BIC are not comparable (R flexmix stops with \
                         'Log-likelihood: NaN')",
                        orig[j] + 1
                    ));
                }
            }
            sigma.push(s);
            coefficients.push(beta.to_vec());
            resid_rows.push(residuals);
        }

        // E-step.
        for j in 0..k {
            let constant = prior[j].ln() - HALF_LN_2PI - sigma[j].ln();
            let inv = 1.0 / sigma[j];
            for (v, r) in post.column_mut(j).iter_mut().zip(resid_rows[j].iter()) {
                let z = r * inv;
                *v = constant - 0.5 * z * z;
            }
        }
        let llh_old = llh;
        llh = 0.0;
        for mut row in post.rows_mut() {
            let top = row.iter().fold(f64::NEG_INFINITY, |a, v| a.max(*v));
            let sum_exp: f64 = row.iter().map(|v| (v - top).exp()).sum();
            let lse = top + sum_exp.ln();
            llh += lse;
            row.iter_mut().for_each(|v| *v = (*v - lse).exp());
        }
        if !llh.is_finite() {
            return Err(spec(format!(
                "flexmix: the log-likelihood is not finite at iteration {iter} (R flexmix stops \
                 with 'Log-likelihood: {llh}')"
            )));
        }

        change = deviance_change(llh, llh_old);
        if change < config.tolerance {
            converged = true;
            break;
        }
    }

    let convergence = Convergence::new(
        converged,
        iterations,
        config.max_iterations,
        "relative change in log-likelihood (flexmix)",
        change,
        config.tolerance,
    );
    if let Some(note) = convergence.note("flexmix") {
        warnings.push(note);
    }

    let k = post.ncols();
    let prior: Vec<f64> = post
        .columns()
        .into_iter()
        .map(|c| c.sum() / n as f64)
        .collect();
    let n_parameters = (k - 1) + k * (p + 1);
    let df = n_parameters as f64;
    Ok(FlexMixResult {
        cluster: hard_labels(&post),
        posterior: post,
        coefficients,
        sigma,
        prior,
        loglik: llh,
        bic: -2.0 * llh + df * (n as f64).ln(),
        aic: -2.0 * llh + 2.0 * df,
        n_iterations: iterations,
        converged,
        k,
        n,
        p,
        k_initial: k0,
        minprior: config.minprior,
        n_parameters,
        removed_components: removed,
        degenerate_components: degenerate,
        warnings,
        convergence,
    })
}

/// Fit a finite mixture regression model with the EM algorithm ([`flexmix_with`] with a random
/// start and the positional arguments).
///
/// # Arguments
/// * `y` - Response variable (n x 1; one column)
/// * `x` - Predictor matrix (n x p), should include intercept column if desired
/// * `k` - Number of mixture components at the start
/// * `max_iter` - Maximum EM iterations (default: 200)
/// * `tol` - Stopping rule on `|dl| / (|l| + 0.1)` (default: 1e-6)
/// * `seed` - Seed of the random start (default: [`FLEXMIX_DEFAULT_SEED`])
///
/// # Returns
/// * `FlexMixResult` with fitted parameters and model diagnostics
///
/// # References
///
/// - Leisch (2004). "FlexMix: A General Framework for Finite Mixture Models".
pub fn flexmix(
    y: ArrayView2<f64>,
    x: ArrayView2<f64>,
    k: usize,
    max_iter: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<FlexMixResult, String> {
    if y.ncols() != 1 {
        return Err(format!("y must have one column, got {}", y.ncols()));
    }
    let mut config = FlexMixConfig::new(k);
    if let Some(m) = max_iter {
        config.max_iterations = m;
    }
    if let Some(t) = tol {
        config.tolerance = t;
    }
    config.seed = seed;
    flexmix_with(y.column(0), x, &config).map_err(|e| e.to_string())
}

/// Convenience wrapper for flexmix.
pub fn run_flexmix(
    y: ArrayView2<f64>,
    x: ArrayView2<f64>,
    k: usize,
    max_iter: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<FlexMixResult, String> {
    flexmix(y, x, k, max_iter, tol, seed)
}

// =============================================================================
// Mixture Models (mixtools)
// =============================================================================

/// Result of normal mixture EM fitting.
///
/// # References
///
/// - McLachlan, G. J., & Peel, D. (2000). "Finite Mixture Models." Wiley.
/// - R mixtools package documentation
///   Source: <https://www.rdocumentation.org/packages/mixtools/versions/2.0.0/topics/normalmixEM>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalMixEMResult {
    /// Component means
    pub mu: Vec<f64>,
    /// Component standard deviations
    pub sigma: Vec<f64>,
    /// Component mixing proportions
    pub lambda: Vec<f64>,
    /// Posterior probabilities (n x k)
    #[serde(skip)]
    pub posterior: Array2<f64>,
    /// Hard cluster assignments
    pub labels: Vec<usize>,
    /// Log-likelihood at convergence (the normal log-likelihood, with its `-0.5 ln(2 pi)` per
    /// observation)
    pub loglik: f64,
    /// Log-likelihood history of the last start
    pub loglik_history: Vec<f64>,
    /// Number of iterations of the last start
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Number of random restarts after a collapsed variance
    #[serde(default)]
    pub restarts: usize,
    /// What the fit had to do or could not do
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The stopping rule's final state
    #[serde(default)]
    pub convergence: Convergence,
}

impl ResultNotes for NormalMixEMResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl std::fmt::Display for NormalMixEMResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Normal Mixture EM Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Number of components: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Log-likelihood: {:.4}", self.loglik)?;
        writeln!(f)?;
        writeln!(f, "Component parameters:")?;
        for i in 0..self.k {
            writeln!(
                f,
                "  Component {}: mu={:.4}, sigma={:.4}, lambda={:.4}",
                i + 1,
                self.mu[i],
                self.sigma[i],
                self.lambda[i]
            )?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Starting values of [`normal_mix_em_with`] (R `lambda`, `mu`, `sigma`).
#[derive(Debug, Clone, PartialEq)]
pub struct NormalMixStart {
    /// Mixing proportions, positive, summing to 1.
    pub lambda: Vec<f64>,
    /// Component means.
    pub mu: Vec<f64>,
    /// Component standard deviations, positive.
    pub sigma: Vec<f64>,
}

/// Configuration of [`normal_mix_em_with`].
#[derive(Debug, Clone, PartialEq)]
pub struct NormalMixConfig {
    /// Number of components.
    pub k: usize,
    /// Starting values. `None`: the deterministic start when `seed` is `None`, otherwise
    /// mixtools' random `normalmix.init` from the seed.
    pub start: Option<NormalMixStart>,
    /// Maximum number of iterations per start (default 1000, `normalmixEM`'s `maxit`).
    pub max_iterations: usize,
    /// Stopping rule `|l - l_prev| < tolerance` (default 1e-8, `normalmixEM`'s `epsilon`).
    pub tolerance: f64,
    /// Seed of the random start and restarts; restarts use [`MIXTURE_DEFAULT_SEED`] when `None`.
    pub seed: Option<u64>,
}

impl NormalMixConfig {
    /// The defaults for `k` components.
    pub fn new(k: usize) -> Self {
        NormalMixConfig {
            k,
            start: None,
            max_iterations: 1000,
            tolerance: 1e-8,
            seed: None,
        }
    }
}

/// A draw of mixtools' `normalmix.init` for k components (`arbmean = arbvar = TRUE`), in a fixed
/// order that is not R's (p2a's generator cannot reproduce R's draws): for bin j of the sorted x,
/// `s_j = sd(bin) / E` with `E ~ Exp(1)`, then `mu_j ~ Normal(mean(bin), s_j)`; lambda uniform and
/// renormalised, redrawn while a proportion is below `min(0.05, 1 / (2 k))`. R has no such lower
/// limit for large k, where its redraw never ends; after 100 redraws the proportions are equal.
fn normalmix_init(sorted: &[f64], k: usize, sd_x: f64, rng: &mut StdRng) -> NormalMixStart {
    let n = sorted.len();
    let mut mu = Vec::with_capacity(k);
    let mut sigma = Vec::with_capacity(k);
    for j in 1..=k {
        let lo = ((j - 1) * n / k).max(1) - 1;
        let hi = (j * n).div_ceil(k) - 1;
        let bin = &sorted[lo..=hi];
        let m = bin.iter().sum::<f64>() / bin.len() as f64;
        let hyper = if bin.len() > 1 {
            (bin.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (bin.len() - 1) as f64).sqrt()
        } else {
            0.0
        };
        // A tied bin has no spread of its own: mixtools draws it uniformly from (0, sd(x)).
        let hyper = if hyper > 0.0 {
            hyper
        } else {
            (1.0 - rng.r#gen::<f64>()) * sd_x
        };
        let e: f64 = rng.sample(Exp1);
        let mut s = hyper / e;
        if !s.is_finite() || s <= 0.0 {
            s = hyper;
        }
        let draw = Normal::new(m, s).map(|d| d.sample(rng)).unwrap_or(m);
        mu.push(draw);
        sigma.push(s);
    }
    let limit = 0.05f64.min(0.5 / k as f64);
    let mut lambda = vec![1.0 / k as f64; k];
    for _ in 0..100 {
        let u: Vec<f64> = (0..k).map(|_| rng.r#gen::<f64>()).collect();
        let total: f64 = u.iter().sum();
        if total > 0.0 && u.iter().all(|v| v / total >= limit) {
            lambda = u.iter().map(|v| v / total).collect();
            break;
        }
    }
    NormalMixStart { lambda, mu, sigma }
}

/// Fit a univariate normal mixture by EM, to the fixed point `mixtools::normalmixEM` reaches.
///
/// The log-likelihood includes the normal constant `-0.5 ln(2 pi)` per observation. Without `start`,
/// `seed = None` gives the deterministic start (equal weights, the population sd, means at
/// `min + (j + 0.5) / k (max - min)`) and `seed = Some(_)` mixtools' random `normalmix.init`. When a
/// component sd falls below [`MIXTOOLS_SIGMA_COLLAPSE_REL`]` x sd(x)` or a component loses its mass,
/// the fit restarts from new random values as `normalmixEM` does, at most
/// [`MIXTOOLS_MAX_RESTARTS`] times; past that it returns an error (mixtools: "Too many tries!").
///
/// `normalmixEM` updates sigma with the previous iteration's mu; p2a runs the standard M-step. Both
/// stop at the same fixed point, which is what the tests pin (`validation/ml/mixtures.md`).
pub fn normal_mix_em_with(data: &[f64], config: &NormalMixConfig) -> EconResult<NormalMixEMResult> {
    let n = data.len();
    let k = config.k;
    check_em_limits(k, n, config.max_iterations, config.tolerance)?;
    if let Some((i, v)) = data.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(spec(format!(
            "normal mixture: data must contain only finite values (element {i} is {v})"
        )));
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let ss: f64 = data.iter().map(|v| (v - mean) * (v - mean)).sum();
    if ss <= 0.0 || n < 2 {
        return Err(spec("normal mixture: data have zero variance".into()));
    }
    if !ss.is_finite() {
        return Err(spec(
            "normal mixture: the data are too large in magnitude for their variance to be finite"
                .into(),
        ));
    }
    let sd_x = (ss / (n - 1) as f64).sqrt();
    let collapse = MIXTOOLS_SIGMA_COLLAPSE_REL * sd_x;

    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or(MIXTURE_DEFAULT_SEED));

    let NormalMixStart {
        mut lambda,
        mut mu,
        mut sigma,
    } = match &config.start {
        Some(s) => {
            if s.mu.len() != k || s.mu.iter().any(|v| !v.is_finite()) {
                return Err(spec(format!("start mu must have {k} finite values")));
            }
            if s.sigma.len() != k || s.sigma.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(spec(format!(
                    "start sigma must have {k} finite, positive values"
                )));
            }
            NormalMixStart {
                lambda: checked_lambda(&s.lambda, k)?,
                mu: s.mu.clone(),
                sigma: s.sigma.clone(),
            }
        }
        None if config.seed.is_some() => normalmix_init(&sorted, k, sd_x, &mut rng),
        None => {
            let (min, max) = (sorted[0], sorted[n - 1]);
            NormalMixStart {
                lambda: vec![1.0 / k as f64; k],
                mu: (0..k)
                    .map(|j| min + (j as f64 + 0.5) / k as f64 * (max - min))
                    .collect(),
                sigma: vec![(ss / n as f64).sqrt(); k],
            }
        }
    };

    let mut posterior = try_zeros2(n, k, "normal mixture posteriors")?;
    let mut loglik_history: Vec<f64> = Vec::new();
    let mut restarts = 0;
    let mut lp = vec![0.0; k];
    let mut constants = vec![0.0; k];
    let mut inv_sigma = vec![0.0; k];
    let mut nj = vec![0.0; k];
    let mut sum_x = vec![0.0; k];
    let mut sum_xx = vec![0.0; k];
    let mut ll_prev = f64::NEG_INFINITY;
    let mut last_change = f64::INFINITY;
    let mut iter = 0;
    let mut converged = false;

    loop {
        checkpoint()?;
        iter += 1;
        // E-step.
        for j in 0..k {
            constants[j] = lambda[j].ln() - sigma[j].ln() - HALF_LN_2PI;
            inv_sigma[j] = 1.0 / sigma[j];
        }
        nj.fill(0.0);
        sum_x.fill(0.0);
        sum_xx.fill(0.0);
        let mut ll = 0.0;
        for (&x, mut post) in data.iter().zip(posterior.rows_mut()) {
            let mut top = f64::NEG_INFINITY;
            for j in 0..k {
                let z = (x - mu[j]) * inv_sigma[j];
                let v = constants[j] - 0.5 * z * z;
                lp[j] = v;
                top = top.max(v);
            }
            let sum_exp: f64 = lp.iter().map(|v| (v - top).exp()).sum();
            let lse = top + sum_exp.ln();
            ll += lse;
            for (j, p) in post.iter_mut().enumerate() {
                let w = (lp[j] - lse).exp();
                *p = w;
                nj[j] += w;
                sum_x[j] += w * x;
                sum_xx[j] += w * x * x;
            }
        }
        if !ll.is_finite() {
            return Err(EconError::Computation(format!(
                "normal mixture: the log-likelihood is not finite at iteration {iter}"
            )));
        }
        loglik_history.push(ll);
        if iter > 1 {
            last_change = (ll - ll_prev).abs();
            if last_change < config.tolerance {
                converged = true;
                break;
            }
        }
        ll_prev = ll;
        if iter == config.max_iterations {
            break;
        }

        // M-step: weights, means and the standard sd about the new means.
        let mut collapsed = false;
        for j in 0..k {
            if nj[j] <= n as f64 * f64::EPSILON {
                collapsed = true;
                break;
            }
            let m = sum_x[j] / nj[j];
            let var = (sum_xx[j] / nj[j] - m * m).max(0.0);
            lambda[j] = nj[j] / n as f64;
            mu[j] = m;
            sigma[j] = var.sqrt();
            if sigma[j] < collapse {
                collapsed = true;
            }
        }
        if collapsed {
            restarts += 1;
            if restarts > MIXTOOLS_MAX_RESTARTS {
                return Err(EconError::ConvergenceFailure {
                    iterations: iter,
                    last_change: f64::NAN,
                    suggestion: format!(
                        "a component variance collapsed in {} starts, {MIXTOOLS_MAX_RESTARTS} of \
                         them restarts (mixtools stops with 'Too many tries!'); reduce k or \
                         remove tied values",
                        MIXTOOLS_MAX_RESTARTS + 1
                    ),
                });
            }
            let s = normalmix_init(&sorted, k, sd_x, &mut rng);
            lambda = s.lambda;
            mu = s.mu;
            sigma = s.sigma;
            loglik_history.clear();
            ll_prev = f64::NEG_INFINITY;
            last_change = f64::INFINITY;
            iter = 0;
        }
    }

    let convergence = Convergence::new(
        converged,
        iter,
        config.max_iterations,
        "absolute change in log-likelihood",
        last_change,
        config.tolerance,
    );
    let mut warnings = Vec::new();
    if restarts > 0 {
        warnings.push(format!(
            "restarted from random values {restarts} time(s) after a component variance collapsed \
             (as mixtools)"
        ));
    }
    if let Some(note) = convergence.note("normal mixture EM") {
        warnings.push(note);
    }
    let loglik = loglik_history.last().copied().unwrap_or(f64::NEG_INFINITY);
    Ok(NormalMixEMResult {
        mu,
        sigma,
        lambda,
        labels: hard_labels(&posterior),
        posterior,
        loglik,
        loglik_history,
        n_iterations: iter,
        converged,
        k,
        n,
        restarts,
        warnings,
        convergence,
    })
}

/// Fit a univariate normal mixture model by EM ([`normal_mix_em_with`] with the positional
/// arguments).
///
/// # Arguments
/// * `data` - Univariate data (vector)
/// * `k` - Number of mixture components
/// * `max_iterations` - Maximum EM iterations per start (default: 1000, `normalmixEM`'s)
/// * `tol` - Stopping rule on `|l - l_prev|` (default: 1e-8, `normalmixEM`'s)
/// * `seed` - `None`: the deterministic start. `Some(_)`: mixtools' random start from the seed
///
/// # Returns
/// * `NormalMixEMResult` containing fitted parameters
pub fn normal_mix_em(
    data: &[f64],
    k: usize,
    max_iterations: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<NormalMixEMResult, String> {
    let mut config = NormalMixConfig::new(k);
    if let Some(m) = max_iterations {
        config.max_iterations = m;
    }
    if let Some(t) = tol {
        config.tolerance = t;
    }
    config.seed = seed;
    normal_mix_em_with(data, &config).map_err(|e| e.to_string())
}

/// Result of multivariate normal mixture EM fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultivariateNormalMixEMResult {
    /// Component means (k x d)
    #[serde(skip)]
    pub mu: Array2<f64>,
    /// Component covariance matrices (k matrices, each d x d)
    #[serde(skip)]
    pub sigma: Vec<Array2<f64>>,
    /// Component mixing proportions
    pub lambda: Vec<f64>,
    /// Posterior probabilities (n x k)
    #[serde(skip)]
    pub posterior: Array2<f64>,
    /// Hard cluster assignments
    pub labels: Vec<usize>,
    /// Log-likelihood at convergence
    pub loglik: f64,
    /// Number of iterations of the last start
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Number of dimensions
    pub d: usize,
    /// Number of random restarts after a singular covariance
    #[serde(default)]
    pub restarts: usize,
    /// What the fit had to do or could not do
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The stopping rule's final state
    #[serde(default)]
    pub convergence: Convergence,
}

impl ResultNotes for MultivariateNormalMixEMResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl std::fmt::Display for MultivariateNormalMixEMResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multivariate Normal Mixture EM Results")?;
        writeln!(f, "=======================================")?;
        writeln!(f, "Number of components: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Dimensions: {}", self.d)?;
        writeln!(f, "Log-likelihood: {:.4}", self.loglik)?;
        writeln!(f)?;
        writeln!(f, "Mixing proportions:")?;
        for (i, &l) in self.lambda.iter().enumerate() {
            writeln!(f, "  Component {}: {:.4}", i + 1, l)?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Starting values of [`mvnorm_mix_em_with`] (R `lambda`, `mu`, `sigma`).
#[derive(Debug, Clone, PartialEq)]
pub struct MvNormalMixStart {
    /// Mixing proportions, positive, summing to 1.
    pub lambda: Vec<f64>,
    /// Component means (k x d).
    pub mu: Vec<Vec<f64>>,
    /// Component covariance matrices (k x d x d), symmetric positive definite.
    pub sigma: Vec<Vec<Vec<f64>>>,
}

/// Configuration of [`mvnorm_mix_em_with`].
#[derive(Debug, Clone, PartialEq)]
pub struct MvNormalMixConfig {
    /// Number of components.
    pub k: usize,
    /// Starting values; `None` is a k-means++ start from the seed.
    pub start: Option<MvNormalMixStart>,
    /// Maximum number of iterations per start (default 10000, `mvnormalmixEM`'s `maxit`).
    pub max_iterations: usize,
    /// Stopping rule `|l - l_prev| < tolerance` (default 1e-8, `mvnormalmixEM`'s `epsilon`).
    pub tolerance: f64,
    /// Seed of the k-means++ start and of restarts; `None` means [`MIXTURE_DEFAULT_SEED`].
    pub seed: Option<u64>,
}

impl MvNormalMixConfig {
    /// The defaults for `k` components.
    pub fn new(k: usize) -> Self {
        MvNormalMixConfig {
            k,
            start: None,
            max_iterations: 10_000,
            tolerance: 1e-8,
            seed: None,
        }
    }
}

/// k-means++ centres, covariances `global_var * I` and equal weights.
fn mv_random_start(
    data: ArrayView2<'_, f64>,
    k: usize,
    global_var: f64,
    rng: &mut StdRng,
) -> EconResult<(Vec<f64>, Vec<Component>)> {
    let (n, d) = data.dim();
    let mut centres: Vec<Vec<f64>> = Vec::with_capacity(k);
    centres.push(data.row(rng.gen_range(0..n)).to_vec());
    let mut distances = vec![f64::INFINITY; n];
    for c in 1..k {
        let last = &centres[c - 1];
        for (dist, row) in distances.iter_mut().zip(data.rows()) {
            let to_last: f64 = row.iter().zip(last).map(|(a, b)| (a - b) * (a - b)).sum();
            *dist = dist.min(to_last);
        }
        let total: f64 = distances.iter().sum();
        let threshold = rng.r#gen::<f64>() * total;
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for (i, &dist) in distances.iter().enumerate() {
            cumsum += dist;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centres.push(data.row(chosen).to_vec());
    }
    let mut comps = Vec::with_capacity(k);
    for mean in centres {
        let mut cov = try_zeros2(d, d, "normal mixture covariance")?;
        cov.diag_mut().fill(global_var);
        let Some(c) = Component::new(mean, cov) else {
            return Err(spec(
                "normal mixture: the data's variance is too small for a start covariance".into(),
            ));
        };
        comps.push(c);
    }
    Ok((vec![1.0 / k as f64; k], comps))
}

/// Fit a multivariate normal mixture with full covariances by EM, to the fixed point
/// `mixtools::mvnormalmixEM` reaches.
///
/// E- and M-steps run on the Cholesky components shared with
/// [`gaussian_mixture`](super::gmm::gaussian_mixture). Without `start`, the start is k-means++ from
/// `seed` (default [`MIXTURE_DEFAULT_SEED`]). When an M-step gives a covariance that is not positive
/// definite or a weight below 1e-8, the fit restarts from new k-means++ centres as `mvnormalmixEM`
/// restarts from new random values, at most [`MVMIXTOOLS_MAX_RESTARTS`] times; past that it returns
/// an error. See `validation/ml/mixtures.md`.
pub fn mvnorm_mix_em_with(
    data: ArrayView2<'_, f64>,
    config: &MvNormalMixConfig,
) -> EconResult<MultivariateNormalMixEMResult> {
    let (n, d) = data.dim();
    let k = config.k;
    check_em_limits(k, n, config.max_iterations, config.tolerance)?;
    if d == 0 {
        return Err(spec("normal mixture: the data have no columns".into()));
    }
    crate::ml::extract::require_finite(&data, "normal mixture").map_err(spec)?;

    let global_var: f64 = data
        .columns()
        .into_iter()
        .map(|c| {
            let m = c.sum() / n as f64;
            c.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64
        })
        .sum::<f64>()
        / d as f64;
    if !global_var.is_finite() {
        return Err(spec(
            "normal mixture: the data are too large in magnitude for their variance to be finite"
                .into(),
        ));
    }
    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or(MIXTURE_DEFAULT_SEED));

    let (mut lambda, mut comps) = match &config.start {
        Some(s) => {
            let lambda = checked_lambda(&s.lambda, k)?;
            if s.mu.len() != k
                || s.mu
                    .iter()
                    .any(|m| m.len() != d || m.iter().any(|v| !v.is_finite()))
            {
                return Err(spec(format!("start mu must be a finite {k} x {d} matrix")));
            }
            if s.sigma.len() != k {
                return Err(spec(format!(
                    "start sigma must have {k} matrices, got {}",
                    s.sigma.len()
                )));
            }
            let mut comps = Vec::with_capacity(k);
            for (j, (m, sig)) in s.mu.iter().zip(&s.sigma).enumerate() {
                if sig.len() != d
                    || sig
                        .iter()
                        .any(|r| r.len() != d || r.iter().any(|v| !v.is_finite()))
                {
                    return Err(spec(format!(
                        "start sigma[{j}] must be a finite {d} x {d} matrix"
                    )));
                }
                let big = sig.iter().flatten().fold(1.0f64, |a, v| a.max(v.abs()));
                let mut cov = try_zeros2(d, d, "normal mixture covariance")?;
                for a in 0..d {
                    for b in 0..d {
                        if (sig[a][b] - sig[b][a]).abs() > 1e-10 * big {
                            return Err(spec(format!("start sigma[{j}] is not symmetric")));
                        }
                        cov[[a, b]] = sig[a][b];
                    }
                }
                let Some(c) = Component::new(m.clone(), cov) else {
                    return Err(spec(format!("start sigma[{j}] is not positive definite")));
                };
                comps.push(c);
            }
            (lambda, comps)
        }
        None => {
            if global_var <= 0.0 {
                return Err(spec("normal mixture: data have zero variance".into()));
            }
            mv_random_start(data, k, global_var, &mut rng)?
        }
    };

    let mut posterior = try_zeros2(n, k, "normal mixture posteriors")?;
    let mut restarts = 0;
    let mut ll = f64::NEG_INFINITY;
    let mut last_change = f64::INFINITY;
    let mut iter = 0;
    let mut converged = false;

    loop {
        checkpoint()?;
        iter += 1;
        let ll_prev = ll;
        ll = gmm::e_step(data, &lambda, &comps, &mut posterior);
        if !ll.is_finite() {
            return Err(EconError::Computation(format!(
                "normal mixture: the log-likelihood is not finite at iteration {iter}"
            )));
        }
        if iter > 1 {
            last_change = (ll - ll_prev).abs();
            if last_change < config.tolerance {
                converged = true;
                break;
            }
        }
        if iter == config.max_iterations {
            break;
        }

        // M-step. A component without mass, a negligible weight or a singular covariance restarts.
        let mut next = None;
        match gmm::m_step(data, posterior.view(), CovarianceType::Full) {
            Ok((w, means, covs)) => {
                if w.iter().all(|v| *v >= MV_WEIGHT_COLLAPSE) {
                    let built: Option<Vec<Component>> = means
                        .into_iter()
                        .zip(covs)
                        .map(|(m, c)| Component::new(m, c))
                        .collect();
                    next = built.map(|c| (w, c));
                }
            }
            Err(EconError::InsufficientData { .. }) => {}
            Err(e) => return Err(e),
        }
        if let Some((w, c)) = next {
            lambda = w;
            comps = c;
            continue;
        }
        restarts += 1;
        if restarts > MVMIXTOOLS_MAX_RESTARTS {
            return Err(EconError::ConvergenceFailure {
                iterations: iter,
                last_change: f64::NAN,
                suggestion: format!(
                    "a component covariance was singular in {} starts, {MVMIXTOOLS_MAX_RESTARTS} \
                     of them restarts (mixtools stops with 'Too many tries!'); reduce k or remove \
                     duplicated points",
                    MVMIXTOOLS_MAX_RESTARTS + 1
                ),
            });
        }
        if global_var <= 0.0 {
            return Err(spec("normal mixture: data have zero variance".into()));
        }
        (lambda, comps) = mv_random_start(data, k, global_var, &mut rng)?;
        ll = f64::NEG_INFINITY;
        last_change = f64::INFINITY;
        iter = 0;
    }

    let convergence = Convergence::new(
        converged,
        iter,
        config.max_iterations,
        "absolute change in log-likelihood",
        last_change,
        config.tolerance,
    );
    let mut warnings = Vec::new();
    if restarts > 0 {
        warnings.push(format!(
            "restarted from random values {restarts} time(s) after a singular component \
             covariance (as mixtools)"
        ));
    }
    if let Some(note) = convergence.note("multivariate normal mixture EM") {
        warnings.push(note);
    }
    let mut mu = try_zeros2(k, d, "normal mixture means")?;
    for (mut row, c) in mu.rows_mut().into_iter().zip(&comps) {
        row.iter_mut().zip(&c.mean).for_each(|(a, b)| *a = *b);
    }
    Ok(MultivariateNormalMixEMResult {
        mu,
        sigma: comps.into_iter().map(|c| c.cov).collect(),
        lambda,
        labels: hard_labels(&posterior),
        posterior,
        loglik: ll,
        n_iterations: iter,
        converged,
        k,
        n,
        d,
        restarts,
        warnings,
        convergence,
    })
}

/// Fit a multivariate normal mixture model by EM ([`mvnorm_mix_em_with`] with the positional
/// arguments).
///
/// # Arguments
/// * `data` - Multivariate data (n x d)
/// * `k` - Number of mixture components
/// * `max_iterations` - Maximum EM iterations per start (default: 10000, `mvnormalmixEM`'s)
/// * `tol` - Stopping rule on `|l - l_prev|` (default: 1e-8, `mvnormalmixEM`'s)
/// * `seed` - Seed of the k-means++ start (default: [`MIXTURE_DEFAULT_SEED`])
///
/// # Returns
/// * `MultivariateNormalMixEMResult` containing fitted parameters
pub fn mvnorm_mix_em(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> Result<MultivariateNormalMixEMResult, String> {
    let mut config = MvNormalMixConfig::new(k);
    if let Some(m) = max_iterations {
        config.max_iterations = m;
    }
    if let Some(t) = tol {
        config.tolerance = t;
    }
    config.seed = seed;
    mvnorm_mix_em_with(data, &config).map_err(|e| e.to_string())
}

/// Convenience wrapper for normal_mix_em.
pub fn run_normal_mix_em(
    data: &[f64],
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<NormalMixEMResult, String> {
    normal_mix_em(data, k, max_iterations, None, seed)
}

/// Convenience wrapper for mvnorm_mix_em.
pub fn run_mvnorm_mix_em(
    data: ArrayView2<f64>,
    k: usize,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> Result<MultivariateNormalMixEMResult, String> {
    mvnorm_mix_em(data, k, max_iterations, None, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Case, Check, Tol};

    fn case_matrix(ds: &crate::data::Dataset, cols: &[&str]) -> Array2<f64> {
        let mut m = Array2::zeros((ds.nrows(), cols.len()));
        for (j, c) in cols.iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            m.column_mut(j).assign(&ndarray::Array1::from_vec(v));
        }
        m
    }
    /// Positions that sort `keys` ascending (the case orders components this way).
    fn order_by(keys: &[f64]) -> Vec<usize> {
        let mut o: Vec<usize> = (0..keys.len()).collect();
        o.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
        o
    }
    /// 1-based position of each label in `order` (R's `match(cluster, o)`).
    fn relabel(labels: &[usize], order: &[usize]) -> Vec<f64> {
        let mut pos = vec![0usize; order.len()];
        for (p, &k) in order.iter().enumerate() {
            pos[k] = p;
        }
        labels.iter().map(|&l| (pos[l] + 1) as f64).collect()
    }

    const EM_PARAM: Tol = Tol::new(1e-5, 1e-5);
    const EM_LOGLIK: Tol = Tol::new(1e-12, 0.0);
    const FIXED_POINT: Tol = Tol::new(1e-10, 1e-12);
    const FLEX_PRUNE: Tol = Tol::new(1e-9, 1e-12); // same algorithm, same stop
    const FLEX_KEEP: Tol = Tol::new(1e-7, 1e-9); // a borderline stop: one iteration moves 2.2e-9

    /// A generated CSV for the property tests.
    fn csv(name: &str) -> crate::data::Dataset {
        let path = p2a_validation::repo_path(&format!("validation/datasets/generated/{name}.csv"));
        crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    fn x1(ds: &crate::data::Dataset) -> Vec<f64> {
        numeric_column(ds.df(), "x1", NaPolicy::Fail).unwrap().0
    }
    fn pick(v: &[f64], o: &[usize]) -> Vec<f64> {
        o.iter().map(|&k| v[k]).collect()
    }

    #[test]
    fn test_validate_mixtools_uni() {
        // ML-58: the old log-likelihood is 150 ln(2 pi) = 275.68155996 too high here.
        let (case, ds) = crate::test_support::load_case("mixtools_em");
        let x = x1(&ds);
        let r = normal_mix_em(&x, 3, Some(100_000), Some(1e-12), None).unwrap(); // seed None: deterministic start
        assert!(r.converged);
        let o = order_by(&r.mu);
        Check::new(&case)
            .scalar("uni_tight_loglik", r.loglik, EM_LOGLIK)
            .vector("uni_tight_lambda", &pick(&r.lambda, &o), EM_PARAM)
            .vector("uni_tight_mu", &pick(&r.mu, &o), EM_PARAM)
            .vector("uni_tight_sigma", &pick(&r.sigma, &o), EM_PARAM)
            .scalar(
                "uni_det_start_loglik",
                r.loglik_history[0],
                Tol::new(1e-12, 0.0),
            )
            .scalar("uni_loglik", r.loglik, Tol::new(1e-12, 1e-10)) // normalmixEM's own stop
            .finish();
    }

    /// Means (component-major), proportions and covariances of a multivariate fit in the order `o`.
    fn mv_values(r: &MultivariateNormalMixEMResult, o: &[usize]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            pick(&r.lambda, o),
            o.iter().flat_map(|&k| r.mu.row(k).to_vec()).collect(),
            o.iter()
                .flat_map(|&k| r.sigma[k].iter().copied().collect::<Vec<_>>())
                .collect(),
        )
    }
    /// The case's mv_tight_* sorted by the first mean coordinate.
    fn mv_tight_sorted(case: &Case) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let mu = case.vector("mv_tight_mu");
        let o = order_by(&mu.chunks(3).map(|c| c[0]).collect::<Vec<_>>());
        let lam = case.vector("mv_tight_lambda");
        let sig = case.vector("mv_tight_sigma");
        (
            pick(&lam, &o),
            o.iter()
                .flat_map(|&k| mu[3 * k..3 * k + 3].to_vec())
                .collect(),
            o.iter()
                .flat_map(|&k| sig[9 * k..9 * k + 9].to_vec())
                .collect(),
            case.scalar("mv_tight_loglik"),
        )
    }

    #[test]
    fn test_validate_mixtools_mv() {
        // ML-59: the old fit was axis-aligned (off-diagonals exactly 0) with a different logLik.
        let (case, ds) = crate::test_support::load_case("mixtools_em");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        let best = (1..=5)
            .map(|s| mvnorm_mix_em(x.view(), 3, Some(10_000), Some(1e-12), Some(s)).unwrap())
            .max_by(|a, b| a.loglik.total_cmp(&b.loglik))
            .unwrap();
        let o = order_by(&(0..3).map(|k| best.mu[[k, 0]]).collect::<Vec<_>>());
        let (lam, mu, sig) = mv_values(&best, &o);
        let (tlam, tmu, tsig, tll) = mv_tight_sorted(&case);
        assert!(
            (best.loglik - tll).abs() <= 1e-12 * tll.abs(),
            "{} vs {tll}",
            best.loglik
        );
        for (got, want) in [(&lam, &tlam), (&mu, &tmu), (&sig, &tsig)] {
            for (a, b) in got.iter().zip(want.iter()) {
                assert!(EM_PARAM.allows(*a, *b), "{a} vs {b}");
            }
        }
        assert!(
            sig[1].abs() > 0.3,
            "off-diagonal covariance is fitted: {}",
            sig[1]
        ); // 0.5338 in R
    }

    #[test]
    fn test_validate_mixtools_uni_start() {
        let (case, ds) = crate::test_support::load_case("mixtools_em");
        let x = x1(&ds);
        let mut cfg = NormalMixConfig::new(3);
        cfg.start = Some(NormalMixStart {
            lambda: vec![0.4, 0.3, 0.3],
            mu: vec![-0.5, 1.5, 4.0],
            sigma: vec![1.0; 3],
        });
        cfg.tolerance = 1e-12;
        cfg.max_iterations = 100_000;
        let r = normal_mix_em_with(&x, &cfg).unwrap();
        Check::new(&case)
            .scalar("uni_tight_loglik", r.loglik, EM_LOGLIK)
            .vector("uni_tight_lambda", &r.lambda, EM_PARAM) // start order = ascending mu here
            .vector("uni_tight_mu", &r.mu, EM_PARAM)
            .vector("uni_tight_sigma", &r.sigma, EM_PARAM)
            .vector("uni_mu", &r.mu, EM_PARAM)
            .finish();
    }

    #[test]
    fn test_validate_mixtools_uni_fixed_point() {
        let (case, ds) = crate::test_support::load_case("mixtools_em");
        let x = x1(&ds);
        let mut cfg = NormalMixConfig::new(3);
        cfg.start = Some(NormalMixStart {
            lambda: case.vector("uni_tight_lambda"),
            mu: case.vector("uni_tight_mu"),
            sigma: case.vector("uni_tight_sigma"),
        });
        cfg.tolerance = 1e-12;
        let r = normal_mix_em_with(&x, &cfg).unwrap();
        assert!(
            r.converged && r.n_iterations <= 3,
            "{} iterations",
            r.n_iterations
        );
        Check::new(&case)
            .scalar("uni_tight_loglik", r.loglik, Tol::new(1e-13, 0.0))
            .vector("uni_tight_lambda", &r.lambda, FIXED_POINT)
            .vector("uni_tight_mu", &r.mu, FIXED_POINT)
            .vector("uni_tight_sigma", &r.sigma, FIXED_POINT)
            .finish();
    }

    #[test]
    fn test_validate_mixtools_mv_start() {
        let (case, ds) = crate::test_support::load_case("mixtools_em");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        let mut cfg = MvNormalMixConfig::new(3);
        let eye = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        cfg.start = Some(MvNormalMixStart {
            lambda: vec![0.4, 0.33, 0.27],
            mu: vec![
                vec![0.0, 0.0, 0.0],
                vec![4.0, 1.0, -1.0],
                vec![1.0, 5.0, 2.0],
            ],
            sigma: vec![eye.clone(), eye.clone(), eye],
        });
        cfg.tolerance = 1e-12;
        let r = mvnorm_mix_em_with(x.view(), &cfg).unwrap();
        let (lam, mu, sig) = mv_values(&r, &[0, 1, 2]);
        Check::new(&case)
            .scalar("mv_tight_loglik", r.loglik, EM_LOGLIK)
            .vector("mv_tight_lambda", &lam, EM_PARAM)
            .vector("mv_tight_mu", &mu, EM_PARAM)
            .vector("mv_tight_sigma", &sig, EM_PARAM)
            .vector("mv_mu", &mu, EM_PARAM) // mvnormalmixEM's own stop
            .vector("mv_sigma", &sig, EM_PARAM)
            .finish();
    }

    #[test]
    fn test_normal_mix_em_uses_its_seed() {
        // ML-111: the seed was bound to `_rng` and never used.
        let x = x1(&csv("mixture_3d"));
        let fit = |s| normal_mix_em(&x, 3, Some(5), Some(1e-12), s).unwrap();
        let (a, b, c, d, e) = (
            fit(Some(1)),
            fit(Some(1)),
            fit(Some(2)),
            fit(None),
            fit(None),
        );
        assert_eq!(a.loglik_history[0].to_bits(), b.loglik_history[0].to_bits());
        assert_ne!(
            a.loglik_history[0], c.loglik_history[0],
            "different seeds give the same start"
        );
        assert_ne!(
            a.loglik_history[0], d.loglik_history[0],
            "None is the deterministic start"
        );
        assert_eq!(d.loglik_history[0].to_bits(), e.loglik_history[0].to_bits());
    }

    #[test]
    fn test_normalmix_init_draws_valid_starts_for_any_k() {
        // R's redraw of lambda never ends for k > 20; this must, with proportions that sum to 1.
        let sorted: Vec<f64> = (0..60).map(|i| (i / 2) as f64).collect();
        let mut rng = StdRng::seed_from_u64(9);
        for k in [1, 2, 3, 30, 60] {
            let s = normalmix_init(&sorted, k, 10.0, &mut rng);
            assert_eq!((s.lambda.len(), s.mu.len(), s.sigma.len()), (k, k, k));
            assert!((s.lambda.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(
                s.sigma.iter().all(|v| v.is_finite() && *v > 0.0),
                "{:?}",
                s.sigma
            );
            assert!(s.mu.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_mixture_em_stopping_rule_is_scale_free() {
        // |l - l_prev| is invariant to rescaling (l shifts by -n ln c); 2^k scales exactly.
        let x = x1(&csv("mixture_3d"));
        let base = normal_mix_em(&x, 3, None, None, None).unwrap();
        for c in [2f64.powi(-20), 2f64.powi(20)] {
            let xs: Vec<f64> = x.iter().map(|v| v * c).collect();
            let r = normal_mix_em(&xs, 3, None, None, None).unwrap();
            assert!(
                r.n_iterations.abs_diff(base.n_iterations) <= 1,
                "{} vs {}",
                r.n_iterations,
                base.n_iterations
            );
            let want = base.loglik - 300.0 * c.ln();
            assert!((r.loglik - want).abs() <= 1e-11 * want.abs());
        }
    }

    #[test]
    fn test_normal_mix_em_restarts_when_a_variance_collapses() {
        // Three tied values far from the rest: a component on them has variance 0 (mixtools
        // restarts).
        let mut x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.618).sin()).collect();
        x.extend([10.0, 10.0, 10.0]);
        let mut cfg = NormalMixConfig::new(2);
        cfg.start = Some(NormalMixStart {
            lambda: vec![0.9, 0.1],
            mu: vec![0.0, 10.0],
            sigma: vec![1.0, 1.0],
        });
        cfg.seed = Some(3);
        let outcome = |r: EconResult<NormalMixEMResult>| match r {
            Ok(r) => format!(
                "ok loglik={} restarts={} warnings={:?}",
                r.loglik, r.restarts, r.warnings
            ),
            Err(e) => e.to_string(),
        };
        let (a, b) = (
            outcome(normal_mix_em_with(&x, &cfg)),
            outcome(normal_mix_em_with(&x, &cfg)),
        );
        assert_eq!(a, b, "seeded restarts are reproducible");
        assert!(a.contains("restart"), "{a}");
        assert!(!a.contains("restarts=0 "), "{a}");
    }

    #[test]
    fn test_mvnorm_mix_em_default_seed_is_reproducible() {
        let x = case_matrix(&csv("mixture_3d"), &["x1", "x2", "x3"]);
        let a = mvnorm_mix_em(x.view(), 3, Some(50), None, None).unwrap();
        let b = mvnorm_mix_em(x.view(), 3, Some(50), None, Some(MIXTURE_DEFAULT_SEED)).unwrap();
        assert_eq!(a.loglik.to_bits(), b.loglik.to_bits());
    }

    #[test]
    fn test_mixtools_rejects_bad_parameters() {
        let x: Vec<f64> = (0..20)
            .map(|i| (i as f64 * 0.7).sin() + if i > 9 { 4.0 } else { 0.0 })
            .collect();
        let e = |r: Result<NormalMixEMResult, String>| r.unwrap_err();
        assert!(e(normal_mix_em(&x, 0, None, None, None)).contains("k must"));
        assert!(e(normal_mix_em(&x, 21, None, None, None)).contains("k must"));
        assert!(e(normal_mix_em(&x, 2, Some(0), None, None)).contains("max_iterations"));
        assert!(e(normal_mix_em(&x, 2, None, Some(f64::NAN), None)).contains("tolerance"));
        assert!(e(normal_mix_em(&[1.0, f64::NAN, 2.0], 1, None, None, None)).contains("finite"));
        let mut cfg = NormalMixConfig::new(2);
        cfg.start = Some(NormalMixStart {
            lambda: vec![0.5, 0.5],
            mu: vec![0.0, 4.0],
            sigma: vec![1.0, 0.0],
        });
        assert!(
            normal_mix_em_with(&x, &cfg)
                .unwrap_err()
                .to_string()
                .contains("sigma")
        );
        let xm = Array2::from_shape_fn((20, 2), |(i, j)| x[i] + j as f64);
        let mut mc = MvNormalMixConfig::new(2);
        mc.start = Some(MvNormalMixStart {
            lambda: vec![0.5, 0.5],
            mu: vec![vec![0.0, 0.0], vec![4.0, 4.0]],
            sigma: vec![vec![vec![1.0, 2.0], vec![2.0, 1.0]]; 2], // not PD
        });
        assert!(
            mvnorm_mix_em_with(xm.view(), &mc)
                .unwrap_err()
                .to_string()
                .contains("positive definite")
        );
    }

    fn flexmix_data(ds: &crate::data::Dataset) -> (Array1<f64>, Array2<f64>, Vec<usize>) {
        let col = |c: &str| numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0;
        let xv = col("x");
        let x = Array2::from_shape_fn((xv.len(), 2), |(i, j)| if j == 0 { 1.0 } else { xv[i] });
        (
            Array1::from_vec(col("y")),
            x,
            col("start").iter().map(|&s| s as usize - 1).collect(),
        )
    }

    fn check_flex(case: &Case, tag: &str, r: &FlexMixResult, tol: Tol, ll_tol: Tol) {
        let o = order_by(&r.coefficients.iter().map(|b| b[0]).collect::<Vec<_>>());
        let coef: Vec<f64> = o.iter().flat_map(|&k| r.coefficients[k].clone()).collect();
        let k = |s: &str| format!("{tag}_{s}");
        Check::new(case)
            .scalar(&k("k"), r.k as f64, Tol::exact())
            .scalar(&k("k0"), r.k_initial as f64, Tol::exact())
            .scalar(&k("loglik"), r.loglik, ll_tol)
            .scalar(&k("df"), r.n_parameters as f64, Tol::exact())
            .scalar(&k("aic"), r.aic, ll_tol)
            .scalar(&k("bic"), r.bic, ll_tol)
            .vector(&k("prior"), &pick(&r.prior, &o), tol)
            .vector(&k("coef"), &coef, tol)
            .vector(&k("sigma"), &pick(&r.sigma, &o), tol)
            .vector(&k("cluster"), &relabel(&r.cluster, &o), Tol::exact())
            .finish();
        // not pinned: iteration path (iterations, the iteration of a removal).
    }

    #[test]
    fn test_validate_flexmix_minprior() {
        let (case, ds) = crate::test_support::load_case("flexmix_minprior");
        let (y, x, start) = flexmix_data(&ds);
        let mut cfg = FlexMixConfig::new(3);
        cfg.start = FlexMixStart::Clusters(start);
        cfg.minprior = 0.1;
        cfg.tolerance = 1e-12;
        cfg.max_iterations = 5000;
        let r = flexmix_with(y.view(), x.view(), &cfg).unwrap();
        assert!(r.converged);
        assert_eq!(r.removed_components.len(), 1);
        assert_eq!(r.removed_components[0].component, 2); // the third start component
        assert!(r.removed_components[0].prior < cfg.minprior);
        assert!(
            r.warnings.iter().any(|w| w.contains("removed component 3")),
            "{:?}",
            r.warnings
        );
        check_flex(&case, "prune", &r, FLEX_PRUNE, Tol::new(1e-12, 0.0));
    }

    #[test]
    fn test_validate_flexmix_keep() {
        let (case, ds) = crate::test_support::load_case("flexmix_minprior");
        let (y, x, start) = flexmix_data(&ds);
        let mut cfg = FlexMixConfig::new(3); // minprior 0.05 (flexmix's default)
        cfg.start = FlexMixStart::Clusters(start);
        cfg.tolerance = 1e-12;
        cfg.max_iterations = 5000;
        let r = flexmix_with(y.view(), x.view(), &cfg).unwrap();
        assert!(r.converged && r.removed_components.is_empty());
        check_flex(&case, "keep", &r, FLEX_KEEP, Tol::new(1e-11, 0.0));
    }

    #[test]
    fn test_flexmix_flags_degenerate_components() {
        // ML-53's scenario: 30 points from two lines, k = 3, random starts. Before: 6 of 20 seeds
        // returned a component with sigma ~3e-8 and converged = true, without any flag.
        let (y, x, _) = flexmix_data(&csv("flexmix_lines"));
        let (y30, x30) = (
            y.slice(ndarray::s![..30]).to_owned(),
            x.slice(ndarray::s![..30, ..]).to_owned(),
        );
        let sd = 1.5681710772805115; // sd(y[1:30]) in R
        for seed in 0..20 {
            let r = flexmix(
                y30.view().insert_axis(Axis(1)),
                x30.view(),
                3,
                None,
                None,
                Some(seed),
            )
            .unwrap();
            for (j, &s) in r.sigma.iter().enumerate() {
                assert!(
                    s >= FLEXMIX_SIGMA_FLOOR_REL * sd * (1.0 - 1e-12),
                    "seed {seed}: sigma {s}"
                );
                assert!(
                    s > 1e-4 * sd || r.warnings.iter().any(|w| w.contains("degenerate")),
                    "seed {seed}: component {j} sigma {s} unflagged"
                );
            }
        }
    }

    #[test]
    fn test_flexmix_exact_fit_is_floored_and_flagged() {
        // Component 3 starts on rows 1-2 only: the weighted fit is exact there. R flexmix stops
        // with "Log-likelihood: NaN" (stopifnot in the case); p2a floors sigma and warns.
        let (y, x, start) = flexmix_data(&csv("flexmix_lines"));
        let mut p = Array2::zeros((200, 3));
        for i in 0..200 {
            p[[i, if start[i] == 2 { 0 } else { start[i] }]] = 1.0;
        }
        for i in 0..2 {
            p.row_mut(i).fill(0.0);
            p[[i, 2]] = 1.0;
        }
        let mut cfg = FlexMixConfig::new(3);
        cfg.start = FlexMixStart::Posterior(p);
        cfg.minprior = 0.0;
        cfg.max_iterations = 50;
        let r = flexmix_with(y.view(), x.view(), &cfg).unwrap();
        let floor = FLEXMIX_SIGMA_FLOOR_REL * 1.5302207343058354; // sd(y) in R
        assert!(
            (r.sigma[2] - floor).abs() <= 1e-12 * floor,
            "{}",
            r.sigma[2]
        );
        assert_eq!(r.degenerate_components, vec![2]);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("component 3 is degenerate")),
            "{:?}",
            r.warnings
        );
        assert!(r.loglik.is_finite());
    }

    #[test]
    fn test_flexmix_rejects_bad_parameters() {
        let (y, x, _) = flexmix_data(&csv("flexmix_lines"));
        let run = |f: &dyn Fn(&mut FlexMixConfig)| {
            let mut c = FlexMixConfig::new(3);
            f(&mut c);
            flexmix_with(y.view(), x.view(), &c)
                .unwrap_err()
                .to_string()
        };
        assert!(run(&|c| c.k = 0).contains("k must"));
        assert!(run(&|c| c.k = 201).contains("k must"));
        assert!(run(&|c| c.max_iterations = 0).contains("max_iterations"));
        assert!(run(&|c| c.tolerance = 0.0).contains("tolerance"));
        assert!(run(&|c| c.minprior = 1.0).contains("minprior"));
        assert!(run(&|c| c.minprior = f64::NAN).contains("minprior"));
        assert!(run(&|c| c.start = FlexMixStart::Clusters(vec![0; 5])).contains("length"));
        assert!(run(&|c| c.start = FlexMixStart::Clusters(vec![3; 200])).contains("cluster"));
        assert!(
            run(&|c| c.start = FlexMixStart::Posterior(Array2::zeros((200, 3))))
                .contains("posterior")
        );
        assert!(run(&|c| c.minprior = 0.9).contains("all components removed")); // flexmix stops the same way
        let y2 = Array2::from_shape_fn((200, 2), |(i, _)| y[i]);
        assert!(
            flexmix(y2.view(), x.view(), 2, None, None, Some(1))
                .unwrap_err()
                .contains("one column")
        );
        let tiny = flexmix_with(
            y.slice(ndarray::s![..2]),
            x.slice(ndarray::s![..2, ..]),
            &FlexMixConfig::new(1),
        );
        assert!(tiny.unwrap_err().to_string().contains("observations"));
    }

    #[test]
    fn test_flexmix_rank_deficient_component_is_refused() {
        // Two identical columns: the QR that the normal equations fall back to names the rank.
        let t: Vec<f64> = (0..30).map(|i| (i as f64 * 0.9).sin()).collect();
        let x = Array2::from_shape_fn((30, 2), |(i, _)| t[i]);
        let y = Array1::from_shape_fn(30, |i| t[i] + 0.1 * (i as f64 * 1.7).cos());
        let e = flexmix_with(y.view(), x.view(), &FlexMixConfig::new(1)).unwrap_err();
        assert!(e.to_string().contains("rank deficient"), "{e}");
    }

    #[test]
    fn test_flexmix_basic() {
        // Create mixture regression data: two groups with different slopes
        // Group 1: y = 1 + 2*x + noise
        // Group 2: y = 5 - 1*x + noise
        let x = array![
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
        ];
        let y = array![
            [2.1], // Group 1: 1 + 2*0.5 + noise
            [3.0], // Group 1: 1 + 2*1.0 + noise
            [4.1], // Group 1: 1 + 2*1.5 + noise
            [5.0], // Group 1: 1 + 2*2.0 + noise
            [4.4], // Group 2: 5 - 1*0.5 + noise
            [3.9], // Group 2: 5 - 1*1.0 + noise
            [3.6], // Group 2: 5 - 1*1.5 + noise
            [3.1], // Group 2: 5 - 1*2.0 + noise
        ];

        let mut cfg = FlexMixConfig::new(2);
        cfg.start = FlexMixStart::Clusters(vec![0, 0, 0, 0, 1, 1, 1, 1]);
        cfg.max_iterations = 100;
        cfg.tolerance = 1e-4;
        let result = flexmix_with(y.column(0), x.view(), &cfg).unwrap();

        assert_eq!(result.cluster.len(), 8);
        assert_eq!(result.k, 2);
        assert_eq!(result.p, 2);
        assert_eq!(result.coefficients.len(), 2);
        assert!((result.prior.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!(result.bic.is_finite());
        assert!(result.aic.is_finite());

        // Random starts on 8 points: minprior may legitimately remove a component.
        let smoke = flexmix(y.view(), x.view(), 2, Some(100), Some(1e-4), Some(42)).unwrap();
        assert!(smoke.k <= 2);
        assert!(smoke.bic.is_finite());
    }

    #[test]
    fn test_flexmix_single_component() {
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0],];
        let y = array![[2.1], [4.0], [5.9], [8.1],];

        let result = flexmix(y.view(), x.view(), 1, Some(50), None, Some(42)).unwrap();

        assert_eq!(result.k, 1);
        assert_eq!(result.prior.len(), 1);
        assert!((result.prior[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normal_mix_em_basic() {
        // Create bimodal data
        let data: Vec<f64> = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, // Cluster around 0.3
            9.5, 9.6, 9.7, 9.8, 9.9, // Cluster around 9.7
        ];

        let result = normal_mix_em(&data, 2, Some(100), Some(1e-6), None).unwrap();

        assert_eq!(result.k, 2);
        assert_eq!(result.n, 10);
        assert_eq!(result.mu.len(), 2);
        assert_eq!(result.sigma.len(), 2);
        assert_eq!(result.lambda.len(), 2);
        assert_eq!(result.posterior.nrows(), 10);
        assert_eq!(result.posterior.ncols(), 2);

        // Mixing proportions should sum to 1
        let lambda_sum: f64 = result.lambda.iter().sum();
        assert!((lambda_sum - 1.0).abs() < 1e-6);

        // Means should be separated
        let mu_diff = (result.mu[0] - result.mu[1]).abs();
        assert!(
            mu_diff > 5.0,
            "Means should be well separated: {:?}",
            result.mu
        );
    }

    #[test]
    fn test_normal_mix_em_convergence() {
        let data: Vec<f64> = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2];

        let result = normal_mix_em(&data, 2, Some(500), Some(1e-8), None).unwrap();

        // Should converge with reasonable tolerance
        assert!(result.n_iterations <= 500);
        // Log-likelihood should increase (roughly)
        if result.loglik_history.len() > 1 {
            let first = result.loglik_history[0];
            let last = *result.loglik_history.last().unwrap();
            assert!(
                last >= first - 1e-6,
                "Log-likelihood should not decrease significantly"
            );
        }
    }

    #[test]
    fn test_mvnorm_mix_em_basic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];

        let result = mvnorm_mix_em(data.view(), 2, Some(100), Some(1e-6), Some(42)).unwrap();

        assert_eq!(result.k, 2);
        assert_eq!(result.n, 6);
        assert_eq!(result.d, 2);
        assert_eq!(result.mu.nrows(), 2);
        assert_eq!(result.mu.ncols(), 2);
        assert_eq!(result.labels.len(), 6);

        // Lambda should sum to 1
        let lambda_sum: f64 = result.lambda.iter().sum();
        assert!((lambda_sum - 1.0).abs() < 1e-6);

        // Points should be assigned to correct clusters
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }
}
