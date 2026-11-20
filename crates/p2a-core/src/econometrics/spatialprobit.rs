//! Spatial Probit Models for Binary Outcomes with Spatial Dependence.
//!
//! Implements spatial autoregressive (SAR) probit and spatial error (SEM) probit models
//! for binary dependent variables with spatial dependence, by the Bayesian Gibbs sampler of
//! LeSage & Pace (2009, ch. 10), the scheme of R's `spatialprobit` (`sarprobit`, `semprobit`).
//! The deterministic pieces (the full conditionals, the marginal probabilities and the
//! impacts at given parameters) are pinned to R written out in base R; the posterior itself
//! is not R-pinned (an MCMC run cannot reproduce spatialprobit's draws) and is checked by
//! recovering simulated parameters. See `validation/econometrics/spatialprobit.md`.
//!
//! # Models
//!
//! ## SAR Probit Model
//!
//! The spatial autoregressive probit model is:
//!
//! ```text
//! y* = rho * W * y* + X * beta + epsilon,  epsilon ~ N(0, I)
//! y_i = 1 if y*_i > 0, else y_i = 0
//! ```
//!
//! where rho is the spatial autoregressive parameter and W is the spatial weights matrix.
//! This model captures spatial spillover effects in binary outcomes.
//!
//! ## SEM Probit Model
//!
//! The spatial error probit model is:
//!
//! ```text
//! y* = X * beta + u,  u = lambda * W * u + epsilon,  epsilon ~ N(0, I)
//! y_i = 1 if y*_i > 0, else y_i = 0
//! ```
//!
//! where lambda is the spatial error parameter. This model accounts for spatial
//! correlation in the error term.
//!
//! # Data requirements
//!
//! Row i of the dataset is unit i of the weights: one row per unit of W, and no row can be
//! dropped. Missing values (null or NaN) in y or X are refused with
//! `EconError::NullValues` naming the column (spatialprobit, like spatialreg, has no
//! `na.omit` route); remove the incomplete rows and rebuild W first. Integer and Boolean
//! columns are accepted. The outcome is exactly 0/1: an Int64, Float64 or Boolean column with
//! both values present; any other value is refused, not dichotomised.
//!
//! `n_draws` must be at least 2 (a posterior standard deviation needs two retained draws),
//! `thin` at least 1, and `compute_impacts` needs at least one regressor besides the intercept.
//!
//! # Estimation Method
//!
//! Gibbs sampling with data augmentation (Albert & Chib 1993), A = I - ρW:
//!
//! 1. The latent outcomes z, one single-site sweep per iteration: z_i given the others is
//!    N(m_i, 1/H_ii) with H = A'A, m_i = z_i - (A'(A z - c))_i / H_ii (c = Xβ for SAR, AXβ for
//!    SEM), truncated to (0, ∞) where y_i = 1 and (-∞, 0] where y_i = 0. Each draw is exact
//!    (normal rejection near the mean, Robert's (1995) exponential rejection in the far tail),
//!    and a sweep costs O(links) with no solve. spatialprobit runs m = 10 inner sweeps of its
//!    sparse truncated-multivariate-normal sampler per iteration; both are valid Gibbs schemes.
//! 2. β | z, ρ: conjugate normal (prior precision `prior_beta_precision`, default 0.01).
//! 3. ρ | z, β: griddy Gibbs on 2,000 cells of the prior range (intersected with the range
//!    where I - ρW is invertible), with the log kernel ln|I - ρW| - ||A z - c||² / 2;
//!    ln|I - ρW| comes from the eigenvalues of W once per fit, and the draw inverts the
//!    trapezoid CDF within its cell. There is no rejection step, so `acceptance_rate` is 1.
//!
//! The summaries at the posterior mean use the marginal distribution of each latent outcome,
//! z_i ~ N(μ_i, σ_i²) with μ = A⁻¹Xβ (SAR) or Xβ (SEM) and σ_i² = [(A'A)⁻¹]_ii (one dense
//! inverse of A): `fitted_prob` = Φ(μ_i/σ_i), the composite log-likelihood Σ ln Φ(±μ_i/σ_i)
//! (the joint likelihood is an n-dimensional orthant probability, which is not computed), the
//! pseudo R² and the percent correctly predicted. `dic` and `log_marginal_likelihood` are NaN.
//!
//! # Marginal Effects
//!
//! Per posterior draw (β, ρ), with A = I - ρW and the intercept excluded:
//!
//! - **SAR** (spatialprobit's and LeSage & Pace's convention): μ = A⁻¹Xβ, φ_i = dnorm(μ_i),
//!   direct_k = β_k mean_i(φ_i [A⁻¹]_ii), total_k = β_k mean_i(φ_i [A⁻¹1]_i), indirect =
//!   total - direct.
//! - **SEM**: the derivative of the reported probability Φ(μ_i/σ_i), μ = Xβ:
//!   direct_k = β_k mean_i(φ(μ_i/σ_i)/σ_i); a change in x_j moves no other unit's probability,
//!   so indirect is 0 and total = direct.
//!
//! The reported impacts are the posterior means and SDs over the draws. A⁻¹ is exact, not a
//! truncated series: the dense inverse is formed at Chebyshev nodes over the range of the ρ
//! draws (9, then 17 or 33) and interpolated barycentrically to each draw. A node count is
//! accepted when the interpolant agrees to 1e-10 with the next coarser one at the midpoints of
//! its cells, or else with the dense inverse at the midpoints of its two end cells (where the
//! error of Chebyshev interpolation peaks; a heuristic, not a bound over the whole range); a
//! warning reports the accuracy when 33 nodes miss it.
//!
//! # References
//!
//! - LeSage, J.P. & Pace, R.K. (2009). "Introduction to Spatial Econometrics".
//!   CRC Press. Chapter 10: Spatial Probit and Logit Models.
//!   ISBN: 978-1420064247.
//!
//! - Albert, J.H. & Chib, S. (1993). "Bayesian analysis of binary and polychotomous
//!   response data". *Journal of the American Statistical Association*, 88(422), 669-679.
//!   https://doi.org/10.1080/01621459.1993.10476321
//!
//! - Geweke, J. (1991). "Efficient simulation from the multivariate normal and
//!   Student-t distributions subject to linear constraints and the evaluation
//!   of constraint probabilities". Computing Science and Statistics: Proc. 23rd
//!   Symposium on the Interface, 571-578.
//!
//! - Beron, K.J. & Vijverberg, W.P.M. (2004). "Probit in a Spatial Context:
//!   A Monte Carlo Analysis". In Anselin, L., Florax, R.J.G.M. & Rey, S.J. (Eds.),
//!   Advances in Spatial Econometrics (pp. 169-195). Springer.
//!
//! - Robert, C.P. (1995). "Simulation of truncated normal variables". *Statistics and
//!   Computing*, 5(2), 121-125. https://doi.org/10.1007/BF00143942
//!
//! - R package `spatialprobit`: Wilhelm, S. & de Matos, M.G. (2013).
//!   https://cran.r-project.org/package=spatialprobit

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{matrix_inverse, xtx, xty};
use crate::resource::{try_zeros1, try_zeros2};
use crate::spatial::SpatialWeights;
use crate::stats::robust::percentile_interval;
use crate::traits::estimator::{normal_cdf, normal_p_value_two_sided, normal_pdf};

/// Model type for spatial probit estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialProbitModel {
    /// SAR probit: y* = rho*W*y* + X*beta + epsilon
    Sar,
    /// SEM probit: y* = X*beta + u, u = lambda*W*u + epsilon
    Sem,
}

impl std::fmt::Display for SpatialProbitModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpatialProbitModel::Sar => write!(f, "SAR Probit"),
            SpatialProbitModel::Sem => write!(f, "SEM Probit"),
        }
    }
}

/// The seed of a run whose configuration sets none.
pub const SPATIAL_PROBIT_DEFAULT_SEED: u64 = 42;

/// Configuration for spatial probit model estimation.
#[derive(Debug, Clone)]
pub struct SpatialProbitConfig {
    /// Number of MCMC draws (after burn-in)
    pub n_draws: usize,
    /// Number of burn-in draws to discard
    pub burn_in: usize,
    /// Thinning interval (keep every nth draw)
    pub thin: usize,
    /// Prior mean for beta (default: 0)
    pub prior_beta_mean: Option<Array1<f64>>,
    /// Prior precision for beta (default: 0.01 * I)
    pub prior_beta_precision: Option<f64>,
    /// Prior for spatial parameter: uniform on (a, b)
    /// Default: (-0.99, 0.99) for row-standardized weights
    pub rho_prior_range: (f64, f64),
    /// Compute marginal effects
    pub compute_impacts: bool,
    /// Random seed; `None` uses [`SPATIAL_PROBIT_DEFAULT_SEED`], so every run is reproducible.
    pub seed: Option<u64>,
    /// Show progress (for long MCMC runs)
    pub verbose: bool,
}

impl Default for SpatialProbitConfig {
    fn default() -> Self {
        Self {
            n_draws: 1000,
            burn_in: 200,
            thin: 1,
            prior_beta_mean: None,
            prior_beta_precision: Some(0.01),
            rho_prior_range: (-0.99, 0.99),
            compute_impacts: true,
            seed: None,
            verbose: false,
        }
    }
}

/// Spatial marginal effects for probit models.
///
/// In spatial probit models, the effect of a change in x_k on the probability
/// of y=1 involves both direct and indirect (spillover) effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialProbitImpacts {
    /// Direct effects: average effect on own probability
    pub direct: Vec<f64>,
    /// Indirect effects: average spillover effect
    pub indirect: Vec<f64>,
    /// Total effects: direct + indirect
    pub total: Vec<f64>,
    /// Standard errors for direct effects
    pub direct_se: Vec<f64>,
    /// Standard errors for indirect effects
    pub indirect_se: Vec<f64>,
    /// Standard errors for total effects
    pub total_se: Vec<f64>,
    /// Variable names
    pub var_names: Vec<String>,
    /// Notes on the computation (the interpolation accuracy when it misses its target)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Result from spatial probit model estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialProbitResult {
    /// Model type (SAR or SEM)
    pub model_type: SpatialProbitModel,

    /// Posterior mean of beta coefficients
    pub coefficients: Vec<f64>,
    /// Coefficient names
    pub coef_names: Vec<String>,
    /// Posterior standard deviation of beta
    pub std_errors: Vec<f64>,
    /// Z-values (coef / se)
    pub z_values: Vec<f64>,
    /// Two-sided p-values
    pub p_values: Vec<f64>,

    /// Posterior mean of spatial parameter (rho for SAR, lambda for SEM)
    pub rho: f64,
    /// Posterior standard deviation of spatial parameter
    pub rho_se: f64,
    /// Z-value for spatial parameter
    pub rho_z: f64,
    /// P-value for spatial parameter
    pub rho_p: f64,

    /// Composite (marginal) log-likelihood at the posterior mean, Σ_i ln Φ(±μ_i / σ_i) with
    /// σ_i² = [((I - ρW)'(I - ρW))^-1]_ii; not the joint likelihood, an n-dimensional orthant
    /// probability
    pub log_likelihood: f64,
    /// Not computed (NaN): the joint likelihood is an orthant probability, and the harmonic
    /// mean estimator is unreliable; spatialprobit reports none
    pub log_marginal_likelihood: f64,
    /// Not computed (NaN), for the same reason; spatialprobit reports none
    pub dic: f64,
    /// McFadden's pseudo R-squared of the composite log-likelihood against the
    /// intercept-only probit
    pub pseudo_r_squared: f64,

    /// Marginal probabilities Φ(μ_i / σ_i) at the posterior mean
    pub fitted_prob: Vec<f64>,
    /// Percent correctly predicted
    pub pcp: f64,

    /// Spatial impacts (if computed)
    pub impacts: Option<SpatialProbitImpacts>,

    /// Number of observations
    pub n_obs: usize,
    /// Number of positive outcomes (y=1)
    pub n_positive: usize,
    /// Number of MCMC draws used
    pub n_draws: usize,
    /// Always 1: the griddy-Gibbs step for the spatial parameter has no rejection step
    pub acceptance_rate: f64,

    /// MCMC diagnostics
    #[serde(skip)]
    pub beta_draws: Array2<f64>,
    #[serde(skip)]
    pub rho_draws: Array1<f64>,
}

/// The R type-7 credible interval of `draws`; NaN for a level outside (0, 1) or no draws.
fn credible_interval(draws: &[f64], level: f64) -> (f64, f64) {
    percentile_interval(draws, level).unwrap_or((f64::NAN, f64::NAN))
}

impl SpatialProbitResult {
    /// The central credible interval of coefficient `idx` at `level` (R type-7 quantiles of the
    /// retained draws); NaN for an index that does not exist or a level outside (0, 1).
    pub fn credible_interval_beta(&self, idx: usize, level: f64) -> (f64, f64) {
        if idx >= self.beta_draws.ncols() {
            return (f64::NAN, f64::NAN);
        }
        credible_interval(&self.beta_draws.column(idx).to_vec(), level)
    }

    /// The central credible interval of the spatial parameter at `level` (R type-7 quantiles of
    /// the retained draws); NaN for a level outside (0, 1).
    pub fn credible_interval_rho(&self, level: f64) -> (f64, f64) {
        credible_interval(&self.rho_draws.to_vec(), level)
    }
}

impl std::fmt::Display for SpatialProbitImpacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Spatial impacts (probability scale; posterior mean and SD):"
        )?;
        if self.var_names.is_empty() {
            return writeln!(f, "(no regressors)");
        }
        writeln!(
            f,
            "{:>15} {:>12} {:>10} {:>12} {:>10} {:>12} {:>10}",
            "Variable", "Direct", "(SD)", "Indirect", "(SD)", "Total", "(SD)"
        )?;
        for (i, name) in self.var_names.iter().enumerate() {
            writeln!(
                f,
                "{name:>15} {:>12.6} {:>10.6} {:>12.6} {:>10.6} {:>12.6} {:>10.6}",
                self.direct[i],
                self.direct_se[i],
                self.indirect[i],
                self.indirect_se[i],
                self.total[i],
                self.total_se[i]
            )?;
        }
        for w in &self.warnings {
            writeln!(f, "Warning: {w}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for SpatialProbitResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let spatial = match self.model_type {
            SpatialProbitModel::Sar => "rho",
            SpatialProbitModel::Sem => "lambda",
        };
        writeln!(f, "{} (Bayesian MCMC)", self.model_type)?;
        writeln!(f, "Retained draws: {}", self.n_draws)?;
        writeln!(f)?;
        let (low, high) = self.credible_interval_rho(0.95);
        writeln!(
            f,
            "{spatial}: {:.6} (posterior SD {:.6}, z = {:.3}, p = {:.4}; 95% credible interval [{low:.6}, {high:.6}])",
            self.rho, self.rho_se, self.rho_z, self.rho_p
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Mean", "Std.Dev", "z", "P>|z|"
        )?;
        for (i, name) in self.coef_names.iter().enumerate() {
            writeln!(
                f,
                "{name:>15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                self.coefficients[i], self.std_errors[i], self.z_values[i], self.p_values[i]
            )?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Composite log-likelihood (marginal probabilities): {:.4}",
            self.log_likelihood
        )?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "Percent correctly predicted: {:.2}", self.pcp)?;
        writeln!(f, "N: {}", self.n_obs)?;
        if let Some(impacts) = &self.impacts {
            writeln!(f)?;
            write!(f, "{impacts}")?;
        }
        Ok(())
    }
}

/// Run SAR probit model: y* = rho*W*y* + X*beta + epsilon, y = 1(y* > 0).
///
/// Estimates a spatial autoregressive probit model using Bayesian MCMC with
/// data augmentation for the latent variable.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the variables
/// * `y_col` - Name of the binary dependent variable (0/1)
/// * `x_cols` - Names of the independent variables
/// * `listw` - Spatial weights matrix
/// * `config` - Model configuration
///
/// # Returns
///
/// Estimation results including posterior means, standard deviations, and impacts
///
/// # Example
///
/// ```ignore
/// use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
/// use p2a_core::econometrics::spatialprobit::{run_sar_probit, SpatialProbitConfig};
///
/// // Assuming dataset and coords are already set up
/// let nb = Neighbors::from_knn(&coords, 5);
/// let mut listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
///
/// let result = run_sar_probit(&dataset, "y", &["x1", "x2"], &mut listw,
///                             SpatialProbitConfig::default())?;
/// println!("rho = {} (SE = {})", result.rho, result.rho_se);
/// ```
pub fn run_sar_probit(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SpatialProbitConfig,
) -> EconResult<SpatialProbitResult> {
    run_spatial_probit_internal(
        dataset,
        y_col,
        x_cols,
        listw,
        config,
        SpatialProbitModel::Sar,
    )
}

/// Run SEM probit model: y* = X*beta + u, u = lambda*W*u + epsilon, y = 1(y* > 0).
///
/// Estimates a spatial error probit model using Bayesian MCMC.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the variables
/// * `y_col` - Name of the binary dependent variable (0/1)
/// * `x_cols` - Names of the independent variables
/// * `listw` - Spatial weights matrix
/// * `config` - Model configuration
///
/// # Returns
///
/// Estimation results including posterior means and spatial error parameter lambda
pub fn run_sem_probit(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SpatialProbitConfig,
) -> EconResult<SpatialProbitResult> {
    run_spatial_probit_internal(
        dataset,
        y_col,
        x_cols,
        listw,
        config,
        SpatialProbitModel::Sem,
    )
}

/// Main spatial probit estimation function (internal).
fn run_spatial_probit_internal(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SpatialProbitConfig,
    model_type: SpatialProbitModel,
) -> EconResult<SpatialProbitResult> {
    // Refuse impossible draw counts before any work.
    total_mcmc_draws(&config)?;
    if config.compute_impacts && x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "compute_impacts needs at least one regressor besides the intercept; pass x columns or set compute_impacts = false".to_string(),
        });
    }

    // y (0/1, integer or Boolean) and [1, X]; one row per unit of W, no missing values
    let (y, x) = super::spatial::spatial_model_data(dataset, y_col, x_cols, listw)?;
    let n = y.len();

    // Validate y is binary
    let is_non_binary = |v: f64| v != 0.0 && v != 1.0;
    if let Some(row) = y.iter().position(|&v| is_non_binary(v)) {
        let count = y.iter().filter(|&&v| is_non_binary(v)).count();
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dependent variable '{y_col}' must be binary (0/1); row {row} is {} ({count} non-binary values)",
                y[row]
            ),
        });
    }
    let n_positive = y.iter().filter(|&&v| v == 1.0).count();
    if n_positive == 0 || n_positive == n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dependent variable '{}' must be binary with both 0 and 1 values. Found {} ones out of {} observations.",
                y_col, n_positive, n
            ),
        });
    }

    // The latent full conditional uses H_ii = 1 + rho^2 sum_j w_ji^2, which holds only for a
    // zero diagonal.
    if let Some(i) = crate::spatial::first_self_link(listw) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "spatial probit: unit {} lists itself as a neighbour; W must have a zero diagonal",
                i + 1
            ),
        });
    }

    // Get valid range for spatial parameter
    let (rho_min, rho_max) = listw.try_rho_range()?;
    let (prior_lo, prior_hi) = config.rho_prior_range;
    if prior_lo.is_nan() || prior_hi.is_nan() {
        return Err(EconError::InvalidSpecification {
            message: format!("rho_prior_range ({prior_lo}, {prior_hi}) contains NaN"),
        });
    }
    let rho_range = (rho_min.max(prior_lo), rho_max.min(prior_hi));
    // The rho grid needs a non-empty, finite interval: a reversed prior, or one outside W's
    // range, would give a decreasing grid and meaningless draws.
    if !(rho_range.0.is_finite() && rho_range.1.is_finite() && rho_range.0 < rho_range.1) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "rho_prior_range ({prior_lo}, {prior_hi}) does not overlap the range \
                 ({rho_min}, {rho_max}) where I - rho W is invertible"
            ),
        });
    }

    // Run MCMC estimation
    let mcmc_result = run_mcmc_spatial_probit(&y, &x, listw, model_type, &config, rho_range)?;

    // Compute posterior statistics
    let n_draws = mcmc_result.beta_draws.nrows();
    let beta_mean: Array1<f64> = mcmc_result.beta_draws.mean_axis(Axis(0)).unwrap();
    let beta_std: Array1<f64> = compute_std_axis0(&mcmc_result.beta_draws);
    let rho_mean = mcmc_result.rho_draws.mean().unwrap_or(0.0);
    let rho_std = compute_std(&mcmc_result.rho_draws);

    // Z-values and p-values
    let z_values: Vec<f64> = beta_mean
        .iter()
        .zip(beta_std.iter())
        .map(|(&b, &se)| if se > 1e-10 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    let rho_z = if rho_std > 1e-10 {
        rho_mean / rho_std
    } else {
        0.0
    };
    let rho_p = normal_p_value_two_sided(rho_z);

    // Build coefficient names
    let mut coef_names = vec!["(Intercept)".to_string()];
    for &name in x_cols {
        coef_names.push(name.to_string());
    }

    // Marginal probabilities and the composite log-likelihood at the posterior mean
    let summary = marginal_summary(&y, &x, listw, &beta_mean, rho_mean, model_type)?;
    let ll_mean = summary.composite_log_likelihood;

    // Null log-likelihood (intercept only, no spatial dependence) and McFadden's pseudo R2
    let p_bar = n_positive as f64 / n as f64;
    let ll_null = (n_positive as f64) * p_bar.ln() + ((n - n_positive) as f64) * (1.0 - p_bar).ln();
    let pseudo_r2 = 1.0 - ll_mean / ll_null;

    let fitted_prob = summary.prob;
    let correct: usize = y
        .iter()
        .zip(fitted_prob.iter())
        .filter(|&(&yi, &pi)| (yi >= 0.5) == (pi >= 0.5))
        .count();
    let pcp = 100.0 * correct as f64 / n as f64;

    // Compute spatial impacts if requested
    let impacts = if config.compute_impacts {
        Some(compute_spatial_probit_impacts(
            &x,
            listw,
            &mcmc_result.beta_draws,
            &mcmc_result.rho_draws,
            model_type,
            &coef_names[1..], // Exclude intercept
        )?)
    } else {
        None
    };

    Ok(SpatialProbitResult {
        model_type,
        coefficients: beta_mean.to_vec(),
        coef_names,
        std_errors: beta_std.to_vec(),
        z_values,
        p_values,
        rho: rho_mean,
        rho_se: rho_std,
        rho_z,
        rho_p,
        log_likelihood: ll_mean,
        log_marginal_likelihood: f64::NAN,
        dic: f64::NAN,
        pseudo_r_squared: pseudo_r2,
        fitted_prob: fitted_prob.to_vec(),
        pcp,
        impacts,
        n_obs: n,
        n_positive,
        n_draws,
        acceptance_rate: 1.0,
        beta_draws: mcmc_result.beta_draws,
        rho_draws: mcmc_result.rho_draws,
    })
}

/// The number of Gibbs iterations, burn_in + n_draws * thin, after checking that at least
/// two draws are kept, the thinning interval is positive and the total does not overflow.
fn total_mcmc_draws(config: &SpatialProbitConfig) -> EconResult<usize> {
    let (n_draws, burn_in, thin) = (config.n_draws, config.burn_in, config.thin);
    if n_draws < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "n_draws must be at least 2 (posterior standard deviations need two retained draws), got {n_draws}"
            ),
        });
    }
    if thin == 0 {
        return Err(EconError::InvalidSpecification {
            message: "thin must be at least 1".to_string(),
        });
    }
    n_draws
        .checked_mul(thin)
        .and_then(|v| v.checked_add(burn_in))
        .ok_or_else(|| {
            EconError::ResourceLimit(format!(
                "burn_in + n_draws * thin overflows ({burn_in} + {n_draws} * {thin})"
            ))
        })
}

/// MCMC result structure (internal).
struct McmcResult {
    beta_draws: Array2<f64>,
    rho_draws: Array1<f64>,
}

/// Cells of the griddy-Gibbs grid for ρ (LeSage & Pace's spacing of about 0.001 on (-1, 1)).
const RHO_GRID_CELLS: usize = 2000;

/// The columns of W: `cols[i]` lists (j, w_ji) over the rows j with a link to i, so that a
/// column of I - ρW is available in O(links of i).
fn weight_columns(listw: &SpatialWeights) -> EconResult<Vec<Vec<(usize, f64)>>> {
    let n = listw.n();
    let mut cols: Vec<Vec<(usize, f64)>> = crate::resource::try_vec(n, Vec::new(), "columns of W")?;
    for j in 0..n {
        let sw = listw.get_weights(j);
        for (&i, &w) in sw.indices.iter().zip(sw.weights.iter()) {
            cols[i].push((j, w));
        }
    }
    Ok(cols)
}

/// c in the latent model A z = c + ε: X β (SAR) or A X β (SEM), A = I - ρW.
fn latent_offset(
    xb: &Array1<f64>,
    listw: &SpatialWeights,
    rho: f64,
    model_type: SpatialProbitModel,
) -> Array1<f64> {
    match model_type {
        SpatialProbitModel::Sar => xb.clone(),
        SpatialProbitModel::Sem => xb - &(rho * &listw.lag(xb)),
    }
}

/// The mean of the full conditional of z_i given the other latent values, with r = A z - c and
/// H = A'A: m_i = z_i - (A' r)_i / H_ii (the solve-free form of
/// μ_i - Σ_{j≠i} H_ij (z_j - μ_j) / H_ii); its variance is 1 / H_ii, H_ii = 1 + ρ² Σ_j w_ji².
fn latent_conditional_mean(
    i: usize,
    z_i: f64,
    r: &Array1<f64>,
    col: &[(usize, f64)],
    rho: f64,
    h_ii: f64,
) -> f64 {
    let g = r[i] - rho * col.iter().map(|&(j, w)| w * r[j]).sum::<f64>();
    z_i - g / h_ii
}

/// One single-site Gibbs sweep over the latent outcomes: each z_i is drawn from its full
/// conditional N(m_i, 1 / H_ii), truncated to (0, inf) where y_i = 1 and (-inf, 0] where
/// y_i = 0. The residual r = A z - c is formed exactly at the start and updated along column
/// i of A after each draw: O(links) per sweep, no solve.
#[allow(clippy::too_many_arguments)]
fn sample_latent_sweep<R: rand::Rng>(
    y: &Array1<f64>,
    z: &mut Array1<f64>,
    c: &Array1<f64>,
    listw: &SpatialWeights,
    cols: &[Vec<(usize, f64)>],
    col_sq: &[f64],
    rho: f64,
    rng: &mut R,
) {
    let mut r = &*z - &(rho * &listw.lag(z)) - c;
    for i in 0..z.len() {
        let h_ii = 1.0 + rho * rho * col_sq[i];
        let m = latent_conditional_mean(i, z[i], &r, &cols[i], rho, h_ii);
        let sd = h_ii.sqrt().recip();
        let new = if y[i] >= 0.5 {
            sample_truncated_normal_above(m, sd, 0.0, rng)
        } else {
            sample_truncated_normal_below(m, sd, 0.0, rng)
        };
        let dz = new - z[i];
        z[i] = new;
        r[i] += dz;
        for &(j, w) in &cols[i] {
            r[j] -= rho * w * dz;
        }
    }
}

/// The three inner products of the ρ full conditional: with a = z - Xβ and b = Wz (SAR) or
/// Wa (SEM), ||A z - c||² = a'a - 2ρ a'b + ρ² b'b.
fn rho_kernel_terms(
    z: &Array1<f64>,
    xb: &Array1<f64>,
    listw: &SpatialWeights,
    model_type: SpatialProbitModel,
) -> (f64, f64, f64) {
    let a = z - xb;
    let b = match model_type {
        SpatialProbitModel::Sar => listw.lag(z),
        SpatialProbitModel::Sem => listw.lag(&a),
    };
    (a.dot(&a), a.dot(&b), b.dot(&b))
}

/// ln k(ρ) = ln|I - ρW| - ||A z - c||² / 2, the log kernel of ρ | z, β under a flat prior.
fn rho_log_kernel(log_det: f64, rho: f64, (aa, ab, bb): (f64, f64, f64)) -> f64 {
    log_det - 0.5 * (aa - 2.0 * rho * ab + rho * rho * bb)
}

/// A draw from the density whose log is `log_k` at the equally spaced `grid` and linear
/// between neighbouring grid points: the trapezoid CDF inverted exactly within the cell
/// (LeSage & Pace's griddy Gibbs `draw_rho`). No rejection step, so every draw moves.
fn draw_from_grid<R: rand::Rng>(grid: &[f64], log_k: &[f64], rng: &mut R) -> f64 {
    let max = log_k.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // a cell below exp(-60) of the peak adds under 1e-23 of the total over the whole grid:
    // zero without the exponential, as most of the grid is for a concentrated posterior
    let f: Vec<f64> = log_k
        .iter()
        .map(|&l| {
            let d = l - max;
            if d < -60.0 { 0.0 } else { d.exp() }
        })
        .collect();
    let h = grid[1] - grid[0];
    let areas: Vec<f64> = f.windows(2).map(|p| 0.5 * h * (p[0] + p[1])).collect();
    let total: f64 = areas.iter().sum();
    let mut u = rng.r#gen::<f64>() * total;
    for (g, &area) in areas.iter().enumerate() {
        if u <= area || g + 1 == areas.len() {
            let u = u.min(area);
            // ∫_0^t h (f0 + (f1 - f0) s) ds = u, t in [0, 1]: the stable root of
            // (h (f1 - f0) / 2) t² + h f0 t - u = 0
            let (a, b) = (0.5 * h * (f[g + 1] - f[g]), h * f[g]);
            let disc = (b * b + 4.0 * a * u).max(0.0);
            let t = if u > 0.0 {
                (2.0 * u / (b + disc.sqrt())).clamp(0.0, 1.0)
            } else {
                0.0
            };
            return grid[g] + t * h;
        }
        u -= area;
    }
    grid[grid.len() - 1]
}

/// The griddy-Gibbs grid for ρ: the midpoints of [`RHO_GRID_CELLS`] equal cells on `range`.
fn rho_grid(range: (f64, f64)) -> Vec<f64> {
    let h = (range.1 - range.0) / RHO_GRID_CELLS as f64;
    (0..RHO_GRID_CELLS)
        .map(|g| range.0 + (g as f64 + 0.5) * h)
        .collect()
}

/// Run MCMC for the spatial probit (LeSage & Pace 2009, ch. 10): Gibbs sampling over
/// - the latent outcomes, one single-site sweep per iteration from their full conditional
///   (precision (I - ρW)'(I - ρW)), truncated by the observed outcome;
/// - β | z, ρ, conjugate normal;
/// - ρ | z, β, griddy Gibbs on 2,000 cells with ln|I - ρW| from the eigenvalues.
fn run_mcmc_spatial_probit(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &mut SpatialWeights,
    model_type: SpatialProbitModel,
    config: &SpatialProbitConfig,
    rho_range: (f64, f64),
) -> EconResult<McmcResult> {
    let k = x.ncols();
    let total_draws = total_mcmc_draws(config)?;

    use rand::SeedableRng;
    let mut rng =
        rand::rngs::StdRng::seed_from_u64(config.seed.unwrap_or(SPATIAL_PROBIT_DEFAULT_SEED));

    // Prior parameters
    let prior_precision = config.prior_beta_precision.unwrap_or(0.01);
    let beta0 = config
        .prior_beta_mean
        .clone()
        .unwrap_or_else(|| Array1::zeros(k));

    // Start: beta from OLS on the pseudo-outcome ±0.5 (consistent with y), rho = 0
    let pseudo_y: Array1<f64> = y.mapv(|yi| if yi >= 0.5 { 0.5 } else { -0.5 });
    let xtx_mat = xtx(&x.view());
    let xty_vec = xty(&x.view(), &pseudo_y);
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;
    let mut beta = xtx_inv.dot(&xty_vec);
    let mut rho = 0.0;
    let mut z = pseudo_y;

    // Once per fit: WX, the columns of W, ln|I - rho W| on the grid
    let mut wx = Array2::zeros((x.nrows(), k));
    for col in 0..k {
        wx.column_mut(col)
            .assign(&listw.lag(&x.column(col).to_owned()));
    }
    let cols = weight_columns(listw)?;
    let col_sq: Vec<f64> = cols
        .iter()
        .map(|c| c.iter().map(|&(_, w)| w * w).sum())
        .collect();
    let grid = rho_grid(rho_range);
    let log_det = listw.try_spectrum()?.log_det_grid(&grid);
    if !log_det.iter().all(|v| v.is_finite()) {
        return Err(EconError::Computation(
            "spatial probit: ln|I - rho W| is not finite on the rho grid".to_string(),
        ));
    }
    let mut log_k = vec![0.0; grid.len()];

    let keep_draws = config.n_draws;
    let mut beta_draws = try_zeros2(keep_draws, k, "posterior draws of beta")?;
    let mut rho_draws = try_zeros1(keep_draws, "posterior draws of rho")?;
    let mut draw_idx = 0;

    for iter in 0..total_draws {
        crate::resource::checkpoint()?;

        // --- z | beta, rho ---
        let xb = x.dot(&beta);
        let c = latent_offset(&xb, listw, rho, model_type);
        sample_latent_sweep(y, &mut z, &c, listw, &cols, &col_sq, rho, &mut rng);

        // --- beta | z, rho ---
        beta = sample_beta(
            &z,
            x,
            &wx,
            listw,
            rho,
            model_type,
            &beta0,
            prior_precision,
            &mut rng,
        )?;
        if !beta.iter().all(|v| v.is_finite()) {
            return Err(EconError::Computation(format!(
                "spatial probit: non-finite coefficient draw at iteration {iter}"
            )));
        }

        // --- rho | z, beta (griddy Gibbs) ---
        let terms = rho_kernel_terms(&z, &x.dot(&beta), listw, model_type);
        for ((lk, &ld), &r) in log_k.iter_mut().zip(&log_det).zip(&grid) {
            *lk = rho_log_kernel(ld, r, terms);
        }
        rho = draw_from_grid(&grid, &log_k, &mut rng);

        if iter >= config.burn_in
            && (iter - config.burn_in).is_multiple_of(config.thin)
            && draw_idx < keep_draws
        {
            beta_draws.row_mut(draw_idx).assign(&beta);
            rho_draws[draw_idx] = rho;
            draw_idx += 1;
        }
    }

    Ok(McmcResult {
        beta_draws,
        rho_draws,
    })
}

/// Sample beta from conditional posterior.
#[allow(clippy::too_many_arguments)]
fn sample_beta<R: rand::Rng>(
    y_star: &Array1<f64>,
    x: &Array2<f64>,
    wx: &Array2<f64>,
    listw: &SpatialWeights,
    rho: f64,
    model_type: SpatialProbitModel,
    beta0: &Array1<f64>,
    prior_precision: f64,
    rng: &mut R,
) -> EconResult<Array1<f64>> {
    let k = x.ncols();

    // Transform y* based on model type
    let (y_tilde, x_tilde) = match model_type {
        SpatialProbitModel::Sar => {
            // (I - rho*W)*y* = X*beta + epsilon
            let y_t: Array1<f64> = y_star - rho * &listw.lag(y_star);
            (y_t, x.clone())
        }
        SpatialProbitModel::Sem => {
            // y* = X*beta + (I - lambda*W)^{-1}*epsilon
            // Transform: (I - lambda*W)*y* = (I - lambda*W)*X*beta + epsilon
            let y_t: Array1<f64> = y_star - rho * &listw.lag(y_star);
            let x_t = x - &(rho * wx);
            (y_t, x_t)
        }
    };

    // Posterior precision: X'X + prior_precision * I
    let mut post_precision = xtx(&x_tilde.view());
    for i in 0..k {
        post_precision[[i, i]] += prior_precision;
    }

    // Posterior mean: post_var * (X'y_tilde + prior_precision * beta0)
    let xty_vec = xty(&x_tilde.view(), &y_tilde);
    let post_mean_unnorm = &xty_vec + &beta0.mapv(|b| b * prior_precision);

    let post_var = matrix_inverse(&post_precision.view())?;
    let post_mean = post_var.dot(&post_mean_unnorm);

    // beta = post_mean + L e, L L' = post_var
    let l = crate::linalg::matrix_ops::cholesky(&post_var.view())?;
    let e: Array1<f64> = Array1::from_iter((0..k).map(|_| {
        use rand_distr::{Distribution, StandardNormal};
        StandardNormal.sample(rng)
    }));
    Ok(&post_mean + &l.dot(&e))
}

/// ln Φ(x), accurate in both tails: ln(1 - Φ(-x)) above 0, ln Φ(x) from the upper-tail
/// function down to -30, and the asymptotic series below.
fn ln_normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        (-0.5 * normal_p_value_two_sided(x)).ln_1p()
    } else if x > -30.0 {
        (0.5 * normal_p_value_two_sided(x)).ln()
    } else {
        // Φ(x) = φ(x)/|x| (1 - 1/x² + 3/x⁴ - 15/x⁶ + 105/x⁸ - ...), error below 1e-12 here
        let q = (x * x).recip();
        -0.5 * x * x - (-x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            + (1.0 - q * (1.0 - q * (3.0 - q * (15.0 - 105.0 * q)))).ln()
    }
}

/// The marginal summaries at (β, ρ): σ_i² = [((I - ρW)'(I - ρW))^-1]_ii, μ = (I - ρW)^-1 Xβ
/// (SAR) or Xβ (SEM), the probabilities Φ(μ_i / σ_i) and the composite (marginal)
/// log-likelihood Σ_i ln Φ(±μ_i / σ_i). One dense inverse of I - ρW.
struct MarginalSummary {
    sigma: Array1<f64>,
    prob: Array1<f64>,
    composite_log_likelihood: f64,
}

fn marginal_summary(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    beta: &Array1<f64>,
    rho: f64,
    model_type: SpatialProbitModel,
) -> EconResult<MarginalSummary> {
    let parts = listw.dense_inverse_parts(rho, x.view())?;
    let sigma = parts.row_sq_norms.mapv(f64::sqrt);
    let mu = match model_type {
        SpatialProbitModel::Sar => parts.solved.dot(beta),
        SpatialProbitModel::Sem => x.dot(beta),
    };
    let t = &mu / &sigma;
    let prob = t.mapv(normal_cdf);
    let composite_log_likelihood = y
        .iter()
        .zip(t.iter())
        .map(|(&yi, &ti)| ln_normal_cdf(if yi >= 0.5 { ti } else { -ti }))
        .sum();
    Ok(MarginalSummary {
        sigma,
        prob,
        composite_log_likelihood,
    })
}

/// Accuracy the impact interpolant must reach: the largest difference between two nested
/// interpolants, relative to the largest value of the quantity.
const IMPACT_INTERPOLATION_TOL: f64 = 1e-10;

/// Chebyshev points of the second kind on [a, b]: (a + b)/2 + (b - a)/2 cos(jπ/m), j = 0..=m.
fn chebyshev_nodes(a: f64, b: f64, m: usize) -> Vec<f64> {
    (0..=m)
        .map(|j| 0.5 * (a + b) + 0.5 * (b - a) * (std::f64::consts::PI * j as f64 / m as f64).cos())
        .collect()
}

/// The parts of (I - ρW)^-1 the impacts need, interpolated in ρ: barycentric interpolation on
/// Chebyshev points of the second kind over the range of the ρ draws, from one dense inverse
/// per node ([`SpatialWeights::dense_inverse_parts`]). The nodes double (9, 17, 33) until two
/// nested interpolants agree to [`IMPACT_INTERPOLATION_TOL`] at the midpoints of the finer
/// one, or the finer one agrees with the dense inverse to that tolerance in its two end
/// cells, where its error peaks; the finer one is kept. (A check of the 9 nodes against the
/// dense inverse costs two more inverses and fails for the posterior ranges of n up to a few
/// thousand, so it is not made.) When 33 nodes miss the target a warning reports the
/// accuracy measured in the end cells.
struct InverseInterpolant {
    nodes: Vec<f64>,
    /// Barycentric weights (-1)^j, halved at both ends; `[1]` for a single node
    bary: Vec<f64>,
    /// diag((I - ρW)^-1) at each node
    diag: Vec<Array1<f64>>,
    /// [((I - ρW)'(I - ρW))^-1]_ii at each node
    var: Vec<Array1<f64>>,
    /// (I - ρW)^-1 X at each node
    ax: Vec<Array2<f64>>,
    /// The attained accuracy when 33 nodes do not reach the tolerance
    warning: Option<String>,
}

impl InverseInterpolant {
    fn empty() -> Self {
        InverseInterpolant {
            nodes: Vec::new(),
            bary: Vec::new(),
            diag: Vec::new(),
            var: Vec::new(),
            ax: Vec::new(),
            warning: None,
        }
    }

    fn build(listw: &SpatialWeights, x: &Array2<f64>, lo: f64, hi: f64) -> EconResult<Self> {
        let mut it = InverseInterpolant::empty();
        if hi - lo <= 1e-12 * hi.abs().max(1.0) {
            it.push_node(listw, x, 0.5 * (lo + hi))?;
            it.bary = vec![1.0];
            return Ok(it);
        }
        // 9 nodes, then the nested 17 and 33 (every other node of the finer set is a node of
        // the coarser one)
        for &r in &chebyshev_nodes(lo, hi, 8) {
            it.push_node(listw, x, r)?;
        }
        it.set_weights();
        let mut m = 8;
        loop {
            let finer = chebyshev_nodes(lo, hi, 2 * m);
            let mut next = InverseInterpolant::empty();
            for (j, &r) in finer.iter().enumerate() {
                if j % 2 == 0 {
                    let c = j / 2;
                    next.nodes.push(it.nodes[c]);
                    next.diag.push(it.diag[c].clone());
                    next.var.push(it.var[c].clone());
                    next.ax.push(it.ax[c].clone());
                } else {
                    next.push_node(listw, x, r)?;
                }
            }
            next.set_weights();
            let midpoints: Vec<f64> = next.nodes.windows(2).map(|p| 0.5 * (p[0] + p[1])).collect();
            let diff = it.max_relative_difference(&next, &midpoints);
            m *= 2;
            it = next;
            if diff <= IMPACT_INTERPOLATION_TOL {
                return Ok(it);
            }
            // The nested difference measures the coarser interpolant, which overstates the
            // error of the kept one (geometric convergence squares it when the nodes double):
            // measure the kept one against the dense inverse in its two end cells, where its
            // error peaks. Two inverses instead of 16 more nodes when 17 suffice.
            let err = it.end_cell_error(listw, x)?;
            if err <= IMPACT_INTERPOLATION_TOL {
                return Ok(it);
            }
            if m >= 32 {
                it.warning = Some(format!(
                    "the spatial multiplier is interpolated over the rho draws to a relative \
                     accuracy of {err:.1e} (33 nodes), above the target \
                     {IMPACT_INTERPOLATION_TOL:e}"
                ));
                return Ok(it);
            }
        }
    }

    /// The largest relative difference from the dense inverse at the midpoints of the two
    /// end cells (two dense inverses).
    fn end_cell_error(&self, listw: &SpatialWeights, x: &Array2<f64>) -> EconResult<f64> {
        let last = self.nodes.len() - 1;
        let mut err = 0.0f64;
        for r in [
            0.5 * (self.nodes[0] + self.nodes[1]),
            0.5 * (self.nodes[last - 1] + self.nodes[last]),
        ] {
            let mut exact = InverseInterpolant::empty();
            exact.push_node(listw, x, r)?;
            exact.bary = vec![1.0];
            err = err.max(self.max_relative_difference(&exact, &[r]));
        }
        Ok(err)
    }

    fn push_node(&mut self, listw: &SpatialWeights, x: &Array2<f64>, rho: f64) -> EconResult<()> {
        crate::resource::checkpoint()?;
        let parts = listw.dense_inverse_parts(rho, x.view())?;
        self.nodes.push(rho);
        self.diag.push(parts.diag);
        self.var.push(parts.row_sq_norms);
        self.ax.push(parts.solved);
        Ok(())
    }

    fn set_weights(&mut self) {
        let m = self.nodes.len() - 1;
        self.bary = (0..=m)
            .map(|j| {
                let s = if j % 2 == 0 { 1.0 } else { -1.0 };
                if j == 0 || j == m { 0.5 * s } else { s }
            })
            .collect();
    }

    /// The interpolation coefficients at ρ (sum 1; a node gets weight 1).
    fn coefficients(&self, rho: f64) -> Vec<f64> {
        if let Some(j) = self.nodes.iter().position(|&r| r == rho) {
            let mut c = vec![0.0; self.nodes.len()];
            c[j] = 1.0;
            return c;
        }
        let t: Vec<f64> = self
            .nodes
            .iter()
            .zip(&self.bary)
            .map(|(&r, &w)| w / (rho - r))
            .collect();
        let s: f64 = t.iter().sum();
        t.into_iter().map(|v| v / s).collect()
    }

    /// diag, the marginal variances and (I - ρW)^-1 X at ρ.
    fn at(&self, rho: f64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let c = self.coefficients(rho);
        let mut diag = Array1::zeros(self.diag[0].len());
        let mut var = Array1::zeros(diag.len());
        let mut ax = Array2::zeros(self.ax[0].raw_dim());
        for (j, &cj) in c.iter().enumerate() {
            if cj != 0.0 {
                diag.scaled_add(cj, &self.diag[j]);
                var.scaled_add(cj, &self.var[j]);
                ax.scaled_add(cj, &self.ax[j]);
            }
        }
        (diag, var, ax)
    }

    /// The largest difference between this interpolant and `other` at `points`, relative to
    /// the largest value of each quantity at the nodes of `other`.
    fn max_relative_difference(&self, other: &InverseInterpolant, points: &[f64]) -> f64 {
        let finer = other;
        let scale = |vals: &[Array1<f64>]| {
            vals.iter()
                .flat_map(|v| v.iter())
                .fold(0.0f64, |m, v| m.max(v.abs()))
                .max(f64::MIN_POSITIVE)
        };
        let s_diag = scale(&finer.diag);
        let s_var = scale(&finer.var);
        let s_ax = finer
            .ax
            .iter()
            .flat_map(|v| v.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        // element-wise over any memory layout (a non-contiguous array has no slice)
        fn rel<'a>(
            a: impl Iterator<Item = &'a f64>,
            b: impl Iterator<Item = &'a f64>,
            s: f64,
        ) -> f64 {
            a.zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs() / s))
        }
        let mut worst = 0.0f64;
        for &r in points {
            let (d1, v1, a1) = self.at(r);
            let (d2, v2, a2) = finer.at(r);
            worst = worst
                .max(rel(d1.iter(), d2.iter(), s_diag))
                .max(rel(v1.iter(), v2.iter(), s_var))
                .max(rel(a1.iter(), a2.iter(), s_ax));
        }
        if worst.is_nan() { f64::INFINITY } else { worst }
    }
}

/// Direct, indirect and total impacts of one draw.
///
/// SAR (spatialprobit's and LeSage & Pace's convention): with μ = (I - ρW)^-1 Xβ and
/// φ = dnorm(μ), direct_k = β_k mean(φ_i [(I - ρW)^-1]_ii) and total_k = β_k mean(φ_i
/// [(I - ρW)^-1 1]_i). SEM: the derivative of the reported probability Φ(μ_i/σ_i), μ = Xβ,
/// direct_k = β_k mean(φ(μ_i/σ_i)/σ_i), and no spillover (indirect 0, total = direct).
fn probit_impacts_one_draw(
    x: &Array2<f64>,
    interp: &InverseInterpolant,
    beta: &Array1<f64>,
    rho: f64,
    model_type: SpatialProbitModel,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = x.ncols() - 1; // the intercept has no impact
    let n_f = x.nrows() as f64;
    let (diag, var, ax) = interp.at(rho);
    match model_type {
        SpatialProbitModel::Sar => {
            let mu = ax.dot(beta);
            // column 0 of X is the intercept, so column 0 of A^-1 X is A^-1 1
            let row = ax.column(0);
            let (mut d, mut t) = (0.0, 0.0);
            for i in 0..mu.len() {
                let phi = normal_pdf(mu[i]);
                d += phi * diag[i];
                t += phi * row[i];
            }
            let direct: Vec<f64> = (0..k).map(|j| beta[j + 1] * d / n_f).collect();
            let total: Vec<f64> = (0..k).map(|j| beta[j + 1] * t / n_f).collect();
            let indirect = total.iter().zip(&direct).map(|(t, d)| t - d).collect();
            (direct, indirect, total)
        }
        SpatialProbitModel::Sem => {
            let mu = x.dot(beta);
            let s: f64 = mu
                .iter()
                .zip(var.iter())
                .map(|(&m, &v)| {
                    let sigma = v.sqrt();
                    normal_pdf(m / sigma) / sigma
                })
                .sum();
            let direct: Vec<f64> = (0..k).map(|j| beta[j + 1] * s / n_f).collect();
            (direct.clone(), vec![0.0; k], direct)
        }
    }
}

/// Posterior impacts: per draw from the exact spatial multiplier (interpolated in ρ to
/// 1e-10, see [`InverseInterpolant`]), then their means and standard deviations.
fn compute_spatial_probit_impacts(
    x: &Array2<f64>,
    listw: &SpatialWeights,
    beta_draws: &Array2<f64>,
    rho_draws: &Array1<f64>,
    model_type: SpatialProbitModel,
    var_names: &[String],
) -> EconResult<SpatialProbitImpacts> {
    let k = x.ncols() - 1;
    let n_draws = beta_draws.nrows();
    let lo = rho_draws.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = rho_draws.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let interp = InverseInterpolant::build(listw, x, lo, hi)?;

    let mut direct_draws = try_zeros2(n_draws, k, "direct impact draws")?;
    let mut indirect_draws = try_zeros2(n_draws, k, "indirect impact draws")?;
    let mut total_draws = try_zeros2(n_draws, k, "total impact draws")?;
    for d in 0..n_draws {
        crate::resource::checkpoint()?;
        let beta: Array1<f64> = beta_draws.row(d).to_owned();
        let (direct, indirect, total) =
            probit_impacts_one_draw(x, &interp, &beta, rho_draws[d], model_type);
        for j in 0..k {
            direct_draws[[d, j]] = direct[j];
            indirect_draws[[d, j]] = indirect[j];
            total_draws[[d, j]] = total[j];
        }
    }

    let mean = |a: &Array2<f64>| -> Vec<f64> {
        (0..k).map(|j| a.column(j).mean().unwrap_or(0.0)).collect()
    };
    let sd = |a: &Array2<f64>| -> Vec<f64> {
        (0..k)
            .map(|j| compute_std(&a.column(j).to_owned()))
            .collect()
    };
    Ok(SpatialProbitImpacts {
        direct: mean(&direct_draws),
        indirect: mean(&indirect_draws),
        total: mean(&total_draws),
        direct_se: sd(&direct_draws),
        indirect_se: sd(&indirect_draws),
        total_se: sd(&total_draws),
        var_names: var_names.to_vec(),
        warnings: interp.warning.into_iter().collect(),
    })
}

/// Exact draw from N(0, 1) truncated to [a, inf). For a <= 0, normal rejection (acceptance
/// Phi(-a) >= 1/2). Otherwise Robert's (1995) exponential rejection with the optimal rate
/// lambda = (a + sqrt(a^2 + 4)) / 2, accepting z = a + E/lambda with probability
/// exp(-(z - lambda)^2 / 2) (acceptance above 0.76 for a > 0). NaN, +inf and a bound whose
/// square overflows return a (no loop); the Gibbs loop refuses non-finite draws.
fn standard_normal_truncated_below<R: rand::Rng>(a: f64, rng: &mut R) -> f64 {
    use rand_distr::{Distribution, Exp1, StandardNormal};
    // NaN or +inf, and a bound so far out that a² overflows: the mass lies within 1/a of a
    if a.is_nan() || (a > 0.0 && !(a * a).is_finite()) {
        return a;
    }
    if a <= 0.0 {
        loop {
            let z: f64 = StandardNormal.sample(rng);
            if z >= a {
                return z;
            }
        }
    }
    let lambda = 0.5 * (a + (a * a + 4.0).sqrt());
    loop {
        let e: f64 = Exp1.sample(rng);
        let z = a + e / lambda;
        let u: f64 = rng.r#gen();
        if u <= (-0.5 * (z - lambda) * (z - lambda)).exp() {
            return z;
        }
    }
}

/// Draw from N(mu, sigma^2) truncated to [lower, inf).
fn sample_truncated_normal_above<R: rand::Rng>(
    mu: f64,
    sigma: f64,
    lower: f64,
    rng: &mut R,
) -> f64 {
    mu + sigma * standard_normal_truncated_below((lower - mu) / sigma, rng)
}

/// Draw from N(mu, sigma^2) truncated to (-inf, upper].
fn sample_truncated_normal_below<R: rand::Rng>(
    mu: f64,
    sigma: f64,
    upper: f64,
    rng: &mut R,
) -> f64 {
    mu - sigma * standard_normal_truncated_below((mu - upper) / sigma, rng)
}

/// Compute standard deviation of an array.
fn compute_std(arr: &Array1<f64>) -> f64 {
    let n = arr.len() as f64;
    if n <= 1.0 {
        return 0.0;
    }
    let mean = arr.mean().unwrap_or(0.0);
    let var: f64 = arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// Compute column-wise standard deviation.
fn compute_std_axis0(arr: &Array2<f64>) -> Array1<f64> {
    let n = arr.nrows() as f64;
    let k = arr.ncols();
    let mean = arr.mean_axis(Axis(0)).unwrap();

    let mut std = Array1::zeros(k);
    for j in 0..k {
        let var: f64 = arr
            .column(j)
            .iter()
            .map(|&x| (x - mean[j]).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        std[j] = var.sqrt();
    }
    std
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Neighbors, WeightStyle};
    use polars::prelude::*;

    /// The probit-pieces case (spatial_diag_t4.R): W the 4-NN row-standardised weights of
    /// spatial_sim, X = [1, x1, x2], the outcome yb, the latent z and beta.
    struct Pieces {
        case: p2a_validation::Case,
        w: SpatialWeights,
        x: Array2<f64>,
        yb: Array1<f64>,
        z: Array1<f64>,
        b: Array1<f64>,
    }

    fn pieces(id: &str) -> Pieces {
        let (case, ds) = crate::test_support::load_case(id);
        let c = |name: &str| {
            crate::data::columns::numeric_column(
                ds.df(),
                name,
                crate::data::columns::NaPolicy::Fail,
            )
            .unwrap()
            .0
        };
        let coords: Vec<(f64, f64)> = c("cx").into_iter().zip(c("cy")).collect();
        let w = SpatialWeights::from_neighbors(
            &Neighbors::try_from_knn(&coords, 4).unwrap(),
            WeightStyle::RowStd,
        );
        let (x1, x2) = (c("x1"), c("x2"));
        let x = Array2::from_shape_fn((x1.len(), 3), |(i, j)| match j {
            0 => 1.0,
            1 => x1[i],
            _ => x2[i],
        });
        Pieces {
            yb: Array1::from(case.vector("yb")),
            z: Array1::from(case.vector("z")),
            b: Array1::from(case.vector("beta")),
            case,
            w,
            x,
        }
    }

    /// Closed-form pieces: exact linear algebra on both sides.
    fn pieces_tol() -> p2a_validation::Tol {
        p2a_validation::Tol::new(1e-8, 1e-12)
    }

    /// R: logdet(rho) - ||A z - c||^2 / 2 on the grid -0.9, ..., 0.9, SAR and SEM.
    #[test]
    fn test_validate_probit_rho_kernel_spatial_sim() {
        let mut p = pieces("spatial_probit_pieces_t4");
        let grid = p.case.vector("grid");
        let log_det = p.w.try_spectrum().unwrap().log_det_grid(&grid);
        let xb = p.x.dot(&p.b);
        let mut chk = p2a_validation::Check::new(&p.case);
        chk.vector("logdet_grid", &log_det, pieces_tol());
        for (model, key) in [
            (SpatialProbitModel::Sar, "sar_rho_kernel"),
            (SpatialProbitModel::Sem, "sem_rho_kernel"),
        ] {
            let terms = rho_kernel_terms(&p.z, &xb, &p.w, model);
            let k: Vec<f64> = grid
                .iter()
                .zip(&log_det)
                .map(|(&r, &ld)| rho_log_kernel(ld, r, terms))
                .collect();
            chk.vector(key, &k, pieces_tol());
        }
        chk.finish();
    }

    /// R: the full conditional of each z_i given the others at rho = 0.6: mean
    /// mu_i - sum_{j != i} H_ij (z_j - mu_j) / H_ii and variance 1 / H_ii, H = A'A.
    #[test]
    fn test_validate_probit_latent_conditional_spatial_sim() {
        let p = pieces("spatial_probit_pieces_t4");
        let rho = 0.6;
        let cols = weight_columns(&p.w).unwrap();
        let xb = p.x.dot(&p.b);
        let mut chk = p2a_validation::Check::new(&p.case);
        let h: Vec<f64> = cols
            .iter()
            .map(|c| 1.0 + rho * rho * c.iter().map(|&(_, w)| w * w).sum::<f64>())
            .collect();
        for (model, key) in [
            (SpatialProbitModel::Sar, "sar_latent_mean_rho06"),
            (SpatialProbitModel::Sem, "sem_latent_mean_rho06"),
        ] {
            let c = latent_offset(&xb, &p.w, rho, model);
            let r = &p.z - &(rho * &p.w.lag(&p.z)) - &c;
            let m: Vec<f64> = (0..p.z.len())
                .map(|i| latent_conditional_mean(i, p.z[i], &r, &cols[i], rho, h[i]))
                .collect();
            chk.vector(key, &m, pieces_tol());
        }
        let var: Vec<f64> = h.iter().map(|v| v.recip()).collect();
        chk.vector("latent_var_rho06", &var, pieces_tol()).finish();
    }

    /// R: sigma_i = sqrt([(A'A)^-1]_ii), Phi(mu_i / sigma_i) and the composite log-likelihood,
    /// SAR at rho = 0.5 and 0.9 and SEM at lambda = 0.6.
    #[test]
    fn test_validate_probit_marginal_probabilities_spatial_sim() {
        let p = pieces("spatial_probit_pieces_t4");
        let mut chk = p2a_validation::Check::new(&p.case);
        for (model, rho, tag) in [
            (SpatialProbitModel::Sar, 0.5, "sar_%_rho05"),
            (SpatialProbitModel::Sar, 0.9, "sar_%_rho09"),
            (SpatialProbitModel::Sem, 0.6, "sem_%_rho06"),
        ] {
            let s = marginal_summary(&p.yb, &p.x, &p.w, &p.b, rho, model).unwrap();
            let key = |what: &str| tag.replace('%', what);
            chk.vector(&key("sigma"), s.sigma.as_slice().unwrap(), pieces_tol())
                .vector(&key("prob"), s.prob.as_slice().unwrap(), pieces_tol())
                .scalar(&key("cll"), s.composite_log_likelihood, pieces_tol());
        }
        chk.finish();
    }

    /// R: impacts of one draw at rho = 0.5 and 0.9 from the exact inverse (SAR: phi(mu) with
    /// mu = A^-1 X b, direct from diag(A^-1), total from A^-1 1), SEM direct effects at 0.6.
    #[test]
    fn test_validate_probit_impacts_exact_spatial_sim() {
        let p = pieces("spatial_probit_pieces_t4");
        let mut chk = p2a_validation::Check::new(&p.case);
        for (rho, tag) in [(0.5, "rho05"), (0.9, "rho09")] {
            let interp = InverseInterpolant::build(&p.w, &p.x, rho, rho).unwrap();
            let (d, i, t) =
                probit_impacts_one_draw(&p.x, &interp, &p.b, rho, SpatialProbitModel::Sar);
            chk.vector(&format!("sar_direct_{tag}"), &d, pieces_tol())
                .vector(&format!("sar_indirect_{tag}"), &i, pieces_tol())
                .vector(&format!("sar_total_{tag}"), &t, pieces_tol());
        }
        chk.finish();
    }

    #[test]
    fn test_validate_sem_probit_direct_effects_spatial_sim() {
        let p = pieces("spatial_probit_pieces_t4");
        let interp = InverseInterpolant::build(&p.w, &p.x, 0.6, 0.6).unwrap();
        let (d, _, _) = probit_impacts_one_draw(&p.x, &interp, &p.b, 0.6, SpatialProbitModel::Sem);
        p2a_validation::Check::new(&p.case)
            .vector("sem_direct_rho06", &d, pieces_tol())
            .finish();
    }

    #[test]
    fn sem_probit_impacts_have_no_spillover() {
        let p = pieces("spatial_probit_pieces_t4");
        let interp = InverseInterpolant::build(&p.w, &p.x, 0.2, 0.7).unwrap();
        for rho in [0.2, 0.45, 0.7] {
            let (d, i, t) =
                probit_impacts_one_draw(&p.x, &interp, &p.b, rho, SpatialProbitModel::Sem);
            assert!(i.iter().all(|&v| v == 0.0));
            assert_eq!(d, t);
        }
    }

    /// Between the nodes, the interpolated multiplier gives the impacts of the dense inverse
    /// at that rho to 1e-9 relative.
    #[test]
    fn probit_impacts_interpolation_matches_dense() {
        let p = pieces("spatial_probit_pieces_t4");
        let interp = InverseInterpolant::build(&p.w, &p.x, -0.2, 0.85).unwrap();
        assert!(interp.warning.is_none());
        // a narrower range stops at 17 nodes, accepted by the dense check in its end cells, and
        // a very narrow one at 17 too, by the nested comparison
        let narrow = InverseInterpolant::build(&p.w, &p.x, 0.4, 0.7).unwrap();
        assert!(narrow.nodes.len() == 17 && narrow.warning.is_none());
        let narrower = InverseInterpolant::build(&p.w, &p.x, 0.30, 0.34).unwrap();
        assert!(narrower.nodes.len() == 17 && narrower.warning.is_none());
        for r in [0.30, 0.3123, 0.34] {
            let mut exact = InverseInterpolant::empty();
            exact.push_node(&p.w, &p.x, r).unwrap();
            exact.bary = vec![1.0];
            assert!(narrower.max_relative_difference(&exact, &[r]) <= 1e-9);
        }
        for rho in [-0.199, -0.1, 0.123_456, 0.5, 0.77, 0.849] {
            let exact = InverseInterpolant::build(&p.w, &p.x, rho, rho).unwrap();
            for model in [SpatialProbitModel::Sar, SpatialProbitModel::Sem] {
                let a = probit_impacts_one_draw(&p.x, &interp, &p.b, rho, model);
                let b = probit_impacts_one_draw(&p.x, &exact, &p.b, rho, model);
                for (u, v) in [(a.0, b.0), (a.2, b.2)] {
                    for (x, y) in u.iter().zip(&v) {
                        assert!(
                            (x - y).abs() <= 1e-9 * y.abs().max(1e-3),
                            "{model} rho={rho}: {x} vs {y}"
                        );
                    }
                }
            }
        }
    }

    /// A range reaching close to 1/lambda_max (row-standardised W: rho = 1) converges slowly;
    /// 33 nodes stop the doubling and the warning reports the measured accuracy.
    #[test]
    fn probit_impacts_interpolation_warns_near_the_singular_rho() {
        let p = pieces("spatial_probit_pieces_t4");
        let interp = InverseInterpolant::build(&p.w, &p.x, -0.9, 0.97).unwrap();
        assert_eq!(interp.nodes.len(), 33);
        let w = interp.warning.as_deref().unwrap();
        assert!(w.contains("33 nodes"), "{w}");
    }

    #[test]
    fn ln_normal_cdf_is_accurate_in_both_tails() {
        for x in [-40.0, -31.0, -29.0, -5.0, -0.3, 0.0, 0.7, 6.0, 9.0] {
            let reference = if x > -35.0 {
                normal_cdf(x).ln()
            } else {
                f64::NAN
            };
            let v = ln_normal_cdf(x);
            if x > 5.0 {
                // ln(1 - tiny): normal_cdf rounds to 1
                assert!(v < 0.0 && v > -1e-8, "{x}: {v}");
            } else if reference.is_finite() {
                assert!(
                    (v - reference).abs() <= 1e-9 * reference.abs().max(1e-300),
                    "{x}: {v}"
                );
            }
        }
        // the series against the exact value at -31 and -40 (R: pnorm(x, log.p = TRUE))
        assert!((ln_normal_cdf(-40.0) - (-804.608_442_013_754)).abs() < 1e-9);
        assert!((ln_normal_cdf(-31.0) - (-484.853_963_627_179_3)).abs() < 1e-9);
    }

    /// The grid draw targets the kernel: 20,000 draws of rho at fixed z and beta have the
    /// kernel's mean and SD (by quadrature) within 4 Monte Carlo SEs.
    #[test]
    fn probit_rho_step_draws_from_the_kernel() {
        use rand::SeedableRng;
        let mut p = pieces("spatial_probit_pieces_t4");
        let (lo, hi) = p.w.try_rho_range().unwrap();
        let grid = rho_grid((lo.max(-0.99), hi.min(0.99)));
        let spectrum = p.w.try_spectrum().unwrap();
        let log_det = spectrum.log_det_grid(&grid);
        let terms = rho_kernel_terms(&p.z, &p.x.dot(&p.b), &p.w, SpatialProbitModel::Sar);
        let log_k: Vec<f64> = grid
            .iter()
            .zip(&log_det)
            .map(|(&r, &ld)| rho_log_kernel(ld, r, terms))
            .collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
        let n_draws = 20_000;
        let draws: Vec<f64> = (0..n_draws)
            .map(|_| draw_from_grid(&grid, &log_k, &mut rng))
            .collect();
        let mean = draws.iter().sum::<f64>() / n_draws as f64;
        let sd =
            (draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n_draws - 1) as f64).sqrt();
        // quadrature of the exact kernel on [grid[0], grid[last]]
        let (a, b) = (grid[0], grid[grid.len() - 1]);
        let m = 200_000;
        let pts: Vec<f64> = (0..=m).map(|i| a + (b - a) * i as f64 / m as f64).collect();
        let lk: Vec<f64> = pts
            .iter()
            .map(|&r| rho_log_kernel(spectrum.log_det(r), r, terms))
            .collect();
        let top = lk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let f: Vec<f64> = lk.iter().map(|l| (l - top).exp()).collect();
        let trap = |g: &dyn Fn(usize) -> f64| {
            (0..m).map(|i| 0.5 * (g(i) + g(i + 1))).sum::<f64>() * (b - a) / m as f64
        };
        let z0 = trap(&|i| f[i]);
        let mu = trap(&|i| f[i] * pts[i]) / z0;
        let var = trap(&|i| f[i] * (pts[i] - mu).powi(2)) / z0;
        let se_mean = (var / n_draws as f64).sqrt();
        let se_sd = (var / (2.0 * n_draws as f64)).sqrt();
        assert!((mean - mu).abs() <= 4.0 * se_mean, "mean {mean} vs {mu}");
        assert!(
            (sd - var.sqrt()).abs() <= 4.0 * se_sd,
            "sd {sd} vs {}",
            var.sqrt()
        );
    }

    #[test]
    fn truncated_normal_far_bound_returns_the_bound() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        assert_eq!(standard_normal_truncated_below(1e200, &mut rng), 1e200);
        assert_eq!(
            sample_truncated_normal_below(-1e200, 1.0, 0.0, &mut rng),
            -1e200
        );
    }

    /// A 20 x 20 rook grid (row-standardised), x ~ N(0, 1), beta = (0, 1), latent outcomes from
    /// the SAR or SEM model with spatial parameter `rho`, y = 1(z > 0).
    fn probit_dgp(model: SpatialProbitModel, rho: f64, seed: u64) -> (Dataset, SpatialWeights) {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        let side = 20;
        let lists = (0..side * side)
            .map(|i| {
                let (r, c) = (i / side, i % side);
                let mut v = Vec::new();
                if r > 0 {
                    v.push(i - side);
                }
                if c > 0 {
                    v.push(i - 1);
                }
                if c + 1 < side {
                    v.push(i + 1);
                }
                if r + 1 < side {
                    v.push(i + side);
                }
                v
            })
            .collect();
        let w =
            SpatialWeights::from_neighbors(&Neighbors::from_indices(lists), WeightStyle::RowStd);
        let n = side * side;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let x: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let e: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
        let z: Vec<f64> = match model {
            SpatialProbitModel::Sar => {
                let rhs = Array2::from_shape_fn((n, 1), |(i, _)| x[i] + e[i]);
                w.dense_inverse_parts(rho, rhs.view())
                    .unwrap()
                    .solved
                    .column(0)
                    .to_vec()
            }
            SpatialProbitModel::Sem => {
                let rhs = Array2::from_shape_fn((n, 1), |(i, _)| e[i]);
                let u = w.dense_inverse_parts(rho, rhs.view()).unwrap().solved;
                (0..n).map(|i| x[i] + u[[i, 0]]).collect()
            }
        };
        let y: Vec<f64> = z.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
        (Dataset::new(df! { "y" => y, "x" => x }.unwrap()), w)
    }

    fn check_recovery(model: SpatialProbitModel, rho: f64) {
        for seed in [11, 23, 37] {
            let (d, mut w) = probit_dgp(model, rho, seed);
            let config = SpatialProbitConfig {
                n_draws: 2000,
                burn_in: 500,
                compute_impacts: false,
                seed: Some(seed),
                ..Default::default()
            };
            let r = match model {
                SpatialProbitModel::Sar => run_sar_probit(&d, "y", &["x"], &mut w, config),
                SpatialProbitModel::Sem => run_sem_probit(&d, "y", &["x"], &mut w, config),
            }
            .unwrap();
            let (lo, hi) = r.credible_interval_rho(0.95);
            assert!(
                (r.rho - rho).abs() <= 0.15 && lo <= rho && rho <= hi,
                "{model} rho = {rho}, seed {seed}: posterior mean {} [{lo}, {hi}]",
                r.rho
            );
        }
    }

    #[test]
    fn sem_probit_recovers_dgp() {
        check_recovery(SpatialProbitModel::Sem, 0.6);
    }

    #[test]
    fn sar_probit_recovers_dgp() {
        check_recovery(SpatialProbitModel::Sar, 0.5);
        check_recovery(SpatialProbitModel::Sar, 0.8);
    }

    fn create_test_spatial_binary_data() -> (Dataset, SpatialWeights) {
        // Create a 5x5 grid (25 observations)
        let n = 25;
        let coords: Vec<(f64, f64)> = (0..5)
            .flat_map(|i| (0..5).map(move |j| (i as f64, j as f64)))
            .collect();

        // Create neighbors (4-nearest neighbors)
        let nb = Neighbors::from_knn(&coords, 4);
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        // Generate data with spatial correlation
        // True model: P(y=1) = Phi(0.5 + 0.5*x + spatial_effect)
        let mut x_vec: Vec<f64> = Vec::with_capacity(n);
        let mut y_vec: Vec<f64> = Vec::with_capacity(n);

        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal, Uniform};
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::new(0.0, 1.0);

        for i in 0..n {
            let row = i / 5;
            let col = i % 5;
            let x_i = normal.sample(&mut rng);
            x_vec.push(x_i);

            // Add spatial pattern
            let spatial_effect = 0.2 * ((row + col) as f64 - 4.0);
            let z_i = 0.5 + 0.5 * x_i + spatial_effect + normal.sample(&mut rng) * 0.5;
            let p_i = normal_cdf(z_i);
            let y_i = if uniform.sample(&mut rng) < p_i {
                1.0
            } else {
                0.0
            };
            y_vec.push(y_i);
        }

        let df = df! {
            "y" => &y_vec,
            "x" => &x_vec,
        }
        .unwrap();

        (Dataset::new(df), listw)
    }

    #[test]
    fn test_sar_probit_basic() {
        let (dataset, mut listw) = create_test_spatial_binary_data();

        let config = SpatialProbitConfig {
            n_draws: 100,
            burn_in: 50,
            compute_impacts: false,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_sar_probit(&dataset, "y", &["x"], &mut listw, config).unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 25);
        assert_eq!(result.model_type, SpatialProbitModel::Sar);
        assert!(result.rho > -1.0 && result.rho < 1.0);
        assert!(result.coefficients.len() == 2); // Intercept + x
        assert!(result.pcp >= 0.0 && result.pcp <= 100.0);
    }

    #[test]
    fn test_sem_probit_basic() {
        let (dataset, mut listw) = create_test_spatial_binary_data();

        let config = SpatialProbitConfig {
            n_draws: 100,
            burn_in: 50,
            compute_impacts: false,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_sem_probit(&dataset, "y", &["x"], &mut listw, config).unwrap();

        assert_eq!(result.n_obs, 25);
        assert_eq!(result.model_type, SpatialProbitModel::Sem);
        assert!(result.rho > -1.0 && result.rho < 1.0);
    }

    #[test]
    fn test_spatial_probit_impacts() {
        let (dataset, mut listw) = create_test_spatial_binary_data();

        let config = SpatialProbitConfig {
            n_draws: 50,
            burn_in: 25,
            compute_impacts: true,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_sar_probit(&dataset, "y", &["x"], &mut listw, config).unwrap();

        let impacts = result.impacts.unwrap();
        assert_eq!(impacts.var_names.len(), 1); // Only x (not intercept)
        assert_eq!(impacts.direct.len(), 1);
        assert_eq!(impacts.indirect.len(), 1);
        assert_eq!(impacts.total.len(), 1);

        // Total should approximately equal direct + indirect
        let total_check = impacts.direct[0] + impacts.indirect[0];
        assert!((impacts.total[0] - total_check).abs() < 1e-6);
    }

    #[test]
    fn test_credible_intervals() {
        let (dataset, mut listw) = create_test_spatial_binary_data();

        let config = SpatialProbitConfig {
            n_draws: 100,
            burn_in: 50,
            compute_impacts: false,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_sar_probit(&dataset, "y", &["x"], &mut listw, config).unwrap();

        let (lower, upper) = result.credible_interval_rho(0.95);
        assert!(lower < result.rho);
        assert!(upper > result.rho);

        let (lower_b, upper_b) = result.credible_interval_beta(0, 0.95);
        assert!(lower_b < result.coefficients[0]);
        assert!(upper_b > result.coefficients[0]);
    }

    #[test]
    fn test_truncated_normal_sampling() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        // Sample above 0 with mean 1
        let samples: Vec<f64> = (0..1000)
            .map(|_| sample_truncated_normal_above(1.0, 1.0, 0.0, &mut rng))
            .collect();

        // All samples should be positive
        assert!(samples.iter().all(|&x| x > 0.0));

        // Mean should be greater than 1 (truncation shifts mean up)
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        assert!(mean > 1.0);
    }

    fn cfg(n_draws: usize, burn_in: usize, thin: usize) -> SpatialProbitConfig {
        SpatialProbitConfig {
            n_draws,
            burn_in,
            thin,
            compute_impacts: false,
            seed: Some(42),
            ..Default::default()
        }
    }

    #[test]
    fn probit_rejects_zero_draws_and_zero_thinning() {
        let (d, mut w) = create_test_spatial_binary_data();
        for (c, word) in [(cfg(0, 10, 1), "n_draws"), (cfg(10, 10, 0), "thin")] {
            let e = run_sar_probit(&d, "y", &["x"], &mut w, c).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains(word)),
                "{e}"
            );
        }
    }

    #[test]
    fn probit_rejects_draw_counts_that_overflow_or_cannot_be_stored() {
        let (d, mut w) = create_test_spatial_binary_data();
        // overflow of burn_in + n_draws * thin, and an n_draws x k buffer larger than isize::MAX bytes
        for c in [
            cfg(usize::MAX / 2, 0, 3),
            // n_draws = 1 is refused before the overflow check
            cfg(2, usize::MAX, 1),
            cfg(usize::MAX / 8, 0, 1),
        ] {
            assert!(matches!(
                run_sem_probit(&d, "y", &["x"], &mut w, c),
                Err(EconError::ResourceLimit(_))
            ));
        }
    }

    #[test]
    fn probit_sampler_honours_cancellation() {
        let (d, mut w) = create_test_spatial_binary_data();
        let t = crate::resource::CancelToken::new();
        t.cancel();
        let r = crate::resource::with_budget(crate::resource::Budget::new(None, t), || {
            run_sar_probit(&d, "y", &["x"], &mut w, cfg(10, 5, 1))
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

    #[test]
    fn seeded_probit_runs_are_unchanged() {
        use crate::spatial::test_fixtures::{dataset, knn3};
        let d = dataset();
        let c = |compute_impacts| SpatialProbitConfig {
            n_draws: 200,
            burn_in: 50,
            compute_impacts,
            seed: Some(42),
            ..Default::default()
        };
        // Pinned at the values the sampler printed at seed 42 (single-site latent Gibbs, griddy
        // rho with the log-determinant; impacts from the exact inverse). Summation order may
        // change the floating-point noise, so 1e-8 relative.
        let close = |a: f64, b: f64, what: &str| {
            assert!(
                (a - b).abs() <= 1e-8 * b.abs().max(1.0),
                "{what}: {a:.17e} vs {b:.17e}"
            );
        };
        let mut w = knn3(WeightStyle::RowStd);
        let sar = run_sar_probit(&d, "y_bin", &["x1"], &mut w, c(true)).unwrap();
        let mut w = knn3(WeightStyle::RowStd);
        let sem = run_sem_probit(&d, "y_bin", &["x1"], &mut w, c(false)).unwrap();
        // [rho, intercept, slope, se(intercept), se(slope), log-likelihood, pcp]
        for (r, pins) in [
            (
                &sar,
                [
                    0.33935990444742364,
                    -1.8920669011406606,
                    1.0458502912753576,
                    0.6916706203357502,
                    0.3288943171809251,
                    -10.58827818212202,
                    80.0,
                ],
            ),
            (
                &sem,
                [
                    0.7388000568415172,
                    -3.5195331030245915,
                    2.1441699735020996,
                    2.1289762066771036,
                    0.6266556880570991,
                    -10.93023963401941,
                    83.33333333333333,
                ],
            ),
        ] {
            close(r.rho, pins[0], "rho");
            close(r.coefficients[0], pins[1], "intercept");
            close(r.coefficients[1], pins[2], "slope");
            close(r.std_errors[0], pins[3], "se(intercept)");
            close(r.std_errors[1], pins[4], "se(slope)");
            close(r.log_likelihood, pins[5], "log-likelihood");
            close(r.pcp, pins[6], "pcp");
        }
        let imp = sar.impacts.as_ref().unwrap();
        close(imp.direct[0], 0.17366779759349335, "direct");
        close(imp.indirect[0], 0.08944798156287251, "indirect");
        close(imp.total[0], 0.26311577915636586, "total");
        close(imp.direct_se[0], 0.03203089403860428, "direct se");
        close(imp.indirect_se[0], 0.06302894554825618, "indirect se");
        close(imp.total_se[0], 0.07557935355488735, "total se");
        assert!(sem.impacts.is_none());
    }

    fn probit_config(n_draws: usize, impacts: bool) -> SpatialProbitConfig {
        let mut c = SpatialProbitConfig::default();
        c.n_draws = n_draws;
        c.burn_in = 10;
        c.compute_impacts = impacts;
        c.seed = Some(42);
        c
    }

    #[test]
    fn test_spatial_probit_rejects_n_draws_0() {
        // n_draws = 1 returned Ok with NaN posterior SDs (0/0) before.
        let (d, mut w) = create_test_spatial_binary_data();
        for n_draws in [0, 1] {
            for run in [run_sar_probit, run_sem_probit] {
                let e = run(&d, "y", &["x"], &mut w, probit_config(n_draws, false)).unwrap_err();
                assert!(
                    matches!(&e, EconError::InvalidSpecification { message } if message.contains("n_draws")),
                    "{e}"
                );
            }
        }
    }

    #[test]
    fn test_spatial_probit_rejects_thin_0() {
        let (d, mut w) = create_test_spatial_binary_data();
        let mut c = probit_config(10, false);
        c.thin = 0;
        let e = run_sar_probit(&d, "y", &["x"], &mut w, c).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("thin")),
            "{e}"
        );
    }

    #[test]
    fn test_spatial_probit_rejects_non_binary() {
        // Before: y in {0, 1.7} (and counts) were thresholded at 0.5 without a word.
        let (d, mut w) = create_test_spatial_binary_data();
        let y: Vec<f64> = outcome(&d).iter().map(|v| 1.7 * v).collect();
        let mut df = d.df().clone();
        df.with_column(Column::new("y".into(), y)).unwrap();
        let e = run_sem_probit(
            &Dataset::new(df),
            "y",
            &["x"],
            &mut w,
            probit_config(10, false),
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("binary") && message.contains("'y'")),
            "{e}"
        );
    }

    #[test]
    fn test_spatial_probit_rejects_impacts_without_regressors() {
        // Before: Ok with empty impact vectors, which a caller reading variable 0 cannot index.
        let (d, mut w) = create_test_spatial_binary_data();
        let e = run_sar_probit(&d, "y", &[], &mut w, probit_config(10, true)).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("compute_impacts")),
            "{e}"
        );
        assert!(run_sar_probit(&d, "y", &[], &mut w, probit_config(10, false)).is_ok());
    }

    #[test]
    fn spatial_probit_rejects_an_empty_rho_range() {
        let (d, mut w) = create_test_spatial_binary_data();
        for prior in [(0.5, -0.5), (0.3, 0.3), (2.0, 3.0), (f64::NAN, 0.9)] {
            let mut c = probit_config(10, false);
            c.rho_prior_range = prior;
            for run in [run_sar_probit, run_sem_probit] {
                let e = run(&d, "y", &["x"], &mut w, c.clone()).unwrap_err();
                assert!(
                    matches!(&e, EconError::InvalidSpecification { message } if message.contains("rho_prior_range")),
                    "{prior:?}: {e}"
                );
            }
        }
    }

    #[test]
    fn spatial_probit_rejects_a_self_link() {
        let (d, _) = create_test_spatial_binary_data();
        let mut lists: Vec<Vec<usize>> = (0..25)
            .map(|i| {
                let mut v = Vec::new();
                if i % 5 > 0 {
                    v.push(i - 1);
                }
                if i % 5 < 4 {
                    v.push(i + 1);
                }
                v
            })
            .collect();
        lists[3].push(3);
        let mut w =
            SpatialWeights::from_neighbors(&Neighbors::from_indices(lists), WeightStyle::RowStd);
        for run in [run_sar_probit, run_sem_probit] {
            let e = run(&d, "y", &["x"], &mut w, probit_config(10, false)).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("unit 4 lists itself")),
                "{e}"
            );
        }
    }

    /// Exact CDF of N(mu, 1) truncated to [0, inf) (above) or (-inf, 0] (below), through upper tails.
    fn truncnorm_cdf(x: f64, mu: f64, above: bool) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};
        let nd = Normal::new(0.0, 1.0).unwrap();
        if above {
            1.0 - nd.sf(x - mu) / nd.sf(-mu)
        } else {
            nd.sf(mu - x) / nd.sf(mu)
        }
    }

    fn ks_check(mu: f64, above: bool, mean: f64, sd: f64, seed: u64) {
        use rand::SeedableRng;
        const N: usize = 20_000;
        // sqrt(-log(0.5e-6) / 2) / sqrt(N): P(D > D_CRIT) = 1e-6
        const D_CRIT: f64 = 0.019045116000253333;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut draws: Vec<f64> = (0..N)
            .map(|_| {
                if above {
                    sample_truncated_normal_above(mu, 1.0, 0.0, &mut rng)
                } else {
                    sample_truncated_normal_below(mu, 1.0, 0.0, &mut rng)
                }
            })
            .collect();
        assert!(
            draws
                .iter()
                .all(|&v| if above { v >= 0.0 } else { v <= 0.0 }),
            "mu={mu}: a draw violates the bound"
        );
        let m = draws.iter().sum::<f64>() / N as f64;
        assert!(
            (m - mean).abs() < 5.0 * sd / (N as f64).sqrt(),
            "mu={mu}: mean {m} vs {mean}"
        );
        draws.sort_by(|a, b| a.total_cmp(b));
        let d = draws.iter().enumerate().fold(0.0_f64, |acc, (i, &x)| {
            let f = truncnorm_cdf(x, mu, above);
            acc.max(f - i as f64 / N as f64)
                .max((i + 1) as f64 / N as f64 - f)
        });
        assert!(d < D_CRIT, "mu={mu}: KS D = {d}");
    }

    #[test]
    fn test_truncated_normal_sampler_ks() {
        // Exact means/SDs from R (lam = dnorm(a) / pnorm(a, lower.tail = FALSE)). The inverse-CDF
        // sampler drew 7 + qnorm(U(1e-10, 2e-10)) in (0.639, 0.746) at mu = 7 (y = 0), all above
        // the bound, and returned the constant 0.01 at mu = -8 (y = 1).
        for (c, &(mu, above, mean, sd)) in [
            (1.0, true, 1.2875999709391783, 0.7935277473262076),
            (-2.5, true, 0.3227447976639053, 0.29828476565376677),
            (-8.0, true, 0.12136811223612298, 0.11968660511206433),
            (7.0, false, -0.1375456132264974, 0.13513664083682866),
            (-0.5, false, -1.0091604338370335, 0.6972628168032245),
        ]
        .iter()
        .enumerate()
        {
            ks_check(mu, above, mean, sd, 20_261_022 + c as u64);
        }
    }

    #[test]
    fn test_truncated_normal_sampler_tail_6_sigma() {
        // mu = -6.2 for y = 1: the inverse-CDF sampler worked on Uniform(Phi(6.2), 1 - 1e-10), an
        // interval of width 1.8e-10.
        ks_check(
            -6.2,
            true,
            0.1538203828034117,
            0.15050885838734468,
            20_261_029,
        );
    }

    #[test]
    fn test_spatial_probit_credible_intervals_are_type7() {
        // R quantile(type = 7) at 0.025 / 0.975 of 10 sorted draws: x1 + 0.225 (x2 - x1),
        // x9 + 0.775 (x10 - x9). Before: draws[floor(10 * 0.025)] = x1 and draws[floor(10 * 0.975)] = x10.
        let (d, mut w) = create_test_spatial_binary_data();
        let mut c = probit_config(10, false);
        c.burn_in = 5;
        let r = run_sar_probit(&d, "y", &["x"], &mut w, c).unwrap();
        let mut s = r.rho_draws.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        let expect = (s[0] + 0.225 * (s[1] - s[0]), s[8] + 0.775 * (s[9] - s[8]));
        let got = r.credible_interval_rho(0.95);
        assert!(
            (got.0 - expect.0).abs() <= 1e-12 && (got.1 - expect.1).abs() <= 1e-12,
            "{got:?} vs {expect:?}"
        );
        assert!(r.credible_interval_beta(5, 0.95).0.is_nan());
        assert!(r.credible_interval_rho(1.5).0.is_nan());
    }

    #[test]
    fn test_spatial_probit_unseeded_runs_are_reproducible() {
        // seed None uses SPATIAL_PROBIT_DEFAULT_SEED (before: entropy, so two runs differed).
        let (d, w) = create_test_spatial_binary_data();
        let run = || {
            let mut c = probit_config(20, false);
            c.seed = None;
            run_sem_probit(&d, "y", &["x"], &mut w.clone(), c).unwrap()
        };
        assert_eq!(format!("{:?}", run()), format!("{:?}", run()));
    }
    #[test]
    fn test_spatial_probit_display_lists_every_impact_se() {
        // Every variable's own posterior SDs, not one "Std.Errors" row.
        let (d, mut w) = create_test_spatial_binary_data();
        let mut df = d.df().clone();
        let x2: Vec<f64> = (0..25).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
        df.with_column(Column::new("x2".into(), x2)).unwrap();
        let mut c = probit_config(50, true);
        c.burn_in = 25;
        let r = run_sar_probit(&Dataset::new(df), "y", &["x", "x2"], &mut w, c).unwrap();
        let imp = r.impacts.as_ref().unwrap();
        let text = r.to_string();
        for (i, name) in imp.var_names.iter().enumerate() {
            let total = format!("{:.6}", imp.total[i]);
            let row = text
                .lines()
                .find(|l| l.trim_start().starts_with(name.as_str()) && l.contains(&total))
                .unwrap_or_else(|| panic!("no impact row for {name}:\n{text}"));
            for v in [imp.direct_se[i], imp.indirect_se[i], imp.total_se[i]] {
                assert!(row.contains(&format!("{v:.6}")), "{row}");
            }
        }
        assert!(!text.contains("Std.Errors"), "{text}");
    }

    fn short_chain() -> SpatialProbitConfig {
        SpatialProbitConfig {
            n_draws: 100,
            burn_in: 50,
            compute_impacts: false,
            seed: Some(42),
            ..Default::default()
        }
    }

    /// The outcome of the binary fixture as stored (Float64 0/1).
    fn outcome(ds: &Dataset) -> Vec<f64> {
        ds.df()
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    #[test]
    fn test_na_spatial_probit_int64_and_boolean_outcome() {
        let (ds, w) = create_test_spatial_binary_data();
        let y = outcome(&ds);
        let x = ds.df().column("x").unwrap().clone();
        let y_i64: Vec<i64> = y.iter().map(|&v| v as i64).collect();
        let y_bool: Vec<bool> = y.iter().map(|&v| v > 0.5).collect();
        let as_i64 = DataFrame::new(vec![Column::new("y".into(), y_i64), x.clone()]).unwrap();
        let as_bool = DataFrame::new(vec![Column::new("y".into(), y_bool), x]).unwrap();
        for run in [run_sar_probit, run_sem_probit] {
            let f = run(&ds, "y", &["x"], &mut w.clone(), short_chain());
            assert!(f.is_ok(), "{f:?}");
            for df in [&as_i64, &as_bool] {
                let r = run(
                    &Dataset::new(df.clone()),
                    "y",
                    &["x"],
                    &mut w.clone(),
                    short_chain(),
                );
                assert_eq!(format!("{r:?}"), format!("{f:?}"));
            }
        }
    }

    #[test]
    fn test_na_spatial_probit_missing_outcome() {
        let (ds, w) = create_test_spatial_binary_data();
        let y: Vec<Option<f64>> = outcome(&ds)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i != 4).then_some(v))
            .collect();
        let mut df = ds.df().clone();
        df.with_column(Column::new("y".into(), y)).unwrap();
        let e = run_sar_probit(
            &Dataset::new(df),
            "y",
            &["x"],
            &mut w.clone(),
            short_chain(),
        )
        .unwrap_err();
        assert!(
            matches!(e, EconError::NullValues { ref column, count: 1 } if column == "y"),
            "{e:?}"
        );
    }
}
