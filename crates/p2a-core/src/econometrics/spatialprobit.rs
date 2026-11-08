//! Spatial Probit Models for Binary Outcomes with Spatial Dependence.
//!
//! Implements spatial autoregressive (SAR) probit and spatial error (SEM) probit models
//! for binary dependent variables with spatial dependence, equivalent to R's `spatialprobit`
//! package.
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
//! Uses a Bayesian Markov Chain Monte Carlo (MCMC) approach with data augmentation,
//! following LeSage & Pace (2009). The algorithm uses:
//!
//! 1. Albert & Chib (1993) data augmentation for latent y*
//! 2. Exact one-sided truncated normal draws for the latent outcomes, in both tails: normal
//!    rejection when the bound is on the near side of the mean, Robert's (1995) exponential
//!    rejection when it is on the far side
//! 3. Gibbs sampling for parameters
//!
//! # Marginal Effects
//!
//! For spatial probit models, marginal effects are computed as:
//!
//! - **Direct effect**: Average diagonal element of S_k * phi(A^{-1}*X*beta)
//! - **Indirect effect**: Average off-diagonal element of S_k * phi(A^{-1}*X*beta)
//! - **Total effect**: Average column sum of S_k * phi(A^{-1}*X*beta)
//!
//! where S_k = A^{-1} * beta_k, A = I - rho*W, and phi is the standard normal PDF.
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

    /// Log-likelihood at posterior mean
    pub log_likelihood: f64,
    /// Log marginal likelihood (approximation via harmonic mean)
    pub log_marginal_likelihood: f64,
    /// Deviance Information Criterion
    pub dic: f64,
    /// Pseudo R-squared (McFadden)
    pub pseudo_r_squared: f64,

    /// Predicted probabilities at posterior mean
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
    /// Acceptance rate for spatial parameter (if Metropolis-Hastings used)
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
        writeln!(f, "Log-likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "DIC: {:.4}", self.dic)?;
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

    // Get valid range for spatial parameter
    let (rho_min, rho_max) = listw.try_rho_range()?;
    let rho_range = (
        rho_min.max(config.rho_prior_range.0),
        rho_max.min(config.rho_prior_range.1),
    );

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

    // Compute log-likelihood at posterior mean
    let ll_mean =
        compute_spatial_probit_log_likelihood(&y, &x, listw, &beta_mean, rho_mean, model_type);

    // Compute null log-likelihood (intercept only, no spatial)
    let p_bar = n_positive as f64 / n as f64;
    let ll_null = (n_positive as f64) * p_bar.ln() + ((n - n_positive) as f64) * (1.0 - p_bar).ln();

    // Pseudo R-squared
    let pseudo_r2 = 1.0 - ll_mean / ll_null;

    // Compute fitted probabilities at posterior mean
    let fitted_prob = compute_fitted_probabilities(&x, listw, &beta_mean, rho_mean, model_type);

    // Percent correctly predicted
    let correct: usize = y
        .iter()
        .zip(fitted_prob.iter())
        .filter(|&(&yi, &pi)| (yi >= 0.5) == (pi >= 0.5))
        .count();
    let pcp = 100.0 * correct as f64 / n as f64;

    // Approximate log marginal likelihood via harmonic mean estimator
    // (Newton & Raftery, 1994) - biased but simple
    let log_marginal_ll = compute_log_marginal_likelihood_approx(&mcmc_result.log_lik_draws);

    // DIC = D_bar + p_D where p_D = D_bar - D(theta_bar)
    let d_bar = -2.0 * mcmc_result.log_lik_draws.mean().unwrap_or(ll_mean);
    let d_theta_bar = -2.0 * ll_mean;
    let p_d = d_bar - d_theta_bar;
    let dic = d_bar + p_d;

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
        log_marginal_likelihood: log_marginal_ll,
        dic,
        pseudo_r_squared: pseudo_r2,
        fitted_prob: fitted_prob.to_vec(),
        pcp,
        impacts,
        n_obs: n,
        n_positive,
        n_draws,
        acceptance_rate: mcmc_result.acceptance_rate,
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
    log_lik_draws: Array1<f64>,
    acceptance_rate: f64,
}

/// Run MCMC for spatial probit model.
///
/// Uses Gibbs sampling with:
/// - Data augmentation for latent y* (Albert & Chib, 1993)
/// - Conjugate update for beta
/// - Griddy Gibbs for rho (Ritter & Tanner, 1992)
fn run_mcmc_spatial_probit(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    model_type: SpatialProbitModel,
    config: &SpatialProbitConfig,
    rho_range: (f64, f64),
) -> EconResult<McmcResult> {
    let _n = y.len();
    let k = x.ncols();
    let total_draws = total_mcmc_draws(config)?;

    // Initialize RNG
    use rand::SeedableRng;
    let mut rng =
        rand::rngs::StdRng::seed_from_u64(config.seed.unwrap_or(SPATIAL_PROBIT_DEFAULT_SEED));

    // Prior parameters
    let prior_precision = config.prior_beta_precision.unwrap_or(0.01);
    let beta0 = config
        .prior_beta_mean
        .clone()
        .unwrap_or_else(|| Array1::zeros(k));

    // Initialize parameters
    // Start beta from OLS on pseudo-y
    let pseudo_y: Array1<f64> = y.mapv(|yi| if yi >= 0.5 { 0.5 } else { -0.5 });
    let xtx_mat = xtx(&x.view());
    let xty_vec = xty(&x.view(), &pseudo_y);
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;
    let mut beta = xtx_inv.dot(&xty_vec);

    let mut rho = 0.0; // Start at no spatial dependence
    let mut y_star = pseudo_y.clone(); // Latent variable

    // WX does not change between iterations
    let mut wx = Array2::zeros((x.nrows(), k));
    for col in 0..k {
        wx.column_mut(col)
            .assign(&listw.lag(&x.column(col).to_owned()));
    }

    // Storage for draws
    let keep_draws = config.n_draws;
    let mut beta_draws = try_zeros2(keep_draws, k, "posterior draws of beta")?;
    let mut rho_draws = try_zeros1(keep_draws, "posterior draws of rho")?;
    let mut log_lik_draws = try_zeros1(keep_draws, "posterior log-likelihood draws")?;

    // MCMC iterations
    let mut n_accepted = 0;
    let mut draw_idx = 0;

    for iter in 0..total_draws {
        crate::resource::checkpoint()?;

        // --- Step 1: Sample latent y* given (beta, rho) ---
        y_star = sample_latent_y(y, x, listw, &beta, rho, model_type, &mut rng);

        // --- Step 2: Sample beta given (y*, rho) ---
        beta = sample_beta(
            &y_star,
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

        // --- Step 3: Sample rho given (y*, beta) using griddy Gibbs ---
        let (new_rho, accepted) = sample_rho_griddy_gibbs(
            &y_star, x, listw, &beta, rho, model_type, rho_range, &mut rng,
        );
        rho = new_rho;
        if accepted {
            n_accepted += 1;
        }

        // Store draws after burn-in, with thinning
        if iter >= config.burn_in
            && (iter - config.burn_in).is_multiple_of(config.thin)
            && draw_idx < keep_draws
        {
            for j in 0..k {
                beta_draws[[draw_idx, j]] = beta[j];
            }
            rho_draws[draw_idx] = rho;
            log_lik_draws[draw_idx] =
                compute_spatial_probit_log_likelihood(y, x, listw, &beta, rho, model_type);
            draw_idx += 1;
        }
    }

    let acceptance_rate = n_accepted as f64 / total_draws as f64;

    Ok(McmcResult {
        beta_draws,
        rho_draws,
        log_lik_draws,
        acceptance_rate,
    })
}

/// Sample latent y* from truncated normal given observed y, parameters.
fn sample_latent_y<R: rand::Rng>(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    beta: &Array1<f64>,
    rho: f64,
    model_type: SpatialProbitModel,
    rng: &mut R,
) -> Array1<f64> {
    let n = y.len();
    let mut y_star = Array1::zeros(n);

    // Compute mean of y* based on model type
    let mu = match model_type {
        SpatialProbitModel::Sar => {
            // y* = (I - rho*W)^{-1} * X * beta + (I - rho*W)^{-1} * epsilon
            // Mean: (I - rho*W)^{-1} * X * beta
            // For sampling, we use iterative approach

            // Approximate: mu_i = (X*beta)_i + rho * sum_j(w_ij * y*_j)
            // This is a simplified approximation
            x.dot(beta)
        }
        SpatialProbitModel::Sem => {
            // y* = X*beta + u, Var(y*) = (I - lambda*W)^{-1} * (I - lambda*W')^{-1}
            // Mean is simply X*beta
            x.dot(beta)
        }
    };

    // Sample from truncated normal for each observation
    for i in 0..n {
        let mu_i = mu[i];
        let sigma = 1.0; // Unit variance in probit

        if y[i] >= 0.5 {
            // y_i = 1: sample from TN(mu, sigma) truncated at (0, inf)
            y_star[i] = sample_truncated_normal_above(mu_i, sigma, 0.0, rng);
        } else {
            // y_i = 0: sample from TN(mu, sigma) truncated at (-inf, 0)
            y_star[i] = sample_truncated_normal_below(mu_i, sigma, 0.0, rng);
        }
    }

    // For SAR model, iteratively adjust for spatial lag
    if model_type == SpatialProbitModel::Sar && rho.abs() > 1e-6 {
        // Apply one iteration of adjustment: y* = X*beta + rho*W*y*_old
        let xb = x.dot(beta);
        for _ in 0..3 {
            // A few iterations for convergence
            let wy = listw.lag(&y_star);
            for i in 0..n {
                let new_mu = xb[i] + rho * wy[i];
                if y[i] >= 0.5 {
                    y_star[i] = sample_truncated_normal_above(new_mu, 1.0, 0.0, rng);
                } else {
                    y_star[i] = sample_truncated_normal_below(new_mu, 1.0, 0.0, rng);
                }
            }
        }
    }

    y_star
}

/// Sample beta from conditional posterior.
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
            let mut x_t = x.clone();
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    x_t[[i, j]] -= rho * wx[[i, j]];
                }
            }
            (y_t, x_t)
        }
    };

    // Posterior precision: X'X + prior_precision * I
    let xtx_mat = xtx(&x_tilde.view());
    let mut post_precision = xtx_mat.clone();
    for i in 0..k {
        post_precision[[i, i]] += prior_precision;
    }

    // Posterior mean: post_var * (X'y_tilde + prior_precision * beta0)
    let xty_vec = xty(&x_tilde.view(), &y_tilde);
    let prior_term = beta0.mapv(|b| b * prior_precision);
    let post_mean_unnorm = &xty_vec + &prior_term;

    let post_var = matrix_inverse(&post_precision.view())?;
    let post_mean = post_var.dot(&post_mean_unnorm);

    // Sample from N(post_mean, post_var)
    // Using Cholesky: beta = post_mean + L * z where L*L' = post_var
    let l = crate::linalg::matrix_ops::cholesky(&post_var.view())?;
    let z: Array1<f64> = Array1::from_iter((0..k).map(|_| {
        use rand_distr::{Distribution, StandardNormal};
        StandardNormal.sample(rng)
    }));
    let beta = &post_mean + &l.dot(&z);

    Ok(beta)
}

/// Sample rho using griddy Gibbs (grid-based evaluation).
fn sample_rho_griddy_gibbs<R: rand::Rng>(
    y_star: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    beta: &Array1<f64>,
    current_rho: f64,
    model_type: SpatialProbitModel,
    rho_range: (f64, f64),
    rng: &mut R,
) -> (f64, bool) {
    // Create grid of rho values
    let n_grid = 50;
    let step = (rho_range.1 - rho_range.0) / (n_grid as f64);
    let grid: Vec<f64> = (0..=n_grid)
        .map(|i| rho_range.0 + i as f64 * step)
        .collect();

    // Evaluate log-likelihood kernel at each grid point
    let xb = x.dot(beta);
    let wy_star = listw.lag(y_star);
    // W y* - W X beta, for the SEM residual; computed once, not per grid point
    let dw = match model_type {
        SpatialProbitModel::Sar => None,
        SpatialProbitModel::Sem => Some(&wy_star - &listw.lag(&xb)),
    };

    let log_probs: Vec<f64> = grid
        .iter()
        .map(|&rho| {
            let resid: Array1<f64> = match &dw {
                None => y_star - rho * &wy_star - &xb,
                Some(dw) => y_star - &xb - rho * dw,
            };
            // Log-likelihood contribution (ignoring normalizing constants)
            -0.5 * resid.iter().map(|&r| r * r).sum::<f64>()
        })
        .collect();

    // Convert to probabilities (softmax-like)
    let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_log).exp()).collect();
    let sum_probs: f64 = probs.iter().sum();
    let probs_normalized: Vec<f64> = probs.iter().map(|&p| p / sum_probs).collect();

    // Sample from discrete distribution
    use rand_distr::{Distribution, Uniform};
    let u: f64 = Uniform::new(0.0, 1.0).sample(rng);
    let mut cumsum = 0.0;
    let mut selected_idx = 0;
    for (i, &p) in probs_normalized.iter().enumerate() {
        cumsum += p;
        if u <= cumsum {
            selected_idx = i;
            break;
        }
    }

    // Add small jitter within grid cell
    let new_rho = if selected_idx == 0 {
        grid[0] + Uniform::new(0.0, step).sample(rng) * 0.5
    } else if selected_idx == n_grid {
        grid[n_grid] - Uniform::new(0.0, step).sample(rng) * 0.5
    } else {
        grid[selected_idx] + Uniform::new(-step * 0.5, step * 0.5).sample(rng)
    };

    // Clamp to valid range
    let new_rho_clamped = new_rho.max(rho_range.0 + 0.001).min(rho_range.1 - 0.001);

    (
        new_rho_clamped,
        (new_rho_clamped - current_rho).abs() > 1e-8,
    )
}

/// Compute log-likelihood for spatial probit model.
fn compute_spatial_probit_log_likelihood(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    beta: &Array1<f64>,
    rho: f64,
    model_type: SpatialProbitModel,
) -> f64 {
    let xb = x.dot(beta);

    // Compute z = A^{-1} * X * beta where A = I - rho*W
    let z: Array1<f64> = match model_type {
        SpatialProbitModel::Sar => {
            // Use iterative approximation for (I - rho*W)^{-1} * X * beta
            let mut z_approx = xb.clone();
            for _ in 0..5 {
                let wz = listw.lag(&z_approx);
                z_approx = &xb + rho * &wz;
            }
            z_approx
        }
        SpatialProbitModel::Sem => {
            // For SEM, mean is just X*beta
            xb
        }
    };

    // Log-likelihood: sum of log(Phi(z_i)) for y_i=1, log(1-Phi(z_i)) for y_i=0
    let ll: f64 = y
        .iter()
        .zip(z.iter())
        .map(|(&yi, &zi)| {
            let p = normal_cdf(zi);
            let p_clipped = p.max(1e-10).min(1.0 - 1e-10);
            if yi >= 0.5 {
                p_clipped.ln()
            } else {
                (1.0 - p_clipped).ln()
            }
        })
        .sum();

    ll
}

/// Compute fitted probabilities.
fn compute_fitted_probabilities(
    x: &Array2<f64>,
    listw: &SpatialWeights,
    beta: &Array1<f64>,
    rho: f64,
    model_type: SpatialProbitModel,
) -> Array1<f64> {
    let xb = x.dot(beta);

    let z: Array1<f64> = match model_type {
        SpatialProbitModel::Sar => {
            // Approximate (I - rho*W)^{-1} * X * beta iteratively
            let mut z_approx = xb.clone();
            for _ in 0..10 {
                let wz = listw.lag(&z_approx);
                z_approx = &xb + rho * &wz;
            }
            z_approx
        }
        SpatialProbitModel::Sem => xb,
    };

    z.mapv(normal_cdf)
}

/// Highest power of W in the truncated Neumann series the probit impacts use.
const NEUMANN_MAX_POWER: usize = 10;

/// `diag(W^p)` and `W^p 1` for p = 0..=[`NEUMANN_MAX_POWER`], one row per power.
struct NeumannTerms {
    /// `diag[[p, i]] = (W^p)_ii`
    diag: Array2<f64>,
    /// `rowsum[[p, i]] = (W^p 1)_i`
    rowsum: Array2<f64>,
}

/// The powers of W the impacts need, computed once per fit from the sparse rows: the row
/// sums by repeated lags, and each diagonal entry by propagating the row vector e_i' W^p.
/// Costs at most O(10 n links), against ten dense n x n products per draw before.
fn neumann_terms(listw: &SpatialWeights) -> EconResult<NeumannTerms> {
    let n = listw.n();
    let p_max = NEUMANN_MAX_POWER;
    let mut diag = try_zeros2(p_max + 1, n, "diagonals of the powers of W")?;
    let mut rowsum = try_zeros2(p_max + 1, n, "row sums of the powers of W")?;

    // W^p 1 by repeated lags
    let mut v = Array1::ones(n);
    rowsum.row_mut(0).assign(&v);
    for p in 1..=p_max {
        v = listw.lag(&v);
        rowsum.row_mut(p).assign(&v);
    }

    // (W^p)_ii: propagate r = e_i' W^p through the sparse rows. `cur`/`next` are dense
    // scratch rows; the index lists name their non-zero entries so they reset cheaply.
    let mut cur = vec![0.0; n];
    let mut next = vec![0.0; n];
    let mut in_next = vec![false; n];
    let mut cur_idx: Vec<usize> = Vec::new();
    let mut next_idx: Vec<usize> = Vec::new();
    for i in 0..n {
        crate::resource::checkpoint()?;
        diag[[0, i]] = 1.0;
        cur[i] = 1.0;
        cur_idx.clear();
        cur_idx.push(i);
        for p in 1..=p_max {
            for &m in &cur_idx {
                let r = cur[m];
                let sw = listw.get_weights(m);
                for (&j, &w) in sw.indices.iter().zip(sw.weights.iter()) {
                    if !in_next[j] {
                        in_next[j] = true;
                        next_idx.push(j);
                    }
                    next[j] += r * w;
                }
                cur[m] = 0.0;
            }
            std::mem::swap(&mut cur, &mut next);
            std::mem::swap(&mut cur_idx, &mut next_idx);
            next_idx.clear();
            for &j in &cur_idx {
                in_next[j] = false;
            }
            diag[[p, i]] = cur[i];
        }
        for &m in &cur_idx {
            cur[m] = 0.0;
        }
    }
    Ok(NeumannTerms { diag, rowsum })
}

/// Direct, indirect and total impacts of one posterior draw, from the truncated Neumann
/// series A = I + Σ_p ρ^p W^p (the terms up to the first with |ρ^(p+1)| < 1e-10, at most
/// [`NEUMANN_MAX_POWER`]): z = A Xβ, φ = dnorm(z), direct_j = β_j Σ_i φ_i A_ii / n and
/// total_j = β_j Σ_i φ_i (A 1)_i / n.
fn probit_impacts_one_draw(
    x: &Array2<f64>,
    listw: &SpatialWeights,
    terms: &NeumannTerms,
    beta: &Array1<f64>,
    rho: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.nrows();
    let k = x.ncols() - 1; // Exclude intercept

    // Series coefficients ρ^p by repeated multiplication, as the dense loop built them
    let mut coefs = vec![1.0];
    let mut c = rho;
    for _ in 0..NEUMANN_MAX_POWER {
        coefs.push(c);
        c *= rho;
        if c.abs() < 1e-10 {
            break;
        }
    }

    // z = A Xβ by repeated lags
    let xb = x.dot(beta);
    let mut z = xb.clone();
    let mut v = xb;
    for &cp in &coefs[1..] {
        v = listw.lag(&v);
        z.scaled_add(cp, &v);
    }
    let phi_z: Array1<f64> = z.mapv(normal_pdf);

    // Σ_i φ_i A_ii and Σ_i φ_i (A 1)_i
    let mut diag_sum = 0.0;
    let mut row_sum = 0.0;
    for i in 0..n {
        let mut a_ii = 0.0;
        let mut a_i1 = 0.0;
        for (p, &cp) in coefs.iter().enumerate() {
            a_ii += cp * terms.diag[[p, i]];
            a_i1 += cp * terms.rowsum[[p, i]];
        }
        diag_sum += phi_z[i] * a_ii;
        row_sum += phi_z[i] * a_i1;
    }

    let n_f = n as f64;
    let direct: Vec<f64> = (0..k).map(|j| beta[j + 1] * diag_sum / n_f).collect();
    let total: Vec<f64> = (0..k).map(|j| beta[j + 1] * row_sum / n_f).collect();
    let indirect: Vec<f64> = total.iter().zip(&direct).map(|(t, d)| t - d).collect();
    (direct, indirect, total)
}

/// Compute spatial impacts for probit model.
fn compute_spatial_probit_impacts(
    x: &Array2<f64>,
    listw: &SpatialWeights,
    beta_draws: &Array2<f64>,
    rho_draws: &Array1<f64>,
    _model_type: SpatialProbitModel,
    var_names: &[String],
) -> EconResult<SpatialProbitImpacts> {
    let k = x.ncols() - 1; // Exclude intercept
    let n_draws = beta_draws.nrows();

    let terms = neumann_terms(listw)?;

    // Storage for impacts across draws
    let mut direct_draws = try_zeros2(n_draws, k, "direct impact draws")?;
    let mut indirect_draws = try_zeros2(n_draws, k, "indirect impact draws")?;
    let mut total_draws = try_zeros2(n_draws, k, "total impact draws")?;

    // For each draw, compute impacts
    for d in 0..n_draws {
        crate::resource::checkpoint()?;
        let beta: Array1<f64> = beta_draws.row(d).to_owned();
        let (direct, indirect, total) =
            probit_impacts_one_draw(x, listw, &terms, &beta, rho_draws[d]);
        for j in 0..k {
            direct_draws[[d, j]] = direct[j];
            indirect_draws[[d, j]] = indirect[j];
            total_draws[[d, j]] = total[j];
        }
    }

    // Compute posterior means and standard deviations
    let direct: Vec<f64> = (0..k)
        .map(|j| direct_draws.column(j).mean().unwrap_or(0.0))
        .collect();
    let indirect: Vec<f64> = (0..k)
        .map(|j| indirect_draws.column(j).mean().unwrap_or(0.0))
        .collect();
    let total: Vec<f64> = (0..k)
        .map(|j| total_draws.column(j).mean().unwrap_or(0.0))
        .collect();

    let direct_se: Vec<f64> = (0..k)
        .map(|j| compute_std(&direct_draws.column(j).to_owned()))
        .collect();
    let indirect_se: Vec<f64> = (0..k)
        .map(|j| compute_std(&indirect_draws.column(j).to_owned()))
        .collect();
    let total_se: Vec<f64> = (0..k)
        .map(|j| compute_std(&total_draws.column(j).to_owned()))
        .collect();

    Ok(SpatialProbitImpacts {
        direct,
        indirect,
        total,
        direct_se,
        indirect_se,
        total_se,
        var_names: var_names.to_vec(),
    })
}

/// Exact draw from N(0, 1) truncated to [a, inf). For a <= 0, normal rejection (acceptance
/// Phi(-a) >= 1/2). Otherwise Robert's (1995) exponential rejection with the optimal rate
/// lambda = (a + sqrt(a^2 + 4)) / 2, accepting z = a + E/lambda with probability
/// exp(-(z - lambda)^2 / 2) (acceptance above 0.76 for a > 0). NaN or +inf returns a (no loop);
/// the Gibbs loop refuses non-finite draws.
fn standard_normal_truncated_below<R: rand::Rng>(a: f64, rng: &mut R) -> f64 {
    use rand_distr::{Distribution, Exp1, StandardNormal};
    if a.is_nan() || a == f64::INFINITY {
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

/// Approximate log marginal likelihood using harmonic mean estimator.
fn compute_log_marginal_likelihood_approx(log_lik_draws: &Array1<f64>) -> f64 {
    // Harmonic mean estimator: 1 / (1/n * sum(1/L))
    // In log: -log(mean(exp(-log_lik)))
    let max_ll = log_lik_draws
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_exp_neg: f64 = log_lik_draws
        .iter()
        .map(|&ll| (-(ll - max_ll)).exp())
        .sum::<f64>()
        / log_lik_draws.len() as f64;

    max_ll - mean_exp_neg.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Neighbors, WeightStyle};
    use polars::prelude::*;

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

    // R 4.3.3 (fixture): X <- cbind(1, x1); xb <- X %*% beta; A <- diag(n); Wp <- W; rp <- rho
    //   for (it in 1:10) { A <- A + rp*Wp; Wp <- W %*% Wp; rp <- rp*rho; if (abs(rp) < 1e-10) break }
    //   z <- A %*% xb; phi <- dnorm(z); direct <- sum(beta[2]*diag(A)*phi)/n; total <- sum(beta[2]*A*phi)/n
    //   (the existing truncated Neumann series, transcribed)
    #[test]
    fn probit_draw_impacts_match_r_reference() {
        use crate::spatial::test_fixtures::{X1, knn3};
        use ndarray::array;
        let mut x = Array2::ones((30, 2));
        for (i, &v) in X1.iter().enumerate() {
            x[[i, 1]] = v;
        }
        for (w, rho, d, i, t) in [
            (
                knn3(WeightStyle::RowStd),
                0.4,
                0.07423545257844562,
                0.04334337166509111,
                0.11757882424353673,
            ),
            (
                knn3(WeightStyle::RowStd),
                0.0,
                0.1181903507676808,
                0.0,
                0.1181903507676808,
            ),
            (
                knn3(WeightStyle::Binary),
                -0.3,
                0.17835073842548715,
                -0.07465542591941698,
                0.10369531250607017,
            ),
        ] {
            let terms = neumann_terms(&w).unwrap();
            let (dd, ii, tt) = probit_impacts_one_draw(&x, &w, &terms, &array![0.2, 0.7], rho);
            assert!(
                (dd[0] - d).abs() < 1e-12 && (ii[0] - i).abs() < 1e-12 && (tt[0] - t).abs() < 1e-12,
                "rho={rho}: {dd:?} {ii:?} {tt:?}"
            );
        }
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
        // Pinned at the values the dense-W sampler printed. Summation order may change the
        // floating-point noise, so 1e-8 relative.
        // Re-pinned for the exact truncated-normal sampler (Robert 1995, SPAT-6).
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
                    0.21561581519656442,
                    -1.7620719814393688,
                    1.0100875460961094,
                    0.6419059802897408,
                    0.31414857742601277,
                    -10.554279814396638,
                    83.33333333333333,
                ],
            ),
            (
                &sem,
                [
                    0.375057484757089,
                    -8.136770598618815,
                    6.6014133446280585,
                    7.0002900603296965,
                    5.28338700058015,
                    -80.85099173669074,
                    76.66666666666667,
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
        close(imp.direct[0], 0.17744956884009333, "direct");
        close(imp.indirect[0], 0.05606513641769403, "indirect");
        close(imp.total[0], 0.23351470525778736, "total");
        close(imp.direct_se[0], 0.03314944949117544, "direct se");
        close(imp.indirect_se[0], 0.06918302638820466, "indirect se");
        close(imp.total_se[0], 0.07512769921233899, "total se");
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
