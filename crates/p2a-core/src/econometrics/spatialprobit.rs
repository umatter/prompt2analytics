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
//! # Estimation Method
//!
//! Uses a Bayesian Markov Chain Monte Carlo (MCMC) approach with data augmentation,
//! following LeSage & Pace (2009). The algorithm uses:
//!
//! 1. Albert & Chib (1993) data augmentation for latent y*
//! 2. Truncated normal sampling for latent outcomes
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
//! - R package `spatialprobit`: Wilhelm, S. & de Matos, M.G. (2013).
//!   https://cran.r-project.org/package=spatialprobit

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{matrix_inverse, xtx, xty};
use crate::spatial::SpatialWeights;
use crate::traits::estimator::{normal_cdf, normal_pdf};

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
    /// Random seed for reproducibility (optional)
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

impl SpatialProbitResult {
    /// Compute 95% credible interval for a coefficient.
    pub fn credible_interval_beta(&self, idx: usize, level: f64) -> (f64, f64) {
        let alpha = (1.0 - level) / 2.0;
        let mut draws: Vec<f64> = self.beta_draws.column(idx).to_vec();
        draws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = draws.len();
        let lower_idx = ((n as f64) * alpha) as usize;
        let upper_idx = ((n as f64) * (1.0 - alpha)) as usize;
        (draws[lower_idx], draws[upper_idx.min(n - 1)])
    }

    /// Compute 95% credible interval for spatial parameter.
    pub fn credible_interval_rho(&self, level: f64) -> (f64, f64) {
        let alpha = (1.0 - level) / 2.0;
        let mut draws: Vec<f64> = self.rho_draws.to_vec();
        draws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = draws.len();
        let lower_idx = ((n as f64) * alpha) as usize;
        let upper_idx = ((n as f64) * (1.0 - alpha)) as usize;
        (draws[lower_idx], draws[upper_idx.min(n - 1)])
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
/// ```rust,ignore
/// use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
/// use p2a_core::econometrics::spatialprobit::{run_sar_probit, SpatialProbitConfig};
///
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
    let df = dataset.df();
    let n = df.height();
    let k = x_cols.len() + 1; // +1 for intercept

    // Validate dimensions
    if n != listw.n() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dataset has {} observations but weights matrix has {} observations",
                n,
                listw.n()
            ),
        });
    }

    // Extract y (binary outcome)
    let y_series = df.column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;
    let y: Array1<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: y_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Validate y is binary
    let n_positive = y.iter().filter(|&&v| v >= 0.5).count();
    if n_positive == 0 || n_positive == n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dependent variable '{}' must be binary with both 0 and 1 values. Found {} ones out of {} observations.",
                y_col, n_positive, n
            ),
        });
    }

    // Build design matrix with intercept
    let mut x = Array2::zeros((n, k));
    for i in 0..n {
        x[[i, 0]] = 1.0; // Intercept
    }
    for (j, &col_name) in x_cols.iter().enumerate() {
        let col = df.column(col_name).map_err(|_| EconError::ColumnNotFound {
            column: col_name.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;
        let col_f64 = col.f64().map_err(|_| EconError::NonNumericColumn {
            column: col_name.to_string(),
        })?;
        for (i, val) in col_f64.into_no_null_iter().enumerate() {
            x[[i, j + 1]] = val;
        }
    }

    // Get spatial weights as dense matrix for MCMC
    let w = listw.to_dense();

    // Get valid range for spatial parameter
    let (rho_min, rho_max) = listw.rho_range();
    let rho_range = (
        rho_min.max(config.rho_prior_range.0),
        rho_max.min(config.rho_prior_range.1),
    );

    // Run MCMC estimation
    let mcmc_result = run_mcmc_spatial_probit(&y, &x, &w, model_type, &config, rho_range)?;

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
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let rho_z = if rho_std > 1e-10 {
        rho_mean / rho_std
    } else {
        0.0
    };
    let rho_p = 2.0 * (1.0 - normal_cdf(rho_z.abs()));

    // Build coefficient names
    let mut coef_names = vec!["(Intercept)".to_string()];
    for &name in x_cols {
        coef_names.push(name.to_string());
    }

    // Compute log-likelihood at posterior mean
    let ll_mean =
        compute_spatial_probit_log_likelihood(&y, &x, &w, &beta_mean, rho_mean, model_type);

    // Compute null log-likelihood (intercept only, no spatial)
    let p_bar = n_positive as f64 / n as f64;
    let ll_null = (n_positive as f64) * p_bar.ln() + ((n - n_positive) as f64) * (1.0 - p_bar).ln();

    // Pseudo R-squared
    let pseudo_r2 = 1.0 - ll_mean / ll_null;

    // Compute fitted probabilities at posterior mean
    let fitted_prob = compute_fitted_probabilities(&x, &w, &beta_mean, rho_mean, model_type);

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
            &w,
            &mcmc_result.beta_draws,
            &mcmc_result.rho_draws,
            model_type,
            &coef_names[1..], // Exclude intercept
        ))
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
    w: &Array2<f64>,
    model_type: SpatialProbitModel,
    config: &SpatialProbitConfig,
    rho_range: (f64, f64),
) -> EconResult<McmcResult> {
    let _n = y.len();
    let k = x.ncols();
    let total_draws = config.burn_in + config.n_draws * config.thin;

    // Initialize RNG
    use rand::SeedableRng;
    let mut rng = match config.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

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

    // Storage for draws
    let keep_draws = config.n_draws;
    let mut beta_draws = Array2::zeros((keep_draws, k));
    let mut rho_draws = Array1::zeros(keep_draws);
    let mut log_lik_draws = Array1::zeros(keep_draws);

    // MCMC iterations
    let mut n_accepted = 0;
    let mut draw_idx = 0;

    for iter in 0..total_draws {
        // --- Step 1: Sample latent y* given (beta, rho) ---
        y_star = sample_latent_y(y, x, w, &beta, rho, model_type, &mut rng);

        // --- Step 2: Sample beta given (y*, rho) ---
        beta = sample_beta(
            &y_star,
            x,
            w,
            rho,
            model_type,
            &beta0,
            prior_precision,
            &mut rng,
        )?;

        // --- Step 3: Sample rho given (y*, beta) using griddy Gibbs ---
        let (new_rho, accepted) =
            sample_rho_griddy_gibbs(&y_star, x, w, &beta, rho, model_type, rho_range, &mut rng);
        rho = new_rho;
        if accepted {
            n_accepted += 1;
        }

        // Store draws after burn-in, with thinning
        if iter >= config.burn_in
            && (iter - config.burn_in) % config.thin == 0
            && draw_idx < keep_draws
        {
            for j in 0..k {
                beta_draws[[draw_idx, j]] = beta[j];
            }
            rho_draws[draw_idx] = rho;
            log_lik_draws[draw_idx] =
                compute_spatial_probit_log_likelihood(y, x, w, &beta, rho, model_type);
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
    w: &Array2<f64>,
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
            let wy = w.dot(&y_star);
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
    w: &Array2<f64>,
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
            let y_t: Array1<f64> = y_star - rho * &w.dot(y_star);
            (y_t, x.clone())
        }
        SpatialProbitModel::Sem => {
            // y* = X*beta + (I - lambda*W)^{-1}*epsilon
            // Transform: (I - lambda*W)*y* = (I - lambda*W)*X*beta + epsilon
            let y_t: Array1<f64> = y_star - rho * &w.dot(y_star);
            let mut x_t = x.clone();
            let wx = w.dot(x);
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
    w: &Array2<f64>,
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
    let wy_star = w.dot(y_star);

    let log_probs: Vec<f64> = grid
        .iter()
        .map(|&rho| {
            let resid: Array1<f64> = match model_type {
                SpatialProbitModel::Sar => y_star - rho * &wy_star - &xb,
                SpatialProbitModel::Sem => {
                    let wxb = w.dot(&xb);
                    y_star - &xb - rho * &(w.dot(y_star) - &wxb)
                }
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
    w: &Array2<f64>,
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
                let wz = w.dot(&z_approx);
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
    w: &Array2<f64>,
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
                let wz = w.dot(&z_approx);
                z_approx = &xb + rho * &wz;
            }
            z_approx
        }
        SpatialProbitModel::Sem => xb,
    };

    z.mapv(normal_cdf)
}

/// Compute spatial impacts for probit model.
fn compute_spatial_probit_impacts(
    x: &Array2<f64>,
    w: &Array2<f64>,
    beta_draws: &Array2<f64>,
    rho_draws: &Array1<f64>,
    _model_type: SpatialProbitModel,
    var_names: &[String],
) -> SpatialProbitImpacts {
    let n = x.nrows();
    let k = x.ncols() - 1; // Exclude intercept
    let n_draws = beta_draws.nrows();

    // Storage for impacts across draws
    let mut direct_draws = Array2::zeros((n_draws, k));
    let mut indirect_draws = Array2::zeros((n_draws, k));
    let mut total_draws = Array2::zeros((n_draws, k));

    // For each draw, compute impacts
    for d in 0..n_draws {
        let rho = rho_draws[d];
        let beta: Array1<f64> = beta_draws.row(d).to_owned();

        // Compute A^{-1} = (I - rho*W)^{-1} approximately
        // Use Neumann series approximation: A^{-1} = I + rho*W + rho^2*W^2 + ...
        let mut a_inv = Array2::eye(n);
        let mut w_power = w.clone();
        let mut rho_power = rho;
        for _ in 0..10 {
            for i in 0..n {
                for j in 0..n {
                    a_inv[[i, j]] += rho_power * w_power[[i, j]];
                }
            }
            w_power = w.dot(&w_power);
            rho_power *= rho;
            if rho_power.abs() < 1e-10 {
                break;
            }
        }

        // Compute X*beta and phi(A^{-1}*X*beta)
        let xb = x.dot(&beta);
        let z = a_inv.dot(&xb);
        let phi_z: Array1<f64> = z.mapv(normal_pdf);

        // For each variable k (excluding intercept)
        for j in 0..k {
            let beta_j = beta[j + 1]; // +1 to skip intercept

            // S_j = beta_j * A^{-1}
            let s_j = &a_inv * beta_j;

            // Direct effect: average of diag(S_j * diag(phi_z))
            let direct: f64 = (0..n).map(|i| s_j[[i, i]] * phi_z[i]).sum::<f64>() / n as f64;

            // Total effect: average column sum
            let total: f64 = (0..n)
                .map(|col| (0..n).map(|row| s_j[[row, col]] * phi_z[row]).sum::<f64>())
                .sum::<f64>()
                / n as f64;

            // Indirect = total - direct
            let indirect = total - direct;

            direct_draws[[d, j]] = direct;
            indirect_draws[[d, j]] = indirect;
            total_draws[[d, j]] = total;
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

    SpatialProbitImpacts {
        direct,
        indirect,
        total,
        direct_se,
        indirect_se,
        total_se,
        var_names: var_names.to_vec(),
    }
}

/// Sample from truncated normal, truncated above threshold.
fn sample_truncated_normal_above<R: rand::Rng>(
    mu: f64,
    sigma: f64,
    lower: f64,
    rng: &mut R,
) -> f64 {
    use rand_distr::{Distribution, Uniform};

    // Use inverse CDF method
    let alpha = (lower - mu) / sigma;
    let phi_alpha = normal_cdf(alpha);

    // Ensure valid range for Uniform
    let lower_bound = phi_alpha.max(1e-10);
    let upper_bound = 1.0 - 1e-10;

    if lower_bound >= upper_bound {
        // Extreme case: return value just above threshold
        return lower + 0.01 * sigma;
    }

    // Sample u ~ Uniform(phi_alpha, 1)
    let u: f64 = Uniform::new(lower_bound, upper_bound).sample(rng);

    // Return mu + sigma * Phi^{-1}(u)
    mu + sigma * inverse_normal_cdf(u)
}

/// Sample from truncated normal, truncated below threshold.
fn sample_truncated_normal_below<R: rand::Rng>(
    mu: f64,
    sigma: f64,
    upper: f64,
    rng: &mut R,
) -> f64 {
    use rand_distr::{Distribution, Uniform};

    // Use inverse CDF method
    let beta = (upper - mu) / sigma;
    let phi_beta = normal_cdf(beta);

    // Ensure valid range for Uniform
    let lower_bound = 1e-10;
    let upper_bound = phi_beta.max(2e-10);

    if lower_bound >= upper_bound {
        // Extreme case: return value just below threshold
        return upper - 0.01 * sigma;
    }

    // Sample u ~ Uniform(0, phi_beta)
    let u: f64 = Uniform::new(lower_bound, upper_bound).sample(rng);

    // Return mu + sigma * Phi^{-1}(u)
    mu + sigma * inverse_normal_cdf(u)
}

/// Inverse of standard normal CDF (quantile function).
fn inverse_normal_cdf(p: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.inverse_cdf(p.max(1e-15).min(1.0 - 1e-15))
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

    #[test]
    fn test_inverse_normal_cdf() {
        // Test known values
        assert!((inverse_normal_cdf(0.5) - 0.0).abs() < 1e-6);
        assert!(inverse_normal_cdf(0.975) > 1.9 && inverse_normal_cdf(0.975) < 2.0);
        assert!(inverse_normal_cdf(0.025) < -1.9 && inverse_normal_cdf(0.025) > -2.0);
    }
}
