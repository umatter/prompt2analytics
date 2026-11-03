//! Finite mixtures: normal mixtures by EM ([`normal_mix_em`], [`mvnorm_mix_em`]; R `mixtools`) and mixtures of regressions ([`flexmix`]; R `flexmix`).

use faer::linalg::solvers::Solve;
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// =============================================================================
// FlexMix - Finite Mixture Regression
// =============================================================================

/// Result of FlexMix finite mixture regression.
///
/// # References
///
/// - Leisch, F. (2004). "FlexMix: A General Framework for Finite Mixture Models
///   and Latent Class Regression in R". Journal of Statistical Software, 11(8).
/// - R flexmix package documentation
///   Source: https://www.rdocumentation.org/packages/flexmix/versions/2.3-20/topics/flexmix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexMixResult {
    /// Component membership (posterior) probabilities for each observation
    #[serde(skip)]
    pub posterior: Array2<f64>,
    /// Hard cluster assignments (0-indexed)
    pub cluster: Vec<usize>,
    /// Regression coefficients for each component
    pub coefficients: Vec<Vec<f64>>,
    /// Residual standard deviations for each component
    pub sigma: Vec<f64>,
    /// Component weights (mixing proportions)
    pub prior: Vec<f64>,
    /// Log-likelihood
    pub loglik: f64,
    /// BIC (Bayesian Information Criterion)
    pub bic: f64,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
    /// Number of iterations
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Number of predictors (including intercept)
    pub p: usize,
}

impl std::fmt::Display for FlexMixResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FlexMix Finite Mixture Regression")?;
        writeln!(f, "==================================")?;
        writeln!(f, "Components: {}", self.k)?;
        writeln!(f, "Observations: {}", self.n)?;
        writeln!(f, "Predictors: {}", self.p)?;
        writeln!(
            f,
            "Converged: {} (iterations: {})",
            self.converged, self.n_iterations
        )?;
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
        Ok(())
    }
}

/// Fit a finite mixture regression model using the EM algorithm.
///
/// FlexMix fits a mixture of linear regression models, where each component
/// has its own regression coefficients and error variance.
///
/// # Arguments
/// * `y` - Response variable (n x 1)
/// * `x` - Predictor matrix (n x p), should include intercept column if desired
/// * `k` - Number of mixture components
/// * `max_iter` - Maximum EM iterations (default: 200)
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `seed` - Random seed for initialization
///
/// # Returns
/// * `FlexMixResult` with fitted parameters and model diagnostics
///
/// # Algorithm
///
/// 1. Initialize: Random soft assignments and fit weighted OLS per component
/// 2. E-step: Compute posterior probabilities of component membership
/// 3. M-step: Update component parameters using weighted OLS
/// 4. Repeat until convergence
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
    let n = y.nrows();
    let p = x.ncols();

    if n != x.nrows() {
        return Err(format!(
            "y ({}) and x ({}) must have same number of rows",
            n,
            x.nrows()
        ));
    }
    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n {
        return Err(format!("k ({}) cannot exceed n ({})", k, n));
    }

    let max_iterations = max_iter.unwrap_or(200);
    let tolerance = tol.unwrap_or(1e-6);

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Initialize posterior probabilities randomly
    let mut posterior = Array2::zeros((n, k));
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..k {
            posterior[[i, j]] = rng.r#gen::<f64>() + 0.1;
            sum += posterior[[i, j]];
        }
        for j in 0..k {
            posterior[[i, j]] /= sum;
        }
    }

    // Initialize component parameters
    let mut coefficients: Vec<Vec<f64>> = vec![vec![0.0; p]; k];
    let mut sigma: Vec<f64> = vec![1.0; k];
    let mut prior: Vec<f64> = vec![1.0 / k as f64; k];

    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iterations = 0;

    // Flatten y for easier access
    let y_vec: Vec<f64> = y.iter().copied().collect();

    for iter in 0..max_iterations {
        n_iterations = iter + 1;

        // M-step: Update parameters for each component
        for j in 0..k {
            // Update prior (mixing proportion)
            let nj: f64 = posterior.column(j).sum();
            prior[j] = (nj / n as f64).max(1e-10);

            // Weighted least squares for component j
            let weights: Vec<f64> = posterior.column(j).iter().copied().collect();

            // Fit weighted OLS: (X'WX)^-1 X'Wy
            let (beta, sigma_j) = weighted_ols(&x, &y_vec, &weights)?;
            coefficients[j] = beta;
            sigma[j] = sigma_j.max(1e-10);
        }

        // E-step: Update posterior probabilities
        let mut loglik = 0.0;
        for i in 0..n {
            let mut densities: Vec<f64> = Vec::with_capacity(k);
            let mut max_log_density = f64::NEG_INFINITY;

            for j in 0..k {
                // Compute predicted value
                let mut pred = 0.0;
                for l in 0..p {
                    pred += coefficients[j][l] * x[[i, l]];
                }
                let resid = y_vec[i] - pred;

                // Log Gaussian density
                let log_density = -0.5 * (resid / sigma[j]).powi(2)
                    - sigma[j].ln()
                    - 0.5 * (2.0 * std::f64::consts::PI).ln()
                    + prior[j].ln();

                densities.push(log_density);
                max_log_density = max_log_density.max(log_density);
            }

            // Log-sum-exp for numerical stability
            let log_sum: f64 = densities
                .iter()
                .map(|&d| (d - max_log_density).exp())
                .sum::<f64>()
                .ln()
                + max_log_density;

            // Update posteriors
            for j in 0..k {
                posterior[[i, j]] = (densities[j] - log_sum).exp();
            }

            loglik += log_sum;
        }

        // Check convergence
        if (loglik - prev_loglik).abs() < tolerance {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    // Hard cluster assignments
    let cluster: Vec<usize> = (0..n)
        .map(|i| {
            (0..k)
                .max_by(|&a, &b| {
                    posterior[[i, a]]
                        .partial_cmp(&posterior[[i, b]])
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0)
        })
        .collect();

    // Compute information criteria
    let n_params = k * (p + 1) + k - 1; // coefficients + sigmas + mixing proportions
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n as f64).ln();

    Ok(FlexMixResult {
        posterior,
        cluster,
        coefficients,
        sigma,
        prior,
        loglik: prev_loglik,
        bic,
        aic,
        n_iterations,
        converged,
        k,
        n,
        p,
    })
}

/// Weighted OLS: returns (coefficients, residual_std)
fn weighted_ols(
    x: &ArrayView2<f64>,
    y: &[f64],
    weights: &[f64],
) -> Result<(Vec<f64>, f64), String> {
    let n = x.nrows();
    let p = x.ncols();

    if n == 0 || p == 0 {
        return Err("Empty data".to_string());
    }

    // Check if weights sum to something meaningful
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum < 1e-10 {
        // Return zeros if no weight
        return Ok((vec![0.0; p], 1.0));
    }

    // Compute X'WX
    let mut xtwx: Array2<f64> = Array2::zeros((p, p));
    for i in 0..n {
        let w = weights[i];
        for j in 0..p {
            for l in 0..p {
                xtwx[[j, l]] += w * x[[i, j]] * x[[i, l]];
            }
        }
    }

    // Compute X'Wy
    let mut xtwy = vec![0.0; p];
    for i in 0..n {
        let w = weights[i];
        for j in 0..p {
            xtwy[j] += w * x[[i, j]] * y[i];
        }
    }

    // Solve using Cholesky (add small regularization for stability)
    for j in 0..p {
        xtwx[[j, j]] += 1e-8;
    }

    // Simple matrix solve using faer
    let xtwx_faer = faer::Mat::from_fn(p, p, |i, j| xtwx[[i, j]]);
    let chol = xtwx_faer
        .llt(faer::Side::Lower)
        .map_err(|_| "Cholesky decomposition failed - singular matrix")?;

    let xtwy_faer = faer::Mat::from_fn(p, 1, |i, _| xtwy[i]);
    let beta_faer = chol.solve(&xtwy_faer);

    let beta: Vec<f64> = (0..p).map(|i| beta_faer[(i, 0)]).collect();

    // Compute weighted residual standard deviation
    let mut wss = 0.0;
    for i in 0..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += beta[j] * x[[i, j]];
        }
        wss += weights[i] * (y[i] - pred).powi(2);
    }
    let sigma = (wss / weight_sum.max(1.0)).sqrt();

    Ok((beta, sigma))
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
///   Source: https://www.rdocumentation.org/packages/mixtools/versions/2.0.0/topics/normalmixEM
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
    /// Log-likelihood at convergence
    pub loglik: f64,
    /// Log-likelihood history
    pub loglik_history: Vec<f64>,
    /// Number of iterations
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components
    pub k: usize,
    /// Number of observations
    pub n: usize,
}

impl std::fmt::Display for NormalMixEMResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Normal Mixture EM Results")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Number of components: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
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
        Ok(())
    }
}

/// Fit a univariate normal mixture model using EM algorithm.
///
/// # Arguments
/// * `data` - Univariate data (vector)
/// * `k` - Number of mixture components
/// * `max_iterations` - Maximum EM iterations (default: 500)
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `seed` - Optional random seed
///
/// # Algorithm
///
/// EM algorithm for Gaussian mixtures:
/// 1. E-step: Compute posterior probabilities
/// 2. M-step: Update means, variances, and mixing proportions
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
    let n = data.len();

    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n {
        return Err(format!("k ({}) cannot exceed n ({})", k, n));
    }

    let max_iter = max_iterations.unwrap_or(500);
    let tolerance = tol.unwrap_or(1e-6);

    let _rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Compute data statistics for initialization
    let data_mean: f64 = data.iter().sum::<f64>() / n as f64;
    let data_var: f64 = data.iter().map(|x| (x - data_mean).powi(2)).sum::<f64>() / n as f64;
    let data_sd = data_var.sqrt();

    // Initialize parameters
    let mut mu: Vec<f64> = Vec::with_capacity(k);
    let mut sigma: Vec<f64> = vec![data_sd; k];
    let mut lambda: Vec<f64> = vec![1.0 / k as f64; k];

    // Initialize means spread across data range
    let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for i in 0..k {
        let t = (i as f64 + 0.5) / k as f64;
        mu.push(data_min + t * (data_max - data_min));
    }

    let mut posterior = Array2::zeros((n, k));
    let mut loglik_history = Vec::new();
    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iterations = 0;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // E-step: compute posterior probabilities
        let mut loglik = 0.0;
        for i in 0..n {
            let x = data[i];
            let mut log_probs = vec![0.0f64; k];
            let mut max_log_prob = f64::NEG_INFINITY;

            for j in 0..k {
                // Log of normal density weighted by mixing proportion
                let z = (x - mu[j]) / sigma[j];
                log_probs[j] = lambda[j].ln() - sigma[j].ln() - 0.5 * z * z;
                max_log_prob = max_log_prob.max(log_probs[j]);
            }

            // Log-sum-exp trick for numerical stability
            let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log_prob).exp()).sum();
            let log_total = max_log_prob + sum_exp.ln();
            loglik += log_total;

            // Posterior probabilities
            for j in 0..k {
                posterior[[i, j]] = (log_probs[j] - log_total).exp();
            }
        }

        loglik_history.push(loglik);

        // Check convergence
        if (loglik - prev_loglik).abs() < tolerance {
            converged = true;
            break;
        }
        prev_loglik = loglik;

        // M-step: update parameters
        for j in 0..k {
            let n_j: f64 = (0..n).map(|i| posterior[[i, j]]).sum();

            if n_j > 1e-10 {
                // Update mixing proportion
                lambda[j] = n_j / n as f64;

                // Update mean
                mu[j] = (0..n).map(|i| posterior[[i, j]] * data[i]).sum::<f64>() / n_j;

                // Update variance
                let var_j: f64 = (0..n)
                    .map(|i| posterior[[i, j]] * (data[i] - mu[j]).powi(2))
                    .sum::<f64>()
                    / n_j;
                sigma[j] = var_j.sqrt().max(1e-6);
            }
        }
    }

    // Compute hard assignments
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let mut max_prob = 0.0;
        for j in 0..k {
            if posterior[[i, j]] > max_prob {
                max_prob = posterior[[i, j]];
                labels[i] = j;
            }
        }
    }

    let loglik = loglik_history.last().copied().unwrap_or(f64::NEG_INFINITY);

    Ok(NormalMixEMResult {
        mu,
        sigma,
        lambda,
        posterior,
        labels,
        loglik,
        loglik_history,
        n_iterations,
        converged,
        k,
        n,
    })
}

/// Result of multivariate normal mixture EM fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultivariateNormalMixEMResult {
    /// Component means (k x d)
    #[serde(skip)]
    pub mu: Array2<f64>,
    /// Component covariance matrices (flattened: k x d x d stored as k x (d*d))
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
    /// Number of iterations
    pub n_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Number of components
    pub k: usize,
    /// Number of observations
    pub n: usize,
    /// Number of dimensions
    pub d: usize,
}

impl std::fmt::Display for MultivariateNormalMixEMResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multivariate Normal Mixture EM Results")?;
        writeln!(f, "=======================================")?;
        writeln!(f, "Number of components: {}", self.k)?;
        writeln!(f, "Number of observations: {}", self.n)?;
        writeln!(f, "Dimensions: {}", self.d)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(f, "Converged: {}", self.converged)?;
        writeln!(f, "Log-likelihood: {:.4}", self.loglik)?;
        writeln!(f)?;
        writeln!(f, "Mixing proportions:")?;
        for (i, &l) in self.lambda.iter().enumerate() {
            writeln!(f, "  Component {}: {:.4}", i + 1, l)?;
        }
        Ok(())
    }
}

/// Fit a multivariate normal mixture model using EM algorithm.
///
/// # Arguments
/// * `data` - Multivariate data (n x d)
/// * `k` - Number of mixture components
/// * `max_iterations` - Maximum EM iterations (default: 500)
/// * `tol` - Convergence tolerance (default: 1e-6)
/// * `seed` - Optional random seed
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
    let n = data.nrows();
    let d = data.ncols();

    if k == 0 {
        return Err("k must be at least 1".to_string());
    }
    if k > n {
        return Err(format!("k ({}) cannot exceed n ({})", k, n));
    }

    let max_iter = max_iterations.unwrap_or(500);
    let tolerance = tol.unwrap_or(1e-6);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Initialize with k-means++ style initialization
    let mut mu = Array2::zeros((k, d));
    let mut sigma: Vec<Array2<f64>> = vec![Array2::eye(d); k];
    let mut lambda = vec![1.0 / k as f64; k];

    // Choose first center randomly
    let first_idx = rng.gen_range(0..n);
    mu.row_mut(0).assign(&data.row(first_idx));

    // Choose remaining centers proportional to squared distance
    for c in 1..k {
        let mut distances = vec![f64::INFINITY; n];
        for i in 0..n {
            for j in 0..c {
                let dist: f64 = (0..d).map(|l| (data[[i, l]] - mu[[j, l]]).powi(2)).sum();
                distances[i] = distances[i].min(dist);
            }
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
        mu.row_mut(c).assign(&data.row(chosen));
    }

    // Initialize covariances as scaled identity
    let global_var: f64 = (0..d)
        .map(|j| {
            let col_mean: f64 = data.column(j).sum() / n as f64;
            data.column(j)
                .iter()
                .map(|&x| (x - col_mean).powi(2))
                .sum::<f64>()
                / n as f64
        })
        .sum::<f64>()
        / d as f64;

    for c in 0..k {
        sigma[c] = Array2::eye(d) * global_var.max(1e-6);
    }

    let mut posterior = Array2::zeros((n, k));
    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iterations = 0;
    let mut loglik = f64::NEG_INFINITY;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        // E-step: compute posterior probabilities
        loglik = 0.0;
        for i in 0..n {
            let x: Vec<f64> = data.row(i).to_vec();
            let mut log_probs = vec![f64::NEG_INFINITY; k];

            for j in 0..k {
                // Compute log of multivariate normal density
                let mu_j: Vec<f64> = mu.row(j).to_vec();
                let diff: Vec<f64> = x.iter().zip(mu_j.iter()).map(|(a, b)| a - b).collect();

                // Simple diagonal approximation for stability
                let sigma_diag: Vec<f64> = (0..d).map(|l| sigma[j][[l, l]].max(1e-6)).collect();
                let log_det: f64 = sigma_diag.iter().map(|s| s.ln()).sum();
                let mahal: f64 = diff
                    .iter()
                    .zip(sigma_diag.iter())
                    .map(|(di, si)| di * di / si)
                    .sum();

                log_probs[j] = lambda[j].ln()
                    - 0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + mahal);
            }

            // Log-sum-exp
            let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_log_prob.is_finite() {
                let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log_prob).exp()).sum();
                let log_total = max_log_prob + sum_exp.ln();
                loglik += log_total;

                for j in 0..k {
                    posterior[[i, j]] = (log_probs[j] - log_total).exp();
                }
            } else {
                // Fallback: uniform assignment
                for j in 0..k {
                    posterior[[i, j]] = 1.0 / k as f64;
                }
            }
        }

        // Check convergence
        if (loglik - prev_loglik).abs() < tolerance {
            converged = true;
            break;
        }
        prev_loglik = loglik;

        // M-step
        for j in 0..k {
            let n_j: f64 = (0..n).map(|i| posterior[[i, j]]).sum();

            if n_j > 1e-10 {
                lambda[j] = n_j / n as f64;

                // Update mean
                for l in 0..d {
                    mu[[j, l]] = (0..n)
                        .map(|i| posterior[[i, j]] * data[[i, l]])
                        .sum::<f64>()
                        / n_j;
                }

                // Update covariance (diagonal for stability)
                for l in 0..d {
                    let var_l: f64 = (0..n)
                        .map(|i| posterior[[i, j]] * (data[[i, l]] - mu[[j, l]]).powi(2))
                        .sum::<f64>()
                        / n_j;
                    sigma[j][[l, l]] = var_l.max(1e-6);
                }
            }
        }
    }

    // Hard assignments
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let mut max_prob = 0.0;
        for j in 0..k {
            if posterior[[i, j]] > max_prob {
                max_prob = posterior[[i, j]];
                labels[i] = j;
            }
        }
    }

    Ok(MultivariateNormalMixEMResult {
        mu,
        sigma,
        lambda,
        posterior,
        labels,
        loglik,
        n_iterations,
        converged,
        k,
        n,
        d,
    })
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

        let result = flexmix(y.view(), x.view(), 2, Some(100), Some(1e-4), Some(42)).unwrap();

        assert_eq!(result.cluster.len(), 8);
        assert_eq!(result.k, 2);
        assert_eq!(result.p, 2);
        assert_eq!(result.coefficients.len(), 2);
        assert!(result.prior.iter().sum::<f64>() - 1.0 < 1e-6);
        assert!(result.bic.is_finite());
        assert!(result.aic.is_finite());
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

        let result = normal_mix_em(&data, 2, Some(100), Some(1e-6), Some(42)).unwrap();

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

        let result = normal_mix_em(&data, 2, Some(500), Some(1e-8), Some(42)).unwrap();

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
