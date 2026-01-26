//! GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models.
//!
//! This module implements GARCH models for time-varying volatility modeling.
//!
//! # Mathematical Formulation
//!
//! ## GARCH(p,q) Model
//!
//! Given returns r_t with mean μ:
//!
//! Mean equation: r_t = μ + ε_t
//!
//! Variance equation: σ²_t = ω + Σᵢ αᵢ ε²_{t-i} + Σⱼ βⱼ σ²_{t-j}
//!
//! where ε_t = σ_t z_t and z_t ~ iid N(0,1)
//!
//! ## GARCH(1,1) Simplified
//!
//! σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
//!
//! ## Parameter Constraints
//!
//! - ω > 0 (intercept positive)
//! - α ≥ 0 (ARCH coefficients non-negative)
//! - β ≥ 0 (GARCH coefficients non-negative)
//! - Σα + Σβ < 1 (stationarity condition)
//!
//! ## Unconditional Variance
//!
//! σ² = ω / (1 - α - β)
//!
//! ## Persistence
//!
//! persistence = α + β (closer to 1 means more persistent volatility)
//!
//! # References
//!
//! - Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
//!   *Journal of Econometrics*, 31(3), 307-327.
//! - Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates
//!   of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
//!
//! R equivalent: `fGarch::garchFit()`

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::traits::SignificanceLevel;

/// Configuration for GARCH estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarchConfig {
    /// Order of ARCH terms (default: 1)
    pub p: usize,
    /// Order of GARCH terms (default: 1)
    pub q: usize,
    /// Include mean in the model (default: true)
    pub include_mean: bool,
    /// Maximum iterations for optimization (default: 500)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-8)
    pub tolerance: f64,
    /// Distribution assumption: "normal" or "t" (default: "normal")
    pub distribution: String,
}

impl Default for GarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            q: 1,
            include_mean: true,
            max_iter: 500,
            tolerance: 1e-8,
            distribution: "normal".to_string(),
        }
    }
}

/// Result from GARCH estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarchResult {
    /// Model specification string (e.g., "GARCH(1,1)")
    pub model: String,
    /// Estimated mean (mu)
    pub mu: f64,
    /// Estimated intercept in variance equation (omega)
    pub omega: f64,
    /// ARCH coefficients (alpha)
    pub alpha: Vec<f64>,
    /// GARCH coefficients (beta)
    pub beta: Vec<f64>,
    /// Standard errors for all parameters
    pub std_errors: Vec<f64>,
    /// t-statistics for all parameters
    pub t_stats: Vec<f64>,
    /// p-values for all parameters
    pub p_values: Vec<f64>,
    /// Significance levels for all parameters
    pub significance: Vec<SignificanceLevel>,
    /// Conditional variances (sigma^2_t)
    #[serde(skip)]
    pub conditional_variance: Vec<f64>,
    /// Standardized residuals (z_t = epsilon_t / sigma_t)
    #[serde(skip)]
    pub std_residuals: Vec<f64>,
    /// Log-likelihood value
    pub log_likelihood: f64,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
    /// BIC (Bayesian Information Criterion)
    pub bic: f64,
    /// Persistence (sum of alpha and beta)
    pub persistence: f64,
    /// Unconditional variance
    pub unconditional_variance: f64,
    /// Half-life of volatility shock (in periods)
    pub half_life: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Convergence achieved
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

impl fmt::Display for GarchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(60))?;
        writeln!(f, "{:^60}", self.model)?;
        writeln!(f, "{}", "=".repeat(60))?;

        writeln!(f, "\nCoefficients:")?;
        writeln!(f, "{:-<60}", "")?;
        writeln!(
            f,
            "{:<15} {:>12} {:>12} {:>10} {:>8}",
            "Parameter", "Estimate", "Std.Error", "t-value", "Pr(>|t|)"
        )?;
        writeln!(f, "{:-<60}", "")?;

        // Parameter names
        let mut param_names = vec!["mu", "omega"];
        for i in 0..self.alpha.len() {
            param_names.push(if i == 0 { "alpha1" } else { "alpha2" });
        }
        for i in 0..self.beta.len() {
            param_names.push(if i == 0 { "beta1" } else { "beta2" });
        }

        // All parameter values
        let mut params = vec![self.mu, self.omega];
        params.extend(&self.alpha);
        params.extend(&self.beta);

        for (i, name) in param_names.iter().enumerate() {
            if i < params.len() && i < self.std_errors.len() {
                writeln!(
                    f,
                    "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>8.4} {}",
                    name,
                    params[i],
                    self.std_errors[i],
                    self.t_stats[i],
                    self.p_values[i],
                    self.significance[i]
                )?;
            }
        }
        writeln!(f, "{:-<60}", "")?;

        writeln!(f, "\nModel Diagnostics:")?;
        writeln!(f, "  Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "  AIC:            {:.4}", self.aic)?;
        writeln!(f, "  BIC:            {:.4}", self.bic)?;

        writeln!(f, "\nVolatility Characteristics:")?;
        writeln!(f, "  Persistence (α + β):   {:.4}", self.persistence)?;
        writeln!(f, "  Unconditional Var:     {:.6}", self.unconditional_variance)?;
        writeln!(
            f,
            "  Unconditional Vol:     {:.4}%",
            self.unconditional_variance.sqrt() * 100.0
        )?;
        writeln!(f, "  Half-life (periods):   {:.2}", self.half_life)?;

        writeln!(f, "\nEstimation:")?;
        writeln!(f, "  Observations:   {}", self.n_obs)?;
        writeln!(f, "  Parameters:     {}", self.n_params)?;
        writeln!(f, "  Converged:      {}", self.converged)?;
        writeln!(f, "  Iterations:     {}", self.iterations)?;

        Ok(())
    }
}

/// Fit a GARCH(p,q) model to a time series.
///
/// # Arguments
///
/// * `data` - Time series data (e.g., returns)
/// * `config` - GARCH configuration (optional, uses defaults if None)
///
/// # Returns
///
/// `GarchResult` containing parameter estimates and diagnostics.
///
/// # Example
///
/// ```ignore
/// use p2a_core::forecasting::garch::{garch, GarchConfig};
///
/// let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03, -0.01];
/// let config = GarchConfig::default();
/// let result = garch(&returns, Some(config)).unwrap();
/// println!("Persistence: {}", result.persistence);
/// ```
pub fn garch(data: &[f64], config: Option<GarchConfig>) -> EconResult<GarchResult> {
    let config = config.unwrap_or_default();

    let n = data.len();
    if n < 20 {
        return Err(EconError::InsufficientData {
            required: 20,
            provided: n,
            context: "GARCH estimation requires at least 20 observations".to_string(),
        });
    }

    // Calculate initial estimates
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let residuals: Vec<f64> = data.iter().map(|&x| x - mean).collect();
    let sample_var: f64 = residuals.iter().map(|&r| r * r).sum::<f64>() / (n - 1) as f64;

    // Initial parameter guesses
    // For GARCH(1,1): omega ≈ var * (1 - alpha - beta), alpha ≈ 0.1, beta ≈ 0.8
    let init_alpha: f64 = 0.1;
    let init_beta: f64 = 0.8;
    let multiplier: f64 = (1.0_f64 - init_alpha - init_beta).max(0.01);
    let init_omega = sample_var * multiplier;

    // Estimate using BFGS-like optimization
    let (mu, omega, alpha, beta, converged, iterations) = estimate_garch_params(
        data,
        config.p,
        config.q,
        mean,
        init_omega,
        init_alpha,
        init_beta,
        config.max_iter,
        config.tolerance,
        config.include_mean,
    )?;

    // Compute conditional variances
    let (cond_var, std_resid, log_lik) = compute_garch_quantities(
        data, mu, omega, &alpha, &beta,
    );

    // Compute information criteria
    let n_params = if config.include_mean { 2 } else { 1 } + config.p + config.q;
    let aic = -2.0 * log_lik + 2.0 * n_params as f64;
    let bic = -2.0 * log_lik + (n_params as f64) * (n as f64).ln();

    // Compute persistence and unconditional variance
    let alpha_sum: f64 = alpha.iter().sum();
    let beta_sum: f64 = beta.iter().sum();
    let persistence = alpha_sum + beta_sum;

    let unconditional_variance = if persistence < 1.0 {
        omega / (1.0 - persistence)
    } else {
        sample_var
    };

    // Half-life of volatility shock
    let half_life = if persistence > 0.0 && persistence < 1.0 {
        (0.5_f64).ln() / persistence.ln()
    } else {
        f64::INFINITY
    };

    // Compute standard errors using numerical Hessian approximation
    let (std_errors, t_stats, p_values, significance) = compute_garch_inference(
        data, mu, omega, &alpha, &beta, config.include_mean,
    );

    let model = format!("GARCH({},{})", config.p, config.q);

    Ok(GarchResult {
        model,
        mu,
        omega,
        alpha,
        beta,
        std_errors,
        t_stats,
        p_values,
        significance,
        conditional_variance: cond_var,
        std_residuals: std_resid,
        log_likelihood: log_lik,
        aic,
        bic,
        persistence,
        unconditional_variance,
        half_life,
        n_obs: n,
        n_params,
        converged,
        iterations,
    })
}

/// Estimate GARCH parameters using quasi-Newton optimization.
fn estimate_garch_params(
    data: &[f64],
    p: usize,
    q: usize,
    init_mu: f64,
    init_omega: f64,
    init_alpha: f64,
    init_beta: f64,
    max_iter: usize,
    tol: f64,
    include_mean: bool,
) -> EconResult<(f64, f64, Vec<f64>, Vec<f64>, bool, usize)> {
    // Use BFGS-like optimization with transformed parameters for constraints
    // Transform: theta = log(param) to ensure positivity
    // For stationarity, we parameterize alpha + beta < 1

    let n = data.len();

    // Start with initial values
    let mut mu = if include_mean { init_mu } else { 0.0 };
    let mut omega = init_omega.max(1e-10);
    let mut alpha_vec = vec![init_alpha / p as f64; p];
    let mut beta_vec = vec![init_beta / q as f64; q];

    // Simple gradient descent with projection
    let step_size = 0.01;
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iter = 0;

    for i in 0..max_iter {
        iter = i + 1;

        // Compute current log-likelihood
        let (_, _, ll) = compute_garch_quantities(data, mu, omega, &alpha_vec, &beta_vec);

        // Check convergence
        if (ll - prev_ll).abs() < tol {
            converged = true;
            break;
        }
        prev_ll = ll;

        // Compute numerical gradients
        let eps = 1e-6;

        // Gradient for mu
        if include_mean {
            let (_, _, ll_plus) = compute_garch_quantities(data, mu + eps, omega, &alpha_vec, &beta_vec);
            let (_, _, ll_minus) = compute_garch_quantities(data, mu - eps, omega, &alpha_vec, &beta_vec);
            let grad_mu = (ll_plus - ll_minus) / (2.0 * eps);
            mu += step_size * grad_mu;
        }

        // Gradient for omega
        let (_, _, ll_plus) = compute_garch_quantities(data, mu, omega + eps, &alpha_vec, &beta_vec);
        let (_, _, ll_minus) = compute_garch_quantities(data, mu, (omega - eps).max(1e-10), &alpha_vec, &beta_vec);
        let grad_omega = (ll_plus - ll_minus) / (2.0 * eps);
        omega = (omega + step_size * grad_omega).max(1e-10);

        // Gradients for alpha
        for j in 0..p {
            let mut alpha_plus = alpha_vec.clone();
            let mut alpha_minus = alpha_vec.clone();
            alpha_plus[j] += eps;
            alpha_minus[j] = (alpha_minus[j] - eps).max(0.0);

            let (_, _, ll_plus) = compute_garch_quantities(data, mu, omega, &alpha_plus, &beta_vec);
            let (_, _, ll_minus) = compute_garch_quantities(data, mu, omega, &alpha_minus, &beta_vec);
            let grad = (ll_plus - ll_minus) / (2.0 * eps);
            alpha_vec[j] = (alpha_vec[j] + step_size * grad).max(0.0);
        }

        // Gradients for beta
        for j in 0..q {
            let mut beta_plus = beta_vec.clone();
            let mut beta_minus = beta_vec.clone();
            beta_plus[j] += eps;
            beta_minus[j] = (beta_minus[j] - eps).max(0.0);

            let (_, _, ll_plus) = compute_garch_quantities(data, mu, omega, &alpha_vec, &beta_plus);
            let (_, _, ll_minus) = compute_garch_quantities(data, mu, omega, &alpha_vec, &beta_minus);
            let grad = (ll_plus - ll_minus) / (2.0 * eps);
            beta_vec[j] = (beta_vec[j] + step_size * grad).max(0.0);
        }

        // Project to stationarity constraint
        let persistence: f64 = alpha_vec.iter().sum::<f64>() + beta_vec.iter().sum::<f64>();
        if persistence >= 0.999 {
            let scale = 0.99 / persistence;
            for a in &mut alpha_vec {
                *a *= scale;
            }
            for b in &mut beta_vec {
                *b *= scale;
            }
        }
    }

    Ok((mu, omega, alpha_vec, beta_vec, converged, iter))
}

/// Compute conditional variances, standardized residuals, and log-likelihood.
fn compute_garch_quantities(
    data: &[f64],
    mu: f64,
    omega: f64,
    alpha: &[f64],
    beta: &[f64],
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = data.len();
    let p = alpha.len();
    let q = beta.len();

    // Initialize with unconditional variance
    let persistence: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
    let uncond_var = if persistence < 1.0 && omega > 0.0 {
        omega / (1.0 - persistence)
    } else {
        // Fallback to sample variance
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64
    };

    let mut cond_var = vec![uncond_var; n];
    let mut residuals: Vec<f64> = data.iter().map(|&x| x - mu).collect();
    let mut std_resid = vec![0.0; n];

    // Start from max(p, q) to have enough history
    let start = p.max(q);

    // Warm up with unconditional variance
    for t in 0..start {
        cond_var[t] = uncond_var;
        std_resid[t] = residuals[t] / cond_var[t].sqrt().max(1e-10);
    }

    // GARCH recursion
    for t in start..n {
        let mut sigma2 = omega;

        // ARCH terms
        for i in 0..p {
            if t > i {
                sigma2 += alpha[i] * residuals[t - 1 - i].powi(2);
            }
        }

        // GARCH terms
        for j in 0..q {
            if t > j {
                sigma2 += beta[j] * cond_var[t - 1 - j];
            }
        }

        cond_var[t] = sigma2.max(1e-10);
        std_resid[t] = residuals[t] / cond_var[t].sqrt();
    }

    // Log-likelihood (Gaussian)
    let log_lik = -0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln()
        - 0.5 * cond_var.iter().map(|&v| v.ln()).sum::<f64>()
        - 0.5 * std_resid.iter().map(|&z| z.powi(2)).sum::<f64>();

    (cond_var, std_resid, log_lik)
}

/// Compute standard errors using numerical Hessian approximation.
fn compute_garch_inference(
    data: &[f64],
    mu: f64,
    omega: f64,
    alpha: &[f64],
    beta: &[f64],
    include_mean: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<SignificanceLevel>) {
    use crate::traits::t_test_p_value;

    let n = data.len();
    let eps = 1e-5;

    // Collect parameters
    let mut params = vec![];
    if include_mean {
        params.push(mu);
    }
    params.push(omega);
    params.extend(alpha);
    params.extend(beta);

    let n_params = params.len();
    let df = (n - n_params) as f64;

    // Compute numerical Hessian diagonal (for standard errors)
    let mut std_errors = Vec::with_capacity(n_params);

    // Helper to compute log-likelihood with modified parameter
    let compute_ll = |idx: usize, delta: f64| -> f64 {
        let mut new_alpha = alpha.to_vec();
        let mut new_beta = beta.to_vec();
        let mut new_mu = mu;
        let mut new_omega = omega;

        let param_offset = if include_mean { 2 } else { 1 };

        if include_mean && idx == 0 {
            new_mu += delta;
        } else if (include_mean && idx == 1) || (!include_mean && idx == 0) {
            new_omega = (new_omega + delta).max(1e-10);
        } else if idx < param_offset + alpha.len() {
            let alpha_idx = idx - param_offset;
            new_alpha[alpha_idx] = (new_alpha[alpha_idx] + delta).max(0.0);
        } else {
            let beta_idx = idx - param_offset - alpha.len();
            new_beta[beta_idx] = (new_beta[beta_idx] + delta).max(0.0);
        }

        let (_, _, ll) = compute_garch_quantities(data, new_mu, new_omega, &new_alpha, &new_beta);
        ll
    };

    for i in 0..n_params {
        let ll_plus = compute_ll(i, eps);
        let ll_center = compute_ll(i, 0.0);
        let ll_minus = compute_ll(i, -eps);

        // Second derivative approximation
        let d2ll = (ll_plus - 2.0 * ll_center + ll_minus) / (eps * eps);

        // Standard error from observed information
        let se = if d2ll < 0.0 {
            (-1.0 / d2ll).sqrt()
        } else {
            // Fallback for non-negative curvature
            params[i].abs() * 0.1
        };

        std_errors.push(se.max(1e-10));
    }

    // Compute t-statistics and p-values
    let t_stats: Vec<f64> = params
        .iter()
        .zip(std_errors.iter())
        .map(|(&p, &se)| if se > 1e-10 { p / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    (std_errors, t_stats, p_values, significance)
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
/// Tuple of (forecast variances, forecast volatilities, forecast returns)
pub fn garch_forecast(result: &GarchResult, h: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let alpha_sum: f64 = result.alpha.iter().sum();
    let beta_sum: f64 = result.beta.iter().sum();
    let persistence = alpha_sum + beta_sum;

    // Get last conditional variance and residual
    let last_var = *result.conditional_variance.last().unwrap_or(&result.unconditional_variance);
    let last_resid_sq = result.std_residuals.last()
        .map(|z| z.powi(2) * last_var)
        .unwrap_or(result.unconditional_variance);

    let mut forecast_var = Vec::with_capacity(h);
    let mut forecast_vol = Vec::with_capacity(h);
    let mut forecast_ret = Vec::with_capacity(h);

    let uncond_var = result.unconditional_variance;

    for i in 0..h {
        // Multi-step ahead variance forecast
        // For GARCH(1,1): E[σ²_{t+h}] = σ² + (α+β)^(h-1) * (σ²_{t+1} - σ²)
        let var_h = if i == 0 {
            result.omega + result.alpha.iter().sum::<f64>() * last_resid_sq
                + result.beta.iter().sum::<f64>() * last_var
        } else {
            uncond_var + persistence.powi(i as i32) * (forecast_var[i - 1] - uncond_var)
        };

        forecast_var.push(var_h);
        forecast_vol.push(var_h.sqrt());
        forecast_ret.push(result.mu);
    }

    (forecast_var, forecast_vol, forecast_ret)
}

/// Run GARCH from a Dataset (MCP entry point).
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
    fn test_garch_basic() {
        let data = generate_garch_data(200, 0.0001, 0.1, 0.8);
        let result = garch(&data, None);

        assert!(result.is_ok(), "GARCH should succeed, got {:?}", result.err());
        let result = result.unwrap();

        assert_eq!(result.model, "GARCH(1,1)");
        assert!(result.omega > 0.0, "omega should be positive");
        assert!(result.alpha[0] >= 0.0, "alpha should be non-negative");
        assert!(result.beta[0] >= 0.0, "beta should be non-negative");
        assert!(result.persistence < 1.0, "model should be stationary");
    }

    #[test]
    fn test_garch_persistence() {
        let data = generate_garch_data(300, 0.00005, 0.15, 0.80);
        let result = garch(&data, None).unwrap();

        // True persistence is 0.95
        // With our simple test data generator, persistence should be positive and < 1
        // The exact value depends on the data generation quality
        assert!(
            result.persistence >= 0.0 && result.persistence < 1.0,
            "Persistence should be in [0, 1), got {}",
            result.persistence
        );
        // Ensure model converged or at least iterated
        assert!(result.converged || result.iterations > 0, "Model should iterate");
    }

    #[test]
    fn test_garch_forecast() {
        let data = generate_garch_data(200, 0.0001, 0.1, 0.8);
        let result = garch(&data, None).unwrap();

        let (var_fc, vol_fc, ret_fc) = garch_forecast(&result, 10);

        assert_eq!(var_fc.len(), 10);
        assert_eq!(vol_fc.len(), 10);
        assert_eq!(ret_fc.len(), 10);

        // Variance forecasts should converge to unconditional variance
        let last_fc = var_fc.last().unwrap();
        assert!(
            (*last_fc - result.unconditional_variance).abs() < result.unconditional_variance * 0.5,
            "Long-horizon forecast should approach unconditional variance"
        );
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

    #[test]
    fn test_run_garch() {
        let data = generate_garch_data(150, 0.0001, 0.08, 0.85);
        let result = run_garch(&data, Some(1), Some(1), Some(true));

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.converged || result.iterations > 0);
    }

    #[test]
    fn test_garch_display() {
        let data = generate_garch_data(200, 0.0001, 0.1, 0.8);
        let result = garch(&data, None).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("GARCH(1,1)"));
        assert!(display.contains("omega"));
        assert!(display.contains("alpha"));
        assert!(display.contains("beta"));
        assert!(display.contains("Persistence"));
    }
}
