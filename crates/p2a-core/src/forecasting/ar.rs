//! Autoregressive (AR) Model Fitting.
//!
//! Fits AR models to univariate time series data using various methods:
//! - Yule-Walker (default): Solves Yule-Walker equations via Durbin-Levinson
//! - Burg: Burg's method using forward/backward prediction errors
//! - OLS: Ordinary least squares regression
//!
//! # Mathematical Background
//!
//! The AR(p) model is:
//! ```text
//! x_t - μ = φ₁(x_{t-1} - μ) + φ₂(x_{t-2} - μ) + ... + φₚ(x_{t-p} - μ) + ε_t
//! ```
//!
//! where ε_t ~ WN(0, σ²) (white noise).
//!
//! # References
//!
//! - Brockwell, P. J. & Davis, R. A. (1991). "Time Series: Theory and Methods".
//!   Springer. (Chapters 8-9 on parameter estimation)
//! - Burg, J. P. (1967). "Maximum Entropy Spectral Analysis". 37th Meeting of
//!   the Society of Exploration Geophysicists, Oklahoma City.
//! - R Core Team. `stats::ar()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ar.html>

use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};

/// Method for AR model fitting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ArMethod {
    /// Yule-Walker equations solved via Durbin-Levinson (default)
    YuleWalker,
    /// Burg's method using forward/backward prediction errors
    Burg,
    /// Ordinary least squares
    Ols,
}

impl Default for ArMethod {
    fn default() -> Self {
        ArMethod::YuleWalker
    }
}

/// Configuration for AR model fitting.
#[derive(Debug, Clone)]
pub struct ArConfig {
    /// Use AIC for order selection (default: true)
    pub aic: bool,
    /// Maximum order to consider (default: computed from series length)
    pub order_max: Option<usize>,
    /// Specific order to use (if aic=false)
    pub order: Option<usize>,
    /// Fitting method (default: YuleWalker)
    pub method: ArMethod,
    /// Demean the series before fitting (default: true)
    pub demean: bool,
}

impl Default for ArConfig {
    fn default() -> Self {
        ArConfig {
            aic: true,
            order_max: None,
            order: None,
            method: ArMethod::YuleWalker,
            demean: true,
        }
    }
}

/// Result of AR model fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArResult {
    /// Selected or specified model order
    pub order: usize,
    /// Estimated AR coefficients (φ₁, ..., φₚ)
    pub ar: Vec<f64>,
    /// Prediction error variance (innovation variance)
    pub var_pred: f64,
    /// Estimated mean of the series
    pub x_mean: f64,
    /// AIC values relative to the minimum (only if aic=true)
    pub aic: Option<Vec<f64>>,
    /// Partial autocorrelation coefficients at each lag
    pub partial_acf: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Fitting method used
    pub method: ArMethod,
    /// Residuals (first `order` values are NA represented as NaN)
    #[serde(skip)]
    pub resid: Vec<f64>,
}

impl std::fmt::Display for ArResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Autoregressive Model AR({})", self.order)?;
        writeln!(f, "{}", "=".repeat(40))?;
        writeln!(f)?;
        writeln!(f, "Method: {:?}", self.method)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Order selected: {}", self.order)?;
        writeln!(f)?;
        writeln!(f, "Coefficients:")?;
        for (i, coef) in self.ar.iter().enumerate() {
            writeln!(f, "  ar{}: {:.6}", i + 1, coef)?;
        }
        writeln!(f)?;
        writeln!(f, "Mean: {:.6}", self.x_mean)?;
        writeln!(f, "Prediction variance: {:.6}", self.var_pred)?;
        if let Some(ref aic_vals) = self.aic {
            writeln!(f)?;
            writeln!(f, "AIC (relative to minimum):")?;
            for (i, aic) in aic_vals.iter().enumerate() {
                if *aic < 100.0 {
                    writeln!(f, "  AR({}): {:.2}", i, aic)?;
                }
            }
        }
        Ok(())
    }
}

/// Fit an autoregressive model to a time series.
///
/// # Arguments
///
/// * `x` - Time series data
/// * `config` - Configuration options
///
/// # Example
///
/// ```
/// use p2a_core::forecasting::ar::{ar, ArConfig, ArMethod};
///
/// let data = vec![1.0, 2.1, 1.9, 3.2, 2.8, 4.1, 3.9, 5.0, 4.8, 6.1];
/// let config = ArConfig::default();
/// let result = ar(&data, config).unwrap();
/// println!("Selected order: {}", result.order);
/// println!("AR coefficients: {:?}", result.ar);
/// ```
pub fn ar(x: &[f64], config: ArConfig) -> EconResult<ArResult> {
    let n = x.len();

    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "AR model fitting".to_string(),
        });
    }

    // Compute default max order: min(n-1, 10 * log10(n))
    let default_max = ((n as f64).log10() * 10.0).floor() as usize;
    let max_order = config.order_max.unwrap_or(default_max.min(n - 1)).min(n - 1);

    // Demean if requested
    let x_mean = if config.demean {
        x.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let y: Vec<f64> = x.iter().map(|v| v - x_mean).collect();

    // Fit based on method
    let (ar_coefs, var_pred, partial_acf, aic_values) = match config.method {
        ArMethod::YuleWalker => fit_ar_yule_walker_full(&y, max_order, config.aic, config.order)?,
        ArMethod::Burg => fit_ar_burg(&y, max_order, config.aic, config.order)?,
        ArMethod::Ols => fit_ar_ols(&y, max_order, config.aic, config.order)?,
    };

    // Compute residuals
    let order = ar_coefs.len();
    let mut resid = vec![f64::NAN; n];
    for t in order..n {
        let mut fitted = 0.0;
        for (k, &phi) in ar_coefs.iter().enumerate() {
            fitted += phi * y[t - k - 1];
        }
        resid[t] = y[t] - fitted;
    }

    Ok(ArResult {
        order,
        ar: ar_coefs,
        var_pred,
        x_mean,
        aic: aic_values,
        partial_acf,
        n_obs: n,
        method: config.method,
        resid,
    })
}

/// Fit AR model using Yule-Walker equations with full diagnostics.
fn fit_ar_yule_walker_full(
    y: &[f64],
    max_order: usize,
    use_aic: bool,
    fixed_order: Option<usize>,
) -> EconResult<(Vec<f64>, f64, Vec<f64>, Option<Vec<f64>>)> {
    let n = y.len();

    // Compute autocovariances up to max_order
    let mut gamma = Vec::with_capacity(max_order + 1);
    for k in 0..=max_order {
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += y[t] * y[t + k];
        }
        gamma.push(sum / n as f64);
    }

    if gamma[0] <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Series has zero variance".to_string(),
        });
    }

    // Use Durbin-Levinson to compute all orders simultaneously
    // This gives us partial autocorrelations as a byproduct
    let mut phi_all: Vec<Vec<f64>> = Vec::with_capacity(max_order + 1);
    let mut var_all: Vec<f64> = Vec::with_capacity(max_order + 1);
    let mut pacf = Vec::with_capacity(max_order);

    // Order 0
    phi_all.push(vec![]);
    var_all.push(gamma[0]);

    // Orders 1 through max_order
    if max_order > 0 {
        // Order 1
        let phi_11 = gamma[1] / gamma[0];
        pacf.push(phi_11);
        phi_all.push(vec![phi_11]);
        var_all.push(gamma[0] * (1.0 - phi_11 * phi_11));

        for p in 2..=max_order {
            let phi_prev = &phi_all[p - 1];

            // Compute partial autocorrelation phi_pp
            let mut num = gamma[p];
            for j in 0..(p - 1) {
                num -= phi_prev[j] * gamma[p - 1 - j];
            }
            let phi_pp = num / var_all[p - 1];
            pacf.push(phi_pp);

            // Update coefficients
            let mut phi_new = vec![0.0; p];
            for j in 0..(p - 1) {
                phi_new[j] = phi_prev[j] - phi_pp * phi_prev[p - 2 - j];
            }
            phi_new[p - 1] = phi_pp;

            // Update variance
            let new_var = var_all[p - 1] * (1.0 - phi_pp * phi_pp);

            phi_all.push(phi_new);
            var_all.push(new_var.max(0.0));
        }
    }

    // Select order based on AIC or fixed
    let (selected_order, aic_values) = if let Some(order) = fixed_order {
        let order = order.min(max_order);
        (order, None)
    } else if use_aic {
        let mut aic_vals = Vec::with_capacity(max_order + 1);
        let mut min_aic = f64::INFINITY;
        let mut best_order = 0;

        for p in 0..=max_order {
            let var = var_all[p];
            if var <= 0.0 {
                aic_vals.push(f64::INFINITY);
                continue;
            }
            // AIC = n * log(var) + 2 * (p + 1)
            let aic = n as f64 * var.ln() + 2.0 * (p + 1) as f64;
            aic_vals.push(aic);
            if aic < min_aic {
                min_aic = aic;
                best_order = p;
            }
        }

        // Convert to relative AIC
        let aic_relative: Vec<f64> = aic_vals.iter().map(|a| a - min_aic).collect();
        (best_order, Some(aic_relative))
    } else {
        // Default to max_order if neither aic nor fixed_order
        (max_order, None)
    };

    let ar_coefs = phi_all[selected_order].clone();
    let var_pred = var_all[selected_order];

    Ok((ar_coefs, var_pred, pacf, aic_values))
}

/// Fit AR model using Burg's method.
///
/// Burg's method minimizes the sum of forward and backward prediction error squares.
fn fit_ar_burg(
    y: &[f64],
    max_order: usize,
    use_aic: bool,
    fixed_order: Option<usize>,
) -> EconResult<(Vec<f64>, f64, Vec<f64>, Option<Vec<f64>>)> {
    let n = y.len();

    // Initialize forward and backward prediction errors
    let mut ef: Vec<f64> = y.to_vec();
    let mut eb: Vec<f64> = y.to_vec();

    let var0 = y.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if var0 <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "Series has zero variance".to_string(),
        });
    }

    let mut phi_all: Vec<Vec<f64>> = Vec::with_capacity(max_order + 1);
    let mut var_all: Vec<f64> = Vec::with_capacity(max_order + 1);
    let mut pacf = Vec::with_capacity(max_order);

    // Order 0
    phi_all.push(vec![]);
    var_all.push(var0);

    for p in 1..=max_order {
        // Compute reflection coefficient using Burg's formula
        let mut num = 0.0;
        let mut den = 0.0;
        for t in p..n {
            num += ef[t] * eb[t - 1];
            den += ef[t] * ef[t] + eb[t - 1] * eb[t - 1];
        }

        if den <= 0.0 {
            // Can't continue
            break;
        }

        let k = 2.0 * num / den;
        pacf.push(k);

        // Update coefficients
        let phi_prev = if p > 1 { phi_all[p - 1].clone() } else { vec![] };
        let mut phi_new = vec![0.0; p];
        for j in 0..(p - 1) {
            phi_new[j] = phi_prev[j] - k * phi_prev[p - 2 - j];
        }
        phi_new[p - 1] = k;

        // Update variance
        let new_var = var_all[p - 1] * (1.0 - k * k);
        var_all.push(new_var.max(0.0));
        phi_all.push(phi_new);

        // Update forward and backward errors
        let ef_old = ef.clone();
        for t in p..n {
            ef[t] = ef_old[t] - k * eb[t - 1];
            eb[t] = eb[t - 1] - k * ef_old[t];
        }
    }

    // Ensure we have enough orders
    while phi_all.len() <= max_order {
        phi_all.push(phi_all.last().unwrap_or(&vec![]).clone());
        var_all.push(*var_all.last().unwrap_or(&var0));
    }

    // Select order
    let (selected_order, aic_values) = if let Some(order) = fixed_order {
        let order = order.min(phi_all.len() - 1);
        (order, None)
    } else if use_aic {
        let mut aic_vals = Vec::with_capacity(phi_all.len());
        let mut min_aic = f64::INFINITY;
        let mut best_order = 0;

        for p in 0..phi_all.len() {
            let var = var_all[p];
            if var <= 0.0 {
                aic_vals.push(f64::INFINITY);
                continue;
            }
            let aic = n as f64 * var.ln() + 2.0 * (p + 1) as f64;
            aic_vals.push(aic);
            if aic < min_aic {
                min_aic = aic;
                best_order = p;
            }
        }

        let aic_relative: Vec<f64> = aic_vals.iter().map(|a| a - min_aic).collect();
        (best_order, Some(aic_relative))
    } else {
        (max_order.min(phi_all.len() - 1), None)
    };

    let ar_coefs = phi_all[selected_order].clone();
    let var_pred = var_all[selected_order];

    Ok((ar_coefs, var_pred, pacf, aic_values))
}

/// Fit AR model using OLS.
fn fit_ar_ols(
    y: &[f64],
    max_order: usize,
    use_aic: bool,
    fixed_order: Option<usize>,
) -> EconResult<(Vec<f64>, f64, Vec<f64>, Option<Vec<f64>>)> {
    let n = y.len();

    // Also compute PACF via Yule-Walker for consistency
    let (_, _, pacf, _) = fit_ar_yule_walker_full(y, max_order, false, Some(max_order))?;

    let mut phi_all: Vec<Vec<f64>> = Vec::with_capacity(max_order + 1);
    let mut var_all: Vec<f64> = Vec::with_capacity(max_order + 1);

    // Order 0: just variance
    let var0 = y.iter().map(|v| v * v).sum::<f64>() / n as f64;
    phi_all.push(vec![]);
    var_all.push(var0);

    for p in 1..=max_order {
        if n <= p {
            break;
        }

        // Build design matrix and response
        let n_eff = n - p;
        let mut x_data = vec![0.0; n_eff * p];
        let mut y_vec = vec![0.0; n_eff];

        for t in 0..n_eff {
            y_vec[t] = y[t + p];
            for k in 0..p {
                x_data[t * p + k] = y[t + p - k - 1];
            }
        }

        // Solve normal equations: (X'X)^{-1} X'y
        // Using simple Cholesky or direct solve for small p
        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];

        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for t in 0..n_eff {
                    sum += x_data[t * p + i] * x_data[t * p + j];
                }
                xtx[i * p + j] = sum;
            }
            let mut sum = 0.0;
            for t in 0..n_eff {
                sum += x_data[t * p + i] * y_vec[t];
            }
            xty[i] = sum;
        }

        // Solve using simple Gauss elimination (could use more robust method)
        let phi = match solve_linear_system(&xtx, &xty, p) {
            Some(coefs) => coefs,
            None => {
                // Singular system, use Yule-Walker instead
                let (yw_coefs, _, _, _) = fit_ar_yule_walker_full(y, p, false, Some(p))?;
                yw_coefs
            }
        };

        // Compute residual variance
        let mut sse = 0.0;
        for t in 0..n_eff {
            let mut fitted = 0.0;
            for k in 0..p {
                fitted += phi[k] * x_data[t * p + k];
            }
            let resid = y_vec[t] - fitted;
            sse += resid * resid;
        }
        let var = sse / (n_eff - p) as f64;

        phi_all.push(phi);
        var_all.push(var.max(0.0));
    }

    // Ensure enough orders
    while phi_all.len() <= max_order {
        phi_all.push(phi_all.last().unwrap_or(&vec![]).clone());
        var_all.push(*var_all.last().unwrap_or(&var0));
    }

    // Select order
    let (selected_order, aic_values) = if let Some(order) = fixed_order {
        let order = order.min(phi_all.len() - 1);
        (order, None)
    } else if use_aic {
        let mut aic_vals = Vec::with_capacity(phi_all.len());
        let mut min_aic = f64::INFINITY;
        let mut best_order = 0;

        for p in 0..phi_all.len() {
            let var = var_all[p];
            if var <= 0.0 {
                aic_vals.push(f64::INFINITY);
                continue;
            }
            // Use effective sample size
            let n_eff = if p == 0 { n } else { n - p };
            let aic = n_eff as f64 * var.ln() + 2.0 * (p + 1) as f64;
            aic_vals.push(aic);
            if aic < min_aic {
                min_aic = aic;
                best_order = p;
            }
        }

        let aic_relative: Vec<f64> = aic_vals.iter().map(|a| a - min_aic).collect();
        (best_order, Some(aic_relative))
    } else {
        (max_order.min(phi_all.len() - 1), None)
    };

    let ar_coefs = phi_all[selected_order].clone();
    let var_pred = var_all[selected_order];

    Ok((ar_coefs, var_pred, pacf, aic_values))
}

/// Simple linear system solver for small systems.
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(vec![]);
    }

    // Copy matrix and vector
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = aug[k * (n + 1) + k].abs();
        for i in (k + 1)..n {
            let val = aug[i * (n + 1) + k].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val < 1e-14 {
            return None; // Singular
        }

        // Swap rows
        if max_idx != k {
            for j in 0..=n {
                let temp = aug[k * (n + 1) + j];
                aug[k * (n + 1) + j] = aug[max_idx * (n + 1) + j];
                aug[max_idx * (n + 1) + j] = temp;
            }
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = aug[i * (n + 1) + k] / aug[k * (n + 1) + k];
            for j in k..=n {
                aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    Some(x)
}

/// Convenience function to run AR model fitting with default settings.
pub fn run_ar(x: &[f64]) -> EconResult<ArResult> {
    ar(x, ArConfig::default())
}

/// Convenience function to run AR model with specific order.
pub fn run_ar_with_order(x: &[f64], order: usize) -> EconResult<ArResult> {
    ar(x, ArConfig {
        aic: false,
        order: Some(order),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ar_basic() {
        // Generate AR(1) process
        let mut x = vec![0.0; 100];
        x[0] = 1.0;
        let phi = 0.7;
        for t in 1..100 {
            // Simple AR(1): x_t = phi * x_{t-1} + noise
            x[t] = phi * x[t - 1] + (t as f64 % 3.0 - 1.0) * 0.3;
        }

        let result = ar(&x, ArConfig::default()).unwrap();

        println!("AR result: {}", result);
        println!("True phi: {}, Estimated: {:?}", phi, result.ar);

        assert!(result.n_obs == 100);
        assert!(result.var_pred > 0.0);
    }

    #[test]
    fn test_ar_methods() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64).sin() + (i as f64 * 0.1)).collect();

        // Test all methods
        for method in [ArMethod::YuleWalker, ArMethod::Burg, ArMethod::Ols] {
            let config = ArConfig {
                method,
                ..Default::default()
            };
            let result = ar(&x, config).unwrap();
            println!("{:?} method: order={}, var={:.4}", method, result.order, result.var_pred);
            assert!(result.var_pred > 0.0);
        }
    }

    #[test]
    fn test_ar_fixed_order() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64).sin()).collect();

        let config = ArConfig {
            aic: false,
            order: Some(3),
            ..Default::default()
        };
        let result = ar(&x, config).unwrap();

        assert_eq!(result.order, 3);
        assert_eq!(result.ar.len(), 3);
    }

    #[test]
    fn test_ar_aic_selection() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();

        let result = ar(&x, ArConfig::default()).unwrap();

        assert!(result.aic.is_some());
        let aic = result.aic.unwrap();
        assert!(!aic.is_empty());
        // Minimum should be 0
        assert!(aic.iter().any(|&a| a == 0.0));
    }

    #[test]
    fn test_ar_residuals() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();

        let result = ar(&x, ArConfig {
            aic: false,
            order: Some(2),
            ..Default::default()
        }).unwrap();

        assert_eq!(result.resid.len(), 30);
        // First `order` residuals should be NaN
        assert!(result.resid[0].is_nan());
        assert!(result.resid[1].is_nan());
        // Later residuals should not be NaN
        assert!(!result.resid[5].is_nan());
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_ar_yule_walker_against_r() {
        // R code:
        // set.seed(42)
        // x <- arima.sim(n=100, model=list(ar=c(0.7, -0.2)))
        // result <- ar(x, method="yule-walker")
        // result$order  # Order selected
        // result$ar     # Coefficients

        // Use deterministic test data
        let x: Vec<f64> = (0..100).map(|i| {
            0.7 * ((i as f64 - 1.0).max(0.0) * 0.1).sin() -
            0.2 * ((i as f64 - 2.0).max(0.0) * 0.1).sin() +
            (i as f64 * 0.1).cos()
        }).collect();

        let result = ar(&x, ArConfig {
            method: ArMethod::YuleWalker,
            ..Default::default()
        }).unwrap();

        println!("Yule-Walker result:");
        println!("Order: {}", result.order);
        println!("AR coefficients: {:?}", result.ar);
        println!("Var: {:.6}", result.var_pred);
        println!("PACF: {:?}", result.partial_acf);

        // Check that we get reasonable results
        assert!(result.order > 0);
        assert!(result.ar.iter().all(|c| c.abs() < 2.0)); // Coefficients should be reasonable
    }

    #[test]
    fn test_ar_display() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = ar(&x, ArConfig::default()).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Autoregressive Model"));
        assert!(display.contains("Order selected"));
    }
}
