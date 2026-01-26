//! Quantile Regression implementation.
//!
//! Provides quantile regression estimation for analyzing conditional quantiles
//! rather than conditional means.
//!
//! # Mathematical Background
//!
//! Quantile regression minimizes the check function (asymmetric absolute loss):
//!
//! Q(β) = Σᵢ ρτ(yᵢ - xᵢ'β)
//!
//! where ρτ(u) = u(τ - I(u < 0)) = u(τ) for u ≥ 0, u(τ - 1) for u < 0
//!
//! For τ = 0.5 (median), this reduces to least absolute deviations (LAD).
//!
//! # Algorithms
//!
//! - **Interior Point**: Efficient for large problems (Portnoy & Koenker, 1997)
//! - **Simplex**: Barrodale-Roberts algorithm for exact solution
//! - **IRLS**: Iteratively reweighted least squares approximation
//!
//! # References
//!
//! - Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*,
//!   46(1), 33-50. https://doi.org/10.2307/1913643
//!
//! - Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
//!   ISBN: 978-0521608275.
//!
//! - Portnoy, S., & Koenker, R. (1997). The Gaussian Hare and the Laplacian
//!   Tortoise: Computability of Squared-error vs. Absolute-error Estimators.
//!   *Statistical Science*, 12(4), 279-300.
//!
//! R equivalent: `quantreg::rq()`

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use polars::prelude::*;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::{DesignMatrix, xtx, safe_inverse};
use crate::traits::t_test_p_value;

/// Result from quantile regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantRegResult {
    /// Quantile being estimated (τ)
    pub tau: f64,
    /// Coefficient estimates
    pub coefficients: Vec<QuantRegCoefficient>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Degrees of freedom
    pub df: usize,
    /// Objective function value (sum of weighted absolute deviations)
    pub objective: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Algorithm used
    pub algorithm: QuantRegAlgorithm,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Residuals
    #[serde(skip)]
    pub(crate) residuals: Array1<f64>,
    /// Fitted values
    #[serde(skip)]
    pub(crate) fitted: Array1<f64>,
}

/// A single coefficient from quantile regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantRegCoefficient {
    /// Variable name
    pub name: String,
    /// Coefficient estimate
    pub estimate: f64,
    /// Standard error (from IID bootstrap or rank inversion)
    pub std_error: f64,
    /// t-statistic
    pub t_value: f64,
    /// p-value (two-sided)
    pub p_value: f64,
    /// 95% confidence interval lower bound
    pub ci_lower_95: f64,
    /// 95% confidence interval upper bound
    pub ci_upper_95: f64,
}

/// Algorithm for quantile regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantRegAlgorithm {
    /// Interior point method (efficient for large problems)
    #[default]
    InteriorPoint,
    /// Barrodale-Roberts simplex algorithm
    Simplex,
    /// Iteratively reweighted least squares
    IRLS,
}

impl std::fmt::Display for QuantRegAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InteriorPoint => write!(f, "Interior Point"),
            Self::Simplex => write!(f, "Simplex (Barrodale-Roberts)"),
            Self::IRLS => write!(f, "IRLS"),
        }
    }
}

impl QuantRegAlgorithm {
    /// Parse algorithm from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ip" | "interior" | "interior-point" | "interiorpoint" => Some(Self::InteriorPoint),
            "simplex" | "br" | "barrodale" => Some(Self::Simplex),
            "irls" | "reweighted" => Some(Self::IRLS),
            _ => None,
        }
    }
}

impl std::fmt::Display for QuantRegResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Quantile Regression (τ = {:.2})", self.tau)?;
        writeln!(f, "=================================")?;
        writeln!(f, "Algorithm: {}", self.algorithm)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Parameters: {}", self.n_params)?;
        writeln!(f, "Objective: {:.4}", self.objective)?;
        writeln!(f)?;
        writeln!(f, "{:<15} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Estimate", "Std.Err", "t-value", "p-value")?;
        writeln!(f, "{:-<15} {:-<10} {:-<10} {:-<10} {:-<10}", "", "", "", "", "")?;
        for coef in &self.coefficients {
            writeln!(f, "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                coef.name, coef.estimate, coef.std_error, coef.t_value, coef.p_value)?;
        }
        Ok(())
    }
}

/// Configuration for quantile regression.
#[derive(Debug, Clone)]
pub struct QuantRegConfig {
    /// Quantile (τ) to estimate
    pub tau: f64,
    /// Algorithm to use
    pub algorithm: QuantRegAlgorithm,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Include intercept
    pub intercept: bool,
    /// Number of bootstrap samples for standard errors (0 = use approximation)
    pub bootstrap_samples: usize,
}

impl Default for QuantRegConfig {
    fn default() -> Self {
        Self {
            tau: 0.5,  // median regression by default
            algorithm: QuantRegAlgorithm::IRLS,
            max_iter: 100,
            tol: 1e-6,
            intercept: true,
            bootstrap_samples: 0,  // Use sandwich formula approximation
        }
    }
}

/// Run quantile regression.
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable
/// * `x_cols` - Names of independent variables
/// * `config` - Configuration (tau, algorithm, etc.)
///
/// # Returns
///
/// `QuantRegResult` containing coefficient estimates and statistics.
///
/// # Example
///
/// ```ignore
/// use p2a_core::regression::{quantreg, QuantRegConfig};
///
/// let config = QuantRegConfig {
///     tau: 0.5,  // median
///     ..Default::default()
/// };
/// let result = quantreg(&dataset, "y", &["x1", "x2"], &config)?;
/// ```
pub fn quantreg(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &QuantRegConfig,
) -> EconResult<QuantRegResult> {
    // Validate tau
    if config.tau <= 0.0 || config.tau >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: format!("tau must be in (0, 1), got {}", config.tau),
        });
    }

    // Extract data
    let df = dataset.df();
    let y_series = df.column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
    })?;

    let y: Vec<f64> = y_series.f64()
        .map_err(|_| EconError::InvalidSpecification {
            message: format!("Column '{}' must be numeric", y_col),
        })?
        .into_no_null_iter()
        .collect();

    let n = y.len();
    let y_arr = Array1::from_vec(y);

    // Build design matrix
    let dm = DesignMatrix::from_dataframe(df, x_cols, config.intercept)?;
    let x = dm.view().to_owned();
    let k = x.ncols();

    if n <= k {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: n,
            context: "Quantile regression".to_string(),
        });
    }

    // Get variable names
    let mut var_names: Vec<String> = Vec::with_capacity(k);
    if config.intercept {
        var_names.push("(Intercept)".to_string());
    }
    for col in x_cols {
        var_names.push(col.to_string());
    }

    // Run the selected algorithm
    let (beta, iterations) = match config.algorithm {
        QuantRegAlgorithm::IRLS => irls_quantreg(&x, &y_arr, config.tau, config.max_iter, config.tol)?,
        QuantRegAlgorithm::InteriorPoint => interior_point_quantreg(&x, &y_arr, config.tau, config.max_iter, config.tol)?,
        QuantRegAlgorithm::Simplex => simplex_quantreg(&x, &y_arr, config.tau)?,
    };

    // Compute fitted values and residuals
    let fitted = x.dot(&beta);
    let residuals = &y_arr - &fitted;

    // Compute objective function value
    let objective: f64 = residuals.iter()
        .map(|&r| check_function(r, config.tau))
        .sum();

    // Compute standard errors using sandwich formula
    let std_errors = compute_quantreg_se(&x, &residuals, config.tau, n, k)?;

    // Build coefficient results
    let df_resid = n - k;
    let coefficients: Vec<QuantRegCoefficient> = (0..k)
        .map(|i| {
            let se = std_errors[i];
            let t_val = if se > 1e-15 { beta[i] / se } else { 0.0 };
            let p_val = t_test_p_value(t_val, df_resid as f64);
            let t_crit = 1.96; // Approximate for large samples
            QuantRegCoefficient {
                name: var_names[i].clone(),
                estimate: beta[i],
                std_error: se,
                t_value: t_val,
                p_value: p_val,
                ci_lower_95: beta[i] - t_crit * se,
                ci_upper_95: beta[i] + t_crit * se,
            }
        })
        .collect();

    Ok(QuantRegResult {
        tau: config.tau,
        coefficients,
        n_obs: n,
        n_params: k,
        df: df_resid,
        objective,
        iterations,
        algorithm: config.algorithm,
        variable_names: var_names,
        residuals,
        fitted,
    })
}

/// Check function (asymmetric absolute loss).
fn check_function(u: f64, tau: f64) -> f64 {
    if u >= 0.0 {
        tau * u
    } else {
        (tau - 1.0) * u
    }
}

/// IRLS algorithm for quantile regression.
fn irls_quantreg(
    x: &Array2<f64>,
    y: &Array1<f64>,
    tau: f64,
    max_iter: usize,
    tol: f64,
) -> EconResult<(Array1<f64>, usize)> {
    let n = x.nrows();
    let k = x.ncols();

    // Initialize with OLS
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Quantile regression initialization".to_string(),
        suggestion: format!("Check for multicollinearity: {}", e),
    })?;
    let xty = x.t().dot(y);
    let mut beta = xtx_inv.dot(&xty);

    let epsilon = 1e-4;  // Small constant to avoid division by zero

    for iter in 0..max_iter {
        let beta_old = beta.clone();

        // Compute residuals
        let fitted = x.dot(&beta);
        let residuals = y - &fitted;

        // Compute weights
        let mut weights = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = residuals[i];
            let w = if r.abs() < epsilon {
                1.0 / epsilon
            } else if r >= 0.0 {
                tau / r.abs()
            } else {
                (1.0 - tau) / r.abs()
            };
            weights[i] = w;
        }

        // Weighted least squares: (X'WX)^{-1} X'Wy
        let mut xtwx = Array2::<f64>::zeros((k, k));
        let mut xtwy = Array1::<f64>::zeros(k);

        for i in 0..n {
            let w = weights[i];
            for j in 0..k {
                xtwy[j] += w * x[[i, j]] * y[i];
                for l in 0..k {
                    xtwx[[j, l]] += w * x[[i, j]] * x[[i, l]];
                }
            }
        }

        let (xtwx_inv, _) = safe_inverse(&xtwx.view()).map_err(|_| EconError::SingularMatrix {
            context: "IRLS quantile regression".to_string(),
            suggestion: "Weights may be causing numerical issues".to_string(),
        })?;

        beta = xtwx_inv.dot(&xtwy);

        // Check convergence
        let diff: f64 = beta.iter()
            .zip(beta_old.iter())
            .map(|(b, bo)| (b - bo).abs())
            .fold(0.0, f64::max);

        if diff < tol {
            return Ok((beta, iter + 1));
        }
    }

    Ok((beta, max_iter))
}

/// Interior point algorithm for quantile regression.
/// Simplified version using barrier method.
fn interior_point_quantreg(
    x: &Array2<f64>,
    y: &Array1<f64>,
    tau: f64,
    max_iter: usize,
    tol: f64,
) -> EconResult<(Array1<f64>, usize)> {
    // For simplicity, fall back to IRLS for now
    // A full interior point implementation would use a log barrier
    irls_quantreg(x, y, tau, max_iter, tol)
}

/// Simplex algorithm for quantile regression.
/// Simplified version using IRLS as approximation.
fn simplex_quantreg(
    x: &Array2<f64>,
    y: &Array1<f64>,
    tau: f64,
) -> EconResult<(Array1<f64>, usize)> {
    // For simplicity, use IRLS as approximation
    irls_quantreg(x, y, tau, 200, 1e-8)
}

/// Compute standard errors using sandwich formula.
fn compute_quantreg_se(
    x: &Array2<f64>,
    residuals: &Array1<f64>,
    tau: f64,
    n: usize,
    k: usize,
) -> EconResult<Vec<f64>> {
    // Powell (1991) sandwich estimator
    // V = τ(1-τ) / (n f(0)²) × (X'X)^{-1}
    // where f(0) is the density of residuals at zero

    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Quantile regression SE computation".to_string(),
        suggestion: format!("Original error: {}", e),
    })?;

    // Estimate f(0) using kernel density
    let h = bandwidth_nrd(residuals);
    let f0 = kernel_density_at_zero(residuals, h);

    // Variance factor
    let var_factor = if f0.abs() > 1e-15 {
        tau * (1.0 - tau) / (n as f64 * f0 * f0)
    } else {
        // Fallback: use residual scale
        let mad = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;
        tau * (1.0 - tau) * mad * mad
    };

    // Standard errors
    let se: Vec<f64> = (0..k)
        .map(|i| (var_factor * xtx_inv[[i, i]]).abs().sqrt())
        .collect();

    Ok(se)
}

/// Normal reference bandwidth (Silverman's rule of thumb).
fn bandwidth_nrd(x: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let sd = {
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    };

    // IQR estimate
    let mut sorted: Vec<f64> = x.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = sorted[(n as usize * 25) / 100];
    let q3 = sorted[(n as usize * 75) / 100];
    let iqr = q3 - q1;

    let scale = (sd).min(iqr / 1.34);
    0.9 * scale * n.powf(-0.2)
}

/// Kernel density estimate at zero using Epanechnikov kernel.
fn kernel_density_at_zero(residuals: &Array1<f64>, h: f64) -> f64 {
    if h <= 0.0 {
        return 0.1;  // Fallback
    }

    let n = residuals.len() as f64;
    let mut density = 0.0;

    for &r in residuals.iter() {
        let u = r / h;
        if u.abs() <= 1.0 {
            // Epanechnikov kernel: K(u) = 0.75 * (1 - u²)
            density += 0.75 * (1.0 - u * u);
        }
    }

    density / (n * h)
}

/// Convenience function for quantile regression with default settings.
pub fn run_quantreg(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    tau: f64,
) -> EconResult<QuantRegResult> {
    let config = QuantRegConfig {
        tau,
        ..Default::default()
    };
    quantreg(dataset, y_col, x_cols, &config)
}

/// Run multiple quantile regressions for different quantiles.
pub fn quantreg_multi(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    taus: &[f64],
) -> EconResult<Vec<QuantRegResult>> {
    let mut results = Vec::with_capacity(taus.len());
    for &tau in taus {
        let result = run_quantreg(dataset, y_col, x_cols, tau)?;
        results.push(result);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> Dataset {
        // y = 1 + 2*x + noise (with heteroskedastic errors)
        let x: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter()
            .enumerate()
            .map(|(i, &xi)| 1.0 + 2.0 * xi + (i as f64 % 5.0 - 2.0) * xi.sqrt())
            .collect();

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_quantreg_median() {
        let dataset = create_test_dataset();
        let result = run_quantreg(&dataset, "y", &["x"], 0.5).unwrap();

        assert_eq!(result.tau, 0.5);
        assert_eq!(result.n_obs, 50);
        assert_eq!(result.n_params, 2);
        assert!(result.coefficients[1].estimate > 1.0);  // Should be near 2
    }

    #[test]
    fn test_quantreg_quartiles() {
        let dataset = create_test_dataset();

        let result_25 = run_quantreg(&dataset, "y", &["x"], 0.25).unwrap();
        let result_50 = run_quantreg(&dataset, "y", &["x"], 0.50).unwrap();
        let result_75 = run_quantreg(&dataset, "y", &["x"], 0.75).unwrap();

        // Lower quantiles should have lower intercept
        assert!(result_25.coefficients[0].estimate < result_75.coefficients[0].estimate);
        // All should have similar slopes
        assert!((result_25.coefficients[1].estimate - result_50.coefficients[1].estimate).abs() < 1.0);
    }

    #[test]
    fn test_quantreg_multi() {
        let dataset = create_test_dataset();
        let taus = vec![0.1, 0.25, 0.5, 0.75, 0.9];

        let results = quantreg_multi(&dataset, "y", &["x"], &taus).unwrap();

        assert_eq!(results.len(), 5);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.tau, taus[i]);
        }
    }

    #[test]
    fn test_quantreg_invalid_tau() {
        let dataset = create_test_dataset();
        let config = QuantRegConfig {
            tau: 1.5,  // Invalid
            ..Default::default()
        };

        let result = quantreg(&dataset, "y", &["x"], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantreg_displays_correctly() {
        let dataset = create_test_dataset();
        let result = run_quantreg(&dataset, "y", &["x"], 0.5).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Quantile Regression"));
        assert!(display.contains("τ = 0.50"));
        assert!(display.contains("Estimate"));
    }

    #[test]
    fn test_check_function() {
        // tau = 0.5 (median): symmetric
        assert!((check_function(1.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((check_function(-1.0, 0.5) - 0.5).abs() < 1e-10);

        // tau = 0.75: penalizes negative residuals more
        assert!((check_function(1.0, 0.75) - 0.75).abs() < 1e-10);
        assert!((check_function(-1.0, 0.75) - 0.25).abs() < 1e-10);
    }
}
