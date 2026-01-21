//! Generalized Least Squares (GLS) regression.
//!
//! GLS extends OLS to handle correlated and heteroscedastic errors.
//! The model is Y = Xβ + ε where Var(ε) = σ²Ω.

use ndarray::{Array1, Array2, ArrayView2, s};
use serde::{Deserialize, Serialize};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{xtx, safe_inverse};
use crate::traits::estimator::t_test_p_value;

/// Correlation structure for GLS.
#[derive(Debug, Clone)]
pub enum CorrelationStructure {
    /// AR(1) correlation: Ω[i,j] = ρ^|i-j|
    AR1 { rho: f64 },
    /// Compound symmetry: Ω[i,j] = ρ for i≠j, 1 for i=j
    CompoundSymmetry { rho: f64 },
    /// Known correlation matrix (user-provided)
    Known { omega: Array2<f64> },
    /// Identity (equivalent to OLS)
    Identity,
}

/// Result of GLS estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlsResult {
    /// Coefficient estimates
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_values: Vec<f64>,
    /// p-values (two-sided)
    pub p_values: Vec<f64>,
    /// Residuals
    #[serde(skip)]
    pub residuals: Vec<f64>,
    /// Fitted values
    #[serde(skip)]
    pub fitted: Vec<f64>,
    /// Residual standard error (σ)
    pub sigma: f64,
    /// R-squared (generalized)
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Degrees of freedom for residuals
    pub df_residual: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Correlation structure used
    pub correlation: String,
    /// Estimated correlation parameter (if applicable)
    pub correlation_param: Option<f64>,
}

/// Fit a GLS model.
///
/// Estimates Y = Xβ + ε where Var(ε) = σ²Ω using generalized least squares.
///
/// # Arguments
///
/// * `y` - Response vector (n × 1)
/// * `x` - Design matrix (n × p), should include intercept if needed
/// * `correlation` - Correlation structure for errors
///
/// # Returns
///
/// A `GlsResult` containing coefficient estimates and diagnostics.
pub fn gls(
    y: &ndarray::ArrayView1<f64>,
    x: &ArrayView2<f64>,
    correlation: CorrelationStructure,
) -> EconResult<GlsResult> {
    let n = y.len();
    let p = x.ncols();

    if x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: "X and y must have the same number of observations".to_string(),
        });
    }

    if n <= p {
        return Err(EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: "GLS regression".to_string(),
        });
    }

    // Build the correlation matrix Ω
    let omega = build_correlation_matrix(n, &correlation)?;

    // Compute Ω^(-1/2) via Cholesky: Ω = LL', so Ω^(-1/2) = L^(-T)
    let omega_inv_sqrt = compute_omega_inv_sqrt(&omega)?;

    // Transform data: y* = Ω^(-1/2) y, X* = Ω^(-1/2) X
    let y_arr = y.to_owned();
    let y_star = omega_inv_sqrt.dot(&y_arr);
    let x_star = omega_inv_sqrt.dot(x);

    // Now apply OLS to transformed data
    let xtx_star = xtx(&x_star.view());
    let xty_star: Array1<f64> = x_star.t().dot(&y_star);

    let (xtx_inv, _cond) = safe_inverse(&xtx_star.view())?;

    // Coefficients: β = (X*'X*)^(-1) X*'y*
    let beta = xtx_inv.dot(&xty_star);

    // Fitted values in original space
    let fitted: Array1<f64> = x.dot(&beta);

    // Residuals
    let residuals: Array1<f64> = y.to_owned() - &fitted;

    // Transformed residuals for sigma estimation
    let residuals_star: Array1<f64> = &y_star - &x_star.dot(&beta);

    // Residual variance: σ² = (e*'e*) / (n - p)
    let df_residual = n - p;
    let sse: f64 = residuals_star.dot(&residuals_star);
    let sigma_sq = sse / df_residual as f64;
    let sigma = sigma_sq.sqrt();

    // Variance-covariance of β: Var(β) = σ² (X*'X*)^(-1)
    let vcov = &xtx_inv * sigma_sq;

    // Standard errors
    let std_errors: Array1<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt());

    // t-statistics
    let t_values: Array1<f64> = &beta / &std_errors;

    // p-values
    let p_values: Array1<f64> = t_values.mapv(|t| t_test_p_value(t, df_residual as f64));

    // R-squared (using original residuals)
    let y_mean = y.sum() / n as f64;
    let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.dot(&residuals);
    let r_squared = if sst > 0.0 { 1.0 - ss_res / sst } else { 0.0 };
    let adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_residual as f64;

    // Log-likelihood
    let log_likelihood = -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma_sq.ln());

    // AIC and BIC
    let n_params = p + 1; // +1 for sigma
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n as f64).ln() * n_params as f64;

    // Extract correlation parameter if applicable
    let (corr_name, corr_param) = match &correlation {
        CorrelationStructure::AR1 { rho } => ("AR(1)".to_string(), Some(*rho)),
        CorrelationStructure::CompoundSymmetry { rho } => ("Compound Symmetry".to_string(), Some(*rho)),
        CorrelationStructure::Known { .. } => ("User-specified".to_string(), None),
        CorrelationStructure::Identity => ("Identity (OLS)".to_string(), None),
    };

    Ok(GlsResult {
        coefficients: beta.to_vec(),
        std_errors: std_errors.to_vec(),
        t_values: t_values.to_vec(),
        p_values: p_values.to_vec(),
        residuals: residuals.to_vec(),
        fitted: fitted.to_vec(),
        sigma,
        r_squared,
        adj_r_squared,
        log_likelihood,
        aic,
        bic,
        n_obs: n,
        df_residual,
        n_params,
        correlation: corr_name,
        correlation_param: corr_param,
    })
}

/// Build correlation matrix Ω from the specified structure.
fn build_correlation_matrix(n: usize, structure: &CorrelationStructure) -> EconResult<Array2<f64>> {
    match structure {
        CorrelationStructure::AR1 { rho } => {
            if rho.abs() >= 1.0 {
                return Err(EconError::InvalidSpecification {
                    message: "AR(1) correlation |rho| must be < 1".to_string(),
                });
            }
            let mut omega = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let lag = (i as i64 - j as i64).unsigned_abs() as i32;
                    omega[[i, j]] = rho.powi(lag);
                }
            }
            Ok(omega)
        }
        CorrelationStructure::CompoundSymmetry { rho } => {
            if *rho <= -1.0 / (n as f64 - 1.0) || *rho >= 1.0 {
                return Err(EconError::InvalidSpecification {
                    message: "Compound symmetry rho out of valid range".to_string(),
                });
            }
            let mut omega = Array2::from_elem((n, n), *rho);
            for i in 0..n {
                omega[[i, i]] = 1.0;
            }
            Ok(omega)
        }
        CorrelationStructure::Known { omega } => {
            if omega.nrows() != n || omega.ncols() != n {
                return Err(EconError::InvalidSpecification {
                    message: format!("Known Omega must be {}x{}", n, n),
                });
            }
            Ok(omega.clone())
        }
        CorrelationStructure::Identity => {
            Ok(Array2::eye(n))
        }
    }
}

/// Compute Ω^(-1/2) using Cholesky decomposition.
fn compute_omega_inv_sqrt(omega: &Array2<f64>) -> EconResult<Array2<f64>> {
    let n = omega.nrows();

    // For identity matrix, return identity
    if is_identity(omega) {
        return Ok(Array2::eye(n));
    }

    // Cholesky decomposition: Ω = LL'
    let l = cholesky_lower(omega)?;

    // Compute L^(-1) by forward substitution
    let l_inv = lower_triangular_inverse(&l)?;

    // Ω^(-1/2) = L^(-T) = (L^(-1))'
    Ok(l_inv.t().to_owned())
}

/// Check if matrix is identity.
fn is_identity(m: &Array2<f64>) -> bool {
    let n = m.nrows();
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (m[[i, j]] - expected).abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// Cholesky decomposition (lower triangular L such that A = LL').
fn cholesky_lower(a: &Array2<f64>) -> EconResult<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }

            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    return Err(EconError::SingularMatrix {
                        context: "Cholesky decomposition failed: matrix not positive definite".to_string(),
                        suggestion: "Check that the correlation matrix is positive definite".to_string(),
                    });
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Compute inverse of lower triangular matrix.
fn lower_triangular_inverse(l: &Array2<f64>) -> EconResult<Array2<f64>> {
    let n = l.nrows();
    let mut inv = Array2::zeros((n, n));

    for i in 0..n {
        if l[[i, i]].abs() < 1e-15 {
            return Err(EconError::SingularMatrix {
                context: "Lower triangular matrix has zero on diagonal".to_string(),
                suggestion: "Check matrix conditioning".to_string(),
            });
        }
        inv[[i, i]] = 1.0 / l[[i, i]];

        for j in 0..i {
            let mut sum = 0.0;
            for k in j..i {
                sum += l[[i, k]] * inv[[k, j]];
            }
            inv[[i, j]] = -sum / l[[i, i]];
        }
    }

    Ok(inv)
}

/// Fit GLS with automatic AR(1) correlation parameter estimation.
///
/// First fits OLS, then estimates rho from residuals, then fits GLS.
pub fn gls_ar1_auto(
    y: &ndarray::ArrayView1<f64>,
    x: &ArrayView2<f64>,
) -> EconResult<GlsResult> {
    // Step 1: OLS to get residuals
    let xtx_mat = xtx(x);
    let xty_vec: Array1<f64> = x.t().dot(&y.to_owned());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view())?;
    let beta_ols = xtx_inv.dot(&xty_vec);
    let residuals_ols: Array1<f64> = y.to_owned() - x.dot(&beta_ols);

    // Step 2: Estimate rho from residuals: rho = cor(e_t, e_{t-1})
    let e_lag: Vec<f64> = residuals_ols.slice(s![..-1]).to_vec();
    let e_lead: Vec<f64> = residuals_ols.slice(s![1..]).to_vec();

    let n_lag = e_lag.len() as f64;
    let mean_lag: f64 = e_lag.iter().sum::<f64>() / n_lag;
    let mean_lead: f64 = e_lead.iter().sum::<f64>() / n_lag;

    let mut cov = 0.0;
    let mut var_lag = 0.0;
    let mut var_lead = 0.0;

    for i in 0..e_lag.len() {
        let d_lag = e_lag[i] - mean_lag;
        let d_lead = e_lead[i] - mean_lead;
        cov += d_lag * d_lead;
        var_lag += d_lag * d_lag;
        var_lead += d_lead * d_lead;
    }

    let rho = if var_lag > 0.0 && var_lead > 0.0 {
        cov / (var_lag * var_lead).sqrt()
    } else {
        0.0
    };

    // Bound rho to valid range
    let rho = rho.max(-0.99).min(0.99);

    // Step 3: Fit GLS with estimated rho
    gls(y, x, CorrelationStructure::AR1 { rho })
}

/// Run GLS with string correlation specification (MCP wrapper).
pub fn run_gls(
    y: &[f64],
    x: &[f64],
    n_cols: usize,
    correlation_type: &str,
    correlation_param: Option<f64>,
) -> EconResult<GlsResult> {
    let n = y.len();
    let y_arr = Array1::from_vec(y.to_vec());
    let x_arr = Array2::from_shape_vec((n, n_cols), x.to_vec())
        .map_err(|e| EconError::InvalidSpecification {
            message: format!("Invalid X matrix shape: {}", e),
        })?;

    let correlation = match correlation_type.to_lowercase().as_str() {
        "ar1" => {
            let rho = correlation_param.ok_or_else(|| EconError::InvalidSpecification {
                message: "AR(1) requires rho parameter".to_string(),
            })?;
            CorrelationStructure::AR1 { rho }
        }
        "ar1_auto" => {
            return gls_ar1_auto(&y_arr.view(), &x_arr.view());
        }
        "compound_symmetry" | "cs" => {
            let rho = correlation_param.ok_or_else(|| EconError::InvalidSpecification {
                message: "Compound symmetry requires rho parameter".to_string(),
            })?;
            CorrelationStructure::CompoundSymmetry { rho }
        }
        "identity" | "ols" => CorrelationStructure::Identity,
        _ => return Err(EconError::InvalidSpecification {
            message: format!("Unknown correlation type: {}", correlation_type),
        }),
    };

    gls(&y_arr.view(), &x_arr.view(), correlation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gls_identity() {
        // With identity correlation, GLS should equal OLS
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0];
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];

        let result = gls(&y.view(), &x.view(), CorrelationStructure::Identity).unwrap();

        // Slope should be close to 1
        assert!((result.coefficients[1] - 1.0).abs() < 0.2);
        assert!(result.r_squared > 0.9);
    }

    #[test]
    fn test_gls_ar1() {
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0, 6.0, 7.1, 8.0, 9.0, 10.1];
        let x = array![
            [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
            [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0], [1.0, 10.0]
        ];

        let result = gls(&y.view(), &x.view(), CorrelationStructure::AR1 { rho: 0.5 }).unwrap();

        // Should still get reasonable estimates
        assert!((result.coefficients[1] - 1.0).abs() < 0.3);
        assert!(result.r_squared > 0.9);
    }

    #[test]
    fn test_gls_compound_symmetry() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];

        let result = gls(&y.view(), &x.view(), CorrelationStructure::CompoundSymmetry { rho: 0.3 }).unwrap();

        assert!((result.coefficients[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_gls_ar1_auto() {
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0, 6.0, 7.1, 8.0, 9.0, 10.1];
        let x = array![
            [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
            [1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0], [1.0, 10.0]
        ];

        let result = gls_ar1_auto(&y.view(), &x.view()).unwrap();

        // Should estimate rho automatically
        assert!(result.correlation_param.is_some());
        assert!(result.coefficients[1] > 0.5);
    }

    #[test]
    fn test_build_ar1_matrix() {
        let omega = build_correlation_matrix(4, &CorrelationStructure::AR1 { rho: 0.5 }).unwrap();

        // Check structure: Ω[i,j] = 0.5^|i-j|
        assert!((omega[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((omega[[0, 1]] - 0.5).abs() < 1e-10);
        assert!((omega[[0, 2]] - 0.25).abs() < 1e-10);
        assert!((omega[[0, 3]] - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        let a = array![[4.0, 2.0], [2.0, 5.0]];
        let l = cholesky_lower(&a).unwrap();

        // Verify LL' = A
        let reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10);
            }
        }
    }
}
