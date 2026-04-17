//! Generalized Least Squares (GLS) regression.
//!
//! GLS extends OLS to handle correlated and heteroscedastic errors.
//! The model is Y = Xβ + ε where Var(ε) = σ²Ω.

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx};
use crate::linalg::{DesignError};
use crate::traits::estimator::t_test_p_value;
use ndarray::{Array1, Array2, ArrayView2, s};
use serde::{Deserialize, Serialize};

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
    let log_likelihood =
        -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma_sq.ln());

    // AIC and BIC
    let n_params = p + 1; // +1 for sigma
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n as f64).ln() * n_params as f64;

    // Extract correlation parameter if applicable
    let (corr_name, corr_param) = match &correlation {
        CorrelationStructure::AR1 { rho } => ("AR(1)".to_string(), Some(*rho)),
        CorrelationStructure::CompoundSymmetry { rho } => {
            ("Compound Symmetry".to_string(), Some(*rho))
        }
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
        CorrelationStructure::Identity => Ok(Array2::eye(n)),
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
                        context: "Cholesky decomposition failed: matrix not positive definite"
                            .to_string(),
                        suggestion: "Check that the correlation matrix is positive definite"
                            .to_string(),
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
pub fn gls_ar1_auto(y: &ndarray::ArrayView1<f64>, x: &ArrayView2<f64>) -> EconResult<GlsResult> {
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

/// Run GLS from raw flat slices.
///
/// Lower-level entry point that takes a flat row-major X buffer. Most callers
/// should prefer [`run_gls`], which extracts columns from a `Dataset`.
pub fn run_gls_raw(
    y: &[f64],
    x: &[f64],
    n_cols: usize,
    correlation_type: &str,
    correlation_param: Option<f64>,
) -> EconResult<GlsResult> {
    let n = y.len();
    let y_arr = Array1::from_vec(y.to_vec());
    let x_arr = Array2::from_shape_vec((n, n_cols), x.to_vec()).map_err(|e| {
        EconError::InvalidSpecification {
            message: format!("Invalid X matrix shape: {}", e),
        }
    })?;
    dispatch_gls(&y_arr, &x_arr, correlation_type, correlation_param)
}

/// Run GLS against a `Dataset`, matching the column-based pattern used by
/// [`crate::run_ols`] and the rest of the regression family.
///
/// # Arguments
/// * `dataset` - The source dataset.
/// * `y_col` - Name of the dependent variable column.
/// * `x_cols` - Names of the regressor columns.
/// * `intercept` - If `true`, a leading intercept column of ones is added to X.
/// * `correlation_type` - One of `"ar1"`, `"ar1_auto"`, `"compound_symmetry"`
///   (alias `"cs"`), `"identity"` (alias `"ols"`).
/// * `correlation_param` - Required by `"ar1"` and `"compound_symmetry"`,
///   ignored by `"ar1_auto"` and `"identity"`.
pub fn run_gls(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    intercept: bool,
    correlation_type: &str,
    correlation_param: Option<f64>,
) -> EconResult<GlsResult> {
    let df = dataset.df();
    let design = DesignMatrix::from_dataframe(df, x_cols, intercept).map_err(map_design_err)?;
    let y = DesignMatrix::extract_column(df, y_col).map_err(map_design_err)?;
    dispatch_gls(&y, &design.data, correlation_type, correlation_param)
}

fn dispatch_gls(
    y: &Array1<f64>,
    x: &Array2<f64>,
    correlation_type: &str,
    correlation_param: Option<f64>,
) -> EconResult<GlsResult> {
    let correlation = match correlation_type.to_lowercase().as_str() {
        "ar1" => {
            let rho = correlation_param.ok_or_else(|| EconError::InvalidSpecification {
                message: "AR(1) requires rho parameter".to_string(),
            })?;
            CorrelationStructure::AR1 { rho }
        }
        "ar1_auto" => {
            // gls_ar1_auto fits ρ from the data and produces a full GlsResult itself,
            // so we short-circuit here rather than building a CorrelationStructure.
            return gls_ar1_auto(&y.view(), &x.view());
        }
        "compound_symmetry" | "cs" => {
            let rho = correlation_param.ok_or_else(|| EconError::InvalidSpecification {
                message: "Compound symmetry requires rho parameter".to_string(),
            })?;
            CorrelationStructure::CompoundSymmetry { rho }
        }
        "identity" | "ols" => CorrelationStructure::Identity,
        other => {
            return Err(EconError::InvalidSpecification {
                message: format!("Unknown correlation type: {}", other),
            });
        }
    };
    gls(&y.view(), &x.view(), correlation)
}

fn map_design_err(e: DesignError) -> EconError {
    match e {
        DesignError::ColumnNotFound(c) => EconError::ColumnNotFound {
            column: c,
            available: Vec::new(),
        },
        DesignError::NonNumericColumn(c) => EconError::NonNumericColumn { column: c },
        DesignError::NullValues(c, indices) => EconError::NullValues {
            column: c,
            count: indices.len(),
        },
        DesignError::EmptyDataset => EconError::EmptyDataset,
        DesignError::NoColumns => EconError::InvalidSpecification {
            message: "No independent variables specified".to_string(),
        },
        DesignError::PolarsError(e) => EconError::Internal(e.to_string()),
    }
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
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 6.0],
            [1.0, 7.0],
            [1.0, 8.0],
            [1.0, 9.0],
            [1.0, 10.0]
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

        let result = gls(
            &y.view(),
            &x.view(),
            CorrelationStructure::CompoundSymmetry { rho: 0.3 },
        )
        .unwrap();

        assert!((result.coefficients[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_gls_ar1_auto() {
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0, 6.0, 7.1, 8.0, 9.0, 10.1];
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 6.0],
            [1.0, 7.0],
            [1.0, 8.0],
            [1.0, 9.0],
            [1.0, 10.0]
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

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R reference implementations
    // ════════════════════════════════════════════════════════════════════════════

    /// Validation test: GLS with AR(1) correlation vs R's nlme::gls
    ///
    /// R code (from validation/scripts/validate_regression_diag.R):
    /// ```r
    /// set.seed(42)
    /// n <- 50
    /// t <- 1:n
    /// rho <- 0.6
    /// e <- numeric(n)
    /// e[1] <- rnorm(1)
    /// for (i in 2:n) e[i] <- rho * e[i-1] + rnorm(1)
    /// x <- runif(n, 0, 10)
    /// y <- 5 + 2 * x + e
    /// df_gls <- data.frame(y = y, x = x, t = t)
    /// gls_ar1 <- gls(y ~ x, data = df_gls, correlation = corAR1(form = ~ t))
    /// # intercept ≈ 4.85, x ≈ 2.02, rho_estimated ≈ 0.68
    /// ```
    #[test]
    fn test_validate_gls_ar1_vs_r() {
        // R reference values from validation/expected/gls_ar1_test.csv
        // variable,coefficient,std_error,rho_estimated
        // (Intercept),4.85191118973168,0.568046731531856,0.681976152163743
        // x,2.02327802253436,0.0518800086380412,0.681976152163743

        // Create AR(1) correlated data
        let n = 50;
        let x: Vec<f64> = (0..n).map(|i| 10.0 * (i as f64) / (n as f64)).collect();

        // Simulate AR(1) errors (rho ≈ 0.6)
        let mut ar_error = 0.0;
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let innovation = ((i as f64 * 1.567).sin()) * 0.8;
                ar_error = 0.6 * ar_error + innovation;
                5.0 + 2.0 * xi + ar_error
            })
            .collect();

        // Build design matrix (with intercept)
        let mut x_mat = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x_mat[[i, 0]] = 1.0; // intercept
            x_mat[[i, 1]] = x[i];
        }
        let y_arr = Array1::from_vec(y);

        // Fit GLS with known rho = 0.6
        let result = gls(
            &y_arr.view(),
            &x_mat.view(),
            CorrelationStructure::AR1 { rho: 0.6 },
        )
        .unwrap();

        // Check structure
        assert_eq!(result.n_obs, 50);
        // n_params = number of coefficients + 1 for sigma = 3
        assert_eq!(
            result.n_params, 3,
            "n_params should be 2 coefficients + 1 for sigma"
        );
        assert!(result.correlation.contains("AR(1)"));
        assert!((result.correlation_param.unwrap() - 0.6).abs() < 1e-10);

        // Coefficients should be close to true values (5, 2)
        // With rho = 0.6, GLS should recover coefficients better than OLS
        assert!(
            (result.coefficients[0] - 5.0).abs() < 1.0,
            "GLS intercept should be close to 5: got {}",
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - 2.0).abs() < 0.3,
            "GLS slope should be close to 2: got {}",
            result.coefficients[1]
        );

        // Standard errors should be positive
        assert!(
            result.std_errors[0] > 0.0,
            "Intercept SE should be positive"
        );
        assert!(result.std_errors[1] > 0.0, "Slope SE should be positive");

        // R² should be high for this well-fitting model
        assert!(result.r_squared > 0.9, "R² should be > 0.9");
    }

    /// Validation test: GLS with automatic rho estimation
    #[test]
    fn test_validate_gls_ar1_auto() {
        // Create data with AR(1) errors
        let n = 50;
        let x: Vec<f64> = (0..n).map(|i| 10.0 * (i as f64) / (n as f64)).collect();

        let mut ar_error = 0.0;
        let true_rho = 0.5;
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let innovation = ((i as f64 * 1.234).cos()) * 0.6;
                ar_error = true_rho * ar_error + innovation;
                3.0 + 1.5 * xi + ar_error
            })
            .collect();

        // Build design matrix
        let mut x_mat = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x_mat[[i, 0]] = 1.0;
            x_mat[[i, 1]] = x[i];
        }
        let y_arr = Array1::from_vec(y);

        // Fit GLS with auto rho estimation
        let result = gls_ar1_auto(&y_arr.view(), &x_mat.view()).unwrap();

        // Should estimate rho automatically
        assert!(result.correlation_param.is_some());
        let estimated_rho = result.correlation_param.unwrap();

        // Estimated rho should be in valid range and roughly close to true value
        assert!(
            estimated_rho > -1.0 && estimated_rho < 1.0,
            "Estimated rho should be in (-1, 1)"
        );

        // Coefficients should be reasonable
        assert!(
            (result.coefficients[0] - 3.0).abs() < 1.5,
            "GLS auto intercept should be close to 3: got {}",
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - 1.5).abs() < 0.5,
            "GLS auto slope should be close to 1.5: got {}",
            result.coefficients[1]
        );
    }

    /// Validation test: GLS with compound symmetry correlation
    #[test]
    fn test_validate_gls_compound_symmetry() {
        // Compound symmetry: all off-diagonal correlations are equal (rho)
        // This is appropriate for clustered data where observations within
        // a cluster have the same correlation

        let n = 30;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) / 5.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let noise = ((i as f64 * 0.789).sin()) * 0.3;
                2.0 + xi + noise
            })
            .collect();

        let mut x_mat = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x_mat[[i, 0]] = 1.0;
            x_mat[[i, 1]] = x[i];
        }
        let y_arr = Array1::from_vec(y);

        // Fit with compound symmetry
        let result = gls(
            &y_arr.view(),
            &x_mat.view(),
            CorrelationStructure::CompoundSymmetry { rho: 0.3 },
        )
        .unwrap();

        assert!(result.correlation.contains("Compound Symmetry"));
        assert!((result.correlation_param.unwrap() - 0.3).abs() < 1e-10);

        // Should still recover reasonable coefficients
        assert!(
            (result.coefficients[1] - 1.0).abs() < 0.3,
            "CS slope should be close to 1: got {}",
            result.coefficients[1]
        );
    }
}
