//! Generalized Least Squares (GLS) regression.
//!
//! GLS extends OLS to handle correlated and heteroscedastic errors.
//! The model is Y = Xβ + ε where Var(ε) = σ²Ω.

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::DesignError;
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx};
use crate::resource;
use crate::traits::estimator::t_test_p_value;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
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
/// The data are whitened with a matrix `P` such that `P'P = Ω⁻¹` and OLS is fit
/// to `(P y, P X)`, which gives exactly the GLS estimator, its standard errors
/// and σ. `P` is the Prais-Winsten transform for AR(1), a closed-form symmetric
/// square root of `Ω⁻¹` for compound symmetry (both O(n p)), no transform for
/// the identity, and `L⁻¹` applied by forward substitution, where `Ω = L L'`, for
/// a known `Ω` (an O(n³) Cholesky factor, allocated fallibly, then O(n² p)
/// whitening).
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

    // Transform data: y* = P y, X* = P X with P'P = Ω⁻¹
    let (y_star, x_star) = whiten(y, x, &correlation)?;

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

/// Whitened data `(P y, P X)` for a whitening matrix `P` with `P'P = Ω⁻¹`.
///
/// Any such `P` turns GLS into OLS on the transformed data. Checks the
/// cancellation budget before any structure-specific work (rho validation,
/// the Cholesky factor, or the copy).
fn whiten(
    y: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    structure: &CorrelationStructure,
) -> EconResult<(Array1<f64>, Array2<f64>)> {
    resource::checkpoint()?;
    let n = y.len();
    match structure {
        // P = I: a plain copy, allocated fallibly like every other X*. The buffer
        // takes x's own shape, so the copy cannot mismatch whatever the caller checked.
        CorrelationStructure::Identity => {
            let (rows, cols) = x.dim();
            let mut x_star = resource::try_zeros2(rows, cols, WHITENED_DESIGN)?;
            x_star.assign(x);
            Ok((y.to_owned(), x_star))
        }
        CorrelationStructure::AR1 { rho } => {
            let rho = *rho;
            if !rho.is_finite() || rho.abs() >= 1.0 {
                return Err(EconError::InvalidSpecification {
                    message: format!("AR(1) correlation |rho| must be < 1 (got {rho})"),
                });
            }
            // Prais-Winsten: exactly L⁻¹ for the Cholesky factor of Ω[i,j] = ρ^|i-j|
            let scale = (1.0 - rho * rho).sqrt();
            let transform = |v: &mut [f64]| {
                for t in (1..v.len()).rev() {
                    v[t] = (v[t] - rho * v[t - 1]) / scale;
                }
            };
            let mut y_star = y.to_vec();
            transform(&mut y_star);
            let x_star = whiten_columns(x, |col| {
                transform(col);
                Ok(())
            })?;
            Ok((Array1::from_vec(y_star), x_star))
        }
        CorrelationStructure::CompoundSymmetry { rho } => {
            let rho = *rho;
            if !rho.is_finite() || rho <= -1.0 / (n as f64 - 1.0) || rho >= 1.0 {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "compound symmetry rho must lie in (-1/(n - 1), 1) = ({:.6}, 1) \
                         for n = {n} (got {rho})",
                        -1.0 / (n as f64 - 1.0)
                    ),
                });
            }
            // Ω = (1 - ρ) I + ρ 11' has eigenvalue 1 + (n - 1)ρ on 1 and 1 - ρ on its
            // complement, so P = c (I - a 11'/n) is symmetric with P² = Ω⁻¹.
            let a = 1.0 - ((1.0 - rho) / (1.0 + (n as f64 - 1.0) * rho)).sqrt();
            let c = 1.0 / (1.0 - rho).sqrt();
            let transform = |v: &mut [f64]| {
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                for e in v.iter_mut() {
                    *e = c * (*e - a * mean);
                }
            };
            let mut y_star = y.to_vec();
            transform(&mut y_star);
            let x_star = whiten_columns(x, |col| {
                transform(col);
                Ok(())
            })?;
            Ok((Array1::from_vec(y_star), x_star))
        }
        CorrelationStructure::Known { omega } => {
            if omega.nrows() != n || omega.ncols() != n {
                return Err(EconError::InvalidSpecification {
                    message: format!("Known Omega must be {}x{}", n, n),
                });
            }
            // Ω = L L', P = L⁻¹: solve L z = v by forward substitution
            let l = cholesky_lower(omega)?;
            let mut y_star = y.to_vec();
            forward_substitute(&l, &mut y_star)?;
            let x_star = whiten_columns(x, |col| forward_substitute(&l, col))?;
            Ok((Array1::from_vec(y_star), x_star))
        }
    }
}

/// Allocation label for the whitened design `X*`, shared by every structure.
const WHITENED_DESIGN: &str = "GLS whitened design";

/// Apply a whitening transform to each column of `x`, checking the
/// cancellation budget once per column. The output is allocated fallibly.
fn whiten_columns(
    x: &ArrayView2<f64>,
    mut transform: impl FnMut(&mut [f64]) -> EconResult<()>,
) -> EconResult<Array2<f64>> {
    let (n, p) = x.dim();
    let mut x_star = resource::try_zeros2(n, p, WHITENED_DESIGN)?;
    let mut col = vec![0.0; n];
    for j in 0..p {
        resource::checkpoint()?;
        for (c, &v) in col.iter_mut().zip(x.column(j)) {
            *c = v;
        }
        transform(&mut col)?;
        for (dst, &v) in x_star.column_mut(j).iter_mut().zip(&col) {
            *dst = v;
        }
    }
    Ok(x_star)
}

/// Solve `L z = v` in place for a lower-triangular `L` (forward substitution).
fn forward_substitute(l: &Array2<f64>, v: &mut [f64]) -> EconResult<()> {
    for i in 0..v.len() {
        let diag = l[[i, i]];
        if diag.abs() < 1e-15 {
            return Err(EconError::SingularMatrix {
                context: "Lower triangular matrix has zero on diagonal".to_string(),
                suggestion: "Check matrix conditioning".to_string(),
            });
        }
        let mut sum = v[i];
        for k in 0..i {
            sum -= l[[i, k]] * v[k];
        }
        v[i] = sum / diag;
    }
    Ok(())
}

/// Cholesky decomposition (lower triangular L such that A = LL').
fn cholesky_lower(a: &Array2<f64>) -> EconResult<Array2<f64>> {
    let n = a.nrows();
    let mut l = resource::try_zeros2(n, n, "GLS Cholesky factor")?;

    for i in 0..n {
        resource::checkpoint()?;
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

    /// Dense n x n correlation matrix (the pre-T0-WP14 `build_correlation_matrix`),
    /// kept as a test reference for the closed-form transforms.
    fn dense_correlation_matrix(n: usize, structure: &CorrelationStructure) -> Array2<f64> {
        match structure {
            CorrelationStructure::AR1 { rho } => {
                let mut omega = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let lag = (i as i64 - j as i64).unsigned_abs() as i32;
                        omega[[i, j]] = rho.powi(lag);
                    }
                }
                omega
            }
            CorrelationStructure::CompoundSymmetry { rho } => {
                let mut omega = Array2::from_elem((n, n), *rho);
                for i in 0..n {
                    omega[[i, i]] = 1.0;
                }
                omega
            }
            CorrelationStructure::Known { omega } => omega.clone(),
            CorrelationStructure::Identity => Array2::eye(n),
        }
    }

    /// Textbook GLS with an explicit inverse: beta = (X'O^-1 X)^-1 X'O^-1 y,
    /// sigma^2 = e'O^-1 e / (n - p). Returns (beta, standard errors, sigma).
    fn dense_gls_reference(
        y: &Array1<f64>,
        x: &Array2<f64>,
        omega: &Array2<f64>,
    ) -> (Array1<f64>, Array1<f64>, f64) {
        let (omega_inv, _) = safe_inverse(&omega.view()).unwrap();
        let xto = x.t().dot(&omega_inv);
        let (a_inv, _) = safe_inverse(&xto.dot(x).view()).unwrap();
        let beta = a_inv.dot(&xto.dot(y));
        let e = y - &x.dot(&beta);
        let (n, p) = x.dim();
        let sigma2 = e.dot(&omega_inv.dot(&e)) / (n - p) as f64;
        let se = a_inv.diag().mapv(|v| (v * sigma2).sqrt());
        (beta, se, sigma2.sqrt())
    }

    const NLME_X: [f64; 30] = [
        5.637, 8.080, 7.150, 8.753, 6.938, 6.813, 8.448, 7.718, 7.320, 6.537, 2.562, 2.370, 2.185,
        0.415, 6.587, 0.424, 9.626, 7.693, 8.135, 7.260, 3.562, 6.383, 3.659, 6.179, 5.827, 9.956,
        1.355, 9.545, 1.014, 3.836,
    ];
    const NLME_Y: [f64; 30] = [
        6.5899, 7.4509, 8.7175, 8.8798, 7.3817, 6.2093, 7.3718, 8.4644, 8.4831, 8.5294, 4.1718,
        3.3226, 3.6097, 2.7661, 8.2699, 2.9248, 9.6094, 8.4130, 8.1559, 7.5124, 5.1513, 7.7384,
        4.7780, 7.3021, 5.5900, 9.2503, 1.8103, 8.8721, 2.9977, 4.4985,
    ];

    fn nlme_xy() -> (Array1<f64>, Array2<f64>) {
        let x = Array2::from_shape_fn((30, 2), |(i, j)| if j == 0 { 1.0 } else { NLME_X[i] });
        (Array1::from_vec(NLME_Y.to_vec()), x)
    }

    /// GLS with a fixed correlation vs `nlme::gls` (nlme 3.1.164; REML sigma).
    ///
    /// ```r
    /// library(nlme); set.seed(20261021); n <- 30
    /// x <- round(runif(n, 0, 10), 3); e <- numeric(n); e[1] <- rnorm(1)
    /// for (i in 2:n) e[i] <- 0.6 * e[i - 1] + rnorm(1, sd = 0.8)
    /// y <- round(1.5 + 0.8 * x + e, 4); d <- data.frame(y = y, x = x)
    /// gls(y ~ x, data = d, correlation = corAR1(0.6, fixed = TRUE))
    /// # and corAR1(-0.4), corCompSymm(0.3), corCompSymm(-0.02), no correlation
    /// # reported: coef(fit), summary(fit)$tTable[, 2], fit$sigma
    /// ```
    #[test]
    fn test_validate_gls_fixed_correlation_vs_nlme() {
        let (y, x) = nlme_xy();
        let cases = [
            (
                CorrelationStructure::AR1 { rho: 0.6 },
                [
                    1.941977840640,
                    0.791557641587,
                    0.387453755473,
                    0.040120184694,
                    0.903229314501,
                ],
            ),
            (
                CorrelationStructure::AR1 { rho: -0.4 },
                [
                    2.213916554461,
                    0.747378376163,
                    0.307612193194,
                    0.049986850996,
                    0.906511290861,
                ],
            ),
            (
                CorrelationStructure::CompoundSymmetry { rho: 0.3 },
                [
                    2.093925805991,
                    0.767614285417,
                    0.552217510279,
                    0.046135388346,
                    0.852538446062,
                ],
            ),
            (
                CorrelationStructure::CompoundSymmetry { rho: -0.02 },
                [
                    2.093925805991,
                    0.767614285417,
                    0.277347530251,
                    0.046135388346,
                    0.706257230683,
                ],
            ),
            (
                CorrelationStructure::Identity,
                [
                    2.093925805991,
                    0.767614285417,
                    0.294784053256,
                    0.046135388346,
                    0.713284838904,
                ],
            ),
        ];
        for (corr, r) in cases {
            let label = format!("{corr:?}");
            let g = gls(&y.view(), &x.view(), corr).unwrap();
            let got = [
                g.coefficients[0],
                g.coefficients[1],
                g.std_errors[0],
                g.std_errors[1],
                g.sigma,
            ];
            for (a, e) in got.iter().zip(r) {
                assert!((a - e).abs() < 1e-8, "{label}: got {got:?}, nlme {r:?}");
            }
        }
    }

    #[test]
    fn test_gls_closed_forms_match_dense_reference() {
        let n = 25;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => (i as f64 * 0.7).sin() * 3.0 + i as f64 * 0.1,
            _ => ((i * 13) % 7) as f64 - 3.0,
        });
        let y = Array1::from_shape_fn(n, |i| {
            1.0 + 0.5 * x[[i, 1]] - 0.3 * x[[i, 2]] + (i as f64 * 1.9).cos() * 0.4
        });
        for corr in [
            CorrelationStructure::AR1 { rho: 0.9 },
            CorrelationStructure::AR1 { rho: 0.3 },
            CorrelationStructure::AR1 { rho: -0.7 },
            CorrelationStructure::CompoundSymmetry { rho: 0.5 },
            CorrelationStructure::CompoundSymmetry { rho: -0.03 },
            CorrelationStructure::Identity,
        ] {
            let omega = dense_correlation_matrix(n, &corr);
            let (b_ref, se_ref, sigma_ref) = dense_gls_reference(&y, &x, &omega);
            let g = gls(&y.view(), &x.view(), corr.clone()).unwrap();
            for j in 0..3 {
                assert!(
                    (g.coefficients[j] - b_ref[j]).abs() < 1e-10
                        && (g.std_errors[j] - se_ref[j]).abs() < 1e-10,
                    "{corr:?}: {:?} vs {b_ref}, {:?} vs {se_ref}",
                    g.coefficients,
                    g.std_errors
                );
            }
            assert!((g.sigma - sigma_ref).abs() < 1e-10, "{corr:?}");
            let known = gls(&y.view(), &x.view(), CorrelationStructure::Known { omega }).unwrap();
            for j in 0..3 {
                assert!(
                    (known.coefficients[j] - g.coefficients[j]).abs() < 1e-10,
                    "Known vs closed form {corr:?}"
                );
            }
        }
    }

    #[test]
    fn test_gls_compound_symmetry_rho_error_names_value_and_bound() {
        // n = 5: rho must lie in (-0.25, 1)
        let y = array![1.0, 2.1, 2.9, 4.1, 5.0];
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
        let e = gls(
            &y.view(),
            &x.view(),
            CorrelationStructure::CompoundSymmetry { rho: -0.3 },
        )
        .unwrap_err();
        match e {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("(-0.250000, 1)")
                    && message.contains("n = 5")
                    && message.contains("got -0.3"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn test_gls_rejects_non_finite_rho() {
        let (y, x) = nlme_xy();
        for corr in [
            CorrelationStructure::AR1 { rho: f64::NAN },
            CorrelationStructure::CompoundSymmetry { rho: f64::NAN },
        ] {
            let e = gls(&y.view(), &x.view(), corr).unwrap_err();
            assert!(matches!(e, EconError::InvalidSpecification { .. }), "{e:?}");
        }
    }

    #[test]
    fn test_gls_whitened_design_respects_alloc_ceiling() {
        // X* is 2000 x 3 x 8 B = 48,000 B, above a 32 KiB ceiling
        let n = 2000;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => (i as f64 * 0.37).sin() * 5.0,
            _ => ((i * 13) % 7) as f64 - 3.0,
        });
        let y = Array1::from_shape_fn(n, |i| 2.0 + 0.5 * x[[i, 1]] + (i as f64 * 2.1).cos());
        for corr in [
            CorrelationStructure::AR1 { rho: 0.5 },
            CorrelationStructure::CompoundSymmetry { rho: 0.3 },
            CorrelationStructure::Identity,
        ] {
            let label = format!("{corr:?}");
            let e =
                resource::with_alloc_ceiling(Some(32 << 10), || gls(&y.view(), &x.view(), corr))
                    .unwrap_err();
            assert!(
                matches!(&e, EconError::ResourceLimit(m) if m.contains("GLS whitened design")),
                "{label}: {e:?}"
            );
        }
    }

    fn cancelled() -> resource::Budget {
        let t = resource::CancelToken::new();
        t.cancel();
        resource::Budget::new(None, t)
    }

    #[test]
    fn test_gls_stops_when_cancelled() {
        let (y, x) = nlme_xy();
        let omega = dense_correlation_matrix(y.len(), &CorrelationStructure::AR1 { rho: 0.5 });
        for corr in [
            CorrelationStructure::AR1 { rho: 0.5 },
            CorrelationStructure::CompoundSymmetry { rho: 0.3 },
            CorrelationStructure::Known { omega },
            CorrelationStructure::Identity,
        ] {
            let label = match &corr {
                CorrelationStructure::Known { .. } => "Known".to_string(),
                c => format!("{c:?}"),
            };
            let e =
                resource::with_budget(cancelled(), || gls(&y.view(), &x.view(), corr)).unwrap_err();
            assert!(matches!(e, EconError::Cancelled(_)), "{label}: {e:?}");
        }
        // The string entry points, including the identity aliases
        let x_flat: Vec<f64> = x.iter().copied().collect();
        for kind in ["identity", "ols", "ar1_auto"] {
            let e = resource::with_budget(cancelled(), || {
                run_gls_raw(y.as_slice().unwrap(), &x_flat, 2, kind, None)
            })
            .unwrap_err();
            assert!(matches!(e, EconError::Cancelled(_)), "{kind}: {e:?}");
        }
    }

    #[test]
    fn test_gls_known_omega_respects_alloc_ceiling() {
        let n = 400;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                (i as f64 * 0.37).sin() * 5.0
            }
        });
        let y = Array1::from_shape_fn(n, |i| 2.0 + 0.5 * x[[i, 1]] + (i as f64 * 2.1).cos());
        let omega = dense_correlation_matrix(n, &CorrelationStructure::AR1 { rho: 0.5 });
        // L is 400*400*8 = 1.28 MB
        let e = crate::resource::with_alloc_ceiling(Some(256 << 10), || {
            gls(&y.view(), &x.view(), CorrelationStructure::Known { omega })
        })
        .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_gls_ar1_large_n_is_linear() {
        // Post-fix guard. Do NOT run against the dense code: it allocates
        // several 320 GB matrices.
        let n = 200_000;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                (i % 1000) as f64 / 100.0
            }
        });
        let y = Array1::from_shape_fn(n, |i| {
            1.0 + 0.8 * x[[i, 1]] + ((i * 7919) % 1009) as f64 / 1009.0
        });
        let g = gls(&y.view(), &x.view(), CorrelationStructure::AR1 { rho: 0.5 }).unwrap();
        assert_eq!(g.n_obs, n);
        assert!((g.coefficients[1] - 0.8).abs() < 0.05);
    }

    #[test]
    fn test_build_ar1_matrix() {
        let omega = dense_correlation_matrix(4, &CorrelationStructure::AR1 { rho: 0.5 });

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

    /// Case `gls_ar1` (nlme::gls REML on the legacy Test 2 DGP; validation/r/cases/regression_legacy.R):
    /// coefficients, SEs, t, p and sigma at nlme's phi.
    #[test]
    fn test_validate_gls_ar1_nlme() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("gls_ar1");
        let n = case.scalar("n_obs") as usize;
        let phi = case.scalar("phi");
        let names = ["(Intercept)", "x"];
        let g = run_gls(&ds, "y", &["x"], true, "ar1", Some(phi)).unwrap();
        Check::new(&case)
            .named("coef", &names, &g.coefficients, Tol::coef(n))
            .named("se", &names, &g.std_errors, Tol::se(n))
            .named("t", &names, &g.t_values, Tol::stat(n))
            .named("p", &names, &g.p_values, Tol::p_value(n))
            .scalar("sigma", g.sigma, Tol::stat(n))
            .scalar("df_resid", g.df_residual as f64, Tol::exact())
            .scalar("n_obs", g.n_obs as f64, Tol::exact())
            .finish();
        // deviation REG-9: log_likelihood, aic, bic (log_lik_reml, aic_reml, bic_reml, log_lik_ml).
        // deviation REG-43: "ar1_auto" estimates rho as the lag-1 correlation of OLS residuals,
        //   not REML/ML (phi, phi_ml).
        assert_eq!(g.n_params, 3);
        assert!(g.correlation.contains("AR(1)") && g.correlation_param == Some(phi));
        let auto = run_gls(&ds, "y", &["x"], true, "ar1_auto", None).unwrap();
        let rho = auto.correlation_param.unwrap();
        assert!(
            rho > 0.0 && rho < 1.0,
            "AR(1) errors with rho = 0.6 give a positive estimate: {rho}"
        );
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
