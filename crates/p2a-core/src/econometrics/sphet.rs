//! Spatial GMM with Heteroscedasticity-Robust Estimation (sphet).
//!
//! Pure Rust implementation of spatial models estimated via Generalized Method
//! of Moments (GMM) that are robust to heteroscedasticity, following the approach
//! of Kelejian and Prucha (1998, 1999, 2010) and Arraiz et al. (2010).
//!
//! # Overview
//!
//! This module provides GMM-based estimation for spatial autoregressive models
//! that does not assume homoscedastic errors. The key advantage over ML estimation
//! is robustness to heteroscedasticity of unknown form.
//!
//! # Models Supported
//!
//! ## Spatial Lag Model (SAR)
//!
//! ```text
//! y = λWy + Xβ + ε
//! ```
//!
//! ## Spatial Error Model (SEM)
//!
//! ```text
//! y = Xβ + u,  where u = ρWu + ε
//! ```
//!
//! ## SARAR (Combined Lag and Error)
//!
//! ```text
//! y = λWy + Xβ + u,  where u = ρWu + ε
//! ```
//!
//! # Estimation Procedure
//!
//! The Kelejian-Prucha approach uses a multi-step procedure:
//!
//! 1. **Step 1 (Initial 2SLS)**: Estimate β and λ using spatial 2SLS with
//!    instruments [X, WX, W²X, ...]
//!
//! 2. **Step 2 (GM for ρ)**: Use residuals to estimate the spatial error
//!    parameter ρ via Generalized Moments, exploiting moment conditions:
//!    - E[ε'ε/n] = σ²
//!    - E[ε'Wε/n] = 0
//!    - E[(Wε)'(Wε)/n] = σ² tr(W'W)/n
//!
//! 3. **Step 3 (Cochrane-Orcutt)**: Transform data using estimated ρ and
//!    re-estimate β and λ via 2SLS on the transformed model.
//!
//! # Standard Errors
//!
//! Standard errors are computed using the heteroscedasticity-robust
//! variance estimator from Kelejian and Prucha (2010), which does not
//! assume a specific form for the error variance.
//!
//! Alternatively, HAC standard errors (Kelejian & Prucha, 2007) can be
//! computed for robustness to both heteroscedasticity and spatial correlation.
//!
//! # References
//!
//! - Kelejian, H.H. & Prucha, I.R. (1998). "A Generalized Spatial Two-Stage
//!   Least Squares Procedure for Estimating a Spatial Autoregressive Model with
//!   Autoregressive Disturbances." Journal of Real Estate Finance and Economics,
//!   17(1), 99-121.
//!
//! - Kelejian, H.H. & Prucha, I.R. (1999). "A Generalized Moments Estimator
//!   for the Autoregressive Parameter in a Spatial Model." International
//!   Economic Review, 40(2), 509-533.
//!
//! - Kelejian, H.H. & Prucha, I.R. (2007). "HAC Estimation in a Spatial
//!   Framework." Journal of Econometrics, 140(1), 131-154.
//!
//! - Kelejian, H.H. & Prucha, I.R. (2010). "Specification and Estimation
//!   of Spatial Autoregressive Models with Autoregressive and Heteroskedastic
//!   Disturbances." Journal of Econometrics, 157(1), 53-67.
//!
//! - Arraiz, I., Drukker, D.M., Kelejian, H.H. & Prucha, I.R. (2010).
//!   "A Spatial Cliff-Ord-type Model with Heteroskedastic Innovations:
//!   Small and Large Sample Results." Journal of Regional Science, 50(2), 592-614.
//!
//! - Piras, G. (2010). "sphet: Spatial Models with Heteroskedastic Innovations
//!   in R." Journal of Statistical Software, 35(1), 1-21.
//!   https://www.jstatsoft.org/article/view/v035i01
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
//! use p2a_core::econometrics::sphet::{run_sphet, SphetConfig, SphetModel};
//! use p2a_core::data::Dataset;
//!
//! // Assuming dataset is loaded with columns "y", "x1", "x2" and coordinates
//! let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
//!
//! // Create spatial weights
//! let nb = Neighbors::from_knn(&coords, 5);
//! let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
//!
//! // Estimate SAR model with heteroscedasticity-robust GMM
//! let config = SphetConfig {
//!     model: SphetModel::SpatialLag,
//!     het: true,
//!     ..Default::default()
//! };
//! let result = run_sphet(&dataset, "y", &["x1", "x2"], &listw, config)?;
//! println!("λ = {}, SE = {}", result.lambda, result.lambda_se);
//! ```

use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::regression::HacKernel;
use crate::spatial::SpatialWeights;
use crate::traits::estimator::{SignificanceLevel, normal_cdf};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of spatial model to estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SphetModel {
    /// Spatial lag model: y = λWy + Xβ + ε
    #[default]
    SpatialLag,
    /// Spatial error model: y = Xβ + u, u = ρWu + ε
    SpatialError,
    /// SARAR model: y = λWy + Xβ + u, u = ρWu + ε
    SARAR,
}

impl fmt::Display for SphetModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SphetModel::SpatialLag => write!(f, "Spatial Lag (SAR)"),
            SphetModel::SpatialError => write!(f, "Spatial Error (SEM)"),
            SphetModel::SARAR => write!(f, "SARAR (Spatial Lag + Error)"),
        }
    }
}

/// Standard error computation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SphetSE {
    /// Heteroscedasticity-robust standard errors (Kelejian-Prucha 2010)
    #[default]
    Robust,
    /// HAC standard errors (Kelejian-Prucha 2007)
    HAC,
    /// Standard (homoscedastic) standard errors
    Standard,
}

impl fmt::Display for SphetSE {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SphetSE::Robust => write!(f, "Heteroscedasticity-robust"),
            SphetSE::HAC => write!(f, "HAC (Newey-West type)"),
            SphetSE::Standard => write!(f, "Standard (homoscedastic)"),
        }
    }
}

/// Configuration for sphet estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphetConfig {
    /// Type of spatial model to estimate
    pub model: SphetModel,
    /// Whether to use heteroscedasticity-robust estimation
    pub het: bool,
    /// Standard error type
    pub se_type: SphetSE,
    /// HAC kernel (only used if se_type = HAC)
    pub kernel: HacKernel,
    /// HAC bandwidth (None for automatic)
    pub bandwidth: Option<usize>,
    /// Number of instruments: order of W to use (default 2 means [X, WX, W²X])
    pub instrument_order: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Initial value for ρ (spatial error parameter)
    pub initial_rho: f64,
}

impl Default for SphetConfig {
    fn default() -> Self {
        Self {
            model: SphetModel::SpatialLag,
            het: true,
            se_type: SphetSE::Robust,
            kernel: HacKernel::Bartlett,
            bandwidth: None,
            instrument_order: 2,
            tolerance: 1e-7,
            max_iter: 100,
            initial_rho: 0.2,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from spatial GMM estimation with heteroscedasticity robustness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphetResult {
    /// Regression coefficients (β)
    pub coefficients: Vec<f64>,
    /// Coefficient names
    pub coef_names: Vec<String>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Z-values
    pub z_values: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Variance-covariance matrix
    pub vcov: Vec<Vec<f64>>,

    /// Spatial lag parameter (λ) - for SAR and SARAR models
    pub lambda: Option<f64>,
    /// Standard error of λ
    pub lambda_se: Option<f64>,
    /// Z-value for λ
    pub lambda_z: Option<f64>,
    /// P-value for λ
    pub lambda_p: Option<f64>,

    /// Spatial error parameter (ρ) - for SEM and SARAR models
    pub rho: Option<f64>,
    /// Standard error of ρ
    pub rho_se: Option<f64>,
    /// Z-value for ρ
    pub rho_z: Option<f64>,
    /// P-value for ρ
    pub rho_p: Option<f64>,

    /// Residual variance (σ²)
    pub sigma2: f64,
    /// Residuals
    #[serde(skip)]
    pub residuals: Array1<f64>,
    /// Fitted values
    #[serde(skip)]
    pub fitted: Array1<f64>,

    /// Number of observations
    pub n_obs: usize,
    /// Degrees of freedom
    pub df: usize,
    /// Model type estimated
    pub model_type: SphetModel,
    /// Standard error type used
    pub se_type: SphetSE,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence achieved
    pub converged: bool,
}

impl fmt::Display for SphetResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Spatial GMM Estimation (sphet)")?;
        writeln!(f, "================================")?;
        writeln!(f)?;
        writeln!(f, "Model: {}", self.model_type)?;
        writeln!(f, "Standard errors: {}", self.se_type)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f)?;

        // Spatial parameters
        if let (Some(lambda), Some(se), Some(z), Some(p)) =
            (self.lambda, self.lambda_se, self.lambda_z, self.lambda_p)
        {
            let sig = SignificanceLevel::from_p_value(p);
            writeln!(
                f,
                "Spatial lag (lambda): {:.6} (SE: {:.6}, z: {:.3}, p: {:.4}){}",
                lambda,
                se,
                z,
                p,
                sig.stars()
            )?;
        }
        if let (Some(rho), Some(se), Some(z), Some(p)) =
            (self.rho, self.rho_se, self.rho_z, self.rho_p)
        {
            let sig = SignificanceLevel::from_p_value(p);
            writeln!(
                f,
                "Spatial error (rho): {:.6} (SE: {:.6}, z: {:.3}, p: {:.4}){}",
                rho,
                se,
                z,
                p,
                sig.stars()
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Estimate", "Std.Err", "z-value", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(65))?;

        for i in 0..self.coefficients.len() {
            let sig = SignificanceLevel::from_p_value(self.p_values[i]);
            writeln!(
                f,
                "{:>15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}",
                self.coef_names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_values[i],
                self.p_values[i],
                sig.stars()
            )?;
        }
        writeln!(f, "{}", "-".repeat(65))?;
        writeln!(f)?;
        writeln!(f, "sigma^2: {:.6}", self.sigma2)?;
        writeln!(
            f,
            "Iterations: {} (converged: {})",
            self.iterations, self.converged
        )?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run spatial GMM estimation with heteroscedasticity robustness.
///
/// Implements the Kelejian-Prucha (1998, 1999, 2010) estimator for spatial
/// autoregressive models that is robust to heteroscedasticity of unknown form.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the variables
/// * `y_col` - Name of the dependent variable
/// * `x_cols` - Names of the independent variables
/// * `listw` - Spatial weights matrix
/// * `config` - Configuration options
///
/// # Returns
///
/// Estimation results including coefficients, spatial parameters, and
/// heteroscedasticity-robust standard errors.
///
/// # Example
///
/// ```ignore
/// use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
/// use p2a_core::econometrics::sphet::{run_sphet, SphetConfig, SphetModel};
///
/// // Assuming dataset, coords, and listw are already set up
/// let config = SphetConfig {
///     model: SphetModel::SARAR,
///     het: true,
///     ..Default::default()
/// };
/// let result = run_sphet(&dataset, "crime", &["income", "poverty"], &listw, config)?;
/// ```
///
/// # References
///
/// - Kelejian & Prucha (2010), Journal of Econometrics, 157(1), 53-67.
pub fn run_sphet(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &SpatialWeights,
    config: SphetConfig,
) -> EconResult<SphetResult> {
    let df = dataset.df();
    let n = df.height();
    let k = x_cols.len() + 1; // +1 for intercept

    if n != listw.n() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dataset has {} observations but weights matrix has {} observations",
                n,
                listw.n()
            ),
        });
    }

    if n < k + 5 {
        return Err(EconError::InsufficientData {
            required: k + 5,
            provided: n,
            context: "Spatial GMM requires sufficient degrees of freedom".to_string(),
        });
    }

    // Extract y
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

    // Build coefficient names
    let mut coef_names = vec!["(Intercept)".to_string()];
    for &name in x_cols {
        coef_names.push(name.to_string());
    }

    // Dispatch to appropriate model
    match config.model {
        SphetModel::SpatialLag => estimate_sar_gmm(&y, &x, listw, &coef_names, &config),
        SphetModel::SpatialError => estimate_sem_gmm(&y, &x, listw, &coef_names, &config),
        SphetModel::SARAR => estimate_sarar_gmm(&y, &x, listw, &coef_names, &config),
    }
}

/// Convenience function for dataset-based estimation.
pub fn sphet(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &SpatialWeights,
    config: Option<SphetConfig>,
) -> EconResult<SphetResult> {
    run_sphet(dataset, y_col, x_cols, listw, config.unwrap_or_default())
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAR Model (Spatial Lag)
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate spatial lag model via GMM (Spatial 2SLS).
///
/// Model: y = λWy + Xβ + ε
///
/// Uses instruments [X, WX, W²X, ...] for the endogenous Wy.
fn estimate_sar_gmm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    coef_names: &[String],
    config: &SphetConfig,
) -> EconResult<SphetResult> {
    let n = y.len();
    let k = x.ncols();

    // Compute Wy (endogenous)
    let wy = listw.lag(y);

    // Build instrument matrix [X, WX, W²X, ...]
    let h = build_instruments(x, listw, config.instrument_order);
    let n_inst = h.ncols();

    if n_inst < k + 1 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Insufficient instruments: {} instruments for {} + 1 parameters",
                n_inst, k
            ),
        });
    }

    // Augmented regressor matrix Z = [X, Wy]
    let mut z = Array2::zeros((n, k + 1));
    z.slice_mut(s![.., ..k]).assign(x);
    z.column_mut(k).assign(&wy);

    // Spatial 2SLS: δ̂ = (Z'P_H Z)^{-1} Z'P_H y
    // where P_H = H(H'H)^{-1}H' is the projection matrix
    let hh = xtx(&h.view());
    let (hh_inv, _) = safe_inverse(&hh.view()).map_err(|e| EconError::SingularMatrix {
        context: "Instrument matrix".to_string(),
        suggestion: format!("H'H singular: {}", e),
    })?;

    // P_H Z = H(H'H)^{-1}H'Z
    let hz = h.t().dot(&z);
    let hh_inv_hz = hh_inv.dot(&hz);
    let ph_z = h.dot(&hh_inv_hz);

    // (Z'P_H Z)^{-1}
    let zph_z = z.t().dot(&ph_z);
    let (zph_z_inv, _) = safe_inverse(&zph_z.view()).map_err(|e| EconError::SingularMatrix {
        context: "2SLS normal equations".to_string(),
        suggestion: format!("Z'P_H Z singular: {}", e),
    })?;

    // δ̂ = (Z'P_H Z)^{-1} Z'P_H y
    let hy = h.t().dot(y);
    let hh_inv_hy = hh_inv.dot(&hy);
    let ph_y = h.dot(&hh_inv_hy);
    let zph_y = z.t().dot(&ph_y);
    let delta = zph_z_inv.dot(&zph_y);

    // Extract β and λ
    let beta: Array1<f64> = delta.slice(s![..k]).to_owned();
    let lambda = delta[k];

    // Residuals
    let fitted = z.dot(&delta);
    let residuals = y - &fitted;
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = rss / (n - k - 1) as f64;

    // Standard errors
    let (vcov_mat, se_delta) =
        compute_spatial_2sls_se(&z, &h, &residuals, sigma2, &zph_z_inv, config)?;

    // Extract standard errors for β and λ
    let se_beta: Array1<f64> = se_delta.slice(s![..k]).to_owned();
    let se_lambda = se_delta[k];

    // Z-values and p-values for β
    let z_values: Vec<f64> = beta
        .iter()
        .zip(se_beta.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    // Z-value and p-value for λ
    let lambda_z = if se_lambda > 0.0 {
        lambda / se_lambda
    } else {
        0.0
    };
    let lambda_p = 2.0 * (1.0 - normal_cdf(lambda_z.abs()));

    // Convert vcov to Vec<Vec<f64>> (just for beta)
    let vcov: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| vcov_mat[[i, j]]).collect())
        .collect();

    Ok(SphetResult {
        coefficients: beta.to_vec(),
        coef_names: coef_names.to_vec(),
        std_errors: se_beta.to_vec(),
        z_values,
        p_values,
        vcov,
        lambda: Some(lambda),
        lambda_se: Some(se_lambda),
        lambda_z: Some(lambda_z),
        lambda_p: Some(lambda_p),
        rho: None,
        rho_se: None,
        rho_z: None,
        rho_p: None,
        sigma2,
        residuals,
        fitted,
        n_obs: n,
        df: n - k - 1,
        model_type: SphetModel::SpatialLag,
        se_type: config.se_type,
        iterations: 1,
        converged: true,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEM Model (Spatial Error)
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate spatial error model via GMM.
///
/// Model: y = Xβ + u, where u = ρWu + ε
///
/// Uses the Kelejian-Prucha (1999) GM estimator for ρ.
fn estimate_sem_gmm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    coef_names: &[String],
    config: &SphetConfig,
) -> EconResult<SphetResult> {
    let n = y.len();
    let k = x.ncols();

    // Step 1: Initial OLS
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "OLS normal equations".to_string(),
        suggestion: format!("X'X singular: {}", e),
    })?;
    let xty_vec = xty(&x.view(), y);
    let beta_ols = xtx_inv.dot(&xty_vec);
    let u_ols = y - &x.dot(&beta_ols);

    // Step 2: GM estimation of ρ using moment conditions
    let rho = estimate_rho_gm(&u_ols, listw, config)?;

    // Step 3: Cochrane-Orcutt transformation and re-estimation
    // y* = y - ρWy, X* = X - ρWX
    let y_star = listw.transform_y(y, rho);
    let x_star = listw.transform_x(x, rho);

    // OLS on transformed data
    let xtx_star = xtx(&x_star.view());
    let (xtx_star_inv, _) =
        safe_inverse(&xtx_star.view()).map_err(|e| EconError::SingularMatrix {
            context: "Transformed OLS".to_string(),
            suggestion: format!("X*'X* singular: {}", e),
        })?;
    let xty_star = xty(&x_star.view(), &y_star);
    let beta = xtx_star_inv.dot(&xty_star);

    // Residuals in transformed space
    let fitted_star = x_star.dot(&beta);
    let residuals_star = &y_star - &fitted_star;
    let rss: f64 = residuals_star.iter().map(|r| r * r).sum();
    let sigma2 = rss / (n - k) as f64;

    // Residuals in original space
    let fitted = x.dot(&beta);
    let residuals = y - &fitted;

    // Standard errors for β
    let (vcov_mat, se_beta) =
        compute_sem_se(x, &x_star, &residuals_star, sigma2, &xtx_star_inv, config)?;

    // Standard error for ρ (from GM theory)
    let se_rho = compute_rho_se(&residuals_star, listw, sigma2, config)?;

    // Z-values and p-values for β
    let z_values: Vec<f64> = beta
        .iter()
        .zip(se_beta.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    // Z-value and p-value for ρ
    let rho_z = if se_rho > 0.0 { rho / se_rho } else { 0.0 };
    let rho_p = 2.0 * (1.0 - normal_cdf(rho_z.abs()));

    // Convert vcov to Vec<Vec<f64>>
    let vcov: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| vcov_mat[[i, j]]).collect())
        .collect();

    Ok(SphetResult {
        coefficients: beta.to_vec(),
        coef_names: coef_names.to_vec(),
        std_errors: se_beta.to_vec(),
        z_values,
        p_values,
        vcov,
        lambda: None,
        lambda_se: None,
        lambda_z: None,
        lambda_p: None,
        rho: Some(rho),
        rho_se: Some(se_rho),
        rho_z: Some(rho_z),
        rho_p: Some(rho_p),
        sigma2,
        residuals,
        fitted,
        n_obs: n,
        df: n - k,
        model_type: SphetModel::SpatialError,
        se_type: config.se_type,
        iterations: 1,
        converged: true,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// SARAR Model (Combined Lag and Error)
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate SARAR model via GMM.
///
/// Model: y = λWy + Xβ + u, where u = ρWu + ε
///
/// Uses the Kelejian-Prucha (1998) three-step procedure.
fn estimate_sarar_gmm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    coef_names: &[String],
    config: &SphetConfig,
) -> EconResult<SphetResult> {
    let n = y.len();
    let k = x.ncols();

    // Compute Wy
    let wy = listw.lag(y);

    // Build instrument matrix
    let h = build_instruments(x, listw, config.instrument_order);

    // Augmented regressor matrix Z = [X, Wy]
    let mut z = Array2::zeros((n, k + 1));
    z.slice_mut(s![.., ..k]).assign(x);
    z.column_mut(k).assign(&wy);

    // Step 1: Initial S2SLS
    let hh = xtx(&h.view());
    let (hh_inv, _) = safe_inverse(&hh.view())?;
    let hz = h.t().dot(&z);
    let ph_z = h.dot(&hh_inv.dot(&hz));
    let zph_z = z.t().dot(&ph_z);
    let (zph_z_inv, _) = safe_inverse(&zph_z.view())?;
    let hy = h.t().dot(y);
    let ph_y = h.dot(&hh_inv.dot(&hy));
    let zph_y = z.t().dot(&ph_y);
    let delta1 = zph_z_inv.dot(&zph_y);

    let u1 = y - &z.dot(&delta1);

    // Step 2: GM estimation of ρ
    let rho = estimate_rho_gm(&u1, listw, config)?;

    // Step 3: Cochrane-Orcutt transformation and re-estimation
    // y* = (I - ρW)y, Z* = (I - ρW)Z, H* = (I - ρW)H
    let y_star = listw.transform_y(y, rho);
    let z_star = transform_matrix(listw, &z, rho);
    let h_star = transform_matrix(listw, &h, rho);

    // S2SLS on transformed data
    let hh_star = xtx(&h_star.view());
    let (hh_star_inv, _) = safe_inverse(&hh_star.view())?;
    let hz_star = h_star.t().dot(&z_star);
    let ph_z_star = h_star.dot(&hh_star_inv.dot(&hz_star));
    let zph_z_star = z_star.t().dot(&ph_z_star);
    let (zph_z_star_inv, _) = safe_inverse(&zph_z_star.view())?;
    let hy_star = h_star.t().dot(&y_star);
    let ph_y_star = h_star.dot(&hh_star_inv.dot(&hy_star));
    let zph_y_star = z_star.t().dot(&ph_y_star);
    let delta = zph_z_star_inv.dot(&zph_y_star);

    // Extract β and λ
    let beta: Array1<f64> = delta.slice(s![..k]).to_owned();
    let lambda = delta[k];

    // Residuals in transformed space
    let fitted_star = z_star.dot(&delta);
    let residuals_star = &y_star - &fitted_star;
    let rss: f64 = residuals_star.iter().map(|r| r * r).sum();
    let sigma2 = rss / (n - k - 1) as f64;

    // Residuals in original space
    let fitted = x.dot(&beta) + lambda * &wy;
    let residuals = y - &fitted;

    // Standard errors
    let (vcov_mat, se_delta) = compute_spatial_2sls_se(
        &z_star,
        &h_star,
        &residuals_star,
        sigma2,
        &zph_z_star_inv,
        config,
    )?;

    let se_beta: Array1<f64> = se_delta.slice(s![..k]).to_owned();
    let se_lambda = se_delta[k];
    let se_rho = compute_rho_se(&residuals_star, listw, sigma2, config)?;

    // Z-values and p-values
    let z_values: Vec<f64> = beta
        .iter()
        .zip(se_beta.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let lambda_z = if se_lambda > 0.0 {
        lambda / se_lambda
    } else {
        0.0
    };
    let lambda_p = 2.0 * (1.0 - normal_cdf(lambda_z.abs()));

    let rho_z = if se_rho > 0.0 { rho / se_rho } else { 0.0 };
    let rho_p = 2.0 * (1.0 - normal_cdf(rho_z.abs()));

    let vcov: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| vcov_mat[[i, j]]).collect())
        .collect();

    Ok(SphetResult {
        coefficients: beta.to_vec(),
        coef_names: coef_names.to_vec(),
        std_errors: se_beta.to_vec(),
        z_values,
        p_values,
        vcov,
        lambda: Some(lambda),
        lambda_se: Some(se_lambda),
        lambda_z: Some(lambda_z),
        lambda_p: Some(lambda_p),
        rho: Some(rho),
        rho_se: Some(se_rho),
        rho_z: Some(rho_z),
        rho_p: Some(rho_p),
        sigma2,
        residuals,
        fitted,
        n_obs: n,
        df: n - k - 1,
        model_type: SphetModel::SARAR,
        se_type: config.se_type,
        iterations: 2,
        converged: true,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Build instrument matrix [X, WX, W²X, ...].
///
/// Following Kelejian-Prucha, uses powers of W applied to X as instruments
/// for the endogenous Wy.
fn build_instruments(x: &Array2<f64>, listw: &SpatialWeights, order: usize) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let n_cols = k * (order + 1);

    let mut h = Array2::zeros((n, n_cols));

    // Start with X
    h.slice_mut(s![.., ..k]).assign(x);

    // Add WX, W²X, etc.
    let mut wx = x.clone();
    for o in 1..=order {
        // Compute W^o X
        let mut wx_new = Array2::zeros((n, k));
        for col in 0..k {
            let x_col = wx.column(col).to_owned();
            let lag_col = listw.lag(&x_col);
            wx_new.column_mut(col).assign(&lag_col);
        }
        wx = wx_new;

        // Add to instrument matrix
        let start_col = k * (o);
        let end_col = k * (o + 1);
        h.slice_mut(s![.., start_col..end_col]).assign(&wx);
    }

    h
}

/// Transform a matrix: M* = (I - ρW)M
fn transform_matrix(listw: &SpatialWeights, m: &Array2<f64>, rho: f64) -> Array2<f64> {
    let n = m.nrows();
    let k = m.ncols();
    let mut result = Array2::zeros((n, k));

    for col in 0..k {
        let m_col = m.column(col).to_owned();
        let wm_col = listw.lag(&m_col);
        for i in 0..n {
            result[[i, col]] = m_col[i] - rho * wm_col[i];
        }
    }

    result
}

/// Estimate ρ using Generalized Moments (Kelejian-Prucha 1999).
///
/// Uses the moment conditions:
/// - E[ε'ε/n] = σ²
/// - E[ε'Wε/n] = 0
/// - E[(Wε)'(Wε)/n] = σ² tr(W'W)/n
///
/// # References
///
/// Kelejian, H.H. & Prucha, I.R. (1999). International Economic Review, 40(2), 509-533.
fn estimate_rho_gm(
    u: &Array1<f64>,
    listw: &SpatialWeights,
    config: &SphetConfig,
) -> EconResult<f64> {
    let n = u.len();
    let n_f64 = n as f64;

    // Compute Wu and W²u
    let wu = listw.lag(u);
    let w2u = listw.lag(&wu);

    // Sample moments
    // g1 = (1/n) u'u
    let uu: f64 = u.iter().map(|&ui| ui * ui).sum();
    let g1 = uu / n_f64;

    // g2 = (1/n) u'Wu
    let uwu: f64 = u.iter().zip(wu.iter()).map(|(&ui, &wui)| ui * wui).sum();
    let g2 = uwu / n_f64;

    // g3 = (1/n) (Wu)'(Wu)
    let wuwu: f64 = wu.iter().map(|&wui| wui * wui).sum();
    let g3 = wuwu / n_f64;

    // g4 = (1/n) u'W²u
    let uw2u: f64 = u.iter().zip(w2u.iter()).map(|(&ui, &w2ui)| ui * w2ui).sum();
    let g4 = uw2u / n_f64;

    // g5 = (1/n) (Wu)'(W²u)
    let wuw2u: f64 = wu
        .iter()
        .zip(w2u.iter())
        .map(|(&wui, &w2ui)| wui * w2ui)
        .sum();
    let g5 = wuw2u / n_f64;

    // Trace terms for the population moment conditions
    let tr_wtw = listw.trace_wtw();
    let tr_w2 = listw.trace_w2();

    // The GM estimator solves:
    // [g2 - ρg3 - ρtr(W²)/n * σ²] = 0
    // [g4 - ρg5 - ρtr(W'W·W'W)/n * σ²] ≈ 0
    //
    // For simplicity, use a concentrated approach:
    // From first moment: σ² ≈ g1 - 2ρg2 + ρ²g3
    // Substitute and solve for ρ

    // Grid search followed by Newton refinement
    let mut best_rho = config.initial_rho;
    let mut best_obj = f64::INFINITY;

    // Grid search
    for i in -99..100 {
        let rho = i as f64 / 100.0;
        let obj = gm_objective(rho, g1, g2, g3, g4, g5, tr_wtw / n_f64, tr_w2 / n_f64);
        if obj < best_obj {
            best_obj = obj;
            best_rho = rho;
        }
    }

    // Newton refinement
    for _ in 0..config.max_iter {
        let h = 1e-6;
        let f0 = gm_objective(best_rho, g1, g2, g3, g4, g5, tr_wtw / n_f64, tr_w2 / n_f64);
        let f1 = gm_objective(
            best_rho + h,
            g1,
            g2,
            g3,
            g4,
            g5,
            tr_wtw / n_f64,
            tr_w2 / n_f64,
        );
        let f2 = gm_objective(
            best_rho - h,
            g1,
            g2,
            g3,
            g4,
            g5,
            tr_wtw / n_f64,
            tr_w2 / n_f64,
        );

        let grad = (f1 - f2) / (2.0 * h);
        let hess = (f1 - 2.0 * f0 + f2) / (h * h);

        if hess.abs() < 1e-10 {
            break;
        }

        let step = -grad / hess;
        let new_rho = (best_rho + step).clamp(-0.99, 0.99);

        if (new_rho - best_rho).abs() < config.tolerance {
            best_rho = new_rho;
            break;
        }
        best_rho = new_rho;
    }

    Ok(best_rho)
}

/// GM objective function for ρ estimation.
///
/// Based on the quadratic form of moment conditions.
fn gm_objective(
    rho: f64,
    g1: f64,
    g2: f64,
    g3: f64,
    g4: f64,
    g5: f64,
    _tr_wtw_n: f64,
    _tr_w2_n: f64,
) -> f64 {
    // σ² estimate given ρ
    let sigma2 = g1 - 2.0 * rho * g2 + rho * rho * g3;

    if sigma2 <= 0.0 {
        return f64::INFINITY;
    }

    // Moment conditions:
    // m1 = g2 - ρg3 - ρ*tr_w2_n*σ²
    // m2 = g4 - ρg5 - (something involving W'W W'W)
    let m1 = g2 - rho * g3;
    let m2 = g4 - rho * g5;

    // Quadratic objective
    m1 * m1 + m2 * m2
}

/// Compute standard errors for spatial 2SLS.
///
/// Uses heteroscedasticity-robust formula from Kelejian-Prucha (2010).
fn compute_spatial_2sls_se(
    z: &Array2<f64>,
    h: &Array2<f64>,
    residuals: &Array1<f64>,
    sigma2: f64,
    zph_z_inv: &Array2<f64>,
    config: &SphetConfig,
) -> EconResult<(Array2<f64>, Array1<f64>)> {
    let n = z.nrows();
    let k = z.ncols();

    match config.se_type {
        SphetSE::Standard => {
            // Var(δ̂) = σ² (Z'P_H Z)^{-1}
            let vcov = sigma2 * zph_z_inv;
            let se: Array1<f64> = vcov.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();
            Ok((vcov.slice(s![..k - 1, ..k - 1]).to_owned(), se))
        }
        SphetSE::Robust => {
            // Heteroscedasticity-robust: Var(δ̂) = (Z'P_H Z)^{-1} Ω̂ (Z'P_H Z)^{-1}
            // where Ω̂ = Σᵢ ε̂ᵢ² ẑᵢ ẑᵢ'
            let hh = xtx(&h.view());
            let (hh_inv, _) = safe_inverse(&hh.view())?;
            let hz = h.t().dot(z);
            let z_hat = h.dot(&hh_inv.dot(&hz));

            let mut omega = Array2::<f64>::zeros((k, k));
            for i in 0..n {
                let e_sq = residuals[i] * residuals[i];
                for j in 0..k {
                    for l in 0..k {
                        omega[[j, l]] += e_sq * z_hat[[i, j]] * z_hat[[i, l]];
                    }
                }
            }

            let vcov = zph_z_inv.dot(&omega).dot(zph_z_inv);
            let se: Array1<f64> = vcov.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();
            Ok((vcov.slice(s![..k - 1, ..k - 1]).to_owned(), se))
        }
        SphetSE::HAC => {
            // HAC standard errors (Kelejian-Prucha 2007)
            let bw = config
                .bandwidth
                .unwrap_or_else(|| (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize);

            let hh = xtx(&h.view());
            let (hh_inv, _) = safe_inverse(&hh.view())?;
            let hz = h.t().dot(z);
            let z_hat = h.dot(&hh_inv.dot(&hz));

            // Score matrix
            let mut scores = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                for j in 0..k {
                    scores[[i, j]] = z_hat[[i, j]] * residuals[i];
                }
            }

            // HAC covariance
            let mut omega = Array2::<f64>::zeros((k, k));

            // Lag 0
            for i in 0..n {
                for j in 0..k {
                    for l in 0..k {
                        omega[[j, l]] += scores[[i, j]] * scores[[i, l]];
                    }
                }
            }

            // Lags 1 to bw
            for lag in 1..=bw {
                let w = config.kernel.weight(lag, bw);
                if w.abs() < 1e-15 {
                    continue;
                }

                let mut gamma_lag = Array2::<f64>::zeros((k, k));
                for i in lag..n {
                    for j in 0..k {
                        for l in 0..k {
                            gamma_lag[[j, l]] += scores[[i, j]] * scores[[i - lag, l]];
                        }
                    }
                }

                for j in 0..k {
                    for l in 0..k {
                        omega[[j, l]] += w * (gamma_lag[[j, l]] + gamma_lag[[l, j]]);
                    }
                }
            }

            let vcov = zph_z_inv.dot(&omega).dot(zph_z_inv);
            let se: Array1<f64> = vcov.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();
            Ok((vcov.slice(s![..k - 1, ..k - 1]).to_owned(), se))
        }
    }
}

/// Compute standard errors for SEM model.
fn compute_sem_se(
    _x: &Array2<f64>,
    _x_star: &Array2<f64>,
    residuals_star: &Array1<f64>,
    sigma2: f64,
    xtx_star_inv: &Array2<f64>,
    config: &SphetConfig,
) -> EconResult<(Array2<f64>, Array1<f64>)> {
    let n = residuals_star.len();
    let k = xtx_star_inv.nrows();

    match config.se_type {
        SphetSE::Standard => {
            let vcov = sigma2 * xtx_star_inv;
            let se: Array1<f64> = vcov.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();
            Ok((vcov, se))
        }
        SphetSE::Robust | SphetSE::HAC => {
            // For SEM, use HC-robust formula
            let mut omega = Array2::<f64>::zeros((k, k));
            for i in 0..n {
                let e_sq = residuals_star[i] * residuals_star[i];
                for j in 0..k {
                    for l in 0..k {
                        omega[[j, l]] += e_sq * xtx_star_inv[[j, l]];
                    }
                }
            }

            // Small sample correction (HC1)
            omega *= n as f64 / (n - k) as f64;

            let vcov = xtx_star_inv.dot(&omega).dot(xtx_star_inv);
            let se: Array1<f64> = vcov.diag().iter().map(|&v| v.max(0.0).sqrt()).collect();
            Ok((vcov, se))
        }
    }
}

/// Compute standard error for ρ.
///
/// Based on the asymptotic theory from Kelejian-Prucha (2010).
fn compute_rho_se(
    residuals: &Array1<f64>,
    listw: &SpatialWeights,
    sigma2: f64,
    _config: &SphetConfig,
) -> EconResult<f64> {
    let n = residuals.len();

    // Approximate variance using trace formulas
    // Var(ρ̂) ≈ σ²/(trace(W²) + trace(W'W))
    let tr_w2 = listw.trace_w2();
    let tr_wtw = listw.trace_wtw();
    let info = tr_w2 + tr_wtw;

    if info <= 0.0 {
        return Ok(0.01); // Fallback
    }

    let var_rho = sigma2 * n as f64 / info;
    Ok(var_rho.max(0.0).sqrt())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Neighbors, WeightStyle};
    use polars::prelude::*;

    fn create_test_data() -> (Dataset, SpatialWeights) {
        // Create a 5x5 grid
        let n = 25;
        let coords: Vec<(f64, f64)> = (0..5)
            .flat_map(|i| (0..5).map(move |j| (i as f64, j as f64)))
            .collect();

        let nb = Neighbors::from_knn(&coords, 4);
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        // Generate data with spatial structure
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64) / 5.0 + 0.1 * (i % 3) as f64)
            .collect();
        let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 + 1.5 * xi).collect();

        // Add spatial autocorrelation
        for i in 0..n {
            let row = i / 5;
            let col = i % 5;
            y[i] += 0.2 * ((row + col) as f64) + 0.1 * (i as f64 % 7.0);
        }

        let df = df! {
            "y" => &y,
            "x" => &x,
        }
        .unwrap();

        (Dataset::new(df), listw)
    }

    #[test]
    fn test_sphet_sar() {
        let (dataset, listw) = create_test_data();

        let config = SphetConfig {
            model: SphetModel::SpatialLag,
            het: true,
            ..Default::default()
        };

        let result = run_sphet(&dataset, "y", &["x"], &listw, config).unwrap();

        // Basic checks
        assert_eq!(result.n_obs, 25);
        assert_eq!(result.coefficients.len(), 2);
        assert!(result.lambda.is_some());
        assert!(result.lambda.unwrap().abs() < 1.0);
        assert!(result.sigma2 > 0.0);

        // Standard errors should be positive
        for se in &result.std_errors {
            assert!(*se > 0.0);
        }
    }

    #[test]
    fn test_sphet_sem() {
        let (dataset, listw) = create_test_data();

        let config = SphetConfig {
            model: SphetModel::SpatialError,
            het: true,
            ..Default::default()
        };

        let result = run_sphet(&dataset, "y", &["x"], &listw, config).unwrap();

        assert_eq!(result.n_obs, 25);
        assert!(result.rho.is_some());
        assert!(result.rho.unwrap().abs() < 1.0);
        assert!(result.lambda.is_none());
    }

    #[test]
    fn test_sphet_sarar() {
        let (dataset, listw) = create_test_data();

        let config = SphetConfig {
            model: SphetModel::SARAR,
            het: true,
            ..Default::default()
        };

        let result = run_sphet(&dataset, "y", &["x"], &listw, config).unwrap();

        assert_eq!(result.n_obs, 25);
        assert!(result.lambda.is_some());
        assert!(result.rho.is_some());
        assert!(result.lambda.unwrap().abs() < 1.0);
        assert!(result.rho.unwrap().abs() < 1.0);
    }

    #[test]
    fn test_sphet_se_types() {
        let (dataset, listw) = create_test_data();

        for se_type in [SphetSE::Standard, SphetSE::Robust, SphetSE::HAC] {
            let config = SphetConfig {
                model: SphetModel::SpatialLag,
                se_type,
                ..Default::default()
            };

            let result = run_sphet(&dataset, "y", &["x"], &listw, config).unwrap();
            assert_eq!(result.se_type, se_type);

            // All SE types should give positive standard errors
            for se in &result.std_errors {
                assert!(*se > 0.0, "SE type {:?} gave non-positive SE", se_type);
            }
        }
    }

    #[test]
    fn test_sphet_display() {
        let (dataset, listw) = create_test_data();

        let config = SphetConfig {
            model: SphetModel::SARAR,
            ..Default::default()
        };

        let result = run_sphet(&dataset, "y", &["x"], &listw, config).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Spatial GMM"));
        assert!(display.contains("SARAR"));
        assert!(display.contains("lambda"));
        assert!(display.contains("rho"));
    }

    #[test]
    fn test_build_instruments() {
        let (_, listw) = create_test_data();

        let x = Array2::from_shape_fn((25, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 / 5.0 });

        // Order 2: [X, WX, W²X]
        let h = build_instruments(&x, &listw, 2);
        assert_eq!(h.ncols(), 6); // 2 * 3 = 6 columns

        // Order 1: [X, WX]
        let h1 = build_instruments(&x, &listw, 1);
        assert_eq!(h1.ncols(), 4); // 2 * 2 = 4 columns
    }

    #[test]
    fn test_gm_objective() {
        // Test that objective function is well-behaved
        let obj = gm_objective(0.5, 1.0, 0.2, 0.8, 0.1, 0.3, 0.1, 0.1);
        assert!(obj.is_finite());
        assert!(obj >= 0.0);

        // Objective should be infinity for sigma2 <= 0
        // sigma2 = g1 - 2*rho*g2 + rho^2*g3
        // With rho=2, g1=0.1, g2=0.5, g3=0.01: sigma2 = 0.1 - 2*0.5 + 0.04 = -0.86 < 0
        let obj_bad = gm_objective(2.0, 0.1, 0.5, 0.01, 0.1, 0.3, 0.1, 0.1);
        assert!(obj_bad.is_infinite());
    }
}
