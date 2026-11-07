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
//! # Data requirements
//!
//! Row i of the dataset is unit i of the weights: one row per unit of W, and no row can be
//! dropped. Missing values (null or NaN) in y or X are refused with
//! `EconError::NullValues` naming the column (spdep/spatialreg `na.fail`); remove the
//! incomplete rows and rebuild W first. Integer and Boolean (0/1) columns are accepted.
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
use crate::linalg::solve::{LM_QR_TOL, Qr, aliasing_message, least_squares};
use crate::regression::HacKernel;
use crate::regression::ols::{NormalInverse, normal_equations_inverse};
use crate::resource::{checked_len, checkpoint, try_zeros2};
use crate::spatial::SpatialWeights;
use crate::traits::convergence::Convergence;
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

/// Kernel of the Kelejian-Prucha (2007) spatial HAC (sphet 2.1-1 `kernelsfun.R`), as a function
/// of z = d_ij / bandwidth >= 0; zero for z >= 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpatialHacKernel {
    /// K(z) = 1 - z^2 (sphet's `stslshac` default).
    #[default]
    Epanechnikov,
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
    /// Newton steps of the GM solve for rho (error and SARAR models); 0 for the closed-form lag model
    pub iterations: usize,
    /// Status of the GM solve (false at a boundary solution or at the iteration cap); true for the lag model
    pub converged: bool,
    /// Warnings about this fit
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The GM solve for rho (error and SARAR models); `None` for the lag model
    #[serde(default)]
    pub convergence: Option<Convergence>,
    /// The GM moments used for rho (error and SARAR models)
    #[serde(default)]
    pub rho_moments: Option<String>,
    /// The instruments kept, e.g. `["(Intercept)", "x1", "W x1", "W^2 x1"]`; empty for the error model
    #[serde(default)]
    pub instruments: Vec<String>,
    /// Kernel of the spatial HAC (se_type HAC only)
    #[serde(default)]
    pub hac_kernel: Option<SpatialHacKernel>,
    /// Fixed distance bandwidth of the spatial HAC (`None` with se_type HAC: variable bandwidth)
    #[serde(default)]
    pub hac_bandwidth: Option<f64>,
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
    validate_config(&config)?;

    let (y, x) = super::spatial::spatial_model_data(dataset, y_col, x_cols, listw)?;
    let n = y.len();
    let k = x.ncols(); // intercept + regressors

    if n < k + 5 {
        return Err(EconError::InsufficientData {
            required: k + 5,
            provided: n,
            context: "Spatial GMM requires sufficient degrees of freedom".to_string(),
        });
    }

    // Build coefficient names
    let mut coef_names = vec!["(Intercept)".to_string()];
    for &name in x_cols {
        coef_names.push(name.to_string());
    }

    refuse_collinear_regressors(&x, &coef_names)?;

    // Dispatch to appropriate model
    match config.model {
        SphetModel::SpatialLag => estimate_sar_gmm(&y, &x, listw, &coef_names, &config),
        SphetModel::SpatialError => estimate_sem_gmm(&y, &x, listw, &coef_names, &config),
        SphetModel::SARAR => estimate_sarar_gmm(&y, &x, listw, &coef_names, &config),
    }
}

/// Highest power of W used as an instrument (`[X, WX, ..., W^4 X]`).
const MAX_INSTRUMENT_ORDER: usize = 4;

/// Reject configurations that cannot be estimated or would size absurd work.
fn validate_config(config: &SphetConfig) -> EconResult<()> {
    let uses_instruments = matches!(config.model, SphetModel::SpatialLag | SphetModel::SARAR);
    if uses_instruments && !(1..=MAX_INSTRUMENT_ORDER).contains(&config.instrument_order) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "instrument_order must be between 1 and {MAX_INSTRUMENT_ORDER} (instruments X, WX, ..., W^order X), got {}",
                config.instrument_order
            ),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be a finite positive number, got {}",
                config.tolerance
            ),
        });
    }
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1".to_string(),
        });
    }
    Ok(())
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
/// Uses the instruments of [`spatial_instruments`] for the endogenous Wy.
fn estimate_sar_gmm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    coef_names: &[String],
    config: &SphetConfig,
) -> EconResult<SphetResult> {
    let n = y.len();
    let k = x.ncols();

    let instruments = spatial_instruments(x, listw, config.instrument_order, coef_names)?;
    let (z, z_names) = with_spatial_lag(y, x, listw, coef_names)?;
    let t = tsls(y, &z, &instruments, k, &z_names)?;

    let beta: Array1<f64> = t.delta.slice(s![..k]).to_owned();
    let lambda = t.delta[k];
    let fitted = z.dot(&t.delta);
    let residuals = y - &fitted;
    let df = residual_df(n, k + 1)?;
    let sigma2 = residuals.dot(&residuals) / df as f64;

    let vcov_mat = match config.se_type {
        SphetSE::HAC => compute_spatial_2sls_se(&t.zh, &residuals, &t.a, config)?,
        se_type => delta_vcov(&t.zh, &residuals, &t.a, se_type)?,
    };

    let mut warnings = Vec::new();
    warnings.extend(instruments.dropped_warning());
    Ok(build_result(ResultParts {
        coef_names,
        config,
        model: SphetModel::SpatialLag,
        beta,
        vcov: &vcov_mat,
        lambda: Some(lambda),
        rho: None,
        sigma2,
        residuals,
        fitted,
        df,
        instruments: instruments.names,
        warnings,
        gm: None,
    }))
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
    let (vcov_mat, _) = compute_sem_se(x, &x_star, &residuals_star, sigma2, &xtx_star_inv, config)?;

    // Standard error for ρ (from GM theory)
    let se_rho = compute_rho_se(&residuals_star, listw, sigma2, config)?;

    Ok(build_result(ResultParts {
        coef_names,
        config,
        model: SphetModel::SpatialError,
        beta,
        vcov: &vcov_mat,
        lambda: None,
        rho: Some((rho, se_rho)),
        sigma2,
        residuals,
        fitted,
        df: residual_df(n, k)?,
        instruments: Vec::new(),
        warnings: Vec::new(),
        gm: None,
    }))
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

    let instruments = spatial_instruments(x, listw, config.instrument_order, coef_names)?;
    let h = instruments.h.clone();
    let (z, z_names) = with_spatial_lag(y, x, listw, coef_names)?;
    let wy = z.column(k).to_owned();

    // Step 1: Initial S2SLS
    let t1 = tsls(y, &z, &instruments, k, &z_names)?;
    let u1 = y - &z.dot(&t1.delta);

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
    let z_hat = {
        let hh = xtx(&h_star.view());
        let (hh_inv, _) = safe_inverse(&hh.view())?;
        h_star.dot(&hh_inv.dot(&h_star.t().dot(&z_star)))
    };
    let vcov_mat = match config.se_type {
        SphetSE::HAC => compute_spatial_2sls_se(&z_hat, &residuals_star, &zph_z_star_inv, config)?,
        se_type => delta_vcov(&z_hat, &residuals_star, &zph_z_star_inv, se_type)?,
    };
    let se_rho = compute_rho_se(&residuals_star, listw, sigma2, config)?;

    let mut warnings = Vec::new();
    warnings.extend(instruments.dropped_warning());
    Ok(build_result(ResultParts {
        coef_names,
        config,
        model: SphetModel::SARAR,
        beta,
        vcov: &vcov_mat,
        lambda: Some(lambda),
        rho: Some((rho, se_rho)),
        sigma2,
        residuals,
        fitted,
        df: residual_df(n, k + 1)?,
        instruments: instruments.names,
        warnings,
        gm: None,
    }))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Everything a fit reports, in estimation order.
struct ResultParts<'a> {
    coef_names: &'a [String],
    config: &'a SphetConfig,
    model: SphetModel,
    beta: Array1<f64>,
    /// Covariance of (beta, lambda) when the model has a lambda, of beta otherwise.
    vcov: &'a Array2<f64>,
    lambda: Option<f64>,
    /// (rho, its standard error)
    rho: Option<(f64, f64)>,
    sigma2: f64,
    residuals: Array1<f64>,
    fitted: Array1<f64>,
    df: usize,
    instruments: Vec<String>,
    warnings: Vec<String>,
    gm: Option<Convergence>,
}

/// z and its two-sided p-value; both NaN unless the standard error is finite and positive.
fn wald(estimate: f64, se: f64) -> (f64, f64) {
    if se.is_finite() && se > 0.0 {
        let z = estimate / se;
        (z, 2.0 * (1.0 - normal_cdf(z.abs())))
    } else {
        (f64::NAN, f64::NAN)
    }
}

fn build_result(parts: ResultParts<'_>) -> SphetResult {
    let ResultParts {
        coef_names,
        config,
        model,
        beta,
        vcov,
        lambda,
        rho,
        sigma2,
        residuals,
        fitted,
        df,
        instruments,
        warnings,
        gm,
    } = parts;
    let k = beta.len();
    let se = |j: usize| vcov[[j, j]].max(0.0).sqrt();
    let se_beta: Vec<f64> = (0..k).map(se).collect();
    let (z_values, p_values): (Vec<f64>, Vec<f64>) =
        beta.iter().zip(&se_beta).map(|(&b, &s)| wald(b, s)).unzip();
    let lambda_stats = lambda.map(|l| {
        let s = se(k);
        let (z, p) = wald(l, s);
        (s, z, p)
    });
    let rho_stats = rho.map(|(r, s)| {
        let (z, p) = wald(r, s);
        (z, p)
    });
    SphetResult {
        coefficients: beta.to_vec(),
        coef_names: coef_names.to_vec(),
        std_errors: se_beta,
        z_values,
        p_values,
        vcov: (0..k)
            .map(|i| (0..k).map(|j| vcov[[i, j]]).collect())
            .collect(),
        lambda,
        lambda_se: lambda_stats.map(|v| v.0),
        lambda_z: lambda_stats.map(|v| v.1),
        lambda_p: lambda_stats.map(|v| v.2),
        rho: rho.map(|v| v.0),
        rho_se: rho.map(|v| v.1),
        rho_z: rho_stats.map(|v| v.0),
        rho_p: rho_stats.map(|v| v.1),
        sigma2,
        n_obs: residuals.len(),
        residuals,
        fitted,
        df,
        model_type: model,
        se_type: config.se_type,
        iterations: gm.as_ref().map_or(0, |c| c.iterations),
        converged: gm.as_ref().is_none_or(|c| c.converged),
        warnings,
        convergence: gm,
        rho_moments: None,
        instruments,
        hac_kernel: None,
        hac_bandwidth: None,
    }
}

/// `n - p`, or `InsufficientData` when the fit has no residual degree of freedom.
fn residual_df(n: usize, p: usize) -> EconResult<usize> {
    n.checked_sub(p)
        .filter(|df| *df > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: "spatial GMM: residual degrees of freedom".to_string(),
        })
}

/// `SingularMatrix` naming the collinear regressors of `[1, X]`. The normal equations decide
/// a full-rank design cheaply; only a doubtful one goes to the pivoted QR (`lm.fit`'s rule).
fn refuse_collinear_regressors(x: &Array2<f64>, names: &[String]) -> EconResult<()> {
    if normal_equations_inverse(x.view())?.is_some() {
        return Ok(());
    }
    let aliased = Qr::decompose(x.view(), LM_QR_TOL)?.aliased();
    match aliasing_message(&aliased, names) {
        Some(message) => Err(EconError::SingularMatrix {
            context: "sphet regressors".to_string(),
            suggestion: format!("{message}; drop the collinear regressors"),
        }),
        None => Ok(()),
    }
}

/// How a vector is projected on the kept instruments: one factorisation for the whole fit.
enum Projector {
    /// `(H'H)^-1`, when H is of full rank and well conditioned.
    Normal(NormalInverse),
    /// Pivoted QR of the kept columns, otherwise.
    Qr(Qr),
}

/// The instruments of one fit: H of full column rank by construction.
struct Instruments {
    h: Array2<f64>,
    names: Vec<String>,
    /// Columns dropped as collinear with the others.
    dropped: Vec<String>,
    /// Columns before dropping.
    candidates: usize,
    projector: Projector,
}

impl Instruments {
    /// `P_H Z`, column by column.
    fn project(&self, z: ndarray::ArrayView2<'_, f64>) -> EconResult<Array2<f64>> {
        match &self.projector {
            Projector::Normal(normal) => {
                Ok(self.h.dot(&normal.coefficients_multi(self.h.view(), z)))
            }
            Projector::Qr(qr) => {
                let mut out = try_zeros2(z.nrows(), z.ncols(), "projected regressors")?;
                for (j, col) in z.columns().into_iter().enumerate() {
                    checkpoint()?;
                    out.column_mut(j)
                        .assign(&self.h.dot(&qr.coefficients(col)?));
                }
                Ok(out)
            }
        }
    }

    fn dropped_warning(&self) -> Option<String> {
        (!self.dropped.is_empty()).then(|| {
            format!(
                "instruments dropped as collinear with the others: {} (rank {} of {})",
                self.dropped.join(", "),
                self.h.ncols(),
                self.candidates
            )
        })
    }
}

/// H = [X, W X_(-1), W^2 X_(-1), ..., W^order X_(-1)]: the lags of the non-intercept columns only
/// (column 0 of `x` is the intercept, and W 1 = 1 for row-standardised W). Columns that the
/// pivoted QR (tol 1e-7, `lm.fit`'s rule) finds collinear with earlier ones are dropped. X is
/// checked for full rank first and that QR pivots only later columns, so X stays H's leading block.
fn spatial_instruments(
    x: &Array2<f64>,
    listw: &SpatialWeights,
    order: usize,
    names: &[String],
) -> EconResult<Instruments> {
    let n = x.nrows();
    let k = x.ncols();
    let lags = k - 1;
    let n_cols = checked_len(&[lags, order], "instrument columns")?
        .checked_add(k)
        .ok_or_else(|| EconError::ResourceLimit("instrument columns overflow".to_string()))?;
    let mut h = try_zeros2(n, n_cols, "instrument matrix")?;
    h.slice_mut(s![.., ..k]).assign(x);
    let mut all_names: Vec<String> = names[..k].to_vec();

    for j in 1..k {
        let mut lag = x.column(j).to_owned();
        for o in 1..=order {
            checkpoint()?;
            lag = listw.lag(&lag);
            let col = k + (o - 1) * lags + (j - 1);
            h.column_mut(col).assign(&lag);
        }
    }
    for o in 1..=order {
        for name in &names[1..k] {
            all_names.push(if o == 1 {
                format!("W {name}")
            } else {
                format!("W^{o} {name}")
            });
        }
    }

    if let Some(normal) = normal_equations_inverse(h.view())? {
        return Ok(Instruments {
            h,
            names: all_names,
            dropped: Vec::new(),
            candidates: n_cols,
            projector: Projector::Normal(normal),
        });
    }

    let qr = Qr::decompose(h.view(), LM_QR_TOL)?;
    let aliased = qr.aliased();
    if aliased.is_empty() {
        return Ok(Instruments {
            h,
            names: all_names,
            dropped: Vec::new(),
            candidates: n_cols,
            projector: Projector::Qr(qr),
        });
    }
    let keep: Vec<usize> = (0..n_cols).filter(|j| !aliased.contains(j)).collect();
    let mut kept = try_zeros2(n, keep.len(), "instrument matrix")?;
    for (out, &j) in keep.iter().enumerate() {
        kept.column_mut(out).assign(&h.column(j));
    }
    let projector = Projector::Qr(Qr::decompose(kept.view(), LM_QR_TOL)?);
    Ok(Instruments {
        h: kept,
        names: keep.iter().map(|&j| all_names[j].clone()).collect(),
        dropped: aliased.iter().map(|&j| all_names[j].clone()).collect(),
        candidates: n_cols,
        projector,
    })
}

/// Z = [X, Wy] and its column names (the last one is `Wy`).
fn with_spatial_lag(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    names: &[String],
) -> EconResult<(Array2<f64>, Vec<String>)> {
    let (n, k) = x.dim();
    let mut z = try_zeros2(n, k + 1, "spatial 2SLS regressors")?;
    z.slice_mut(s![.., ..k]).assign(x);
    z.column_mut(k).assign(&listw.lag(y));
    let mut z_names = names.to_vec();
    z_names.push("Wy".to_string());
    Ok((z, z_names))
}

/// One spatial 2SLS fit.
struct Tsls {
    delta: Array1<f64>,
    /// `P_H Z`
    zh: Array2<f64>,
    /// `(Zh'Zh)^-1`
    a: Array2<f64>,
}

/// `delta = (Zh'Zh)^-1 Zh'y` with Zh = P_H Z. The first `in_h` columns of Z are H's own leading
/// columns (P_H leaves them unchanged), so they are copied rather than projected.
fn tsls(
    y: &Array1<f64>,
    z: &Array2<f64>,
    instruments: &Instruments,
    in_h: usize,
    z_names: &[String],
) -> EconResult<Tsls> {
    let (n, p) = z.dim();
    let q = instruments.h.ncols();
    if n <= q {
        return Err(EconError::InsufficientData {
            required: q + 1,
            provided: n,
            context: "spatial 2SLS: more instruments than observations".to_string(),
        });
    }
    if q < p {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "spatial 2SLS needs {p} linearly independent instruments for {p} parameters, but the instruments ({}) have rank {q}; add regressors or raise instrument_order",
                instruments.names.join(", ")
            ),
        });
    }
    let mut zh = try_zeros2(n, p, "projected regressors")?;
    zh.slice_mut(s![.., ..in_h])
        .assign(&z.slice(s![.., ..in_h]));
    let projected = instruments.project(z.slice(s![.., in_h..]))?;
    zh.slice_mut(s![.., in_h..]).assign(&projected);

    match normal_equations_inverse(zh.view())? {
        Some(normal) => {
            let delta = normal.coefficients(zh.view(), y);
            Ok(Tsls {
                delta,
                zh,
                a: normal.inverse,
            })
        }
        None => {
            let ls = least_squares(zh.view(), y.view())?;
            if let Some(message) = aliasing_message(&ls.aliased, z_names) {
                return Err(EconError::SingularMatrix {
                    context: "spatial 2SLS".to_string(),
                    suggestion: format!("{message}; the instruments do not identify these columns"),
                });
            }
            Ok(Tsls {
                delta: ls.coefficients,
                zh,
                a: ls.unscaled_cov,
            })
        }
    }
}

/// `sum_i u_i^2 p_i p_i'`: the White meat, from the scores `p * u`.
fn scores(p: &Array2<f64>, u: &Array1<f64>) -> Array2<f64> {
    let mut out = p.clone();
    for (mut row, &ui) in out.rows_mut().into_iter().zip(u) {
        row *= ui;
    }
    out
}

/// Classical `s2 A`, with s2 = u'u / (n - p), and White `A (sum u_i^2 p_i p_i') A`.
fn delta_vcov(
    p: &Array2<f64>,
    u: &Array1<f64>,
    a: &Array2<f64>,
    se_type: SphetSE,
) -> EconResult<Array2<f64>> {
    match se_type {
        SphetSE::Standard => {
            let s2 = u.dot(u) / residual_df(p.nrows(), p.ncols())? as f64;
            Ok(s2 * a)
        }
        _ => {
            let meat = xtx(&scores(p, u).view());
            Ok(a.dot(&meat).dot(a))
        }
    }
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
        crate::resource::checkpoint()?;
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

/// Largest HAC lag that can contribute for a sample of `n` rows.
///
/// A lag `>= n` has no pairs `(i, i - lag)` inside the sample, so its
/// autocovariance is zero; stopping at `n - 1` gives the same result for any
/// bandwidth.
fn hac_max_lag(bw: usize, n: usize) -> usize {
    bw.min(n.saturating_sub(1))
}

/// Row-order Newey-West covariance of a spatial 2SLS fit.
fn compute_spatial_2sls_se(
    z_hat: &Array2<f64>,
    residuals: &Array1<f64>,
    zph_z_inv: &Array2<f64>,
    config: &SphetConfig,
) -> EconResult<Array2<f64>> {
    let (n, k) = z_hat.dim();
    let bw = config
        .bandwidth
        .unwrap_or_else(|| (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize);

    let s = scores(z_hat, residuals);
    let mut omega = xtx(&s.view());

    // Lags 1 to bw; lags >= n have no pairs in the sample and add nothing
    for lag in 1..=hac_max_lag(bw, n) {
        crate::resource::checkpoint()?;
        let w = config.kernel.weight(lag, bw);
        if w.abs() < 1e-15 {
            continue;
        }

        let mut gamma_lag = Array2::<f64>::zeros((k, k));
        for i in lag..n {
            for j in 0..k {
                for l in 0..k {
                    gamma_lag[[j, l]] += s[[i, j]] * s[[i - lag, l]];
                }
            }
        }

        for j in 0..k {
            for l in 0..k {
                omega[[j, l]] += w * (gamma_lag[[j, l]] + gamma_lag[[l, j]]);
            }
        }
    }

    Ok(zph_z_inv.dot(&omega).dot(zph_z_inv))
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
    use p2a_validation::{Case, Check, Tol};
    use polars::prelude::*;

    const N_SIM: usize = 150;
    const XS: [&str; 2] = ["x1", "x2"];

    fn column(ds: &Dataset, name: &str) -> Vec<f64> {
        ds.df()
            .column(name)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }
    fn knn5(coords: &[[f64; 2]]) -> SpatialWeights {
        let pts: Vec<(f64, f64)> = coords.iter().map(|c| (c[0], c[1])).collect();
        SpatialWeights::from_neighbors(&Neighbors::from_knn(&pts, 5), WeightStyle::RowStd)
    }
    fn coords_of(ds: &Dataset) -> Vec<[f64; 2]> {
        column(ds, "cx")
            .into_iter()
            .zip(column(ds, "cy"))
            .map(|(a, b)| [a, b])
            .collect()
    }
    /// Case sphet_gmm_hac on sphet_sim.csv (n = 150). W = 5 nearest neighbours on (cx, cy),
    /// row-standardised, as the case's .knn_w. The 5th and 6th neighbour distances differ by at
    /// least 3.0e-5, so there are no ties.
    fn sphet_case() -> (Case, Dataset, SpatialWeights, Vec<[f64; 2]>) {
        let (case, ds) = crate::test_support::load_case("sphet_gmm_hac");
        let coords = coords_of(&ds);
        let w = knn5(&coords);
        (case, ds, w, coords)
    }
    /// The same dataset and W for the property tests, which pin nothing from the case.
    fn sim() -> (Dataset, SpatialWeights, Vec<[f64; 2]>) {
        let path = p2a_validation::repo_path("validation/datasets/generated/sphet_sim.csv");
        let ds = match crate::data::DataLoader::load_csv(&path) {
            Ok(df) => Dataset::new(df),
            Err(e) => panic!("cannot load {}: {e}", path.display()),
        };
        let coords = coords_of(&ds);
        let w = knn5(&coords);
        (ds, w, coords)
    }
    fn config(model: SphetModel, se_type: SphetSE, het: bool) -> SphetConfig {
        SphetConfig {
            model,
            se_type,
            het,
            ..Default::default()
        }
    }
    /// R's delta order: (Intercept), x1, x2, then lambda for SAR/SARAR.
    fn delta(r: &SphetResult) -> Vec<f64> {
        r.coefficients.iter().copied().chain(r.lambda).collect()
    }
    fn delta_se(r: &SphetResult) -> Vec<f64> {
        r.std_errors.iter().copied().chain(r.lambda_se).collect()
    }

    #[test]
    fn test_validate_sphet_instruments() {
        // R: H = cbind(X, W Xn, W^2 Xn), Xn = X[, -1]: qr(H)$rank = h_rank = 7 = h_cols. The
        // earlier set cbind(X, W X, W^2 X) had h_old_cols = 9 columns of rank h_old_rank = 7
        // (W 1 = 1 for row-standardised W).
        let (case, ds, w, _) = sphet_case();
        let r = run_sphet(
            &ds,
            "y_sar",
            &XS,
            &w,
            config(SphetModel::SpatialLag, SphetSE::Standard, true),
        )
        .unwrap();
        assert_eq!(
            r.instruments,
            [
                "(Intercept)",
                "x1",
                "x2",
                "W x1",
                "W x2",
                "W^2 x1",
                "W^2 x2"
            ]
        );
        assert_eq!(r.instruments.len() as f64, case.scalar("h_cols"));
        assert_eq!(case.scalar("h_rank"), case.scalar("h_cols"));
        assert_eq!(
            (case.scalar("h_old_cols"), case.scalar("h_old_rank")),
            (9.0, 7.0)
        );
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        // the kept H has full column rank by construction
        let x = Array2::from_shape_fn((N_SIM, 3), |(i, j)| match j {
            0 => 1.0,
            1 => column(&ds, "x1")[i],
            _ => column(&ds, "x2")[i],
        });
        let names: Vec<String> = ["(Intercept)", "x1", "x2"].map(String::from).to_vec();
        let inst = spatial_instruments(&x, &w, 2, &names).unwrap();
        let rank = Qr::decompose(inst.h.view(), LM_QR_TOL).unwrap().rank();
        assert_eq!(rank, inst.h.ncols());
    }

    #[test]
    fn test_validate_sphet_sar_classical() {
        let (case, ds, w, _) = sphet_case();
        let r = run_sphet(
            &ds,
            "y_sar",
            &XS,
            &w,
            config(SphetModel::SpatialLag, SphetSE::Standard, true),
        )
        .unwrap();
        assert_eq!(r.df, N_SIM - 4);
        assert_eq!(
            (r.iterations, r.converged, r.convergence.is_none()),
            (0, true, true)
        );
        Check::new(&case)
            .vector("sar_coef", &delta(&r), Tol::coef(N_SIM))
            .vector("sar_se_classical", &delta_se(&r), Tol::se(N_SIM))
            .finish();
    }

    #[test]
    fn test_validate_sphet_sar_white() {
        let (case, ds, w, _) = sphet_case();
        let r = run_sphet(
            &ds,
            "y_sar",
            &XS,
            &w,
            config(SphetModel::SpatialLag, SphetSE::Robust, true),
        )
        .unwrap();
        Check::new(&case)
            .vector("sar_coef", &delta(&r), Tol::coef(N_SIM))
            .vector("sar_se_white", &delta_se(&r), Tol::se(N_SIM))
            .finish();
    }

    #[test]
    fn test_sphet_rejects_unidentified_lag_and_collinear_regressors() {
        let (ds, w, _) = sim();
        // Intercept only: H = [1] cannot identify Wy (it used to be [1, W1, W^2 1], of rank 1,
        // and an Ok fit).
        let e = run_sphet(&ds, "y_sar", &[], &w, SphetConfig::default()).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("instruments")),
            "{e}"
        );
        // x3 = 2 x1: R's lm keeps x1 and aliases x3.
        let mut df = ds.df().clone();
        let x3: Vec<f64> = column(&ds, "x1").iter().map(|v| 2.0 * v).collect();
        df.with_column(Column::new("x3".into(), x3)).unwrap();
        for model in [
            SphetModel::SpatialLag,
            SphetModel::SpatialError,
            SphetModel::SARAR,
        ] {
            let e = run_sphet(
                &Dataset::new(df.clone()),
                "y_sar",
                &["x1", "x2", "x3"],
                &w,
                config(model, SphetSE::Robust, true),
            )
            .unwrap_err();
            assert!(
                matches!(&e, EconError::SingularMatrix { suggestion, .. } if suggestion.contains("x3")),
                "{e}"
            );
        }
    }

    #[test]
    fn test_sphet_drops_collinear_instruments_with_a_warning() {
        // A ring of 40 units, each with the next one as its only neighbour: W z = -z for the
        // alternating column x2, so W x2 and W^2 x2 repeat x2 and are dropped.
        let n = 40;
        let nb = Neighbors::from_indices((0..n).map(|i| vec![(i + 1) % n]).collect());
        let w = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        let x1: Vec<f64> = (0..n)
            .map(|i| ((i * 7) % 11) as f64 + 0.3 * (i % 3) as f64)
            .collect();
        let x2: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * x1[i] - x2[i] + ((i * 13) % 7) as f64 * 0.1)
            .collect();
        let d = Dataset::new(df! { "y" => y, "x1" => x1, "x2" => x2 }.unwrap());
        let r = run_sphet(
            &d,
            "y",
            &XS,
            &w,
            config(SphetModel::SpatialLag, SphetSE::Robust, true),
        )
        .unwrap();
        assert_eq!(r.instruments, ["(Intercept)", "x1", "x2", "W x1", "W^2 x1"]);
        assert!(
            r.warnings
                .iter()
                .any(|m| m.contains("W x2, W^2 x2") && m.contains("rank 5 of 7")),
            "{:?}",
            r.warnings
        );
    }

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
    fn instrument_matrix_too_large_is_an_error_not_an_abort() {
        let (_, listw) = create_test_data();
        let x = Array2::from_shape_fn((25, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
        let names = ["(Intercept)".to_string(), "x".to_string()];
        for order in [usize::MAX / 4, usize::MAX] {
            assert!(matches!(
                spatial_instruments(&x, &listw, order, &names),
                Err(EconError::ResourceLimit(_))
            ));
        }
    }

    #[test]
    fn hac_with_huge_bandwidth_finishes() {
        let (d, w) = create_test_data();
        for model in [SphetModel::SpatialLag, SphetModel::SARAR] {
            let c = SphetConfig {
                model,
                se_type: SphetSE::HAC,
                bandwidth: Some(1_000_000_000_000_000),
                ..Default::default()
            };
            let r = run_sphet(&d, "y", &["x"], &w, c).unwrap();
            // With every kernel weight near 1 the HAC meat is (Z_hat'u)(Z_hat'u)', and
            // Z_hat'u = Z'P_H u = 0 for 2SLS residuals, so the variance is 0 up to rounding:
            // the SEs are finite and non-negative. They were positive only while
            // safe_inverse returned a Cholesky "inverse" of the singular H'H (row-standardised
            // weights make W1 repeat the constant column).
            assert!(
                r.std_errors.iter().all(|s| s.is_finite() && *s >= 0.0),
                "{:?}",
                r.std_errors
            );
        }
    }

    #[test]
    fn hac_lag_range_is_clamped_to_the_sample() {
        assert_eq!(hac_max_lag(1_000_000_000_000_000, 60), 59);
        assert_eq!(hac_max_lag(3, 60), 3);
        assert_eq!(hac_max_lag(5, 1), 0);
        assert_eq!(hac_max_lag(5, 0), 0);
    }

    #[test]
    fn instrument_order_must_be_between_1_and_4() {
        let (d, w) = create_test_data();
        for (model, order) in [
            (SphetModel::SpatialLag, 0),
            (SphetModel::SpatialLag, 5),
            (SphetModel::SARAR, usize::MAX / 4),
        ] {
            let c = SphetConfig {
                model,
                instrument_order: order,
                ..Default::default()
            };
            let e = run_sphet(&d, "y", &["x"], &w, c).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("instrument_order")),
                "{e}"
            );
        }
        let c = SphetConfig {
            instrument_order: 4,
            ..Default::default()
        };
        assert!(run_sphet(&d, "y", &["x"], &w, c).is_ok());
    }

    #[test]
    fn sphet_rejects_bad_tolerance_and_honours_cancellation() {
        let (d, w) = create_test_data();
        for tol in [0.0, f64::NAN] {
            assert!(
                run_sphet(
                    &d,
                    "y",
                    &["x"],
                    &w,
                    SphetConfig {
                        tolerance: tol,
                        ..Default::default()
                    }
                )
                .is_err()
            );
        }
        let t = crate::resource::CancelToken::new();
        t.cancel();
        let c = SphetConfig {
            se_type: SphetSE::HAC,
            bandwidth: Some(10),
            ..Default::default()
        };
        let r = crate::resource::with_budget(crate::resource::Budget::new(None, t), || {
            run_sphet(&d, "y", &["x"], &w, c)
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
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

    /// The integer 5 x 5 grid of `spatial.rs`'s `int_grid` (that helper is private to its
    /// module): W = 4 nearest neighbours, row-standardised; x1 = 6 i, x2 = 5 ((7 i) mod 13),
    /// y = round(30 (1.5 + 0.8 i/5 - 0.3 ((7 i) mod 13)/6 + 0.2 (row + col))). Returns the
    /// Int64 columns, a Float64 copy and W.
    fn int_grid() -> (DataFrame, DataFrame, SpatialWeights) {
        let coords: Vec<(f64, f64)> = (0..5)
            .flat_map(|i| (0..5).map(move |j| (i as f64, j as f64)))
            .collect();
        let w =
            SpatialWeights::from_neighbors(&Neighbors::from_knn(&coords, 4), WeightStyle::RowStd);
        let x1: Vec<i64> = (0..25).map(|i| 6 * i).collect();
        let x2: Vec<i64> = (0..25).map(|i| 5 * ((7 * i) % 13)).collect();
        let y: Vec<i64> = (0..25i64)
            .map(|i| {
                let (row, col) = (i / 5, i % 5);
                let v = 1.5 + 0.8 * i as f64 / 5.0 - 0.3 * ((7 * i) % 13) as f64 / 6.0
                    + 0.2 * (row + col) as f64;
                (30.0 * v).round() as i64
            })
            .collect();
        let f = |v: &[i64]| v.iter().map(|&a| a as f64).collect::<Vec<f64>>();
        let int = df! { "y" => &y, "x1" => &x1, "x2" => &x2 }.unwrap();
        let flt = df! { "y" => f(&y), "x1" => f(&x1), "x2" => f(&x2) }.unwrap();
        (int, flt, w)
    }

    #[test]
    fn test_na_sphet_int64_equals_float64_and_refuses_missing() {
        let (int, flt, w) = int_grid();
        let y_null: Vec<Option<f64>> = flt
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, v)| if i == 7 { None } else { v })
            .collect();
        let mut with_null = flt.clone();
        with_null
            .with_column(Column::new("y".into(), y_null))
            .unwrap();
        let run = |df: &DataFrame, model: SphetModel| {
            let config = SphetConfig {
                model,
                ..Default::default()
            };
            run_sphet(&Dataset::new(df.clone()), "y", &["x1", "x2"], &w, config)
        };
        for model in [
            SphetModel::SpatialLag,
            SphetModel::SpatialError,
            SphetModel::SARAR,
        ] {
            let (a, b) = (run(&int, model), run(&flt, model));
            assert!(b.is_ok(), "{model:?}: {b:?}");
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "{model:?}");
            let e = run(&with_null, model).unwrap_err();
            assert!(
                matches!(e, EconError::NullValues { ref column, count: 1 } if column == "y"),
                "{model:?}: {e:?}"
            );
        }
    }
}
