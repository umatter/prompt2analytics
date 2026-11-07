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
//! Alternatively, spatial HAC standard errors (Kelejian & Prucha, 2007) are robust to both
//! heteroscedasticity and spatial correlation of unknown form: the score cross-products of units
//! i and j are weighted by a kernel of their distance d_ij, from unit coordinates or a distance
//! matrix, with a fixed distance bandwidth or each unit's farthest W-neighbour as its bandwidth.
//! They do not depend on the order of the rows.
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
    /// White (HC0) sandwich: robust to heteroskedasticity of unknown form
    #[default]
    Robust,
    /// Spatial HAC (Kelejian-Prucha 2007): kernel of the distance between units, also robust to
    /// spatial correlation of unknown form. Needs `coords` or `distance`.
    HAC,
    /// Classical (homoskedastic) standard errors
    Standard,
}

impl fmt::Display for SphetSE {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SphetSE::Robust => write!(f, "Heteroskedasticity-robust (White)"),
            SphetSE::HAC => write!(f, "Spatial HAC (Kelejian-Prucha 2007)"),
            SphetSE::Standard => write!(f, "Classical (homoskedastic)"),
        }
    }
}

/// Kernel of the Kelejian-Prucha (2007) spatial HAC (sphet 2.1-1 `kernelsfun.R`), as a function
/// of z = d_ij / bandwidth >= 0; zero for z >= 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpatialHacKernel {
    /// K(z) = 1 - z (the Bartlett kernel in distance).
    Triangular,
    /// K(z) = 1. Not positive semi-definite: a variance can come out negative.
    Rectangular,
    /// K(z) = 1 - z^2 (sphet's `stslshac` default).
    #[default]
    Epanechnikov,
    /// K(z) = (1 - z^2)^2.
    Bisquare,
    /// K(z) = 1 - 6 z^2 + 6 z^3 for z <= 0.5, 2 (1 - z)^3 otherwise.
    Parzen,
    /// K(z) = (1 + cos(pi z)) / 2.
    TukeyHanning,
    /// K(z) = 25 / (12 pi^2 z^2) (sin(a) / a - cos(a)), a = 6 pi z / 5, with K(0) = 1 its limit,
    /// truncated at z = 1 as sphet does. Truncated, it is not positive semi-definite.
    QuadraticSpectral,
}

impl SpatialHacKernel {
    /// The weight at z = d_ij / bandwidth >= 0.
    pub fn weight(self, z: f64) -> f64 {
        if z >= 1.0 {
            return 0.0;
        }
        match self {
            SpatialHacKernel::Triangular => 1.0 - z,
            SpatialHacKernel::Rectangular => 1.0,
            SpatialHacKernel::Epanechnikov => 1.0 - z * z,
            SpatialHacKernel::Bisquare => {
                let t = 1.0 - z * z;
                t * t
            }
            SpatialHacKernel::Parzen => {
                if z <= 0.5 {
                    1.0 - 6.0 * z * z + 6.0 * z * z * z
                } else {
                    let t = 1.0 - z;
                    2.0 * t * t * t
                }
            }
            SpatialHacKernel::TukeyHanning => 0.5 * (1.0 + (std::f64::consts::PI * z).cos()),
            SpatialHacKernel::QuadraticSpectral => {
                let a = 1.2 * std::f64::consts::PI * z;
                if a < 1e-2 {
                    // The closed form loses all digits as z -> 0; 1 - a^2/10 + a^4/280 is exact
                    // to rounding here.
                    let a2 = a * a;
                    1.0 - a2 / 10.0 + a2 * a2 / 280.0
                } else {
                    3.0 / (a * a) * (a.sin() / a - a.cos())
                }
            }
        }
    }
}

crate::option_enum!(
    SpatialHacKernel,
    crate::options::Choices {
        option: "hac_kernel",
        values: &[
            (
                &["triangular", "bartlett", "tri"],
                SpatialHacKernel::Triangular
            ),
            (
                &["rectangular", "truncated", "uniform"],
                SpatialHacKernel::Rectangular
            ),
            (&["epanechnikov", "epan"], SpatialHacKernel::Epanechnikov),
            (
                &["bisquare", "biweight", "bisq"],
                SpatialHacKernel::Bisquare
            ),
            (&["parzen"], SpatialHacKernel::Parzen),
            (&["tukey_hanning", "th"], SpatialHacKernel::TukeyHanning),
            (
                &["quadratic_spectral", "qs"],
                SpatialHacKernel::QuadraticSpectral
            ),
        ]
    }
);

/// Configuration for sphet estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphetConfig {
    /// Type of spatial model to estimate
    pub model: SphetModel,
    /// Whether to use heteroscedasticity-robust estimation
    pub het: bool,
    /// Standard error type
    pub se_type: SphetSE,
    /// Deprecated: the kernel of the removed row-order Newey-West. Must stay Bartlett with
    /// se_type HAC; set `hac_kernel`.
    pub kernel: HacKernel,
    /// Deprecated: the lag count of the removed row-order Newey-West. Must stay `None` with
    /// se_type HAC; set `hac_bandwidth`.
    pub bandwidth: Option<usize>,
    /// Number of instruments: order of W to use (default 2 means [X, WX, W²X])
    pub instrument_order: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Unit coordinates for the spatial HAC (row i = unit i of W). Set this or `distance`.
    #[serde(default)]
    pub coords: Option<Vec<[f64; 2]>>,
    /// Alternatively an n x n distance matrix (finite, >= 0, symmetric, zero diagonal).
    #[serde(skip)]
    pub distance: Option<Array2<f64>>,
    /// Fixed distance bandwidth of the spatial HAC: pairs with d_ij < d get weight K(d_ij / d).
    /// `None` is the variable bandwidth: unit i's bandwidth is its largest distance to its
    /// W-neighbours, and only those pairs get a weight (sphet's default, with W's neighbour sets
    /// as the distance object).
    #[serde(default)]
    pub hac_bandwidth: Option<f64>,
    /// Kernel of the spatial HAC.
    #[serde(default)]
    pub hac_kernel: SpatialHacKernel,
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
            coords: None,
            distance: None,
            hac_bandwidth: None,
            hac_kernel: SpatialHacKernel::default(),
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

    let hac = validate_hac(&config, n)?;
    refuse_collinear_regressors(&x, &coef_names)?;

    // Dispatch to appropriate model
    let hac = hac.as_ref();
    match config.model {
        SphetModel::SpatialLag => estimate_sar_gmm(&y, &x, listw, &coef_names, &config, hac),
        SphetModel::SpatialError => estimate_sem_gmm(&y, &x, listw, &coef_names, &config, hac),
        SphetModel::SARAR => estimate_sarar_gmm(&y, &x, listw, &coef_names, &config, hac),
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
    hac: Option<&HacSpec<'_>>,
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

    let vcov_mat = delta_vcov(&t.zh, &residuals, &t.a, config.se_type, hac, listw)?;

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
    hac: Option<&HacSpec<'_>>,
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
    hac: Option<&HacSpec<'_>>,
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
    let vcov_mat = delta_vcov(
        &z_hat,
        &residuals_star,
        &zph_z_star_inv,
        config.se_type,
        hac,
        listw,
    )?;
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
        hac_kernel: (config.se_type == SphetSE::HAC).then_some(config.hac_kernel),
        hac_bandwidth: config
            .hac_bandwidth
            .filter(|_| config.se_type == SphetSE::HAC),
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

/// Classical `s2 A`, with s2 = u'u / (n - p); White `A (sum u_i^2 p_i p_i') A`; spatial HAC
/// `A (spatial_hac_meat) A`, with no degrees-of-freedom factor (sphet's `tsls`).
fn delta_vcov(
    p: &Array2<f64>,
    u: &Array1<f64>,
    a: &Array2<f64>,
    se_type: SphetSE,
    hac: Option<&HacSpec<'_>>,
    listw: &SpatialWeights,
) -> EconResult<Array2<f64>> {
    let meat = match (se_type, hac) {
        (SphetSE::Standard, _) => {
            let s2 = u.dot(u) / residual_df(p.nrows(), p.ncols())? as f64;
            return Ok(s2 * a);
        }
        (SphetSE::HAC, Some(hac)) => spatial_hac_meat(&scores(p, u), hac, listw)?,
        _ => xtx(&scores(p, u).view()),
    };
    Ok(a.dot(&meat).dot(a))
}

/// Where the distances of the spatial HAC come from.
enum HacSource<'a> {
    Coords(&'a [[f64; 2]]),
    Distance(&'a Array2<f64>),
}

impl HacSource<'_> {
    fn distance(&self, i: usize, j: usize) -> f64 {
        match self {
            HacSource::Coords(c) => {
                let (dx, dy) = (c[i][0] - c[j][0], c[i][1] - c[j][1]);
                (dx * dx + dy * dy).sqrt()
            }
            HacSource::Distance(d) => d[[i, j]],
        }
    }
}

/// A validated spatial HAC request.
struct HacSpec<'a> {
    source: HacSource<'a>,
    /// `None`: the variable bandwidth of each unit's W-neighbours.
    bandwidth: Option<f64>,
    kernel: SpatialHacKernel,
}

/// The spatial HAC request of `config`, checked against the n units of the weights; `None`
/// unless se_type is HAC.
fn validate_hac(config: &SphetConfig, n: usize) -> EconResult<Option<HacSpec<'_>>> {
    let invalid = |message: String| EconError::InvalidSpecification { message };
    if config.se_type != SphetSE::HAC {
        return Ok(None);
    }
    if let Some(b) = config.bandwidth {
        return Err(invalid(format!(
            "bandwidth ({b}) was the lag count of the removed row-order Newey-West HAC, which ignored the spatial layout; set hac_bandwidth, a distance in the units of the coordinates"
        )));
    }
    if config.kernel != HacKernel::Bartlett {
        return Err(invalid(format!(
            "kernel {} is a time-series HAC kernel and is no longer used; set hac_kernel ({})",
            config.kernel,
            SpatialHacKernel::choices().names().join(" | ")
        )));
    }
    let source = match (&config.coords, &config.distance) {
        (None, None) => {
            return Err(invalid(
                "se_type HAC needs the unit coordinates (coords, one [x, y] per row of the weights) or an n x n distance matrix (distance)"
                    .to_string(),
            ));
        }
        (Some(_), Some(_)) => {
            return Err(invalid(
                "set only one of coords and distance for the spatial HAC".to_string(),
            ));
        }
        (Some(coords), None) => {
            if coords.len() != n {
                return Err(invalid(format!(
                    "coords has {} points but the weights have {n} units",
                    coords.len()
                )));
            }
            if let Some(i) = coords
                .iter()
                .position(|c| !(c[0].is_finite() && c[1].is_finite()))
            {
                return Err(invalid(format!(
                    "coords contains a non-finite value at row {i}"
                )));
            }
            HacSource::Coords(coords)
        }
        (None, Some(d)) => {
            validate_distance_matrix(d, n)?;
            HacSource::Distance(d)
        }
    };
    if let Some(d) = config.hac_bandwidth
        && !(d.is_finite() && d > 0.0)
    {
        return Err(invalid(format!(
            "hac_bandwidth must be a finite distance > 0 for se_type HAC, got {d}"
        )));
    }
    Ok(Some(HacSpec {
        source,
        bandwidth: config.hac_bandwidth,
        kernel: config.hac_kernel,
    }))
}

fn validate_distance_matrix(d: &Array2<f64>, n: usize) -> EconResult<()> {
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    let (r, c) = d.dim();
    if (r, c) != (n, n) {
        return invalid(format!("distance must be {n} x {n}, got {r} x {c}"));
    }
    let mut largest = 0.0_f64;
    for ((i, j), &v) in d.indexed_iter() {
        if !(v.is_finite() && v >= 0.0) {
            return invalid(format!(
                "distance must be finite and >= 0 (row {i}, column {j} is {v})"
            ));
        }
        largest = largest.max(v);
    }
    let tol = 1e-12 * largest.max(1.0);
    for i in 0..n {
        checkpoint()?;
        if d[[i, i]] != 0.0 {
            return invalid(format!(
                "distance must have a zero diagonal (row {i} is {})",
                d[[i, i]]
            ));
        }
        for j in i + 1..n {
            if (d[[i, j]] - d[[j, i]]).abs() > tol {
                return invalid(format!(
                    "distance must be symmetric (d[{i}][{j}] = {} but d[{j}][{i}] = {})",
                    d[[i, j]],
                    d[[j, i]]
                ));
            }
        }
    }
    Ok(())
}

/// Accumulates `sum w s_i s_j'` over pairs of score rows.
struct PairMeat<'a> {
    scores: ndarray::ArrayView2<'a, f64>,
    p: usize,
    sum: Vec<f64>,
}

impl PairMeat<'_> {
    fn add(&mut self, i: usize, j: usize, w: f64) {
        let (si, sj) = (self.scores.row(i), self.scores.row(j));
        for (a, &sa) in si.iter().enumerate() {
            let scaled = w * sa;
            let row = &mut self.sum[a * self.p..(a + 1) * self.p];
            for (cell, &sb) in row.iter_mut().zip(sj.iter()) {
                *cell += scaled * sb;
            }
        }
    }

    /// `white + factor (sum + sum')`.
    fn finish(self, white: Array2<f64>, factor: f64) -> Array2<f64> {
        let p = self.p;
        let mut meat = white;
        for a in 0..p {
            for b in 0..p {
                meat[[a, b]] += factor * (self.sum[a * p + b] + self.sum[b * p + a]);
            }
        }
        meat
    }
}

/// Kelejian-Prucha (2007): the White meat of the scores s_i = p_i u_i, plus the kernel-weighted
/// cross-products of the pairs of units.
///
/// Fixed bandwidth d: every pair i < j with d_ij < d adds K(d_ij / d) (s_i s_j' + s_j s_i'), and
/// coordinates are swept in x order, so the cost is O(n log n + candidate pairs) and no pair list
/// is stored. Variable bandwidth: each unit i adds K(d_ij / d_i) s_i s_j' for its W-neighbours j,
/// with d_i its largest distance to them, symmetrised as (M + M') / 2 (the diagonal SEs do not
/// change); the cost is O(nnz(W) p^2).
fn spatial_hac_meat(
    scores: &Array2<f64>,
    hac: &HacSpec<'_>,
    listw: &SpatialWeights,
) -> EconResult<Array2<f64>> {
    const SCAN_CHECKPOINT: usize = 1 << 16;
    let (n, p) = scores.dim();
    let white = xtx(&scores.view());
    let mut pairs = PairMeat {
        scores: scores.view(),
        p,
        sum: crate::resource::try_vec(
            checked_len(&[p, p], "spatial HAC meat")?,
            0.0,
            "spatial HAC meat",
        )?,
    };
    let kernel = hac.kernel;

    let Some(bandwidth) = hac.bandwidth else {
        for i in 0..n {
            if i.is_multiple_of(256) {
                checkpoint()?;
            }
            let sw = listw.get_weights(i);
            let others = || {
                sw.indices
                    .iter()
                    .zip(&sw.weights)
                    .filter(|&(&j, &w)| j != i && w != 0.0)
                    .map(|(&j, _)| j)
            };
            let d_i = others().fold(0.0_f64, |m, j| m.max(hac.source.distance(i, j)));
            for j in others() {
                // Coincident neighbours only (d_i = 0): z = 0.
                let z = if d_i > 0.0 {
                    hac.source.distance(i, j) / d_i
                } else {
                    0.0
                };
                let w = kernel.weight(z);
                if w != 0.0 {
                    pairs.add(i, j, w);
                }
            }
        }
        return Ok(pairs.finish(white, 0.5));
    };

    match &hac.source {
        HacSource::Coords(coords) => {
            let mut order = crate::resource::try_with_capacity::<usize>(n, "spatial HAC sweep")?;
            order.extend(0..n);
            order.sort_by(|&a, &b| coords[a][0].total_cmp(&coords[b][0]).then(a.cmp(&b)));
            let mut scanned = SCAN_CHECKPOINT;
            for (pos, &i) in order.iter().enumerate() {
                if scanned >= SCAN_CHECKPOINT {
                    checkpoint()?;
                    scanned = 0;
                }
                let [xi, yi] = coords[i];
                for &j in &order[pos + 1..] {
                    let [xj, yj] = coords[j];
                    let dx = xj - xi;
                    if dx >= bandwidth {
                        break;
                    }
                    scanned += 1;
                    let dy = yj - yi;
                    let d = (dx * dx + dy * dy).sqrt();
                    if d < bandwidth {
                        let w = kernel.weight(d / bandwidth);
                        if w != 0.0 {
                            pairs.add(i, j, w);
                        }
                    }
                }
            }
        }
        HacSource::Distance(dm) => {
            for i in 0..n {
                checkpoint()?;
                for j in i + 1..n {
                    let d = dm[[i, j]];
                    if d < bandwidth {
                        let w = kernel.weight(d / bandwidth);
                        if w != 0.0 {
                            pairs.add(i, j, w);
                        }
                    }
                }
            }
        }
    }
    Ok(pairs.finish(white, 1.0))
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
    let mut best_rho = 0.0;
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
    const TH_025: f64 = 0.8535533905932737;
    const TH_075: f64 = 0.14644660940672627;
    const QS_025: f64 = 0.9139455782435691;
    const QS_075: f64 = 0.3979103991034253;
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

    fn hac_config(
        model: SphetModel,
        het: bool,
        coords: &[[f64; 2]],
        bandwidth: f64,
        kernel: SpatialHacKernel,
    ) -> SphetConfig {
        SphetConfig {
            coords: Some(coords.to_vec()),
            hac_bandwidth: Some(bandwidth),
            hac_kernel: kernel,
            ..config(model, SphetSE::HAC, het)
        }
    }

    #[test]
    fn test_validate_sphet_sar_hac() {
        let (case, ds, w, coords) = sphet_case();
        let mut check = Check::new(&case);
        for (key, d, k) in [
            ("sar_se_hac_tri_015", 0.15, SpatialHacKernel::Triangular),
            ("sar_se_hac_epa_015", 0.15, SpatialHacKernel::Epanechnikov),
            ("sar_se_hac_tri_030", 0.30, SpatialHacKernel::Triangular),
        ] {
            let r = run_sphet(
                &ds,
                "y_sar",
                &XS,
                &w,
                hac_config(SphetModel::SpatialLag, true, &coords, d, k),
            )
            .unwrap();
            assert_eq!((r.hac_kernel, r.hac_bandwidth), (Some(k), Some(d)));
            check
                .vector("sar_coef", &delta(&r), Tol::coef(N_SIM))
                .vector(key, &delta_se(&r), Tol::se(N_SIM));
        }
        check.finish();
    }

    #[test]
    fn test_validate_sphet_sar_hac_kernels() {
        let (case, ds, w, coords) = sphet_case();
        let mut check = Check::new(&case);
        for (key, k) in [
            ("sar_se_hac_rect_015", SpatialHacKernel::Rectangular),
            ("sar_se_hac_bisq_015", SpatialHacKernel::Bisquare),
            ("sar_se_hac_parzen_015", SpatialHacKernel::Parzen),
            ("sar_se_hac_th_015", SpatialHacKernel::TukeyHanning),
            ("sar_se_hac_qs_015", SpatialHacKernel::QuadraticSpectral),
        ] {
            let r = run_sphet(
                &ds,
                "y_sar",
                &XS,
                &w,
                hac_config(SphetModel::SpatialLag, true, &coords, 0.15, k),
            )
            .unwrap();
            check.vector(key, &delta_se(&r), Tol::se(N_SIM));
        }
        check.finish();
    }

    /// The rows sorted by the x coordinate, with W rebuilt on the sorted coordinates.
    fn sorted_by_cx(ds: &Dataset, coords: &[[f64; 2]]) -> (Dataset, SpatialWeights, Vec<[f64; 2]>) {
        let cx = column(ds, "cx");
        let mut order: Vec<usize> = (0..N_SIM).collect();
        order.sort_by(|&a, &b| cx[a].total_cmp(&cx[b]));
        let pick = |name: &str| {
            let v = column(ds, name);
            order.iter().map(|&i| v[i]).collect::<Vec<f64>>()
        };
        let df = df! { "y_sar" => pick("y_sar"), "x1" => pick("x1"), "x2" => pick("x2") }.unwrap();
        let pc: Vec<[f64; 2]> = order.iter().map(|&i| coords[i]).collect();
        let w = knn5(&pc);
        (Dataset::new(df), w, pc)
    }

    fn assert_same_fit(base: &SphetResult, other: &SphetResult, rel: f64) {
        let tol = Tol::new(rel, 1e-15);
        let a = [delta(base), delta_se(base)].concat();
        let b = [delta(other), delta_se(other)].concat();
        for (x, y) in a.iter().zip(&b) {
            assert!(tol.allows(*y, *x), "{x} vs {y}");
        }
    }

    #[test]
    fn test_validate_sphet_sar_hac_variable() {
        // The variable bandwidth: each unit's farthest W-neighbour; invariant to the row order.
        let (case, ds, w, coords) = sphet_case();
        let cfg = |c: &[[f64; 2]]| SphetConfig {
            hac_bandwidth: None,
            ..hac_config(
                SphetModel::SpatialLag,
                true,
                c,
                1.0,
                SpatialHacKernel::Epanechnikov,
            )
        };
        let r = run_sphet(&ds, "y_sar", &XS, &w, cfg(&coords)).unwrap();
        assert_eq!(
            (r.hac_kernel, r.hac_bandwidth),
            (Some(SpatialHacKernel::Epanechnikov), None)
        );
        Check::new(&case)
            .vector("sar_se_hac_epa_var", &delta_se(&r), Tol::se(N_SIM))
            .finish();
        let (sorted, sw, pc) = sorted_by_cx(&ds, &coords);
        let perm = run_sphet(&sorted, "y_sar", &XS, &sw, cfg(&pc)).unwrap();
        assert_same_fit(&r, &perm, 1e-11);
    }

    #[test]
    fn test_sphet_hac_row_permutation_invariant() {
        // SPAT-20: the old HAC paired row i with row i - l. With bandwidth 4 the intercept SE was
        // 0.3946365283 in file order and 0.4161269832 sorted by cx (base-R emulation). The spatial
        // HAC depends only on the distances.
        let (ds, w, coords) = sim();
        let cfg = |c: &[[f64; 2]]| {
            hac_config(
                SphetModel::SpatialLag,
                true,
                c,
                0.15,
                SpatialHacKernel::Triangular,
            )
        };
        let base = run_sphet(&ds, "y_sar", &XS, &w, cfg(&coords)).unwrap();
        let (sorted, sw, pc) = sorted_by_cx(&ds, &coords);
        let perm = run_sphet(&sorted, "y_sar", &XS, &sw, cfg(&pc)).unwrap();
        assert_same_fit(&base, &perm, 1e-11);
    }

    #[test]
    fn test_sphet_hac_tiny_bandwidth_equals_white() {
        // A bandwidth below the smallest pair distance keeps only K(0) = 1 on the diagonal: White exactly.
        let (ds, w, coords) = sim();
        let mut min_d = f64::INFINITY;
        for (i, a) in coords.iter().enumerate() {
            for b in &coords[i + 1..] {
                min_d = min_d.min((a[0] - b[0]).hypot(a[1] - b[1]));
            }
        }
        let hac = run_sphet(
            &ds,
            "y_sar",
            &XS,
            &w,
            hac_config(
                SphetModel::SpatialLag,
                true,
                &coords,
                0.5 * min_d,
                SpatialHacKernel::Triangular,
            ),
        )
        .unwrap();
        let white = run_sphet(
            &ds,
            "y_sar",
            &XS,
            &w,
            config(SphetModel::SpatialLag, SphetSE::Robust, true),
        )
        .unwrap();
        assert_eq!(delta_se(&hac), delta_se(&white));
    }

    #[test]
    fn test_sphet_hac_distance_matrix_equals_coords() {
        let (ds, w, coords) = sim();
        let n = coords.len();
        let dm = Array2::from_shape_fn((n, n), |(i, j)| {
            let (dx, dy) = (coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]);
            (dx * dx + dy * dy).sqrt()
        });
        let kernel = SpatialHacKernel::Epanechnikov;
        for bandwidth in [Some(0.15), None] {
            let via_coords = SphetConfig {
                hac_bandwidth: bandwidth,
                ..hac_config(SphetModel::SpatialLag, true, &coords, 1.0, kernel)
            };
            let via_matrix = SphetConfig {
                distance: Some(dm.clone()),
                hac_bandwidth: bandwidth,
                hac_kernel: kernel,
                ..config(SphetModel::SpatialLag, SphetSE::HAC, true)
            };
            let a = run_sphet(&ds, "y_sar", &XS, &w, via_coords).unwrap();
            let b = run_sphet(&ds, "y_sar", &XS, &w, via_matrix).unwrap();
            let tol = Tol::new(1e-12, 1e-15);
            for (x, y) in delta_se(&a).iter().zip(delta_se(&b)) {
                assert!(tol.allows(y, *x), "{x} vs {y}");
            }
        }
    }

    #[test]
    fn test_sphet_hac_rejects_bad_inputs() {
        let (ds, w, coords) = sim();
        let msg = |c: SphetConfig| match run_sphet(&ds, "y_sar", &XS, &w, c) {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        };
        let tri = SpatialHacKernel::Triangular;
        let lag = SphetModel::SpatialLag;
        assert!(msg(config(lag, SphetSE::HAC, true)).contains("coords"));
        for d in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                msg(hac_config(lag, true, &coords, d, tri)).contains("hac_bandwidth"),
                "{d}"
            );
        }
        assert!(msg(hac_config(lag, true, &coords[..149], 0.15, tri)).contains("coords"));
        let mut bad = coords.clone();
        bad[7][1] = f64::NAN;
        assert!(msg(hac_config(lag, true, &bad, 0.15, tri)).contains("coords"));
        let both = SphetConfig {
            distance: Some(Array2::zeros((150, 150))),
            ..hac_config(lag, true, &coords, 0.15, tri)
        };
        assert!(msg(both).contains("distance"));
        let with_matrix = |dm: Array2<f64>| SphetConfig {
            distance: Some(dm),
            hac_bandwidth: Some(0.15),
            ..config(lag, SphetSE::HAC, true)
        };
        let mut dm = Array2::<f64>::zeros((150, 150));
        dm[[0, 1]] = 1.0;
        assert!(msg(with_matrix(dm)).contains("symmetric"));
        assert!(msg(with_matrix(Array2::zeros((3, 3)))).contains("150 x 150"));
        let mut dm = Array2::<f64>::zeros((150, 150));
        dm[[2, 2]] = 1.0;
        assert!(msg(with_matrix(dm)).contains("diagonal"));
        let mut dm = Array2::<f64>::zeros((150, 150));
        dm[[1, 2]] = -1.0;
        dm[[2, 1]] = -1.0;
        assert!(msg(with_matrix(dm)).contains(">= 0"));
        // the removed row-order HAC's fields
        let legacy = SphetConfig {
            bandwidth: Some(4),
            ..hac_config(lag, true, &coords, 0.15, tri)
        };
        assert!(msg(legacy).contains("hac_bandwidth"));
        let legacy = SphetConfig {
            kernel: HacKernel::Parzen,
            ..hac_config(lag, true, &coords, 0.15, tri)
        };
        assert!(msg(legacy).contains("hac_kernel"));
    }

    #[test]
    fn test_spatial_hac_kernel_parses_aliases_and_rejects_unknown() {
        use SpatialHacKernel as K;
        K::choices().validate().unwrap();
        for (s, k) in [
            ("triangular", K::Triangular),
            ("Triangular", K::Triangular),
            ("bartlett", K::Triangular),
            ("tri", K::Triangular),
            ("rectangular", K::Rectangular),
            ("truncated", K::Rectangular),
            ("uniform", K::Rectangular),
            ("Epanechnikov", K::Epanechnikov),
            ("epan", K::Epanechnikov),
            ("bisquare", K::Bisquare),
            ("biweight", K::Bisquare),
            ("bisq", K::Bisquare),
            ("parzen", K::Parzen),
            ("tukey_hanning", K::TukeyHanning),
            ("tukey-hanning", K::TukeyHanning),
            ("th", K::TukeyHanning),
            ("quadratic_spectral", K::QuadraticSpectral),
            ("qs", K::QuadraticSpectral),
        ] {
            assert_eq!(s.parse::<K>().unwrap(), k, "{s}");
        }
        let e = "gaussian".parse::<K>().unwrap_err().to_string();
        for name in [
            "hac_kernel",
            "triangular",
            "rectangular",
            "epanechnikov",
            "bisquare",
            "parzen",
            "tukey_hanning",
            "quadratic_spectral",
        ] {
            assert!(e.contains(name), "{e}");
        }
        assert_eq!(K::default(), K::Epanechnikov); // sphet::stslshac's default
        // Hand-computed K(0.25), K(0.75).
        for (k, a, b) in [
            (K::Triangular, 0.75, 0.25),
            (K::Rectangular, 1.0, 1.0),
            (K::Epanechnikov, 0.9375, 0.4375),
            (K::Bisquare, 0.87890625, 0.19140625),
            (K::Parzen, 0.71875, 0.03125),
            (K::TukeyHanning, TH_025, TH_075),
            (K::QuadraticSpectral, QS_025, QS_075),
        ] {
            assert!(
                (k.weight(0.25) - a).abs() < 1e-15,
                "{k}: {}",
                k.weight(0.25)
            );
            assert!(
                (k.weight(0.75) - b).abs() < 1e-15,
                "{k}: {}",
                k.weight(0.75)
            );
            assert_eq!(k.weight(0.0), 1.0, "{k}");
            assert_eq!(k.weight(1.0), 0.0, "{k}");
            assert_eq!(k.weight(1.5), 0.0, "{k}");
        }
        // The QS series below the cut-over agrees with the closed form above it.
        let z = 1e-2 / (1.2 * std::f64::consts::PI);
        assert!(
            (K::QuadraticSpectral.weight(z * 0.999) - K::QuadraticSpectral.weight(z * 1.001)).abs()
                < 1e-7
        );
        assert_eq!(K::QuadraticSpectral.weight(1e-300), 1.0);
    }

    /// The layout of `create_test_data`.
    fn grid_coords() -> Vec<[f64; 2]> {
        (0..5)
            .flat_map(|i| (0..5).map(move |j| [i as f64, j as f64]))
            .collect()
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
                coords: (se_type == SphetSE::HAC).then(grid_coords),
                hac_bandwidth: (se_type == SphetSE::HAC).then_some(1.5),
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
                coords: Some(grid_coords()),
                hac_bandwidth: Some(1e300),
                ..Default::default()
            };
            let r = run_sphet(&d, "y", &["x"], &w, c).unwrap();
            // With every kernel weight near 1 the HAC meat is (Z_hat'u)(Z_hat'u)', and
            // Z_hat'u = Z'P_H u = 0 for 2SLS residuals, so the variance is 0 up to rounding:
            // the SEs are finite and non-negative (test_sphet_se_types has positive ones).
            assert!(
                r.std_errors.iter().all(|s| s.is_finite() && *s >= 0.0),
                "{:?}",
                r.std_errors
            );
        }
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
            coords: Some(grid_coords()),
            hac_bandwidth: Some(2.0),
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
