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
use crate::linalg::solve::{
    LM_QR_TOL, Qr, SingularPolicy, aliasing_message, least_squares, sym_inverse,
};
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
    /// true (default): the GM moments of Kelejian-Prucha (2010), robust to heteroskedasticity,
    /// for rho and its SE; false: the three homoskedastic moments of Kelejian-Prucha (1999).
    /// No effect on the lag model.
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
    /// Relative Newton-step tolerance of the GM root solve for rho
    pub tolerance: f64,
    /// Iteration cap of the GM root solve, shared by its brackets
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
            tolerance: 1e-10,
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
    let x_normal = refuse_collinear_regressors(&x, &coef_names)?;
    if config.model != SphetModel::SpatialLag {
        refuse_self_neighbours(listw)?;
    }

    // Dispatch to appropriate model
    let hac = hac.as_ref();
    match config.model {
        SphetModel::SpatialLag => estimate_sar_gmm(&y, &x, listw, &coef_names, &config, hac),
        SphetModel::SpatialError => {
            estimate_sem_gmm(&y, &x, x_normal, listw, &coef_names, &config, hac)
        }
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
        rho_moments: None,
    }))
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEM Model (Spatial Error)
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate spatial error model: the GM estimator of rho on the OLS residuals, then feasible GLS.
///
/// Model: y = Xβ + u, where u = ρWu + ε
fn estimate_sem_gmm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    x_normal: Option<NormalInverse>,
    listw: &SpatialWeights,
    coef_names: &[String],
    config: &SphetConfig,
    hac: Option<&HacSpec<'_>>,
) -> EconResult<SphetResult> {
    let n = y.len();
    let k = x.ncols();

    // Step 1: OLS
    let first = ols_fit(x, y, x_normal, coef_names)?;

    // Step 2: GM estimation of rho
    let gm = estimate_rho_gm(&first.residuals, listw, config.het, config)?;
    let rho = gm.rho;

    // Step 3: feasible GLS: OLS of (I - rho W) y on (I - rho W) X
    let y_star = listw.transform_y(y, rho);
    let x_star = listw.transform_x(x, rho);
    let normal = normal_equations_inverse(x_star.view())?;
    let second = ols_fit(&x_star, &y_star, normal, coef_names)?;

    let df = residual_df(n, k)?;
    let sigma2 = second.residuals.dot(&second.residuals) / df as f64;
    let fitted = x.dot(&second.beta);
    let residuals = y - &fitted;

    // Standard errors for beta: on the transformed data
    let vcov_mat = delta_vcov(
        &x_star,
        &second.residuals,
        &second.unscaled,
        config.se_type,
        hac,
        listw,
    )?;

    let (se_rho, rho_note) = gm_rho_se(&gm, &first.residuals, None, listw, config.het)?;
    let mut warnings = gm.warnings;
    warnings.extend(rho_note);

    Ok(build_result(ResultParts {
        coef_names,
        config,
        model: SphetModel::SpatialError,
        beta: second.beta,
        vcov: &vcov_mat,
        lambda: None,
        rho: Some((rho, se_rho)),
        sigma2,
        residuals,
        fitted,
        df,
        instruments: Vec::new(),
        warnings,
        gm: Some(gm.convergence),
        rho_moments: Some(gm_moments_name(config.het).to_string()),
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

    // Step 2: GM estimation of rho
    let gm = estimate_rho_gm(&u1, listw, config.het, config)?;
    let rho = gm.rho;

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
    let (se_rho, rho_note) = gm_rho_se(
        &gm,
        &u1,
        Some(FirstStep {
            z: &z,
            zh: &t1.zh,
            a: &t1.a,
        }),
        listw,
        config.het,
    )?;

    let mut warnings = gm.warnings;
    warnings.extend(rho_note);
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
        gm: Some(gm.convergence),
        rho_moments: Some(gm_moments_name(config.het).to_string()),
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
    rho_moments: Option<String>,
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
        rho_moments,
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
        rho_moments,
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
/// a full-rank design cheaply and their inverse is returned; only a doubtful design goes to the
/// pivoted QR (`lm.fit`'s rule).
fn refuse_collinear_regressors(
    x: &Array2<f64>,
    names: &[String],
) -> EconResult<Option<NormalInverse>> {
    if let Some(normal) = normal_equations_inverse(x.view())? {
        return Ok(Some(normal));
    }
    let aliased = Qr::decompose(x.view(), LM_QR_TOL)?.aliased();
    match aliasing_message(&aliased, names) {
        Some(message) => Err(EconError::SingularMatrix {
            context: "sphet regressors".to_string(),
            suggestion: format!("{message}; drop the collinear regressors"),
        }),
        None => Ok(None),
    }
}

/// The Kelejian-Prucha GM moments assume w_ii = 0.
fn refuse_self_neighbours(listw: &SpatialWeights) -> EconResult<()> {
    for i in 0..listw.n() {
        if listw.get_weights(i).indices.contains(&i) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "spatial weights have a non-zero diagonal (unit {i} is its own neighbour); the Kelejian-Prucha GM moments need w_ii = 0"
                ),
            });
        }
    }
    Ok(())
}

/// One least-squares fit of the error model.
struct OlsFit {
    beta: Array1<f64>,
    residuals: Array1<f64>,
    /// `(X'X)^-1`
    unscaled: Array2<f64>,
}

/// `y` on `x`: from `normal` (the normal equations' inverse of `x`), otherwise by pivoted QR.
fn ols_fit(
    x: &Array2<f64>,
    y: &Array1<f64>,
    normal: Option<NormalInverse>,
    names: &[String],
) -> EconResult<OlsFit> {
    match normal {
        Some(normal) => {
            let beta = normal.coefficients(x.view(), y);
            let residuals = y - &x.dot(&beta);
            Ok(OlsFit {
                beta,
                residuals,
                unscaled: normal.inverse,
            })
        }
        None => {
            let ls = least_squares(x.view(), y.view())?;
            if let Some(message) = aliasing_message(&ls.aliased, names) {
                return Err(EconError::SingularMatrix {
                    context: "sphet regressors".to_string(),
                    suggestion: format!("{message}; drop the collinear regressors"),
                });
            }
            Ok(OlsFit {
                beta: ls.coefficients,
                residuals: ls.residuals,
                unscaled: ls.unscaled_cov,
            })
        }
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

/// The GM estimate of rho is searched on [-GM_RHO_BOUND, GM_RHO_BOUND].
const GM_RHO_BOUND: f64 = 0.99;

fn gm_moments_name(het: bool) -> &'static str {
    if het {
        "Kelejian-Prucha (2010) heteroskedasticity-robust GM, 2 moments"
    } else {
        "Kelejian-Prucha (1999) GM, 3 moments (homoskedastic)"
    }
}

/// The GM moments m(theta) = g - G (rho, rho^2[, sigma^2])'.
struct GmParts {
    g: Array1<f64>,
    gmat: Array2<f64>,
}

/// Homoskedastic (Kelejian-Prucha 1999, three moments with sigma^2) or heteroskedastic
/// (Kelejian-Prucha 2010, two moments: m1 = eb'eb - e' diag(W'W) e and m2 = eb'e, sphet's `gg_het`).
fn gm_parts(u: &Array1<f64>, listw: &SpatialWeights, het: bool) -> GmParts {
    let n = u.len() as f64;
    let ub = listw.lag(u);
    let ubb = listw.lag(&ub);
    let (u_ub, ub_ub, u_ubb) = (u.dot(&ub), ub.dot(&ub), u.dot(&ubb));
    let (ub_ubb, ubb_ubb) = (ub.dot(&ubb), ubb.dot(&ubb));
    if het {
        // diag(W'W)_j = sum_i w_ij^2
        let mut dg = Array1::<f64>::zeros(u.len());
        for i in 0..u.len() {
            let sw = listw.get_weights(i);
            for (&j, &w) in sw.indices.iter().zip(&sw.weights) {
                dg[j] += w * w;
            }
        }
        let (mut d_uu, mut d_uub, mut d_ubub) = (0.0, 0.0, 0.0);
        for i in 0..u.len() {
            d_uu += dg[i] * u[i] * u[i];
            d_uub += dg[i] * u[i] * ub[i];
            d_ubub += dg[i] * ub[i] * ub[i];
        }
        GmParts {
            g: ndarray::array![(ub_ub - d_uu) / n, u_ub / n],
            gmat: ndarray::array![
                [2.0 * (ub_ubb - d_uub) / n, -(ubb_ubb - d_ubub) / n],
                [(u_ubb + ub_ub) / n, -ub_ubb / n]
            ],
        }
    } else {
        let tr = listw.trace_wtw();
        GmParts {
            g: ndarray::array![u.dot(u) / n, ub_ub / n, u_ub / n],
            gmat: ndarray::array![
                [2.0 * u_ub / n, -ub_ub / n, 1.0],
                [2.0 * ub_ubb / n, -ubb_ubb / n, tr / n],
                [(u_ubb + ub_ub) / n, -ub_ubb / n, 0.0]
            ],
        }
    }
}

/// The minimum of a GM objective on an interval.
#[derive(Debug)]
struct QuarticMin {
    rho: f64,
    objective: f64,
    iterations: usize,
    /// No bracket was cut off by the iteration cap.
    converged: bool,
    on_boundary: bool,
    final_change: f64,
}

/// Minimiser of `f(r) = ||a - b r - c r^2||^2` on [lo, hi], exactly: the interior roots of the
/// derivative cubic, by safeguarded Newton in the windows on which f' is monotone, and the two
/// bounds, whichever gives the smallest f. Ties go to the smaller |r|, then the smaller r.
fn minimise_quartic(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    lo: f64,
    hi: f64,
    tol: f64,
    max_iter: usize,
) -> EconResult<QuarticMin> {
    let flat = || {
        EconError::InvalidSpecification {
        message: "the GM objective for rho is flat (the first-step residuals are zero or carry no spatial information); rho is not identified"
            .to_string(),
    }
    };
    // On the unit scale: the path and the stopping rule do not depend on the units of y.
    let scale = a
        .iter()
        .chain(b)
        .chain(c)
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return Err(flat());
    }
    let unit = |v: &[f64]| v.iter().map(|x| x / scale).collect::<Vec<f64>>();
    let (a, b, c) = (unit(a), unit(b), unit(c));
    let dot = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(x, y)| x * y).sum::<f64>();
    // f'(r) / 2 = d0 + d1 r + d2 r^2 + d3 r^3
    let d0 = -dot(&b, &a);
    let d1 = dot(&b, &b) - 2.0 * dot(&c, &a);
    let d2 = 3.0 * dot(&b, &c);
    let d3 = 2.0 * dot(&c, &c);
    if d0 == 0.0 && d1 == 0.0 && d2 == 0.0 && d3 == 0.0 {
        return Err(flat());
    }
    let f = |r: f64| {
        a.iter()
            .zip(&b)
            .zip(&c)
            .map(|((&ai, &bi), &ci)| {
                let m = ai - bi * r - ci * r * r;
                m * m
            })
            .sum::<f64>()
    };
    let df = |r: f64| d0 + r * (d1 + r * (d2 + r * d3));
    let ddf = |r: f64| d1 + r * (2.0 * d2 + r * 3.0 * d3);

    // The turning points of f' split [lo, hi] into windows on which f' is monotone.
    let mut knots = vec![lo];
    let mut turning: Vec<f64> = Vec::with_capacity(2);
    if d3 != 0.0 {
        let disc = d2 * d2 - 3.0 * d1 * d3;
        if disc > 0.0 {
            let q = -(d2 + (if d2 < 0.0 { -1.0 } else { 1.0 }) * disc.sqrt());
            turning.push(q / (3.0 * d3));
            if q != 0.0 {
                turning.push(d1 / q);
            }
        }
    } else if d2 != 0.0 {
        turning.push(-d1 / (2.0 * d2));
    }
    turning.sort_by(f64::total_cmp);
    knots.extend(turning.into_iter().filter(|&t| t > lo && t < hi));
    knots.push(hi);

    let mut candidates = vec![lo, hi];
    let mut iterations = 0;
    let mut converged = true;
    let mut final_change = 0.0;
    for window in knots.windows(2) {
        let (p, q) = (window[0], window[1]);
        let (dp, dq) = (df(p), df(q));
        if dp == 0.0 || dp * dq > 0.0 {
            continue;
        }
        if dq == 0.0 {
            candidates.push(q);
            continue;
        }
        let (mut left, mut right) = (p, q);
        let left_sign = dp.is_sign_positive();
        let mut x = 0.5 * (p + q);
        loop {
            if iterations >= max_iter {
                converged = false;
                candidates.push(x);
                break;
            }
            iterations += 1;
            checkpoint()?;
            let dx = df(x);
            if dx == 0.0 {
                final_change = 0.0;
                candidates.push(x);
                break;
            }
            if dx.is_sign_positive() == left_sign {
                left = x;
            } else {
                right = x;
            }
            let slope = ddf(x);
            let newton = x - dx / slope;
            let next = if slope != 0.0 && newton > left && newton < right {
                newton
            } else {
                0.5 * (left + right)
            };
            final_change = (next - x).abs() / next.abs().max(1.0);
            x = next;
            if final_change <= tol {
                candidates.push(x);
                break;
            }
        }
    }

    let key = |r: f64| (f(r), r.abs(), r);
    let rho = candidates
        .iter()
        .copied()
        .min_by(|&p, &q| {
            let (kp, kq) = (key(p), key(q));
            kp.0.total_cmp(&kq.0)
                .then(kp.1.total_cmp(&kq.1))
                .then(kp.2.total_cmp(&kq.2))
        })
        .unwrap_or(lo);
    Ok(QuarticMin {
        rho,
        objective: f(rho) * scale * scale,
        iterations,
        converged,
        on_boundary: rho == lo || rho == hi,
        final_change,
    })
}

/// The GM estimate of rho and what the rest of the fit needs from it.
struct GmSolve {
    rho: f64,
    /// sigma^2 of the homoskedastic moments
    sigma2: Option<f64>,
    objective: f64,
    parts: GmParts,
    on_boundary: bool,
    convergence: Convergence,
    warnings: Vec<String>,
}

/// The GM estimator of rho on the first-step residuals `u`. sigma^2 enters the homoskedastic
/// moments linearly and is concentrated out, so both objectives are quartics in rho.
fn estimate_rho_gm(
    u: &Array1<f64>,
    listw: &SpatialWeights,
    het: bool,
    config: &SphetConfig,
) -> EconResult<GmSolve> {
    let parts = gm_parts(u, listw, het);
    let (g, gmat) = (&parts.g, &parts.gmat);
    let (g0, g1) = (gmat.column(0), gmat.column(1));
    let (min, sigma2) = if het {
        let min = minimise_quartic(
            &g.to_vec(),
            &g0.to_vec(),
            &g1.to_vec(),
            -GM_RHO_BOUND,
            GM_RHO_BOUND,
            config.tolerance,
            config.max_iter,
        )?;
        (min, None)
    } else {
        let g2 = gmat.column(2);
        let g2g2 = g2.dot(&g2);
        let project = |v: ndarray::ArrayView1<'_, f64>| -> Vec<f64> {
            let t = g2.dot(&v) / g2g2;
            v.iter().zip(g2.iter()).map(|(x, y)| x - t * y).collect()
        };
        let min = minimise_quartic(
            &project(g.view()),
            &project(g0),
            &project(g1),
            -GM_RHO_BOUND,
            GM_RHO_BOUND,
            config.tolerance,
            config.max_iter,
        )?;
        let fitted = g - &(min.rho * &g0) - &(min.rho * min.rho * &g1);
        let sigma2 = g2.dot(&fitted) / g2g2;
        (min, Some(sigma2))
    };
    const CRITERION: &str = "relative step of the GM first-order condition (safeguarded Newton)";
    let record = |converged: bool| {
        Convergence::new(
            converged,
            min.iterations,
            config.max_iter,
            CRITERION,
            min.final_change,
            config.tolerance,
        )
    };
    let mut warnings = Vec::new();
    if !min.converged {
        warnings.extend(record(false).note("GM estimation of rho"));
    }
    if min.on_boundary {
        warnings.push(format!(
            "GM estimate of rho is at the boundary {} of the search interval [-{GM_RHO_BOUND}, {GM_RHO_BOUND}]: the GM objective has no interior minimum, so rho and the estimates that use it are unreliable",
            min.rho
        ));
    }
    let convergence = record(min.converged && !min.on_boundary);
    Ok(GmSolve {
        rho: min.rho,
        sigma2,
        objective: min.objective,
        on_boundary: min.on_boundary,
        convergence,
        warnings,
        parts,
    })
}

/// A sparse symmetric matrix in compressed rows, with each row sorted by column.
struct SparseSym {
    ptr: Vec<usize>,
    idx: Vec<usize>,
    val: Vec<f64>,
}

impl SparseSym {
    fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.ptr[i], self.ptr[i + 1]);
        (&self.idx[a..b], &self.val[a..b])
    }

    fn n(&self) -> usize {
        self.ptr.len() - 1
    }

    /// `v I`
    fn identity(n: usize, v: f64) -> EconResult<Self> {
        Ok(SparseSym {
            ptr: (0..=n).collect(),
            idx: (0..n).collect(),
            val: crate::resource::try_vec(n, v, "GM moment matrix")?,
        })
    }

    /// `W + W'`
    fn w_plus_wt(listw: &SpatialWeights) -> EconResult<Self> {
        let n = listw.n();
        let t = RowsOfTranspose::of(listw)?;
        let nnz: usize = (0..n).map(|i| listw.get_weights(i).indices.len()).sum();
        let mut out = SparseSym {
            ptr: Vec::with_capacity(n + 1),
            idx: crate::resource::try_with_capacity(2 * nnz, "W + W'")?,
            val: crate::resource::try_with_capacity(2 * nnz, "W + W'")?,
        };
        out.ptr.push(0);
        for i in 0..n {
            let sw = listw.get_weights(i);
            let (rows, weights) = t.row(i);
            // merge row i of W with column i of W, both sorted
            let (mut a, mut b) = (0, 0);
            while a < sw.indices.len() || b < rows.len() {
                let ja = sw.indices.get(a).copied().unwrap_or(usize::MAX);
                let jb = rows.get(b).copied().unwrap_or(usize::MAX);
                let j = ja.min(jb);
                let mut v = 0.0;
                if ja == j {
                    v += sw.weights[a];
                    a += 1;
                }
                if jb == j {
                    v += weights[b];
                    b += 1;
                }
                out.idx.push(j);
                out.val.push(v);
            }
            out.ptr.push(out.idx.len());
        }
        Ok(out)
    }

    /// `scale (W'W - diag(W'W))` when `zero_diagonal`, `scale W'W` otherwise: entry (j, k) is
    /// the sum of w_ij w_ik over the rows i.
    fn wtw(listw: &SpatialWeights, zero_diagonal: bool, scale: f64) -> EconResult<Self> {
        let n = listw.n();
        let t = RowsOfTranspose::of(listw)?;
        let too_many = || {
            EconError::ResourceLimit(
                "W'W for the GM variance of rho has too many entries".to_string(),
            )
        };
        let bound = (0..n).try_fold(0_usize, |m, i| {
            let k = listw.get_weights(i).indices.len();
            k.checked_mul(k).and_then(|kk| m.checked_add(kk))
        });
        let bound = bound.ok_or_else(too_many)?;
        let bound = bound.min(n.checked_mul(n).ok_or_else(too_many)?);
        let what = format!("W'W for the GM variance of rho needs up to {bound} entries");
        let mut out = SparseSym {
            ptr: Vec::with_capacity(n + 1),
            idx: crate::resource::try_with_capacity(bound, &what)?,
            val: crate::resource::try_with_capacity(bound, &what)?,
        };
        out.ptr.push(0);
        let mut dense = crate::resource::try_vec(n, 0.0_f64, &what)?;
        let mut seen = crate::resource::try_vec(n, false, &what)?;
        let mut touched: Vec<usize> = Vec::new();
        for j in 0..n {
            if j.is_multiple_of(1024) {
                checkpoint()?;
            }
            let (rows, weights) = t.row(j);
            for (&i, &w_ij) in rows.iter().zip(weights) {
                let sw = listw.get_weights(i);
                for (&k, &w_ik) in sw.indices.iter().zip(&sw.weights) {
                    if !seen[k] {
                        seen[k] = true;
                        touched.push(k);
                    }
                    dense[k] += w_ij * w_ik;
                }
            }
            touched.sort_unstable();
            for &k in &touched {
                if !(zero_diagonal && k == j) {
                    out.idx.push(k);
                    out.val.push(scale * dense[k]);
                }
                dense[k] = 0.0;
                seen[k] = false;
            }
            touched.clear();
            out.ptr.push(out.idx.len());
        }
        Ok(out)
    }

    fn mul_vec(&self, x: &Array1<f64>) -> Array1<f64> {
        Array1::from_iter((0..self.n()).map(|i| {
            let (idx, val) = self.row(i);
            idx.iter().zip(val).map(|(&j, &v)| v * x[j]).sum::<f64>()
        }))
    }

    fn diag(&self) -> Array1<f64> {
        Array1::from_iter((0..self.n()).map(|i| {
            let (idx, val) = self.row(i);
            idx.binary_search(&i).map_or(0.0, |pos| val[pos])
        }))
    }

    /// `sum_ij self_ij other_ij g_i g_j` (g = 1 when `None`): tr(self G other G) for symmetric
    /// matrices.
    fn weighted_inner(&self, other: &SparseSym, g: Option<&Array1<f64>>) -> f64 {
        let mut total = 0.0;
        for i in 0..self.n() {
            let (ia, va) = self.row(i);
            let (ib, vb) = other.row(i);
            let (mut a, mut b) = (0, 0);
            let mut row = 0.0;
            while a < ia.len() && b < ib.len() {
                match ia[a].cmp(&ib[b]) {
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                    std::cmp::Ordering::Equal => {
                        let cell = va[a] * vb[b];
                        row += g.map_or(cell, |g| cell * g[ia[a]]);
                        a += 1;
                        b += 1;
                    }
                }
            }
            total += g.map_or(row, |g| row * g[i]);
        }
        total
    }
}

/// The columns of W as rows: for each j, the units i with w_ij != 0 in ascending order.
struct RowsOfTranspose {
    ptr: Vec<usize>,
    idx: Vec<usize>,
    val: Vec<f64>,
}

impl RowsOfTranspose {
    fn of(listw: &SpatialWeights) -> EconResult<Self> {
        let n = listw.n();
        let mut ptr = crate::resource::try_vec(n + 1, 0_usize, "W'")?;
        for i in 0..n {
            for &j in &listw.get_weights(i).indices {
                ptr[j + 1] += 1;
            }
        }
        for j in 0..n {
            ptr[j + 1] += ptr[j];
        }
        let nnz = ptr[n];
        let mut idx = crate::resource::try_vec(nnz, 0_usize, "W'")?;
        let mut val = crate::resource::try_vec(nnz, 0.0_f64, "W'")?;
        let mut next = ptr.clone();
        for i in 0..n {
            let sw = listw.get_weights(i);
            for (&j, &w) in sw.indices.iter().zip(&sw.weights) {
                idx[next[j]] = i;
                val[next[j]] = w;
                next[j] += 1;
            }
        }
        Ok(RowsOfTranspose { ptr, idx, val })
    }

    fn row(&self, j: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.ptr[j], self.ptr[j + 1]);
        (&self.idx[a..b], &self.val[a..b])
    }
}

/// The endogenous first step of a SARAR fit (Z, P_H Z and (Zh'Zh)^-1), which the variance of
/// the moments has to account for. The error model's first step is OLS on exogenous X, so
/// alpha_r = -n^-1 E[X~'(A_r + A_r') eps] = 0 (Kelejian-Prucha 2010, Theorem 2) and it has none.
struct FirstStep<'a> {
    z: &'a Array2<f64>,
    zh: &'a Array2<f64>,
    a: &'a Array2<f64>,
}

/// `(I - rho W')^-1 b` by the Neumann series `sum_p (rho W')^p b`, stopping when the term falls
/// below 1e-12 of the partial sum. `None` when it diverges. For a row-standardised W a Krylov
/// solver would converge at the same rate, rho^p.
fn solve_i_minus_rho_wt(
    listw: &SpatialWeights,
    rho: f64,
    b: &Array1<f64>,
) -> EconResult<Option<Array1<f64>>> {
    const MAX_TERMS: usize = 100_000;
    let largest = |v: &Array1<f64>| v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let b_max = largest(b);
    let mut sum = b.clone();
    if b_max == 0.0 {
        return Ok(Some(sum));
    }
    let mut term = b.clone();
    for p in 1..=MAX_TERMS {
        if p.is_multiple_of(64) {
            checkpoint()?;
        }
        term = rho * &listw.lag_transpose(&term);
        let t_max = largest(&term);
        if !t_max.is_finite() || t_max > 1e12 * b_max {
            return Ok(None);
        }
        sum += &term;
        if t_max <= 1e-12 * largest(&sum) {
            return Ok(Some(sum));
        }
    }
    Ok(None)
}

/// SE of the GM rho: sqrt([(J'J)^-1 J' Psi J (J'J)^-1]_00 / n), the GMM sandwich of the
/// identity-weighted estimator, with J = d m / d theta and Psi the variance of the moments of
/// Kelejian-Prucha (2010, Theorem 2) as sphet's `psirhorho_het` computes it (`het`), or its
/// homoskedastic analogue with the mu3 and mu4 terms of `psirhorho_hom` for the three moments
/// A1 = I, A2 = W'W, A3 = W. `u` holds the first-step residuals. Returns NaN and a warning when
/// the asymptotic variance does not apply or cannot be formed.
fn gm_rho_se(
    gm: &GmSolve,
    u: &Array1<f64>,
    first: Option<FirstStep<'_>>,
    listw: &SpatialWeights,
    het: bool,
) -> EconResult<(f64, Option<String>)> {
    let rho = gm.rho;
    if gm.on_boundary {
        return Ok((
            f64::NAN,
            Some(
                "the SE of rho is not reported: the GM estimate is at the boundary of the search interval, where its asymptotic variance does not apply"
                    .to_string(),
            ),
        ));
    }
    let n = u.len();
    let eps = listw.transform_y(u, rho);
    let (bs, ds): (Vec<SparseSym>, Vec<Array1<f64>>) = if het {
        (
            vec![
                SparseSym::wtw(listw, true, 2.0)?,
                SparseSym::w_plus_wt(listw)?,
            ],
            Vec::new(),
        )
    } else {
        let wtw = SparseSym::wtw(listw, false, 2.0)?;
        let diag_wtw = 0.5 * &wtw.diag();
        (
            vec![
                SparseSym::identity(n, 2.0)?,
                wtw,
                SparseSym::w_plus_wt(listw)?,
            ],
            vec![Array1::ones(n), diag_wtw, Array1::zeros(n)],
        )
    };

    // a_r = (I - rho W')^-1 T alpha_r, T = n Zh (Zh'Zh)^-1, alpha_r = -n^-1 Z~'(A_r + A_r') eps
    let mut avec: Vec<Array1<f64>> = Vec::new();
    if let Some(first) = &first {
        let zt = listw.transform_x(first.z, rho);
        for b in &bs {
            let t_r = -first.zh.dot(&first.a.dot(&zt.t().dot(&b.mul_vec(&eps))));
            match solve_i_minus_rho_wt(listw, rho, &t_r)? {
                Some(a_r) => avec.push(a_r),
                None => {
                    return Ok((
                        f64::NAN,
                        Some(format!(
                            "the SE of rho is not available: (I - rho W')^-1 has no convergent Neumann series at rho = {rho}"
                        )),
                    ));
                }
            }
        }
    }

    let m = bs.len();
    let nf = n as f64;
    let mut psi = Array2::<f64>::zeros((m, m));
    let ga = eps.mapv(|e| e * e);
    let s2 = eps.dot(&eps) / nf;
    let mu3 = eps.iter().map(|e| e * e * e).sum::<f64>() / nf;
    let mu4 = eps.iter().map(|e| e * e * e * e).sum::<f64>() / nf;
    for r in 0..m {
        for t in r..m {
            let mut v = if het {
                0.5 * bs[r].weighted_inner(&bs[t], Some(&ga))
            } else {
                s2 * s2 * 0.5 * bs[r].weighted_inner(&bs[t], None)
                    + (mu4 - 3.0 * s2 * s2) * ds[r].dot(&ds[t])
            };
            if !avec.is_empty() {
                let (ar, at) = (&avec[r], &avec[t]);
                if het {
                    v += ar
                        .iter()
                        .zip(&ga)
                        .zip(at)
                        .map(|((a, g), b)| a * g * b)
                        .sum::<f64>();
                } else {
                    v += s2 * ar.dot(at) + mu3 * (ar.dot(&ds[t]) + at.dot(&ds[r]));
                }
            }
            psi[[r, t]] = v / nf;
            psi[[t, r]] = v / nf;
        }
    }

    let gmat = &gm.parts.gmat;
    let slope = &gmat.column(0) + &(2.0 * rho * &gmat.column(1));
    let mut j = Array2::<f64>::zeros((m, if het { 1 } else { 2 }));
    j.column_mut(0).assign(&slope);
    if !het {
        j.column_mut(1).assign(&gmat.column(2));
    }
    let jj = j.t().dot(&j);
    let jj_inv = match sym_inverse(jj.view(), SingularPolicy::Error, "GM Jacobian J'J") {
        Ok(inv) => inv.inverse,
        Err(e @ (EconError::Cancelled(_) | EconError::ResourceLimit(_))) => return Err(e),
        Err(_) => {
            return Ok((
                f64::NAN,
                Some("the SE of rho is not available: the GM Jacobian is singular".to_string()),
            ));
        }
    };
    let bread = jj_inv.dot(&j.t());
    let v = bread.dot(&psi).dot(&bread.t());
    Ok(((v[[0, 0]] / nf).max(0.0).sqrt(), None))
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

    fn design_sim(ds: &Dataset) -> Array2<f64> {
        let (x1, x2) = (column(ds, "x1"), column(ds, "x2"));
        Array2::from_shape_fn((x1.len(), 3), |(i, j)| match j {
            0 => 1.0,
            1 => x1[i],
            _ => x2[i],
        })
    }
    fn ols_residuals(ds: &Dataset, y: &str) -> Array1<f64> {
        let y = Array1::from(column(ds, y));
        least_squares(design_sim(ds).view(), y.view())
            .unwrap()
            .residuals
    }

    #[test]
    fn test_validate_sphet_sem_gm_hom() {
        // R .gm_rho: KP (1999), three moments, sigma^2 concentrated out, exact minimiser on
        // [-0.99, 0.99]. The former invalid moments (u'Wu - rho (Wu)'Wu)^2 + (u'W^2u - rho
        // (Wu)'W^2u)^2 gave 0.8995203819178312.
        let (case, ds, w, _) = sphet_case();
        let gm = estimate_rho_gm(
            &ols_residuals(&ds, "y_sem"),
            &w,
            false,
            &SphetConfig::default(),
        )
        .unwrap();
        let r = run_sphet(
            &ds,
            "y_sem",
            &XS,
            &w,
            config(SphetModel::SpatialError, SphetSE::Standard, false),
        )
        .unwrap();
        // run_sphet's first step takes the normal equations, this one the QR: rounding apart
        assert!(Tol::new(1e-11, 0.0).allows(r.rho.unwrap(), gm.rho));
        let conv = r.convergence.clone().expect("GM convergence");
        assert!(
            conv.converged && conv.iterations >= 1 && conv.iterations <= conv.max_iterations,
            "{conv:?}"
        );
        assert_eq!((r.iterations, r.converged), (conv.iterations, true));
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert!(r.rho_moments.as_deref().unwrap().contains("1999"));
        Check::new(&case)
            .scalar("sem_hom_rho", gm.rho, Tol::iterative(1e-8))
            .scalar("sem_hom_gm_objective", gm.objective, Tol::iterative(1e-8))
            .scalar(
                "sem_hom_gm_sigma2",
                gm.sigma2.unwrap(),
                Tol::iterative(1e-8),
            )
            .vector("sem_hom_coef", &r.coefficients, Tol::iterative(1e-8))
            .vector("sem_hom_se_classical", &r.std_errors, Tol::se(N_SIM))
            .finish();
    }

    #[test]
    fn test_validate_sphet_sem_gm_het() {
        // R .gm_rho_het: KP (2010), m1 = eb'eb - e' diag(W'W) e, m2 = eb'e (sphet's gg_het).
        let (case, ds, w, _) = sphet_case();
        let gm = estimate_rho_gm(
            &ols_residuals(&ds, "y_sem"),
            &w,
            true,
            &SphetConfig::default(),
        )
        .unwrap();
        assert!(gm.sigma2.is_none());
        let r = run_sphet(
            &ds,
            "y_sem",
            &XS,
            &w,
            config(SphetModel::SpatialError, SphetSE::Standard, true),
        )
        .unwrap();
        // run_sphet's first step takes the normal equations, this one the QR: rounding apart
        assert!(Tol::new(1e-11, 0.0).allows(r.rho.unwrap(), gm.rho));
        assert!(r.converged && r.warnings.is_empty(), "{:?}", r.warnings);
        assert!(r.rho_moments.as_deref().unwrap().contains("2010"));
        Check::new(&case)
            .scalar("sem_het_rho", gm.rho, Tol::iterative(1e-8))
            .scalar("sem_het_gm_objective", gm.objective, Tol::iterative(1e-8))
            .vector("sem_het_coef", &r.coefficients, Tol::iterative(1e-8))
            .vector("sem_het_se_classical", &r.std_errors, Tol::se(N_SIM))
            .finish();
    }

    #[test]
    fn test_sphet_het_option_changes_rho() {
        // KP (2010) vs KP (1999) give rho 0.48159 vs 0.47874 (SEM) and 0.42520 vs 0.41717 (SARAR);
        // het used to be ignored, and both runs were identical.
        let (ds, w, _) = sim();
        for (model, y) in [
            (SphetModel::SpatialError, "y_sem"),
            (SphetModel::SARAR, "y_sarar"),
        ] {
            let a = run_sphet(&ds, y, &XS, &w, config(model, SphetSE::Standard, true)).unwrap();
            let b = run_sphet(&ds, y, &XS, &w, config(model, SphetSE::Standard, false)).unwrap();
            assert!(
                (a.rho.unwrap() - b.rho.unwrap()).abs() > 1e-3,
                "{model:?}: {:?} vs {:?}",
                a.rho,
                b.rho
            );
            assert_ne!(a.rho_moments, b.rho_moments);
        }
    }

    #[test]
    fn test_sphet_gm_flags_boundary_solution() {
        // y = 1 + 0.8 x1 - 0.5 x2 + cx + cy: the OLS residuals are a smooth trend. R
        // (.sphet_quartic_min on [-5, 5]) puts the GM minimum at 1.047192 (het) / 1.054467 (hom),
        // outside [-0.99, 0.99].
        let (ds, w, _) = sim();
        let (x1, x2, cx, cy) = (
            column(&ds, "x1"),
            column(&ds, "x2"),
            column(&ds, "cx"),
            column(&ds, "cy"),
        );
        let y: Vec<f64> = (0..N_SIM)
            .map(|i| 1.0 + 0.8 * x1[i] - 0.5 * x2[i] + cx[i] + cy[i])
            .collect();
        let df = df! { "y" => y, "x1" => x1, "x2" => x2 }.unwrap();
        for het in [true, false] {
            let r = run_sphet(
                &Dataset::new(df.clone()),
                "y",
                &XS,
                &w,
                config(SphetModel::SpatialError, SphetSE::Standard, het),
            )
            .unwrap();
            assert_eq!(r.rho, Some(0.99), "het={het}");
            assert!(!r.converged && !r.convergence.as_ref().unwrap().converged);
            assert!(
                r.warnings.iter().any(|m| m.contains("boundary")),
                "{:?}",
                r.warnings
            );
        }
    }

    #[test]
    fn test_sphet_gm_reports_iteration_limit() {
        let (ds, w, _) = sim();
        let c = SphetConfig {
            max_iter: 1,
            ..config(SphetModel::SpatialError, SphetSE::Standard, true)
        };
        let r = run_sphet(&ds, "y_sem", &XS, &w, c).unwrap();
        assert!(!r.converged);
        assert!(
            r.warnings.iter().any(|m| m.contains("did not converge")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_gm_quartic_minimiser_interior_boundary_and_flat() {
        // Hand-computed: f = (0.5 - r)^2 -> 0.5; f = (1.5 - r)^2 -> bound 0.99 with f = 0.2601;
        // f = (0.09 - r^2)^2 + (0.03 - 0.1 r)^2 = 0 at r = 0.3 (the other stationary points
        // -0.017712 and -0.282288 have f > 0.0034); a, b, c = 0 is flat.
        let m = minimise_quartic(&[0.5], &[1.0], &[0.0], -0.99, 0.99, 1e-10, 100).unwrap();
        assert!(
            (m.rho - 0.5).abs() < 1e-15 && m.objective < 1e-30 && m.converged && !m.on_boundary,
            "{m:?}"
        );
        let m = minimise_quartic(&[1.5], &[1.0], &[0.0], -0.99, 0.99, 1e-10, 100).unwrap();
        assert!(
            m.rho == 0.99 && m.on_boundary && (m.objective - 0.2601).abs() < 1e-14,
            "{m:?}"
        );
        let m = minimise_quartic(
            &[0.09, 0.03],
            &[0.0, 0.1],
            &[1.0, 0.0],
            -0.99,
            0.99,
            1e-10,
            100,
        )
        .unwrap();
        assert!(
            (m.rho - 0.3).abs() < 1e-12 && m.objective < 1e-24 && m.converged && !m.on_boundary,
            "{m:?}"
        );
        assert!(m.iterations >= 1 && m.iterations <= 100);
        let e = minimise_quartic(&[0.0], &[0.0], &[0.0], -0.99, 0.99, 1e-10, 100).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("flat")),
            "{e}"
        );
    }

    #[test]
    fn test_sphet_gm_scale_invariant() {
        // The GM moments scale by c^2, so rho, the Newton path and the iteration count do not
        // depend on the units of y. The former Newton loop stopped on an absolute |hess| < 1e-10
        // at small scales.
        let (ds, w, _) = sim();
        let (x1, x2, y) = (column(&ds, "x1"), column(&ds, "x2"), column(&ds, "y_sem"));
        for het in [true, false] {
            let fit = |s: f64| {
                let df = df! {
                    "y" => y.iter().map(|v| v * s).collect::<Vec<f64>>(),
                    "x1" => x1.clone(),
                    "x2" => x2.clone(),
                }
                .unwrap();
                run_sphet(
                    &Dataset::new(df),
                    "y",
                    &XS,
                    &w,
                    config(SphetModel::SpatialError, SphetSE::Standard, het),
                )
                .unwrap()
            };
            let base = fit(1.0);
            for s in [1e-6, 1e6] {
                let r = fit(s);
                assert!(
                    Tol::new(1e-10, 0.0).allows(r.rho.unwrap(), base.rho.unwrap()),
                    "het={het} s={s}"
                );
                assert!(
                    r.iterations.abs_diff(base.iterations) <= 1,
                    "{} vs {}",
                    r.iterations,
                    base.iterations
                );
                for (a, b) in r.coefficients.iter().zip(&base.coefficients) {
                    assert!(Tol::new(1e-9, 0.0).allows(*a, b * s));
                }
            }
        }
    }

    #[test]
    fn test_sphet_rejects_self_neighbours_for_gm() {
        let (d, _) = create_test_data();
        let nb = Neighbors::from_indices((0..25).map(|i| vec![i, (i + 1) % 25]).collect());
        let w = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        for model in [SphetModel::SpatialError, SphetModel::SARAR] {
            let e =
                run_sphet(&d, "y", &["x"], &w, config(model, SphetSE::Standard, true)).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("diagonal")),
                "{e}"
            );
        }
    }

    #[test]
    fn test_validate_sphet_sem_se_hom() {
        sem_se_case(false, "sem_hom");
    }

    #[test]
    fn test_validate_sphet_sem_se_het() {
        sem_se_case(true, "sem_het");
    }

    fn sem_se_case(het: bool, prefix: &str) {
        // The former "robust" and HAC arms were (sum e^2) A^3 n/(n-k) = (0.0485, 0.0063, 0.0079)
        // at the hom rho, against White (0.1582, 0.1164, 0.0828).
        let (case, ds, w, coords) = sphet_case();
        let mut check = Check::new(&case);
        let tri = SpatialHacKernel::Triangular;
        for (cfg, key) in [
            (
                config(SphetModel::SpatialError, SphetSE::Standard, het),
                "se_classical",
            ),
            (
                config(SphetModel::SpatialError, SphetSE::Robust, het),
                "se_white",
            ),
            (
                hac_config(SphetModel::SpatialError, het, &coords, 0.15, tri),
                "se_hac_tri_015",
            ),
        ] {
            let r = run_sphet(&ds, "y_sem", &XS, &w, cfg).unwrap();
            check
                .vector(
                    &format!("{prefix}_coef"),
                    &r.coefficients,
                    Tol::iterative(1e-8),
                )
                .vector(&format!("{prefix}_{key}"), &r.std_errors, Tol::se(N_SIM));
        }
        check.finish();
    }

    #[test]
    fn test_sphet_sem_white_se_is_scale_equivariant() {
        // SPAT-9: x1 * 0.1 must multiply the x1 SE by exactly 10 (it was 20.4 in the finding's repro).
        let (ds, w, _) = sim();
        let mut df = ds.df().clone();
        df.with_column(Column::new(
            "x1".into(),
            column(&ds, "x1")
                .iter()
                .map(|v| 0.1 * v)
                .collect::<Vec<f64>>(),
        ))
        .unwrap();
        let c = || config(SphetModel::SpatialError, SphetSE::Robust, true);
        let a = run_sphet(&ds, "y_sem", &XS, &w, c()).unwrap();
        let b = run_sphet(&Dataset::new(df), "y_sem", &XS, &w, c()).unwrap();
        assert!(Tol::new(1e-9, 0.0).allows(b.std_errors[1], 10.0 * a.std_errors[1]));
        assert!(Tol::new(1e-9, 0.0).allows(b.std_errors[0], a.std_errors[0]));
    }

    #[test]
    fn test_validate_sphet_rho_se() {
        // R .sphet_rho_se (the SEM with alpha_r = 0). The former formula,
        // sqrt(sigma2 n / (tr W^2 + tr W'W)), gave 1.699 for the SEM.
        let (case, ds, w, _) = sphet_case();
        let mut check = Check::new(&case);
        for (y, model, het, key) in [
            ("y_sem", SphetModel::SpatialError, false, "sem_hom_rho_se"),
            ("y_sem", SphetModel::SpatialError, true, "sem_het_rho_se"),
            ("y_sarar", SphetModel::SARAR, false, "sarar_rho_se"),
            ("y_sarar", SphetModel::SARAR, true, "sarar_het_rho_se"),
        ] {
            // the rho SE does not depend on se_type
            for se in [SphetSE::Standard, SphetSE::Robust] {
                let r = run_sphet(&ds, y, &XS, &w, config(model, se, het)).unwrap();
                check.scalar(key, r.rho_se.unwrap(), Tol::se(N_SIM));
            }
        }
        check.finish();
    }

    #[test]
    fn test_sphet_rho_se_scale_invariant() {
        // The former SE scaled with y.
        let (ds, w, _) = sim();
        let (x1, x2, y) = (column(&ds, "x1"), column(&ds, "x2"), column(&ds, "y_sem"));
        for het in [true, false] {
            let se = |s: f64| {
                let df = df! {
                    "y" => y.iter().map(|v| v * s).collect::<Vec<f64>>(),
                    "x1" => x1.clone(),
                    "x2" => x2.clone(),
                }
                .unwrap();
                run_sphet(
                    &Dataset::new(df),
                    "y",
                    &XS,
                    &w,
                    config(SphetModel::SpatialError, SphetSE::Standard, het),
                )
                .unwrap()
                .rho_se
                .unwrap()
            };
            let base = se(1.0);
            for s in [1e-6, 10.0, 1e6] {
                assert!(Tol::new(1e-8, 0.0).allows(se(s), base), "het={het} s={s}");
            }
        }
    }

    /// SEM: y = 1 + 0.8 x + u, u = (I - 0.4 W)^-1 e, e ~ N(0, 1), coordinates U(0, 1)^2,
    /// W = 5-NN row-standardised.
    fn simulated_sem(n: usize, seed: u64) -> (Dataset, SpatialWeights) {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal, Uniform};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let unif = Uniform::new(0.0, 1.0);
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|_| [unif.sample(&mut rng), unif.sample(&mut rng)])
            .collect();
        let w = knn5(&coords);
        let x: Vec<f64> = (0..n)
            .map(|_| {
                let v: f64 = StandardNormal.sample(&mut rng);
                v
            })
            .collect();
        let e = Array1::from_iter((0..n).map(|_| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v
        }));
        let mut u = e.clone();
        for _ in 0..200 {
            u = &e + &(0.4 * &w.lag(&u)); // 0.4^200: exact to rounding
        }
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.8 * x[i] + u[i]).collect();
        (Dataset::new(df! { "y" => y, "x" => x }.unwrap()), w)
    }

    #[test]
    fn test_sphet_rho_se_shrinks_with_n() {
        // Seeded property. The GM variance is O(1/n). Base-R Monte Carlo of this design: n = 150,
        // SD 0.10 / mean SE 0.11; n = 600, SD 0.0525 / mean SE 0.0523. The former SE was about 1.3
        // at both n = 225 and n = 900.
        for het in [true, false] {
            let se = |n: usize| {
                let (d, w) = simulated_sem(n, 7);
                run_sphet(
                    &d,
                    "y",
                    &["x"],
                    &w,
                    config(SphetModel::SpatialError, SphetSE::Standard, het),
                )
                .unwrap()
                .rho_se
                .unwrap()
            };
            let ratio = se(600) / se(150);
            assert!((0.3..0.75).contains(&ratio), "het={het}: SE ratio {ratio}");
        }
    }

    #[test]
    fn test_sphet_rho_se_is_nan_at_the_boundary_and_for_a_divergent_series() {
        let (ds, w, _) = sim();
        let (x1, x2, cx, cy) = (
            column(&ds, "x1"),
            column(&ds, "x2"),
            column(&ds, "cx"),
            column(&ds, "cy"),
        );
        let y: Vec<f64> = (0..N_SIM)
            .map(|i| 1.0 + 0.8 * x1[i] - 0.5 * x2[i] + cx[i] + cy[i])
            .collect();
        let df = df! { "y" => y, "x1" => x1, "x2" => x2 }.unwrap();
        let r = run_sphet(
            &Dataset::new(df),
            "y",
            &XS,
            &w,
            config(SphetModel::SpatialError, SphetSE::Standard, true),
        )
        .unwrap();
        assert!(r.rho_se.unwrap().is_nan() && r.rho_z.unwrap().is_nan());
        assert!(r.rho_p.unwrap().is_nan());
        assert!(
            r.warnings
                .iter()
                .any(|m| m.contains("SE of rho is not reported")),
            "{:?}",
            r.warnings
        );
        // Binary weights have spectral radius about 5: at rho = 0.9 the series diverges.
        let coords = coords_of(&ds);
        let pts: Vec<(f64, f64)> = coords.iter().map(|c| (c[0], c[1])).collect();
        let b = SpatialWeights::from_neighbors(&Neighbors::from_knn(&pts, 5), WeightStyle::Binary);
        let v = Array1::from(column(&ds, "x1"));
        assert!(solve_i_minus_rho_wt(&b, 0.9, &v).unwrap().is_none());
        let exact = solve_i_minus_rho_wt(&w, 0.4, &v).unwrap().unwrap();
        // (I - 0.4 W') exact = v
        let back = &exact - &(0.4 * &w.lag_transpose(&exact));
        for (a, b) in back.iter().zip(&v) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
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
