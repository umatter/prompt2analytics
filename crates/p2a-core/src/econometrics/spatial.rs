//! Spatial regression models.
//!
//! Provides maximum likelihood estimation of spatial autoregressive (SAR) and
//! spatial error (SEM) models, equivalent to R's `lagsarlm` and `errorsarlm`
//! from the spatialreg package.
//!
//! # Models
//!
//! ## Spatial Autoregressive Lag Model (SAR)
//!
//! y = ρWy + Xβ + ε
//!
//! where ρ is the spatial autoregressive parameter, W is the spatial weights
//! matrix, and ε ~ N(0, σ²I).
//!
//! ## Spatial Error Model (SEM)
//!
//! y = Xβ + u, where u = λWu + ε
//!
//! where λ is the spatial error parameter.
//!
//! # Likelihood
//!
//! Both models and their combination (SAC) maximise the concentrated log-likelihood with
//! log|I - ρW| = Re Σ log(1 - ρλ_i) over the **complex** eigenvalues of W (spatialreg method
//! "eigen"). The estimates are pinned to spatialreg's formulas written out in base R
//! (`validation/econometrics/spatial.md`).
//!
//! # Standard errors
//!
//! From the observed information of the full log-likelihood at the MLE: analytic for the SAR and
//! the spatial Durbin model, central differences of the analytic gradient for the SAC. The SEM
//! keeps errorsarlm's block-diagonal form: `sqrt(s2 diag((X*'X*)^-1))` for beta and the
//! (lambda, s2) block for lambda. spatialreg's method "eigen" reports the expected information
//! instead, which differs at finite n. An information matrix that is not positive definite at
//! the estimate is an error.
//!
//! # Data requirements
//!
//! Row i of the dataset is unit i of the weights, so the dataset has exactly one row per
//! unit of W and no row can be dropped. Missing values (null or NaN) in the outcome or a
//! regressor are refused with `EconError::NullValues` naming the column (spatialreg's
//! `na.action = na.fail`); remove the incomplete rows and rebuild W first. Integer and
//! Boolean (0/1) columns are accepted, text is `NonNumericColumn` and `±Inf` is
//! `NonFiniteValues`. The same rules hold for `sphet` and `spatialprobit`.
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
//! use p2a_core::econometrics::spatial::{run_sar, SarConfig};
//! use p2a_core::data::Dataset;
//!
//! // Assuming dataset is loaded with columns "y", "x1", "x2" and coordinates
//! let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
//!
//! // Create spatial weights from coordinates
//! let nb = Neighbors::from_knn(&coords, 5);
//! let mut listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
//!
//! // Fit SAR model
//! let result = run_sar(&dataset, "y", &["x1", "x2"], &mut listw, SarConfig::default())?;
//! println!("ρ = {}, p-value = {}", result.rho, result.rho_p);
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{cholesky, matrix_inverse, xtx, xty};
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, sym_inverse};
use crate::regression::ols::{OlsSolve, normal_equations_inverse, solve_ols};
use crate::spatial::SpatialWeights;
use crate::spatial::weights::WeightsSpectrum;
use crate::traits::estimator::normal_p_value_two_sided;

/// Outcome `y` and design `[1, X]` (intercept first) of a spatial cross-section model (SAR,
/// SEM, SAC, sphet, spatial probit).
///
/// Row i of the dataset is unit i of `listw`, so no row can be dropped. Every column is read
/// through `ColumnRequest` with `NaPolicy::Fail` (spatialreg/spdep: `na.action = na.fail`):
/// - a missing value (null or NaN) is `EconError::NullValues` for the first such column,
///   `y_col` first;
/// - Integer and Boolean (0/1) columns are accepted;
/// - text is `NonNumericColumn`, and `±Inf` is `NonFiniteValues`.
///
/// `n != listw.n()` is `InvalidSpecification` ("Dataset has {n} observations but weights
/// matrix has {m} observations"); that check runs first.
pub(crate) fn spatial_model_data(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &SpatialWeights,
) -> EconResult<(Array1<f64>, Array2<f64>)> {
    let df = dataset.df();
    let n = df.height();
    if n != listw.n() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dataset has {} observations but weights matrix has {} observations",
                n,
                listw.n()
            ),
        });
    }
    let cols = ColumnRequest::new(df)
        .numeric(y_col)
        .numerics(x_cols)
        .na(NaPolicy::Fail)
        .extract()?;
    Ok((cols.numeric_array(y_col), cols.matrix(x_cols, true)?.data))
}

/// spatialreg (lagsarlm, errorsarlm, sacsarlm) stops with "There are aliased coefficients in the
/// model"; so does this crate, naming the columns.
fn aliased_error(aliased: &[usize], names: &[String]) -> EconError {
    EconError::PerfectMulticollinearity {
        columns: aliased
            .iter()
            .filter_map(|&j| names.get(j).cloned())
            .collect(),
    }
}

/// The regressions of y and Wy on X behind the concentrated SAR and SDM likelihood (Ord 1975):
/// beta(rho) = b_y - rho b_wy and e(rho) = e_y - rho e_wy, linear in rho.
struct OrdRegressions {
    b_y: Array1<f64>,
    b_wy: Array1<f64>,
    e_y: Array1<f64>,
    e_wy: Array1<f64>,
}

/// One factorisation of X for both regressions: the normal equations on a well-conditioned
/// design, the pivoted QR (which decides rank and aliasing) otherwise.
fn ord_regressions(
    x: &Array2<f64>,
    y: &Array1<f64>,
    wy: &Array1<f64>,
    names: &[String],
) -> EconResult<OrdRegressions> {
    let (b_y, b_wy) = match normal_equations_inverse(x.view())? {
        Some(normal) => {
            let mut ys = Array2::zeros((y.len(), 2));
            ys.column_mut(0).assign(y);
            ys.column_mut(1).assign(wy);
            let b = normal.coefficients_multi(x.view(), ys.view());
            (b.column(0).to_owned(), b.column(1).to_owned())
        }
        None => {
            let qr = Qr::decompose(x.view(), LM_QR_TOL)?;
            if qr.rank() < x.ncols() {
                return Err(aliased_error(&qr.aliased(), names));
            }
            (qr.coefficients(y.view())?, qr.coefficients(wy.view())?)
        }
    };
    let e_y = y - &x.dot(&b_y);
    let e_wy = wy - &x.dot(&b_wy);
    Ok(OrdRegressions {
        b_y,
        b_wy,
        e_y,
        e_wy,
    })
}

/// The final least-squares fit on the transformed system of the SEM and the SAC.
fn transformed_fit(
    x_star: &Array2<f64>,
    y_star: &Array1<f64>,
    names: &[String],
) -> EconResult<OlsSolve> {
    let fit = solve_ols(x_star.view(), y_star.view())?;
    if fit.rank < x_star.ncols() {
        return Err(aliased_error(&fit.aliased, names));
    }
    Ok(fit)
}

/// Refuses an optimiser tolerance that is not finite and positive, and `max_iter == 0`.
/// Shared by the cross-section estimators here and the spatial panel ones in `splm`.
pub(crate) fn validate_optimizer(tol: f64, max_iter: usize) -> EconResult<()> {
    if !(tol.is_finite() && tol > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("tol must be a finite positive number, got {tol}"),
        });
    }
    if max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1".to_string(),
        });
    }
    Ok(())
}

/// Configuration for SAR model estimation.
#[derive(Debug, Clone)]
pub struct SarConfig {
    /// Include spatially lagged covariates (Spatial Durbin Model)
    pub durbin: bool,
    /// Tolerance for optimization
    pub tol: f64,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Compute spatial impacts (SAR and spatial Durbin model)
    pub compute_impacts: bool,
}

impl Default for SarConfig {
    fn default() -> Self {
        Self {
            durbin: false,
            tol: 1e-8,
            max_iter: 100,
            compute_impacts: true,
        }
    }
}

/// Configuration for SEM model estimation.
#[derive(Debug, Clone)]
pub struct SemConfig {
    /// Tolerance for optimization
    pub tol: f64,
    /// Maximum iterations for optimization
    pub max_iter: usize,
}

impl Default for SemConfig {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 100,
        }
    }
}

/// Spatial impacts: direct, indirect, and total effects (spatialreg's exact impacts).
///
/// For the SAR and the spatial Durbin model, the interpretation of coefficients is complicated
/// by the feedback loop (Wy appears on both sides). With S_k = (I - ρW)^{-1} (β_k I + θ_k W)
/// (θ_k = 0 for the SAR), the direct effect is the mean of diag(S_k), the total effect the mean
/// row sum of S_k and the indirect effect their difference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialImpacts {
    /// Direct effects, mean(diag(S_k))
    pub direct: Vec<f64>,
    /// Indirect effects (total - direct)
    pub indirect: Vec<f64>,
    /// Total effects, mean(rowSums(S_k))
    pub total: Vec<f64>,
    /// Variable names
    pub var_names: Vec<String>,
    /// Non-empty only when I - ρW is singular at the estimate (min |1 - ρλ| at or below 1e-10) or
    /// the total effect could not be solved; the effects are then `NaN`.
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Result from SAR (Spatial Autoregressive Lag) model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarResult {
    /// Regression coefficients (β)
    pub coefficients: Vec<f64>,
    /// Coefficient names
    pub coef_names: Vec<String>,
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// Z-values
    pub z_values: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,

    /// Spatial autoregressive coefficient (ρ)
    pub rho: f64,
    /// Standard error of ρ
    pub rho_se: f64,
    /// Z-value for ρ
    pub rho_z: f64,
    /// P-value for ρ
    pub rho_p: f64,

    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Residual variance (σ²)
    pub sigma2: f64,

    /// Residuals
    #[serde(skip)]
    pub residuals: Array1<f64>,
    /// Fitted values
    #[serde(skip)]
    pub fitted: Array1<f64>,

    /// Spatial impacts (if computed)
    pub impacts: Option<SpatialImpacts>,

    /// Number of observations
    pub n_obs: usize,
    /// Degrees of freedom
    pub df: usize,

    /// Whether the Spatial Durbin Model was estimated
    pub is_durbin: bool,
}

/// Result from SEM (Spatial Error Model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemResult {
    /// Regression coefficients (β)
    pub coefficients: Vec<f64>,
    /// Coefficient names
    pub coef_names: Vec<String>,
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// Z-values
    pub z_values: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,

    /// Spatial error coefficient (λ)
    pub lambda: f64,
    /// Standard error of λ
    pub lambda_se: f64,
    /// Z-value for λ
    pub lambda_z: f64,
    /// P-value for λ
    pub lambda_p: f64,

    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
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
}

/// Fit a Spatial Autoregressive Lag (SAR) model.
///
/// Estimates the model: y = ρWy + Xβ + ε
///
/// Uses concentrated maximum likelihood estimation where ρ is found
/// via univariate optimization and β is then computed via GLS.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the variables
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `listw` - Spatial weights matrix (will be modified to cache eigenvalues)
/// * `config` - Model configuration
///
/// # Returns
///
/// SAR estimation results
pub fn run_sar(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SarConfig,
) -> EconResult<SarResult> {
    validate_optimizer(config.tol, config.max_iter)?;
    let (y, x) = spatial_model_data(dataset, y_col, x_cols, listw)?;
    let n = y.len();
    let k = x.ncols(); // intercept + regressors

    // Add spatially lagged X for Spatial Durbin Model
    let x_final = if config.durbin {
        let mut x_durbin = Array2::zeros((n, k + x_cols.len()));
        for i in 0..n {
            for j in 0..k {
                x_durbin[[i, j]] = x[[i, j]];
            }
        }
        // Add WX for each non-intercept column
        for (j, _) in x_cols.iter().enumerate() {
            let x_col = x.column(j + 1).to_owned();
            let wx_col = listw.lag(&x_col);
            for i in 0..n {
                x_durbin[[i, k + j]] = wx_col[i];
            }
        }
        x_durbin
    } else {
        x
    };

    let k_final = x_final.ncols();

    // Build coefficient names
    let mut coef_names = vec!["(Intercept)".to_string()];
    for &name in x_cols {
        coef_names.push(name.to_string());
    }
    if config.durbin {
        for &name in x_cols {
            coef_names.push(format!("lag.{}", name));
        }
    }

    // Compute Wy
    let wy = listw.lag(&y);

    let ord = ord_regressions(&x_final, &y, &wy, &coef_names)?;
    let spectrum = listw.try_spectrum()?;

    // Get valid range for ρ
    let (rho_min, rho_max) = listw.try_rho_range()?;
    let rho_min = rho_min.max(-0.999);
    let rho_max = rho_max.min(0.999);

    // Optimize concentrated log-likelihood over ρ
    let (rho_opt, _ll_opt) = optimize_rho_sar(
        &ord,
        &spectrum,
        rho_min,
        rho_max,
        config.tol,
        config.max_iter,
    )?;

    // Final estimates at the optimal ρ; the regression of y - ρWy on X is linear in ρ
    let beta = &ord.b_y - &(rho_opt * &ord.b_wy);
    let residuals = &ord.e_y - &(rho_opt * &ord.e_wy);
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let sigma2 = rss / n as f64;

    // Fitted values in original scale
    let fitted = &y - &residuals;

    // Log-likelihood at convergence
    let log_det = spectrum.log_det(rho_opt);
    let ll = -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln()) + log_det;

    // Standard errors from the observed information at the MLE
    let info = sar_observed_information(&x_final, &wy, &residuals, sigma2, rho_opt, &spectrum);
    let se = information_se(
        &info,
        if config.durbin { "SDM" } else { "SAR" },
        k_final + 1,
    )?;
    let se_beta = Array1::from_iter(se[..k_final].iter().copied());
    let se_rho = se[k_final];

    // Z-values and p-values
    let z_values: Vec<f64> = beta
        .iter()
        .zip(se_beta.iter())
        .map(|(&b, &se)| b / se)
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    let rho_z = rho_opt / se_rho;
    let rho_p = normal_p_value_two_sided(rho_z);

    // AIC and BIC
    let n_params = k_final + 2; // beta + rho + sigma2
    let aic = -2.0 * ll + 2.0 * n_params as f64;
    let bic = -2.0 * ll + (n_params as f64) * (n as f64).ln();

    // Compute impacts if requested
    let impacts = if config.compute_impacts {
        let p = x_cols.len();
        let b = beta.to_vec();
        let lags = config.durbin.then(|| &b[1 + p..1 + 2 * p]);
        Some(impacts_exact(
            &b[1..1 + p],
            lags,
            rho_opt,
            listw,
            &coef_names[1..1 + p],
        )?)
    } else {
        None
    };

    Ok(SarResult {
        coefficients: beta.to_vec(),
        coef_names,
        std_errors: se_beta.to_vec(),
        z_values,
        p_values,
        rho: rho_opt,
        rho_se: se_rho,
        rho_z,
        rho_p,
        log_likelihood: ll,
        aic,
        bic,
        sigma2,
        residuals,
        fitted,
        impacts,
        n_obs: n,
        df: n - k_final - 1,
        is_durbin: config.durbin,
    })
}

/// Fit a Spatial Error Model (SEM).
///
/// Estimates the model: y = Xβ + u, where u = λWu + ε
///
/// Uses concentrated maximum likelihood estimation.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the variables
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `listw` - Spatial weights matrix
/// * `config` - Model configuration
///
/// # Returns
///
/// SEM estimation results
pub fn run_sem(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SemConfig,
) -> EconResult<SemResult> {
    validate_optimizer(config.tol, config.max_iter)?;
    let (y, x) = spatial_model_data(dataset, y_col, x_cols, listw)?;
    let n = y.len();
    let k = x.ncols(); // intercept + regressors

    // Coefficient names
    let mut coef_names = vec!["(Intercept)".to_string()];
    for &name in x_cols {
        coef_names.push(name.to_string());
    }

    let spectrum = listw.try_spectrum()?;

    // Get valid range for λ
    let (lambda_min, lambda_max) = listw.try_rho_range()?;
    let lambda_min = lambda_min.max(-0.999);
    let lambda_max = lambda_max.min(0.999);

    // Optimize concentrated log-likelihood over λ
    let (lambda_opt, _ll_opt) = optimize_lambda_sem(
        &y,
        &x,
        listw,
        &spectrum,
        lambda_min,
        lambda_max,
        config.tol,
        config.max_iter,
    )?;

    // Compute final estimates at optimal λ
    // Transform: (I - λW)y = (I - λW)X β + ε
    let y_star = listw.transform_y(&y, lambda_opt);
    let x_star = listw.transform_x(&x, lambda_opt);

    // β = (X*'X*)^{-1} X*'y*
    let fit = transformed_fit(&x_star, &y_star, &coef_names)?;
    let beta = fit.beta;
    let rss: f64 = fit.residuals.iter().map(|&r| r * r).sum();
    let sigma2 = rss / n as f64;

    // Residuals in original space
    let fitted = x.dot(&beta);
    let residuals = &y - &fitted;

    // Log-likelihood
    let log_det = spectrum.log_det(lambda_opt);
    let ll = -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln()) + log_det;

    // Standard errors: errorsarlm's sqrt(s2 diag((X*'X*)^-1)) for beta, the (lambda, s2) block of
    // the observed information for lambda
    let se_beta: Array1<f64> = fit.unscaled.diag().mapv(|v| (sigma2 * v).sqrt());
    let wu = listw.lag(&residuals);
    let eps_wu = fit.residuals.dot(&wu);
    let info = ndarray::array![
        [
            spectrum.trace_w_inverse_squared(lambda_opt) + wu.dot(&wu) / sigma2,
            eps_wu / (sigma2 * sigma2)
        ],
        [
            eps_wu / (sigma2 * sigma2),
            -(n as f64) / (2.0 * sigma2 * sigma2) + rss / sigma2.powi(3)
        ],
    ];
    let se_lambda = information_se(&info, "SEM", 1)?[0];

    // Z-values and p-values
    let z_values: Vec<f64> = beta
        .iter()
        .zip(se_beta.iter())
        .map(|(&b, &se)| b / se)
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    let lambda_z = lambda_opt / se_lambda;
    let lambda_p = normal_p_value_two_sided(lambda_z);

    // AIC and BIC
    let n_params = k + 2; // beta + lambda + sigma2
    let aic = -2.0 * ll + 2.0 * n_params as f64;
    let bic = -2.0 * ll + (n_params as f64) * (n as f64).ln();

    Ok(SemResult {
        coefficients: beta.to_vec(),
        coef_names,
        std_errors: se_beta.to_vec(),
        z_values,
        p_values,
        lambda: lambda_opt,
        lambda_se: se_lambda,
        lambda_z,
        lambda_p,
        log_likelihood: ll,
        aic,
        bic,
        sigma2,
        residuals,
        fitted,
        n_obs: n,
        df: n - k - 1,
    })
}

/// Dataset-based wrapper for SAR estimation.
pub fn run_sar_dataset(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SarConfig,
) -> EconResult<SarResult> {
    run_sar(dataset, y_col, x_cols, listw, config)
}

/// Dataset-based wrapper for SEM estimation.
pub fn run_sem_dataset(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SemConfig,
) -> EconResult<SemResult> {
    run_sem(dataset, y_col, x_cols, listw, config)
}

// ============================================================================
// SAC (SARAR) Model - Combined spatial lag and spatial error
// ============================================================================

/// Configuration for SAC (SARAR) model estimation.
#[derive(Debug, Clone)]
pub struct SacConfig {
    /// Tolerance for optimization
    pub tol: f64,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Grid search resolution for initial parameter search
    pub grid_resolution: usize,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            max_iter: 100,
            grid_resolution: 10,
        }
    }
}

/// Result from SAC (SARAR) model - combined spatial lag and error.
///
/// Model: y = ρWy + Xβ + u, where u = λWu + ε
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacResult {
    /// Regression coefficients (β)
    pub coefficients: Vec<f64>,
    /// Coefficient names
    pub coef_names: Vec<String>,
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// Z-values
    pub z_values: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,

    /// Spatial lag coefficient (ρ)
    pub rho: f64,
    /// Standard error of ρ
    pub rho_se: f64,
    /// Z-value for ρ
    pub rho_z: f64,
    /// P-value for ρ
    pub rho_p: f64,

    /// Spatial error coefficient (λ)
    pub lambda: f64,
    /// Standard error of λ
    pub lambda_se: f64,
    /// Z-value for λ
    pub lambda_z: f64,
    /// P-value for λ
    pub lambda_p: f64,

    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
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
}

/// Fit a SAC (SARAR) model - combined spatial lag and spatial error.
///
/// Estimates the model: y = ρWy + Xβ + u, where u = λWu + ε
///
/// Uses 2D concentrated maximum likelihood estimation with grid search
/// for initial values followed by coordinate descent optimization.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the variables
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `listw` - Spatial weights matrix (same W used for both lag and error)
/// * `config` - Model configuration
///
/// # Returns
///
/// SAC estimation results
pub fn run_sac(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SacConfig,
) -> EconResult<SacResult> {
    validate_optimizer(config.tol, config.max_iter)?;
    let (y, x) = spatial_model_data(dataset, y_col, x_cols, listw)?;
    let n = y.len();
    let k = x.ncols(); // intercept + regressors

    // Coefficient names
    let mut coef_names = vec!["(Intercept)".to_string()];
    for &name in x_cols {
        coef_names.push(name.to_string());
    }

    // Pre-compute eigenvalues
    let spectrum = listw.try_spectrum()?;

    // Get valid range for ρ and λ
    let (param_min, param_max) = listw.try_rho_range()?;
    let param_min = param_min.max(-0.99);
    let param_max = param_max.min(0.99);

    // Pre-compute Wy, W²y, and WX
    let wy = listw.lag(&y);
    let wwy = listw.lag(&wy); // W²y = W(Wy)
    let mut wx = Array2::zeros((n, k));
    for col in 0..k {
        let x_col = x.column(col).to_owned();
        let wx_col = listw.lag(&x_col);
        for i in 0..n {
            wx[[i, col]] = wx_col[i];
        }
    }

    // Optimize using coordinate descent with grid search initialization
    let (rho_opt, lambda_opt, _ll_opt) = optimize_sac(
        &y,
        &wy,
        &wwy,
        &x,
        &wx,
        &spectrum,
        param_min,
        param_max,
        config.tol,
        config.max_iter,
        config.grid_resolution,
    )?;

    // Compute final estimates at optimal (ρ, λ)
    // Transform: (I - λW)(I - ρW)y = (I - λW)Xβ + ε
    let n_f64 = n as f64;

    // y* = (I - λW)[(I - ρW)y] = (I - λW)[y - ρWy]
    let y_rho = &y - rho_opt * &wy;
    let y_star = &y_rho - lambda_opt * &listw.lag(&y_rho);

    // X* = (I - λW)X
    let x_star = &x - lambda_opt * &wx;

    // β = (X*'X*)^{-1} X*'y*
    let fit = transformed_fit(&x_star, &y_star, &coef_names)?;
    let beta = fit.beta;
    let rss: f64 = fit.residuals.iter().map(|&r| r * r).sum();
    let sigma2 = rss / n_f64;

    // Residuals in original space
    let fitted = x.dot(&beta);
    let residuals = &y - &fitted - rho_opt * &wy;

    // Log-likelihood at convergence
    let ll = -0.5 * n_f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln())
        + spectrum.log_det(rho_opt)
        + spectrum.log_det(lambda_opt);

    // Standard errors from the observed information at the MLE
    let theta: Vec<f64> = beta
        .iter()
        .copied()
        .chain([rho_opt, lambda_opt, sigma2])
        .collect();
    let info = sac_observed_information(&y, &wy, &wwy, &x, &wx, &spectrum, &theta)?;
    let se = information_se(&info, "SAC", k + 2)?;
    let se_beta = Array1::from_iter(se[..k].iter().copied());
    let (se_rho, se_lambda) = (se[k], se[k + 1]);

    // Z-values and p-values
    let z_values: Vec<f64> = beta
        .iter()
        .zip(se_beta.iter())
        .map(|(&b, &se)| b / se)
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    let rho_z = rho_opt / se_rho;
    let rho_p = normal_p_value_two_sided(rho_z);

    let lambda_z = lambda_opt / se_lambda;
    let lambda_p = normal_p_value_two_sided(lambda_z);

    // AIC and BIC
    let n_params = k + 3; // beta + rho + lambda + sigma2
    let aic = -2.0 * ll + 2.0 * n_params as f64;
    let bic = -2.0 * ll + (n_params as f64) * (n_f64).ln();

    Ok(SacResult {
        coefficients: beta.to_vec(),
        coef_names,
        std_errors: se_beta.to_vec(),
        z_values,
        p_values,
        rho: rho_opt,
        rho_se: se_rho,
        rho_z,
        rho_p,
        lambda: lambda_opt,
        lambda_se: se_lambda,
        lambda_z,
        lambda_p,
        log_likelihood: ll,
        aic,
        bic,
        sigma2,
        residuals,
        fitted,
        n_obs: n,
        df: n - k - 2,
    })
}

/// Dataset-based wrapper for SAC estimation.
pub fn run_sac_dataset(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    listw: &mut SpatialWeights,
    config: SacConfig,
) -> EconResult<SacResult> {
    run_sac(dataset, y_col, x_cols, listw, config)
}

/// Optimize spatial parameters (ρ, λ) for SAC/SARAR model.
///
/// The Spatial Autoregressive Combined (SAC) model has both spatial lag
/// and spatial error components:
///
///   y = ρWy + Xβ + u, where u = λWu + ε
///
/// # Likelihood
/// Maximizes the concentrated log-likelihood:
///
///   L_c(ρ,λ) = const - (n/2)ln(σ²(ρ,λ)) + ln|I - ρW| + ln|I - λW|
///
/// # Algorithm
/// 1. Pre-compute Wy, W²y (= WWy), and WX (constant across iterations)
/// 2. Grid search over (ρ, λ) parameter space for initial values
/// 3. Coordinate descent refinement:
///    - Fix λ, optimize ρ
///    - Fix ρ, optimize λ
///    - Repeat until convergence
/// 4. For each (ρ, λ) candidate, compute transformed variables:
///    - y* = y - ρWy - λ(Wy - ρW²y)
///    - X* = X - λWX
///    - OLS on transformed system for β
///
/// # Efficiency
/// Pre-computing W²y avoids the expensive sparse matrix product W(Wy).
/// Using eigenvalues for log-determinants avoids matrix decomposition.
///
/// # Arguments
/// * `y` - Outcome vector (n × 1)
/// * `wy` - Pre-computed Wy (n × 1)
/// * `wwy` - Pre-computed W²y (n × 1)
/// * `x` - Design matrix (n × k)
/// * `wx` - Pre-computed WX (n × k)
/// * `spectrum` - Eigenvalues of W for the log-determinants
/// * `param_min` - Lower bound for both ρ and λ
/// * `param_max` - Upper bound for both ρ and λ
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum coordinate descent iterations
/// * `grid_resolution` - Number of grid points per dimension for initial search
///
/// # Returns
/// Tuple of (ρ_optimal, λ_optimal, log_likelihood)
fn optimize_sac(
    y: &Array1<f64>,
    wy: &Array1<f64>,
    wwy: &Array1<f64>, // W²y = W(Wy)
    x: &Array2<f64>,
    wx: &Array2<f64>,
    spectrum: &WeightsSpectrum,
    param_min: f64,
    param_max: f64,
    tol: f64,
    max_iter: usize,
    grid_resolution: usize,
) -> EconResult<(f64, f64, f64)> {
    let n = y.len();
    let n_f64 = n as f64;
    let const_term = 0.5 * n_f64 * (1.0 + (2.0 * std::f64::consts::PI).ln());

    // Negative log-likelihood function
    // For SAC model: y* = (I - λW)(I - ρW)y = (I - λW)(y - ρWy)
    //                   = y - ρWy - λWy + λρW²y
    //                   = y - ρWy - λ(Wy - ρW²y)
    // With pre-computed wy = Wy and wwy = W²y:
    //   y* = y - ρ*wy - λ*(wy - ρ*wwy)
    let neg_ll = |rho: f64, lambda: f64| -> f64 {
        if rho <= param_min || rho >= param_max || lambda <= param_min || lambda >= param_max {
            return f64::INFINITY;
        }

        // Exact computation of y* using pre-computed wy and wwy
        // W(y - ρWy) = Wy - ρW²y = wy - ρ*wwy
        let wy_rho = wy - rho * wwy;
        let y_star: Array1<f64> = y - rho * wy - lambda * &wy_rho;

        // X* = (I - λW)X = X - λWX
        let x_star = x - lambda * wx;

        // β = (X*'X*)^{-1} X*'y*; X is of full rank, so (I - λW)X is on the feasible interval
        let xtx_star = xtx(&x_star.view());
        let xtx_star_inv = match matrix_inverse(&xtx_star.view()) {
            Ok(inv) => inv,
            Err(_) => return f64::INFINITY,
        };
        let xty_star = xty(&x_star.view(), &y_star);
        let beta = xtx_star_inv.dot(&xty_star);

        // RSS
        let fitted = x_star.dot(&beta);
        let residuals = &y_star - &fitted;
        let rss: f64 = residuals.iter().map(|&r| r * r).sum();
        let sigma2 = rss / n_f64;

        if sigma2 <= 0.0 {
            return f64::INFINITY;
        }

        // Log determinants from the complex eigenvalues
        let log_det = spectrum.log_det(rho) + spectrum.log_det(lambda);
        if !log_det.is_finite() {
            return f64::INFINITY;
        }

        const_term + 0.5 * n_f64 * sigma2.ln() - log_det
    };

    // Grid search for initial values
    let mut best_rho = 0.0;
    let mut best_lambda = 0.0;
    let mut best_ll = f64::INFINITY;

    let step = (param_max - param_min) / grid_resolution as f64;
    for i in 1..grid_resolution {
        crate::resource::checkpoint()?;
        let rho = param_min + i as f64 * step;
        for j in 1..grid_resolution {
            let lambda = param_min + j as f64 * step;
            let ll = neg_ll(rho, lambda);
            if ll < best_ll {
                best_ll = ll;
                best_rho = rho;
                best_lambda = lambda;
            }
        }
    }

    // Coordinate descent refinement
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    for _ in 0..max_iter {
        crate::resource::checkpoint()?;
        let old_rho = best_rho;
        let old_lambda = best_lambda;

        // Optimize rho with lambda fixed
        let mut a = param_min;
        let mut b = param_max;
        let mut c = b - (b - a) / phi;
        let mut d = a + (b - a) / phi;

        for _ in 0..50 {
            if (b - a).abs() < tol {
                break;
            }
            let fc = neg_ll(c, best_lambda);
            let fd = neg_ll(d, best_lambda);
            if fc < fd {
                b = d;
                d = c;
                c = b - (b - a) / phi;
            } else {
                a = c;
                c = d;
                d = a + (b - a) / phi;
            }
        }
        best_rho = (a + b) / 2.0;

        // Optimize lambda with rho fixed
        a = param_min;
        b = param_max;
        c = b - (b - a) / phi;
        d = a + (b - a) / phi;

        for _ in 0..50 {
            if (b - a).abs() < tol {
                break;
            }
            let fc = neg_ll(best_rho, c);
            let fd = neg_ll(best_rho, d);
            if fc < fd {
                b = d;
                d = c;
                c = b - (b - a) / phi;
            } else {
                a = c;
                c = d;
                d = a + (b - a) / phi;
            }
        }
        best_lambda = (a + b) / 2.0;

        // Check convergence
        if (best_rho - old_rho).abs() < tol && (best_lambda - old_lambda).abs() < tol {
            break;
        }
    }

    best_ll = neg_ll(best_rho, best_lambda);
    Ok((best_rho, best_lambda, -best_ll))
}

/// The analytic gradient of the full SAC log-likelihood at theta = (beta, rho, lambda, s2):
/// e = (I - λW) u, u = y - ρWy - Xβ; d/dβ = X*'e / s2 with X* = (I - λW) X;
/// d/dρ = -tr(W (I - ρW)^-1) + e'(I - λW) Wy / s2; d/dλ = -tr(W (I - λW)^-1) + e'Wu / s2;
/// d/ds2 = -n / (2 s2) + e'e / (2 s2²).
fn sac_gradient(
    y: &Array1<f64>,
    wy: &Array1<f64>,
    wwy: &Array1<f64>,
    x: &Array2<f64>,
    wx: &Array2<f64>,
    spectrum: &WeightsSpectrum,
    theta: &[f64],
) -> Array1<f64> {
    let k = x.ncols();
    let beta = ndarray::ArrayView1::from(&theta[..k]);
    let (rho, lambda, s2) = (theta[k], theta[k + 1], theta[k + 2]);
    let u = y - &(rho * wy) - x.dot(&beta);
    let w_u = wy - &(rho * wwy) - wx.dot(&beta);
    let e = &u - &(lambda * &w_u);
    let mut g = Array1::zeros(k + 3);
    let score = (x.t().dot(&e) - lambda * wx.t().dot(&e)) / s2;
    g.slice_mut(ndarray::s![..k]).assign(&score);
    g[k] = -spectrum.trace_w_inverse(rho) + (e.dot(wy) - lambda * e.dot(wwy)) / s2;
    g[k + 1] = -spectrum.trace_w_inverse(lambda) + e.dot(&w_u) / s2;
    g[k + 2] = -(y.len() as f64) / (2.0 * s2) + e.dot(&e) / (2.0 * s2 * s2);
    g
}

/// Minus the Hessian of the full SAC log-likelihood at `theta`, by central differences of its
/// analytic gradient with step 1e-5 max(|theta_i|, 1), symmetrised.
fn sac_observed_information(
    y: &Array1<f64>,
    wy: &Array1<f64>,
    wwy: &Array1<f64>,
    x: &Array2<f64>,
    wx: &Array2<f64>,
    spectrum: &WeightsSpectrum,
    theta: &[f64],
) -> EconResult<Array2<f64>> {
    let m = theta.len();
    let mut hessian = Array2::zeros((m, m));
    let mut at = theta.to_vec();
    for i in 0..m {
        crate::resource::checkpoint()?;
        let h = 1e-5 * theta[i].abs().max(1.0);
        at[i] = theta[i] + h;
        let plus = sac_gradient(y, wy, wwy, x, wx, spectrum, &at);
        at[i] = theta[i] - h;
        let minus = sac_gradient(y, wy, wwy, x, wx, spectrum, &at);
        at[i] = theta[i];
        hessian
            .column_mut(i)
            .assign(&((&plus - &minus) / (2.0 * h)));
    }
    Ok((&hessian + &hessian.t()) * -0.5)
}

/// The observed information of the full SAR/SDM log-likelihood at the MLE, theta = (beta, rho, s2),
/// e = y - ρWy - Xβ, A = W (I - ρW)^-1:
/// I_bb = X'X / s2; I_br = X'Wy / s2; I_bs = X'e / s2²; I_rr = tr(A²) + (Wy)'Wy / s2;
/// I_rs = e'Wy / s2²; I_ss = -n / (2 s2²) + e'e / s2³.
fn sar_observed_information(
    x: &Array2<f64>,
    wy: &Array1<f64>,
    e: &Array1<f64>,
    s2: f64,
    rho: f64,
    spectrum: &WeightsSpectrum,
) -> Array2<f64> {
    let k = x.ncols();
    let s4 = s2 * s2;
    let mut info = Array2::zeros((k + 2, k + 2));
    info.slice_mut(ndarray::s![..k, ..k])
        .assign(&(xtx(&x.view()) / s2));
    let x_wy = xty(&x.view(), wy);
    let x_e = xty(&x.view(), e);
    for j in 0..k {
        info[[j, k]] = x_wy[j] / s2;
        info[[k, j]] = info[[j, k]];
        info[[j, k + 1]] = x_e[j] / s4;
        info[[k + 1, j]] = info[[j, k + 1]];
    }
    info[[k, k]] = spectrum.trace_w_inverse_squared(rho) + wy.dot(wy) / s2;
    info[[k, k + 1]] = e.dot(wy) / s4;
    info[[k + 1, k]] = info[[k, k + 1]];
    info[[k + 1, k + 1]] = -(e.len() as f64) / (2.0 * s4) + e.dot(e) / (s2 * s4);
    info
}

/// sqrt(diag(I^-1)) of the first `keep` parameters of an observed information matrix `info`, which
/// must be positive definite (checked on the equilibrated matrix, so the units of y do not matter).
fn information_se(info: &Array2<f64>, model: &str, keep: usize) -> EconResult<Vec<f64>> {
    let not_definite = || {
        EconError::Computation(format!(
            "{model} information matrix is not positive definite at the estimate"
        ))
    };
    let d = info.nrows();
    if !info.iter().all(|v| v.is_finite()) || (0..d).any(|i| info[[i, i]] <= 0.0) {
        return Err(not_definite());
    }
    let mut scaled = info.clone();
    for i in 0..d {
        for j in 0..d {
            scaled[[i, j]] /= (info[[i, i]] * info[[j, j]]).sqrt();
        }
    }
    if cholesky(&scaled.view()).is_err() {
        return Err(not_definite());
    }
    let inverse = match sym_inverse(info.view(), SingularPolicy::Error, "information matrix") {
        Ok(inv) => inv.inverse,
        Err(e @ (EconError::Cancelled(_) | EconError::ResourceLimit(_))) => return Err(e),
        Err(_) => return Err(not_definite()),
    };
    let var = inverse.diag();
    if !var.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err(not_definite());
    }
    Ok(var.iter().take(keep).map(|v| v.sqrt()).collect())
}

// ============================================================================
// Internal optimization functions
// ============================================================================

/// Optimize spatial autoregressive parameter ρ for SAR model.
///
/// Maximizes the concentrated log-likelihood via golden section search:
///
///   L_c(ρ) = const - (n/2)ln(σ²(ρ)) + ln|I - ρW|
///
/// where σ²(ρ) = (1/n) * RSS(ρ) is the concentrated variance.
///
/// # Model
/// SAR: y = ρWy + Xβ + ε
///
/// # Algorithm
/// With e_y and e_wy the residuals of y and Wy on X (Ord 1975), RSS(ρ) = ||e_y - ρ e_wy||² is a
/// quadratic in ρ, so a likelihood evaluation costs three stored dot products and
/// log|I - ρW| over the complex eigenvalues. Golden section search finds the maximum.
///
/// # Arguments
/// * `ord` - The regressions of y and Wy on X
/// * `spectrum` - Eigenvalues of W for the log-determinant
/// * `rho_min` - Lower bound for ρ (typically 1/λ_min)
/// * `rho_max` - Upper bound for ρ (typically 1/λ_max)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Tuple of (ρ_optimal, log_likelihood)
fn optimize_rho_sar(
    ord: &OrdRegressions,
    spectrum: &WeightsSpectrum,
    rho_min: f64,
    rho_max: f64,
    tol: f64,
    max_iter: usize,
) -> EconResult<(f64, f64)> {
    let n_f64 = ord.e_y.len() as f64;
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio

    // Constant term in log-likelihood
    let const_term = 0.5 * n_f64 * (1.0 + (2.0 * std::f64::consts::PI).ln());

    let ey_ey = ord.e_y.dot(&ord.e_y);
    let ey_ewy = ord.e_y.dot(&ord.e_wy);
    let ewy_ewy = ord.e_wy.dot(&ord.e_wy);

    let mut a = rho_min;
    let mut b = rho_max;

    let neg_ll = |rho: f64| -> f64 {
        let rss = ey_ey - 2.0 * rho * ey_ewy + rho * rho * ewy_ewy;
        let sigma2 = rss / n_f64;

        if sigma2 <= 0.0 {
            return f64::INFINITY;
        }

        // Negative concentrated log-likelihood (for minimization)
        const_term + 0.5 * n_f64 * sigma2.ln() - spectrum.log_det(rho)
    };

    // Golden section search (each iteration is O(p) instead of O(n*k))
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;
    let mut fc = neg_ll(c);
    let mut fd = neg_ll(d);

    for _ in 0..max_iter {
        crate::resource::checkpoint()?;
        if (b - a).abs() < tol {
            break;
        }

        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / phi;
            fc = neg_ll(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / phi;
            fd = neg_ll(d);
        }
    }

    let rho_opt = (a + b) / 2.0;
    let ll_opt = -neg_ll(rho_opt);

    Ok((rho_opt, ll_opt))
}

/// Optimize spatial error parameter λ for SEM model.
///
/// Maximizes the concentrated log-likelihood via golden section search:
///
///   L_c(λ) = const - (n/2)ln(σ²(λ)) + ln|I - λW|
///
/// # Model
/// SEM: y = Xβ + u, where u = λWu + ε
/// Equivalently: (I - λW)y = (I - λW)Xβ + ε
///
/// # Algorithm
/// 1. Pre-compute W eigenvalues and Wy, WX (constant across iterations)
/// 2. For each λ candidate:
///    - Transform: y* = y - λWy, X* = X - λWX
///    - Compute β = (X*'X*)⁻¹ X*'y*
///    - Compute RSS and σ²
///    - Compute log|I - λW| = Σ ln(1 - λeᵢ) using eigenvalues eᵢ
/// 3. Use golden section search to find λ minimizing negative log-likelihood
///
/// # Efficiency
/// Pre-computing Wy and WX avoids repeated sparse matrix multiplications.
/// Only the linear combinations change during optimization.
///
/// # Arguments
/// * `y` - Outcome vector (n × 1)
/// * `x` - Design matrix (n × k)
/// * `listw` - Spatial weights (provides the lags)
/// * `spectrum` - Eigenvalues of W for the log-determinant
/// * `lambda_min` - Lower bound for λ
/// * `lambda_max` - Upper bound for λ
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Tuple of (λ_optimal, log_likelihood)
fn optimize_lambda_sem(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    spectrum: &WeightsSpectrum,
    lambda_min: f64,
    lambda_max: f64,
    tol: f64,
    max_iter: usize,
) -> EconResult<(f64, f64)> {
    let n = y.len();
    let n_f64 = n as f64;
    let k = x.ncols();
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    // Pre-compute Wy and WX (these don't change with lambda)
    let wy = listw.lag(y);
    let mut wx = Array2::zeros((n, k));
    for col in 0..k {
        let x_col = x.column(col).to_owned();
        let wx_col = listw.lag(&x_col);
        for i in 0..n {
            wx[[i, col]] = wx_col[i];
        }
    }

    // Constant term in log-likelihood
    let const_term = 0.5 * n_f64 * (1.0 + (2.0 * std::f64::consts::PI).ln());

    // === Pre-compute quadratic coefficients for SEM ===
    // X*(λ) = X - λ*WX, y*(λ) = y - λ*Wy
    //
    // X*'X* = X'X - λ*(X'WX + WX'X) + λ²*WX'WX  [k×k matrices]
    // X*'y* = X'y - λ*(X'Wy + WX'y) + λ²*WX'Wy   [k×1 vectors]
    //
    // Pre-computing these eliminates all O(n*k) operations from the inner loop.
    // Only k×k matrix operations remain (trivial for typical k=2-5).

    let xx = xtx(&x.view()); // X'X [k×k]
    let xwx = x.t().dot(&wx); // X'WX [k×k]
    let wxwx = wx.t().dot(&wx); // WX'WX [k×k]
    let xwx_plus_wxwx_sym = &xwx + &xwx.t(); // X'WX + WX'X [k×k, symmetric term]

    let xy = xty(&x.view(), y); // X'y [k×1]
    let xwy = xty(&x.view(), &wy); // X'Wy [k×1]
    let wxy = xty(&wx.view(), y); // WX'y [k×1]
    let xwy_plus_wxy = &xwy + &wxy; // X'Wy + WX'y [k×1]
    let wxwy = xty(&wx.view(), &wy); // WX'Wy [k×1]

    // For RSS computation, pre-compute dot products:
    // ||y*||² = y'y - 2λ*y'Wy + λ²*Wy'Wy
    let yy: f64 = y.iter().map(|&v| v * v).sum();
    let ywy: f64 = y.iter().zip(wy.iter()).map(|(&a, &b)| a * b).sum();
    let wywy: f64 = wy.iter().map(|&v| v * v).sum();

    let mut a = lambda_min;
    let mut b = lambda_max;

    let neg_ll = |lambda: f64| -> f64 {
        let l2 = lambda * lambda;

        // X*'X* = XX - λ*(X'WX + WX'X) + λ²*WX'WX
        let xtx_star = &xx - lambda * &xwx_plus_wxwx_sym + l2 * &wxwx;

        // X*'y* = X'y - λ*(X'Wy + WX'y) + λ²*WX'Wy
        let xty_star = &xy - lambda * &xwy_plus_wxy + l2 * &wxwy;

        // Solve k×k system for β (tiny for typical k=2-5); X is of full rank, so (I - λW)X is on
        // the feasible interval
        let xtx_inv = match matrix_inverse(&xtx_star.view()) {
            Ok(inv) => inv,
            Err(_) => return f64::INFINITY,
        };
        let beta = xtx_inv.dot(&xty_star);

        // RSS = ||y*||² - 2*β'*(X*'y*) + β'*(X*'X*)*β
        //     = ||y*||² - β'*(X*'y*)  [since β = (X*'X*)^{-1}(X*'y*)]
        let y_star_sq = yy - 2.0 * lambda * ywy + l2 * wywy;
        let beta_xty: f64 = beta.iter().zip(xty_star.iter()).map(|(&b, &x)| b * x).sum();
        let rss = y_star_sq - beta_xty;
        let sigma2 = rss / n_f64;

        if sigma2 <= 0.0 {
            return f64::INFINITY;
        }

        // Log determinant from the complex eigenvalues: O(p) scalar ops
        const_term + 0.5 * n_f64 * sigma2.ln() - spectrum.log_det(lambda)
    };

    // Golden section search (each iteration is now O(k³ + p) instead of O(n*k))
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;
    let mut fc = neg_ll(c);
    let mut fd = neg_ll(d);

    for _ in 0..max_iter {
        crate::resource::checkpoint()?;
        if (b - a).abs() < tol {
            break;
        }

        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / phi;
            fc = neg_ll(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / phi;
            fd = neg_ll(d);
        }
    }

    let lambda_opt = (a + b) / 2.0;
    let ll_opt = -neg_ll(lambda_opt);

    Ok((lambda_opt, ll_opt))
}

/// min |1 - ρλ| at or below which I - ρW is treated as singular (condition number of about 1e10
/// or more, where the dense inverse refuses and spatialreg's solve() stops).
const IMPACT_SINGULAR_TOL: f64 = 1e-10;

/// spatialreg's exact impacts (lagImpactsExact; mixedImpactsExact for the SDM, whose lagged
/// coefficients are `lags`).
///
/// With S_k = (I - ρW)^{-1} (β_k I + θ_k W), the direct effect is tr(S_k) / n from the complex
/// eigenvalues the fit has already cached. The total effect is 1'S_k 1 / n: a closed form when
/// every row of W sums to one value, otherwise one linear solve per term. There is no dense
/// inverse. When I - ρW is singular, or a solve fails, the effects are `NaN` with a warning;
/// cancellation and resource limits return their errors.
fn impacts_exact(
    slopes: &[f64],
    lags: Option<&[f64]>,
    rho: f64,
    listw: &mut SpatialWeights,
    var_names: &[String],
) -> EconResult<SpatialImpacts> {
    let spectrum = listw.try_spectrum()?;
    let n = listw.n();
    let n_f = n as f64;
    let p = slopes.len();
    let degraded = |why: String| SpatialImpacts {
        direct: vec![f64::NAN; p],
        indirect: vec![f64::NAN; p],
        total: vec![f64::NAN; p],
        var_names: var_names.to_vec(),
        warnings: vec![why],
    };

    let m = spectrum.min_modulus(rho);
    if m.is_nan() || m <= IMPACT_SINGULAR_TOL {
        return Ok(degraded(format!(
            "(I - rho W) is singular at rho = {rho} (min |1 - rho lambda| = {m:.1e}); direct, \
             indirect and total impacts are not reported"
        )));
    }

    let tr_a = spectrum.trace_inverse(rho);
    let tr_aw = lags.map_or(0.0, |_| spectrum.trace_w_inverse(rho));
    let (s1, s2) = match listw.constant_row_sum() {
        Some(c) => (n_f / (1.0 - rho * c), n_f * c / (1.0 - rho * c)),
        None => {
            let ones = Array1::ones(n);
            let solved = listw.solve_i_minus_rho_w(rho, &ones).and_then(|v| {
                let s1 = v.sum();
                if lags.is_some() {
                    let lagged = listw.lag(&ones);
                    Ok((s1, listw.solve_i_minus_rho_w(rho, &lagged)?.sum()))
                } else {
                    Ok((s1, 0.0))
                }
            });
            match solved {
                Ok(sums) => sums,
                Err(e @ (EconError::Cancelled(_) | EconError::ResourceLimit(_))) => return Err(e),
                Err(e) => {
                    return Ok(degraded(format!(
                        "(I - rho W) v = b could not be solved at rho = {rho}: {e}; direct, \
                         indirect and total impacts are not reported"
                    )));
                }
            }
        }
    };
    if ![tr_a, tr_aw, s1, s2].iter().all(|v| v.is_finite()) {
        return Ok(degraded(format!(
            "the traces of (I - rho W)^-1 are not finite at rho = {rho}; direct, indirect and \
             total impacts are not reported"
        )));
    }

    let theta = |j: usize| lags.map_or(0.0, |t| t[j]);
    let direct: Vec<f64> = (0..p)
        .map(|j| (slopes[j] * tr_a + theta(j) * tr_aw) / n_f)
        .collect();
    let total: Vec<f64> = (0..p)
        .map(|j| (slopes[j] * s1 + theta(j) * s2) / n_f)
        .collect();
    let indirect: Vec<f64> = total.iter().zip(&direct).map(|(t, d)| t - d).collect();

    Ok(SpatialImpacts {
        direct,
        indirect,
        total,
        var_names: var_names.to_vec(),
        warnings: Vec::new(),
    })
}

/// [`impacts_exact`] of a SAR coefficient vector that holds the intercept first.
#[cfg(test)]
fn compute_impacts(
    beta: &Array1<f64>,
    rho: f64,
    listw: &mut SpatialWeights,
    var_names: &[String],
) -> EconResult<SpatialImpacts> {
    let slopes: Vec<f64> = beta.iter().skip(1).copied().collect();
    impacts_exact(&slopes, None, rho, listw, var_names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Neighbors, WeightStyle};
    use polars::prelude::*;

    fn create_test_data() -> (Dataset, SpatialWeights) {
        // Create a simple spatial dataset
        // 4x4 grid of observations
        let n = 16;

        // Coordinates for a 4x4 grid
        let coords: Vec<(f64, f64)> = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i as f64, j as f64)))
            .collect();

        // Create neighbors (4-nearest neighbors)
        let nb = Neighbors::from_knn(&coords, 4);
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        // Generate y with spatial autocorrelation
        // y = 2 + 0.5*x + spatial_effect + noise
        let x: Vec<f64> = (0..n).map(|i| (i as f64) / 4.0).collect();
        let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 + 0.5 * xi).collect();

        // Add some spatial pattern (nearby values similar)
        for i in 0..n {
            let row = i / 4;
            let col = i % 4;
            y[i] += 0.3 * ((row + col) as f64);
        }

        let df = df! {
            "y" => &y,
            "x" => &x,
        }
        .unwrap();

        let dataset = Dataset::new(df);
        (dataset, listw)
    }

    #[test]
    fn test_sar_basic() {
        let (dataset, mut listw) = create_test_data();

        let config = SarConfig::default();
        let result = run_sar(&dataset, "y", &["x"], &mut listw, config).unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 16);
        assert!(result.rho > -1.0 && result.rho < 1.0);
        assert!(result.sigma2 > 0.0);
        assert!(result.coefficients.len() == 2); // Intercept + x

        // Check that fitted + residuals = y approximately
        // Note: This won't be exact due to the spatial lag
    }

    #[test]
    fn test_sem_basic() {
        let (dataset, mut listw) = create_test_data();

        let config = SemConfig::default();
        let result = run_sem(&dataset, "y", &["x"], &mut listw, config).unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 16);
        assert!(result.lambda > -1.0 && result.lambda < 1.0);
        assert!(result.sigma2 > 0.0);
        assert!(result.coefficients.len() == 2);
    }

    #[test]
    fn test_sar_impacts() {
        let (dataset, mut listw) = create_test_data();

        let config = SarConfig {
            compute_impacts: true,
            ..Default::default()
        };
        let result = run_sar(&dataset, "y", &["x"], &mut listw, config).unwrap();

        let impacts = result.impacts.unwrap();
        assert_eq!(impacts.var_names.len(), 1);
        assert_eq!(impacts.direct.len(), 1);
        assert_eq!(impacts.indirect.len(), 1);
        assert_eq!(impacts.total.len(), 1);

        // Total should be approximately direct + indirect
        let total_check = impacts.direct[0] + impacts.indirect[0];
        assert!((impacts.total[0] - total_check).abs() < 1e-10);
    }

    #[test]
    fn test_sar_durbin() {
        let (dataset, mut listw) = create_test_data();

        let config = SarConfig {
            durbin: true,
            compute_impacts: false,
            ..Default::default()
        };
        let result = run_sar(&dataset, "y", &["x"], &mut listw, config).unwrap();

        // Should have intercept, x, and lag.x
        assert_eq!(result.coefficients.len(), 3);
        assert!(result.is_durbin);
        assert!(result.coef_names.contains(&"lag.x".to_string()));
    }

    #[test]
    fn test_sac_basic() {
        let (dataset, mut listw) = create_test_data();

        let config = SacConfig::default();
        let result = run_sac(&dataset, "y", &["x"], &mut listw, config).unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 16);
        assert!(
            result.rho > -1.0 && result.rho < 1.0,
            "rho = {} out of bounds",
            result.rho
        );
        assert!(
            result.lambda > -1.0 && result.lambda < 1.0,
            "lambda = {} out of bounds",
            result.lambda
        );
        assert!(result.sigma2 > 0.0);
        assert_eq!(result.coefficients.len(), 2); // Intercept + x
        assert_eq!(result.coef_names.len(), 2);
        assert_eq!(result.coef_names[0], "(Intercept)");
        assert_eq!(result.coef_names[1], "x");
    }

    #[test]
    fn test_sac_standard_errors() {
        let (dataset, mut listw) = create_test_data();

        let config = SacConfig::default();
        let result = run_sac(&dataset, "y", &["x"], &mut listw, config).unwrap();

        // Standard errors should be positive
        for se in &result.std_errors {
            assert!(*se > 0.0, "Standard error should be positive, got {}", se);
        }
        assert!(result.rho_se > 0.0, "rho SE should be positive");
        assert!(result.lambda_se > 0.0, "lambda SE should be positive");

        // P-values should be in [0, 1]
        for p in &result.p_values {
            assert!(
                *p >= 0.0 && *p <= 1.0,
                "P-value should be in [0,1], got {}",
                p
            );
        }
        assert!(result.rho_p >= 0.0 && result.rho_p <= 1.0);
        assert!(result.lambda_p >= 0.0 && result.lambda_p <= 1.0);
    }

    #[test]
    fn test_sac_model_fit() {
        let (dataset, mut listw) = create_test_data();

        let config = SacConfig::default();
        let result = run_sac(&dataset, "y", &["x"], &mut listw, config).unwrap();

        // AIC and BIC should be finite
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());

        // Residuals and fitted should have correct length
        assert_eq!(result.residuals.len(), 16);
        assert_eq!(result.fitted.len(), 16);
    }

    #[test]
    fn test_sac_vs_sar_sem() {
        // SAC model should fit at least as well as SAR or SEM
        // since it's a generalization of both
        let (dataset, mut listw) = create_test_data();

        let sar_result = run_sar(
            &dataset,
            "y",
            &["x"],
            &mut listw.clone(),
            SarConfig {
                compute_impacts: false,
                ..Default::default()
            },
        )
        .unwrap();

        let sem_result = run_sem(
            &dataset,
            "y",
            &["x"],
            &mut listw.clone(),
            SemConfig::default(),
        )
        .unwrap();

        let sac_result = run_sac(&dataset, "y", &["x"], &mut listw, SacConfig::default()).unwrap();

        // SAC should have at least as good log-likelihood as either nested model
        // (allowing for some numerical tolerance)
        let min_nested_ll = sar_result.log_likelihood.min(sem_result.log_likelihood);
        // Note: Due to optimization, SAC might not always dominate, but should be close
        assert!(
            sac_result.log_likelihood >= min_nested_ll - 1.0,
            "SAC ll {} should be >= min(SAR ll {}, SEM ll {}) - 1.0",
            sac_result.log_likelihood,
            sar_result.log_likelihood,
            sem_result.log_likelihood
        );
    }

    use crate::data::columns::numeric_column;

    fn col(ds: &Dataset, name: &str) -> Vec<f64> {
        numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0
    }

    fn coords_of(ds: &Dataset) -> Vec<(f64, f64)> {
        col(ds, "cx").into_iter().zip(col(ds, "cy")).collect()
    }

    /// The case's W: 4 nearest neighbours on (cx, cy), row-standardised (spdep style "W").
    fn knn4(ds: &Dataset) -> SpatialWeights {
        SpatialWeights::from_neighbors(
            &Neighbors::try_from_knn(&coords_of(ds), 4).unwrap(),
            WeightStyle::RowStd,
        )
    }

    /// R: concentrated lagsarlm (Durbin = FALSE/TRUE) and errorsarlm likelihoods with the complex
    /// log-determinant, written out (spatialreg not installed); spatial_ml_weights.R, case
    /// spatial_sar_sdm_sem.
    #[test]
    fn test_validate_sar_sdm_sem_concentrated_ml() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("spatial_sar_sdm_sem");
        let x = ["x1", "x2"];
        let mut w = knn4(&ds);
        let sar = run_sar(&ds, "y", &x, &mut w, SarConfig::default()).unwrap();
        let durbin = SarConfig {
            durbin: true,
            ..Default::default()
        };
        let sdm = run_sar(&ds, "y", &x, &mut w, durbin).unwrap();
        let sem = run_sem(&ds, "y", &x, &mut w, SemConfig::default()).unwrap();
        let n = sar.n_obs;
        // Golden-section searches on flat concentrated likelihoods: the flatness bound is 1.3e-7
        // in rho, and an R replay of the search reaches 1e-8 relative. Residuals move by at most
        // max|de/drho| (2.24) x 1.3e-7.
        let est = Tol::iterative(1e-6);
        let imp = sar.impacts.as_ref().expect("SAR impacts");
        Check::new(&case)
            .scalar("sar_rho", sar.rho, est)
            .named("sar_beta", &sar.coef_names, &sar.coefficients, est)
            .scalar("sar_s2", sar.sigma2, Tol::stat(n))
            .scalar("sar_loglik", sar.log_likelihood, Tol::stat(n))
            .scalar("sar_aic", sar.aic, Tol::stat(n))
            .scalar("sar_bic", sar.bic, Tol::stat(n))
            .vector(
                "sar_residuals",
                &sar.residuals.to_vec(),
                Tol::new(1e-6, 1e-6),
            )
            .named("sar_impact_direct", &imp.var_names, &imp.direct, est)
            .named("sar_impact_indirect", &imp.var_names, &imp.indirect, est)
            .named("sar_impact_total", &imp.var_names, &imp.total, est)
            .scalar("sdm_rho", sdm.rho, est)
            .named("sdm_beta", &sdm.coef_names, &sdm.coefficients, est)
            .scalar("sdm_s2", sdm.sigma2, Tol::stat(n))
            .scalar("sdm_loglik", sdm.log_likelihood, Tol::stat(n))
            .scalar("sdm_aic", sdm.aic, Tol::stat(n))
            .scalar("sdm_bic", sdm.bic, Tol::stat(n))
            .scalar("sem_lambda", sem.lambda, est)
            .named("sem_beta", &sem.coef_names, &sem.coefficients, est)
            // errorsarlm rest.se = sqrt(s2 diag((X*'X*)^-1)), s2 = SSE / n
            .named("sem_beta_se", &sem.coef_names, &sem.std_errors, Tol::se(n))
            .named("sem_beta_z", &sem.coef_names, &sem.z_values, Tol::stat(n))
            .named(
                "sem_beta_p",
                &sem.coef_names,
                &sem.p_values,
                Tol::p_value(n),
            )
            .scalar("sem_s2", sem.sigma2, Tol::stat(n))
            .scalar("sem_loglik", sem.log_likelihood, Tol::stat(n))
            .scalar("sem_aic", sem.aic, Tol::stat(n))
            .scalar("sem_bic", sem.bic, Tol::stat(n))
            .finish();
        let y = col(&ds, "y");
        for i in 0..n {
            assert!(
                (sar.fitted[i] + sar.residuals[i] - y[i]).abs() <= 1e-12 * (1.0 + y[i].abs()),
                "row {i}"
            );
        }
        assert_eq!((sar.df, sdm.df, sem.df), (n - 4, n - 6, n - 4));
    }

    /// R: concentrated sacsarlm likelihood, BFGS then Newton on the analytic score
    /// (spatial_ml_weights.R).
    #[test]
    fn test_validate_sac_concentrated_ml() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("spatial_sar_sdm_sem");
        let mut w = knn4(&ds);
        let tight = SacConfig {
            tol: 1e-10,
            max_iter: 1000,
            ..Default::default()
        };
        let r = run_sac(&ds, "y", &["x1", "x2"], &mut w, tight).unwrap();
        let n = r.n_obs;
        // Coordinate descent with golden sections: an R replay reaches 3.1e-8 (rho) and 3.6e-7
        // (lambda = -0.209) at tol 1e-10.
        let it = Tol::iterative(1e-5);
        Check::new(&case)
            .scalar("sac_rho", r.rho, it)
            .scalar("sac_lambda", r.lambda, it)
            .named("sac_beta", &r.coef_names, &r.coefficients, it)
            .scalar("sac_s2", r.sigma2, it)
            .scalar("sac_loglik", r.log_likelihood, Tol::stat(n))
            .scalar("sac_aic", r.aic, Tol::stat(n))
            .scalar("sac_bic", r.bic, Tol::stat(n))
            .finish();
        // the package default (tol 1e-6): the replay reaches 1.4e-7 (rho), 1.6e-6 (lambda)
        let d = run_sac(&ds, "y", &["x1", "x2"], &mut w, SacConfig::default()).unwrap();
        Check::new(&case)
            .scalar("sac_rho", d.rho, Tol::iterative(1e-4))
            .scalar("sac_lambda", d.lambda, Tol::iterative(1e-4))
            .finish();
    }

    /// The dataset of the spatial_sar_sdm_sem case, read as a property test's data.
    fn spatial_sim() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/spatial_sim.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    /// `spatial_sim` with y multiplied by `scale`.
    fn scaled_y(ds: &Dataset, scale: f64) -> Dataset {
        let y: Vec<f64> = col(ds, "y").iter().map(|v| v * scale).collect();
        Dataset::new(df! { "y" => y, "x1" => col(ds, "x1"), "x2" => col(ds, "x2") }.unwrap())
    }

    /// R: minus the analytic Hessian of the full log-likelihood at the MLE (SAC: central
    /// differences of its analytic gradient), each cross-checked against stats::optimHess in the
    /// case (spatial_ml_weights.R). The convention reference `sar_se` (expected information,
    /// spatialreg's method "eigen") differs at finite n and is not compared.
    #[test]
    fn test_validate_spatial_ml_se_observed_information() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("spatial_sar_sdm_sem");
        let x = ["x1", "x2"];
        let mut w = knn4(&ds);
        let with_rho = |names: &[String], se: &[f64], last: &[(&str, f64)]| {
            let mut n = names.to_vec();
            let mut v = se.to_vec();
            for (name, value) in last {
                n.push((*name).to_string());
                v.push(*value);
            }
            (n, v)
        };
        // the estimates are good to the flatness bound of the likelihood (1.3e-7 in rho)
        let est = Tol::iterative(1e-6);
        let sar = run_sar(&ds, "y", &x, &mut w, SarConfig::default()).unwrap();
        let durbin = SarConfig {
            durbin: true,
            ..Default::default()
        };
        let sdm = run_sar(&ds, "y", &x, &mut w, durbin).unwrap();
        let sem = run_sem(&ds, "y", &x, &mut w, SemConfig::default()).unwrap();
        let tight = SacConfig {
            tol: 1e-10,
            max_iter: 1000,
            ..Default::default()
        };
        let sac = run_sac(&ds, "y", &x, &mut w, tight).unwrap();
        let (sar_n, sar_v) = with_rho(&sar.coef_names, &sar.std_errors, &[("rho", sar.rho_se)]);
        let (sdm_n, sdm_v) = with_rho(&sdm.coef_names, &sdm.std_errors, &[("rho", sdm.rho_se)]);
        let (sac_n, sac_v) = with_rho(
            &sac.coef_names,
            &sac.std_errors,
            &[("rho", sac.rho_se), ("lambda", sac.lambda_se)],
        );
        Check::new(&case)
            .named("sar_se_obs", &sar_n, &sar_v, est)
            .named("sdm_se_obs", &sdm_n, &sdm_v, est)
            .scalar("sem_lambda_se_obs", sem.lambda_se, est)
            // coordinate descent: lambda is good to 3.6e-7 relative at this tolerance
            .named("sac_se_obs", &sac_n, &sac_v, Tol::iterative(1e-5))
            .finish();
    }

    /// The SEs of a dimensionless rho or lambda do not depend on the units of y, and every beta SE
    /// scales with them. The estimates themselves move within the likelihood's flatness (1.3e-7
    /// in rho), so the comparison leaves room for that.
    #[test]
    fn test_spatial_ml_se_scale_with_y() {
        let ds = spatial_sim();
        let x = ["x1", "x2"];
        let mut w = knn4(&ds);
        let close = |a: f64, b: f64, tol: f64, what: &str| {
            assert!((a - b).abs() <= tol * b.abs(), "{what}: {a} vs {b}");
        };
        let tight = || SacConfig {
            tol: 1e-10,
            max_iter: 1000,
            ..Default::default()
        };
        let durbin = || SarConfig {
            durbin: true,
            ..Default::default()
        };
        let d10 = scaled_y(&ds, 10.0);
        let sar = (
            run_sar(&ds, "y", &x, &mut w, SarConfig::default()).unwrap(),
            run_sar(&d10, "y", &x, &mut w, SarConfig::default()).unwrap(),
        );
        let sdm = (
            run_sar(&ds, "y", &x, &mut w, durbin()).unwrap(),
            run_sar(&d10, "y", &x, &mut w, durbin()).unwrap(),
        );
        for (a, b) in [&sar, &sdm] {
            close(b.rho_se, a.rho_se, 1e-5, "rho se");
            for (u, v) in a.std_errors.iter().zip(&b.std_errors) {
                close(*v, 10.0 * u, 1e-5, "beta se");
            }
        }
        let sem = (
            run_sem(&ds, "y", &x, &mut w, SemConfig::default()).unwrap(),
            run_sem(&d10, "y", &x, &mut w, SemConfig::default()).unwrap(),
        );
        close(sem.1.lambda_se, sem.0.lambda_se, 1e-5, "lambda se");
        for (u, v) in sem.0.std_errors.iter().zip(&sem.1.std_errors) {
            close(*v, 10.0 * u, 1e-5, "sem beta se");
        }
        let sac = (
            run_sac(&ds, "y", &x, &mut w, tight()).unwrap(),
            run_sac(&d10, "y", &x, &mut w, tight()).unwrap(),
        );
        close(sac.1.rho_se, sac.0.rho_se, 1e-4, "sac rho se");
        close(sac.1.lambda_se, sac.0.lambda_se, 1e-4, "sac lambda se");
        for (u, v) in sac.0.std_errors.iter().zip(&sac.1.std_errors) {
            close(*v, 10.0 * u, 1e-4, "sac beta se");
        }
    }

    #[test]
    fn test_sac_rho_and_lambda_se_differ() {
        let ds = spatial_sim();
        let mut w = knn4(&ds);
        let sac = run_sac(&ds, "y", &["x1", "x2"], &mut w, SacConfig::default()).unwrap();
        assert!(sac.rho_se > 0.0 && sac.lambda_se > 0.0);
        assert!((sac.rho_se - sac.lambda_se).abs() > 0.01, "{sac:?}");
    }

    use crate::spatial::test_fixtures::{band2, knn3};
    use ndarray::array;

    /// (weights, rho, direct, indirect, total) for beta = c(1, 2, -0.5).
    type ImpactCase = (SpatialWeights, f64, [f64; 2], [f64; 2], [f64; 2]);

    // R 4.3.3 (fixture): S <- solve(diag(n) - rho * W); direct <- beta[-1]*mean(diag(S)); total <- beta[-1]*sum(S)/n
    // W: knn3 or band2 (0, 2] weights, styles "W" (row-standardised) and "B" (binary); beta = c(1, 2, -0.5)
    fn r_solve_impact_cases() -> [ImpactCase; 7] {
        [
            // (weights, rho, direct, indirect, total)                       path exercised
            // constant row sums
            (
                knn3(WeightStyle::RowStd),
                0.4,
                [2.106615512105862, -0.5266538780264655],
                [1.2267178212274716, -0.3066794553068679],
                [3.3333333333333335, -0.8333333333333334],
            ),
            (
                knn3(WeightStyle::RowStd),
                -0.7,
                [2.2006493621499876, -0.5501623405374969],
                [-1.0241787739146935, 0.25604469347867337],
                [1.1764705882352942, -0.29411764705882354],
            ),
            (
                knn3(WeightStyle::Binary),
                0.12,
                [2.082709614373916, -0.520677403593479],
                [1.042290385626084, -0.260572596406521],
                [3.125, -0.78125],
            ),
            // isolates: Neumann
            (
                band2(WeightStyle::RowStd),
                0.3,
                [2.0517776386547717, -0.5129444096636929],
                [0.7196509327737997, -0.17991273319344991],
                [2.7714285714285714, -0.6928571428571428],
            ),
            // Neumann, q = 0.48
            (
                band2(WeightStyle::Binary),
                0.1,
                [2.094684105504974, -0.5236710263762435],
                [1.101127304917072, -0.275281826229268],
                [3.195811410422046, -0.7989528526055115],
            ),
            // Neumann, q = 0.96
            (
                band2(WeightStyle::Binary),
                -0.2,
                [2.2393344487779285, -0.5598336121944821],
                [-0.9729643888739399, 0.24324109721848497],
                [1.2663700599039887, -0.31659251497599716],
            ),
            // q = 1.68: dense LU solve (q > 0.99)
            (
                band2(WeightStyle::Binary),
                -0.35,
                [3.053661716595591, -0.7634154291488977],
                [-1.990823153699782, 0.4977057884249455],
                [1.0628385628958088, -0.2657096407239522],
            ),
        ]
    }

    /// Asserts that `imp` holds the case's direct, indirect and total effects within 1e-9.
    fn assert_impacts(imp: &SpatialImpacts, rho: f64, d: [f64; 2], i: [f64; 2], t: [f64; 2]) {
        for j in 0..2 {
            assert!(
                (imp.direct[j] - d[j]).abs() < 1e-9
                    && (imp.indirect[j] - i[j]).abs() < 1e-9
                    && (imp.total[j] - t[j]).abs() < 1e-9,
                "rho={rho} j={j}: {imp:?}"
            );
        }
        assert!(imp.warnings.is_empty(), "rho={rho}: {:?}", imp.warnings);
    }

    #[test]
    fn sar_impacts_match_r_solve_reference() {
        for (mut w, rho, d, i, t) in r_solve_impact_cases() {
            let imp = compute_impacts(
                &array![1.0, 2.0, -0.5],
                rho,
                &mut w,
                &["a".into(), "b".into()],
            )
            .unwrap();
            assert_impacts(&imp, rho, d, i, t);
        }
    }

    /// R: spatialreg mixedImpactsExact written out: S_k = (I - rho W)^-1 (beta_k I + theta_k W),
    /// direct = mean(diag(S_k)), total = mean(rowSums(S_k)) (spatial_ml_weights.R).
    #[test]
    fn test_validate_sdm_impacts() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("spatial_sar_sdm_sem");
        let mut w = knn4(&ds);
        let c = SarConfig {
            durbin: true,
            compute_impacts: true,
            ..Default::default()
        };
        let r = run_sar(&ds, "y", &["x1", "x2"], &mut w, c).unwrap();
        let imp = r.impacts.expect("SDM impacts");
        assert!(imp.warnings.is_empty(), "{:?}", imp.warnings);
        // rho is good to the flatness bound of the likelihood (1.3e-7)
        let est = Tol::iterative(1e-6);
        Check::new(&case)
            .named("sdm_impact_direct", &imp.var_names, &imp.direct, est)
            .named("sdm_impact_indirect", &imp.var_names, &imp.indirect, est)
            .named("sdm_impact_total", &imp.var_names, &imp.total, est)
            .finish();
    }

    /// The same impacts through the closed form (constant row sums) and through the solves: the
    /// binary list is rescaled so that the rows do not sum to one value.
    #[test]
    fn test_sdm_impacts_match_the_dense_formula_without_constant_row_sums() {
        let names = vec!["a".to_string()];
        for mut w in [band2(WeightStyle::RowStd), band2(WeightStyle::Binary)] {
            let rho = 0.05;
            let n = w.n();
            let imp = impacts_exact(&[2.0], Some(&[0.7][..]), rho, &mut w, &names).unwrap();
            let d = w.try_to_dense().unwrap();
            let a = Array2::<f64>::eye(n) - &d * rho;
            let ainv = crate::linalg::matrix_ops::matrix_inverse(&a.view()).unwrap();
            let s = (&ainv * 2.0) + ainv.dot(&d) * 0.7;
            let direct = s.diag().sum() / n as f64;
            let total = s.sum() / n as f64;
            assert!((imp.direct[0] - direct).abs() < 1e-9, "{imp:?}");
            assert!((imp.total[0] - total).abs() < 1e-9, "{imp:?}");
            assert!((imp.indirect[0] - (total - direct)).abs() < 1e-9, "{imp:?}");
        }
    }

    /// rho = 1 on a row-standardised W: I - W is singular (W 1 = 1), where spatialreg's solve()
    /// stops. The impacts are NaN with a warning, never pseudo-inverse numbers.
    #[test]
    fn test_impacts_report_a_singular_multiplier() {
        let names = vec!["a".to_string(), "b".to_string()];
        let mut w = knn3(WeightStyle::RowStd);
        for imp in [
            compute_impacts(&array![1.0, 2.0, -0.5], 1.0, &mut w, &names).unwrap(),
            impacts_exact(&[2.0, -0.5], Some(&[0.3, 0.1][..]), 1.0, &mut w, &names).unwrap(),
        ] {
            assert_eq!(imp.warnings.len(), 1, "{:?}", imp.warnings);
            assert!(imp.warnings[0].contains("singular"), "{}", imp.warnings[0]);
            assert!(
                imp.direct
                    .iter()
                    .chain(&imp.indirect)
                    .chain(&imp.total)
                    .all(|v| v.is_nan()),
                "{imp:?}"
            );
            assert_eq!(imp.var_names, names);
        }
    }

    #[test]
    fn estimators_reject_bad_tolerance_and_iterations() {
        let (d, mut w) = create_test_data();
        let is_param_error = |e: EconError, word: &str| matches!(&e, EconError::InvalidSpecification { message } if message.contains(word));
        for tol in [0.0, -1e-8, f64::NAN, f64::INFINITY] {
            let e = run_sar(
                &d,
                "y",
                &["x"],
                &mut w,
                SarConfig {
                    tol,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(is_param_error(e, "tol"), "sar tol={tol}");
            let e = run_sem(
                &d,
                "y",
                &["x"],
                &mut w,
                SemConfig {
                    tol,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(is_param_error(e, "tol"), "sem tol={tol}");
            let e = run_sac(
                &d,
                "y",
                &["x"],
                &mut w,
                SacConfig {
                    tol,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(is_param_error(e, "tol"), "sac tol={tol}");
        }
        let e = run_sar(
            &d,
            "y",
            &["x"],
            &mut w,
            SarConfig {
                max_iter: 0,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(is_param_error(e, "max_iter"));
        let e = run_sem(
            &d,
            "y",
            &["x"],
            &mut w,
            SemConfig {
                max_iter: 0,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(is_param_error(e, "max_iter"));
        let e = run_sac(
            &d,
            "y",
            &["x"],
            &mut w,
            SacConfig {
                max_iter: 0,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(is_param_error(e, "max_iter"));
    }

    #[test]
    fn estimators_honour_cancellation() {
        use crate::resource::{Budget, CancelToken, with_budget};
        let (d, mut w) = create_test_data();
        let t = CancelToken::new();
        t.cancel();
        let budget = || Budget::new(None, t.clone());
        let r = with_budget(budget(), || {
            run_sar(&d, "y", &["x"], &mut w, SarConfig::default())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "sar");
        let r = with_budget(budget(), || {
            run_sem(&d, "y", &["x"], &mut w, SemConfig::default())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "sem");
        let r = with_budget(budget(), || {
            run_sac(&d, "y", &["x"], &mut w, SacConfig::default())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "sac");
    }

    /// 5 x 5 grid, W = 4 nearest neighbours (row-standardised), integer-valued data:
    /// x1 = 6 i, x2 = 5 ((7 i) mod 13),
    /// y = round(30 (1.5 + 0.8 i/5 - 0.3 ((7 i) mod 13)/6 + 0.2 (row + col)))
    /// (the former `create_validation_spatial_data` scaled by 30 and rounded, so the fit stays
    /// well conditioned). Returns the Int64 columns, a Float64 copy and W.
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

    fn sar(df: &DataFrame, w: &SpatialWeights, durbin: bool) -> EconResult<SarResult> {
        let config = SarConfig {
            durbin,
            ..Default::default()
        };
        run_sar(
            &Dataset::new(df.clone()),
            "y",
            &["x1", "x2"],
            &mut w.clone(),
            config,
        )
    }

    fn sem(df: &DataFrame, w: &SpatialWeights) -> EconResult<SemResult> {
        let ds = Dataset::new(df.clone());
        run_sem(
            &ds,
            "y",
            &["x1", "x2"],
            &mut w.clone(),
            SemConfig::default(),
        )
    }

    fn sac(df: &DataFrame, w: &SpatialWeights) -> EconResult<SacResult> {
        let ds = Dataset::new(df.clone());
        run_sac(
            &ds,
            "y",
            &["x1", "x2"],
            &mut w.clone(),
            SacConfig::default(),
        )
    }

    /// A Float64 copy of column `name` of `df`, every value as stored.
    fn f64_values(df: &DataFrame, name: &str) -> Vec<f64> {
        df.column(name)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    #[test]
    fn test_na_spatial_models_int64_equals_float64() {
        let (int, flt, w) = int_grid();
        assert_eq!(int.column("y").unwrap().dtype(), &DataType::Int64);
        for durbin in [false, true] {
            let (a, b) = (sar(&int, &w, durbin), sar(&flt, &w, durbin));
            assert!(b.is_ok(), "{b:?}");
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
        let (a, b) = (sem(&int, &w), sem(&flt, &w));
        assert!(b.is_ok(), "{b:?}");
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        let (a, b) = (sac(&int, &w), sac(&flt, &w));
        assert!(b.is_ok(), "{b:?}");
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        // a Boolean regressor is 0/1
        let flag: Vec<bool> = (0..25).map(|i| i % 3 == 0).collect();
        let mut bdf = flt.clone();
        bdf.with_column(Column::new("x2".into(), flag.clone()))
            .unwrap();
        let mut fdf = flt.clone();
        let flag01: Vec<f64> = flag.iter().map(|&b| f64::from(u8::from(b))).collect();
        fdf.with_column(Column::new("x2".into(), flag01)).unwrap();
        let (a, b) = (sar(&bdf, &w, false), sar(&fdf, &w, false));
        assert!(b.is_ok(), "{b:?}");
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn test_na_spatial_models_missing_values_refused() {
        let (_, flt, w) = int_grid();
        let y_null: Vec<Option<f64>> = f64_values(&flt, "y")
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i != 7).then_some(v))
            .collect();
        let mut d = flt.clone();
        d.with_column(Column::new("y".into(), y_null)).unwrap();
        for e in [
            sar(&d, &w, false).unwrap_err(),
            sem(&d, &w).unwrap_err(),
            sac(&d, &w).unwrap_err(),
        ] {
            assert!(
                matches!(e, EconError::NullValues { ref column, count: 1 } if column == "y"),
                "{e:?}"
            );
        }
        // NaN is missing too (a Rust-built fixture; R has no NaN in a CSV)
        let mut x1 = f64_values(&flt, "x1");
        x1[3] = f64::NAN;
        let mut d = flt.clone();
        d.with_column(Column::new("x1".into(), x1)).unwrap();
        let e = sar(&d, &w, false).unwrap_err();
        assert!(
            matches!(e, EconError::NullValues { ref column, count: 1 } if column == "x1"),
            "{e:?}"
        );
    }

    #[test]
    fn test_na_spatial_models_dtype_errors() {
        let (_, flt, w) = int_grid();
        let mut d = flt.clone();
        let text: Vec<String> = (0..25).map(|i| format!("v{i}")).collect();
        d.with_column(Column::new("x1".into(), text)).unwrap();
        let e = sar(&d, &w, false).unwrap_err();
        assert!(
            matches!(e, EconError::NonNumericColumn { ref column } if column == "x1"),
            "{e:?}"
        );
        let mut x2 = f64_values(&flt, "x2");
        x2[0] = f64::INFINITY;
        let mut d = flt.clone();
        d.with_column(Column::new("x2".into(), x2)).unwrap();
        let e = sem(&d, &w).unwrap_err();
        assert!(
            matches!(e, EconError::NonFiniteValues { ref column, count: 1 } if column == "x2"),
            "{e:?}"
        );
        let r = run_sar(
            &Dataset::new(flt.clone()),
            "y",
            &["nope"],
            &mut w.clone(),
            SarConfig::default(),
        );
        assert!(matches!(r, Err(EconError::ColumnNotFound { .. })), "{r:?}");
        // W of the wrong size (a guard: refused before the change too)
        let three = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let small =
            SpatialWeights::from_neighbors(&Neighbors::from_knn(&three, 1), WeightStyle::RowStd);
        let e = sar(&flt, &small, false).unwrap_err();
        assert!(matches!(e, EconError::InvalidSpecification { .. }), "{e:?}");
    }
}
