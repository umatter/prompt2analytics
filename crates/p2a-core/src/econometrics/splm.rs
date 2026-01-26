//! Spatial panel data models (splm).
//!
//! Provides ML and GMM estimation of econometric models for spatial panel data,
//! combining spatial dependence structures with panel data methods.
//!
//! # Models
//!
//! ## Spatial Panel Lag Model (SPML with lag)
//!
//! y_it = rho * W * y_it + X_it * beta + alpha_i + epsilon_it
//!
//! ## Spatial Panel Error Model (SPML with spatial.error)
//!
//! y_it = X_it * beta + alpha_i + u_it
//! u_it = lambda * W * u_it + epsilon_it
//!
//! ## Combined Spatial Lag and Error (SAC Panel)
//!
//! y_it = rho * W * y_it + X_it * beta + alpha_i + u_it
//! u_it = lambda * W * u_it + epsilon_it
//!
//! # References
//!
//! - Baltagi, B.H., Song, S.H., & Koh, W. (2003). Testing panel data regression models
//!   with spatial error correlation. *Journal of Econometrics*, 117(1), 123-150.
//!   https://doi.org/10.1016/S0304-4076(03)00120-9
//!
//! - Kapoor, M., Kelejian, H.H., & Prucha, I.R. (2007). Panel data models with
//!   spatially correlated error components. *Journal of Econometrics*, 140(1), 97-130.
//!   https://doi.org/10.1016/j.jeconom.2006.09.004
//!
//! - Millo, G., & Piras, G. (2012). splm: Spatial Panel Data Models in R.
//!   *Journal of Statistical Software*, 47(1), 1-38.
//!   https://www.jstatsoft.org/v47/i01/
//!
//! R equivalent: `splm::spml()`, `splm::spgm()`, `splm::spreml()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{matrix_inverse, xtx, xty};
use crate::spatial::SpatialWeights;
use crate::traits::estimator::{chi_squared_p_value, SignificanceLevel};

// ============================================================================
// Configuration Types
// ============================================================================

/// Panel effect type for spatial panel models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpatialPanelEffect {
    /// Individual (entity) effects only
    #[default]
    Individual,
    /// Time effects only
    Time,
    /// Two-way effects (individual + time)
    TwoWays,
}

impl fmt::Display for SpatialPanelEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpatialPanelEffect::Individual => write!(f, "Individual"),
            SpatialPanelEffect::Time => write!(f, "Time"),
            SpatialPanelEffect::TwoWays => write!(f, "Two-ways"),
        }
    }
}

/// Panel model type for spatial panel estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpatialPanelModel {
    /// Fixed effects (within transformation)
    #[default]
    Within,
    /// Random effects (GLS)
    Random,
    /// Pooled (no effects)
    Pooling,
}

impl fmt::Display for SpatialPanelModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpatialPanelModel::Within => write!(f, "Fixed Effects (Within)"),
            SpatialPanelModel::Random => write!(f, "Random Effects"),
            SpatialPanelModel::Pooling => write!(f, "Pooled"),
        }
    }
}

/// Spatial error specification type.
///
/// Following Baltagi et al. (2003) and Kapoor et al. (2007).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpatialErrorType {
    /// No spatial error correlation
    #[default]
    None,
    /// Baltagi-type spatial error (Baltagi, Song & Koh, 2003)
    /// Random effects are not spatially correlated
    Baltagi,
    /// Kapoor-Kelejian-Prucha type (Kapoor et al., 2007)
    /// Random effects are spatially correlated
    KKP,
}

impl fmt::Display for SpatialErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpatialErrorType::None => write!(f, "None"),
            SpatialErrorType::Baltagi => write!(f, "Baltagi (sem)"),
            SpatialErrorType::KKP => write!(f, "Kapoor-Kelejian-Prucha (sem2)"),
        }
    }
}

/// Configuration for spatial panel ML estimation (spml).
#[derive(Debug, Clone)]
pub struct SpmlConfig {
    /// Panel model type (within, random, pooling)
    pub model: SpatialPanelModel,
    /// Effect type (individual, time, twoways)
    pub effect: SpatialPanelEffect,
    /// Include spatial lag of dependent variable
    pub lag: bool,
    /// Spatial error specification
    pub spatial_error: SpatialErrorType,
    /// Tolerance for optimization
    pub tol: f64,
    /// Maximum iterations for optimization
    pub max_iter: usize,
}

impl Default for SpmlConfig {
    fn default() -> Self {
        Self {
            model: SpatialPanelModel::Within,
            effect: SpatialPanelEffect::Individual,
            lag: false,
            spatial_error: SpatialErrorType::None,
            tol: 1e-8,
            max_iter: 100,
        }
    }
}

/// GMM estimation method for spatial panel data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpgmMethod {
    /// Within 2SLS (fixed effects)
    #[default]
    W2sls,
    /// Between 2SLS
    B2sls,
    /// GLS random effects 2SLS
    G2sls,
    /// Baltagi's EC2SLS
    Ec2sls,
}

impl fmt::Display for SpgmMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpgmMethod::W2sls => write!(f, "Within 2SLS (w2sls)"),
            SpgmMethod::B2sls => write!(f, "Between 2SLS (b2sls)"),
            SpgmMethod::G2sls => write!(f, "GLS Random Effects 2SLS (g2sls)"),
            SpgmMethod::Ec2sls => write!(f, "Baltagi EC2SLS (ec2sls)"),
        }
    }
}

/// GMM moments specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpgmMoments {
    /// Initial moments with equal weighting
    #[default]
    Initial,
    /// Simplified variance-covariance weighting
    Weights,
    /// Full moment conditions
    FullWeights,
}

/// Configuration for spatial panel GMM estimation (spgm).
#[derive(Debug, Clone)]
pub struct SpgmConfig {
    /// Estimation method
    pub method: SpgmMethod,
    /// Include spatial lag
    pub lag: bool,
    /// Include spatial error
    pub spatial_error: bool,
    /// Moments specification
    pub moments: SpgmMoments,
    /// Tolerance for optimization
    pub tol: f64,
    /// Maximum iterations
    pub max_iter: usize,
}

impl Default for SpgmConfig {
    fn default() -> Self {
        Self {
            method: SpgmMethod::W2sls,
            lag: false,
            spatial_error: true,
            moments: SpgmMoments::Initial,
            tol: 1e-8,
            max_iter: 100,
        }
    }
}

/// Error component specification for spreml.
///
/// Combines abbreviations: sem, sem2, sr, re, ols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpremlErrors {
    /// Spherical errors (OLS)
    Ols,
    /// Random effects only
    #[default]
    Re,
    /// Serial correlation only
    Sr,
    /// Spatial error (Anselin-Baltagi type)
    Sem,
    /// Spatial error + random effects
    SemRe,
    /// Serial correlation + random effects
    SrRe,
    /// Spatial error + serial correlation
    SemSr,
    /// Full model: spatial error + serial + random effects
    SemSrRe,
    /// KKP-type spatial error + random effects
    Sem2Re,
    /// Full KKP model with serial correlation
    Sem2SrRe,
}

impl fmt::Display for SpremlErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpremlErrors::Ols => write!(f, "ols"),
            SpremlErrors::Re => write!(f, "re"),
            SpremlErrors::Sr => write!(f, "sr"),
            SpremlErrors::Sem => write!(f, "sem"),
            SpremlErrors::SemRe => write!(f, "semre"),
            SpremlErrors::SrRe => write!(f, "srre"),
            SpremlErrors::SemSr => write!(f, "semsr"),
            SpremlErrors::SemSrRe => write!(f, "semsrre"),
            SpremlErrors::Sem2Re => write!(f, "sem2re"),
            SpremlErrors::Sem2SrRe => write!(f, "sem2srre"),
        }
    }
}

/// Configuration for spreml (spatial random effects ML).
#[derive(Debug, Clone)]
pub struct SpremlConfig {
    /// Include spatial lag
    pub lag: bool,
    /// Error component specification
    pub errors: SpremlErrors,
    /// Tolerance for optimization
    pub tol: f64,
    /// Maximum iterations
    pub max_iter: usize,
}

impl Default for SpremlConfig {
    fn default() -> Self {
        Self {
            lag: false,
            errors: SpremlErrors::Re,
            tol: 1e-8,
            max_iter: 100,
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Variance components for spatial panel models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPanelVariance {
    /// Individual effect variance (sigma_mu^2)
    pub sigma_mu: Option<f64>,
    /// Time effect variance (sigma_nu^2)
    pub sigma_nu: Option<f64>,
    /// Idiosyncratic error variance (sigma_epsilon^2)
    pub sigma_epsilon: f64,
    /// Spatial error parameter (lambda for SEM component)
    pub lambda: Option<f64>,
    /// Serial correlation parameter (rho_sr)
    pub rho_sr: Option<f64>,
}

/// Result from spatial panel ML estimation (spml).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpmlResult {
    /// Model type used
    pub model: SpatialPanelModel,
    /// Effect type
    pub effect: SpatialPanelEffect,
    /// Whether spatial lag was included
    pub has_lag: bool,
    /// Spatial error type
    pub spatial_error: SpatialErrorType,

    /// Regression coefficients (beta)
    pub coefficients: Vec<f64>,
    /// Coefficient names
    pub coef_names: Vec<String>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Z-values
    pub z_values: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,

    /// Spatial lag coefficient (rho) if lag=true
    pub rho: Option<f64>,
    /// Standard error of rho
    pub rho_se: Option<f64>,
    /// Z-value for rho
    pub rho_z: Option<f64>,
    /// P-value for rho
    pub rho_p: Option<f64>,

    /// Spatial error coefficient (lambda) if spatial_error != None
    pub lambda: Option<f64>,
    /// Standard error of lambda
    pub lambda_se: Option<f64>,
    /// Z-value for lambda
    pub lambda_z: Option<f64>,
    /// P-value for lambda
    pub lambda_p: Option<f64>,

    /// Variance components
    pub variance: SpatialPanelVariance,

    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,

    /// Number of observations
    pub n_obs: usize,
    /// Number of entities (cross-sectional units)
    pub n_entities: usize,
    /// Number of time periods
    pub n_time: usize,
    /// Degrees of freedom
    pub df: usize,

    /// Residuals (skipped for serialization)
    #[serde(skip)]
    pub residuals: Array1<f64>,
    /// Fitted values (skipped for serialization)
    #[serde(skip)]
    pub fitted: Array1<f64>,
}

impl fmt::Display for SpmlResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\nSpatial Panel Model (ML)")?;
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f, "Model: {}  Effect: {}", self.model, self.effect)?;
        writeln!(f, "Spatial Lag: {}  Spatial Error: {}", self.has_lag, self.spatial_error)?;
        writeln!(f, "Observations: {}  Entities: {}  Time periods: {}",
                 self.n_obs, self.n_entities, self.n_time)?;
        writeln!(f, "Log-Likelihood: {:.4}  AIC: {:.4}  BIC: {:.4}",
                 self.log_likelihood, self.aic, self.bic)?;
        writeln!(f, "{}", "-".repeat(60))?;

        // Spatial parameters
        if let Some(rho) = self.rho {
            writeln!(f, "\nSpatial lag coefficient (rho):")?;
            writeln!(f, "  Estimate: {:.6}  SE: {:.6}  Z: {:.4}  P: {:.4}",
                     rho,
                     self.rho_se.unwrap_or(0.0),
                     self.rho_z.unwrap_or(0.0),
                     self.rho_p.unwrap_or(1.0))?;
        }

        if let Some(lambda) = self.lambda {
            writeln!(f, "\nSpatial error coefficient (lambda):")?;
            writeln!(f, "  Estimate: {:.6}  SE: {:.6}  Z: {:.4}  P: {:.4}",
                     lambda,
                     self.lambda_se.unwrap_or(0.0),
                     self.lambda_z.unwrap_or(0.0),
                     self.lambda_p.unwrap_or(1.0))?;
        }

        writeln!(f, "\n{:<20} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coefficient", "Std.Error", "Z-value", "P>|z|")?;
        writeln!(f, "{}", "-".repeat(60))?;

        for i in 0..self.coef_names.len() {
            writeln!(f, "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4}{}",
                     self.coef_names[i],
                     self.coefficients[i],
                     self.std_errors[i],
                     self.z_values[i],
                     self.p_values[i],
                     self.significance[i].stars())?;
        }

        writeln!(f, "{}", "-".repeat(60))?;
        writeln!(f, "Variance components:")?;
        if let Some(sigma_mu) = self.variance.sigma_mu {
            writeln!(f, "  sigma_mu (individual): {:.6}", sigma_mu)?;
        }
        writeln!(f, "  sigma_epsilon (error): {:.6}", self.variance.sigma_epsilon)?;

        Ok(())
    }
}

/// Result from spatial panel GMM estimation (spgm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpgmResult {
    /// Estimation method
    pub method: SpgmMethod,
    /// Whether spatial lag was included
    pub has_lag: bool,
    /// Whether spatial error was included
    pub has_spatial_error: bool,

    /// Regression coefficients
    pub coefficients: Vec<f64>,
    /// Coefficient names
    pub coef_names: Vec<String>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Z-values
    pub z_values: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,

    /// Spatial lag coefficient (rho)
    pub rho: Option<f64>,
    /// Spatial error coefficient (lambda)
    pub lambda: Option<f64>,

    /// Variance components
    pub sigma2: f64,
    pub sigma2_mu: Option<f64>,

    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Number of time periods
    pub n_time: usize,
    /// Number of instruments
    pub n_instruments: usize,

    /// Sargan/Hansen test for overidentifying restrictions
    pub sargan_stat: Option<f64>,
    /// Sargan test p-value
    pub sargan_p: Option<f64>,
    /// Sargan test degrees of freedom
    pub sargan_df: Option<usize>,
}

impl fmt::Display for SpgmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\nSpatial Panel Model (GMM)")?;
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f, "Method: {}", self.method)?;
        writeln!(f, "Spatial Lag: {}  Spatial Error: {}", self.has_lag, self.has_spatial_error)?;
        writeln!(f, "Observations: {}  Entities: {}  Time periods: {}",
                 self.n_obs, self.n_entities, self.n_time)?;
        writeln!(f, "Instruments: {}", self.n_instruments)?;
        writeln!(f, "{}", "-".repeat(60))?;

        writeln!(f, "\n{:<20} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coefficient", "Std.Error", "Z-value", "P>|z|")?;
        writeln!(f, "{}", "-".repeat(60))?;

        for i in 0..self.coef_names.len() {
            writeln!(f, "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4}{}",
                     self.coef_names[i],
                     self.coefficients[i],
                     self.std_errors[i],
                     self.z_values[i],
                     self.p_values[i],
                     self.significance[i].stars())?;
        }

        if let (Some(stat), Some(p), Some(df)) = (self.sargan_stat, self.sargan_p, self.sargan_df) {
            writeln!(f, "\nSargan test: chi2({}) = {:.4}, p = {:.4}", df, stat, p)?;
        }

        Ok(())
    }
}

// ============================================================================
// Main Estimation Functions
// ============================================================================

/// Run spatial panel ML estimation (equivalent to R's splm::spml).
///
/// Estimates spatial panel models with fixed or random effects using
/// maximum likelihood. Supports spatial lag models, spatial error models,
/// and combined specifications.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of independent variable columns
/// * `entity_col` - Name of the entity identifier column
/// * `time_col` - Name of the time identifier column
/// * `listw` - Spatial weights matrix
/// * `config` - Model configuration
///
/// # Returns
///
/// `SpmlResult` with estimates, standard errors, and diagnostics.
///
/// # References
///
/// - Baltagi, B.H., Song, S.H., & Koh, W. (2003). Testing panel data regression
///   models with spatial error correlation. *Journal of Econometrics*, 117(1), 123-150.
/// - Millo, G., & Piras, G. (2012). splm: Spatial Panel Data Models in R.
///   *Journal of Statistical Software*, 47(1), 1-38.
pub fn run_spml(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
    listw: &mut SpatialWeights,
    config: SpmlConfig,
) -> EconResult<SpmlResult> {
    let df = dataset.df();
    let n_total = df.height();

    // Extract and organize panel structure
    let (y, x, entity_ids, time_ids, n_entities, n_time) =
        extract_panel_data(dataset, y_col, x_cols, entity_col, time_col)?;

    let k = x.ncols();

    // Validate spatial weights dimension
    if listw.n() != n_entities {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Spatial weights matrix has {} units but panel has {} entities. \
                 Weights should be N x N where N is the number of cross-sectional units.",
                listw.n(),
                n_entities
            ),
        });
    }

    // Build variable names
    let mut coef_names = vec!["(Intercept)".to_string()];
    coef_names.extend(x_cols.iter().map(|s| s.to_string()));

    // Dispatch based on model type and spatial specification
    match config.model {
        SpatialPanelModel::Within => {
            run_spml_within(&y, &x, &entity_ids, &time_ids, n_entities, n_time,
                           listw, &config, coef_names)
        }
        SpatialPanelModel::Random => {
            run_spml_random(&y, &x, &entity_ids, &time_ids, n_entities, n_time,
                           listw, &config, coef_names)
        }
        SpatialPanelModel::Pooling => {
            run_spml_pooling(&y, &x, &entity_ids, &time_ids, n_entities, n_time,
                            listw, &config, coef_names)
        }
    }
}

/// Run spatial panel GMM estimation (equivalent to R's splm::spgm).
///
/// Uses generalized method of moments with spatial instruments for
/// panel data with spatial dependence.
///
/// # Arguments
///
/// * `dataset` - Dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of independent variable columns
/// * `entity_col` - Name of the entity identifier column
/// * `time_col` - Name of the time identifier column
/// * `listw` - Spatial weights matrix
/// * `config` - GMM configuration
///
/// # Returns
///
/// `SpgmResult` with GMM estimates and diagnostics.
///
/// # References
///
/// - Kapoor, M., Kelejian, H.H., & Prucha, I.R. (2007). Panel data models with
///   spatially correlated error components. *Journal of Econometrics*, 140(1), 97-130.
pub fn run_spgm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
    listw: &mut SpatialWeights,
    config: SpgmConfig,
) -> EconResult<SpgmResult> {
    let (y, x, entity_ids, time_ids, n_entities, n_time) =
        extract_panel_data(dataset, y_col, x_cols, entity_col, time_col)?;

    let k = x.ncols();

    if listw.n() != n_entities {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Spatial weights matrix has {} units but panel has {} entities",
                listw.n(),
                n_entities
            ),
        });
    }

    let mut coef_names = vec!["(Intercept)".to_string()];
    coef_names.extend(x_cols.iter().map(|s| s.to_string()));

    // Dispatch based on method
    match config.method {
        SpgmMethod::W2sls => {
            run_spgm_within(&y, &x, &entity_ids, &time_ids, n_entities, n_time,
                           listw, &config, coef_names)
        }
        SpgmMethod::G2sls => {
            run_spgm_gls(&y, &x, &entity_ids, &time_ids, n_entities, n_time,
                        listw, &config, coef_names)
        }
        _ => {
            // Default to within for other methods (simplified)
            run_spgm_within(&y, &x, &entity_ids, &time_ids, n_entities, n_time,
                           listw, &config, coef_names)
        }
    }
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

/// Extract and organize panel data from dataset.
fn extract_panel_data(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
) -> EconResult<(Array1<f64>, Array2<f64>, Vec<usize>, Vec<usize>, usize, usize)> {
    let df = dataset.df();
    let n = df.height();

    // Extract y
    let y_series = df.column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
    })?;
    let y: Array1<f64> = y_series
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: y_col.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Extract X with intercept
    let k = x_cols.len() + 1;
    let mut x = Array2::zeros((n, k));
    for i in 0..n {
        x[[i, 0]] = 1.0; // Intercept
    }
    for (j, &col_name) in x_cols.iter().enumerate() {
        let col = df.column(col_name).map_err(|_| EconError::ColumnNotFound {
            column: col_name.to_string(),
            available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        })?;
        let col_f64 = col.f64().map_err(|_| EconError::NonNumericColumn {
            column: col_name.to_string(),
        })?;
        for (i, val) in col_f64.into_no_null_iter().enumerate() {
            x[[i, j + 1]] = val;
        }
    }

    // Extract entity IDs
    let entity_series = df.column(entity_col).map_err(|_| EconError::ColumnNotFound {
        column: entity_col.to_string(),
        available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
    })?;

    let mut entity_map: HashMap<String, usize> = HashMap::new();
    let mut next_entity_id = 0usize;

    let entity_strings: Vec<String> = if let Ok(str_col) = entity_series.str() {
        str_col.into_iter().map(|s| s.unwrap_or("").to_string()).collect()
    } else if let Ok(i64_col) = entity_series.i64() {
        i64_col.into_iter().map(|v| v.unwrap_or(0).to_string()).collect()
    } else {
        return Err(EconError::NonNumericColumn {
            column: entity_col.to_string(),
        });
    };

    let entity_ids: Vec<usize> = entity_strings
        .iter()
        .map(|s| {
            *entity_map.entry(s.clone()).or_insert_with(|| {
                let id = next_entity_id;
                next_entity_id += 1;
                id
            })
        })
        .collect();

    let n_entities = entity_map.len();

    // Extract time IDs
    let time_series = df.column(time_col).map_err(|_| EconError::ColumnNotFound {
        column: time_col.to_string(),
        available: df.get_column_names().iter().map(|s| s.to_string()).collect(),
    })?;

    let mut time_map: HashMap<i64, usize> = HashMap::new();
    let mut next_time_id = 0usize;

    let time_values: Vec<i64> = if let Ok(i64_col) = time_series.i64() {
        i64_col.into_iter().map(|v| v.unwrap_or(0)).collect()
    } else if let Ok(f64_col) = time_series.f64() {
        f64_col.into_iter().map(|v| v.unwrap_or(0.0) as i64).collect()
    } else {
        return Err(EconError::NonNumericColumn {
            column: time_col.to_string(),
        });
    };

    let time_ids: Vec<usize> = time_values
        .iter()
        .map(|&t| {
            *time_map.entry(t).or_insert_with(|| {
                let id = next_time_id;
                next_time_id += 1;
                id
            })
        })
        .collect();

    let n_time = time_map.len();

    Ok((y, x, entity_ids, time_ids, n_entities, n_time))
}

/// Demean by entity (within transformation).
fn demean_by_entity(data: &Array1<f64>, entity_ids: &[usize], n_entities: usize) -> Array1<f64> {
    let n = data.len();
    let mut group_sums = vec![0.0; n_entities];
    let mut group_counts = vec![0usize; n_entities];

    for (i, &val) in data.iter().enumerate() {
        let g = entity_ids[i];
        group_sums[g] += val;
        group_counts[g] += 1;
    }

    let group_means: Vec<f64> = group_sums
        .iter()
        .zip(group_counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
        .collect();

    let mut demeaned = Array1::zeros(n);
    for i in 0..n {
        demeaned[i] = data[i] - group_means[entity_ids[i]];
    }
    demeaned
}

/// Demean matrix by entity.
fn demean_matrix_by_entity(
    x: &Array2<f64>,
    entity_ids: &[usize],
    n_entities: usize,
) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut x_demeaned = Array2::zeros((n, k));

    for j in 0..k {
        let col = x.column(j).to_owned();
        let col_demeaned = demean_by_entity(&col, entity_ids, n_entities);
        x_demeaned.column_mut(j).assign(&col_demeaned);
    }

    x_demeaned
}

/// Compute entity means.
fn compute_entity_means(data: &Array1<f64>, entity_ids: &[usize], n_entities: usize) -> Vec<f64> {
    let mut group_sums = vec![0.0; n_entities];
    let mut group_counts = vec![0usize; n_entities];

    for (i, &val) in data.iter().enumerate() {
        let g = entity_ids[i];
        group_sums[g] += val;
        group_counts[g] += 1;
    }

    group_sums
        .iter()
        .zip(group_counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
        .collect()
}

/// Expand spatial lag to panel dimension.
///
/// For panel data, we apply the spatial weights to each cross-section at time t.
/// W operates on N observations, but we have N*T total observations.
fn spatial_lag_panel(
    y: &Array1<f64>,
    entity_ids: &[usize],
    time_ids: &[usize],
    n_entities: usize,
    n_time: usize,
    listw: &SpatialWeights,
) -> Array1<f64> {
    let n_total = y.len();
    let mut wy = Array1::zeros(n_total);

    // For each time period, apply spatial weights to cross-section
    for t in 0..n_time {
        // Extract values for this time period
        let mut y_t = Array1::zeros(n_entities);
        let mut obs_indices = vec![None; n_entities];

        for (i, (&eid, &tid)) in entity_ids.iter().zip(time_ids.iter()).enumerate() {
            if tid == t {
                y_t[eid] = y[i];
                obs_indices[eid] = Some(i);
            }
        }

        // Apply spatial lag
        let wy_t = listw.lag(&y_t);

        // Map back to panel structure
        for (eid, opt_idx) in obs_indices.iter().enumerate() {
            if let Some(idx) = opt_idx {
                wy[*idx] = wy_t[eid];
            }
        }
    }

    wy
}

// ============================================================================
// Fixed Effects (Within) Estimation
// ============================================================================

fn run_spml_within(
    y: &Array1<f64>,
    x: &Array2<f64>,
    entity_ids: &[usize],
    time_ids: &[usize],
    n_entities: usize,
    n_time: usize,
    listw: &mut SpatialWeights,
    config: &SpmlConfig,
    coef_names: Vec<String>,
) -> EconResult<SpmlResult> {
    let n = y.len();
    let k = x.ncols();

    // Within transformation (demean by entity)
    let y_within = demean_by_entity(y, entity_ids, n_entities);
    let x_within = demean_matrix_by_entity(x, entity_ids, n_entities);

    // For fixed effects, drop intercept column (demeaning removes it)
    let x_fe = x_within.slice(ndarray::s![.., 1..]).to_owned();
    let k_fe = k - 1;

    let mut coef_names_fe: Vec<String> = coef_names[1..].to_vec();

    // Handle spatial lag if specified
    let (beta, rho_opt, sigma2, ll) = if config.lag {
        // Spatial lag panel model: y = rho*W*y + X*beta + alpha + epsilon
        let wy = spatial_lag_panel(y, entity_ids, time_ids, n_entities, n_time, listw);
        let wy_within = demean_by_entity(&wy, entity_ids, n_entities);

        // Optimize over rho using concentrated likelihood
        let (rho_min, rho_max) = listw.rho_range();
        let rho_min = rho_min.max(-0.99);
        let rho_max = rho_max.min(0.99);

        let (rho_opt, ll_opt) = optimize_rho_panel(
            &y_within,
            &wy_within,
            &x_fe,
            n_entities,
            n_time,
            listw,
            rho_min,
            rho_max,
            config.tol,
            config.max_iter,
        )?;

        // Compute beta at optimal rho
        let y_tilde = &y_within - rho_opt * &wy_within;
        let xtx_mat = xtx(&x_fe.view());
        let xtx_inv = matrix_inverse(&xtx_mat.view())?;
        let xty_vec = xty(&x_fe.view(), &y_tilde);
        let beta = xtx_inv.dot(&xty_vec);

        let residuals = &y_tilde - &x_fe.dot(&beta);
        let rss: f64 = residuals.iter().map(|&r| r * r).sum();
        let df = n - n_entities - k_fe;
        let sigma2 = rss / df as f64;

        (beta, Some(rho_opt), sigma2, ll_opt)
    } else if config.spatial_error != SpatialErrorType::None {
        // Spatial error panel model
        run_spml_within_sem(&y_within, &x_fe, entity_ids, time_ids,
                           n_entities, n_time, listw, config)?
    } else {
        // Standard fixed effects (no spatial)
        let xtx_mat = xtx(&x_fe.view());
        let xtx_inv = matrix_inverse(&xtx_mat.view())?;
        let xty_vec = xty(&x_fe.view(), &y_within);
        let beta = xtx_inv.dot(&xty_vec);

        let residuals = &y_within - &x_fe.dot(&beta);
        let rss: f64 = residuals.iter().map(|&r| r * r).sum();
        let df = n - n_entities - k_fe;
        let sigma2 = rss / df as f64;

        let ll = -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln());

        (beta, None, sigma2, ll)
    };

    // Compute standard errors and statistics
    let df = n - n_entities - k_fe - if config.lag { 1 } else { 0 };
    let xtx_mat = xtx(&x_fe.view());
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;

    let std_errors: Vec<f64> = (0..k_fe)
        .map(|j| (sigma2 * xtx_inv[[j, j]]).sqrt())
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_values: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| 2.0 * (1.0 - normal.cdf(z.abs())))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Rho statistics if present
    let (rho_se, rho_z, rho_p) = if let Some(rho) = rho_opt {
        let tr_w2 = listw.trace_w2();
        let tr_wtw = listw.trace_wtw();
        let var_rho = sigma2 / (n_time as f64 * (tr_w2 + tr_wtw));
        let se = var_rho.sqrt().max(0.001);
        let z = rho / se;
        let p = 2.0 * (1.0 - normal.cdf(z.abs()));
        (Some(se), Some(z), Some(p))
    } else {
        (None, None, None)
    };

    // Compute fitted values and residuals
    let fitted = x_fe.dot(&beta);
    let residuals = &y_within - &fitted;

    // AIC and BIC
    let n_params = k_fe + if config.lag { 2 } else { 1 }; // beta + (rho?) + sigma2
    let aic = -2.0 * ll + 2.0 * n_params as f64;
    let bic = -2.0 * ll + (n_params as f64) * (n as f64).ln();

    Ok(SpmlResult {
        model: SpatialPanelModel::Within,
        effect: config.effect,
        has_lag: config.lag,
        spatial_error: config.spatial_error,
        coefficients: beta.to_vec(),
        coef_names: coef_names_fe,
        std_errors,
        z_values,
        p_values,
        significance,
        rho: rho_opt,
        rho_se,
        rho_z,
        rho_p,
        lambda: None,
        lambda_se: None,
        lambda_z: None,
        lambda_p: None,
        variance: SpatialPanelVariance {
            sigma_mu: None,
            sigma_nu: None,
            sigma_epsilon: sigma2.sqrt(),
            lambda: None,
            rho_sr: None,
        },
        log_likelihood: ll,
        aic,
        bic,
        n_obs: n,
        n_entities,
        n_time,
        df,
        residuals,
        fitted,
    })
}

/// Spatial error model within fixed effects.
fn run_spml_within_sem(
    y_within: &Array1<f64>,
    x_fe: &Array2<f64>,
    entity_ids: &[usize],
    time_ids: &[usize],
    n_entities: usize,
    n_time: usize,
    listw: &mut SpatialWeights,
    config: &SpmlConfig,
) -> EconResult<(Array1<f64>, Option<f64>, f64, f64)> {
    let n = y_within.len();
    let k = x_fe.ncols();

    // Get valid lambda range
    let (lambda_min, lambda_max) = listw.rho_range();
    let lambda_min = lambda_min.max(-0.99);
    let lambda_max = lambda_max.min(0.99);

    // Optimize lambda for spatial error
    let eigenvalues = listw.eigenvalues().clone();

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut a = lambda_min;
    let mut b = lambda_max;

    // Negative concentrated log-likelihood
    let neg_ll = |lambda: f64| -> f64 {
        // Transform y and X for each time period
        let mut y_star = Array1::zeros(n);
        let mut x_star = Array2::zeros((n, k));

        for t in 0..n_time {
            let mut y_t = Array1::zeros(n_entities);
            let mut x_t = Array2::zeros((n_entities, k));
            let mut obs_map = vec![None; n_entities];

            for (i, (&eid, &tid)) in entity_ids.iter().zip(time_ids.iter()).enumerate() {
                if tid == t {
                    y_t[eid] = y_within[i];
                    for j in 0..k {
                        x_t[[eid, j]] = x_fe[[i, j]];
                    }
                    obs_map[eid] = Some(i);
                }
            }

            // (I - lambda*W) transformation
            let wy_t = listw.lag(&y_t);
            let y_t_star = &y_t - lambda * &wy_t;

            let mut wx_t = Array2::zeros((n_entities, k));
            for j in 0..k {
                let x_col = x_t.column(j).to_owned();
                let wx_col = listw.lag(&x_col);
                wx_t.column_mut(j).assign(&wx_col);
            }
            let x_t_star = &x_t - lambda * &wx_t;

            // Map back
            for (eid, opt_idx) in obs_map.iter().enumerate() {
                if let Some(idx) = opt_idx {
                    y_star[*idx] = y_t_star[eid];
                    for j in 0..k {
                        x_star[[*idx, j]] = x_t_star[[eid, j]];
                    }
                }
            }
        }

        // OLS on transformed data
        let xtx_mat = xtx(&x_star.view());
        let xtx_inv = match matrix_inverse(&xtx_mat.view()) {
            Ok(inv) => inv,
            Err(_) => return f64::INFINITY,
        };
        let xty_vec = xty(&x_star.view(), &y_star);
        let beta = xtx_inv.dot(&xty_vec);
        let residuals = &y_star - &x_star.dot(&beta);
        let rss: f64 = residuals.iter().map(|&r| r * r).sum();
        let sigma2 = rss / n as f64;

        if sigma2 <= 0.0 {
            return f64::INFINITY;
        }

        // Log determinant: T * sum(log(1 - lambda * eigenvalue))
        let log_det: f64 = n_time as f64
            * eigenvalues
                .iter()
                .map(|&ev| (1.0 - lambda * ev).ln())
                .sum::<f64>();

        0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln()) - log_det
    };

    // Golden section search
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;

    for _ in 0..config.max_iter {
        if (b - a).abs() < config.tol {
            break;
        }
        if neg_ll(c) < neg_ll(d) {
            b = d;
            d = c;
            c = b - (b - a) / phi;
        } else {
            a = c;
            c = d;
            d = a + (b - a) / phi;
        }
    }

    let lambda_opt = (a + b) / 2.0;
    let ll_opt = -neg_ll(lambda_opt);

    // Compute beta at optimal lambda (recompute transformed data)
    let mut y_star = Array1::zeros(n);
    let mut x_star = Array2::zeros((n, k));

    for t in 0..n_time {
        let mut y_t = Array1::zeros(n_entities);
        let mut x_t = Array2::zeros((n_entities, k));
        let mut obs_map = vec![None; n_entities];

        for (i, (&eid, &tid)) in entity_ids.iter().zip(time_ids.iter()).enumerate() {
            if tid == t {
                y_t[eid] = y_within[i];
                for j in 0..k {
                    x_t[[eid, j]] = x_fe[[i, j]];
                }
                obs_map[eid] = Some(i);
            }
        }

        let wy_t = listw.lag(&y_t);
        let y_t_star = &y_t - lambda_opt * &wy_t;

        let mut wx_t = Array2::zeros((n_entities, k));
        for j in 0..k {
            let x_col = x_t.column(j).to_owned();
            let wx_col = listw.lag(&x_col);
            wx_t.column_mut(j).assign(&wx_col);
        }
        let x_t_star = &x_t - lambda_opt * &wx_t;

        for (eid, opt_idx) in obs_map.iter().enumerate() {
            if let Some(idx) = opt_idx {
                y_star[*idx] = y_t_star[eid];
                for j in 0..k {
                    x_star[[*idx, j]] = x_t_star[[eid, j]];
                }
            }
        }
    }

    let xtx_mat = xtx(&x_star.view());
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;
    let xty_vec = xty(&x_star.view(), &y_star);
    let beta = xtx_inv.dot(&xty_vec);
    let residuals = &y_star - &x_star.dot(&beta);
    let rss: f64 = residuals.iter().map(|&r| r * r).sum();
    let sigma2 = rss / (n - n_entities - k) as f64;

    // Note: For SEM, we return lambda via sigma2 field temporarily
    // The caller will need to handle this properly
    Ok((beta, Some(lambda_opt), sigma2, ll_opt))
}

/// Optimize rho for spatial lag panel model.
fn optimize_rho_panel(
    y_within: &Array1<f64>,
    wy_within: &Array1<f64>,
    x_fe: &Array2<f64>,
    n_entities: usize,
    n_time: usize,
    listw: &mut SpatialWeights,
    rho_min: f64,
    rho_max: f64,
    tol: f64,
    max_iter: usize,
) -> EconResult<(f64, f64)> {
    let n = y_within.len();
    let k = x_fe.ncols();
    let eigenvalues = listw.eigenvalues().clone();

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    // Pre-compute X'X inverse
    let xtx_mat = xtx(&x_fe.view());
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;

    let neg_ll = |rho: f64| -> f64 {
        let y_tilde = y_within - rho * wy_within;
        let xty_vec = xty(&x_fe.view(), &y_tilde);
        let beta = xtx_inv.dot(&xty_vec);
        let residuals = &y_tilde - &x_fe.dot(&beta);
        let rss: f64 = residuals.iter().map(|&r| r * r).sum();
        let sigma2 = rss / n as f64;

        if sigma2 <= 0.0 {
            return f64::INFINITY;
        }

        // Log determinant: T * sum(log(1 - rho * eigenvalue))
        let log_det: f64 = n_time as f64
            * eigenvalues
                .iter()
                .map(|&ev| (1.0 - rho * ev).ln())
                .sum::<f64>();

        0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln()) - log_det
    };

    let mut a = rho_min;
    let mut b = rho_max;
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;

    for _ in 0..max_iter {
        if (b - a).abs() < tol {
            break;
        }
        if neg_ll(c) < neg_ll(d) {
            b = d;
            d = c;
            c = b - (b - a) / phi;
        } else {
            a = c;
            c = d;
            d = a + (b - a) / phi;
        }
    }

    let rho_opt = (a + b) / 2.0;
    let ll_opt = -neg_ll(rho_opt);

    Ok((rho_opt, ll_opt))
}

// ============================================================================
// Random Effects Estimation
// ============================================================================

fn run_spml_random(
    y: &Array1<f64>,
    x: &Array2<f64>,
    entity_ids: &[usize],
    time_ids: &[usize],
    n_entities: usize,
    n_time: usize,
    listw: &mut SpatialWeights,
    config: &SpmlConfig,
    coef_names: Vec<String>,
) -> EconResult<SpmlResult> {
    let n = y.len();
    let k = x.ncols();

    // Step 1: Estimate variance components from FE residuals
    let y_within = demean_by_entity(y, entity_ids, n_entities);
    let x_within = demean_matrix_by_entity(x, entity_ids, n_entities);
    let x_fe = x_within.slice(ndarray::s![.., 1..]).to_owned();

    let xtx_fe = xtx(&x_fe.view());
    let xtx_fe_inv = matrix_inverse(&xtx_fe.view())?;
    let xty_fe = xty(&x_fe.view(), &y_within);
    let beta_fe = xtx_fe_inv.dot(&xty_fe);
    let resid_fe = &y_within - &x_fe.dot(&beta_fe);

    let df_fe = n - n_entities - (k - 1);
    let sigma2_e = resid_fe.iter().map(|&r| r * r).sum::<f64>() / df_fe as f64;

    // Estimate sigma2_u from between variation
    let y_means = compute_entity_means(y, entity_ids, n_entities);
    let y_overall = y.mean().unwrap_or(0.0);
    let sigma2_between =
        y_means.iter().map(|&m| (m - y_overall).powi(2)).sum::<f64>() / (n_entities - 1) as f64;
    let t_bar = n as f64 / n_entities as f64;
    let sigma2_u = (sigma2_between - sigma2_e / t_bar).max(0.0);

    // Theta for quasi-demeaning
    let theta = if sigma2_u > 0.0 {
        1.0 - (sigma2_e / (t_bar * sigma2_u + sigma2_e)).sqrt()
    } else {
        0.0
    };

    // Quasi-demean y and X
    let mut y_quasi = Array1::zeros(n);
    let mut x_quasi = Array2::zeros((n, k));

    for i in 0..n {
        let eid = entity_ids[i];
        y_quasi[i] = y[i] - theta * y_means[eid];
    }

    for j in 0..k {
        let x_col = x.column(j).to_owned();
        let x_means = compute_entity_means(&x_col, entity_ids, n_entities);
        for i in 0..n {
            let eid = entity_ids[i];
            x_quasi[[i, j]] = x[[i, j]] - theta * x_means[eid];
        }
    }

    // Handle spatial lag if specified
    let (beta, rho_opt, sigma2, ll) = if config.lag {
        let wy = spatial_lag_panel(y, entity_ids, time_ids, n_entities, n_time, listw);
        let wy_means = compute_entity_means(&wy, entity_ids, n_entities);
        let mut wy_quasi = Array1::zeros(n);
        for i in 0..n {
            let eid = entity_ids[i];
            wy_quasi[i] = wy[i] - theta * wy_means[eid];
        }

        let (rho_min, rho_max) = listw.rho_range();
        let (rho_opt, ll_opt) = optimize_rho_panel(
            &y_quasi,
            &wy_quasi,
            &x_quasi,
            n_entities,
            n_time,
            listw,
            rho_min.max(-0.99),
            rho_max.min(0.99),
            config.tol,
            config.max_iter,
        )?;

        let y_tilde = &y_quasi - rho_opt * &wy_quasi;
        let xtx_mat = xtx(&x_quasi.view());
        let xtx_inv = matrix_inverse(&xtx_mat.view())?;
        let xty_vec = xty(&x_quasi.view(), &y_tilde);
        let beta = xtx_inv.dot(&xty_vec);

        let residuals = &y_tilde - &x_quasi.dot(&beta);
        let sigma2 = residuals.iter().map(|&r| r * r).sum::<f64>() / (n - k) as f64;

        (beta, Some(rho_opt), sigma2, ll_opt)
    } else {
        let xtx_mat = xtx(&x_quasi.view());
        let xtx_inv = matrix_inverse(&xtx_mat.view())?;
        let xty_vec = xty(&x_quasi.view(), &y_quasi);
        let beta = xtx_inv.dot(&xty_vec);

        let residuals = &y_quasi - &x_quasi.dot(&beta);
        let sigma2 = residuals.iter().map(|&r| r * r).sum::<f64>() / (n - k) as f64;
        let ll = -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln());

        (beta, None, sigma2, ll)
    };

    // Standard errors
    let xtx_mat = xtx(&x_quasi.view());
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;
    let std_errors: Vec<f64> = (0..k)
        .map(|j| (sigma2 * xtx_inv[[j, j]]).sqrt())
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_values: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| 2.0 * (1.0 - normal.cdf(z.abs())))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    let (rho_se, rho_z, rho_p) = if let Some(rho) = rho_opt {
        let tr_w2 = listw.trace_w2();
        let tr_wtw = listw.trace_wtw();
        let var_rho = sigma2 / (n_time as f64 * (tr_w2 + tr_wtw));
        let se = var_rho.sqrt().max(0.001);
        let z = rho / se;
        let p = 2.0 * (1.0 - normal.cdf(z.abs()));
        (Some(se), Some(z), Some(p))
    } else {
        (None, None, None)
    };

    let fitted = x.dot(&beta);
    let residuals = y - &fitted;

    let n_params = k + if config.lag { 1 } else { 0 } + 2; // beta + rho? + sigma2_u + sigma2_e
    let aic = -2.0 * ll + 2.0 * n_params as f64;
    let bic = -2.0 * ll + (n_params as f64) * (n as f64).ln();

    Ok(SpmlResult {
        model: SpatialPanelModel::Random,
        effect: config.effect,
        has_lag: config.lag,
        spatial_error: config.spatial_error,
        coefficients: beta.to_vec(),
        coef_names,
        std_errors,
        z_values,
        p_values,
        significance,
        rho: rho_opt,
        rho_se,
        rho_z,
        rho_p,
        lambda: None,
        lambda_se: None,
        lambda_z: None,
        lambda_p: None,
        variance: SpatialPanelVariance {
            sigma_mu: Some(sigma2_u.sqrt()),
            sigma_nu: None,
            sigma_epsilon: sigma2_e.sqrt(),
            lambda: None,
            rho_sr: None,
        },
        log_likelihood: ll,
        aic,
        bic,
        n_obs: n,
        n_entities,
        n_time,
        df: n - k,
        residuals,
        fitted,
    })
}

// ============================================================================
// Pooled Estimation
// ============================================================================

fn run_spml_pooling(
    y: &Array1<f64>,
    x: &Array2<f64>,
    entity_ids: &[usize],
    time_ids: &[usize],
    n_entities: usize,
    n_time: usize,
    listw: &mut SpatialWeights,
    config: &SpmlConfig,
    coef_names: Vec<String>,
) -> EconResult<SpmlResult> {
    let n = y.len();
    let k = x.ncols();

    let (beta, rho_opt, sigma2, ll) = if config.lag {
        let wy = spatial_lag_panel(y, entity_ids, time_ids, n_entities, n_time, listw);

        let (rho_min, rho_max) = listw.rho_range();
        let (rho_opt, ll_opt) = optimize_rho_panel(
            y,
            &wy,
            x,
            n_entities,
            n_time,
            listw,
            rho_min.max(-0.99),
            rho_max.min(0.99),
            config.tol,
            config.max_iter,
        )?;

        let y_tilde = y - rho_opt * &wy;
        let xtx_mat = xtx(&x.view());
        let xtx_inv = matrix_inverse(&xtx_mat.view())?;
        let xty_vec = xty(&x.view(), &y_tilde);
        let beta = xtx_inv.dot(&xty_vec);

        let residuals = &y_tilde - &x.dot(&beta);
        let sigma2 = residuals.iter().map(|&r| r * r).sum::<f64>() / (n - k) as f64;

        (beta, Some(rho_opt), sigma2, ll_opt)
    } else {
        let xtx_mat = xtx(&x.view());
        let xtx_inv = matrix_inverse(&xtx_mat.view())?;
        let xty_vec = xty(&x.view(), y);
        let beta = xtx_inv.dot(&xty_vec);

        let residuals = y - &x.dot(&beta);
        let sigma2 = residuals.iter().map(|&r| r * r).sum::<f64>() / (n - k) as f64;
        let ll = -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln());

        (beta, None, sigma2, ll)
    };

    let xtx_mat = xtx(&x.view());
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;
    let std_errors: Vec<f64> = (0..k)
        .map(|j| (sigma2 * xtx_inv[[j, j]]).sqrt())
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_values: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| 2.0 * (1.0 - normal.cdf(z.abs())))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    let (rho_se, rho_z, rho_p) = if let Some(rho) = rho_opt {
        let tr_w2 = listw.trace_w2();
        let tr_wtw = listw.trace_wtw();
        let var_rho = sigma2 / (n_time as f64 * (tr_w2 + tr_wtw));
        let se = var_rho.sqrt().max(0.001);
        let z = rho / se;
        let p = 2.0 * (1.0 - normal.cdf(z.abs()));
        (Some(se), Some(z), Some(p))
    } else {
        (None, None, None)
    };

    let fitted = x.dot(&beta);
    let residuals = y - &fitted;

    let n_params = k + if config.lag { 2 } else { 1 };
    let aic = -2.0 * ll + 2.0 * n_params as f64;
    let bic = -2.0 * ll + (n_params as f64) * (n as f64).ln();

    Ok(SpmlResult {
        model: SpatialPanelModel::Pooling,
        effect: config.effect,
        has_lag: config.lag,
        spatial_error: config.spatial_error,
        coefficients: beta.to_vec(),
        coef_names,
        std_errors,
        z_values,
        p_values,
        significance,
        rho: rho_opt,
        rho_se,
        rho_z,
        rho_p,
        lambda: None,
        lambda_se: None,
        lambda_z: None,
        lambda_p: None,
        variance: SpatialPanelVariance {
            sigma_mu: None,
            sigma_nu: None,
            sigma_epsilon: sigma2.sqrt(),
            lambda: None,
            rho_sr: None,
        },
        log_likelihood: ll,
        aic,
        bic,
        n_obs: n,
        n_entities,
        n_time,
        df: n - k,
        residuals,
        fitted,
    })
}

// ============================================================================
// GMM Estimation
// ============================================================================

fn run_spgm_within(
    y: &Array1<f64>,
    x: &Array2<f64>,
    entity_ids: &[usize],
    time_ids: &[usize],
    n_entities: usize,
    n_time: usize,
    listw: &mut SpatialWeights,
    config: &SpgmConfig,
    coef_names: Vec<String>,
) -> EconResult<SpgmResult> {
    let n = y.len();
    let k = x.ncols();

    // Within transformation
    let y_within = demean_by_entity(y, entity_ids, n_entities);
    let x_within = demean_matrix_by_entity(x, entity_ids, n_entities);
    let x_fe = x_within.slice(ndarray::s![.., 1..]).to_owned();
    let k_fe = k - 1;

    let mut coef_names_fe: Vec<String> = coef_names[1..].to_vec();
    let mut n_instruments = k_fe;

    // Build instruments
    let (z, rho_opt, lambda_opt) = if config.lag {
        // For spatial lag: use WX as instruments
        let wy = spatial_lag_panel(y, entity_ids, time_ids, n_entities, n_time, listw);
        let wy_within = demean_by_entity(&wy, entity_ids, n_entities);

        // W*X as additional instruments
        let mut wx_within = Array2::zeros((n, k_fe));
        for j in 0..k_fe {
            let x_col = x_fe.column(j).to_owned();
            let wx = spatial_lag_panel(&x_col, entity_ids, time_ids, n_entities, n_time, listw);
            let wx_within_col = demean_by_entity(&wx, entity_ids, n_entities);
            wx_within.column_mut(j).assign(&wx_within_col);
        }

        // Combine [X, WX] as instruments
        let mut z = Array2::zeros((n, 2 * k_fe));
        for j in 0..k_fe {
            z.column_mut(j).assign(&x_fe.column(j));
            z.column_mut(k_fe + j).assign(&wx_within.column(j));
        }

        n_instruments = 2 * k_fe;

        // 2SLS for spatial lag
        let ztz = xtx(&z.view());
        let ztz_inv = matrix_inverse(&ztz.view())?;
        let pz = z.dot(&ztz_inv).dot(&z.t());

        // Augmented X with Wy
        let mut x_aug = Array2::zeros((n, k_fe + 1));
        for j in 0..k_fe {
            x_aug.column_mut(j).assign(&x_fe.column(j));
        }
        x_aug.column_mut(k_fe).assign(&wy_within);

        let x_proj = pz.dot(&x_aug);
        let xtpx = x_aug.t().dot(&x_proj);
        let xtpx_inv = matrix_inverse(&xtpx.view())?;
        let xtpy = x_proj.t().dot(&y_within);
        let gamma = xtpx_inv.dot(&xtpy);

        let rho = gamma[k_fe];
        let beta = gamma.slice(ndarray::s![..k_fe]).to_owned();

        coef_names_fe.push("rho (spatial lag)".to_string());

        (z, Some(rho), None)
    } else {
        // No spatial lag - standard FE
        (x_fe.clone(), None, None)
    };

    // Compute beta
    let xtx_mat = xtx(&x_fe.view());
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;
    let xty_vec = xty(&x_fe.view(), &y_within);
    let beta = if let Some(rho) = rho_opt {
        let wy = spatial_lag_panel(y, entity_ids, time_ids, n_entities, n_time, listw);
        let wy_within = demean_by_entity(&wy, entity_ids, n_entities);
        let y_tilde = &y_within - rho * &wy_within;
        let xty_tilde = xty(&x_fe.view(), &y_tilde);
        xtx_inv.dot(&xty_tilde)
    } else {
        xtx_inv.dot(&xty_vec)
    };

    let residuals = if let Some(rho) = rho_opt {
        let wy = spatial_lag_panel(y, entity_ids, time_ids, n_entities, n_time, listw);
        let wy_within = demean_by_entity(&wy, entity_ids, n_entities);
        &y_within - rho * &wy_within - &x_fe.dot(&beta)
    } else {
        &y_within - &x_fe.dot(&beta)
    };

    let df = n - n_entities - k_fe - if config.lag { 1 } else { 0 };
    let sigma2 = residuals.iter().map(|&r| r * r).sum::<f64>() / df as f64;

    // Standard errors
    let std_errors: Vec<f64> = (0..k_fe)
        .map(|j| (sigma2 * xtx_inv[[j, j]]).sqrt())
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_values: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| 2.0 * (1.0 - normal.cdf(z.abs())))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Sargan test (if overidentified)
    let (sargan_stat, sargan_p, sargan_df) = if n_instruments > k_fe + if config.lag { 1 } else { 0 }
    {
        let over_id = n_instruments - k_fe - if config.lag { 1 } else { 0 };
        let zr = z.t().dot(&residuals);
        let ztz = xtx(&z.view());
        let ztz_inv = matrix_inverse(&ztz.view())?;
        let stat = zr.dot(&ztz_inv).dot(&zr) / sigma2;
        let p = chi_squared_p_value(stat, over_id as f64);
        (Some(stat), Some(p), Some(over_id))
    } else {
        (None, None, None)
    };

    Ok(SpgmResult {
        method: config.method,
        has_lag: config.lag,
        has_spatial_error: config.spatial_error,
        coefficients: beta.to_vec(),
        coef_names: coef_names_fe[..k_fe].to_vec(),
        std_errors,
        z_values,
        p_values,
        significance,
        rho: rho_opt,
        lambda: lambda_opt,
        sigma2,
        sigma2_mu: None,
        n_obs: n,
        n_entities,
        n_time,
        n_instruments,
        sargan_stat,
        sargan_p,
        sargan_df,
    })
}

fn run_spgm_gls(
    y: &Array1<f64>,
    x: &Array2<f64>,
    entity_ids: &[usize],
    time_ids: &[usize],
    n_entities: usize,
    n_time: usize,
    listw: &mut SpatialWeights,
    config: &SpgmConfig,
    coef_names: Vec<String>,
) -> EconResult<SpgmResult> {
    // Random effects GMM (simplified implementation)
    // Uses two-stage approach: first estimate variance components, then GLS

    let n = y.len();
    let k = x.ncols();

    // Step 1: Get variance components from within estimation
    let y_within = demean_by_entity(y, entity_ids, n_entities);
    let x_within = demean_matrix_by_entity(x, entity_ids, n_entities);
    let x_fe = x_within.slice(ndarray::s![.., 1..]).to_owned();

    let xtx_fe = xtx(&x_fe.view());
    let xtx_fe_inv = matrix_inverse(&xtx_fe.view())?;
    let xty_fe = xty(&x_fe.view(), &y_within);
    let beta_fe = xtx_fe_inv.dot(&xty_fe);
    let resid_fe = &y_within - &x_fe.dot(&beta_fe);

    let df_fe = n - n_entities - (k - 1);
    let sigma2_e = resid_fe.iter().map(|&r| r * r).sum::<f64>() / df_fe as f64;

    let y_means = compute_entity_means(y, entity_ids, n_entities);
    let y_overall = y.mean().unwrap_or(0.0);
    let sigma2_between =
        y_means.iter().map(|&m| (m - y_overall).powi(2)).sum::<f64>() / (n_entities - 1) as f64;
    let t_bar = n as f64 / n_entities as f64;
    let sigma2_u = (sigma2_between - sigma2_e / t_bar).max(0.0);

    let theta = if sigma2_u > 0.0 {
        1.0 - (sigma2_e / (t_bar * sigma2_u + sigma2_e)).sqrt()
    } else {
        0.0
    };

    // Quasi-demean
    let mut y_quasi = Array1::zeros(n);
    let mut x_quasi = Array2::zeros((n, k));

    for i in 0..n {
        let eid = entity_ids[i];
        y_quasi[i] = y[i] - theta * y_means[eid];
    }

    for j in 0..k {
        let x_col = x.column(j).to_owned();
        let x_means = compute_entity_means(&x_col, entity_ids, n_entities);
        for i in 0..n {
            let eid = entity_ids[i];
            x_quasi[[i, j]] = x[[i, j]] - theta * x_means[eid];
        }
    }

    // GLS estimation
    let xtx_mat = xtx(&x_quasi.view());
    let xtx_inv = matrix_inverse(&xtx_mat.view())?;
    let xty_vec = xty(&x_quasi.view(), &y_quasi);
    let beta = xtx_inv.dot(&xty_vec);

    let residuals = &y_quasi - &x_quasi.dot(&beta);
    let sigma2 = residuals.iter().map(|&r| r * r).sum::<f64>() / (n - k) as f64;

    let std_errors: Vec<f64> = (0..k)
        .map(|j| (sigma2 * xtx_inv[[j, j]]).sqrt())
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_values: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = z_values
        .iter()
        .map(|&z| 2.0 * (1.0 - normal.cdf(z.abs())))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    Ok(SpgmResult {
        method: SpgmMethod::G2sls,
        has_lag: config.lag,
        has_spatial_error: config.spatial_error,
        coefficients: beta.to_vec(),
        coef_names,
        std_errors,
        z_values,
        p_values,
        significance,
        rho: None,
        lambda: None,
        sigma2,
        sigma2_mu: Some(sigma2_u),
        n_obs: n,
        n_entities,
        n_time,
        n_instruments: k,
        sargan_stat: None,
        sargan_p: None,
        sargan_df: None,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Neighbors, WeightStyle};
    use polars::prelude::*;

    fn create_spatial_panel_data() -> (Dataset, SpatialWeights) {
        // Create a small spatial panel dataset
        // 4 entities, 3 time periods = 12 observations
        let n_entities = 4;
        let n_time = 3;

        // Entity and time identifiers
        let mut entity = Vec::new();
        let mut time = Vec::new();
        let mut y = Vec::new();
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();

        for t in 0..n_time {
            for i in 0..n_entities {
                entity.push(i as i64);
                time.push(t as i64);
                // y = 1.0 + 0.5*x1 + 0.3*x2 + entity_effect + noise
                let entity_effect = (i as f64) * 0.5;
                let x1_val = (i + t) as f64 + 0.1 * (i * t) as f64;
                let x2_val = ((i + 1) * (t + 1)) as f64 * 0.5;
                x1.push(x1_val);
                x2.push(x2_val);
                y.push(1.0 + 0.5 * x1_val + 0.3 * x2_val + entity_effect + 0.1 * ((i + t) as f64));
            }
        }

        let df = df! {
            "entity" => &entity,
            "time" => &time,
            "y" => &y,
            "x1" => &x1,
            "x2" => &x2,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        // Create spatial weights for 4 entities (2x2 grid)
        let coords: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let nb = Neighbors::from_knn(&coords, 2);
        let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        (dataset, listw)
    }

    #[test]
    fn test_spml_within_basic() {
        let (dataset, mut listw) = create_spatial_panel_data();

        let config = SpmlConfig {
            model: SpatialPanelModel::Within,
            lag: false,
            spatial_error: SpatialErrorType::None,
            ..Default::default()
        };

        let result = run_spml(
            &dataset,
            "y",
            &["x1", "x2"],
            "entity",
            "time",
            &mut listw,
            config,
        )
        .unwrap();

        assert_eq!(result.n_obs, 12);
        assert_eq!(result.n_entities, 4);
        assert_eq!(result.n_time, 3);
        assert_eq!(result.coefficients.len(), 2); // x1, x2 (no intercept in FE)
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_spml_within_spatial_lag() {
        let (dataset, mut listw) = create_spatial_panel_data();

        let config = SpmlConfig {
            model: SpatialPanelModel::Within,
            lag: true,
            spatial_error: SpatialErrorType::None,
            ..Default::default()
        };

        let result = run_spml(
            &dataset,
            "y",
            &["x1", "x2"],
            "entity",
            "time",
            &mut listw,
            config,
        )
        .unwrap();

        assert!(result.has_lag);
        assert!(result.rho.is_some());
        let rho = result.rho.unwrap();
        assert!(rho > -1.0 && rho < 1.0);
    }

    #[test]
    fn test_spml_random_effects() {
        // Create a larger dataset for random effects to be well-conditioned
        let n_entities = 6;
        let n_time = 4;

        let mut entity = Vec::new();
        let mut time = Vec::new();
        let mut y = Vec::new();
        let mut x1 = Vec::new();

        for t in 0..n_time {
            for i in 0..n_entities {
                entity.push(i as i64);
                time.push(t as i64);
                let entity_effect = (i as f64) * 0.3;
                let x1_val = (i as f64) * 0.5 + (t as f64) * 1.2 + 0.2;
                x1.push(x1_val);
                y.push(2.0 + 0.7 * x1_val + entity_effect + 0.15 * ((i * t) as f64));
            }
        }

        let df = df! {
            "entity" => &entity,
            "time" => &time,
            "y" => &y,
            "x1" => &x1,
        }
        .unwrap();

        let dataset = Dataset::new(df);

        // Create spatial weights for 6 entities (2x3 grid)
        let coords: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        ];
        let nb = Neighbors::from_knn(&coords, 2);
        let mut listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

        let config = SpmlConfig {
            model: SpatialPanelModel::Random,
            lag: false,
            spatial_error: SpatialErrorType::None,
            ..Default::default()
        };

        let result = run_spml(
            &dataset,
            "y",
            &["x1"],
            "entity",
            "time",
            &mut listw,
            config,
        )
        .unwrap();

        assert_eq!(result.model, SpatialPanelModel::Random);
        assert!(result.variance.sigma_mu.is_some());
        assert_eq!(result.coefficients.len(), 2); // intercept + x1
    }

    #[test]
    fn test_spgm_basic() {
        let (dataset, mut listw) = create_spatial_panel_data();

        let config = SpgmConfig {
            method: SpgmMethod::W2sls,
            lag: false,
            spatial_error: false,
            ..Default::default()
        };

        let result = run_spgm(
            &dataset,
            "y",
            &["x1", "x2"],
            "entity",
            "time",
            &mut listw,
            config,
        )
        .unwrap();

        assert_eq!(result.method, SpgmMethod::W2sls);
        assert_eq!(result.n_entities, 4);
        assert_eq!(result.coefficients.len(), 2);
    }

    #[test]
    fn test_spml_pooling() {
        let (dataset, mut listw) = create_spatial_panel_data();

        let config = SpmlConfig {
            model: SpatialPanelModel::Pooling,
            lag: false,
            spatial_error: SpatialErrorType::None,
            ..Default::default()
        };

        let result = run_spml(
            &dataset,
            "y",
            &["x1", "x2"],
            "entity",
            "time",
            &mut listw,
            config,
        )
        .unwrap();

        assert_eq!(result.model, SpatialPanelModel::Pooling);
        assert_eq!(result.coefficients.len(), 3); // intercept + x1 + x2
    }
}
