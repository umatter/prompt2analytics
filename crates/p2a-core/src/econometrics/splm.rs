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
//!
//! The panel layout the estimators require (balanced, W rows in sorted entity-id order,
//! rows in any order, no missing values) is described under "Panel layout" on [`run_spml`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::cell::RefCell;
use std::f64::consts::PI;
use std::fmt;
use std::rc::Rc;

use super::spatial::validate_optimizer;
use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{eig_symmetric, matrix_inverse, ndarray_to_faer, xtx, xty};
use crate::linalg::solve::{SingularPolicy, least_squares, sym_inverse};
use crate::resource::{checkpoint, try_with_capacity, try_zeros1, try_zeros2};
use crate::spatial::SpatialWeights;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, normal_p_value_two_sided};

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

crate::option_enum!(
    SpatialPanelEffect,
    crate::options::Choices {
        option: "spatial panel effect",
        values: &[
            (
                &["individual", "id", "entity", "unit"],
                SpatialPanelEffect::Individual
            ),
            (&["time", "period"], SpatialPanelEffect::Time),
            (
                &["twoways", "two_ways", "twoway", "two_way", "both"],
                SpatialPanelEffect::TwoWays
            ),
        ],
    },
    no_display
);

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

crate::option_enum!(
    SpatialPanelModel,
    crate::options::Choices {
        option: "spatial panel model",
        values: &[
            (
                &["within", "fe", "fixed", "fixed_effects"],
                SpatialPanelModel::Within
            ),
            (
                &["random", "re", "random_effects"],
                SpatialPanelModel::Random
            ),
            (&["pooling", "pool", "pooled"], SpatialPanelModel::Pooling),
        ],
    },
    no_display
);

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

crate::option_enum!(
    SpatialErrorType,
    crate::options::Choices {
        option: "spatial error",
        values: &[
            (&["none", "no"], SpatialErrorType::None),
            (&["b", "baltagi", "sem"], SpatialErrorType::Baltagi),
            (&["kkp", "sem2", "kapoor"], SpatialErrorType::KKP),
        ],
    },
    no_display
);

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
    /// Standard deviation sigma_mu of the individual effect (random effects; 0 at the
    /// boundary phi = 0). The field has always held the standard deviation, not the variance.
    pub sigma_mu: Option<f64>,
    /// Time effect standard deviation (not estimated by any model; `None`)
    pub sigma_nu: Option<f64>,
    /// ML standard deviation of the idiosyncratic error, sqrt(e'e / (N T)) (of eps in the
    /// random-effects model)
    pub sigma_epsilon: f64,
    /// Spatial error parameter lambda, whenever an error term is estimated
    pub lambda: Option<f64>,
    /// Serial correlation parameter (rho_sr; not estimated, `None`)
    pub rho_sr: Option<f64>,
    /// phi = sigma_mu^2 / sigma_epsilon^2, the random-effects likelihood parameter (splm's
    /// phi); `None` for within and pooled models.
    #[serde(default)]
    pub phi: Option<f64>,
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
    /// Residual degrees of freedom: observations minus absorbed effects, coefficients and
    /// spatial parameters
    pub df: usize,

    /// Residuals A y - X beta on the scale the model is estimated on (demeaned for within
    /// models, the composite error for random effects); skipped for serialization
    #[serde(skip)]
    pub residuals: Array1<f64>,
    /// Fitted values, y - residuals on the same scale (skipped for serialization)
    #[serde(skip)]
    pub fitted: Array1<f64>,

    /// Every estimated parameter, in the order of `parameter_se`: the coefficients, then
    /// "rho", "phi", "lambda" (those estimated) and "sigma2".
    #[serde(default)]
    pub parameter_names: Vec<String>,
    /// Standard errors from the inverse observed information of the full log-likelihood
    /// (NaN for a parameter at a boundary).
    #[serde(default)]
    pub parameter_se: Vec<f64>,
    /// Newton on the concentrated likelihood; `None` for the closed-form non-spatial
    /// within or pooled fit.
    #[serde(default)]
    pub convergence: Option<Convergence>,
    /// Warnings about the fit.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for SpmlResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

impl fmt::Display for SpmlResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\nSpatial Panel Model (ML)")?;
        writeln!(f, "{}", "=".repeat(60))?;
        let effect = if self.model == SpatialPanelModel::Pooling {
            "none (pooled)".to_string()
        } else {
            self.effect.to_string()
        };
        writeln!(f, "Model: {}  Effect: {}", self.model, effect)?;
        let error = if self.spatial_error == SpatialErrorType::None {
            "none".to_string()
        } else {
            self.spatial_error.to_string()
        };
        writeln!(
            f,
            "Spatial lag: {}  Spatial error: {}",
            if self.has_lag { "yes" } else { "no" },
            error
        )?;
        writeln!(
            f,
            "Observations: {}  Entities: {}  Time periods: {}",
            self.n_obs, self.n_entities, self.n_time
        )?;
        writeln!(
            f,
            "Log-Likelihood: {:.4}  AIC: {:.4}  BIC: {:.4}",
            self.log_likelihood, self.aic, self.bic
        )?;
        writeln!(f, "{}", "-".repeat(60))?;

        let block = |f: &mut fmt::Formatter<'_>,
                     title: &str,
                     v: f64,
                     se: Option<f64>,
                     z: Option<f64>,
                     p: Option<f64>|
         -> fmt::Result {
            writeln!(f, "\n{title}:")?;
            writeln!(
                f,
                "  Estimate: {:.6}  SE: {:.6}  Z: {:.4}  P: {:.4}",
                v,
                se.unwrap_or(f64::NAN),
                z.unwrap_or(f64::NAN),
                p.unwrap_or(f64::NAN)
            )
        };
        if let Some(rho) = self.rho {
            block(
                f,
                "Spatial lag coefficient (rho)",
                rho,
                self.rho_se,
                self.rho_z,
                self.rho_p,
            )?;
        }
        if let Some(lambda) = self.lambda {
            block(
                f,
                "Spatial error coefficient (lambda)",
                lambda,
                self.lambda_se,
                self.lambda_z,
                self.lambda_p,
            )?;
        }

        writeln!(
            f,
            "\n{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coefficient", "Std.Error", "Z-value", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(60))?;

        for i in 0..self.coef_names.len() {
            writeln!(
                f,
                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4}{}",
                self.coef_names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_values[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(60))?;
        writeln!(f, "Variance components:")?;
        if let Some(sigma_mu) = self.variance.sigma_mu {
            writeln!(f, "  sigma_mu (individual): {:.6}", sigma_mu)?;
        }
        if let Some(phi) = self.variance.phi {
            let se = self
                .parameter_names
                .iter()
                .position(|n| n == "phi")
                .and_then(|i| self.parameter_se.get(i).copied())
                .unwrap_or(f64::NAN);
            writeln!(f, "  phi = sigma_mu^2/sigma^2: {:.6} (SE {:.6})", phi, se)?;
        }
        writeln!(
            f,
            "  sigma_epsilon (error, ML): {:.6}",
            self.variance.sigma_epsilon
        )?;
        writeln!(
            f,
            "Standard errors: inverse observed information of the full log-likelihood"
        )?;
        write!(f, "{}", notes_text(self))
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
        writeln!(
            f,
            "Spatial Lag: {}  Spatial Error: {}",
            self.has_lag, self.has_spatial_error
        )?;
        writeln!(
            f,
            "Observations: {}  Entities: {}  Time periods: {}",
            self.n_obs, self.n_entities, self.n_time
        )?;
        writeln!(f, "Instruments: {}", self.n_instruments)?;
        writeln!(f, "{}", "-".repeat(60))?;

        writeln!(
            f,
            "\n{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coefficient", "Std.Error", "Z-value", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(60))?;

        for i in 0..self.coef_names.len() {
            writeln!(
                f,
                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4}{}",
                self.coef_names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_values[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
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
/// # Panel layout
///
/// - The panel must be balanced: every entity is observed exactly once in every period.
///   A duplicated (entity, time) pair or a missing cell is `InvalidSpecification` naming
///   the first one (`splm::spml` refuses unbalanced panels too).
/// - W is N x N, and row i of W belongs to the i-th entity in sorted id order: numbers
///   ascending, text in byte order (R `factor()` under `LC_COLLATE=C`, splm's
///   convention; an R session in a UTF-8 locale sorts text case-insensitively). When the
///   weights' neighbours carry region ids, they must equal the sorted entity labels.
/// - Rows may come in any order; the estimates do not depend on it, and `residuals` and
///   `fitted` are returned in the dataset's row order.
/// - Missing values (null or NaN) in y, X, the entity or the time column are refused
///   (`NullValues`); integer and Boolean columns are accepted; time holds whole numbers
///   or dates.
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
    validate_optimizer(config.tol, config.max_iter)?;
    check_spml_config(&config)?;

    // Balanced panel in canonical layout (period blocks of entities in W row order)
    let panel = extract_panel_data(dataset, y_col, x_cols, entity_col, time_col)?;
    check_panel_weights(listw, &panel)?;

    let mut result = match config.model {
        SpatialPanelModel::Random => fit_spml_random(&panel, listw, &config, y_col, x_cols)?,
        SpatialPanelModel::Within | SpatialPanelModel::Pooling => {
            fit_spml_fe(&panel, listw, &config, y_col, x_cols)?
        }
    };

    // Back to the dataset's row order
    result.residuals = to_row_order(&result.residuals, &panel.rows);
    result.fitted = to_row_order(&result.fitted, &panel.rows);
    Ok(result)
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
/// # Panel layout
///
/// As for [`run_spml`]: a balanced panel with unique (entity, time) pairs, rows in any
/// order, W row i = the i-th entity in sorted id order (numbers ascending, text in byte
/// order; region ids, when present, must match), and no missing values in y, X, the
/// entity or the time column.
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
    validate_optimizer(config.tol, config.max_iter)?;
    let panel = extract_panel_data(dataset, y_col, x_cols, entity_col, time_col)?;
    check_panel_weights(listw, &panel)?;

    let mut coef_names = vec!["(Intercept)".to_string()];
    coef_names.extend(x_cols.iter().map(|s| s.to_string()));

    let (y, x, entity_ids) = (&panel.y, &panel.x, &panel.entity_ids);
    let (n_entities, n_time) = (panel.n_entities, panel.n_time);

    // Dispatch based on method
    match config.method {
        SpgmMethod::W2sls => run_spgm_within(
            y, x, entity_ids, n_entities, n_time, listw, &config, coef_names,
        ),
        SpgmMethod::G2sls => run_spgm_gls(
            y, x, entity_ids, n_entities, n_time, listw, &config, coef_names,
        ),
        _ => {
            // Default to within for other methods (simplified)
            run_spgm_within(
                y, x, entity_ids, n_entities, n_time, listw, &config, coef_names,
            )
        }
    }
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

/// A balanced spatial panel in canonical layout: time-major (period 0's N rows, then period
/// 1's, ...), with entities in sorted-id order inside each period, so period t is rows
/// `t*N .. (t+1)*N` and row i of a cross-section is entity i, matching W row i.
#[derive(Debug)]
struct SpatialPanelData {
    /// Outcome, canonical order.
    y: Array1<f64>,
    /// `[1, X]`, canonical order.
    x: Array2<f64>,
    /// Entity code (= W row) of each canonical row.
    entity_ids: Vec<usize>,
    n_entities: usize,
    n_time: usize,
    /// Entity labels in sorted order (numbers numerically, text in byte order).
    entity_labels: Vec<String>,
    /// Dataset row of each canonical row (a permutation of `0..n`).
    rows: Vec<usize>,
}

/// Read a spatial panel (y, `[1, X]`, entity, time) and put it in canonical layout.
///
/// Every column goes through one `ColumnRequest` with `NaPolicy::Fail`: a missing value in
/// y, X, the entity or the time column is `NullValues`; entities are a factor in R
/// `factor()` order; times are a whole-number index. Duplicated (entity, time) pairs and
/// an unbalanced panel are `InvalidSpecification` naming the first offending cell.
fn extract_panel_data(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
) -> EconResult<SpatialPanelData> {
    let df = dataset.df();
    let cols = ColumnRequest::new(df)
        .numeric(y_col)
        .numerics(x_cols)
        .factor(entity_col)
        .time_index(time_col)
        .na(NaPolicy::Fail)
        .extract()?;
    let n = cols.n();
    let entity = cols.factor(entity_col);
    let n_entities = entity.n_levels();
    let time_values = cols.time_index(time_col);

    // Sorted unique periods; a row's time code is its period's position.
    let mut times: Vec<i64> = try_with_capacity(n, "spatial panel periods")?;
    times.extend_from_slice(time_values);
    times.sort_unstable();
    times.dedup();
    let n_time = times.len();
    let mut time_code: Vec<usize> = try_with_capacity(n, "spatial panel time codes")?;
    for &t in time_values {
        let code = times
            .binary_search(&t)
            .map_err(|_| EconError::InvalidSpecification {
                message: format!("time value {t} of column '{time_col}' could not be indexed"),
            })?;
        time_code.push(code);
    }
    let key = |r: usize| (time_code[r], entity.codes[r] as usize);

    // Rows in canonical order: by (time, entity).
    let mut perm: Vec<usize> = try_with_capacity(n, "spatial panel row order")?;
    perm.extend(0..n);
    perm.sort_unstable_by_key(|&r| key(r));

    let cell = |(t, e): (usize, usize)| format!("entity '{}', time {}", entity.levels[e], times[t]);

    // Each (entity, time) pair at most once.
    let mut duplicates = 0usize;
    let mut first_duplicate = None;
    for pair in perm.windows(2) {
        if key(pair[0]) == key(pair[1]) {
            duplicates += 1;
            first_duplicate.get_or_insert(key(pair[1]));
        }
    }
    if let Some(first) = first_duplicate {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Spatial panel has {duplicates} duplicated (entity, time) pair(s) (first: {}); \
                 each entity may appear once per period.",
                cell(first)
            ),
        });
    }

    // Every entity in every period. Without duplicates n < N*T here, so the first canonical
    // position whose row is not the expected (period, entity) cell, or position n when all
    // n rows match, is the first missing cell.
    let cells = n_entities as u128 * n_time as u128;
    if n as u128 != cells {
        let first_missing = (0..n)
            .find(|&p| key(perm[p]) != (p / n_entities, p % n_entities))
            .unwrap_or(n);
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Spatial panel models need a balanced panel: {} of {cells} (entity, time) cells \
                 are missing (first: {}). splm::spml refuses unbalanced panels too; keep the \
                 entities observed in every period (or the periods observed for every entity) \
                 and build W for exactly those entities.",
                cells - n as u128,
                cell((first_missing / n_entities, first_missing % n_entities))
            ),
        });
    }

    // Canonical arrays.
    let k = x_cols.len() + 1;
    let y_values = cols.numeric(y_col);
    let mut y = try_zeros1(n, "spatial panel outcome")?;
    let mut x = try_zeros2(n, k, "spatial panel design")?;
    x.column_mut(0).fill(1.0);
    for (p, &r) in perm.iter().enumerate() {
        y[p] = y_values[r];
    }
    for (j, &name) in x_cols.iter().enumerate() {
        let values = cols.numeric(name);
        for (p, &r) in perm.iter().enumerate() {
            x[[p, j + 1]] = values[r];
        }
    }
    let mut entity_ids: Vec<usize> = try_with_capacity(n, "spatial panel entity codes")?;
    entity_ids.extend(perm.iter().map(|&r| entity.codes[r] as usize));
    let mut rows: Vec<usize> = try_with_capacity(n, "spatial panel rows")?;
    rows.extend(perm.iter().map(|&r| cols.rows()[r]));

    Ok(SpatialPanelData {
        y,
        x,
        entity_ids,
        n_entities,
        n_time,
        entity_labels: entity.levels.clone(),
        rows,
    })
}

/// W must be N x N, and when its neighbours carry region ids they must equal the panel's
/// sorted entity labels, in order.
fn check_panel_weights(listw: &SpatialWeights, panel: &SpatialPanelData) -> EconResult<()> {
    let labels = &panel.entity_labels;
    if listw.n() != panel.n_entities {
        let mut shown: Vec<&str> = labels.iter().take(3).map(String::as_str).collect();
        if labels.len() > 3 {
            shown.push("...");
        }
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Spatial weights matrix has {} units but the panel has {} entities. W must be \
                 N x N, and row i of W belongs to the i-th entity in sorted id order ({}; \
                 numbers ascending, text in byte order).",
                listw.n(),
                panel.n_entities,
                shown.join(", ")
            ),
        });
    }
    if let Some(ids) = listw.neighbors().region_ids()
        && let Some((r, (id, label))) = ids
            .iter()
            .zip(labels)
            .enumerate()
            .find(|(_, (id, label))| id != label)
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Spatial weights row {r} has region id '{id}' but entity {r} of the panel in \
                 sorted id order is '{label}'; W rows must follow the sorted entity ids."
            ),
        });
    }
    Ok(())
}

/// `out[rows[p]] = v[p]`: canonical order back to dataset row order.
fn to_row_order(v: &Array1<f64>, rows: &[usize]) -> Array1<f64> {
    let mut out = Array1::zeros(v.len());
    for (&r, &value) in rows.iter().zip(v.iter()) {
        out[r] = value;
    }
    out
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

/// W applied to each period block of a canonical-layout vector (`SpatialPanelData`):
/// block t is entries `t*N .. (t+1)*N`, entity i at offset i.
fn spatial_lag_panel(
    y: &Array1<f64>,
    n_entities: usize,
    n_time: usize,
    listw: &SpatialWeights,
) -> Array1<f64> {
    debug_assert_eq!(y.len(), n_entities * n_time);
    let mut wy = Array1::zeros(y.len());
    for t in 0..n_time {
        let block = s![t * n_entities..(t + 1) * n_entities];
        let wy_t = listw.lag(&y.slice(block).to_owned());
        wy.slice_mut(block).assign(&wy_t);
    }
    wy
}

// ============================================================================
// Configuration checks and panel helpers
// ============================================================================

/// Combinations of model and effect that `run_spml` does not estimate.
fn check_spml_config(config: &SpmlConfig) -> EconResult<()> {
    let effect = SpatialPanelEffect::choices().canonical(config.effect);
    match (config.model, config.effect) {
        (SpatialPanelModel::Within, _) | (_, SpatialPanelEffect::Individual) => Ok(()),
        (SpatialPanelModel::Pooling, _) => Err(EconError::InvalidSpecification {
            message: format!(
                "model = 'pooling' estimates no individual or time effects, so effect = \
                 '{effect}' does not apply; use model = 'within' with effect = '{effect}', or \
                 leave effect at 'individual'"
            ),
        }),
        (SpatialPanelModel::Random, _) => Err(EconError::InvalidSpecification {
            message: format!(
                "random effects are individual effects (as splm::spml); effect = '{effect}' is \
                 not available with model = 'random'; use model = 'within' for time or two-way \
                 fixed effects"
            ),
        }),
    }
}

/// Effects a model sweeps out of y and X (canonical layout: index s * N + i is entity i in
/// period s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Absorbed {
    None,
    Individual,
    Time,
    TwoWays,
}

impl Absorbed {
    fn of(config: &SpmlConfig) -> Self {
        match (config.model, config.effect) {
            (SpatialPanelModel::Pooling, _) => Absorbed::None,
            (_, SpatialPanelEffect::Individual) => Absorbed::Individual,
            (_, SpatialPanelEffect::Time) => Absorbed::Time,
            (_, SpatialPanelEffect::TwoWays) => Absorbed::TwoWays,
        }
    }

    /// Number of effects absorbed (the degrees of freedom they use).
    fn count(self, n: usize, t: usize) -> usize {
        match self {
            Absorbed::None => 0,
            Absorbed::Individual => n,
            Absorbed::Time => t,
            Absorbed::TwoWays => n + t - 1,
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Absorbed::None => "pooled data",
            Absorbed::Individual => "entity demeaning",
            Absorbed::Time => "period demeaning",
            Absorbed::TwoWays => "two-way demeaning",
        }
    }
}

/// Entity means of a canonical-layout vector (N entries).
fn unit_means(v: ArrayView1<f64>, n: usize, t: usize) -> Array1<f64> {
    let mut m = Array1::<f64>::zeros(n);
    for p in 0..t {
        m += &v.slice(s![p * n..(p + 1) * n]);
    }
    m / t as f64
}

/// R: `v - ave(v, id)` | `v - ave(v, time)` | `v - ave(v, id) - ave(v, time) + mean(v)`
/// (balanced panel, canonical layout).
fn demean_panel(v: ArrayView1<f64>, a: Absorbed, n: usize, t: usize) -> Array1<f64> {
    if a == Absorbed::None {
        return v.to_owned();
    }
    let unit = unit_means(v, n, t);
    let period: Vec<f64> = (0..t)
        .map(|p| v.slice(s![p * n..(p + 1) * n]).sum() / n as f64)
        .collect();
    let grand = v.sum() / (n * t) as f64;
    Array1::from_shape_fn(n * t, |q| {
        let (p, i) = (q / n, q % n);
        match a {
            Absorbed::Individual => v[q] - unit[i],
            Absorbed::Time => v[q] - period[p],
            Absorbed::TwoWays => v[q] - unit[i] - period[p] + grand,
            Absorbed::None => v[q],
        }
    })
}

/// [`demean_panel`] column by column.
fn demean_panel_matrix(
    x: ArrayView2<f64>,
    a: Absorbed,
    n: usize,
    t: usize,
) -> EconResult<Array2<f64>> {
    let mut out = try_zeros2(x.nrows(), x.ncols(), "demeaned spatial panel design")?;
    for (j, col) in x.columns().into_iter().enumerate() {
        out.column_mut(j).assign(&demean_panel(col, a, n, t));
    }
    Ok(out)
}

/// W applied per period to every column of a canonical-layout matrix.
fn lag_columns(
    x: &Array2<f64>,
    n: usize,
    t: usize,
    listw: &SpatialWeights,
) -> EconResult<Array2<f64>> {
    let mut out = try_zeros2(
        x.nrows(),
        x.ncols(),
        "spatially lagged spatial panel design",
    )?;
    for (j, col) in x.columns().into_iter().enumerate() {
        out.column_mut(j)
            .assign(&spatial_lag_panel(&col.to_owned(), n, t, listw));
    }
    Ok(out)
}

/// T >= 2 for entity or two-way demeaning and random effects, N >= 2 for period or two-way
/// demeaning.
fn check_panel_size(a: Absorbed, random: bool, n: usize, t: usize) -> EconResult<()> {
    if (random || matches!(a, Absorbed::Individual | Absorbed::TwoWays)) && t < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: t,
            context: "time periods: within-entity demeaning and random effects need T >= 2".into(),
        });
    }
    if matches!(a, Absorbed::Time | Absorbed::TwoWays) && n < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n,
            context: "entities: demeaning by period needs N >= 2".into(),
        });
    }
    Ok(())
}

/// n - absorbed - p, refused unless positive.
fn residual_df(n: usize, absorbed: usize, p: usize) -> EconResult<usize> {
    let used = absorbed.saturating_add(p);
    match n.checked_sub(used) {
        Some(df) if df > 0 => Ok(df),
        _ => Err(EconError::InsufficientData {
            required: used.saturating_add(1),
            provided: n,
            context: format!(
                "observations: the residual degrees of freedom of the spatial panel model \
                 would be {}",
                n as i128 - used as i128
            ),
        }),
    }
}

/// The relative residual sum of squares below which a least-squares fit counts as exact.
const EXACT_FIT_RSS: f64 = 64.0 * f64::EPSILON * 64.0 * f64::EPSILON;

/// Least squares of y on x before any spatial work: an aliased column is
/// `PerfectMulticollinearity` (when `refuse_aliased`), and an exact fit (residual sum of
/// squares at most (64 eps)^2 y'y, or y'y = 0) leaves the Gaussian likelihood unbounded.
fn require_non_degenerate_fit(
    y: &Array1<f64>,
    x: &Array2<f64>,
    names: &[String],
    y_col: &str,
    what: &str,
    refuse_aliased: bool,
) -> EconResult<()> {
    let yy = y.dot(y);
    let rss = if x.ncols() == 0 {
        yy
    } else {
        let ls = least_squares(x.view(), y.view())?;
        if refuse_aliased && !ls.aliased.is_empty() {
            return Err(EconError::PerfectMulticollinearity {
                columns: ls.aliased.iter().map(|&j| names[j].clone()).collect(),
            });
        }
        ls.residuals.dot(&ls.residuals)
    };
    if yy == 0.0 || rss <= EXACT_FIT_RSS * yy {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the regressors fit {y_col} exactly after the {what} (residual sum of squares \
                 {rss:.1e} against {yy:.1e}); the Gaussian likelihood is unbounded, so the \
                 spatial panel model cannot be estimated"
            ),
        });
    }
    Ok(())
}

/// The eigenvalues of W, kept complex. kNN and other asymmetric weights have complex
/// eigenvalues, and log|I - r W| = Re sum_i ln(1 - r w_i) = 1/2 sum_i ln((1 - r a_i)^2 +
/// (r b_i)^2) needs them (R: `Re(sum(log(1 - r * eigen(W)$values)))`). rho and lambda are
/// searched on (1 / min Re w + 1e-6, 1 / max Re w - 1e-6), as the R reference.
// T3-WP34: build from WeightsSpectrum once WP32 is merged (the weights' cached complex
// spectrum), instead of a fresh dense eigen decomposition per call.
struct PanelSpectrum {
    re: Vec<f64>,
    im: Vec<f64>,
    lower: f64,
    upper: f64,
}

impl PanelSpectrum {
    fn new(listw: &SpatialWeights) -> EconResult<Self> {
        checkpoint()?;
        let w = listw.try_to_dense()?;
        let evs = ndarray_to_faer(&w.view()).eigenvalues().map_err(|_| {
            EconError::Computation(
                "the eigenvalues of the spatial weights matrix could not be computed".into(),
            )
        })?;
        let (re, im): (Vec<f64>, Vec<f64>) = evs.iter().map(|z| (z.re, z.im)).unzip();
        if re.iter().chain(&im).any(|v| !v.is_finite()) {
            return Err(EconError::Computation(
                "the spatial weights matrix has non-finite eigenvalues".into(),
            ));
        }
        let min_re = re.iter().copied().fold(f64::INFINITY, f64::min);
        let max_re = re.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lower = if min_re < 0.0 {
            1.0 / min_re + 1e-6
        } else {
            -1.0
        };
        let upper = if max_re > 0.0 {
            1.0 / max_re - 1e-6
        } else {
            1.0
        };
        Ok(Self {
            re,
            im,
            lower,
            upper,
        })
    }

    /// log|I - r W|.
    fn log_det(&self, r: f64) -> f64 {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(&a, &b)| 0.5 * ((1.0 - r * a).powi(2) + (r * b).powi(2)).ln())
            .sum()
    }

    /// d/dr log|I - r W| = -Re sum w / (1 - r w)
    /// = -sum (a - r (a^2 + b^2)) / ((1 - r a)^2 + (r b)^2).
    fn d_log_det(&self, r: f64) -> f64 {
        -self
            .re
            .iter()
            .zip(&self.im)
            .map(|(&a, &b)| (a - r * (a * a + b * b)) / ((1.0 - r * a).powi(2) + (r * b).powi(2)))
            .sum::<f64>()
    }
}

// ============================================================================
// Maximum likelihood machinery (shared by the within, pooled and random models)
// ============================================================================

/// Value, gradient and parameter vector of a concentrated log-likelihood at psi;
/// `coef_scale[j]` = sqrt(s2 [(X*'X*)^-1]_jj) of the final (G)LS (the step floor of the
/// information below).
struct MlProfile {
    ll: f64,
    grad: Vec<f64>,
    theta: Vec<f64>,
    coef_scale: Vec<f64>,
}

const NEWTON_CRITERION: &str = "max |change| / max(1, |psi|) of rho, log phi and lambda";
/// Step of the central differences of the profile gradient.
const PROFILE_STEP: f64 = 1e-5;

/// Central-difference Hessian of the profile gradient at psi (one-sided towards the
/// interior when psi_j +- h leaves [lower_j, upper_j]), symmetrised.
fn profile_hessian<F>(
    profile: &F,
    psi: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> EconResult<Array2<f64>>
where
    F: Fn(&[f64]) -> EconResult<MlProfile>,
{
    let p = psi.len();
    let mut h = Array2::<f64>::zeros((p, p));
    let at = |v: f64, j: usize| -> EconResult<Vec<f64>> {
        let mut q = psi.to_vec();
        q[j] = v;
        Ok(profile(&q)?.grad)
    };
    let center = profile(psi)?.grad;
    for j in 0..p {
        let (up, down) = (psi[j] + PROFILE_STEP, psi[j] - PROFILE_STEP);
        let (gp, gm, width) = if up > upper[j] {
            (center.clone(), at(down, j)?, PROFILE_STEP)
        } else if down < lower[j] {
            (at(up, j)?, center.clone(), PROFILE_STEP)
        } else {
            (at(up, j)?, at(down, j)?, 2.0 * PROFILE_STEP)
        };
        for i in 0..p {
            h[[i, j]] = (gp[i] - gm[i]) / width;
        }
    }
    Ok((&h + &h.t()) * 0.5)
}

/// -H^-1 g when H is negative definite, else g_j / max(|H_jj|, 1e-8).
fn newton_direction(h: &Array2<f64>, g: &[f64]) -> Vec<f64> {
    let p = g.len();
    let gradient_step =
        || -> Vec<f64> { (0..p).map(|j| g[j] / h[[j, j]].abs().max(1e-8)).collect() };
    if h.iter().any(|v| !v.is_finite()) {
        return gradient_step();
    }
    match eig_symmetric(&h.view()) {
        Ok((vals, vecs)) if vals.iter().all(|&l| l < 0.0) => {
            // delta = V diag(-1/l) V' g
            let vg = vecs.t().dot(&Array1::from(g.to_vec()));
            let scaled = Array1::from_shape_fn(p, |i| -vg[i] / vals[i]);
            vecs.dot(&scaled).to_vec()
        }
        _ => gradient_step(),
    }
}

/// Damped Newton ascent on a concentrated log-likelihood over psi (rho, log phi, lambda:
/// dimensionless), clamped to [lower, upper]. Hessian: [`profile_hessian`]. Step halving
/// (at most 30) until ll(new) >= ll(old) - 1e-12 |ll(old)|; a failed line search stops the
/// loop (converged only when the Newton step itself is below `tol`). Converged when
/// max_j |change_j| / max(1, |psi_j|) <= tol.
fn newton_profile<F>(
    profile: F,
    psi0: &[f64],
    lower: &[f64],
    upper: &[f64],
    tol: f64,
    max_iter: usize,
) -> EconResult<(Vec<f64>, MlProfile, Convergence)>
where
    F: Fn(&[f64]) -> EconResult<MlProfile>,
{
    let p = psi0.len();
    let clamp = |v: &mut [f64]| {
        for j in 0..p {
            v[j] = v[j].clamp(lower[j], upper[j]);
        }
    };
    let rel = |new: &[f64], old: &[f64]| -> f64 {
        (0..p)
            .map(|j| (new[j] - old[j]).abs() / old[j].abs().max(1.0))
            .fold(0.0, f64::max)
    };
    let mut psi = psi0.to_vec();
    clamp(&mut psi);
    let mut cur = profile(&psi)?;
    if !cur.ll.is_finite() || cur.grad.iter().any(|g| !g.is_finite()) {
        return Err(EconError::Computation(
            "the spatial panel log-likelihood is not finite at rho = lambda = 0, phi = 1".into(),
        ));
    }
    let (mut converged, mut iterations, mut change) = (false, 0usize, f64::NAN);
    while iterations < max_iter {
        checkpoint()?;
        iterations += 1;
        let h = profile_hessian(&profile, &psi, lower, upper)?;
        let delta = newton_direction(&h, &cur.grad);
        let step_size = (0..p)
            .map(|j| delta[j].abs() / psi[j].abs().max(1.0))
            .fold(0.0, f64::max);
        let mut accepted = None;
        let mut t = 1.0;
        for _ in 0..=30 {
            let mut cand: Vec<f64> = (0..p).map(|j| psi[j] + t * delta[j]).collect();
            clamp(&mut cand);
            let prof = profile(&cand)?;
            if prof.ll.is_finite()
                && prof.grad.iter().all(|g| g.is_finite())
                && prof.ll >= cur.ll - 1e-12 * cur.ll.abs()
            {
                accepted = Some((cand, prof));
                break;
            }
            t *= 0.5;
        }
        match accepted {
            None => {
                change = step_size;
                converged = step_size <= tol;
                break;
            }
            Some((cand, prof)) => {
                change = rel(&cand, &psi);
                psi = cand;
                cur = prof;
                if change <= tol {
                    converged = true;
                    break;
                }
            }
        }
    }
    let conv = Convergence::new(
        converged,
        iterations,
        max_iter,
        NEWTON_CRITERION,
        change,
        tol,
    );
    Ok((psi, cur, conv))
}

/// Kind of each entry of theta, for the information steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    Coef(usize),
    Spatial,
    Scale,
}

/// Steps h_j of the information: Coef(j) 1e-5 max(|theta_j|, coef_scale_j) (1e-5 if both
/// are 0); Spatial (rho, lambda) 1e-5, made inactive with a warning when theta_j +- 1e-5
/// leaves `interval`; Scale (phi, sigma2) 1e-5 theta_j (a phi of 0 is made inactive by the
/// caller). Returns (steps, active).
fn information_steps(
    prof: &MlProfile,
    kinds: &[ParamKind],
    interval: Option<(f64, f64)>,
    names: &[String],
    warnings: &mut Vec<String>,
) -> (Vec<f64>, Vec<bool>) {
    let mut steps = Vec::with_capacity(kinds.len());
    let mut active = Vec::with_capacity(kinds.len());
    for (j, kind) in kinds.iter().enumerate() {
        let v = prof.theta[j];
        match *kind {
            ParamKind::Coef(c) => {
                let base = v.abs().max(prof.coef_scale[c]);
                steps.push(if base > 0.0 { 1e-5 * base } else { 1e-5 });
                active.push(true);
            }
            ParamKind::Spatial => {
                let h = 1e-5;
                let inside = interval.is_none_or(|(lo, hi)| v - h > lo && v + h < hi);
                if !inside {
                    warnings.push(format!(
                        "{} is within 1e-5 of the boundary of its interval; its standard error \
                         is not reported",
                        names[j]
                    ));
                }
                steps.push(h);
                active.push(inside);
            }
            ParamKind::Scale => {
                steps.push(1e-5 * v.abs());
                active.push(v > 0.0);
            }
        }
    }
    (steps, active)
}

/// Symmetrised central-difference Jacobian of an analytic score at theta over the `active`
/// parameters (inactive rows and columns stay 0).
fn fd_hessian<F>(score: F, theta: &[f64], steps: &[f64], active: &[bool]) -> EconResult<Array2<f64>>
where
    F: Fn(&[f64]) -> EconResult<Array1<f64>>,
{
    let p = theta.len();
    let mut h = Array2::<f64>::zeros((p, p));
    for j in (0..p).filter(|&j| active[j]) {
        checkpoint()?;
        let mut up = theta.to_vec();
        let mut down = theta.to_vec();
        up[j] += steps[j];
        down[j] -= steps[j];
        let (sp, sm) = (score(&up)?, score(&down)?);
        for i in (0..p).filter(|&i| active[i]) {
            h[[i, j]] = (sp[i] - sm[i]) / (2.0 * steps[j]);
        }
    }
    Ok((&h + &h.t()) * 0.5)
}

/// SEs = sqrt(diag((-H_active)^-1)); NaN for inactive parameters. A non-positive eigenvalue
/// of -H_active gives NaN for every parameter and a warning.
fn ml_standard_errors(
    h: &Array2<f64>,
    active: &[bool],
    warnings: &mut Vec<String>,
) -> EconResult<Vec<f64>> {
    let idx: Vec<usize> = (0..active.len()).filter(|&j| active[j]).collect();
    let mut se = vec![f64::NAN; active.len()];
    if idx.is_empty() {
        return Ok(se);
    }
    let info = Array2::from_shape_fn((idx.len(), idx.len()), |(a, b)| -h[[idx[a], idx[b]]]);
    let smallest = if info.iter().all(|v| v.is_finite()) {
        eig_symmetric(&info.view())
            .map(|(vals, _)| vals.iter().copied().fold(f64::INFINITY, f64::min))
            .unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    if smallest.is_nan() || smallest <= 0.0 {
        warnings.push(format!(
            "the observed information is not positive definite at the estimate (smallest \
             eigenvalue {smallest:.3e}); standard errors are not reported"
        ));
        return Ok(se);
    }
    let inv = sym_inverse(
        info.view(),
        SingularPolicy::Error,
        "spml observed information",
    )?;
    for (a, &j) in idx.iter().enumerate() {
        se[j] = inv.inverse[[a, a]].sqrt();
    }
    Ok(se)
}

/// Estimates of one spml fit, for [`assemble_spml`].
struct SpmlFit {
    theta: Vec<f64>,
    se: Vec<f64>,
    k: usize,
    lag: bool,
    phi: Option<f64>,
    err: bool,
    ll: f64,
    df: usize,
    coef_names: Vec<String>,
    convergence: Option<Convergence>,
    warnings: Vec<String>,
    residuals: Array1<f64>,
    fitted: Array1<f64>,
}

/// z and p of an estimate, NaN unless its SE is finite and positive.
fn wald(v: f64, se: f64) -> (f64, f64) {
    if se.is_finite() && se > 0.0 {
        let z = v / se;
        (z, normal_p_value_two_sided(z))
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// theta = [beta (k), rho?, phi?, lambda?, sigma2] into an `SpmlResult`.
fn assemble_spml(
    fit: SpmlFit,
    config: &SpmlConfig,
    n_entities: usize,
    n_time: usize,
) -> SpmlResult {
    let SpmlFit {
        theta,
        se,
        k,
        lag,
        phi,
        err,
        ll,
        df,
        coef_names,
        convergence,
        warnings,
        residuals,
        fitted,
    } = fit;
    let n = residuals.len();
    let coefficients = theta[..k].to_vec();
    let std_errors = se[..k].to_vec();
    let (z_values, p_values): (Vec<f64>, Vec<f64>) =
        (0..k).map(|j| wald(coefficients[j], std_errors[j])).unzip();
    let significance = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();
    let mut names = coef_names.clone();
    let mut i = k;
    let (mut rho, mut rho_se, mut rho_z, mut rho_p) = (None, None, None, None);
    if lag {
        let (z, p) = wald(theta[i], se[i]);
        (rho, rho_se, rho_z, rho_p) = (Some(theta[i]), Some(se[i]), Some(z), Some(p));
        names.push("rho".into());
        i += 1;
    }
    if phi.is_some() {
        names.push("phi".into());
        i += 1;
    }
    let (mut lambda, mut lambda_se, mut lambda_z, mut lambda_p) = (None, None, None, None);
    if err {
        let (z, p) = wald(theta[i], se[i]);
        (lambda, lambda_se, lambda_z, lambda_p) = (Some(theta[i]), Some(se[i]), Some(z), Some(p));
        names.push("lambda".into());
    }
    names.push("sigma2".into());
    let sigma2 = theta[theta.len() - 1];
    let n_par = theta.len() as f64;
    SpmlResult {
        model: config.model,
        effect: config.effect,
        has_lag: lag,
        spatial_error: config.spatial_error,
        coefficients,
        coef_names,
        std_errors,
        z_values,
        p_values,
        significance,
        rho,
        rho_se,
        rho_z,
        rho_p,
        lambda,
        lambda_se,
        lambda_z,
        lambda_p,
        variance: SpatialPanelVariance {
            sigma_mu: phi.map(|ph| (ph * sigma2).sqrt()),
            sigma_nu: None,
            sigma_epsilon: sigma2.sqrt(),
            lambda,
            rho_sr: None,
            phi,
        },
        log_likelihood: ll,
        aic: -2.0 * ll + 2.0 * n_par,
        bic: -2.0 * ll + n_par * (n as f64).ln(),
        n_obs: n,
        n_entities,
        n_time,
        df,
        residuals,
        fitted,
        parameter_names: names,
        parameter_se: se,
        convergence,
        warnings,
    }
}

// ============================================================================
// Within and pooled ML
// ============================================================================

/// The lag-only profile in closed form: with one QR of X, beta(rho) = b0 - rho b1 and
/// e(rho) = e0 - rho e1 (e0, e1 the residuals of y and W y), so a profile evaluation costs
/// the log-determinant only.
struct LagOnly {
    b0: Array1<f64>,
    b1: Array1<f64>,
    e0e0: f64,
    e0e1: f64,
    e1e1: f64,
    unscaled_diag: Array1<f64>,
}

/// Transformed (demeaned or pooled) data and the spatial lags the likelihood needs
/// (canonical layout).
struct FeData {
    y: Array1<f64>,
    x: Array2<f64>,
    wy: Array1<f64>,
    wwy: Array1<f64>,
    wx: Array2<f64>,
    n_time: usize,
    lag_only: Option<LagOnly>,
}

/// psi = (rho?, lambda?): beta = least squares of B A y on B X; s2 = e'e / n;
/// ll = -n/2 (ln(2 pi s2) + 1) + T (log|A| + log|B|); grad_rho = T dlog|A| + e'(B W y)/s2;
/// grad_lambda = T dlog|B| + e'(W r)/s2 with r = A y - X beta (envelope theorem).
fn fe_profile(
    d: &FeData,
    sp: Option<&PanelSpectrum>,
    lag: bool,
    err: bool,
    psi: &[f64],
    names: &[String],
) -> EconResult<MlProfile> {
    let rho = if lag { psi[0] } else { 0.0 };
    let lam = if err { psi[psi.len() - 1] } else { 0.0 };
    let (n, t) = (d.y.len() as f64, d.n_time as f64);
    let ld = |r: f64| sp.map_or(0.0, |s| s.log_det(r));
    if let (Some(lo), false) = (&d.lag_only, err) {
        let ee = lo.e0e0 - 2.0 * rho * lo.e0e1 + rho * rho * lo.e1e1;
        let s2 = ee / n;
        let ll = -0.5 * n * ((2.0 * PI * s2).ln() + 1.0) + t * ld(rho);
        let mut grad = Vec::with_capacity(1);
        if let (Some(s), true) = (sp, lag) {
            // e'(W y) = e'e1 because e is orthogonal to X
            grad.push(t * s.d_log_det(rho) + (lo.e0e1 - rho * lo.e1e1) / s2);
        }
        let mut theta = (&lo.b0 - &(&lo.b1 * rho)).to_vec();
        if lag {
            theta.push(rho);
        }
        theta.push(s2);
        let coef_scale = lo.unscaled_diag.iter().map(|v| (s2 * v).sqrt()).collect();
        return Ok(MlProfile {
            ll,
            grad,
            theta,
            coef_scale,
        });
    }
    let ay = &d.y - &(&d.wy * rho);
    let way = &d.wy - &(&d.wwy * rho);
    let by = &ay - &(&way * lam);
    let bx = &d.x - &(&d.wx * lam);
    let ls = least_squares(bx.view(), by.view())?;
    if !ls.aliased.is_empty() {
        return Err(EconError::PerfectMulticollinearity {
            columns: ls.aliased.iter().map(|&j| names[j].clone()).collect(),
        });
    }
    let e = &ls.residuals;
    let s2 = e.dot(e) / n;
    let ll = -0.5 * n * ((2.0 * PI * s2).ln() + 1.0) + t * (ld(rho) + ld(lam));
    let mut grad = Vec::with_capacity(2);
    if let Some(s) = sp {
        if lag {
            grad.push(t * s.d_log_det(rho) + e.dot(&(&d.wy - &(&d.wwy * lam))) / s2);
        }
        if err {
            grad.push(t * s.d_log_det(lam) + e.dot(&(&way - &d.wx.dot(&ls.coefficients))) / s2);
        }
    }
    let mut theta = ls.coefficients.to_vec();
    if lag {
        theta.push(rho);
    }
    if err {
        theta.push(lam);
    }
    theta.push(s2);
    let coef_scale = ls
        .unscaled_cov
        .diag()
        .iter()
        .map(|v| (s2 * v).sqrt())
        .collect();
    Ok(MlProfile {
        ll,
        grad,
        theta,
        coef_scale,
    })
}

/// Full analytic score at theta = (beta, rho?, lambda?, s2), r = y - rho W y - X beta,
/// e = r - lambda W r: d/dbeta = (X - lambda W X)'e / s2; d/drho = T dlog|A| +
/// e'(W y - lambda W W y)/s2; d/dlambda = T dlog|B| + e'(W r)/s2;
/// d/ds2 = -n/(2 s2) + e'e/(2 s2^2).
fn fe_score(
    d: &FeData,
    sp: Option<&PanelSpectrum>,
    lag: bool,
    err: bool,
    theta: &[f64],
) -> Array1<f64> {
    let k = d.x.ncols();
    let beta = ArrayView1::from(&theta[..k]);
    let rho = if lag { theta[k] } else { 0.0 };
    let lam = if err {
        theta[k + usize::from(lag)]
    } else {
        0.0
    };
    let s2 = theta[theta.len() - 1];
    let (n, t) = (d.y.len() as f64, d.n_time as f64);
    let r = &d.y - &(&d.wy * rho) - d.x.dot(&beta);
    let wr = &d.wy - &(&d.wwy * rho) - d.wx.dot(&beta);
    let e = &r - &(&wr * lam);
    let bx = &d.x - &(&d.wx * lam);
    let mut g: Vec<f64> = bx.t().dot(&e).iter().map(|v| v / s2).collect();
    if let Some(s) = sp {
        if lag {
            g.push(t * s.d_log_det(rho) + e.dot(&(&d.wy - &(&d.wwy * lam))) / s2);
        }
        if err {
            g.push(t * s.d_log_det(lam) + e.dot(&wr) / s2);
        }
    }
    g.push(-n / (2.0 * s2) + e.dot(&e) / (2.0 * s2 * s2));
    Array1::from(g)
}

/// Within (entity, time or two-way demeaning) and pooled lag, error and SARAR models by ML.
fn fit_spml_fe(
    panel: &SpatialPanelData,
    listw: &SpatialWeights,
    config: &SpmlConfig,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<SpmlResult> {
    let (n_ent, n_time) = (panel.n_entities, panel.n_time);
    let n = panel.y.len();
    let a = Absorbed::of(config);
    check_panel_size(a, false, n_ent, n_time)?;
    let lag = config.lag;
    let err = config.spatial_error != SpatialErrorType::None;
    let (x, coef_names): (Array2<f64>, Vec<String>) = if a == Absorbed::None {
        let mut names = vec!["(Intercept)".to_string()];
        names.extend(x_cols.iter().map(|s| s.to_string()));
        (panel.x.clone(), names)
    } else {
        (
            demean_panel_matrix(panel.x.slice(s![.., 1..]), a, n_ent, n_time)?,
            x_cols.iter().map(|s| s.to_string()).collect(),
        )
    };
    let k = x.ncols();
    if k == 0 {
        return Err(EconError::InvalidSpecification {
            message: "the within spatial panel model needs at least one regressor".into(),
        });
    }
    let df = residual_df(
        n,
        a.count(n_ent, n_time),
        k + usize::from(lag) + usize::from(err),
    )?;
    let y = demean_panel(panel.y.view(), a, n_ent, n_time);
    require_non_degenerate_fit(&y, &x, &coef_names, y_col, a.describe(), true)?;

    let sp = if lag || err {
        Some(PanelSpectrum::new(listw)?)
    } else {
        None
    };
    let wy = spatial_lag_panel(&y, n_ent, n_time, listw);
    let wwy = spatial_lag_panel(&wy, n_ent, n_time, listw);
    let wx = lag_columns(&x, n_ent, n_time, listw)?;
    let lag_only = if err {
        None
    } else {
        let f0 = least_squares(x.view(), y.view())?;
        let f1 = least_squares(x.view(), wy.view())?;
        Some(LagOnly {
            e0e0: f0.residuals.dot(&f0.residuals),
            e0e1: f0.residuals.dot(&f1.residuals),
            e1e1: f1.residuals.dot(&f1.residuals),
            unscaled_diag: f0.unscaled_cov.diag().to_owned(),
            b0: f0.coefficients,
            b1: f1.coefficients,
        })
    };
    let data = FeData {
        y,
        x,
        wy,
        wwy,
        wx,
        n_time,
        lag_only,
    };

    let p = usize::from(lag) + usize::from(err);
    let mut warnings = Vec::new();
    let (prof, convergence) = match sp.as_ref() {
        Some(s) if p > 0 => {
            let (_, prof, conv) = newton_profile(
                |psi| fe_profile(&data, Some(s), lag, err, psi, &coef_names),
                &vec![0.0; p],
                &vec![s.lower; p],
                &vec![s.upper; p],
                config.tol,
                config.max_iter,
            )?;
            if let Some(note) = conv.note("Spatial panel ML (spml)") {
                warnings.push(note);
            }
            (prof, Some(conv))
        }
        _ => (
            fe_profile(&data, None, false, false, &[], &coef_names)?,
            None,
        ),
    };

    let mut kinds: Vec<ParamKind> = (0..k).map(ParamKind::Coef).collect();
    let mut names = coef_names.clone();
    if lag {
        kinds.push(ParamKind::Spatial);
        names.push("rho".into());
    }
    if err {
        kinds.push(ParamKind::Spatial);
        names.push("lambda".into());
    }
    kinds.push(ParamKind::Scale);
    names.push("sigma2".into());
    let interval = sp.as_ref().map(|s| (s.lower, s.upper));
    let (steps, active) = information_steps(&prof, &kinds, interval, &names, &mut warnings);
    let hess = fd_hessian(
        |th| Ok(fe_score(&data, sp.as_ref(), lag, err, th)),
        &prof.theta,
        &steps,
        &active,
    )?;
    let se = ml_standard_errors(&hess, &active, &mut warnings)?;

    let theta = prof.theta;
    let beta = ArrayView1::from(&theta[..k]);
    let rho = if lag { theta[k] } else { 0.0 };
    let residuals = &data.y - &(&data.wy * rho) - data.x.dot(&beta);
    let fitted = &data.y - &residuals;
    Ok(assemble_spml(
        SpmlFit {
            theta,
            se,
            k,
            lag,
            phi: None,
            err,
            ll: prof.ll,
            df,
            coef_names,
            convergence,
            warnings,
            residuals,
            fitted,
        },
        config,
        n_ent,
        n_time,
    ))
}

// ============================================================================
// Random effects ML
// ============================================================================

/// K = B'B = V diag(d) V' at one lambda (B = I - lambda W), and k'_i = v_i' K' v_i with
/// K' = dK/dlambda = -(W + W') + 2 lambda W'W.
struct BaltagiEigen {
    v: Array2<f64>,
    d: Array1<f64>,
    kp: Array1<f64>,
}

/// Number of decompositions of K the Baltagi path keeps (by the bits of lambda): every
/// evaluation that moves only rho, log phi, beta or sigma2 reuses one.
const BALTAGI_CACHE: usize = 3;

/// The random-effects data: y (raw), x = [1, X], wy = W y per period, and what the
/// spatial error needs.
struct ReData<'a> {
    y: Array1<f64>,
    x: Array2<f64>,
    wy: Array1<f64>,
    n: usize,
    t: usize,
    lag: bool,
    err: SpatialErrorType,
    sp: Option<PanelSpectrum>,
    listw: &'a SpatialWeights,
    cache: RefCell<Vec<(u64, Rc<BaltagiEigen>)>>,
}

impl ReData<'_> {
    /// B v = v - lambda W v (an N-vector).
    fn b(&self, lambda: f64, v: &Array1<f64>) -> Array1<f64> {
        v - &(self.listw.lag(v) * lambda)
    }

    /// K v = B'B v.
    fn k(&self, lambda: f64, v: &Array1<f64>) -> Array1<f64> {
        let bv = self.b(lambda, v);
        &bv - &(self.listw.lag_transpose(&bv) * lambda)
    }

    /// v'K'v = -2 v'W v + 2 lambda |W v|^2.
    fn kq(&self, lambda: f64, v: &Array1<f64>) -> f64 {
        let wv = self.listw.lag(v);
        -2.0 * v.dot(&wv) + 2.0 * lambda * wv.dot(&wv)
    }

    /// The eigendecomposition of K = B'B at lambda (cached), or `None` when K is not
    /// positive definite. K is formed from the sparse W (O(N k^2)) and K'V as a sparse
    /// times dense product.
    fn baltagi_eigen(&self, lambda: f64) -> EconResult<Option<Rc<BaltagiEigen>>> {
        let key = lambda.to_bits();
        if let Some((_, e)) = self.cache.borrow().iter().find(|(k, _)| *k == key) {
            return Ok(Some(Rc::clone(e)));
        }
        checkpoint()?;
        let n = self.n;
        let mut kmat = try_zeros2(n, n, "spatial panel K = B'B")?;
        for i in 0..n {
            kmat[[i, i]] = 1.0;
        }
        for r in 0..n {
            let sw = self.listw.get_weights(r);
            for (&j, &w) in sw.indices.iter().zip(&sw.weights) {
                kmat[[r, j]] -= lambda * w;
                kmat[[j, r]] -= lambda * w;
                for (&l, &w2) in sw.indices.iter().zip(&sw.weights) {
                    kmat[[j, l]] += lambda * lambda * w * w2;
                }
            }
        }
        let (d, v) = eig_symmetric(&kmat.view())?;
        drop(kmat);
        if d.iter().any(|&x| !(x.is_finite() && x > 0.0)) {
            return Ok(None);
        }
        let mut kp = Array1::<f64>::zeros(n);
        for i in 0..n {
            kp[i] = self.kq(lambda, &v.column(i).to_owned());
        }
        let e = Rc::new(BaltagiEigen { v, d, kp });
        let mut cache = self.cache.borrow_mut();
        if cache.len() >= BALTAGI_CACHE {
            cache.remove(0);
        }
        cache.push((key, Rc::clone(&e)));
        Ok(Some(e))
    }
}

/// Omega~^-1 = Jbar_T (x) M + E_T (x) K (Omega = s2 Omega~) at one (phi, lambda), c = T phi:
///   Baltagi: K = V diag(d) V', M = V diag(m) V', m = d / (c d + 1),
///            ldet = sum ln(c d + 1) - T sum ln d
///   KKP:     M = K / (c + 1), ldet = N ln(c + 1) - 2 T log|B|
///   none:    K = I, M = I / (c + 1), ldet = N ln(c + 1)
/// Square root S = Jbar_T (x) S1 + E_T (x) B with S1'S1 = M: M^1/2 (Baltagi), B / sqrt(c + 1)
/// (KKP), I / sqrt(c + 1) (none); the KKP and no-error kernels need no N x N work.
struct ReKernel {
    c: f64,
    lambda: f64,
    eig: Option<Rc<BaltagiEigen>>,
    m: Option<Array1<f64>>,
    ldet: f64,
}

impl ReKernel {
    /// `None` when K is not positive definite.
    fn new(d: &ReData, phi: f64, lambda: f64) -> EconResult<Option<Self>> {
        let c = d.t as f64 * phi;
        let nf = d.n as f64;
        let kernel = match d.err {
            SpatialErrorType::None => ReKernel {
                c,
                lambda: 0.0,
                eig: None,
                m: None,
                ldet: nf * (c + 1.0).ln(),
            },
            SpatialErrorType::KKP => {
                let ld = d.sp.as_ref().map_or(0.0, |s| s.log_det(lambda));
                if !ld.is_finite() {
                    return Ok(None);
                }
                ReKernel {
                    c,
                    lambda,
                    eig: None,
                    m: None,
                    ldet: nf * (c + 1.0).ln() - 2.0 * d.t as f64 * ld,
                }
            }
            SpatialErrorType::Baltagi => {
                let Some(e) = d.baltagi_eigen(lambda)? else {
                    return Ok(None);
                };
                let m = e.d.mapv(|x| x / (c * x + 1.0));
                let ldet = e.d.iter().map(|&x| (c * x + 1.0).ln()).sum::<f64>()
                    - d.t as f64 * e.d.iter().map(|x| x.ln()).sum::<f64>();
                ReKernel {
                    c,
                    lambda,
                    eig: Some(e),
                    m: Some(m),
                    ldet,
                }
            }
        };
        Ok(Some(kernel))
    }

    /// V diag(f) V' v.
    fn spectral(e: &BaltagiEigen, f: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let a = e.v.t().dot(v) * f;
        e.v.dot(&a)
    }

    /// S2 v: B v, or v without spatial error.
    fn s2(&self, d: &ReData, v: &Array1<f64>) -> Array1<f64> {
        match d.err {
            SpatialErrorType::None => v.clone(),
            _ => d.b(self.lambda, v),
        }
    }

    /// S1 v with S1'S1 = M.
    fn s1(&self, d: &ReData, v: &Array1<f64>) -> Array1<f64> {
        let scale = 1.0 / (self.c + 1.0).sqrt();
        match (&self.eig, &self.m) {
            (Some(e), Some(m)) => Self::spectral(e, &m.mapv(f64::sqrt), v),
            _ => self.s2(d, v) * scale,
        }
    }

    /// M v.
    fn apply_m(&self, d: &ReData, v: &Array1<f64>) -> Array1<f64> {
        match (&self.eig, &self.m, d.err) {
            (Some(e), Some(m), _) => Self::spectral(e, m, v),
            (_, _, SpatialErrorType::None) => v / (self.c + 1.0),
            _ => d.k(self.lambda, v) / (self.c + 1.0),
        }
    }

    /// K v.
    fn apply_k(&self, d: &ReData, v: &Array1<f64>) -> Array1<f64> {
        match d.err {
            SpatialErrorType::None => v.clone(),
            _ => d.k(self.lambda, v),
        }
    }
}

/// (vbar, blocks v_t - vbar) of a canonical-layout vector.
fn re_split(v: ArrayView1<f64>, n: usize, t: usize) -> (Array1<f64>, Vec<Array1<f64>>) {
    let vbar = unit_means(v, n, t);
    let blocks = (0..t)
        .map(|p| &v.slice(s![p * n..(p + 1) * n]) - &vbar)
        .collect();
    (vbar, blocks)
}

/// u'Omega~^-1 u = T |S1 ubar|^2 + sum_t |S2 (u_t - ubar)|^2.
fn re_quad(k: &ReKernel, d: &ReData, u: ArrayView1<f64>) -> f64 {
    let (ubar, blocks) = re_split(u, d.n, d.t);
    let s1 = k.s1(d, &ubar);
    d.t as f64 * s1.dot(&s1)
        + blocks
            .iter()
            .map(|e| {
                let s = k.s2(d, e);
                s.dot(&s)
            })
            .sum::<f64>()
}

/// Omega~^-1/2 v: block t = S2 (v_t - vbar) + S1 vbar.
fn re_whiten(k: &ReKernel, d: &ReData, v: ArrayView1<f64>) -> Array1<f64> {
    let (vbar, blocks) = re_split(v, d.n, d.t);
    let s1 = k.s1(d, &vbar);
    let mut out = Array1::<f64>::zeros(d.n * d.t);
    for (p, e) in blocks.iter().enumerate() {
        out.slice_mut(s![p * d.n..(p + 1) * d.n])
            .assign(&(&k.s2(d, e) + &s1));
    }
    out
}

/// Omega~^-1 v: block t = K (v_t - vbar) + M vbar.
fn re_omega_inv(k: &ReKernel, d: &ReData, v: ArrayView1<f64>) -> Array1<f64> {
    let (vbar, blocks) = re_split(v, d.n, d.t);
    let mv = k.apply_m(d, &vbar);
    let mut out = Array1::<f64>::zeros(d.n * d.t);
    for (p, e) in blocks.iter().enumerate() {
        out.slice_mut(s![p * d.n..(p + 1) * d.n])
            .assign(&(&k.apply_k(d, e) + &mv));
    }
    out
}

fn infeasible_kernel() -> EconError {
    EconError::Computation(
        "B'B is not positive definite at this lambda; the random-effects likelihood is not \
         defined there"
            .into(),
    )
}

/// Full analytic score at theta = (beta, rho?, phi, lambda?, s2); u = y - rho W y - X beta,
/// ubar its entity means, a = V'ubar, E_t = u_t - ubar:
///   d/dbeta   = X'Omega~^-1 u / s2
///   d/drho    = T dlog|A| + (W y)'Omega~^-1 u / s2
///   d/dphi    = T (-1/2 dldet/dc - dQ/dc / (2 s2)),
///               dldet/dc = sum d/(c d + 1) (b) | N/(c + 1) (kkp, none);
///               dQ/dc    = -T sum m^2 a^2 (b) | -T |B ubar|^2/(c + 1)^2 (kkp)
///                          | -T ubar'ubar/(c + 1)^2 (none)
///   d/dlambda = -1/2 dldet/dlambda - dQ/dlambda / (2 s2), k'_i = v_i'K'v_i:
///               dldet/dlambda = c sum k'_i/(c d_i + 1) - T sum k'_i/d_i (b)
///                             | -2 T dlog|B| (kkp);
///               dQ/dlambda = sum_t E_t'K'E_t + T (G ubar)'K'(G ubar),
///                            G = V diag(1/(c d + 1)) V' (b)
///                          | sum_t E_t'K'E_t + T ubar'K'ubar/(c + 1) (kkp)
///   d/ds2     = -n/(2 s2) + Q/(2 s2^2)
fn re_score(d: &ReData, theta: &[f64]) -> EconResult<Array1<f64>> {
    let k = d.x.ncols();
    let err = d.err != SpatialErrorType::None;
    let beta = ArrayView1::from(&theta[..k]);
    let rho = if d.lag { theta[k] } else { 0.0 };
    let i_phi = k + usize::from(d.lag);
    let phi = theta[i_phi];
    let lambda = if err { theta[i_phi + 1] } else { 0.0 };
    let s2 = theta[theta.len() - 1];
    let kern = ReKernel::new(d, phi, lambda)?.ok_or_else(infeasible_kernel)?;
    let (tf, nf, nobs) = (d.t as f64, d.n as f64, (d.n * d.t) as f64);
    let u = &d.y - &(&d.wy * rho) - d.x.dot(&beta);
    let oiu = re_omega_inv(&kern, d, u.view());
    let q = re_quad(&kern, d, u.view());
    let (ubar, blocks) = re_split(u.view(), d.n, d.t);
    let c = kern.c;

    let mut g: Vec<f64> = d.x.t().dot(&oiu).iter().map(|v| v / s2).collect();
    if d.lag {
        let dld = d.sp.as_ref().map_or(0.0, |s| s.d_log_det(rho));
        g.push(tf * dld + d.wy.dot(&oiu) / s2);
    }
    let (dldet_dc, dq_dc) = match (&kern.eig, &kern.m) {
        (Some(e), Some(m)) => {
            let a = e.v.t().dot(&ubar);
            (
                e.d.iter().map(|&x| x / (c * x + 1.0)).sum::<f64>(),
                -tf * m
                    .iter()
                    .zip(&a)
                    .map(|(mi, ai)| mi * mi * ai * ai)
                    .sum::<f64>(),
            )
        }
        _ => {
            let bu = kern.s2(d, &ubar);
            (nf / (c + 1.0), -tf * bu.dot(&bu) / (c + 1.0).powi(2))
        }
    };
    g.push(tf * (-0.5 * dldet_dc - dq_dc / (2.0 * s2)));
    if err {
        let within: f64 = blocks.iter().map(|e| d.kq(lambda, e)).sum();
        let (dldet, dq) = match &kern.eig {
            Some(e) => {
                let dldet = c * e
                    .kp
                    .iter()
                    .zip(&e.d)
                    .map(|(kp, x)| kp / (c * x + 1.0))
                    .sum::<f64>()
                    - tf * e.kp.iter().zip(&e.d).map(|(kp, x)| kp / x).sum::<f64>();
                let a = e.v.t().dot(&ubar);
                let gu = e.v.dot(&(&a / &e.d.mapv(|x| c * x + 1.0)));
                (dldet, within + tf * d.kq(lambda, &gu))
            }
            None => {
                let dld = d.sp.as_ref().map_or(0.0, |s| s.d_log_det(lambda));
                (
                    -2.0 * tf * dld,
                    within + tf * d.kq(lambda, &ubar) / (c + 1.0),
                )
            }
        };
        g.push(-0.5 * dldet - dq / (2.0 * s2));
    }
    g.push(-nobs / (2.0 * s2) + q / (2.0 * s2 * s2));
    Ok(Array1::from(g))
}

/// Profile at (rho, phi, lambda) in natural units: beta = least squares of Omega~^-1/2 X on
/// Omega~^-1/2 A y (aliased -> PerfectMulticollinearity), u = A y - X beta,
/// s2 = u'Omega~^-1 u / n, ll = -n/2 (ln(2 pi s2) + 1) + T log|A| - ldet/2; the gradient is
/// the score at theta (the log phi component times phi). An infeasible kernel gives
/// ll = -inf, which the line search rejects.
fn re_profile(
    d: &ReData,
    rho: f64,
    phi: f64,
    lambda: f64,
    names: &[String],
) -> EconResult<MlProfile> {
    let err = d.err != SpatialErrorType::None;
    let n_grad = usize::from(d.lag) + 1 + usize::from(err);
    let Some(kern) = ReKernel::new(d, phi, lambda)? else {
        return Ok(MlProfile {
            ll: f64::NEG_INFINITY,
            grad: vec![f64::NAN; n_grad],
            theta: vec![],
            coef_scale: vec![],
        });
    };
    let k = d.x.ncols();
    let nobs = (d.n * d.t) as f64;
    let ay = &d.y - &(&d.wy * rho);
    let ys = re_whiten(&kern, d, ay.view());
    let mut xs = try_zeros2(d.x.nrows(), k, "whitened spatial panel design")?;
    for (j, col) in d.x.columns().into_iter().enumerate() {
        xs.column_mut(j).assign(&re_whiten(&kern, d, col));
    }
    let ls = least_squares(xs.view(), ys.view())?;
    if !ls.aliased.is_empty() {
        return Err(EconError::PerfectMulticollinearity {
            columns: ls.aliased.iter().map(|&j| names[j].clone()).collect(),
        });
    }
    let u = &ay - &d.x.dot(&ls.coefficients);
    let s2 = re_quad(&kern, d, u.view()) / nobs;
    let ld_rho = d.sp.as_ref().map_or(0.0, |s| s.log_det(rho));
    let ll = -0.5 * nobs * ((2.0 * PI * s2).ln() + 1.0) + d.t as f64 * ld_rho - 0.5 * kern.ldet;
    let mut theta = ls.coefficients.to_vec();
    if d.lag {
        theta.push(rho);
    }
    theta.push(phi);
    if err {
        theta.push(lambda);
    }
    theta.push(s2);
    let score = re_score(d, &theta)?;
    let mut grad = Vec::with_capacity(n_grad);
    let i_phi = k + usize::from(d.lag);
    if d.lag {
        grad.push(score[k]);
    }
    grad.push(score[i_phi] * phi);
    if err {
        grad.push(score[i_phi + 1]);
    }
    let coef_scale = ls
        .unscaled_cov
        .diag()
        .iter()
        .map(|v| (s2 * v).sqrt())
        .collect();
    Ok(MlProfile {
        ll,
        grad,
        theta,
        coef_scale,
    })
}

/// Lower and upper bound of log phi.
const LOG_PHI_BOUND: f64 = 30.0;

/// Individual random effects by ML, with no spatial error, the Baltagi error (mu not
/// spatially correlated) or the KKP error (mu spatially correlated), each with or without
/// a spatial lag.
fn fit_spml_random(
    panel: &SpatialPanelData,
    listw: &SpatialWeights,
    config: &SpmlConfig,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<SpmlResult> {
    let (n_ent, n_time) = (panel.n_entities, panel.n_time);
    let n = panel.y.len();
    check_panel_size(Absorbed::Individual, true, n_ent, n_time)?;
    let lag = config.lag;
    let err = config.spatial_error != SpatialErrorType::None;
    let mut coef_names = vec!["(Intercept)".to_string()];
    coef_names.extend(x_cols.iter().map(|s| s.to_string()));
    let x = panel.x.clone();
    let k = x.ncols();
    let df = residual_df(n, 0, k + usize::from(lag) + usize::from(err))?;
    let full = least_squares(x.view(), panel.y.view())?;
    if !full.aliased.is_empty() {
        return Err(EconError::PerfectMulticollinearity {
            columns: full
                .aliased
                .iter()
                .map(|&j| coef_names[j].clone())
                .collect(),
        });
    }
    // With an exact within fit the likelihood is unbounded as phi grows.
    let yw = demean_panel(panel.y.view(), Absorbed::Individual, n_ent, n_time);
    let xw = demean_panel_matrix(x.slice(s![.., 1..]), Absorbed::Individual, n_ent, n_time)?;
    require_non_degenerate_fit(&yw, &xw, &coef_names[1..], y_col, "entity demeaning", false)?;

    let sp = if lag || err {
        Some(PanelSpectrum::new(listw)?)
    } else {
        None
    };
    let (lo, hi) = sp.as_ref().map_or((-1.0, 1.0), |s| (s.lower, s.upper));
    let data = ReData {
        wy: spatial_lag_panel(&panel.y, n_ent, n_time, listw),
        y: panel.y.clone(),
        x,
        n: n_ent,
        t: n_time,
        lag,
        err: config.spatial_error,
        sp,
        listw,
        cache: RefCell::new(Vec::new()),
    };

    // psi = (rho?, log phi, lambda?)
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    if lag {
        lower.push(lo);
        upper.push(hi);
    }
    lower.push(-LOG_PHI_BOUND);
    upper.push(LOG_PHI_BOUND);
    if err {
        lower.push(lo);
        upper.push(hi);
    }
    let i_phi = usize::from(lag);
    let unpack = |psi: &[f64]| {
        (
            if lag { psi[0] } else { 0.0 },
            psi[i_phi].exp(),
            if err { psi[psi.len() - 1] } else { 0.0 },
        )
    };
    let mut warnings = Vec::new();
    let (psi, mut prof, conv) = newton_profile(
        |psi| {
            let (rho, phi, lambda) = unpack(psi);
            re_profile(&data, rho, phi, lambda, &coef_names)
        },
        &vec![0.0; lower.len()],
        &lower,
        &upper,
        config.tol,
        config.max_iter,
    )?;
    if let Some(note) = conv.note("Spatial panel ML (spml)") {
        warnings.push(note);
    }
    let (rho, mut phi, lambda) = unpack(&psi);
    if psi[i_phi] <= -LOG_PHI_BOUND + 1e-9 {
        phi = 0.0;
        prof = re_profile(&data, rho, 0.0, lambda, &coef_names)?;
        warnings.push(
            "phi = sigma_mu^2 / sigma^2 is estimated at the boundary 0: the likelihood \
             increases as phi decreases to 0, so the fit equals the pooled model; phi has no \
             standard error"
                .into(),
        );
    }

    let mut kinds: Vec<ParamKind> = (0..k).map(ParamKind::Coef).collect();
    let mut names = coef_names.clone();
    if lag {
        kinds.push(ParamKind::Spatial);
        names.push("rho".into());
    }
    kinds.push(ParamKind::Scale);
    names.push("phi".into());
    if err {
        kinds.push(ParamKind::Spatial);
        names.push("lambda".into());
    }
    kinds.push(ParamKind::Scale);
    names.push("sigma2".into());
    let (steps, active) = information_steps(&prof, &kinds, Some((lo, hi)), &names, &mut warnings);
    let hess = fd_hessian(|th| re_score(&data, th), &prof.theta, &steps, &active)?;
    let se = ml_standard_errors(&hess, &active, &mut warnings)?;

    let theta = prof.theta;
    let beta = ArrayView1::from(&theta[..k]);
    let residuals = &data.y - &(&data.wy * rho) - data.x.dot(&beta);
    let fitted = &data.y - &residuals;
    Ok(assemble_spml(
        SpmlFit {
            theta,
            se,
            k,
            lag,
            phi: Some(phi),
            err,
            ll: prof.ll,
            df,
            coef_names,
            convergence: Some(conv),
            warnings,
            residuals,
            fitted,
        },
        config,
        n_ent,
        n_time,
    ))
}

// ============================================================================
// GMM Estimation
// ============================================================================

fn run_spgm_within(
    y: &Array1<f64>,
    x: &Array2<f64>,
    entity_ids: &[usize],
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
        let wy = spatial_lag_panel(y, n_entities, n_time, listw);
        let wy_within = demean_by_entity(&wy, entity_ids, n_entities);

        // W*X as additional instruments
        let mut wx_within = Array2::zeros((n, k_fe));
        for j in 0..k_fe {
            let x_col = x_fe.column(j).to_owned();
            let wx = spatial_lag_panel(&x_col, n_entities, n_time, listw);
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

        // Augmented X with Wy
        let mut x_aug = Array2::zeros((n, k_fe + 1));
        for j in 0..k_fe {
            x_aug.column_mut(j).assign(&x_fe.column(j));
        }
        x_aug.column_mut(k_fe).assign(&wy_within);

        // P_Z X = Z ((Z'Z)^-1 (Z'X)): O(NT k) memory instead of the NT x NT projection
        let x_proj = z.dot(&ztz_inv.dot(&z.t().dot(&x_aug)));
        let xtpx = x_aug.t().dot(&x_proj);
        let xtpx_inv = matrix_inverse(&xtpx.view())?;
        let xtpy = x_proj.t().dot(&y_within);
        let gamma = xtpx_inv.dot(&xtpy);

        let rho = gamma[k_fe];
        let _beta = gamma.slice(ndarray::s![..k_fe]).to_owned();

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
        let wy = spatial_lag_panel(y, n_entities, n_time, listw);
        let wy_within = demean_by_entity(&wy, entity_ids, n_entities);
        let y_tilde = &y_within - rho * &wy_within;
        let xty_tilde = xty(&x_fe.view(), &y_tilde);
        xtx_inv.dot(&xty_tilde)
    } else {
        xtx_inv.dot(&xty_vec)
    };

    let residuals = if let Some(rho) = rho_opt {
        let wy = spatial_lag_panel(y, n_entities, n_time, listw);
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
    let (sargan_stat, sargan_p, sargan_df) =
        if n_instruments > k_fe + if config.lag { 1 } else { 0 } {
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
    n_entities: usize,
    n_time: usize,
    _listw: &mut SpatialWeights,
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
    let sigma2_between = y_means
        .iter()
        .map(|&m| (m - y_overall).powi(2))
        .sum::<f64>()
        / (n_entities - 1) as f64;
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

    let std_errors: Vec<f64> = (0..k).map(|j| (sigma2 * xtx_inv[[j, j]]).sqrt()).collect();

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
    use p2a_validation::{Case, Check, Tol};
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

    /// N = 30 entities on the shared fixture's kNN-3 weights, T = 4 periods.
    fn panel_from_fixture() -> (Dataset, SpatialWeights) {
        use crate::spatial::test_fixtures::{CY, X1, Y, knn3};
        let (mut entity, mut time, mut y, mut x1, mut x2) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for t in 0..4 {
            for i in 0..30 {
                let (fi, ft) = (i as f64, t as f64);
                let v1 = X1[i] + 0.1 * ft + 0.05 * (fi * ft).sin();
                let v2 = CY[i] / 10.0 + 0.2 * (1.3 * fi + ft).cos();
                entity.push(i as i64);
                time.push(t as i64);
                x1.push(v1);
                x2.push(v2);
                y.push(Y[i] / 4.0 + 0.6 * v1 - 0.4 * v2 + 0.15 * (2.1 * fi + 0.7 * ft).sin());
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
        (Dataset::new(df), knn3(WeightStyle::RowStd))
    }

    fn assert_rel(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:.17e} vs {b:.17e}"
        );
    }

    #[test]
    fn spgm_lag_results_are_unchanged() {
        let (d, mut w) = panel_from_fixture();
        let config = SpgmConfig {
            method: SpgmMethod::W2sls,
            lag: true,
            spatial_error: false,
            ..Default::default()
        };
        let r = run_spgm(&d, "y", &["x1", "x2"], "entity", "time", &mut w, config).unwrap();
        // Values printed by the code that formed the dense NT x NT projection P_Z.
        assert_rel(r.rho.unwrap(), -0.18808132677669764, 1e-10);
        let coefficients = [0.6778198161002749, -0.366417317278429];
        let std_errors = [0.06930267708140747, 0.0631177888519706];
        for j in 0..2 {
            assert_rel(r.coefficients[j], coefficients[j], 1e-10);
            assert_rel(r.std_errors[j], std_errors[j], 1e-10);
        }
        assert_rel(r.sigma2, 0.007531180095604837, 1e-10);
        assert_rel(r.sargan_stat.unwrap(), 2.355851571465689, 1e-10);
        assert_eq!((r.n_instruments, r.sargan_df), (4, Some(1)));
    }

    #[test]
    fn panel_estimators_reject_bad_tolerance_and_honour_cancellation() {
        let (d, mut w) = panel_from_fixture();
        let invalid = |e: EconError| matches!(e, EconError::InvalidSpecification { .. });
        for (tol, max_iter) in [(0.0, 100), (-1e-8, 100), (f64::NAN, 100), (1e-8, 0)] {
            let c = SpmlConfig {
                lag: true,
                tol,
                max_iter,
                ..Default::default()
            };
            let e = run_spml(&d, "y", &["x1", "x2"], "entity", "time", &mut w, c).unwrap_err();
            assert!(invalid(e), "spml tol={tol} max_iter={max_iter}");
            let c = SpgmConfig {
                lag: true,
                tol,
                max_iter,
                ..Default::default()
            };
            let e = run_spgm(&d, "y", &["x1", "x2"], "entity", "time", &mut w, c).unwrap_err();
            assert!(invalid(e), "spgm tol={tol} max_iter={max_iter}");
        }
        let t = crate::resource::CancelToken::new();
        t.cancel();
        let c = SpmlConfig {
            lag: true,
            ..Default::default()
        };
        let r = crate::resource::with_budget(crate::resource::Budget::new(None, t), || {
            run_spml(&d, "y", &["x1", "x2"], "entity", "time", &mut w, c)
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{r:?}");
    }

    // ------------------------------------------------------------------------
    // Panel layout (T2-WP14): balanced panels, sorted-id W rows, any row order.
    // R: spml_entity_order.R (base R; splm is not installed), reproduced in
    // validation/econometrics/splm.md.
    // ------------------------------------------------------------------------

    /// Dataset A of the R script and its W (entities in id order): entity k = 1..=12 on a
    /// 3 x 4 integer grid at ((k-1) mod 4, (k-1) div 4), rook W (`from_distance(0, 1)`),
    /// row-standardised; Int64 time 2001..=2005 (t = 1..=5);
    /// x = sin(0.7 k + 1.3 t) + 0.1 k, alpha = 0.5 cos(1.1 k), e = 0.3 sin(2.3 k t + 0.4),
    /// y_t solves (I - 0.5 W) y_t = 1.5 x_t + alpha + e_t by the Neumann iteration
    /// v <- b + 0.5 W v from v = b, 200 times (0.5^200 is below rounding; R uses solve()).
    /// Rows are entity-major, as R's `A`.
    fn grid_panel() -> (DataFrame, SpatialWeights) {
        let coords: Vec<(f64, f64)> = (0..12).map(|i| ((i % 4) as f64, (i / 4) as f64)).collect();
        let w = SpatialWeights::from_neighbors(
            &Neighbors::from_distance(&coords, 0.0, 1.0),
            WeightStyle::RowStd,
        );
        let (n, tt) = (12usize, 5usize);
        let x = |k: f64, t: f64| (0.7 * k + 1.3 * t).sin() + 0.1 * k;
        let mut y = Array2::<f64>::zeros((n, tt));
        for t in 1..=tt {
            let b = Array1::from_iter((1..=n).map(|k| {
                let (k, t) = (k as f64, t as f64);
                1.5 * x(k, t) + 0.5 * (1.1 * k).cos() + 0.3 * (2.3 * k * t + 0.4).sin()
            }));
            let mut v = b.clone();
            for _ in 0..200 {
                v = &b + 0.5 * &w.lag(&v);
            }
            y.column_mut(t - 1).assign(&v);
        }
        let (mut entity, mut time, mut yv, mut xv) = (vec![], vec![], vec![], vec![]);
        for k in 1..=n {
            for t in 1..=tt {
                entity.push(k as i64);
                time.push(2000 + t as i64);
                yv.push(y[[k - 1, t - 1]]);
                xv.push(x(k as f64, t as f64));
            }
        }
        let df = df! { "entity" => entity, "time" => time, "y" => yv, "x" => xv }.unwrap();
        (df, w)
    }

    /// Row i of the result = row (7 i + 3) mod 60 of `df`.
    fn shuffled(df: &DataFrame) -> DataFrame {
        let idx: Vec<IdxSize> = (0..60).map(|i| ((7 * i + 3) % 60) as IdxSize).collect();
        df.take(&IdxCa::from_vec("idx".into(), idx)).unwrap()
    }

    const C_LABELS: [&str; 12] = [
        "u1", "u10", "u11", "u12", "u2", "u3", "u4", "u5", "u6", "u7", "u8", "u9",
    ];

    /// Entity k (Int64) -> `C_LABELS[k - 1]` (String), so the k-th entity in sorted id
    /// order is still entity k (R: `labs[Bd$entity]`).
    fn with_labels(df: &DataFrame) -> DataFrame {
        let labels: Vec<&str> = df
            .column("entity")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .map(|k| C_LABELS[(k - 1) as usize])
            .collect();
        let mut out = df.clone();
        out.with_column(Column::new("entity".into(), labels))
            .unwrap();
        out
    }

    fn lag_cfg() -> SpmlConfig {
        SpmlConfig {
            lag: true,
            ..Default::default()
        }
    }

    fn spml(df: &DataFrame, w: &SpatialWeights, cfg: SpmlConfig) -> EconResult<SpmlResult> {
        let ds = Dataset::new(df.clone());
        run_spml(&ds, "y", &["x"], "entity", "time", &mut w.clone(), cfg)
    }

    fn spgm(df: &DataFrame, w: &SpatialWeights, cfg: SpgmConfig) -> EconResult<SpgmResult> {
        let ds = Dataset::new(df.clone());
        run_spgm(&ds, "y", &["x"], "entity", "time", &mut w.clone(), cfg)
    }

    /// Column `name` of `df` as stored (Float64).
    fn f64_values(df: &DataFrame, name: &str) -> Vec<f64> {
        df.column(name)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    /// R: spml_entity_order.R (the script is reproduced in
    /// validation/econometrics/splm.md). Base-R re-implementation of the spml within-lag
    /// concentrated likelihood, not splm itself (not installed), so this test is not
    /// R-pinned in the registry's sense.
    #[test]
    fn test_spml_entity_order_r_cross_check() {
        let (a, w) = grid_panel();
        // The fixture reproduces R's first row of A.
        let (y0, x0) = (f64_values(&a, "y")[0], f64_values(&a, "x")[0]);
        assert!((y0 - 2.313_251_294_567_377).abs() < 1e-13, "{y0:.17}");
        assert!((x0 - 1.009_297_426_825_681_8).abs() < 1e-15, "{x0:.17}");
        for df in [a.clone(), shuffled(&a), with_labels(&shuffled(&a))] {
            let r = spml(&df, &w, lag_cfg()).unwrap();
            // Newton on the concentrated likelihood; the T2 reference stops optimize at
            // tol 1e-15
            let rho = r.rho.unwrap();
            assert!(
                Tol::iterative(1e-6).allows(rho, 0.415_229_776_053_281_8),
                "rho {rho}"
            );
            let beta = r.coefficients[0];
            assert!(
                Tol::iterative(1e-6).allows(beta, 1.453_594_665_020_881_5),
                "{beta}"
            );
            let ll = r.log_likelihood;
            assert!(
                Tol::iterative(1e-9).allows(ll, 13.520_649_774_963_92),
                "{ll}"
            );
            let sigma2 = r.variance.sigma_epsilon.powi(2);
            // sigma_epsilon is the ML estimate RSS / (N T) (was RSS / (n - N - k)); same R
            // script (spml_entity_order.R): 0.044630452161004024 * 47 / 60.
            assert!(
                Tol::iterative(1e-6).allows(sigma2, 0.03496052085945315),
                "{sigma2}"
            );
            assert_eq!((r.n_obs, r.n_entities, r.n_time, r.df), (60, 12, 5, 46));
        }
    }

    #[test]
    fn test_spml_spgm_row_order_invariant() {
        let (a, w) = grid_panel();
        let b = shuffled(&a);
        let perm: Vec<usize> = (0..60).map(|i| (7 * i + 3) % 60).collect();
        for cfg in [
            lag_cfg(),
            SpmlConfig {
                spatial_error: SpatialErrorType::Baltagi,
                ..Default::default()
            },
            SpmlConfig {
                model: SpatialPanelModel::Random,
                lag: true,
                ..Default::default()
            },
            SpmlConfig {
                model: SpatialPanelModel::Pooling,
                lag: true,
                ..Default::default()
            },
        ] {
            let (ra, rb) = (
                spml(&a, &w, cfg.clone()).unwrap(),
                spml(&b, &w, cfg).unwrap(),
            );
            assert_eq!(ra.coefficients, rb.coefficients);
            assert_eq!(ra.std_errors, rb.std_errors);
            assert_eq!(
                (ra.rho, ra.lambda, ra.log_likelihood),
                (rb.rho, rb.lambda, rb.log_likelihood)
            );
            // residuals and fitted values come back in each dataset's row order
            for i in 0..60 {
                assert_eq!(rb.residuals[i], ra.residuals[perm[i]], "residual {i}");
                assert_eq!(rb.fitted[i], ra.fitted[perm[i]], "fitted {i}");
            }
        }
        for cfg in [
            SpgmConfig {
                lag: true,
                spatial_error: false,
                ..Default::default()
            },
            SpgmConfig::default(),
            SpgmConfig {
                method: SpgmMethod::G2sls,
                ..Default::default()
            },
        ] {
            let (ra, rb) = (
                spgm(&a, &w, cfg.clone()).unwrap(),
                spgm(&b, &w, cfg).unwrap(),
            );
            assert_eq!(
                (ra.coefficients, ra.std_errors, ra.rho, ra.lambda, ra.sigma2),
                (rb.coefficients, rb.std_errors, rb.rho, rb.lambda, rb.sigma2)
            );
        }
    }

    #[test]
    fn test_spatial_lag_panel_follows_sorted_entity_ids() {
        let df = df! {
            "entity" => ["b", "a", "c", "a", "b", "c"],
            "time" => [1i64, 1, 1, 2, 2, 2],
            "y" => [10.0, 1.0, 100.0, 2.0, 20.0, 200.0],
        }
        .unwrap();
        let panel = extract_panel_data(&Dataset::new(df), "y", &[], "entity", "time").unwrap();
        assert_eq!(panel.entity_labels, ["a", "b", "c"]);
        let w = SpatialWeights::from_neighbors(
            &Neighbors::from_indices(vec![vec![1], vec![0, 2], vec![1]]),
            WeightStyle::Binary,
        );
        let wy = to_row_order(&spatial_lag_panel(&panel.y, 3, 2, &w), &panel.rows);
        // R (spml_entity_order.R, 3-entity example): 101 10 10 20 202 20
        assert_eq!(wy.to_vec(), vec![101.0, 10.0, 10.0, 20.0, 202.0, 20.0]);
    }

    #[test]
    fn test_spml_unbalanced_and_duplicated_panels_refused() {
        let (a, w) = grid_panel();
        let keep: Vec<bool> = (0..60).map(|r| r != 13).collect(); // entity 3, time 2004
        let unb = a
            .filter(&BooleanChunked::from_slice("m".into(), &keep))
            .unwrap();
        let dup = a.vstack(&a.slice(0, 1)).unwrap(); // entity 1, time 2001 twice
        for (df, parts) in [
            (&unb, ["balanced panel", "1 of 60", "entity '3', time 2004"]),
            (
                &dup,
                ["1 duplicated", "(entity, time)", "entity '1', time 2001"],
            ),
        ] {
            let e1 = spml(df, &w, lag_cfg()).unwrap_err();
            let e2 = spgm(df, &w, SpgmConfig::default()).unwrap_err();
            for e in [e1, e2] {
                let m = e.to_string();
                assert!(
                    matches!(e, EconError::InvalidSpecification { .. })
                        && parts.iter().all(|p| m.contains(p)),
                    "{m}"
                );
            }
        }
    }

    #[test]
    fn test_na_spml_missing_and_invalid_ids() {
        let (a, w) = grid_panel();
        let null_at_5 = |name: &str| {
            let col = a.column(name).unwrap();
            let mut out = a.clone();
            let replaced = match col.dtype() {
                DataType::Int64 => {
                    let v: Vec<Option<i64>> = col
                        .i64()
                        .unwrap()
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| if i == 5 { None } else { v })
                        .collect();
                    Column::new(name.into(), v)
                }
                _ => {
                    let v: Vec<Option<f64>> = col
                        .f64()
                        .unwrap()
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| if i == 5 { None } else { v })
                        .collect();
                    Column::new(name.into(), v)
                }
            };
            out.with_column(replaced).unwrap();
            out
        };
        for name in ["y", "entity", "time"] {
            let e = spml(&null_at_5(name), &w, lag_cfg()).unwrap_err();
            assert!(
                matches!(e, EconError::NullValues { ref column, count: 1 } if column == name),
                "{name}: {e:?}"
            );
            let e = spgm(&null_at_5(name), &w, SpgmConfig::default()).unwrap_err();
            assert!(
                matches!(e, EconError::NullValues { ref column, count: 1 } if column == name),
                "{name}: {e:?}"
            );
        }
        // a fractional time
        let mut times: Vec<f64> = a
            .column("time")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .map(|t| t as f64)
            .collect();
        times[4] = 2001.5;
        let mut d = a.clone();
        d.with_column(Column::new("time".into(), times)).unwrap();
        let e = spml(&d, &w, lag_cfg()).unwrap_err();
        let m = e.to_string();
        assert!(
            matches!(e, EconError::InvalidSpecification { .. })
                && m.contains("Time column 'time' must hold whole numbers"),
            "{m}"
        );
        // a text outcome
        let mut d = a.clone();
        let text: Vec<String> = (0..60).map(|i| format!("v{i}")).collect();
        d.with_column(Column::new("y".into(), text)).unwrap();
        let e = spml(&d, &w, lag_cfg()).unwrap_err();
        assert!(
            matches!(e, EconError::NonNumericColumn { ref column } if column == "y"),
            "{e:?}"
        );
    }

    #[test]
    fn test_na_spml_int64_equals_float64() {
        let (a, w) = grid_panel();
        let scaled = |name: &str| -> Vec<i64> {
            f64_values(&a, name)
                .iter()
                .map(|v| (10.0 * v).round() as i64)
                .collect()
        };
        let (y_i, x_i) = (scaled("y"), scaled("x"));
        let as_f64 = |v: &[i64]| v.iter().map(|&a| a as f64).collect::<Vec<f64>>();
        let mut int = a.clone();
        int.with_column(Column::new("y".into(), y_i.clone()))
            .unwrap();
        int.with_column(Column::new("x".into(), x_i.clone()))
            .unwrap();
        let mut flt = a.clone();
        flt.with_column(Column::new("y".into(), as_f64(&y_i)))
            .unwrap();
        flt.with_column(Column::new("x".into(), as_f64(&x_i)))
            .unwrap();
        assert_eq!(int.column("y").unwrap().dtype(), &DataType::Int64);
        let (ri, rf) = (spml(&int, &w, lag_cfg()), spml(&flt, &w, lag_cfg()));
        assert!(rf.is_ok(), "{rf:?}");
        assert_eq!(format!("{ri:?}"), format!("{rf:?}"));
        let (gi, gf) = (
            spgm(&int, &w, SpgmConfig::default()),
            spgm(&flt, &w, SpgmConfig::default()),
        );
        assert!(gf.is_ok(), "{gf:?}");
        assert_eq!(format!("{gi:?}"), format!("{gf:?}"));
    }

    #[test]
    fn test_spml_weights_must_follow_sorted_entities() {
        let (a, w) = grid_panel();
        let c = with_labels(&shuffled(&a));
        let coords: Vec<(f64, f64)> = (0..12).map(|k| ((k % 4) as f64, (k / 4) as f64)).collect();
        // an 11 x 11 W for 12 entities
        let w11 = SpatialWeights::from_neighbors(
            &Neighbors::from_distance(&coords[..11], 0.0, 1.0),
            WeightStyle::RowStd,
        );
        let m = spml(&c, &w11, lag_cfg()).unwrap_err().to_string();
        assert!(
            m.contains("12 entities")
                && m.contains("sorted id order")
                && m.contains("u1, u10, u11, ..."),
            "{m}"
        );
        let m = spgm(&c, &w11, SpgmConfig::default())
            .unwrap_err()
            .to_string();
        assert!(m.contains("sorted id order"), "{m}");
        // region ids equal to the sorted labels: accepted, the same result as without ids
        let mut nb = Neighbors::from_distance(&coords, 0.0, 1.0);
        nb.set_region_ids(C_LABELS.iter().map(|s| s.to_string()).collect());
        let w_ids = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        assert_eq!(
            format!("{:?}", spml(&c, &w_ids, lag_cfg()).unwrap().coefficients),
            format!("{:?}", spml(&c, &w, lag_cfg()).unwrap().coefficients)
        );
        // region ids in another order: refused
        let mut nb = Neighbors::from_distance(&coords, 0.0, 1.0);
        nb.set_region_ids(C_LABELS.iter().rev().map(|s| s.to_string()).collect());
        let reversed = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        let m = spml(&c, &reversed, lag_cfg()).unwrap_err().to_string();
        assert!(
            m.contains("row 0 has region id 'u9'") && m.contains("'u1'"),
            "{m}"
        );
    }

    // ------------------------------------------------------------------------
    // spml and spgm against the written-out references (validation/r/cases/spatial_splm.R)
    // ------------------------------------------------------------------------

    /// splm_sim (validation/r/cases/spatial_splm.R): N = 49, T = 6, rows shuffled; W = 4
    /// nearest neighbours, row-standardised, on the period-1 coordinates in id order (W row
    /// i = i-th entity in sorted id order). R: no kNN ties (smallest 5th-4th neighbour
    /// distance gap 4.06e-4); neighbours of entity 1 = 17 39 40 45. Called as
    /// `splm_case("splm_spml_spgm")`: the case id is a literal in each test body.
    fn splm_case(id: &str) -> (Case, Dataset, SpatialWeights) {
        let (case, ds) = crate::test_support::load_case(id);
        let w = splm_weights(&ds);
        (case, ds, w)
    }

    /// The same data for the property tests, which compare no R value and so load no R
    /// case: `validation/datasets/generated/splm_sim.csv`, read as `p2a data load` reads it.
    fn splm_sim() -> (Dataset, SpatialWeights) {
        let path = p2a_validation::repo_path("validation/datasets/generated/splm_sim.csv");
        let ds = Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        let w = splm_weights(&ds);
        (ds, w)
    }

    /// W of splm_sim: 4 nearest neighbours of the period-1 coordinates in id order.
    fn splm_weights(ds: &Dataset) -> SpatialWeights {
        let col = |c: &str| -> Vec<f64> {
            ds.df()
                .column(c)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .map(|v| v.unwrap())
                .collect()
        };
        let (id, time, cx, cy) = (col("id"), col("time"), col("cx"), col("cy"));
        let mut pts: Vec<(f64, (f64, f64))> = (0..id.len())
            .filter(|&r| time[r] == 1.0)
            .map(|r| (id[r], (cx[r], cy[r])))
            .collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let coords: Vec<(f64, f64)> = pts.into_iter().map(|p| p.1).collect();
        assert_eq!(coords.len(), 49);
        let w =
            SpatialWeights::from_neighbors(&Neighbors::from_knn(&coords, 4), WeightStyle::RowStd);
        assert_eq!(w.neighbors().get_neighbors(0), &[16, 38, 39, 44]);
        w
    }

    fn spml_cfg(
        model: SpatialPanelModel,
        effect: SpatialPanelEffect,
        lag: bool,
        err: SpatialErrorType,
    ) -> SpmlConfig {
        let mut c = SpmlConfig::default();
        c.model = model;
        c.effect = effect;
        c.lag = lag;
        c.spatial_error = err;
        c
    }

    fn fit(ds: &Dataset, w: &SpatialWeights, y: &str, cfg: SpmlConfig) -> SpmlResult {
        run_spml(ds, y, &["x1", "x2"], "id", "time", &mut w.clone(), cfg).unwrap()
    }

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Within/pooled pins to `<tag>_{coef, rho, lambda, s2, se, loglik}`. Newton converges
    /// quadratically, so parameters at Tol::iterative(1e-8); the SEs are a central difference
    /// of the analytic score against R's Richardson Jacobian: Tol::iterative(1e-6).
    fn check_spml_fe(case: &Case, r: &SpmlResult, tag: &str) {
        let k = |s: &str| format!("{tag}_{s}");
        Check::new(case)
            .vector(&k("coef"), &r.coefficients, Tol::iterative(1e-8))
            .scalar(&k("rho"), r.rho.unwrap_or(0.0), Tol::iterative(1e-8))
            .scalar(&k("lambda"), r.lambda.unwrap_or(0.0), Tol::iterative(1e-8))
            .scalar(
                &k("s2"),
                r.variance.sigma_epsilon.powi(2),
                Tol::iterative(1e-8),
            )
            .vector(&k("se"), &r.parameter_se, Tol::iterative(1e-6))
            .scalar(&k("loglik"), r.log_likelihood, Tol::iterative(1e-10))
            .finish();
        assert!(r.convergence.as_ref().unwrap().converged, "{tag}");
    }

    /// `ds` with `cols` multiplied by `f`.
    fn scaled(ds: &Dataset, cols: &[&str], f: f64) -> Dataset {
        let mut df = ds.df().clone();
        for &c in cols {
            let s = (df.column(c).unwrap().f64().unwrap() * f)
                .with_name(c.into())
                .into_series();
            df.with_column(s).unwrap();
        }
        Dataset::new(df)
    }

    const LAG_NAMES: [&str; 4] = ["x1", "x2", "rho", "sigma2"];
    const ERR_NAMES: [&str; 4] = ["x1", "x2", "lambda", "sigma2"];

    #[test]
    fn test_validate_spml_individual_lag() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_lag",
            spml_cfg(Within, Individual, true, SpatialErrorType::None),
        );
        check_spml_fe(&case, &r, "individual_lag");
        assert_eq!(r.parameter_names, strings(&LAG_NAMES));
    }

    #[test]
    fn test_validate_spml_individual_error() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_err",
            spml_cfg(Within, Individual, false, SpatialErrorType::Baltagi),
        );
        check_spml_fe(&case, &r, "individual_error");
        assert_eq!(r.parameter_names, strings(&ERR_NAMES));
    }

    #[test]
    fn test_validate_spml_individual_sarar() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_lag",
            spml_cfg(Within, Individual, true, SpatialErrorType::Baltagi),
        );
        check_spml_fe(&case, &r, "individual_sarar");
        assert_eq!(
            r.parameter_names,
            strings(&["x1", "x2", "rho", "lambda", "sigma2"])
        );
    }

    #[test]
    fn test_validate_spml_time_lag() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_lag",
            spml_cfg(Within, Time, true, SpatialErrorType::None),
        );
        check_spml_fe(&case, &r, "time_lag");
        assert_eq!(r.parameter_names, strings(&LAG_NAMES));
    }

    #[test]
    fn test_validate_spml_time_error() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_err",
            spml_cfg(Within, Time, false, SpatialErrorType::Baltagi),
        );
        check_spml_fe(&case, &r, "time_error");
        assert_eq!(r.parameter_names, strings(&ERR_NAMES));
    }

    #[test]
    fn test_validate_spml_twoways_lag() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_lag",
            spml_cfg(Within, TwoWays, true, SpatialErrorType::None),
        );
        check_spml_fe(&case, &r, "twoways_lag");
        assert_eq!(r.parameter_names, strings(&LAG_NAMES));
    }

    #[test]
    fn test_validate_spml_twoways_error() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_err",
            spml_cfg(Within, TwoWays, false, SpatialErrorType::Baltagi),
        );
        check_spml_fe(&case, &r, "twoways_error");
        assert_eq!(r.parameter_names, strings(&ERR_NAMES));
    }

    #[test]
    fn test_validate_spml_pooling_lag() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_lag",
            spml_cfg(Pooling, Individual, true, SpatialErrorType::None),
        );
        check_spml_fe(&case, &r, "pooling_lag");
        assert_eq!(
            r.parameter_names,
            strings(&["(Intercept)", "x1", "x2", "rho", "sigma2"])
        );
    }

    #[test]
    fn test_validate_spml_pooling_error() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        use SpatialPanelModel::*;
        let r = fit(
            &ds,
            &w,
            "y_err",
            spml_cfg(Pooling, Individual, false, SpatialErrorType::Baltagi),
        );
        check_spml_fe(&case, &r, "pooling_error");
        assert_eq!(
            r.parameter_names,
            strings(&["(Intercept)", "x1", "x2", "lambda", "sigma2"])
        );
    }

    #[test]
    fn test_validate_spml_within_none_plm() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        use SpatialPanelEffect::*;
        for (effect, tag) in [
            (Individual, "individual"),
            (Time, "time"),
            (TwoWays, "twoways"),
        ] {
            let r = fit(
                &ds,
                &w,
                "y_err",
                spml_cfg(
                    SpatialPanelModel::Within,
                    effect,
                    false,
                    SpatialErrorType::None,
                ),
            );
            assert!(r.convergence.is_none() && r.rho.is_none() && r.lambda.is_none());
            Check::new(&case)
                .vector(
                    &format!("{tag}_none_coef_plm"),
                    &r.coefficients,
                    Tol::coef(294),
                )
                .scalar(
                    &format!("{tag}_none_s2"),
                    r.variance.sigma_epsilon.powi(2),
                    Tol::stat(294),
                )
                .vector(&format!("{tag}_none_se"), &r.parameter_se, Tol::se(294))
                .scalar(
                    &format!("{tag}_none_loglik"),
                    r.log_likelihood,
                    Tol::stat(294),
                )
                .finish();
        }
    }

    #[test]
    fn test_spml_error_lambda_not_rho() {
        let (ds, w) = splm_sim();
        let cfg = |e| {
            spml_cfg(
                SpatialPanelModel::Within,
                SpatialPanelEffect::Individual,
                false,
                e,
            )
        };
        let r = fit(&ds, &w, "y_err", cfg(SpatialErrorType::Baltagi));
        assert!(!r.has_lag && r.rho.is_none() && r.rho_se.is_none());
        assert!(Tol::iterative(1e-8).allows(r.lambda.unwrap(), 0.430180548080978));
        assert!(Tol::iterative(1e-6).allows(r.lambda_se.unwrap(), 0.06205389304743795));
        assert_eq!(r.variance.lambda, r.lambda);
        let text = r.to_string();
        assert!(
            text.contains("Spatial error coefficient (lambda)"),
            "{text}"
        );
        assert!(!text.contains("Spatial lag coefficient (rho)"), "{text}");
        // within models sweep mu out: the KKP and Baltagi errors give the same fit
        let k = fit(&ds, &w, "y_err", cfg(SpatialErrorType::KKP));
        assert_eq!(
            (k.coefficients, k.lambda, k.log_likelihood),
            (r.coefficients, r.lambda, r.log_likelihood)
        );
    }

    #[test]
    fn test_spml_effects_change_the_estimate() {
        let (ds, w) = splm_sim();
        let rho = |e| {
            fit(
                &ds,
                &w,
                "y_lag",
                spml_cfg(SpatialPanelModel::Within, e, true, SpatialErrorType::None),
            )
            .rho
            .unwrap()
        };
        let (i, t, tw) = (
            rho(SpatialPanelEffect::Individual),
            rho(SpatialPanelEffect::Time),
            rho(SpatialPanelEffect::TwoWays),
        );
        // R: 0.43022, 0.3739, 0.40971
        assert!(
            (i - t).abs() > 0.05 && (i - tw).abs() > 0.01,
            "{i} {t} {tw}"
        );
    }

    #[test]
    fn test_spml_convergence_is_scale_free() {
        let (ds, w) = splm_sim();
        let cfg = || {
            spml_cfg(
                SpatialPanelModel::Within,
                SpatialPanelEffect::Individual,
                true,
                SpatialErrorType::Baltagi,
            )
        };
        let base = fit(&ds, &w, "y_err", cfg());
        let it = base.convergence.as_ref().unwrap().iterations;
        for f in [1e-6, 1e6] {
            let r = fit(&scaled(&ds, &["y_err", "x1", "x2"], f), &w, "y_err", cfg());
            let tol = Tol::iterative(1e-8);
            assert!(tol.allows(r.rho.unwrap(), base.rho.unwrap()), "{f}");
            assert!(tol.allows(r.lambda.unwrap(), base.lambda.unwrap()), "{f}");
            assert!(
                tol.allows(r.variance.sigma_epsilon, f * base.variance.sigma_epsilon),
                "{f}"
            );
            assert!(r.convergence.unwrap().iterations.abs_diff(it) <= 1, "{f}");
        }
    }

    #[test]
    fn test_spml_pooling_rejects_effects() {
        let (ds, w) = splm_sim();
        for e in [SpatialPanelEffect::Time, SpatialPanelEffect::TwoWays] {
            let cfg = spml_cfg(SpatialPanelModel::Pooling, e, true, SpatialErrorType::None);
            let err = run_spml(
                &ds,
                "y_lag",
                &["x1", "x2"],
                "id",
                "time",
                &mut w.clone(),
                cfg,
            )
            .unwrap_err();
            let m = err.to_string();
            assert!(
                matches!(err, EconError::InvalidSpecification { .. })
                    && m.contains("pooling")
                    && m.contains("effect"),
                "{m}"
            );
        }
    }

    fn check_spml_random(case: &Case, r: &SpmlResult, tag: &str, lag: bool, err: bool) {
        let k = |s: &str| format!("{tag}_{s}");
        let mut c = Check::new(case);
        c.vector(&k("coef"), &r.coefficients, Tol::iterative(1e-8))
            .scalar(&k("phi"), r.variance.phi.unwrap(), Tol::iterative(1e-8))
            .scalar(
                &k("s2"),
                r.variance.sigma_epsilon.powi(2),
                Tol::iterative(1e-8),
            )
            .scalar(
                &k("sigma2_mu"),
                r.variance.sigma_mu.unwrap().powi(2),
                Tol::iterative(1e-8),
            )
            .vector(&k("se"), &r.parameter_se, Tol::iterative(1e-6))
            .scalar(&k("loglik"), r.log_likelihood, Tol::iterative(1e-10));
        if lag {
            c.scalar(&k("rho"), r.rho.unwrap(), Tol::iterative(1e-8));
        }
        if err {
            c.scalar(&k("lambda"), r.lambda.unwrap(), Tol::iterative(1e-8));
        }
        c.finish();
        assert!(r.convergence.as_ref().unwrap().converged, "{tag}");
    }

    fn random_cfg(lag: bool, err: SpatialErrorType) -> SpmlConfig {
        spml_cfg(
            SpatialPanelModel::Random,
            SpatialPanelEffect::Individual,
            lag,
            err,
        )
    }

    /// Hand fixture: no individual effect and shrunken between variation, so the
    /// random-effects ML phi is at its boundary 0 (nlme::lme ML: sigma_mu^2 = 2.8e-11); at
    /// phi = 0 the fit is pooled OLS (R lm).
    fn boundary_panel() -> (Dataset, SpatialWeights) {
        let (mut id, mut time, mut x, mut y) = (vec![], vec![], vec![], vec![]);
        for i in 1..=20i64 {
            let fi = i as f64;
            let e: Vec<f64> = (1..=5)
                .map(|t| 0.5 * (2.3 * fi * t as f64 + 0.4).cos())
                .collect();
            let ebar = e.iter().sum::<f64>() / 5.0;
            for t in 1..=5i64 {
                let xv = (fi + 1.7 * t as f64).sin() + 0.1 * fi;
                id.push(i);
                time.push(t);
                x.push(xv);
                y.push(1.0 + 2.0 * xv + e[(t - 1) as usize] - 0.95 * ebar);
            }
        }
        let df = df! { "id" => id, "time" => time, "x" => x, "y" => y }.unwrap();
        let coords: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, 0.0)).collect();
        (
            Dataset::new(df),
            SpatialWeights::from_neighbors(&Neighbors::from_knn(&coords, 2), WeightStyle::RowStd),
        )
    }

    #[test]
    fn test_validate_spml_random_baltagi() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        let r = fit(
            &ds,
            &w,
            "y_re",
            random_cfg(false, SpatialErrorType::Baltagi),
        );
        check_spml_random(&case, &r, "random_b", false, true);
        assert_eq!(
            r.parameter_names,
            strings(&["(Intercept)", "x1", "x2", "phi", "lambda", "sigma2"])
        );
    }

    #[test]
    fn test_validate_spml_random_kkp() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        let r = fit(&ds, &w, "y_re", random_cfg(false, SpatialErrorType::KKP));
        check_spml_random(&case, &r, "random_kkp", false, true);
    }

    #[test]
    fn test_validate_spml_random_none_lme() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        let r = fit(&ds, &w, "y_re", random_cfg(false, SpatialErrorType::None));
        check_spml_random(&case, &r, "random_none", false, false);
        // nlme::lme(method = "ML"), equal to the written-out likelihood to 1e-15 in R
        Check::new(&case)
            .vector(
                "random_none_lme_coef",
                &r.coefficients,
                Tol::iterative(1e-8),
            )
            .scalar(
                "random_none_lme_s2",
                r.variance.sigma_epsilon.powi(2),
                Tol::iterative(1e-8),
            )
            .scalar(
                "random_none_lme_sigma2_mu",
                r.variance.sigma_mu.unwrap().powi(2),
                Tol::iterative(1e-8),
            )
            .scalar(
                "random_none_lme_loglik",
                r.log_likelihood,
                Tol::iterative(1e-10),
            )
            .finish();
        assert_eq!(
            r.parameter_names,
            strings(&["(Intercept)", "x1", "x2", "phi", "sigma2"])
        );
    }

    #[test]
    fn test_validate_spml_random_lag() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        let r = fit(&ds, &w, "y_lag", random_cfg(true, SpatialErrorType::None));
        check_spml_random(&case, &r, "random_lag", true, false);
    }

    #[test]
    fn test_validate_spml_random_sarar_baltagi() {
        let (case, ds, w) = splm_case("splm_spml_spgm");
        let r = fit(
            &ds,
            &w,
            "y_lag",
            random_cfg(true, SpatialErrorType::Baltagi),
        );
        check_spml_random(&case, &r, "random_sarar_b", true, true);
        assert_eq!(
            r.parameter_names,
            strings(&["(Intercept)", "x1", "x2", "rho", "phi", "lambda", "sigma2"])
        );
    }

    #[test]
    fn test_spml_random_error_types_differ() {
        let (ds, w) = splm_sim();
        let b = fit(
            &ds,
            &w,
            "y_re",
            random_cfg(false, SpatialErrorType::Baltagi),
        );
        let k = fit(&ds, &w, "y_re", random_cfg(false, SpatialErrorType::KKP));
        let none = fit(&ds, &w, "y_re", random_cfg(false, SpatialErrorType::None));
        // R: lambda 0.38557 (b) against 0.34525 (kkp); phi 0.8677 (none) against 1.0388 (b)
        assert!((b.lambda.unwrap() - k.lambda.unwrap()).abs() > 0.02);
        assert!((none.variance.phi.unwrap() - b.variance.phi.unwrap()).abs() > 0.1);
    }

    #[test]
    fn test_spml_random_phi_at_boundary() {
        let (ds, w) = boundary_panel();
        // R: the first row (i = 1, t = 1)
        let x0 = ds.df().column("x").unwrap().f64().unwrap().get(0).unwrap();
        let y0 = ds.df().column("y").unwrap().f64().unwrap().get(0).unwrap();
        assert!((x0 - 0.52737988023382976).abs() < 1e-15, "{x0}");
        assert!((y0 - 1.6305539393475814).abs() < 1e-14, "{y0}");
        let r = run_spml(
            &ds,
            "y",
            &["x"],
            "id",
            "time",
            &mut w.clone(),
            random_cfg(false, SpatialErrorType::None),
        )
        .unwrap();
        assert_eq!(r.variance.phi, Some(0.0));
        assert_eq!(r.variance.sigma_mu, Some(0.0));
        assert!(
            r.warnings.iter().any(|w| w.contains("boundary 0")),
            "{:?}",
            r.warnings
        );
        let lm = [0.97369334871553836, 2.0246769270507219];
        for j in 0..2 {
            assert!(Tol::coef(100).allows(r.coefficients[j], lm[j]), "{j}");
        }
        assert!(Tol::stat(100).allows(r.variance.sigma_epsilon.powi(2), 0.1026975911947393));
        assert!(r.parameter_se[2].is_nan());
        for (j, v) in [
            (0, 0.048334128739317243),
            (1, 0.034412295940336811),
            (3, 0.014523632629064805),
        ] {
            assert!(
                Tol::iterative(1e-6).allows(r.parameter_se[j], v),
                "{j}: {}",
                r.parameter_se[j]
            );
        }
        assert!(Tol::iterative(1e-10).allows(r.log_likelihood, -28.095522465646837));
    }

    #[test]
    fn test_spml_random_rejects_time_effects() {
        let (ds, w) = splm_sim();
        for e in [SpatialPanelEffect::Time, SpatialPanelEffect::TwoWays] {
            let cfg = spml_cfg(SpatialPanelModel::Random, e, false, SpatialErrorType::None);
            let err = run_spml(
                &ds,
                "y_re",
                &["x1", "x2"],
                "id",
                "time",
                &mut w.clone(),
                cfg,
            )
            .unwrap_err();
            let m = err.to_string();
            assert!(
                matches!(err, EconError::InvalidSpecification { .. })
                    && m.contains("random")
                    && m.contains("individual"),
                "{m}"
            );
        }
    }

    #[test]
    fn test_spml_random_convergence_is_scale_free() {
        let (ds, w) = splm_sim();
        let cfg = || random_cfg(false, SpatialErrorType::Baltagi);
        let base = fit(&ds, &w, "y_re", cfg());
        let it = base.convergence.as_ref().unwrap().iterations;
        for f in [1e-6, 1e6] {
            let r = fit(&scaled(&ds, &["y_re", "x1", "x2"], f), &w, "y_re", cfg());
            let tol = Tol::iterative(1e-8);
            assert!(tol.allows(r.lambda.unwrap(), base.lambda.unwrap()), "{f}");
            assert!(
                tol.allows(r.variance.phi.unwrap(), base.variance.phi.unwrap()),
                "{f}"
            );
            assert!(r.convergence.unwrap().iterations.abs_diff(it) <= 1, "{f}");
        }
    }

    #[test]
    fn test_spml_perfect_fit_is_refused() {
        // y is exact in x once the individual effects are taken out
        let (ds, w) = create_spatial_panel_data();
        for model in [SpatialPanelModel::Within, SpatialPanelModel::Random] {
            let cfg = spml_cfg(
                model,
                SpatialPanelEffect::Individual,
                true,
                SpatialErrorType::None,
            );
            let e = run_spml(
                &ds,
                "y",
                &["x1", "x2"],
                "entity",
                "time",
                &mut w.clone(),
                cfg,
            )
            .unwrap_err();
            let m = e.to_string();
            assert!(
                matches!(e, EconError::InvalidSpecification { .. }) && m.contains("fit y exactly"),
                "{model:?}: {m}"
            );
        }
    }

    #[test]
    fn test_spml_rejects_too_small_panels() {
        // T = 1
        let df = df! {
            "entity" => [1i64, 2, 3, 4, 5],
            "time" => [1i64, 1, 1, 1, 1],
            "y" => [1.1, 2.3, 2.9, 4.2, 5.1],
            "x1" => [0.4, 1.3, 1.7, 2.2, 3.1],
        }
        .unwrap();
        let coords: Vec<(f64, f64)> = (1..=5).map(|i| (i as f64, 0.1 * (i * i) as f64)).collect();
        let mut w =
            SpatialWeights::from_neighbors(&Neighbors::from_knn(&coords, 2), WeightStyle::RowStd);
        let cfg = spml_cfg(
            SpatialPanelModel::Within,
            SpatialPanelEffect::Individual,
            true,
            SpatialErrorType::None,
        );
        let e = run_spml(
            &Dataset::new(df),
            "y",
            &["x1"],
            "entity",
            "time",
            &mut w,
            cfg,
        )
        .unwrap_err();
        assert!(
            matches!(e, EconError::InsufficientData { required: 2, provided: 1, ref context } if context.contains("time periods")),
            "{e:?}"
        );
        // two-way effects on N = 4, T = 2 leave no residual degrees of freedom
        let x1 = [0.3, 1.2, 2.2, 2.9, 0.8, 1.9, 2.4, 3.6];
        let x2 = [1.0, 0.4, 0.9, 1.7, 0.2, 1.3, 0.5, 1.1];
        let noise = [0.1, -0.2, 0.15, 0.05, -0.1, 0.2, -0.05, 0.12];
        let y: Vec<f64> = (0..8)
            .map(|i| 0.5 * x1[i] - 0.3 * x2[i] + noise[i])
            .collect();
        let df = df! {
            "entity" => [0i64, 1, 2, 3, 0, 1, 2, 3],
            "time" => [0i64, 0, 0, 0, 1, 1, 1, 1],
            "y" => y,
            "x1" => x1,
            "x2" => x2,
        }
        .unwrap();
        let (_, mut w4) = create_spatial_panel_data();
        let cfg = spml_cfg(
            SpatialPanelModel::Within,
            SpatialPanelEffect::TwoWays,
            true,
            SpatialErrorType::None,
        );
        let e = run_spml(
            &Dataset::new(df),
            "y",
            &["x1", "x2"],
            "entity",
            "time",
            &mut w4,
            cfg,
        )
        .unwrap_err();
        assert!(
            matches!(e, EconError::InsufficientData { ref context, .. } if context.contains("residual degrees of freedom")),
            "{e:?}"
        );
    }

    #[test]
    fn test_spatial_panel_model_parses_aliases_and_rejects_unknown() {
        SpatialPanelModel::choices().validate().unwrap();
        for (s, v) in [
            ("within", SpatialPanelModel::Within),
            ("FE", SpatialPanelModel::Within),
            ("fixed effects", SpatialPanelModel::Within),
            ("random", SpatialPanelModel::Random),
            ("RE", SpatialPanelModel::Random),
            ("random-effects", SpatialPanelModel::Random),
            ("pooling", SpatialPanelModel::Pooling),
            ("pooled", SpatialPanelModel::Pooling),
            ("Pool", SpatialPanelModel::Pooling),
        ] {
            assert_eq!(s.parse::<SpatialPanelModel>().unwrap(), v, "{s}");
        }
        let m = "between"
            .parse::<SpatialPanelModel>()
            .unwrap_err()
            .to_string();
        for part in ["between", "within", "random", "pooling"] {
            assert!(m.contains(part), "{m}");
        }
    }

    #[test]
    fn test_spatial_panel_effect_parses_aliases_and_rejects_unknown() {
        SpatialPanelEffect::choices().validate().unwrap();
        for (s, v) in [
            ("individual", SpatialPanelEffect::Individual),
            ("ID", SpatialPanelEffect::Individual),
            ("entity", SpatialPanelEffect::Individual),
            ("time", SpatialPanelEffect::Time),
            ("Period", SpatialPanelEffect::Time),
            ("twoways", SpatialPanelEffect::TwoWays),
            ("two-ways", SpatialPanelEffect::TwoWays),
            ("two_way", SpatialPanelEffect::TwoWays),
            ("both", SpatialPanelEffect::TwoWays),
        ] {
            assert_eq!(s.parse::<SpatialPanelEffect>().unwrap(), v, "{s}");
        }
        assert!("nested".parse::<SpatialPanelEffect>().is_err());
    }

    #[test]
    fn test_spatial_error_type_parses_aliases_and_rejects_unknown() {
        SpatialErrorType::choices().validate().unwrap();
        for (s, v) in [
            ("none", SpatialErrorType::None),
            ("b", SpatialErrorType::Baltagi),
            ("Baltagi", SpatialErrorType::Baltagi),
            ("SEM", SpatialErrorType::Baltagi),
            ("kkp", SpatialErrorType::KKP),
            ("sem2", SpatialErrorType::KKP),
            ("Kapoor", SpatialErrorType::KKP),
        ] {
            assert_eq!(s.parse::<SpatialErrorType>().unwrap(), v, "{s}");
        }
        let m = "sac".parse::<SpatialErrorType>().unwrap_err().to_string();
        for part in ["none", "b", "kkp"] {
            assert!(m.contains(part), "{m}");
        }
    }

    #[test]
    fn test_panel_spectrum_log_det_uses_complex_eigenvalues() {
        let (_ds, w) = splm_sim();
        let s = PanelSpectrum::new(&w).unwrap();
        // R: 10 complex eigenvalues (5 pairs, |Im| >= 0.016) and a repeated real eigenvalue
        // that R reports with |Im| = 7.6e-17; the dense decomposition may resolve that
        // defective pair with an imaginary part near sqrt(eps), which moves log|I - rW| by
        // its square only.
        let mut im: Vec<f64> = s.im.iter().map(|v| v.abs()).collect();
        im.sort_by(f64::total_cmp);
        assert_eq!(im.iter().filter(|&&v| v > 1e-6).count(), 10, "{im:?}");
        assert!(im[im.len() - 11] < 1e-6, "{im:?}");
        let ld = s.log_det(0.43);
        assert!(
            Tol::new(1e-12, 1e-14).allows(ld, -1.020559182335464),
            "{ld}"
        );
        // the real-parts formula gives -1.02797700329204
        assert!((ld + 1.02797700329204).abs() > 1e-3);
        assert!(
            Tol::new(1e-12, 0.0).allows(s.lower, -2.160195419270105),
            "{}",
            s.lower
        );
        assert!(
            Tol::new(1e-12, 0.0).allows(s.upper, 0.9999989999999964),
            "{}",
            s.upper
        );
        let fd = (s.log_det(0.43 + 1e-6) - s.log_det(0.43 - 1e-6)) / 2e-6;
        assert!((s.d_log_det(0.43) - fd).abs() < 1e-7);
    }
}
