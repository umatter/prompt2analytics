//! Regression Standardization (G-computation) for causal effects estimation.
//!
//! This module implements the parametric g-formula for single time point interventions,
//! also known as regression standardization or g-computation. It estimates marginal
//! causal effects by fitting an outcome model and averaging predictions under different
//! treatment values over the covariate distribution.
//!
//! # Algorithm
//!
//! 1. **Fit outcome model**: E[Y|A,X] = m(A, X; beta)
//! 2. **Create counterfactual datasets**: Set A=1 for all, set A=0 for all
//! 3. **Predict outcomes under each scenario**: Y_hat(1), Y_hat(0)
//! 4. **Average predictions** to get E[Y(1)] and E[Y(0)]
//! 5. **Compute treatment effect**: ATE = E[Y(1)] - E[Y(0)]
//! 6. **Compute SE via bootstrap** (or delta method)
//!
//! # Outcome models
//!
//! The linear model is least squares (`regression::ols::solve_ols`: the normal equations,
//! the pivoted QR of R's `lm` when X'X is ill-conditioned); the logistic and Poisson models
//! are `econometrics::glm` (R's `glm.fit`: `mustart` start values, step halving, the relative
//! deviance rule with `StdRegConfig::tolerance` as `epsilon` and `max_iter` as `maxit`). The
//! fit's convergence is reported (`StdRegResult::convergence`, the "Converged:" line), with
//! glm's warnings and, when fitted probabilities are numerically 0 or 1, a separation
//! sentence: the maximum likelihood estimates then do not exist and the standardised
//! effects are not reliable.
//!
//! Aliased columns get a `NaN` coefficient and the warning "Coefficients not defined because
//! of singularities: ..." (as `lm`/`glm`); predictions use the identified columns. An aliased
//! treatment column, or a treatment interaction aliased while its covariate is not, leaves
//! the treatment effect unidentified and is an error.
//!
//! Standard errors: the bootstrap (percentile intervals with R's type-7 quantiles) is the
//! reference (each replicate refits coefficients only; see `fit_replicate`); the delta-method and sandwich SEs are approximations on the treatment
//! coefficient (CAUS-39, Tier 4), not stdReg's stacked estimating equations.
//!
//! # Estimands
//!
//! - **ATE (Average Treatment Effect)**: E[Y(1) - Y(0)]
//!   Effect averaged over the entire population.
//!
//! - **ATT (Average Treatment Effect on the Treated)**: E[Y(1) - Y(0) | A=1]
//!   Effect averaged over those who received treatment.
//!
//! - **ATC (Average Treatment Effect on the Controls)**: E[Y(1) - Y(0) | A=0]
//!   Effect averaged over those who did not receive treatment.
//!
//! # Mathematical Framework
//!
//! Under the assumptions of consistency, no unmeasured confounding (conditional
//! exchangeability), and positivity, the g-formula identifies the causal effect:
//!
//! ```text
//! E[Y(a)] = sum_x E[Y|A=a, X=x] * P(X=x)
//!         approx (1/n) * sum_i E[Y|A=a, X_i]
//! ```
//!
//! For ATE:
//! ```text
//! ATE = E[Y(1) - Y(0)]
//!     = (1/n) * sum_i [E[Y|A=1, X_i] - E[Y|A=0, X_i]]
//!     = (1/n) * sum_i [Y_hat_i(1) - Y_hat_i(0)]
//! ```
//!
//! For ATT:
//! ```text
//! ATT = E[Y(1) - Y(0) | A=1]
//!     = (1/n1) * sum_{i:A_i=1} [Y_hat_i(1) - Y_hat_i(0)]
//! ```
//!
//! # References
//!
//! - Robins, J.M. (1986). "A new approach to causal inference in mortality studies with
//!   a sustained exposure period -- application to control of the healthy worker survivor
//!   effect." *Mathematical Modelling*, 7(9-12), 1393-1512.
//!   https://doi.org/10.1016/0270-0255(86)90088-6
//!
//! - Snowden, J.M., Rose, S., & Mortimer, K.M. (2011). "Implementation of G-computation
//!   on a simulated dataset: Demonstration of a causal inference technique."
//!   *American Journal of Epidemiology*, 173(7), 731-738.
//!   https://doi.org/10.1093/aje/kwq472
//!
//! - Hernan, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
//!   https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
//!
//! - R package `stdReg` (Sjolander, 2016):
//!   https://cran.r-project.org/package=stdReg
//!
//! - R package `margins` (Leeper, 2021):
//!   https://cran.r-project.org/package=margins
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::econometrics::{run_stdreg, StdRegConfig, StdRegModel, StdRegEstimand};
//!
//! let config = StdRegConfig {
//!     model_type: StdRegModel::Linear,
//!     estimand: StdRegEstimand::ATE,
//!     se_method: SEMethod::Bootstrap { n: 999 },
//!     seed: Some(42),
//!     ..Default::default()
//! };
//!
//! let result = run_stdreg(&dataset, "outcome", "treatment", &["x1", "x2"], config)?;
//! println!("ATE: {:.4} (SE: {:.4})", result.ate, result.se);
//! println!("95% CI: [{:.4}, {:.4}]", result.ci_lower, result.ci_upper);
//! println!("E[Y(1)]: {:.4}, E[Y(0)]: {:.4}", result.ey1, result.ey0);
//! ```

use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
use crate::econometrics::glm::{GlmSpec, fit_glm, plogis};
use crate::econometrics::tmle::{GlmControl, LeanStart, check_binary_treatment, lean_logit};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{xtx, xty};
use crate::linalg::solve::{SingularPolicy, aliasing_message, sym_inverse};
use crate::options::Choices;
use crate::regression::ols::{normal_equations_inverse, solve_ols};
use crate::resource;
use crate::stats::robust::percentile_interval;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::{
    SignificanceLevel, normal_p_value_two_sided, validate_level, z_critical,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Outcome model type for regression standardization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StdRegModel {
    /// Linear regression for continuous outcomes (default)
    #[default]
    Linear,
    /// Logistic regression for binary outcomes
    Logistic,
    /// Poisson regression for count outcomes
    Poisson,
}

impl fmt::Display for StdRegModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StdRegModel::Linear => write!(f, "Linear (OLS)"),
            StdRegModel::Logistic => write!(f, "Logistic (MLE)"),
            StdRegModel::Poisson => write!(f, "Poisson (MLE)"),
        }
    }
}

/// Target estimand for regression standardization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StdRegEstimand {
    /// Average Treatment Effect (ATE): E[Y(1) - Y(0)]
    /// Effect averaged over the entire population.
    #[default]
    ATE,
    /// Average Treatment Effect on the Treated (ATT): E[Y(1) - Y(0) | A=1]
    /// Effect averaged over those who received treatment.
    ATT,
    /// Average Treatment Effect on the Controls (ATC): E[Y(1) - Y(0) | A=0]
    /// Effect averaged over those who did not receive treatment.
    ATC,
    /// Return E[Y(1)] and E[Y(0)] separately (potential outcomes)
    Levels,
}

impl fmt::Display for StdRegEstimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StdRegEstimand::ATE => write!(f, "ATE (Average Treatment Effect)"),
            StdRegEstimand::ATT => write!(f, "ATT (Average Treatment Effect on Treated)"),
            StdRegEstimand::ATC => write!(f, "ATC (Average Treatment Effect on Controls)"),
            StdRegEstimand::Levels => write!(f, "Levels (E[Y(1)] and E[Y(0)])"),
        }
    }
}

/// Method for computing standard errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SEMethod {
    /// Bootstrap standard errors (default, recommended)
    #[default]
    Bootstrap,
    /// Delta method (analytical, faster but assumes normality)
    Delta,
    /// Sandwich (robust) standard errors
    Sandwich,
}

impl fmt::Display for SEMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SEMethod::Bootstrap => write!(f, "Bootstrap"),
            SEMethod::Delta => write!(f, "Delta Method"),
            SEMethod::Sandwich => write!(f, "Sandwich (Robust)"),
        }
    }
}

crate::option_enum!(
    StdRegModel,
    Choices {
        option: "model_type",
        values: &[
            (
                &["linear", "ols", "gaussian", "continuous"],
                StdRegModel::Linear
            ),
            (
                &["logistic", "logit", "binomial", "binary"],
                StdRegModel::Logistic
            ),
            (
                &["poisson", "count", "counts", "log_linear"],
                StdRegModel::Poisson
            ),
        ],
    },
    no_display
);

crate::option_enum!(
    StdRegEstimand,
    Choices {
        option: "estimand",
        values: &[
            (&["ate"], StdRegEstimand::ATE),
            (&["att"], StdRegEstimand::ATT),
            (&["atc"], StdRegEstimand::ATC),
            (&["levels", "potential_outcomes"], StdRegEstimand::Levels),
        ],
    },
    no_display
);

crate::option_enum!(
    SEMethod,
    Choices {
        option: "se_method",
        values: &[
            (&["bootstrap", "boot"], SEMethod::Bootstrap),
            (&["delta", "delta_method"], SEMethod::Delta),
            (&["sandwich", "robust"], SEMethod::Sandwich),
        ],
    },
    no_display
);

/// Configuration for regression standardization.
#[derive(Debug, Clone)]
pub struct StdRegConfig {
    /// Outcome model type (Linear, Logistic, or Poisson)
    pub model_type: StdRegModel,
    /// Target estimand (ATE, ATT, ATC, or Levels)
    pub estimand: StdRegEstimand,
    /// Method for computing standard errors
    pub se_method: SEMethod,
    /// Number of bootstrap replications (if using bootstrap SE). Must be at least 2
    /// for bootstrap standard errors; a count whose result buffers cannot be allocated
    /// is refused with `EconError::ResourceLimit`.
    pub n_bootstrap: usize,
    /// Confidence level for intervals, in (0, 1) (default: 0.95)
    pub confidence_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// `glm.fit`'s `maxit` for the logistic and Poisson fits; at least 1 (default 100)
    pub max_iter: usize,
    /// `glm.fit`'s `epsilon` (relative change in deviance) for the logistic and Poisson
    /// fits; finite and > 0 (default 1e-8, R's `glm.control` default)
    pub tolerance: f64,
    /// Whether to include treatment-covariate interactions in outcome model
    pub include_interactions: bool,
}

impl Default for StdRegConfig {
    fn default() -> Self {
        Self {
            model_type: StdRegModel::Linear,
            estimand: StdRegEstimand::ATE,
            se_method: SEMethod::Bootstrap,
            n_bootstrap: 999,
            confidence_level: 0.95,
            seed: None,
            max_iter: 100,
            tolerance: 1e-8,
            include_interactions: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Effect estimate for a specific subgroup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgroupEffect {
    /// Identifier for the subgroup
    pub subgroup_name: String,
    /// Value defining the subgroup
    pub subgroup_value: f64,
    /// Treatment effect estimate
    pub effect: f64,
    /// Standard error
    pub se: f64,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
    /// Number of observations in this subgroup
    pub n: usize,
}

/// Result from regression standardization (g-computation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdRegResult {
    /// Target estimand
    pub estimand: StdRegEstimand,
    /// Outcome model type used
    pub model_type: StdRegModel,
    /// SE method used
    pub se_method: SEMethod,

    // ═══════════════════════════════════════════════════════════════════════
    // Treatment Effect
    // ═══════════════════════════════════════════════════════════════════════
    /// Average treatment effect (or appropriate estimand)
    pub ate: f64,
    /// Standard error
    pub se: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Confidence level used (e.g., 0.95)
    pub confidence_level: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Potential Outcomes
    // ═══════════════════════════════════════════════════════════════════════
    /// E[Y(1)] - Expected outcome under treatment
    pub ey1: f64,
    /// E[Y(0)] - Expected outcome under control
    pub ey0: f64,
    /// SE of E[Y(1)]
    pub ey1_se: f64,
    /// SE of E[Y(0)]
    pub ey0_se: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Effect Measures (for binary outcomes)
    // ═══════════════════════════════════════════════════════════════════════
    /// Risk Ratio: E[Y(1)] / E[Y(0)] (for binary outcomes)
    pub risk_ratio: Option<f64>,
    /// Risk Ratio confidence interval
    pub risk_ratio_ci: Option<(f64, f64)>,
    /// Odds Ratio: [E[Y(1)]/(1-E[Y(1)])] / [E[Y(0)]/(1-E[Y(0)])]
    pub odds_ratio: Option<f64>,
    /// Odds Ratio confidence interval
    pub odds_ratio_ci: Option<(f64, f64)>,
    /// Number Needed to Treat: 1 / ATE (for binary outcomes)
    pub nnt: Option<f64>,

    // ═══════════════════════════════════════════════════════════════════════
    // Inference
    // ═══════════════════════════════════════════════════════════════════════
    /// Z-statistic (effect / se)
    pub z_stat: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,

    // ═══════════════════════════════════════════════════════════════════════
    // Sample Info
    // ═══════════════════════════════════════════════════════════════════════
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Number of bootstrap replications used
    pub n_bootstrap: usize,

    // ═══════════════════════════════════════════════════════════════════════
    // Model Fit
    // ═══════════════════════════════════════════════════════════════════════
    /// R-squared (for linear model) or pseudo-R-squared (for GLM)
    pub r_squared: Option<f64>,
    /// Outcome model coefficients (`NaN` for aliased columns, `null` in JSON)
    #[serde(with = "nan_vec")]
    pub outcome_coefs: Vec<f64>,
    /// Outcome model coefficient names
    pub outcome_coef_names: Vec<String>,

    // ═══════════════════════════════════════════════════════════════════════
    // Subgroup Effects (optional)
    // ═══════════════════════════════════════════════════════════════════════
    /// Effects by subgroup (if requested)
    pub subgroup_effects: Option<Vec<SubgroupEffect>>,

    // ═══════════════════════════════════════════════════════════════════════
    // Individual-Level Predictions (optional, not serialized)
    // ═══════════════════════════════════════════════════════════════════════
    /// Y_hat(1) for each observation
    #[serde(skip)]
    pub y_hat_1: Vec<f64>,
    /// Y_hat(0) for each observation
    #[serde(skip)]
    pub y_hat_0: Vec<f64>,
    /// Individual treatment effects: Y_hat(1) - Y_hat(0)
    #[serde(skip)]
    pub individual_effects: Vec<f64>,

    /// Warnings generated during estimation
    pub warnings: Vec<String>,

    /// Convergence of the outcome model's IRLS (logistic and Poisson); `None` for the linear
    /// model.
    #[serde(default)]
    pub convergence: Option<Convergence>,
}

impl ResultNotes for StdRegResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

/// `Vec<f64>` as a JSON array with `NaN` as `null`.
mod nan_vec {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(v.iter().map(|x| if x.is_nan() { None } else { Some(*x) }))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        let v = Vec::<Option<f64>>::deserialize(d)?;
        Ok(v.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect())
    }
}

impl fmt::Display for StdRegResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Regression Standardization (G-Computation)")?;
        writeln!(f, "===========================================")?;
        writeln!(f)?;
        writeln!(f, "Specification:")?;
        writeln!(f, "  Outcome Model:  {}", self.model_type)?;
        writeln!(f, "  Estimand:       {}", self.estimand)?;
        if self.se_method == SEMethod::Bootstrap {
            writeln!(
                f,
                "  SE Method:      {} (n = {})",
                self.se_method, self.n_bootstrap
            )?;
        } else {
            writeln!(f, "  SE Method:      {}", self.se_method)?;
        }
        writeln!(f)?;
        writeln!(f, "Treatment Effect:")?;
        writeln!(f, "  Effect:     {:>12.4}", self.ate)?;
        writeln!(f, "  Std. Error: {:>12.4}", self.se)?;
        writeln!(f, "  z-stat:     {:>12.2}", self.z_stat)?;
        writeln!(
            f,
            "  p-value:    {:>12.4}{}",
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(
            f,
            "  {:.0}% CI:     [{:.4}, {:.4}]",
            self.confidence_level * 100.0,
            self.ci_lower,
            self.ci_upper
        )?;
        writeln!(f)?;
        writeln!(f, "Potential Outcomes:")?;
        writeln!(
            f,
            "  E[Y(1)]:    {:>12.4} (SE: {:.4})",
            self.ey1, self.ey1_se
        )?;
        writeln!(
            f,
            "  E[Y(0)]:    {:>12.4} (SE: {:.4})",
            self.ey0, self.ey0_se
        )?;
        writeln!(f)?;

        // Additional measures for binary outcomes
        if let Some(rr) = self.risk_ratio {
            writeln!(f, "Additional Measures (Binary Outcome):")?;
            writeln!(f, "  Risk Ratio:     {:>8.4}", rr)?;
            if let Some((lo, hi)) = self.risk_ratio_ci {
                writeln!(f, "                  [{:.4}, {:.4}]", lo, hi)?;
            }
            if let Some(or) = self.odds_ratio {
                writeln!(f, "  Odds Ratio:     {:>8.4}", or)?;
                if let Some((lo, hi)) = self.odds_ratio_ci {
                    writeln!(f, "                  [{:.4}, {:.4}]", lo, hi)?;
                }
            }
            if let Some(nnt) = self.nnt {
                if nnt.is_finite() && nnt > 0.0 {
                    writeln!(f, "  NNT:            {:>8.1}", nnt)?;
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  Observations:  {} (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(f)?;

        if let Some(r2) = self.r_squared {
            writeln!(f, "Model Fit:")?;
            writeln!(f, "  R-squared:     {:.4}", r2)?;
            writeln!(f)?;
        }

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        let notes = notes_text(self);
        if !notes.is_empty() {
            writeln!(f)?;
            write!(f, "{notes}")?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run regression standardization (G-computation) for causal effect estimation.
///
/// Estimates treatment effects by fitting an outcome model and averaging predictions
/// under different treatment values over the covariate distribution.
///
/// # Arguments
///
/// * `dataset` - Dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable column
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of covariate columns
/// * `config` - Configuration options
///
/// # Returns
///
/// `StdRegResult` containing the treatment effect estimate, standard error,
/// confidence interval, potential outcomes, and diagnostics.
///
/// # Algorithm
///
/// 1. Fit outcome model: Y ~ A + X (optionally with A*X interactions)
/// 2. For each observation i, predict:
///    - Y_hat_i(1) = E[Y | A=1, X_i]
///    - Y_hat_i(0) = E[Y | A=0, X_i]
/// 3. Compute estimand:
///    - ATE = (1/n) * sum_i [Y_hat_i(1) - Y_hat_i(0)]
///    - ATT = (1/n1) * sum_{i:A_i=1} [Y_hat_i(1) - Y_hat_i(0)]
///    - ATC = (1/n0) * sum_{i:A_i=0} [Y_hat_i(1) - Y_hat_i(0)]
/// 4. Compute SE via bootstrap or delta method
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::{run_stdreg, StdRegConfig, StdRegModel};
///
/// let config = StdRegConfig {
///     model_type: StdRegModel::Linear,
///     ..Default::default()
/// };
///
/// let result = run_stdreg(&dataset, "outcome", "treatment", &["age", "sex"], config)?;
/// println!("ATE: {:.4} (95% CI: [{:.4}, {:.4}])",
///          result.ate, result.ci_lower, result.ci_upper);
/// ```
///
/// # References
///
/// - Robins (1986) for the g-computation formula
/// - Snowden, Rose & Mortimer (2011) for implementation guidance
/// - Hernan & Robins (2020), "Causal Inference: What If", Chapter 13
pub fn run_stdreg(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: StdRegConfig,
) -> EconResult<StdRegResult> {
    let mut warnings = Vec::new();

    validate_stdreg_config(&config)?;

    // ═══════════════════════════════════════════════════════════════════════
    // Extract Data
    // ═══════════════════════════════════════════════════════════════════════

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    // Extract outcome variable Y
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    // Extract treatment variable A
    let a = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    let n = y.len();

    // The treatment must be coded 0/1: the counterfactual designs set it to 1 and 0.
    let (n_treated, n_control) = check_binary_treatment(a.view(), treatment_col)?;

    // A logistic outcome model needs Y in [0, 1] (glm's binomial family).
    if config.model_type == StdRegModel::Logistic {
        let lo = y.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo < 0.0 || hi > 1.0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "model_type logistic needs an outcome in [0, 1]; '{outcome_col}' ranges from {lo} to {hi}"
                ),
            });
        }
    }
    // A Poisson outcome model needs a non-negative Y (glm's poisson family).
    if config.model_type == StdRegModel::Poisson {
        let lo = y.iter().copied().fold(f64::INFINITY, f64::min);
        if lo < 0.0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "model_type poisson needs a non-negative outcome; '{outcome_col}' has minimum {lo}"
                ),
            });
        }
    }

    // Build design matrix W for covariates (with intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, true)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let w = design.data;
    let k_w = w.ncols();

    // Full design matrix: [intercept, X, A] (or with A:X interactions)
    let (x_full, coef_names) = build_design_matrix(
        &w,
        &a,
        covariate_cols,
        treatment_col,
        config.include_interactions,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // Fit Outcome Model
    // ═══════════════════════════════════════════════════════════════════════

    let fit = fit_outcome_model(
        &x_full,
        &y,
        config.model_type,
        config.max_iter,
        config.tolerance,
        None,
    )?;
    check_treatment_identified(
        &fit.aliased,
        &coef_names,
        k_w,
        config.include_interactions,
        treatment_col,
    )?;
    warnings.extend(fit.warnings.iter().map(|w| format!("outcome model: {w}")));
    if let Some(msg) = aliasing_message(&fit.aliased, &coef_names) {
        warnings.push(format!("outcome model: {msg}"));
    }
    if fit.separation {
        warnings.push(format!(
            "Separation: fitted probabilities are numerically 0 or 1 for {} of {n} observations; \
             the covariates (almost) perfectly predict '{outcome_col}', so the outcome model's \
             maximum likelihood estimates do not exist and the standardised effects and their \
             standard errors are not reliable",
            fit.n_extreme
        ));
    }
    let convergence = fit.glm.then(|| {
        Convergence::new(
            fit.converged,
            fit.iterations,
            config.max_iter,
            "relative change in deviance (glm.fit)",
            f64::NAN,
            config.tolerance,
        )
    });
    if let Some(note) = convergence
        .as_ref()
        .and_then(|c| c.note("stdreg outcome model (glm)"))
    {
        warnings.push(note);
    }
    let beta = fit.coefficients.clone();
    let y_hat = fit.fitted.clone();
    let r_squared = fit.r_squared;

    // ═══════════════════════════════════════════════════════════════════════
    // Compute Counterfactual Predictions
    // ═══════════════════════════════════════════════════════════════════════

    // X(1): design matrix with A=1 for all; X(0): with A=0 for all
    let (x_1, x_0) = create_counterfactual_matrices(&w, n, k_w, config.include_interactions);
    let y_hat_1: Array1<f64> = predict(&x_1, &beta, config.model_type);
    let y_hat_0: Array1<f64> = predict(&x_0, &beta, config.model_type);

    // ═══════════════════════════════════════════════════════════════════════
    // Compute Point Estimates
    // ═══════════════════════════════════════════════════════════════════════

    let (ate, ey1, ey0) = compute_estimand(&y_hat_1, &y_hat_0, &a, config.estimand);

    // Individual treatment effects
    let individual_effects: Vec<f64> = y_hat_1
        .iter()
        .zip(y_hat_0.iter())
        .map(|(&y1, &y0)| y1 - y0)
        .collect();

    // ═══════════════════════════════════════════════════════════════════════
    // Compute Standard Errors
    // ═══════════════════════════════════════════════════════════════════════

    // the identified columns (the delta and sandwich breads)
    let identified: Vec<usize> = (0..x_full.ncols())
        .filter(|j| !fit.aliased.contains(j))
        .collect();
    let x_id = x_full.select(Axis(1), &identified);

    let (se, ey1_se, ey0_se, boot) = match config.se_method {
        SEMethod::Bootstrap => {
            let boot = compute_bootstrap_se(
                &y,
                &a,
                &w,
                k_w,
                config.model_type,
                config.estimand,
                config.n_bootstrap,
                config.seed,
                config.include_interactions,
                config.max_iter,
                config.tolerance,
                fit.glm
                    .then(|| beta.mapv(|v| if v.is_nan() { 0.0 } else { v })),
            )?;
            let b = config.n_bootstrap;
            if boot.not_converged > 0 {
                warnings.push(format!(
                    "outcome model did not converge in {} of {b} bootstrap replicates",
                    boot.not_converged
                ));
            }
            if boot.separated > 0 {
                warnings.push(format!(
                    "fitted probabilities numerically 0 or 1 in {} of {b} bootstrap replicates",
                    boot.separated
                ));
            }
            if boot.failed > 0 {
                warnings.push(format!(
                    "{} of {b} bootstrap replicates failed and were skipped",
                    boot.failed
                ));
            }
            (boot.se, boot.ey1_se, boot.ey0_se, Some(boot))
        }
        SEMethod::Delta => {
            let (se, s1, s0) = compute_delta_se(&x_id, &y, &y_hat, config.model_type)?;
            (se, s1, s0, None)
        }
        SEMethod::Sandwich => {
            let (se, s1, s0) = compute_sandwich_se(&x_id, &y, &y_hat)?;
            (se, s1, s0, None)
        }
    };
    let (boot_ates, boot_ey1s, boot_ey0s) = match &boot {
        Some(b) => (b.ates.as_slice(), b.ey1s.as_slice(), b.ey0s.as_slice()),
        None => (&[][..], &[][..], &[][..]),
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Compute Confidence Intervals
    // ═══════════════════════════════════════════════════════════════════════

    let (ci_lower, ci_upper) = if config.se_method == SEMethod::Bootstrap && boot_ates.len() >= 20 {
        // Percentile bootstrap CI (R's type-7 quantiles)
        bootstrap_ci(boot_ates, config.confidence_level)?
    } else {
        // Normal approximation CI
        let z_crit = z_critical(config.confidence_level)?;
        (ate - z_crit * se, ate + z_crit * se)
    };

    // Inference
    let z_stat = if se > 0.0 && se.is_finite() {
        ate / se
    } else {
        0.0
    };
    let p_value = normal_p_value_two_sided(z_stat);
    let significance = SignificanceLevel::from_p_value(p_value);

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Effect Measures (Binary Outcomes)
    // ═══════════════════════════════════════════════════════════════════════

    let (risk_ratio, risk_ratio_ci, odds_ratio, odds_ratio_ci, nnt) =
        if config.model_type == StdRegModel::Logistic || is_binary_outcome(&y) {
            compute_binary_effect_measures(ey1, ey0, boot_ey1s, boot_ey0s, config.confidence_level)?
        } else {
            (None, None, None, None, None)
        };

    // ═══════════════════════════════════════════════════════════════════════
    // Construct Result
    // ═══════════════════════════════════════════════════════════════════════

    Ok(StdRegResult {
        estimand: config.estimand,
        model_type: config.model_type,
        se_method: config.se_method,
        ate,
        se,
        ci_lower,
        ci_upper,
        confidence_level: config.confidence_level,
        ey1,
        ey0,
        ey1_se,
        ey0_se,
        risk_ratio,
        risk_ratio_ci,
        odds_ratio,
        odds_ratio_ci,
        nnt,
        z_stat,
        p_value,
        significance,
        n_obs: n,
        n_treated,
        n_control,
        n_bootstrap: config.n_bootstrap,
        r_squared: Some(r_squared),
        outcome_coefs: beta.to_vec(),
        outcome_coef_names: coef_names,
        subgroup_effects: None,
        y_hat_1: y_hat_1.to_vec(),
        y_hat_0: y_hat_0.to_vec(),
        individual_effects,
        warnings,
        convergence,
    })
}

/// Run regression standardization with default configuration.
///
/// Uses linear outcome model, ATE estimand, and bootstrap standard errors.
pub fn stdreg(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
) -> EconResult<StdRegResult> {
    run_stdreg(
        dataset,
        outcome_col,
        treatment_col,
        covariate_cols,
        StdRegConfig::default(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the full design matrix [1, X, A] or [1, X, A, A*X].
fn build_design_matrix(
    w: &Array2<f64>,
    a: &Array1<f64>,
    covariate_names: &[&str],
    treatment_name: &str,
    include_interactions: bool,
) -> (Array2<f64>, Vec<String>) {
    let n = w.nrows();
    let k_w = w.ncols(); // includes intercept

    let k = if include_interactions {
        k_w + 1 + (k_w - 1) // intercept (in w) + covariates (in w) + A + A*X interactions
    } else {
        k_w + 1 // intercept (in w) + covariates (in w) + A
    };

    let mut x = Array2::zeros((n, k));

    // Copy W (intercept + covariates)
    for i in 0..n {
        for j in 0..k_w {
            x[[i, j]] = w[[i, j]];
        }
    }

    // Add treatment
    let a_col = k_w;
    for i in 0..n {
        x[[i, a_col]] = a[i];
    }

    // Build coefficient names
    let mut names = vec!["(Intercept)".to_string()];
    for name in covariate_names {
        names.push(name.to_string());
    }
    names.push(treatment_name.to_string());

    // Add interactions if requested
    if include_interactions {
        let mut col = a_col + 1;
        for j in 1..k_w {
            // Skip intercept (j=0)
            for i in 0..n {
                x[[i, col]] = a[i] * w[[i, j]];
            }
            names.push(format!("{treatment_name}:{}", covariate_names[j - 1]));
            col += 1;
        }
    }

    (x, names)
}

/// Create counterfactual design matrices X(1) and X(0).
fn create_counterfactual_matrices(
    w: &Array2<f64>,
    n: usize,
    k_w: usize,
    include_interactions: bool,
) -> (Array2<f64>, Array2<f64>) {
    let k = if include_interactions {
        k_w + 1 + (k_w - 1)
    } else {
        k_w + 1
    };

    let mut x_1 = Array2::zeros((n, k));
    let mut x_0 = Array2::zeros((n, k));

    // Copy W (intercept + covariates)
    for i in 0..n {
        for j in 0..k_w {
            x_1[[i, j]] = w[[i, j]];
            x_0[[i, j]] = w[[i, j]];
        }
    }

    // Set treatment: A=1 for x_1, A=0 for x_0
    let a_col = k_w;
    for i in 0..n {
        x_1[[i, a_col]] = 1.0;
        x_0[[i, a_col]] = 0.0;
    }

    // Add interactions if needed
    if include_interactions {
        let mut col = a_col + 1;
        for j in 1..k_w {
            for i in 0..n {
                x_1[[i, col]] = 1.0 * w[[i, j]]; // A=1
                x_0[[i, col]] = 0.0 * w[[i, j]]; // A=0 (all zeros)
            }
            col += 1;
        }
    }

    (x_1, x_0)
}

/// A fitted outcome model.
struct OutcomeFit {
    /// `NaN` for aliased columns.
    coefficients: Array1<f64>,
    fitted: Array1<f64>,
    /// `1 - SSE / SST` (lm; 0 when SST is 0) or `1 - deviance / null deviance` (glm).
    r_squared: f64,
    converged: bool,
    iterations: usize,
    /// Binomial: some fitted probability within `10 eps` of 0 or 1.
    separation: bool,
    /// Logistic: the number of fitted probabilities within `10 eps` of 0 or 1.
    n_extreme: usize,
    /// Indices of the aliased columns.
    aliased: Vec<usize>,
    /// glm's warnings except "glm.fit: algorithm did not converge" (carried by `converged`).
    warnings: Vec<String>,
    /// Logistic or Poisson (an IRLS fit).
    glm: bool,
}

/// The outcome model: least squares (`solve_ols`) or `econometrics::glm` with `tolerance`
/// as glm.fit's `epsilon` and `max_iter` as its `maxit`. A GLM starts IRLS from `start` when
/// given, from glm.fit's `mustart` otherwise. Bootstrap replicates start from the full-sample
/// fit: a replicate that converges reaches the same maximum-likelihood estimates as from
/// `mustart`, in fewer iterations; a separated or non-converged replicate (counted in the
/// warnings) can stop at a different point along the diverging direction than a cold start
/// would.
fn fit_outcome_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model_type: StdRegModel,
    max_iter: usize,
    tolerance: f64,
    start: Option<&Array1<f64>>,
) -> EconResult<OutcomeFit> {
    resource::checkpoint()?;
    match model_type {
        StdRegModel::Linear => {
            let s = solve_ols(x.view(), y.view())?;
            let fitted = y - &s.residuals;
            let y_mean = y.mean().unwrap_or(0.0);
            let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
            let sse: f64 = s.residuals.iter().map(|e| e * e).sum();
            Ok(OutcomeFit {
                coefficients: s.beta,
                fitted,
                r_squared: if sst > 0.0 { 1.0 - sse / sst } else { 0.0 },
                converged: true,
                iterations: 1,
                separation: false,
                n_extreme: 0,
                aliased: s.aliased,
                warnings: Vec::new(),
                glm: false,
            })
        }
        StdRegModel::Logistic | StdRegModel::Poisson => {
            let logistic = model_type == StdRegModel::Logistic;
            let mut spec = if logistic {
                GlmSpec::logit()
            } else {
                GlmSpec::poisson()
            };
            spec.epsilon = tolerance;
            spec.max_iter = max_iter;
            spec.start = start.map(|b| b.view());
            let fit = fit_glm(x.view(), y.view(), &spec)?;
            let eps10 = 10.0 * f64::EPSILON;
            let n_extreme = if logistic {
                fit.fitted
                    .iter()
                    .filter(|&&m| m < eps10 || m > 1.0 - eps10)
                    .count()
            } else {
                0
            };
            let r_squared = if fit.null_deviance > 0.0 {
                1.0 - fit.deviance / fit.null_deviance
            } else {
                0.0
            };
            Ok(OutcomeFit {
                warnings: fit
                    .warnings
                    .iter()
                    .filter(|w| w.as_str() != "glm.fit: algorithm did not converge")
                    .cloned()
                    .collect(),
                coefficients: fit.coefficients,
                fitted: fit.fitted,
                r_squared,
                converged: fit.converged,
                iterations: fit.iterations,
                separation: fit.separation,
                n_extreme,
                aliased: fit.aliased,
                glm: true,
            })
        }
    }
}

/// The treatment effect is not identified when the treatment column is aliased, or when an
/// interaction `A:W_j` is aliased while `W_j` is not.
fn check_treatment_identified(
    aliased: &[usize],
    names: &[String],
    k_w: usize,
    include_interactions: bool,
    treatment: &str,
) -> EconResult<()> {
    let interaction_aliased = include_interactions
        && (1..k_w).any(|j| aliased.contains(&(k_w + j)) && !aliased.contains(&j));
    if aliased.contains(&k_w) || interaction_aliased {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the treatment '{treatment}' is collinear with the covariates in the outcome \
                 model ({}); the treatment effect is not identified",
                aliasing_message(aliased, names).unwrap_or_default()
            ),
        });
    }
    Ok(())
}

/// The outcome model's inverse link at the linear predictor `z`.
fn inverse_link(z: f64, model_type: StdRegModel) -> f64 {
    match model_type {
        StdRegModel::Linear => z,
        StdRegModel::Logistic => plogis(z),
        // R's poisson()$linkinv: pmax(exp(eta), .Machine$double.eps)
        StdRegModel::Poisson => z.exp().max(f64::EPSILON),
    }
}

/// Predicted means; aliased (`NaN`) coefficients count as 0.
fn predict(x: &Array2<f64>, beta: &Array1<f64>, model_type: StdRegModel) -> Array1<f64> {
    let b = beta.mapv(|v| if v.is_nan() { 0.0 } else { v });
    let z: Array1<f64> = x.dot(&b);
    z.mapv(|zi| inverse_link(zi, model_type))
}

/// Compute the target estimand from counterfactual predictions.
fn compute_estimand(
    y_hat_1: &Array1<f64>,
    y_hat_0: &Array1<f64>,
    a: &Array1<f64>,
    estimand: StdRegEstimand,
) -> (f64, f64, f64) {
    let n = y_hat_1.len();

    match estimand {
        StdRegEstimand::ATE | StdRegEstimand::Levels => {
            // Average over all observations
            let ey1: f64 = y_hat_1.iter().sum::<f64>() / n as f64;
            let ey0: f64 = y_hat_0.iter().sum::<f64>() / n as f64;
            let ate = ey1 - ey0;
            (ate, ey1, ey0)
        }
        StdRegEstimand::ATT => {
            // Average over treated observations
            let n1 = a.iter().filter(|&&ai| ai >= 0.5).count();
            let (sum_y1, sum_y0): (f64, f64) = a
                .iter()
                .zip(y_hat_1.iter())
                .zip(y_hat_0.iter())
                .filter(|((ai, _), _)| **ai >= 0.5)
                .map(|((_, y1), y0)| (*y1, *y0))
                .fold((0.0, 0.0), |(s1, s0), (y1, y0)| (s1 + y1, s0 + y0));
            let ey1 = sum_y1 / n1 as f64;
            let ey0 = sum_y0 / n1 as f64;
            let att = ey1 - ey0;
            (att, ey1, ey0)
        }
        StdRegEstimand::ATC => {
            // Average over control observations
            let n0 = a.iter().filter(|&&ai| ai < 0.5).count();
            let (sum_y1, sum_y0): (f64, f64) = a
                .iter()
                .zip(y_hat_1.iter())
                .zip(y_hat_0.iter())
                .filter(|((ai, _), _)| **ai < 0.5)
                .map(|((_, y1), y0)| (*y1, *y0))
                .fold((0.0, 0.0), |(s1, s0), (y1, y0)| (s1 + y1, s0 + y0));
            let ey1 = sum_y1 / n0 as f64;
            let ey0 = sum_y0 / n0 as f64;
            let atc = ey1 - ey0;
            (atc, ey1, ey0)
        }
    }
}

/// Checks the replication count, the iteration controls and the confidence level before any
/// work.
fn validate_stdreg_config(config: &StdRegConfig) -> EconResult<()> {
    validate_level(config.confidence_level, "confidence_level")?;
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1; got 0".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("tolerance must be finite and > 0; got {}", config.tolerance),
        });
    }
    if config.se_method == SEMethod::Bootstrap && config.n_bootstrap < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "n_bootstrap must be at least 2 for bootstrap standard errors; got {}",
                config.n_bootstrap
            ),
        });
    }
    Ok(())
}

/// Bootstrap draws and the replicates' fit diagnostics.
struct BootDraws {
    se: f64,
    ey1_se: f64,
    ey0_se: f64,
    ates: Vec<f64>,
    ey1s: Vec<f64>,
    ey0s: Vec<f64>,
    not_converged: usize,
    separated: usize,
    failed: usize,
}

/// The sample standard deviation (`n - 1`), 0 for a single draw.
fn draw_sd(draws: &[f64]) -> f64 {
    let m = draws.iter().sum::<f64>() / draws.len() as f64;
    let ss: f64 = draws.iter().map(|&e| (e - m).powi(2)).sum();
    (ss / (draws.len() - 1).max(1) as f64).sqrt()
}

/// Compute bootstrap standard errors: every replicate refits the outcome model
/// ([`fit_replicate`]; a GLM from `start`, the full-sample coefficients with aliased ones set
/// to 0) on a design buffer that the replicates share, and standardises over the resample
/// ([`resample_estimand`]).
#[allow(clippy::too_many_arguments)]
fn compute_bootstrap_se(
    y: &Array1<f64>,
    a: &Array1<f64>,
    w: &Array2<f64>,
    k_w: usize,
    model_type: StdRegModel,
    estimand: StdRegEstimand,
    n_bootstrap: usize,
    seed: Option<u64>,
    include_interactions: bool,
    max_iter: usize,
    tolerance: f64,
    start: Option<Array1<f64>>,
) -> EconResult<BootDraws> {
    let n = y.len();

    let mut rng: rand::rngs::StdRng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let what = "bootstrap replications (n_bootstrap)";
    let mut boot_ates = resource::try_with_capacity::<f64>(n_bootstrap, what)?;
    let mut boot_ey1s = resource::try_with_capacity::<f64>(n_bootstrap, what)?;
    let mut boot_ey0s = resource::try_with_capacity::<f64>(n_bootstrap, what)?;
    let (mut not_converged, mut separated, mut failed) = (0usize, 0usize, 0usize);
    let names: Vec<String> = {
        let mut v = vec!["(Intercept)".to_string()];
        v.extend((1..k_w).map(|j| format!("x{j}")));
        v.push("a".to_string());
        v.extend((1..k_w).map(|j| format!("a:x{j}")));
        v
    };
    // One resample's design `[w | a | a w]` and outcome, rewritten by every replicate.
    let k = if include_interactions {
        2 * k_w
    } else {
        k_w + 1
    };
    let what = "stdreg bootstrap design";
    let mut x_boot = resource::try_zeros2(n, k, what)?;
    let mut y_boot = Array1::from(resource::try_vec(n, 0.0, what)?);
    let mut a_boot = resource::try_vec(n, 0.0, what)?;
    let mut indices = resource::try_vec(n, 0usize, what)?;
    let ctl = GlmControl {
        epsilon: tolerance,
        max_iter,
    };

    for _ in 0..n_bootstrap {
        resource::checkpoint()?;

        // Resample with replacement
        for i in indices.iter_mut() {
            *i = rng.gen_range(0..n);
        }

        // Check that bootstrap sample has both treated and control
        let n_t_boot = indices.iter().filter(|&&i| a[i] >= 0.5).count();
        if n_t_boot == 0 || n_t_boot == n {
            continue;
        }

        // The resample's design, as build_design_matrix lays it out
        for (r, &i) in indices.iter().enumerate() {
            let ai = a[i];
            let wi = w.row(i);
            let mut xr = x_boot.row_mut(r);
            for j in 0..k_w {
                xr[j] = wi[j];
            }
            xr[k_w] = ai;
            if include_interactions {
                for j in 1..k_w {
                    xr[k_w + j] = ai * wi[j];
                }
            }
            y_boot[r] = y[i];
            a_boot[r] = ai;
        }

        let fit =
            match fit_replicate(&x_boot, &y_boot, model_type, ctl, start.as_ref()).and_then(|f| {
                check_treatment_identified(&f.aliased, &names, k_w, include_interactions, "a")
                    .map(|_| f)
            }) {
                Ok(f) => f,
                Err(e @ (EconError::Cancelled(_) | EconError::ResourceLimit(_))) => return Err(e),
                Err(_) => {
                    failed += 1;
                    continue;
                }
            };
        if !fit.converged {
            not_converged += 1;
        }
        if fit.separation {
            separated += 1;
        }

        let (ate_boot, ey1_boot, ey0_boot) = resample_estimand(
            w,
            &indices,
            &a_boot,
            &fit.coefficients,
            k_w,
            include_interactions,
            model_type,
            estimand,
        );

        if ate_boot.is_finite() {
            boot_ates.push(ate_boot);
            boot_ey1s.push(ey1_boot);
            boot_ey0s.push(ey0_boot);
        }
    }

    if boot_ates.is_empty() {
        return Err(EconError::Computation(
            "All bootstrap replications failed".to_string(),
        ));
    }

    Ok(BootDraws {
        se: draw_sd(&boot_ates),
        ey1_se: draw_sd(&boot_ey1s),
        ey0_se: draw_sd(&boot_ey0s),
        ates: boot_ates,
        ey1s: boot_ey1s,
        ey0s: boot_ey0s,
        not_converged,
        separated,
        failed,
    })
}

/// The coefficients of one bootstrap replicate and what the replicate counts.
struct ReplicateFit {
    /// `NaN` for aliased columns.
    coefficients: Array1<f64>,
    aliased: Vec<usize>,
    converged: bool,
    separation: bool,
}

/// One bootstrap replicate's outcome model, coefficients only (the replicate needs no
/// fitted values, R-squared or warnings):
///
/// - Linear: the normal equations on a full-rank, well-conditioned design (OLS's decision,
///   `normal_equations_inverse`) without OLS's refinement step, as the fixed-design
///   bootstrap of `regression::ols` does: a replicate's coefficients then carry a relative
///   error of order `eps / rcond` (at most about 1e-8 at the decision's threshold), far below
///   the Monte Carlo error of the standard error. `solve_ols` (the pivoted QR) otherwise.
/// - Logistic: `tmle::lean_logit` from `start` (glm.fit's deviance rule and iterates, without
///   the shared GLM's covariance, deviances and warnings), and the shared GLM when the fit
///   leaves that path (separation, aliasing, ill conditioning, no convergence), so those
///   replicates are counted as before.
/// - Poisson: the shared GLM.
fn fit_replicate(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model_type: StdRegModel,
    ctl: GlmControl,
    start: Option<&Array1<f64>>,
) -> EconResult<ReplicateFit> {
    let full = |fit: OutcomeFit| ReplicateFit {
        coefficients: fit.coefficients,
        aliased: fit.aliased,
        converged: fit.converged,
        separation: fit.separation,
    };
    match model_type {
        StdRegModel::Linear => {
            resource::checkpoint()?;
            if let Some(normal) = normal_equations_inverse(x.view())? {
                return Ok(ReplicateFit {
                    coefficients: normal.inverse.dot(&xty(&x.view(), y)),
                    aliased: Vec::new(),
                    converged: true,
                    separation: false,
                });
            }
            let s = solve_ols(x.view(), y.view())?;
            Ok(ReplicateFit {
                coefficients: s.beta,
                aliased: s.aliased,
                converged: true,
                separation: false,
            })
        }
        StdRegModel::Logistic => {
            let from = start.map_or(LeanStart::Mustart, |b| LeanStart::Coef(b.view()));
            if let Some(fit) = lean_logit(x.view(), y.view(), None, from, ctl)? {
                return Ok(ReplicateFit {
                    coefficients: fit.coefficients,
                    aliased: Vec::new(),
                    converged: true,
                    separation: false,
                });
            }
            fit_outcome_model(x, y, model_type, ctl.max_iter, ctl.epsilon, start).map(full)
        }
        StdRegModel::Poisson => {
            fit_outcome_model(x, y, model_type, ctl.max_iter, ctl.epsilon, start).map(full)
        }
    }
}

/// [`compute_estimand`] for one resample, from `w`'s rows `indices` and the treatment
/// `a_boot` of the resample, without building the counterfactual designs: the linear
/// predictor under A = 0 is `w_i b_w`, under A = 1 it adds `b_a` and the interaction terms.
/// Aliased (`NaN`) coefficients count as 0, as in [`predict`].
#[allow(clippy::too_many_arguments)]
fn resample_estimand(
    w: &Array2<f64>,
    indices: &[usize],
    a_boot: &[f64],
    beta: &Array1<f64>,
    k_w: usize,
    include_interactions: bool,
    model_type: StdRegModel,
    estimand: StdRegEstimand,
) -> (f64, f64, f64) {
    let b = |j: usize| if beta[j].is_nan() { 0.0 } else { beta[j] };
    let b_a = b(k_w);
    let (mut s1, mut s0, mut m) = (0.0, 0.0, 0usize);
    for (&i, &ai) in indices.iter().zip(a_boot) {
        let keep = match estimand {
            StdRegEstimand::ATE | StdRegEstimand::Levels => true,
            StdRegEstimand::ATT => ai >= 0.5,
            StdRegEstimand::ATC => ai < 0.5,
        };
        if !keep {
            continue;
        }
        let wi = w.row(i);
        let mut z0 = 0.0;
        for j in 0..k_w {
            z0 += wi[j] * b(j);
        }
        let mut z1 = z0 + b_a;
        if include_interactions {
            for j in 1..k_w {
                z1 += wi[j] * b(k_w + j);
            }
        }
        s1 += inverse_link(z1, model_type);
        s0 += inverse_link(z0, model_type);
        m += 1;
    }
    let (ey1, ey0) = (s1 / m as f64, s0 / m as f64);
    (ey1 - ey0, ey1, ey0)
}

/// The percentile interval of the finite bootstrap draws (R's type-7 quantiles).
fn bootstrap_ci(draws: &[f64], level: f64) -> EconResult<(f64, f64)> {
    let finite: Vec<f64> = draws.iter().copied().filter(|v| v.is_finite()).collect();
    percentile_interval(&finite, level)
}

/// `(X'X)^{-1}` of the identified columns.
fn identified_bread(x_id: &Array2<f64>) -> EconResult<Array2<f64>> {
    let xtx_mat = xtx(&x_id.view());
    Ok(sym_inverse(
        xtx_mat.view(),
        SingularPolicy::Error,
        "stdreg outcome model X'X",
    )?
    .inverse)
}

/// Compute delta method standard errors on the identified columns `x_id`.
/// For linear model: SE(ATE) derived from coefficient variance.
// CAUS-39 (Tier 4): the formula is an approximation (the treatment coefficient's variance,
// the last identified column; with interactions that is not the effect's SE), unchanged here.
fn compute_delta_se(
    x_id: &Array2<f64>,
    y: &Array1<f64>,
    y_hat: &Array1<f64>,
    model_type: StdRegModel,
) -> EconResult<(f64, f64, f64)> {
    let n = y.len();
    let rank = x_id.ncols();
    let xtx_inv = identified_bread(x_id)?;

    let sigma2 = match model_type {
        StdRegModel::Linear => {
            let df = n.checked_sub(rank).filter(|&d| d > 0).ok_or_else(|| {
                EconError::InsufficientData {
                    required: rank + 1,
                    provided: n,
                    context: "stdreg delta-method residual variance".to_string(),
                }
            })?;
            let sse: f64 = y
                .iter()
                .zip(y_hat.iter())
                .map(|(&yi, &yhi)| (yi - yhi).powi(2))
                .sum();
            sse / df as f64
        }
        // For GLMs, use asymptotic variance approximation
        _ => 1.0,
    };

    let a_col_idx = rank - 1;
    let se = (sigma2 * xtx_inv[[a_col_idx, a_col_idx]]).sqrt();

    // For potential outcomes, rough approximation
    let ey1_se = se / 2.0_f64.sqrt();
    let ey0_se = se / 2.0_f64.sqrt();

    Ok((se, ey1_se, ey0_se))
}

/// Compute sandwich (robust) standard errors on the identified columns `x_id`.
// CAUS-39 (Tier 4): an HC0 sandwich on the treatment coefficient (the last identified column),
// unchanged here.
fn compute_sandwich_se(
    x_id: &Array2<f64>,
    y: &Array1<f64>,
    y_hat: &Array1<f64>,
) -> EconResult<(f64, f64, f64)> {
    let n = y.len();
    let k = x_id.ncols();

    // Bread: (X'X)^{-1}
    let bread = identified_bread(x_id)?;

    // Meat: X' diag(e^2) X
    let residuals: Array1<f64> = y - y_hat;
    let mut meat = Array2::zeros((k, k));
    for i in 0..n {
        let e_i_sq = residuals[i].powi(2);
        for j in 0..k {
            for l in 0..k {
                meat[[j, l]] += e_i_sq * x_id[[i, j]] * x_id[[i, l]];
            }
        }
    }

    // Sandwich: (X'X)^{-1} X'diag(e^2)X (X'X)^{-1}
    let sandwich = bread.dot(&meat).dot(&bread);

    let a_col_idx = k - 1;
    let se = sandwich[[a_col_idx, a_col_idx]].sqrt();

    let ey1_se = se / 2.0_f64.sqrt();
    let ey0_se = se / 2.0_f64.sqrt();

    Ok((se, ey1_se, ey0_se))
}

/// Compute additional effect measures for binary outcomes.
fn compute_binary_effect_measures(
    ey1: f64,
    ey0: f64,
    boot_ey1s: &[f64],
    boot_ey0s: &[f64],
    confidence_level: f64,
) -> EconResult<(
    Option<f64>,
    Option<(f64, f64)>,
    Option<f64>,
    Option<(f64, f64)>,
    Option<f64>,
)> {
    // Risk Ratio
    let rr = if ey0 > 1e-10 { Some(ey1 / ey0) } else { None };

    // Odds Ratio
    let odds1 = ey1 / (1.0 - ey1).max(1e-10);
    let odds0 = ey0 / (1.0 - ey0).max(1e-10);
    let or = if odds0 > 1e-10 {
        Some(odds1 / odds0)
    } else {
        None
    };

    // NNT (Number Needed to Treat)
    let rd = ey1 - ey0;
    let nnt = if rd.abs() > 1e-10 {
        Some(1.0 / rd.abs())
    } else {
        None
    };

    // Bootstrap percentile CIs if available
    let rr_ci = if boot_ey1s.len() >= 20 && boot_ey0s.len() >= 20 {
        let boot_rrs: Vec<f64> = boot_ey1s
            .iter()
            .zip(boot_ey0s.iter())
            .filter_map(|(&y1, &y0)| if y0 > 1e-10 { Some(y1 / y0) } else { None })
            .collect();
        if boot_rrs.len() >= 20 {
            Some(bootstrap_ci(&boot_rrs, confidence_level)?)
        } else {
            None
        }
    } else {
        None
    };

    let or_ci = if boot_ey1s.len() >= 20 && boot_ey0s.len() >= 20 {
        let boot_ors: Vec<f64> = boot_ey1s
            .iter()
            .zip(boot_ey0s.iter())
            .filter_map(|(&y1, &y0)| {
                let o1 = y1 / (1.0 - y1).max(1e-10);
                let o0 = y0 / (1.0 - y0).max(1e-10);
                if o0 > 1e-10 { Some(o1 / o0) } else { None }
            })
            .collect();
        if boot_ors.len() >= 20 {
            Some(bootstrap_ci(&boot_ors, confidence_level)?)
        } else {
            None
        }
    } else {
        None
    };

    Ok((rr, rr_ci, or, or_ci, nnt))
}

/// Check if outcome appears to be binary.
fn is_binary_outcome(y: &Array1<f64>) -> bool {
    let unique: std::collections::HashSet<u64> =
        y.iter().map(|&v| (v * 1000.0).round() as u64).collect();
    unique.len() == 2
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    fn expect_invalid_spec(err: EconError, needle: &str) {
        match err {
            EconError::InvalidSpecification { message } => {
                assert!(
                    message.contains(needle),
                    "message `{message}` does not mention `{needle}`"
                )
            }
            other => panic!("expected InvalidSpecification mentioning `{needle}`, got {other:?}"),
        }
    }

    fn cancelled_budget() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn run_test_stdreg(config: StdRegConfig) -> EconResult<StdRegResult> {
        run_stdreg(
            &create_stdreg_test_dataset(),
            "y",
            "treatment",
            &["x1", "x2"],
            config,
        )
    }

    #[test]
    fn test_stdreg_bootstrap_count_is_an_error_not_an_abort() {
        let err = run_test_stdreg(StdRegConfig {
            se_method: SEMethod::Bootstrap,
            n_bootstrap: usize::MAX,
            seed: Some(42),
            ..Default::default()
        })
        .unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "got {err:?}");
    }

    #[test]
    fn test_stdreg_needs_two_replicates() {
        let err = run_test_stdreg(StdRegConfig {
            se_method: SEMethod::Bootstrap,
            n_bootstrap: 1,
            seed: Some(42),
            ..Default::default()
        })
        .unwrap_err();
        expect_invalid_spec(err, "n_bootstrap");
        assert!(
            run_test_stdreg(StdRegConfig {
                se_method: SEMethod::Delta,
                n_bootstrap: 0,
                ..Default::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn test_stdreg_rejects_bad_iteration_controls() {
        let base = || StdRegConfig {
            se_method: SEMethod::Delta,
            ..Default::default()
        };
        expect_invalid_spec(
            run_test_stdreg(StdRegConfig {
                max_iter: 0,
                ..base()
            })
            .unwrap_err(),
            "max_iter",
        );
        expect_invalid_spec(
            run_test_stdreg(StdRegConfig {
                tolerance: f64::NAN,
                ..base()
            })
            .unwrap_err(),
            "tolerance",
        );
        expect_invalid_spec(
            run_test_stdreg(StdRegConfig {
                tolerance: 0.0,
                ..base()
            })
            .unwrap_err(),
            "tolerance",
        );
    }

    #[test]
    fn test_stdreg_honours_cancellation() {
        let r = resource::with_budget(cancelled_budget(), || {
            run_test_stdreg(StdRegConfig {
                model_type: StdRegModel::Linear,
                se_method: SEMethod::Bootstrap,
                n_bootstrap: 20,
                seed: Some(42),
                ..Default::default()
            })
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
        // the IRLS loop of the logistic fit is a checkpoint too (no bootstrap here)
        let r = resource::with_budget(cancelled_budget(), || {
            run_stdreg(
                &create_binary_outcome_dataset(),
                "y",
                "treatment",
                &["x1"],
                StdRegConfig {
                    model_type: StdRegModel::Logistic,
                    se_method: SEMethod::Delta,
                    ..Default::default()
                },
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
    }

    /// Create a test dataset with known treatment effect.
    ///
    /// DGP:
    /// - X ~ Uniform(0, 1)
    /// - A | X ~ Bernoulli(P(A=1) depending on X)
    /// - Y = 0.5 + 0.3*X + 0.4*A + noise
    ///
    /// True ATE = 0.4
    fn create_stdreg_test_dataset() -> Dataset {
        let df = df! {
            "y" => [
                // Treated observations: Y approx 0.5 + 0.3*X + 0.4 + noise
                1.1, 1.3, 1.0, 1.4, 1.15, 1.35, 1.05, 1.45, 1.12, 1.28,
                0.95, 1.5, 1.18, 1.38, 1.02, 1.48, 1.2, 1.32, 0.98, 1.52,
                // Control observations: Y approx 0.5 + 0.3*X + noise
                0.6, 0.85, 0.55, 0.95, 0.7, 0.9, 0.58, 0.92, 0.65, 0.88,
                0.5, 1.0, 0.72, 0.82, 0.52, 0.98, 0.75, 0.78, 0.48, 1.02
            ],
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "x1" => [
                // Covariates with overlapping distribution
                0.3, 0.7, 0.2, 0.8, 0.35, 0.75, 0.25, 0.85, 0.32, 0.68,
                0.15, 0.9, 0.4, 0.6, 0.22, 0.78, 0.45, 0.55, 0.18, 0.82,
                0.25, 0.65, 0.15, 0.75, 0.3, 0.7, 0.2, 0.8, 0.28, 0.62,
                0.1, 0.85, 0.35, 0.58, 0.18, 0.72, 0.4, 0.55, 0.12, 0.78
            ],
            "x2" => [
                0.5, 0.6, 0.45, 0.65, 0.52, 0.58, 0.48, 0.62, 0.5, 0.6,
                0.4, 0.7, 0.55, 0.55, 0.42, 0.68, 0.58, 0.52, 0.38, 0.72,
                0.48, 0.62, 0.42, 0.68, 0.5, 0.6, 0.45, 0.65, 0.46, 0.64,
                0.35, 0.75, 0.52, 0.58, 0.38, 0.72, 0.55, 0.55, 0.32, 0.78
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_stdreg_linear_ate() {
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Linear,
            estimand: StdRegEstimand::ATE,
            se_method: SEMethod::Bootstrap,
            n_bootstrap: 100, // Fewer for faster tests
            seed: Some(42),
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 40);
        assert_eq!(result.n_treated, 20);
        assert_eq!(result.n_control, 20);

        // ATE should be approximately 0.4 (the true value)
        assert!(
            result.ate > 0.2 && result.ate < 0.6,
            "ATE should be around 0.4, got {}",
            result.ate
        );

        // Standard error should be positive and finite
        assert!(
            result.se > 0.0 && result.se.is_finite(),
            "SE should be positive and finite, got {}",
            result.se
        );

        // E[Y(1)] should be higher than E[Y(0)]
        assert!(
            result.ey1 > result.ey0,
            "E[Y(1)] = {} should be > E[Y(0)] = {}",
            result.ey1,
            result.ey0
        );

        // R-squared should be reasonable
        assert!(
            result.r_squared.unwrap() > 0.5,
            "R-squared should be high for this DGP"
        );
    }

    #[test]
    fn test_stdreg_linear_att() {
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Linear,
            estimand: StdRegEstimand::ATT,
            se_method: SEMethod::Bootstrap,
            n_bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // ATT should also be positive
        assert!(
            result.ate > 0.0,
            "ATT should be positive, got {}",
            result.ate
        );
        assert_eq!(result.estimand, StdRegEstimand::ATT);
    }

    #[test]
    fn test_stdreg_linear_atc() {
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Linear,
            estimand: StdRegEstimand::ATC,
            se_method: SEMethod::Bootstrap,
            n_bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        assert!(result.ate > 0.0, "ATC should be positive");
        assert_eq!(result.estimand, StdRegEstimand::ATC);
    }

    #[test]
    fn test_stdreg_delta_se_runs() {
        // the delta-method formula is CAUS-39 (Tier 4); stdReg's values are recorded in
        // the case stdreg_glm (*_stdreg_ate_se)
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Linear,
            estimand: StdRegEstimand::ATE,
            se_method: SEMethod::Delta,
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Delta method should give positive SE
        assert!(result.se > 0.0 && result.se.is_finite());
    }

    #[test]
    fn test_stdreg_sandwich_se() {
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Linear,
            estimand: StdRegEstimand::ATE,
            se_method: SEMethod::Sandwich,
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Sandwich SE should be positive
        assert!(result.se > 0.0 && result.se.is_finite());
    }

    #[test]
    fn test_stdreg_confidence_interval() {
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Linear,
            n_bootstrap: 200,
            seed: Some(42),
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // CI should bracket the point estimate
        assert!(
            result.ci_lower < result.ate && result.ate < result.ci_upper,
            "ATE {} should be within CI [{}, {}]",
            result.ate,
            result.ci_lower,
            result.ci_upper
        );

        // CI should be reasonably wide but not too wide
        let ci_width = result.ci_upper - result.ci_lower;
        assert!(
            ci_width > 0.0 && ci_width < 1.0,
            "CI width {} seems unreasonable",
            ci_width
        );
    }

    #[test]
    fn test_stdreg_individual_effects() {
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Linear,
            n_bootstrap: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        // Individual effects should be available
        assert_eq!(result.individual_effects.len(), result.n_obs);
        assert_eq!(result.y_hat_1.len(), result.n_obs);
        assert_eq!(result.y_hat_0.len(), result.n_obs);

        // Average of individual effects should match ATE
        let mean_ie: f64 = result.individual_effects.iter().sum::<f64>() / result.n_obs as f64;
        assert!(
            (mean_ie - result.ate).abs() < 1e-10,
            "Mean individual effect {} should match ATE {}",
            mean_ie,
            result.ate
        );
    }

    #[test]
    fn test_stdreg_missing_column() {
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig::default();

        let result = run_stdreg(&dataset, "nonexistent", "treatment", &["x1"], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_stdreg_display() {
        let dataset = create_stdreg_test_dataset();
        let config = StdRegConfig {
            n_bootstrap: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("Regression Standardization"));
        assert!(output.contains("Effect:"));
        assert!(output.contains("Std. Error:"));
        assert!(output.contains("E[Y(1)]"));
        assert!(output.contains("E[Y(0)]"));
    }

    #[test]
    fn test_stdreg_default_function() {
        let dataset = create_stdreg_test_dataset();

        // Test the convenience function
        let result = stdreg(&dataset, "y", "treatment", &["x1", "x2"]).unwrap();

        assert_eq!(result.estimand, StdRegEstimand::ATE);
        assert_eq!(result.model_type, StdRegModel::Linear);
    }

    /// Create a binary outcome dataset for logistic model testing.
    fn create_binary_outcome_dataset() -> Dataset {
        let df = df! {
            "y" => [
                // Treated: higher probability of Y=1
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                // Control: lower probability of Y=1
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0
            ],
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "x1" => [
                0.3, 0.7, 0.2, 0.8, 0.35, 0.75, 0.25, 0.85, 0.32, 0.68,
                0.15, 0.9, 0.4, 0.6, 0.22, 0.78, 0.45, 0.55, 0.18, 0.82,
                0.25, 0.65, 0.15, 0.75, 0.3, 0.7, 0.2, 0.8, 0.28, 0.62,
                0.1, 0.85, 0.35, 0.58, 0.18, 0.72, 0.4, 0.55, 0.12, 0.78
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_stdreg_logistic_basic() {
        let dataset = create_binary_outcome_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Logistic,
            estimand: StdRegEstimand::ATE,
            se_method: SEMethod::Bootstrap,
            n_bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1"], config).unwrap();

        // ATE should be positive (treatment increases probability of Y=1)
        assert!(
            result.ate > 0.0,
            "ATE should be positive for this DGP, got {}",
            result.ate
        );

        // Risk ratio and odds ratio should be available
        assert!(
            result.risk_ratio.is_some(),
            "Risk ratio should be computed for binary outcome"
        );
        assert!(
            result.odds_ratio.is_some(),
            "Odds ratio should be computed for binary outcome"
        );

        // Risk ratio should be > 1 (treatment is beneficial)
        assert!(result.risk_ratio.unwrap() > 1.0, "Risk ratio should be > 1");
    }

    #[test]
    fn test_stdreg_logistic_effect_measures() {
        let dataset = create_binary_outcome_dataset();
        let config = StdRegConfig {
            model_type: StdRegModel::Logistic,
            n_bootstrap: 200,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_stdreg(&dataset, "y", "treatment", &["x1"], config).unwrap();

        // Check that CIs are computed for RR and OR
        if let Some((lo, hi)) = result.risk_ratio_ci {
            assert!(lo < result.risk_ratio.unwrap() && result.risk_ratio.unwrap() < hi);
        }

        if let Some((lo, hi)) = result.odds_ratio_ci {
            assert!(lo < result.odds_ratio.unwrap() && result.odds_ratio.unwrap() < hi);
        }

        // NNT should be positive and finite
        if let Some(nnt) = result.nnt {
            assert!(nnt > 0.0 && nnt.is_finite());
        }
    }

    // ── T2-WP13: missing values and column types (R case causal_mediation_na) ──

    #[test]
    fn test_validate_na_stdreg_columns() {
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, y: &str, d: &str, x: &[&str]| {
            run_stdreg(
                ds,
                y,
                d,
                x,
                StdRegConfig {
                    se_method: SEMethod::Sandwich,
                    ..Default::default()
                },
            )
        };
        nf::assert_refusals(&case, &ds, true, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| vec![r.ate, r.se]);
    }

    // ── T3-WP29: the R reference cases stdreg_glm, stdreg_linear, stdreg_separation
    //    (validation/r/cases/causal_stdreg.R) ──

    fn col(ds: &Dataset, name: &str) -> Vec<f64> {
        ds.df()
            .column(name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    /// The cases' dataset read directly, for the tests that compare no R value.
    fn stdreg_sim() -> Dataset {
        crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/stdreg_sim.csv",
        )
    }

    fn with_col(ds: &Dataset, name: &str, values: Vec<f64>) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(Column::new(name.into(), values)).unwrap();
        Dataset::new(df)
    }

    fn fit_case(
        ds: &Dataset,
        y: &str,
        model: StdRegModel,
        inter: bool,
        est: StdRegEstimand,
        tolerance: f64,
    ) -> StdRegResult {
        let c = StdRegConfig {
            model_type: model,
            estimand: est,
            include_interactions: inter,
            // point estimates only; the SEs are CAUS-39 (Tier 4)
            se_method: SEMethod::Sandwich,
            tolerance,
            ..Default::default()
        };
        run_stdreg(ds, y, "a", &["x1", "x2"], c).unwrap()
    }

    #[test]
    fn test_validate_stdreg_glm() {
        let (case, ds) = crate::test_support::load_case("stdreg_glm");
        let mut check = Check::new(&case);
        for (p, y, m, inter) in [
            ("logit", "yb", StdRegModel::Logistic, false),
            ("logit_int", "yb", StdRegModel::Logistic, true),
            ("pois", "yp", StdRegModel::Poisson, false),
        ] {
            let f = |e| fit_case(&ds, y, m, inter, e, 1e-14); // the case's glm epsilon
            let (ate, att, atc) = (
                f(StdRegEstimand::ATE),
                f(StdRegEstimand::ATT),
                f(StdRegEstimand::ATC),
            );
            let t = Tol::coef(ate.n_obs);
            check
                .vector(&format!("{p}_coef"), &ate.outcome_coefs, t)
                .scalar(&format!("{p}_ey1"), ate.ey1, t)
                .scalar(&format!("{p}_ey0"), ate.ey0, t)
                .scalar(&format!("{p}_ate"), ate.ate, t)
                .scalar(&format!("{p}_att"), att.ate, t)
                .scalar(&format!("{p}_atc"), atc.ate, t)
                .scalar(&format!("{p}_att_ey1"), att.ey1, t)
                .scalar(&format!("{p}_att_ey0"), att.ey0, t);
            assert!(ate.convergence.as_ref().unwrap().converged, "{p}");
            if !inter {
                check.scalar(
                    &format!("{p}_r2"),
                    ate.r_squared.unwrap(),
                    Tol::stat(ate.n_obs),
                );
            }
        }
        check.finish();
    }

    #[test]
    fn test_validate_stdreg_glm_default_control() {
        // R's glm.control() epsilon 1e-8 (StdRegConfig's default) stops within about 6e-10 of
        // the epsilon-1e-14 optimum the case records
        let (case, ds) = crate::test_support::load_case("stdreg_glm");
        let t = Tol::iterative(1e-8);
        let mut check = Check::new(&case);
        for (p, y, m) in [
            ("logit", "yb", StdRegModel::Logistic),
            ("pois", "yp", StdRegModel::Poisson),
        ] {
            let r = fit_case(&ds, y, m, false, StdRegEstimand::ATE, 1e-8);
            check
                .vector(&format!("{p}_coef"), &r.outcome_coefs, t)
                .scalar(&format!("{p}_ate"), r.ate, t);
            assert!(r.convergence.as_ref().unwrap().converged, "{p}");
            assert!(
                r.warnings.iter().all(|w| !w.contains("did not converge")),
                "{p}: {:?}",
                r.warnings
            );
        }
        check.finish();
    }

    #[test]
    fn test_validate_stdreg_linear() {
        let (case, ds) = crate::test_support::load_case("stdreg_linear");
        let f = |e| {
            let c = StdRegConfig {
                model_type: StdRegModel::Linear,
                estimand: e,
                include_interactions: true,
                se_method: SEMethod::Sandwich,
                ..Default::default()
            };
            run_stdreg(&ds, "yc", "a", &["x1", "x2"], c).unwrap()
        };
        let (ate, att, atc) = (
            f(StdRegEstimand::ATE),
            f(StdRegEstimand::ATT),
            f(StdRegEstimand::ATC),
        );
        assert!(ate.convergence.is_none()); // least squares
        let t = Tol::coef(300);
        let alias = run_stdreg(
            &ds,
            "yc",
            "a",
            &["x1", "x2", "x1dup"],
            StdRegConfig {
                se_method: SEMethod::Sandwich,
                ..Default::default()
            },
        )
        .unwrap();
        Check::new(&case)
            .vector("int_coef", &ate.outcome_coefs, t)
            .scalar("int_ey1", ate.ey1, t)
            .scalar("int_ey0", ate.ey0, t)
            .scalar("int_ate", ate.ate, t)
            .scalar("int_att", att.ate, t)
            .scalar("int_atc", atc.ate, t)
            .scalar("int_r2", ate.r_squared.unwrap(), Tol::stat(300))
            .vector("alias_coef", &alias.outcome_coefs, t) // NaN (R's NA) for x1dup
            .scalar("alias_ey1", alias.ey1, t)
            .scalar("alias_ey0", alias.ey0, t)
            .scalar("alias_ate", alias.ate, t)
            .finish();
        assert!(alias.outcome_coefs[3].is_nan());
        assert!(
            alias
                .warnings
                .iter()
                .any(|w| w
                    == "outcome model: Coefficients not defined because of singularities: x1dup"),
            "{:?}",
            alias.warnings
        );
        assert!(alias.warnings.iter().all(|w| !w.contains("rank-deficient")));
        // null in JSON, NaN again when read back
        let json = serde_json::to_value(&alias).unwrap();
        assert!(json["outcome_coefs"][3].is_null());
        let back: StdRegResult = serde_json::from_value(json).unwrap();
        assert!(back.outcome_coefs[3].is_nan());
    }

    #[test]
    fn test_validate_stdreg_separation_warns() {
        let (case, ds) = crate::test_support::load_case("stdreg_separation");
        let c = StdRegConfig {
            model_type: StdRegModel::Logistic,
            se_method: SEMethod::Sandwich,
            ..Default::default()
        };
        let r = run_stdreg(&ds, "ys", "a", &["x1", "x2"], c).unwrap();
        Check::new(&case)
            .scalar("sep_ey1", r.ey1, Tol::coef(300))
            .scalar("sep_ey0", r.ey0, Tol::coef(300))
            .scalar(
                "sep_converged",
                r.convergence.as_ref().unwrap().converged as u8 as f64,
                Tol::exact(),
            )
            .finish();
        assert!(r.ate.abs() <= 1e-8, "{}", r.ate); // R 2.17e-11
        let fitted01 = "outcome model: glm.fit: fitted probabilities numerically 0 or 1 occurred";
        assert_eq!(
            r.warnings.iter().filter(|w| *w == fitted01).count() as f64,
            case.scalar("sep_warn_fitted01"),
            "{:?}",
            r.warnings
        );
        let k = case.scalar("sep_fitted_extreme") as usize;
        let sentence = format!(
            "Separation: fitted probabilities are numerically 0 or 1 for {k} of 300 observations"
        );
        assert_eq!(
            r.warnings
                .iter()
                .filter(|w| w.starts_with(&sentence))
                .count(),
            1,
            "{:?}",
            r.warnings
        );
        assert_eq!(case.scalar("sep_warn_noconv"), 0.0);
        assert!(r.warnings.iter().all(|w| !w.contains("did not converge")));
        assert!(r.to_string().contains("Converged: yes ("), "{r}");
    }

    #[test]
    fn test_validate_stdreg_poisson_scale_free() {
        let (case, ds) = crate::test_support::load_case("stdreg_glm");
        let base = fit_case(
            &ds,
            "yp",
            StdRegModel::Poisson,
            false,
            StdRegEstimand::ATE,
            1e-8,
        );
        for k in [1e-6, 1e6] {
            let x1: Vec<f64> = col(&ds, "x1").iter().map(|v| v * k).collect();
            let x2: Vec<f64> = col(&ds, "x2").iter().map(|v| v * k).collect();
            let scaled = with_col(&with_col(&ds, "x1", x1), "x2", x2);
            let r = fit_case(
                &scaled,
                "yp",
                StdRegModel::Poisson,
                false,
                StdRegEstimand::ATE,
                1e-8,
            );
            assert!(r.convergence.as_ref().unwrap().converged, "{k}");
            assert!(
                Tol::new(1e-9, 0.0).allows(r.ate, base.ate),
                "{k}: {}",
                r.ate
            );
            assert!(
                Tol::iterative(1e-8).allows(r.ate, case.scalar("pois_ate")),
                "{k}: {}",
                r.ate
            );
        }
    }

    #[test]
    fn test_stdreg_replicate_fit_matches_the_outcome_model() {
        // A bootstrap replicate's coefficients-only fit (fit_replicate) and its standardisation
        // without counterfactual designs (resample_estimand) against the full fit and
        // compute_estimand on the counterfactual designs, on a resample of stdreg_sim.
        let ds = stdreg_sim();
        let (a, x1, x2) = (col(&ds, "a"), col(&ds, "x1"), col(&ds, "x2"));
        let n = a.len();
        let w = Array2::from_shape_fn((n, 3), |(i, j)| [1.0, x1[i], x2[i]][j]);
        let a = Array1::from(a);
        let idx: Vec<usize> = (0..n).map(|i| (i * 7 + i / 3) % n).collect();
        let w_b = w.select(Axis(0), &idx);
        let a_b = a.select(Axis(0), &idx);
        let ctl = GlmControl {
            epsilon: 1e-10,
            max_iter: 100,
        };
        for (yname, model) in [
            ("yc", StdRegModel::Linear),
            ("yb", StdRegModel::Logistic),
            ("yp", StdRegModel::Poisson),
        ] {
            let y = Array1::from(col(&ds, yname)).select(Axis(0), &idx);
            for inter in [false, true] {
                let (x, _) = build_design_matrix(&w_b, &a_b, &["x1", "x2"], "a", inter);
                let full = fit_outcome_model(&x, &y, model, 100, 1e-10, None).unwrap();
                let start = full.coefficients.mapv(|v| v * 0.9);
                let rep = fit_replicate(&x, &y, model, ctl, Some(&start)).unwrap();
                assert!(rep.converged && !rep.separation && rep.aliased.is_empty());
                for (u, v) in rep.coefficients.iter().zip(&full.coefficients) {
                    assert!(Tol::new(1e-8, 1e-10).allows(*u, *v), "{model:?} {u} vs {v}");
                }
                let (x1c, x0c) = create_counterfactual_matrices(&w_b, n, 3, inter);
                let (p1, p0) = (
                    predict(&x1c, &rep.coefficients, model),
                    predict(&x0c, &rep.coefficients, model),
                );
                for est in [
                    StdRegEstimand::ATE,
                    StdRegEstimand::ATT,
                    StdRegEstimand::ATC,
                ] {
                    let want = compute_estimand(&p1, &p0, &a_b, est);
                    let got = resample_estimand(
                        &w,
                        &idx,
                        a_b.as_slice().unwrap(),
                        &rep.coefficients,
                        3,
                        inter,
                        model,
                        est,
                    );
                    let t = Tol::new(1e-12, 1e-14);
                    assert!(
                        t.allows(got.0, want.0)
                            && t.allows(got.1, want.1)
                            && t.allows(got.2, want.2),
                        "{model:?} {est:?} {got:?} vs {want:?}"
                    );
                }
            }
        }
        // a separated replicate leaves the lean logit and is counted as before
        let ys = Array1::from(col(&ds, "ys")).select(Axis(0), &idx);
        let (x, _) = build_design_matrix(&w_b, &a_b, &["x1", "x2"], "a", false);
        let rep = fit_replicate(&x, &ys, StdRegModel::Logistic, ctl, None).unwrap();
        assert!(rep.separation);
        // an aliased covariate takes the QR and reports the column
        let mut xa = Array2::<f64>::zeros((n, 5));
        xa.slice_mut(ndarray::s![.., ..4]).assign(&x);
        xa.column_mut(4)
            .assign(&x.column(1).mapv(|v| 2.0 * v + 1.0));
        let yc = Array1::from(col(&ds, "yc")).select(Axis(0), &idx);
        let rep = fit_replicate(&xa, &yc, StdRegModel::Linear, ctl, None).unwrap();
        assert_eq!(rep.aliased, vec![4]);
    }

    #[test]
    fn test_stdreg_aliased_treatment_is_an_error() {
        let ds = stdreg_sim();
        let a = col(&ds, "a");
        let lin = |inter| StdRegConfig {
            include_interactions: inter,
            se_method: SEMethod::Sandwich,
            ..Default::default()
        };
        let d = with_col(&ds, "acopy", a.clone());
        let err = run_stdreg(&d, "yc", "a", &["x1", "x2", "acopy"], lin(false)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("'a'") && msg.contains("not identified"),
            "{msg}"
        );
        expect_invalid_spec(err, "collinear with the covariates");
        // a:x1const = a while x1const itself is identified
        let x1 = col(&ds, "x1");
        let x1const: Vec<f64> = a
            .iter()
            .zip(&x1)
            .map(|(&ai, &xi)| if ai == 1.0 { 1.0 } else { xi })
            .collect();
        let d = with_col(&ds, "x1const", x1const);
        let err = run_stdreg(&d, "yc", "a", &["x1const", "x2"], lin(true)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("'a'") && msg.contains("not identified"),
            "{msg}"
        );
        expect_invalid_spec(err, "collinear with the covariates");
    }

    #[test]
    fn test_stdreg_aliased_covariate_keeps_sandwich_se() {
        let ds = stdreg_sim();
        for se_method in [SEMethod::Sandwich, SEMethod::Delta] {
            let c = StdRegConfig {
                se_method,
                ..Default::default()
            };
            let full = run_stdreg(&ds, "yc", "a", &["x1", "x2", "x1dup"], c.clone()).unwrap();
            let id = run_stdreg(&ds, "yc", "a", &["x1", "x2"], c).unwrap();
            let t = Tol::new(1e-12, 0.0);
            assert!(t.allows(full.ate, id.ate), "{se_method}");
            assert!(
                t.allows(full.se, id.se),
                "{se_method}: {} {}",
                full.se,
                id.se
            );
        }
    }

    #[test]
    fn test_stdreg_bootstrap_ci_is_type_7() {
        let draws: Vec<f64> = (1..=10).map(f64::from).collect();
        let (lo, hi) = bootstrap_ci(&draws, 0.9).unwrap();
        // R: quantile(1:10, c(.05, .95))
        assert!(
            (lo - 1.45).abs() < 1e-12 && (hi - 9.55).abs() < 1e-12,
            "{lo} {hi}"
        );
        // non-finite draws are dropped
        let mut with_nan = draws.clone();
        with_nan.push(f64::NAN);
        with_nan.push(f64::INFINITY);
        assert_eq!(bootstrap_ci(&with_nan, 0.9).unwrap(), (lo, hi));
    }

    #[test]
    fn test_stdreg_options_parse_aliases_and_reject_unknown() {
        StdRegModel::choices().validate().unwrap();
        StdRegEstimand::choices().validate().unwrap();
        SEMethod::choices().validate().unwrap();
        assert_eq!(
            "logit".parse::<StdRegModel>().unwrap(),
            StdRegModel::Logistic
        );
        assert_eq!(
            "count".parse::<StdRegModel>().unwrap(),
            StdRegModel::Poisson
        );
        assert_eq!("OLS".parse::<StdRegModel>().unwrap(), StdRegModel::Linear);
        assert_eq!(
            "ATT".parse::<StdRegEstimand>().unwrap(),
            StdRegEstimand::ATT
        );
        assert_eq!("robust".parse::<SEMethod>().unwrap(), SEMethod::Sandwich);
        assert_eq!("delta-method".parse::<SEMethod>().unwrap(), SEMethod::Delta);
        expect_invalid_spec(
            "probit".parse::<StdRegModel>().unwrap_err(),
            "unknown model_type 'probit'",
        );
        expect_invalid_spec(
            "ate1".parse::<StdRegEstimand>().unwrap_err(),
            "unknown estimand 'ate1'",
        );
        expect_invalid_spec(
            "jackknife".parse::<SEMethod>().unwrap_err(),
            "unknown se_method 'jackknife'",
        );
    }

    #[test]
    fn test_stdreg_rejects_bad_level() {
        let ds = stdreg_sim();
        for level in [1.0, f64::NAN] {
            let c = StdRegConfig {
                confidence_level: level,
                se_method: SEMethod::Delta,
                ..Default::default()
            };
            let err = run_stdreg(&ds, "yc", "a", &["x1", "x2"], c).unwrap_err();
            assert!(err.to_string().contains("confidence_level"), "{err}");
        }
    }

    #[test]
    fn test_stdreg_logistic_needs_unit_outcome() {
        let ds = stdreg_sim();
        let c = StdRegConfig {
            model_type: StdRegModel::Logistic,
            se_method: SEMethod::Sandwich,
            ..Default::default()
        };
        expect_invalid_spec(
            run_stdreg(&ds, "yp", "a", &["x1", "x2"], c).unwrap_err(),
            "model_type logistic needs an outcome in [0, 1]; 'yp' ranges from",
        );
    }

    #[test]
    fn test_stdreg_poisson_needs_nonnegative_outcome() {
        let ds = stdreg_sim();
        let mut yneg = col(&ds, "yp");
        yneg[3] = -1.5;
        let c = StdRegConfig {
            model_type: StdRegModel::Poisson,
            se_method: SEMethod::Sandwich,
            ..Default::default()
        };
        expect_invalid_spec(
            run_stdreg(&with_col(&ds, "yneg", yneg), "yneg", "a", &["x1", "x2"], c).unwrap_err(),
            "model_type poisson needs a non-negative outcome; 'yneg' has minimum -1.5",
        );
    }

    #[test]
    fn test_stdreg_refuses_non_binary_treatment() {
        // a 0/2 treatment would be fitted on its raw values while the counterfactual designs
        // set it to 1 and 0
        let ds = stdreg_sim();
        let a2: Vec<f64> = col(&ds, "a").iter().map(|v| 2.0 * v).collect();
        let c = StdRegConfig {
            se_method: SEMethod::Sandwich,
            ..Default::default()
        };
        // the message names the dataset row (0-based) and fits no propensity model
        let first = a2.iter().position(|&v| v == 2.0).unwrap();
        let e = run_stdreg(&with_col(&ds, "a2", a2), "yc", "a2", &["x1", "x2"], c).unwrap_err();
        assert!(!e.to_string().contains("propensity"), "{e}");
        expect_invalid_spec(
            e,
            &format!("treatment 'a2' must be coded 0/1 (0-based row {first} is 2)"),
        );
    }

    #[test]
    fn test_stdreg_notes_follow_display() {
        let ds = stdreg_sim();
        let r = fit_case(
            &ds,
            "yb",
            StdRegModel::Logistic,
            false,
            StdRegEstimand::ATE,
            1e-8,
        );
        let text = r.to_string();
        assert!(
            text.contains("SE Method:      Sandwich (Robust)\n"),
            "{text}"
        );
        assert!(text.ends_with(&crate::traits::convergence::notes_text(&r)));
        assert!(text.contains("Converged: yes ("));
        let b = run_stdreg(
            &ds,
            "yc",
            "a",
            &["x1", "x2"],
            StdRegConfig {
                n_bootstrap: 30,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(b.to_string().contains("SE Method:      Bootstrap (n = 30)"));
    }
}
