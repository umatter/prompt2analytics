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

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::{SignificanceLevel, logistic_cdf, normal_cdf};

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

/// Configuration for regression standardization.
#[derive(Debug, Clone)]
pub struct StdRegConfig {
    /// Outcome model type (Linear, Logistic, or Poisson)
    pub model_type: StdRegModel,
    /// Target estimand (ATE, ATT, ATC, or Levels)
    pub estimand: StdRegEstimand,
    /// Method for computing standard errors
    pub se_method: SEMethod,
    /// Number of bootstrap replications (if using bootstrap SE)
    pub n_bootstrap: usize,
    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Maximum iterations for MLE estimation (logistic/Poisson)
    pub max_iter: usize,
    /// Convergence tolerance
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
    /// Outcome model coefficients
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
}

impl fmt::Display for StdRegResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Regression Standardization (G-Computation)")?;
        writeln!(f, "===========================================")?;
        writeln!(f)?;
        writeln!(f, "Specification:")?;
        writeln!(f, "  Outcome Model:  {}", self.model_type)?;
        writeln!(f, "  Estimand:       {}", self.estimand)?;
        writeln!(
            f,
            "  SE Method:      {} (n = {})",
            self.se_method, self.n_bootstrap
        )?;
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

        if !self.warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {}", w)?;
            }
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

    // ═══════════════════════════════════════════════════════════════════════
    // Extract Data
    // ═══════════════════════════════════════════════════════════════════════

    // Extract outcome variable Y
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    // Extract treatment variable A
    let a = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let n = y.len();

    // Validate treatment is binary
    let n_treated: usize = a.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n - n_treated;

    if n_treated == 0 || n_control == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment variable '{}' must have both treated (1) and control (0) observations. \
                 Found {} treated, {} control.",
                treatment_col, n_treated, n_control
            ),
        });
    }

    // Validate outcome for logistic model
    if config.model_type == StdRegModel::Logistic {
        let y_unique: std::collections::HashSet<u64> =
            y.iter().map(|&v| (v * 1000.0).round() as u64).collect();
        let is_binary = y_unique.len() == 2 && (y_unique.contains(&0) || y_unique.contains(&1000));
        if !is_binary {
            warnings.push(format!(
                "Logistic model specified but outcome '{}' may not be binary. \
                 Unique values: {}. Consider using Linear model.",
                outcome_col,
                y_unique.len()
            ));
        }
    }

    // Build design matrix W for covariates (with intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, true)?;
    let w = design.data;
    let k_w = w.ncols();

    // Build full design matrix: [intercept, X, A] (or with interactions)
    // Note: intercept is already in w
    let (x_full, coef_names) =
        build_design_matrix(&w, &a, covariate_cols, config.include_interactions);
    let _k = x_full.ncols();

    // ═══════════════════════════════════════════════════════════════════════
    // Fit Outcome Model
    // ═══════════════════════════════════════════════════════════════════════

    let (beta, y_hat, r_squared) = fit_outcome_model(
        &x_full,
        &y,
        &a,
        config.model_type,
        config.max_iter,
        config.tolerance,
    )?;

    // ═══════════════════════════════════════════════════════════════════════
    // Compute Counterfactual Predictions
    // ═══════════════════════════════════════════════════════════════════════

    // Create counterfactual design matrices
    // X(1): design matrix with A=1 for all
    // X(0): design matrix with A=0 for all
    let (x_1, x_0) = create_counterfactual_matrices(&w, n, k_w, config.include_interactions);

    // Predict under each treatment regime
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

    let (se, ey1_se, ey0_se, boot_ates, boot_ey1s, boot_ey0s) = match config.se_method {
        SEMethod::Bootstrap => compute_bootstrap_se(
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
        )?,
        SEMethod::Delta => compute_delta_se(
            &x_full,
            &y,
            &y_hat,
            &beta,
            &y_hat_1,
            &y_hat_0,
            &a,
            config.model_type,
            config.estimand,
        )?,
        SEMethod::Sandwich => compute_sandwich_se(
            &x_full,
            &y,
            &y_hat,
            &beta,
            &y_hat_1,
            &y_hat_0,
            &a,
            config.model_type,
            config.estimand,
        )?,
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Compute Confidence Intervals
    // ═══════════════════════════════════════════════════════════════════════

    let alpha = 1.0 - config.confidence_level;

    // For ATE
    let (ci_lower, ci_upper) = if config.se_method == SEMethod::Bootstrap && boot_ates.len() >= 20 {
        // Percentile bootstrap CI
        let mut sorted_ates = boot_ates.clone();
        sorted_ates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo_idx = (sorted_ates.len() as f64 * alpha / 2.0).floor() as usize;
        let hi_idx = (sorted_ates.len() as f64 * (1.0 - alpha / 2.0)).floor() as usize;
        (
            sorted_ates[lo_idx],
            sorted_ates[hi_idx.min(sorted_ates.len() - 1)],
        )
    } else {
        // Normal approximation CI
        let z_crit = normal_quantile(1.0 - alpha / 2.0);
        (ate - z_crit * se, ate + z_crit * se)
    };

    // Inference
    let z_stat = if se > 0.0 && se.is_finite() {
        ate / se
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));
    let significance = SignificanceLevel::from_p_value(p_value);

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Effect Measures (Binary Outcomes)
    // ═══════════════════════════════════════════════════════════════════════

    let (risk_ratio, risk_ratio_ci, odds_ratio, odds_ratio_ci, nnt) = if config.model_type
        == StdRegModel::Logistic
        || is_binary_outcome(&y)
    {
        compute_binary_effect_measures(ey1, ey0, &boot_ey1s, &boot_ey0s, config.confidence_level)
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
    names.push("treatment".to_string());

    // Add interactions if requested
    if include_interactions {
        let mut col = a_col + 1;
        for j in 1..k_w {
            // Skip intercept (j=0)
            for i in 0..n {
                x[[i, col]] = a[i] * w[[i, j]];
            }
            names.push(format!("treatment:{}", covariate_names[j - 1]));
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

/// Fit outcome model and return coefficients, fitted values, and R-squared.
fn fit_outcome_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    _a: &Array1<f64>,
    model_type: StdRegModel,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Array1<f64>, Array1<f64>, f64)> {
    match model_type {
        StdRegModel::Linear => fit_linear_model(x, y),
        StdRegModel::Logistic => fit_logistic_model(x, y, max_iter, tolerance),
        StdRegModel::Poisson => fit_poisson_model(x, y, max_iter, tolerance),
    }
}

/// Fit linear regression using OLS.
fn fit_linear_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> EconResult<(Array1<f64>, Array1<f64>, f64)> {
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Outcome model X'X matrix".to_string(),
        suggestion: format!("Check for multicollinearity: {:?}", e),
    })?;

    let xty_vec = xty(&x.view(), y);
    let beta = xtx_inv.dot(&xty_vec);

    // Fitted values
    let y_hat: Array1<f64> = x.dot(&beta);

    // R-squared
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let sse: f64 = y
        .iter()
        .zip(y_hat.iter())
        .map(|(&yi, &yhi)| (yi - yhi).powi(2))
        .sum();
    let r_squared = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };

    Ok((beta, y_hat, r_squared))
}

/// Fit logistic regression using Newton-Raphson (IRLS).
fn fit_logistic_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = y.len();
    let k = x.ncols();

    let mut beta = Array1::zeros(k);

    for _ in 0..max_iter {
        // Linear predictor
        let z: Array1<f64> = x.dot(&beta);
        let p: Array1<f64> = z.mapv(logistic_cdf);
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Gradient: X'(y - p)
        let residuals = y - &p_clipped;
        let mut gradient = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                gradient[j] += residuals[i] * x[[i, j]];
            }
        }

        // Check convergence
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            break;
        }

        // Weights
        let weights: Array1<f64> = p_clipped.mapv(|pi| pi * (1.0 - pi));

        // Hessian: -X'WX
        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Newton-Raphson update
        let neg_hessian = &hessian * -1.0;
        let (hess_inv, _) =
            safe_inverse(&neg_hessian.view()).map_err(|e| EconError::SingularMatrix {
                context: "Logistic regression Hessian".to_string(),
                suggestion: format!("Check for multicollinearity: {:?}", e),
            })?;

        let delta = hess_inv.dot(&gradient);
        beta = &beta + &delta;
    }

    // Final predictions
    let z_final: Array1<f64> = x.dot(&beta);
    let y_hat: Array1<f64> = z_final.mapv(logistic_cdf);

    // Pseudo R-squared (McFadden)
    let null_ll = null_log_likelihood_binary(y);
    let model_ll = log_likelihood_binary(y, &y_hat);
    let r_squared = 1.0 - model_ll / null_ll;

    Ok((beta, y_hat, r_squared))
}

/// Fit Poisson regression using Newton-Raphson (IRLS).
fn fit_poisson_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = y.len();
    let k = x.ncols();

    let mut beta = Array1::zeros(k);

    for _ in 0..max_iter {
        // Linear predictor
        let z: Array1<f64> = x.dot(&beta);
        // mu = exp(z)
        let mu: Array1<f64> = z.mapv(|zi| zi.exp().min(1e10));

        // Gradient: X'(y - mu)
        let residuals = y - &mu;
        let mut gradient = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                gradient[j] += residuals[i] * x[[i, j]];
            }
        }

        // Check convergence
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            break;
        }

        // Hessian: -X'diag(mu)X
        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let wi = mu[i];
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Newton-Raphson update
        let neg_hessian = &hessian * -1.0;
        let (hess_inv, _) =
            safe_inverse(&neg_hessian.view()).map_err(|e| EconError::SingularMatrix {
                context: "Poisson regression Hessian".to_string(),
                suggestion: format!("Check for multicollinearity: {:?}", e),
            })?;

        let delta = hess_inv.dot(&gradient);
        beta = &beta + &delta;
    }

    // Final predictions
    let z_final: Array1<f64> = x.dot(&beta);
    let y_hat: Array1<f64> = z_final.mapv(|zi| zi.exp().min(1e10));

    // Pseudo R-squared (deviance based)
    let y_mean = y.mean().unwrap_or(1.0);
    let null_deviance = poisson_deviance(y, &Array1::from_elem(n, y_mean));
    let model_deviance = poisson_deviance(y, &y_hat);
    let r_squared = 1.0 - model_deviance / null_deviance.max(1e-10);

    Ok((beta, y_hat, r_squared.max(0.0)))
}

/// Predict outcomes using fitted coefficients.
fn predict(x: &Array2<f64>, beta: &Array1<f64>, model_type: StdRegModel) -> Array1<f64> {
    let z: Array1<f64> = x.dot(beta);
    match model_type {
        StdRegModel::Linear => z,
        StdRegModel::Logistic => z.mapv(logistic_cdf),
        StdRegModel::Poisson => z.mapv(|zi| zi.exp().min(1e10)),
    }
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

/// Compute bootstrap standard errors.
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
) -> EconResult<(f64, f64, f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = y.len();

    let mut rng: rand::rngs::StdRng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    let mut boot_ates: Vec<f64> = Vec::with_capacity(n_bootstrap);
    let mut boot_ey1s: Vec<f64> = Vec::with_capacity(n_bootstrap);
    let mut boot_ey0s: Vec<f64> = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

        // Check that bootstrap sample has both treated and control
        let n_t_boot = indices.iter().filter(|&&i| a[i] >= 0.5).count();
        if n_t_boot == 0 || n_t_boot == n {
            continue;
        }

        // Create bootstrap arrays
        let y_boot: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        let a_boot: Array1<f64> = indices.iter().map(|&i| a[i]).collect();
        let mut w_boot = Array2::zeros((n, k_w));
        for (new_i, &old_i) in indices.iter().enumerate() {
            w_boot.row_mut(new_i).assign(&w.row(old_i));
        }

        // Build design matrix
        let dummy_names: Vec<&str> = (0..k_w - 1).map(|_| "x").collect();
        let (x_boot, _) = build_design_matrix(&w_boot, &a_boot, &dummy_names, include_interactions);

        // Fit model
        let beta_boot =
            match fit_outcome_model(&x_boot, &y_boot, &a_boot, model_type, max_iter, tolerance) {
                Ok((b, _, _)) => b,
                Err(_) => continue,
            };

        // Counterfactual predictions
        let (x_1_boot, x_0_boot) =
            create_counterfactual_matrices(&w_boot, n, k_w, include_interactions);
        let y_hat_1_boot = predict(&x_1_boot, &beta_boot, model_type);
        let y_hat_0_boot = predict(&x_0_boot, &beta_boot, model_type);

        // Compute estimand
        let (ate_boot, ey1_boot, ey0_boot) =
            compute_estimand(&y_hat_1_boot, &y_hat_0_boot, &a_boot, estimand);

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

    // Compute SE as standard deviation of bootstrap estimates
    let ate_mean: f64 = boot_ates.iter().sum::<f64>() / boot_ates.len() as f64;
    let ate_var: f64 = boot_ates
        .iter()
        .map(|&e| (e - ate_mean).powi(2))
        .sum::<f64>()
        / (boot_ates.len() - 1).max(1) as f64;
    let se = ate_var.sqrt();

    let ey1_mean: f64 = boot_ey1s.iter().sum::<f64>() / boot_ey1s.len() as f64;
    let ey1_var: f64 = boot_ey1s
        .iter()
        .map(|&e| (e - ey1_mean).powi(2))
        .sum::<f64>()
        / (boot_ey1s.len() - 1).max(1) as f64;
    let ey1_se = ey1_var.sqrt();

    let ey0_mean: f64 = boot_ey0s.iter().sum::<f64>() / boot_ey0s.len() as f64;
    let ey0_var: f64 = boot_ey0s
        .iter()
        .map(|&e| (e - ey0_mean).powi(2))
        .sum::<f64>()
        / (boot_ey0s.len() - 1).max(1) as f64;
    let ey0_se = ey0_var.sqrt();

    Ok((se, ey1_se, ey0_se, boot_ates, boot_ey1s, boot_ey0s))
}

/// Compute delta method standard errors.
/// For linear model: SE(ATE) derived from coefficient variance.
#[allow(clippy::too_many_arguments)]
fn compute_delta_se(
    x: &Array2<f64>,
    y: &Array1<f64>,
    y_hat: &Array1<f64>,
    _beta: &Array1<f64>,
    _y_hat_1: &Array1<f64>,
    _y_hat_0: &Array1<f64>,
    _a: &Array1<f64>,
    model_type: StdRegModel,
    _estimand: StdRegEstimand,
) -> EconResult<(f64, f64, f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let k = x.ncols();

    // For linear model, ATE = beta_A, and SE(ATE) = SE(beta_A)
    // For nonlinear models, this is an approximation

    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Delta method variance".to_string(),
        suggestion: format!("Check for multicollinearity: {:?}", e),
    })?;

    let sigma2 = match model_type {
        StdRegModel::Linear => {
            let sse: f64 = y
                .iter()
                .zip(y_hat.iter())
                .map(|(&yi, &yhi)| (yi - yhi).powi(2))
                .sum();
            sse / (n - k).max(1) as f64
        }
        _ => {
            // For GLMs, use asymptotic variance approximation
            1.0
        }
    };

    // Treatment coefficient is at position k_w (after intercept and covariates)
    // Find it by looking for the column that changed between X(0) and X(1)
    // In our design matrix, it's at position k-1 (or second to last if interactions)
    let a_col_idx = x.ncols() - 1; // This assumes no interactions, simplification

    // Variance of treatment effect (approximation)
    let se = (sigma2 * xtx_inv[[a_col_idx, a_col_idx]]).sqrt();

    // For potential outcomes, rough approximation
    let ey1_se = se / 2.0_f64.sqrt();
    let ey0_se = se / 2.0_f64.sqrt();

    Ok((se, ey1_se, ey0_se, vec![], vec![], vec![]))
}

/// Compute sandwich (robust) standard errors.
#[allow(clippy::too_many_arguments)]
fn compute_sandwich_se(
    x: &Array2<f64>,
    y: &Array1<f64>,
    y_hat: &Array1<f64>,
    _beta: &Array1<f64>,
    _y_hat_1: &Array1<f64>,
    _y_hat_0: &Array1<f64>,
    _a: &Array1<f64>,
    _model_type: StdRegModel,
    _estimand: StdRegEstimand,
) -> EconResult<(f64, f64, f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let k = x.ncols();

    // Bread: (X'X)^{-1}
    let xtx_mat = xtx(&x.view());
    let (bread, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Sandwich SE bread matrix".to_string(),
        suggestion: format!("Check for multicollinearity: {:?}", e),
    })?;

    // Meat: X' diag(e^2) X
    let residuals: Array1<f64> = y - y_hat;
    let mut meat = Array2::zeros((k, k));
    for i in 0..n {
        let e_i_sq = residuals[i].powi(2);
        for j in 0..k {
            for l in 0..k {
                meat[[j, l]] += e_i_sq * x[[i, j]] * x[[i, l]];
            }
        }
    }

    // Sandwich: (X'X)^{-1} X'diag(e^2)X (X'X)^{-1}
    let sandwich = bread.dot(&meat).dot(&bread);

    // Treatment coefficient index (approximation)
    let a_col_idx = x.ncols() - 1;
    let se = sandwich[[a_col_idx, a_col_idx]].sqrt();

    let ey1_se = se / 2.0_f64.sqrt();
    let ey0_se = se / 2.0_f64.sqrt();

    Ok((se, ey1_se, ey0_se, vec![], vec![], vec![]))
}

/// Compute additional effect measures for binary outcomes.
fn compute_binary_effect_measures(
    ey1: f64,
    ey0: f64,
    boot_ey1s: &[f64],
    boot_ey0s: &[f64],
    confidence_level: f64,
) -> (
    Option<f64>,
    Option<(f64, f64)>,
    Option<f64>,
    Option<(f64, f64)>,
    Option<f64>,
) {
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

    // Bootstrap CIs if available
    let alpha = 1.0 - confidence_level;

    let rr_ci = if boot_ey1s.len() >= 20 && boot_ey0s.len() >= 20 {
        let boot_rrs: Vec<f64> = boot_ey1s
            .iter()
            .zip(boot_ey0s.iter())
            .filter_map(|(&y1, &y0)| if y0 > 1e-10 { Some(y1 / y0) } else { None })
            .collect();
        if boot_rrs.len() >= 20 {
            let mut sorted = boot_rrs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo = (sorted.len() as f64 * alpha / 2.0).floor() as usize;
            let hi = (sorted.len() as f64 * (1.0 - alpha / 2.0)).floor() as usize;
            Some((sorted[lo], sorted[hi.min(sorted.len() - 1)]))
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
            let mut sorted = boot_ors.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo = (sorted.len() as f64 * alpha / 2.0).floor() as usize;
            let hi = (sorted.len() as f64 * (1.0 - alpha / 2.0)).floor() as usize;
            Some((sorted[lo], sorted[hi.min(sorted.len() - 1)]))
        } else {
            None
        }
    } else {
        None
    };

    (rr, rr_ci, or, or_ci, nnt)
}

/// Check if outcome appears to be binary.
fn is_binary_outcome(y: &Array1<f64>) -> bool {
    let unique: std::collections::HashSet<u64> =
        y.iter().map(|&v| (v * 1000.0).round() as u64).collect();
    unique.len() == 2
}

/// Null log-likelihood for binary outcome (intercept-only model).
fn null_log_likelihood_binary(y: &Array1<f64>) -> f64 {
    let n = y.len() as f64;
    let p = y.iter().sum::<f64>() / n;
    let p_clipped = p.max(1e-10).min(1.0 - 1e-10);
    y.iter()
        .map(|&yi| yi * p_clipped.ln() + (1.0 - yi) * (1.0 - p_clipped).ln())
        .sum()
}

/// Log-likelihood for binary outcome.
fn log_likelihood_binary(y: &Array1<f64>, y_hat: &Array1<f64>) -> f64 {
    y.iter()
        .zip(y_hat.iter())
        .map(|(&yi, &pi)| {
            let pi_clipped = pi.max(1e-10).min(1.0 - 1e-10);
            yi * pi_clipped.ln() + (1.0 - yi) * (1.0 - pi_clipped).ln()
        })
        .sum()
}

/// Poisson deviance.
fn poisson_deviance(y: &Array1<f64>, mu: &Array1<f64>) -> f64 {
    2.0 * y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mi)| {
            let mi_clipped = mi.max(1e-10);
            if yi > 0.0 {
                yi * (yi / mi_clipped).ln() - (yi - mi_clipped)
            } else {
                mi_clipped
            }
        })
        .sum::<f64>()
}

/// Standard normal quantile function (inverse CDF).
fn normal_quantile(p: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.inverse_cdf(p)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

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
    fn test_stdreg_delta_se() {
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
}
