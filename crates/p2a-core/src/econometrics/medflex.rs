//! Natural Effect Models for Causal Mediation Analysis (medflex approach).
//!
//! Implements natural effect models that allow for exposure-mediator interactions,
//! following the methodology of Lange, Vansteelandt, and Bekaert (2012) and the
//! R package `medflex` by Steen et al. (2017).
//!
//! # Overview
//!
//! Natural effect models decompose the total effect of a treatment on an outcome
//! into Natural Direct Effects (NDE) and Natural Indirect Effects (NIE) while
//! allowing for treatment-mediator interactions.
//!
//! # Key Concepts
//!
//! - **Natural Direct Effect (NDE)**: Effect of treatment on outcome holding the
//!   mediator at the level it would naturally attain under the control condition
//!   NDE = E[Y(1, M(0))] - E[Y(0, M(0))]
//!
//! - **Natural Indirect Effect (NIE)**: Effect of changing the mediator from its
//!   natural level under control to its natural level under treatment, while
//!   holding treatment at the treatment level
//!   NIE = E[Y(1, M(1))] - E[Y(1, M(0))]
//!
//! - **Total Effect (TE) = NDE + NIE** (on the difference scale)
//!
//! # Mathematical Framework
//!
//! For treatment A, mediator M, outcome Y, and confounders C:
//!
//! 1. Fit mediator model:
//!    M = alpha_0 + alpha_1*A + alpha_2'*C + epsilon_M
//!
//! 2. Fit outcome model with interaction:
//!    Y = beta_0 + beta_1*A + beta_2*M + beta_3*A*M + beta_4'*C + epsilon_Y
//!
//! 3. Compute effects:
//!    - Without interaction (beta_3 = 0):
//!      NDE = beta_1
//!      NIE = alpha_1 * beta_2
//!
//!    - With interaction:
//!      NDE = beta_1 + beta_3 * E[M|A=0]
//!      NIE = alpha_1 * (beta_2 + beta_3)
//!
//! # Identification Assumptions
//!
//! 1. No unmeasured confounding of treatment-outcome relationship given C
//! 2. No unmeasured confounding of mediator-outcome relationship given (A, C)
//! 3. No unmeasured confounding of treatment-mediator relationship given C
//! 4. No effect of treatment on mediator-outcome confounders (cross-world independence)
//!
//! # References
//!
//! - Lange, T., Vansteelandt, S., & Bekaert, M. (2012). "Choice of effect measure
//!   for mediation analysis." *Epidemiology*, 23(6), 889-897.
//!   https://doi.org/10.1097/EDE.0b013e31826c2107
//!
//! - VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for
//!   Mediation and Interaction*. Oxford University Press.
//!
//! - Steen, J., Loeys, T., Moerkerke, B., & Vansteelandt, S. (2017). "medflex:
//!   An R Package for Flexible Mediation Analysis using Natural Effect Models."
//!   *Journal of Statistical Software*, 76(11), 1-46.
//!   https://doi.org/10.18637/jss.v076.i11
//!
//! - R package medflex: https://CRAN.R-project.org/package=medflex

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::linalg::{safe_inverse, xtx, xty};
use crate::traits::estimator::normal_cdf;

/// Scale for reporting mediation effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EffectScale {
    /// Effects as differences (additive scale) - default for continuous outcomes
    #[default]
    Difference,
    /// Effects as ratios (multiplicative scale) - for binary outcomes with log link
    Ratio,
    /// Effects as odds ratios - for binary outcomes with logit link
    OddsRatio,
}

impl fmt::Display for EffectScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Difference => write!(f, "Difference"),
            Self::Ratio => write!(f, "Ratio"),
            Self::OddsRatio => write!(f, "Odds Ratio"),
        }
    }
}

/// Configuration for Natural Effect Models (medflex).
#[derive(Debug, Clone)]
pub struct MedflexConfig {
    /// Allow treatment-mediator interaction in outcome model (default: true)
    pub allow_interaction: bool,
    /// Use bootstrap for confidence intervals (default: true)
    pub bootstrap_ci: bool,
    /// Number of bootstrap samples (default: 1000)
    pub n_bootstrap: usize,
    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,
    /// Effect scale for reporting (default: Difference)
    pub scale: EffectScale,
    /// Random seed for reproducibility (optional)
    pub seed: Option<u64>,
}

impl Default for MedflexConfig {
    fn default() -> Self {
        Self {
            allow_interaction: true,
            bootstrap_ci: true,
            n_bootstrap: 1000,
            confidence_level: 0.95,
            scale: EffectScale::Difference,
            seed: None,
        }
    }
}

/// Results from Natural Effect Models analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedflexResult {
    // ═══════════════════════════════════════════════════════════════════
    // Core Effect Estimates
    // ═══════════════════════════════════════════════════════════════════
    /// Total Effect (TE = NDE + NIE)
    pub total_effect: f64,
    /// Natural Direct Effect (NDE)
    pub natural_direct_effect: f64,
    /// Natural Indirect Effect (NIE)
    pub natural_indirect_effect: f64,
    /// Proportion of total effect mediated (NIE / TE)
    pub proportion_mediated: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Standard Errors
    // ═══════════════════════════════════════════════════════════════════
    /// Standard error of total effect
    pub te_se: f64,
    /// Standard error of NDE
    pub nde_se: f64,
    /// Standard error of NIE
    pub nie_se: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Confidence Intervals
    // ═══════════════════════════════════════════════════════════════════
    /// Confidence interval for total effect
    pub te_ci: (f64, f64),
    /// Confidence interval for NDE
    pub nde_ci: (f64, f64),
    /// Confidence interval for NIE
    pub nie_ci: (f64, f64),

    // ═══════════════════════════════════════════════════════════════════
    // p-values
    // ═══════════════════════════════════════════════════════════════════
    /// p-value for TE (two-sided test H0: TE = 0)
    pub te_p_value: f64,
    /// p-value for NDE (two-sided test H0: NDE = 0)
    pub nde_p_value: f64,
    /// p-value for NIE (two-sided test H0: NIE = 0)
    pub nie_p_value: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Model Coefficients
    // ═══════════════════════════════════════════════════════════════════
    /// Treatment effect on mediator (alpha_1 from mediator model)
    pub mediator_coef: f64,
    /// Treatment effect on mediator SE
    pub mediator_coef_se: f64,
    /// Direct treatment effect in outcome model (beta_1)
    pub direct_coef: f64,
    /// Direct treatment effect SE
    pub direct_coef_se: f64,
    /// Mediator effect on outcome (beta_2)
    pub mediator_outcome_coef: f64,
    /// Mediator effect on outcome SE
    pub mediator_outcome_coef_se: f64,
    /// Interaction coefficient (beta_3, None if no interaction)
    pub interaction_coef: Option<f64>,
    /// Interaction coefficient SE
    pub interaction_coef_se: Option<f64>,

    // ═══════════════════════════════════════════════════════════════════
    // Model Fit Statistics
    // ═══════════════════════════════════════════════════════════════════
    /// R-squared for mediator model
    pub mediator_r_squared: f64,
    /// R-squared for outcome model
    pub outcome_r_squared: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Sample Information
    // ═══════════════════════════════════════════════════════════════════
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Mean of mediator in control group (E[M|A=0])
    pub mediator_mean_control: f64,
    /// Mean of mediator in treated group (E[M|A=1])
    pub mediator_mean_treated: f64,

    // ═══════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════
    /// Whether interaction was included
    pub interaction_included: bool,
    /// Effect scale used
    pub scale: EffectScale,
    /// Confidence level
    pub confidence_level: f64,
}

impl fmt::Display for MedflexResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Natural Effect Models (medflex) ===")?;
        writeln!(f)?;
        writeln!(f, "Effect Decomposition ({}):", self.scale)?;
        writeln!(f, "{}", "─".repeat(70))?;
        writeln!(
            f,
            "{:<25} {:>12} {:>12} {:>16} {:>8}",
            "Effect",
            "Estimate",
            "Std.Err",
            &format!("{:.0}% CI", self.confidence_level * 100.0),
            "p-value"
        )?;
        writeln!(f, "{}", "─".repeat(70))?;

        let stars = |p: f64| -> &str {
            if p < 0.001 {
                "***"
            } else if p < 0.01 {
                "**"
            } else if p < 0.05 {
                "*"
            } else if p < 0.10 {
                "."
            } else {
                ""
            }
        };

        writeln!(
            f,
            "{:<25} {:>12.4} {:>12.4} [{:>6.3}, {:>6.3}] {:>7.4}{}",
            "Total Effect (TE)",
            self.total_effect,
            self.te_se,
            self.te_ci.0,
            self.te_ci.1,
            self.te_p_value,
            stars(self.te_p_value)
        )?;

        writeln!(
            f,
            "{:<25} {:>12.4} {:>12.4} [{:>6.3}, {:>6.3}] {:>7.4}{}",
            "Natural Direct Effect",
            self.natural_direct_effect,
            self.nde_se,
            self.nde_ci.0,
            self.nde_ci.1,
            self.nde_p_value,
            stars(self.nde_p_value)
        )?;

        writeln!(
            f,
            "{:<25} {:>12.4} {:>12.4} [{:>6.3}, {:>6.3}] {:>7.4}{}",
            "Natural Indirect Effect",
            self.natural_indirect_effect,
            self.nie_se,
            self.nie_ci.0,
            self.nie_ci.1,
            self.nie_p_value,
            stars(self.nie_p_value)
        )?;

        writeln!(f, "{}", "─".repeat(70))?;
        writeln!(f)?;

        writeln!(
            f,
            "Proportion Mediated (NIE/TE): {:.1}%",
            self.proportion_mediated * 100.0
        )?;
        writeln!(f)?;

        writeln!(f, "Model Components:")?;
        writeln!(f, "  Mediator Model (M ~ A + C):")?;
        writeln!(
            f,
            "    Treatment -> Mediator: {:.4} (SE = {:.4})",
            self.mediator_coef, self.mediator_coef_se
        )?;
        writeln!(f, "    R-squared: {:.4}", self.mediator_r_squared)?;
        writeln!(f)?;

        if self.interaction_included {
            writeln!(f, "  Outcome Model (Y ~ A + M + A*M + C):")?;
        } else {
            writeln!(f, "  Outcome Model (Y ~ A + M + C):")?;
        }
        writeln!(
            f,
            "    Treatment -> Outcome: {:.4} (SE = {:.4})",
            self.direct_coef, self.direct_coef_se
        )?;
        writeln!(
            f,
            "    Mediator -> Outcome:  {:.4} (SE = {:.4})",
            self.mediator_outcome_coef, self.mediator_outcome_coef_se
        )?;
        if let (Some(int_coef), Some(int_se)) = (self.interaction_coef, self.interaction_coef_se) {
            writeln!(
                f,
                "    A*M Interaction:      {:.4} (SE = {:.4})",
                int_coef, int_se
            )?;
        }
        writeln!(f, "    R-squared: {:.4}", self.outcome_r_squared)?;
        writeln!(f)?;

        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  N = {} (treated: {}, control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(
            f,
            "  E[M|A=0] = {:.4}, E[M|A=1] = {:.4}",
            self.mediator_mean_control, self.mediator_mean_treated
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Signif. codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10"
        )?;

        Ok(())
    }
}

/// Run Natural Effect Models for causal mediation analysis.
///
/// Implements the regression-based approach to mediation analysis with
/// exposure-mediator interactions, following the medflex methodology.
///
/// # Arguments
///
/// * `y` - Outcome variable (n x 1)
/// * `treatment` - Treatment/exposure variable (n x 1), typically binary
/// * `mediator` - Mediator variable (n x 1)
/// * `confounders` - Confounder matrix (n x p), can be empty (n x 0)
/// * `config` - Configuration options
///
/// # Returns
///
/// `MedflexResult` containing effect estimates, standard errors, CIs, and model statistics
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::medflex::{run_medflex, MedflexConfig};
/// use ndarray::array;
///
/// let y = array![1.2, 2.1, 1.5, 3.0, 2.8, 1.1, 2.5, 3.2];
/// let treatment = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
/// let mediator = array![0.5, 1.2, 0.6, 1.5, 1.3, 0.4, 1.4, 1.6];
/// let confounders = Array2::zeros((8, 0));  // No confounders
///
/// let config = MedflexConfig::default();
/// let result = run_medflex(&y.view(), &treatment.view(), &mediator.view(),
///                          &confounders.view(), config)?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// - Steen, J., et al. (2017). "medflex: An R Package for Flexible Mediation
///   Analysis using Natural Effect Models." Journal of Statistical Software.
/// - VanderWeele, T. J. (2015). Explanation in Causal Inference. Oxford Univ. Press.
pub fn run_medflex(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    mediator: &ArrayView1<f64>,
    confounders: &ArrayView2<f64>,
    config: MedflexConfig,
) -> EconResult<MedflexResult> {
    let n = y.len();

    // Validate inputs
    if treatment.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment length ({}) must match outcome length ({})",
                treatment.len(),
                n
            ),
        });
    }
    if mediator.len() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Mediator length ({}) must match outcome length ({})",
                mediator.len(),
                n
            ),
        });
    }
    if confounders.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Confounders rows ({}) must match outcome length ({})",
                confounders.nrows(),
                n
            ),
        });
    }

    let min_obs = if config.allow_interaction { 10 } else { 8 };
    if n < min_obs {
        return Err(EconError::InsufficientData {
            required: min_obs,
            provided: n,
            context: "Natural effect models require sufficient observations".to_string(),
        });
    }

    // Count treated and control observations
    let n_treated = treatment.iter().filter(|&&a| a >= 0.5).count();
    let n_control = n - n_treated;

    if n_treated < 3 || n_control < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n_treated.min(n_control),
            context: "Need at least 3 observations in each treatment group".to_string(),
        });
    }

    // Compute mediator means by treatment group
    let mut sum_m_treated = 0.0;
    let mut sum_m_control = 0.0;
    for i in 0..n {
        if treatment[i] >= 0.5 {
            sum_m_treated += mediator[i];
        } else {
            sum_m_control += mediator[i];
        }
    }
    let mediator_mean_treated = sum_m_treated / (n_treated as f64);
    let mediator_mean_control = sum_m_control / (n_control as f64);

    // Step 1: Fit mediator model: M = alpha_0 + alpha_1*A + alpha_2'*C + epsilon
    let (alpha, alpha_se, mediator_r2) = fit_mediator_model(mediator, treatment, confounders)?;

    // alpha[0] = intercept, alpha[1] = treatment effect, alpha[2..] = confounders
    let alpha_1 = alpha[1];
    let alpha_1_se = alpha_se[1];

    // Step 2: Fit outcome model with or without interaction
    let (beta, beta_se, outcome_r2) = fit_outcome_model(
        y,
        treatment,
        mediator,
        confounders,
        config.allow_interaction,
    )?;

    // Extract coefficients
    // Without interaction: Y = beta_0 + beta_1*A + beta_2*M + beta_3'*C
    // With interaction:    Y = beta_0 + beta_1*A + beta_2*M + beta_3*A*M + beta_4'*C
    let beta_1 = beta[1]; // Treatment coefficient
    let beta_1_se = beta_se[1];
    let beta_2 = beta[2]; // Mediator coefficient
    let beta_2_se = beta_se[2];

    let (beta_3, beta_3_se) = if config.allow_interaction {
        (Some(beta[3]), Some(beta_se[3]))
    } else {
        (None, None)
    };

    // Step 3: Compute Natural Direct and Indirect Effects
    // Using formulas from VanderWeele (2015) and medflex documentation
    let (nde, nie, te) =
        compute_natural_effects(alpha_1, beta_1, beta_2, beta_3, mediator_mean_control);

    // Step 4: Bootstrap for standard errors and confidence intervals
    let (te_se, nde_se, nie_se, te_ci, nde_ci, nie_ci) = if config.bootstrap_ci {
        bootstrap_effects(
            y,
            treatment,
            mediator,
            confounders,
            config.allow_interaction,
            config.n_bootstrap,
            config.confidence_level,
            config.seed,
        )?
    } else {
        // Delta method approximation for standard errors
        delta_method_se(
            alpha_1,
            alpha_1_se,
            beta_1,
            beta_1_se,
            beta_2,
            beta_2_se,
            beta_3,
            beta_3_se,
            mediator_mean_control,
            config.confidence_level,
        )
    };

    // Compute p-values (two-sided test against zero)
    let te_p = compute_p_value(te, te_se);
    let nde_p = compute_p_value(nde, nde_se);
    let nie_p = compute_p_value(nie, nie_se);

    // Proportion mediated
    let proportion_mediated = if te.abs() > 1e-10 {
        (nie / te).clamp(-2.0, 2.0) // Clamp to reasonable range
    } else {
        f64::NAN
    };

    Ok(MedflexResult {
        total_effect: te,
        natural_direct_effect: nde,
        natural_indirect_effect: nie,
        proportion_mediated,
        te_se,
        nde_se,
        nie_se,
        te_ci,
        nde_ci,
        nie_ci,
        te_p_value: te_p,
        nde_p_value: nde_p,
        nie_p_value: nie_p,
        mediator_coef: alpha_1,
        mediator_coef_se: alpha_1_se,
        direct_coef: beta_1,
        direct_coef_se: beta_1_se,
        mediator_outcome_coef: beta_2,
        mediator_outcome_coef_se: beta_2_se,
        interaction_coef: beta_3,
        interaction_coef_se: beta_3_se,
        mediator_r_squared: mediator_r2,
        outcome_r_squared: outcome_r2,
        n_obs: n,
        n_treated,
        n_control,
        mediator_mean_control,
        mediator_mean_treated,
        interaction_included: config.allow_interaction,
        scale: config.scale,
        confidence_level: config.confidence_level,
    })
}

/// Fit the mediator model: M = alpha_0 + alpha_1*A + alpha_2'*C + epsilon
///
/// Returns (coefficients, standard_errors, r_squared)
fn fit_mediator_model(
    mediator: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    confounders: &ArrayView2<f64>,
) -> EconResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = mediator.len();
    let p_conf = confounders.ncols();
    let k = 2 + p_conf; // intercept + treatment + confounders

    // Build design matrix X = [1, A, C]
    let mut x = Array2::zeros((n, k));
    for i in 0..n {
        x[[i, 0]] = 1.0; // Intercept
        x[[i, 1]] = treatment[i];
        for j in 0..p_conf {
            x[[i, 2 + j]] = confounders[[i, j]];
        }
    }

    // Convert mediator to Array1
    let m: Array1<f64> = mediator.iter().cloned().collect();

    // OLS: beta = (X'X)^{-1} X'y
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|_| EconError::SingularMatrix {
        context: "X'X in mediator model".to_string(),
        suggestion: "Check for collinearity among treatment and confounders".to_string(),
    })?;

    let xty_vec = xty(&x.view(), &m);
    let alpha = xtx_inv.dot(&xty_vec);

    // Compute residuals and sigma^2
    let m_hat = x.dot(&alpha);
    let residuals: Array1<f64> = &m - &m_hat;
    let ssr: f64 = residuals.iter().map(|&e| e * e).sum();
    let df_resid = n - k;
    let sigma2 = ssr / (df_resid as f64);

    // Standard errors
    let vcov = &xtx_inv * sigma2;
    let se: Array1<f64> = (0..k).map(|i| vcov[[i, i]].sqrt()).collect();

    // R-squared
    let m_mean = m.mean().unwrap_or(0.0);
    let sst: f64 = m.iter().map(|&mi| (mi - m_mean).powi(2)).sum();
    let r2 = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    Ok((alpha, se, r2))
}

/// Fit the outcome model with or without treatment-mediator interaction.
///
/// Without interaction: Y = beta_0 + beta_1*A + beta_2*M + beta_3'*C
/// With interaction:    Y = beta_0 + beta_1*A + beta_2*M + beta_3*A*M + beta_4'*C
///
/// Returns (coefficients, standard_errors, r_squared)
fn fit_outcome_model(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    mediator: &ArrayView1<f64>,
    confounders: &ArrayView2<f64>,
    include_interaction: bool,
) -> EconResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = y.len();
    let p_conf = confounders.ncols();
    let k = if include_interaction {
        4 + p_conf // intercept + A + M + A*M + confounders
    } else {
        3 + p_conf // intercept + A + M + confounders
    };

    // Build design matrix
    let mut x = Array2::zeros((n, k));
    for i in 0..n {
        x[[i, 0]] = 1.0; // Intercept
        x[[i, 1]] = treatment[i];
        x[[i, 2]] = mediator[i];

        let mut col_idx = 3;
        if include_interaction {
            x[[i, 3]] = treatment[i] * mediator[i]; // A*M interaction
            col_idx = 4;
        }

        for j in 0..p_conf {
            x[[i, col_idx + j]] = confounders[[i, j]];
        }
    }

    // Convert y to Array1
    let y_arr: Array1<f64> = y.iter().cloned().collect();

    // OLS: beta = (X'X)^{-1} X'y
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|_| EconError::SingularMatrix {
        context: "X'X in outcome model".to_string(),
        suggestion: "Check for collinearity in treatment, mediator, and confounders".to_string(),
    })?;

    let xty_vec = xty(&x.view(), &y_arr);
    let beta = xtx_inv.dot(&xty_vec);

    // Compute residuals and sigma^2
    let y_hat = x.dot(&beta);
    let residuals: Array1<f64> = &y_arr - &y_hat;
    let ssr: f64 = residuals.iter().map(|&e| e * e).sum();
    let df_resid = n - k;
    let sigma2 = ssr / (df_resid as f64);

    // Standard errors
    let vcov = &xtx_inv * sigma2;
    let se: Array1<f64> = (0..k).map(|i| vcov[[i, i]].sqrt()).collect();

    // R-squared
    let y_mean = y_arr.mean().unwrap_or(0.0);
    let sst: f64 = y_arr.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r2 = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    Ok((beta, se, r2))
}

/// Compute Natural Direct and Indirect Effects.
///
/// Formulas from VanderWeele (2015), Chapter 2:
/// - Without interaction (beta_3 = 0):
///   NDE = beta_1
///   NIE = alpha_1 * beta_2
///
/// - With interaction:
///   NDE = beta_1 + beta_3 * E[M|A=0]
///   NIE = alpha_1 * (beta_2 + beta_3)
///
/// Total Effect = NDE + NIE
fn compute_natural_effects(
    alpha_1: f64,               // Treatment effect on mediator
    beta_1: f64,                // Direct treatment effect on outcome
    beta_2: f64,                // Mediator effect on outcome
    beta_3: Option<f64>,        // Interaction coefficient (if included)
    mediator_mean_control: f64, // E[M|A=0]
) -> (f64, f64, f64) {
    match beta_3 {
        Some(interaction) => {
            // With interaction (VanderWeele 2015, Eq. 2.6-2.7)
            // NDE = beta_1 + beta_3 * E[M|A=0]
            let nde = beta_1 + interaction * mediator_mean_control;
            // NIE = alpha_1 * (beta_2 + beta_3)
            let nie = alpha_1 * (beta_2 + interaction);
            let te = nde + nie;
            (nde, nie, te)
        }
        None => {
            // Without interaction (product of coefficients method)
            // NDE = beta_1
            let nde = beta_1;
            // NIE = alpha_1 * beta_2
            let nie = alpha_1 * beta_2;
            let te = nde + nie;
            (nde, nie, te)
        }
    }
}

/// Bootstrap confidence intervals for natural effects.
fn bootstrap_effects(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    mediator: &ArrayView1<f64>,
    confounders: &ArrayView2<f64>,
    include_interaction: bool,
    n_bootstrap: usize,
    confidence_level: f64,
    seed: Option<u64>,
) -> EconResult<(f64, f64, f64, (f64, f64), (f64, f64), (f64, f64))> {
    let n = y.len();

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    let mut boot_te = Vec::with_capacity(n_bootstrap);
    let mut boot_nde = Vec::with_capacity(n_bootstrap);
    let mut boot_nie = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

        // Create bootstrap sample
        let y_boot: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        let t_boot: Array1<f64> = indices.iter().map(|&i| treatment[i]).collect();
        let m_boot: Array1<f64> = indices.iter().map(|&i| mediator[i]).collect();
        let c_boot: Array2<f64> = Array2::from_shape_fn((n, confounders.ncols()), |(row, col)| {
            confounders[[indices[row], col]]
        });

        // Compute mediator mean in control group for bootstrap sample
        let mut sum_m_ctrl = 0.0;
        let mut n_ctrl = 0;
        for i in 0..n {
            if t_boot[i] < 0.5 {
                sum_m_ctrl += m_boot[i];
                n_ctrl += 1;
            }
        }
        let m_mean_ctrl = if n_ctrl > 0 {
            sum_m_ctrl / (n_ctrl as f64)
        } else {
            0.0
        };

        // Fit models on bootstrap sample
        if let (Ok((alpha, _, _)), Ok((beta, _, _))) = (
            fit_mediator_model(&m_boot.view(), &t_boot.view(), &c_boot.view()),
            fit_outcome_model(
                &y_boot.view(),
                &t_boot.view(),
                &m_boot.view(),
                &c_boot.view(),
                include_interaction,
            ),
        ) {
            let alpha_1 = alpha[1];
            let beta_1 = beta[1];
            let beta_2 = beta[2];
            let beta_3 = if include_interaction {
                Some(beta[3])
            } else {
                None
            };

            let (nde, nie, te) =
                compute_natural_effects(alpha_1, beta_1, beta_2, beta_3, m_mean_ctrl);

            if te.is_finite() && nde.is_finite() && nie.is_finite() {
                boot_te.push(te);
                boot_nde.push(nde);
                boot_nie.push(nie);
            }
        }
    }

    if boot_te.len() < 50 {
        return Err(EconError::ConvergenceFailure {
            iterations: n_bootstrap,
            last_change: 0.0,
            suggestion: "Too few valid bootstrap samples. Check data for issues.".to_string(),
        });
    }

    // Compute standard errors and percentile CIs
    let (te_se, te_ci) = bootstrap_stats(&boot_te, confidence_level);
    let (nde_se, nde_ci) = bootstrap_stats(&boot_nde, confidence_level);
    let (nie_se, nie_ci) = bootstrap_stats(&boot_nie, confidence_level);

    Ok((te_se, nde_se, nie_se, te_ci, nde_ci, nie_ci))
}

/// Compute bootstrap standard error and percentile confidence interval.
fn bootstrap_stats(samples: &[f64], confidence_level: f64) -> (f64, (f64, f64)) {
    if samples.is_empty() {
        return (f64::NAN, (f64::NAN, f64::NAN));
    }

    let n = samples.len() as f64;
    let mean: f64 = samples.iter().sum::<f64>() / n;
    let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = variance.sqrt();

    // Percentile CI
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0 * n) as usize).max(0).min(samples.len() - 1);
    let upper_idx = (((1.0 - alpha / 2.0) * n) as usize)
        .max(0)
        .min(samples.len() - 1);

    (se, (sorted[lower_idx], sorted[upper_idx]))
}

/// Delta method approximation for standard errors (when bootstrap is disabled).
fn delta_method_se(
    alpha_1: f64,
    alpha_1_se: f64,
    beta_1: f64,
    beta_1_se: f64,
    beta_2: f64,
    beta_2_se: f64,
    beta_3: Option<f64>,
    beta_3_se: Option<f64>,
    mediator_mean_control: f64,
    confidence_level: f64,
) -> (f64, f64, f64, (f64, f64), (f64, f64), (f64, f64)) {
    // Using delta method for product of independent estimates
    // Var(ab) approx= a^2 * Var(b) + b^2 * Var(a)

    match (beta_3, beta_3_se) {
        (Some(b3), Some(b3_se)) => {
            // With interaction:
            // NDE = beta_1 + beta_3 * E[M|A=0]
            // NIE = alpha_1 * (beta_2 + beta_3)

            // SE(NDE) using delta method (ignoring uncertainty in E[M|A=0])
            let nde_var = beta_1_se.powi(2) + (mediator_mean_control * b3_se).powi(2);
            let nde_se = nde_var.sqrt();

            // SE(NIE) = SE(alpha_1 * (beta_2 + beta_3))
            // Using delta method: approximately alpha_1^2 * (Var(beta_2) + Var(beta_3)) + (beta_2+beta_3)^2 * Var(alpha_1)
            let sum_beta = beta_2 + b3;
            let nie_var = alpha_1.powi(2) * (beta_2_se.powi(2) + b3_se.powi(2))
                + sum_beta.powi(2) * alpha_1_se.powi(2);
            let nie_se = nie_var.sqrt();

            // SE(TE) = sqrt(Var(NDE) + Var(NIE) + 2*Cov(NDE,NIE))
            // Ignoring covariance as approximation
            let te_se = (nde_se.powi(2) + nie_se.powi(2)).sqrt();

            let z = z_critical(confidence_level);
            let nde = beta_1 + b3 * mediator_mean_control;
            let nie = alpha_1 * sum_beta;
            let te = nde + nie;

            (
                te_se,
                nde_se,
                nie_se,
                (te - z * te_se, te + z * te_se),
                (nde - z * nde_se, nde + z * nde_se),
                (nie - z * nie_se, nie + z * nie_se),
            )
        }
        _ => {
            // Without interaction:
            // NDE = beta_1
            // NIE = alpha_1 * beta_2

            let nde_se = beta_1_se;

            // SE(NIE) = SE(alpha_1 * beta_2) using delta method
            let nie_var = alpha_1.powi(2) * beta_2_se.powi(2) + beta_2.powi(2) * alpha_1_se.powi(2);
            let nie_se = nie_var.sqrt();

            let te_se = (nde_se.powi(2) + nie_se.powi(2)).sqrt();

            let z = z_critical(confidence_level);
            let nde = beta_1;
            let nie = alpha_1 * beta_2;
            let te = nde + nie;

            (
                te_se,
                nde_se,
                nie_se,
                (te - z * te_se, te + z * te_se),
                (nde - z * nde_se, nde + z * nde_se),
                (nie - z * nie_se, nie + z * nie_se),
            )
        }
    }
}

/// Compute z critical value for given confidence level.
fn z_critical(confidence_level: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.inverse_cdf(1.0 - (1.0 - confidence_level) / 2.0)
}

/// Compute two-sided p-value assuming normal distribution.
fn compute_p_value(estimate: f64, se: f64) -> f64 {
    if se <= 0.0 || !se.is_finite() || !estimate.is_finite() {
        return f64::NAN;
    }
    let z = estimate.abs() / se;
    2.0 * (1.0 - normal_cdf(z))
}

// ============================================================================
// Dataset-based interface
// ============================================================================

use crate::data::Dataset;

/// Run Natural Effect Models on a dataset.
///
/// This is a convenience wrapper around `run_medflex` that extracts data
/// from a Dataset using column names.
///
/// # Arguments
///
/// * `dataset` - The dataset containing all variables
/// * `outcome` - Name of the outcome variable column
/// * `treatment` - Name of the treatment variable column
/// * `mediator` - Name of the mediator variable column
/// * `confounders` - Names of confounder columns (can be empty)
/// * `config` - Configuration options
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::medflex::{run_medflex_dataset, MedflexConfig};
///
/// let config = MedflexConfig {
///     allow_interaction: true,
///     n_bootstrap: 500,
///     ..Default::default()
/// };
///
/// let result = run_medflex_dataset(
///     &dataset, "outcome", "treatment", "mediator",
///     &["age", "gender"], config
/// )?;
/// println!("{}", result);
/// ```
pub fn run_medflex_dataset(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    mediator: &str,
    confounders: &[&str],
    config: MedflexConfig,
) -> EconResult<MedflexResult> {
    let df = dataset.df();
    let n = df.height();
    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Extract outcome
    let y_col = df.column(outcome).map_err(|_| EconError::ColumnNotFound {
        column: outcome.to_string(),
        available: available_cols.clone(),
    })?;
    let y: Array1<f64> = y_col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: outcome.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Extract treatment
    let t_col = df
        .column(treatment)
        .map_err(|_| EconError::ColumnNotFound {
            column: treatment.to_string(),
            available: available_cols.clone(),
        })?;
    let t: Array1<f64> = t_col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: treatment.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Extract mediator
    let m_col = df.column(mediator).map_err(|_| EconError::ColumnNotFound {
        column: mediator.to_string(),
        available: available_cols.clone(),
    })?;
    let m: Array1<f64> = m_col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: mediator.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Extract confounders
    let mut c_data: Vec<f64> = Vec::with_capacity(n * confounders.len());
    for conf_name in confounders {
        let col = df
            .column(conf_name)
            .map_err(|_| EconError::ColumnNotFound {
                column: conf_name.to_string(),
                available: available_cols.clone(),
            })?;
        let vals: Vec<f64> = col
            .f64()
            .map_err(|_| EconError::NonNumericColumn {
                column: conf_name.to_string(),
            })?
            .into_no_null_iter()
            .collect();
        c_data.extend(vals);
    }

    let c = if confounders.is_empty() {
        Array2::zeros((n, 0))
    } else {
        // Data is column-major from extraction, reshape accordingly
        Array2::from_shape_vec((confounders.len(), n), c_data)
            .map_err(|e| EconError::Internal(format!("Failed to create confounder matrix: {}", e)))?
            .t()
            .to_owned()
    };

    run_medflex(&y.view(), &t.view(), &m.view(), &c.view(), config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use polars::prelude::*;

    fn create_test_data() -> (Array1<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
        // Synthetic data with known mediation structure:
        // True DGP:
        //   M = 0.5 + 0.6*A + noise  (alpha_1 = 0.6)
        //   Y = 1.0 + 0.4*A + 0.5*M + 0.2*A*M + noise
        //
        // Expected (approximately):
        //   Without interaction: NDE = 0.4, NIE = 0.6*0.5 = 0.3, TE = 0.7
        //   With interaction: more complex

        let n = 100;
        let y = array![
            // Treated (A=1) with higher Y on average
            2.1, 2.3, 2.0, 2.5, 2.2, 2.4, 2.1, 2.6, 2.3, 2.2, 2.0, 2.4, 2.5, 2.1, 2.3, 2.6, 2.2,
            2.4, 2.0, 2.5, 2.3, 2.1, 2.4, 2.2, 2.5, 2.0, 2.3, 2.6, 2.1, 2.4, 2.2, 2.5, 2.0, 2.3,
            2.1, 2.6, 2.4, 2.2, 2.5, 2.3, 2.1, 2.4, 2.2, 2.0, 2.5, 2.3, 2.6, 2.1, 2.4, 2.2,
            // Control (A=0) with lower Y on average
            1.3, 1.5, 1.2, 1.6, 1.4, 1.3, 1.5, 1.2, 1.4, 1.6, 1.3, 1.5, 1.4, 1.2, 1.6, 1.3, 1.5,
            1.4, 1.2, 1.6, 1.4, 1.3, 1.5, 1.2, 1.6, 1.4, 1.3, 1.5, 1.2, 1.6, 1.5, 1.3, 1.4, 1.2,
            1.6, 1.5, 1.3, 1.4, 1.2, 1.6, 1.4, 1.5, 1.3, 1.2, 1.6, 1.4, 1.5, 1.3, 1.2, 1.6
        ];

        let treatment = array![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ];

        let mediator = array![
            // Treated have higher M (around 1.1 = 0.5 + 0.6)
            1.0, 1.2, 0.9, 1.3, 1.1, 1.2, 1.0, 1.4, 1.1, 1.0, 0.9, 1.2, 1.3, 1.0, 1.1, 1.4, 1.0,
            1.2, 0.9, 1.3, 1.1, 0.9, 1.2, 1.0, 1.3, 0.9, 1.1, 1.4, 0.9, 1.2, 1.0, 1.3, 0.9, 1.1,
            0.9, 1.4, 1.2, 1.0, 1.3, 1.1, 0.9, 1.2, 1.0, 0.9, 1.3, 1.1, 1.4, 0.9, 1.2, 1.0,
            // Control have lower M (around 0.5)
            0.4, 0.6, 0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 0.5, 0.7, 0.4, 0.6, 0.5, 0.3, 0.7, 0.4, 0.6,
            0.5, 0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 0.7, 0.6, 0.4, 0.5, 0.3,
            0.7, 0.6, 0.4, 0.5, 0.3, 0.7, 0.5, 0.6, 0.4, 0.3, 0.7, 0.5, 0.6, 0.4, 0.3, 0.7
        ];

        // No confounders for simplicity
        let confounders = Array2::zeros((n, 0));

        (y, treatment, mediator, confounders)
    }

    #[test]
    fn test_medflex_basic() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            allow_interaction: false,
            bootstrap_ci: true,
            n_bootstrap: 200, // Reduced for faster tests
            seed: Some(42),
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        // Check basic properties
        assert!(result.total_effect > 0.0, "Total effect should be positive");
        assert!(result.n_obs == 100, "Should have 100 observations");
        assert!(result.n_treated == 50, "Should have 50 treated");
        assert!(result.n_control == 50, "Should have 50 control");

        // Check decomposition: TE = NDE + NIE
        let decomp_error =
            (result.total_effect - result.natural_direct_effect - result.natural_indirect_effect)
                .abs();
        assert!(
            decomp_error < 1e-10,
            "Decomposition error {} too large",
            decomp_error
        );

        // Check that SEs are positive
        assert!(result.te_se > 0.0, "TE SE should be positive");
        assert!(result.nde_se > 0.0, "NDE SE should be positive");
        assert!(result.nie_se > 0.0, "NIE SE should be positive");

        // Check confidence intervals contain point estimates
        assert!(result.te_ci.0 <= result.total_effect, "TE CI lower bound");
        assert!(result.te_ci.1 >= result.total_effect, "TE CI upper bound");
    }

    #[test]
    fn test_medflex_with_interaction() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            allow_interaction: true,
            bootstrap_ci: true,
            n_bootstrap: 200,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        // Check that interaction coefficient is present
        assert!(
            result.interaction_coef.is_some(),
            "Interaction coefficient should be present"
        );
        assert!(
            result.interaction_included,
            "Should flag interaction as included"
        );

        // Decomposition should still hold
        let decomp_error =
            (result.total_effect - result.natural_direct_effect - result.natural_indirect_effect)
                .abs();
        assert!(
            decomp_error < 1e-10,
            "Decomposition error {} too large",
            decomp_error
        );
    }

    #[test]
    fn test_medflex_with_confounders() {
        // Create data with a confounder
        let n = 80;
        let y = Array1::from_iter((0..n).map(|i| {
            if i < 40 {
                2.0 + 0.1 * (i as f64 % 10.0)
            } else {
                1.3 + 0.1 * (i as f64 % 10.0)
            }
        }));
        let treatment = Array1::from_iter((0..n).map(|i| if i < 40 { 1.0 } else { 0.0 }));
        let mediator = Array1::from_iter((0..n).map(|i| {
            if i < 40 {
                1.1 + 0.05 * (i as f64 % 10.0)
            } else {
                0.5 + 0.05 * (i as f64 % 10.0)
            }
        }));
        let confounders = Array2::from_shape_fn((n, 1), |(i, _)| 0.5 + 0.1 * (i as f64 % 10.0));

        let config = MedflexConfig {
            allow_interaction: false,
            bootstrap_ci: false, // Use delta method for speed
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        // Should complete without error
        assert!(result.total_effect.is_finite());
        assert!(result.natural_direct_effect.is_finite());
        assert!(result.natural_indirect_effect.is_finite());
    }

    #[test]
    fn test_medflex_delta_method() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            allow_interaction: false,
            bootstrap_ci: false, // Use delta method
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        // Standard errors should be computed
        assert!(result.te_se > 0.0, "TE SE should be positive");
        assert!(result.nde_se > 0.0, "NDE SE should be positive");
        assert!(result.nie_se > 0.0, "NIE SE should be positive");

        // CIs should be valid
        assert!(
            result.te_ci.0 < result.te_ci.1,
            "CI should have lower < upper"
        );
    }

    #[test]
    fn test_medflex_display() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            n_bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        let display = result.to_string();
        assert!(display.contains("Natural Effect Models"));
        assert!(display.contains("Total Effect"));
        assert!(display.contains("Natural Direct Effect"));
        assert!(display.contains("Natural Indirect Effect"));
        assert!(display.contains("Proportion Mediated"));
    }

    #[test]
    fn test_medflex_dataset_interface() {
        let df = df! {
            "y" => [2.1, 2.3, 2.0, 1.3, 1.5, 1.2, 2.2, 2.4, 1.4, 1.6,
                   2.0, 2.5, 1.3, 1.5, 2.3, 2.1, 1.4, 1.2, 2.4, 2.2],
            "treatment" => [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                           1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "mediator" => [1.0, 1.2, 0.9, 0.4, 0.6, 0.3, 1.1, 1.3, 0.5, 0.7,
                          0.9, 1.3, 0.4, 0.6, 1.1, 0.9, 0.5, 0.3, 1.2, 1.0],
            "age" => [30.0, 35.0, 28.0, 42.0, 38.0, 31.0, 33.0, 40.0, 36.0, 29.0,
                     34.0, 37.0, 41.0, 32.0, 39.0, 27.0, 43.0, 35.0, 30.0, 38.0]
        }
        .unwrap();

        let ds = Dataset::new(df);

        let config = MedflexConfig {
            bootstrap_ci: false,
            ..Default::default()
        };

        let result =
            run_medflex_dataset(&ds, "y", "treatment", "mediator", &["age"], config).unwrap();

        assert_eq!(result.n_obs, 20);
        assert!(result.total_effect.is_finite());
    }

    #[test]
    fn test_medflex_insufficient_data() {
        // Too few observations
        let y = array![1.0, 2.0, 1.5];
        let treatment = array![1.0, 0.0, 1.0];
        let mediator = array![0.5, 0.3, 0.6];
        let confounders = Array2::zeros((3, 0));

        let config = MedflexConfig::default();
        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        );

        assert!(result.is_err());
        if let Err(EconError::InsufficientData { .. }) = result {
            // Expected error type
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_proportion_mediated_bounds() {
        let (y, treatment, mediator, confounders) = create_test_data();

        let config = MedflexConfig {
            n_bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = run_medflex(
            &y.view(),
            &treatment.view(),
            &mediator.view(),
            &confounders.view(),
            config,
        )
        .unwrap();

        // Proportion mediated should be between -2 and 2 (clamped)
        assert!(
            result.proportion_mediated >= -2.0 && result.proportion_mediated <= 2.0,
            "Proportion mediated {} out of bounds",
            result.proportion_mediated
        );
    }
}
