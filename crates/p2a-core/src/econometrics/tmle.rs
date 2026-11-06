//! Targeted Maximum Likelihood Estimation (TMLE) for causal inference.
//!
//! TMLE is a doubly robust, semiparametric efficient estimator for causal effects.
//! It uses a two-stage procedure:
//! 1. Fit initial outcome and propensity score models
//! 2. Apply a "targeting" step that fluctuates the outcome model to optimize
//!    the bias-variance tradeoff for the specific target parameter (ATE)
//!
//! The targeting step is what distinguishes TMLE from standard AIPW and makes it
//! locally efficient (achieves the semiparametric efficiency bound).
//!
//! # Algorithm (ATE, as R's `tmle` 2.1.1 with glm Q and g)
//!
//! **Step 1: Initial Estimates**
//! - Outcome model Q(A,W) = E[Y|A,W]: a logistic glm (`QModel::Logistic`, the default) or
//!   least squares (`QModel::Linear`)
//! - Propensity score g(W) = P(A=1|W): a logistic glm, bounded to `truncate_ps`
//!
//! The fits are [`crate::econometrics::glm::fit_glm`] (R's `glm.fit`: its convergence rule,
//! separation and aliasing warnings) and OLS's least-squares solve (pivoted QR for a
//! rank-deficient design). An aliased covariate gets R's "Coefficients not defined because of
//! singularities" warning; an aliased treatment is an error, because the effect is not
//! identified. The treatment must be coded 0/1.
//!
//! The logistic Q needs an outcome in [0, 1]. A continuous outcome outside it is rescaled to
//! `Y* = (Y - min) / (max - min)` first, as R's `tmle` and `ltmle` do (Gruber & van der Laan
//! 2010); the estimates, the influence curve and the Q vectors are reported on the original
//! scale and the fluctuation coefficients on the logit scale.
//!
//! **Step 2: Clever Covariates**
//! - H1(A,W) = A/g(W) and H0(A,W) = (1-A)/(1-g(W)); H = H1 - H0
//!
//! **Step 3: Targeting Step (Fluctuation)**
//! - Fit (e0, e1) in logit(Q*(A,W)) = logit(Q(A,W)) + e0 H0 + e1 H1 (quasibinomial, no
//!   intercept; least squares of Y - Q on H0 and H1 for the linear Q), as `tmle` 2.1.1
//!
//! **Step 4: Update Predictions and Compute Estimate**
//! - logit Q*(1,W) = logit Q(1,W) + e1 / g(W); logit Q*(0,W) = logit Q(0,W) + e0 / (1 - g(W))
//! - ATE = (1/n) * sum[ Q*(1,W_i) - Q*(0,W_i) ]
//!
//! **Step 5: Influence Curve for Variance**
//! - IC(O) = H(A,W)*(Y - Q*(A,W)) + Q*(1,W) - Q*(0,W) - ATE
//! - Var(ATE) = Var(IC) / n
//!
//! # References
//!
//! - van der Laan, M.J. & Rose, S. (2011). *Targeted Learning: Causal Inference for
//!   Observational and Experimental Data*. Springer.
//!   https://doi.org/10.1007/978-1-4419-9782-1
//!
//! - van der Laan, M.J. & Rubin, D. (2006). Targeted Maximum Likelihood Learning.
//!   *The International Journal of Biostatistics*, 2(1), Article 11.
//!   https://doi.org/10.2202/1557-4679.1043
//!
//! - Gruber, S. & van der Laan, M.J. (2010). A targeted maximum likelihood estimator of a
//!   causal effect on a bounded continuous outcome. *The International Journal of
//!   Biostatistics*, 6(1), Article 26. https://doi.org/10.2202/1557-4679.1260
//!
//! - Gruber, S. & van der Laan, M.J. (2012). tmle: An R Package for Targeted Maximum
//!   Likelihood Estimation. *Journal of Statistical Software*, 51(13), 1-35.
//!   https://doi.org/10.18637/jss.v051.i13
//!
//! - R package `tmle`: https://cran.r-project.org/package=tmle
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::econometrics::{run_tmle, TmleConfig, QModel, GModel};
//!
//! let config = TmleConfig::default();
//! let result = run_tmle(&dataset, "outcome", "treatment", &["x1", "x2"], config)?;
//! println!("ATE: {:.4} (SE: {:.4})", result.ate, result.ate_se);
//! println!("95% CI: [{:.4}, {:.4}]", result.ate_ci_lower, result.ate_ci_upper);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
// The lean logit IRLS lives in the shared GLM; C-TMLE and stdreg import it from here.
pub(crate) use crate::econometrics::glm::{GlmControl, LeanStart, lean_logit};
use crate::econometrics::glm::{
    GlmFamily, GlmFit, GlmLink, GlmSpec, LeanLogit, bound, fit_glm, plogis, qlogis,
};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::solve::aliasing_message;
use crate::regression::ols::solve_ols;
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, normal_p_value_two_sided, z_critical};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Outcome model specification for TMLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QModel {
    /// Linear regression for continuous Y
    Linear,
    /// Logistic regression (default): a 0/1 outcome, a proportion in [0, 1], or a continuous
    /// outcome rescaled to [0, 1] by its range
    #[default]
    Logistic,
}

impl fmt::Display for QModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QModel::Linear => write!(f, "Linear"),
            QModel::Logistic => write!(f, "Logistic"),
        }
    }
}

// FromStr (unknown names are an error listing the accepted ones) and choices(); the Display
// above stays.
crate::option_enum!(
    QModel,
    crate::options::Choices {
        option: "q_model",
        values: &[
            (
                &["logistic", "logit", "binomial", "binary"],
                QModel::Logistic
            ),
            (&["linear", "gaussian", "continuous", "ols"], QModel::Linear),
        ],
    },
    no_display
);

/// Propensity score model specification for TMLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GModel {
    /// Logistic regression (default)
    #[default]
    Logistic,
}

impl fmt::Display for GModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GModel::Logistic => write!(f, "Logistic"),
        }
    }
}

crate::option_enum!(
    GModel,
    crate::options::Choices {
        option: "g_model",
        values: &[(&["logistic", "logit", "glm"], GModel::Logistic)],
    },
    no_display
);

/// Configuration for TMLE estimation.
#[derive(Debug, Clone)]
pub struct TmleConfig {
    /// Outcome model specification
    pub q_model: QModel,
    /// Propensity score model specification
    pub g_model: GModel,
    /// Truncation bounds for propensity scores (min, max), `0 < min < max < 1`.
    /// Default: (0.01, 0.99) to avoid extreme weights
    pub truncate_ps: (f64, f64),
    /// `glm.fit`'s `maxit` for the logistic Q, g and fluctuation fits (default 100; R's
    /// `glm.control` default is 25).
    pub max_iter: usize,
    /// `glm.fit`'s `epsilon` for the logistic Q, g and fluctuation fits: the relative change
    /// in deviance at which IRLS stops (default 1e-8, R's `glm.control` default).
    pub tolerance: f64,
}

impl Default for TmleConfig {
    fn default() -> Self {
        Self {
            q_model: QModel::Logistic,
            g_model: GModel::Logistic,
            truncate_ps: (0.01, 0.99),
            max_iter: 100,
            tolerance: 1e-8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from TMLE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmleResult {
    /// Average treatment effect estimate
    pub ate: f64,
    /// Standard error of ATE (from influence curve)
    pub ate_se: f64,
    /// 95% confidence interval lower bound
    pub ate_ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ate_ci_upper: f64,
    /// Two-sided p-value for H0: ATE = 0
    pub ate_p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Z-statistic (ATE / SE)
    pub z_stat: f64,

    /// Propensity scores g(W) for each observation
    pub propensity_scores: Vec<f64>,
    /// Initial outcome predictions Q(A,W)
    pub initial_outcome: Vec<f64>,
    /// Targeted (fluctuated) outcome predictions Q*(A,W)
    pub targeted_outcome: Vec<f64>,
    /// Clever covariate of the influence curve, H(A,W) = A/g(W) - (1-A)/(1-g(W))
    pub clever_covariate: Vec<f64>,
    /// Fluctuation coefficient e1 of the treated clever covariate H1 = A/g (logit scale for
    /// the logistic Q)
    pub fluctuation_coef: f64,
    /// Influence curve IC(O) for each observation
    pub influence_curve: Vec<f64>,

    /// Counterfactual predictions Q*(1,W) for each observation
    pub q_star_1: Vec<f64>,
    /// Counterfactual predictions Q*(0,W) for each observation
    pub q_star_0: Vec<f64>,

    /// Whether the outcome model converged
    pub q_model_converged: bool,
    /// Number of iterations for outcome model
    pub q_model_iterations: usize,
    /// Whether the propensity score model converged
    pub g_model_converged: bool,
    /// Number of iterations for propensity model
    pub g_model_iterations: usize,
    /// Whether the targeting step converged
    pub targeting_converged: bool,

    /// Number of observations
    pub n_obs: usize,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Number of observations with truncated propensity scores
    pub n_truncated: usize,

    /// Outcome model type used
    pub q_model_type: QModel,
    /// Propensity model type used
    pub g_model_type: GModel,
    /// Propensity score truncation bounds used
    pub truncate_bounds: (f64, f64),

    /// Warnings generated during estimation
    pub warnings: Vec<String>,

    /// IRLS iterations of the targeting step (1 for the linear Q)
    #[serde(default)]
    pub targeting_iterations: usize,
    /// Fluctuation coefficient e0 of the control clever covariate H0 = (1-A)/(1-g)
    #[serde(default)]
    pub fluctuation_coef_control: f64,
    /// `(min, max)` of the outcome when the logistic Q rescaled it to [0, 1]; `None` otherwise
    #[serde(default)]
    pub outcome_range: Option<(f64, f64)>,
}

impl ResultNotes for TmleResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for TmleResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Targeted Maximum Likelihood Estimation (TMLE)")?;
        writeln!(f, "==============================================")?;
        writeln!(f)?;
        writeln!(f, "Treatment Effect (ATE):")?;
        writeln!(f, "  Estimate:   {:>12.4}", self.ate)?;
        writeln!(f, "  Std. Error: {:>12.4}", self.ate_se)?;
        writeln!(f, "  z-stat:     {:>12.2}", self.z_stat)?;
        writeln!(
            f,
            "  p-value:    {:>12.4}{}",
            self.ate_p_value,
            self.significance.stars()
        )?;
        writeln!(
            f,
            "  95% CI:     [{:.4}, {:.4}]",
            self.ate_ci_lower, self.ate_ci_upper
        )?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  Observations:  {} (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(f, "  PS Truncated:  {}", self.n_truncated)?;
        writeln!(f)?;
        writeln!(f, "Model Specification:")?;
        writeln!(f, "  Outcome Model (Q):    {}", self.q_model_type)?;
        writeln!(f, "  Propensity Model (g): {}", self.g_model_type)?;
        writeln!(
            f,
            "  PS Truncation:        [{:.2}, {:.2}]",
            self.truncate_bounds.0, self.truncate_bounds.1
        )?;
        if let Some((lo, hi)) = self.outcome_range {
            writeln!(
                f,
                "  Outcome rescaled:     [{lo:.4}, {hi:.4}] to [0, 1] for the logistic Q"
            )?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Targeting Step (clever covariates H1 = A/g, H0 = (1-A)/(1-g)):"
        )?;
        writeln!(
            f,
            "  Fluctuation coef (epsilon, H1): {:.6}",
            self.fluctuation_coef
        )?;
        writeln!(
            f,
            "  Fluctuation coef (epsilon, H0): {:.6}",
            self.fluctuation_coef_control
        )?;
        writeln!(f)?;
        writeln!(f, "Convergence:")?;
        writeln!(
            f,
            "  Q model: {} (iterations: {})",
            if self.q_model_converged { "Yes" } else { "No" },
            self.q_model_iterations
        )?;
        writeln!(
            f,
            "  g model: {} (iterations: {})",
            if self.g_model_converged { "Yes" } else { "No" },
            self.g_model_iterations
        )?;
        writeln!(
            f,
            "  Targeting: {} (iterations: {})",
            if self.targeting_converged {
                "Yes"
            } else {
                "No"
            },
            self.targeting_iterations
        )?;
        writeln!(f)?;
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
// Main TMLE Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run TMLE with full configuration.
///
/// Estimates the Average Treatment Effect (ATE) using Targeted Maximum Likelihood
/// Estimation. TMLE is doubly robust and locally efficient.
///
/// # Arguments
/// * `dataset` - Dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable column
/// * `treatment_col` - Name of the binary treatment variable (coded 0/1)
/// * `covariate_cols` - Names of covariate columns for both Q and g models
/// * `config` - TMLE configuration options
///
/// # Returns
/// `TmleResult` containing ATE estimate, standard error, confidence interval,
/// and diagnostic information.
///
/// # Errors
/// `InvalidSpecification` for an invalid config, a treatment not coded 0/1 or without both
/// groups, a constant outcome that the logistic Q would have to rescale, and a treatment
/// aliased with the covariates in the outcome model; the column errors of the extraction.
///
/// # Algorithm
///
/// 1. Fit initial outcome model Q(A,W) = E[Y|A,W]
/// 2. Fit propensity score model g(W) = P(A=1|W)
/// 3. Clever covariates H1 = A/g(W), H0 = (1-A)/(1-g(W))
/// 4. Targeting step: fit (e0, e1) in logit(Q*) = logit(Q) + e0 H0 + e1 H1
/// 5. Compute targeted predictions Q*(1,W) and Q*(0,W)
/// 6. ATE = mean(Q*(1,W) - Q*(0,W))
/// 7. Variance from influence curve: Var(ATE) = Var(IC)/n
///
/// # References
///
/// - van der Laan & Rose (2011), "Targeted Learning", Chapter 4
/// - Gruber & van der Laan (2012), JSS 51(13), Algorithm 1
pub fn tmle(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: TmleConfig,
) -> EconResult<TmleResult> {
    let mut warnings = Vec::new();
    validate_tmle_config(&config)?;

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    // Extract outcome variable Y
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    // Extract treatment variable A
    let a = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    let n = y.len();
    let (n_treated, n_control) = check_binary_treatment(a.view(), treatment_col)?;
    let scale = outcome_scale(y.view(), config.q_model, outcome_col)?;
    let ys = scale.apply(&y);
    if let Some(note) = scale.note("TMLE", outcome_col) {
        warnings.push(note);
    }

    // Build design matrix W for covariates (with intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, true)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let w = design.data;
    let k = w.ncols();
    let mut names = design.column_names;
    names.push(treatment_col.to_string());
    let ctl = GlmControl {
        epsilon: config.tolerance,
        max_iter: config.max_iter,
    };

    // Step 1: initial outcome model Q(A,W), with Q(1,W) and Q(0,W)
    let (qfit, q) = fit_outcome(
        w.view(),
        a.view(),
        ys.view(),
        config.q_model,
        None,
        &names,
        ctl,
        None,
    )?;

    // Step 2: propensity score g(W) = P(A=1|W), bounded (van der Laan & Rose 2011, 4.2.2)
    let (gfit, g_raw) = match config.g_model {
        GModel::Logistic => fit_treatment(w.view(), a.view(), None, &names[..k], ctl, None)?,
    };
    let (g, n_truncated) = bound_scores(&g_raw, config.truncate_ps);

    warnings.extend(prefixed("Q model", &qfit.warnings));
    warnings.extend(prefixed("g model", &gfit.warnings));
    if n_truncated > n / 10 {
        warnings.push(format!(
            "Many propensity scores truncated ({}/{}). Consider wider truncation bounds.",
            n_truncated, n
        ));
    }
    let g_min = g.iter().cloned().fold(f64::INFINITY, f64::min);
    let g_max = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if g_min < 0.05 || g_max > 0.95 {
        warnings.push(format!(
            "Propensity scores have extreme values (min: {:.4}, max: {:.4}). \
             This may indicate positivity violations.",
            g_min, g_max
        ));
    }

    // Steps 3-5: the bivariate fluctuation of tmle 2.1.1 (van der Laan & Rose 2011,
    // Algorithm 4.1, Step 3) and the targeted counterfactual predictions
    let fl = fluctuate(
        ys.view(),
        a.view(),
        &q,
        g.view(),
        config.q_model,
        None,
        ctl,
        FluctuationForm::Bivariate,
        false,
    )?;
    warnings.extend(prefixed("targeting step", &fl.warnings));

    // Steps 6-7: substitution estimator and influence-curve variance, on the original scale
    let (psi, ic) = ate_and_ic(ys.view(), &fl);
    let ate = scale.width * psi;
    let ic = ic * scale.width;
    let ate_se = (sample_variance(&ic.to_vec()) / n as f64).sqrt();
    let (z_stat, ate_p_value, ate_ci_lower, ate_ci_upper) = wald95(ate, ate_se)?;
    let significance = SignificanceLevel::from_p_value(ate_p_value);

    for (converged, iterations, method) in [
        (qfit.converged, qfit.iterations, "TMLE outcome model (Q)"),
        (gfit.converged, gfit.iterations, "TMLE propensity model (g)"),
        (fl.converged, fl.iterations, "TMLE targeting step"),
    ] {
        if let Some(note) = glm_convergence(converged, iterations, ctl).note(method) {
            warnings.push(note);
        }
    }

    let back = |v: &Array1<f64>| -> Vec<f64> { v.iter().map(|&x| scale.unscale(x)).collect() };
    Ok(TmleResult {
        ate,
        ate_se,
        ate_ci_lower,
        ate_ci_upper,
        ate_p_value,
        significance,
        z_stat,
        propensity_scores: g.to_vec(),
        initial_outcome: back(&q.qa),
        targeted_outcome: back(&fl.q_star.qa),
        clever_covariate: fl.h.to_vec(),
        fluctuation_coef: fl.epsilon,
        influence_curve: ic.to_vec(),
        q_star_1: back(&fl.q_star.q1),
        q_star_0: back(&fl.q_star.q0),
        q_model_converged: qfit.converged,
        q_model_iterations: qfit.iterations,
        g_model_converged: gfit.converged,
        g_model_iterations: gfit.iterations,
        targeting_converged: fl.converged,
        n_obs: n,
        n_treated,
        n_control,
        n_truncated,
        q_model_type: config.q_model,
        g_model_type: config.g_model,
        truncate_bounds: config.truncate_ps,
        warnings,
        targeting_iterations: fl.iterations,
        fluctuation_coef_control: fl.epsilon_control,
        outcome_range: scale.range(),
    })
}

/// Run TMLE with default configuration.
///
/// Convenience function that uses logistic regression for both the outcome model
/// and propensity score model, with standard propensity score truncation at [0.01, 0.99].
///
/// # Arguments
/// * `dataset` - Dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable column (binary or continuous)
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of covariate columns
///
/// # Example
/// ```ignore
/// let result = run_tmle(&dataset, "outcome", "treatment", &["age", "sex", "income"])?;
/// println!("ATE: {:.4} (95% CI: [{:.4}, {:.4}])",
///          result.ate, result.ate_ci_lower, result.ate_ci_upper);
/// ```
pub fn run_tmle(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
) -> EconResult<TmleResult> {
    tmle(
        dataset,
        outcome_col,
        treatment_col,
        covariate_cols,
        TmleConfig::default(),
    )
}

/// `max_iter >= 1`, `tolerance` finite and positive, `0 < truncate_ps.0 < truncate_ps.1 < 1`.
fn validate_tmle_config(config: &TmleConfig) -> EconResult<()> {
    if config.max_iter == 0 {
        return Err(spec("max_iter must be at least 1; got 0".to_string()));
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(spec(format!(
            "tolerance must be finite and positive; got {}",
            config.tolerance
        )));
    }
    let (lo, hi) = config.truncate_ps;
    if !(lo.is_finite() && hi.is_finite() && 0.0 < lo && lo < hi && hi < 1.0) {
        return Err(spec(format!(
            "truncate_ps must satisfy 0 < lower < upper < 1; got ({lo}, {hi})"
        )));
    }
    Ok(())
}

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

// ═══════════════════════════════════════════════════════════════════════════════
// The TMLE helper set (shared with C-TMLE)
// ═══════════════════════════════════════════════════════════════════════════════

/// The convergence record of one nuisance fit (glm.fit's deviance rule).
pub(crate) fn glm_convergence(converged: bool, iterations: usize, ctl: GlmControl) -> Convergence {
    Convergence::new(
        converged,
        iterations,
        ctl.max_iter,
        "relative change in deviance (glm.fit)",
        f64::NAN,
        ctl.epsilon,
    )
}

/// Guard applied to Q before `qlogis`.
pub(crate) const Q_BOUND: f64 = 1e-10;

/// Q(A,W), Q(1,W), Q(0,W) for every row.
#[derive(Debug, Clone)]
pub(crate) struct QPredictions {
    pub qa: Array1<f64>,
    pub q1: Array1<f64>,
    pub q0: Array1<f64>,
}

/// A fitted nuisance model.
#[derive(Debug, Clone)]
pub(crate) struct NuisanceFit {
    /// NaN where aliased.
    pub coefficients: Array1<f64>,
    /// `true` for least squares.
    pub converged: bool,
    /// 1 for least squares.
    pub iterations: usize,
    pub separation: bool,
    /// Indices of the aliased columns.
    pub aliased: Vec<usize>,
    /// The fit's warnings except "glm.fit: algorithm did not converge" (carried by
    /// `converged`), then the aliasing message; unprefixed.
    pub warnings: Vec<String>,
}

impl NuisanceFit {
    /// The shared GLM's fit with its warnings as [`lean_logit`]'s path would report them:
    /// non-convergence is carried by `converged`, and the dispersion warning about
    /// zero-weight rows (the training rows of a fold fit) concerns a dispersion that no
    /// nuisance fit reads, so a fold fit reports the same notes on either path.
    fn from_glm(fit: GlmFit, names: &[String]) -> NuisanceFit {
        let mut warnings: Vec<String> = fit
            .warnings
            .into_iter()
            .filter(|w| {
                w != "glm.fit: algorithm did not converge"
                    && w != "observations with zero weight not used for calculating dispersion"
            })
            .collect();
        warnings.extend(aliasing_message(&fit.aliased, names));
        NuisanceFit {
            coefficients: fit.coefficients,
            converged: fit.converged,
            iterations: fit.iterations,
            separation: fit.separation,
            aliased: fit.aliased,
            warnings,
        }
    }

    /// A fit of [`lean_logit`]'s path: converged, full rank, nothing to report.
    fn from_lean(fit: LeanLogit) -> NuisanceFit {
        NuisanceFit {
            coefficients: fit.coefficients,
            converged: true,
            iterations: fit.iterations,
            separation: false,
            aliased: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Coefficients with NaN set to 0 (R's `predict` on a rank-deficient fit).
    pub(crate) fn zeroed(&self) -> Array1<f64> {
        self.coefficients.mapv(|b| if b.is_nan() { 0.0 } else { b })
    }
}

/// The rows with positive training weight, or `None` for all rows.
fn positive_rows(train: Option<ArrayView1<'_, f64>>) -> Option<Vec<usize>> {
    train.map(|t| {
        t.iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(i, _)| i)
            .collect()
    })
}

/// A logit nuisance fit of `y` on `x` over the rows with positive weight of `train` (all rows
/// when `None`): [`lean_logit`] from `start` (glm.fit's `mustart` when `None`), and the shared
/// GLM (`family`, `ctl`, glm.fit's start) when the fit leaves the lean path, so that the GLM
/// reports separation, aliasing and non-convergence.
fn logit_nuisance(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    family: GlmFamily,
    train: Option<ArrayView1<'_, f64>>,
    names: &[String],
    ctl: GlmControl,
    start: Option<ArrayView1<'_, f64>>,
) -> EconResult<NuisanceFit> {
    let start = start.map_or(LeanStart::Mustart, LeanStart::Coef);
    let lean = match positive_rows(train) {
        None => lean_logit(x, y, None, start, ctl)?,
        Some(rows) => {
            let xs = x.select(Axis(0), &rows);
            let ys = y.select(Axis(0), &rows);
            lean_logit(xs.view(), ys.view(), None, start, ctl)?
        }
    };
    if let Some(fit) = lean {
        return Ok(NuisanceFit::from_lean(fit));
    }
    let spec = ctl.spec(family, GlmLink::Logit, train);
    Ok(NuisanceFit::from_glm(fit_glm(x, y, &spec)?, names))
}

/// Least squares of `y` on `x` over the rows with positive weight of `train` (all rows when
/// `None`): OLS's solve (normal equations on a well-conditioned full-rank design, `lm.fit`'s
/// pivoted QR otherwise).
fn least_squares_fit(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    train: Option<ArrayView1<'_, f64>>,
    names: &[String],
) -> EconResult<NuisanceFit> {
    let solve = match positive_rows(train) {
        None => solve_ols(x, y)?,
        Some(rows) => {
            if rows.is_empty() {
                return Err(EconError::InsufficientData {
                    required: 1,
                    provided: 0,
                    context: "least squares: no observation has a positive weight".into(),
                });
            }
            let xs = x.select(Axis(0), &rows);
            let ys = y.select(Axis(0), &rows);
            solve_ols(xs.view(), ys.view())?
        }
    };
    Ok(NuisanceFit {
        warnings: aliasing_message(&solve.aliased, names)
            .into_iter()
            .collect(),
        coefficients: solve.beta,
        converged: true,
        iterations: 1,
        separation: false,
        aliased: solve.aliased,
    })
}

/// Outcome model on `[w | a]` (`w` has the intercept), fitted on the rows with positive
/// weight of `train` (all rows when `None`): Logistic is a logit glm (binomial for a 0/1
/// outcome, quasibinomial for one in [0, 1], which gives the binomial estimates without the
/// non-integer warning), Linear least squares. Predictions for every row: Q(A,W) and Q(1,W),
/// Q(0,W) from the coefficients with NaN set to 0. `names` names `w`'s columns, then the
/// treatment.
///
/// An aliased treatment is `InvalidSpecification`: the effect is not identified.
pub(crate) fn fit_outcome(
    w: ArrayView2<'_, f64>,
    a: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    model: QModel,
    train: Option<ArrayView1<'_, f64>>,
    names: &[String],
    ctl: GlmControl,
    start: Option<ArrayView1<'_, f64>>,
) -> EconResult<(NuisanceFit, QPredictions)> {
    crate::resource::checkpoint()?;
    let (n, k) = w.dim();
    let mut x = crate::resource::try_zeros2(n, k + 1, "outcome model design")?;
    x.slice_mut(s![.., ..k]).assign(&w);
    x.column_mut(k).assign(&a);
    let fit = match model {
        QModel::Logistic => {
            let family = if y.iter().all(|&v| v == 0.0 || v == 1.0) {
                GlmFamily::Binomial
            } else {
                GlmFamily::QuasiBinomial
            };
            logit_nuisance(x.view(), y, family, train, names, ctl, start)?
        }
        QModel::Linear => least_squares_fit(x.view(), y, train, names)?,
    };
    if fit.aliased.contains(&k) {
        let name = names.get(k).map(String::as_str).unwrap_or("treatment");
        return Err(spec(format!(
            "the treatment '{name}' is collinear with the covariates in the outcome model ({}); \
             the treatment effect is not identified",
            aliasing_message(&fit.aliased, names).unwrap_or_default()
        )));
    }
    let b = fit.zeroed();
    let bw = b.slice(s![..k]);
    let ba = b[k];
    let eta0 = w.dot(&bw);
    let eta1 = &eta0 + ba;
    let eta_a = &eta0 + &(&a * ba);
    let link = |e: Array1<f64>| match model {
        QModel::Logistic => e.mapv(plogis),
        QModel::Linear => e,
    };
    let q = QPredictions {
        qa: link(eta_a),
        q1: link(eta1),
        q0: link(eta0),
    };
    Ok((fit, q))
}

/// Propensity score g(W) = P(A = 1 | W): a binomial logit glm on the rows with positive
/// weight of `train`, raw scores for every row. The 0/1 check is the caller's.
pub(crate) fn fit_treatment(
    w: ArrayView2<'_, f64>,
    a: ArrayView1<'_, f64>,
    train: Option<ArrayView1<'_, f64>>,
    names: &[String],
    ctl: GlmControl,
    start: Option<ArrayView1<'_, f64>>,
) -> EconResult<(NuisanceFit, Array1<f64>)> {
    crate::resource::checkpoint()?;
    let fit = logit_nuisance(w, a, GlmFamily::Binomial, train, names, ctl, start)?;
    let scores = w.dot(&fit.zeroed()).mapv(plogis);
    Ok((fit, scores))
}

/// Clamp scores to `[lo, hi]`; also return how many were outside.
pub(crate) fn bound_scores(g: &Array1<f64>, bounds: (f64, f64)) -> (Array1<f64>, usize) {
    let (lo, hi) = bounds;
    let n_out = g.iter().filter(|&&v| v < lo || v > hi).count();
    (g.mapv(|v| bound(v, lo, hi)), n_out)
}

/// The clever covariates of the targeting step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FluctuationForm {
    /// One clever covariate H = A/g - (1-A)/(1-g) (van der Laan & Rose 2011; ctmle).
    OneDim,
    /// Two clever covariates H1 = A/g and H0 = (1-A)/(1-g) (tmle 2.1.1).
    Bivariate,
}

/// A fitted targeting step.
#[derive(Debug, Clone)]
pub(crate) struct Fluctuation {
    /// The coefficient of H (one-dimensional) or of H1 (bivariate).
    pub epsilon: f64,
    /// The coefficient of H0 (bivariate); NaN for the one-dimensional form.
    pub epsilon_control: f64,
    /// H = A/g - (1-A)/(1-g), the influence curve's clever covariate, for every row.
    pub h: Array1<f64>,
    /// Q*(A,W), Q*(1,W), Q*(0,W) for every row.
    pub q_star: QPredictions,
    pub converged: bool,
    pub iterations: usize,
    /// Warnings of the fluctuation glm (unprefixed; not the non-convergence one).
    pub warnings: Vec<String>,
}

/// `qlogis` of a Q prediction bounded to `[Q_BOUND, 1 - Q_BOUND]`: the offset of the logistic
/// fluctuation.
pub(crate) fn logit_q(v: f64) -> f64 {
    qlogis(bound(v, Q_BOUND, 1.0 - Q_BOUND))
}

/// The fitted fluctuation coefficients (one per column of the clever-covariate design).
pub(crate) struct EpsilonFit {
    pub coefficients: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    /// glm's warnings except "glm.fit: algorithm did not converge"; unprefixed.
    pub warnings: Vec<String>,
}

/// The targeting regression on the clever-covariate columns `hx` (no intercept), over the rows
/// with positive weight of `train` (all rows when `None`). Logistic: the quasibinomial logit glm
/// of `y` with offset `base` (= `logit_q(Q(A,W))`), fitted by [`lean_logit`] (from glm.fit's
/// start, or from epsilon = 0 when `from_zero`; the maximum-likelihood estimate is the same)
/// and by the shared GLM when the fit leaves the lean path. The shared GLM fit is built here
/// rather than by `glm::fit_logistic_fluctuation`, which does not take the glm control
/// (`epsilon`, `max_iter`). Linear: per column `sum c (y - base) / sum c^2` with `base` =
/// Q(A,W) (the columns of the bivariate design have disjoint supports, so this is the joint
/// least-squares fit).
pub(crate) fn fit_epsilon(
    y: ArrayView1<'_, f64>,
    hx: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
    model: QModel,
    train: Option<ArrayView1<'_, f64>>,
    ctl: GlmControl,
    from_zero: bool,
) -> EconResult<EpsilonFit> {
    match model {
        QModel::Logistic => {
            let start = if from_zero {
                LeanStart::Zero
            } else {
                LeanStart::Mustart
            };
            let lean = match positive_rows(train) {
                None => lean_logit(hx, y, Some(base), start, ctl)?,
                Some(rows) => {
                    let hs = hx.select(Axis(0), &rows);
                    let ys = y.select(Axis(0), &rows);
                    let bs = base.select(Axis(0), &rows);
                    lean_logit(hs.view(), ys.view(), Some(bs.view()), start, ctl)?
                }
            };
            if let Some(fit) = lean {
                return Ok(EpsilonFit {
                    coefficients: fit.coefficients.to_vec(),
                    converged: true,
                    iterations: fit.iterations,
                    warnings: Vec::new(),
                });
            }
            let spec = ctl
                .spec(
                    GlmFamily::QuasiBinomial,
                    GlmLink::Logit,
                    train.as_ref().map(|t| t.view()),
                )
                .offset(base.view());
            let fit = NuisanceFit::from_glm(fit_glm(hx, y, &spec)?, &[]);
            if fit.coefficients.iter().any(|c| !c.is_finite()) {
                return Err(EconError::Computation(
                    "TMLE targeting step: the clever covariate is degenerate".into(),
                ));
            }
            Ok(EpsilonFit {
                coefficients: fit.coefficients.to_vec(),
                converged: fit.converged,
                iterations: fit.iterations,
                warnings: fit.warnings,
            })
        }
        QModel::Linear => {
            let n = y.len();
            let in_fit = |i: usize| train.as_ref().is_none_or(|t| t[i] > 0.0);
            let mut coefficients = Vec::with_capacity(hx.ncols());
            for c in hx.columns() {
                let (mut num, mut den) = (0.0, 0.0);
                for i in (0..n).filter(|&i| in_fit(i)) {
                    num += c[i] * (y[i] - base[i]);
                    den += c[i] * c[i];
                }
                if den.abs() < 1e-10 {
                    return Err(EconError::SingularMatrix {
                        context: "TMLE targeting step".to_string(),
                        suggestion: "the clever covariate is zero on the fitting rows (no \
                                     treated or no control observation)"
                            .to_string(),
                    });
                }
                coefficients.push(num / den);
            }
            Ok(EpsilonFit {
                coefficients,
                converged: true,
                iterations: 1,
                warnings: Vec::new(),
            })
        }
    }
}

/// The targeting step, fitted on the rows with positive weight of `train`:
///
/// - Logistic: `glm(y ~ -1 + H, offset = qlogis(Q), quasibinomial)` (OneDim) or
///   `glm(y ~ -1 + H0 + H1, offset = qlogis(Q), quasibinomial)` (Bivariate), with Q bounded
///   to `[Q_BOUND, 1 - Q_BOUND]` before `qlogis`; logit Q*(1) = logit Q(1) + e/g (e1/g),
///   logit Q*(0) = logit Q(0) - e/(1-g) (+ e0/(1-g)).
/// - Linear: least squares of `y - Q(A,W)` on H (on H0 and H1, which have disjoint support,
///   so each coefficient is its own ratio); Q* updated the same way on the identity scale.
///
/// `from_zero`: the logistic fit starts at epsilon = 0 (see [`fit_epsilon`]); the estimate is
/// the same, the iteration count is not glm.fit's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fluctuate(
    y: ArrayView1<'_, f64>,
    a: ArrayView1<'_, f64>,
    q: &QPredictions,
    g: ArrayView1<'_, f64>,
    model: QModel,
    train: Option<ArrayView1<'_, f64>>,
    ctl: GlmControl,
    form: FluctuationForm,
    from_zero: bool,
) -> EconResult<Fluctuation> {
    crate::resource::checkpoint()?;
    let n = y.len();
    let h1: Array1<f64> = Array1::from_iter((0..n).map(|i| a[i] / g[i]));
    let h0: Array1<f64> = Array1::from_iter((0..n).map(|i| (1.0 - a[i]) / (1.0 - g[i])));
    let h = &h1 - &h0;
    // (e1, e0): the coefficients applied to Q(1) / g and Q(0) / (1 - g)
    let base = match model {
        QModel::Logistic => q.qa.mapv(logit_q),
        QModel::Linear => q.qa.clone(),
    };
    let hx: Array2<f64> = match form {
        FluctuationForm::OneDim => h.clone().insert_axis(Axis(1)),
        FluctuationForm::Bivariate => {
            let mut m = crate::resource::try_zeros2(n, 2, "fluctuation design")?;
            m.column_mut(0).assign(&h0);
            m.column_mut(1).assign(&h1);
            m
        }
    };
    let fit = fit_epsilon(y, hx.view(), base.view(), model, train, ctl, from_zero)?;
    let (eps, eps_control, e1, e0) = match form {
        FluctuationForm::OneDim => {
            let e = fit.coefficients[0];
            (e, f64::NAN, e, -e)
        }
        FluctuationForm::Bivariate => {
            let (e0, e1) = (fit.coefficients[0], fit.coefficients[1]);
            (e1, e0, e1, e0)
        }
    };
    let (converged, iterations, warnings) = (fit.converged, fit.iterations, fit.warnings);
    let q_star = match model {
        QModel::Logistic => QPredictions {
            qa: Array1::from_iter((0..n).map(|i| plogis(base[i] + e0 * h0[i] + e1 * h1[i]))),
            q1: Array1::from_iter((0..n).map(|i| plogis(logit_q(q.q1[i]) + e1 / g[i]))),
            q0: Array1::from_iter((0..n).map(|i| plogis(logit_q(q.q0[i]) + e0 / (1.0 - g[i])))),
        },
        QModel::Linear => QPredictions {
            qa: Array1::from_iter((0..n).map(|i| q.qa[i] + e0 * h0[i] + e1 * h1[i])),
            q1: Array1::from_iter((0..n).map(|i| q.q1[i] + e1 / g[i])),
            q0: Array1::from_iter((0..n).map(|i| q.q0[i] + e0 / (1.0 - g[i]))),
        },
    };
    Ok(Fluctuation {
        epsilon: eps,
        epsilon_control: eps_control,
        h,
        q_star,
        converged,
        iterations,
        warnings,
    })
}

/// psi = mean(Q*(1) - Q*(0)) over all rows and IC_i = H_i (y_i - Q*(A)_i) + Q*(1)_i - Q*(0)_i - psi.
pub(crate) fn ate_and_ic(y: ArrayView1<'_, f64>, fl: &Fluctuation) -> (f64, Array1<f64>) {
    let n = y.len();
    let qs = &fl.q_star;
    let psi = (0..n).map(|i| qs.q1[i] - qs.q0[i]).sum::<f64>() / n as f64;
    let ic =
        Array1::from_iter((0..n).map(|i| fl.h[i] * (y[i] - qs.qa[i]) + qs.q1[i] - qs.q0[i] - psi));
    (psi, ic)
}

/// R's `var`: `sum (x - mean)^2 / (n - 1)`; NaN for n < 2.
pub(crate) fn sample_variance(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
}

/// `(z, p, lower, upper)` at level 0.95: `z = est / se` if `se` is positive and finite, else
/// 0; `p` the two-sided normal p-value; bounds `est -/+ qnorm(0.975) se`.
pub(crate) fn wald95(est: f64, se: f64) -> EconResult<(f64, f64, f64, f64)> {
    let z = if se > 0.0 && se.is_finite() {
        est / se
    } else {
        0.0
    };
    let zc = z_critical(0.95)?;
    Ok((z, normal_p_value_two_sided(z), est - zc * se, est + zc * se))
}

/// Every value exactly 0 or 1 and both present: `Ok((n_treated, n_control))`. A value that is
/// not 0 or 1 is reported by its 0-based position in `a`. TMLE, C-TMLE and stdreg pass the
/// whole column (they refuse missing values), so that position is the dataset row.
pub(crate) fn check_binary_treatment(
    a: ArrayView1<'_, f64>,
    name: &str,
) -> EconResult<(usize, usize)> {
    if let Some((i, v)) = a.iter().enumerate().find(|(_, v)| **v != 0.0 && **v != 1.0) {
        return Err(spec(format!(
            "treatment '{name}' must be coded 0/1 (0-based row {i} is {v})"
        )));
    }
    let n_treated = a.iter().filter(|&&v| v == 1.0).count();
    let n_control = a.len() - n_treated;
    if n_treated == 0 || n_control == 0 {
        return Err(spec(format!(
            "Treatment variable '{name}' must have both treated (1) and control (0) observations. \
             Found {n_treated} treated, {n_control} control."
        )));
    }
    Ok((n_treated, n_control))
}

/// How the outcome enters the Q model: `Y* = (Y - lower) / width`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OutcomeScale {
    pub lower: f64,
    pub width: f64,
    pub rescaled: bool,
    /// `(min, max)` of the outcome.
    pub min: f64,
    pub max: f64,
}

impl OutcomeScale {
    /// `Y*` for every value (a copy when not rescaled).
    pub fn apply(&self, y: &Array1<f64>) -> Array1<f64> {
        if self.rescaled {
            y.mapv(|v| (v - self.lower) / self.width)
        } else {
            y.clone()
        }
    }

    /// A fitted value on the original scale.
    pub fn unscale(&self, v: f64) -> f64 {
        if self.rescaled {
            self.lower + self.width * v
        } else {
            v
        }
    }

    /// `Some((min, max))` when rescaled.
    pub fn range(&self) -> Option<(f64, f64)> {
        self.rescaled.then_some((self.min, self.max))
    }

    /// The warning that reports the rescaling.
    pub fn note(&self, method: &str, name: &str) -> Option<String> {
        self.rescaled.then(|| {
            format!(
                "{method}: outcome '{name}' rescaled from [{:.4}, {:.4}] to [0, 1] for the \
                 logistic Q model (as R's tmle and ltmle); estimates are on the original scale",
                self.min, self.max
            )
        })
    }
}

/// For a logistic Q with an outcome outside [0, 1]: rescale by the range (`InvalidSpecification`
/// "outcome '<name>' is constant" when the range is 0, "contains a non-finite value" when it is
/// not finite); otherwise the identity.
pub(crate) fn outcome_scale(
    y: ArrayView1<'_, f64>,
    model: QModel,
    name: &str,
) -> EconResult<OutcomeScale> {
    let min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let identity = OutcomeScale {
        lower: 0.0,
        width: 1.0,
        rescaled: false,
        min,
        max,
    };
    if model != QModel::Logistic || (min >= 0.0 && max <= 1.0) {
        return Ok(identity);
    }
    let width = max - min;
    if !width.is_finite() || y.iter().any(|v| v.is_nan()) {
        return Err(spec(format!(
            "outcome '{name}' contains a non-finite value"
        )));
    }
    if width == 0.0 {
        return Err(spec(format!(
            "outcome '{name}' is constant ({min}); nothing to estimate"
        )));
    }
    Ok(OutcomeScale {
        lower: min,
        width,
        rescaled: true,
        min,
        max,
    })
}

/// `"<prefix>: <w>"` for each warning.
pub(crate) fn prefixed<'a>(
    prefix: &'a str,
    warnings: &'a [String],
) -> impl Iterator<Item = String> + 'a {
    warnings.iter().map(move |w| format!("{prefix}: {w}"))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    /// Create a test dataset with known treatment effect (binary outcomes).
    ///
    /// DGP for binary Y:
    /// - W ~ Uniform(0, 1)
    /// - A | W ~ Bernoulli(expit(W))
    /// - P(Y=1 | A, W) = expit(-1 + 2*A + W)
    ///
    /// This creates treated units with higher P(Y=1) than control units.
    /// True ATE on probability scale is approximately 0.3-0.5.
    fn create_tmle_test_dataset() -> Dataset {
        // Binary outcomes appropriate for logistic regression
        // Treated (A=1): Higher probability of Y=1
        // Control (A=0): Lower probability of Y=1
        let df = df! {
            "y" => [
                // Treated observations (A=1): ~70-90% are Y=1
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                // Control observations (A=0): ~30-50% are Y=1
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0
            ],
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "w1" => [
                // Overlapping covariate distribution
                0.3, 0.7, 0.2, 0.8, 0.35, 0.75, 0.25, 0.85, 0.32, 0.68,
                0.15, 0.9, 0.4, 0.6, 0.22, 0.78, 0.45, 0.55, 0.18, 0.82,
                0.25, 0.65, 0.15, 0.75, 0.3, 0.7, 0.2, 0.8, 0.28, 0.62,
                0.1, 0.85, 0.35, 0.6, 0.18, 0.72, 0.4, 0.58, 0.12, 0.78
            ],
            "w2" => [
                0.4, 0.6, 0.35, 0.65, 0.45, 0.55, 0.38, 0.62, 0.42, 0.58,
                0.3, 0.7, 0.48, 0.52, 0.32, 0.68, 0.5, 0.5, 0.28, 0.72,
                0.38, 0.62, 0.32, 0.68, 0.4, 0.6, 0.35, 0.65, 0.36, 0.64,
                0.25, 0.75, 0.42, 0.58, 0.28, 0.72, 0.45, 0.55, 0.22, 0.78
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    /// Create a test dataset with continuous outcomes for linear Q model.
    fn create_tmle_continuous_dataset() -> Dataset {
        // Continuous outcomes: Y = 0.3*W + 0.5*A + noise
        // True ATE = 0.5
        let df = df! {
            "y" => [
                // Treated observations (A=1): Y approx 0.3*W + 0.5 + noise
                0.7, 0.9, 0.6, 0.95, 0.75, 0.85, 0.65, 0.92, 0.72, 0.88,
                0.55, 0.98, 0.78, 0.82, 0.62, 0.90, 0.80, 0.76, 0.58, 0.94,
                // Control observations (A=0): Y approx 0.3*W + noise
                0.15, 0.35, 0.10, 0.40, 0.18, 0.38, 0.12, 0.42, 0.16, 0.32,
                0.08, 0.45, 0.22, 0.30, 0.11, 0.36, 0.25, 0.28, 0.09, 0.38
            ],
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            "w1" => [
                0.3, 0.7, 0.2, 0.8, 0.35, 0.75, 0.25, 0.85, 0.32, 0.68,
                0.15, 0.9, 0.4, 0.6, 0.22, 0.78, 0.45, 0.55, 0.18, 0.82,
                0.25, 0.65, 0.15, 0.75, 0.3, 0.7, 0.2, 0.8, 0.28, 0.62,
                0.1, 0.85, 0.35, 0.6, 0.18, 0.72, 0.4, 0.58, 0.12, 0.78
            ],
            "w2" => [
                0.4, 0.6, 0.35, 0.65, 0.45, 0.55, 0.38, 0.62, 0.42, 0.58,
                0.3, 0.7, 0.48, 0.52, 0.32, 0.68, 0.5, 0.5, 0.28, 0.72,
                0.38, 0.62, 0.32, 0.68, 0.4, 0.6, 0.35, 0.65, 0.36, 0.64,
                0.25, 0.75, 0.42, 0.58, 0.28, 0.72, 0.45, 0.55, 0.22, 0.78
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_tmle_basic() {
        let dataset = create_tmle_test_dataset();
        let result = run_tmle(&dataset, "y", "treatment", &["w1", "w2"]).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 40);
        assert_eq!(result.n_treated, 20);
        assert_eq!(result.n_control, 20);

        // Binary outcome data: treated has 17/20 Y=1 (85%), control has 7/20 Y=1 (35%)
        // Raw difference is 0.5, TMLE estimate should be in reasonable range
        assert!(
            result.ate > 0.2,
            "ATE should be positive, got {}",
            result.ate
        );
        assert!(
            result.ate < 0.8,
            "ATE should be around 0.3-0.5, got {}",
            result.ate
        );

        // Standard error should be positive and reasonable
        assert!(
            result.ate_se > 0.0 && result.ate_se.is_finite(),
            "SE should be positive and finite, got {}",
            result.ate_se
        );
        assert!(result.ate_se < 0.5, "SE seems too large: {}", result.ate_se);

        // Check that models converged
        assert!(result.q_model_converged, "Q model should converge");
        assert!(result.g_model_converged, "g model should converge");
        assert!(result.targeting_converged, "Targeting step should converge");

        // Check propensity scores are in valid range
        for &ps in &result.propensity_scores {
            assert!(
                (0.01..=0.99).contains(&ps),
                "Propensity score {} outside truncation bounds",
                ps
            );
        }

        // Fluctuation coefficient should be small (good initial models)
        assert!(
            result.fluctuation_coef.abs() < 10.0,
            "Fluctuation coefficient seems too large: {}",
            result.fluctuation_coef
        );
    }

    #[test]
    fn test_tmle_with_linear_outcome() {
        // Use continuous outcome dataset for linear Q model
        let dataset = create_tmle_continuous_dataset();
        let config = TmleConfig {
            q_model: QModel::Linear,
            ..Default::default()
        };

        let result = tmle(&dataset, "y", "treatment", &["w1", "w2"], config).unwrap();

        // True ATE is approximately 0.5 (from DGP: Y = 0.3*W + 0.5*A + noise)
        assert!(
            result.ate > 0.3 && result.ate < 0.7,
            "ATE with linear Q model should be ~0.5, got: {}",
            result.ate
        );
        assert!(result.ate_se > 0.0 && result.ate_se.is_finite());
    }

    #[test]
    fn test_tmle_propensity_truncation() {
        let dataset = create_tmle_test_dataset();

        // Test with tighter truncation
        let config = TmleConfig {
            truncate_ps: (0.1, 0.9),
            ..Default::default()
        };

        let result = tmle(&dataset, "y", "treatment", &["w1", "w2"], config).unwrap();

        // All propensity scores should be in [0.1, 0.9]
        for &ps in &result.propensity_scores {
            assert!(
                (0.1..=0.9).contains(&ps),
                "PS {} outside truncation bounds [0.1, 0.9]",
                ps
            );
        }
    }

    #[test]
    fn test_tmle_influence_curve() {
        let dataset = create_tmle_test_dataset();
        let result = run_tmle(&dataset, "y", "treatment", &["w1", "w2"]).unwrap();

        // The influence curve should have mean close to zero (property of EIC)
        let ic_mean: f64 = result.influence_curve.iter().sum::<f64>() / result.n_obs as f64;
        assert!(
            ic_mean.abs() < 0.1,
            "IC mean should be close to zero, got {}",
            ic_mean
        );

        // IC should have the same length as n_obs
        assert_eq!(result.influence_curve.len(), result.n_obs);
    }

    #[test]
    fn test_tmle_counterfactuals() {
        let dataset = create_tmle_test_dataset();
        let result = run_tmle(&dataset, "y", "treatment", &["w1", "w2"]).unwrap();

        // Counterfactual predictions should be available for all observations
        assert_eq!(result.q_star_1.len(), result.n_obs);
        assert_eq!(result.q_star_0.len(), result.n_obs);

        // Q*(1,W) - Q*(0,W) should average to ATE
        let ate_from_cf: f64 = result
            .q_star_1
            .iter()
            .zip(result.q_star_0.iter())
            .map(|(q1, q0)| q1 - q0)
            .sum::<f64>()
            / result.n_obs as f64;

        assert!(
            (ate_from_cf - result.ate).abs() < 1e-10,
            "ATE from counterfactuals ({}) should match reported ATE ({})",
            ate_from_cf,
            result.ate
        );
    }

    #[test]
    fn test_tmle_missing_column_error() {
        let dataset = create_tmle_test_dataset();

        // Missing outcome column
        let result = run_tmle(&dataset, "nonexistent", "treatment", &["w1"]);
        assert!(result.is_err());

        // Missing treatment column
        let result = run_tmle(&dataset, "y", "nonexistent", &["w1"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tmle_display() {
        let dataset = create_tmle_test_dataset();
        let result = run_tmle(&dataset, "y", "treatment", &["w1", "w2"]).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("Targeted Maximum Likelihood"));
        assert!(output.contains("ATE"));
        assert!(output.contains("Std. Error"));
        assert!(output.contains("95% CI"));
        assert!(output.contains("Fluctuation coef"));
    }

    /// Test that TMLE provides efficiency gains over simple IPW/AIPW.
    ///
    /// Due to the targeting step, TMLE should have variance closer to
    /// the semiparametric efficiency bound.
    #[test]
    fn test_tmle_variance_properties() {
        let dataset = create_tmle_test_dataset();
        let result = run_tmle(&dataset, "y", "treatment", &["w1", "w2"]).unwrap();

        // Variance should be Var(IC)/n
        let ic_var: f64 = {
            let mean: f64 = result.influence_curve.iter().sum::<f64>() / result.n_obs as f64;
            result
                .influence_curve
                .iter()
                .map(|&ic| (ic - mean).powi(2))
                .sum::<f64>()
                / (result.n_obs - 1) as f64
        };
        let expected_var = ic_var / result.n_obs as f64;
        let reported_var = result.ate_se.powi(2);

        assert!(
            (expected_var - reported_var).abs() / expected_var < 0.01,
            "Variance calculation mismatch: expected {}, got {}",
            expected_var,
            reported_var
        );
    }

    // =========================================================================
    // R Validation Tests (Phase 5)
    // =========================================================================

    /// Simple LCG for deterministic random numbers
    fn lcg_rand(seed: &mut u64) -> f64 {
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);
        *seed = (a.wrapping_mul(*seed).wrapping_add(c)) % m;
        (*seed as f64) / (m as f64)
    }

    /// Create validation dataset matching R's tmle package example.
    fn create_tmle_validation_dataset() -> Dataset {
        let n = 500;
        let mut seed: u64 = 42;

        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            // Generate covariates
            let u1 = lcg_rand(&mut seed).max(1e-10);
            let u2 = lcg_rand(&mut seed);
            let z1 = ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos();
            let z2 = ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).sin();
            x1.push(z1);
            x2.push(z2);

            // Propensity score model
            let ps = 1.0 / (1.0 + (-(-0.5 + 0.6 * z1 + 0.3 * z2)).exp());
            let t = if lcg_rand(&mut seed) < ps { 1.0 } else { 0.0 };
            treatment.push(t);

            // Binary outcome with true ATE approximately 0.15
            let y_prob = 1.0 / (1.0 + (-(-1.0 + 0.5 * z1 - 0.3 * z2 + 0.5 * t)).exp());
            let y_i = if lcg_rand(&mut seed) < y_prob {
                1.0
            } else {
                0.0
            };
            y.push(y_i);
        }

        let df = df! {
            "y" => y,
            "treatment" => treatment,
            "x1" => x1,
            "x2" => x2
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_tmle_targeting_step_converges() {
        // The targeting step on a well-specified binary-outcome fit
        let dataset = create_tmle_validation_dataset();
        let config = TmleConfig::default();

        let result = tmle(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();

        assert!(result.fluctuation_coef.is_finite());
        assert!(result.fluctuation_coef_control.is_finite());
        assert!(
            result.fluctuation_coef.abs() < 10.0 && result.fluctuation_coef_control.abs() < 10.0,
            "fluctuation coefficients {:.4} / {:.4} seem too large",
            result.fluctuation_coef,
            result.fluctuation_coef_control
        );
        assert!(result.targeting_converged, "Targeting should converge");
        assert!(result.targeting_iterations >= 1);
    }

    #[test]
    fn test_tmle_clever_covariate_formula() {
        // H(A,W) = A/g(W) - (1-A)/(1-g(W))
        let dataset = create_tmle_validation_dataset();
        let result = tmle(
            &dataset,
            "y",
            "treatment",
            &["x1", "x2"],
            TmleConfig::default(),
        )
        .unwrap();
        let treatment = col(&dataset, "treatment");
        for i in 0..result.n_obs {
            let (h, g, a) = (
                result.clever_covariate[i],
                result.propensity_scores[i],
                treatment[i],
            );
            let expected = a / g - (1.0 - a) / (1.0 - g);
            assert!(
                (h - expected).abs() <= 1e-12 * expected.abs(),
                "clever covariate at {i}: {h} vs {expected}"
            );
        }
    }

    #[test]
    fn test_tmle_influence_curve_mean_zero() {
        // The targeted fit solves the score equations, so the IC has mean 0 (at glm epsilon 1e-8)
        let dataset = create_tmle_validation_dataset();
        let result = tmle(
            &dataset,
            "y",
            "treatment",
            &["x1", "x2"],
            TmleConfig::default(),
        )
        .unwrap();
        let ic_mean: f64 = result.influence_curve.iter().sum::<f64>() / result.n_obs as f64;
        assert!(ic_mean.abs() < 1e-8, "IC mean {ic_mean:e}");
        assert_eq!(result.influence_curve.len(), result.n_obs);
    }

    #[test]
    fn test_tmle_counterfactuals_bounded() {
        let dataset = create_tmle_validation_dataset();
        let result = tmle(
            &dataset,
            "y",
            "treatment",
            &["x1", "x2"],
            TmleConfig::default(),
        )
        .unwrap();
        for i in 0..result.n_obs {
            assert!((0.0..=1.0).contains(&result.q_star_1[i]));
            assert!((0.0..=1.0).contains(&result.q_star_0[i]));
        }
        let ate_from_cf: f64 = result
            .q_star_1
            .iter()
            .zip(result.q_star_0.iter())
            .map(|(q1, q0)| q1 - q0)
            .sum::<f64>()
            / result.n_obs as f64;
        assert!((ate_from_cf - result.ate).abs() < 1e-12);
    }

    #[test]
    fn test_tmle_ps_truncation_bounds() {
        let dataset = create_tmle_validation_dataset();
        let config = TmleConfig {
            truncate_ps: (0.05, 0.95),
            ..Default::default()
        };
        let result = tmle(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();
        for &ps in &result.propensity_scores {
            assert!((0.05..=0.95).contains(&ps), "PS {ps} outside [0.05, 0.95]");
        }
        assert!(result.truncate_bounds == (0.05, 0.95));
    }

    #[test]
    fn test_tmle_linear_q_on_binary_outcome() {
        let dataset = create_tmle_validation_dataset();
        let config = TmleConfig {
            q_model: QModel::Linear,
            ..Default::default()
        };
        let r = tmle(&dataset, "y", "treatment", &["x1", "x2"], config).unwrap();
        assert!(r.ate.is_finite());
        assert!(r.ate_se > 0.0 && r.ate_se.is_finite());
        assert_eq!((r.q_model_iterations, r.targeting_iterations), (1, 1));
    }

    #[test]
    fn test_tmle_z_and_p_consistent() {
        let dataset = create_tmle_validation_dataset();
        let r = tmle(
            &dataset,
            "y",
            "treatment",
            &["x1", "x2"],
            TmleConfig::default(),
        )
        .unwrap();
        assert!((r.z_stat - r.ate / r.ate_se).abs() <= 1e-12 * r.z_stat.abs());
        assert_eq!(r.ate_p_value, normal_p_value_two_sided(r.z_stat));
        let zc = z_critical(0.95).unwrap();
        assert_eq!(r.ate_ci_lower, r.ate - zc * r.ate_se);
        assert_eq!(r.ate_ci_upper, r.ate + zc * r.ate_se);
    }

    // ── T3-WP29: the R reference case tmle_glm (validation/r/cases/causal_stdreg.R) ──

    const W: [&str; 4] = ["w1", "w2", "w3", "w4"];

    /// The case's dataset read directly, for the property tests that compare no R value.
    fn ctmle_sim() -> Dataset {
        crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/ctmle_sim.csv",
        )
    }

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

    fn with_col(ds: &Dataset, name: &str, values: Vec<f64>) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(Column::new(name.into(), values)).unwrap();
        Dataset::new(df)
    }

    fn tight() -> TmleConfig {
        // the case's glm.control(epsilon = 1e-14)
        TmleConfig {
            tolerance: 1e-14,
            ..Default::default()
        }
    }

    /// Every number the case records for one fit (`prefix` logit or linear).
    fn check_fit(case: &p2a_validation::Case, prefix: &str, r: &TmleResult) {
        let n = r.n_obs;
        let at: Vec<usize> = case
            .vector(&format!("{prefix}_rows_at"))
            .iter()
            .map(|&v| v as usize)
            .collect();
        let pick = |v: &[f64]| at.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let k = |s: &str| format!("{prefix}_{s}");
        Check::new(case)
            .scalar(&k("epsilon1"), r.fluctuation_coef, Tol::coef(n))
            .scalar(&k("epsilon0"), r.fluctuation_coef_control, Tol::coef(n))
            .scalar(&k("ate"), r.ate, Tol::coef(n))
            .scalar(&k("se"), r.ate_se, Tol::se(n))
            .scalar(&k("ci_lower"), r.ate_ci_lower, Tol::stat(n))
            .scalar(&k("ci_upper"), r.ate_ci_upper, Tol::stat(n))
            .scalar(&k("z"), r.z_stat, Tol::stat(n))
            .scalar(&k("p"), r.ate_p_value, Tol::p_value(n))
            .scalar(&k("n_truncated"), r.n_truncated as f64, Tol::exact())
            .vector(&k("q_init_at"), &pick(&r.initial_outcome), Tol::coef(n))
            .vector(&k("q_star_at"), &pick(&r.targeted_outcome), Tol::coef(n))
            .vector(&k("q_star1_at"), &pick(&r.q_star_1), Tol::coef(n))
            .vector(&k("q_star0_at"), &pick(&r.q_star_0), Tol::coef(n))
            .vector(&k("g_at"), &pick(&r.propensity_scores), Tol::coef(n))
            .vector(&k("ic_at"), &pick(&r.influence_curve), Tol::coef(n))
            .scalar(
                &k("ic_mean"),
                r.influence_curve.iter().sum::<f64>() / n as f64,
                Tol::new(0.0, 1e-12),
            )
            .finish();
    }

    #[test]
    fn test_validate_tmle_glm_logit() {
        let (case, ds) = crate::test_support::load_case("tmle_glm");
        let r = tmle(&ds, "yb", "a", &W, tight()).unwrap();
        check_fit(&case, "logit", &r);
        // tmle::tmle 2.1.1 itself (two clever covariates, cvQinit = FALSE, gbound as here)
        let n = r.n_obs;
        Check::new(&case)
            .scalar("tmle_pkg_ate", r.ate, Tol::coef(n))
            .scalar("tmle_pkg_se", r.ate_se, Tol::se(n))
            .finish();
        assert!(r.q_model_converged && r.g_model_converged && r.targeting_converged);
        assert!(r.outcome_range.is_none());
    }

    #[test]
    fn test_validate_tmle_glm_linear() {
        let (case, ds) = crate::test_support::load_case("tmle_glm");
        let c = TmleConfig {
            q_model: QModel::Linear,
            ..tight()
        };
        let r = tmle(&ds, "yc", "a", &W, c).unwrap();
        check_fit(&case, "linear", &r);
        assert!(r.g_model_converged && r.targeting_converged);
    }

    #[test]
    fn test_validate_tmle_glm_default_control() {
        // glm.control()'s epsilon 1e-8: within Tol::iterative(1e-8) of the epsilon-1e-14 fit
        let (case, ds) = crate::test_support::load_case("tmle_glm");
        let r = tmle(&ds, "yb", "a", &W, TmleConfig::default()).unwrap();
        let t = Tol::iterative(1e-8);
        Check::new(&case)
            .scalar("logit_ate", r.ate, t)
            .scalar("logit_se", r.ate_se, t)
            .scalar("logit_epsilon1", r.fluctuation_coef, t)
            .scalar("logit_epsilon0", r.fluctuation_coef_control, t)
            .finish();
        assert!(r.q_model_converged && r.g_model_converged && r.targeting_converged);
    }

    #[test]
    fn test_tmle_q_separation_warns() {
        // w3 separates ysep completely: glm warns (and converges), as R does
        let ds = ctmle_sim();
        let ysep: Vec<f64> = col(&ds, "w3")
            .iter()
            .map(|&v| if v > 0.2 { 1.0 } else { 0.0 })
            .collect();
        let r = tmle(
            &with_col(&ds, "ysep", ysep),
            "ysep",
            "a",
            &W,
            TmleConfig::default(),
        )
        .unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "Q model: glm.fit: fitted probabilities numerically 0 or 1 occurred"),
            "{:?}",
            r.warnings
        );
        assert!(r.q_model_converged, "{:?}", r.warnings);
        assert!(r.ate.is_finite());
    }

    #[test]
    fn test_tmle_nonconvergence_is_reported() {
        let ds = ctmle_sim();
        let c = TmleConfig {
            max_iter: 2,
            ..Default::default()
        };
        let r = tmle(&ds, "yb", "a", &W, c).unwrap();
        assert!(!r.q_model_converged && !r.g_model_converged);
        for method in ["TMLE outcome model (Q)", "TMLE propensity model (g)"] {
            let head = format!("{method} did not converge after 2 iterations");
            assert!(
                r.warnings.iter().any(|w| w.starts_with(&head)),
                "{head}: {:?}",
                r.warnings
            );
        }
        assert!(
            !r.warnings
                .iter()
                .any(|w| w.contains("algorithm did not converge"))
        );
        let text = r.to_string();
        assert!(text.contains("Warning: TMLE outcome model (Q) did not converge"));
    }

    #[test]
    fn test_validate_tmle_aliased_covariate_and_treatment() {
        // CAUS-73: an exact duplicate covariate is aliased (NaN, R's warning) and the identified
        // fit is unchanged; a covariate equal to the treatment leaves the effect unidentified.
        let (case, ds) = crate::test_support::load_case("tmle_glm");
        let dup: Vec<f64> = col(&ds, "w1").iter().map(|v| 2.0 * v + 1.0).collect();
        let covs = ["w1", "w2", "w3", "w4", "w1dup"];
        let r = tmle(
            &with_col(&ds, "w1dup", dup),
            "yb",
            "a",
            &covs,
            TmleConfig::default(),
        )
        .unwrap();
        Check::new(&case)
            .scalar("logit_ate", r.ate, Tol::iterative(1e-8))
            .finish();
        for model in ["Q model", "g model"] {
            let w = format!("{model}: Coefficients not defined because of singularities: w1dup");
            assert!(r.warnings.contains(&w), "{w}: {:?}", r.warnings);
        }
        let acopy = col(&ds, "a");
        let covs = ["w1", "w2", "w3", "w4", "acopy"];
        let e = tmle(
            &with_col(&ds, "acopy", acopy),
            "yb",
            "a",
            &covs,
            TmleConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("'a'") && message.contains("not identified")),
            "{e:?}"
        );
    }

    #[test]
    fn test_validate_tmle_logistic_q_rescales_continuous_outcome() {
        // CAUS-45. Before: the logistic Q on yc (outside [0, 1]) diverged and returned ATE +-1
        // with a non-convergence note. Now Y* = (Y - min) / (max - min), quasibinomial Q,
        // logistic fluctuation, estimates scaled back.
        let (case, ds) = crate::test_support::load_case("tmle_glm");
        let r = tmle(&ds, "yc", "a", &W, tight()).unwrap();
        let n = r.n_obs;
        let (lo, hi) = r.outcome_range.expect("yc is rescaled");
        Check::new(&case)
            .vector("rescaled_range", &[lo, hi], Tol::exact())
            .scalar("rescaled_ate", r.ate, Tol::coef(n))
            .scalar("rescaled_se", r.ate_se, Tol::se(n))
            .scalar("rescaled_epsilon1", r.fluctuation_coef, Tol::coef(n))
            .scalar(
                "rescaled_epsilon0",
                r.fluctuation_coef_control,
                Tol::coef(n),
            )
            .finish();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("rescaled from [") && w.contains("for the logistic Q model")),
            "{:?}",
            r.warnings
        );
        assert!(!r.warnings.iter().any(|w| w.contains("non-integer")));
        // property: the run on the manually rescaled outcome, scaled back
        let ys: Vec<f64> = col(&ds, "yc")
            .iter()
            .map(|v| (v - lo) / (hi - lo))
            .collect();
        let s = tmle(&with_col(&ds, "ycs", ys), "ycs", "a", &W, tight()).unwrap();
        assert!(s.outcome_range.is_none()); // already in [0, 1]: not rescaled again
        let t = Tol::new(1e-12, 1e-15);
        assert!(t.allows(r.ate, (hi - lo) * s.ate) && t.allows(r.ate_se, (hi - lo) * s.ate_se));
        assert!(t.allows(r.fluctuation_coef, s.fluctuation_coef));
        for i in 0..n {
            assert!(t.allows(r.q_star_1[i], lo + (hi - lo) * s.q_star_1[i]));
            assert!(t.allows(r.influence_curve[i], (hi - lo) * s.influence_curve[i]));
        }
        // a constant outcome cannot be rescaled
        let e = tmle(
            &with_col(&ds, "yk", vec![3.0; n]),
            "yk",
            "a",
            &W,
            TmleConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("is constant")),
            "{e:?}"
        );
    }

    /// A noisy logit design: intercept and two covariates, 0/1 and fractional outcomes, offset.
    fn lean_case(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let mut seed = 11_u64;
        let mut x = Array2::<f64>::ones((n, 3));
        let (mut yb, mut yf, mut off) = (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
        for i in 0..n {
            x[[i, 1]] = 2.0 * lcg_rand(&mut seed) - 1.0;
            x[[i, 2]] = 3.0 * lcg_rand(&mut seed);
            let p = plogis(-0.4 + 0.8 * x[[i, 1]] - 0.3 * x[[i, 2]]);
            yb[i] = if lcg_rand(&mut seed) < p { 1.0 } else { 0.0 };
            yf[i] = (0.6 * p + 0.4 * lcg_rand(&mut seed)).clamp(0.0, 1.0);
            off[i] = 0.5 * lcg_rand(&mut seed) - 0.25;
        }
        (x, yb, yf, off)
    }

    #[test]
    fn test_lean_logit_matches_shared_glm() {
        let (x, yb, yf, off) = lean_case(400);
        for ctl in [
            GlmControl {
                epsilon: 1e-8,
                max_iter: 100,
            },
            GlmControl {
                epsilon: 1e-14,
                max_iter: 100,
            },
        ] {
            for (y, family) in [(&yb, GlmFamily::Binomial), (&yf, GlmFamily::QuasiBinomial)] {
                for offset in [None, Some(off.view())] {
                    let mut spec = ctl.spec(family, GlmLink::Logit, None);
                    spec.offset = offset;
                    let glm = fit_glm(x.view(), y.view(), &spec).unwrap();
                    let lean = lean_logit(x.view(), y.view(), offset, LeanStart::Mustart, ctl)
                        .unwrap()
                        .expect("lean path");
                    assert_eq!(lean.iterations, glm.iterations);
                    for (a, b) in lean.coefficients.iter().zip(&glm.coefficients) {
                        assert!(Tol::new(1e-10, 1e-12).allows(*a, *b), "{a} vs {b}");
                    }
                    // from zero: the same maximum-likelihood estimate
                    let z = lean_logit(x.view(), y.view(), offset, LeanStart::Zero, ctl)
                        .unwrap()
                        .expect("lean path");
                    let tol = if ctl.epsilon < 1e-10 { 1e-9 } else { 1e-5 };
                    for (a, b) in z.coefficients.iter().zip(&glm.coefficients) {
                        assert!(Tol::new(tol, tol).allows(*a, *b), "{a} vs {b}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_well_conditioned_inverse_matches_sym_inverse() {
        use crate::econometrics::glm::well_conditioned_inverse;
        use crate::linalg::solve::{SingularPolicy, sym_inverse};
        let k = 3;
        let mut chol = vec![0.0; k * k];
        let (mut ys, mut rs) = (vec![0.0; k], vec![0.0; k]);
        let mut inv = Array2::<f64>::zeros((k, k));
        // well conditioned: the inverse of sym_inverse
        let g = [4.1, 0.7, -0.3, 0.0, 2.9, 0.45, 0.0, 0.0, 1.7];
        assert!(well_conditioned_inverse(
            &g, k, &mut chol, &mut ys, &mut rs, &mut inv
        ));
        let full = Array2::from_shape_fn(
            (k, k),
            |(a, c)| {
                if c >= a { g[a * k + c] } else { g[c * k + a] }
            },
        );
        let reference = sym_inverse(full.view(), SingularPolicy::Error, "g").unwrap();
        for (a, b) in inv.iter().zip(reference.inverse.iter()) {
            assert!(Tol::new(1e-13, 1e-15).allows(*a, *b), "{a} vs {b}");
        }
        // nearly collinear (equilibrated condition number above 100), singular, indefinite
        let near = [1.0, 0.999, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let singular = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let indefinite = [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for g in [near, singular, indefinite] {
            assert!(!well_conditioned_inverse(
                &g, k, &mut chol, &mut ys, &mut rs, &mut inv
            ));
        }
    }

    #[test]
    fn test_lean_logit_leaves_the_path_where_glm_reports() {
        let (x, yb, _, _) = lean_case(200);
        let ctl = GlmControl {
            epsilon: 1e-8,
            max_iter: 100,
        };
        // complete separation on x1
        let ysep = x.column(1).mapv(|v| if v > 0.1 { 1.0 } else { 0.0 });
        assert!(
            lean_logit(x.view(), ysep.view(), None, LeanStart::Mustart, ctl)
                .unwrap()
                .is_none()
        );
        // an aliased column
        let mut xa = Array2::<f64>::ones((200, 4));
        xa.slice_mut(s![.., ..3]).assign(&x);
        xa.column_mut(3)
            .assign(&x.column(1).mapv(|v| 2.0 * v + 1.0));
        assert!(
            lean_logit(xa.view(), yb.view(), None, LeanStart::Mustart, ctl)
                .unwrap()
                .is_none()
        );
        // no convergence within max_iter
        let short = GlmControl {
            epsilon: 1e-8,
            max_iter: 1,
        };
        assert!(
            lean_logit(x.view(), yb.view(), None, LeanStart::Mustart, short)
                .unwrap()
                .is_none()
        );
        // a constant outcome
        let ones = Array1::<f64>::ones(200);
        assert!(
            lean_logit(x.view(), ones.view(), None, LeanStart::Mustart, ctl)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_logit_nuisance_fold_fit_notes_do_not_depend_on_the_path() {
        // A fractional outcome fitted on a fold's training rows (zero weights elsewhere): the
        // lean path reports nothing, and the shared GLM's fallback (here forced by a design
        // too ill conditioned for the lean Cholesky) must not add summary.glm's dispersion
        // warning about the zero-weight rows.
        let (x, _, yf, _) = lean_case(300);
        let mut xn = Array2::<f64>::ones((300, 4));
        xn.slice_mut(s![.., ..3]).assign(&x);
        let mut seed = 7_u64;
        for i in 0..300 {
            xn[[i, 3]] = x[[i, 1]] + 0.02 * (lcg_rand(&mut seed) - 0.5);
        }
        let train = Array1::from_shape_fn(300, |i| if i % 5 == 0 { 0.0 } else { 1.0 });
        let names: Vec<String> = ["(Intercept)", "x1", "x2", "x1b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ctl = GlmControl {
            epsilon: 1e-8,
            max_iter: 100,
        };
        let rows: Vec<usize> = (0..300).filter(|i| i % 5 != 0).collect();
        let (xs, ys) = (xn.select(Axis(0), &rows), yf.select(Axis(0), &rows));
        assert!(
            lean_logit(xs.view(), ys.view(), None, LeanStart::Mustart, ctl)
                .unwrap()
                .is_none(),
            "the design must leave the lean path"
        );
        let spec = ctl.spec(GlmFamily::QuasiBinomial, GlmLink::Logit, Some(train.view()));
        let glm = fit_glm(xn.view(), yf.view(), &spec).unwrap();
        assert!(glm.warnings.iter().any(|w| w.contains("zero weight")));
        let fit = logit_nuisance(
            xn.view(),
            yf.view(),
            GlmFamily::QuasiBinomial,
            Some(train.view()),
            &names,
            ctl,
            None,
        )
        .unwrap();
        assert!(fit.converged && fit.aliased.is_empty());
        assert!(fit.warnings.is_empty(), "{:?}", fit.warnings);
    }

    #[test]
    fn test_outcome_scale_names_constant_and_non_finite_outcomes() {
        let msg = |y: &[f64]| match outcome_scale(ArrayView1::from(y), QModel::Logistic, "y") {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            msg(&[2.5, 2.5, 2.5]),
            "outcome 'y' is constant (2.5); nothing to estimate"
        );
        assert_eq!(
            msg(&[0.3, f64::INFINITY, 1.7]),
            "outcome 'y' contains a non-finite value"
        );
        assert_eq!(
            msg(&[0.3, f64::NAN, 1.7]),
            "outcome 'y' contains a non-finite value"
        );
    }

    #[test]
    fn test_tmle_refuses_non_binary_treatment() {
        let ds = ctmle_sim();
        let a2: Vec<f64> = col(&ds, "a").iter().map(|v| 2.0 * v).collect();
        let first = a2.iter().position(|&v| v == 2.0).unwrap();
        let e = tmle(
            &with_col(&ds, "a2", a2),
            "yb",
            "a2",
            &W,
            TmleConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains(&format!("treatment 'a2' must be coded 0/1 (0-based row {first} is 2)"))),
            "{e:?}"
        );
    }

    #[test]
    fn test_tmle_rejects_bad_config() {
        let ds = ctmle_sim();
        let bad = [
            TmleConfig {
                max_iter: 0,
                ..Default::default()
            },
            TmleConfig {
                tolerance: f64::NAN,
                ..Default::default()
            },
            TmleConfig {
                tolerance: 0.0,
                ..Default::default()
            },
            TmleConfig {
                truncate_ps: (0.5, 0.5),
                ..Default::default()
            },
            TmleConfig {
                truncate_ps: (0.0, 0.9),
                ..Default::default()
            },
        ];
        for c in bad {
            let e = tmle(&ds, "yb", "a", &W, c.clone()).unwrap_err();
            assert!(
                matches!(e, EconError::InvalidSpecification { .. }),
                "{c:?}: {e:?}"
            );
        }
    }

    #[test]
    fn test_validate_tmle_scale_free() {
        // glm.fit's deviance rule does not depend on the scale of the covariates
        let (case, ds) = crate::test_support::load_case("tmle_glm");
        let base = tmle(&ds, "yb", "a", &W, TmleConfig::default()).unwrap();
        for k in [1e-6, 1e6] {
            let mut scaled = ds.clone();
            for w in W {
                let v: Vec<f64> = col(&ds, w).iter().map(|x| k * x).collect();
                scaled = with_col(&scaled, w, v);
            }
            let r = tmle(&scaled, "yb", "a", &W, TmleConfig::default()).unwrap();
            assert!(
                r.q_model_converged && r.g_model_converged && r.targeting_converged,
                "k = {k}: {:?}",
                r.warnings
            );
            assert!(
                Tol::new(1e-9, 0.0).allows(r.ate, base.ate),
                "k = {k}: {} vs {}",
                r.ate,
                base.ate
            );
            Check::new(&case)
                .scalar("logit_ate", r.ate, Tol::iterative(1e-8))
                .finish();
        }
    }

    #[test]
    fn test_tmle_bivariate_fluctuation_solves_both_scores() {
        // tmle 2.1.1's fluctuation solves sum H1 (y - Q*) = 0 and sum H0 (y - Q*) = 0 separately
        let ds = ctmle_sim();
        let (y, a) = (col(&ds, "yb"), col(&ds, "a"));
        for (outcome, q_model) in [("yb", QModel::Logistic), ("yc", QModel::Linear)] {
            let c = TmleConfig { q_model, ..tight() };
            let r = tmle(&ds, outcome, "a", &W, c).unwrap();
            let yv = if outcome == "yb" {
                y.clone()
            } else {
                col(&ds, "yc")
            };
            let (mut s1, mut s0, mut m1, mut m0) = (0.0, 0.0, 0.0_f64, 0.0_f64);
            for i in 0..r.n_obs {
                let g = r.propensity_scores[i];
                let res = yv[i] - r.targeted_outcome[i];
                let (h1, h0) = (a[i] / g, (1.0 - a[i]) / (1.0 - g));
                s1 += h1 * res;
                s0 += h0 * res;
                m1 = m1.max((h1 * res).abs());
                m0 = m0.max((h0 * res).abs());
            }
            assert!(s1.abs() <= 1e-10 * m1 * r.n_obs as f64, "{outcome}: {s1:e}");
            assert!(s0.abs() <= 1e-10 * m0 * r.n_obs as f64, "{outcome}: {s0:e}");
        }
    }

    #[test]
    fn test_q_model_parses_aliases_and_rejects_unknown() {
        QModel::choices().validate().unwrap();
        GModel::choices().validate().unwrap();
        for s in ["Linear", "gaussian", "continuous"] {
            assert_eq!(s.parse::<QModel>().unwrap(), QModel::Linear);
        }
        for s in ["LOGIT", "binary", "binomial"] {
            assert_eq!(s.parse::<QModel>().unwrap(), QModel::Logistic);
        }
        assert_eq!("glm".parse::<GModel>().unwrap(), GModel::Logistic);
        let e = "probit".parse::<QModel>().unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("unknown q_model 'probit'") && message.contains("logistic, linear")),
            "{e:?}"
        );
    }

    // ── T2-WP13: missing values and column types (R case causal_mediation_na) ──

    #[test]
    fn test_validate_na_tmle_columns() {
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        // continuous y (the logistic default is CAUS-45)
        let run = |ds: &Dataset, y: &str, d: &str, x: &[&str]| {
            tmle(
                ds,
                y,
                d,
                x,
                TmleConfig {
                    q_model: QModel::Linear,
                    ..Default::default()
                },
            )
        };
        nf::assert_refusals(&case, &ds, true, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| vec![r.ate, r.ate_se]);
    }
}
