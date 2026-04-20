//! Longitudinal Targeted Maximum Likelihood Estimation (LTMLE) for causal inference.
//!
//! LTMLE extends TMLE to longitudinal settings with time-varying treatments and
//! confounders. It estimates causal effects under dynamic treatment regimes using
//! a sequential regression approach.
//!
//! # Algorithm Overview
//!
//! LTMLE performs estimation by iterating backwards through time points T to 1:
//!
//! 1. **Initial Step (t = T)**: Fit outcome model Q_T(A_T, L_T) = E[Y_T | A_T, L_T]
//!
//! 2. **Sequential Regression (t = T-1, ..., 1)**:
//!    - Use predictions from t+1 as pseudo-outcomes
//!    - Fit Q_t(A_t, L_t) = E[Q_{t+1}^* | A_t, L_t]
//!    - Apply targeting step with clever covariate
//!
//! 3. **Targeting Step at Each Time Point**:
//!    - Compute clever covariate H_t = cumulative product of 1/g(A_s | L_s)
//!    - Fluctuate Q_t using: Q_t^* = expit(logit(Q_t) + epsilon * H_t)
//!
//! 4. **Final Estimate**: E[Y^{a_1,...,a_T}] = mean(Q_1^*)
//!
//! # Intervention Types
//!
//! - **Static "Always Treat"**: Set all A_t = 1
//! - **Static "Never Treat"**: Set all A_t = 0
//! - **Dynamic**: User-specified treatment rule based on history
//!
//! # References
//!
//! - van der Laan, M.J. & Gruber, S. (2012). "Targeted Minimum Loss Based
//!   Estimation of Causal Effects of Multiple Time Point Interventions."
//!   *The International Journal of Biostatistics*, 8(1), Article 9.
//!   https://doi.org/10.1515/1557-4679.1370
//!
//! - Lendle, S.D., Schwab, J., Petersen, M.L., & van der Laan, M.J. (2017).
//!   "ltmle: An R Package Implementing Targeted Minimum Loss-Based Estimation
//!   for Longitudinal Data." *Journal of Statistical Software*, 81(1), 1-21.
//!   https://doi.org/10.18637/jss.v081.i01
//!
//! - van der Laan, M.J. & Rose, S. (2011). *Targeted Learning: Causal Inference
//!   for Observational and Experimental Data*. Springer. Chapter 6.
//!   https://doi.org/10.1007/978-1-4419-9782-1
//!
//! - R package `ltmle`: https://cran.r-project.org/package=ltmle
//!   (Schwab, Lendle, Petersen, van der Laan)
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::econometrics::{run_ltmle, LtmleConfig, InterventionType, LtmleData};
//!
//! // Create longitudinal data with 2 time points
//! let data = LtmleData {
//!     outcomes: vec![y_1.clone(), y_2.clone()],
//!     treatments: vec![a_1.clone(), a_2.clone()],
//!     covariates: vec![l_1.clone(), l_2.clone()],
//! };
//!
//! let config = LtmleConfig {
//!     intervention: InterventionType::AlwaysTreat,
//!     gbounds: (0.01, 0.99),
//!     ..Default::default()
//! };
//!
//! let result = run_ltmle(&data, config)?;
//! println!("ATE: {:.4} (SE: {:.4})", result.ate, result.se);
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::{SignificanceLevel, logistic_cdf, normal_cdf};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of intervention for LTMLE.
///
/// Specifies the counterfactual treatment regime to estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InterventionType {
    /// Always treat: A_t = 1 for all t
    #[default]
    AlwaysTreat,
    /// Never treat: A_t = 0 for all t
    NeverTreat,
    /// Custom static intervention: A_t = specified value for all t
    Static(bool),
}

impl fmt::Display for InterventionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterventionType::AlwaysTreat => write!(f, "Always Treat (A=1)"),
            InterventionType::NeverTreat => write!(f, "Never Treat (A=0)"),
            InterventionType::Static(treat) => {
                write!(f, "Static (A={})", if *treat { 1 } else { 0 })
            }
        }
    }
}

/// Outcome model specification for LTMLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LtmleQModel {
    /// Logistic regression for binary outcomes (default)
    #[default]
    Logistic,
    /// Linear regression for continuous outcomes
    Linear,
}

impl fmt::Display for LtmleQModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LtmleQModel::Logistic => write!(f, "Logistic"),
            LtmleQModel::Linear => write!(f, "Linear"),
        }
    }
}

/// Configuration for LTMLE estimation.
#[derive(Debug, Clone)]
pub struct LtmleConfig {
    /// Type of intervention to estimate
    pub intervention: InterventionType,
    /// Outcome model specification
    pub q_model: LtmleQModel,
    /// Propensity score truncation bounds (min, max)
    /// Default: (0.01, 0.99) to avoid extreme weights
    pub gbounds: (f64, f64),
    /// Maximum iterations for logistic regression
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for LtmleConfig {
    fn default() -> Self {
        Self {
            intervention: InterventionType::AlwaysTreat,
            q_model: LtmleQModel::Logistic,
            gbounds: (0.01, 0.99),
            max_iter: 100,
            tolerance: 1e-8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Longitudinal data structure for LTMLE.
///
/// Data is organized by time point (t = 1, ..., T):
/// - `outcomes[t-1]`: Outcome Y_t at time t (only last time point typically observed)
/// - `treatments[t-1]`: Treatment A_t at time t (binary: 0 or 1)
/// - `covariates[t-1]`: Time-varying covariates L_t at time t (n x k_t matrix)
///
/// The covariates L_t may be affected by prior treatment A_{t-1}, which is the
/// defining characteristic of time-varying confounding.
///
/// # Temporal Ordering
///
/// At each time point t, the ordering is: L_t -> A_t -> Y_t
///
/// The causal structure is:
/// L_1 -> A_1 -> L_2 -> A_2 -> ... -> L_T -> A_T -> Y_T
#[derive(Debug, Clone)]
pub struct LtmleData {
    /// Outcome at each time point. For survival/final outcomes, only the last
    /// element may contain the actual outcome; earlier elements can be zeros
    /// or intermediate outcomes.
    pub outcomes: Vec<Array1<f64>>,
    /// Treatment indicator at each time point (0 or 1)
    pub treatments: Vec<Array1<f64>>,
    /// Time-varying covariates at each time point (n x k_t matrix)
    pub covariates: Vec<Array2<f64>>,
}

impl LtmleData {
    /// Create new LTMLE data from arrays.
    pub fn new(
        outcomes: Vec<Array1<f64>>,
        treatments: Vec<Array1<f64>>,
        covariates: Vec<Array2<f64>>,
    ) -> EconResult<Self> {
        let t_count = outcomes.len();

        // Validate consistency
        if treatments.len() != t_count || covariates.len() != t_count {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Number of time points must be consistent across outcomes ({}), \
                     treatments ({}), and covariates ({})",
                    outcomes.len(),
                    treatments.len(),
                    covariates.len()
                ),
            });
        }

        if t_count == 0 {
            return Err(EconError::EmptyDataset);
        }

        // Get sample size from first time point
        let n = outcomes[0].len();

        // Validate all arrays have same number of observations
        for (t, (y, (a, l))) in outcomes
            .iter()
            .zip(treatments.iter().zip(covariates.iter()))
            .enumerate()
        {
            if y.len() != n || a.len() != n || l.nrows() != n {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Time point {} has inconsistent sample sizes: \
                         outcomes={}, treatments={}, covariates={}",
                        t + 1,
                        y.len(),
                        a.len(),
                        l.nrows()
                    ),
                });
            }
        }

        // Validate treatments are binary
        for (t, a) in treatments.iter().enumerate() {
            for val in a.iter() {
                if !(*val == 0.0
                    || *val == 1.0
                    || (*val - 0.0).abs() < 1e-10
                    || (*val - 1.0).abs() < 1e-10)
                {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "Treatment at time {} must be binary (0 or 1), found value {}",
                            t + 1,
                            val
                        ),
                    });
                }
            }
        }

        Ok(Self {
            outcomes,
            treatments,
            covariates,
        })
    }

    /// Number of time points T.
    pub fn time_points(&self) -> usize {
        self.outcomes.len()
    }

    /// Number of observations n.
    pub fn n_obs(&self) -> usize {
        self.outcomes[0].len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from LTMLE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtmleResult {
    /// Average treatment effect (ATE) estimate: E[Y^{always treat}] - E[Y^{never treat}]
    pub ate: f64,
    /// Standard error of ATE (from influence curve)
    pub se: f64,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
    /// Two-sided p-value for H0: ATE = 0
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Z-statistic (ATE / SE)
    pub z_stat: f64,

    /// Counterfactual mean under treatment: E[Y^{always treat}]
    pub psi_treated: f64,
    /// Standard error for psi_treated
    pub psi_treated_se: f64,
    /// Counterfactual mean under control: E[Y^{never treat}]
    pub psi_control: f64,
    /// Standard error for psi_control
    pub psi_control_se: f64,

    /// Influence curve for the ATE (for each observation)
    pub influence_curve: Vec<f64>,

    /// Propensity scores at each time point: g(A_t | L_t)
    pub propensity_scores: Vec<Vec<f64>>,
    /// Cumulative inverse probability weights H_t at each time point
    pub clever_covariates: Vec<Vec<f64>>,
    /// Targeted predictions Q_t^* at each time point
    pub targeted_predictions: Vec<Vec<f64>>,
    /// Fluctuation coefficients (epsilon) at each time point
    pub fluctuation_coefs: Vec<f64>,

    /// Number of observations
    pub n_obs: usize,
    /// Number of time points
    pub time_points: usize,
    /// Number of treated at each time point
    pub n_treated_by_time: Vec<usize>,
    /// Number of observations with truncated propensity scores at each time point
    pub n_truncated_by_time: Vec<usize>,

    /// Intervention type used for treated counterfactual
    pub intervention_treated: InterventionType,
    /// Intervention type used for control counterfactual
    pub intervention_control: InterventionType,
    /// Outcome model type used
    pub q_model_type: LtmleQModel,
    /// Propensity score truncation bounds used
    pub gbounds: (f64, f64),

    /// Whether all models converged
    pub converged: bool,
    /// Warnings generated during estimation
    pub warnings: Vec<String>,
}

impl fmt::Display for LtmleResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Longitudinal Targeted Maximum Likelihood Estimation (LTMLE)"
        )?;
        writeln!(
            f,
            "============================================================"
        )?;
        writeln!(f)?;
        writeln!(f, "Time Points: {}", self.time_points)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f)?;
        writeln!(f, "Counterfactual Means:")?;
        writeln!(
            f,
            "  E[Y^{{always treat}}]: {:>10.4} (SE: {:.4})",
            self.psi_treated, self.psi_treated_se
        )?;
        writeln!(
            f,
            "  E[Y^{{never treat}}]:  {:>10.4} (SE: {:.4})",
            self.psi_control, self.psi_control_se
        )?;
        writeln!(f)?;
        writeln!(f, "Average Treatment Effect (ATE):")?;
        writeln!(f, "  Estimate:   {:>12.4}", self.ate)?;
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
            "  95% CI:     [{:.4}, {:.4}]",
            self.ci_lower, self.ci_upper
        )?;
        writeln!(f)?;
        writeln!(f, "Treatment Summary by Time:")?;
        for (t, &n) in self.n_treated_by_time.iter().enumerate() {
            writeln!(
                f,
                "  Time {}: {} treated, {} truncated PS",
                t + 1,
                n,
                self.n_truncated_by_time[t]
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Model Specification:")?;
        writeln!(f, "  Outcome Model (Q): {}", self.q_model_type)?;
        writeln!(
            f,
            "  PS Truncation:     [{:.2}, {:.2}]",
            self.gbounds.0, self.gbounds.1
        )?;
        writeln!(f)?;
        writeln!(f, "Targeting Step:")?;
        for (t, &eps) in self.fluctuation_coefs.iter().enumerate() {
            writeln!(f, "  Time {}: epsilon = {:.6}", t + 1, eps)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Convergence: {}",
            if self.converged { "Yes" } else { "No" }
        )?;
        writeln!(f)?;
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
// Main LTMLE Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run Longitudinal TMLE with full configuration.
///
/// Estimates counterfactual means and the Average Treatment Effect under
/// specified intervention regimes. The default computes E[Y^{always treat}]
/// vs E[Y^{never treat}].
///
/// # Algorithm (van der Laan & Gruber 2012, Lendle et al. 2017)
///
/// For T time points with L_t (covariates), A_t (treatments), Y_T (outcome):
///
/// 1. **Propensity Score Estimation**: At each t, fit g_t(L_t) = P(A_t = 1 | L_t)
///
/// 2. **Backward Sequential Regression**:
///    - At t = T: Fit Q_T = E[Y_T | A_T, L_T]
///    - At t = T-1, ..., 1: Fit Q_t = E[Q_{t+1}^* | A_t, L_t]
///
/// 3. **Targeting at Each Time Point**:
///    - Compute clever covariate: H_t = prod_{s<=t} 1/g_s(A_s | L_s)
///    - Fit epsilon in: logit(Q_t^*) = logit(Q_t) + epsilon * H_t
///
/// 4. **Compute Estimates**:
///    - psi = mean(Q_1^*) under the intervention regime
///    - Standard errors from influence curve
///
/// # Arguments
///
/// * `data` - Longitudinal data with outcomes, treatments, and covariates
/// * `config` - LTMLE configuration (intervention type, model spec, etc.)
///
/// # Returns
///
/// `LtmleResult` containing ATE estimate, standard errors, counterfactual means,
/// and diagnostic information.
///
/// # References
///
/// - van der Laan & Gruber (2012), Int J Biostat 8(1), Article 9
/// - Lendle et al. (2017), JSS 81(1), Algorithm 1
pub fn run_ltmle(data: &LtmleData, config: LtmleConfig) -> EconResult<LtmleResult> {
    let mut warnings = Vec::new();
    let n = data.n_obs();
    let t_max = data.time_points();

    if t_max < 2 {
        return Err(EconError::InvalidSpecification {
            message:
                "LTMLE requires at least 2 time points. For single time point, use standard TMLE."
                    .to_string(),
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Estimate counterfactual mean under "always treat" (psi_1)
    // ═══════════════════════════════════════════════════════════════════════

    let (
        psi_treated,
        ic_treated,
        ps_treated,
        h_treated,
        q_star_treated,
        eps_treated,
        converged_treated,
        n_treated_by_time,
        n_truncated_treated,
    ) = estimate_counterfactual_mean(data, InterventionType::AlwaysTreat, &config, &mut warnings)?;

    // ═══════════════════════════════════════════════════════════════════════
    // Estimate counterfactual mean under "never treat" (psi_0)
    // ═══════════════════════════════════════════════════════════════════════

    let (
        psi_control,
        ic_control,
        _ps_control,
        _h_control,
        _q_star_control,
        _eps_control,
        converged_control,
        _n_control_by_time,
        n_truncated_control,
    ) = estimate_counterfactual_mean(data, InterventionType::NeverTreat, &config, &mut warnings)?;

    // ═══════════════════════════════════════════════════════════════════════
    // Compute ATE = psi_1 - psi_0 and its standard error
    // ═══════════════════════════════════════════════════════════════════════

    let ate = psi_treated - psi_control;

    // Influence curve for ATE: IC_ATE = IC_1 - IC_0
    // (van der Laan & Rose 2011, Chapter 6)
    let ic_ate: Array1<f64> = ic_treated
        .iter()
        .zip(ic_control.iter())
        .map(|(&ic1, &ic0)| ic1 - ic0)
        .collect();

    // Variance from influence curve: Var(ATE) = Var(IC) / n
    let ic_mean: f64 = ic_ate.iter().sum::<f64>() / n as f64;
    let ic_var: f64 =
        ic_ate.iter().map(|&ic| (ic - ic_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let ate_var = ic_var / n as f64;
    let se = ate_var.sqrt();

    // Standard errors for individual counterfactual means
    let ic_1_mean: f64 = ic_treated.iter().sum::<f64>() / n as f64;
    let ic_1_var: f64 = ic_treated
        .iter()
        .map(|&ic| (ic - ic_1_mean).powi(2))
        .sum::<f64>()
        / (n - 1).max(1) as f64;
    let psi_treated_se = (ic_1_var / n as f64).sqrt();

    let ic_0_mean: f64 = ic_control.iter().sum::<f64>() / n as f64;
    let ic_0_var: f64 = ic_control
        .iter()
        .map(|&ic| (ic - ic_0_mean).powi(2))
        .sum::<f64>()
        / (n - 1).max(1) as f64;
    let psi_control_se = (ic_0_var / n as f64).sqrt();

    // Wald-type inference
    let z_stat = if se > 0.0 && se.is_finite() {
        ate / se
    } else {
        0.0
    };

    let z_crit = 1.96;
    let ci_lower = ate - z_crit * se;
    let ci_upper = ate + z_crit * se;

    let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));
    let significance = SignificanceLevel::from_p_value(p_value);

    // ═══════════════════════════════════════════════════════════════════════
    // Merge truncation counts
    // ═══════════════════════════════════════════════════════════════════════

    let n_truncated_by_time: Vec<usize> = n_truncated_treated
        .iter()
        .zip(n_truncated_control.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    Ok(LtmleResult {
        ate,
        se,
        ci_lower,
        ci_upper,
        p_value,
        significance,
        z_stat,
        psi_treated,
        psi_treated_se,
        psi_control,
        psi_control_se,
        influence_curve: ic_ate.to_vec(),
        propensity_scores: ps_treated.iter().map(|p| p.to_vec()).collect(),
        clever_covariates: h_treated.iter().map(|h| h.to_vec()).collect(),
        targeted_predictions: q_star_treated.iter().map(|q| q.to_vec()).collect(),
        fluctuation_coefs: eps_treated,
        n_obs: n,
        time_points: t_max,
        n_treated_by_time,
        n_truncated_by_time,
        intervention_treated: InterventionType::AlwaysTreat,
        intervention_control: InterventionType::NeverTreat,
        q_model_type: config.q_model,
        gbounds: config.gbounds,
        converged: converged_treated && converged_control,
        warnings,
    })
}

/// Estimate a single counterfactual mean E[Y^d] under intervention d.
///
/// This is the core LTMLE estimation for one intervention regime.
/// The algorithm follows a simplified g-computation with TMLE targeting:
///
/// 1. Fit propensity scores at each time point
/// 2. Iterate backwards, fitting outcome models and targeting
/// 3. Return the mean of the initial (t=1) counterfactual predictions
///
/// Returns: (psi, influence_curve, propensity_scores, clever_covariates,
///           targeted_predictions, fluctuation_coefs, converged, n_treated, n_truncated)
#[allow(clippy::type_complexity)]
fn estimate_counterfactual_mean(
    data: &LtmleData,
    intervention: InterventionType,
    config: &LtmleConfig,
    warnings: &mut Vec<String>,
) -> EconResult<(
    f64,              // psi
    Array1<f64>,      // influence_curve
    Vec<Array1<f64>>, // propensity_scores
    Vec<Array1<f64>>, // clever_covariates
    Vec<Array1<f64>>, // targeted_predictions
    Vec<f64>,         // fluctuation_coefs
    bool,             // converged
    Vec<usize>,       // n_treated_by_time
    Vec<usize>,       // n_truncated_by_time
)> {
    let n = data.n_obs();
    let t_max = data.time_points();

    // Get the counterfactual treatment value
    let d = match intervention {
        InterventionType::AlwaysTreat | InterventionType::Static(true) => 1.0,
        InterventionType::NeverTreat | InterventionType::Static(false) => 0.0,
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Step 1: Estimate propensity scores g_t(L_t) = P(A_t = 1 | L_t)
    // ═══════════════════════════════════════════════════════════════════════

    let mut propensity_scores: Vec<Array1<f64>> = Vec::with_capacity(t_max);
    let mut n_treated_by_time: Vec<usize> = Vec::with_capacity(t_max);
    let mut n_truncated_by_time: Vec<usize> = Vec::with_capacity(t_max);
    let mut overall_converged = true;

    for t in 0..t_max {
        let a_t = &data.treatments[t];
        let l_t = &data.covariates[t];

        // Add intercept to covariates
        let x_t = add_intercept(l_t);

        // Fit logistic regression for propensity score
        let (ps_raw, _beta, converged, _iter) =
            fit_logistic_model(&x_t, a_t, config.max_iter, config.tolerance)?;

        if !converged {
            warnings.push(format!(
                "Propensity model at time {} did not converge",
                t + 1
            ));
            overall_converged = false;
        }

        // Truncate propensity scores
        let (ps_min, ps_max) = config.gbounds;
        let mut n_truncated = 0;
        let ps_truncated: Array1<f64> = ps_raw.mapv(|p| {
            if p < ps_min || p > ps_max {
                n_truncated += 1;
            }
            p.max(ps_min).min(ps_max)
        });

        n_treated_by_time.push(a_t.iter().filter(|&&v| v >= 0.5).count());
        n_truncated_by_time.push(n_truncated);
        propensity_scores.push(ps_truncated);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 2: Compute clever covariates for targeting
    // ═══════════════════════════════════════════════════════════════════════

    // For LTMLE, the clever covariate at time t is the cumulative IPW weight
    // H_t = prod_{s=1}^{t} 1/g(d|L_s) where g(d|L) is the probability of receiving d
    //
    // For d=1 (always treat): g(1|L) = propensity score
    // For d=0 (never treat): g(0|L) = 1 - propensity score

    let mut clever_covariates: Vec<Array1<f64>> = Vec::with_capacity(t_max);

    for t in 0..t_max {
        let h_t: Array1<f64> = (0..n)
            .map(|i| {
                // Compute cumulative product of 1/g(d|L_s) for s=1..t
                let mut h = 1.0;
                for s in 0..=t {
                    let g_s = propensity_scores[s][i];
                    let g_d = if d == 1.0 { g_s } else { 1.0 - g_s };
                    h *= 1.0 / g_d;
                }
                h
            })
            .collect();

        clever_covariates.push(h_t);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 3: Sequential regression with targeting (backward from T to 1)
    // ═══════════════════════════════════════════════════════════════════════

    let mut targeted_predictions: Vec<Array1<f64>> = vec![Array1::zeros(n); t_max];
    let mut fluctuation_coefs: Vec<f64> = vec![0.0; t_max];

    // Initialize: at t=T, the pseudo-outcome is the actual outcome
    let y_final = data.outcomes[t_max - 1].clone();
    let mut pseudo_outcome = y_final.clone();

    // Backward iteration: t = T, T-1, ..., 1
    for t in (0..t_max).rev() {
        let a_t = &data.treatments[t];
        let l_t = &data.covariates[t];
        let h_t = &clever_covariates[t];

        // Build design matrix with intercept
        let x_t = add_intercept(l_t);
        let k = x_t.ncols();

        // Build design matrix with OBSERVED treatment
        let mut x_with_a = Array2::zeros((n, k + 1));
        for i in 0..n {
            for j in 0..k {
                x_with_a[[i, j]] = x_t[[i, j]];
            }
            x_with_a[[i, k]] = a_t[i]; // Use OBSERVED treatment
        }

        // Fit initial outcome model Q_t(A_t, L_t) using observed treatment
        let (_q_init_obs, beta_q, q_converged, _iter) = match config.q_model {
            LtmleQModel::Logistic => fit_logistic_model(
                &x_with_a,
                &pseudo_outcome,
                config.max_iter,
                config.tolerance,
            )?,
            LtmleQModel::Linear => fit_linear_model(&x_with_a, &pseudo_outcome)?,
        };

        if !q_converged {
            warnings.push(format!("Outcome model at time {} did not converge", t + 1));
            overall_converged = false;
        }

        // Predict under COUNTERFACTUAL treatment (A_t = d)
        let mut x_with_d = x_with_a.clone();
        for i in 0..n {
            x_with_d[[i, k]] = d;
        }

        let q_init: Array1<f64> = match config.q_model {
            LtmleQModel::Logistic => {
                let z = x_with_d.dot(&beta_q);
                z.mapv(logistic_cdf)
            }
            LtmleQModel::Linear => x_with_d.dot(&beta_q),
        };

        // ═══════════════════════════════════════════════════════════════════
        // Targeting Step: Fluctuate Q_t using clever covariate H_t
        // ═══════════════════════════════════════════════════════════════════

        // For the targeting step, we use the counterfactual predictions
        let (epsilon, targeting_converged) =
            fit_targeting_model(&pseudo_outcome, &q_init, h_t, config.q_model)?;

        if !targeting_converged {
            warnings.push(format!("Targeting step at time {} did not converge", t + 1));
            overall_converged = false;
        }

        fluctuation_coefs[t] = epsilon;

        // Compute targeted predictions Q_t^* under counterfactual
        let q_star: Array1<f64> = match config.q_model {
            LtmleQModel::Logistic => (0..n)
                .map(|i| {
                    let logit_q = logit(q_init[i]);
                    logistic_cdf(logit_q + epsilon * h_t[i])
                })
                .collect(),
            LtmleQModel::Linear => (0..n).map(|i| q_init[i] + epsilon * h_t[i]).collect(),
        };

        targeted_predictions[t] = q_star.clone();

        // Update pseudo-outcome for next iteration (going backward)
        pseudo_outcome = q_star;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 4: Compute final estimate and influence curve
    // ═══════════════════════════════════════════════════════════════════════

    // The final estimate is the mean of Q_1^* (targeted prediction at first time point)
    let psi: f64 = targeted_predictions[0].iter().sum::<f64>() / n as f64;

    // Simplified influence curve for g-computation estimator
    // IC(O) = Q_1^*(O) - psi
    // This is the influence curve for the substitution estimator
    let influence_curve: Array1<f64> = (0..n).map(|i| targeted_predictions[0][i] - psi).collect();

    Ok((
        psi,
        influence_curve,
        propensity_scores,
        clever_covariates,
        targeted_predictions,
        fluctuation_coefs,
        overall_converged,
        n_treated_by_time,
        n_truncated_by_time,
    ))
}

/// Run LTMLE with default configuration (always treat vs never treat).
///
/// Convenience function using default settings: logistic outcome model,
/// propensity truncation at [0.01, 0.99], and estimating the ATE comparing
/// always-treat to never-treat regimes.
///
/// # Arguments
///
/// * `data` - Longitudinal data structure with outcomes, treatments, and covariates
///
/// # Example
///
/// ```ignore
/// let data = LtmleData::new(outcomes, treatments, covariates)?;
/// let result = ltmle(&data)?;
/// println!("ATE: {:.4} (SE: {:.4})", result.ate, result.se);
/// ```
pub fn ltmle(data: &LtmleData) -> EconResult<LtmleResult> {
    run_ltmle(data, LtmleConfig::default())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Add intercept column to design matrix.
fn add_intercept(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let mut x_with_intercept = Array2::zeros((n, k + 1));

    // First column is intercept (all 1s)
    for i in 0..n {
        x_with_intercept[[i, 0]] = 1.0;
        for j in 0..k {
            x_with_intercept[[i, j + 1]] = x[[i, j]];
        }
    }

    x_with_intercept
}

/// Logit (log-odds) function: logit(p) = log(p / (1-p))
#[inline]
fn logit(p: f64) -> f64 {
    let p_clipped = p.max(1e-10).min(1.0 - 1e-10);
    (p_clipped / (1.0 - p_clipped)).ln()
}

/// Fit logistic regression using Newton-Raphson (IRLS).
///
/// Returns (predictions, coefficients, converged, iterations).
fn fit_logistic_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Array1<f64>, Array1<f64>, bool, usize)> {
    let n = y.len();
    let k = x.ncols();

    // Initialize coefficients to zero
    let mut beta = Array1::zeros(k);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Linear predictor z = X*beta
        let z: Array1<f64> = x.dot(&beta);

        // Probabilities p = expit(z)
        let p: Array1<f64> = z.mapv(logistic_cdf);
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Gradient: g = X'(y - p)
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
            converged = true;
            break;
        }

        // Weights: w = p*(1-p)
        let weights: Array1<f64> = p_clipped.mapv(|pi| pi * (1.0 - pi));

        // Hessian: H = -X' diag(w) X
        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Newton-Raphson update: beta_new = beta - H^{-1} * g
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
    let p_final: Array1<f64> = z_final.mapv(logistic_cdf);

    Ok((p_final, beta, converged, iterations))
}

/// Fit linear regression using OLS.
///
/// Returns (predictions, coefficients, converged=true, iterations=1).
fn fit_linear_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> EconResult<(Array1<f64>, Array1<f64>, bool, usize)> {
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Linear regression X'X matrix".to_string(),
        suggestion: format!("Check for multicollinearity: {:?}", e),
    })?;

    let xty_vec = xty(&x.view(), y);
    let beta = xtx_inv.dot(&xty_vec);

    // Predictions
    let y_hat = x.dot(&beta);

    Ok((y_hat, beta, true, 1))
}

/// Fit the targeting model to estimate the fluctuation parameter epsilon.
///
/// For the targeting step, we fit:
/// - Logistic: logit(E[Y|H]) = logit(Q) + epsilon * H
/// - Linear: E[Y|H] = Q + epsilon * H
///
/// # References
/// - van der Laan & Rose (2011), Algorithm 4.1, Step 3
/// - Gruber & van der Laan (2012), Section 2.2
fn fit_targeting_model(
    y: &Array1<f64>,
    q_init: &Array1<f64>,
    h: &Array1<f64>,
    q_model: LtmleQModel,
) -> EconResult<(f64, bool)> {
    let n = y.len();

    match q_model {
        LtmleQModel::Logistic => {
            // Newton-Raphson for single parameter epsilon
            let mut epsilon = 0.0;
            let mut converged = false;
            let max_iter = 50;
            let tolerance = 1e-8;

            for _ in 0..max_iter {
                // Current predictions: p = expit(logit(Q) + epsilon * H)
                let p: Array1<f64> = (0..n)
                    .map(|i| {
                        let logit_q = logit(q_init[i]);
                        logistic_cdf(logit_q + epsilon * h[i])
                    })
                    .collect();
                let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

                // Score: dL/d(epsilon) = sum(H * (Y - p))
                let score: f64 = (0..n).map(|i| h[i] * (y[i] - p_clipped[i])).sum();

                // Check convergence
                if score.abs() < tolerance {
                    converged = true;
                    break;
                }

                // Information: -d^2L/d(epsilon)^2 = sum(H^2 * p * (1-p))
                let info: f64 = (0..n)
                    .map(|i| {
                        let pi = p_clipped[i];
                        h[i] * h[i] * pi * (1.0 - pi)
                    })
                    .sum();

                // Newton-Raphson update
                if info.abs() > 1e-10 {
                    epsilon += score / info;
                } else {
                    epsilon += 0.1 * score.signum();
                }
            }

            Ok((epsilon, converged))
        }
        LtmleQModel::Linear => {
            // OLS with offset Q and covariate H (no intercept)
            // epsilon = sum(H * (Y - Q)) / sum(H^2)
            let numerator: f64 = (0..n).map(|i| h[i] * (y[i] - q_init[i])).sum();
            let denominator: f64 = h.iter().map(|hi| hi * hi).sum::<f64>();

            if denominator.abs() < 1e-10 {
                return Err(EconError::SingularMatrix {
                    context: "Targeting step".to_string(),
                    suggestion: "Clever covariate has near-zero variance".to_string(),
                });
            }

            let epsilon = numerator / denominator;
            Ok((epsilon, true))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Create test data with 2 time points and known structure.
    ///
    /// DGP:
    /// - L_1 ~ Uniform(0, 1)
    /// - A_1 | L_1 ~ Bernoulli(expit(0.5 * L_1))
    /// - L_2 | A_1, L_1 ~ L_1 + 0.3 * A_1 + noise
    /// - A_2 | L_2 ~ Bernoulli(expit(0.5 * L_2))
    /// - Y_2 | A_1, A_2, L_1, L_2 = 0.2*L_1 + 0.3*L_2 + 0.4*A_1 + 0.5*A_2 + noise
    ///
    /// True ATE (always treat vs never treat) should be approximately 0.4 + 0.5 = 0.9
    fn create_ltmle_test_data() -> LtmleData {
        let n = 100;

        // Time 1 covariates (single covariate)
        let l_1_vec: Vec<f64> = (0..n).map(|i| (i as f64 % 10.0) / 10.0 + 0.05).collect();
        let l_1 = Array2::from_shape_vec((n, 1), l_1_vec).unwrap();

        // Time 1 treatments (based on L_1)
        let a_1: Array1<f64> = l_1
            .column(0)
            .iter()
            .enumerate()
            .map(|(i, l)| {
                // Deterministic assignment based on index for reproducibility
                if (*l > 0.5) ^ (i % 3 == 0) { 1.0 } else { 0.0 }
            })
            .collect();

        // Time 2 covariates (depend on A_1 and L_1)
        let l_2_vec: Vec<f64> = l_1
            .column(0)
            .iter()
            .zip(a_1.iter())
            .enumerate()
            .map(|(i, (l1, a1))| *l1 + 0.3 * *a1 + (i as f64 % 5.0) / 50.0 - 0.05)
            .collect();
        let l_2 = Array2::from_shape_vec((n, 1), l_2_vec).unwrap();

        // Time 2 treatments (based on L_2)
        let a_2: Array1<f64> = l_2
            .column(0)
            .iter()
            .enumerate()
            .map(|(i, l)| if (*l > 0.5) ^ (i % 4 == 0) { 1.0 } else { 0.0 })
            .collect();

        // Outcome at time 2
        let y_2: Array1<f64> = (0..n)
            .map(|i| {
                let l1 = l_1[[i, 0]];
                let l2 = l_2[[i, 0]];
                let a1 = a_1[i];
                let a2 = a_2[i];
                let noise = (i as f64 % 7.0) / 70.0 - 0.05;
                0.2 * l1 + 0.3 * l2 + 0.4 * a1 + 0.5 * a2 + noise
            })
            .collect();

        // Placeholder outcome at time 1 (not used in this setup)
        let y_1 = Array1::zeros(n);

        LtmleData {
            outcomes: vec![y_1, y_2],
            treatments: vec![a_1, a_2],
            covariates: vec![l_1, l_2],
        }
    }

    #[test]
    fn test_ltmle_data_validation() {
        let n = 10;

        // Valid data
        let outcomes = vec![Array1::zeros(n), Array1::zeros(n)];
        let treatments = vec![Array1::zeros(n), Array1::zeros(n)];
        let covariates = vec![Array2::zeros((n, 2)), Array2::zeros((n, 2))];

        let data = LtmleData::new(outcomes, treatments, covariates);
        assert!(data.is_ok());

        let data = data.unwrap();
        assert_eq!(data.time_points(), 2);
        assert_eq!(data.n_obs(), 10);
    }

    #[test]
    fn test_ltmle_data_validation_errors() {
        let n = 10;

        // Mismatched time points
        let outcomes = vec![Array1::zeros(n), Array1::zeros(n)];
        let treatments = vec![Array1::zeros(n)]; // Only 1 time point
        let covariates = vec![Array2::zeros((n, 2)), Array2::zeros((n, 2))];

        let result = LtmleData::new(outcomes, treatments, covariates);
        assert!(result.is_err());

        // Non-binary treatment
        let outcomes = vec![Array1::zeros(n)];
        let treatments = vec![Array1::from_vec(vec![0.5; n])]; // Not binary
        let covariates = vec![Array2::zeros((n, 2))];

        let result = LtmleData::new(outcomes, treatments, covariates);
        assert!(result.is_err());
    }

    #[test]
    fn test_ltmle_basic() {
        let data = create_ltmle_test_data();
        // Use Linear model since outcome is continuous
        let config = LtmleConfig {
            q_model: LtmleQModel::Linear,
            ..Default::default()
        };
        let result = run_ltmle(&data, config).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 100);
        assert_eq!(result.time_points, 2);

        // ATE should be positive (treatment effect is positive in DGP)
        // The true ATE is approximately 0.4 + 0.5 = 0.9 based on the DGP
        assert!(
            result.ate > 0.0,
            "ATE should be positive, got {}",
            result.ate
        );

        // Standard error should be positive and finite
        assert!(
            result.se > 0.0 && result.se.is_finite(),
            "SE should be positive and finite, got {}",
            result.se
        );

        // Confidence interval should be finite
        assert!(
            result.ci_lower.is_finite() && result.ci_upper.is_finite(),
            "CI should be finite: [{}, {}]",
            result.ci_lower,
            result.ci_upper
        );

        // Counterfactual means should be ordered: treated > control
        // (since treatment effect is positive)
        assert!(
            result.psi_treated > result.psi_control,
            "E[Y^1] = {} should be > E[Y^0] = {}",
            result.psi_treated,
            result.psi_control
        );
    }

    #[test]
    fn test_ltmle_with_linear_outcome() {
        let data = create_ltmle_test_data();
        let config = LtmleConfig {
            q_model: LtmleQModel::Linear,
            ..Default::default()
        };

        let result = run_ltmle(&data, config).unwrap();

        // Should still produce reasonable results
        assert!(
            result.ate.is_finite(),
            "ATE should be finite: {}",
            result.ate
        );
        assert!(
            result.se > 0.0 && result.se.is_finite(),
            "SE should be positive and finite: {}",
            result.se
        );
    }

    #[test]
    fn test_ltmle_propensity_truncation() {
        let data = create_ltmle_test_data();
        let config = LtmleConfig {
            gbounds: (0.1, 0.9),
            ..Default::default()
        };

        let result = run_ltmle(&data, config).unwrap();

        // All propensity scores should be within bounds
        for (t, ps) in result.propensity_scores.iter().enumerate() {
            for &p in ps.iter() {
                assert!(
                    (0.1..=0.9).contains(&p),
                    "PS at time {} outside bounds [0.1, 0.9]: {}",
                    t + 1,
                    p
                );
            }
        }
    }

    #[test]
    fn test_ltmle_influence_curve() {
        let data = create_ltmle_test_data();
        let result = ltmle(&data).unwrap();

        // Influence curve should have mean close to zero
        let ic_mean: f64 = result.influence_curve.iter().sum::<f64>() / result.n_obs as f64;
        assert!(
            ic_mean.abs() < 0.2,
            "IC mean should be close to zero, got {}",
            ic_mean
        );

        // IC should have correct length
        assert_eq!(result.influence_curve.len(), result.n_obs);
    }

    #[test]
    fn test_ltmle_fluctuation_coefs() {
        let data = create_ltmle_test_data();
        let result = ltmle(&data).unwrap();

        // Should have one fluctuation coefficient per time point
        assert_eq!(result.fluctuation_coefs.len(), result.time_points);

        // Fluctuation coefficients should be finite
        for (t, &eps) in result.fluctuation_coefs.iter().enumerate() {
            assert!(
                eps.is_finite(),
                "Fluctuation coef at time {} should be finite: {}",
                t + 1,
                eps
            );
        }
    }

    #[test]
    fn test_ltmle_display() {
        let data = create_ltmle_test_data();
        let result = ltmle(&data).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("Longitudinal Targeted Maximum Likelihood"));
        assert!(output.contains("ATE"));
        assert!(output.contains("Time Points"));
        assert!(output.contains("E[Y^{always treat}]"));
        assert!(output.contains("E[Y^{never treat}]"));
    }

    #[test]
    fn test_intervention_type_display() {
        assert_eq!(
            format!("{}", InterventionType::AlwaysTreat),
            "Always Treat (A=1)"
        );
        assert_eq!(
            format!("{}", InterventionType::NeverTreat),
            "Never Treat (A=0)"
        );
        assert_eq!(
            format!("{}", InterventionType::Static(true)),
            "Static (A=1)"
        );
        assert_eq!(
            format!("{}", InterventionType::Static(false)),
            "Static (A=0)"
        );
    }

    #[test]
    fn test_ltmle_single_time_point_error() {
        let n = 10;
        let outcomes = vec![Array1::zeros(n)];
        let treatments = vec![Array1::zeros(n)];
        let covariates = vec![Array2::zeros((n, 2))];

        let data = LtmleData::new(outcomes, treatments, covariates).unwrap();
        let result = ltmle(&data);

        // Should error because LTMLE requires >= 2 time points
        assert!(result.is_err());
        if let Err(EconError::InvalidSpecification { message }) = result {
            assert!(message.contains("2 time points"));
        }
    }

    #[test]
    fn test_add_intercept() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x_int = add_intercept(&x);

        assert_eq!(x_int.shape(), &[3, 3]);
        // First column should be all 1s
        assert_eq!(x_int.column(0).to_vec(), vec![1.0, 1.0, 1.0]);
        // Other columns should be original data
        assert_eq!(x_int.column(1).to_vec(), vec![1.0, 3.0, 5.0]);
        assert_eq!(x_int.column(2).to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_logit() {
        assert!((logit(0.5) - 0.0).abs() < 1e-10);
        assert!(logit(0.01).is_finite());
        assert!(logit(0.99).is_finite());
        assert!(logit(0.731) > 0.9 && logit(0.731) < 1.1);
    }

    // =========================================================================
    // Validation tests (test_validate_* prefix)
    // =========================================================================

    /// Create a larger longitudinal dataset with known DGP for validation.
    ///
    /// DGP with 2 time points, n=200:
    ///   L_1 ~ Uniform(0,1)
    ///   A_1 | L_1 ~ Bernoulli(based on L_1 threshold)
    ///   L_2 = L_1 + 0.3*A_1 + noise
    ///   A_2 | L_2 ~ Bernoulli(based on L_2 threshold)
    ///   Y = 0.2*L_1 + 0.3*L_2 + 0.4*A_1 + 0.5*A_2 + noise
    ///
    /// The treatment effects for always-treat vs never-treat:
    ///   E[Y^{1,1}] - E[Y^{0,0}] = 0.4 + 0.5 + indirect effects through L_2
    ///   Expected ATE ~ 0.9 + indirect effect of A_1 on L_2 (0.3*0.3 = 0.09)
    ///   So roughly ATE ~ 1.0
    fn create_validate_ltmle_data() -> LtmleData {
        let n = 200;

        // L_1: covariate at time 1
        let l_1_vec: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0 + 0.005)
            .collect();
        let l_1 = Array2::from_shape_vec((n, 1), l_1_vec).unwrap();

        // A_1: treatment at time 1 (depends on L_1)
        let a_1: Array1<f64> = l_1
            .column(0)
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let ps = 1.0 / (1.0 + (-0.5 * l + 0.25).exp());
                let threshold = ((i * 37 + 11) % 100) as f64 / 100.0;
                if threshold < ps { 1.0 } else { 0.0 }
            })
            .collect();

        // L_2: covariate at time 2 (depends on L_1 and A_1)
        let l_2_vec: Vec<f64> = (0..n)
            .map(|i| {
                let noise = ((i * 31 + 5) % 100) as f64 / 500.0 - 0.1;
                l_1[[i, 0]] + 0.3 * a_1[i] + noise
            })
            .collect();
        let l_2 = Array2::from_shape_vec((n, 1), l_2_vec).unwrap();

        // A_2: treatment at time 2 (depends on L_2)
        let a_2: Array1<f64> = l_2
            .column(0)
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let ps = 1.0 / (1.0 + (-0.5 * l + 0.25).exp());
                let threshold = ((i * 41 + 17) % 100) as f64 / 100.0;
                if threshold < ps { 1.0 } else { 0.0 }
            })
            .collect();

        // Y: final outcome
        let y_2: Array1<f64> = (0..n)
            .map(|i| {
                let noise = ((i * 23 + 9) % 100) as f64 / 500.0 - 0.1;
                0.2 * l_1[[i, 0]] + 0.3 * l_2[[i, 0]] + 0.4 * a_1[i] + 0.5 * a_2[i] + noise
            })
            .collect();

        let y_1 = Array1::zeros(n);

        LtmleData {
            outcomes: vec![y_1, y_2],
            treatments: vec![a_1, a_2],
            covariates: vec![l_1, l_2],
        }
    }

    /// Validate LTMLE estimates a positive ATE consistent with the DGP.
    #[test]
    fn test_validate_ltmle_ate_direction() {
        let data = create_validate_ltmle_data();
        let config = LtmleConfig {
            q_model: LtmleQModel::Linear,
            ..Default::default()
        };

        let result = run_ltmle(&data, config).unwrap();

        // ATE should be positive (treatment has positive effects in DGP)
        assert!(
            result.ate > 0.0,
            "ATE = {:.4} should be positive",
            result.ate
        );

        // ATE should be in a reasonable range (true ATE ~ 1.0)
        assert!(
            result.ate < 3.0,
            "ATE = {:.4} should not be too large (true ~ 1.0)",
            result.ate
        );

        // Counterfactual means should be ordered: treated > control
        assert!(
            result.psi_treated > result.psi_control,
            "E[Y^{{always treat}}] = {:.4} should be > E[Y^{{never treat}}] = {:.4}",
            result.psi_treated,
            result.psi_control
        );

        // ATE should equal the difference of counterfactual means
        let ate_from_psi = result.psi_treated - result.psi_control;
        assert!(
            (result.ate - ate_from_psi).abs() < 1e-8,
            "ATE = {:.4} should equal psi_treated - psi_control = {:.4}",
            result.ate,
            ate_from_psi
        );
    }

    /// Validate LTMLE standard errors and confidence intervals.
    #[test]
    fn test_validate_ltmle_inference() {
        let data = create_validate_ltmle_data();
        let config = LtmleConfig {
            q_model: LtmleQModel::Linear,
            ..Default::default()
        };

        let result = run_ltmle(&data, config).unwrap();

        // SE should be positive and finite
        assert!(
            result.se > 0.0 && result.se.is_finite(),
            "SE = {:.4} should be positive and finite",
            result.se
        );

        // CI should be properly ordered
        assert!(
            result.ci_lower < result.ci_upper,
            "CI lower ({:.4}) should be < CI upper ({:.4})",
            result.ci_lower,
            result.ci_upper
        );

        // CI should bracket ATE
        assert!(
            result.ci_lower < result.ate && result.ate < result.ci_upper,
            "CI [{:.4}, {:.4}] should bracket ATE = {:.4}",
            result.ci_lower,
            result.ci_upper,
            result.ate
        );

        // z-stat should be consistent with ATE/SE
        assert!(
            (result.z_stat - result.ate / result.se).abs() < 1e-8,
            "z_stat = {:.4} should equal ATE/SE = {:.4}",
            result.z_stat,
            result.ate / result.se
        );

        // p-value should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&result.p_value),
            "p-value = {} should be in [0, 1]",
            result.p_value
        );

        // Counterfactual mean SEs should be positive
        assert!(
            result.psi_treated_se > 0.0,
            "psi_treated_se should be positive"
        );
        assert!(
            result.psi_control_se > 0.0,
            "psi_control_se should be positive"
        );
    }

    /// Validate LTMLE targeting step produces finite fluctuation coefficients
    /// and properly structured output.
    #[test]
    fn test_validate_ltmle_targeting_convergence() {
        let data = create_validate_ltmle_data();
        let config = LtmleConfig {
            q_model: LtmleQModel::Linear,
            ..Default::default()
        };

        let result = run_ltmle(&data, config).unwrap();

        // Should have one fluctuation coefficient per time point
        assert_eq!(
            result.fluctuation_coefs.len(),
            2,
            "Should have 2 fluctuation coefficients for 2 time points"
        );

        // All fluctuation coefficients should be finite
        for (t, &eps) in result.fluctuation_coefs.iter().enumerate() {
            assert!(
                eps.is_finite(),
                "Fluctuation coef at time {} = {:.6} should be finite",
                t + 1,
                eps
            );
        }

        // Clever covariates should exist for each time point
        assert_eq!(result.clever_covariates.len(), 2);
        for (t, hc) in result.clever_covariates.iter().enumerate() {
            assert_eq!(
                hc.len(),
                200,
                "Clever covariates at time {} should have n=200 entries",
                t + 1
            );
            // All should be finite
            assert!(
                hc.iter().all(|h| h.is_finite()),
                "All clever covariates at time {} should be finite",
                t + 1
            );
        }

        // Targeted predictions should exist for each time point
        assert_eq!(result.targeted_predictions.len(), 2);

        // Model should have converged
        assert!(result.converged, "LTMLE should report convergence");
    }

    /// Validate LTMLE propensity score truncation.
    #[test]
    fn test_validate_ltmle_propensity_truncation() {
        let data = create_validate_ltmle_data();

        let gbounds = (0.05, 0.95);
        let config = LtmleConfig {
            q_model: LtmleQModel::Linear,
            gbounds,
            ..Default::default()
        };

        let result = run_ltmle(&data, config).unwrap();

        // All propensity scores should be within the specified bounds
        for (t, ps_vec) in result.propensity_scores.iter().enumerate() {
            for (i, &ps) in ps_vec.iter().enumerate() {
                assert!(
                    ps >= gbounds.0 - 1e-10 && ps <= gbounds.1 + 1e-10,
                    "PS at time {}, obs {} = {:.4} should be in [{}, {}]",
                    t + 1,
                    i,
                    ps,
                    gbounds.0,
                    gbounds.1
                );
            }
        }

        // Should report truncation counts per time point
        assert_eq!(result.n_truncated_by_time.len(), 2);

        // Should report treatment counts per time point
        assert_eq!(result.n_treated_by_time.len(), 2);
        for &nt in &result.n_treated_by_time {
            assert!(
                nt > 0,
                "Should have some treated observations at each time point"
            );
        }
    }

    /// Validate influence curve has mean approximately zero.
    #[test]
    fn test_validate_ltmle_influence_curve_properties() {
        let data = create_validate_ltmle_data();
        let config = LtmleConfig {
            q_model: LtmleQModel::Linear,
            ..Default::default()
        };

        let result = run_ltmle(&data, config).unwrap();

        // IC should have n entries
        assert_eq!(result.influence_curve.len(), 200);

        // All IC values should be finite
        assert!(
            result.influence_curve.iter().all(|ic| ic.is_finite()),
            "All IC values must be finite"
        );

        // Mean of IC should be close to zero (targeting condition)
        let ic_mean: f64 = result.influence_curve.iter().sum::<f64>() / 200.0;
        assert!(
            ic_mean.abs() < 0.5,
            "IC mean = {:.6} should be close to zero",
            ic_mean
        );

        // IC variance should be positive (used for SE)
        let ic_var: f64 = result
            .influence_curve
            .iter()
            .map(|ic| (ic - ic_mean).powi(2))
            .sum::<f64>()
            / 199.0;
        assert!(
            ic_var > 0.0,
            "IC variance should be positive, got {:.6}",
            ic_var
        );
    }
}
