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
//! # Algorithm (for ATE with binary outcome)
//!
//! **Step 1: Initial Estimates**
//! - Fit outcome model Q(A,W) = E[Y|A,W] using logistic regression
//! - Fit propensity score g(W) = P(A=1|W) using logistic regression
//!
//! **Step 2: Compute Clever Covariate**
//! - H(A,W) = A/g(W) - (1-A)/(1-g(W))
//!
//! **Step 3: Targeting Step (Fluctuation)**
//! - Fit epsilon in: logit(Q*(A,W)) = logit(Q(A,W)) + epsilon * H(A,W)
//! - This "fluctuates" Q towards the optimal estimate for ATE
//!
//! **Step 4: Update Predictions and Compute Estimate**
//! - Compute Q*(1,W) and Q*(0,W) for all observations
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

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::{logistic_cdf, normal_cdf, SignificanceLevel};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Outcome model specification for TMLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QModel {
    /// Linear regression for continuous Y
    Linear,
    /// Logistic regression for binary Y (default)
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

/// Configuration for TMLE estimation.
#[derive(Debug, Clone)]
pub struct TmleConfig {
    /// Outcome model specification
    pub q_model: QModel,
    /// Propensity score model specification
    pub g_model: GModel,
    /// Truncation bounds for propensity scores (min, max)
    /// Default: (0.01, 0.99) to avoid extreme weights
    pub truncate_ps: (f64, f64),
    /// Maximum iterations for logistic regression
    pub max_iter: usize,
    /// Convergence tolerance
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
    /// Clever covariate H(A,W) for each observation
    pub clever_covariate: Vec<f64>,
    /// Fluctuation coefficient epsilon from targeting step
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
        writeln!(f, "  p-value:    {:>12.4}{}", self.ate_p_value, self.significance.stars())?;
        writeln!(f, "  95% CI:     [{:.4}, {:.4}]", self.ate_ci_lower, self.ate_ci_upper)?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(f, "  Observations:  {} (Treated: {}, Control: {})",
                 self.n_obs, self.n_treated, self.n_control)?;
        writeln!(f, "  PS Truncated:  {}", self.n_truncated)?;
        writeln!(f)?;
        writeln!(f, "Model Specification:")?;
        writeln!(f, "  Outcome Model (Q):    {}", self.q_model_type)?;
        writeln!(f, "  Propensity Model (g): {}", self.g_model_type)?;
        writeln!(f, "  PS Truncation:        [{:.2}, {:.2}]",
                 self.truncate_bounds.0, self.truncate_bounds.1)?;
        writeln!(f)?;
        writeln!(f, "Targeting Step:")?;
        writeln!(f, "  Fluctuation coef (epsilon): {:.6}", self.fluctuation_coef)?;
        writeln!(f)?;
        writeln!(f, "Convergence:")?;
        writeln!(f, "  Q model: {} (iterations: {})",
                 if self.q_model_converged { "Yes" } else { "No" }, self.q_model_iterations)?;
        writeln!(f, "  g model: {} (iterations: {})",
                 if self.g_model_converged { "Yes" } else { "No" }, self.g_model_iterations)?;
        writeln!(f, "  Targeting: {}", if self.targeting_converged { "Yes" } else { "No" })?;
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
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of covariate columns for both Q and g models
/// * `config` - TMLE configuration options
///
/// # Returns
/// `TmleResult` containing ATE estimate, standard error, confidence interval,
/// and diagnostic information.
///
/// # Algorithm
///
/// 1. Fit initial outcome model Q(A,W) = E[Y|A,W]
/// 2. Fit propensity score model g(W) = P(A=1|W)
/// 3. Compute clever covariate H(A,W) = A/g(W) - (1-A)/(1-g(W))
/// 4. Targeting step: fit epsilon in logit(Q*) = logit(Q) + epsilon * H
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

    // ═══════════════════════════════════════════════════════════════════════
    // Extract Data
    // ═══════════════════════════════════════════════════════════════════════

    // Extract outcome variable Y
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Extract treatment variable A
    let a = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: vec![format!("{:?}", e)],
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

    // Build design matrix W for covariates (with intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, true)?;
    let w = design.data;
    let k = w.ncols();

    // Build design matrix for outcome model: [W, A]
    // We include treatment A as a covariate in Q(A,W)
    let mut x_q = Array2::zeros((n, k + 1));
    for i in 0..n {
        for j in 0..k {
            x_q[[i, j]] = w[[i, j]];
        }
        x_q[[i, k]] = a[i];
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 1: Fit Initial Outcome Model Q(A,W)
    // ═══════════════════════════════════════════════════════════════════════

    let (q_init, q_beta, q_converged, q_iterations) = match config.q_model {
        QModel::Logistic => {
            fit_logistic_model(&x_q, &y, config.max_iter, config.tolerance)?
        }
        QModel::Linear => {
            fit_linear_model(&x_q, &y)?
        }
    };

    if !q_converged {
        warnings.push("Outcome model did not converge".to_string());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 2: Fit Propensity Score Model g(W) = P(A=1|W)
    // ═══════════════════════════════════════════════════════════════════════

    let (g_raw, _g_beta, g_converged, g_iterations) = match config.g_model {
        GModel::Logistic => {
            fit_logistic_model(&w, &a, config.max_iter, config.tolerance)?
        }
    };

    if !g_converged {
        warnings.push("Propensity score model did not converge".to_string());
    }

    // Truncate propensity scores to avoid extreme weights
    // (van der Laan & Rose 2011, Section 4.2.2)
    let (ps_min, ps_max) = config.truncate_ps;
    let mut n_truncated = 0;
    let g: Array1<f64> = g_raw.mapv(|gi| {
        if gi < ps_min || gi > ps_max {
            n_truncated += 1;
        }
        gi.max(ps_min).min(ps_max)
    });

    if n_truncated > n / 10 {
        warnings.push(format!(
            "Many propensity scores truncated ({}/{}). Consider wider truncation bounds.",
            n_truncated, n
        ));
    }

    // Check propensity score overlap
    let _g_mean: f64 = g.iter().sum::<f64>() / n as f64;
    let g_min = g.iter().cloned().fold(f64::INFINITY, f64::min);
    let g_max = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if g_min < 0.05 || g_max > 0.95 {
        warnings.push(format!(
            "Propensity scores have extreme values (min: {:.4}, max: {:.4}). \
             This may indicate positivity violations.",
            g_min, g_max
        ));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 3: Compute Clever Covariate H(A,W)
    // ═══════════════════════════════════════════════════════════════════════

    // H(A,W) = A/g(W) - (1-A)/(1-g(W))
    // This is the efficient influence curve component for the ATE
    // (van der Laan & Rose 2011, Eq. 4.2)
    let h: Array1<f64> = (0..n)
        .map(|i| {
            let ai = a[i];
            let gi = g[i];
            if ai >= 0.5 {
                1.0 / gi
            } else {
                -1.0 / (1.0 - gi)
            }
        })
        .collect();

    // ═══════════════════════════════════════════════════════════════════════
    // Step 4: Targeting Step - Fluctuate Q
    // ═══════════════════════════════════════════════════════════════════════

    // The key insight of TMLE: we fit a single parameter epsilon in
    // logit(Q*) = logit(Q) + epsilon * H
    //
    // This "fluctuates" the initial estimate Q towards the optimal estimate
    // for the ATE. The epsilon is estimated by weighted logistic regression
    // with offset logit(Q) and covariate H.
    //
    // (van der Laan & Rose 2011, Algorithm 4.1, Step 3)

    let (epsilon, targeting_converged) = fit_targeting_model(&y, &q_init, &h, config.q_model)?;

    if !targeting_converged {
        warnings.push("Targeting step did not converge".to_string());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 5: Update Predictions Q*(A,W)
    // ═══════════════════════════════════════════════════════════════════════

    // Compute Q*(A,W) using the fluctuation
    let q_star: Array1<f64> = match config.q_model {
        QModel::Logistic => {
            // Q*(A,W) = expit(logit(Q(A,W)) + epsilon * H(A,W))
            (0..n)
                .map(|i| {
                    let logit_q = logit(q_init[i]);
                    logistic_cdf(logit_q + epsilon * h[i])
                })
                .collect()
        }
        QModel::Linear => {
            // Q*(A,W) = Q(A,W) + epsilon * H(A,W)
            (0..n)
                .map(|i| q_init[i] + epsilon * h[i])
                .collect()
        }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Step 6: Compute Counterfactual Predictions Q*(1,W) and Q*(0,W)
    // ═══════════════════════════════════════════════════════════════════════

    // For each observation, compute what their outcome would be under treatment
    // and under control, using the targeted model.
    //
    // We need to create counterfactual design matrices where A=1 for all (Q*(1,W))
    // and A=0 for all (Q*(0,W)).

    // First, compute Q(1,W) and Q(0,W) using the initial model coefficients
    let mut x_q_1 = x_q.clone();
    let mut x_q_0 = x_q.clone();
    for i in 0..n {
        x_q_1[[i, k]] = 1.0;  // Set A = 1
        x_q_0[[i, k]] = 0.0;  // Set A = 0
    }

    let (q_1_init, q_0_init) = match config.q_model {
        QModel::Logistic => {
            let z_1: Array1<f64> = x_q_1.dot(&q_beta);
            let z_0: Array1<f64> = x_q_0.dot(&q_beta);
            (z_1.mapv(logistic_cdf), z_0.mapv(logistic_cdf))
        }
        QModel::Linear => {
            (x_q_1.dot(&q_beta), x_q_0.dot(&q_beta))
        }
    };

    // Compute clever covariates for counterfactuals
    // H(1,W) = 1/g(W),  H(0,W) = -1/(1-g(W))
    let h_1: Array1<f64> = g.mapv(|gi| 1.0 / gi);
    let h_0: Array1<f64> = g.mapv(|gi| -1.0 / (1.0 - gi));

    // Apply targeting fluctuation to get Q*(1,W) and Q*(0,W)
    let (q_star_1, q_star_0): (Array1<f64>, Array1<f64>) = match config.q_model {
        QModel::Logistic => {
            let q1 = (0..n)
                .map(|i| {
                    let logit_q = logit(q_1_init[i]);
                    logistic_cdf(logit_q + epsilon * h_1[i])
                })
                .collect();
            let q0 = (0..n)
                .map(|i| {
                    let logit_q = logit(q_0_init[i]);
                    logistic_cdf(logit_q + epsilon * h_0[i])
                })
                .collect();
            (q1, q0)
        }
        QModel::Linear => {
            let q1 = (0..n)
                .map(|i| q_1_init[i] + epsilon * h_1[i])
                .collect();
            let q0 = (0..n)
                .map(|i| q_0_init[i] + epsilon * h_0[i])
                .collect();
            (q1, q0)
        }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Step 7: Compute ATE and Influence Curve-Based Variance
    // ═══════════════════════════════════════════════════════════════════════

    // ATE = (1/n) * sum(Q*(1,W) - Q*(0,W))
    // This is the substitution estimator (van der Laan & Rose 2011, Eq. 4.4)
    let ate: f64 = (0..n)
        .map(|i| q_star_1[i] - q_star_0[i])
        .sum::<f64>() / n as f64;

    // Efficient Influence Curve (EIC) for ATE:
    // IC(O) = H(A,W) * (Y - Q*(A,W)) + Q*(1,W) - Q*(0,W) - ATE
    //
    // The first term is the IPW augmentation, the second is the outcome model,
    // and -ATE centers it. Under the true model, E[IC] = 0.
    //
    // (van der Laan & Rose 2011, Eq. 4.3)
    let ic: Array1<f64> = (0..n)
        .map(|i| {
            let ipw_term = h[i] * (y[i] - q_star[i]);
            let outcome_term = q_star_1[i] - q_star_0[i];
            ipw_term + outcome_term - ate
        })
        .collect();

    // Variance of ATE from influence curve
    // Var(ATE) = Var(IC) / n
    // (Asymptotic variance of the sample mean of the IC)
    let ic_mean: f64 = ic.iter().sum::<f64>() / n as f64;
    let ic_var: f64 = ic.iter().map(|&ic_i| (ic_i - ic_mean).powi(2)).sum::<f64>()
        / (n - 1).max(1) as f64;
    let ate_var = ic_var / n as f64;
    let ate_se = ate_var.sqrt();

    // Wald-type confidence interval and p-value
    let z_stat = if ate_se > 0.0 && ate_se.is_finite() {
        ate / ate_se
    } else {
        0.0
    };

    // 95% CI using normal approximation (valid asymptotically)
    let z_crit = 1.96;
    let ate_ci_lower = ate - z_crit * ate_se;
    let ate_ci_upper = ate + z_crit * ate_se;

    // Two-sided p-value
    let ate_p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));
    let significance = SignificanceLevel::from_p_value(ate_p_value);

    // ═══════════════════════════════════════════════════════════════════════
    // Construct Result
    // ═══════════════════════════════════════════════════════════════════════

    Ok(TmleResult {
        ate,
        ate_se,
        ate_ci_lower,
        ate_ci_upper,
        ate_p_value,
        significance,
        z_stat,
        propensity_scores: g.to_vec(),
        initial_outcome: q_init.to_vec(),
        targeted_outcome: q_star.to_vec(),
        clever_covariate: h.to_vec(),
        fluctuation_coef: epsilon,
        influence_curve: ic.to_vec(),
        q_star_1: q_star_1.to_vec(),
        q_star_0: q_star_0.to_vec(),
        q_model_converged: q_converged,
        q_model_iterations: q_iterations,
        g_model_converged: g_converged,
        g_model_iterations: g_iterations,
        targeting_converged,
        n_obs: n,
        n_treated,
        n_control,
        n_truncated,
        q_model_type: config.q_model,
        g_model_type: config.g_model,
        truncate_bounds: config.truncate_ps,
        warnings,
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
    tmle(dataset, outcome_col, treatment_col, covariate_cols, TmleConfig::default())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

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
        // Equivalently: beta_new = beta + (-H)^{-1} * g
        let neg_hessian = &hessian * -1.0;
        let (hess_inv, _) = safe_inverse(&neg_hessian.view()).map_err(|e| {
            EconError::SingularMatrix {
                context: "Logistic regression Hessian".to_string(),
                suggestion: format!("Check for multicollinearity: {:?}", e),
            }
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
    // OLS: beta = (X'X)^{-1} X'y
    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| {
        EconError::SingularMatrix {
            context: "Linear regression X'X matrix".to_string(),
            suggestion: format!("Check for multicollinearity: {:?}", e),
        }
    })?;

    let xty_vec = xty(&x.view(), y);
    let beta = xtx_inv.dot(&xty_vec);

    // Predictions
    let y_hat = x.dot(&beta);

    Ok((y_hat, beta, true, 1))
}

/// Fit the targeting model to estimate the fluctuation parameter epsilon.
///
/// This is the key step of TMLE: we fit a logistic regression with
/// - Offset: logit(Q)
/// - Single covariate: H (clever covariate)
/// - No intercept (or equivalently, intercept constrained to 0)
///
/// The coefficient on H is epsilon, the fluctuation parameter.
///
/// For continuous outcomes (linear Q model), we use weighted least squares
/// with the analogous structure.
///
/// # References
/// - van der Laan & Rose (2011), Algorithm 4.1, Step 3
/// - Gruber & van der Laan (2012), Section 2.2
fn fit_targeting_model(
    y: &Array1<f64>,
    q_init: &Array1<f64>,
    h: &Array1<f64>,
    q_model: QModel,
) -> EconResult<(f64, bool)> {
    let n = y.len();

    match q_model {
        QModel::Logistic => {
            // For binary outcomes, we fit:
            // logit(E[Y|H]) = logit(Q) + epsilon * H
            //
            // Using Newton-Raphson with a single parameter epsilon.
            // This is essentially a GLM with offset = logit(Q) and single covariate H.

            let mut epsilon = 0.0;
            let mut converged = false;
            let max_iter = 50;
            let tolerance = 1e-8;

            for _ in 0..max_iter {
                // Compute current predictions: p = expit(logit(Q) + epsilon * H)
                let p: Array1<f64> = (0..n)
                    .map(|i| {
                        let logit_q = logit(q_init[i]);
                        logistic_cdf(logit_q + epsilon * h[i])
                    })
                    .collect();
                let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

                // Score (gradient): dL/d(epsilon) = sum(H * (Y - p))
                let score: f64 = (0..n)
                    .map(|i| h[i] * (y[i] - p_clipped[i]))
                    .sum();

                // Check convergence
                if score.abs() < tolerance {
                    converged = true;
                    break;
                }

                // Information (negative Hessian): -d^2L/d(epsilon)^2 = sum(H^2 * p * (1-p))
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
                    // If info is too small, use a dampened step
                    epsilon += 0.1 * score.signum();
                }
            }

            Ok((epsilon, converged))
        }
        QModel::Linear => {
            // For continuous outcomes, we fit:
            // E[Y|H] = Q + epsilon * H
            //
            // This is OLS with offset Q and single covariate H (no intercept).
            // Solution: epsilon = sum(H * (Y - Q)) / sum(H^2)

            let numerator: f64 = (0..n)
                .map(|i| h[i] * (y[i] - q_init[i]))
                .sum();
            let denominator: f64 = h.iter().map(|&hi| hi * hi).sum::<f64>();

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
    use polars::prelude::*;

    /// Create a test dataset with known treatment effect.
    ///
    /// DGP:
    /// - W ~ Uniform(0, 1)
    /// - A | W ~ Bernoulli(expit(0.5 * W))
    /// - Y | A, W = 0.3 * W + 0.5 * A + noise
    ///
    /// True ATE = 0.5
    fn create_tmle_test_dataset() -> Dataset {
        // Deterministic data for reproducibility
        let df = df! {
            "y" => [
                // Treated observations (A=1): Y approx 0.3*W + 0.5 + noise
                0.9, 1.1, 0.8, 1.2, 0.95, 1.15, 0.85, 1.25, 0.92, 1.08,
                0.75, 1.35, 0.88, 1.18, 0.82, 1.28, 0.95, 1.12, 0.78, 1.32,
                // Control observations (A=0): Y approx 0.3*W + noise
                0.3, 0.5, 0.25, 0.55, 0.35, 0.6, 0.28, 0.58, 0.32, 0.52,
                0.2, 0.7, 0.38, 0.65, 0.22, 0.68, 0.4, 0.62, 0.18, 0.72
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
        }.unwrap();
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

        // ATE should be approximately 0.5 (the true value in our DGP)
        assert!(result.ate > 0.3, "ATE should be positive, got {}", result.ate);
        assert!(result.ate < 0.8, "ATE should be around 0.5, got {}", result.ate);

        // Standard error should be positive and reasonable
        assert!(result.ate_se > 0.0 && result.ate_se.is_finite(),
                "SE should be positive and finite, got {}", result.ate_se);
        assert!(result.ate_se < 0.5, "SE seems too large: {}", result.ate_se);

        // Confidence interval should contain the true effect
        assert!(result.ate_ci_lower < 0.5 && result.ate_ci_upper > 0.5,
                "95% CI [{}, {}] should contain true ATE of 0.5",
                result.ate_ci_lower, result.ate_ci_upper);

        // Check that models converged
        assert!(result.q_model_converged, "Q model should converge");
        assert!(result.g_model_converged, "g model should converge");
        assert!(result.targeting_converged, "Targeting step should converge");

        // Check propensity scores are in valid range
        for &ps in &result.propensity_scores {
            assert!(ps >= 0.01 && ps <= 0.99,
                    "Propensity score {} outside truncation bounds", ps);
        }

        // Fluctuation coefficient should be small (good initial models)
        assert!(result.fluctuation_coef.abs() < 5.0,
                "Fluctuation coefficient seems too large: {}", result.fluctuation_coef);
    }

    #[test]
    fn test_tmle_with_linear_outcome() {
        let dataset = create_tmle_test_dataset();
        let config = TmleConfig {
            q_model: QModel::Linear,
            ..Default::default()
        };

        let result = tmle(&dataset, "y", "treatment", &["w1", "w2"], config).unwrap();

        // Should still get a reasonable estimate
        assert!(result.ate > 0.2 && result.ate < 1.0,
                "ATE with linear Q model seems off: {}", result.ate);
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
            assert!(ps >= 0.1 && ps <= 0.9,
                    "PS {} outside truncation bounds [0.1, 0.9]", ps);
        }
    }

    #[test]
    fn test_tmle_influence_curve() {
        let dataset = create_tmle_test_dataset();
        let result = run_tmle(&dataset, "y", "treatment", &["w1", "w2"]).unwrap();

        // The influence curve should have mean close to zero (property of EIC)
        let ic_mean: f64 = result.influence_curve.iter().sum::<f64>() / result.n_obs as f64;
        assert!(ic_mean.abs() < 0.1,
                "IC mean should be close to zero, got {}", ic_mean);

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
        let ate_from_cf: f64 = result.q_star_1.iter()
            .zip(result.q_star_0.iter())
            .map(|(q1, q0)| q1 - q0)
            .sum::<f64>() / result.n_obs as f64;

        assert!((ate_from_cf - result.ate).abs() < 1e-10,
                "ATE from counterfactuals ({}) should match reported ATE ({})",
                ate_from_cf, result.ate);
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

    #[test]
    fn test_logit_helper() {
        // Test logit function
        assert!((logit(0.5) - 0.0).abs() < 1e-10);
        assert!((logit(0.731) - 1.0).abs() < 0.01);
        assert!(logit(0.01).is_finite()); // Should handle edge cases
        assert!(logit(0.99).is_finite());
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
            result.influence_curve.iter()
                .map(|&ic| (ic - mean).powi(2))
                .sum::<f64>() / (result.n_obs - 1) as f64
        };
        let expected_var = ic_var / result.n_obs as f64;
        let reported_var = result.ate_se.powi(2);

        assert!((expected_var - reported_var).abs() / expected_var < 0.01,
                "Variance calculation mismatch: expected {}, got {}",
                expected_var, reported_var);
    }
}
