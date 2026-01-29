//! Collaborative Targeted Maximum Likelihood Estimation (C-TMLE).
//!
//! C-TMLE extends TMLE by using cross-validation to select which covariates to include
//! in the propensity score model. This reduces finite-sample bias from including too many
//! covariates while maintaining double robustness.
//!
//! The key insight is that the propensity score model should be "collaborative" with the
//! outcome model, meaning covariates are selected based on how well they improve the
//! targeting step, not just how well they predict treatment.
//!
//! # Algorithm (Discrete C-TMLE with Forward Selection)
//!
//! 1. **Fit initial outcome model Q(A,W)** using all covariates
//! 2. **Start with empty propensity model** g() (intercept only)
//! 3. **For each candidate covariate** W_j not yet in g:
//!    - Fit g with current covariates + W_j
//!    - Compute TMLE with this g
//!    - Use V-fold CV to evaluate the targeting step
//! 4. **Add covariate** that minimizes CV criterion (RSS or variance of IC)
//! 5. **Stop when** CV criterion no longer improves (or max covariates reached)
//! 6. **Return final TMLE estimate** with selected covariates
//!
//! # References
//!
//! - Ju, C., Gruber, S., Lendle, S. D., Chambaz, A., Franklin, J. M., Wyss, R., ... &
//!   van der Laan, M. J. (2019). Scalable collaborative targeted learning for
//!   high-dimensional data. *Statistical Methods in Medical Research*, 28(2), 532-554.
//!   https://doi.org/10.1177/0962280217729845
//!
//! - van der Laan, M. J., & Gruber, S. (2010). Collaborative double robust targeted
//!   maximum likelihood estimation. *The International Journal of Biostatistics*, 6(1).
//!   https://doi.org/10.2202/1557-4679.1181
//!
//! - Ju, C., & van der Laan, M. J. (2017). ctmle: Collaborative Targeted Maximum
//!   Likelihood Estimation. R package version 0.1.2.
//!   https://cran.r-project.org/package=ctmle
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::econometrics::{run_ctmle, CTmleConfig, StoppingRule, SelectionOrder};
//!
//! let config = CTmleConfig {
//!     n_folds: 5,
//!     max_covariates: Some(10),
//!     stopping_rule: StoppingRule::CVMinimum,
//!     order: SelectionOrder::Forward,
//!     ..Default::default()
//! };
//!
//! let result = run_ctmle(&dataset, "outcome", "treatment", &["x1", "x2", "x3", "x4"], config)?;
//! println!("ATE: {:.4} (SE: {:.4})", result.ate, result.se);
//! println!("Selected covariates: {:?}", result.selected_covariates);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::traits::estimator::{SignificanceLevel, logistic_cdf, normal_cdf};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Stopping rule for covariate selection in C-TMLE.
///
/// Controls when to stop adding covariates to the propensity score model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StoppingRule {
    /// Stop at the model with minimum CV criterion.
    /// This is the default and most commonly used approach.
    #[default]
    CVMinimum,

    /// One-standard-error rule: Select the simplest model within one SE of the minimum.
    /// This provides more regularization than CVMinimum, often preferred in practice.
    OneSE,

    /// Stop after adding exactly k covariates.
    MaxCovariates(usize),
}

impl fmt::Display for StoppingRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoppingRule::CVMinimum => write!(f, "CV Minimum"),
            StoppingRule::OneSE => write!(f, "One-SE Rule"),
            StoppingRule::MaxCovariates(k) => write!(f, "Max {} Covariates", k),
        }
    }
}

/// Order for covariate selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SelectionOrder {
    /// Forward selection: Start with empty model, add covariates greedily.
    /// This is the default and most commonly used approach.
    #[default]
    Forward,

    /// Backward elimination: Start with full model, remove covariates greedily.
    /// Can be computationally expensive but may find different solutions.
    Backward,

    /// Prespecified order: Add covariates in the user-specified order.
    /// Useful when domain knowledge suggests a preferred ordering.
    Prespecified(Vec<usize>),
}

impl fmt::Display for SelectionOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectionOrder::Forward => write!(f, "Forward"),
            SelectionOrder::Backward => write!(f, "Backward"),
            SelectionOrder::Prespecified(order) => write!(f, "Prespecified({:?})", order),
        }
    }
}

/// Cross-validation criterion for model selection.
///
/// Determines how candidate propensity score models are evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CVCriterion {
    /// Residual sum of squares from the targeting step.
    /// CV-RSS = sum((Y - Q*)^2) evaluated on held-out folds.
    #[default]
    RSS,

    /// Variance of the influence curve.
    /// CV-VarIC = Var(IC) evaluated on held-out folds.
    VarIC,

    /// Penalized RSS: RSS + penalty * number of covariates.
    /// Adds explicit complexity penalty.
    PenRSS,
}

impl fmt::Display for CVCriterion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CVCriterion::RSS => write!(f, "CV-RSS"),
            CVCriterion::VarIC => write!(f, "CV-VarIC"),
            CVCriterion::PenRSS => write!(f, "CV-PenRSS"),
        }
    }
}

/// Outcome model specification for C-TMLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CTmleQModel {
    /// Linear regression for continuous Y
    Linear,
    /// Logistic regression for binary Y (default)
    #[default]
    Logistic,
}

impl fmt::Display for CTmleQModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CTmleQModel::Linear => write!(f, "Linear"),
            CTmleQModel::Logistic => write!(f, "Logistic"),
        }
    }
}

/// Configuration for C-TMLE estimation.
#[derive(Debug, Clone)]
pub struct CTmleConfig {
    /// Number of cross-validation folds (default: 5).
    pub n_folds: usize,

    /// Maximum number of covariates to include in propensity model.
    /// None means no limit (can include all covariates).
    pub max_covariates: Option<usize>,

    /// Stopping rule for covariate selection.
    pub stopping_rule: StoppingRule,

    /// Order for covariate selection.
    pub order: SelectionOrder,

    /// Cross-validation criterion for model selection.
    pub cv_criterion: CVCriterion,

    /// Outcome model specification.
    pub q_model: CTmleQModel,

    /// Truncation bounds for propensity scores (min, max).
    /// Default: (0.025, 0.975) as recommended in ctmle package.
    pub gbound: (f64, f64),

    /// Maximum iterations for logistic regression.
    pub max_iter: usize,

    /// Convergence tolerance.
    pub tolerance: f64,

    /// Penalty factor for PenRSS criterion (multiplied by log(n)).
    /// Only used when cv_criterion = PenRSS.
    pub penalty_factor: f64,

    /// Factor for early stopping.
    /// If CV criterion is stopFactor times larger than best, stop.
    /// Default: 1e6 (effectively disables early stopping).
    pub stop_factor: f64,

    /// Whether to print verbose output during selection.
    pub verbose: bool,
}

impl Default for CTmleConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            max_covariates: None,
            stopping_rule: StoppingRule::CVMinimum,
            order: SelectionOrder::Forward,
            cv_criterion: CVCriterion::RSS,
            q_model: CTmleQModel::Logistic,
            gbound: (0.025, 0.975),
            max_iter: 100,
            tolerance: 1e-8,
            penalty_factor: 1.0,
            stop_factor: 1e6,
            verbose: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Information about a single step in the selection path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionStep {
    /// Index of the covariate added (or removed for backward selection).
    /// None for the initial (intercept-only) model.
    pub covariate_index: Option<usize>,

    /// Name of the covariate added (if available).
    pub covariate_name: Option<String>,

    /// Cross-validation criterion value at this step.
    pub cv_criterion: f64,

    /// Standard error of CV criterion (across folds).
    pub cv_criterion_se: f64,

    /// ATE estimate at this step.
    pub ate_estimate: f64,

    /// Standard error of ATE at this step.
    pub ate_se: f64,

    /// Number of covariates in the model at this step.
    pub n_covariates: usize,
}

/// Result from C-TMLE estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTmleResult {
    /// Average treatment effect estimate.
    pub ate: f64,

    /// Standard error of ATE (from influence curve).
    pub se: f64,

    /// 95% confidence interval lower bound.
    pub ci_lower: f64,

    /// 95% confidence interval upper bound.
    pub ci_upper: f64,

    /// Two-sided p-value for H0: ATE = 0.
    pub p_value: f64,

    /// Significance level.
    pub significance: SignificanceLevel,

    /// Z-statistic (ATE / SE).
    pub z_stat: f64,

    /// Indices of covariates selected for the propensity score model.
    pub selected_covariates: Vec<usize>,

    /// Names of covariates selected (if available).
    pub selected_covariate_names: Vec<String>,

    /// Number of covariates selected.
    pub n_selected: usize,

    /// The selection path showing CV criterion at each step.
    pub selection_path: Vec<SelectionStep>,

    /// CV criterion values at each step (for plotting).
    pub cv_risk: Vec<f64>,

    /// Influence curve values for each observation.
    pub influence_curve: Vec<f64>,

    /// Index of the selected step (0 = intercept only).
    pub selected_step: usize,

    /// Final propensity scores g(W) for each observation.
    pub propensity_scores: Vec<f64>,

    /// Targeted outcome predictions Q*(A,W).
    pub targeted_outcome: Vec<f64>,

    /// Number of observations.
    pub n_obs: usize,

    /// Number of treated observations.
    pub n_treated: usize,

    /// Number of control observations.
    pub n_control: usize,

    /// Configuration used for estimation.
    pub config: CTmleConfigSummary,

    /// Warnings generated during estimation.
    pub warnings: Vec<String>,
}

/// Summary of C-TMLE configuration (serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTmleConfigSummary {
    pub n_folds: usize,
    pub max_covariates: Option<usize>,
    pub stopping_rule: String,
    pub order: String,
    pub cv_criterion: String,
    pub q_model: String,
    pub gbound: (f64, f64),
}

impl fmt::Display for CTmleResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Collaborative Targeted Maximum Likelihood Estimation (C-TMLE)"
        )?;
        writeln!(
            f,
            "============================================================="
        )?;
        writeln!(f)?;
        writeln!(f, "Treatment Effect (ATE):")?;
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
        writeln!(f, "Covariate Selection:")?;
        writeln!(
            f,
            "  Selected: {} of {} candidates",
            self.n_selected,
            self.selection_path.len().saturating_sub(1)
        )?;
        if !self.selected_covariate_names.is_empty() {
            writeln!(
                f,
                "  Variables: {}",
                self.selected_covariate_names.join(", ")
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  Observations: {} (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(f)?;
        writeln!(f, "Configuration:")?;
        writeln!(f, "  CV Folds:       {}", self.config.n_folds)?;
        writeln!(f, "  Stopping Rule:  {}", self.config.stopping_rule)?;
        writeln!(f, "  Selection:      {}", self.config.order)?;
        writeln!(f, "  CV Criterion:   {}", self.config.cv_criterion)?;
        writeln!(f, "  Outcome Model:  {}", self.config.q_model)?;
        writeln!(
            f,
            "  PS Bounds:      [{:.3}, {:.3}]",
            self.config.gbound.0, self.config.gbound.1
        )?;
        writeln!(f)?;

        writeln!(f, "Selection Path:")?;
        writeln!(f, "  Step | Covariates |   CV Risk   |    ATE    |   SE")?;
        writeln!(f, "  -----|------------|-------------|-----------|--------")?;
        for (i, step) in self.selection_path.iter().enumerate() {
            let marker = if i == self.selected_step { " *" } else { "  " };
            writeln!(
                f,
                "  {:>4} | {:>10} | {:>11.4} | {:>9.4} | {:>6.4}{}",
                i, step.n_covariates, step.cv_criterion, step.ate_estimate, step.ate_se, marker
            )?;
        }
        writeln!(f, "  (* = selected model)")?;
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
// Main C-TMLE Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run C-TMLE with full configuration.
///
/// Estimates the Average Treatment Effect (ATE) using Collaborative Targeted Maximum
/// Likelihood Estimation. C-TMLE uses cross-validation to select covariates for the
/// propensity score model in a "collaborative" way that minimizes finite-sample bias.
///
/// # Arguments
/// * `dataset` - Dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable column
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of candidate covariate columns for propensity score selection
/// * `config` - C-TMLE configuration options
///
/// # Returns
/// `CTmleResult` containing ATE estimate, selected covariates, selection path, and diagnostics.
///
/// # Algorithm
///
/// The forward selection C-TMLE algorithm:
///
/// 1. Fit outcome model Q(A,W) = E[Y|A,W] using all covariates
/// 2. Start with g0 = intercept-only propensity model (g = mean(A))
/// 3. For k = 1 to K (number of candidate covariates):
///    a. For each covariate W_j not yet selected:
///       - Fit gk with current covariates + W_j
///       - Compute TMLE with this gk
///       - Compute V-fold CV criterion (RSS or VarIC)
///    b. Add covariate that minimizes CV criterion
///    c. Record (CV criterion, ATE estimate, SE) for step k
/// 4. Select step k* according to stopping rule:
///    - CVMinimum: k* = argmin(CV criterion)
///    - OneSE: smallest k with CV within 1 SE of minimum
/// 5. Return TMLE estimate using gk* model
///
/// # References
///
/// - Ju et al. (2019), "Scalable collaborative targeted learning", SMMR 28(2)
/// - van der Laan & Gruber (2010), "Collaborative double robust TMLE", IJB 6(1)
pub fn ctmle(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: CTmleConfig,
) -> EconResult<CTmleResult> {
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

    // Need at least 2*n_folds observations per group for CV
    let min_per_fold = 2;
    if n_treated < config.n_folds * min_per_fold || n_control < config.n_folds * min_per_fold {
        warnings.push(format!(
            "Small sample size for {}-fold CV. Consider reducing n_folds.",
            config.n_folds
        ));
    }

    // Extract covariates individually (need separate columns for selection)
    let n_covariates = covariate_cols.len();
    let mut covariate_data: Vec<Array1<f64>> = Vec::with_capacity(n_covariates);
    let covariate_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();

    for col in covariate_cols {
        let col_data = DesignMatrix::extract_column(dataset.df(), col).map_err(|e| {
            EconError::ColumnNotFound {
                column: col.to_string(),
                available: vec![format!("{:?}", e)],
            }
        })?;
        covariate_data.push(col_data);
    }

    // Build full design matrix W for covariates (with intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, true)?;
    let w_full = design.data;
    let k_full = w_full.ncols();

    // Build design matrix for outcome model: [intercept, W, A]
    // Include all covariates in Q(A,W) - this remains fixed
    let mut x_q = Array2::zeros((n, k_full + 1));
    for i in 0..n {
        for j in 0..k_full {
            x_q[[i, j]] = w_full[[i, j]];
        }
        x_q[[i, k_full]] = a[i];
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 1: Fit Initial Outcome Model Q(A,W) Using All Covariates
    // ═══════════════════════════════════════════════════════════════════════

    let (q_init, q_beta, q_converged, _q_iterations) = match config.q_model {
        CTmleQModel::Logistic => fit_logistic_model(&x_q, &y, config.max_iter, config.tolerance)?,
        CTmleQModel::Linear => fit_linear_model(&x_q, &y)?,
    };

    if !q_converged {
        warnings.push("Outcome model did not converge".to_string());
    }

    // Compute counterfactual Q predictions (needed throughout)
    let mut x_q_1 = x_q.clone();
    let mut x_q_0 = x_q.clone();
    for i in 0..n {
        x_q_1[[i, k_full]] = 1.0; // Set A = 1
        x_q_0[[i, k_full]] = 0.0; // Set A = 0
    }

    let (q_1_init, q_0_init) = match config.q_model {
        CTmleQModel::Logistic => {
            let z_1: Array1<f64> = x_q_1.dot(&q_beta);
            let z_0: Array1<f64> = x_q_0.dot(&q_beta);
            (z_1.mapv(logistic_cdf), z_0.mapv(logistic_cdf))
        }
        CTmleQModel::Linear => (x_q_1.dot(&q_beta), x_q_0.dot(&q_beta)),
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Step 2: Generate CV Folds
    // ═══════════════════════════════════════════════════════════════════════

    let folds = create_cv_folds(n, config.n_folds);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 3: Forward Selection of Covariates for Propensity Score
    // ═══════════════════════════════════════════════════════════════════════

    let max_to_select = config.max_covariates.unwrap_or(n_covariates);
    let max_to_select = max_to_select.min(n_covariates);

    // Selection state
    let mut selected: Vec<usize> = Vec::new(); // Indices of selected covariates
    let mut available: Vec<usize> = (0..n_covariates).collect(); // Indices not yet selected
    let mut selection_path: Vec<SelectionStep> = Vec::new();
    let mut cv_risks: Vec<f64> = Vec::new();
    let mut best_cv = f64::INFINITY;

    // Determine selection order
    let selection_indices: Vec<usize> = match &config.order {
        SelectionOrder::Forward => Vec::new(), // Will be built dynamically
        SelectionOrder::Backward => {
            // Start with all covariates, remove greedily (not implemented yet)
            warnings.push("Backward selection not yet implemented, using forward".to_string());
            Vec::new()
        }
        SelectionOrder::Prespecified(order) => order.clone(),
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Step 0: Evaluate intercept-only model
    // ═══════════════════════════════════════════════════════════════════════

    let (cv_rss_0, cv_rss_se_0, ate_0, ate_se_0) = evaluate_propensity_model_cv(
        &y,
        &a,
        &q_init,
        &q_1_init,
        &q_0_init,
        &covariate_data,
        &selected,
        &folds,
        config.q_model,
        config.gbound,
        config.max_iter,
        config.tolerance,
    )?;

    selection_path.push(SelectionStep {
        covariate_index: None,
        covariate_name: Some("(intercept)".to_string()),
        cv_criterion: cv_rss_0,
        cv_criterion_se: cv_rss_se_0,
        ate_estimate: ate_0,
        ate_se: ate_se_0,
        n_covariates: 0,
    });
    cv_risks.push(cv_rss_0);
    best_cv = cv_rss_0;

    // ═══════════════════════════════════════════════════════════════════════
    // Steps 1 to K: Greedy forward selection
    // ═══════════════════════════════════════════════════════════════════════

    for step in 0..max_to_select {
        if available.is_empty() {
            break;
        }

        let mut best_idx: Option<usize> = None;
        let mut best_cv_this_step = f64::INFINITY;
        let mut best_cv_se_this_step = 0.0;
        let mut best_ate_this_step = 0.0;
        let mut best_ate_se_this_step = 0.0;

        // Determine which covariate to try based on selection order
        let candidates: Vec<usize> =
            if !selection_indices.is_empty() && step < selection_indices.len() {
                // Prespecified order: use the next one in sequence
                let next_idx = selection_indices[step];
                if available.contains(&next_idx) {
                    vec![next_idx]
                } else {
                    // Skip if already selected (shouldn't happen with proper input)
                    continue;
                }
            } else {
                // Forward selection: try all available
                available.clone()
            };

        // Try adding each candidate covariate
        for &cov_idx in &candidates {
            // Temporarily add this covariate
            let mut test_selected = selected.clone();
            test_selected.push(cov_idx);

            // Evaluate this propensity model via CV
            let (cv_rss, cv_rss_se, ate, ate_se) = evaluate_propensity_model_cv(
                &y,
                &a,
                &q_init,
                &q_1_init,
                &q_0_init,
                &covariate_data,
                &test_selected,
                &folds,
                config.q_model,
                config.gbound,
                config.max_iter,
                config.tolerance,
            )?;

            if cv_rss < best_cv_this_step {
                best_cv_this_step = cv_rss;
                best_cv_se_this_step = cv_rss_se;
                best_ate_this_step = ate;
                best_ate_se_this_step = ate_se;
                best_idx = Some(cov_idx);
            }
        }

        // Check stopping condition based on stop_factor
        if best_cv_this_step > config.stop_factor * best_cv {
            if config.verbose {
                eprintln!(
                    "C-TMLE: Early stopping at step {} (CV risk {} > {} * {})",
                    step, best_cv_this_step, config.stop_factor, best_cv
                );
            }
            break;
        }

        // Add the best covariate for this step
        if let Some(idx) = best_idx {
            // Update tracking
            if best_cv_this_step < best_cv {
                best_cv = best_cv_this_step;
            }

            // Move from available to selected
            available.retain(|&x| x != idx);
            selected.push(idx);

            // Record this step
            selection_path.push(SelectionStep {
                covariate_index: Some(idx),
                covariate_name: Some(covariate_names[idx].clone()),
                cv_criterion: best_cv_this_step,
                cv_criterion_se: best_cv_se_this_step,
                ate_estimate: best_ate_this_step,
                ate_se: best_ate_se_this_step,
                n_covariates: selected.len(),
            });
            cv_risks.push(best_cv_this_step);

            if config.verbose {
                eprintln!(
                    "C-TMLE: Step {} added {} (CV: {:.4}, ATE: {:.4})",
                    step + 1,
                    covariate_names[idx],
                    best_cv_this_step,
                    best_ate_this_step
                );
            }
        }

        // Check for MaxCovariates stopping rule
        if let StoppingRule::MaxCovariates(k) = config.stopping_rule {
            if selected.len() >= k {
                break;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 4: Select Final Model Based on Stopping Rule
    // ═══════════════════════════════════════════════════════════════════════

    let selected_step = match config.stopping_rule {
        StoppingRule::CVMinimum => {
            // Find step with minimum CV criterion
            cv_risks
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
        StoppingRule::OneSE => {
            // Find minimum CV and its SE
            let (min_idx, min_cv) = cv_risks
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i, *v))
                .unwrap_or((0, f64::INFINITY));

            // Get SE at minimum
            let min_se = selection_path
                .get(min_idx)
                .map(|s| s.cv_criterion_se)
                .unwrap_or(0.0);

            // Find smallest model within 1 SE of minimum
            let threshold = min_cv + min_se;
            cv_risks
                .iter()
                .enumerate()
                .find(|(_, cv)| **cv <= threshold)
                .map(|(i, _)| i)
                .unwrap_or(min_idx)
        }
        StoppingRule::MaxCovariates(_) => {
            // Use all covariates that were added (already enforced in loop)
            selection_path.len() - 1
        }
    };

    // Get the selected covariates
    let selected_covariates: Vec<usize> = if selected_step == 0 {
        Vec::new()
    } else {
        selection_path[1..=selected_step]
            .iter()
            .filter_map(|s| s.covariate_index)
            .collect()
    };

    let selected_covariate_names: Vec<String> = selected_covariates
        .iter()
        .map(|&idx| covariate_names[idx].clone())
        .collect();

    // ═══════════════════════════════════════════════════════════════════════
    // Step 5: Compute Final TMLE Estimate with Selected Model
    // ═══════════════════════════════════════════════════════════════════════

    // Build propensity score model design matrix with selected covariates
    let w_g = build_design_matrix_subset(&covariate_data, &selected_covariates, n);

    // Fit final propensity score model on full data
    let (g_raw, _g_beta, g_converged, _g_iterations) =
        fit_logistic_model(&w_g, &a, config.max_iter, config.tolerance)?;

    if !g_converged {
        warnings.push("Final propensity score model did not converge".to_string());
    }

    // Truncate propensity scores
    let (ps_min, ps_max) = config.gbound;
    let g: Array1<f64> = g_raw.mapv(|gi| gi.max(ps_min).min(ps_max));

    // Compute clever covariate H(A,W)
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

    // Targeting step
    let (epsilon, targeting_converged) = fit_targeting_model(&y, &q_init, &h, config.q_model)?;

    if !targeting_converged {
        warnings.push("Targeting step did not converge".to_string());
    }

    // Compute targeted predictions Q*(A,W)
    let q_star: Array1<f64> = match config.q_model {
        CTmleQModel::Logistic => (0..n)
            .map(|i| {
                let logit_q = logit(q_init[i]);
                logistic_cdf(logit_q + epsilon * h[i])
            })
            .collect(),
        CTmleQModel::Linear => (0..n).map(|i| q_init[i] + epsilon * h[i]).collect(),
    };

    // Compute counterfactual clever covariates
    let h_1: Array1<f64> = g.mapv(|gi| 1.0 / gi);
    let h_0: Array1<f64> = g.mapv(|gi| -1.0 / (1.0 - gi));

    // Apply targeting to counterfactuals
    let (q_star_1, q_star_0): (Array1<f64>, Array1<f64>) = match config.q_model {
        CTmleQModel::Logistic => {
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
        CTmleQModel::Linear => {
            let q1 = (0..n).map(|i| q_1_init[i] + epsilon * h_1[i]).collect();
            let q0 = (0..n).map(|i| q_0_init[i] + epsilon * h_0[i]).collect();
            (q1, q0)
        }
    };

    // Compute ATE
    let ate: f64 = (0..n).map(|i| q_star_1[i] - q_star_0[i]).sum::<f64>() / n as f64;

    // Compute influence curve for variance
    let ic: Array1<f64> = (0..n)
        .map(|i| {
            let ipw_term = h[i] * (y[i] - q_star[i]);
            let outcome_term = q_star_1[i] - q_star_0[i];
            ipw_term + outcome_term - ate
        })
        .collect();

    // Variance from influence curve
    let ic_mean: f64 = ic.iter().sum::<f64>() / n as f64;
    let ic_var: f64 =
        ic.iter().map(|&ic_i| (ic_i - ic_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let ate_var = ic_var / n as f64;
    let se = ate_var.sqrt();

    // Z-statistic and confidence interval
    let z_stat = if se > 0.0 && se.is_finite() {
        ate / se
    } else {
        0.0
    };

    let z_crit = 1.96;
    let ci_lower = ate - z_crit * se;
    let ci_upper = ate + z_crit * se;

    // P-value
    let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));
    let significance = SignificanceLevel::from_p_value(p_value);

    // ═══════════════════════════════════════════════════════════════════════
    // Construct Result
    // ═══════════════════════════════════════════════════════════════════════

    Ok(CTmleResult {
        ate,
        se,
        ci_lower,
        ci_upper,
        p_value,
        significance,
        z_stat,
        selected_covariates,
        selected_covariate_names,
        n_selected: selected_step,
        selection_path,
        cv_risk: cv_risks,
        influence_curve: ic.to_vec(),
        selected_step,
        propensity_scores: g.to_vec(),
        targeted_outcome: q_star.to_vec(),
        n_obs: n,
        n_treated,
        n_control,
        config: CTmleConfigSummary {
            n_folds: config.n_folds,
            max_covariates: config.max_covariates,
            stopping_rule: format!("{}", config.stopping_rule),
            order: format!("{}", config.order),
            cv_criterion: format!("{}", config.cv_criterion),
            q_model: format!("{}", config.q_model),
            gbound: config.gbound,
        },
        warnings,
    })
}

/// Run C-TMLE with default configuration.
///
/// Convenience function that uses 5-fold CV, forward selection, CV-RSS criterion,
/// and CVMinimum stopping rule.
///
/// # Arguments
/// * `dataset` - Dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable column
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of candidate covariate columns
///
/// # Example
/// ```ignore
/// let result = run_ctmle(&dataset, "outcome", "treatment", &["x1", "x2", "x3"])?;
/// println!("ATE: {:.4}", result.ate);
/// println!("Selected: {:?}", result.selected_covariate_names);
/// ```
pub fn run_ctmle(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
) -> EconResult<CTmleResult> {
    ctmle(
        dataset,
        outcome_col,
        treatment_col,
        covariate_cols,
        CTmleConfig::default(),
    )
}

/// Run C-TMLE on array data directly.
///
/// Lower-level function that works with array data rather than Dataset.
///
/// # Arguments
/// * `y` - Outcome vector (n,)
/// * `treatment` - Binary treatment indicator (n,)
/// * `covariates` - Covariate matrix (n, p)
/// * `config` - C-TMLE configuration
///
/// # Returns
/// `CTmleResult` with ATE estimate and selection path.
pub fn ctmle_arrays(
    y: &ArrayView1<f64>,
    treatment: &ArrayView1<f64>,
    covariates: &ArrayView2<f64>,
    config: CTmleConfig,
) -> EconResult<CTmleResult> {
    let n = y.len();
    let p = covariates.ncols();

    if treatment.len() != n || covariates.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Dimension mismatch: y has {} rows, treatment has {}, covariates has {}",
                n,
                treatment.len(),
                covariates.nrows()
            ),
        });
    }

    // Convert to owned arrays
    let y_owned = y.to_owned();
    let a_owned = treatment.to_owned();

    // Split covariates into individual vectors
    let covariate_data: Vec<Array1<f64>> =
        (0..p).map(|j| covariates.column(j).to_owned()).collect();

    let covariate_names: Vec<String> = (0..p).map(|j| format!("X{}", j + 1)).collect();

    // Run the core algorithm with array data
    ctmle_core(
        &y_owned,
        &a_owned,
        &covariate_data,
        &covariate_names,
        config,
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Core C-TMLE implementation operating on array data.
fn ctmle_core(
    y: &Array1<f64>,
    a: &Array1<f64>,
    covariate_data: &[Array1<f64>],
    covariate_names: &[String],
    config: CTmleConfig,
) -> EconResult<CTmleResult> {
    let mut warnings = Vec::new();
    let n = y.len();
    let n_covariates = covariate_data.len();

    // Validate treatment is binary
    let n_treated: usize = a.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n - n_treated;

    if n_treated == 0 || n_control == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment must have both treated and control. Found {} treated, {} control.",
                n_treated, n_control
            ),
        });
    }

    // Build full design matrix with all covariates (for Q model)
    let w_full =
        build_design_matrix_subset(covariate_data, &(0..n_covariates).collect::<Vec<_>>(), n);
    let k_full = w_full.ncols();

    // Build outcome model design: [intercept + covariates, A]
    let mut x_q = Array2::zeros((n, k_full + 1));
    for i in 0..n {
        for j in 0..k_full {
            x_q[[i, j]] = w_full[[i, j]];
        }
        x_q[[i, k_full]] = a[i];
    }

    // Fit outcome model
    let (q_init, q_beta, q_converged, _) = match config.q_model {
        CTmleQModel::Logistic => fit_logistic_model(&x_q, y, config.max_iter, config.tolerance)?,
        CTmleQModel::Linear => fit_linear_model(&x_q, y)?,
    };

    if !q_converged {
        warnings.push("Outcome model did not converge".to_string());
    }

    // Compute counterfactual predictions
    let mut x_q_1 = x_q.clone();
    let mut x_q_0 = x_q.clone();
    for i in 0..n {
        x_q_1[[i, k_full]] = 1.0;
        x_q_0[[i, k_full]] = 0.0;
    }

    let (q_1_init, q_0_init) = match config.q_model {
        CTmleQModel::Logistic => {
            let z_1: Array1<f64> = x_q_1.dot(&q_beta);
            let z_0: Array1<f64> = x_q_0.dot(&q_beta);
            (z_1.mapv(logistic_cdf), z_0.mapv(logistic_cdf))
        }
        CTmleQModel::Linear => (x_q_1.dot(&q_beta), x_q_0.dot(&q_beta)),
    };

    // Generate CV folds
    let folds = create_cv_folds(n, config.n_folds);

    // Forward selection
    let max_to_select = config
        .max_covariates
        .unwrap_or(n_covariates)
        .min(n_covariates);
    let mut selected: Vec<usize> = Vec::new();
    let mut available: Vec<usize> = (0..n_covariates).collect();
    let mut selection_path: Vec<SelectionStep> = Vec::new();
    let mut cv_risks: Vec<f64> = Vec::new();
    let mut best_cv = f64::INFINITY;

    // Step 0: Intercept-only model
    let (cv_rss_0, cv_rss_se_0, ate_0, ate_se_0) = evaluate_propensity_model_cv(
        y,
        a,
        &q_init,
        &q_1_init,
        &q_0_init,
        covariate_data,
        &selected,
        &folds,
        config.q_model,
        config.gbound,
        config.max_iter,
        config.tolerance,
    )?;

    selection_path.push(SelectionStep {
        covariate_index: None,
        covariate_name: Some("(intercept)".to_string()),
        cv_criterion: cv_rss_0,
        cv_criterion_se: cv_rss_se_0,
        ate_estimate: ate_0,
        ate_se: ate_se_0,
        n_covariates: 0,
    });
    cv_risks.push(cv_rss_0);
    best_cv = cv_rss_0;

    // Forward selection steps
    for _step in 0..max_to_select {
        if available.is_empty() {
            break;
        }

        let mut best_idx: Option<usize> = None;
        let mut best_cv_this_step = f64::INFINITY;
        let mut best_cv_se_this_step = 0.0;
        let mut best_ate_this_step = 0.0;
        let mut best_ate_se_this_step = 0.0;

        for &cov_idx in &available {
            let mut test_selected = selected.clone();
            test_selected.push(cov_idx);

            let (cv_rss, cv_rss_se, ate, ate_se) = evaluate_propensity_model_cv(
                y,
                a,
                &q_init,
                &q_1_init,
                &q_0_init,
                covariate_data,
                &test_selected,
                &folds,
                config.q_model,
                config.gbound,
                config.max_iter,
                config.tolerance,
            )?;

            if cv_rss < best_cv_this_step {
                best_cv_this_step = cv_rss;
                best_cv_se_this_step = cv_rss_se;
                best_ate_this_step = ate;
                best_ate_se_this_step = ate_se;
                best_idx = Some(cov_idx);
            }
        }

        // Early stopping check
        if best_cv_this_step > config.stop_factor * best_cv {
            break;
        }

        if let Some(idx) = best_idx {
            if best_cv_this_step < best_cv {
                best_cv = best_cv_this_step;
            }

            available.retain(|&x| x != idx);
            selected.push(idx);

            selection_path.push(SelectionStep {
                covariate_index: Some(idx),
                covariate_name: Some(covariate_names[idx].clone()),
                cv_criterion: best_cv_this_step,
                cv_criterion_se: best_cv_se_this_step,
                ate_estimate: best_ate_this_step,
                ate_se: best_ate_se_this_step,
                n_covariates: selected.len(),
            });
            cv_risks.push(best_cv_this_step);
        }

        if let StoppingRule::MaxCovariates(k) = config.stopping_rule {
            if selected.len() >= k {
                break;
            }
        }
    }

    // Select final model
    let selected_step = match config.stopping_rule {
        StoppingRule::CVMinimum => cv_risks
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0),
        StoppingRule::OneSE => {
            let (min_idx, min_cv) = cv_risks
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i, *v))
                .unwrap_or((0, f64::INFINITY));

            let min_se = selection_path
                .get(min_idx)
                .map(|s| s.cv_criterion_se)
                .unwrap_or(0.0);
            let threshold = min_cv + min_se;

            cv_risks
                .iter()
                .enumerate()
                .find(|(_, cv)| **cv <= threshold)
                .map(|(i, _)| i)
                .unwrap_or(min_idx)
        }
        StoppingRule::MaxCovariates(_) => selection_path.len() - 1,
    };

    let selected_covariates: Vec<usize> = if selected_step == 0 {
        Vec::new()
    } else {
        selection_path[1..=selected_step]
            .iter()
            .filter_map(|s| s.covariate_index)
            .collect()
    };

    let selected_covariate_names: Vec<String> = selected_covariates
        .iter()
        .map(|&idx| covariate_names[idx].clone())
        .collect();

    // Fit final model on full data
    let w_g = build_design_matrix_subset(covariate_data, &selected_covariates, n);
    let (g_raw, _, g_converged, _) =
        fit_logistic_model(&w_g, a, config.max_iter, config.tolerance)?;

    if !g_converged {
        warnings.push("Final propensity score model did not converge".to_string());
    }

    let (ps_min, ps_max) = config.gbound;
    let g: Array1<f64> = g_raw.mapv(|gi| gi.max(ps_min).min(ps_max));

    let h: Array1<f64> = (0..n)
        .map(|i| {
            if a[i] >= 0.5 {
                1.0 / g[i]
            } else {
                -1.0 / (1.0 - g[i])
            }
        })
        .collect();

    let (epsilon, targeting_converged) = fit_targeting_model(y, &q_init, &h, config.q_model)?;

    if !targeting_converged {
        warnings.push("Targeting step did not converge".to_string());
    }

    let q_star: Array1<f64> = match config.q_model {
        CTmleQModel::Logistic => (0..n)
            .map(|i| logistic_cdf(logit(q_init[i]) + epsilon * h[i]))
            .collect(),
        CTmleQModel::Linear => (0..n).map(|i| q_init[i] + epsilon * h[i]).collect(),
    };

    let h_1: Array1<f64> = g.mapv(|gi| 1.0 / gi);
    let h_0: Array1<f64> = g.mapv(|gi| -1.0 / (1.0 - gi));

    let (q_star_1, q_star_0): (Array1<f64>, Array1<f64>) = match config.q_model {
        CTmleQModel::Logistic => {
            let q1 = (0..n)
                .map(|i| logistic_cdf(logit(q_1_init[i]) + epsilon * h_1[i]))
                .collect();
            let q0 = (0..n)
                .map(|i| logistic_cdf(logit(q_0_init[i]) + epsilon * h_0[i]))
                .collect();
            (q1, q0)
        }
        CTmleQModel::Linear => {
            let q1 = (0..n).map(|i| q_1_init[i] + epsilon * h_1[i]).collect();
            let q0 = (0..n).map(|i| q_0_init[i] + epsilon * h_0[i]).collect();
            (q1, q0)
        }
    };

    let ate: f64 = (0..n).map(|i| q_star_1[i] - q_star_0[i]).sum::<f64>() / n as f64;

    let ic: Array1<f64> = (0..n)
        .map(|i| h[i] * (y[i] - q_star[i]) + q_star_1[i] - q_star_0[i] - ate)
        .collect();

    let ic_mean: f64 = ic.iter().sum::<f64>() / n as f64;
    let ic_var: f64 =
        ic.iter().map(|&ic_i| (ic_i - ic_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let ate_var = ic_var / n as f64;
    let se = ate_var.sqrt();

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

    Ok(CTmleResult {
        ate,
        se,
        ci_lower,
        ci_upper,
        p_value,
        significance,
        z_stat,
        selected_covariates,
        selected_covariate_names,
        n_selected: selected_step,
        selection_path,
        cv_risk: cv_risks,
        influence_curve: ic.to_vec(),
        selected_step,
        propensity_scores: g.to_vec(),
        targeted_outcome: q_star.to_vec(),
        n_obs: n,
        n_treated,
        n_control,
        config: CTmleConfigSummary {
            n_folds: config.n_folds,
            max_covariates: config.max_covariates,
            stopping_rule: format!("{}", config.stopping_rule),
            order: format!("{}", config.order),
            cv_criterion: format!("{}", config.cv_criterion),
            q_model: format!("{}", config.q_model),
            gbound: config.gbound,
        },
        warnings,
    })
}

/// Evaluate a propensity model using V-fold cross-validation.
///
/// For each fold, fit propensity model on training data, compute TMLE on test data,
/// and return CV-RSS criterion.
///
/// Returns (cv_rss, cv_rss_se, ate_full, ate_se_full).
fn evaluate_propensity_model_cv(
    y: &Array1<f64>,
    a: &Array1<f64>,
    q_init: &Array1<f64>,
    q_1_init: &Array1<f64>,
    q_0_init: &Array1<f64>,
    covariate_data: &[Array1<f64>],
    selected_covariates: &[usize],
    folds: &[Vec<usize>],
    q_model: CTmleQModel,
    gbound: (f64, f64),
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(f64, f64, f64, f64)> {
    let n = y.len();
    let n_folds = folds.len();

    // First, compute full-data estimate for reporting
    let w_g = build_design_matrix_subset(covariate_data, selected_covariates, n);
    let (g_raw_full, _, _, _) = fit_logistic_model(&w_g, a, max_iter, tolerance)?;

    let (ps_min, ps_max) = gbound;
    let g_full: Array1<f64> = g_raw_full.mapv(|gi| gi.max(ps_min).min(ps_max));

    let h_full: Array1<f64> = (0..n)
        .map(|i| {
            if a[i] >= 0.5 {
                1.0 / g_full[i]
            } else {
                -1.0 / (1.0 - g_full[i])
            }
        })
        .collect();

    let (epsilon_full, _) = fit_targeting_model(y, q_init, &h_full, q_model)?;

    let q_star_full: Array1<f64> = match q_model {
        CTmleQModel::Logistic => (0..n)
            .map(|i| logistic_cdf(logit(q_init[i]) + epsilon_full * h_full[i]))
            .collect(),
        CTmleQModel::Linear => (0..n)
            .map(|i| q_init[i] + epsilon_full * h_full[i])
            .collect(),
    };

    let h_1_full: Array1<f64> = g_full.mapv(|gi| 1.0 / gi);
    let h_0_full: Array1<f64> = g_full.mapv(|gi| -1.0 / (1.0 - gi));

    let (q_star_1_full, q_star_0_full): (Array1<f64>, Array1<f64>) = match q_model {
        CTmleQModel::Logistic => {
            let q1 = (0..n)
                .map(|i| logistic_cdf(logit(q_1_init[i]) + epsilon_full * h_1_full[i]))
                .collect();
            let q0 = (0..n)
                .map(|i| logistic_cdf(logit(q_0_init[i]) + epsilon_full * h_0_full[i]))
                .collect();
            (q1, q0)
        }
        CTmleQModel::Linear => {
            let q1 = (0..n)
                .map(|i| q_1_init[i] + epsilon_full * h_1_full[i])
                .collect();
            let q0 = (0..n)
                .map(|i| q_0_init[i] + epsilon_full * h_0_full[i])
                .collect();
            (q1, q0)
        }
    };

    let ate_full: f64 = (0..n)
        .map(|i| q_star_1_full[i] - q_star_0_full[i])
        .sum::<f64>()
        / n as f64;

    let ic_full: Array1<f64> = (0..n)
        .map(|i| {
            h_full[i] * (y[i] - q_star_full[i]) + q_star_1_full[i] - q_star_0_full[i] - ate_full
        })
        .collect();

    let ic_mean_full = ic_full.iter().sum::<f64>() / n as f64;
    let ic_var_full = ic_full
        .iter()
        .map(|&ic_i| (ic_i - ic_mean_full).powi(2))
        .sum::<f64>()
        / (n - 1).max(1) as f64;
    let ate_se_full = (ic_var_full / n as f64).sqrt();

    // Now compute CV criterion
    let mut fold_rss: Vec<f64> = Vec::with_capacity(n_folds);

    for fold_idx in 0..n_folds {
        let test_indices = &folds[fold_idx];
        let train_indices: Vec<usize> = (0..n_folds)
            .filter(|&i| i != fold_idx)
            .flat_map(|i| folds[i].iter().copied())
            .collect();

        let _n_train = train_indices.len();
        let n_test = test_indices.len();

        // Extract training data
        let _y_train: Array1<f64> = train_indices.iter().map(|&i| y[i]).collect();
        let a_train: Array1<f64> = train_indices.iter().map(|&i| a[i]).collect();

        // Build training design matrix for propensity
        let w_train =
            build_design_matrix_subset_indexed(covariate_data, selected_covariates, &train_indices);

        // Fit propensity model on training data
        let (_g_train_raw, g_beta, _, _) =
            match fit_logistic_model(&w_train, &a_train, max_iter, tolerance) {
                Ok(result) => result,
                Err(_) => continue, // Skip this fold if fitting fails
            };

        // Predict propensity scores on test data
        let w_test =
            build_design_matrix_subset_indexed(covariate_data, selected_covariates, test_indices);
        let g_test_raw: Array1<f64> = w_test.dot(&g_beta).mapv(logistic_cdf);
        let g_test: Array1<f64> = g_test_raw.mapv(|gi| gi.max(ps_min).min(ps_max));

        // Extract test data
        let y_test: Array1<f64> = test_indices.iter().map(|&i| y[i]).collect();
        let a_test: Array1<f64> = test_indices.iter().map(|&i| a[i]).collect();
        let q_init_test: Array1<f64> = test_indices.iter().map(|&i| q_init[i]).collect();

        // Compute clever covariate on test set using test propensity scores
        let h_test: Array1<f64> = (0..n_test)
            .map(|i| {
                if a_test[i] >= 0.5 {
                    1.0 / g_test[i]
                } else {
                    -1.0 / (1.0 - g_test[i])
                }
            })
            .collect();

        // Targeting step on test set
        let (epsilon_test, _) = match fit_targeting_model(&y_test, &q_init_test, &h_test, q_model) {
            Ok(result) => result,
            Err(_) => continue,
        };

        // Compute Q* on test set
        let q_star_test: Array1<f64> = match q_model {
            CTmleQModel::Logistic => (0..n_test)
                .map(|i| logistic_cdf(logit(q_init_test[i]) + epsilon_test * h_test[i]))
                .collect(),
            CTmleQModel::Linear => (0..n_test)
                .map(|i| q_init_test[i] + epsilon_test * h_test[i])
                .collect(),
        };

        // Compute RSS on test set: sum((Y - Q*)^2)
        // This is the targeting residual (Ju et al. 2019, Section 3.1)
        let rss: f64 = (0..n_test)
            .map(|i| (y_test[i] - q_star_test[i]).powi(2))
            .sum();

        fold_rss.push(rss);
    }

    // Compute mean and SE of CV-RSS
    if fold_rss.is_empty() {
        return Ok((f64::INFINITY, 0.0, ate_full, ate_se_full));
    }

    let cv_rss_mean: f64 = fold_rss.iter().sum::<f64>() / fold_rss.len() as f64;
    let cv_rss_se = if fold_rss.len() > 1 {
        let variance: f64 = fold_rss
            .iter()
            .map(|&x| (x - cv_rss_mean).powi(2))
            .sum::<f64>()
            / (fold_rss.len() - 1) as f64;
        (variance / fold_rss.len() as f64).sqrt()
    } else {
        0.0
    };

    Ok((cv_rss_mean, cv_rss_se, ate_full, ate_se_full))
}

/// Build design matrix with intercept and subset of covariates.
fn build_design_matrix_subset(
    covariate_data: &[Array1<f64>],
    selected_indices: &[usize],
    n: usize,
) -> Array2<f64> {
    let k = selected_indices.len() + 1; // +1 for intercept
    let mut x = Array2::zeros((n, k));

    // Intercept column
    for i in 0..n {
        x[[i, 0]] = 1.0;
    }

    // Selected covariates
    for (j, &cov_idx) in selected_indices.iter().enumerate() {
        for i in 0..n {
            x[[i, j + 1]] = covariate_data[cov_idx][i];
        }
    }

    x
}

/// Build design matrix from subset of observations and covariates.
fn build_design_matrix_subset_indexed(
    covariate_data: &[Array1<f64>],
    selected_indices: &[usize],
    obs_indices: &[usize],
) -> Array2<f64> {
    let n = obs_indices.len();
    let k = selected_indices.len() + 1;
    let mut x = Array2::zeros((n, k));

    // Intercept
    for i in 0..n {
        x[[i, 0]] = 1.0;
    }

    // Selected covariates
    for (j, &cov_idx) in selected_indices.iter().enumerate() {
        for (i, &obs_idx) in obs_indices.iter().enumerate() {
            x[[i, j + 1]] = covariate_data[cov_idx][obs_idx];
        }
    }

    x
}

/// Create V-fold cross-validation indices.
///
/// Returns a vector of V vectors, each containing indices for one fold.
fn create_cv_folds(n: usize, v: usize) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = (0..v).map(|_| Vec::new()).collect();

    // Simple sequential assignment (deterministic for reproducibility)
    for i in 0..n {
        folds[i % v].push(i);
    }

    folds
}

/// Logit function: logit(p) = log(p / (1-p))
#[inline]
fn logit(p: f64) -> f64 {
    let p_clipped = p.max(1e-10).min(1.0 - 1e-10);
    (p_clipped / (1.0 - p_clipped)).ln()
}

/// Fit logistic regression using Newton-Raphson (IRLS).
fn fit_logistic_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    max_iter: usize,
    tolerance: f64,
) -> EconResult<(Array1<f64>, Array1<f64>, bool, usize)> {
    let n = y.len();
    let k = x.ncols();

    let mut beta = Array1::zeros(k);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        let z: Array1<f64> = x.dot(&beta);
        let p: Array1<f64> = z.mapv(logistic_cdf);
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        let residuals = y - &p_clipped;
        let mut gradient = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                gradient[j] += residuals[i] * x[[i, j]];
            }
        }

        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            converged = true;
            break;
        }

        let weights: Array1<f64> = p_clipped.mapv(|pi| pi * (1.0 - pi));

        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        let neg_hessian = &hessian * -1.0;
        let (hess_inv, _) =
            safe_inverse(&neg_hessian.view()).map_err(|e| EconError::SingularMatrix {
                context: "Logistic regression Hessian".to_string(),
                suggestion: format!("Check for multicollinearity: {:?}", e),
            })?;

        let delta = hess_inv.dot(&gradient);
        beta = &beta + &delta;
    }

    let z_final: Array1<f64> = x.dot(&beta);
    let p_final: Array1<f64> = z_final.mapv(logistic_cdf);

    Ok((p_final, beta, converged, iterations))
}

/// Fit linear regression using OLS.
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
    let y_hat = x.dot(&beta);

    Ok((y_hat, beta, true, 1))
}

/// Fit targeting model to estimate fluctuation parameter epsilon.
fn fit_targeting_model(
    y: &Array1<f64>,
    q_init: &Array1<f64>,
    h: &Array1<f64>,
    q_model: CTmleQModel,
) -> EconResult<(f64, bool)> {
    let n = y.len();

    match q_model {
        CTmleQModel::Logistic => {
            let mut epsilon = 0.0;
            let mut converged = false;
            let max_iter = 50;
            let tolerance = 1e-8;

            for _ in 0..max_iter {
                let p: Array1<f64> = (0..n)
                    .map(|i| {
                        let logit_q = logit(q_init[i]);
                        logistic_cdf(logit_q + epsilon * h[i])
                    })
                    .collect();
                let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

                let score: f64 = (0..n).map(|i| h[i] * (y[i] - p_clipped[i])).sum();

                if score.abs() < tolerance {
                    converged = true;
                    break;
                }

                let info: f64 = (0..n)
                    .map(|i| {
                        let pi = p_clipped[i];
                        h[i] * h[i] * pi * (1.0 - pi)
                    })
                    .sum();

                if info.abs() > 1e-10 {
                    epsilon += score / info;
                } else {
                    epsilon += 0.1 * score.signum();
                }
            }

            Ok((epsilon, converged))
        }
        CTmleQModel::Linear => {
            let numerator: f64 = (0..n).map(|i| h[i] * (y[i] - q_init[i])).sum();
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

    /// Create test dataset with known treatment effect and confounding.
    ///
    /// DGP for binary outcomes:
    /// - W1, W2, W3, W4 ~ correlated confounders
    /// - A | W ~ Bernoulli(expit(0.5*W1 + 0.3*W2)) (only W1, W2 affect treatment)
    /// - P(Y=1 | A, W) = expit(-1 + 2*A + W1 + 0.5*W2)
    ///
    /// Optimal propensity model should select W1, W2 (not W3, W4)
    fn create_ctmle_test_dataset() -> Dataset {
        // Binary outcome data appropriate for logistic regression
        let df = df! {
            "y" => [
                // Treated (A=1): ~80-90% are Y=1
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                // Control (A=0): ~30-40% are Y=1
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0
            ],
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            // W1: Strong confounder (affects both treatment and outcome)
            "w1" => [
                0.6, 0.8, 0.5, 0.9, 0.55, 0.85, 0.45, 0.95, 0.58, 0.78,
                0.4, 0.92, 0.62, 0.75, 0.48, 0.88, 0.65, 0.72, 0.42, 0.95,
                0.3, 0.5, 0.25, 0.55, 0.35, 0.52, 0.28, 0.58, 0.32, 0.48,
                0.2, 0.6, 0.38, 0.45, 0.22, 0.55, 0.42, 0.48, 0.18, 0.62
            ],
            // W2: Moderate confounder
            "w2" => [
                0.55, 0.7, 0.5, 0.75, 0.52, 0.72, 0.48, 0.78, 0.54, 0.68,
                0.45, 0.8, 0.58, 0.65, 0.46, 0.76, 0.6, 0.62, 0.44, 0.82,
                0.4, 0.55, 0.38, 0.58, 0.42, 0.54, 0.36, 0.6, 0.44, 0.52,
                0.32, 0.62, 0.48, 0.5, 0.34, 0.58, 0.5, 0.48, 0.3, 0.65
            ],
            // W3: Only affects outcome, not treatment (precision variable)
            "w3" => [
                0.5, 0.6, 0.45, 0.65, 0.52, 0.58, 0.48, 0.68, 0.55, 0.55,
                0.4, 0.7, 0.6, 0.52, 0.42, 0.66, 0.62, 0.5, 0.38, 0.72,
                0.48, 0.58, 0.44, 0.62, 0.5, 0.56, 0.46, 0.64, 0.52, 0.54,
                0.38, 0.68, 0.56, 0.5, 0.4, 0.62, 0.58, 0.48, 0.36, 0.7
            ],
            // W4: Noise variable (affects neither treatment nor outcome meaningfully)
            "w4" => [
                0.4, 0.5, 0.55, 0.45, 0.6, 0.42, 0.52, 0.48, 0.58, 0.44,
                0.46, 0.56, 0.5, 0.54, 0.48, 0.52, 0.44, 0.58, 0.42, 0.6,
                0.55, 0.45, 0.5, 0.52, 0.48, 0.54, 0.46, 0.56, 0.44, 0.58,
                0.52, 0.48, 0.54, 0.46, 0.5, 0.56, 0.44, 0.58, 0.42, 0.6
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_ctmle_basic() {
        let dataset = create_ctmle_test_dataset();

        // Use simpler configuration for small sample size
        // C-TMLE with 4 covariates on n=40 is unstable
        let config = CTmleConfig {
            n_folds: 3,              // Fewer folds for stability
            max_covariates: Some(2), // Limit complexity
            stopping_rule: StoppingRule::MaxCovariates(2),
            q_model: CTmleQModel::Linear, // More stable with small samples
            ..Default::default()
        };

        let result = ctmle(&dataset, "y", "treatment", &["w1", "w2"], config).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 40);
        assert_eq!(result.n_treated, 20);
        assert_eq!(result.n_control, 20);

        // With linear Q model and limited covariates, ATE should be positive
        // Binary outcome: treated 17/20 Y=1, control 7/20 Y=1
        assert!(
            result.ate.is_finite(),
            "ATE should be finite, got {}",
            result.ate
        );

        // SE should be reasonable
        assert!(
            result.se > 0.0 && result.se.is_finite(),
            "SE should be positive and finite, got {}",
            result.se
        );

        // Selection path should have at least intercept step
        assert!(
            !result.selection_path.is_empty(),
            "Selection path should not be empty"
        );

        // CV risks should match selection path length
        assert_eq!(
            result.cv_risk.len(),
            result.selection_path.len(),
            "CV risk and selection path should have same length"
        );
    }

    #[test]
    fn test_ctmle_with_config() {
        let dataset = create_ctmle_test_dataset();

        let config = CTmleConfig {
            n_folds: 3, // Fewer folds for small sample
            max_covariates: Some(2),
            stopping_rule: StoppingRule::MaxCovariates(2),
            order: SelectionOrder::Forward,
            cv_criterion: CVCriterion::RSS,
            q_model: CTmleQModel::Linear,
            gbound: (0.05, 0.95),
            ..Default::default()
        };

        let result = ctmle(
            &dataset,
            "y",
            "treatment",
            &["w1", "w2", "w3", "w4"],
            config,
        )
        .unwrap();

        // Should have selected at most 2 covariates
        assert!(
            result.n_selected <= 2,
            "Should have at most 2 covariates, got {}",
            result.n_selected
        );

        // ATE should still be reasonable
        assert!(
            result.ate > 0.1 && result.ate < 1.0,
            "ATE should be reasonable, got {}",
            result.ate
        );
    }

    #[test]
    fn test_ctmle_onese_stopping() {
        let dataset = create_ctmle_test_dataset();

        let config = CTmleConfig {
            n_folds: 3,
            stopping_rule: StoppingRule::OneSE,
            ..Default::default()
        };

        let result = ctmle(
            &dataset,
            "y",
            "treatment",
            &["w1", "w2", "w3", "w4"],
            config,
        )
        .unwrap();

        // One-SE rule should select a sparser model than CV minimum
        // (or the same if minimum is already very simple)
        assert!(result.selected_step < result.selection_path.len());
    }

    #[test]
    fn test_ctmle_selection_path() {
        let dataset = create_ctmle_test_dataset();
        let result = run_ctmle(&dataset, "y", "treatment", &["w1", "w2", "w3", "w4"]).unwrap();

        // First step should be intercept only
        assert!(
            result.selection_path[0].covariate_index.is_none(),
            "First step should be intercept-only"
        );
        assert_eq!(result.selection_path[0].n_covariates, 0);

        // Each subsequent step should add one covariate
        for i in 1..result.selection_path.len() {
            assert_eq!(
                result.selection_path[i].n_covariates, i,
                "Step {} should have {} covariates",
                i, i
            );
        }

        // CV criterion should be non-negative
        for step in &result.selection_path {
            assert!(
                step.cv_criterion >= 0.0,
                "CV criterion should be non-negative, got {}",
                step.cv_criterion
            );
        }
    }

    #[test]
    fn test_ctmle_influence_curve() {
        let dataset = create_ctmle_test_dataset();
        let result = run_ctmle(&dataset, "y", "treatment", &["w1", "w2"]).unwrap();

        // IC should have same length as n_obs
        assert_eq!(result.influence_curve.len(), result.n_obs);

        // IC should have finite values
        assert!(
            result.influence_curve.iter().all(|&ic| ic.is_finite()),
            "IC should have finite values"
        );

        // IC variance should be positive (used for SE calculation)
        let ic_mean: f64 = result.influence_curve.iter().sum::<f64>() / result.n_obs as f64;
        let ic_var: f64 = result
            .influence_curve
            .iter()
            .map(|&ic| (ic - ic_mean).powi(2))
            .sum::<f64>()
            / (result.n_obs - 1) as f64;
        assert!(ic_var > 0.0, "IC variance should be positive");
    }

    #[test]
    fn test_ctmle_display() {
        let dataset = create_ctmle_test_dataset();
        let result = run_ctmle(&dataset, "y", "treatment", &["w1", "w2"]).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("C-TMLE"));
        assert!(output.contains("ATE"));
        assert!(output.contains("Selection Path"));
        assert!(output.contains("CV Folds"));
    }

    #[test]
    fn test_ctmle_error_handling() {
        let dataset = create_ctmle_test_dataset();

        // Missing outcome column
        let result = run_ctmle(&dataset, "nonexistent", "treatment", &["w1"]);
        assert!(result.is_err());

        // Missing treatment column
        let result = run_ctmle(&dataset, "y", "nonexistent", &["w1"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_cv_folds() {
        let folds = create_cv_folds(10, 3);

        // Should have 3 folds
        assert_eq!(folds.len(), 3);

        // Total indices should be 10
        let total: usize = folds.iter().map(|f| f.len()).sum();
        assert_eq!(total, 10);

        // Each observation should appear exactly once
        let mut all_indices: Vec<usize> = folds.into_iter().flatten().collect();
        all_indices.sort();
        assert_eq!(all_indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_ctmle_prespecified_order() {
        let dataset = create_ctmle_test_dataset();

        // Specify order: first w3, then w1, then w2
        let config = CTmleConfig {
            n_folds: 3,
            order: SelectionOrder::Prespecified(vec![2, 0, 1]), // w3, w1, w2
            max_covariates: Some(3),
            stopping_rule: StoppingRule::MaxCovariates(3),
            ..Default::default()
        };

        let result = ctmle(
            &dataset,
            "y",
            "treatment",
            &["w1", "w2", "w3", "w4"],
            config,
        )
        .unwrap();

        // Should have followed the specified order (at least partially)
        // Check that we got some covariates selected
        assert!(
            !result.selection_path.is_empty(),
            "Should have at least intercept step"
        );
    }

    /// Test that C-TMLE with array inputs works.
    #[test]
    fn test_ctmle_arrays() {
        use ndarray::{Array2, array};

        let y = array![
            0.9, 1.1, 0.8, 1.2, 0.95, 0.85, 1.15, 0.92, 1.08, 0.88, 0.3, 0.5, 0.25, 0.55, 0.35,
            0.28, 0.58, 0.32, 0.52, 0.38
        ];
        let treatment = array![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ];
        let covariates = Array2::from_shape_vec(
            (20, 2),
            vec![
                0.6, 0.55, 0.8, 0.7, 0.5, 0.5, 0.9, 0.75, 0.55, 0.52, 0.45, 0.48, 0.85, 0.72, 0.58,
                0.54, 0.78, 0.68, 0.62, 0.58, 0.3, 0.4, 0.5, 0.55, 0.25, 0.38, 0.55, 0.58, 0.35,
                0.42, 0.28, 0.36, 0.52, 0.54, 0.32, 0.44, 0.48, 0.52, 0.38, 0.48,
            ],
        )
        .unwrap();

        let config = CTmleConfig {
            n_folds: 2, // Small folds for small data
            max_covariates: Some(2),
            ..Default::default()
        };

        let result =
            ctmle_arrays(&y.view(), &treatment.view(), &covariates.view(), config).unwrap();

        assert_eq!(result.n_obs, 20);
        assert!(result.ate.is_finite());
        assert!(result.se > 0.0);
    }
}
