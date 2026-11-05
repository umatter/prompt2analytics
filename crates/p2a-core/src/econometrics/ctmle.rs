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
//! # Algorithm (greedy selection of g with a fixed initial Q)
//!
//! 1. **Fit the initial outcome model Q(A,W)** on all candidate covariates (a logistic glm,
//!    or least squares for `CTmleQModel::Linear`), on the full data and on the training rows
//!    of every fold (folds: observation `i` is in fold `i mod V`).
//! 2. **Build a path of propensity models g.** Forward: start from the intercept-only g and
//!    add, at every step, the candidate with the smallest CV criterion. Backward: start from
//!    all candidates and remove one per step. Prespecified: add the covariates in exactly the
//!    given order (an empty list means the listed order).
//! 3. **CV criterion of a candidate g** (the honest V-fold risk): for every fold v, Q, g and
//!    the fluctuation epsilon (one clever covariate H = A/g - (1-A)/(1-g)) are fitted on the
//!    training folds and the loss is evaluated on fold v.
//!    - `RSS`: the sum over folds of the validation-fold RSS (ctmle's likelihood), SE
//!      sqrt(V) sd(fold RSS);
//!    - `VarIC`: the mean over folds of the validation-fold variance of the influence curve
//!      (ctmle's varIC.cv), SE sd(fold varIC) / sqrt(V);
//!    - `PenRSS`: RSS + penalty_factor * ln(n) * k for a g with k covariates, SE as RSS.
//! 4. **Select a step** (CV minimum, the one-SE rule, or a fixed number of covariates) and
//!    report the full-data TMLE with that g.
//!
//! Deviations from `ctmle::ctmleDiscrete` 0.1.2: the initial Q is fixed (ctmle updates Q
//! between clever covariates); the initial Q is a parametric glm / lm on all candidates, not
//! SuperLearner; the folds are deterministic (`i mod V`); g is fitted on the training folds
//! (ctmle fits g on all rows); a continuous outcome is fitted with the linear fluctuation
//! (`CTmleQModel::Linear`) or, under the logistic Q, rescaled once to [0, 1] by its range
//! (the criteria are then on that scale, the estimates on the original scale); the
//! propensity bounds are symmetric.
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
//! use p2a_core::econometrics::{ctmle, CTmleConfig, StoppingRule, SelectionOrder};
//!
//! let config = CTmleConfig {
//!     n_folds: 5,
//!     max_covariates: Some(10),
//!     stopping_rule: StoppingRule::CVMinimum,
//!     order: SelectionOrder::Forward,
//!     ..Default::default()
//! };
//!
//! let result = ctmle(&dataset, "outcome", "treatment", &["x1", "x2", "x3", "x4"], config)?;
//! println!("ATE: {:.4} (SE: {:.4})", result.ate, result.se);
//! println!("Selected covariates: {:?}", result.selected_covariate_names);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
use crate::econometrics::glm::plogis;
use crate::econometrics::tmle::{
    FluctuationForm, GlmControl, NuisanceFit, QModel, QPredictions, ate_and_ic, bound_scores,
    check_binary_treatment, fit_epsilon, fit_outcome, fit_treatment, fluctuate, glm_convergence,
    logit_q, outcome_scale, prefixed, sample_variance, wald95,
};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::options::Choices;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::SignificanceLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Stopping rule for covariate selection in C-TMLE.
///
/// Controls which step of the selection path is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StoppingRule {
    /// The step with the minimum CV criterion (the first one on ties).
    /// This is the default and most commonly used approach.
    #[default]
    CVMinimum,

    /// One-standard-error rule: the step with the fewest covariates whose criterion is
    /// within one SE of the minimum.
    OneSE,

    /// Forward and prespecified selection stop after k additions, backward elimination after
    /// reaching k covariates; the last step is selected.
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum RuleKind {
    CvMinimum,
    OneSe,
    MaxCovariates,
}

const RULE_CHOICES: Choices<RuleKind> = Choices {
    option: "stopping_rule",
    values: &[
        (
            &["cv_minimum", "cvmin", "cv_min", "minimum", "min"],
            RuleKind::CvMinimum,
        ),
        (
            &["one_se", "onese", "1se", "one_standard_error"],
            RuleKind::OneSe,
        ),
        (
            &["max_covariates", "maxcovariates", "max"],
            RuleKind::MaxCovariates,
        ),
    ],
};

impl StoppingRule {
    /// Parse a user string; `max_covariates` needs the number of covariates to keep.
    pub fn parse_with(s: &str, max_covariates: Option<usize>) -> EconResult<StoppingRule> {
        Ok(match RULE_CHOICES.parse(s)? {
            RuleKind::CvMinimum => StoppingRule::CVMinimum,
            RuleKind::OneSe => StoppingRule::OneSE,
            RuleKind::MaxCovariates => match max_covariates {
                Some(k) => StoppingRule::MaxCovariates(k),
                None => {
                    return Err(EconError::InvalidSpecification {
                        message: "stopping_rule 'max_covariates' needs max_covariates (the \
                                  number of covariates to keep)"
                            .to_string(),
                    });
                }
            },
        })
    }

    /// "cv_minimum | one_se | max_covariates", for CLI help and tool descriptions.
    pub fn help() -> String {
        RULE_CHOICES.help()
    }
}

impl std::str::FromStr for StoppingRule {
    type Err = EconError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        StoppingRule::parse_with(s, None)
    }
}

/// Order for covariate selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SelectionOrder {
    /// Forward selection: start from the intercept-only g and add, per step, the candidate
    /// with the best criterion. This is the default.
    #[default]
    Forward,

    /// Backward elimination: start from all candidates and remove, per step, the candidate
    /// whose removal gives the best criterion.
    Backward,

    /// Prespecified order: covariates are added in exactly this order, one per step; the path
    /// ends when the list is used up. An empty list means the order in which the covariates
    /// are listed (ctmle's preOrder default). Indices must be below the number of candidates
    /// and distinct.
    Prespecified(Vec<usize>),
}

impl fmt::Display for SelectionOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectionOrder::Forward => write!(f, "Forward"),
            SelectionOrder::Backward => write!(f, "Backward"),
            SelectionOrder::Prespecified(order) if order.is_empty() => {
                write!(f, "Prespecified (listed order)")
            }
            SelectionOrder::Prespecified(order) => write!(f, "Prespecified({:?})", order),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum OrderKind {
    Forward,
    Backward,
    Prespecified,
}

const ORDER_CHOICES: Choices<OrderKind> = Choices {
    option: "order",
    values: &[
        (&["forward", "fwd"], OrderKind::Forward),
        (
            &["backward", "bwd", "backward_elimination"],
            OrderKind::Backward,
        ),
        (
            &["prespecified", "preorder", "listed", "pre_ordered"],
            OrderKind::Prespecified,
        ),
    ],
};

impl SelectionOrder {
    /// "forward | backward | prespecified", for CLI help and tool descriptions.
    pub fn help() -> String {
        ORDER_CHOICES.help()
    }
}

impl std::str::FromStr for SelectionOrder {
    type Err = EconError;
    /// "prespecified" gives the listed order (`Prespecified(vec![])`).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match ORDER_CHOICES.parse(s)? {
            OrderKind::Forward => SelectionOrder::Forward,
            OrderKind::Backward => SelectionOrder::Backward,
            OrderKind::Prespecified => SelectionOrder::Prespecified(Vec::new()),
        })
    }
}

/// Cross-validation criterion for model selection. Every criterion is computed on the
/// validation folds of a V-fold split in which Q, g and the fluctuation are fitted on the
/// training folds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CVCriterion {
    /// The sum over folds of the validation-fold RSS of Q* (ctmle's likelihood); SE
    /// sqrt(V) sd(fold RSS).
    #[default]
    RSS,

    /// The mean over folds of the validation-fold variance of the influence curve (ctmle's
    /// varIC.cv); SE sd(fold varIC) / sqrt(V).
    VarIC,

    /// Penalized RSS: RSS + penalty_factor * ln(n) * k for a g with k covariates; SE as RSS.
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

crate::option_enum!(
    CVCriterion,
    Choices {
        option: "cv_criterion",
        values: &[
            (&["rss", "cv_rss", "likelihood"], CVCriterion::RSS),
            (
                &["varic", "var_ic", "cv_varic", "variance_ic"],
                CVCriterion::VarIC
            ),
            (
                &[
                    "penrss",
                    "pen_rss",
                    "cv_penrss",
                    "penalized_rss",
                    "penalised_rss"
                ],
                CVCriterion::PenRSS
            ),
        ],
    },
    no_display
);

/// Outcome model specification for C-TMLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CTmleQModel {
    /// Linear regression for continuous Y
    Linear,
    /// Logistic regression (default): a 0/1 outcome, a proportion in [0, 1], or a continuous
    /// outcome rescaled to [0, 1] by its range
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

crate::option_enum!(
    CTmleQModel,
    Choices {
        option: "q_model",
        values: &[
            (
                &["logistic", "logit", "binomial", "binary"],
                CTmleQModel::Logistic
            ),
            (
                &["linear", "gaussian", "continuous", "ols"],
                CTmleQModel::Linear
            ),
        ],
    },
    no_display
);

/// Configuration for C-TMLE estimation.
#[derive(Debug, Clone)]
pub struct CTmleConfig {
    /// Number of cross-validation folds (default: 5). Must be between 2 and n/2, so
    /// that every fold holds at least 2 observations.
    pub n_folds: usize,

    /// Only path steps with at most this many covariates can be selected; forward and
    /// prespecified selection stop there. None means no limit.
    pub max_covariates: Option<usize>,

    /// Stopping rule for covariate selection.
    pub stopping_rule: StoppingRule,

    /// Order for covariate selection.
    pub order: SelectionOrder,

    /// Cross-validation criterion for model selection.
    pub cv_criterion: CVCriterion,

    /// Outcome model specification.
    pub q_model: CTmleQModel,

    /// Truncation bounds for propensity scores (min, max), with 0 < min < max < 1.
    /// Default: (0.025, 0.975) as recommended in ctmle package.
    pub gbound: (f64, f64),

    /// `glm.fit`'s `maxit` for the logistic fits (at least 1; default 100).
    pub max_iter: usize,

    /// `glm.fit`'s `epsilon` for the logistic fits (finite and > 0; default 1e-8, R's
    /// `glm.control` default).
    pub tolerance: f64,

    /// Used by PenRSS: RSS + penalty_factor * ln(n) * k (finite and >= 0).
    pub penalty_factor: f64,

    /// Factor for early stopping (finite and > 0): the path stops when the best criterion of
    /// a step exceeds stop_factor times the best so far (ctmle's stopFactor).
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

fn nan() -> f64 {
    f64::NAN
}

/// Information about a single step in the selection path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionStep {
    /// Index of the covariate added (or removed for backward selection).
    /// None for the initial model.
    pub covariate_index: Option<usize>,

    /// Name of the covariate added (if available).
    pub covariate_name: Option<String>,

    /// Cross-validation criterion value at this step (on the fitting scale).
    pub cv_criterion: f64,

    /// Standard error of CV criterion (across folds).
    pub cv_criterion_se: f64,

    /// ATE estimate at this step (full-data TMLE with this step's g).
    pub ate_estimate: f64,

    /// Standard error of ATE at this step.
    pub ate_se: f64,

    /// Number of covariates in the model at this step.
    pub n_covariates: usize,

    /// Total validation RSS of this step's g (whatever the criterion).
    #[serde(default = "nan")]
    pub cv_rss: f64,

    /// CV variance of the influence curve of this step's g.
    #[serde(default = "nan")]
    pub cv_var_ic: f64,
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

    /// Indices of covariates selected for the propensity score model (in addition order;
    /// ascending for backward elimination).
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

    /// Index of the selected step (0 = the first model of the path).
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

    /// Number of candidate covariates.
    #[serde(default)]
    pub n_candidates: usize,

    /// Epsilon of the final full-data targeting step (logit scale for the logistic Q).
    #[serde(default)]
    pub fluctuation_coef: f64,

    /// `(min, max)` of the outcome when the logistic Q rescaled it to [0, 1]; `None` otherwise.
    #[serde(default)]
    pub outcome_range: Option<(f64, f64)>,
}

impl ResultNotes for CTmleResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
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
    #[serde(default)]
    pub penalty_factor: f64,
    /// The algorithm actually run.
    #[serde(default)]
    pub algorithm: String,
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
            self.n_selected, self.n_candidates
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
        writeln!(f, "  Algorithm:      {}", self.config.algorithm)?;
        writeln!(f, "  CV Folds:       {}", self.config.n_folds)?;
        writeln!(f, "  Stopping Rule:  {}", self.config.stopping_rule)?;
        writeln!(f, "  Selection:      {}", self.config.order)?;
        if self.config.cv_criterion == CVCriterion::PenRSS.to_string() {
            writeln!(
                f,
                "  CV Criterion:   {} (penalty factor {})",
                self.config.cv_criterion, self.config.penalty_factor
            )?;
        } else {
            writeln!(f, "  CV Criterion:   {}", self.config.cv_criterion)?;
        }
        writeln!(f, "  Outcome Model:  {}", self.config.q_model)?;
        if let Some((lo, hi)) = self.outcome_range {
            writeln!(
                f,
                "  Outcome:        rescaled from [{lo:.4}, {hi:.4}] to [0, 1]; criteria on that scale"
            )?;
        }
        writeln!(
            f,
            "  PS Bounds:      [{:.3}, {:.3}]",
            self.config.gbound.0, self.config.gbound.1
        )?;
        writeln!(f)?;

        writeln!(f, "Selection Path:")?;
        writeln!(
            f,
            "  Step | Covariates |  Criterion  |  Crit. SE  |    ATE    |   SE"
        )?;
        writeln!(
            f,
            "  -----|------------|-------------|------------|-----------|--------"
        )?;
        for (i, step) in self.selection_path.iter().enumerate() {
            let marker = if i == self.selected_step { " *" } else { "  " };
            writeln!(
                f,
                "  {:>4} | {:>10} | {:>11.4} | {:>10.4} | {:>9.4} | {:>6.4}{}",
                i,
                step.n_covariates,
                step.cv_criterion,
                step.cv_criterion_se,
                step.ate_estimate,
                step.ate_se,
                marker
            )?;
        }
        writeln!(f, "  (* = selected model)")?;
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
// Main C-TMLE Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run C-TMLE with full configuration.
///
/// Estimates the Average Treatment Effect (ATE) using Collaborative Targeted Maximum
/// Likelihood Estimation. C-TMLE uses cross-validation to select covariates for the
/// propensity score model in a "collaborative" way that minimizes finite-sample bias.
/// See the module documentation for the algorithm and the criteria.
///
/// # Arguments
/// * `dataset` - Dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable column
/// * `treatment_col` - Name of the binary treatment variable (coded 0/1)
/// * `covariate_cols` - Names of candidate covariate columns for propensity score selection
/// * `config` - C-TMLE configuration options
///
/// # Returns
/// `CTmleResult` containing ATE estimate, selected covariates, selection path, and diagnostics.
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
    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let a = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let mut covariate_data: Vec<Array1<f64>> = Vec::with_capacity(covariate_cols.len());
    for col in covariate_cols {
        let col_data = DesignMatrix::extract_column(dataset.df(), col)
            .map_err(|e| econ_error_from_design(e, dataset.df()))?;
        covariate_data.push(col_data);
    }
    let covariate_names: Vec<String> = covariate_cols.iter().map(|s| s.to_string()).collect();
    ctmle_core(
        y.view(),
        a.view(),
        &covariate_data,
        &covariate_names,
        (outcome_col, treatment_col),
        config,
    )
}

/// Run C-TMLE with default configuration.
///
/// Uses forward selection with CV minimum stopping rule.
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

/// Run C-TMLE on array data directly (the same engine as [`ctmle()`]; covariates are named
/// X1, X2, ...).
///
/// # Arguments
/// * `y` - Outcome vector (n,)
/// * `treatment` - Binary treatment indicator (n,), coded 0/1
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
    for (what, ok) in [
        ("y", y.iter().all(|v| v.is_finite())),
        ("treatment", treatment.iter().all(|v| v.is_finite())),
        ("covariates", covariates.iter().all(|v| v.is_finite())),
    ] {
        if !ok {
            return Err(EconError::InvalidSpecification {
                message: format!("{what} contains a missing or non-finite value"),
            });
        }
    }

    let covariate_data: Vec<Array1<f64>> =
        (0..p).map(|j| covariates.column(j).to_owned()).collect();
    let covariate_names: Vec<String> = (0..p).map(|j| format!("X{}", j + 1)).collect();

    ctmle_core(
        y.view(),
        treatment.view(),
        &covariate_data,
        &covariate_names,
        ("outcome", "treatment"),
        config,
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Checks the fold count, the iteration controls, the propensity bounds, the penalty and the
/// stop factor against the sample size before any work.
fn validate_ctmle_config(config: &CTmleConfig, n: usize) -> EconResult<()> {
    let max_folds = n / 2;
    if config.n_folds < 2 || config.n_folds > max_folds {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "n_folds must be between 2 and n/2 = {max_folds} for {n} observations (each fold needs at least 2); got {}",
                config.n_folds
            ),
        });
    }
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
    let (lo, hi) = config.gbound;
    if !(lo.is_finite() && hi.is_finite() && 0.0 < lo && lo < hi && hi < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "gbound must satisfy 0 < lower < upper < 1 (propensity truncation bounds); got ({lo}, {hi})"
            ),
        });
    }
    if !(config.penalty_factor.is_finite() && config.penalty_factor >= 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "penalty_factor must be finite and non-negative; got {}",
                config.penalty_factor
            ),
        });
    }
    if !(config.stop_factor.is_finite() && config.stop_factor > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "stop_factor must be finite and positive; got {}",
                config.stop_factor
            ),
        });
    }
    Ok(())
}

/// The effective prespecified list (empty means `0..p`); empty for Forward and Backward.
fn validate_order(order: &SelectionOrder, p: usize) -> EconResult<Vec<usize>> {
    let SelectionOrder::Prespecified(list) = order else {
        return Ok(Vec::new());
    };
    if list.is_empty() {
        return Ok((0..p).collect());
    }
    let mut seen = vec![false; p];
    for &j in list {
        if j >= p {
            return Err(EconError::InvalidSpecification {
                message: format!("order index {j} is out of range for {p} covariates"),
            });
        }
        if seen[j] {
            return Err(EconError::InvalidSpecification {
                message: format!("order lists covariate {j} twice"),
            });
        }
        seen[j] = true;
    }
    Ok(list.clone())
}

fn qmodel(m: CTmleQModel) -> QModel {
    match m {
        CTmleQModel::Linear => QModel::Linear,
        CTmleQModel::Logistic => QModel::Logistic,
    }
}

/// The intercept column, then `covariates[j]` for `j` in `set` order.
fn design_with_intercept(covariates: &[Array1<f64>], set: &[usize], n: usize) -> Array2<f64> {
    let mut x = Array2::ones((n, set.len() + 1));
    for (c, &j) in set.iter().enumerate() {
        x.column_mut(c + 1).assign(&covariates[j]);
    }
    x
}

/// Create V-fold cross-validation indices: observation `i` is in fold `i mod v`.
fn create_cv_folds(n: usize, v: usize) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = (0..v).map(|_| Vec::new()).collect();
    for i in 0..n {
        folds[i % v].push(i);
    }
    folds
}

/// Aggregated warnings of the many CV fits: (scope, text) -> count, scope -> number of fits,
/// scope -> number of fits that did not converge.
#[derive(Default)]
struct FitNotes {
    counts: BTreeMap<(&'static str, String), usize>,
    fits: BTreeMap<&'static str, usize>,
    not_converged: BTreeMap<&'static str, usize>,
}

impl FitNotes {
    fn record(&mut self, scope: &'static str, warnings: &[String], converged: bool) {
        *self.fits.entry(scope).or_default() += 1;
        for w in warnings {
            *self.counts.entry((scope, w.clone())).or_default() += 1;
        }
        if !converged {
            *self.not_converged.entry(scope).or_default() += 1;
        }
    }

    fn lines(&self, max_iter: usize) -> Vec<String> {
        let mut out = Vec::new();
        for ((scope, text), k) in &self.counts {
            let m = self.fits.get(scope).copied().unwrap_or(0);
            out.push(format!("{scope}: {text} ({k} of {m} fits)"));
        }
        for (scope, k) in &self.not_converged {
            let m = self.fits.get(scope).copied().unwrap_or(0);
            out.push(format!(
                "{scope}: did not converge within max_iter = {max_iter} ({k} of {m} fits)"
            ));
        }
        out
    }
}

/// Validation-fold predictions of one fold: rows, y, Q*(A), Q*(1), Q*(0), H on those rows.
#[allow(dead_code)]
struct FoldEval {
    rows: Vec<usize>,
    y: Vec<f64>,
    q_star_a: Vec<f64>,
    q_star_1: Vec<f64>,
    q_star_0: Vec<f64>,
    h: Vec<f64>,
}

/// The V-fold risk of one candidate g (CAUS-57).
struct CvRisk {
    criterion: f64,
    se: f64,
    rss: f64,
    #[allow(dead_code)]
    rss_se: f64,
    var_ic: f64,
    #[allow(dead_code)]
    var_ic_se: f64,
    #[allow(dead_code)]
    fold_rss: Vec<f64>,
    #[allow(dead_code)]
    fold_var_ic: Vec<f64>,
    #[allow(dead_code)]
    fold_epsilon: Vec<f64>,
    #[allow(dead_code)]
    folds: Vec<FoldEval>,
}

/// The full-data TMLE with one g.
struct FullTmle {
    ate: f64,
    se: f64,
    epsilon: f64,
    g: Array1<f64>,
    q_star_a: Array1<f64>,
    ic: Array1<f64>,
    g_fit: NuisanceFit,
    fl_warnings: Vec<String>,
    fl_converged: bool,
    fl_iterations: usize,
}

/// One fold of the V-fold risk: its training and validation rows, and the fold's Q (fitted on
/// the training rows) on the fitting scale of the fluctuation (`logit_q` for the logistic Q,
/// Q itself for the linear Q), computed once for all candidate g.
struct FoldCache {
    train_rows: Vec<usize>,
    y_train: Array1<f64>,
    a_train: Array1<f64>,
    /// Q(A,W) on the training rows (the fluctuation's offset).
    base_train: Array1<f64>,
    y_val: Array1<f64>,
    a_val: Array1<f64>,
    /// Q(A,W), Q(1,W), Q(0,W) on the validation rows.
    base_val: [Array1<f64>; 3],
}

/// Shared state of the V-fold risk: data, folds (i % V), the per-fold cache, the full-data Q,
/// the criterion, the penalty and the glm control.
struct CvSetup<'a> {
    y: ArrayView1<'a, f64>,
    a: ArrayView1<'a, f64>,
    covariates: &'a [Array1<f64>],
    names: &'a [String],
    folds: Vec<Vec<usize>>,
    cache: Vec<FoldCache>,
    q_full: QPredictions,
    q_full_fit: NuisanceFit,
    model: QModel,
    gbound: (f64, f64),
    criterion: CVCriterion,
    penalty: f64,
    ctl: GlmControl,
}

impl<'a> CvSetup<'a> {
    fn new(
        y: ArrayView1<'a, f64>,
        a: ArrayView1<'a, f64>,
        covariates: &'a [Array1<f64>],
        names: &'a [String],
        treatment: &str,
        config: &CTmleConfig,
        notes: &mut FitNotes,
    ) -> EconResult<CvSetup<'a>> {
        let n = y.len();
        let p = covariates.len();
        let v = config.n_folds;
        let model = qmodel(config.q_model);
        let ctl = GlmControl {
            epsilon: config.tolerance,
            max_iter: config.max_iter,
        };
        let folds = create_cv_folds(n, v);
        let train: Vec<Array1<f64>> = (0..v)
            .map(|f| Array1::from_iter((0..n).map(|i| if i % v != f { 1.0 } else { 0.0 })))
            .collect();
        let all: Vec<usize> = (0..p).collect();
        let w = design_with_intercept(covariates, &all, n);
        let mut qnames = Vec::with_capacity(p + 2);
        qnames.push("(Intercept)".to_string());
        qnames.extend(names.iter().cloned());
        qnames.push(treatment.to_string());
        let scale = |q: &Array1<f64>, rows: &[usize]| -> Array1<f64> {
            match model {
                QModel::Logistic => rows.iter().map(|&i| logit_q(q[i])).collect(),
                QModel::Linear => rows.iter().map(|&i| q[i]).collect(),
            }
        };
        let pick = |x: ArrayView1<'_, f64>, rows: &[usize]| -> Array1<f64> {
            rows.iter().map(|&i| x[i]).collect()
        };
        let mut cache = Vec::with_capacity(v);
        for (t, val) in train.iter().zip(&folds) {
            let (fit, q) = fit_outcome(w.view(), a, y, model, Some(t.view()), &qnames, ctl)?;
            notes.record("outcome model (Q), CV folds", &fit.warnings, fit.converged);
            let train_rows: Vec<usize> = (0..n).filter(|&i| t[i] > 0.0).collect();
            cache.push(FoldCache {
                y_train: pick(y, &train_rows),
                a_train: pick(a, &train_rows),
                base_train: scale(&q.qa, &train_rows),
                y_val: pick(y, val),
                a_val: pick(a, val),
                base_val: [scale(&q.qa, val), scale(&q.q1, val), scale(&q.q0, val)],
                train_rows,
            });
        }
        let (q_full_fit, q_full) = fit_outcome(w.view(), a, y, model, None, &qnames, ctl)?;
        Ok(CvSetup {
            y,
            a,
            covariates,
            names,
            folds,
            cache,
            q_full,
            q_full_fit,
            model,
            gbound: config.gbound,
            criterion: config.cv_criterion,
            penalty: config.penalty_factor,
            ctl,
        })
    }

    fn g_names(&self, set: &[usize]) -> Vec<String> {
        std::iter::once("(Intercept)".to_string())
            .chain(set.iter().map(|&j| self.names[j].clone()))
            .collect()
    }

    /// The CV risk of the g-model with covariates `set` (CAUS-57). For every fold v, the
    /// initial Q (the Q of the training folds, the honest initial estimate for fold v), g and
    /// epsilon are fitted on the training rows and the loss is evaluated on fold v. Unlike
    /// ctmle::ctmleDiscrete, which fits g on all rows, g is fitted on the training rows too.
    fn risk(&self, set: &[usize], notes: &mut FitNotes) -> EconResult<CvRisk> {
        let n = self.y.len();
        let v = self.folds.len();
        let xg = design_with_intercept(self.covariates, set, n);
        let gnames = self.g_names(set);
        let link = |x: f64| match self.model {
            QModel::Logistic => plogis(x),
            QModel::Linear => x,
        };
        let clever = |a: f64, g: f64| a / g - (1.0 - a) / (1.0 - g);
        let mut fold_rss = Vec::with_capacity(v);
        let mut fold_var_ic = Vec::with_capacity(v);
        let mut fold_epsilon = Vec::with_capacity(v);
        let mut evals = Vec::with_capacity(v);
        for (f, rows) in self.folds.iter().enumerate() {
            resource::checkpoint()?;
            let fc = &self.cache[f];
            let x_train = xg.select(Axis(0), &fc.train_rows);
            let (gfit, g_raw) =
                fit_treatment(x_train.view(), fc.a_train.view(), None, &gnames, self.ctl)?;
            notes.record(
                "propensity model (g), CV fits",
                &gfit.warnings,
                gfit.converged,
            );
            let g_train = bound_scores(&g_raw, self.gbound).0;
            let h_train: Array1<f64> =
                Array1::from_iter((0..g_train.len()).map(|r| clever(fc.a_train[r], g_train[r])));
            let fl = fit_epsilon(
                fc.y_train.view(),
                h_train.insert_axis(Axis(1)).view(),
                fc.base_train.view(),
                self.model,
                None,
                self.ctl,
            )?;
            notes.record("fluctuation, CV fits", &fl.warnings, fl.converged);
            let e = fl.coefficients[0];
            let g_val = bound_scores(
                &xg.select(Axis(0), rows).dot(&gfit.zeroed()).mapv(plogis),
                self.gbound,
            )
            .0;
            let [qa, q1, q0] = &fc.base_val;
            let m = rows.len();
            let h: Vec<f64> = (0..m).map(|r| clever(fc.a_val[r], g_val[r])).collect();
            let ev = FoldEval {
                rows: rows.clone(),
                y: fc.y_val.to_vec(),
                q_star_a: (0..m).map(|r| link(qa[r] + e * h[r])).collect(),
                q_star_1: (0..m).map(|r| link(q1[r] + e / g_val[r])).collect(),
                q_star_0: (0..m).map(|r| link(q0[r] - e / (1.0 - g_val[r]))).collect(),
                h,
            };
            let rss: f64 = (0..m).map(|r| (ev.y[r] - ev.q_star_a[r]).powi(2)).sum();
            let ic: Vec<f64> = (0..m)
                .map(|r| ev.h[r] * (ev.y[r] - ev.q_star_a[r]) + ev.q_star_1[r] - ev.q_star_0[r])
                .collect();
            fold_rss.push(rss);
            fold_var_ic.push(sample_variance(&ic));
            fold_epsilon.push(e);
            evals.push(ev);
        }
        let vf = v as f64;
        let rss: f64 = fold_rss.iter().sum();
        let rss_se = vf.sqrt() * sample_variance(&fold_rss).sqrt();
        let var_ic = fold_var_ic.iter().sum::<f64>() / vf;
        let var_ic_se = (sample_variance(&fold_var_ic) / vf).sqrt();
        let k = set.len() as f64;
        let (criterion, se) = match self.criterion {
            CVCriterion::RSS => (rss, rss_se),
            CVCriterion::VarIC => (var_ic, var_ic_se),
            CVCriterion::PenRSS => (rss + self.penalty * (n as f64).ln() * k, rss_se),
        };
        Ok(CvRisk {
            criterion,
            se,
            rss,
            rss_se,
            var_ic,
            var_ic_se,
            fold_rss,
            fold_var_ic,
            fold_epsilon,
            folds: evals,
        })
    }

    /// The full-data TMLE with g on `set` (path and final estimates); on the fitting scale.
    fn full_tmle(&self, set: &[usize]) -> EconResult<FullTmle> {
        let n = self.y.len();
        let xg = design_with_intercept(self.covariates, set, n);
        let (g_fit, graw) = fit_treatment(xg.view(), self.a, None, &self.g_names(set), self.ctl)?;
        let g = bound_scores(&graw, self.gbound).0;
        let fl = fluctuate(
            self.y,
            self.a,
            &self.q_full,
            g.view(),
            self.model,
            None,
            self.ctl,
            FluctuationForm::OneDim,
        )?;
        let (ate, ic) = ate_and_ic(self.y, &fl);
        let se = (sample_variance(&ic.to_vec()) / n as f64).sqrt();
        Ok(FullTmle {
            ate,
            se,
            epsilon: fl.epsilon,
            g,
            q_star_a: fl.q_star.qa,
            ic,
            g_fit,
            fl_warnings: fl.warnings,
            fl_converged: fl.converged,
            fl_iterations: fl.iterations,
        })
    }
}

/// One recorded step of the selection path.
struct PathStep {
    moved: Option<usize>,
    set: Vec<usize>,
    risk: CvRisk,
    est: FullTmle,
}

impl PathStep {
    fn new(
        setup: &CvSetup<'_>,
        moved: Option<usize>,
        set: Vec<usize>,
        risk: CvRisk,
        notes: &mut FitNotes,
    ) -> EconResult<PathStep> {
        let est = setup.full_tmle(&set)?;
        notes.record(
            "propensity model (g), path fits",
            &est.g_fit.warnings,
            est.g_fit.converged,
        );
        notes.record("fluctuation, path fits", &est.fl_warnings, est.fl_converged);
        Ok(PathStep {
            moved,
            set,
            risk,
            est,
        })
    }
}

/// The selection path, one loop for every order.
fn select_path(
    setup: &CvSetup<'_>,
    config: &CTmleConfig,
    order: &[usize],
    p: usize,
    notes: &mut FitNotes,
) -> EconResult<Vec<PathStep>> {
    let backward = config.order == SelectionOrder::Backward;
    let prespecified = matches!(config.order, SelectionOrder::Prespecified(_));
    let cap = config.max_covariates.unwrap_or(p).min(p);
    let max_k = match config.stopping_rule {
        StoppingRule::MaxCovariates(k) => Some(k),
        _ => None,
    };
    let mut current: Vec<usize> = if backward {
        (0..p).collect()
    } else {
        Vec::new()
    };
    let risk0 = setup.risk(&current, notes)?;
    let mut best = risk0.criterion;
    let mut path = vec![PathStep::new(setup, None, current.clone(), risk0, notes)?];
    loop {
        resource::checkpoint()?;
        let stop = if backward {
            current.is_empty() || max_k.is_some_and(|k| current.len() <= k)
        } else {
            current.len() >= cap || max_k.is_some_and(|k| current.len() >= k)
        };
        if stop {
            break;
        }
        let candidates: Vec<usize> = if backward {
            current.clone()
        } else if prespecified {
            match order.get(path.len() - 1) {
                Some(&j) => vec![j],
                None => break,
            }
        } else {
            (0..p).filter(|j| !current.contains(j)).collect()
        };
        if candidates.is_empty() {
            break;
        }
        let mut chosen: Option<(usize, Vec<usize>, CvRisk)> = None;
        for &j in &candidates {
            resource::checkpoint()?;
            let set: Vec<usize> = if backward {
                current.iter().copied().filter(|&c| c != j).collect()
            } else {
                let mut s = current.clone();
                s.push(j);
                s
            };
            let risk = setup.risk(&set, notes)?;
            let better = match &chosen {
                None => true,
                Some((_, _, r)) => risk.criterion < r.criterion,
            };
            if better {
                chosen = Some((j, set, risk));
            }
        }
        let Some((j, set, risk)) = chosen else {
            break;
        };
        if risk.criterion > config.stop_factor * best {
            if config.verbose {
                eprintln!(
                    "C-TMLE: Early stopping at step {} (CV criterion {} > {} * {})",
                    path.len(),
                    risk.criterion,
                    config.stop_factor,
                    best
                );
            }
            break;
        }
        best = best.min(risk.criterion);
        current = set;
        let step = PathStep::new(setup, Some(j), current.clone(), risk, notes)?;
        if config.verbose {
            eprintln!(
                "C-TMLE: Step {} {} {} (CV: {:.4}, ATE: {:.4})",
                path.len(),
                if backward { "removed" } else { "added" },
                setup.names[j],
                step.risk.criterion,
                step.est.ate
            );
        }
        path.push(step);
    }
    Ok(path)
}

/// The selected step: among the steps with at most `cap` covariates, the first minimum of the
/// criterion (CVMinimum), the step with the fewest covariates within one SE of it (OneSE), or
/// the last step (MaxCovariates).
fn choose_step(path: &[PathStep], rule: StoppingRule, cap: usize) -> EconResult<usize> {
    for (i, s) in path.iter().enumerate() {
        if !s.risk.criterion.is_finite() {
            return Err(EconError::Computation(format!(
                "C-TMLE criterion is not finite at step {i}"
            )));
        }
    }
    let eligible: Vec<usize> = (0..path.len())
        .filter(|&i| path[i].set.len() <= cap)
        .collect();
    let Some(&first) = eligible.first() else {
        let k = path.last().map(|s| s.set.len()).unwrap_or(0);
        return Err(EconError::InvalidSpecification {
            message: format!(
                "max_covariates {cap} cannot be honoured: backward elimination stopped early \
                 (stop_factor) at {k} covariates"
            ),
        });
    };
    let crit = |i: usize| path[i].risk.criterion;
    let min_idx = eligible
        .iter()
        .copied()
        .fold(first, |b, i| if crit(i) < crit(b) { i } else { b });
    Ok(match rule {
        StoppingRule::CVMinimum => min_idx,
        StoppingRule::OneSE => {
            let thr = crit(min_idx) + path[min_idx].risk.se;
            eligible
                .iter()
                .copied()
                .filter(|&i| crit(i) <= thr)
                .fold(None::<usize>, |b, i| match b {
                    Some(b) if path[b].set.len() <= path[i].set.len() => Some(b),
                    _ => Some(i),
                })
                .unwrap_or(min_idx)
        }
        StoppingRule::MaxCovariates(_) => {
            let last = path.len() - 1;
            if path[last].set.len() <= cap {
                last
            } else {
                *eligible.last().unwrap_or(&first)
            }
        }
    })
}

/// The C-TMLE engine of [`ctmle()`] and [`ctmle_arrays`]; `labels` names the outcome and
/// the treatment in messages.
fn ctmle_core(
    y: ArrayView1<'_, f64>,
    a: ArrayView1<'_, f64>,
    covariate_data: &[Array1<f64>],
    covariate_names: &[String],
    labels: (&str, &str),
    config: CTmleConfig,
) -> EconResult<CTmleResult> {
    let mut warnings = Vec::new();
    let n = y.len();
    let p = covariate_data.len();
    validate_ctmle_config(&config, n)?;
    let order = validate_order(&config.order, p)?;
    let (n_treated, n_control) = check_binary_treatment(a, labels.1)?;
    let model = qmodel(config.q_model);
    let scale = outcome_scale(y, model, labels.0)?;
    let ys = scale.apply(&y.to_owned());
    warnings.extend(scale.note("C-TMLE", labels.0));

    // Need at least 2*n_folds observations per group for CV
    let min_per_fold = 2;
    if n_treated < config.n_folds * min_per_fold || n_control < config.n_folds * min_per_fold {
        warnings.push(format!(
            "Small sample size for {}-fold CV. Consider reducing n_folds.",
            config.n_folds
        ));
    }

    let mut notes = FitNotes::default();
    let setup = CvSetup::new(
        ys.view(),
        a.view(),
        covariate_data,
        covariate_names,
        labels.1,
        &config,
        &mut notes,
    )?;
    let path = select_path(&setup, &config, &order, p, &mut notes)?;
    let cap = config.max_covariates.unwrap_or(p).min(p);
    let selected_step = choose_step(&path, config.stopping_rule, cap)?;

    let backward = config.order == SelectionOrder::Backward;
    let selected_covariates: Vec<usize> = if backward {
        path[selected_step].set.clone()
    } else {
        path[1..=selected_step]
            .iter()
            .filter_map(|s| s.moved)
            .collect()
    };
    let selected_covariate_names: Vec<String> = selected_covariates
        .iter()
        .map(|&idx| covariate_names[idx].clone())
        .collect();

    let fin = &path[selected_step].est;
    let ctl = setup.ctl;
    warnings.extend(prefixed("outcome model (Q)", &setup.q_full_fit.warnings));
    warnings.extend(prefixed("final propensity model (g)", &fin.g_fit.warnings));
    warnings.extend(prefixed("final targeting step", &fin.fl_warnings));
    for (converged, iterations, method) in [
        (
            setup.q_full_fit.converged,
            setup.q_full_fit.iterations,
            "C-TMLE outcome model (Q)",
        ),
        (
            fin.g_fit.converged,
            fin.g_fit.iterations,
            "C-TMLE final propensity model (g)",
        ),
        (
            fin.fl_converged,
            fin.fl_iterations,
            "C-TMLE final targeting step",
        ),
    ] {
        warnings.extend(glm_convergence(converged, iterations, ctl).note(method));
    }
    warnings.extend(notes.lines(config.max_iter));

    let width = scale.width;
    let ate = width * fin.ate;
    let se = width * fin.se;
    let (z_stat, p_value, ci_lower, ci_upper) = wald95(ate, se)?;
    let significance = SignificanceLevel::from_p_value(p_value);

    let selection_path: Vec<SelectionStep> = path
        .iter()
        .map(|s| SelectionStep {
            covariate_index: s.moved,
            covariate_name: Some(match s.moved {
                Some(j) => covariate_names[j].clone(),
                None if backward => "(all candidates)".to_string(),
                None => "(intercept)".to_string(),
            }),
            cv_criterion: s.risk.criterion,
            cv_criterion_se: s.risk.se,
            ate_estimate: width * s.est.ate,
            ate_se: width * s.est.se,
            n_covariates: s.set.len(),
            cv_rss: s.risk.rss,
            cv_var_ic: s.risk.var_ic,
        })
        .collect();
    let cv_risk: Vec<f64> = path.iter().map(|s| s.risk.criterion).collect();

    let order_word = match config.order {
        SelectionOrder::Forward => "forward",
        SelectionOrder::Backward => "backward",
        SelectionOrder::Prespecified(_) => "prespecified",
    };
    Ok(CTmleResult {
        ate,
        se,
        ci_lower,
        ci_upper,
        p_value,
        significance,
        z_stat,
        n_selected: selected_covariates.len(),
        selected_covariates,
        selected_covariate_names,
        selection_path,
        cv_risk,
        influence_curve: fin.ic.iter().map(|v| width * v).collect(),
        selected_step,
        propensity_scores: fin.g.to_vec(),
        targeted_outcome: fin.q_star_a.iter().map(|&v| scale.unscale(v)).collect(),
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
            penalty_factor: config.penalty_factor,
            algorithm: format!(
                "greedy {order_word} selection of g, fixed initial Q, {}-fold CV (Q, g and \
                 epsilon fitted on the training folds)",
                config.n_folds
            ),
        },
        warnings,
        n_candidates: p,
        fluctuation_coef: fin.epsilon,
        outcome_range: scale.range(),
    })
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

    fn small_config() -> CTmleConfig {
        CTmleConfig {
            n_folds: 3,
            max_covariates: Some(2),
            q_model: CTmleQModel::Linear,
            ..Default::default()
        }
    }

    #[test]
    fn test_ctmle_rejects_out_of_range_folds() {
        let ds = create_ctmle_test_dataset(); // n = 40, so 2..=20 is valid
        for v in [0, 1, 21, usize::MAX] {
            let cfg = CTmleConfig {
                n_folds: v,
                ..small_config()
            };
            expect_invalid_spec(
                ctmle(&ds, "y", "treatment", &["w1", "w2"], cfg).unwrap_err(),
                "n_folds",
            );
        }
        assert!(
            ctmle(
                &ds,
                "y",
                "treatment",
                &["w1", "w2"],
                CTmleConfig {
                    n_folds: 20,
                    ..small_config()
                }
            )
            .is_ok()
        );
        // the array entry point validates too (same 40 rows as arrays)
        let y = DesignMatrix::extract_column(ds.df(), "y").unwrap();
        let a = DesignMatrix::extract_column(ds.df(), "treatment").unwrap();
        let w = DesignMatrix::from_dataframe(ds.df(), &["w1", "w2"], false)
            .unwrap()
            .data;
        for v in [0, 21, usize::MAX] {
            let cfg = CTmleConfig {
                n_folds: v,
                ..small_config()
            };
            expect_invalid_spec(
                ctmle_arrays(&y.view(), &a.view(), &w.view(), cfg).unwrap_err(),
                "n_folds",
            );
        }
    }

    #[test]
    fn test_ctmle_rejects_bad_controls() {
        let ds = create_ctmle_test_dataset();
        let cases = [
            (
                CTmleConfig {
                    max_iter: 0,
                    ..small_config()
                },
                "max_iter",
            ),
            (
                CTmleConfig {
                    tolerance: 0.0,
                    ..small_config()
                },
                "tolerance",
            ),
            (
                CTmleConfig {
                    tolerance: f64::NAN,
                    ..small_config()
                },
                "tolerance",
            ),
            (
                CTmleConfig {
                    gbound: (0.9, 0.1),
                    ..small_config()
                },
                "gbound",
            ),
            (
                CTmleConfig {
                    gbound: (0.0, 1.0),
                    ..small_config()
                },
                "gbound",
            ),
            (
                CTmleConfig {
                    gbound: (f64::NAN, 0.9),
                    ..small_config()
                },
                "gbound",
            ),
        ];
        for (cfg, needle) in cases {
            expect_invalid_spec(
                ctmle(&ds, "y", "treatment", &["w1", "w2"], cfg).unwrap_err(),
                needle,
            );
        }
    }

    #[test]
    fn test_ctmle_honours_cancellation() {
        let ds = create_ctmle_test_dataset();
        let r = resource::with_budget(cancelled_budget(), || {
            ctmle(&ds, "y", "treatment", &["w1", "w2"], small_config())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
        // the logistic outcome model's IRLS loop is a checkpoint too
        let r = resource::with_budget(cancelled_budget(), || {
            ctmle(
                &ds,
                "y",
                "treatment",
                &["w1", "w2"],
                CTmleConfig {
                    q_model: CTmleQModel::Logistic,
                    ..small_config()
                },
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
    }

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

        // The path stops at 2 covariates and selects them. On these 40 rows w1 and w2
        // (almost) separate the treatment, so g sits at its bounds and the TMLE moves far
        // from the Q estimate: R's lm(y ~ w1 + w2 + w3 + w4 + a), glm(a ~ w1 + w2, binomial),
        // g bounded to [0.05, 0.95], lm(y ~ -1 + H, offset = Q(A,W)) gives 0.04024225 (quoted
        // value, not an R case). The old range 0.1..1 matched the in-sample risk of CAUS-57.
        let mut selected = result.selected_covariates.clone();
        selected.sort_unstable();
        assert_eq!(selected, vec![0, 1]);
        assert!((result.ate - 0.04024225).abs() < 1e-7, "ATE {}", result.ate);
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
        // w3, then w1, then w2
        let config = CTmleConfig {
            n_folds: 3,
            order: SelectionOrder::Prespecified(vec![2, 0, 1]),
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
        let moved: Vec<usize> = result.selection_path[1..]
            .iter()
            .map(|s| s.covariate_index.unwrap())
            .collect();
        assert_eq!(moved, vec![2, 0, 1]);
        assert_eq!(result.selection_path.len(), 4);
        assert_eq!(result.selected_covariates, vec![2, 0, 1]);
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
            q_model: CTmleQModel::Linear, // y is continuous
            ..Default::default()
        };

        let result =
            ctmle_arrays(&y.view(), &treatment.view(), &covariates.view(), config).unwrap();

        assert_eq!(result.n_obs, 20);
        assert!(result.ate.is_finite());
        assert!(result.se > 0.0);
    }

    // =========================================================================
    // Validation tests (test_validate_* prefix)
    // =========================================================================

    /// Create a larger confounded dataset for validation with known positive ATE.
    ///
    /// DGP (continuous outcome for stability):
    ///   W1 ~ Uniform(0, 1), W2 ~ Uniform(0, 1)
    ///   A | W ~ Bernoulli(expit(0.5*W1 + 0.3*W2 - 0.4))
    ///   Y = ATE*A + 0.8*W1 + 0.4*W2 + noise
    ///
    /// True ATE = 2.0
    fn create_validate_ctmle_dataset() -> (Dataset, f64) {
        let true_ate = 2.0;
        let n = 200;

        // Deterministic pseudo-random using modular arithmetic
        let w1: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0 + 0.005)
            .collect();
        let w2: Vec<f64> = (0..n)
            .map(|i| ((i * 13 + 7) % 100) as f64 / 100.0 + 0.005)
            .collect();

        let treatment: Vec<f64> = (0..n)
            .map(|i| {
                let ps = 1.0 / (1.0 + (-0.5 * w1[i] - 0.3 * w2[i] + 0.4).exp());
                let threshold = ((i * 37 + 11) % 100) as f64 / 100.0;
                if threshold < ps { 1.0 } else { 0.0 }
            })
            .collect();

        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = ((i * 31 + 5) % 100) as f64 / 500.0 - 0.1;
                true_ate * treatment[i] + 0.8 * w1[i] + 0.4 * w2[i] + noise
            })
            .collect();

        let df = df! {
            "y" => y,
            "treatment" => treatment,
            "w1" => w1,
            "w2" => w2,
        }
        .unwrap();

        (Dataset::new(df), true_ate)
    }

    fn dgp_config() -> CTmleConfig {
        CTmleConfig {
            n_folds: 3,
            max_covariates: Some(2),
            stopping_rule: StoppingRule::MaxCovariates(2),
            q_model: CTmleQModel::Linear,
            ..Default::default()
        }
    }

    /// The C-TMLE ATE is in the range of the DGP's effect (not an R comparison).
    #[test]
    fn test_ctmle_ate_recovers_dgp() {
        let (dataset, true_ate) = create_validate_ctmle_dataset();
        let result = ctmle(&dataset, "y", "treatment", &["w1", "w2"], dgp_config()).unwrap();
        assert!(
            result.ate > 0.5 && result.ate < 4.0,
            "ATE = {:.4} (true = {true_ate})",
            result.ate
        );
        assert!(result.se > 0.0 && result.se.is_finite());
        assert!(result.ci_lower < result.ate && result.ate < result.ci_upper);
    }

    #[test]
    fn test_ctmle_selection_path_properties() {
        let (dataset, _) = create_validate_ctmle_dataset();
        let result = ctmle(&dataset, "y", "treatment", &["w1", "w2"], dgp_config()).unwrap();
        assert!(result.selection_path[0].covariate_index.is_none());
        assert_eq!(result.selection_path[0].n_covariates, 0);
        for i in 1..result.selection_path.len() {
            assert_eq!(result.selection_path[i].n_covariates, i);
            assert!(result.selection_path[i].covariate_index.is_some());
        }
        for step in &result.selection_path {
            assert!(step.cv_criterion >= 0.0 && step.cv_rss >= 0.0 && step.cv_var_ic >= 0.0);
        }
        assert_eq!(result.cv_risk.len(), result.selection_path.len());
    }

    #[test]
    fn test_ctmle_se_is_ic_variance() {
        let (dataset, _) = create_validate_ctmle_dataset();
        let r = ctmle(&dataset, "y", "treatment", &["w1", "w2"], dgp_config()).unwrap();
        assert_eq!(r.influence_curve.len(), r.n_obs);
        let n = r.n_obs as f64;
        let mean: f64 = r.influence_curve.iter().sum::<f64>() / n;
        let var: f64 = r
            .influence_curve
            .iter()
            .map(|ic| (ic - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        assert!((r.se - (var / n).sqrt()).abs() <= 1e-12 * r.se);
    }

    #[test]
    fn test_ctmle_propensity_bounds() {
        let (dataset, _) = create_validate_ctmle_dataset();
        let gbound = (0.05, 0.95);
        let config = CTmleConfig {
            gbound,
            ..dgp_config()
        };
        let result = ctmle(&dataset, "y", "treatment", &["w1", "w2"], config).unwrap();
        for &ps in &result.propensity_scores {
            assert!((gbound.0..=gbound.1).contains(&ps), "{ps}");
        }
        assert_eq!(result.n_treated + result.n_control, result.n_obs);
    }

    // ── T3-WP29: the R reference case ctmle_cv (validation/r/cases/causal_stdreg.R) ──

    const W: [&str; 4] = ["w1", "w2", "w3", "w4"];

    /// The case's dataset read directly, for the property tests that compare no R value.
    fn ctmle_sim() -> Dataset {
        crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/ctmle_sim.csv",
        )
    }

    fn col(ds: &Dataset, name: &str) -> Array1<f64> {
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

    fn w_matrix(ds: &Dataset) -> Array2<f64> {
        let cols: Vec<Array1<f64>> = W.iter().map(|w| col(ds, w)).collect();
        Array2::from_shape_fn((cols[0].len(), 4), |(i, j)| cols[j][i])
    }

    fn tight() -> CTmleConfig {
        // the case's glm.control(epsilon = 1e-14)
        CTmleConfig {
            tolerance: 1e-14,
            ..Default::default()
        }
    }

    fn mask(s: &[usize]) -> Vec<f64> {
        (0..4)
            .map(|j| if s.contains(&j) { 1.0 } else { 0.0 })
            .collect()
    }

    /// Every number the case records for one run (`prefix`), and its one-SE selection.
    fn check_run(
        case: &p2a_validation::Case,
        prefix: &str,
        r: &CTmleResult,
        onese: Option<&CTmleResult>,
    ) {
        let t = Tol::iterative(1e-8); // both sides at glm epsilon 1e-14
        let k = |s: &str| format!("{prefix}_{s}");
        let st =
            |f: fn(&SelectionStep) -> f64| r.selection_path.iter().map(f).collect::<Vec<f64>>();
        let moved: Vec<f64> = r.selection_path[1..]
            .iter()
            .map(|s| s.covariate_index.unwrap() as f64)
            .collect();
        let mut c = Check::new(case);
        c.vector(&k("path_crit"), &st(|s| s.cv_criterion), t)
            .vector(&k("path_se"), &st(|s| s.cv_criterion_se), t)
            .vector(&k("path_rss"), &st(|s| s.cv_rss), t)
            .vector(&k("path_varic"), &st(|s| s.cv_var_ic), t)
            .vector(
                &k("path_ncov"),
                &st(|s| s.n_covariates as f64),
                Tol::exact(),
            )
            .vector(&k("moved"), &moved, Tol::exact())
            .scalar(&k("selected_step"), r.selected_step as f64, Tol::exact())
            .vector(
                &k("selected_mask"),
                &mask(&r.selected_covariates),
                Tol::exact(),
            )
            .scalar(&k("ate"), r.ate, t)
            .scalar(&k("se"), r.se, t)
            .scalar(&k("epsilon"), r.fluctuation_coef, t)
            .scalar(&k("ci_lower"), r.ci_lower, t)
            .scalar(&k("ci_upper"), r.ci_upper, t)
            .scalar(&k("p"), r.p_value, Tol::p_value(300))
            .vector(&k("path_ate"), &st(|s| s.ate_estimate), t)
            .vector(&k("path_ate_se"), &st(|s| s.ate_se), t);
        if let Some(o) = onese {
            c.scalar(&k("onese_step"), o.selected_step as f64, Tol::exact())
                .vector(
                    &k("onese_selected_mask"),
                    &mask(&o.selected_covariates),
                    Tol::exact(),
                )
                .scalar(&k("onese_ate"), o.ate, t);
        }
        c.finish();
    }

    fn run(ds: &Dataset, y: &str, c: CTmleConfig) -> (CTmleResult, CTmleResult) {
        let r = ctmle(ds, y, "a", &W, c.clone()).unwrap();
        let o = ctmle(
            ds,
            y,
            "a",
            &W,
            CTmleConfig {
                stopping_rule: StoppingRule::OneSE,
                ..c
            },
        )
        .unwrap();
        (r, o)
    }

    #[test]
    fn test_validate_ctmle_prespecified_order_in_arrays_path() {
        let (case, ds) = crate::test_support::load_case("ctmle_cv");
        let (y, a, x) = (col(&ds, "yb"), col(&ds, "a"), w_matrix(&ds));
        let mut c = tight();
        c.order = SelectionOrder::Prespecified(vec![2, 0]);
        let r = ctmle_arrays(&y.view(), &a.view(), &x.view(), c.clone()).unwrap();
        let moved: Vec<usize> = r.selection_path[1..]
            .iter()
            .map(|s| s.covariate_index.unwrap())
            .collect();
        assert_eq!(moved, vec![2, 0]);
        assert_eq!(r.selection_path.len(), 3); // no fallback to forward once the list is used up
        assert_eq!(r.config.order, "Prespecified([2, 0])");
        assert!(
            r.warnings
                .iter()
                .all(|w| !w.contains("not yet implemented"))
        );
        let o = ctmle_arrays(
            &y.view(),
            &a.view(),
            &x.view(),
            CTmleConfig {
                stopping_rule: StoppingRule::OneSE,
                ..c.clone()
            },
        )
        .unwrap();
        check_run(&case, "logit_pre31", &r, Some(&o));
        // the dataset entry point: same engine, same numbers
        let d = ctmle(&ds, "yb", "a", &W, c.clone()).unwrap();
        assert_eq!(
            (
                d.ate,
                d.se,
                d.cv_risk.clone(),
                d.selected_covariates.clone()
            ),
            (
                r.ate,
                r.se,
                r.cv_risk.clone(),
                r.selected_covariates.clone()
            )
        );
        c.order = SelectionOrder::Backward;
        let b = ctmle_arrays(&y.view(), &a.view(), &x.view(), c.clone()).unwrap();
        assert_eq!(
            b.selection_path
                .iter()
                .map(|s| s.n_covariates)
                .collect::<Vec<_>>(),
            vec![4, 3, 2, 1, 0]
        );
        assert_eq!(b.selection_path[0].covariate_index, None);
        let bo = ctmle_arrays(
            &y.view(),
            &a.view(),
            &x.view(),
            CTmleConfig {
                stopping_rule: StoppingRule::OneSE,
                ..c.clone()
            },
        )
        .unwrap();
        check_run(&case, "logit_backward", &b, Some(&bo));
        c.order = SelectionOrder::Prespecified(vec![]); // the listed order (ctmle's preOrder default)
        let l = ctmle_arrays(&y.view(), &a.view(), &x.view(), c).unwrap();
        assert_eq!(
            l.selection_path[1..]
                .iter()
                .map(|s| s.covariate_index.unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(l.config.order, "Prespecified (listed order)");
    }

    #[test]
    fn test_ctmle_prespecified_order_is_validated() {
        let ds = ctmle_sim();
        for (list, needle) in [
            (vec![4], "order index 4 is out of range for 4 covariates"),
            (vec![1, 1], "order lists covariate 1 twice"),
        ] {
            let c = CTmleConfig {
                order: SelectionOrder::Prespecified(list),
                ..Default::default()
            };
            expect_invalid_spec(ctmle(&ds, "yb", "a", &W, c).unwrap_err(), needle);
        }
    }

    #[test]
    fn test_ctmle_max_covariates_caps_every_order() {
        let ds = ctmle_sim();
        let base = CTmleConfig {
            max_covariates: Some(2),
            ..Default::default()
        };
        let f = ctmle(&ds, "yb", "a", &W, base.clone()).unwrap();
        assert_eq!(f.selection_path.len(), 3);
        assert!(f.n_selected <= 2);
        let b = ctmle(
            &ds,
            "yb",
            "a",
            &W,
            CTmleConfig {
                order: SelectionOrder::Backward,
                ..base.clone()
            },
        )
        .unwrap();
        assert_eq!(b.selection_path.len(), 5);
        assert!(b.n_selected <= 2, "{}", b.n_selected);
        assert_eq!(b.n_selected, b.selected_covariates.len());
        let m = ctmle(
            &ds,
            "yb",
            "a",
            &W,
            CTmleConfig {
                order: SelectionOrder::Backward,
                stopping_rule: StoppingRule::MaxCovariates(2),
                ..base
            },
        )
        .unwrap();
        assert_eq!(m.selected_step, m.selection_path.len() - 1);
        assert_eq!(m.selection_path[m.selected_step].n_covariates, 2);
        assert_eq!(m.n_selected, 2);
    }

    #[test]
    fn test_ctmle_n_folds_checked_in_core() {
        let ds = ctmle_sim();
        let (y, a, x) = (col(&ds, "yb"), col(&ds, "a"), w_matrix(&ds));
        for v in [0, 1, 151] {
            let c = CTmleConfig {
                n_folds: v,
                ..Default::default()
            };
            expect_invalid_spec(ctmle(&ds, "yb", "a", &W, c.clone()).unwrap_err(), "n_folds");
            expect_invalid_spec(
                ctmle_arrays(&y.view(), &a.view(), &x.view(), c).unwrap_err(),
                "n_folds",
            );
        }
    }

    #[test]
    fn test_ctmle_refuses_non_binary_treatment() {
        let ds = ctmle_sim();
        let (y, x) = (col(&ds, "yb"), w_matrix(&ds));
        let a2 = col(&ds, "a").mapv(|v| 2.0 * v);
        expect_invalid_spec(
            ctmle_arrays(&y.view(), &a2.view(), &x.view(), CTmleConfig::default()).unwrap_err(),
            "must be coded 0/1",
        );
    }

    #[test]
    fn test_ctmle_logistic_q_rescales_continuous_outcome() {
        // CAUS-45 (ctmle() and ctmle_core). Before: the logistic Q on yc diverged (ATE +-1).
        // Now Y is rescaled once on the full data (the same lower/width in every fold);
        // criteria on the fitting scale; estimates scaled back.
        let ds = ctmle_sim();
        let r = ctmle(&ds, "yc", "a", &W, tight()).unwrap();
        let y = col(&ds, "yc");
        let lo = y.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(r.outcome_range, Some((lo, hi)));
        let ys: Vec<f64> = y.iter().map(|v| (v - lo) / (hi - lo)).collect();
        let s = ctmle(&with_col(&ds, "ycs", ys), "ycs", "a", &W, tight()).unwrap();
        assert!(s.outcome_range.is_none());
        assert_eq!(r.selected_step, s.selected_step);
        assert_eq!(r.selected_covariates, s.selected_covariates);
        let t = Tol::new(1e-12, 1e-15);
        assert!(t.allows(r.ate, (hi - lo) * s.ate) && t.allows(r.se, (hi - lo) * s.se));
        for (a, b) in r.selection_path.iter().zip(&s.selection_path) {
            assert!(t.allows(a.cv_criterion, b.cv_criterion)); // on the fitting scale
            assert!(t.allows(a.ate_estimate, (hi - lo) * b.ate_estimate));
        }
        assert!(
            r.warnings.iter().any(|w| w.contains("rescaled from [")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_validate_ctmle_cv_risk_uses_training_fluctuation() {
        let (case, ds) = crate::test_support::load_case("ctmle_cv");
        let (y, a) = (col(&ds, "yb"), col(&ds, "a"));
        let covs: Vec<Array1<f64>> = W.iter().map(|w| col(&ds, w)).collect();
        let names: Vec<String> = W.iter().map(|s| s.to_string()).collect();
        let cfg = CTmleConfig {
            cv_criterion: CVCriterion::PenRSS,
            penalty_factor: 2.0,
            ..tight()
        };
        let mut notes = FitNotes::default();
        let setup = CvSetup::new(y.view(), a.view(), &covs, &names, "a", &cfg, &mut notes).unwrap();
        let risk = setup.risk(&[0, 2], &mut notes).unwrap();
        let t = Tol::iterative(1e-8); // both sides at glm epsilon 1e-14
        Check::new(&case)
            .vector("cv13_fold_rss", &risk.fold_rss, t)
            .vector("cv13_fold_varic", &risk.fold_var_ic, t)
            .vector("cv13_fold_epsilon", &risk.fold_epsilon, t)
            .scalar("cv13_rss", risk.rss, t)
            .scalar("cv13_rss_se", risk.rss_se, t)
            .scalar("cv13_varic", risk.var_ic, t)
            .scalar("cv13_varic_se", risk.var_ic_se, t)
            .scalar("cv13_penrss2", risk.criterion, t)
            .finish();
        // fold 0's Q, g and epsilon use only its training rows: flipping fold 0's outcomes
        // leaves them unchanged
        let mut y2 = y.clone();
        for i in (0..y2.len()).step_by(5) {
            y2[i] = 1.0 - y2[i];
        }
        let setup2 =
            CvSetup::new(y2.view(), a.view(), &covs, &names, "a", &cfg, &mut notes).unwrap();
        let risk2 = setup2.risk(&[0, 2], &mut notes).unwrap();
        assert_eq!(risk2.fold_epsilon[0], risk.fold_epsilon[0]); // bit-identical
        assert!((risk2.fold_rss[0] - risk.fold_rss[0]).abs() > 1.0); // the loss sees the flips
        assert_ne!(risk2.fold_epsilon[1], risk.fold_epsilon[1]); // fold 1 trains on fold 0's rows
        // VarIC recomputed from the validation-fold predictions (two-pass variance)
        for (v, ev) in risk.folds.iter().enumerate() {
            let ic: Vec<f64> = (0..ev.rows.len())
                .map(|r| ev.h[r] * (ev.y[r] - ev.q_star_a[r]) + ev.q_star_1[r] - ev.q_star_0[r])
                .collect();
            let m = ic.iter().sum::<f64>() / ic.len() as f64;
            let var = ic.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (ic.len() - 1) as f64;
            assert!(Tol::new(1e-12, 0.0).allows(risk.fold_var_ic[v], var));
        }
    }

    #[test]
    fn test_validate_ctmle_cv_logit_rss() {
        let (case, ds) = crate::test_support::load_case("ctmle_cv");
        let (r, o) = run(&ds, "yb", tight());
        check_run(&case, "logit_rss", &r, Some(&o));
    }

    #[test]
    fn test_validate_ctmle_cv_linear_rss() {
        let (case, ds) = crate::test_support::load_case("ctmle_cv");
        let c = CTmleConfig {
            q_model: CTmleQModel::Linear,
            ..tight()
        };
        let (r, o) = run(&ds, "yc", c);
        check_run(&case, "linear_rss", &r, Some(&o));
    }

    #[test]
    fn test_validate_ctmle_cv_backward_and_prespecified() {
        let (case, ds) = crate::test_support::load_case("ctmle_cv");
        let c = CTmleConfig {
            order: SelectionOrder::Backward,
            ..tight()
        };
        let (b, bo) = run(&ds, "yb", c);
        check_run(&case, "logit_backward", &b, Some(&bo));
        let c = CTmleConfig {
            order: SelectionOrder::Prespecified(vec![2, 0]),
            ..tight()
        };
        let (p, po) = run(&ds, "yb", c);
        check_run(&case, "logit_pre31", &p, Some(&po));
    }

    #[test]
    fn test_validate_ctmle_criteria_varic() {
        let (case, ds) = crate::test_support::load_case("ctmle_cv");
        let c = CTmleConfig {
            cv_criterion: CVCriterion::VarIC,
            ..tight()
        };
        let (r, o) = run(&ds, "yb", c.clone());
        check_run(&case, "logit_varic", &r, Some(&o));
        for s in &r.selection_path {
            assert_eq!(s.cv_criterion, s.cv_var_ic);
        }
        let lc = CTmleConfig {
            q_model: CTmleQModel::Linear,
            ..c
        };
        let (l, lo) = run(&ds, "yc", lc);
        check_run(&case, "linear_varic", &l, Some(&lo));
        // the phantom pair: RSS and VarIC select different steps and estimates
        let rss = ctmle(&ds, "yb", "a", &W, tight()).unwrap();
        assert_ne!(rss.selected_step, r.selected_step);
        assert!((rss.ate - r.ate).abs() > 1e-3);
    }

    #[test]
    fn test_validate_ctmle_criteria_penrss() {
        let (case, ds) = crate::test_support::load_case("ctmle_cv");
        let c = CTmleConfig {
            cv_criterion: CVCriterion::PenRSS,
            penalty_factor: 2.0,
            ..tight()
        };
        let (r, o) = run(&ds, "yb", c.clone());
        check_run(&case, "logit_penrss", &r, Some(&o));
        for s in &r.selection_path {
            let want = s.cv_rss + 2.0 * 300f64.ln() * s.n_covariates as f64;
            assert!((s.cv_criterion - want).abs() <= 1e-12 * s.cv_criterion);
        }
        // penalty 0 is the RSS run
        let z = ctmle(
            &ds,
            "yb",
            "a",
            &W,
            CTmleConfig {
                penalty_factor: 0.0,
                ..c
            },
        )
        .unwrap();
        let rss = ctmle(&ds, "yb", "a", &W, tight()).unwrap();
        assert_eq!(z.selected_step, rss.selected_step);
        assert_eq!(z.cv_risk, rss.cv_risk);
    }

    #[test]
    fn test_ctmle_penalty_and_stop_factor_are_validated() {
        let ds = ctmle_sim();
        for pf in [-1.0, f64::NAN] {
            let c = CTmleConfig {
                cv_criterion: CVCriterion::PenRSS,
                penalty_factor: pf,
                ..Default::default()
            };
            expect_invalid_spec(ctmle(&ds, "yb", "a", &W, c).unwrap_err(), "penalty_factor");
        }
        for sf in [0.0, f64::NAN] {
            let c = CTmleConfig {
                stop_factor: sf,
                ..Default::default()
            };
            expect_invalid_spec(ctmle(&ds, "yb", "a", &W, c).unwrap_err(), "stop_factor");
        }
    }

    #[test]
    fn test_cv_criterion_parses_aliases_and_rejects_unknown() {
        CVCriterion::choices().validate().unwrap();
        for s in ["RSS", "cv-rss", "likelihood"] {
            assert_eq!(s.parse::<CVCriterion>().unwrap(), CVCriterion::RSS);
        }
        for s in ["VarIC", "var_ic", "CV-VarIC"] {
            assert_eq!(s.parse::<CVCriterion>().unwrap(), CVCriterion::VarIC);
        }
        for s in ["penrss", "penalised RSS", "CV-PenRSS"] {
            assert_eq!(s.parse::<CVCriterion>().unwrap(), CVCriterion::PenRSS);
        }
        expect_invalid_spec(
            "aic".parse::<CVCriterion>().unwrap_err(),
            "unknown cv_criterion 'aic'",
        );
        expect_invalid_spec(
            "aic".parse::<CVCriterion>().unwrap_err(),
            "rss, varic, penrss",
        );
        for c in [CVCriterion::RSS, CVCriterion::VarIC, CVCriterion::PenRSS] {
            assert_eq!(c.to_string().parse::<CVCriterion>().unwrap(), c);
        }
    }

    #[test]
    fn test_selection_order_parses_aliases_and_rejects_unknown() {
        ORDER_CHOICES.validate().unwrap();
        for s in ["forward", "FWD"] {
            assert_eq!(
                s.parse::<SelectionOrder>().unwrap(),
                SelectionOrder::Forward
            );
        }
        for s in ["Backward", "backward-elimination"] {
            assert_eq!(
                s.parse::<SelectionOrder>().unwrap(),
                SelectionOrder::Backward
            );
        }
        for s in ["prespecified", "listed"] {
            assert_eq!(
                s.parse::<SelectionOrder>().unwrap(),
                SelectionOrder::Prespecified(vec![])
            );
        }
        expect_invalid_spec(
            "random".parse::<SelectionOrder>().unwrap_err(),
            "forward, backward, prespecified",
        );
        assert_eq!(SelectionOrder::help(), "forward | backward | prespecified");
    }

    #[test]
    fn test_stopping_rule_parses_aliases_and_rejects_unknown() {
        RULE_CHOICES.validate().unwrap();
        for s in ["cv_minimum", "min"] {
            assert_eq!(s.parse::<StoppingRule>().unwrap(), StoppingRule::CVMinimum);
        }
        for s in ["1se", "one-SE"] {
            assert_eq!(s.parse::<StoppingRule>().unwrap(), StoppingRule::OneSE);
        }
        assert_eq!(
            StoppingRule::parse_with("max_covariates", Some(3)).unwrap(),
            StoppingRule::MaxCovariates(3)
        );
        expect_invalid_spec(
            "max_covariates".parse::<StoppingRule>().unwrap_err(),
            "needs max_covariates",
        );
        expect_invalid_spec(
            "aic".parse::<StoppingRule>().unwrap_err(),
            "cv_minimum, one_se, max_covariates",
        );
        assert_eq!(StoppingRule::help(), "cv_minimum | one_se | max_covariates");
    }

    #[test]
    fn test_ctmle_q_model_parses_aliases_and_rejects_unknown() {
        CTmleQModel::choices().validate().unwrap();
        for s in ["Linear", "gaussian", "continuous"] {
            assert_eq!(s.parse::<CTmleQModel>().unwrap(), CTmleQModel::Linear);
        }
        for s in ["LOGIT", "binary", "binomial"] {
            assert_eq!(s.parse::<CTmleQModel>().unwrap(), CTmleQModel::Logistic);
        }
        expect_invalid_spec(
            "probit".parse::<CTmleQModel>().unwrap_err(),
            "unknown q_model 'probit'",
        );
    }

    #[test]
    fn test_ctmle_display_reports_algorithm_and_criterion() {
        let ds = ctmle_sim();
        let r = ctmle(
            &ds,
            "yb",
            "a",
            &W,
            CTmleConfig {
                cv_criterion: CVCriterion::VarIC,
                ..Default::default()
            },
        )
        .unwrap();
        let text = r.to_string();
        assert!(text.contains("CV Criterion:   CV-VarIC\n"), "{text}");
        assert!(text.contains(&format!("Algorithm:      {}", r.config.algorithm)));
        assert!(r.config.algorithm.contains("forward selection of g"));
        assert!(text.contains("Criterion"));
        assert!(text.contains("Crit. SE"));
        assert!(
            text.contains(&format!("Selected: {} of 4 candidates", r.n_selected)),
            "{text}"
        );
        let p = ctmle(
            &ds,
            "yb",
            "a",
            &W,
            CTmleConfig {
                cv_criterion: CVCriterion::PenRSS,
                penalty_factor: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(p.to_string().contains("CV-PenRSS (penalty factor 2)"));
        // warnings: the small-sample note on n = 40 with 5 folds
        let small = create_ctmle_test_dataset();
        let s = ctmle(
            &small,
            "y",
            "treatment",
            &["w1", "w2"],
            CTmleConfig {
                n_folds: 15, // 20 treated < 2 * 15: the small-sample warning
                q_model: CTmleQModel::Linear,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!s.warnings.is_empty());
        let notes = crate::traits::convergence::notes_text(&s);
        assert!(s.to_string().ends_with(&notes));
        assert_eq!(
            crate::traits::convergence::notes_json(&s)["warnings"],
            serde_json::Value::from(s.warnings.clone())
        );
    }

    // ── T2-WP13: missing values and column types (R case causal_mediation_na) ──

    #[test]
    fn test_validate_na_ctmle_columns() {
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, y: &str, d: &str, x: &[&str]| {
            ctmle(
                ds,
                y,
                d,
                x,
                CTmleConfig {
                    q_model: CTmleQModel::Linear,
                    ..Default::default()
                },
            )
        };
        // step 3 reaches the per-covariate extraction
        nf::assert_refusals(&case, &ds, true, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| vec![r.ate, r.se]);
    }
}
