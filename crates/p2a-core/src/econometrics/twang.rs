//! Toolkit for Weighting and Analysis of Nonequivalent Groups (twang).
//!
//! Implements GBM-based propensity score estimation with balance optimization, as the twang
//! R package does with `gbm`: Bernoulli boosting with Newton leaf values and trees of
//! `interaction_depth` splits, and the iteration chosen to minimise a balance criterion.
//!
//! # Overview
//!
//! The twang approach differs from standard propensity score estimation in that
//! it uses machine learning (GBM) instead of logistic regression, and it automatically
//! tunes the number of iterations to optimize covariate balance rather than
//! prediction accuracy.
//!
//! # Algorithm
//!
//! `gbm(distribution = "bernoulli", bag.fraction = 1, train.fraction = 1)`:
//!
//! 1. Initialize: F_0 = log(n_treated / n_control) (gbm's `initF`).
//! 2. For m = 1 to M:
//!    - Residuals r_i = t_i - p_i with p_i = sigmoid(F_{m-1}(x_i)).
//!    - Grow a regression tree on r best-first, as gbm does: every terminal node keeps its
//!      best split (the improvement n_L n_R / n (mean_L - mean_R)^2 over the midpoints of
//!      consecutive distinct values, left `x < midpoint`, both children with at least
//!      `min_node_size` rows, gbm's `n.minobsinnode`), and the terminal node with the largest
//!      improvement is split, `interaction_depth` times or until no node can be split.
//!    - Newton leaf values: sum_leaf r / sum_leaf p (1 - p) (0 when the denominator is 0).
//!    - Update: F_m = F_{m-1} + shrinkage * tree(x).
//!    - Evaluate the stop method's balance criterion with the weights of p_m.
//! 3. The optimal iteration is the first minimum of the criterion over iterations
//!    max(1, min_iterations)..=M (twang searches a grid and refines; the global minimum is
//!    never worse). With `balance_threshold > 0` the run stops early at the first iteration
//!    >= `min_iterations` whose criterion is below the threshold.
//!
//! # Stopping Rules
//!
//! Effect sizes are weighted mean differences over an SD fixed from the unweighted sample, as
//! twang's `ps.summary.new2`: the treated SD for ATT, the SD of the full sample for ATE, the
//! control SD for ATC (n - 1, every covariate treated as continuous). KS is the weighted
//! two-sample Kolmogorov-Smirnov statistic.
//!
//! - **ESMean**: Mean absolute standardized effect size across all covariates
//! - **ESMax**: Maximum absolute standardized effect size
//! - **KSMean**: Mean Kolmogorov-Smirnov statistic
//! - **KSMax**: Maximum KS statistic
//!
//! # References
//!
//! - Ridgeway, G., McCaffrey, D., Morral, A., Burgette, L., & Griffin, B.A. (2017).
//!   "Toolkit for Weighting and Analysis of Nonequivalent Groups: A Tutorial".
//!   RAND Corporation.
//!
//! - McCaffrey, D.F., Ridgeway, G., & Morral, A.R. (2004). "Propensity Score
//!   Estimation with Boosted Regression for Evaluating Causal Effects in
//!   Observational Studies". *Psychological Methods*, 9(4), 403-425.
//!
//! - Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting
//!   Machine". *Annals of Statistics*, 29(5), 1189-1232.
//!
//! - R package twang: Ridgeway, G., et al. (2024). twang: Toolkit for Weighting
//!   and Analysis of Nonequivalent Groups. https://cran.r-project.org/package=twang

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::econ_error_from_design;
use crate::econometrics::weightit::{SdDenominator, balance_sd, finite_abs_mean_max, weighted_var};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::options::Choices;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Stopping rule for determining optimal number of iterations.
///
/// The stopping rule determines which balance metric is optimized to select
/// the optimal number of boosting iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StopMethod {
    /// Mean absolute standardized effect size (default)
    ///
    /// Optimizes the average balance across all covariates.
    /// Good for overall balance when all covariates are equally important.
    #[default]
    ESMean,

    /// Maximum absolute standardized effect size
    ///
    /// Optimizes the worst-case balance across covariates.
    /// Use when you need good balance on all covariates.
    ESMax,

    /// Mean Kolmogorov-Smirnov statistic
    ///
    /// KS statistic captures differences in the entire distribution, not just means.
    /// Good when distributional balance matters.
    KSMean,

    /// Maximum Kolmogorov-Smirnov statistic
    ///
    /// Optimizes worst-case distributional balance.
    KSMax,
}

// The accepted spellings (twang's names); the first of each entry is canonical.
crate::option_enum!(
    StopMethod,
    Choices {
        option: "stop_method",
        values: &[
            (&["es.mean", "esmean"], StopMethod::ESMean),
            (&["es.max", "esmax"], StopMethod::ESMax),
            (&["ks.mean", "ksmean"], StopMethod::KSMean),
            (&["ks.max", "ksmax"], StopMethod::KSMax),
        ],
    },
    no_display
);

impl fmt::Display for StopMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopMethod::ESMean => write!(f, "es.mean (Mean Std. Effect Size)"),
            StopMethod::ESMax => write!(f, "es.max (Max Std. Effect Size)"),
            StopMethod::KSMean => write!(f, "ks.mean (Mean KS Statistic)"),
            StopMethod::KSMax => write!(f, "ks.max (Max KS Statistic)"),
        }
    }
}

/// Target estimand for treatment effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TwangEstimand {
    /// Average Treatment Effect on the Treated (default)
    ///
    /// Weights control units to match the treated group distribution.
    #[default]
    ATT,

    /// Average Treatment Effect (population)
    ///
    /// Weights both groups to match the overall population.
    ATE,

    /// Average Treatment Effect on the Control
    ///
    /// Weights treated units to match the control group distribution.
    ATC,
}

// The accepted spellings (twang's names); the first of each entry is canonical.
crate::option_enum!(
    TwangEstimand,
    Choices {
        option: "estimand",
        values: &[
            (&["att"], TwangEstimand::ATT),
            (&["ate"], TwangEstimand::ATE),
            (&["atc"], TwangEstimand::ATC),
        ],
    },
    no_display
);

impl fmt::Display for TwangEstimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TwangEstimand::ATT => write!(f, "ATT (Effect on Treated)"),
            TwangEstimand::ATE => write!(f, "ATE (Average Treatment Effect)"),
            TwangEstimand::ATC => write!(f, "ATC (Effect on Control)"),
        }
    }
}

/// Configuration for twang propensity score estimation.
#[derive(Debug, Clone)]
pub struct TwangConfig {
    /// Maximum number of boosting iterations (default: 3000). Must be at least 1; a
    /// count whose tree buffer cannot be allocated is refused with
    /// `EconError::ResourceLimit`.
    pub n_trees: usize,

    /// Learning rate / shrinkage (default: 0.01)
    ///
    /// Smaller values require more iterations but often give better results.
    /// Typical values: 0.001 to 0.1. Must be finite and in (0, 1].
    pub shrinkage: f64,

    /// Stopping rule for selecting optimal iterations (default: ESMean)
    pub stop_method: StopMethod,

    /// Target estimand (default: ATT)
    pub estimand: TwangEstimand,

    /// Optional early stop (default: 0, off)
    ///
    /// With a value above 0, boosting stops at the first iteration >= `min_iterations` whose
    /// balance criterion falls below it, and the optimum is chosen among the iterations run so
    /// far. By default every iteration up to `n_trees` is evaluated and the balance-optimal one
    /// is chosen. Must be finite and >= 0.
    pub balance_threshold: f64,

    /// The first iteration the selection (and the early stop) considers (default: 1); at most
    /// `n_trees`. 0 is treated as 1 (the constant model, iteration 0, is never chosen).
    pub min_iterations: usize,

    /// Splits per tree (gbm's `interaction.depth`; default: 1, stumps; twang's default is 3).
    /// Must be at least 1. Each tree is grown best-first to at most this many splits.
    pub interaction_depth: usize,

    /// Minimum observations per terminal node (default: 10, gbm's `n.minobsinnode`)
    pub min_node_size: usize,
}

impl Default for TwangConfig {
    fn default() -> Self {
        Self {
            n_trees: 3000,
            shrinkage: 0.01,
            stop_method: StopMethod::ESMean,
            estimand: TwangEstimand::ATT,
            balance_threshold: 0.0,
            min_iterations: 1,
            interaction_depth: 1,
            min_node_size: 10,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Balance statistics for a single covariate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwangCovariateBalance {
    /// Covariate name
    pub name: String,

    /// Mean in treated group (weighted for after)
    pub mean_treated: f64,

    /// Mean in control group (weighted for after)
    pub mean_control: f64,

    /// Standardized effect size: (mean_t - mean_c) / SD of the unweighted sample (treated SD
    /// for ATT, full-sample SD for ATE, control SD for ATC; twang's definition)
    pub std_eff_size: f64,

    /// Weighted two-sample Kolmogorov-Smirnov statistic
    pub ks_statistic: f64,

    /// Variance ratio var_t / var_c of the weighted variances (cobalt's formula)
    pub var_ratio: f64,
}

/// Balance table with summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwangBalanceTable {
    /// Per-covariate balance statistics
    pub covariates: Vec<TwangCovariateBalance>,

    /// Mean absolute standardized effect size
    pub es_mean: f64,

    /// Maximum absolute standardized effect size
    pub es_max: f64,

    /// Mean KS statistic
    pub ks_mean: f64,

    /// Maximum KS statistic
    pub ks_max: f64,

    /// Number of covariates with |std_eff_size| < 0.1
    pub n_balanced: usize,
}

impl TwangBalanceTable {
    /// Create an empty balance table.
    pub fn new() -> Self {
        Self {
            covariates: Vec::new(),
            es_mean: 0.0,
            es_max: 0.0,
            ks_mean: 0.0,
            ks_max: 0.0,
            n_balanced: 0,
        }
    }

    /// Get the balance metric based on stop method.
    pub fn get_metric(&self, method: StopMethod) -> f64 {
        match method {
            StopMethod::ESMean => self.es_mean,
            StopMethod::ESMax => self.es_max,
            StopMethod::KSMean => self.ks_mean,
            StopMethod::KSMax => self.ks_max,
        }
    }
}

impl Default for TwangBalanceTable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TwangBalanceTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>10} {:>10}",
            "Covariate", "Mean(T)", "Mean(C)", "Std.Eff", "KS", "VarRatio"
        )?;
        writeln!(f, "{}", "-".repeat(78))?;

        for cov in &self.covariates {
            let balanced = if cov.std_eff_size.abs() < 0.1 {
                " "
            } else {
                "*"
            };
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>12.4}{} {:>10.4} {:>10.4}",
                cov.name,
                cov.mean_treated,
                cov.mean_control,
                cov.std_eff_size,
                balanced,
                cov.ks_statistic,
                cov.var_ratio
            )?;
        }

        writeln!(f, "{}", "-".repeat(78))?;
        writeln!(
            f,
            "ES.Mean: {:.4}  ES.Max: {:.4}  KS.Mean: {:.4}  KS.Max: {:.4}",
            self.es_mean, self.es_max, self.ks_mean, self.ks_max
        )?;
        writeln!(
            f,
            "Balanced (|ES| < 0.1): {}/{}",
            self.n_balanced,
            self.covariates.len()
        )?;

        Ok(())
    }
}

/// Decision stump for gradient boosting.
///
/// A decision stump is a tree with a single split. It predicts one value
/// for observations where x[feature] <= threshold, and another value otherwise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionStump {
    /// Index of the feature to split on
    pub feature_idx: usize,

    /// Split threshold
    pub threshold: f64,

    /// Prediction for left branch (x[feature] <= threshold)
    pub left_value: f64,

    /// Prediction for right branch (x[feature] > threshold)
    pub right_value: f64,
}

impl DecisionStump {
    /// Predict for a single observation.
    pub fn predict(&self, x: &[f64]) -> f64 {
        if x[self.feature_idx] <= self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }

    /// Predict for all observations.
    pub fn predict_all(&self, x: &Array2<f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut predictions = Array1::zeros(n);
        for i in 0..n {
            predictions[i] = if x[[i, self.feature_idx]] <= self.threshold {
                self.left_value
            } else {
                self.right_value
            };
        }
        predictions
    }
}

/// One boosting tree (gbm's regression tree with up to `interaction_depth` splits): split
/// nodes and Newton leaf values. Node 0 is the root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoostTree {
    /// The nodes; a split's children are indices into this vector.
    pub nodes: Vec<BoostNode>,
}

/// A node of a [`BoostTree`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoostNode {
    /// Rows with `x[feature] <= threshold` go to `left`, the others to `right`.
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    /// A terminal node and its (unshrunk) value.
    Leaf { value: f64 },
}

impl BoostTree {
    /// The tree's value for one observation.
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.leaf_value(self.leaf_of(|f| x[f]))
    }

    /// Index of the leaf an observation falls in, given its feature values.
    fn leaf_of(&self, value: impl Fn(usize) -> f64) -> usize {
        let mut i = 0;
        while let BoostNode::Split {
            feature,
            threshold,
            left,
            right,
        } = self.nodes[i]
        {
            i = if value(feature) <= threshold {
                left
            } else {
                right
            };
        }
        i
    }

    fn leaf_value(&self, i: usize) -> f64 {
        match self.nodes[i] {
            BoostNode::Leaf { value } => value,
            BoostNode::Split { .. } => f64::NAN,
        }
    }

    /// The tree as a stump (depth 1); a tree without a split predicts its leaf everywhere.
    fn as_stump(&self) -> DecisionStump {
        match self.nodes[0] {
            BoostNode::Split {
                feature,
                threshold,
                left,
                right,
            } => DecisionStump {
                feature_idx: feature,
                threshold,
                left_value: self.leaf_value(left),
                right_value: self.leaf_value(right),
            },
            BoostNode::Leaf { value } => DecisionStump {
                feature_idx: 0,
                threshold: 0.0,
                left_value: value,
                right_value: value,
            },
        }
    }
}

/// Result from twang propensity score estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwangResult {
    /// Estimated propensity scores P(T=1|X)
    pub propensity_scores: Vec<f64>,

    /// IPW weights for treatment effect estimation
    pub weights: Vec<f64>,

    /// Optimal number of boosting iterations
    pub optimal_n_trees: usize,

    /// Balance before weighting
    pub balance_before: TwangBalanceTable,

    /// Balance after weighting (at optimal iteration)
    pub balance_after: TwangBalanceTable,

    /// Balance metric at each iteration (for diagnostics)
    pub balance_history: Vec<f64>,

    /// Stopping method used
    pub stop_method: StopMethod,

    /// Target estimand
    pub estimand: TwangEstimand,

    /// Effective sample size (sum of weights squared / sum of squared weights)
    pub effective_sample_size: f64,

    /// ESS for treated group
    pub ess_treated: f64,

    /// ESS for control group
    pub ess_control: f64,

    /// Number of observations
    pub n_obs: usize,

    /// Number of treated
    pub n_treated: usize,

    /// Number of control
    pub n_control: usize,

    /// Covariate names
    pub covariate_names: Vec<String>,

    /// Maximum weight
    pub max_weight: f64,

    /// Minimum weight (among non-zero)
    pub min_weight: f64,

    /// Learned stumps (internal, for prediction); filled only when `interaction_depth` is 1
    #[serde(skip)]
    pub stumps: Vec<DecisionStump>,

    /// Every boosting tree, in order (internal, for prediction)
    #[serde(skip)]
    pub trees: Vec<BoostTree>,

    /// Splits per tree used
    #[serde(default)]
    pub interaction_depth: usize,

    /// The first iteration the selection considered (max(1, min_iterations))
    #[serde(default)]
    pub search_from: usize,

    /// Warnings (optimum at the last iteration, early stop)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,

    /// Initial F0 value
    #[serde(skip)]
    pub f0: f64,

    /// Shrinkage used
    pub shrinkage: f64,
}

impl fmt::Display for TwangResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "twang: GBM Propensity Score Estimation")?;
        writeln!(f, "=======================================")?;
        writeln!(f)?;
        writeln!(f, "Stop Method: {}", self.stop_method)?;
        writeln!(f, "Estimand:    {}", self.estimand)?;
        writeln!(f)?;
        writeln!(f, "Sample:")?;
        writeln!(
            f,
            "  Total:    {}  (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(f)?;
        writeln!(f, "GBM Tuning:")?;
        writeln!(
            f,
            "  Optimal Iterations: {} (minimum of {} over iterations {}..{})",
            self.optimal_n_trees,
            stop_label(self.stop_method),
            self.search_from,
            self.balance_history.len().saturating_sub(1)
        )?;
        writeln!(f, "  Interaction Depth:  {}", self.interaction_depth)?;
        writeln!(f, "  Shrinkage:          {}", self.shrinkage)?;
        writeln!(f)?;

        // Propensity score summary
        let ps_min = self
            .propensity_scores
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let ps_max = self
            .propensity_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let ps_mean: f64 = self.propensity_scores.iter().sum::<f64>() / self.n_obs as f64;
        writeln!(f, "Propensity Score Summary:")?;
        writeln!(
            f,
            "  Mean: {:.4}  Min: {:.4}  Max: {:.4}",
            ps_mean, ps_min, ps_max
        )?;
        writeln!(f)?;

        writeln!(f, "Weight Summary:")?;
        writeln!(
            f,
            "  Range:   [{:.4}, {:.4}]",
            self.min_weight, self.max_weight
        )?;
        writeln!(
            f,
            "  ESS:     {:.1} (Treated: {:.1}, Control: {:.1})",
            self.effective_sample_size, self.ess_treated, self.ess_control
        )?;
        writeln!(f)?;

        writeln!(f, "Balance Before Weighting:")?;
        writeln!(
            f,
            "  {} = {:.4}",
            match self.stop_method {
                StopMethod::ESMean => "ES.Mean",
                StopMethod::ESMax => "ES.Max",
                StopMethod::KSMean => "KS.Mean",
                StopMethod::KSMax => "KS.Max",
            },
            self.balance_before.get_metric(self.stop_method)
        )?;
        writeln!(f)?;

        writeln!(f, "Balance After Weighting:")?;
        writeln!(
            f,
            "  {} = {:.4}",
            match self.stop_method {
                StopMethod::ESMean => "ES.Mean",
                StopMethod::ESMax => "ES.Max",
                StopMethod::KSMean => "KS.Mean",
                StopMethod::KSMax => "KS.Max",
            },
            self.balance_after.get_metric(self.stop_method)
        )?;
        writeln!(f)?;

        writeln!(f, "Covariate Balance (After Weighting):")?;
        write!(f, "{}", self.balance_after)?;
        write!(f, "{}", notes_text(self))?;

        Ok(())
    }
}

impl ResultNotes for TwangResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// twang's name of a stop method.
fn stop_label(m: StopMethod) -> &'static str {
    match m {
        StopMethod::ESMean => "es.mean",
        StopMethod::ESMax => "es.max",
        StopMethod::KSMean => "ks.mean",
        StopMethod::KSMax => "ks.max",
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run twang GBM propensity score estimation.
///
/// This function estimates propensity scores by gbm's Bernoulli boosting (trees of
/// `interaction_depth` splits, Newton leaf values) and selects the iteration that minimises
/// the balance criterion of `stop_method` over every evaluated iteration.
///
/// # Arguments
///
/// * `dataset` - Dataset containing treatment and covariates
/// * `treatment_col` - Name of binary treatment column (0/1)
/// * `covariate_cols` - Names of covariate columns for propensity model
/// * `config` - Configuration options (optional, uses defaults if None)
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::twang::{run_twang, TwangConfig, StopMethod};
///
/// let config = TwangConfig {
///     n_trees: 3000,
///     shrinkage: 0.01,
///     stop_method: StopMethod::ESMean,
///     ..Default::default()
/// };
///
/// let result = run_twang(&dataset, "treatment", &["age", "income"], Some(config))?;
/// println!("Optimal iterations: {}", result.optimal_n_trees);
/// println!("Balance (ES.Mean): {:.4}", result.balance_after.es_mean);
/// ```
///
/// # References
///
/// - McCaffrey, D.F., Ridgeway, G., & Morral, A.R. (2004). "Propensity Score
///   Estimation with Boosted Regression for Evaluating Causal Effects".
///   *Psychological Methods*, 9(4), 403-425.
pub fn run_twang(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: Option<TwangConfig>,
) -> EconResult<TwangResult> {
    let config = config.unwrap_or_default();

    // Missing values are refused (MatchIt, WeightIt and tmle refuse them).
    // Extract treatment variable
    let t = DesignMatrix::extract_column(dataset.df(), treatment_col)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;

    let n = t.len();

    // Validate treatment is binary
    let n_treated: usize = t.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n - n_treated;

    if n_treated == 0 || n_control == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment variable '{}' must have both treated and control observations. Found {} treated, {} control.",
                treatment_col, n_treated, n_control
            ),
        });
    }

    // Build covariate matrix (no intercept for tree-based methods)
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)
        .map_err(|e| econ_error_from_design(e, dataset.df()))?;
    let x = design.data;
    let covariate_names = design.column_names.clone();

    validate_twang_config(&config)?;
    if t.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "treatment '{treatment_col}' contains NaN or infinite values; twang needs a finite 0/1 treatment"
            ),
        });
    }
    for (j, name) in covariate_names.iter().enumerate() {
        if x.column(j).iter().any(|v| !v.is_finite()) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "covariate '{name}' contains NaN or infinite values; twang needs finite covariates"
                ),
            });
        }
    }

    let sorted = SortedFeatures::new(&x);
    let crit = Criterion::new(&x, &t, &sorted, config.estimand);

    // Run GBM boosting
    let boost = run_gbm_boosting(&x, &t, &sorted, &crit, &config)?;
    let propensity_scores = boost.propensity_scores;

    // Compute final weights at optimal iteration
    let weights = compute_twang_weights(&propensity_scores, &t, config.estimand);

    // Compute balance before (unweighted)
    let balance_before = compute_twang_balance(&x, &crit, &covariate_names, None);

    // Compute balance after (weighted)
    let balance_after = compute_twang_balance(&x, &crit, &covariate_names, Some(&weights));

    // Compute effective sample sizes
    let (ess_total, ess_treated, ess_control) = compute_twang_ess(&weights, &t);

    // Weight summary
    let max_weight = weights
        .iter()
        .filter(|&&w| w > 0.0)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_weight = weights
        .iter()
        .filter(|&&w| w > 0.0)
        .cloned()
        .fold(f64::INFINITY, f64::min);

    let stumps = if config.interaction_depth == 1 {
        boost.trees.iter().map(BoostTree::as_stump).collect()
    } else {
        Vec::new()
    };

    Ok(TwangResult {
        propensity_scores,
        weights,
        optimal_n_trees: boost.optimal_iter,
        balance_before,
        balance_after,
        balance_history: boost.balance_history,
        stop_method: config.stop_method,
        estimand: config.estimand,
        effective_sample_size: ess_total,
        ess_treated,
        ess_control,
        n_obs: n,
        n_treated,
        n_control,
        covariate_names,
        max_weight,
        min_weight,
        stumps,
        trees: boost.trees,
        interaction_depth: config.interaction_depth,
        search_from: config.min_iterations.max(1),
        warnings: boost.warnings,
        f0: boost.f0,
        shrinkage: config.shrinkage,
    })
}

/// Convenience wrapper with default configuration.
pub fn twang(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    stop_method: StopMethod,
    estimand: TwangEstimand,
) -> EconResult<TwangResult> {
    run_twang(
        dataset,
        treatment_col,
        covariate_cols,
        Some(TwangConfig {
            stop_method,
            estimand,
            ..Default::default()
        }),
    )
}

/// Checks the boosting parameters before any work.
fn validate_twang_config(config: &TwangConfig) -> EconResult<()> {
    if config.n_trees == 0 {
        return Err(EconError::InvalidSpecification {
            message: "n_trees must be at least 1; got 0".to_string(),
        });
    }
    if !(config.shrinkage.is_finite() && config.shrinkage > 0.0 && config.shrinkage <= 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "shrinkage must be a finite learning rate in (0, 1]; got {}",
                config.shrinkage
            ),
        });
    }
    if !(config.balance_threshold.is_finite() && config.balance_threshold >= 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "balance_threshold must be finite and >= 0 (0 disables early stopping); got {}",
                config.balance_threshold
            ),
        });
    }
    if config.interaction_depth == 0 {
        return Err(EconError::InvalidSpecification {
            message: "interaction_depth must be at least 1 (splits per tree); got 0".to_string(),
        });
    }
    if config.min_iterations > config.n_trees {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "min_iterations must be at most n_trees ({}); got {}",
                config.n_trees, config.min_iterations
            ),
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// GBM Boosting Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// What a boosting run returns.
struct BoostOutcome {
    trees: Vec<BoostTree>,
    f0: f64,
    balance_history: Vec<f64>,
    optimal_iter: usize,
    propensity_scores: Vec<f64>,
    warnings: Vec<String>,
}

/// gbm's Bernoulli boosting state: the link F, the presorted features and the tree settings.
struct Booster<'a> {
    x: &'a Array2<f64>,
    t: &'a [f64],
    sorted: &'a SortedFeatures,
    depth: usize,
    min_node_size: usize,
    shrinkage: f64,
    f: Vec<f64>,
    // scratch, reused across iterations
    p: Vec<f64>,
    z: Vec<f64>,
    leaf: Vec<usize>,
}

impl<'a> Booster<'a> {
    fn new(
        x: &'a Array2<f64>,
        t: &'a [f64],
        sorted: &'a SortedFeatures,
        config: &TwangConfig,
    ) -> EconResult<(Self, f64)> {
        let n = t.len();
        let n_treated = t.iter().filter(|&&v| v >= 0.5).count() as f64;
        let n_control = n as f64 - n_treated;
        // gbm's initF: the log-odds of treatment
        let f0 = (n_treated / n_control).ln();
        let booster = Booster {
            x,
            t,
            sorted,
            depth: config.interaction_depth,
            min_node_size: config.min_node_size,
            shrinkage: config.shrinkage,
            f: resource::try_vec(n, f0, "boosting link")?,
            p: resource::try_vec(n, 0.0, "boosting probabilities")?,
            z: resource::try_vec(n, 0.0, "boosting residuals")?,
            leaf: resource::try_vec(n, 0usize, "boosting leaf index")?,
        };
        Ok((booster, f0))
    }

    /// One boosting iteration: residuals, a tree, Newton leaf values, the link update.
    fn step(&mut self) -> BoostTree {
        for i in 0..self.t.len() {
            let p = sigmoid(self.f[i]);
            self.p[i] = p;
            self.z[i] = self.t[i] - p;
        }
        let mut tree = grow_tree(self.x, self.sorted, &self.z, self.depth, self.min_node_size);
        set_newton_leaves(&mut tree, self.x, &self.z, &self.p, &mut self.leaf);
        for i in 0..self.t.len() {
            self.f[i] += self.shrinkage * tree.leaf_value(self.leaf[i]);
        }
        tree
    }
}

/// gbm's Newton leaf values (Bernoulli): sum_leaf z / sum_leaf p (1 - p), summed in index
/// order, 0 when the denominator is 0. `leaf` receives each row's leaf index.
fn set_newton_leaves(
    tree: &mut BoostTree,
    x: &Array2<f64>,
    z: &[f64],
    p: &[f64],
    leaf: &mut [usize],
) {
    let n_nodes = tree.nodes.len();
    let (mut num, mut den) = (vec![0.0; n_nodes], vec![0.0; n_nodes]);
    for (i, li) in leaf.iter_mut().enumerate() {
        let l = tree.leaf_of(|j| x[[i, j]]);
        *li = l;
        num[l] += z[i];
        den[l] += p[i] * (1.0 - p[i]);
    }
    for (l, node) in tree.nodes.iter_mut().enumerate() {
        if let BoostNode::Leaf { value } = node {
            *value = if den[l] != 0.0 { num[l] / den[l] } else { 0.0 };
        }
    }
}

/// Run gbm's Bernoulli boosting, evaluate the balance criterion at every iteration and pick
/// the first minimum over `max(1, min_iterations)..=evaluated`; with `balance_threshold > 0`
/// stop early at the first iteration >= `min_iterations` below it.
fn run_gbm_boosting(
    x: &Array2<f64>,
    t: &Array1<f64>,
    sorted: &SortedFeatures,
    crit: &Criterion,
    config: &TwangConfig,
) -> EconResult<BoostOutcome> {
    let t_vec = t.to_vec();
    let mut trees =
        resource::try_with_capacity::<BoostTree>(config.n_trees, "boosting trees (n_trees)")?;
    let history_len = config
        .n_trees
        .checked_add(1)
        .ok_or_else(|| EconError::ResourceLimit("n_trees is too large".to_string()))?;
    let mut balance_history =
        resource::try_with_capacity::<f64>(history_len, "balance history (n_trees + 1)")?;
    let (mut booster, f0) = Booster::new(x, &t_vec, sorted, config)?;
    let mut weights = vec![0.0; t_vec.len()];

    // Iteration 0: the constant model (recorded, never chosen).
    let ps0: Vec<f64> = booster.f.iter().map(|&f| sigmoid(f)).collect();
    twang_weights_into(&ps0, &crit.groups, config.estimand, &mut weights);
    balance_history.push(crit.value(x, &weights, config.stop_method));

    let start = config.min_iterations.max(1);
    let mut best = (f64::INFINITY, 0usize);
    let mut f_best = booster.f.clone();
    let mut early_stop = None;
    let mut ps = ps0;
    for m in 1..=config.n_trees {
        resource::checkpoint()?;
        trees.push(booster.step());
        for (p, &f) in ps.iter_mut().zip(&booster.f) {
            *p = sigmoid(f);
        }
        twang_weights_into(&ps, &crit.groups, config.estimand, &mut weights);
        let value = crit.value(x, &weights, config.stop_method);
        balance_history.push(value);
        if m >= start && value < best.0 {
            best = (value, m);
            f_best.copy_from_slice(&booster.f);
        }
        if config.balance_threshold > 0.0
            && m >= config.min_iterations
            && value < config.balance_threshold
        {
            early_stop = Some(m);
            break;
        }
    }
    let evaluated = trees.len();

    let mut warnings = Vec::new();
    let optimal_iter = if best.1 == 0 {
        // Every criterion value was NaN: report the last iteration.
        warnings.push(format!(
            "the balance criterion ({}) was not finite at any iteration; the last iteration is reported",
            stop_label(config.stop_method)
        ));
        f_best.copy_from_slice(&booster.f);
        evaluated
    } else {
        best.1
    };
    match early_stop {
        Some(m) => warnings.push(format!(
            "boosting stopped early at iteration {m} (criterion below balance_threshold = {}); the optimum is over iterations {start}..{m} only",
            config.balance_threshold
        )),
        None if best.1 != 0 && optimal_iter == evaluated => warnings.push(format!(
            "the balance criterion was still decreasing at n_trees = {}; increase n_trees",
            config.n_trees
        )),
        None => {}
    }

    let propensity_scores: Vec<f64> = f_best.iter().map(|&f| sigmoid(f)).collect();
    Ok(BoostOutcome {
        trees,
        f0,
        balance_history,
        optimal_iter,
        propensity_scores,
        warnings,
    })
}

/// gbm's `initF` and the link after each of the iterations `at` (ascending), for the R pins.
#[cfg(test)]
fn gbm_link_path(
    x: &Array2<f64>,
    t: &Array1<f64>,
    config: &TwangConfig,
    at: &[usize],
) -> (f64, Vec<Vec<f64>>) {
    let t_vec = t.to_vec();
    let sorted = SortedFeatures::new(x);
    let (mut booster, f0) = Booster::new(x, &t_vec, &sorted, config).unwrap();
    let mut out = Vec::new();
    for m in 1..=at.iter().copied().max().unwrap_or(0) {
        booster.step();
        if at.contains(&m) {
            out.push(booster.f.clone());
        }
    }
    (f0, out)
}

/// Relative margin (of the node's total sum of squares) by which a later split candidate must
/// beat the best one so far. Two features that induce the same partition give criteria that
/// differ only in summation order; the margin keeps the first such candidate.
const SSE_TIE_REL: f64 = 1e-10;

/// Row indices of each feature column in ascending order of value, computed once per
/// boosting run (O(k n log n)) so that every split search is a linear sweep.
struct SortedFeatures {
    order: Vec<Vec<usize>>,
    /// 0..n, the root's rows in index order.
    rows: Vec<usize>,
}

impl SortedFeatures {
    /// Sorts each column of `x` (stable, `total_cmp`; `x` is finite, validated by `run_twang`).
    fn new(x: &Array2<f64>) -> Self {
        let order = (0..x.ncols())
            .map(|j| {
                let col = x.column(j);
                let mut idx: Vec<usize> = (0..x.nrows()).collect();
                idx.sort_by(|&a, &b| col[a].total_cmp(&col[b]));
                idx
            })
            .collect();
        Self {
            order,
            rows: (0..x.nrows()).collect(),
        }
    }
}

/// The best split of one node.
#[derive(Debug, Clone, Copy)]
struct NodeSplit {
    feature: usize,
    threshold: f64,
    /// gbm's improvement n_L n_R / n (mean_L - mean_R)^2 = S_L^2 / n_L + S_R^2 / n_R - S^2 / n.
    improvement: f64,
}

/// The best split of the rows `rows` (any order; the node's totals are summed in this order)
/// whose per-feature sorted lists are `lists`. Candidates are the midpoints between
/// consecutive distinct values of each feature, in ascending feature and value order, with at
/// least `min_node_size` rows on each side; the criterion is the SSE after the split
/// (`sum r^2 - S_L^2 / n_L - S_R^2 / n_R`) and the first best wins (within `SSE_TIE_REL`).
fn best_split(
    x: &Array2<f64>,
    rows: &[usize],
    lists: &[&[usize]],
    z: &[f64],
    min_node_size: usize,
) -> Option<NodeSplit> {
    let n = rows.len();
    let total: f64 = rows.iter().map(|&i| z[i]).sum();
    let total_sq: f64 = rows.iter().map(|&i| z[i] * z[i]).sum();
    let tie_margin = SSE_TIE_REL * total_sq;

    let mut best: Option<(usize, f64, f64)> = None;
    let mut best_sse = f64::INFINITY;
    for (j, order) in lists.iter().enumerate() {
        let mut left_sum = 0.0;
        let mut left_count = 0usize;
        for pos in 0..n.saturating_sub(1) {
            let i = order[pos];
            left_sum += z[i];
            left_count += 1;

            // Candidates sit only at group boundaries (`-0.0` and `0.0` are one group).
            let v = x[[i, j]];
            let v_next = x[[order[pos + 1], j]];
            if v == v_next {
                continue;
            }
            let right_count = n - left_count;
            if left_count < min_node_size || right_count < min_node_size {
                continue;
            }
            let right_sum = total - left_sum;
            let gain = left_sum * left_sum / left_count as f64
                + right_sum * right_sum / right_count as f64;
            let sse = total_sq - gain;
            if sse < best_sse - tie_margin {
                // The midpoint must separate the two groups; with adjacent floats or an
                // overflowing sum it does not, and the lower value is the exact boundary.
                let mut threshold = (v + v_next) / 2.0;
                if !(threshold >= v && threshold < v_next) {
                    threshold = v;
                }
                best_sse = sse;
                best = Some((j, threshold, gain));
            }
        }
    }
    best.map(|(feature, threshold, gain)| NodeSplit {
        feature,
        threshold,
        improvement: gain - total * total / n as f64,
    })
}

/// A terminal node while a tree grows: its rows (in index order, and sorted per feature) and
/// its best split. The root borrows the run's presorted lists.
struct Terminal {
    node: usize,
    rows: Option<(Vec<usize>, Vec<Vec<usize>>)>,
    split: Option<NodeSplit>,
}

/// Grow one regression tree on the residuals `z` best-first, as gbm does: split the terminal
/// node with the largest improvement (the first on ties), `depth` times or until no node can
/// be split. A split's rows are partitioned stably into its children, so every list stays
/// sorted and a tree costs O(depth k n). Leaf values are left at 0 (the caller sets them).
fn grow_tree(
    x: &Array2<f64>,
    sorted: &SortedFeatures,
    z: &[f64],
    depth: usize,
    min_node_size: usize,
) -> BoostTree {
    let root_rows = &sorted.rows;
    let root_lists: Vec<&[usize]> = sorted.order.iter().map(|v| v.as_slice()).collect();
    let mut nodes = vec![BoostNode::Leaf { value: 0.0 }];
    let mut terminals = vec![Terminal {
        node: 0,
        rows: None,
        split: best_split(x, root_rows, &root_lists, z, min_node_size),
    }];

    for level in 0..depth {
        // The terminal with the largest improvement; strict `>` keeps the first.
        let mut pick: Option<(usize, NodeSplit)> = None;
        for (b, term) in terminals.iter().enumerate() {
            if let Some(sp) = term.split
                && pick.is_none_or(|(_, best)| sp.improvement > best.improvement)
            {
                pick = Some((b, sp));
            }
        }
        let Some((b, sp)) = pick else { break };
        let (left_id, right_id) = (nodes.len(), nodes.len() + 1);
        nodes[terminals[b].node] = BoostNode::Split {
            feature: sp.feature,
            threshold: sp.threshold,
            left: left_id,
            right: right_id,
        };
        nodes.push(BoostNode::Leaf { value: 0.0 });
        nodes.push(BoostNode::Leaf { value: 0.0 });

        let last = level + 1 == depth;
        let (left, right) = if last {
            // No further split: the children's rows are not needed.
            (None, None)
        } else {
            let goes_left = |i: usize| x[[i, sp.feature]] <= sp.threshold;
            let part = |v: &[usize]| -> (Vec<usize>, Vec<usize>) {
                v.iter().partition(|&&i| goes_left(i))
            };
            let (index, lists): (&[usize], Vec<&[usize]>) = match &terminals[b].rows {
                Some((index, lists)) => (index, lists.iter().map(|v| v.as_slice()).collect()),
                None => (root_rows, root_lists.clone()),
            };
            let (li, ri) = part(index);
            let (mut ll, mut rl) = (Vec::with_capacity(lists.len()), Vec::new());
            for list in &lists {
                let (a, c) = part(list);
                ll.push(a);
                rl.push(c);
            }
            (Some((li, ll)), Some((ri, rl)))
        };
        let search = |rows: &Option<(Vec<usize>, Vec<Vec<usize>>)>| {
            rows.as_ref().and_then(|(index, lists)| {
                let lists: Vec<&[usize]> = lists.iter().map(|v| v.as_slice()).collect();
                best_split(x, index, &lists, z, min_node_size)
            })
        };
        let (left_split, right_split) = (search(&left), search(&right));
        terminals[b] = Terminal {
            node: left_id,
            rows: left,
            split: left_split,
        };
        terminals.push(Terminal {
            node: right_id,
            rows: right,
            split: right_split,
        });
    }
    BoostTree { nodes }
}

/// Sigmoid function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weight and Balance Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// The treatment groups, built once per run.
struct Groups {
    treated: Vec<bool>,
    treated_idx: Vec<usize>,
    control_idx: Vec<usize>,
}

impl Groups {
    fn new(t: &Array1<f64>) -> Self {
        let treated: Vec<bool> = t.iter().map(|&v| v >= 0.5).collect();
        let treated_idx = (0..t.len()).filter(|&i| treated[i]).collect();
        let control_idx = (0..t.len()).filter(|&i| !treated[i]).collect();
        Self {
            treated,
            treated_idx,
            control_idx,
        }
    }
}

/// Compute IPW weights based on propensity scores and estimand.
fn compute_twang_weights(
    propensity_scores: &[f64],
    t: &Array1<f64>,
    estimand: TwangEstimand,
) -> Vec<f64> {
    let mut weights = vec![0.0; t.len()];
    twang_weights_into(propensity_scores, &Groups::new(t), estimand, &mut weights);
    weights
}

/// IPW weights of the estimand (scores clipped to [1e-10, 1 - 1e-10]), each group normalised
/// to sum to its size.
fn twang_weights_into(
    propensity_scores: &[f64],
    groups: &Groups,
    estimand: TwangEstimand,
    weights: &mut [f64],
) {
    for (i, w) in weights.iter_mut().enumerate() {
        let p = propensity_scores[i].clamp(1e-10, 1.0 - 1e-10);
        let treated = groups.treated[i];
        *w = match estimand {
            // ATT: treated 1, controls p / (1 - p)
            TwangEstimand::ATT => {
                if treated {
                    1.0
                } else {
                    p / (1.0 - p)
                }
            }
            // ATE: treated 1 / p, controls 1 / (1 - p)
            TwangEstimand::ATE => {
                if treated {
                    1.0 / p
                } else {
                    1.0 / (1.0 - p)
                }
            }
            // ATC: treated (1 - p) / p, controls 1
            TwangEstimand::ATC => {
                if treated {
                    (1.0 - p) / p
                } else {
                    1.0
                }
            }
        };
    }
    for idx in [&groups.treated_idx, &groups.control_idx] {
        let sum: f64 = idx.iter().map(|&i| weights[i]).sum();
        if sum > 0.0 {
            let scale = idx.len() as f64 / sum;
            for &i in idx.iter() {
                weights[i] *= scale;
            }
        }
    }
}

/// twang's balance criterion: the groups, the presorted features and, per covariate, the SD
/// fixed from the unweighted sample (treated SD for ATT, full-sample SD for ATE, control SD for
/// ATC; n - 1, every covariate continuous, as `ps.summary.new2`).
struct Criterion<'a> {
    groups: Groups,
    sorted: &'a SortedFeatures,
    sd: Vec<f64>,
}

impl<'a> Criterion<'a> {
    fn new(
        x: &Array2<f64>,
        t: &Array1<f64>,
        sorted: &'a SortedFeatures,
        estimand: TwangEstimand,
    ) -> Self {
        let groups = Groups::new(t);
        let denom = match estimand {
            TwangEstimand::ATT => SdDenominator::Treated,
            TwangEstimand::ATE => SdDenominator::All,
            TwangEstimand::ATC => SdDenominator::Control,
        };
        let sd = (0..x.ncols())
            .map(|j| balance_sd(&x.column(j).to_vec(), &groups.treated, denom, false))
            .collect();
        Criterion { groups, sorted, sd }
    }

    /// Weighted means of covariate `j` in the treated and control groups.
    fn means(&self, x: &Array2<f64>, j: usize, w: &[f64]) -> (f64, f64) {
        let mean = |idx: &[usize]| {
            let (mut s, mut sw) = (0.0, 0.0);
            for &i in idx {
                s += w[i] * x[[i, j]];
                sw += w[i];
            }
            if sw > 0.0 { s / sw } else { f64::NAN }
        };
        (
            mean(&self.groups.treated_idx),
            mean(&self.groups.control_idx),
        )
    }

    /// Standardized effect size of covariate `j`.
    fn effect_size(&self, x: &Array2<f64>, j: usize, w: &[f64]) -> f64 {
        let (mt, mc) = self.means(x, j, w);
        (mt - mc) / self.sd[j]
    }

    /// Weighted two-sample KS statistic of covariate `j`, swept along its presorted order:
    /// the group totals and running sums are accumulated in sorted order and the difference is
    /// taken at the last unit of each run of equal values. 0 when a group has no weight.
    fn ks(&self, x: &Array2<f64>, j: usize, w: &[f64]) -> f64 {
        let order = &self.sorted.order[j];
        let treated = &self.groups.treated;
        let (mut t_total, mut c_total) = (0.0, 0.0);
        for &i in order {
            if treated[i] {
                t_total += w[i];
            } else {
                c_total += w[i];
            }
        }
        if t_total == 0.0 || c_total == 0.0 {
            return 0.0;
        }
        let (mut t_cum, mut c_cum, mut max_diff) = (0.0, 0.0, 0.0_f64);
        for (pos, &i) in order.iter().enumerate() {
            if treated[i] {
                t_cum += w[i];
            } else {
                c_cum += w[i];
            }
            let last = pos + 1 == order.len() || x[[order[pos + 1], j]] != x[[i, j]];
            if last {
                let diff = (t_cum / t_total - c_cum / c_total).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        max_diff
    }

    /// The stop method's statistic only: effect sizes for ES, KS for KS; mean and max over
    /// the finite values.
    fn value(&self, x: &Array2<f64>, w: &[f64], method: StopMethod) -> f64 {
        let k = x.ncols();
        let (mean, max) = match method {
            StopMethod::ESMean | StopMethod::ESMax => {
                finite_abs_mean_max((0..k).map(|j| self.effect_size(x, j, w)))
            }
            StopMethod::KSMean | StopMethod::KSMax => {
                finite_abs_mean_max((0..k).map(|j| self.ks(x, j, w)))
            }
        };
        match method {
            StopMethod::ESMean | StopMethod::KSMean => mean,
            StopMethod::ESMax | StopMethod::KSMax => max,
        }
    }
}

/// Full balance table: weighted means, effect sizes, KS and weighted variance ratios.
fn compute_twang_balance(
    x: &Array2<f64>,
    crit: &Criterion,
    names: &[String],
    weights: Option<&[f64]>,
) -> TwangBalanceTable {
    let ones;
    let w = match weights {
        Some(w) => w,
        None => {
            ones = vec![1.0; x.nrows()];
            &ones
        }
    };
    let g = &crit.groups;
    let covariates: Vec<TwangCovariateBalance> = (0..x.ncols())
        .map(|j| {
            let (mean_t, mean_c) = crit.means(x, j, w);
            let pick = |idx: &[usize]| -> (Vec<f64>, Vec<f64>) {
                idx.iter().map(|&i| (x[[i, j]], w[i])).unzip()
            };
            let ((xt, wt), (xc, wc)) = (pick(&g.treated_idx), pick(&g.control_idx));
            TwangCovariateBalance {
                name: names.get(j).cloned().unwrap_or_else(|| format!("X{}", j)),
                mean_treated: mean_t,
                mean_control: mean_c,
                std_eff_size: (mean_t - mean_c) / crit.sd[j],
                ks_statistic: crit.ks(x, j, w),
                var_ratio: weighted_var(&xt, &wt) / weighted_var(&xc, &wc),
            }
        })
        .collect();

    let (es_mean, es_max) = finite_abs_mean_max(covariates.iter().map(|c| c.std_eff_size));
    let (ks_mean, ks_max) = finite_abs_mean_max(covariates.iter().map(|c| c.ks_statistic));
    let n_balanced = covariates
        .iter()
        .filter(|c| c.std_eff_size.abs() < 0.1)
        .count();

    TwangBalanceTable {
        covariates,
        es_mean,
        es_max,
        ks_mean,
        ks_max,
        n_balanced,
    }
}

/// Compute effective sample size.
fn compute_twang_ess(weights: &[f64], t: &Array1<f64>) -> (f64, f64, f64) {
    let n = t.len();

    let treated_idx: Vec<usize> = (0..n).filter(|&i| t[i] >= 0.5).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| t[i] < 0.5).collect();

    // ESS = (sum of weights)^2 / sum of squared weights
    let compute_ess = |indices: &[usize]| -> f64 {
        let w_sum: f64 = indices.iter().map(|&i| weights[i]).sum();
        let w_sq_sum: f64 = indices.iter().map(|&i| weights[i].powi(2)).sum();
        if w_sq_sum > 0.0 {
            w_sum.powi(2) / w_sq_sum
        } else {
            0.0
        }
    };

    let ess_t = compute_ess(&treated_idx);
    let ess_c = compute_ess(&control_idx);
    let ess_total = ess_t + ess_c;

    (ess_total, ess_t, ess_c)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use polars::prelude::*;

    #[test]
    fn test_stop_method_and_estimand_parse_aliases_and_reject_unknown() {
        StopMethod::choices().validate().unwrap();
        TwangEstimand::choices().validate().unwrap();
        assert_eq!("ES.Mean".parse::<StopMethod>().unwrap(), StopMethod::ESMean);
        assert_eq!("ksmax".parse::<StopMethod>().unwrap(), StopMethod::KSMax);
        assert_eq!("es-max".parse::<StopMethod>().unwrap(), StopMethod::ESMax);
        assert_eq!("ATC".parse::<TwangEstimand>().unwrap(), TwangEstimand::ATC);
        match "es.median".parse::<StopMethod>() {
            Err(crate::errors::EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "unknown stop_method 'es.median'; expected one of: es.mean, es.max, ks.mean, ks.max"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        assert!("atu".parse::<TwangEstimand>().is_err());
    }

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

    /// `create_test_dataset()` with `x1[3] = NaN`.
    fn dataset_with_nan_x1() -> Dataset {
        let ds = create_test_dataset();
        let mut x1: Vec<f64> = ds
            .df()
            .column("x1")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        x1[3] = f64::NAN;
        let mut df = ds.df().clone();
        df.with_column(Series::new("x1".into(), x1)).unwrap();
        Dataset::new(df)
    }

    /// The pre-sweep `fit_decision_stump` (O(k n^2) per tree), kept as the reference for the
    /// prefix-sum sweep, with gbm's Newton leaf values sum r / sum p (1 - p) in place of the
    /// leaf means (the split comparisons are unchanged).
    fn naive_stump(
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        p: &[f64],
        min_node_size: usize,
    ) -> DecisionStump {
        let newton = |left: bool, j: usize, threshold: f64| {
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..x.nrows() {
                if (x[[i, j]] <= threshold) == left {
                    num += residuals[i];
                    den += p[i] * (1.0 - p[i]);
                }
            }
            if den != 0.0 { num / den } else { 0.0 }
        };
        let n = x.nrows();
        let k = x.ncols();

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_sse = f64::INFINITY;

        // Try each feature
        for j in 0..k {
            // Get unique sorted values for this feature
            let mut values: Vec<f64> = x.column(j).to_vec();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Try midpoints as thresholds
            for window in values.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;

                // Count observations in each partition
                let mut left_count = 0;
                let mut right_count = 0;
                let mut left_sum = 0.0;
                let mut right_sum = 0.0;

                for i in 0..n {
                    if x[[i, j]] <= threshold {
                        left_count += 1;
                        left_sum += residuals[i];
                    } else {
                        right_count += 1;
                        right_sum += residuals[i];
                    }
                }

                // Skip if partition too small
                if left_count < min_node_size || right_count < min_node_size {
                    continue;
                }

                // Compute mean predictions for each partition
                let left_mean = left_sum / left_count as f64;
                let right_mean = right_sum / right_count as f64;

                // Compute SSE
                let mut sse = 0.0;
                for i in 0..n {
                    let pred = if x[[i, j]] <= threshold {
                        left_mean
                    } else {
                        right_mean
                    };
                    sse += (residuals[i] - pred).powi(2);
                }

                if sse < best_sse {
                    best_sse = sse;
                    best_feature = j;
                    best_threshold = threshold;
                }
            }
        }

        // If no valid split found, one leaf for every row
        if best_sse == f64::INFINITY {
            let (num, den) = (0..n).fold((0.0, 0.0), |(a, b), i| {
                (a + residuals[i], b + p[i] * (1.0 - p[i]))
            });
            let v = if den != 0.0 { num / den } else { 0.0 };
            return DecisionStump {
                feature_idx: 0,
                threshold: 0.0,
                left_value: v,
                right_value: v,
            };
        }

        DecisionStump {
            feature_idx: best_feature,
            threshold: best_threshold,
            left_value: newton(true, best_feature, best_threshold),
            right_value: newton(false, best_feature, best_threshold),
        }
    }

    /// The fast depth-1 tree with Newton leaves, as a stump.
    fn fit_stump(
        x: &Array2<f64>,
        sorted: &SortedFeatures,
        z: &Array1<f64>,
        p: &[f64],
        min_node_size: usize,
    ) -> DecisionStump {
        fit_tree(x, sorted, z, p, 1, min_node_size).as_stump()
    }

    fn fit_tree(
        x: &Array2<f64>,
        sorted: &SortedFeatures,
        z: &Array1<f64>,
        p: &[f64],
        depth: usize,
        min_node_size: usize,
    ) -> BoostTree {
        let z = z.to_vec();
        let mut tree = grow_tree(x, sorted, &z, depth, min_node_size);
        let mut leaf = vec![0; x.nrows()];
        set_newton_leaves(&mut tree, x, &z, p, &mut leaf);
        tree
    }

    /// A port of gbm's best-first growth (WP14 prototype `grow`): an exhaustive search per
    /// terminal node by improvement n_L n_R / n (mean_L - mean_R)^2 (strict `>`, the first best
    /// wins), split at the midpoint, left `x < s`, both children >= `min_node_size`; the node
    /// with the largest improvement splits, `depth` times. Returns the splits in order and the
    /// leaves' rows.
    fn naive_grow(
        x: &Array2<f64>,
        z: &[f64],
        depth: usize,
        min_node: usize,
    ) -> (Vec<(usize, f64)>, Vec<Vec<usize>>) {
        let best = |idx: &[usize]| -> Option<(f64, usize, f64)> {
            let mut best: Option<(f64, usize, f64)> = None;
            for j in 0..x.ncols() {
                let mut o = idx.to_vec();
                o.sort_by(|&a, &b| x[[a, j]].total_cmp(&x[[b, j]]));
                let n = o.len();
                let tot: f64 = o.iter().map(|&i| z[i]).sum();
                let mut cl = 0.0;
                for pos in 0..n.saturating_sub(1) {
                    cl += z[o[pos]];
                    let (a, b) = (x[[o[pos], j]], x[[o[pos + 1], j]]);
                    if a == b {
                        continue;
                    }
                    let (nl, nr) = (pos + 1, n - pos - 1);
                    if nl < min_node || nr < min_node {
                        continue;
                    }
                    let (ml, mr) = (cl / nl as f64, (tot - cl) / nr as f64);
                    let imp = (nl * nr) as f64 / n as f64 * (ml - mr).powi(2);
                    if best.is_none_or(|(bi, _, _)| imp > bi) {
                        best = Some((imp, j, (a + b) / 2.0));
                    }
                }
            }
            best
        };
        let mut leaves = vec![(0..x.nrows()).collect::<Vec<usize>>()];
        let mut cand = vec![best(&leaves[0])];
        let mut splits = Vec::new();
        for _ in 0..depth {
            let mut pick: Option<(usize, f64)> = None;
            for (b, c) in cand.iter().enumerate() {
                if let Some((imp, _, _)) = c
                    && pick.is_none_or(|(_, bi)| *imp > bi)
                {
                    pick = Some((b, *imp));
                }
            }
            let Some((b, _)) = pick else { break };
            let (_, j, sp) = cand[b].unwrap();
            splits.push((j, sp));
            let (l, r): (Vec<usize>, Vec<usize>) = leaves[b].iter().partition(|&&i| x[[i, j]] < sp);
            cand[b] = best(&l);
            leaves[b] = l;
            cand.push(best(&r));
            leaves.push(r);
        }
        (splits, leaves)
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = 1_103_515_245u64.wrapping_mul(*seed).wrapping_add(12_345) % 2_147_483_648;
        *seed as f64 / 2_147_483_648.0
    }

    #[test]
    fn test_stump_sweep_matches_naive_search() {
        let (n, k) = (300, 4);
        let mut s = 99u64;
        // features 0,1 continuous; 2 and 3 heavily tied
        let x = Array2::from_shape_fn((n, k), |(i, j)| match j {
            2 => (i % 7) as f64,
            3 => ((i * 13) % 5) as f64 * 0.5,
            _ => lcg(&mut s),
        });
        let r = Array1::from_shape_fn(n, |_| lcg(&mut s) - 0.5);
        let p: Vec<f64> = (0..n).map(|_| 0.05 + 0.9 * lcg(&mut s)).collect();
        let sorted = SortedFeatures::new(&x);
        for min_node in [1, 10, 40, 151] {
            // 151 > n/2: no valid split, one leaf
            let (a, b) = (
                naive_stump(&x, &r, &p, min_node),
                fit_stump(&x, &sorted, &r, &p, min_node),
            );
            assert_eq!(
                (a.feature_idx, a.threshold),
                (b.feature_idx, b.threshold),
                "min_node {min_node}"
            );
            assert_eq!(
                (a.left_value, a.right_value),
                (b.left_value, b.right_value),
                "min_node {min_node}"
            );
        }
        // Each tied feature on its own, so the sweep's group boundaries are exercised
        // without the continuous features winning every time.
        for j in [2, 3] {
            let xj = x.slice(ndarray::s![.., j..j + 1]).to_owned();
            let sorted = SortedFeatures::new(&xj);
            for min_node in [1, 10, 40] {
                let (a, b) = (
                    naive_stump(&xj, &r, &p, min_node),
                    fit_stump(&xj, &sorted, &r, &p, min_node),
                );
                assert_eq!(
                    (a.feature_idx, a.threshold, a.left_value, a.right_value),
                    (b.feature_idx, b.threshold, b.left_value, b.right_value),
                    "feature {j}, min_node {min_node}"
                );
            }
        }
        // Twang test data: x1 and x2 both separate the groups, so two features induce the
        // same partition and the naive search sees an exact SSE tie (first feature wins).
        let ds = create_test_dataset();
        let x = DesignMatrix::from_dataframe(ds.df(), &["x1", "x2"], false)
            .unwrap()
            .data;
        let t = DesignMatrix::extract_column(ds.df(), "treatment").unwrap();
        let r0 = t.mapv(|v| v - 0.5); // first-iteration residuals: f0 = ln(20/20) = 0, p = 0.5
        let p0 = vec![0.5; t.len()];
        let sorted = SortedFeatures::new(&x);
        let (a, b) = (
            naive_stump(&x, &r0, &p0, 10),
            fit_stump(&x, &sorted, &r0, &p0, 10),
        );
        assert_eq!(
            (a.feature_idx, a.threshold, a.left_value, a.right_value),
            (b.feature_idx, b.threshold, b.left_value, b.right_value)
        );
    }

    #[test]
    fn test_tree_growth_matches_naive_best_first() {
        let (n, k) = (300, 4);
        let mut s = 99u64;
        let x = Array2::from_shape_fn((n, k), |(i, j)| match j {
            2 => (i % 7) as f64,
            3 => ((i * 13) % 5) as f64 * 0.5,
            _ => lcg(&mut s),
        });
        let z: Vec<f64> = (0..n).map(|_| lcg(&mut s) - 0.5).collect();
        let sorted = SortedFeatures::new(&x);
        let cases = [(1, 10), (2, 10), (3, 10), (6, 10), (3, 1), (6, 1), (3, 151)];
        for (depth, min_node) in cases {
            let tree = grow_tree(&x, &sorted, &z, depth, min_node);
            let (splits, leaves) = naive_grow(&x, &z, depth, min_node);
            // The fast tree's splits in the order they were made (split k creates nodes
            // 2k + 1 and 2k + 2).
            let mut fast: Vec<(usize, usize, f64)> = tree
                .nodes
                .iter()
                .filter_map(|nd| match *nd {
                    BoostNode::Split {
                        feature,
                        threshold,
                        left,
                        ..
                    } => Some((left, feature, threshold)),
                    BoostNode::Leaf { .. } => None,
                })
                .collect();
            fast.sort_by_key(|f| f.0);
            let fast: Vec<(usize, f64)> = fast.iter().map(|f| (f.1, f.2)).collect();
            assert_eq!(fast, splits, "depth {depth}, min_node {min_node}");
            let mut fast_leaves: Vec<Vec<usize>> = vec![Vec::new(); tree.nodes.len()];
            for i in 0..n {
                fast_leaves[tree.leaf_of(|j| x[[i, j]])].push(i);
            }
            let mut fast_leaves: Vec<Vec<usize>> =
                fast_leaves.into_iter().filter(|v| !v.is_empty()).collect();
            let mut leaves = leaves;
            fast_leaves.sort();
            leaves.sort();
            assert_eq!(fast_leaves, leaves, "depth {depth}, min_node {min_node}");
        }
    }

    /// The KS reference for the presorted sweep: the per-call sort the balance code used
    /// before the sweep. Compute two-sample Kolmogorov-Smirnov statistic.
    ///
    /// This measures the maximum difference between the weighted ECDFs
    /// of treated and control groups for a given covariate.
    fn compute_ks_statistic(
        x: &Array2<f64>,
        col: usize,
        treated_idx: &[usize],
        control_idx: &[usize],
        weights: &[f64],
    ) -> f64 {
        // Get weighted values for each group
        let mut t_vals: Vec<(f64, f64)> = treated_idx
            .iter()
            .map(|&i| (x[[i, col]], weights[i]))
            .collect();
        let mut c_vals: Vec<(f64, f64)> = control_idx
            .iter()
            .map(|&i| (x[[i, col]], weights[i]))
            .collect();

        // Sort by value
        t_vals.sort_by(|a, b| a.0.total_cmp(&b.0));
        c_vals.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Compute total weights
        let t_total: f64 = t_vals.iter().map(|(_, w)| w).sum();
        let c_total: f64 = c_vals.iter().map(|(_, w)| w).sum();

        if t_total == 0.0 || c_total == 0.0 {
            return 0.0;
        }

        // Merge sorted values to compute ECDF differences
        let mut all_vals: Vec<f64> = t_vals
            .iter()
            .map(|(v, _)| *v)
            .chain(c_vals.iter().map(|(v, _)| *v))
            .collect();
        all_vals.sort_by(f64::total_cmp);
        all_vals.dedup();

        let mut max_diff: f64 = 0.0;
        let mut t_cum = 0.0;
        let mut c_cum = 0.0;
        let mut t_idx = 0;
        let mut c_idx = 0;

        for &val in &all_vals {
            // Update cumulative sums for values <= current
            while t_idx < t_vals.len() && t_vals[t_idx].0 <= val {
                t_cum += t_vals[t_idx].1;
                t_idx += 1;
            }
            while c_idx < c_vals.len() && c_vals[c_idx].0 <= val {
                c_cum += c_vals[c_idx].1;
                c_idx += 1;
            }

            // ECDF values
            let t_ecdf = t_cum / t_total;
            let c_ecdf = c_cum / c_total;

            let diff = (t_ecdf - c_ecdf).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        max_diff
    }

    #[test]
    fn test_twang_ks_presorted_matches_naive() {
        let n = 200;
        let mut s = 5u64;
        let t = Array1::from_shape_fn(n, |i| ((i * 7) % 3 == 0) as u8 as f64);
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => lcg(&mut s),
            1 => (i % 6) as f64,
            _ => ((i % 2) as f64) * lcg(&mut s).round(),
        });
        let w: Vec<f64> = (0..n).map(|_| 0.2 + 3.0 * lcg(&mut s)).collect();
        let sorted = SortedFeatures::new(&x);
        let crit = Criterion::new(&x, &t, &sorted, TwangEstimand::ATT);
        let g = &crit.groups;
        for j in 0..3 {
            let fast = crit.ks(&x, j, &w);
            let slow = compute_ks_statistic(&x, j, &g.treated_idx, &g.control_idx, &w);
            assert_eq!(
                fast.to_bits(),
                slow.to_bits(),
                "column {j}: {fast} vs {slow}"
            );
        }
    }

    #[test]
    fn test_twang_es_uses_fixed_sd() {
        // twang's ps.summary.new2: the effect size divides the weighted mean difference by an SD
        // of the unweighted sample, n - 1, binary columns as continuous. CAUS-108.
        let t = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.5, 0.0, 3.0, 1.0, 4.5, 1.0, 0.5, 0.0, 1.5, 1.0, 2.0, 0.0, 0.7, 0.0,
                3.3, 1.0,
            ],
        )
        .unwrap();
        let w = [1.0, 1.0, 1.0, 1.0, 0.4, 2.2, 1.3, 0.8, 0.3];
        let var = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)
        };
        let wmean = |v: &[f64], w: &[f64]| {
            v.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>()
        };
        let sorted = SortedFeatures::new(&x);
        for est in [TwangEstimand::ATT, TwangEstimand::ATE, TwangEstimand::ATC] {
            let crit = Criterion::new(&x, &t, &sorted, est);
            let names = vec!["a".to_string(), "b".to_string()];
            let table = compute_twang_balance(&x, &crit, &names, Some(&w));
            for j in 0..2 {
                let col: Vec<f64> = x.column(j).to_vec();
                let sd = match est {
                    TwangEstimand::ATT => var(&col[..4]),
                    TwangEstimand::ATE => var(&col),
                    TwangEstimand::ATC => var(&col[4..]),
                }
                .sqrt();
                let expect = (wmean(&col[..4], &w[..4]) - wmean(&col[4..], &w[4..])) / sd;
                let got = table.covariates[j].std_eff_size;
                assert!((got - expect).abs() <= 1e-14 * expect.abs(), "{est} {j}");
                let es = crit.effect_size(&x, j, &w);
                assert_eq!(es, got);
            }
        }
    }

    /// The twang_gbm_sim runs: (key prefix, depth, estimand, stop method).
    const GBM_RUNS: [(&str, usize, TwangEstimand, StopMethod); 3] = [
        ("d1_att_esmean", 1, TwangEstimand::ATT, StopMethod::ESMean),
        ("d3_ate_ksmax", 3, TwangEstimand::ATE, StopMethod::KSMax),
        ("d1_atc_esmax", 1, TwangEstimand::ATC, StopMethod::ESMax),
    ];

    fn gbm_config(depth: usize, est: TwangEstimand, stop: StopMethod) -> TwangConfig {
        let mut c = TwangConfig::default();
        c.n_trees = 3000;
        c.shrinkage = 0.01;
        c.interaction_depth = depth;
        c.estimand = est;
        c.stop_method = stop;
        c
    }

    #[test]
    fn test_validate_twang_gbm_path() {
        // R: gbm(distribution = "bernoulli", n.trees = 3000, interaction.depth = 1 | 3,
        // shrinkage = 0.01, bag.fraction = 1, n.minobsinnode = 10) (case twang_gbm_sim).
        // CAUS-93: the leaves were mean residuals and interaction_depth was ignored.
        let (case, ds) = crate::test_support::load_case("twang_gbm_sim");
        let covs = ["x1", "x2", "x3"];
        let x = DesignMatrix::from_dataframe(ds.df(), &covs, false)
            .unwrap()
            .data;
        let t = DesignMatrix::extract_column(ds.df(), "treat").unwrap();
        let n = t.len();
        let at = [1, 10, 100, 1000, 3000];
        let mut check = p2a_validation::Check::new(&case);
        for (tag, depth, est, stop) in &GBM_RUNS[..2] {
            let (f0, links) = gbm_link_path(&x, &t, &gbm_config(*depth, *est, *stop), &at);
            check.scalar(&format!("{tag}_initf"), f0, p2a_validation::Tol::coef(n));
            for (m, f) in at.iter().zip(&links) {
                check.vector(
                    &format!("{tag}_link_m{m}_head"),
                    &f[..10],
                    p2a_validation::Tol::coef(n),
                );
            }
        }
        check.finish();
    }

    #[test]
    fn test_validate_twang_selection() {
        // R: gbm's path + twang's ES/KS criterion (ps.summary.new2), global argmin over
        // 1..n.trees (case twang_gbm_sim). CAUS-93: the run stopped at the first iteration
        // under 0.1 after min_iterations.
        use p2a_validation::Tol;
        let (case, ds) = crate::test_support::load_case("twang_gbm_sim");
        let covs = ["x1", "x2", "x3"];
        let mut check = p2a_validation::Check::new(&case);
        for (tag, depth, est, stop) in GBM_RUNS {
            let r = run_twang(&ds, "treat", &covs, Some(gbm_config(depth, est, stop))).unwrap();
            let n = r.n_obs;
            let key = |s: &str| format!("{tag}_{s}");
            assert_eq!(
                r.optimal_n_trees as f64,
                case.scalar(&key("optimal_n_trees")),
                "{tag}"
            );
            assert_eq!(
                r.balance_history.len(),
                3001,
                "{tag}: every iteration evaluated"
            );
            assert_eq!(
                r.balance_history[r.optimal_n_trees],
                r.balance_after.get_metric(stop)
            );
            let es_before: Vec<f64> = r
                .balance_before
                .covariates
                .iter()
                .map(|c| c.std_eff_size)
                .collect();
            let es: Vec<f64> = r
                .balance_after
                .covariates
                .iter()
                .map(|c| c.std_eff_size)
                .collect();
            let ks: Vec<f64> = r
                .balance_after
                .covariates
                .iter()
                .map(|c| c.ks_statistic)
                .collect();
            check
                .scalar(
                    &key("criterion_at_opt"),
                    r.balance_after.get_metric(stop),
                    Tol::coef(n),
                )
                .scalar(&key("criterion_m1"), r.balance_history[1], Tol::coef(n))
                .scalar(
                    &key("criterion_m3000"),
                    r.balance_history[3000],
                    Tol::coef(n),
                )
                .vector(&key("ps_head"), &r.propensity_scores[..10], Tol::coef(n))
                .vector(&key("es_after"), &es, Tol::coef(n))
                .vector(&key("ks_after"), &ks, Tol::coef(n))
                .vector(&key("es_before"), &es_before, Tol::coef(n))
                .scalar(&key("ess_t"), r.ess_treated, Tol::coef(n))
                .scalar(&key("ess_c"), r.ess_control, Tol::coef(n));
            let boundary = r.warnings.iter().any(|w| w.contains("still decreasing"));
            assert_eq!(boundary, tag == "d1_att_esmean", "{tag}: {:?}", r.warnings);
            if depth == 1 {
                assert_eq!(r.stumps.len(), 3000);
            } else {
                assert!(r.stumps.is_empty());
            }
            assert_eq!(r.trees.len(), 3000);
        }
        check.finish();
    }

    #[test]
    fn test_twang_early_stop_is_opt_in() {
        let ds = {
            let path = p2a_validation::repo_path("validation/datasets/generated/weighting_sim.csv");
            Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
        };
        let mut c = gbm_config(1, TwangEstimand::ATT, StopMethod::ESMean);
        c.balance_threshold = 0.1;
        c.min_iterations = 100;
        let r = run_twang(&ds, "treat", &["x1", "x2", "x3"], Some(c)).unwrap();
        let h = &r.balance_history;
        let stop = h.len() - 1;
        assert!(stop >= 100 && stop < 3000, "stopped at {stop}");
        assert!(h[stop] < 0.1);
        assert!(h[100..stop].iter().all(|&v| v >= 0.1));
        // The optimum is the first minimum over the evaluated iterations from min_iterations.
        let (mut best, mut arg) = (f64::INFINITY, 0);
        for (m, &v) in h.iter().enumerate().take(stop + 1).skip(100) {
            if v < best {
                best = v;
                arg = m;
            }
        }
        assert_eq!(r.optimal_n_trees, arg);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains(&format!("stopped early at iteration {stop}"))),
            "{:?}",
            r.warnings
        );
        let text = r.to_string();
        assert!(text.contains("Warning: boosting stopped early"));
        assert!(text.contains(&format!("over iterations 100..{stop}")));
    }

    #[test]
    fn test_twang_large_n_is_tractable() {
        // 20,000 rows x 5 covariates, 3 trees. Do not run before the fix: the naive search
        // needs ~1e9 comparisons per feature per tree here. No timing assertion.
        let n = 20_000;
        let mut s = 7u64;
        let treatment: Vec<f64> = (0..n).map(|i| (i % 3 == 0) as u8 as f64).collect();
        let mut cols: Vec<Vec<f64>> = (0..5).map(|_| Vec::with_capacity(n)).collect();
        for i in 0..n {
            for (j, col) in cols.iter_mut().enumerate() {
                let v = lcg(&mut s);
                col.push(if j == 0 { v + 0.3 * treatment[i] } else { v });
            }
        }
        let mut columns = vec![Column::new("treatment".into(), treatment)];
        for (j, col) in cols.into_iter().enumerate() {
            columns.push(Column::new(format!("x{j}").into(), col));
        }
        let big = Dataset::new(DataFrame::new(columns).unwrap());
        let cfg = TwangConfig {
            n_trees: 3,
            min_iterations: 0,
            ..Default::default()
        };
        let r = run_twang(
            &big,
            "treatment",
            &["x0", "x1", "x2", "x3", "x4"],
            Some(cfg),
        )
        .unwrap();
        assert_eq!(r.stumps.len(), 3);
        assert_eq!(r.n_obs, 20_000);
    }

    #[test]
    fn test_twang_n_trees_is_an_error_not_an_abort() {
        let cfg = TwangConfig {
            n_trees: usize::MAX,
            ..Default::default()
        };
        let err = run_twang(
            &create_test_dataset(),
            "treatment",
            &["x1", "x2"],
            Some(cfg),
        )
        .unwrap_err();
        assert!(matches!(err, EconError::ResourceLimit(_)), "got {err:?}");
    }

    #[test]
    fn test_twang_rejects_invalid_parameters() {
        let d = TwangConfig::default;
        let cases = [
            (TwangConfig { n_trees: 0, ..d() }, "n_trees"),
            (
                TwangConfig {
                    shrinkage: 0.0,
                    ..d()
                },
                "shrinkage",
            ),
            (
                TwangConfig {
                    shrinkage: f64::NAN,
                    ..d()
                },
                "shrinkage",
            ),
            (
                TwangConfig {
                    shrinkage: 1.5,
                    ..d()
                },
                "shrinkage",
            ),
            (
                TwangConfig {
                    balance_threshold: f64::NAN,
                    ..d()
                },
                "balance_threshold",
            ),
            (
                TwangConfig {
                    balance_threshold: -0.1,
                    ..d()
                },
                "balance_threshold",
            ),
            (
                TwangConfig {
                    interaction_depth: 0,
                    ..d()
                },
                "interaction_depth",
            ),
            (
                TwangConfig {
                    n_trees: 50,
                    min_iterations: 51,
                    ..d()
                },
                "min_iterations",
            ),
        ];
        for (cfg, needle) in cases {
            expect_invalid_spec(
                run_twang(
                    &create_test_dataset(),
                    "treatment",
                    &["x1", "x2"],
                    Some(cfg),
                )
                .unwrap_err(),
                needle,
            );
        }
    }

    #[test]
    fn test_twang_rejects_non_finite_covariates() {
        // DesignMatrix refuses the NaN covariate by name before twang's own finite check.
        match run_twang(&dataset_with_nan_x1(), "treatment", &["x1", "x2"], None) {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "x1");
                assert_eq!(count, 1);
            }
            other => panic!("expected NullValues for x1, got {other:?}"),
        }
    }

    #[test]
    fn test_twang_names_a_missing_treatment_value() {
        // A NaN treatment is a missing value in an existing column, not "Column not found".
        let ds = create_test_dataset();
        let mut t: Vec<f64> = ds
            .df()
            .column("treatment")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        t[3] = f64::NAN;
        let mut df = ds.df().clone();
        df.with_column(Series::new("treatment".into(), t)).unwrap();
        match run_twang(&Dataset::new(df), "treatment", &["x1", "x2"], None) {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "treatment");
                assert_eq!(count, 1);
            }
            other => panic!("expected NullValues for treatment, got {other:?}"),
        }
    }

    #[test]
    fn test_twang_honours_cancellation() {
        let r = resource::with_budget(cancelled_budget(), || {
            run_twang(&create_test_dataset(), "treatment", &["x1", "x2"], None)
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "got {r:?}");
    }

    /// Create test dataset with known imbalance.
    fn create_test_dataset() -> Dataset {
        // Treated group has higher means on covariates
        let df = df! {
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            // Treated has higher x1 (mean ~1.5 vs ~0.5)
            "x1" => [
                1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.1, 1.7, 1.5, 1.4,
                1.9, 2.0, 1.6, 1.8, 1.3, 1.7, 1.4, 2.1, 1.5, 1.6,
                0.3, 0.5, 0.4, 0.7, 0.2, 0.6, 0.1, 0.8, 0.4, 0.5,
                0.6, 0.7, 0.3, 0.9, 0.4, 0.5, 0.2, 0.8, 0.5, 0.6
            ],
            // Similar imbalance on x2
            "x2" => [
                0.8, 0.9, 0.7, 1.0, 0.85, 0.95, 0.75, 1.1, 0.9, 0.85,
                1.0, 1.1, 0.9, 1.05, 0.8, 0.95, 0.85, 1.15, 0.9, 0.92,
                0.3, 0.4, 0.35, 0.5, 0.25, 0.45, 0.2, 0.55, 0.4, 0.35,
                0.45, 0.5, 0.3, 0.6, 0.35, 0.4, 0.25, 0.55, 0.4, 0.42
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_twang_basic() {
        let dataset = create_test_dataset();

        let config = TwangConfig {
            n_trees: 500,
            shrinkage: 0.05,
            stop_method: StopMethod::ESMean,
            min_iterations: 50,
            ..Default::default()
        };

        let result = run_twang(&dataset, "treatment", &["x1", "x2"], Some(config)).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 40);
        assert_eq!(result.n_treated, 20);
        assert_eq!(result.n_control, 20);
        assert_eq!(result.propensity_scores.len(), 40);
        assert_eq!(result.weights.len(), 40);

        // Propensity scores should be in (0, 1)
        assert!(result.propensity_scores.iter().all(|&p| p > 0.0 && p < 1.0));

        // Weights should be positive
        assert!(result.weights.iter().all(|&w| w > 0.0));

        // Balance should improve
        assert!(result.balance_after.es_mean <= result.balance_before.es_mean + 0.1);
    }

    #[test]
    fn test_twang_stop_methods() {
        let dataset = create_test_dataset();

        for stop_method in [
            StopMethod::ESMean,
            StopMethod::ESMax,
            StopMethod::KSMean,
            StopMethod::KSMax,
        ] {
            let config = TwangConfig {
                n_trees: 200,
                shrinkage: 0.1,
                stop_method,
                min_iterations: 20,
                ..Default::default()
            };

            let result = run_twang(&dataset, "treatment", &["x1", "x2"], Some(config)).unwrap();
            assert!(result.optimal_n_trees > 0);
            assert!(!result.balance_history.is_empty());
        }
    }

    #[test]
    fn test_twang_estimands() {
        let dataset = create_test_dataset();

        for estimand in [TwangEstimand::ATT, TwangEstimand::ATE, TwangEstimand::ATC] {
            let config = TwangConfig {
                n_trees: 200,
                shrinkage: 0.1,
                estimand,
                min_iterations: 20,
                ..Default::default()
            };

            let result = run_twang(&dataset, "treatment", &["x1", "x2"], Some(config)).unwrap();

            // For ATT, treated weights should be 1 (after normalization to n_treated)
            if estimand == TwangEstimand::ATT {
                let treated_weights: Vec<f64> = (0..40)
                    .filter(|&i| i < 20)
                    .map(|i| result.weights[i])
                    .collect();
                // All treated weights should be equal (normalized)
                let first = treated_weights[0];
                assert!(treated_weights.iter().all(|&w| (w - first).abs() < 0.01));
            }
        }
    }

    #[test]
    fn test_decision_stump() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.5, 2.0, 0.6, 3.0, 0.7, 4.0, 0.8, 5.0, 0.9, 6.0, 1.0],
        )
        .unwrap();

        // Residuals: positive for first 3, negative for last 3
        let residuals = Array1::from_vec(vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);

        let sorted = SortedFeatures::new(&x);
        let stump = fit_stump(&x, &sorted, &residuals, &[0.5; 6], 2);

        // Should split on x[0] (feature 0) around 3.5
        assert_eq!(stump.feature_idx, 0);
        assert!(stump.threshold > 2.5 && stump.threshold < 4.5);
        assert!(stump.left_value > 0.0); // Positive residuals on left
        assert!(stump.right_value < 0.0); // Negative residuals on right
    }

    #[test]
    fn test_ks_statistic() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let treated_idx = vec![0, 1, 2];
        let control_idx = vec![3, 4, 5];
        let weights = vec![1.0; 6];

        // Treated: 1, 2, 3. Control: 4, 5, 6
        // Distributions don't overlap, so KS should be 1.0
        let ks = compute_ks_statistic(&x, 0, &treated_idx, &control_idx, &weights);
        assert!((ks - 1.0).abs() < 0.01);

        // Same distribution should have KS = 0
        let same_idx = vec![0, 1, 2, 3, 4, 5];
        let ks_same = compute_ks_statistic(&x, 0, &same_idx, &same_idx, &weights);
        assert!(ks_same.abs() < 0.01);
    }

    #[test]
    fn test_twang_display() {
        let dataset = create_test_dataset();

        let config = TwangConfig {
            n_trees: 100,
            shrinkage: 0.1,
            min_iterations: 10,
            ..Default::default()
        };

        let result = run_twang(&dataset, "treatment", &["x1", "x2"], Some(config)).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("twang"));
        assert!(output.contains("GBM"));
        assert!(output.contains("Optimal Iterations"));
        assert!(output.contains("Balance"));
    }

    #[test]
    fn test_balance_history() {
        let dataset = create_test_dataset();

        let config = TwangConfig {
            n_trees: 100,
            shrinkage: 0.1,
            min_iterations: 10,
            ..Default::default()
        };

        let result = run_twang(&dataset, "treatment", &["x1", "x2"], Some(config)).unwrap();

        // Balance history should have entries for each iteration + initial
        assert!(!result.balance_history.is_empty());
        assert!(result.balance_history.len() <= 101); // n_trees + 1

        // Balance should generally decrease (with possible fluctuations)
        let initial = result.balance_history[0];
        let final_balance = result.balance_after.get_metric(result.stop_method);
        assert!(final_balance <= initial + 0.5); // Some tolerance for fluctuation
    }

    #[test]
    fn test_twang_missing_column() {
        let dataset = create_test_dataset();
        let result = run_twang(&dataset, "nonexistent", &["x1"], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_twang_no_treated() {
        let df = df! {
            "treatment" => [0.0, 0.0, 0.0, 0.0, 0.0],
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_twang(&dataset, "treatment", &["x1"], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ess_computation() {
        // Uniform weights should give ESS = n
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let t = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);

        let (ess_total, ess_t, ess_c) = compute_twang_ess(&weights, &t);

        assert!((ess_total - 4.0).abs() < 0.01);
        assert!((ess_t - 2.0).abs() < 0.01);
        assert!((ess_c - 2.0).abs() < 0.01);

        // Unequal weights should give ESS < n
        let weights2 = vec![1.0, 3.0, 1.0, 3.0];
        let (ess_total2, _, _) = compute_twang_ess(&weights2, &t);
        assert!(ess_total2 < 4.0);
    }

    #[test]
    fn test_convenience_function() {
        let dataset = create_test_dataset();

        let result = twang(
            &dataset,
            "treatment",
            &["x1", "x2"],
            StopMethod::ESMax,
            TwangEstimand::ATT,
        )
        .unwrap();

        assert_eq!(result.stop_method, StopMethod::ESMax);
        assert_eq!(result.estimand, TwangEstimand::ATT);
    }

    // ── T2-WP13: missing values and column types (R case causal_mediation_na) ──

    #[test]
    fn test_validate_na_twang_columns() {
        // T2-WP01 already maps twang's extraction through econ_error_from_design; this
        // holds twang to the same refusals and dtype rule as the other weighting methods.
        use crate::econometrics::treatment::na_fixtures as nf;
        let (case, ds) = crate::test_support::load_case("causal_mediation_na");
        let run = |ds: &Dataset, _y: &str, d: &str, x: &[&str]| {
            run_twang(
                ds,
                d,
                x,
                Some(TwangConfig {
                    n_trees: 300,
                    ..Default::default()
                }),
            )
        };
        nf::assert_refusals(&case, &ds, false, run);
        nf::assert_treatment_dtypes_agree(&case, &ds, run, |r| r.weights.clone());
    }
}
