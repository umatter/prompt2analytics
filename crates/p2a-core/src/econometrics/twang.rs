//! Toolkit for Weighting and Analysis of Nonequivalent Groups (twang).
//!
//! Implements GBM-based propensity score estimation with balance optimization,
//! similar to the twang R package. Since we don't have a full gradient boosting
//! machine library, we implement a simplified version using boosted decision stumps.
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
//! 1. Initialize: F_0(x) = log(n_treated / n_control)
//! 2. For m = 1 to M:
//!    - Compute pseudo-residuals: r_i = y_i - p_i where p_i = sigmoid(F_{m-1}(x_i))
//!    - Fit decision stump (single split) to residuals on best feature
//!    - Update: F_m(x) = F_{m-1}(x) + shrinkage * stump(x)
//!    - Compute balance metrics with current propensity-based weights
//!    - Track optimal stopping point based on stop_method
//! 3. Return propensity scores from optimal iteration
//!
//! # Stopping Rules
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
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};

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
    /// Maximum number of boosting iterations (default: 3000)
    pub n_trees: usize,

    /// Learning rate / shrinkage (default: 0.01)
    ///
    /// Smaller values require more iterations but often give better results.
    /// Typical values: 0.001 to 0.1
    pub shrinkage: f64,

    /// Stopping rule for selecting optimal iterations (default: ESMean)
    pub stop_method: StopMethod,

    /// Target estimand (default: ATT)
    pub estimand: TwangEstimand,

    /// Balance threshold for early stopping (default: 0.1)
    ///
    /// If the balance metric falls below this threshold, stop early.
    pub balance_threshold: f64,

    /// Minimum number of iterations before early stopping (default: 100)
    pub min_iterations: usize,

    /// Interaction depth for decision stumps (default: 1, meaning single split)
    ///
    /// Value of 1 means single-split stumps. Higher values allow more complex
    /// trees but increase computation and risk of overfitting.
    pub interaction_depth: usize,

    /// Minimum observations per terminal node (default: 10)
    pub min_node_size: usize,
}

impl Default for TwangConfig {
    fn default() -> Self {
        Self {
            n_trees: 3000,
            shrinkage: 0.01,
            stop_method: StopMethod::ESMean,
            estimand: TwangEstimand::ATT,
            balance_threshold: 0.1,
            min_iterations: 100,
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

    /// Standardized effect size: (mean_t - mean_c) / sd_pooled
    pub std_eff_size: f64,

    /// Kolmogorov-Smirnov statistic
    pub ks_statistic: f64,

    /// Variance ratio: var_t / var_c
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

    /// Learned stumps (internal, for prediction)
    #[serde(skip)]
    pub stumps: Vec<DecisionStump>,

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
        writeln!(f, "  Optimal Iterations: {}", self.optimal_n_trees)?;
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

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run twang GBM propensity score estimation.
///
/// This function estimates propensity scores using gradient boosted stumps
/// and automatically selects the optimal number of iterations based on
/// covariate balance.
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

    // Extract treatment variable
    let t = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

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
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, false)?;
    let x = design.data;
    let covariate_names = design.column_names.clone();

    // Run GBM boosting
    let (stumps, f0, balance_history, optimal_iter, propensity_scores) =
        run_gbm_boosting(&x, &t, &config)?;

    // Compute final weights at optimal iteration
    let weights = compute_twang_weights(&propensity_scores, &t, config.estimand);

    // Compute balance before (unweighted)
    let balance_before = compute_twang_balance(&x, &t, &covariate_names, None);

    // Compute balance after (weighted)
    let balance_after = compute_twang_balance(&x, &t, &covariate_names, Some(&weights));

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

    Ok(TwangResult {
        propensity_scores,
        weights,
        optimal_n_trees: optimal_iter,
        balance_before,
        balance_after,
        balance_history,
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
        f0,
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

// ═══════════════════════════════════════════════════════════════════════════════
// GBM Boosting Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Run gradient boosting with decision stumps.
///
/// Returns (stumps, f0, balance_history, optimal_iter, propensity_scores).
fn run_gbm_boosting(
    x: &Array2<f64>,
    t: &Array1<f64>,
    config: &TwangConfig,
) -> EconResult<(Vec<DecisionStump>, f64, Vec<f64>, usize, Vec<f64>)> {
    let n = t.len();
    let _k = x.ncols();

    let n_treated: f64 = t.iter().filter(|&&v| v >= 0.5).count() as f64;
    let n_control: f64 = (n as f64) - n_treated;

    // Initialize: F_0(x) = log(p / (1-p)) where p = n_treated / n
    // This is the log-odds of being treated
    let f0 = (n_treated / n_control).ln();

    // Current ensemble prediction F(x)
    let mut f_values = Array1::from_elem(n, f0);

    // Store stumps
    let mut stumps: Vec<DecisionStump> = Vec::with_capacity(config.n_trees);

    // Track balance at each iteration
    let mut balance_history: Vec<f64> = Vec::with_capacity(config.n_trees + 1);

    // Initial balance (iteration 0)
    let initial_ps = f_values.mapv(sigmoid);
    let initial_weights = compute_twang_weights_internal(&initial_ps, t, config.estimand);
    let initial_balance = compute_balance_metric(x, t, &initial_weights, config.stop_method);
    balance_history.push(initial_balance);

    // Track best balance
    let mut best_balance = initial_balance;
    let mut optimal_iter = 0;

    // Gradient boosting iterations
    for m in 0..config.n_trees {
        // Compute current probabilities
        let p: Array1<f64> = f_values.mapv(sigmoid);

        // Clip probabilities for numerical stability
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Compute pseudo-residuals (gradient of binomial deviance)
        // For logistic loss: r_i = y_i - p_i
        let residuals: Array1<f64> = t - &p_clipped;

        // Fit decision stump to residuals
        let stump = fit_decision_stump(x, &residuals, config.min_node_size);

        // Update ensemble with shrinkage
        let stump_predictions = stump.predict_all(x);
        f_values = &f_values + &(&stump_predictions * config.shrinkage);

        // Store stump
        stumps.push(stump);

        // Compute propensity scores and weights
        let ps = f_values.mapv(sigmoid);
        let weights = compute_twang_weights_internal(&ps, t, config.estimand);

        // Compute balance metric
        let balance = compute_balance_metric(x, t, &weights, config.stop_method);
        balance_history.push(balance);

        // Update optimal iteration if balance improved
        if m >= config.min_iterations && balance < best_balance {
            best_balance = balance;
            optimal_iter = m + 1;
        }

        // Early stopping if balance threshold reached
        if m >= config.min_iterations && balance < config.balance_threshold {
            break;
        }
    }

    // If no iteration beat initial, use last iteration
    if optimal_iter == 0 {
        optimal_iter = stumps.len();
    }

    // Recompute propensity scores at optimal iteration
    let mut f_optimal = Array1::from_elem(n, f0);
    for stump in stumps.iter().take(optimal_iter) {
        let predictions = stump.predict_all(x);
        f_optimal = &f_optimal + &(&predictions * config.shrinkage);
    }
    let propensity_scores: Vec<f64> = f_optimal.mapv(sigmoid).to_vec();

    Ok((stumps, f0, balance_history, optimal_iter, propensity_scores))
}

/// Fit a decision stump (single-split tree) to residuals.
///
/// Finds the best feature and threshold to split on, minimizing
/// the sum of squared residuals in each partition.
fn fit_decision_stump(
    x: &Array2<f64>,
    residuals: &Array1<f64>,
    min_node_size: usize,
) -> DecisionStump {
    let n = x.nrows();
    let k = x.ncols();

    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_sse = f64::INFINITY;
    let mut best_left_value = 0.0;
    let mut best_right_value = 0.0;

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
                best_left_value = left_mean;
                best_right_value = right_mean;
            }
        }
    }

    // If no valid split found, return a stump that predicts the mean
    if best_sse == f64::INFINITY {
        let mean = residuals.sum() / n as f64;
        return DecisionStump {
            feature_idx: 0,
            threshold: 0.0,
            left_value: mean,
            right_value: mean,
        };
    }

    DecisionStump {
        feature_idx: best_feature,
        threshold: best_threshold,
        left_value: best_left_value,
        right_value: best_right_value,
    }
}

/// Sigmoid function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weight and Balance Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute IPW weights based on propensity scores and estimand.
fn compute_twang_weights(
    propensity_scores: &[f64],
    t: &Array1<f64>,
    estimand: TwangEstimand,
) -> Vec<f64> {
    let ps_array = Array1::from_vec(propensity_scores.to_vec());
    compute_twang_weights_internal(&ps_array, t, estimand)
}

/// Internal weight computation from Array1.
fn compute_twang_weights_internal(
    propensity_scores: &Array1<f64>,
    t: &Array1<f64>,
    estimand: TwangEstimand,
) -> Vec<f64> {
    let n = t.len();
    let mut weights = vec![0.0; n];

    // Clip propensity scores
    let ps_clipped: Array1<f64> = propensity_scores.mapv(|p| p.max(1e-10).min(1.0 - 1e-10));

    match estimand {
        TwangEstimand::ATT => {
            // ATT: Treated get weight 1, control get w = ps / (1-ps)
            for i in 0..n {
                if t[i] >= 0.5 {
                    weights[i] = 1.0;
                } else {
                    weights[i] = ps_clipped[i] / (1.0 - ps_clipped[i]);
                }
            }
        }
        TwangEstimand::ATE => {
            // ATE: Treated get 1/ps, control get 1/(1-ps)
            for i in 0..n {
                if t[i] >= 0.5 {
                    weights[i] = 1.0 / ps_clipped[i];
                } else {
                    weights[i] = 1.0 / (1.0 - ps_clipped[i]);
                }
            }
        }
        TwangEstimand::ATC => {
            // ATC: Treated get (1-ps)/ps, control get 1
            for i in 0..n {
                if t[i] >= 0.5 {
                    weights[i] = (1.0 - ps_clipped[i]) / ps_clipped[i];
                } else {
                    weights[i] = 1.0;
                }
            }
        }
    }

    // Normalize weights within each group to sum to group size
    let treated_idx: Vec<usize> = (0..n).filter(|&i| t[i] >= 0.5).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| t[i] < 0.5).collect();

    let sum_t: f64 = treated_idx.iter().map(|&i| weights[i]).sum();
    let sum_c: f64 = control_idx.iter().map(|&i| weights[i]).sum();

    let n_t = treated_idx.len() as f64;
    let n_c = control_idx.len() as f64;

    if sum_t > 0.0 {
        for &i in &treated_idx {
            weights[i] *= n_t / sum_t;
        }
    }
    if sum_c > 0.0 {
        for &i in &control_idx {
            weights[i] *= n_c / sum_c;
        }
    }

    weights
}

/// Compute balance metric for stopping rule.
fn compute_balance_metric(
    x: &Array2<f64>,
    t: &Array1<f64>,
    weights: &[f64],
    stop_method: StopMethod,
) -> f64 {
    let balance = compute_twang_balance_internal(x, t, weights);

    match stop_method {
        StopMethod::ESMean => balance.es_mean,
        StopMethod::ESMax => balance.es_max,
        StopMethod::KSMean => balance.ks_mean,
        StopMethod::KSMax => balance.ks_max,
    }
}

/// Compute full balance table.
fn compute_twang_balance(
    x: &Array2<f64>,
    t: &Array1<f64>,
    names: &[String],
    weights: Option<&[f64]>,
) -> TwangBalanceTable {
    let uniform_weights: Vec<f64>;
    let w = match weights {
        Some(w) => w,
        None => {
            uniform_weights = vec![1.0; t.len()];
            &uniform_weights
        }
    };

    let mut table = compute_twang_balance_internal(x, t, w);

    // Add names
    for (i, cov) in table.covariates.iter_mut().enumerate() {
        cov.name = names.get(i).cloned().unwrap_or_else(|| format!("X{}", i));
    }

    table
}

/// Internal balance computation.
fn compute_twang_balance_internal(
    x: &Array2<f64>,
    t: &Array1<f64>,
    weights: &[f64],
) -> TwangBalanceTable {
    let n = t.len();
    let k = x.ncols();

    let mut covariates = Vec::with_capacity(k);

    // Identify groups
    let treated_idx: Vec<usize> = (0..n).filter(|&i| t[i] >= 0.5).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| t[i] < 0.5).collect();

    for j in 0..k {
        // Compute weighted means
        let mut sum_t = 0.0;
        let mut sum_c = 0.0;
        let mut w_sum_t = 0.0;
        let mut w_sum_c = 0.0;

        for &i in &treated_idx {
            sum_t += weights[i] * x[[i, j]];
            w_sum_t += weights[i];
        }
        for &i in &control_idx {
            sum_c += weights[i] * x[[i, j]];
            w_sum_c += weights[i];
        }

        let mean_t = if w_sum_t > 0.0 { sum_t / w_sum_t } else { 0.0 };
        let mean_c = if w_sum_c > 0.0 { sum_c / w_sum_c } else { 0.0 };

        // Compute weighted variances
        let mut var_t = 0.0;
        let mut var_c = 0.0;

        for &i in &treated_idx {
            var_t += weights[i] * (x[[i, j]] - mean_t).powi(2);
        }
        for &i in &control_idx {
            var_c += weights[i] * (x[[i, j]] - mean_c).powi(2);
        }

        var_t = if w_sum_t > 1.0 {
            var_t / (w_sum_t - 1.0)
        } else {
            0.0
        };
        var_c = if w_sum_c > 1.0 {
            var_c / (w_sum_c - 1.0)
        } else {
            0.0
        };

        // Standardized effect size (using pooled SD)
        let pooled_var = (var_t + var_c) / 2.0;
        let pooled_sd = pooled_var.sqrt().max(1e-10);
        let std_eff_size = (mean_t - mean_c) / pooled_sd;

        // Variance ratio
        let var_ratio = if var_c > 1e-10 { var_t / var_c } else { 1.0 };

        // KS statistic (simplified: compare ECDFs)
        let ks_stat = compute_ks_statistic(x, j, &treated_idx, &control_idx, weights);

        covariates.push(TwangCovariateBalance {
            name: format!("X{}", j),
            mean_treated: mean_t,
            mean_control: mean_c,
            std_eff_size,
            ks_statistic: ks_stat,
            var_ratio,
        });
    }

    // Compute summary statistics
    let es_abs: Vec<f64> = covariates.iter().map(|c| c.std_eff_size.abs()).collect();
    let ks_vals: Vec<f64> = covariates.iter().map(|c| c.ks_statistic).collect();

    let es_mean = es_abs.iter().sum::<f64>() / k as f64;
    let es_max = es_abs.iter().cloned().fold(0.0, f64::max);
    let ks_mean = ks_vals.iter().sum::<f64>() / k as f64;
    let ks_max = ks_vals.iter().cloned().fold(0.0, f64::max);

    let n_balanced = es_abs.iter().filter(|&&e| e < 0.1).count();

    TwangBalanceTable {
        covariates,
        es_mean,
        es_max,
        ks_mean,
        ks_max,
        n_balanced,
    }
}

/// Compute two-sample Kolmogorov-Smirnov statistic.
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
    t_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    c_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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
    all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
    use polars::prelude::*;

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

        let stump = fit_decision_stump(&x, &residuals, 2);

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
}
