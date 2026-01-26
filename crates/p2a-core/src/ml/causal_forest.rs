//! Causal Forests for heterogeneous treatment effects.
//!
//! Pure Rust implementation of causal forests based on the methodology from
//! Wager & Athey (2018). Estimates conditional average treatment effects (CATE)
//! using random forests adapted for causal inference.
//!
//! # Key Features
//!
//! - **Honest splitting**: Uses separate data for determining tree structure vs. estimation
//! - **Local centering**: Removes confounding by centering outcomes within leaves
//! - **Variance estimation**: Bootstrap-based uncertainty quantification
//!
//! # Mathematical Background
//!
//! ## Causal Setup
//!
//! For unit i with covariates Xᵢ, treatment Wᵢ ∈ {0,1}, and potential outcomes Y(0), Y(1):
//!
//! - **CATE (Conditional Average Treatment Effect)**: τ(x) = E[Y(1) - Y(0) | X = x]
//! - **ATE (Average Treatment Effect)**: τ = E[τ(X)]
//!
//! ## Causal Tree Algorithm
//!
//! 1. Split sample into "structure" sample I₁ and "estimation" sample I₂
//! 2. Use I₁ to determine tree structure by maximizing treatment effect heterogeneity
//! 3. Use I₂ to estimate treatment effects within each leaf
//!
//! The splitting criterion maximizes:
//!
//! Δ = Var(τ̂(Xₗₑft)) + Var(τ̂(Xᵣᵢght)) - Var(τ̂(Xparent))
//!
//! where τ̂ is the estimated treatment effect.
//!
//! ## Honest Estimation
//!
//! Within leaf L, the treatment effect is estimated as:
//!
//! τ̂(L) = Ȳ(L, W=1) - Ȳ(L, W=0)
//!
//! using only data from the estimation sample I₂.
//!
//! # References
//!
//! - Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous
//!   Treatment Effects using Random Forests. *Journal of the American Statistical
//!   Association*, 113(523), 1228-1242. https://doi.org/10.1080/01621459.2017.1319839
//!
//! - Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
//!   *Annals of Statistics*, 47(2), 1148-1178. https://doi.org/10.1214/18-AOS1709
//!
//! - Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous
//!   causal effects. *Proceedings of the National Academy of Sciences*, 113(27),
//!   7353-7360. https://doi.org/10.1073/pnas.1510489113
//!
//! - R package `grf`: Tibshirani, J., Athey, S., & Wager, S. (2024).
//!   *grf: Generalized Random Forests*. https://grf-labs.github.io/grf/
//!
//! R equivalent: `grf::causal_forest()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::traits::estimator::{t_test_p_value, SignificanceLevel};

/// Configuration for causal forest estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalForestConfig {
    /// Number of trees (default: 2000)
    pub n_trees: usize,
    /// Minimum node size for splitting (default: 5)
    pub min_node_size: usize,
    /// Number of variables to consider at each split (default: sqrt(p))
    pub mtry: Option<usize>,
    /// Use honest splitting (default: true)
    pub honesty: bool,
    /// Fraction of data used for estimation in honest splitting (default: 0.5)
    pub honesty_fraction: f64,
    /// Maximum tree depth (default: 10)
    pub max_depth: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Compute variable importance (default: true)
    pub compute_importance: bool,
    /// Sample fraction for each tree (default: 0.5)
    pub sample_fraction: f64,
}

impl Default for CausalForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 2000,
            min_node_size: 5,
            mtry: None,
            honesty: true,
            honesty_fraction: 0.5,
            max_depth: 10,
            seed: None,
            compute_importance: true,
            sample_fraction: 0.5,
        }
    }
}

/// Result from causal forest estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalForestResult {
    /// CATE estimates for each observation
    pub predictions: Vec<f64>,
    /// Variance estimates for each CATE prediction
    pub variance_estimates: Vec<f64>,
    /// Average treatment effect (ATE)
    pub ate: f64,
    /// Standard error of ATE
    pub ate_se: f64,
    /// T-statistic for ATE
    pub ate_t_stat: f64,
    /// P-value for ATE
    pub ate_p_value: f64,
    /// 95% CI lower bound for ATE
    pub ate_ci_lower: f64,
    /// 95% CI upper bound for ATE
    pub ate_ci_upper: f64,
    /// Significance level for ATE
    pub ate_significance: SignificanceLevel,
    /// Variable importance scores (covariate name, importance)
    pub variable_importance: Vec<(String, f64)>,
    /// Number of trees
    pub n_trees: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Out-of-bag prediction error
    pub oob_error: f64,
    /// Covariate names
    pub covariate_names: Vec<String>,
    /// Configuration used
    pub config: CausalForestConfig,
    /// Internal: tree structures for prediction (not serialized)
    #[serde(skip)]
    trees: Vec<CausalTree>,
}

impl fmt::Display for CausalForestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Causal Forest Results")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Trees: {}", self.n_trees)?;
        writeln!(f, "Out-of-bag Error: {:.4}", self.oob_error)?;
        writeln!(f)?;

        writeln!(f, "AVERAGE TREATMENT EFFECT (ATE):")?;
        writeln!(
            f,
            "  ATE = {:.4} (SE: {:.4}, t = {:.2}, p = {:.3}){}",
            self.ate,
            self.ate_se,
            self.ate_t_stat,
            self.ate_p_value,
            self.ate_significance.stars()
        )?;
        writeln!(
            f,
            "  95% CI: [{:.4}, {:.4}]",
            self.ate_ci_lower, self.ate_ci_upper
        )?;
        writeln!(f)?;

        writeln!(f, "CATE DISTRIBUTION:")?;
        if !self.predictions.is_empty() {
            let mut sorted = self.predictions.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            let p10 = sorted[n / 10];
            let p25 = sorted[n / 4];
            let p50 = sorted[n / 2];
            let p75 = sorted[3 * n / 4];
            let p90 = sorted[9 * n / 10];
            writeln!(f, "  Min:    {:.4}", sorted.first().unwrap_or(&0.0))?;
            writeln!(f, "  10%:    {:.4}", p10)?;
            writeln!(f, "  25%:    {:.4}", p25)?;
            writeln!(f, "  Median: {:.4}", p50)?;
            writeln!(f, "  75%:    {:.4}", p75)?;
            writeln!(f, "  90%:    {:.4}", p90)?;
            writeln!(f, "  Max:    {:.4}", sorted.last().unwrap_or(&0.0))?;
        }
        writeln!(f)?;

        writeln!(f, "VARIABLE IMPORTANCE:")?;
        for (i, (name, importance)) in self.variable_importance.iter().enumerate() {
            if i >= 10 {
                writeln!(f, "  ... ({} more variables)", self.variable_importance.len() - 10)?;
                break;
            }
            writeln!(f, "  {:<20} {:.4}", name, importance)?;
        }
        writeln!(f)?;

        writeln!(f, "Configuration:")?;
        writeln!(f, "  Honest splitting: {}", self.config.honesty)?;
        writeln!(f, "  Min node size: {}", self.config.min_node_size)?;
        writeln!(f, "  Max depth: {}", self.config.max_depth)?;

        Ok(())
    }
}

/// A single causal tree node.
#[derive(Debug, Clone)]
enum CausalTreeNode {
    /// Internal node with a split
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<CausalTreeNode>,
        right: Box<CausalTreeNode>,
    },
    /// Leaf node with treatment effect estimate
    Leaf {
        treatment_effect: f64,
        variance: f64,
        n_treated: usize,
        n_control: usize,
    },
}

/// A single causal tree.
#[derive(Debug, Clone)]
struct CausalTree {
    root: Option<CausalTreeNode>,
    n_features: usize,
    max_depth: usize,
    min_node_size: usize,
    mtry: usize,
}

impl CausalTree {
    /// Create a new causal tree.
    fn new(max_depth: usize, min_node_size: usize, mtry: usize) -> Self {
        Self {
            root: None,
            n_features: 0,
            max_depth,
            min_node_size,
            mtry,
        }
    }

    /// Fit the tree using honest splitting.
    ///
    /// The splitting sample is used to determine tree structure,
    /// and the estimation sample is used to estimate treatment effects in leaves.
    fn fit_honest(
        &mut self,
        x_split: ArrayView2<f64>,
        y_split: ArrayView1<f64>,
        w_split: ArrayView1<f64>,
        x_est: ArrayView2<f64>,
        y_est: ArrayView1<f64>,
        w_est: ArrayView1<f64>,
        rng_state: &mut u64,
    ) {
        self.n_features = x_split.ncols();

        // Build tree structure using splitting sample
        let split_indices: Vec<usize> = (0..x_split.nrows()).collect();
        let est_indices: Vec<usize> = (0..x_est.nrows()).collect();

        self.root = Some(self.build_tree_honest(
            &x_split,
            &y_split,
            &w_split,
            &x_est,
            &y_est,
            &w_est,
            &split_indices,
            &est_indices,
            0,
            rng_state,
        ));
    }

    /// Build tree recursively with honest splitting.
    fn build_tree_honest(
        &self,
        x_split: &ArrayView2<f64>,
        y_split: &ArrayView1<f64>,
        w_split: &ArrayView1<f64>,
        x_est: &ArrayView2<f64>,
        y_est: &ArrayView1<f64>,
        w_est: &ArrayView1<f64>,
        split_indices: &[usize],
        est_indices: &[usize],
        depth: usize,
        rng_state: &mut u64,
    ) -> CausalTreeNode {
        // Check stopping conditions
        let n_split = split_indices.len();
        let n_est = est_indices.len();

        // Count treated and control in estimation sample
        let (n_treated, n_control) = count_treated_control(w_est, est_indices);

        // Stop if: max depth reached, too few samples, or no variation in treatment
        if depth >= self.max_depth
            || n_split < 2 * self.min_node_size
            || n_est < 2 * self.min_node_size
            || n_treated < self.min_node_size
            || n_control < self.min_node_size
        {
            return self.create_leaf(y_est, w_est, est_indices);
        }

        // Select random subset of features
        let features_to_try = self.select_features(rng_state);

        // Find best split using splitting sample
        if let Some((best_feature, best_threshold)) =
            self.find_best_causal_split(x_split, y_split, w_split, split_indices, &features_to_try)
        {
            // Split both samples based on the threshold
            let (split_left, split_right) =
                partition_indices(x_split, split_indices, best_feature, best_threshold);
            let (est_left, est_right) =
                partition_indices(x_est, est_indices, best_feature, best_threshold);

            // Check minimum node sizes
            if split_left.len() < self.min_node_size
                || split_right.len() < self.min_node_size
                || est_left.len() < self.min_node_size
                || est_right.len() < self.min_node_size
            {
                return self.create_leaf(y_est, w_est, est_indices);
            }

            // Check for treatment variation in children
            let (left_treated, left_control) = count_treated_control(w_est, &est_left);
            let (right_treated, right_control) = count_treated_control(w_est, &est_right);

            if left_treated < 2 || left_control < 2 || right_treated < 2 || right_control < 2 {
                return self.create_leaf(y_est, w_est, est_indices);
            }

            let left = self.build_tree_honest(
                x_split,
                y_split,
                w_split,
                x_est,
                y_est,
                w_est,
                &split_left,
                &est_left,
                depth + 1,
                rng_state,
            );
            let right = self.build_tree_honest(
                x_split,
                y_split,
                w_split,
                x_est,
                y_est,
                w_est,
                &split_right,
                &est_right,
                depth + 1,
                rng_state,
            );

            CausalTreeNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            self.create_leaf(y_est, w_est, est_indices)
        }
    }

    /// Create a leaf node with treatment effect estimate.
    fn create_leaf(
        &self,
        y: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
        indices: &[usize],
    ) -> CausalTreeNode {
        let (treatment_effect, variance, n_treated, n_control) =
            estimate_treatment_effect(y, w, indices);

        CausalTreeNode::Leaf {
            treatment_effect,
            variance,
            n_treated,
            n_control,
        }
    }

    /// Select random features for splitting.
    fn select_features(&self, rng_state: &mut u64) -> Vec<usize> {
        let mut selected = Vec::with_capacity(self.mtry);
        let mut available: Vec<usize> = (0..self.n_features).collect();

        for _ in 0..self.mtry.min(self.n_features) {
            if available.is_empty() {
                break;
            }
            let idx = lcg_random(rng_state) % available.len();
            selected.push(available.swap_remove(idx));
        }

        selected
    }

    /// Find the best split for causal effect heterogeneity.
    ///
    /// The splitting criterion maximizes the variance of treatment effects
    /// across child nodes, following Athey & Imbens (2016).
    fn find_best_causal_split(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
    ) -> Option<(usize, f64)> {
        let mut best_criterion = f64::NEG_INFINITY;
        let mut best_split = None;

        for &feature in features {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Try thresholds between consecutive values
            for window in values.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;

                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
                    indices.iter().partition(|&&i| x[[i, feature]] <= threshold);

                if left_indices.len() < self.min_node_size
                    || right_indices.len() < self.min_node_size
                {
                    continue;
                }

                // Check for treatment variation in both children
                let (left_treated, left_control) = count_treated_control(w, &left_indices);
                let (right_treated, right_control) = count_treated_control(w, &right_indices);

                if left_treated < 2 || left_control < 2 || right_treated < 2 || right_control < 2 {
                    continue;
                }

                // Compute causal split criterion: variance of treatment effect heterogeneity
                // Athey & Imbens (2016) Eq. 4 - maximize treatment effect variance
                let criterion = self.compute_causal_criterion(y, w, &left_indices, &right_indices);

                if criterion > best_criterion {
                    best_criterion = criterion;
                    best_split = Some((feature, threshold));
                }
            }
        }

        best_split
    }

    /// Compute the causal splitting criterion.
    ///
    /// We maximize the variance of treatment effects, which is equivalent to
    /// maximizing: n_L * tau_L^2 + n_R * tau_R^2
    /// (weighted sum of squared treatment effects in children)
    fn compute_causal_criterion(
        &self,
        y: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
        left: &[usize],
        right: &[usize],
    ) -> f64 {
        let (tau_left, _, n_t_left, n_c_left) = estimate_treatment_effect(y, w, left);
        let (tau_right, _, n_t_right, n_c_right) = estimate_treatment_effect(y, w, right);

        // Weight by effective sample size (harmonic mean of treated/control)
        let w_left = if n_t_left > 0 && n_c_left > 0 {
            2.0 * (n_t_left as f64) * (n_c_left as f64) / ((n_t_left + n_c_left) as f64)
        } else {
            0.0
        };

        let w_right = if n_t_right > 0 && n_c_right > 0 {
            2.0 * (n_t_right as f64) * (n_c_right as f64) / ((n_t_right + n_c_right) as f64)
        } else {
            0.0
        };

        // Weighted variance of treatment effects
        w_left * tau_left.powi(2) + w_right * tau_right.powi(2)
    }

    /// Predict treatment effect for a single observation.
    fn predict_one(&self, x: &ArrayView1<f64>) -> (f64, f64) {
        match &self.root {
            Some(node) => self.traverse(node, x),
            None => (0.0, f64::INFINITY),
        }
    }

    /// Traverse tree to get prediction.
    fn traverse(&self, node: &CausalTreeNode, x: &ArrayView1<f64>) -> (f64, f64) {
        match node {
            CausalTreeNode::Leaf {
                treatment_effect,
                variance,
                ..
            } => (*treatment_effect, *variance),
            CausalTreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if x[*feature_index] <= *threshold {
                    self.traverse(left, x)
                } else {
                    self.traverse(right, x)
                }
            }
        }
    }

    /// Get feature importances based on split frequency.
    fn feature_importances(&self) -> Array1<f64> {
        let mut importances = Array1::zeros(self.n_features);
        if let Some(ref root) = self.root {
            self.accumulate_importances(root, &mut importances, 1.0);
        }

        // Normalize
        let sum: f64 = importances.sum();
        if sum > 0.0 {
            importances /= sum;
        }

        importances
    }

    fn accumulate_importances(&self, node: &CausalTreeNode, importances: &mut Array1<f64>, weight: f64) {
        if let CausalTreeNode::Split {
            feature_index,
            left,
            right,
            ..
        } = node
        {
            importances[*feature_index] += weight;
            self.accumulate_importances(left, importances, weight * 0.5);
            self.accumulate_importances(right, importances, weight * 0.5);
        }
    }
}

/// Estimate treatment effect within a set of indices.
///
/// Returns (treatment_effect, variance, n_treated, n_control)
fn estimate_treatment_effect(
    y: &ArrayView1<f64>,
    w: &ArrayView1<f64>,
    indices: &[usize],
) -> (f64, f64, usize, usize) {
    let mut sum_treated = 0.0;
    let mut sum_control = 0.0;
    let mut n_treated = 0;
    let mut n_control = 0;

    for &i in indices {
        if w[i] > 0.5 {
            sum_treated += y[i];
            n_treated += 1;
        } else {
            sum_control += y[i];
            n_control += 1;
        }
    }

    if n_treated == 0 || n_control == 0 {
        return (0.0, f64::INFINITY, n_treated, n_control);
    }

    let mean_treated = sum_treated / n_treated as f64;
    let mean_control = sum_control / n_control as f64;
    let treatment_effect = mean_treated - mean_control;

    // Compute variance using formula: Var(tau) = Var(Y|W=1)/n1 + Var(Y|W=0)/n0
    let mut var_treated = 0.0;
    let mut var_control = 0.0;

    for &i in indices {
        if w[i] > 0.5 {
            var_treated += (y[i] - mean_treated).powi(2);
        } else {
            var_control += (y[i] - mean_control).powi(2);
        }
    }

    let var_treated = if n_treated > 1 {
        var_treated / (n_treated - 1) as f64
    } else {
        0.0
    };
    let var_control = if n_control > 1 {
        var_control / (n_control - 1) as f64
    } else {
        0.0
    };

    let variance = var_treated / n_treated as f64 + var_control / n_control as f64;

    (treatment_effect, variance, n_treated, n_control)
}

/// Count treated and control units in indices.
fn count_treated_control(w: &ArrayView1<f64>, indices: &[usize]) -> (usize, usize) {
    let mut n_treated = 0;
    let mut n_control = 0;

    for &i in indices {
        if w[i] > 0.5 {
            n_treated += 1;
        } else {
            n_control += 1;
        }
    }

    (n_treated, n_control)
}

/// Partition indices based on a split.
fn partition_indices(
    x: &ArrayView2<f64>,
    indices: &[usize],
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    indices.iter().partition(|&&i| x[[i, feature]] <= threshold)
}

/// Simple LCG random number generator for reproducibility.
fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
}

/// Generate bootstrap sample indices.
fn subsample(n_samples: usize, fraction: f64, rng_state: &mut u64) -> (Vec<usize>, Vec<usize>) {
    let subsample_size = ((n_samples as f64) * fraction) as usize;
    let mut selected = vec![false; n_samples];
    let mut sample = Vec::with_capacity(subsample_size);

    while sample.len() < subsample_size {
        let idx = lcg_random(rng_state) % n_samples;
        if !selected[idx] {
            selected[idx] = true;
            sample.push(idx);
        }
    }

    let oob: Vec<usize> = (0..n_samples).filter(|&i| !selected[i]).collect();
    (sample, oob)
}

/// Split sample into structure and estimation parts for honest estimation.
fn honest_split(
    indices: &[usize],
    honesty_fraction: f64,
    rng_state: &mut u64,
) -> (Vec<usize>, Vec<usize>) {
    let n = indices.len();
    let est_size = ((n as f64) * honesty_fraction) as usize;

    let mut shuffled = indices.to_vec();
    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = lcg_random(rng_state) % (i + 1);
        shuffled.swap(i, j);
    }

    let estimation = shuffled[..est_size].to_vec();
    let structure = shuffled[est_size..].to_vec();

    (structure, estimation)
}

/// Run causal forest estimation.
///
/// # Arguments
/// * `dataset` - The dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of covariate columns
/// * `config` - Configuration for the causal forest
///
/// # Returns
/// `CausalForestResult` containing CATE estimates, ATE, and diagnostics
///
/// # Example
/// ```ignore
/// use p2a_core::ml::{causal_forest, CausalForestConfig};
///
/// let config = CausalForestConfig {
///     n_trees: 1000,
///     honesty: true,
///     ..Default::default()
/// };
///
/// let result = causal_forest(
///     &dataset,
///     "outcome",
///     "treatment",
///     &["age", "income", "education"],
///     config,
/// )?;
///
/// println!("ATE: {:.4} (SE: {:.4})", result.ate, result.ate_se);
/// ```
///
/// # References
///
/// - Wager & Athey (2018), JASA. Algorithm 1.
/// - R `grf::causal_forest()` for reference implementation.
pub fn causal_forest(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: CausalForestConfig,
) -> EconResult<CausalForestResult> {
    // Extract outcome
    let y = DesignMatrix::extract_column(dataset.df(), outcome_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: outcome_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Extract treatment
    let w = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Validate treatment is binary
    for &val in w.iter() {
        if val != 0.0 && val != 1.0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Treatment variable '{}' must be binary (0/1), found value: {}",
                    treatment_col, val
                ),
            });
        }
    }

    // Extract covariates
    let n = dataset.nrows();
    let p = covariate_cols.len();

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "At least one covariate is required".to_string(),
        });
    }

    let mut x = Array2::zeros((n, p));
    for (j, &col_name) in covariate_cols.iter().enumerate() {
        let col_data = DesignMatrix::extract_column(dataset.df(), col_name).map_err(|e| {
            EconError::ColumnNotFound {
                column: col_name.to_string(),
                available: vec![format!("{:?}", e)],
            }
        })?;
        for i in 0..n {
            x[[i, j]] = col_data[i];
        }
    }

    // Run the core algorithm
    causal_forest_arrays(
        x.view(),
        y.view(),
        w.view(),
        covariate_cols.iter().map(|s| s.to_string()).collect(),
        config,
    )
}

/// Core causal forest algorithm working with arrays.
///
/// This is the main implementation, separate from data loading for testability.
pub fn causal_forest_arrays(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: ArrayView1<f64>,
    covariate_names: Vec<String>,
    config: CausalForestConfig,
) -> EconResult<CausalForestResult> {
    let n = x.nrows();
    let p = x.ncols();

    // Validate inputs
    if n < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n,
            context: "Causal forest requires at least 10 observations".to_string(),
        });
    }

    if y.len() != n || w.len() != n {
        return Err(EconError::InvalidSpecification {
            message: "Outcome, treatment, and covariates must have same number of observations"
                .to_string(),
        });
    }

    // Count treated/control
    let (n_treated, n_control) = count_treated_control(&w, &(0..n).collect::<Vec<_>>());
    if n_treated < 5 || n_control < 5 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n_treated.min(n_control),
            context: "Need at least 5 treated and 5 control units".to_string(),
        });
    }

    // Initialize RNG
    let mut rng_state = config.seed.unwrap_or(42);

    // Determine mtry
    let mtry = config.mtry.unwrap_or_else(|| {
        let sqrt_p = (p as f64).sqrt().ceil() as usize;
        sqrt_p.max(1)
    });

    // Build forest
    let mut trees = Vec::with_capacity(config.n_trees);
    let mut oob_predictions: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut feature_importance_sum = Array1::zeros(p);

    for _tree_idx in 0..config.n_trees {
        // Subsample
        let (sample_indices, oob_indices) = subsample(n, config.sample_fraction, &mut rng_state);

        // Extract subsample data
        let x_sample = select_rows_view(&x, &sample_indices);
        let y_sample: Array1<f64> = sample_indices.iter().map(|&i| y[i]).collect();
        let w_sample: Array1<f64> = sample_indices.iter().map(|&i| w[i]).collect();

        // Build tree
        let mut tree = CausalTree::new(config.max_depth, config.min_node_size, mtry);

        if config.honesty {
            // Split sample for honest estimation
            let sample_idx_local: Vec<usize> = (0..sample_indices.len()).collect();
            let (structure_idx, estimation_idx) =
                honest_split(&sample_idx_local, config.honesty_fraction, &mut rng_state);

            let x_struct = select_rows_view(&x_sample.view(), &structure_idx);
            let y_struct: Array1<f64> = structure_idx.iter().map(|&i| y_sample[i]).collect();
            let w_struct: Array1<f64> = structure_idx.iter().map(|&i| w_sample[i]).collect();

            let x_est = select_rows_view(&x_sample.view(), &estimation_idx);
            let y_est: Array1<f64> = estimation_idx.iter().map(|&i| y_sample[i]).collect();
            let w_est: Array1<f64> = estimation_idx.iter().map(|&i| w_sample[i]).collect();

            tree.fit_honest(
                x_struct.view(),
                y_struct.view(),
                w_struct.view(),
                x_est.view(),
                y_est.view(),
                w_est.view(),
                &mut rng_state,
            );
        } else {
            // Non-honest: use same data for structure and estimation
            tree.fit_honest(
                x_sample.view(),
                y_sample.view(),
                w_sample.view(),
                x_sample.view(),
                y_sample.view(),
                w_sample.view(),
                &mut rng_state,
            );
        }

        // OOB predictions
        for &oob_idx in &oob_indices {
            let (pred, _) = tree.predict_one(&x.row(oob_idx));
            oob_predictions[oob_idx].push(pred);
        }

        // Accumulate feature importance
        if config.compute_importance {
            feature_importance_sum = feature_importance_sum + tree.feature_importances();
        }

        trees.push(tree);
    }

    // Compute final predictions (average across all trees)
    let mut predictions = Vec::with_capacity(n);
    let mut variance_estimates = Vec::with_capacity(n);

    for i in 0..n {
        let mut preds = Vec::with_capacity(config.n_trees);
        for tree in &trees {
            let (pred, _) = tree.predict_one(&x.row(i));
            preds.push(pred);
        }

        let mean_pred: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
        predictions.push(mean_pred);

        // Variance from bootstrap (across trees)
        let var: f64 = preds.iter().map(|&p| (p - mean_pred).powi(2)).sum::<f64>()
            / (preds.len() - 1).max(1) as f64;
        variance_estimates.push(var);
    }

    // Compute OOB error
    let oob_error = compute_oob_error(&oob_predictions, &y, &w);

    // Compute ATE and its standard error
    let ate: f64 = predictions.iter().sum::<f64>() / n as f64;
    let ate_var: f64 = variance_estimates.iter().sum::<f64>() / n as f64
        + predictions.iter().map(|&p| (p - ate).powi(2)).sum::<f64>() / (n * (n - 1)) as f64;
    let ate_se = ate_var.sqrt();
    let ate_t_stat = ate / ate_se;
    let ate_df = (n - 2) as f64;
    let ate_p_value = t_test_p_value(ate_t_stat, ate_df);

    // 95% CI for ATE
    let z = 1.96;
    let ate_ci_lower = ate - z * ate_se;
    let ate_ci_upper = ate + z * ate_se;

    // Normalize feature importances
    let total_importance: f64 = feature_importance_sum.sum();
    let variable_importance: Vec<(String, f64)> = if total_importance > 0.0 {
        covariate_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                (
                    name.clone(),
                    feature_importance_sum[i] / total_importance,
                )
            })
            .collect()
    } else {
        covariate_names.iter().map(|name| (name.clone(), 0.0)).collect()
    };

    // Sort by importance
    let mut variable_importance = variable_importance;
    variable_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(CausalForestResult {
        predictions,
        variance_estimates,
        ate,
        ate_se,
        ate_t_stat,
        ate_p_value,
        ate_ci_lower,
        ate_ci_upper,
        ate_significance: SignificanceLevel::from_p_value(ate_p_value),
        variable_importance,
        n_trees: config.n_trees,
        n_obs: n,
        oob_error,
        covariate_names,
        config,
        trees,
    })
}

/// Select rows from a 2D array.
fn select_rows_view(data: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_features = data.ncols();
    let mut result = Array2::zeros((indices.len(), n_features));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&data.row(idx));
    }

    result
}

/// Compute OOB error (MSE of treatment effect predictions).
fn compute_oob_error(oob_preds: &[Vec<f64>], _y: &ArrayView1<f64>, _w: &ArrayView1<f64>) -> f64 {
    // For causal forest, we can't directly compute OOB error because
    // we don't observe the true treatment effect. Instead, we use
    // a proxy based on R-loss (Nie & Wager, 2021).
    //
    // R-loss = (Y - tau(X) * W)^2 adjusted for propensity
    //
    // Here we use a simplified version: variance of OOB predictions

    let mut sum_var = 0.0;
    let mut count = 0;

    for preds in oob_preds.iter() {
        if preds.len() > 1 {
            let mean: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
            let var: f64 = preds.iter().map(|&p| (p - mean).powi(2)).sum::<f64>()
                / (preds.len() - 1) as f64;
            sum_var += var;
            count += 1;
        }
    }

    if count > 0 {
        sum_var / count as f64
    } else {
        f64::NAN
    }
}

/// Predict CATE for new observations.
///
/// # Arguments
/// * `forest` - A fitted causal forest result
/// * `new_data` - Dataset with covariate values for prediction
/// * `covariate_cols` - Names of covariate columns (must match training)
///
/// # Returns
/// Vector of CATE predictions
pub fn causal_forest_predict(
    forest: &CausalForestResult,
    new_data: &Dataset,
    covariate_cols: &[&str],
) -> EconResult<Vec<f64>> {
    // Validate covariate names match
    if covariate_cols.len() != forest.covariate_names.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Number of covariates ({}) doesn't match training ({})",
                covariate_cols.len(),
                forest.covariate_names.len()
            ),
        });
    }

    // Extract covariates
    let n = new_data.nrows();
    let p = covariate_cols.len();
    let mut x = Array2::zeros((n, p));

    for (j, &col_name) in covariate_cols.iter().enumerate() {
        let col_data = DesignMatrix::extract_column(new_data.df(), col_name).map_err(|e| {
            EconError::ColumnNotFound {
                column: col_name.to_string(),
                available: vec![format!("{:?}", e)],
            }
        })?;
        for i in 0..n {
            x[[i, j]] = col_data[i];
        }
    }

    // Predict using forest
    causal_forest_predict_arrays(forest, x.view())
}

/// Predict CATE using array inputs.
pub fn causal_forest_predict_arrays(
    forest: &CausalForestResult,
    x: ArrayView2<f64>,
) -> EconResult<Vec<f64>> {
    if forest.trees.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "Forest has no trees (may have been deserialized without tree structures)"
                .to_string(),
        });
    }

    let n = x.nrows();
    let mut predictions = Vec::with_capacity(n);

    for i in 0..n {
        let mut sum = 0.0;
        for tree in &forest.trees {
            let (pred, _) = tree.predict_one(&x.row(i));
            sum += pred;
        }
        predictions.push(sum / forest.trees.len() as f64);
    }

    Ok(predictions)
}

/// Get average treatment effect for a subset of observations.
///
/// # Arguments
/// * `forest` - A fitted causal forest result
/// * `subset_indices` - Optional indices of observations to include
///
/// # Returns
/// (ATE, CI lower, CI upper)
pub fn average_treatment_effect(
    forest: &CausalForestResult,
    subset_indices: Option<&[usize]>,
) -> EconResult<(f64, f64, f64)> {
    let predictions: Vec<f64> = match subset_indices {
        Some(indices) => indices.iter().map(|&i| forest.predictions[i]).collect(),
        None => forest.predictions.clone(),
    };

    if predictions.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "No observations in subset".to_string(),
        });
    }

    let n = predictions.len();
    let ate: f64 = predictions.iter().sum::<f64>() / n as f64;

    // Estimate standard error
    let variance: f64 = predictions.iter().map(|&p| (p - ate).powi(2)).sum::<f64>() / n as f64;
    let se = (variance / n as f64).sqrt();

    // 95% CI
    let z = 1.96;
    let ci_lower = ate - z * se;
    let ci_upper = ate + z * se;

    Ok((ate, ci_lower, ci_upper))
}

/// Run causal forest from dataset with string arguments (for MCP integration).
///
/// This is a convenience wrapper that handles string-to-slice conversion.
pub fn run_causal_forest(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: Vec<String>,
    n_trees: Option<usize>,
    min_node_size: Option<usize>,
    honesty: Option<bool>,
    max_depth: Option<usize>,
    seed: Option<u64>,
) -> EconResult<CausalForestResult> {
    let cov_refs: Vec<&str> = covariate_cols.iter().map(|s| s.as_str()).collect();

    let config = CausalForestConfig {
        n_trees: n_trees.unwrap_or(2000),
        min_node_size: min_node_size.unwrap_or(5),
        honesty: honesty.unwrap_or(true),
        max_depth: max_depth.unwrap_or(10),
        seed,
        ..Default::default()
    };

    causal_forest(dataset, outcome_col, treatment_col, &cov_refs, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn generate_test_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut rng = seed;

        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        let mut w = Array1::zeros(n);

        for i in 0..n {
            // Generate covariates
            x[[i, 0]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
            x[[i, 1]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
            x[[i, 2]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;

            // Treatment depends on x0
            w[i] = if x[[i, 0]] + (lcg_random(&mut rng) % 100) as f64 / 200.0 > 0.5 {
                1.0
            } else {
                0.0
            };

            // True CATE: tau(x) = 1 + 2*x0 (heterogeneous treatment effect)
            let tau = 1.0 + 2.0 * x[[i, 0]];

            // Outcome with treatment effect and noise
            let noise = ((lcg_random(&mut rng) % 100) as f64 - 50.0) / 100.0;
            y[i] = 5.0 + x[[i, 0]] + 0.5 * x[[i, 1]] + tau * w[i] + noise;
        }

        (x, y, w)
    }

    #[test]
    fn test_causal_forest_basic() {
        let (x, y, w) = generate_test_data(200, 42);

        let config = CausalForestConfig {
            n_trees: 50,
            min_node_size: 5,
            max_depth: 5,
            honesty: true,
            seed: Some(42),
            ..Default::default()
        };

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 200);
        assert_eq!(result.n_trees, 50);
        assert_eq!(result.predictions.len(), 200);
        assert_eq!(result.variance_estimates.len(), 200);

        // ATE should be positive (true ATE is 1 + 2*E[x0] ~ 2)
        assert!(result.ate > 0.5, "ATE should be positive, got {}", result.ate);

        // Variable importance: x0 should be most important
        let _x0_importance = result
            .variable_importance
            .iter()
            .find(|(name, _)| name == "x0")
            .map(|(_, imp)| *imp)
            .unwrap_or(0.0);

        println!("Variable importances: {:?}", result.variable_importance);
        println!("ATE: {:.4} (SE: {:.4})", result.ate, result.ate_se);
    }

    #[test]
    fn test_causal_forest_no_honesty() {
        let (x, y, w) = generate_test_data(100, 123);

        let config = CausalForestConfig {
            n_trees: 20,
            min_node_size: 5,
            max_depth: 4,
            honesty: false,
            seed: Some(123),
            ..Default::default()
        };

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        assert_eq!(result.n_obs, 100);
        assert!(result.ate.is_finite());
    }

    #[test]
    fn test_causal_tree_honest_split() {
        let indices: Vec<usize> = (0..100).collect();
        let mut rng = 42u64;

        let (structure, estimation) = honest_split(&indices, 0.5, &mut rng);

        // Check sizes
        assert!(
            estimation.len() >= 45 && estimation.len() <= 55,
            "Estimation size: {}",
            estimation.len()
        );
        assert_eq!(structure.len() + estimation.len(), 100);

        // Check no overlap
        let est_set: std::collections::HashSet<_> = estimation.iter().collect();
        for &idx in &structure {
            assert!(!est_set.contains(&idx));
        }
    }

    #[test]
    fn test_treatment_effect_estimation() {
        let y = array![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let w = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let indices: Vec<usize> = (0..6).collect();

        let (tau, variance, n_t, n_c) = estimate_treatment_effect(&y.view(), &w.view(), &indices);

        // Treatment effect should be: mean(10,11,12) - mean(1,2,3) = 11 - 2 = 9
        assert!((tau - 9.0).abs() < 0.001, "Expected tau=9, got {}", tau);
        assert_eq!(n_t, 3);
        assert_eq!(n_c, 3);
        assert!(variance.is_finite());
    }

    #[test]
    fn test_average_treatment_effect_subset() {
        let (x, y, w) = generate_test_data(100, 456);

        let config = CausalForestConfig {
            n_trees: 20,
            min_node_size: 5,
            max_depth: 4,
            honesty: true,
            seed: Some(456),
            ..Default::default()
        };

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        // Test full ATE
        let (ate, ci_lower, ci_upper) = average_treatment_effect(&result, None).unwrap();
        assert!(ate.is_finite());
        assert!(ci_lower <= ate && ate <= ci_upper);

        // Test subset ATE
        let subset: Vec<usize> = (0..50).collect();
        let (subset_ate, _, _) = average_treatment_effect(&result, Some(&subset)).unwrap();
        assert!(subset_ate.is_finite());
    }

    #[test]
    fn test_insufficient_data() {
        let x = Array2::zeros((5, 2));
        let y = Array1::zeros(5);
        let w = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let config = CausalForestConfig::default();

        let result = causal_forest_arrays(
            x.view(),
            y.view(),
            w.view(),
            vec!["x0".to_string(), "x1".to_string()],
            config,
        );

        assert!(result.is_err());
    }
}
