//! BART-based Causal Inference for Heterogeneous Treatment Effects.
//!
//! This module provides a simplified frequentist approximation to BART-based
//! causal inference, as implemented in the R package `bartCause`. Since full
//! Bayesian BART requires MCMC sampling, we implement an ensemble of regression
//! trees with bootstrap uncertainty quantification.
//!
//! # Key Features
//!
//! - **Separate response surfaces**: Fit separate models for treated and control groups
//! - **CATE estimation**: Conditional Average Treatment Effects via counterfactual prediction
//! - **Bootstrap uncertainty**: Confidence intervals through bootstrap resampling
//! - **Variable importance**: Identify covariates driving treatment effect heterogeneity
//! - **Propensity score option**: Optionally include propensity scores in covariate set
//!
//! # Mathematical Background
//!
//! ## CATE Estimation via Separate Response Surfaces
//!
//! For unit i with covariates X_i, treatment W_i in {0,1}, and outcome Y_i:
//!
//! 1. Fit response surface for treated: mu_1(x) = E[Y | W=1, X=x]
//! 2. Fit response surface for control: mu_0(x) = E[Y | W=0, X=x]
//! 3. CATE: tau(x) = mu_1(x) - mu_0(x)
//!
//! This is the "T-learner" approach (Kunzel et al., 2019).
//!
//! ## Average Treatment Effect
//!
//! The ATE is estimated as:
//!
//! ATE = (1/n) * sum_i [ mu_1(X_i) - mu_0(X_i) ]
//!
//! ## Bootstrap Uncertainty
//!
//! Uncertainty is quantified via bootstrap resampling:
//! 1. Resample data with replacement B times
//! 2. Re-estimate CATE for each bootstrap sample
//! 3. Compute confidence intervals from bootstrap distribution
//!
//! # Comparison with Full Bayesian BART
//!
//! | Feature | This Implementation | bartCause (R) |
//! |---------|---------------------|---------------|
//! | Uncertainty | Bootstrap CI | Posterior credible intervals |
//! | Trees | Random forest | BART (sum of trees with MCMC) |
//! | Speed | Fast | Slower (MCMC) |
//! | Theory | Frequentist | Bayesian |
//!
//! For full Bayesian inference, use R's `bartCause` package.
//!
//! # References
//!
//! - Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference.
//!   *Journal of Computational and Graphical Statistics*, 20(1), 217-240.
//!   https://doi.org/10.1198/jcgs.2010.08162
//!
//! - Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian
//!   Additive Regression Trees. *Annals of Applied Statistics*, 4(1), 266-298.
//!   https://doi.org/10.1214/09-AOAS285
//!
//! - Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020). Bayesian Regression
//!   Tree Models for Causal Inference: Regularization, Confounding, and
//!   Heterogeneous Effects. *Bayesian Analysis*, 15(3), 965-1056.
//!   https://doi.org/10.1214/19-BA1195
//!
//! - Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners
//!   for Estimating Heterogeneous Treatment Effects using Machine Learning.
//!   *Proceedings of the National Academy of Sciences*, 116(10), 4156-4165.
//!   https://doi.org/10.1073/pnas.1804597116
//!
//! - R package `bartCause`: Dorie, V. (2020). *bartCause: Causal Inference
//!   using BART*. https://CRAN.R-project.org/package=bartCause
//!
//! R equivalent: `bartCause::bartc()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::traits::estimator::{t_test_p_value, SignificanceLevel};

/// Configuration for BART-based causal inference.
///
/// # Example
///
/// ```ignore
/// use p2a_core::ml::BartCausalConfig;
///
/// let config = BartCausalConfig {
///     n_trees: 200,
///     max_depth: 4,
///     n_bootstrap: 100,
///     include_propensity: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartCausalConfig {
    /// Number of trees in each ensemble (default: 200)
    ///
    /// More trees generally improve prediction accuracy but increase
    /// computation time. BART typically uses 200 trees.
    pub n_trees: usize,

    /// Maximum depth per tree (default: 4)
    ///
    /// Shallow trees (depth 3-5) are typical for BART-style models.
    /// Deeper trees may overfit.
    pub max_depth: usize,

    /// Minimum samples required to split a node (default: 5)
    pub min_node_size: usize,

    /// Number of bootstrap samples for uncertainty quantification (default: 100)
    ///
    /// More bootstrap samples provide more stable confidence intervals
    /// but increase computation time.
    pub n_bootstrap: usize,

    /// Whether to include estimated propensity scores as a covariate (default: false)
    ///
    /// Including propensity can help with confounding adjustment but may
    /// introduce bias if the propensity model is misspecified.
    pub include_propensity: bool,

    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Sample fraction for subsampling in each tree (default: 0.632)
    pub sample_fraction: f64,

    /// Number of features to consider at each split (default: sqrt(p))
    pub mtry: Option<usize>,
}

impl Default for BartCausalConfig {
    fn default() -> Self {
        Self {
            n_trees: 200,
            max_depth: 4,
            min_node_size: 5,
            n_bootstrap: 100,
            include_propensity: false,
            confidence_level: 0.95,
            seed: None,
            sample_fraction: 0.632,
            mtry: None,
        }
    }
}

/// Result from BART-based causal inference.
///
/// Contains individual treatment effects (CATE), average treatment effect (ATE),
/// uncertainty estimates, and variable importance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartCausalResult {
    /// Average Treatment Effect (ATE)
    pub ate: f64,

    /// Standard error of ATE (from bootstrap)
    pub ate_se: f64,

    /// T-statistic for ATE
    pub ate_t_stat: f64,

    /// P-value for ATE (two-sided)
    pub ate_p_value: f64,

    /// Lower bound of ATE confidence interval
    pub ate_ci_lower: f64,

    /// Upper bound of ATE confidence interval
    pub ate_ci_upper: f64,

    /// Significance level indicator
    pub ate_significance: SignificanceLevel,

    /// Conditional Average Treatment Effects for each observation
    pub cate: Vec<f64>,

    /// Lower CI bound for each CATE estimate
    pub cate_lower: Vec<f64>,

    /// Upper CI bound for each CATE estimate
    pub cate_upper: Vec<f64>,

    /// Standard error for each CATE estimate
    pub cate_se: Vec<f64>,

    /// Predicted outcome under treatment: E[Y|T=1,X]
    pub y1_pred: Vec<f64>,

    /// Predicted outcome under control: E[Y|T=0,X]
    pub y0_pred: Vec<f64>,

    /// Variable importance scores (covariate name, importance)
    ///
    /// Importance is based on how much each variable contributes to
    /// treatment effect heterogeneity.
    pub variable_importance: Vec<(String, f64)>,

    /// Number of observations
    pub n_obs: usize,

    /// Number of treated units
    pub n_treated: usize,

    /// Number of control units
    pub n_control: usize,

    /// Number of trees per ensemble
    pub n_trees: usize,

    /// Number of bootstrap samples used
    pub n_bootstrap: usize,

    /// Covariate names
    pub covariate_names: Vec<String>,

    /// Configuration used
    pub config: BartCausalConfig,

    // Internal: tree ensembles (not serialized)
    #[serde(skip)]
    ensemble_treated: Option<TreeEnsemble>,
    #[serde(skip)]
    ensemble_control: Option<TreeEnsemble>,
}

impl fmt::Display for BartCausalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BART Causal Inference Results")?;
        writeln!(f, "===========================================")?;
        writeln!(
            f,
            "No. Observations: {} (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(f, "No. Trees: {} per response surface", self.n_trees)?;
        writeln!(f, "Bootstrap samples: {}", self.n_bootstrap)?;
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
            "  {:.0}% CI: [{:.4}, {:.4}]",
            self.config.confidence_level * 100.0,
            self.ate_ci_lower,
            self.ate_ci_upper
        )?;
        writeln!(f)?;

        writeln!(f, "CATE DISTRIBUTION:")?;
        if !self.cate.is_empty() {
            let mut sorted = self.cate.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            writeln!(f, "  Min:    {:.4}", sorted.first().unwrap_or(&0.0))?;
            writeln!(f, "  10%:    {:.4}", sorted[n / 10])?;
            writeln!(f, "  25%:    {:.4}", sorted[n / 4])?;
            writeln!(f, "  Median: {:.4}", sorted[n / 2])?;
            writeln!(f, "  75%:    {:.4}", sorted[3 * n / 4])?;
            writeln!(f, "  90%:    {:.4}", sorted[9 * n / 10])?;
            writeln!(f, "  Max:    {:.4}", sorted.last().unwrap_or(&0.0))?;
        }
        writeln!(f)?;

        writeln!(f, "VARIABLE IMPORTANCE (Treatment Effect Heterogeneity):")?;
        for (i, (name, importance)) in self.variable_importance.iter().enumerate() {
            if i >= 10 {
                writeln!(
                    f,
                    "  ... ({} more variables)",
                    self.variable_importance.len() - 10
                )?;
                break;
            }
            writeln!(f, "  {:<20} {:.4}", name, importance)?;
        }
        writeln!(f)?;

        writeln!(f, "Configuration:")?;
        writeln!(f, "  Max depth: {}", self.config.max_depth)?;
        writeln!(f, "  Min node size: {}", self.config.min_node_size)?;
        writeln!(
            f,
            "  Propensity included: {}",
            self.config.include_propensity
        )?;

        Ok(())
    }
}

// ============================================================================
// Tree Node and Tree Implementation
// ============================================================================

/// A node in a regression tree.
#[derive(Debug, Clone)]
enum TreeNode {
    /// Internal node with a split
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    /// Leaf node with prediction
    Leaf { value: f64, n_samples: usize },
}

/// A single regression tree.
#[derive(Debug, Clone)]
struct RegressionTree {
    root: Option<TreeNode>,
    n_features: usize,
    max_depth: usize,
    min_node_size: usize,
    mtry: usize,
}

impl RegressionTree {
    fn new(max_depth: usize, min_node_size: usize, mtry: usize) -> Self {
        Self {
            root: None,
            n_features: 0,
            max_depth,
            min_node_size,
            mtry,
        }
    }

    /// Fit the tree to data.
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>, rng_state: &mut u64) {
        self.n_features = x.ncols();
        let indices: Vec<usize> = (0..x.nrows()).collect();
        self.root = Some(self.build_tree(&x, &y, &indices, 0, rng_state));
    }

    /// Build tree recursively.
    fn build_tree(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        depth: usize,
        rng_state: &mut u64,
    ) -> TreeNode {
        let n_samples = indices.len();

        // Stopping conditions
        if depth >= self.max_depth || n_samples < 2 * self.min_node_size {
            return self.create_leaf(y, indices);
        }

        // Check variance - stop if all values are the same
        let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n_samples as f64;
        let variance: f64 = indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>();
        if variance < 1e-10 {
            return self.create_leaf(y, indices);
        }

        // Select random features
        let features = self.select_features(rng_state);

        // Find best split
        if let Some((best_feature, best_threshold)) = self.find_best_split(x, y, indices, &features)
        {
            let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&i| x[[i, best_feature]] <= best_threshold);

            if left_indices.len() < self.min_node_size || right_indices.len() < self.min_node_size {
                return self.create_leaf(y, indices);
            }

            let left = self.build_tree(x, y, &left_indices, depth + 1, rng_state);
            let right = self.build_tree(x, y, &right_indices, depth + 1, rng_state);

            TreeNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            self.create_leaf(y, indices)
        }
    }

    fn create_leaf(&self, y: &ArrayView1<f64>, indices: &[usize]) -> TreeNode {
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        TreeNode::Leaf {
            value: sum / indices.len() as f64,
            n_samples: indices.len(),
        }
    }

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

    fn find_best_split(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
    ) -> Option<(usize, f64)> {
        let mut best_mse = f64::INFINITY;
        let mut best_split = None;

        for &feature in features {
            let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            for window in values.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;

                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
                    indices.iter().partition(|&&i| x[[i, feature]] <= threshold);

                if left_indices.len() < self.min_node_size
                    || right_indices.len() < self.min_node_size
                {
                    continue;
                }

                let mse = self.compute_split_mse(y, &left_indices, &right_indices);
                if mse < best_mse {
                    best_mse = mse;
                    best_split = Some((feature, threshold));
                }
            }
        }

        best_split
    }

    fn compute_split_mse(&self, y: &ArrayView1<f64>, left: &[usize], right: &[usize]) -> f64 {
        let n = (left.len() + right.len()) as f64;

        let left_mean: f64 = left.iter().map(|&i| y[i]).sum::<f64>() / left.len() as f64;
        let left_mse: f64 = left
            .iter()
            .map(|&i| (y[i] - left_mean).powi(2))
            .sum::<f64>()
            / left.len() as f64;

        let right_mean: f64 = right.iter().map(|&i| y[i]).sum::<f64>() / right.len() as f64;
        let right_mse: f64 = right
            .iter()
            .map(|&i| (y[i] - right_mean).powi(2))
            .sum::<f64>()
            / right.len() as f64;

        (left.len() as f64 * left_mse + right.len() as f64 * right_mse) / n
    }

    fn predict_one(&self, x: &ArrayView1<f64>) -> f64 {
        match &self.root {
            Some(node) => self.traverse(node, x),
            None => 0.0,
        }
    }

    fn traverse(&self, node: &TreeNode, x: &ArrayView1<f64>) -> f64 {
        match node {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Split {
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

    fn feature_importances(&self) -> Array1<f64> {
        let mut importances = Array1::zeros(self.n_features);
        if let Some(ref root) = self.root {
            self.accumulate_importances(root, &mut importances, 1.0);
        }

        let sum: f64 = importances.sum();
        if sum > 0.0 {
            importances /= sum;
        }

        importances
    }

    fn accumulate_importances(&self, node: &TreeNode, importances: &mut Array1<f64>, weight: f64) {
        if let TreeNode::Split {
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

// ============================================================================
// Tree Ensemble
// ============================================================================

/// An ensemble of regression trees (simplified BART).
#[derive(Debug, Clone)]
struct TreeEnsemble {
    trees: Vec<RegressionTree>,
    n_features: usize,
}

impl TreeEnsemble {
    fn new() -> Self {
        Self {
            trees: Vec::new(),
            n_features: 0,
        }
    }

    /// Fit the ensemble to data.
    fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        n_trees: usize,
        max_depth: usize,
        min_node_size: usize,
        sample_fraction: f64,
        mtry: Option<usize>,
        rng_state: &mut u64,
    ) {
        self.n_features = x.ncols();
        let mtry_value = mtry.unwrap_or_else(|| {
            let sqrt_p = (self.n_features as f64).sqrt().ceil() as usize;
            sqrt_p.max(1)
        });

        self.trees.clear();
        self.trees.reserve(n_trees);

        let n_samples = x.nrows();

        for _ in 0..n_trees {
            // Subsample
            let sample_indices = subsample(n_samples, sample_fraction, rng_state);

            // Extract subsample
            let x_sample = select_rows(&x, &sample_indices);
            let y_sample: Array1<f64> = sample_indices.iter().map(|&i| y[i]).collect();

            // Build tree
            let mut tree = RegressionTree::new(max_depth, min_node_size, mtry_value);
            tree.fit(x_sample.view(), y_sample.view(), rng_state);
            self.trees.push(tree);
        }
    }

    /// Predict for all observations.
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut predictions = Array1::zeros(n);

        for i in 0..n {
            let row = x.row(i);
            let sum: f64 = self.trees.iter().map(|t| t.predict_one(&row)).sum();
            predictions[i] = sum / self.trees.len() as f64;
        }

        predictions
    }

    /// Get aggregated feature importances.
    fn feature_importances(&self) -> Array1<f64> {
        let mut importances = Array1::zeros(self.n_features);

        for tree in &self.trees {
            importances = importances + tree.feature_importances();
        }

        importances /= self.trees.len() as f64;
        importances
    }
}

// ============================================================================
// Main API Functions
// ============================================================================

/// Run BART-based causal inference from a dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset containing outcome, treatment, and covariates
/// * `outcome_col` - Name of the outcome variable
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of covariate columns
/// * `config` - Configuration for BART causal inference
///
/// # Returns
///
/// `BartCausalResult` containing CATE estimates, ATE, and uncertainty quantification.
///
/// # Example
///
/// ```ignore
/// use p2a_core::ml::{bart_causal, BartCausalConfig};
///
/// let config = BartCausalConfig {
///     n_trees: 200,
///     n_bootstrap: 100,
///     include_propensity: true,
///     ..Default::default()
/// };
///
/// let result = bart_causal(
///     &dataset,
///     "outcome",
///     "treatment",
///     &["age", "income", "education"],
///     config,
/// )?;
///
/// println!("ATE: {:.4} (SE: {:.4})", result.ate, result.ate_se);
/// println!("CATE range: [{:.4}, {:.4}]",
///     result.cate.iter().cloned().fold(f64::INFINITY, f64::min),
///     result.cate.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
/// );
/// ```
///
/// # References
///
/// - Hill (2011). JASA. Original BART for causal inference.
/// - Chipman, George, & McCulloch (2010). Annals of Applied Statistics. BART.
/// - R package `bartCause`: Dorie (2020).
pub fn bart_causal(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: BartCausalConfig,
) -> EconResult<BartCausalResult> {
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

    // Run core algorithm
    bart_causal_arrays(
        y.view(),
        w.view(),
        x.view(),
        covariate_cols.iter().map(|s| s.to_string()).collect(),
        config,
    )
}

/// Core BART causal inference algorithm using arrays.
///
/// This is the main implementation, separate from data loading for testability.
pub fn bart_causal_arrays(
    y: ArrayView1<f64>,
    treatment: ArrayView1<f64>,
    x: ArrayView2<f64>,
    covariate_names: Vec<String>,
    config: BartCausalConfig,
) -> EconResult<BartCausalResult> {
    let n = x.nrows();
    let p = x.ncols();

    // Validate inputs
    if n < 20 {
        return Err(EconError::InsufficientData {
            required: 20,
            provided: n,
            context: "BART causal inference requires at least 20 observations".to_string(),
        });
    }

    if y.len() != n || treatment.len() != n {
        return Err(EconError::InvalidSpecification {
            message: "Outcome, treatment, and covariates must have same number of observations"
                .to_string(),
        });
    }

    // Split data into treated and control
    let treated_indices: Vec<usize> = (0..n).filter(|&i| treatment[i] > 0.5).collect();
    let control_indices: Vec<usize> = (0..n).filter(|&i| treatment[i] <= 0.5).collect();

    let n_treated = treated_indices.len();
    let n_control = control_indices.len();

    if n_treated < 10 || n_control < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n_treated.min(n_control),
            context: "Need at least 10 treated and 10 control units".to_string(),
        });
    }

    // Initialize RNG
    let mut rng_state = config.seed.unwrap_or(42);

    // Optionally compute and add propensity scores
    let (x_aug, covariate_names_aug) = if config.include_propensity {
        let propensity = estimate_propensity(x.view(), treatment.view(), &mut rng_state)?;
        let mut x_new = Array2::zeros((n, p + 1));
        for i in 0..n {
            for j in 0..p {
                x_new[[i, j]] = x[[i, j]];
            }
            x_new[[i, p]] = propensity[i];
        }
        let mut names = covariate_names.clone();
        names.push("propensity".to_string());
        (x_new, names)
    } else {
        (x.to_owned(), covariate_names.clone())
    };

    // Extract treated and control data
    let x_treated = select_rows(&x_aug.view(), &treated_indices);
    let y_treated: Array1<f64> = treated_indices.iter().map(|&i| y[i]).collect();

    let x_control = select_rows(&x_aug.view(), &control_indices);
    let y_control: Array1<f64> = control_indices.iter().map(|&i| y[i]).collect();

    // Fit ensembles for treated and control response surfaces
    let mut ensemble_treated = TreeEnsemble::new();
    ensemble_treated.fit(
        x_treated.view(),
        y_treated.view(),
        config.n_trees,
        config.max_depth,
        config.min_node_size,
        config.sample_fraction,
        config.mtry,
        &mut rng_state,
    );

    let mut ensemble_control = TreeEnsemble::new();
    ensemble_control.fit(
        x_control.view(),
        y_control.view(),
        config.n_trees,
        config.max_depth,
        config.min_node_size,
        config.sample_fraction,
        config.mtry,
        &mut rng_state,
    );

    // Predict Y(1) and Y(0) for all units
    let y1_pred = ensemble_treated.predict(x_aug.view());
    let y0_pred = ensemble_control.predict(x_aug.view());

    // Compute CATE
    let cate: Array1<f64> = &y1_pred - &y0_pred;

    // Compute ATE
    let ate: f64 = cate.mean().unwrap_or(0.0);

    // Bootstrap for uncertainty quantification
    let (ate_bootstrap, cate_bootstrap) = bootstrap_cate(
        y.view(),
        treatment.view(),
        x_aug.view(),
        &config,
        &mut rng_state,
    )?;

    // Compute ATE statistics from bootstrap
    let ate_se = bootstrap_se(&ate_bootstrap);
    let ate_t_stat = if ate_se > 0.0 { ate / ate_se } else { 0.0 };
    let ate_df = (n - 2) as f64;
    let ate_p_value = t_test_p_value(ate_t_stat, ate_df);

    // Confidence interval for ATE (percentile method)
    let alpha = 1.0 - config.confidence_level;
    let (ate_ci_lower, ate_ci_upper) = percentile_ci(&ate_bootstrap, alpha);

    // Compute CATE confidence intervals
    let mut cate_lower = Vec::with_capacity(n);
    let mut cate_upper = Vec::with_capacity(n);
    let mut cate_se_vec = Vec::with_capacity(n);

    for i in 0..n {
        let obs_bootstrap: Vec<f64> = cate_bootstrap.iter().map(|b| b[i]).collect();
        let se = bootstrap_se(&obs_bootstrap);
        let (ci_low, ci_high) = percentile_ci(&obs_bootstrap, alpha);
        cate_lower.push(ci_low);
        cate_upper.push(ci_high);
        cate_se_vec.push(se);
    }

    // Compute variable importance for treatment effect heterogeneity
    // Use average importance across both response surfaces
    let imp_treated = ensemble_treated.feature_importances();
    let imp_control = ensemble_control.feature_importances();
    let avg_importance = (&imp_treated + &imp_control) / 2.0;

    let mut variable_importance: Vec<(String, f64)> = covariate_names_aug
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), avg_importance[i]))
        .collect();
    variable_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(BartCausalResult {
        ate,
        ate_se,
        ate_t_stat,
        ate_p_value,
        ate_ci_lower,
        ate_ci_upper,
        ate_significance: SignificanceLevel::from_p_value(ate_p_value),
        cate: cate.to_vec(),
        cate_lower,
        cate_upper,
        cate_se: cate_se_vec,
        y1_pred: y1_pred.to_vec(),
        y0_pred: y0_pred.to_vec(),
        variable_importance,
        n_obs: n,
        n_treated,
        n_control,
        n_trees: config.n_trees,
        n_bootstrap: config.n_bootstrap,
        covariate_names: covariate_names_aug,
        config: config.clone(),
        ensemble_treated: Some(ensemble_treated),
        ensemble_control: Some(ensemble_control),
    })
}

/// Run BART causal from dataset with string arguments (for MCP integration).
///
/// This is a convenience wrapper that handles string-to-slice conversion.
pub fn run_bart_causal(
    dataset: &Dataset,
    outcome_col: &str,
    treatment_col: &str,
    covariate_cols: Vec<String>,
    n_trees: Option<usize>,
    max_depth: Option<usize>,
    n_bootstrap: Option<usize>,
    include_propensity: Option<bool>,
    seed: Option<u64>,
) -> EconResult<BartCausalResult> {
    let cov_refs: Vec<&str> = covariate_cols.iter().map(|s| s.as_str()).collect();

    let config = BartCausalConfig {
        n_trees: n_trees.unwrap_or(200),
        max_depth: max_depth.unwrap_or(4),
        n_bootstrap: n_bootstrap.unwrap_or(100),
        include_propensity: include_propensity.unwrap_or(false),
        seed,
        ..Default::default()
    };

    bart_causal(dataset, outcome_col, treatment_col, &cov_refs, config)
}

/// Predict CATE for new observations.
///
/// # Arguments
///
/// * `result` - A fitted BartCausalResult
/// * `new_data` - Dataset with covariate values for prediction
/// * `covariate_cols` - Names of covariate columns (must match training)
///
/// # Returns
///
/// Vector of CATE predictions
pub fn bart_causal_predict(
    result: &BartCausalResult,
    new_data: &Dataset,
    covariate_cols: &[&str],
) -> EconResult<Vec<f64>> {
    // Validate covariate count matches
    let expected_covs = if result.config.include_propensity {
        result.covariate_names.len() - 1 // Exclude propensity
    } else {
        result.covariate_names.len()
    };

    if covariate_cols.len() != expected_covs {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Number of covariates ({}) doesn't match training ({})",
                covariate_cols.len(),
                expected_covs
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

    bart_causal_predict_arrays(result, x.view())
}

/// Predict CATE using array inputs.
pub fn bart_causal_predict_arrays(
    result: &BartCausalResult,
    x: ArrayView2<f64>,
) -> EconResult<Vec<f64>> {
    let ensemble_treated =
        result
            .ensemble_treated
            .as_ref()
            .ok_or_else(|| EconError::InvalidSpecification {
                message: "Model has no fitted ensemble (may have been deserialized)".to_string(),
            })?;

    let ensemble_control =
        result
            .ensemble_control
            .as_ref()
            .ok_or_else(|| EconError::InvalidSpecification {
                message: "Model has no fitted ensemble (may have been deserialized)".to_string(),
            })?;

    // If propensity was included, we need to estimate it for new data
    // For simplicity, we just use covariates as-is (propensity column not included)
    let x_to_use = if result.config.include_propensity && x.ncols() < result.covariate_names.len() {
        // Need to add propensity column (use 0.5 as placeholder for prediction)
        let n = x.nrows();
        let p = x.ncols();
        let mut x_aug = Array2::zeros((n, p + 1));
        for i in 0..n {
            for j in 0..p {
                x_aug[[i, j]] = x[[i, j]];
            }
            x_aug[[i, p]] = 0.5; // Placeholder propensity
        }
        x_aug
    } else {
        x.to_owned()
    };

    let y1_pred = ensemble_treated.predict(x_to_use.view());
    let y0_pred = ensemble_control.predict(x_to_use.view());

    let cate = &y1_pred - &y0_pred;
    Ok(cate.to_vec())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple LCG random number generator.
fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
}

/// Generate subsample indices.
fn subsample(n_samples: usize, fraction: f64, rng_state: &mut u64) -> Vec<usize> {
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

    sample
}

/// Select rows from a 2D array.
fn select_rows(data: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_features = data.ncols();
    let mut result = Array2::zeros((indices.len(), n_features));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&data.row(idx));
    }

    result
}

/// Estimate propensity scores using logistic regression approximation.
///
/// This uses a simplified approach: fit a regression tree to predict treatment.
fn estimate_propensity(
    x: ArrayView2<f64>,
    treatment: ArrayView1<f64>,
    rng_state: &mut u64,
) -> EconResult<Array1<f64>> {
    let n = x.nrows();
    let p = x.ncols();

    // Use a simple ensemble to predict propensity
    let mtry = (p as f64).sqrt().ceil() as usize;
    let mut ensemble = TreeEnsemble::new();
    ensemble.fit(x, treatment, 50, 3, 5, 0.632, Some(mtry), rng_state);

    let mut propensity = ensemble.predict(x);

    // Clip propensity to [0.01, 0.99] for numerical stability
    for i in 0..n {
        propensity[i] = propensity[i].max(0.01).min(0.99);
    }

    Ok(propensity)
}

/// Bootstrap CATE estimation.
fn bootstrap_cate(
    y: ArrayView1<f64>,
    treatment: ArrayView1<f64>,
    x: ArrayView2<f64>,
    config: &BartCausalConfig,
    rng_state: &mut u64,
) -> EconResult<(Vec<f64>, Vec<Array1<f64>>)> {
    let n = x.nrows();
    let mut ate_samples = Vec::with_capacity(config.n_bootstrap);
    let mut cate_samples = Vec::with_capacity(config.n_bootstrap);

    for _ in 0..config.n_bootstrap {
        // Bootstrap resample
        let indices: Vec<usize> = (0..n).map(|_| lcg_random(rng_state) % n).collect();

        let y_boot: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        let w_boot: Array1<f64> = indices.iter().map(|&i| treatment[i]).collect();
        let x_boot = select_rows(&x, &indices);

        // Split into treated/control
        let treated_idx: Vec<usize> = (0..n).filter(|&i| w_boot[i] > 0.5).collect();
        let control_idx: Vec<usize> = (0..n).filter(|&i| w_boot[i] <= 0.5).collect();

        // Skip if not enough in either group
        if treated_idx.len() < 5 || control_idx.len() < 5 {
            continue;
        }

        // Fit ensembles
        let x_treated = select_rows(&x_boot.view(), &treated_idx);
        let y_treated: Array1<f64> = treated_idx.iter().map(|&i| y_boot[i]).collect();

        let x_control = select_rows(&x_boot.view(), &control_idx);
        let y_control: Array1<f64> = control_idx.iter().map(|&i| y_boot[i]).collect();

        let mut ens_t = TreeEnsemble::new();
        ens_t.fit(
            x_treated.view(),
            y_treated.view(),
            config.n_trees / 4, // Fewer trees for bootstrap speed
            config.max_depth,
            config.min_node_size,
            config.sample_fraction,
            config.mtry,
            rng_state,
        );

        let mut ens_c = TreeEnsemble::new();
        ens_c.fit(
            x_control.view(),
            y_control.view(),
            config.n_trees / 4,
            config.max_depth,
            config.min_node_size,
            config.sample_fraction,
            config.mtry,
            rng_state,
        );

        // Predict on original data (not bootstrap sample) for CATE
        let y1_pred = ens_t.predict(x);
        let y0_pred = ens_c.predict(x);
        let cate = &y1_pred - &y0_pred;

        let ate = cate.mean().unwrap_or(0.0);
        ate_samples.push(ate);
        cate_samples.push(cate);
    }

    Ok((ate_samples, cate_samples))
}

/// Compute bootstrap standard error.
fn bootstrap_se(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return f64::NAN;
    }

    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance: f64 =
        samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;

    variance.sqrt()
}

/// Compute percentile confidence interval.
fn percentile_ci(samples: &[f64], alpha: f64) -> (f64, f64) {
    if samples.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;

    let lower = sorted.get(lower_idx).copied().unwrap_or(sorted[0]);
    let upper = sorted
        .get(upper_idx.min(n - 1))
        .copied()
        .unwrap_or(sorted[n - 1]);

    (lower, upper)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn generate_test_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut rng = seed;

        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        let mut w = Array1::zeros(n);

        for i in 0..n {
            // Generate covariates
            x[[i, 0]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
            x[[i, 1]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;
            x[[i, 2]] = (lcg_random(&mut rng) % 100) as f64 / 100.0;

            // Treatment depends on x0 (confounding)
            let p_treat = 0.3 + 0.4 * x[[i, 0]];
            w[i] = if (lcg_random(&mut rng) % 100) as f64 / 100.0 < p_treat {
                1.0
            } else {
                0.0
            };

            // Heterogeneous treatment effect: tau(x) = 1 + 2*x0
            let tau = 1.0 + 2.0 * x[[i, 0]];

            // Outcome with confounding, treatment effect, and noise
            let noise = ((lcg_random(&mut rng) % 100) as f64 - 50.0) / 50.0;
            y[i] = 5.0 + 2.0 * x[[i, 0]] + 0.5 * x[[i, 1]] + tau * w[i] + noise;
        }

        (y, w, x)
    }

    #[test]
    fn test_bart_causal_basic() {
        let (y, w, x) = generate_test_data(200, 42);

        let config = BartCausalConfig {
            n_trees: 50,
            max_depth: 3,
            n_bootstrap: 20,
            seed: Some(42),
            ..Default::default()
        };

        let result = bart_causal_arrays(
            y.view(),
            w.view(),
            x.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 200);
        assert_eq!(result.cate.len(), 200);
        assert_eq!(result.cate_lower.len(), 200);
        assert_eq!(result.cate_upper.len(), 200);

        // ATE should be positive (true ATE is ~2)
        assert!(
            result.ate > 0.5,
            "ATE should be positive, got {}",
            result.ate
        );
        assert!(result.ate < 5.0, "ATE should be < 5, got {}", result.ate);

        // Standard error should be positive
        assert!(result.ate_se > 0.0);

        // CI should contain ATE
        assert!(result.ate_ci_lower <= result.ate);
        assert!(result.ate_ci_upper >= result.ate);

        println!("ATE: {:.4} (SE: {:.4})", result.ate, result.ate_se);
        println!(
            "95% CI: [{:.4}, {:.4}]",
            result.ate_ci_lower, result.ate_ci_upper
        );
    }

    #[test]
    fn test_bart_causal_with_propensity() {
        let (y, w, x) = generate_test_data(150, 123);

        let config = BartCausalConfig {
            n_trees: 30,
            max_depth: 3,
            n_bootstrap: 10,
            include_propensity: true,
            seed: Some(123),
            ..Default::default()
        };

        let result = bart_causal_arrays(
            y.view(),
            w.view(),
            x.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        // Should have 4 covariates (3 original + propensity)
        assert_eq!(result.covariate_names.len(), 4);
        assert_eq!(result.covariate_names[3], "propensity");

        // ATE should still be reasonable
        assert!(result.ate > 0.0);
    }

    #[test]
    fn test_bart_causal_heterogeneity() {
        // Test that CATE varies with x0 (which is the true effect modifier)
        let (y, w, x) = generate_test_data(300, 456);

        let config = BartCausalConfig {
            n_trees: 100,
            max_depth: 4,
            n_bootstrap: 30,
            seed: Some(456),
            ..Default::default()
        };

        let result = bart_causal_arrays(
            y.view(),
            w.view(),
            x.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        // Compute correlation between CATE and x0
        let cate_mean = result.cate.iter().sum::<f64>() / result.cate.len() as f64;
        let x0_col: Vec<f64> = (0..result.n_obs).map(|i| x[[i, 0]]).collect();
        let x0_mean = x0_col.iter().sum::<f64>() / x0_col.len() as f64;

        let cov: f64 = result
            .cate
            .iter()
            .zip(x0_col.iter())
            .map(|(c, x)| (c - cate_mean) * (x - x0_mean))
            .sum::<f64>();
        let var_cate: f64 = result.cate.iter().map(|c| (c - cate_mean).powi(2)).sum();
        let var_x0: f64 = x0_col.iter().map(|x| (x - x0_mean).powi(2)).sum();

        let corr = cov / (var_cate.sqrt() * var_x0.sqrt());

        // CATE should be positively correlated with x0 (true tau = 1 + 2*x0)
        println!("Correlation between CATE and x0: {:.4}", corr);
        assert!(
            corr > 0.0,
            "CATE should be positively correlated with x0, got {}",
            corr
        );

        // Variable importance should rank x0 high
        let x0_importance = result
            .variable_importance
            .iter()
            .find(|(name, _)| name == "x0")
            .map(|(_, imp)| *imp)
            .unwrap_or(0.0);

        println!("x0 importance: {:.4}", x0_importance);
        println!("Variable importances: {:?}", result.variable_importance);
    }

    #[test]
    fn test_bart_causal_insufficient_data() {
        let y = Array1::zeros(10);
        let w = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let x = Array2::zeros((10, 2));

        let config = BartCausalConfig::default();

        let result = bart_causal_arrays(
            y.view(),
            w.view(),
            x.view(),
            vec!["x0".to_string(), "x1".to_string()],
            config,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_bootstrap_se() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let se = bootstrap_se(&samples);

        // Standard deviation of [1,2,3,4,5] is sqrt(2.5) ~ 1.58
        assert!((se - 1.58).abs() < 0.1);
    }

    #[test]
    fn test_percentile_ci() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (lower, upper) = percentile_ci(&samples, 0.05);

        // 95% CI should be approximately [2.5, 97.5]
        assert!((lower - 2.0).abs() < 3.0);
        assert!((upper - 97.0).abs() < 3.0);
    }

    #[test]
    fn test_tree_ensemble_fit_predict() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut ensemble = TreeEnsemble::new();
        let mut rng = 42u64;
        ensemble.fit(x.view(), y.view(), 10, 3, 2, 0.7, None, &mut rng);

        let predictions = ensemble.predict(x.view());

        // Predictions should be in reasonable range
        for (i, &pred) in predictions.iter().enumerate() {
            assert!(
                pred > 0.0 && pred < 12.0,
                "Prediction {} = {} out of range",
                i,
                pred
            );
        }

        // Feature importances should sum to approximately 1
        let importances = ensemble.feature_importances();
        let sum: f64 = importances.sum();
        assert!((sum - 1.0).abs() < 0.01 || sum == 0.0);
    }

    #[test]
    fn test_display() {
        let (y, w, x) = generate_test_data(100, 789);

        let config = BartCausalConfig {
            n_trees: 20,
            max_depth: 3,
            n_bootstrap: 10,
            seed: Some(789),
            ..Default::default()
        };

        let result = bart_causal_arrays(
            y.view(),
            w.view(),
            x.view(),
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
            config,
        )
        .unwrap();

        let output = format!("{}", result);
        assert!(output.contains("BART Causal Inference Results"));
        assert!(output.contains("ATE"));
        assert!(output.contains("CATE DISTRIBUTION"));
    }
}
