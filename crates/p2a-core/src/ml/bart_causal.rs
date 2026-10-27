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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::{default_mtry, heap_chunk_bytes, mul, partition_exact, pow2};
use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::resource::{checkpoint, try_with_capacity, try_zeros2};
use crate::traits::estimator::{SignificanceLevel, t_test_p_value};

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

/// Trees of the propensity model ([`estimate_propensity`]) and their depth.
const PROPENSITY_TREES: usize = 50;
const PROPENSITY_DEPTH: usize = 3;

impl BartCausalConfig {
    /// An upper bound, in bytes, of what [`bart_causal`] holds at once for a
    /// dataset of `n_obs` rows and `n_covariates` covariates under this
    /// configuration: the trees it keeps, its bootstrap draws and the buffers
    /// that are alive together while it runs. Every buffer it counts is
    /// allocated at its exact length, the group indices and the index vectors
    /// of the recursion included, so no vector holds unused capacity. The
    /// allocator's own overhead (chunk headers and free lists) and the dataset
    /// itself come on top. Saturates at `u64::MAX`.
    ///
    /// The bound holds whatever the data are, because each term counts the
    /// most the code can allocate:
    ///
    /// - **Trees.** The two response-surface ensembles of `n_trees` trees
    ///   each, kept for the result, and the two ensembles of `n_trees / 4`
    ///   trees of the bootstrap replicate being fitted; with
    ///   `include_propensity`, the propensity model's trees too. A tree has at
    ///   most `2 L - 1` nodes for `L` leaves; `L` is at most `2^max_depth` and
    ///   at most its subsample divided by `max(min_node_size, 1)`, since every
    ///   child keeps `min_node_size` rows and at least one. A treated tree and
    ///   a control tree are fitted on subsamples that together hold at most
    ///   `sample_fraction` of the rows, so a pair has at most that many rows
    ///   divided by `max(min_node_size, 1)`, plus two, leaves. Every node is
    ///   counted at its heap size.
    /// - **Draws.** The `n_bootstrap x n_obs` matrix of CATE draws, allocated
    ///   before any fit, the ATE draws, and one unit's draws and their sorted
    ///   copy when the intervals are computed.
    /// - **Data and results.** The extracted outcome, treatment and covariates
    ///   (and the column being extracted, cast to `f64` and copied), the group
    ///   indices, the covariates the ensembles use (with the propensity score)
    ///   and their treated and control rows, the predictions; one replicate's
    ///   resample, its rows and groups and its predictions; the intervals and
    ///   standard errors, the result's copies and the sorted copy its summary
    ///   makes.
    /// - **One tree's working copies.** The subsample's flags, indices and
    ///   rows, and what the recursion holds: every level on the path to the
    ///   node being split holds its children's index vectors, together at most
    ///   the subsample, and the covariates its split drew (`mtry`, at most the
    ///   ensembles' covariates). The path is at most `max_depth` levels deep
    ///   and at most the subsample divided by `max(min_node_size, 1)`. The
    ///   split search's sort order adds one subsample, and the list the
    ///   covariates are drawn from one covariate list.
    pub fn memory_bound(&self, n_obs: usize, n_covariates: usize) -> u64 {
        let n = n_obs as u128;
        let p = n_covariates as u128;
        // Covariates of the ensembles: with the propensity score as one more.
        let q_cols = n_covariates.saturating_add(usize::from(self.include_propensity));
        let q = q_cols as u128;
        let s = self.subsample_size(n_obs);
        let leaf = self.min_node_size.max(1) as u128;
        let node = heap_chunk_bytes(size_of::<TreeNode>());
        let tree = size_of::<RegressionTree>() as u128;
        let pairs = (self.n_trees as u128).saturating_add((self.n_trees / 4) as u128);
        let pair = mul(2, self.max_leaves_per_pair(n_obs))
            .saturating_mul(node)
            .saturating_add(2 * tree);
        let mut trees = mul(pairs, pair);
        if self.include_propensity {
            let one = ((1u128 << (PROPENSITY_DEPTH + 1)) - 1) * node + tree;
            trees = trees.saturating_add(PROPENSITY_TREES as u128 * one);
        }
        let b = self.n_bootstrap as u128;
        let draws = mul(mul(8, b), n).saturating_add(mul(24, b));
        // Per row: y, w, and the column being extracted (its cast and its values) (4); the
        // group indices and the outcome by group (2); the predictions y1, y0 and the CATE
        // (3); the replicate's resample indices, y and w (3), its group indices and outcome
        // by group (2) and its predictions (3); the intervals and standard errors (3), the
        // result's copies (3) and the summary's sorted copy (1); the propensity score (1).
        let per_row = 4 + 2 + 3 + 3 + 2 + 3 + 3 + 3 + 1 + 1;
        // Per row and covariate: the extracted covariates, the ensembles' covariates and their
        // treated and control rows, the replicate's rows and their treated and control rows.
        let data = mul(mul(8, n), per_row)
            .saturating_add(mul(mul(8, n), p))
            .saturating_add(mul(mul(32, n), q));
        let levels = (s / leaf).min(self.max_depth as u128) + 1;
        // The covariates one split draws, as `select_features` keeps them.
        let drawn = self
            .mtry
            .unwrap_or_else(|| default_mtry(q_cols))
            .min(q_cols) as u128;
        // Subsample flags, indices and rows; the recursion path (each level's child index
        // vectors and drawn covariates); the split search's order; the list the covariates are
        // drawn from, or the importance sums and one tree's importances.
        let work = n
            .saturating_add(mul(mul(8, s), q.saturating_add(2)))
            .saturating_add(mul(mul(8, s.saturating_add(drawn)), levels))
            .saturating_add(mul(8, s))
            .saturating_add(mul(16, q));
        u64::try_from(
            trees
                .saturating_add(draws)
                .saturating_add(data)
                .saturating_add(work),
        )
        .unwrap_or(u64::MAX)
    }

    /// The most leaves a treated tree and a control tree fitted together can
    /// have for `n_obs` rows ([`Self::memory_bound`]).
    fn max_leaves_per_pair(&self, n_obs: usize) -> u128 {
        let s = self.subsample_size(n_obs);
        let leaf = self.min_node_size.max(1) as u128;
        (s / leaf + 2).min(pow2(self.max_depth).saturating_mul(2))
    }

    /// The rows of a subsample drawn from `n_obs` rows, as `subsample` draws it; a treated and a
    /// control tree draw theirs from groups that together hold `n_obs` rows.
    fn subsample_size(&self, n_obs: usize) -> u128 {
        ((n_obs as f64) * self.sample_fraction) as usize as u128
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
            // Each child's vector at its exact length: every level of the path holds them.
            let (left_indices, right_indices) = partition_exact(indices.iter().copied(), |i| {
                x[[i, best_feature]] <= best_threshold
            });

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
        let mut selected = Vec::with_capacity(self.mtry.min(self.n_features));
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

    /// Best (feature, threshold) by pooled within-child squared error.
    ///
    /// Minimising the children's total sum of squares is maximising
    /// `s_left^2 / n_left + s_right^2 / n_right` for sums of `y - m0` (any
    /// constant `m0`; the node mean keeps the sums small). Each feature's node
    /// values are sorted once and the thresholds (midpoints of consecutive
    /// distinct values, ascending) are swept with a pointer that adds every
    /// observation with `x <= threshold` to the left, the partition the
    /// re-partitioning search this replaces used. `x` must be finite.
    ///
    /// The score is accumulated in sorted-x order, where that search summed
    /// each child's squares around its mean in index order, so the chosen
    /// split is the same except where two candidates have equal within-child
    /// sums of squares, exactly or to within rounding: the two searches may
    /// break those ties either way. A response with repeated values (the 0/1
    /// treatment the propensity ensemble fits, counts, integer outcomes) can
    /// give candidates exactly equal sums of squares, so on such data the
    /// tree can take the other, equally good split at that node and grow
    /// differently below it, and the propensity scores and CATE draws built
    /// on it change with it.
    fn find_best_split(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
    ) -> Option<(usize, f64)> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_split = None;

        let n = indices.len();
        if n < 2 {
            return None;
        }
        let m0 = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
        let s_total: f64 = indices.iter().map(|&i| y[i] - m0).sum();

        let mut order: Vec<usize> = indices.to_vec();
        for &feature in features {
            order.sort_by(|&a, &b| x[[a, feature]].total_cmp(&x[[b, feature]]));

            let mut s_left = 0.0;
            let mut ptr = 0usize;
            let mut lower = x[[order[0], feature]];

            for pos in 1..n {
                let upper = x[[order[pos], feature]];
                if upper == lower {
                    continue;
                }
                let threshold = (lower + upper) / 2.0;
                lower = upper;

                while ptr < n && x[[order[ptr], feature]] <= threshold {
                    s_left += y[order[ptr]] - m0;
                    ptr += 1;
                }

                let (n_left, n_right) = (ptr, n - ptr);
                if n_left == 0
                    || n_right == 0
                    || n_left < self.min_node_size
                    || n_right < self.min_node_size
                {
                    continue;
                }

                let s_right = s_total - s_left;
                let score = s_left * s_left / n_left as f64 + s_right * s_right / n_right as f64;
                if score > best_score {
                    best_score = score;
                    best_split = Some((feature, threshold));
                }
            }
        }

        best_split
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
    ) -> EconResult<()> {
        self.n_features = x.ncols();
        let mtry_value = mtry.unwrap_or_else(|| default_mtry(self.n_features));

        self.trees = try_with_capacity(n_trees, "BART trees")?;

        let n_samples = x.nrows();

        for _ in 0..n_trees {
            checkpoint()?;

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
        Ok(())
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

/// Check the counts, fractions and levels of a BART-causal configuration.
fn validate_bart_causal_config(config: &BartCausalConfig) -> EconResult<()> {
    if config.n_trees == 0 {
        return Err(EconError::InvalidSpecification {
            message: "n_trees must be at least 1 (got 0)".to_string(),
        });
    }
    if config.n_bootstrap < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "n_bootstrap must be at least 2 to estimate a standard error (got {})",
                config.n_bootstrap
            ),
        });
    }
    let sf = config.sample_fraction;
    if !(sf.is_finite() && sf > 0.0 && sf <= 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("sample_fraction must be in (0, 1] (got {sf})"),
        });
    }
    let cl = config.confidence_level;
    if !(cl.is_finite() && cl > 0.0 && cl < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("confidence_level must be in (0, 1) (got {cl})"),
        });
    }
    Ok(())
}

/// `InvalidSpecification` naming the first non-finite value of `values`.
fn check_finite_1d(values: &ArrayView1<f64>, name: &str) -> EconResult<()> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(EconError::InvalidSpecification {
            message: format!(
                "{name} must contain only finite values (row {i} is {})",
                values[i]
            ),
        }),
        None => Ok(()),
    }
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
    validate_bart_causal_config(&config)?;

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

    check_finite_1d(&y, "y")?;
    check_finite_1d(&treatment, "treatment")?;
    if let Some(((i, j), v)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("x must contain only finite values (row {i}, column {j} is {v})"),
        });
    }

    // Split data into treated and control
    let (treated_indices, control_indices) = treated_and_control(&treatment);

    let n_treated = treated_indices.len();
    let n_control = control_indices.len();

    if n_treated < 10 || n_control < 10 {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: n_treated.min(n_control),
            context: "Need at least 10 treated and 10 control units".to_string(),
        });
    }

    // Bootstrap CATE draws (one row per successful replicate), allocated
    // before any ensemble is fitted
    let mut cate_draws = try_zeros2(config.n_bootstrap, n, "BART bootstrap CATE draws")?;

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
    )?;

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
    )?;

    // Predict Y(1) and Y(0) for all units
    let y1_pred = ensemble_treated.predict(x_aug.view());
    let y0_pred = ensemble_control.predict(x_aug.view());

    // Compute CATE
    let cate: Array1<f64> = &y1_pred - &y0_pred;

    // Compute ATE
    let ate: f64 = cate.mean().unwrap_or(0.0);

    // Bootstrap for uncertainty quantification
    let (ate_bootstrap, n_draws) = bootstrap_cate(
        y.view(),
        treatment.view(),
        x_aug.view(),
        &config,
        &mut rng_state,
        &mut cate_draws,
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
        let obs_bootstrap: Vec<f64> = cate_draws.slice(s![..n_draws, i]).to_vec();
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
    let config = run_bart_causal_config(n_trees, max_depth, n_bootstrap, include_propensity, seed);
    bart_causal(dataset, outcome_col, treatment_col, &cov_refs, config)
}

/// The configuration [`run_bart_causal`] fits with for these arguments (the
/// defaults of [`BartCausalConfig`] where an argument is `None`), for example
/// to check its [`BartCausalConfig::memory_bound`] first.
pub fn run_bart_causal_config(
    n_trees: Option<usize>,
    max_depth: Option<usize>,
    n_bootstrap: Option<usize>,
    include_propensity: Option<bool>,
    seed: Option<u64>,
) -> BartCausalConfig {
    let defaults = BartCausalConfig::default();
    BartCausalConfig {
        n_trees: n_trees.unwrap_or(defaults.n_trees),
        max_depth: max_depth.unwrap_or(defaults.max_depth),
        n_bootstrap: n_bootstrap.unwrap_or(defaults.n_bootstrap),
        include_propensity: include_propensity.unwrap_or(defaults.include_propensity),
        seed,
        ..defaults
    }
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

use super::lcg_random;

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

/// The treated rows of `w` (above 0.5) and the control rows (the others), each vector at its
/// exact length, as [`BartCausalConfig::memory_bound`] counts them. `w` is finite (the caller
/// checked the treatment), so each row is either treated or control.
fn treated_and_control(w: &ArrayView1<f64>) -> (Vec<usize>, Vec<usize>) {
    partition_exact(0..w.len(), |i| w[i] > 0.5)
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
    ensemble.fit(
        x,
        treatment,
        PROPENSITY_TREES,
        PROPENSITY_DEPTH,
        5,
        0.632,
        Some(mtry),
        rng_state,
    )?;

    let mut propensity = ensemble.predict(x);

    // Clip propensity to [0.01, 0.99] for numerical stability
    for i in 0..n {
        propensity[i] = propensity[i].max(0.01).min(0.99);
    }

    Ok(propensity)
}

/// Bootstrap CATE estimation.
///
/// Writes the CATE draw of every successful replicate into the next row of
/// `draws` (`n_bootstrap x n`, allocated by the caller) and returns the ATE
/// draws together with the number of rows filled. Replicates with fewer than
/// 5 treated or control units are skipped and leave no row.
fn bootstrap_cate(
    y: ArrayView1<f64>,
    treatment: ArrayView1<f64>,
    x: ArrayView2<f64>,
    config: &BartCausalConfig,
    rng_state: &mut u64,
    draws: &mut Array2<f64>,
) -> EconResult<(Vec<f64>, usize)> {
    let n = x.nrows();
    let mut ate_samples = try_with_capacity(config.n_bootstrap, "BART bootstrap ATE draws")?;
    let mut n_filled = 0usize;

    for _ in 0..config.n_bootstrap {
        checkpoint()?;

        // Bootstrap resample
        let indices: Vec<usize> = (0..n).map(|_| lcg_random(rng_state) % n).collect();

        let y_boot: Array1<f64> = indices.iter().map(|&i| y[i]).collect();
        let w_boot: Array1<f64> = indices.iter().map(|&i| treatment[i]).collect();
        let x_boot = select_rows(&x, &indices);

        // Split into treated/control
        let (treated_idx, control_idx) = treated_and_control(&w_boot.view());

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
        )?;

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
        )?;

        // Predict on original data (not bootstrap sample) for CATE
        let y1_pred = ens_t.predict(x);
        let y0_pred = ens_c.predict(x);
        let cate = &y1_pred - &y0_pred;

        let ate = cate.mean().unwrap_or(0.0);
        ate_samples.push(ate);
        draws.row_mut(n_filled).assign(&cate);
        n_filled += 1;
    }

    Ok((ate_samples, n_filled))
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
    use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
    use ndarray::array;

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn names3() -> Vec<String> {
        vec!["x0".into(), "x1".into(), "x2".into()]
    }

    fn small() -> BartCausalConfig {
        BartCausalConfig {
            n_trees: 8,
            n_bootstrap: 4,
            max_depth: 3,
            seed: Some(5),
            ..Default::default()
        }
    }

    fn count_nodes(node: &TreeNode) -> u128 {
        match node {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => 1 + count_nodes(left) + count_nodes(right),
        }
    }

    fn nodes_of(tree: &RegressionTree) -> u128 {
        tree.root.as_ref().map_or(0, count_nodes)
    }

    /// Distinct covariate values, alternating treatment and an outcome that varies with every
    /// row: every threshold the node-size checks allow is a candidate, so the trees grow as
    /// large as those checks let them.
    fn splittable_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut rng = seed;
        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        let mut w = Array1::zeros(n);
        for i in 0..n {
            for j in 0..3 {
                x[[i, j]] = (lcg_random(&mut rng) % 1_000_003) as f64 + i as f64 * 1e-7;
            }
            w[i] = (i % 2) as f64;
            y[i] = (lcg_random(&mut rng) % 1000) as f64 / 1000.0 + x[[i, 0]] * 1e-6;
        }
        (y, w, x)
    }

    /// The leaf bound behind `memory_bound` holds for every pair of a treated and a control
    /// tree of fitted ensembles, at node sizes 0 to 5 and depths 2 to 100.
    #[test]
    fn memory_bound_counts_at_least_the_nodes_every_tree_pair_grows() {
        for (n, min_node_size, max_depth) in [
            (300, 5, 4),
            (300, 5, 100),
            (300, 1, 100),
            (300, 0, 100),
            (200, 2, 2),
            (800, 0, 100),
        ] {
            let cfg = BartCausalConfig {
                n_trees: 4,
                n_bootstrap: 2,
                min_node_size,
                max_depth,
                seed: Some(9),
                ..Default::default()
            };
            let bound = 2 * cfg.max_leaves_per_pair(n) - 1;
            let (y, w, x) = splittable_data(n, 3 + n as u64);
            let fit =
                bart_causal_arrays(y.view(), w.view(), x.view(), names3(), cfg.clone()).unwrap();
            let (t, c) = (
                fit.ensemble_treated.as_ref().unwrap(),
                fit.ensemble_control.as_ref().unwrap(),
            );
            let most = t
                .trees
                .iter()
                .zip(&c.trees)
                .map(|(a, b)| nodes_of(a) + nodes_of(b))
                .max()
                .unwrap();
            assert!(
                most <= bound,
                "{cfg:?} on {n} rows: a pair has {most} nodes, bound {bound}"
            );
            // The trees fill a good part of what the bound allows (it is not loose by orders
            // of magnitude), also where the depth stops them first.
            assert!(4 * most >= bound, "{cfg:?} on {n} rows: {most} of {bound}");
        }
    }

    /// The group indices of the fit and of every bootstrap replicate are allocated at their
    /// exact lengths, as `memory_bound` counts them (`filter().collect()` grows a vector by
    /// pushing and can leave close to twice its length allocated), and so is a subsample. The
    /// recursion's child vectors come from `partition_exact`, tested in `ml`.
    #[test]
    fn group_and_subsample_indices_are_allocated_at_their_lengths() {
        for (n, every) in [(1_000, 2), (777, 3), (1_000, 997), (61, 1)] {
            let w = Array1::from_shape_fn(n, |i| if i % every == 0 { 1.0 } else { 0.0 });
            let (treated, control) = treated_and_control(&w.view());
            assert_eq!(treated.len() + control.len(), n);
            assert!(treated.iter().all(|&i| w[i] > 0.5) && control.iter().all(|&i| w[i] < 0.5));
            assert_eq!(treated.capacity(), treated.len(), "{n} rows, every {every}");
            assert_eq!(control.capacity(), control.len(), "{n} rows, every {every}");
        }
        let mut rng = 3;
        for (n, fraction) in [(1_000, 0.632), (20, 0.05)] {
            let sample = subsample(n, fraction, &mut rng);
            assert_eq!(
                sample.capacity(),
                sample.len(),
                "{n} rows, fraction {fraction}"
            );
        }
    }

    #[test]
    fn memory_bound_grows_with_its_arguments_and_saturates() {
        let d = BartCausalConfig::default();
        let at = |cfg: &BartCausalConfig, n: usize, p: usize| cfg.memory_bound(n, p);
        let base = at(&d, 10_000, 4);
        let with = |f: fn(&mut BartCausalConfig)| {
            let mut c = d.clone();
            f(&mut c);
            c
        };
        // Every level of the recursion's path keeps the covariates its split drew.
        assert!(at(&with(|c| c.mtry = Some(16)), 10_000, 16) > at(&d, 10_000, 16));
        assert_eq!(
            at(&with(|c| c.mtry = Some(16)), 10_000, 16),
            at(&with(|c| c.mtry = Some(1_000)), 10_000, 16)
        );
        assert!(at(&with(|c| c.n_trees = 300), 10_000, 4) > base);
        assert!(at(&with(|c| c.n_bootstrap = 150), 10_000, 4) > base);
        assert!(at(&with(|c| c.max_depth = 8), 10_000, 4) > base);
        assert!(at(&with(|c| c.include_propensity = true), 10_000, 4) > base);
        assert!(at(&d, 20_000, 4) > base);
        assert!(at(&d, 10_000, 8) > base);
        // The draws alone: n_bootstrap x n_obs doubles.
        assert!(u128::from(base) > 100 * 10_000 * 8);
        // At the default depth a tree pair has at most 2 * 2^4 leaves however many rows.
        assert_eq!(d.max_leaves_per_pair(10_000_000), 32);
        let huge = with(|c| {
            c.n_trees = usize::MAX;
            c.max_depth = usize::MAX;
            c.min_node_size = 0;
            c.n_bootstrap = usize::MAX;
        });
        assert_eq!(at(&huge, usize::MAX, usize::MAX), u64::MAX);
        assert_eq!(at(&d, usize::MAX, 4), u64::MAX);
        assert!(at(&d, 0, 0) > 0);
    }

    /// `run_bart_causal_config` gives the configuration `run_bart_causal` used before it
    /// existed: the arguments, and the defaults where they are `None`.
    #[test]
    fn run_config_takes_the_arguments_and_the_defaults() {
        let c = run_bart_causal_config(None, None, None, None, None);
        let d = BartCausalConfig::default();
        assert_eq!(
            (
                c.n_trees,
                c.max_depth,
                c.n_bootstrap,
                c.include_propensity,
                c.seed
            ),
            (200, 4, 100, false, None)
        );
        assert_eq!(
            (c.min_node_size, c.sample_fraction, c.mtry),
            (d.min_node_size, d.sample_fraction, d.mtry)
        );
        let c = run_bart_causal_config(Some(7), Some(2), Some(3), Some(true), Some(9));
        assert_eq!(
            (
                c.n_trees,
                c.max_depth,
                c.n_bootstrap,
                c.include_propensity,
                c.seed
            ),
            (7, 2, 3, true, Some(9))
        );
    }

    #[test]
    fn bart_causal_huge_counts_are_errors_not_aborts() {
        let (y, w, x) = generate_test_data(60, 3);
        for cfg in [
            BartCausalConfig {
                n_trees: 1usize << 44,
                ..small()
            },
            BartCausalConfig {
                n_bootstrap: 1usize << 44,
                ..small()
            },
        ] {
            let r = with_alloc_ceiling(Some(1 << 30), || {
                bart_causal_arrays(y.view(), w.view(), x.view(), names3(), cfg)
            });
            assert!(
                matches!(r, Err(EconError::ResourceLimit(_))),
                "{:?}",
                r.err()
            );
        }
    }

    #[test]
    fn bart_causal_validates_parameters() {
        let ok = small();
        assert!(validate_bart_causal_config(&ok).is_ok());
        for (cfg, param) in [
            (
                BartCausalConfig {
                    n_trees: 0,
                    ..ok.clone()
                },
                "n_trees",
            ),
            (
                BartCausalConfig {
                    n_bootstrap: 1,
                    ..ok.clone()
                },
                "n_bootstrap",
            ),
            (
                BartCausalConfig {
                    sample_fraction: 1.5,
                    ..ok.clone()
                },
                "sample_fraction",
            ), // looped forever
            (
                BartCausalConfig {
                    sample_fraction: 0.0,
                    ..ok.clone()
                },
                "sample_fraction",
            ),
            (
                BartCausalConfig {
                    confidence_level: 1.0,
                    ..ok.clone()
                },
                "confidence_level",
            ),
            (
                BartCausalConfig {
                    confidence_level: f64::NAN,
                    ..ok.clone()
                },
                "confidence_level",
            ),
        ] {
            let e = validate_bart_causal_config(&cfg).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.starts_with(param)),
                "{e}"
            );
        }
    }

    #[test]
    fn bart_causal_honours_cancellation_and_rejects_non_finite_inputs() {
        let (mut y, w, x) = generate_test_data(60, 3);
        let r = with_budget(cancelled(), || {
            bart_causal_arrays(y.view(), w.view(), x.view(), names3(), small())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{:?}", r.err());
        y[5] = f64::INFINITY;
        let r = bart_causal_arrays(y.view(), w.view(), x.view(), names3(), small());
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message }) if message.contains("finite")),
            "{:?}",
            r.err()
        );
    }

    /// Verbatim copy of `find_best_split` and `compute_split_mse` before the
    /// sorted sweep, kept as the reference for the split sweep's guard tests.
    fn naive_find_best_split(
        min_node_size: usize,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        indices: &[usize],
        features: &[usize],
    ) -> Option<(usize, f64)> {
        fn compute_split_mse(y: &ArrayView1<f64>, left: &[usize], right: &[usize]) -> f64 {
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

                if left_indices.len() < min_node_size || right_indices.len() < min_node_size {
                    continue;
                }

                let mse = compute_split_mse(y, &left_indices, &right_indices);
                if mse < best_mse {
                    best_mse = mse;
                    best_split = Some((feature, threshold));
                }
            }
        }

        best_split
    }

    /// Within-child sum of squares of `split` over `idx`, two-pass; NaN for
    /// no split.
    fn split_sse(
        x: &Array2<f64>,
        y: &Array1<f64>,
        idx: &[usize],
        split: Option<(usize, f64)>,
    ) -> f64 {
        let Some((feature, threshold)) = split else {
            return f64::NAN;
        };
        let (left, right): (Vec<usize>, Vec<usize>) =
            idx.iter().partition(|&&i| x[[i, feature]] <= threshold);
        let ss = |side: &[usize]| {
            let mean = side.iter().map(|&i| y[i]).sum::<f64>() / side.len() as f64;
            side.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>()
        };
        ss(&left) + ss(&right)
    }

    /// Two splits with exactly equal within-child SSE, 0.8: on this 0/1
    /// response x <= 0.5 (1 | 5) and x <= 2 (5 | 1) are mirror images.
    /// Rounding decides which one a search sees as better, so either is a
    /// correct answer; the sweep and the old search must each return one.
    #[test]
    fn regression_split_tie_takes_one_of_the_tied_splits() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 1.0, 1.0, 1.0, 3.0, 0.0]).unwrap();
        let y = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let idx: Vec<usize> = (0..6).collect();
        let tree = RegressionTree {
            n_features: 1,
            ..RegressionTree::new(4, 1, 1)
        };
        for (who, split) in [
            (
                "sweep",
                tree.find_best_split(&x.view(), &y.view(), &idx, &[0]),
            ),
            (
                "reference",
                naive_find_best_split(1, &x.view(), &y.view(), &idx, &[0]),
            ),
        ] {
            assert!(
                matches!(split, Some((0, t)) if t == 0.5 || t == 2.0),
                "{who}: {split:?}"
            );
            let sse = split_sse(&x, &y, &idx, split);
            assert!((sse - 0.8).abs() < 1e-12, "{who}: SSE {sse}");
        }
    }

    /// Reference guard on tie-prone data: covariates on a few levels and a
    /// 0/1 response, as the propensity ensemble sees. The sweep must choose
    /// the old search's split or a different one with the same within-child
    /// SSE.
    #[test]
    fn regression_split_sweep_on_a_binary_response_matches_or_ties_the_reference() {
        let mut s = 2026u64;
        for rep in 0..2000 {
            let n = 6 + lcg_random(&mut s) % 40;
            let p = 1 + lcg_random(&mut s) % 2;
            let levels = 2 + lcg_random(&mut s) % 5;
            let min_node_size = [1usize, 3, 5][lcg_random(&mut s) % 3];
            let x = Array2::from_shape_fn((n, p), |_| (lcg_random(&mut s) % levels) as f64);
            let y = Array1::from_shape_fn(n, |_| (lcg_random(&mut s) % 2) as f64);
            let idx: Vec<usize> = (0..n).collect();
            let features: Vec<usize> = (0..p).collect();
            let tree = RegressionTree {
                n_features: p,
                ..RegressionTree::new(4, min_node_size, p)
            };
            let got = tree.find_best_split(&x.view(), &y.view(), &idx, &features);
            let want = naive_find_best_split(min_node_size, &x.view(), &y.view(), &idx, &features);
            if got != want {
                let (a, b) = (split_sse(&x, &y, &idx, got), split_sse(&x, &y, &idx, want));
                assert!(
                    (a - b).abs() <= 1e-12 * b.max(1.0),
                    "rep={rep}: {got:?} SSE {a} vs {want:?} SSE {b}"
                );
            }
        }
    }

    /// Reference guard on a continuous response (no exact SSE ties): the
    /// sorted sweep must choose the same split as the re-partitioning search
    /// it replaced.
    #[test]
    fn regression_split_sweep_matches_naive_reference() {
        let (y, _w, x) = generate_test_data(400, 77); // x on a 0.01 grid: many tied values
        let mut s = 4321u64;
        for min_node_size in [1usize, 5, 20] {
            let tree = RegressionTree {
                n_features: 3,
                ..RegressionTree::new(4, min_node_size, 3)
            };
            for rep in 0..30 {
                let mut idx: Vec<usize> = (0..400).collect();
                for i in (1..idx.len()).rev() {
                    let j = lcg_random(&mut s) % (i + 1);
                    idx.swap(i, j);
                }
                idx.truncate(20 + lcg_random(&mut s) % 380);
                let got = tree.find_best_split(&x.view(), &y.view(), &idx, &[0, 1, 2]);
                let want =
                    naive_find_best_split(min_node_size, &x.view(), &y.view(), &idx, &[0, 1, 2]);
                assert_eq!(got, want, "min_node_size={min_node_size} rep={rep}");
            }
        }
    }

    /// Golden (characterization) test: captured on the code before the
    /// sweep and the pre-allocated draw matrix, on a continuous outcome and
    /// without the propensity ensemble, so no SSE tie breaks differently; it
    /// passes before and after.
    #[test]
    fn bart_causal_outputs_unchanged_by_the_sweep() {
        const GOLDEN: [u64; 5] = [
            4611755592916943784,
            4598325582365110606,
            4611869894934495320,
            4611929815967930294,
            4611545322639029136,
        ];
        let (y, w, x) = generate_test_data(80, 42);
        let cfg = BartCausalConfig {
            n_trees: 12,
            n_bootstrap: 6,
            max_depth: 4,
            seed: Some(7),
            ..Default::default()
        };
        let r = bart_causal_arrays(y.view(), w.view(), x.view(), names3(), cfg).unwrap();
        let got = [
            r.ate,
            r.ate_se,
            r.cate[0],
            r.cate_lower[0],
            r.cate_upper[79],
        ];
        let bits = got.map(f64::to_bits);
        assert_eq!(bits, GOLDEN, "got {bits:?} = {got:?}");
    }

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
        ensemble
            .fit(x.view(), y.view(), 10, 3, 2, 0.7, None, &mut rng)
            .unwrap();

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
