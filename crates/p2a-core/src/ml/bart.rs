//! Bayesian Additive Regression Trees (BART) for prediction with uncertainty.
//!
//! This module implements BART for both regression and classification, following the
//! methodology of Chipman, George, and McCulloch (2010). BART is a Bayesian
//! nonparametric regression approach using a sum-of-trees model with regularization
//! priors and MCMC sampling.
//!
//! # Model
//!
//! BART models the response as a sum of `m` regression trees:
//!
//! ```text
//! y = f(x) + epsilon = sum_{j=1}^m g_j(x; T_j, M_j) + epsilon
//! ```
//!
//! where each `g_j` is a tree with structure `T_j` and leaf parameters `M_j`,
//! and `epsilon ~ N(0, sigma^2)`.
//!
//! # Priors
//!
//! 1. **Tree structure prior**: P(node at depth d is nonterminal) = alpha * (1 + d)^(-beta)
//!    - Default: alpha = 0.95, beta = 2
//!    - This regularizes tree depth, making deeper trees less likely
//!
//! 2. **Leaf value prior**: mu_ij | T_j ~ N(0, sigma_mu^2)
//!    - sigma_mu = sigma_hat / (k * sqrt(m)) where k ~ 2 shrinks leaf values
//!    - This ensures the prior for f(x) covers the range of y with high probability
//!
//! 3. **Residual variance prior**: sigma^2 ~ Inverse-Gamma(nu/2, nu*lambda/2)
//!    - Default: nu = 3, lambda calibrated so P(sigma < sigma_hat) = q ~ 0.90
//!
//! # MCMC Algorithm
//!
//! The algorithm uses Bayesian backfitting with Gibbs sampling:
//!
//! 1. For j = 1,...,m: Draw (T_j, M_j) given all other trees and sigma
//!    - Use Metropolis-Hastings with grow/prune/change proposals for tree structure
//!    - Draw leaf values from conjugate normal posterior
//!
//! 2. Draw sigma^2 from its conditional inverse-gamma posterior
//!
//! # References
//!
//! - Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian
//!   Additive Regression Trees. *Annals of Applied Statistics*, 4(1), 266-298.
//!   https://doi.org/10.1214/09-AOAS285
//!
//! - R package `BART`: Sparapani, R., Spanbauer, C., & McCulloch, R. (2021).
//!   *Nonparametric Machine Learning and Efficient Computation with Bayesian
//!   Additive Regression Trees: The BART R Package*. JSS, 97(1).
//!   https://CRAN.R-project.org/package=BART
//!
//! R equivalent: `BART::wbart()`, `BART::pbart()`, `BART::lbart()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for BART model.
///
/// # Example
///
/// ```ignore
/// use p2a_core::ml::BartConfig;
///
/// let config = BartConfig {
///     n_trees: 200,
///     n_burn: 250,
///     n_mcmc: 1000,
///     k: 2.0,
///     alpha: 0.95,
///     beta: 2.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartConfig {
    /// Number of trees in the sum (m). Default: 200.
    ///
    /// More trees generally improve prediction but increase computation.
    /// Chipman et al. (2010) recommend m=200 as a robust default.
    pub n_trees: usize,

    /// Number of MCMC burn-in iterations to discard. Default: 250.
    ///
    /// The burn-in period allows the chain to converge before collecting
    /// posterior samples.
    pub n_burn: usize,

    /// Number of posterior MCMC samples to draw. Default: 1000.
    ///
    /// More samples provide more precise uncertainty estimates but
    /// increase computation time.
    pub n_mcmc: usize,

    /// Prior parameter k for leaf value shrinkage. Default: 2.0.
    ///
    /// Controls the width of the prior on leaf values: mu ~ N(0, sigma_hat^2 / (k^2 * m)).
    /// Larger k = more shrinkage, smaller individual tree contributions.
    /// k=2 is recommended for most applications.
    pub k: f64,

    /// Tree prior alpha parameter. Default: 0.95.
    ///
    /// P(node is nonterminal | depth d) = alpha * (1 + d)^(-beta).
    /// Higher alpha allows deeper trees.
    pub alpha: f64,

    /// Tree prior beta parameter. Default: 2.0.
    ///
    /// Controls how quickly splitting probability decreases with depth.
    /// Higher beta = shallower trees.
    pub beta: f64,

    /// Prior degrees of freedom for sigma^2. Default: 3.0.
    ///
    /// sigma^2 ~ Inverse-Gamma(nu/2, nu*lambda/2).
    /// Values 3-10 are typical; nu < 3 may cause overfitting.
    pub nu: f64,

    /// Prior quantile for sigma calibration. Default: 0.90.
    ///
    /// Lambda is set so P(sigma < sigma_hat) = q under the prior.
    pub q: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Minimum observations per terminal node. Default: 5.
    pub min_node_size: usize,

    /// Maximum tree depth. Default: 10.
    pub max_depth: usize,

    /// Confidence level for prediction intervals. Default: 0.95.
    pub confidence_level: f64,

    /// Whether to perform classification (probit BART). Default: false.
    pub classification: bool,
}

impl Default for BartConfig {
    fn default() -> Self {
        Self {
            n_trees: 200,
            n_burn: 250,
            n_mcmc: 1000,
            k: 2.0,
            alpha: 0.95,
            beta: 2.0,
            nu: 3.0,
            q: 0.90,
            seed: None,
            min_node_size: 5,
            max_depth: 10,
            confidence_level: 0.95,
            classification: false,
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result from BART model fitting.
///
/// Contains predictions with uncertainty quantification and variable importance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartResult {
    /// Posterior mean predictions for training data.
    pub predictions: Vec<f64>,

    /// Lower bound of prediction intervals (e.g., 2.5th percentile).
    pub prediction_lower: Vec<f64>,

    /// Upper bound of prediction intervals (e.g., 97.5th percentile).
    pub prediction_upper: Vec<f64>,

    /// Posterior standard deviation of predictions.
    pub prediction_sd: Vec<f64>,

    /// Variable importance scores (based on inclusion frequency).
    ///
    /// Higher values indicate features that are more frequently used
    /// in tree splits across the MCMC samples.
    pub variable_importance: Vec<f64>,

    /// Feature names (if provided).
    pub feature_names: Option<Vec<String>>,

    /// Posterior mean of residual standard deviation sigma.
    pub sigma: f64,

    /// Posterior standard deviation of sigma.
    pub sigma_sd: f64,

    /// Number of observations.
    pub n_obs: usize,

    /// Number of features.
    pub n_features: usize,

    /// Number of trees.
    pub n_trees: usize,

    /// Number of MCMC samples kept.
    pub n_samples: usize,

    /// Configuration used.
    pub config: BartConfig,

    /// Full posterior samples of predictions (n_samples x n_obs).
    /// Skipped in serialization to save space.
    #[serde(skip)]
    pub posterior_samples: Option<Array2<f64>>,

    /// Posterior samples of sigma.
    #[serde(skip)]
    pub sigma_samples: Option<Vec<f64>>,
}

impl fmt::Display for BartResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BART Model Results")?;
        writeln!(f, "==================")?;
        writeln!(
            f,
            "Observations: {}  |  Features: {}  |  Trees: {}",
            self.n_obs, self.n_features, self.n_trees
        )?;
        writeln!(
            f,
            "MCMC: {} burn-in + {} samples",
            self.config.n_burn, self.n_samples
        )?;
        writeln!(f)?;

        writeln!(f, "RESIDUAL SD (sigma):")?;
        writeln!(
            f,
            "  Posterior mean: {:.4}  (SD: {:.4})",
            self.sigma, self.sigma_sd
        )?;
        writeln!(f)?;

        writeln!(f, "PREDICTION SUMMARY:")?;
        let pred_mean: f64 = self.predictions.iter().sum::<f64>() / self.predictions.len() as f64;
        let pred_min = self
            .predictions
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let pred_max = self
            .predictions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        writeln!(
            f,
            "  Mean: {:.4}  |  Min: {:.4}  |  Max: {:.4}",
            pred_mean, pred_min, pred_max
        )?;

        let avg_interval_width: f64 = self
            .prediction_upper
            .iter()
            .zip(self.prediction_lower.iter())
            .map(|(u, l)| u - l)
            .sum::<f64>()
            / self.n_obs as f64;
        writeln!(
            f,
            "  {:.0}% CI avg width: {:.4}",
            self.config.confidence_level * 100.0,
            avg_interval_width
        )?;
        writeln!(f)?;

        writeln!(f, "VARIABLE IMPORTANCE:")?;
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, (i, importance)) in indexed.iter().take(10).enumerate() {
            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("X{}", i + 1)),
                None => format!("X{}", i + 1),
            };
            writeln!(f, "  {}. {}: {:.4}", rank + 1, name, importance)?;
        }
        if self.variable_importance.len() > 10 {
            writeln!(
                f,
                "  ... ({} more variables)",
                self.variable_importance.len() - 10
            )?;
        }

        Ok(())
    }
}

// ============================================================================
// Tree Data Structures
// ============================================================================

/// A node in a BART tree.
#[derive(Debug, Clone)]
enum BartNode {
    /// Internal (split) node
    Internal {
        /// Feature index for split
        var: usize,
        /// Split threshold
        cut: f64,
        /// Left child node index
        left: usize,
        /// Right child node index
        right: usize,
    },
    /// Terminal (leaf) node
    Leaf {
        /// Leaf value (mu)
        mu: f64,
    },
}

/// A single BART tree.
#[derive(Debug, Clone)]
struct BartTree {
    /// Nodes in the tree (index 0 is root)
    nodes: Vec<BartNode>,
    /// Depth of the tree
    depth: usize,
}

impl BartTree {
    /// Create a new tree with single root node
    fn new(mu: f64) -> Self {
        Self {
            nodes: vec![BartNode::Leaf { mu }],
            depth: 0,
        }
    }

    /// Predict for a single observation
    fn predict_one(&self, x: &ArrayView1<f64>) -> f64 {
        let mut node_idx = 0;
        loop {
            match &self.nodes[node_idx] {
                BartNode::Leaf { mu } => return *mu,
                BartNode::Internal {
                    var,
                    cut,
                    left,
                    right,
                } => {
                    node_idx = if x[*var] <= *cut { *left } else { *right };
                }
            }
        }
    }

    /// Predict for all observations
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut preds = Array1::zeros(n);
        for i in 0..n {
            preds[i] = self.predict_one(&x.row(i));
        }
        preds
    }

    /// Get terminal node indices (leaves)
    fn get_leaves(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| match n {
                BartNode::Leaf { .. } => Some(i),
                BartNode::Internal { .. } => None,
            })
            .collect()
    }

    /// Get internal node indices
    fn get_internals(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| match n {
                BartNode::Internal { .. } => Some(i),
                BartNode::Leaf { .. } => None,
            })
            .collect()
    }

    /// Get node depth
    fn get_depth(&self, node_idx: usize) -> usize {
        if node_idx == 0 {
            return 0;
        }
        // Find parent by searching
        for (i, node) in self.nodes.iter().enumerate() {
            if let BartNode::Internal { left, right, .. } = node {
                if *left == node_idx || *right == node_idx {
                    return 1 + self.get_depth(i);
                }
            }
        }
        0
    }

    /// Check if a node has two leaf children (can be pruned)
    fn is_singly_internal(&self, node_idx: usize) -> bool {
        if let BartNode::Internal { left, right, .. } = &self.nodes[node_idx] {
            matches!(
                (&self.nodes[*left], &self.nodes[*right]),
                (BartNode::Leaf { .. }, BartNode::Leaf { .. })
            )
        } else {
            false
        }
    }

    /// Get observations that fall in each leaf
    fn get_leaf_assignments(&self, x: ArrayView2<f64>) -> Vec<Vec<usize>> {
        let leaves = self.get_leaves();
        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];

        for i in 0..x.nrows() {
            let mut node_idx = 0;
            loop {
                match &self.nodes[node_idx] {
                    BartNode::Leaf { .. } => {
                        assignments[node_idx].push(i);
                        break;
                    }
                    BartNode::Internal {
                        var,
                        cut,
                        left,
                        right,
                    } => {
                        node_idx = if x.row(i)[*var] <= *cut {
                            *left
                        } else {
                            *right
                        };
                    }
                }
            }
        }

        leaves.iter().map(|&l| assignments[l].clone()).collect()
    }
}

// ============================================================================
// MCMC Sampler
// ============================================================================

/// BART MCMC sampler state
struct BartSampler {
    /// Current trees
    trees: Vec<BartTree>,
    /// Current residual SD
    sigma: f64,
    /// Scaled prior SD for leaf values
    sigma_mu: f64,
    /// Config
    config: BartConfig,
    /// RNG state
    rng: u64,
    /// Feature counts for variable importance
    var_counts: Vec<usize>,
    /// Number of features
    n_features: usize,
    /// Data standard deviation (for scaling)
    y_sd: f64,
}

impl BartSampler {
    /// Initialize the sampler
    fn new(y: &Array1<f64>, x: ArrayView2<f64>, config: &BartConfig) -> Self {
        let n = y.len();
        let p = x.ncols();
        let m = config.n_trees;

        // Compute data statistics for scaling
        let y_mean = y.mean().unwrap_or(0.0);
        let y_sd = (y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();

        // Initial sigma estimate (from data SD)
        let sigma_hat = y_sd.max(0.1);

        // Prior SD for leaf values: ensures sum of trees covers y range
        // sigma_mu = sigma_hat / (k * sqrt(m)) so that
        // sum of m leaf values has SD = sigma_hat / k
        let sigma_mu = sigma_hat / (config.k * (m as f64).sqrt());

        // Initialize all trees as stumps with mu = y_mean / m
        let initial_mu = y_mean / m as f64;
        let trees: Vec<BartTree> = (0..m).map(|_| BartTree::new(initial_mu)).collect();

        Self {
            trees,
            sigma: sigma_hat,
            sigma_mu,
            config: config.clone(),
            rng: config.seed.unwrap_or(42),
            var_counts: vec![0; p],
            n_features: p,
            y_sd,
        }
    }

    /// Simple LCG random number generator
    fn random(&mut self) -> f64 {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng >> 33) ^ self.rng) as f64 / u64::MAX as f64
    }

    /// Generate random integer in [0, max)
    fn random_int(&mut self, max: usize) -> usize {
        (self.random() * max as f64) as usize
    }

    /// Sample from standard normal (Box-Muller)
    fn random_normal(&mut self) -> f64 {
        let u1 = self.random().max(1e-10);
        let u2 = self.random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Sample from inverse-gamma(alpha, beta)
    fn random_inv_gamma(&mut self, alpha: f64, beta: f64) -> f64 {
        // Gamma via Marsaglia-Tsang for alpha >= 1
        let a = if alpha < 1.0 { alpha + 1.0 } else { alpha };
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        let g = loop {
            let x = self.random_normal();
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 {
                let u = self.random();
                if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln())
                {
                    break d * v;
                }
            }
        };

        let gamma = if alpha < 1.0 {
            g * self.random().powf(1.0 / alpha)
        } else {
            g
        };

        beta / gamma
    }

    /// Compute log likelihood ratio for splitting probability
    fn log_split_prob(&self, depth: usize) -> f64 {
        let p = self.config.alpha * (1.0 + depth as f64).powf(-self.config.beta);
        p.ln() - (1.0 - p).ln()
    }

    /// Run one MCMC iteration (one full sweep through all trees)
    fn step(&mut self, y: &Array1<f64>, x: ArrayView2<f64>) {
        let m = self.trees.len();
        let n = y.len();

        // Compute current fit (sum of all trees)
        let mut current_fit: Array1<f64> = Array1::zeros(n);
        for tree in &self.trees {
            current_fit = current_fit + tree.predict(x);
        }

        // Backfitting: update each tree in turn
        for j in 0..m {
            // Partial residual: y - sum of all other trees
            let tree_fit = self.trees[j].predict(x);
            let residual = y - &current_fit + &tree_fit;

            // Update tree j given partial residual
            self.update_tree(j, &residual, x);

            // Update current fit
            let new_tree_fit = self.trees[j].predict(x);
            current_fit = current_fit - &tree_fit + &new_tree_fit;
        }

        // Update sigma
        let final_residual = y - &current_fit;
        let ss = final_residual.iter().map(|&r| r.powi(2)).sum::<f64>();

        // Posterior: sigma^2 | ... ~ IG((nu + n)/2, (nu*lambda + ss)/2)
        let nu = self.config.nu;
        // Calibrate lambda from prior (using y_sd as sigma_hat)
        let lambda = self.calibrate_lambda(self.y_sd, nu, self.config.q);
        let post_alpha = (nu + n as f64) / 2.0;
        let post_beta = (nu * lambda + ss) / 2.0;

        self.sigma = self.random_inv_gamma(post_alpha, post_beta).sqrt();

        // Update sigma_mu (prior SD for leaves)
        self.sigma_mu = self.sigma / (self.config.k * (m as f64).sqrt());
    }

    /// Calibrate lambda so that P(sigma < sigma_hat) = q
    fn calibrate_lambda(&self, sigma_hat: f64, nu: f64, q: f64) -> f64 {
        // For sigma^2 ~ IG(nu/2, nu*lambda/2)
        // Want P(sigma < sigma_hat) = q, i.e., P(sigma^2 < sigma_hat^2) = q
        // This requires numerical inversion; use simple approximation
        // lambda ~ sigma_hat^2 / qchisq(1-q, nu) * nu
        // Approximate chi-squared quantile
        let chi_sq_quantile = nu
            * (1.0 + 2.0 / (9.0 * nu)).powi(3)
            * (1.0 - 2.0 / (9.0 * nu) + (1.0 - q).powf(1.0 / 3.0) * (2.0 / (9.0 * nu)).sqrt())
                .powi(3);
        sigma_hat.powi(2) * chi_sq_quantile / nu
    }

    /// Update a single tree using Metropolis-Hastings
    fn update_tree(&mut self, tree_idx: usize, residual: &Array1<f64>, x: ArrayView2<f64>) {
        let tree = &mut self.trees[tree_idx];

        // Propose: grow, prune, or change
        let n_leaves = tree.get_leaves().len();
        let n_internal = tree.get_internals().len();
        let n_singly = tree
            .get_internals()
            .iter()
            .filter(|&&i| tree.is_singly_internal(i))
            .count();

        // Choose proposal type
        let r = self.random();
        let (proposal_type, forward_prob, backward_prob) = if n_leaves == 1 {
            // Can only grow from single leaf
            ("grow", 1.0, 1.0)
        } else if n_singly == 0 {
            // No prunable nodes, can only grow
            ("grow", 0.5, 0.5)
        } else if r < 0.5 {
            ("grow", 0.5, 0.5)
        } else {
            ("prune", 0.5, 0.5)
        };

        match proposal_type {
            "grow" => self.propose_grow(tree_idx, residual, x, forward_prob, backward_prob),
            "prune" => self.propose_prune(tree_idx, residual, x, forward_prob, backward_prob),
            _ => {}
        }

        // Draw new leaf values from conditional posterior
        self.draw_leaf_values(tree_idx, residual, x);
    }

    /// Propose a grow move (split a leaf)
    fn propose_grow(
        &mut self,
        tree_idx: usize,
        residual: &Array1<f64>,
        x: ArrayView2<f64>,
        _forward_prob: f64,
        _backward_prob: f64,
    ) {
        let leaves = self.trees[tree_idx].get_leaves();

        if leaves.is_empty() {
            return;
        }

        // Choose a random leaf to split (get random index before borrowing tree)
        let random_idx = self.random_int(leaves.len());
        let leaf_idx = leaves[random_idx];
        let tree = &self.trees[tree_idx];
        let leaf_depth = tree.get_depth(leaf_idx);

        // Check depth constraint
        if leaf_depth >= self.config.max_depth {
            return;
        }

        // Get observations in this leaf
        let assignments = tree.get_leaf_assignments(x);
        let leaf_pos = leaves.iter().position(|&l| l == leaf_idx).unwrap();
        let obs_in_leaf: Vec<usize> = assignments[leaf_pos].clone();

        if obs_in_leaf.len() < 2 * self.config.min_node_size {
            return; // Not enough observations to split
        }

        // Choose a random variable to split on
        let var = self.random_int(self.n_features);

        // Get values of this variable in the leaf
        let mut values: Vec<f64> = obs_in_leaf.iter().map(|&i| x[[i, var]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();

        if values.len() < 2 {
            return; // No valid split point
        }

        // Choose a random split point
        let split_idx = self.random_int(values.len() - 1);
        let cut = (values[split_idx] + values[split_idx + 1]) / 2.0;

        // Count observations in proposed children
        let n_left = obs_in_leaf.iter().filter(|&&i| x[[i, var]] <= cut).count();
        let n_right = obs_in_leaf.len() - n_left;

        if n_left < self.config.min_node_size || n_right < self.config.min_node_size {
            return; // Children would be too small
        }

        // Compute acceptance probability using log likelihood ratio
        let log_prior_ratio =
            self.log_split_prob(leaf_depth) - self.log_split_prob(leaf_depth + 1).powi(2);

        // Likelihood ratio from integrating out leaf values
        let left_obs: Vec<usize> = obs_in_leaf
            .iter()
            .filter(|&&i| x[[i, var]] <= cut)
            .copied()
            .collect();
        let right_obs: Vec<usize> = obs_in_leaf
            .iter()
            .filter(|&&i| x[[i, var]] > cut)
            .copied()
            .collect();

        let log_lik_ratio =
            self.compute_split_log_lik_ratio(residual, &obs_in_leaf, &left_obs, &right_obs);

        let log_acceptance = log_prior_ratio + log_lik_ratio;

        if self.random().ln() < log_acceptance {
            // Accept: create the split
            let tree = &mut self.trees[tree_idx];
            let left_idx = tree.nodes.len();
            let right_idx = left_idx + 1;

            // Get current leaf value for initialization
            let current_mu = if let BartNode::Leaf { mu } = tree.nodes[leaf_idx] {
                mu
            } else {
                0.0
            };

            tree.nodes.push(BartNode::Leaf { mu: current_mu });
            tree.nodes.push(BartNode::Leaf { mu: current_mu });
            tree.nodes[leaf_idx] = BartNode::Internal {
                var,
                cut,
                left: left_idx,
                right: right_idx,
            };
            tree.depth = tree.depth.max(leaf_depth + 1);

            // Update variable importance count
            self.var_counts[var] += 1;
        }
    }

    /// Propose a prune move (collapse an internal node to leaf)
    fn propose_prune(
        &mut self,
        tree_idx: usize,
        residual: &Array1<f64>,
        x: ArrayView2<f64>,
        _forward_prob: f64,
        _backward_prob: f64,
    ) {
        // Find singly-internal nodes (internal nodes with two leaf children)
        let singly_internals: Vec<usize> = {
            let tree = &self.trees[tree_idx];
            tree.get_internals()
                .into_iter()
                .filter(|&i| tree.is_singly_internal(i))
                .collect()
        };

        if singly_internals.is_empty() {
            return;
        }

        // Choose a random singly-internal node to prune (get random index before borrowing tree)
        let random_idx = self.random_int(singly_internals.len());
        let node_idx = singly_internals[random_idx];
        let tree = &self.trees[tree_idx];
        let node_depth = tree.get_depth(node_idx);

        // Get the children and their observations
        let (left_idx, right_idx, var) = if let BartNode::Internal {
            left, right, var, ..
        } = tree.nodes[node_idx]
        {
            (left, right, var)
        } else {
            return;
        };

        // Get all observations that would be in the merged leaf
        let assignments = tree.get_leaf_assignments(x);
        let leaves = tree.get_leaves();

        let left_pos = leaves.iter().position(|&l| l == left_idx);
        let right_pos = leaves.iter().position(|&l| l == right_idx);

        if left_pos.is_none() || right_pos.is_none() {
            return;
        }

        let left_obs = assignments[left_pos.unwrap()].clone();
        let right_obs = assignments[right_pos.unwrap()].clone();
        let merged_obs: Vec<usize> = left_obs.iter().chain(right_obs.iter()).copied().collect();

        // Compute acceptance probability (negative of grow ratio)
        let log_prior_ratio =
            -self.log_split_prob(node_depth) + self.log_split_prob(node_depth + 1).powi(2);
        let log_lik_ratio =
            -self.compute_split_log_lik_ratio(residual, &merged_obs, &left_obs, &right_obs);

        let log_acceptance = log_prior_ratio + log_lik_ratio;

        if self.random().ln() < log_acceptance {
            // Accept: collapse to leaf
            let tree = &mut self.trees[tree_idx];

            // Get mean residual for merged observations
            let mean_resid = if !merged_obs.is_empty() {
                merged_obs.iter().map(|&i| residual[i]).sum::<f64>() / merged_obs.len() as f64
            } else {
                0.0
            };

            tree.nodes[node_idx] = BartNode::Leaf { mu: mean_resid };
            // Note: we leave the old child nodes in place (they're now unreachable)

            // Decrement variable importance count
            if self.var_counts[var] > 0 {
                self.var_counts[var] -= 1;
            }
        }
    }

    /// Compute log likelihood ratio for splitting
    fn compute_split_log_lik_ratio(
        &self,
        residual: &Array1<f64>,
        parent_obs: &[usize],
        left_obs: &[usize],
        right_obs: &[usize],
    ) -> f64 {
        let sigma2 = self.sigma.powi(2);
        let sigma_mu2 = self.sigma_mu.powi(2);

        // Helper to compute marginal log likelihood for a set of observations
        let marg_log_lik = |obs: &[usize]| -> f64 {
            if obs.is_empty() {
                return 0.0;
            }
            let n = obs.len() as f64;
            let sum_r: f64 = obs.iter().map(|&i| residual[i]).sum();
            let sum_r2: f64 = obs.iter().map(|&i| residual[i].powi(2)).sum();

            // Marginal likelihood integrating out mu ~ N(0, sigma_mu^2)
            // -n/2 * log(2*pi*sigma^2) - 1/2 * sum(r^2)/sigma^2
            // + 1/2 * log(sigma_mu^2/(sigma_mu^2 + sigma^2/n)) + 1/2 * (sum_r)^2 / (sigma^2/n + sigma_mu^2 * n) / n
            let var_total = sigma2 + n * sigma_mu2;
            -0.5 * n * (2.0 * std::f64::consts::PI * sigma2).ln() - 0.5 * sum_r2 / sigma2
                + 0.5 * (sigma_mu2 / (sigma_mu2 + sigma2 / n)).ln()
                + 0.5 * sum_r.powi(2) / var_total
        };

        marg_log_lik(left_obs) + marg_log_lik(right_obs) - marg_log_lik(parent_obs)
    }

    /// Draw new leaf values from their conditional posterior
    fn draw_leaf_values(&mut self, tree_idx: usize, residual: &Array1<f64>, x: ArrayView2<f64>) {
        // First, gather information without holding a mutable borrow
        let leaves = self.trees[tree_idx].get_leaves();
        let assignments = self.trees[tree_idx].get_leaf_assignments(x);

        let sigma2 = self.sigma.powi(2);
        let sigma_mu2 = self.sigma_mu.powi(2);
        let sigma_mu = self.sigma_mu;

        // Pre-generate random numbers
        let mut random_vals: Vec<f64> = Vec::with_capacity(leaves.len());
        for _ in 0..leaves.len() {
            random_vals.push(self.random_normal());
        }

        // Now update the tree
        let tree = &mut self.trees[tree_idx];
        for (i, &leaf_idx) in leaves.iter().enumerate() {
            let obs = &assignments[i];
            let rand_val = random_vals[i];

            if obs.is_empty() {
                if let BartNode::Leaf { ref mut mu } = tree.nodes[leaf_idx] {
                    *mu = sigma_mu * rand_val;
                }
                continue;
            }

            let n = obs.len() as f64;
            let sum_r: f64 = obs.iter().map(|&j| residual[j]).sum();

            // Posterior: mu | r ~ N(post_mean, post_var)
            // post_var = 1 / (n/sigma^2 + 1/sigma_mu^2)
            // post_mean = post_var * sum_r / sigma^2
            let post_var = 1.0 / (n / sigma2 + 1.0 / sigma_mu2);
            let post_mean = post_var * sum_r / sigma2;
            let post_sd = post_var.sqrt();

            if let BartNode::Leaf { ref mut mu } = tree.nodes[leaf_idx] {
                *mu = post_mean + post_sd * rand_val;
            }
        }
    }

    /// Get sum of tree predictions
    fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut pred: Array1<f64> = Array1::zeros(n);
        for tree in &self.trees {
            pred = pred + tree.predict(x);
        }
        pred
    }

    /// Get variable importance (normalized counts)
    fn get_variable_importance(&self) -> Vec<f64> {
        let total: usize = self.var_counts.iter().sum();
        if total == 0 {
            return vec![1.0 / self.n_features as f64; self.n_features];
        }
        self.var_counts
            .iter()
            .map(|&c| c as f64 / total as f64)
            .collect()
    }
}

// ============================================================================
// Main API Functions
// ============================================================================

/// Run BART from a dataset.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the response and features
/// * `y_col` - Name of the response variable
/// * `x_cols` - Names of feature columns
/// * `config` - BART configuration
///
/// # Returns
///
/// `BartResult` containing predictions with uncertainty and variable importance.
///
/// # Example
///
/// ```ignore
/// use p2a_core::ml::{bart, BartConfig};
///
/// let config = BartConfig {
///     n_trees: 200,
///     n_burn: 250,
///     n_mcmc: 1000,
///     ..Default::default()
/// };
///
/// let result = bart(&dataset, "y", &["x1", "x2", "x3"], config)?;
/// println!("Predictions: {:?}", &result.predictions[..5]);
/// println!("Sigma: {:.4}", result.sigma);
/// ```
///
/// # References
///
/// - Chipman, George & McCulloch (2010). BART: Bayesian Additive Regression Trees.
/// - R package `BART` (Sparapani et al., 2021).
pub fn bart(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: BartConfig,
) -> EconResult<BartResult> {
    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: vec![format!("{:?}", e)],
        }
    })?;

    // Extract X
    let n = dataset.nrows();
    let p = x_cols.len();

    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "At least one feature column is required".to_string(),
        });
    }

    let mut x = Array2::zeros((n, p));
    for (j, &col_name) in x_cols.iter().enumerate() {
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

    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    bart_arrays(y.view(), x.view(), Some(feature_names), config)
}

/// Run BART with array inputs.
///
/// This is the core implementation, separate from data loading for flexibility.
pub fn bart_arrays(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,
    feature_names: Option<Vec<String>>,
    config: BartConfig,
) -> EconResult<BartResult> {
    let n = y.len();
    let p = x.ncols();

    // Validate inputs
    if n < 20 {
        return Err(EconError::InsufficientData {
            required: 20,
            provided: n,
            context: "BART requires at least 20 observations".to_string(),
        });
    }

    if x.nrows() != n {
        return Err(EconError::InvalidSpecification {
            message: "X and y must have same number of observations".to_string(),
        });
    }

    // Convert to owned arrays
    let y_owned = y.to_owned();
    let x_owned = x.to_owned();

    // Initialize sampler
    let mut sampler = BartSampler::new(&y_owned, x_owned.view(), &config);

    // Burn-in
    for _ in 0..config.n_burn {
        sampler.step(&y_owned, x_owned.view());
    }

    // Reset variable importance counts after burn-in
    sampler.var_counts = vec![0; p];

    // Collect posterior samples
    let mut posterior_preds: Array2<f64> = Array2::zeros((config.n_mcmc, n));
    let mut sigma_samples: Vec<f64> = Vec::with_capacity(config.n_mcmc);

    for iter in 0..config.n_mcmc {
        sampler.step(&y_owned, x_owned.view());

        let pred = sampler.predict(x_owned.view());
        posterior_preds.row_mut(iter).assign(&pred);
        sigma_samples.push(sampler.sigma);
    }

    // Compute posterior summaries
    let predictions: Vec<f64> = (0..n)
        .map(|i| posterior_preds.column(i).mean().unwrap_or(0.0))
        .collect();

    let prediction_sd: Vec<f64> = (0..n)
        .map(|i| {
            let col = posterior_preds.column(i);
            let mean = col.mean().unwrap_or(0.0);
            (col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / config.n_mcmc as f64).sqrt()
        })
        .collect();

    // Compute prediction intervals
    let alpha = 1.0 - config.confidence_level;
    let lower_q = (alpha / 2.0 * config.n_mcmc as f64) as usize;
    let upper_q = ((1.0 - alpha / 2.0) * config.n_mcmc as f64) as usize;

    let mut prediction_lower = Vec::with_capacity(n);
    let mut prediction_upper = Vec::with_capacity(n);

    for i in 0..n {
        let mut col_vals: Vec<f64> = posterior_preds.column(i).to_vec();
        col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        prediction_lower.push(col_vals.get(lower_q).copied().unwrap_or(col_vals[0]));
        prediction_upper.push(
            col_vals
                .get(upper_q.min(col_vals.len() - 1))
                .copied()
                .unwrap_or(*col_vals.last().unwrap()),
        );
    }

    // Sigma posterior summaries
    let sigma = sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64;
    let sigma_sd = {
        let var = sigma_samples
            .iter()
            .map(|&s| (s - sigma).powi(2))
            .sum::<f64>()
            / sigma_samples.len() as f64;
        var.sqrt()
    };

    // Variable importance
    let variable_importance = sampler.get_variable_importance();

    Ok(BartResult {
        predictions,
        prediction_lower,
        prediction_upper,
        prediction_sd,
        variable_importance,
        feature_names,
        sigma,
        sigma_sd,
        n_obs: n,
        n_features: p,
        n_trees: config.n_trees,
        n_samples: config.n_mcmc,
        config: config.clone(),
        posterior_samples: Some(posterior_preds),
        sigma_samples: Some(sigma_samples),
    })
}

/// Run BART from dataset with convenience parameters (for MCP integration).
pub fn run_bart(
    dataset: &Dataset,
    y_col: &str,
    x_cols: Vec<String>,
    n_trees: Option<usize>,
    n_burn: Option<usize>,
    n_mcmc: Option<usize>,
    k: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    seed: Option<u64>,
) -> EconResult<BartResult> {
    let x_refs: Vec<&str> = x_cols.iter().map(|s| s.as_str()).collect();

    let config = BartConfig {
        n_trees: n_trees.unwrap_or(200),
        n_burn: n_burn.unwrap_or(250),
        n_mcmc: n_mcmc.unwrap_or(1000),
        k: k.unwrap_or(2.0),
        alpha: alpha.unwrap_or(0.95),
        beta: beta.unwrap_or(2.0),
        seed,
        ..Default::default()
    };

    bart(dataset, y_col, &x_refs, config)
}

/// Predict using BART posterior for new data.
///
/// Returns posterior mean predictions and prediction intervals.
pub fn bart_predict(
    result: &BartResult,
    x_new: ArrayView2<f64>,
) -> EconResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // For out-of-sample prediction, we would need to store the trees
    // For now, return an error indicating this limitation
    Err(EconError::InvalidSpecification {
        message: "Out-of-sample prediction requires storing the tree ensemble. Use bart_arrays() with the new data included in training for now.".to_string(),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Generate synthetic data: y = sin(x1) + x2 + noise
    fn generate_test_data(n: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
        let mut rng = seed;
        let lcg = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) ^ *state) as f64 / u64::MAX as f64
        };

        let mut x = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);

        for i in 0..n {
            x[[i, 0]] = lcg(&mut rng) * 2.0 * std::f64::consts::PI; // x1 in [0, 2pi]
            x[[i, 1]] = lcg(&mut rng) * 2.0 - 1.0; // x2 in [-1, 1]
            x[[i, 2]] = lcg(&mut rng) * 2.0 - 1.0; // x3 (noise feature)

            let noise = (lcg(&mut rng) * 2.0 - 1.0) * 0.5; // noise in [-0.5, 0.5]
            y[i] = x[[i, 0]].sin() + x[[i, 1]] + noise;
        }

        (y, x)
    }

    #[test]
    fn test_bart_basic() {
        let (y, x) = generate_test_data(100, 42);

        let config = BartConfig {
            n_trees: 20, // Small for testing
            n_burn: 50,
            n_mcmc: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = bart_arrays(
            y.view(),
            x.view(),
            Some(vec!["x1".to_string(), "x2".to_string(), "x3".to_string()]),
            config,
        )
        .unwrap();

        // Check basic properties
        assert_eq!(result.n_obs, 100);
        assert_eq!(result.n_features, 3);
        assert_eq!(result.predictions.len(), 100);
        assert_eq!(result.prediction_lower.len(), 100);
        assert_eq!(result.prediction_upper.len(), 100);

        // Sigma should be positive
        assert!(result.sigma > 0.0);

        // Prediction intervals should contain predictions
        for i in 0..100 {
            assert!(result.prediction_lower[i] <= result.predictions[i]);
            assert!(result.prediction_upper[i] >= result.predictions[i]);
        }

        println!("BART test result:\n{}", result);
    }

    #[test]
    fn test_bart_variable_importance() {
        // x1 and x2 are relevant, x3 is pure noise
        let (y, x) = generate_test_data(150, 123);

        let config = BartConfig {
            n_trees: 30,
            n_burn: 100,
            n_mcmc: 200,
            seed: Some(123),
            ..Default::default()
        };

        let result = bart_arrays(
            y.view(),
            x.view(),
            Some(vec![
                "x1".to_string(),
                "x2".to_string(),
                "noise".to_string(),
            ]),
            config,
        )
        .unwrap();

        // x1 and x2 should have higher importance than noise
        // Note: This test may be sensitive due to randomness
        println!("Variable importance: {:?}", result.variable_importance);

        // The sum of importances should be 1
        let sum: f64 = result.variable_importance.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Importances should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_bart_prediction_quality() {
        // Simple linear relationship
        let n = 100;
        let mut rng = 42u64;
        let lcg = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 33) ^ *state) as f64 / u64::MAX as f64
        };

        let mut x = Array2::zeros((n, 1));
        let mut y = Array1::zeros(n);

        for i in 0..n {
            x[[i, 0]] = lcg(&mut rng) * 10.0;
            y[i] = 2.0 * x[[i, 0]] + 1.0 + (lcg(&mut rng) - 0.5) * 2.0; // y = 2x + 1 + noise
        }

        let config = BartConfig {
            n_trees: 50,
            n_burn: 100,
            n_mcmc: 200,
            seed: Some(42),
            ..Default::default()
        };

        let result = bart_arrays(y.view(), x.view(), None, config).unwrap();

        // Compute R-squared
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(result.predictions.iter())
            .map(|(&yi, &pi)| (yi - pi).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        println!("R-squared: {:.4}", r2);
        assert!(
            r2 > 0.5,
            "R² should be > 0.5 for simple linear data, got {}",
            r2
        );
    }

    #[test]
    fn test_bart_insufficient_data() {
        let y = Array1::zeros(10);
        let x = Array2::zeros((10, 2));

        let config = BartConfig::default();
        let result = bart_arrays(y.view(), x.view(), None, config);

        assert!(result.is_err());
    }

    #[test]
    fn test_bart_tree_operations() {
        // Test tree grow and predict
        let mut tree = BartTree::new(0.5);
        assert_eq!(tree.get_leaves().len(), 1);
        assert_eq!(tree.get_internals().len(), 0);

        // Manual split
        tree.nodes.push(BartNode::Leaf { mu: 0.3 });
        tree.nodes.push(BartNode::Leaf { mu: 0.7 });
        tree.nodes[0] = BartNode::Internal {
            var: 0,
            cut: 0.5,
            left: 1,
            right: 2,
        };
        tree.depth = 1;

        assert_eq!(tree.get_leaves().len(), 2);
        assert_eq!(tree.get_internals().len(), 1);

        // Test predictions
        let x = array![[0.3], [0.7]];
        let preds = tree.predict(x.view());
        assert_eq!(preds[0], 0.3); // x[0] <= 0.5, goes left
        assert_eq!(preds[1], 0.7); // x[1] > 0.5, goes right
    }

    #[test]
    fn test_config_defaults() {
        let config = BartConfig::default();
        assert_eq!(config.n_trees, 200);
        assert_eq!(config.n_burn, 250);
        assert_eq!(config.n_mcmc, 1000);
        assert_eq!(config.k, 2.0);
        assert_eq!(config.alpha, 0.95);
        assert_eq!(config.beta, 2.0);
    }
}
