//! Bayesian Additive Regression Trees (BART)
//!
//! BART is a Bayesian "sum-of-trees" model where each tree is constrained
//! to be a weak learner, and the ensemble is fit using MCMC (Markov Chain
//! Monte Carlo) sampling.
//!
//! # Algorithm
//!
//! BART models the response as:
//! y = f(x) + ε, where f(x) = Σ g(x; T_j, M_j) and ε ~ N(0, σ²)
//!
//! Each tree g(x; T, M) is a regression tree with structure T and terminal
//! node parameters M. The prior on trees encourages small trees with modest
//! effects.
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::ml::bart::{bart, bart_predict, BartConfig};
//!
//! let config = BartConfig {
//!     num_trees: 200,
//!     num_burn: 100,
//!     num_mcmc: 1000,
//!     ..Default::default()
//! };
//!
//! let result = bart(&x, &y, &config)?;
//! let predictions = bart_predict(&result, &x_new);
//! ```

use ndarray::{Array1, Array2};
use rand::Rng;
use rand::prelude::*;
use rand_distr::{Distribution, Gamma, Normal};

use crate::errors::{EconError, EconResult};

/// BART configuration
#[derive(Debug, Clone)]
pub struct BartConfig {
    /// Number of trees in the ensemble
    pub num_trees: usize,
    /// Number of burn-in MCMC iterations
    pub num_burn: usize,
    /// Number of MCMC iterations to keep
    pub num_mcmc: usize,
    /// Base for tree depth prior: P(node at depth d is terminal) = α(1+d)^(-β)
    pub alpha: f64,
    /// Power for tree depth prior
    pub beta: f64,
    /// Prior mean for terminal node parameters
    pub k: f64,
    /// Prior degrees of freedom for σ²
    pub nu: f64,
    /// Prior quantile for σ² (used to set prior scale)
    pub q: f64,
    /// Minimum observations in a leaf node
    pub min_node_size: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for BartConfig {
    fn default() -> Self {
        Self {
            num_trees: 200,
            num_burn: 100,
            num_mcmc: 1000,
            alpha: 0.95,
            beta: 2.0,
            k: 2.0,
            nu: 3.0,
            q: 0.9,
            min_node_size: 5,
            seed: None,
        }
    }
}

/// A node in a BART tree
#[derive(Debug, Clone)]
pub struct BartNode {
    /// Feature index for split (None if leaf)
    pub split_feature: Option<usize>,
    /// Split value (None if leaf)
    pub split_value: Option<f64>,
    /// Left child index (None if leaf)
    pub left_child: Option<usize>,
    /// Right child index (None if leaf)
    pub right_child: Option<usize>,
    /// Terminal node value (prediction)
    pub mu: f64,
    /// Depth of this node
    pub depth: usize,
    /// Parent node index (None for root)
    pub parent: Option<usize>,
    /// Is this a left child?
    pub is_left: bool,
}

impl BartNode {
    fn new_leaf(mu: f64, depth: usize, parent: Option<usize>, is_left: bool) -> Self {
        Self {
            split_feature: None,
            split_value: None,
            left_child: None,
            right_child: None,
            mu,
            depth,
            parent,
            is_left,
        }
    }

    fn is_leaf(&self) -> bool {
        self.split_feature.is_none()
    }
}

/// A single BART tree
#[derive(Debug, Clone)]
pub struct BartTree {
    /// Nodes in the tree
    pub nodes: Vec<BartNode>,
}

impl BartTree {
    fn new(mu: f64) -> Self {
        Self {
            nodes: vec![BartNode::new_leaf(mu, 0, None, false)],
        }
    }

    /// Get prediction for a single observation
    fn predict_one(&self, x: &[f64]) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.mu;
            }
            let split_feat = node.split_feature.unwrap();
            let split_val = node.split_value.unwrap();
            if x[split_feat] <= split_val {
                node_idx = node.left_child.unwrap();
            } else {
                node_idx = node.right_child.unwrap();
            }
        }
    }

    /// Get leaf indices for all observations
    fn get_leaf_indices(&self, x: &Array2<f64>) -> Vec<usize> {
        x.rows()
            .into_iter()
            .map(|row| {
                let mut node_idx = 0;
                loop {
                    let node = &self.nodes[node_idx];
                    if node.is_leaf() {
                        return node_idx;
                    }
                    let split_feat = node.split_feature.unwrap();
                    let split_val = node.split_value.unwrap();
                    if row[split_feat] <= split_val {
                        node_idx = node.left_child.unwrap();
                    } else {
                        node_idx = node.right_child.unwrap();
                    }
                }
            })
            .collect()
    }

    /// Get indices of leaf nodes
    fn get_leaves(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_leaf())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get indices of internal (non-leaf) nodes
    fn get_internal_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.is_leaf())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get indices of singly internal nodes (internal nodes where both children are leaves)
    fn get_singly_internal(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                if n.is_leaf() {
                    return false;
                }
                let left = n.left_child.unwrap();
                let right = n.right_child.unwrap();
                self.nodes[left].is_leaf() && self.nodes[right].is_leaf()
            })
            .map(|(i, _)| i)
            .collect()
    }
}

/// BART model result
#[derive(Debug, Clone)]
pub struct BartResult {
    /// Posterior samples of trees (outer: MCMC iteration, inner: tree index)
    pub tree_samples: Vec<Vec<BartTree>>,
    /// Posterior samples of sigma
    pub sigma_samples: Vec<f64>,
    /// Posterior mean predictions for training data
    pub fitted_values: Array1<f64>,
    /// Posterior prediction intervals (lower, upper)
    pub prediction_intervals: Option<(Array1<f64>, Array1<f64>)>,
    /// Variable inclusion counts (how often each variable was used)
    pub variable_counts: Vec<usize>,
    /// Number of features
    pub n_features: usize,
    /// Configuration used
    pub config: BartConfig,
    /// Minimum y value (for unscaling predictions)
    pub y_min: f64,
    /// Range of y values (for unscaling predictions)
    pub y_range: f64,
}

/// Fit a BART model
pub fn bart(x: &Array2<f64>, y: &Array1<f64>, config: &BartConfig) -> EconResult<BartResult> {
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(EconError::InsufficientData {
            required: n,
            provided: y.len(),
            context: "X and y dimensions must match".to_string(),
        });
    }

    if n < config.min_node_size * 2 {
        return Err(EconError::InsufficientData {
            required: config.min_node_size * 2,
            provided: n,
            context: "BART requires sufficient observations for min_node_size".to_string(),
        });
    }

    // Initialize RNG
    let mut rng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Compute y range for scaling
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = y_max - y_min;

    if y_range < 1e-10 {
        return Err(EconError::SingularMatrix {
            context: "Response has no variation".to_string(),
            suggestion: "Ensure response variable has non-zero variance".to_string(),
        });
    }

    // Scale y to [-0.5, 0.5]
    let y_scaled: Array1<f64> = y.mapv(|yi| (yi - y_min) / y_range - 0.5);

    // Prior on terminal node means: μ ~ N(0, σ_μ²)
    // We set σ_μ such that P(sum of tree means ∈ [-0.5, 0.5]) ≈ 0.95
    let sigma_mu = 0.5 / (config.k * (config.num_trees as f64).sqrt());

    // Initialize sigma² prior
    // Use sample variance as starting point
    let y_mean = y_scaled.mean().unwrap_or(0.0);
    let y_var = y_scaled
        .mapv(|yi| (yi - y_mean).powi(2))
        .mean()
        .unwrap_or(1.0);
    let mut sigma = y_var.sqrt();

    // Prior scale for sigma² (inverse chi-squared)
    let _lambda = compute_sigma_prior_scale(&y_scaled, config.nu, config.q);

    // Precompute split points for each variable
    let split_points = compute_split_points(x, config.min_node_size);

    // Initialize trees with small constant predictions
    let init_mu = 0.0;
    let mut trees: Vec<BartTree> = (0..config.num_trees)
        .map(|_| BartTree::new(init_mu))
        .collect();

    // Current predictions for each tree
    let mut tree_preds: Vec<Array1<f64>> = vec![Array1::zeros(n); config.num_trees];

    // Total predictions
    let mut total_pred: Array1<f64> = Array1::zeros(n);

    // Storage for posterior samples
    let mut tree_samples = Vec::with_capacity(config.num_mcmc);
    let mut sigma_samples = Vec::with_capacity(config.num_mcmc);
    let mut variable_counts = vec![0usize; p];

    // MCMC iterations
    let total_iter = config.num_burn + config.num_mcmc;

    for iter in 0..total_iter {
        // Update each tree
        for j in 0..config.num_trees {
            // Compute partial residuals for this tree
            let partial_resid: Array1<f64> = &y_scaled - &total_pred + &tree_preds[j];

            // Sample new tree structure using Metropolis-Hastings
            let (new_tree, accepted) = sample_tree(
                &trees[j],
                x,
                &partial_resid,
                sigma,
                sigma_mu,
                &split_points,
                config,
                &mut rng,
            );

            if accepted {
                trees[j] = new_tree;
            }

            // Sample terminal node parameters
            sample_mu(&mut trees[j], x, &partial_resid, sigma, sigma_mu, &mut rng);

            // Update predictions for this tree
            let new_preds: Array1<f64> = x
                .rows()
                .into_iter()
                .map(|row| trees[j].predict_one(row.as_slice().unwrap()))
                .collect();

            // Update total predictions
            total_pred = &total_pred - &tree_preds[j] + &new_preds;
            tree_preds[j] = new_preds;
        }

        // Sample sigma
        let residuals = &y_scaled - &total_pred;
        let sse: f64 = residuals.mapv(|r| r * r).sum();
        sigma = sample_sigma(sse, n, config.nu, y_var, &mut rng);

        // Store samples after burn-in
        if iter >= config.num_burn {
            tree_samples.push(trees.clone());
            sigma_samples.push(sigma * y_range);

            // Count variable usage
            for tree in &trees {
                for node in &tree.nodes {
                    if let Some(feat) = node.split_feature {
                        variable_counts[feat] += 1;
                    }
                }
            }
        }
    }

    // Compute posterior mean fitted values
    let mut fitted_sum: Array1<f64> = Array1::zeros(n);
    let mut fitted_sq_sum: Array1<f64> = Array1::zeros(n);

    for sample_trees in &tree_samples {
        let sample_pred: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|row| {
                sample_trees
                    .iter()
                    .map(|t| t.predict_one(row.as_slice().unwrap()))
                    .sum::<f64>()
            })
            .collect();

        fitted_sum = &fitted_sum + &sample_pred;
        fitted_sq_sum = &fitted_sq_sum + &sample_pred.mapv(|v| v * v);
    }

    let n_samples = tree_samples.len() as f64;
    let fitted_mean = fitted_sum.mapv(|v| v / n_samples);

    // Unscale back to original y scale
    let fitted_values = fitted_mean.mapv(|v| (v + 0.5) * y_range + y_min);

    // Compute prediction intervals
    let fitted_var: Array1<f64> =
        &fitted_sq_sum.mapv(|v| v / n_samples) - &fitted_mean.mapv(|v| v * v);
    let fitted_sd: Array1<f64> = fitted_var.mapv(|v: f64| v.max(0.0).sqrt());

    // 95% credible intervals (unscaled)
    let lower =
        (&fitted_mean - &fitted_sd.mapv(|s| 1.96 * s)).mapv(|v| (v + 0.5) * y_range + y_min);
    let upper =
        (&fitted_mean + &fitted_sd.mapv(|s| 1.96 * s)).mapv(|v| (v + 0.5) * y_range + y_min);

    Ok(BartResult {
        tree_samples,
        sigma_samples,
        fitted_values,
        prediction_intervals: Some((lower, upper)),
        variable_counts,
        n_features: p,
        config: config.clone(),
        y_min,
        y_range,
    })
}

/// Make predictions using a fitted BART model
pub fn bart_predict(result: &BartResult, x: &Array2<f64>) -> Array1<f64> {
    let n = x.nrows();
    let n_samples = result.tree_samples.len();

    if n_samples == 0 {
        return Array1::zeros(n);
    }

    // Average predictions over posterior samples
    let mut pred_sum: Array1<f64> = Array1::zeros(n);

    for sample_trees in &result.tree_samples {
        let sample_pred: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|row| {
                sample_trees
                    .iter()
                    .map(|t| t.predict_one(row.as_slice().unwrap()))
                    .sum::<f64>()
            })
            .collect();
        pred_sum = &pred_sum + &sample_pred;
    }

    // Average and unscale predictions back to original y scale
    // Tree predictions are on [-0.5, 0.5] scale
    let y_min = result.y_min;
    let y_range = result.y_range;
    pred_sum.mapv(|v: f64| (v / n_samples as f64 + 0.5) * y_range + y_min)
}

/// Make predictions with credible intervals
pub fn bart_predict_intervals(
    result: &BartResult,
    x: &Array2<f64>,
    alpha: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = x.nrows();
    let n_samples = result.tree_samples.len();

    if n_samples == 0 {
        return (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
    }

    // Collect all predictions
    let mut all_preds: Vec<Array1<f64>> = Vec::with_capacity(n_samples);

    for sample_trees in &result.tree_samples {
        let sample_pred: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|row| {
                sample_trees
                    .iter()
                    .map(|t| t.predict_one(row.as_slice().unwrap()))
                    .sum::<f64>()
            })
            .collect();
        all_preds.push(sample_pred);
    }

    // Compute quantiles for each observation
    let mut mean: Array1<f64> = Array1::zeros(n);
    let mut lower: Array1<f64> = Array1::zeros(n);
    let mut upper: Array1<f64> = Array1::zeros(n);

    let lower_q = alpha / 2.0;
    let upper_q = 1.0 - alpha / 2.0;

    for i in 0..n {
        let mut obs_preds: Vec<f64> = all_preds.iter().map(|p| p[i]).collect();
        obs_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute mean and quantiles on scaled values
        let scaled_mean = obs_preds.iter().sum::<f64>() / n_samples as f64;

        let lower_idx = ((n_samples as f64 * lower_q).floor() as usize).min(n_samples - 1);
        let upper_idx = ((n_samples as f64 * upper_q).ceil() as usize).min(n_samples - 1);

        let scaled_lower = obs_preds[lower_idx];
        let scaled_upper = obs_preds[upper_idx];

        // Unscale to original y scale
        mean[i] = (scaled_mean + 0.5) * result.y_range + result.y_min;
        lower[i] = (scaled_lower + 0.5) * result.y_range + result.y_min;
        upper[i] = (scaled_upper + 0.5) * result.y_range + result.y_min;
    }

    (mean, lower, upper)
}

/// Compute split points for each variable
fn compute_split_points(x: &Array2<f64>, min_node_size: usize) -> Vec<Vec<f64>> {
    let p = x.ncols();
    let mut split_points = Vec::with_capacity(p);

    for j in 0..p {
        let col = x.column(j);
        let mut vals: Vec<f64> = col.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();

        // Keep splits that would leave at least min_node_size in each child
        let points: Vec<f64> = if vals.len() > 2 * min_node_size {
            vals[min_node_size..vals.len() - min_node_size].to_vec()
        } else if vals.len() > 1 {
            // Use midpoints
            vals.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
        } else {
            vec![]
        };

        split_points.push(points);
    }

    split_points
}

/// Compute prior scale for sigma² using quantile matching
fn compute_sigma_prior_scale(y: &Array1<f64>, nu: f64, q: f64) -> f64 {
    let y_mean = y.mean().unwrap_or(0.0);
    let y_var = y.mapv(|yi| (yi - y_mean).powi(2)).mean().unwrap_or(1.0);

    // Find lambda such that P(σ² < y_var) = q under inverse chi-squared prior
    // This is approximate
    let qchisq = statrs::distribution::ChiSquared::new(nu).unwrap();
    use statrs::distribution::ContinuousCDF;
    let chi_q = qchisq.inverse_cdf(1.0 - q);

    nu * y_var / chi_q
}

/// Sample a new tree using Metropolis-Hastings
fn sample_tree(
    tree: &BartTree,
    x: &Array2<f64>,
    r: &Array1<f64>,
    sigma: f64,
    sigma_mu: f64,
    split_points: &[Vec<f64>],
    config: &BartConfig,
    rng: &mut StdRng,
) -> (BartTree, bool) {
    // Choose move type: grow, prune, change, swap
    let u: f64 = rng.r#gen();

    let leaves = tree.get_leaves();
    let singly_internal = tree.get_singly_internal();

    let can_grow = !leaves.is_empty();
    let can_prune = !singly_internal.is_empty();

    let (grow_prob, prune_prob) = if can_grow && can_prune {
        (0.25, 0.25)
    } else if can_grow {
        (0.5, 0.0)
    } else if can_prune {
        (0.0, 0.5)
    } else {
        return (tree.clone(), false);
    };

    if u < grow_prob {
        // Grow: split a leaf
        grow_tree(tree, x, r, sigma, sigma_mu, split_points, config, rng)
    } else if u < grow_prob + prune_prob {
        // Prune: collapse a singly internal node
        prune_tree(tree, x, r, sigma, sigma_mu, config, rng)
    } else {
        // Change: modify split rule at internal node
        change_tree(tree, x, r, sigma, sigma_mu, split_points, config, rng)
    }
}

/// Grow a tree by splitting a leaf
fn grow_tree(
    tree: &BartTree,
    x: &Array2<f64>,
    r: &Array1<f64>,
    sigma: f64,
    sigma_mu: f64,
    split_points: &[Vec<f64>],
    config: &BartConfig,
    rng: &mut StdRng,
) -> (BartTree, bool) {
    let leaves = tree.get_leaves();
    if leaves.is_empty() {
        return (tree.clone(), false);
    }

    // Choose a random leaf
    let leaf_idx = leaves[rng.gen_range(0..leaves.len())];
    let leaf = &tree.nodes[leaf_idx];

    // Get observations in this leaf
    let leaf_indices = tree.get_leaf_indices(x);
    let obs_in_leaf: Vec<usize> = leaf_indices
        .iter()
        .enumerate()
        .filter(|&(_, l)| *l == leaf_idx)
        .map(|(i, _)| i)
        .collect();

    if obs_in_leaf.len() < 2 * config.min_node_size {
        return (tree.clone(), false);
    }

    // Choose a random variable and split point
    let p = x.ncols();
    let available_vars: Vec<usize> = (0..p).filter(|&j| !split_points[j].is_empty()).collect();

    if available_vars.is_empty() {
        return (tree.clone(), false);
    }

    let var_idx = available_vars[rng.gen_range(0..available_vars.len())];
    let var_splits = &split_points[var_idx];

    if var_splits.is_empty() {
        return (tree.clone(), false);
    }

    let split_val = var_splits[rng.gen_range(0..var_splits.len())];

    // Check that split creates valid children
    let left_obs: Vec<usize> = obs_in_leaf
        .iter()
        .filter(|&&i| x[[i, var_idx]] <= split_val)
        .cloned()
        .collect();
    let right_obs: Vec<usize> = obs_in_leaf
        .iter()
        .filter(|&&i| x[[i, var_idx]] > split_val)
        .cloned()
        .collect();

    if left_obs.len() < config.min_node_size || right_obs.len() < config.min_node_size {
        return (tree.clone(), false);
    }

    // Compute log likelihood ratio
    let log_like_old = compute_log_likelihood_leaf(&obs_in_leaf, r, sigma, sigma_mu);
    let log_like_left = compute_log_likelihood_leaf(&left_obs, r, sigma, sigma_mu);
    let log_like_right = compute_log_likelihood_leaf(&right_obs, r, sigma, sigma_mu);
    let log_like_new = log_like_left + log_like_right;

    // Compute tree structure prior ratio
    let depth = leaf.depth;
    let depth_f64: f64 = depth as f64;
    let depth_plus1_f64: f64 = (depth + 1) as f64;
    let p_split = config.alpha * (1.0_f64 + depth_f64).powf(-config.beta);
    let p_not_split = 1.0_f64 - config.alpha * (1.0_f64 + depth_plus1_f64).powf(-config.beta);

    let log_tree_ratio = (p_split * p_not_split * p_not_split / (1.0_f64 - p_split)).ln();

    // Compute proposal ratio
    let n_singly_new = tree.get_singly_internal().len() + 1;
    let n_leaves_old = leaves.len();
    let log_proposal_ratio =
        ((n_leaves_old as f64) / (n_singly_new as f64)).ln() + (var_splits.len() as f64).ln();

    // Accept or reject
    let log_alpha = log_like_new - log_like_old + log_tree_ratio - log_proposal_ratio;

    if rng.r#gen::<f64>().ln() < log_alpha {
        // Accept: create new tree with split
        let mut new_tree = tree.clone();
        let left_idx = new_tree.nodes.len();
        let right_idx = left_idx + 1;

        // Convert leaf to internal node
        new_tree.nodes[leaf_idx].split_feature = Some(var_idx);
        new_tree.nodes[leaf_idx].split_value = Some(split_val);
        new_tree.nodes[leaf_idx].left_child = Some(left_idx);
        new_tree.nodes[leaf_idx].right_child = Some(right_idx);

        // Add new leaves
        new_tree
            .nodes
            .push(BartNode::new_leaf(0.0, depth + 1, Some(leaf_idx), true));
        new_tree
            .nodes
            .push(BartNode::new_leaf(0.0, depth + 1, Some(leaf_idx), false));

        (new_tree, true)
    } else {
        (tree.clone(), false)
    }
}

/// Prune a tree by collapsing a singly internal node
fn prune_tree(
    tree: &BartTree,
    x: &Array2<f64>,
    r: &Array1<f64>,
    sigma: f64,
    sigma_mu: f64,
    config: &BartConfig,
    rng: &mut StdRng,
) -> (BartTree, bool) {
    let singly_internal = tree.get_singly_internal();
    if singly_internal.is_empty() {
        return (tree.clone(), false);
    }

    // Choose a random singly internal node
    let node_idx = singly_internal[rng.gen_range(0..singly_internal.len())];
    let node = &tree.nodes[node_idx];
    let left_idx = node.left_child.unwrap();
    let right_idx = node.right_child.unwrap();

    // Get observations
    let leaf_indices = tree.get_leaf_indices(x);
    let left_obs: Vec<usize> = leaf_indices
        .iter()
        .enumerate()
        .filter(|&(_, l)| *l == left_idx)
        .map(|(i, _)| i)
        .collect();
    let right_obs: Vec<usize> = leaf_indices
        .iter()
        .enumerate()
        .filter(|&(_, l)| *l == right_idx)
        .map(|(i, _)| i)
        .collect();
    let parent_obs: Vec<usize> = left_obs.iter().chain(right_obs.iter()).cloned().collect();

    // Compute log likelihood ratio (inverse of grow)
    let log_like_old = compute_log_likelihood_leaf(&left_obs, r, sigma, sigma_mu)
        + compute_log_likelihood_leaf(&right_obs, r, sigma, sigma_mu);
    let log_like_new = compute_log_likelihood_leaf(&parent_obs, r, sigma, sigma_mu);

    // Tree structure prior ratio (inverse of grow)
    let depth = node.depth;
    let depth_f64: f64 = depth as f64;
    let depth_plus1_f64: f64 = (depth + 1) as f64;
    let p_split = config.alpha * (1.0_f64 + depth_f64).powf(-config.beta);
    let p_not_split = 1.0_f64 - config.alpha * (1.0_f64 + depth_plus1_f64).powf(-config.beta);

    let log_tree_ratio = ((1.0_f64 - p_split) / (p_split * p_not_split * p_not_split)).ln();

    // Proposal ratio (inverse of grow)
    let n_singly_old = singly_internal.len();
    let n_leaves_new = tree.get_leaves().len() - 1;

    // Count unique split values for this feature (avoiding f64 hash issues)
    let n_splits = if let Some(feat) = node.split_feature {
        let mut vals: Vec<f64> = x.column(feat).to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();
        vals.len().saturating_sub(1).max(1)
    } else {
        1
    };

    let log_proposal_ratio =
        ((n_singly_old as f64) / (n_leaves_new as f64)).ln() - (n_splits as f64).ln();

    // Accept or reject
    let log_alpha = log_like_new - log_like_old + log_tree_ratio - log_proposal_ratio;

    if rng.r#gen::<f64>().ln() < log_alpha {
        // Accept: create new tree with node collapsed to leaf
        let mut new_tree = tree.clone();

        // Convert internal node to leaf
        new_tree.nodes[node_idx].split_feature = None;
        new_tree.nodes[node_idx].split_value = None;
        new_tree.nodes[node_idx].left_child = None;
        new_tree.nodes[node_idx].right_child = None;

        // Mark old children as unused (we don't remove to preserve indices)
        // In a production implementation, we'd compact the tree periodically

        (new_tree, true)
    } else {
        (tree.clone(), false)
    }
}

/// Change a split rule at an internal node
fn change_tree(
    tree: &BartTree,
    x: &Array2<f64>,
    r: &Array1<f64>,
    sigma: f64,
    sigma_mu: f64,
    split_points: &[Vec<f64>],
    config: &BartConfig,
    rng: &mut StdRng,
) -> (BartTree, bool) {
    let internal = tree.get_internal_nodes();
    if internal.is_empty() {
        return (tree.clone(), false);
    }

    // Choose a random internal node
    let node_idx = internal[rng.gen_range(0..internal.len())];

    // Choose new split variable and value
    let p = x.ncols();
    let available_vars: Vec<usize> = (0..p).filter(|&j| !split_points[j].is_empty()).collect();

    if available_vars.is_empty() {
        return (tree.clone(), false);
    }

    let new_var = available_vars[rng.gen_range(0..available_vars.len())];
    let var_splits = &split_points[new_var];

    if var_splits.is_empty() {
        return (tree.clone(), false);
    }

    let new_split = var_splits[rng.gen_range(0..var_splits.len())];

    // Compute likelihood under old and new trees
    let log_like_old = compute_tree_log_likelihood(tree, x, r, sigma, sigma_mu);

    let mut new_tree = tree.clone();
    new_tree.nodes[node_idx].split_feature = Some(new_var);
    new_tree.nodes[node_idx].split_value = Some(new_split);

    // Check that children still have enough observations
    let leaf_indices = new_tree.get_leaf_indices(x);
    let leaves = new_tree.get_leaves();

    for &leaf_idx in &leaves {
        let n_in_leaf = leaf_indices.iter().filter(|&&l| l == leaf_idx).count();
        if n_in_leaf < config.min_node_size {
            return (tree.clone(), false);
        }
    }

    let log_like_new = compute_tree_log_likelihood(&new_tree, x, r, sigma, sigma_mu);

    // Accept or reject
    let log_alpha = log_like_new - log_like_old;

    if rng.r#gen::<f64>().ln() < log_alpha {
        (new_tree, true)
    } else {
        (tree.clone(), false)
    }
}

/// Compute log likelihood for a leaf node
fn compute_log_likelihood_leaf(obs: &[usize], r: &Array1<f64>, sigma: f64, sigma_mu: f64) -> f64 {
    let n = obs.len();
    if n == 0 {
        return 0.0;
    }

    let sum_r: f64 = obs.iter().map(|&i| r[i]).sum();
    let sigma2 = sigma * sigma;
    let sigma_mu2 = sigma_mu * sigma_mu;

    // Integrated likelihood with conjugate normal prior
    // p(r | σ², σ_μ²) = ∫ N(r | μ, σ²) N(μ | 0, σ_μ²) dμ
    let posterior_var = 1.0 / (n as f64 / sigma2 + 1.0 / sigma_mu2);
    let posterior_mean = posterior_var * sum_r / sigma2;

    0.5 * (posterior_var / sigma_mu2).ln()
        - 0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln()
        + 0.5 * posterior_mean * posterior_mean / posterior_var
        - 0.5 * obs.iter().map(|&i| r[i] * r[i]).sum::<f64>() / sigma2
}

/// Compute log likelihood for entire tree
fn compute_tree_log_likelihood(
    tree: &BartTree,
    x: &Array2<f64>,
    r: &Array1<f64>,
    sigma: f64,
    sigma_mu: f64,
) -> f64 {
    let leaf_indices = tree.get_leaf_indices(x);
    let leaves = tree.get_leaves();

    leaves
        .iter()
        .map(|&leaf_idx| {
            let obs: Vec<usize> = leaf_indices
                .iter()
                .enumerate()
                .filter(|&(_, l)| *l == leaf_idx)
                .map(|(i, _)| i)
                .collect();
            compute_log_likelihood_leaf(&obs, r, sigma, sigma_mu)
        })
        .sum()
}

/// Sample terminal node parameters
fn sample_mu(
    tree: &mut BartTree,
    x: &Array2<f64>,
    r: &Array1<f64>,
    sigma: f64,
    sigma_mu: f64,
    rng: &mut StdRng,
) {
    let leaf_indices = tree.get_leaf_indices(x);
    let leaves = tree.get_leaves();

    let sigma2 = sigma * sigma;
    let sigma_mu2 = sigma_mu * sigma_mu;

    for leaf_idx in leaves {
        let obs: Vec<usize> = leaf_indices
            .iter()
            .enumerate()
            .filter(|&(_, l)| *l == leaf_idx)
            .map(|(i, _)| i)
            .collect();

        if obs.is_empty() {
            continue;
        }

        let n = obs.len() as f64;
        let sum_r: f64 = obs.iter().map(|&i| r[i]).sum();

        // Posterior: N(posterior_mean, posterior_var)
        let posterior_var = 1.0 / (n / sigma2 + 1.0 / sigma_mu2);
        let posterior_mean = posterior_var * sum_r / sigma2;
        let posterior_sd = posterior_var.sqrt();

        let normal = Normal::new(posterior_mean, posterior_sd).unwrap();
        tree.nodes[leaf_idx].mu = normal.sample(rng);
    }
}

/// Sample sigma from its posterior
fn sample_sigma(sse: f64, n: usize, nu: f64, lambda: f64, rng: &mut StdRng) -> f64 {
    // Posterior: σ² ~ Inv-χ²(ν + n, (νλ + SSE)/(ν + n))
    let post_nu = nu + n as f64;
    let post_scale = (nu * lambda + sse) / post_nu;

    // Sample from scaled inverse chi-squared
    // σ² = post_scale * post_nu / χ²(post_nu)
    let gamma = Gamma::new(post_nu / 2.0, 2.0).unwrap();
    let chi2 = gamma.sample(rng);

    (post_scale * post_nu / chi2).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bart_basic() {
        // Simple regression problem
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);

        let x: Array2<f64> = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                i as f64 / n as f64
            } else {
                rng.r#gen::<f64>()
            }
        });

        // y = 2*x1 + noise
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 * xi)
            + Array1::from_shape_fn(n, |_| {
                let normal = Normal::new(0.0, 0.1).unwrap();
                normal.sample(&mut rng)
            });

        let config = BartConfig {
            num_trees: 50,
            num_burn: 50,
            num_mcmc: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = bart(&x, &y, &config).unwrap();

        // Check that we have predictions
        assert_eq!(result.fitted_values.len(), n);

        // Check that sigma samples were collected
        assert_eq!(result.sigma_samples.len(), config.num_mcmc);

        // Check that variable counts are reasonable
        assert!(result.variable_counts[0] > result.variable_counts[1]);
    }

    #[test]
    fn test_bart_predict() {
        let n = 50;
        let mut rng = StdRng::seed_from_u64(123);

        let x: Array2<f64> = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
        let y: Array1<f64> = x.column(0).mapv(|xi| xi * xi)
            + Array1::from_shape_fn(n, |_| {
                let normal = Normal::new(0.0, 0.05).unwrap();
                normal.sample(&mut rng)
            });

        let config = BartConfig {
            num_trees: 30,
            num_burn: 30,
            num_mcmc: 50,
            seed: Some(123),
            ..Default::default()
        };

        let result = bart(&x, &y, &config).unwrap();

        // Predict on training data
        let preds = bart_predict(&result, &x);
        assert_eq!(preds.len(), n);

        // Predictions should be in a reasonable range
        for &p in preds.iter() {
            assert!(p > -1.0 && p < 2.0);
        }
    }

    #[test]
    fn test_bart_tree_operations() {
        // Test tree grow/prune operations
        let tree = BartTree::new(0.0);

        assert_eq!(tree.nodes.len(), 1);
        assert!(tree.nodes[0].is_leaf());
        assert_eq!(tree.get_leaves().len(), 1);
        assert_eq!(tree.get_internal_nodes().len(), 0);
    }

    #[test]
    fn test_bart_prediction_intervals() {
        let n = 30;
        let mut rng = StdRng::seed_from_u64(456);

        let x: Array2<f64> = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y: Array1<f64> = x.column(0).mapv(|xi| xi)
            + Array1::from_shape_fn(n, |_| {
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.sample(&mut rng)
            });

        let config = BartConfig {
            num_trees: 20,
            num_burn: 20,
            num_mcmc: 50,
            seed: Some(456),
            ..Default::default()
        };

        let result = bart(&x, &y, &config).unwrap();

        let (mean, lower, upper) = bart_predict_intervals(&result, &x, 0.05);

        assert_eq!(mean.len(), n);
        assert_eq!(lower.len(), n);
        assert_eq!(upper.len(), n);

        // Lower should be less than mean, mean less than upper
        for i in 0..n {
            assert!(lower[i] <= mean[i]);
            assert!(mean[i] <= upper[i]);
        }
    }
}
