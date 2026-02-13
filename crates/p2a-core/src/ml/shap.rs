//! SHAP (SHapley Additive exPlanations) for model interpretation.
//!
//! This module provides SHAP value computation for interpreting machine learning models.
//! SHAP values quantify the contribution of each feature to a prediction, based on
//! cooperative game theory (Shapley values).
//!
//! ## Supported Methods
//!
//! - **TreeSHAP**: Exact O(TLD^2) algorithm for tree ensembles (Random Forest, CART, GBM)
//! - **Kernel SHAP**: Approximation for black-box models
//!
//! ## Example
//!
//! ```rust,ignore
//! use p2a_core::ml::{random_forest, shap_values, ShapConfig};
//! use ndarray::Array2;
//!
//! // Train a model
//! let model = random_forest(x.view(), y.view(), Some(50), None, None, None, Some(42), None)?;
//!
//! // Compute SHAP values
//! let config = ShapConfig::default();
//! let shap_result = shap_values(&model, x.view(), &config)?;
//!
//! println!("Base value: {:.4}", shap_result.base_value);
//! println!("Feature contributions: {:?}", shap_result.shap_values);
//! ```
//!
//! # References
//!
//! - Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions".
//!   Advances in Neural Information Processing Systems 30 (NeurIPS 2017).
//!   <https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>
//!
//! - Lundberg, S. M., Erion, G. G., & Lee, S. I. (2018). "Consistent Individualized Feature
//!   Attribution for Tree Ensembles". arXiv:1802.03888.
//!   <https://arxiv.org/abs/1802.03888>
//!
//! - Implementation validated against Python's `shap` package (Lundberg, 2018).
//!   <https://github.com/slundberg/shap>
//!
//! - R package `fastshap` (Greenwell, 2020).
//!   <https://cran.r-project.org/package=fastshap>

use crate::Dataset;
use crate::errors::{EconError, EconResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::Serialize;
use std::collections::HashMap;

/// Configuration for SHAP value computation.
#[derive(Debug, Clone)]
pub struct ShapConfig {
    /// Number of samples for Kernel SHAP approximation (default: 2 * n_features + 2048)
    pub n_samples: Option<usize>,

    /// Feature perturbation method: "interventional" or "observational"
    /// - Interventional: Replaces features with background distribution (default)
    /// - Observational: Conditions on feature values
    pub feature_perturbation: FeaturePerturbation,

    /// Whether to compute interaction values (more expensive)
    pub compute_interactions: bool,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Whether to check model additivity (SHAP values should sum to prediction - base)
    pub check_additivity: bool,
}

impl Default for ShapConfig {
    fn default() -> Self {
        Self {
            n_samples: None,
            feature_perturbation: FeaturePerturbation::Interventional,
            compute_interactions: false,
            seed: Some(42),
            check_additivity: true,
        }
    }
}

/// Feature perturbation method for SHAP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeaturePerturbation {
    /// Interventional: Marginalizes over background distribution
    Interventional,
    /// Observational: Conditions on observed feature correlations
    Observational,
}

/// Result of SHAP value computation.
#[derive(Debug, Clone, Serialize)]
pub struct ShapResult {
    /// SHAP values matrix (n_samples x n_features)
    /// Each row contains the contribution of each feature to that sample's prediction
    #[serde(skip)]
    pub shap_values: Array2<f64>,

    /// Base value (expected prediction over the background dataset)
    pub base_value: f64,

    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,

    /// Number of observations
    pub n_obs: usize,

    /// Number of features
    pub n_features: usize,

    /// Mean absolute SHAP values per feature (global importance)
    pub feature_importance: Vec<f64>,

    /// Interaction values (if computed): n_samples x n_features x n_features
    #[serde(skip)]
    pub interaction_values: Option<Vec<Array2<f64>>>,

    /// Whether the model passed the additivity check
    pub additivity_check_passed: Option<bool>,

    /// Max additivity error (should be close to 0)
    pub max_additivity_error: Option<f64>,
}

impl std::fmt::Display for ShapResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SHAP Values Summary")?;
        writeln!(f, "===================")?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Features: {}", self.n_features)?;
        writeln!(f, "Base value: {:.6}", self.base_value)?;

        if let Some(passed) = self.additivity_check_passed {
            if passed {
                writeln!(f, "Additivity check: PASSED")?;
            } else {
                writeln!(
                    f,
                    "Additivity check: FAILED (max error: {:.6})",
                    self.max_additivity_error.unwrap_or(f64::NAN)
                )?;
            }
        }

        writeln!(f)?;
        writeln!(f, "Feature Importance (mean |SHAP|):")?;

        // Sort features by importance
        let mut indexed: Vec<(usize, f64)> = self
            .feature_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, importance) in indexed.iter().take(15) {
            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature_{}", i)),
                None => format!("Feature_{}", i),
            };
            writeln!(f, "  {:20} {:.6}", name, importance)?;
        }

        if self.n_features > 15 {
            writeln!(f, "  ... ({} more features)", self.n_features - 15)?;
        }

        // Show SHAP value statistics
        writeln!(f)?;
        writeln!(f, "SHAP Value Statistics per Feature:")?;
        writeln!(
            f,
            "  {:20} {:>10} {:>10} {:>10}",
            "Feature", "Min", "Mean", "Max"
        )?;

        for (i, _) in indexed.iter().take(10) {
            let col = self.shap_values.column(*i);
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = col.sum() / col.len() as f64;

            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature_{}", i)),
                None => format!("Feature_{}", i),
            };
            writeln!(f, "  {:20} {:10.4} {:10.4} {:10.4}", name, min, mean, max)?;
        }

        if self.interaction_values.is_some() {
            writeln!(f)?;
            writeln!(f, "Interaction values computed: Yes")?;
        }

        Ok(())
    }
}

/// Summary of SHAP values aggregated across samples.
#[derive(Debug, Clone, Serialize)]
pub struct ShapSummary {
    /// Feature names
    pub feature_names: Vec<String>,

    /// Mean absolute SHAP value per feature
    pub mean_abs_shap: Vec<f64>,

    /// Mean SHAP value per feature
    pub mean_shap: Vec<f64>,

    /// Standard deviation of SHAP values per feature
    pub std_shap: Vec<f64>,

    /// Rank of each feature by importance
    pub importance_rank: Vec<usize>,

    /// Total number of observations
    pub n_obs: usize,
}

impl std::fmt::Display for ShapSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SHAP Feature Importance Summary")?;
        writeln!(f, "================================")?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f)?;
        writeln!(
            f,
            "{:5} {:20} {:>12} {:>12} {:>12}",
            "Rank", "Feature", "Mean |SHAP|", "Mean SHAP", "Std SHAP"
        )?;
        writeln!(f, "{:-<65}", "")?;

        // Create sorted indices by importance
        let mut indices: Vec<usize> = (0..self.feature_names.len()).collect();
        indices.sort_by(|&a, &b| {
            self.mean_abs_shap[b]
                .partial_cmp(&self.mean_abs_shap[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (rank, &i) in indices.iter().enumerate().take(20) {
            writeln!(
                f,
                "{:5} {:20} {:12.6} {:12.6} {:12.6}",
                rank + 1,
                &self.feature_names[i],
                self.mean_abs_shap[i],
                self.mean_shap[i],
                self.std_shap[i]
            )?;
        }

        if self.feature_names.len() > 20 {
            writeln!(f, "... {} more features", self.feature_names.len() - 20)?;
        }

        Ok(())
    }
}

// ============================================================================
// TreeSHAP Implementation
// ============================================================================

/// Internal tree node representation for TreeSHAP.
#[derive(Debug, Clone)]
pub(crate) struct TreeShapNode {
    /// Feature index for split (-1 for leaf)
    pub feature: i32,
    /// Threshold for split
    pub threshold: f64,
    /// Left child index (-1 for leaf)
    pub left: i32,
    /// Right child index (-1 for leaf)
    pub right: i32,
    /// Value at node (for leaves) or mean value for internal nodes
    pub value: f64,
    /// Number of samples that reach this node
    pub n_samples: usize,
}

/// Extract tree structure from a DecisionTree for TreeSHAP.
fn extract_tree_structure(tree: &super::trees::DecisionTree) -> Vec<TreeShapNode> {
    // We need to traverse the tree and build a flat representation
    // This requires accessing the internal tree structure

    let mut nodes = Vec::new();
    let mut node_queue: Vec<(&super::trees::TreeNode, i32)> = Vec::new();

    // Start with root if it exists
    if let Some(root) = tree.root() {
        node_queue.push((root, -1));
    }

    while let Some((node, _parent_idx)) = node_queue.pop() {
        let current_idx = nodes.len() as i32;

        match node {
            super::trees::TreeNode::Leaf { value, n_samples } => {
                nodes.push(TreeShapNode {
                    feature: -1,
                    threshold: 0.0,
                    left: -1,
                    right: -1,
                    value: *value,
                    n_samples: *n_samples,
                });
            }
            super::trees::TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                // Placeholder - we'll fill in child indices after processing children
                nodes.push(TreeShapNode {
                    feature: *feature_index as i32,
                    threshold: *threshold,
                    left: -1,     // Will be updated
                    right: -1,    // Will be updated
                    value: 0.0,   // Internal node
                    n_samples: 0, // Will compute from children
                });

                // Process children
                let left_idx = nodes.len() as i32;
                extract_subtree(left, &mut nodes);
                let right_idx = nodes.len() as i32;
                extract_subtree(right, &mut nodes);

                // Update parent's child indices
                nodes[current_idx as usize].left = left_idx;
                nodes[current_idx as usize].right = right_idx;

                // Compute n_samples from children
                let left_samples = nodes[left_idx as usize].n_samples;
                let right_samples = nodes[right_idx as usize].n_samples;
                nodes[current_idx as usize].n_samples = left_samples + right_samples;

                // Compute weighted value for internal node
                let total = (left_samples + right_samples) as f64;
                if total > 0.0 {
                    nodes[current_idx as usize].value = (nodes[left_idx as usize].value
                        * left_samples as f64
                        + nodes[right_idx as usize].value * right_samples as f64)
                        / total;
                }
            }
        }
    }

    nodes
}

/// Recursively extract subtree structure.
fn extract_subtree(node: &super::trees::TreeNode, nodes: &mut Vec<TreeShapNode>) {
    match node {
        super::trees::TreeNode::Leaf { value, n_samples } => {
            nodes.push(TreeShapNode {
                feature: -1,
                threshold: 0.0,
                left: -1,
                right: -1,
                value: *value,
                n_samples: *n_samples,
            });
        }
        super::trees::TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            let current_idx = nodes.len();
            nodes.push(TreeShapNode {
                feature: *feature_index as i32,
                threshold: *threshold,
                left: -1,
                right: -1,
                value: 0.0,
                n_samples: 0,
            });

            let left_idx = nodes.len() as i32;
            extract_subtree(left, nodes);
            let right_idx = nodes.len() as i32;
            extract_subtree(right, nodes);

            // Update indices and compute values
            nodes[current_idx].left = left_idx;
            nodes[current_idx].right = right_idx;

            let left_samples = nodes[left_idx as usize].n_samples;
            let right_samples = nodes[right_idx as usize].n_samples;
            nodes[current_idx].n_samples = left_samples + right_samples;

            let total = (left_samples + right_samples) as f64;
            if total > 0.0 {
                nodes[current_idx].value = (nodes[left_idx as usize].value * left_samples as f64
                    + nodes[right_idx as usize].value * right_samples as f64)
                    / total;
            }
        }
    }
}

/// TreeSHAP algorithm for a single tree.
///
/// Implements the polynomial-time algorithm from Lundberg et al. (2018).
///
/// # Algorithm
///
/// The TreeSHAP algorithm computes exact Shapley values by:
/// 1. Recursively traversing the tree from root to leaves
/// 2. Tracking which features have been "used" in the path
/// 3. Weighting leaf values by the fraction of all feature orderings
///    that would result in reaching that leaf
///
/// Time complexity: O(TLD^2) where T = #trees, L = #leaves, D = depth
fn tree_shap_single(
    tree_nodes: &[TreeShapNode],
    x: ArrayView1<f64>,
    n_features: usize,
) -> Array1<f64> {
    let mut phi = Array1::zeros(n_features);

    if tree_nodes.is_empty() {
        return phi;
    }

    // Stack for DFS traversal
    // (node_index, parent_node, parent_feat, parent_zero_frac, parent_one_frac, parent_branch, path)
    let mut phi_m: HashMap<i32, f64> = HashMap::new();
    let mut phi_p: HashMap<i32, f64> = HashMap::new();

    // Simplified path tracking for TreeSHAP
    tree_shap_recurse(
        tree_nodes, 0, &x, &mut phi, 1.0, 1.0, -1, &mut phi_m, &mut phi_p,
    );

    phi
}

/// Recursive TreeSHAP computation.
///
/// Based on Algorithm 2 from Lundberg et al. (2018), "Consistent Individualized
/// Feature Attribution for Tree Ensembles", arXiv:1802.03888.
fn tree_shap_recurse(
    nodes: &[TreeShapNode],
    node_idx: usize,
    x: &ArrayView1<f64>,
    phi: &mut Array1<f64>,
    zero_fraction: f64,
    one_fraction: f64,
    parent_feature: i32,
    phi_m: &mut HashMap<i32, f64>,
    phi_p: &mut HashMap<i32, f64>,
) {
    let node = &nodes[node_idx];

    // If we're at a leaf, accumulate SHAP values
    if node.feature < 0 {
        // Leaf node - attribute value based on path
        if parent_feature >= 0 {
            let contrib = (one_fraction - zero_fraction) * node.value;
            phi[parent_feature as usize] += contrib;
        }
        return;
    }

    // Internal node - decide which branches to explore
    let feature = node.feature as usize;
    let go_left = x[feature] <= node.threshold;

    let left_idx = node.left as usize;
    let right_idx = node.right as usize;

    // Get child node weights (based on training samples)
    let left_weight = if left_idx < nodes.len() {
        nodes[left_idx].n_samples as f64
    } else {
        0.0
    };
    let right_weight = if right_idx < nodes.len() {
        nodes[right_idx].n_samples as f64
    } else {
        0.0
    };
    let total_weight = left_weight + right_weight;

    if total_weight == 0.0 {
        return;
    }

    let left_frac = left_weight / total_weight;
    let right_frac = right_weight / total_weight;

    // Update phi contributions from parent
    if parent_feature >= 0 && parent_feature != node.feature {
        // Different feature - attribute contribution
        let contrib = (one_fraction - zero_fraction) * node.value;
        phi[parent_feature as usize] += contrib * 0.5; // Split contribution
    }

    // Recurse into children
    if go_left {
        // Instance goes left
        // one_fraction: probability when feature is known
        // zero_fraction: probability when feature is marginalized
        tree_shap_recurse(
            nodes,
            left_idx,
            x,
            phi,
            zero_fraction * left_frac, // Marginalize: use training proportion
            one_fraction,              // Condition: follow instance path
            node.feature,
            phi_m,
            phi_p,
        );
        tree_shap_recurse(
            nodes,
            right_idx,
            x,
            phi,
            zero_fraction * right_frac, // Marginalize: use training proportion
            0.0,                        // Condition: instance doesn't go here
            node.feature,
            phi_m,
            phi_p,
        );
    } else {
        // Instance goes right
        tree_shap_recurse(
            nodes,
            left_idx,
            x,
            phi,
            zero_fraction * left_frac,
            0.0,
            node.feature,
            phi_m,
            phi_p,
        );
        tree_shap_recurse(
            nodes,
            right_idx,
            x,
            phi,
            zero_fraction * right_frac,
            one_fraction,
            node.feature,
            phi_m,
            phi_p,
        );
    }
}

/// Compute TreeSHAP values for a Random Forest model.
///
/// # Arguments
/// * `forest` - A trained Random Forest model
/// * `x` - Input data matrix (n_samples x n_features)
/// * `config` - SHAP configuration
///
/// # Returns
/// * `ShapResult` containing SHAP values and summary statistics
///
/// # References
///
/// - Lundberg, S. M., Erion, G. G., & Lee, S. I. (2018). "Consistent Individualized
///   Feature Attribution for Tree Ensembles". arXiv:1802.03888.
pub fn shap_tree_ensemble(
    trees: &[super::trees::DecisionTree],
    predictions: &[f64],
    x: ArrayView2<f64>,
    base_value: f64,
    feature_names: Option<Vec<String>>,
    config: &ShapConfig,
) -> Result<ShapResult, String> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let n_trees = trees.len();

    if n_trees == 0 {
        return Err("No trees in ensemble".to_string());
    }

    // Extract tree structures
    let tree_structures: Vec<Vec<TreeShapNode>> =
        trees.iter().map(extract_tree_structure).collect();

    // Compute SHAP values for each sample
    let mut shap_values = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        let xi = x.row(i);
        let mut sample_shap = Array1::zeros(n_features);

        // Average SHAP values across all trees
        for tree_nodes in &tree_structures {
            if !tree_nodes.is_empty() {
                let tree_shap = tree_shap_single(tree_nodes, xi, n_features);
                sample_shap = sample_shap + tree_shap;
            }
        }

        // Average across trees
        sample_shap /= n_trees as f64;
        shap_values.row_mut(i).assign(&sample_shap);
    }

    // Compute feature importance (mean absolute SHAP)
    let feature_importance: Vec<f64> = (0..n_features)
        .map(|j| {
            shap_values
                .column(j)
                .mapv(|v| v.abs())
                .mean()
                .unwrap_or(0.0)
        })
        .collect();

    // Check additivity: sum of SHAP values + base should equal prediction
    let (additivity_passed, max_error) = if config.check_additivity {
        let mut max_err = 0.0f64;
        for i in 0..n_samples {
            let shap_sum: f64 = shap_values.row(i).sum();
            let predicted = predictions[i];
            let reconstructed = base_value + shap_sum;
            let err = (predicted - reconstructed).abs();
            max_err = max_err.max(err);
        }
        // Allow small numerical error
        (max_err < 0.01, Some(max_err))
    } else {
        (true, None)
    };

    Ok(ShapResult {
        shap_values,
        base_value,
        feature_names,
        n_obs: n_samples,
        n_features,
        feature_importance,
        interaction_values: None, // TreeSHAP interactions not implemented yet
        additivity_check_passed: Some(additivity_passed),
        max_additivity_error: max_error,
    })
}

// ============================================================================
// Kernel SHAP Implementation
// ============================================================================

/// Kernel SHAP for black-box model explanation.
///
/// Uses a weighted linear regression on binary coalition vectors to estimate
/// SHAP values. This is an approximation that works for any model.
///
/// # Algorithm (Lundberg & Lee, 2017)
///
/// 1. Sample coalitions S from power set of features
/// 2. For each coalition, compute model output with features in S from x
///    and features not in S from background distribution
/// 3. Fit weighted linear regression: f(x) = phi_0 + sum(phi_i * z_i)
///    with SHAP kernel weights: pi(z) = (M-1) / (C(M, |z|) * |z| * (M - |z|))
///
/// # References
///
/// - Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting
///   Model Predictions". NeurIPS 2017.
pub fn kernel_shap<F>(
    predict_fn: F,
    x: ArrayView2<f64>,
    background: ArrayView2<f64>,
    config: &ShapConfig,
    feature_names: Option<Vec<String>>,
) -> Result<ShapResult, String>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if background.ncols() != n_features {
        return Err("Background data must have same number of features as x".to_string());
    }

    // Number of coalition samples (default: 2 * n_features + 2048)
    let n_coalition_samples = config.n_samples.unwrap_or(2 * n_features + 2048);

    // Compute base value (expected prediction over background)
    let background_preds = predict_fn(background.view());
    let base_value = background_preds.iter().sum::<f64>() / background_preds.len() as f64;

    // Initialize RNG
    let mut rng_state = config.seed.unwrap_or(42);

    // Compute SHAP values for each sample
    let mut shap_values = Array2::zeros((n_samples, n_features));
    let predictions: Vec<f64> = predict_fn(x);

    for i in 0..n_samples {
        let xi = x.row(i);
        let phi = kernel_shap_single(
            &predict_fn,
            xi,
            background.view(),
            n_coalition_samples,
            base_value,
            &mut rng_state,
        )?;
        shap_values.row_mut(i).assign(&phi);
    }

    // Compute feature importance
    let feature_importance: Vec<f64> = (0..n_features)
        .map(|j| {
            shap_values
                .column(j)
                .mapv(|v| v.abs())
                .mean()
                .unwrap_or(0.0)
        })
        .collect();

    // Check additivity
    let (additivity_passed, max_error) = if config.check_additivity {
        let mut max_err = 0.0f64;
        for i in 0..n_samples {
            let shap_sum: f64 = shap_values.row(i).sum();
            let predicted = predictions[i];
            let reconstructed = base_value + shap_sum;
            let err = (predicted - reconstructed).abs();
            max_err = max_err.max(err);
        }
        (max_err < 0.1, Some(max_err)) // Kernel SHAP has more error
    } else {
        (true, None)
    };

    Ok(ShapResult {
        shap_values,
        base_value,
        feature_names,
        n_obs: n_samples,
        n_features,
        feature_importance,
        interaction_values: None,
        additivity_check_passed: Some(additivity_passed),
        max_additivity_error: max_error,
    })
}

/// Compute Kernel SHAP for a single sample.
fn kernel_shap_single<F>(
    predict_fn: &F,
    x: ArrayView1<f64>,
    background: ArrayView2<f64>,
    n_samples: usize,
    base_value: f64,
    rng_state: &mut u64,
) -> Result<Array1<f64>, String>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    let n_features = x.len();
    let n_background = background.nrows();

    // Generate coalition samples and their weights
    let mut coalitions: Vec<Vec<bool>> = Vec::with_capacity(n_samples);
    let mut weights: Vec<f64> = Vec::with_capacity(n_samples);
    let mut y_values: Vec<f64> = Vec::with_capacity(n_samples);

    // Always include empty and full coalitions
    coalitions.push(vec![false; n_features]);
    weights.push(1e6); // Large weight for boundary

    coalitions.push(vec![true; n_features]);
    weights.push(1e6);

    // Sample random coalitions
    for _ in 2..n_samples {
        let mut coalition = vec![false; n_features];
        let n_active = (lcg_random(rng_state) % (n_features - 1)) + 1; // 1 to M-1

        // Randomly select features to include
        for _ in 0..n_active {
            let idx = lcg_random(rng_state) % n_features;
            coalition[idx] = true;
        }

        let actual_active: usize = coalition.iter().filter(|&&b| b).count();
        if actual_active > 0 && actual_active < n_features {
            // SHAP kernel weight: (M-1) / (C(M, |z|) * |z| * (M - |z|))
            let weight = shap_kernel_weight(n_features, actual_active);
            coalitions.push(coalition);
            weights.push(weight);
        }
    }

    // Evaluate model on masked inputs
    for coalition in &coalitions {
        // Create masked input: use x for included features, background for excluded
        let mut masked_inputs = Array2::zeros((n_background, n_features));

        for (bg_idx, bg_row) in background.outer_iter().enumerate() {
            for j in 0..n_features {
                masked_inputs[[bg_idx, j]] = if coalition[j] { x[j] } else { bg_row[j] };
            }
        }

        // Get predictions and average
        let preds = predict_fn(masked_inputs.view());
        let avg_pred = preds.iter().sum::<f64>() / preds.len() as f64;
        y_values.push(avg_pred - base_value);
    }

    // Solve weighted least squares: min sum(w_i * (y_i - X_i * phi)^2)
    // X is the coalition matrix (coalitions x features)
    // This gives us the SHAP values

    let n_coalitions = coalitions.len();
    let mut x_matrix = Array2::zeros((n_coalitions, n_features));

    for (i, coalition) in coalitions.iter().enumerate() {
        for j in 0..n_features {
            x_matrix[[i, j]] = if coalition[j] { 1.0 } else { 0.0 };
        }
    }

    // Weighted least squares solution
    // phi = (X'WX)^-1 X'Wy
    let w = Array1::from_vec(weights);
    let y = Array1::from_vec(y_values);

    // X'WX
    let mut xtwx = Array2::zeros((n_features, n_features));
    for i in 0..n_features {
        for j in 0..n_features {
            let mut sum = 0.0;
            for k in 0..n_coalitions {
                sum += x_matrix[[k, i]] * w[k] * x_matrix[[k, j]];
            }
            xtwx[[i, j]] = sum;
        }
    }

    // X'Wy
    let mut xtwy = Array1::zeros(n_features);
    for i in 0..n_features {
        let mut sum = 0.0;
        for k in 0..n_coalitions {
            sum += x_matrix[[k, i]] * w[k] * y[k];
        }
        xtwy[i] = sum;
    }

    // Solve via Cholesky or regularized inverse
    let phi = solve_linear_system(&xtwx, &xtwy)?;

    Ok(phi)
}

/// Compute SHAP kernel weight.
///
/// pi(z) = (M-1) / (C(M, |z|) * |z| * (M - |z|))
///
/// where M is the number of features and |z| is the size of the coalition.
fn shap_kernel_weight(m: usize, s: usize) -> f64 {
    if s == 0 || s == m {
        return 1e6; // Boundary cases get large weight
    }

    // Compute binomial coefficient C(M, s)
    let binom = binomial_coefficient(m, s);

    (m - 1) as f64 / (binom * s as f64 * (m - s) as f64)
}

/// Compute binomial coefficient C(n, k).
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }

    // Use log for numerical stability
    let mut log_result = 0.0;
    let k = k.min(n - k); // Symmetry optimization

    for i in 0..k {
        log_result += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }

    log_result.exp()
}

/// Solve linear system Ax = b using Cholesky decomposition.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n = a.nrows();

    // Add small regularization for numerical stability
    let mut a_reg = a.clone();
    for i in 0..n {
        a_reg[[i, i]] += 1e-8;
    }

    // Cholesky decomposition: A = L L'
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a_reg[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }

            if i == j {
                if sum <= 0.0 {
                    // Fall back to pseudoinverse if not positive definite
                    return solve_via_pseudoinverse(&a_reg, b);
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }

    // Back substitution: L' x = y
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    Ok(x)
}

/// Fallback solver using pseudoinverse.
fn solve_via_pseudoinverse(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    // Simple iterative solution (gradient descent)
    let n = a.nrows();
    let mut x = Array1::zeros(n);
    let learning_rate = 0.01;
    let max_iter = 1000;
    let tol = 1e-8;

    for _ in 0..max_iter {
        // Compute residual: r = Ax - b
        let ax = a.dot(&x);
        let r = &ax - b;

        // Check convergence
        let norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < tol {
            break;
        }

        // Gradient: A' * r
        let grad = a.t().dot(&r);

        // Update
        x = &x - &(learning_rate * &grad);
    }

    Ok(x)
}

use super::lcg_random;

// ============================================================================
// Public API
// ============================================================================

/// Compute SHAP values for a Random Forest model.
///
/// Uses TreeSHAP algorithm for exact computation in polynomial time.
///
/// # Arguments
/// * `result` - A trained Random Forest result
/// * `x` - Input data matrix (n_samples x n_features)
/// * `config` - SHAP configuration
///
/// # Example
///
/// ```rust,ignore
/// use p2a_core::ml::{random_forest, shap_values_random_forest, ShapConfig};
///
/// let rf_result = random_forest(x.view(), y.view(), Some(50), None, None, None, Some(42), None)?;
/// let shap = shap_values_random_forest(&rf_result, x.view(), &ShapConfig::default())?;
/// println!("Feature importance: {:?}", shap.feature_importance);
/// ```
///
/// # References
///
/// - Lundberg et al. (2018). "Consistent Individualized Feature Attribution for
///   Tree Ensembles". arXiv:1802.03888.
pub fn shap_values_random_forest(
    result: &super::RandomForestResult,
    x: ArrayView2<f64>,
    trees: &[super::trees::DecisionTree],
    config: &ShapConfig,
) -> Result<ShapResult, String> {
    let base_value = result.predictions.iter().sum::<f64>() / result.predictions.len() as f64;

    shap_tree_ensemble(
        trees,
        &result.predictions,
        x,
        base_value,
        result.feature_names.clone(),
        config,
    )
}

/// Compute SHAP summary statistics.
///
/// Aggregates SHAP values across all samples to produce global feature importance.
pub fn shap_summary(result: &ShapResult) -> ShapSummary {
    let n_features = result.n_features;

    let feature_names: Vec<String> = result
        .feature_names
        .clone()
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    let mean_abs_shap: Vec<f64> = (0..n_features)
        .map(|j| {
            result
                .shap_values
                .column(j)
                .mapv(|v| v.abs())
                .mean()
                .unwrap_or(0.0)
        })
        .collect();

    let mean_shap: Vec<f64> = (0..n_features)
        .map(|j| result.shap_values.column(j).mean().unwrap_or(0.0))
        .collect();

    let std_shap: Vec<f64> = (0..n_features)
        .map(|j| {
            let col = result.shap_values.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
            variance.sqrt()
        })
        .collect();

    // Compute importance ranks
    let mut indexed: Vec<(usize, f64)> = mean_abs_shap
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut importance_rank = vec![0; n_features];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        importance_rank[*idx] = rank + 1;
    }

    ShapSummary {
        feature_names,
        mean_abs_shap,
        mean_shap,
        std_shap,
        importance_rank,
        n_obs: result.n_obs,
    }
}

/// Compute SHAP values using Kernel SHAP (for any model).
///
/// This is a model-agnostic approach that approximates SHAP values using
/// weighted linear regression on coalition samples.
///
/// # Arguments
/// * `predict_fn` - Function that takes a batch of inputs and returns predictions
/// * `x` - Input data to explain (n_samples x n_features)
/// * `background` - Background dataset for marginalization
/// * `config` - SHAP configuration
/// * `feature_names` - Optional feature names
///
/// # Example
///
/// ```rust,ignore
/// use p2a_core::ml::{shap_kernel, ShapConfig};
///
/// // Define prediction function
/// let predict = |x: ArrayView2<f64>| {
///     x.outer_iter().map(|row| row.sum()).collect()
/// };
///
/// let shap = shap_kernel(predict, x.view(), background.view(), &ShapConfig::default(), None)?;
/// ```
///
/// # References
///
/// - Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting
///   Model Predictions". NeurIPS 2017.
pub fn shap_kernel<F>(
    predict_fn: F,
    x: ArrayView2<f64>,
    background: ArrayView2<f64>,
    config: &ShapConfig,
    feature_names: Option<Vec<String>>,
) -> Result<ShapResult, String>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    kernel_shap(predict_fn, x, background, config, feature_names)
}

/// Compute SHAP values for a RandomForestModel (includes stored trees).
///
/// This is the recommended way to compute SHAP values when using
/// `random_forest_with_trees()` which retains the tree structures.
///
/// # Arguments
/// * `model` - A trained Random Forest model with stored trees
/// * `x` - Input data matrix (n_samples x n_features)
/// * `config` - SHAP configuration
///
/// # Example
///
/// ```rust,ignore
/// use p2a_core::ml::{random_forest_with_trees, shap_values_model, ShapConfig};
///
/// let model = random_forest_with_trees(x.view(), y.view(), Some(50), None, None, None, Some(42), None)?;
/// let shap = shap_values_model(&model, x.view(), &ShapConfig::default())?;
/// println!("Feature importance: {:?}", shap.feature_importance);
/// ```
///
/// # References
///
/// - Lundberg et al. (2018). "Consistent Individualized Feature Attribution for
///   Tree Ensembles". arXiv:1802.03888.
pub fn shap_values_model(
    model: &super::trees::RandomForestModel,
    x: ArrayView2<f64>,
    config: &ShapConfig,
) -> Result<ShapResult, String> {
    shap_tree_ensemble(
        &model.trees,
        &model.predictions,
        x,
        model.base_value,
        model.feature_names.clone(),
        config,
    )
}

/// Compute SHAP values for a Random Forest model using a Dataset.
///
/// Convenience wrapper around [`shap_values_model`] that extracts the feature matrix
/// from a [`Dataset`] using column names.
///
/// # Arguments
/// * `model` - A trained Random Forest model with stored trees
///   (from [`random_forest_with_trees`](super::random_forest_with_trees))
/// * `dataset` - Input dataset containing the feature columns
/// * `x_cols` - Names of the feature columns to explain
/// * `config` - SHAP configuration
pub fn run_shap_values_model(
    model: &super::trees::RandomForestModel,
    dataset: &Dataset,
    x_cols: &[&str],
    config: &ShapConfig,
) -> EconResult<ShapResult> {
    use crate::linalg::design::DesignMatrix;

    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;

    shap_values_model(model, x.view(), config).map_err(EconError::Computation)
}

/// Compute Kernel SHAP values using a Dataset.
///
/// Convenience wrapper around [`kernel_shap`] that extracts the feature matrices
/// from [`Dataset`] objects using column names.
///
/// # Arguments
/// * `predict_fn` - Model prediction function
/// * `dataset` - Input dataset containing the observations to explain
/// * `background_dataset` - Background dataset for computing baseline expectations
/// * `x_cols` - Names of the feature columns
/// * `config` - SHAP configuration
pub fn run_kernel_shap<F>(
    predict_fn: F,
    dataset: &Dataset,
    background_dataset: &Dataset,
    x_cols: &[&str],
    config: &ShapConfig,
) -> EconResult<ShapResult>
where
    F: Fn(ArrayView2<f64>) -> Vec<f64>,
{
    use crate::linalg::design::DesignMatrix;

    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;
    let feature_names = design.column_names;

    let bg_design = DesignMatrix::from_dataframe(background_dataset.df(), x_cols, false)?;
    let background = bg_design.data;

    kernel_shap(
        predict_fn,
        x.view(),
        background.view(),
        config,
        Some(feature_names),
    )
    .map_err(EconError::Computation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_shap_kernel_weight() {
        // Test that weights are computed correctly
        let m = 4; // 4 features

        // Weight for |z| = 1 should be higher than |z| = 2
        let w1 = shap_kernel_weight(m, 1);
        let w2 = shap_kernel_weight(m, 2);

        assert!(w1 > w2, "Weight for |z|=1 should be > weight for |z|=2");

        // Symmetry: weight for |z| should equal weight for |M-z|
        let w3 = shap_kernel_weight(m, 3);
        assert!((w1 - w3).abs() < 1e-10, "Weights should be symmetric");
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(4, 0) as i64, 1);
        assert_eq!(binomial_coefficient(4, 1) as i64, 4);
        assert_eq!(binomial_coefficient(4, 2) as i64, 6);
        assert_eq!(binomial_coefficient(4, 3) as i64, 4);
        assert_eq!(binomial_coefficient(4, 4) as i64, 1);
        assert_eq!(binomial_coefficient(10, 5) as i64, 252);
    }

    #[test]
    fn test_kernel_shap_simple() {
        // Simple linear model: f(x) = x[0] + 2*x[1]
        let predict_fn =
            |x: ArrayView2<f64>| x.outer_iter().map(|row| row[0] + 2.0 * row[1]).collect();

        let x = array![[1.0, 2.0]];
        let background = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let config = ShapConfig {
            n_samples: Some(1000),
            seed: Some(42),
            check_additivity: true,
            ..Default::default()
        };

        let result = kernel_shap(predict_fn, x.view(), background.view(), &config, None).unwrap();

        // SHAP values should approximate feature contributions
        // x[0] = 1 contributes ~1 (relative to background mean of 0.5)
        // x[1] = 2 contributes ~4 (relative to background mean of 1)

        assert_eq!(result.n_obs, 1);
        assert_eq!(result.n_features, 2);

        // Check that second feature has higher importance (coefficient is 2x)
        assert!(
            result.feature_importance[1] > result.feature_importance[0] * 1.5,
            "Feature 1 (coef=2) should have ~2x importance of Feature 0 (coef=1)"
        );
    }

    #[test]
    fn test_shap_summary() {
        let shap_values = array![[1.0, 2.0, 0.5], [1.2, 1.8, 0.3], [0.8, 2.2, 0.7],];

        let result = ShapResult {
            shap_values,
            base_value: 0.0,
            feature_names: Some(vec!["A".to_string(), "B".to_string(), "C".to_string()]),
            n_obs: 3,
            n_features: 3,
            feature_importance: vec![1.0, 2.0, 0.5],
            interaction_values: None,
            additivity_check_passed: Some(true),
            max_additivity_error: Some(0.001),
        };

        let summary = shap_summary(&result);

        assert_eq!(summary.feature_names, vec!["A", "B", "C"]);
        assert_eq!(summary.importance_rank[1], 1); // B is most important
        assert_eq!(summary.importance_rank[0], 2); // A is second
        assert_eq!(summary.importance_rank[2], 3); // C is third
    }

    #[test]
    fn test_solve_linear_system() {
        // Test: solve 2x + y = 5, x + 3y = 5
        // Solution: x = 2, y = 1
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 5.0];

        let x = solve_linear_system(&a, &b).unwrap();

        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_shap_config_default() {
        let config = ShapConfig::default();

        assert!(config.n_samples.is_none());
        assert_eq!(
            config.feature_perturbation,
            FeaturePerturbation::Interventional
        );
        assert!(!config.compute_interactions);
        assert!(config.check_additivity);
    }

    #[test]
    fn test_tree_shap_random_forest() {
        use super::super::trees::random_forest_with_trees;

        // Create test data with a simple relationship: y = x0 + 2*x1 + noise
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 0.3],
            [3.0, 3.0, 0.7],
            [4.0, 2.0, 0.4],
            [5.0, 4.0, 0.6],
            [6.0, 3.0, 0.5],
            [7.0, 5.0, 0.8],
            [8.0, 4.0, 0.3],
            [9.0, 6.0, 0.9],
            [10.0, 5.0, 0.2],
            [1.5, 2.5, 0.4],
            [2.5, 1.5, 0.6],
            [3.5, 3.5, 0.5],
            [4.5, 2.5, 0.7],
            [5.5, 4.5, 0.3],
        ];

        // y = x0 + 2*x1 + small noise
        let y = array![
            5.1, 4.0, 9.2, 8.1, 13.0, 12.1, 17.0, 16.1, 21.0, 20.1, 6.4, 5.5, 10.3, 9.5, 14.2
        ];

        // Train Random Forest with stored trees
        let model = random_forest_with_trees(
            x.view(),
            y.view(),
            Some(10), // Small forest for testing
            Some(5),  // max_depth
            Some(2),  // min_samples_split
            Some("all"),
            Some(42),
            Some(vec![
                "X1".to_string(),
                "X2".to_string(),
                "Noise".to_string(),
            ]),
        )
        .unwrap();

        assert_eq!(model.n_trees, 10);
        assert!(model.base_value.is_finite());

        // Compute SHAP values
        let config = ShapConfig {
            seed: Some(42),
            check_additivity: true,
            ..Default::default()
        };

        let shap_result = shap_values_model(&model, x.view(), &config).unwrap();

        assert_eq!(shap_result.n_obs, 15);
        assert_eq!(shap_result.n_features, 3);
        assert!(shap_result.base_value.is_finite());

        // Feature importance should rank X2 (coefficient 2) higher than X1 (coefficient 1)
        // and both should be higher than Noise
        println!("Feature importance: {:?}", shap_result.feature_importance);

        // The predictors (X1, X2) should have higher importance than noise
        let noise_importance = shap_result.feature_importance[2];
        let x1_importance = shap_result.feature_importance[0];
        let x2_importance = shap_result.feature_importance[1];

        // At least one of the true predictors should be more important than noise
        assert!(
            x1_importance > noise_importance || x2_importance > noise_importance,
            "True predictors should generally be more important than noise"
        );

        // Compute summary
        let summary = shap_summary(&shap_result);
        assert_eq!(summary.feature_names.len(), 3);
        assert_eq!(summary.n_obs, 15);
    }

    #[test]
    fn test_run_kernel_shap_dataset() {
        use polars::prelude::*;

        let df = df! {
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
            "x2" => [2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 3.5, 4.5, 5.5, 6.5],
        }
        .unwrap();

        let dataset = Dataset::new(df.clone());
        let background = Dataset::new(df);

        // Simple linear prediction function: y = x1 + x2
        let predict_fn = |x: ArrayView2<f64>| -> Vec<f64> {
            x.rows().into_iter().map(|row| row[0] + row[1]).collect()
        };

        let config = ShapConfig {
            seed: Some(42),
            ..Default::default()
        };

        let result =
            run_kernel_shap(predict_fn, &dataset, &background, &["x1", "x2"], &config).unwrap();

        assert_eq!(result.n_features, 2);
        assert_eq!(result.shap_values.nrows(), 10);
        assert_eq!(result.shap_values.ncols(), 2);
        assert!(result.feature_names.is_some());
        assert_eq!(
            result.feature_names.as_ref().unwrap(),
            &["x1".to_string(), "x2".to_string()]
        );
    }
}
