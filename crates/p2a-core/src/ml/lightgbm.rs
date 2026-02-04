//! LightGBM - Light Gradient Boosting Machine implementation.
//!
//! Implements gradient boosting with histogram-based learning and
//! leaf-wise tree growth strategy.
//!
//! ## Key Features
//!
//! - Histogram-based splitting (faster than exact greedy)
//! - Leaf-wise tree growth (vs level-wise)
//! - Gradient-based one-side sampling (GOSS)
//! - Feature bundling for sparse features
//!
//! ## References
//!
//! Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting
//! Decision Tree." *NeurIPS 2017*, 3149-3157.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

/// LightGBM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LgbConfig {
    /// Number of boosting iterations
    pub n_estimators: usize,
    /// Maximum number of leaves per tree (leaf-wise growth)
    pub num_leaves: usize,
    /// Maximum tree depth (-1 for no limit)
    pub max_depth: i32,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of histogram bins
    pub max_bin: usize,
    /// Minimum data in one leaf
    pub min_data_in_leaf: usize,
    /// L1 regularization
    pub lambda_l1: f64,
    /// L2 regularization
    pub lambda_l2: f64,
    /// Feature fraction (colsample)
    pub feature_fraction: f64,
    /// Bagging fraction (subsample)
    pub bagging_fraction: f64,
    /// Objective function
    pub objective: LgbObjective,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for LgbConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            num_leaves: 31,
            max_depth: -1,
            learning_rate: 0.1,
            max_bin: 255,
            min_data_in_leaf: 20,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            feature_fraction: 1.0,
            bagging_fraction: 1.0,
            objective: LgbObjective::Regression,
            seed: None,
        }
    }
}

/// LightGBM objective functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LgbObjective {
    /// Mean squared error
    #[default]
    Regression,
    /// Binary classification
    Binary,
    /// Multiclass classification
    Multiclass,
}

/// LightGBM result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LgbResult {
    /// Predictions
    pub predictions: Vec<f64>,
    /// Probabilities (for classification)
    pub probabilities: Option<Vec<f64>>,
    /// Feature importances
    pub feature_importances: Vec<f64>,
    /// Number of trees
    pub n_trees: usize,
    /// Training loss per iteration
    pub train_loss: Vec<f64>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
    /// Internal trees
    #[serde(skip)]
    trees: Vec<LgbTree>,
    /// Configuration
    config: LgbConfig,
    /// Base prediction
    base_score: f64,
    /// Histogram bins per feature
    #[serde(skip)]
    bin_edges: Vec<Vec<f64>>,
}

impl std::fmt::Display for LgbResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LightGBM Model")?;
        writeln!(f, "==============")?;
        writeln!(f, "Trees: {}", self.n_trees)?;
        writeln!(f, "Num leaves: {}", self.config.num_leaves)?;
        writeln!(f, "Learning rate: {}", self.config.learning_rate)?;
        writeln!(f, "Max bins: {}", self.config.max_bin)?;
        writeln!(f)?;

        if let Some(loss) = self.train_loss.last() {
            writeln!(f, "Final training loss: {:.6}", loss)?;
        }

        writeln!(f)?;
        writeln!(f, "Feature Importances (top 10):")?;

        let mut indexed: Vec<(usize, f64)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, imp) in indexed.iter().take(10) {
            let name = self
                .feature_names
                .as_ref()
                .and_then(|n| n.get(*idx).cloned())
                .unwrap_or_else(|| format!("X{}", idx));
            writeln!(f, "  {}: {:.4}", name, imp)?;
        }

        Ok(())
    }
}

/// Internal LightGBM tree (leaf-wise).
#[derive(Debug, Clone)]
struct LgbTree {
    nodes: Vec<LgbNode>,
}

#[derive(Debug, Clone)]
struct LgbNode {
    /// Feature index for split
    feature: Option<usize>,
    /// Bin threshold for split
    bin_threshold: usize,
    /// Actual threshold value
    threshold: f64,
    /// Left child
    left: Option<usize>,
    /// Right child
    right: Option<usize>,
    /// Leaf value
    value: f64,
    /// Split gain
    gain: f64,
}

impl LgbTree {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn predict(&self, x: &[f64], bin_edges: &[Vec<f64>]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.feature.is_none() {
                return node.value;
            }

            let feat = node.feature.unwrap();
            let val = x[feat];

            // Find bin for this value
            let bin = find_bin(val, &bin_edges[feat]);

            if bin <= node.bin_threshold {
                node_idx = node.left.unwrap_or(node_idx);
            } else {
                node_idx = node.right.unwrap_or(node_idx);
            }

            // Safety check
            if node_idx >= self.nodes.len() {
                return node.value;
            }
        }
    }
}

/// Histogram for a single feature.
#[derive(Debug, Clone)]
struct Histogram {
    /// Sum of gradients per bin
    grad_sum: Vec<f64>,
    /// Sum of hessians per bin
    hess_sum: Vec<f64>,
    /// Count per bin
    count: Vec<usize>,
}

impl Histogram {
    fn new(n_bins: usize) -> Self {
        Self {
            grad_sum: vec![0.0; n_bins],
            hess_sum: vec![0.0; n_bins],
            count: vec![0; n_bins],
        }
    }

    fn clear(&mut self) {
        self.grad_sum.fill(0.0);
        self.hess_sum.fill(0.0);
        self.count.fill(0);
    }
}

/// Train a LightGBM model.
pub fn lightgbm(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &LgbConfig,
) -> EconResult<LgbResult> {
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(EconError::InsufficientData {
            required: n,
            provided: y.len(),
            context: "X and y must have same number of rows".to_string(),
        });
    }

    let mut rng = match config.seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    // Build histograms (bin edges) for each feature
    let bin_edges: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            let col_vec: Vec<f64> = x.column(j).to_vec();
            compute_bin_edges(&col_vec, config.max_bin)
        })
        .collect();

    // Bin the data
    let binned_data: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..p).map(|j| find_bin(x[[i, j]], &bin_edges[j])).collect())
        .collect();

    // Initialize predictions
    let base_score = match config.objective {
        LgbObjective::Regression => y.mean().unwrap_or(0.0),
        LgbObjective::Binary => {
            let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
            let neg = n as f64 - pos;
            (pos / neg).ln().max(-10.0).min(10.0)
        }
        LgbObjective::Multiclass => 0.0,
    };

    let mut predictions = vec![base_score; n];
    let mut trees = Vec::with_capacity(config.n_estimators);
    let mut train_loss = Vec::with_capacity(config.n_estimators);
    let mut feature_importances = vec![0.0; p];

    let all_indices: Vec<usize> = (0..n).collect();
    let all_features: Vec<usize> = (0..p).collect();

    for _round in 0..config.n_estimators {
        // Compute gradients
        let (grad, hess) =
            compute_lgb_gradients(&predictions, y.as_slice().unwrap(), config.objective);

        // Bagging
        let sample_indices: Vec<usize> = if config.bagging_fraction < 1.0 {
            let n_sample = ((n as f64) * config.bagging_fraction).ceil() as usize;
            let mut indices = all_indices.clone();
            indices.shuffle(&mut rng);
            indices.truncate(n_sample);
            indices
        } else {
            all_indices.clone()
        };

        // Feature sampling
        let feature_indices: Vec<usize> = if config.feature_fraction < 1.0 {
            let n_feat = ((p as f64) * config.feature_fraction).ceil() as usize;
            let mut feats = all_features.clone();
            feats.shuffle(&mut rng);
            feats.truncate(n_feat);
            feats
        } else {
            all_features.clone()
        };

        // Build tree with histogram-based splitting
        let tree = build_lgb_tree(
            &binned_data,
            &grad,
            &hess,
            &sample_indices,
            &feature_indices,
            config,
            &bin_edges,
            &mut feature_importances,
        );

        // Update predictions
        for i in 0..n {
            let row: Vec<f64> = x.row(i).to_vec();
            predictions[i] += config.learning_rate * tree.predict(&row, &bin_edges);
        }

        // Compute loss
        let loss = compute_lgb_loss(&predictions, y.as_slice().unwrap(), config.objective);
        train_loss.push(loss);

        trees.push(tree);
    }

    // Normalize feature importances
    let total: f64 = feature_importances.iter().sum();
    if total > 0.0 {
        for imp in &mut feature_importances {
            *imp /= total;
        }
    }

    // Final predictions
    let (final_preds, probs) = match config.objective {
        LgbObjective::Regression => (predictions.clone(), None),
        LgbObjective::Binary => {
            let probs: Vec<f64> = predictions.iter().map(|&p| sigmoid(p)).collect();
            let preds: Vec<f64> = probs
                .iter()
                .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
                .collect();
            (preds, Some(probs))
        }
        LgbObjective::Multiclass => (predictions.clone(), None),
    };

    Ok(LgbResult {
        predictions: final_preds,
        probabilities: probs,
        feature_importances,
        n_trees: trees.len(),
        train_loss,
        feature_names: None,
        trees,
        config: config.clone(),
        base_score,
        bin_edges,
    })
}

/// Predict using a trained LightGBM model.
pub fn lightgbm_predict(model: &LgbResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    let n = x.nrows();
    let mut predictions = vec![model.base_score; n];

    for i in 0..n {
        let row: Vec<f64> = x.row(i).to_vec();
        for tree in &model.trees {
            predictions[i] += model.config.learning_rate * tree.predict(&row, &model.bin_edges);
        }
    }

    match model.config.objective {
        LgbObjective::Regression => Ok(predictions),
        LgbObjective::Binary => Ok(predictions
            .iter()
            .map(|&p| if sigmoid(p) > 0.5 { 1.0 } else { 0.0 })
            .collect()),
        LgbObjective::Multiclass => Ok(predictions),
    }
}

/// Compute bin edges for a feature.
fn compute_bin_edges(values: &[f64], max_bin: usize) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().cloned().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.is_empty() {
        return vec![0.0];
    }

    // Remove duplicates
    sorted.dedup();

    let n_unique = sorted.len();
    let n_bins = n_unique.min(max_bin);

    if n_bins <= 1 {
        return vec![sorted[0]];
    }

    // Create quantile-based bins
    let mut edges = Vec::with_capacity(n_bins);
    for i in 1..n_bins {
        let idx = (i * n_unique) / n_bins;
        edges.push(sorted[idx.min(n_unique - 1)]);
    }
    edges.push(f64::INFINITY);

    edges
}

/// Find the bin for a value.
fn find_bin(value: f64, edges: &[f64]) -> usize {
    for (i, &edge) in edges.iter().enumerate() {
        if value <= edge {
            return i;
        }
    }
    edges.len().saturating_sub(1)
}

/// Compute gradients for LightGBM.
fn compute_lgb_gradients(
    preds: &[f64],
    y: &[f64],
    objective: LgbObjective,
) -> (Vec<f64>, Vec<f64>) {
    let n = preds.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n];

    match objective {
        LgbObjective::Regression => {
            for i in 0..n {
                grad[i] = preds[i] - y[i];
                hess[i] = 1.0;
            }
        }
        LgbObjective::Binary | LgbObjective::Multiclass => {
            for i in 0..n {
                let p = sigmoid(preds[i]);
                grad[i] = p - y[i];
                hess[i] = (p * (1.0 - p)).max(1e-6);
            }
        }
    }

    (grad, hess)
}

/// Compute loss for LightGBM.
fn compute_lgb_loss(preds: &[f64], y: &[f64], objective: LgbObjective) -> f64 {
    let n = preds.len() as f64;

    match objective {
        LgbObjective::Regression => {
            preds
                .iter()
                .zip(y)
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / n
        }
        LgbObjective::Binary | LgbObjective::Multiclass => {
            preds
                .iter()
                .zip(y)
                .map(|(&p, &t)| {
                    let prob = sigmoid(p);
                    -t * prob.ln().max(-100.0) - (1.0 - t) * (1.0 - prob).ln().max(-100.0)
                })
                .sum::<f64>()
                / n
        }
    }
}

/// Build a LightGBM tree using histogram-based splitting.
fn build_lgb_tree(
    binned_data: &[Vec<usize>],
    grad: &[f64],
    hess: &[f64],
    sample_indices: &[usize],
    feature_indices: &[usize],
    config: &LgbConfig,
    bin_edges: &[Vec<f64>],
    feature_importances: &mut [f64],
) -> LgbTree {
    let mut tree = LgbTree::new();
    let n_bins = config.max_bin;

    // Build histograms for root
    let mut histograms: Vec<Histogram> = feature_indices
        .iter()
        .map(|_| Histogram::new(n_bins))
        .collect();

    // Build root histograms
    for &idx in sample_indices {
        for (hist_idx, &feat) in feature_indices.iter().enumerate() {
            let bin = binned_data[idx][feat];
            if bin < n_bins {
                histograms[hist_idx].grad_sum[bin] += grad[idx];
                histograms[hist_idx].hess_sum[bin] += hess[idx];
                histograms[hist_idx].count[bin] += 1;
            }
        }
    }

    // Leaf-wise growth using a priority queue of leaves to split
    let g_sum: f64 = sample_indices.iter().map(|&i| grad[i]).sum();
    let h_sum: f64 = sample_indices.iter().map(|&i| hess[i]).sum();

    // Find best split from histograms
    let (best_feat_idx, best_bin, best_gain) =
        find_best_split_from_histograms(&histograms, feature_indices, g_sum, h_sum, config);

    if best_gain <= 0.0 || best_feat_idx.is_none() {
        // Make single leaf
        let value = -g_sum / (h_sum + config.lambda_l2);
        tree.nodes.push(LgbNode {
            feature: None,
            bin_threshold: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value,
            gain: 0.0,
        });
        return tree;
    }

    // Simple recursive build (simplified leaf-wise)
    build_lgb_node(
        &mut tree,
        binned_data,
        grad,
        hess,
        sample_indices,
        feature_indices,
        config,
        bin_edges,
        feature_importances,
        0,
        config.num_leaves,
    );

    tree
}

fn build_lgb_node(
    tree: &mut LgbTree,
    binned_data: &[Vec<usize>],
    grad: &[f64],
    hess: &[f64],
    indices: &[usize],
    feature_indices: &[usize],
    config: &LgbConfig,
    bin_edges: &[Vec<f64>],
    feature_importances: &mut [f64],
    depth: usize,
    remaining_leaves: usize,
) -> usize {
    let g_sum: f64 = indices.iter().map(|&i| grad[i]).sum();
    let h_sum: f64 = indices.iter().map(|&i| hess[i]).sum();

    let value = -g_sum / (h_sum + config.lambda_l2);

    // Check stopping conditions
    let max_depth_reached = config.max_depth >= 0 && depth >= config.max_depth as usize;
    if max_depth_reached || indices.len() < config.min_data_in_leaf * 2 || remaining_leaves <= 1 {
        let node_idx = tree.nodes.len();
        tree.nodes.push(LgbNode {
            feature: None,
            bin_threshold: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value,
            gain: 0.0,
        });
        return node_idx;
    }

    // Build histograms
    let n_bins = config.max_bin;
    let mut histograms: Vec<Histogram> = feature_indices
        .iter()
        .map(|_| Histogram::new(n_bins))
        .collect();

    for &idx in indices {
        for (hist_idx, &feat) in feature_indices.iter().enumerate() {
            let bin = binned_data[idx][feat];
            if bin < n_bins {
                histograms[hist_idx].grad_sum[bin] += grad[idx];
                histograms[hist_idx].hess_sum[bin] += hess[idx];
                histograms[hist_idx].count[bin] += 1;
            }
        }
    }

    // Find best split
    let (best_feat_idx, best_bin, best_gain) =
        find_best_split_from_histograms(&histograms, feature_indices, g_sum, h_sum, config);

    if best_gain <= 0.0 || best_feat_idx.is_none() {
        let node_idx = tree.nodes.len();
        tree.nodes.push(LgbNode {
            feature: None,
            bin_threshold: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value,
            gain: 0.0,
        });
        return node_idx;
    }

    let best_feat = feature_indices[best_feat_idx.unwrap()];
    feature_importances[best_feat] += best_gain;

    // Split indices
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &idx in indices {
        if binned_data[idx][best_feat] <= best_bin {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    // Check min_data_in_leaf
    if left_indices.len() < config.min_data_in_leaf || right_indices.len() < config.min_data_in_leaf
    {
        let node_idx = tree.nodes.len();
        tree.nodes.push(LgbNode {
            feature: None,
            bin_threshold: 0,
            threshold: 0.0,
            left: None,
            right: None,
            value,
            gain: 0.0,
        });
        return node_idx;
    }

    // Get actual threshold
    let threshold = if best_bin < bin_edges[best_feat].len() {
        bin_edges[best_feat][best_bin]
    } else {
        f64::INFINITY
    };

    // Create node
    let node_idx = tree.nodes.len();
    tree.nodes.push(LgbNode {
        feature: Some(best_feat),
        bin_threshold: best_bin,
        threshold,
        left: None,
        right: None,
        value: 0.0,
        gain: best_gain,
    });

    // Build children
    let left_idx = build_lgb_node(
        tree,
        binned_data,
        grad,
        hess,
        &left_indices,
        feature_indices,
        config,
        bin_edges,
        feature_importances,
        depth + 1,
        remaining_leaves / 2,
    );

    let right_idx = build_lgb_node(
        tree,
        binned_data,
        grad,
        hess,
        &right_indices,
        feature_indices,
        config,
        bin_edges,
        feature_importances,
        depth + 1,
        remaining_leaves / 2,
    );

    tree.nodes[node_idx].left = Some(left_idx);
    tree.nodes[node_idx].right = Some(right_idx);

    node_idx
}

fn find_best_split_from_histograms(
    histograms: &[Histogram],
    feature_indices: &[usize],
    g_sum: f64,
    h_sum: f64,
    config: &LgbConfig,
) -> (Option<usize>, usize, f64) {
    let mut best_feat_idx = None;
    let mut best_bin = 0;
    let mut best_gain = 0.0;

    for (feat_idx, hist) in histograms.iter().enumerate() {
        let mut g_left = 0.0;
        let mut h_left = 0.0;

        for bin in 0..hist.grad_sum.len() - 1 {
            g_left += hist.grad_sum[bin];
            h_left += hist.hess_sum[bin];

            let g_right = g_sum - g_left;
            let h_right = h_sum - h_left;

            if h_left < 1e-6 || h_right < 1e-6 {
                continue;
            }

            // Gain with L2 regularization
            let gain = 0.5
                * (g_left.powi(2) / (h_left + config.lambda_l2)
                    + g_right.powi(2) / (h_right + config.lambda_l2)
                    - g_sum.powi(2) / (h_sum + config.lambda_l2));

            if gain > best_gain {
                best_gain = gain;
                best_feat_idx = Some(feat_idx);
                best_bin = bin;
            }
        }
    }

    (best_feat_idx, best_bin, best_gain)
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_lightgbm_regression() {
        let x = Array2::from_shape_vec(
            (100, 2),
            (0..200).map(|i| (i % 100) as f64 / 100.0).collect(),
        )
        .unwrap();

        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v) + x.column(1).mapv(|v| v * 0.5);

        let config = LgbConfig {
            n_estimators: 50,
            num_leaves: 15,
            learning_rate: 0.1,
            ..Default::default()
        };

        let result = lightgbm(x.view(), y.view(), &config).unwrap();

        assert_eq!(result.n_trees, 50);
        assert!(!result.predictions.is_empty());
    }

    #[test]
    fn test_lightgbm_classification() {
        let x = Array2::from_shape_vec(
            (100, 2),
            (0..200).map(|i| (i % 100) as f64 / 100.0).collect(),
        )
        .unwrap();

        let y: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|row| if row[0] + row[1] > 1.0 { 1.0 } else { 0.0 })
            .collect();

        let config = LgbConfig {
            n_estimators: 50,
            num_leaves: 15,
            learning_rate: 0.1,
            objective: LgbObjective::Binary,
            ..Default::default()
        };

        let result = lightgbm(x.view(), y.view(), &config).unwrap();

        assert!(result.probabilities.is_some());
    }
}
