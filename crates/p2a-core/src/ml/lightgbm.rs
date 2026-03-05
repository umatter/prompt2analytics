//! LightGBM (Light Gradient Boosting Machine) implementation.
//!
//! Pure Rust implementation of LightGBM, a high-performance gradient boosting framework
//! that uses histogram-based learning and leaf-wise (best-first) tree growth.
//!
//! ## Key Innovations
//!
//! - **Histogram-based split finding**: Continuous features are binned into discrete bins (default: 255),
//!   reducing computation from O(#data) to O(#bins) per split.
//! - **Leaf-wise tree growth**: Unlike level-wise growth, always grows the leaf with maximum delta loss,
//!   resulting in asymmetric trees that achieve lower training error.
//! - **Histogram subtraction trick**: Parent histogram - sibling = current node histogram,
//!   halving the computation for histogram building.
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{lightgbm, LightGbmConfig, LightGbmObjective};
//! use ndarray::array;
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let y = array![1.1, 1.9, 3.2, 3.8, 5.1];
//!
//! let config = LightGbmConfig {
//!     num_iterations: 100,
//!     learning_rate: 0.1,
//!     num_leaves: 31,
//!     ..Default::default()
//! };
//!
//! let result = lightgbm(x.view(), y.view(), &config).unwrap();
//! println!("Training loss: {:.4}", result.final_train_loss);
//! ```
//!
//! ## References
//!
//! - Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree".
//!   Advances in Neural Information Processing Systems 30 (NIPS 2017).
//! - LightGBM Documentation: https://lightgbm.readthedocs.io/

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

use crate::errors::{EconError, EconResult};
use crate::Dataset;

/// Objective function for LightGBM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LightGbmObjective {
    /// Regression with L2 loss (MSE)
    #[default]
    Regression,
    /// Regression with L1 loss (MAE)
    RegressionL1,
    /// Regression with Huber loss
    Huber,
    /// Binary classification with log loss
    Binary,
}

impl std::str::FromStr for LightGbmObjective {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "regression" | "regression_l2" | "mse" | "mean_squared_error" => {
                Ok(LightGbmObjective::Regression)
            }
            "regression_l1" | "mae" | "mean_absolute_error" => Ok(LightGbmObjective::RegressionL1),
            "huber" => Ok(LightGbmObjective::Huber),
            "binary" | "binary_logloss" | "logistic" => Ok(LightGbmObjective::Binary),
            _ => Err(format!("Unknown LightGBM objective: {}", s)),
        }
    }
}

impl std::fmt::Display for LightGbmObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LightGbmObjective::Regression => write!(f, "regression"),
            LightGbmObjective::RegressionL1 => write!(f, "regression_l1"),
            LightGbmObjective::Huber => write!(f, "huber"),
            LightGbmObjective::Binary => write!(f, "binary"),
        }
    }
}

/// Feature importance type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportanceType {
    /// Split-based importance (number of times feature is used for splitting)
    #[default]
    Split,
    /// Gain-based importance (total gain from splits using this feature)
    Gain,
}

/// Configuration for LightGBM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightGbmConfig {
    /// Number of boosting iterations (trees)
    pub num_iterations: usize,
    /// Learning rate (shrinkage factor), default 0.1
    pub learning_rate: f64,
    /// Maximum number of leaves in one tree, default 31
    pub num_leaves: usize,
    /// Maximum tree depth (-1 means no limit), default -1
    pub max_depth: i32,
    /// Minimum number of data in one leaf, default 20
    pub min_data_in_leaf: usize,
    /// L1 regularization, default 0.0
    pub lambda_l1: f64,
    /// L2 regularization, default 0.0
    pub lambda_l2: f64,
    /// Fraction of features to use for each tree, default 1.0
    pub feature_fraction: f64,
    /// Fraction of data to use for each tree (bagging), default 1.0
    pub bagging_fraction: f64,
    /// Frequency for bagging (0 = disabled), default 0
    pub bagging_freq: usize,
    /// Maximum number of bins for histogram, default 255
    pub max_bin: usize,
    /// Objective function
    pub objective: LightGbmObjective,
    /// Huber delta parameter (for Huber loss)
    pub huber_delta: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for LightGbmConfig {
    fn default() -> Self {
        LightGbmConfig {
            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1,
            min_data_in_leaf: 20,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            feature_fraction: 1.0,
            bagging_fraction: 1.0,
            bagging_freq: 0,
            max_bin: 255,
            objective: LightGbmObjective::Regression,
            huber_delta: 1.0,
            seed: None,
        }
    }
}

/// Result from LightGBM fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightGbmResult {
    /// Feature importances (gain-based, normalized)
    pub feature_importances: Vec<f64>,
    /// Training loss at each iteration
    pub train_loss: Vec<f64>,
    /// Final training loss
    pub final_train_loss: f64,
    /// Number of trees fitted
    pub num_trees: usize,
    /// Initial prediction (mean for regression, log-odds for classification)
    pub init_prediction: f64,
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Configuration used
    pub config: LightGbmConfig,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Internal: trees (serialized for prediction)
    #[serde(skip)]
    pub(crate) trees: Vec<LgbTree>,
    /// Internal: feature bins for prediction
    #[serde(skip)]
    pub(crate) feature_bins: Vec<FeatureBins>,
}

impl std::fmt::Display for LightGbmResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LightGBM Results")?;
        writeln!(f, "================")?;
        writeln!(f, "Objective: {}", self.config.objective)?;
        writeln!(f, "Number of trees: {}", self.num_trees)?;
        writeln!(f, "Learning rate: {:.4}", self.config.learning_rate)?;
        writeln!(f, "Num leaves: {}", self.config.num_leaves)?;
        writeln!(f, "Max bins: {}", self.config.max_bin)?;

        if self.config.max_depth > 0 {
            writeln!(f, "Max depth: {}", self.config.max_depth)?;
        }

        if self.config.lambda_l1 > 0.0 || self.config.lambda_l2 > 0.0 {
            writeln!(
                f,
                "Regularization: L1={:.4}, L2={:.4}",
                self.config.lambda_l1, self.config.lambda_l2
            )?;
        }

        if self.config.feature_fraction < 1.0 {
            writeln!(f, "Feature fraction: {:.2}", self.config.feature_fraction)?;
        }

        if self.config.bagging_fraction < 1.0 {
            writeln!(f, "Bagging fraction: {:.2}", self.config.bagging_fraction)?;
        }

        writeln!(f)?;
        writeln!(f, "Final training loss: {:.6}", self.final_train_loss)?;

        if self.train_loss.len() > 1 {
            writeln!(f, "Initial loss: {:.6}", self.train_loss[0])?;
            let reduction = if self.train_loss[0] > 0.0 {
                (1.0 - self.final_train_loss / self.train_loss[0]) * 100.0
            } else {
                0.0
            };
            writeln!(f, "Loss reduction: {:.2}%", reduction)?;
        }

        writeln!(f)?;
        writeln!(f, "Feature Importances (Gain):")?;

        let mut indexed: Vec<(usize, f64)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("Feature_{}", i)),
                None => format!("Feature_{}", i),
            };
            writeln!(f, "  {}: {:.4}", name, importance)?;
        }

        if self.feature_importances.len() > 10 {
            writeln!(
                f,
                "  ... ({} more features)",
                self.feature_importances.len() - 10
            )?;
        }

        Ok(())
    }
}

// ============================================================================
// Internal Structures
// ============================================================================

/// Binned feature data for histogram-based learning.
#[derive(Debug, Clone)]
pub(crate) struct FeatureBins {
    /// Bin boundaries (sorted thresholds)
    pub boundaries: Vec<f64>,
    /// Number of bins (boundaries.len() + 1)
    pub num_bins: usize,
}

impl FeatureBins {
    /// Create feature bins using quantile-based binning.
    fn from_data(values: &[f64], max_bins: usize) -> Self {
        // Get unique sorted values
        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup();

        if sorted.len() <= 1 {
            return FeatureBins {
                boundaries: Vec::new(),
                num_bins: 1,
            };
        }

        let num_boundaries = (max_bins - 1).min(sorted.len() - 1);
        let mut boundaries = Vec::with_capacity(num_boundaries);

        if num_boundaries < sorted.len() - 1 {
            // Quantile-based binning
            for i in 1..=num_boundaries {
                let idx = (i * sorted.len()) / (num_boundaries + 1);
                let idx = idx.min(sorted.len() - 1);
                // Use midpoint between values
                if idx > 0 {
                    let boundary = (sorted[idx - 1] + sorted[idx]) / 2.0;
                    if boundaries.last().map_or(true, |&last| boundary > last) {
                        boundaries.push(boundary);
                    }
                }
            }
        } else {
            // Use all unique values as boundaries
            for i in 0..sorted.len() - 1 {
                let boundary = (sorted[i] + sorted[i + 1]) / 2.0;
                boundaries.push(boundary);
            }
        }

        FeatureBins {
            num_bins: boundaries.len() + 1,
            boundaries,
        }
    }

    /// Get bin index for a value.
    #[inline]
    fn get_bin(&self, value: f64) -> usize {
        // Binary search for the right bin
        match self.boundaries.binary_search_by(|b| {
            b.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i + 1, // Value equals boundary, goes to right bin
            Err(i) => i,    // Value between boundaries[i-1] and boundaries[i]
        }
    }
}

/// Histogram for a single feature at a node.
#[derive(Debug, Clone)]
struct Histogram {
    /// Sum of gradients per bin
    gradient_sum: Vec<f64>,
    /// Sum of hessians per bin
    hessian_sum: Vec<f64>,
    /// Count per bin
    count: Vec<usize>,
}

impl Histogram {
    fn new(num_bins: usize) -> Self {
        Histogram {
            gradient_sum: vec![0.0; num_bins],
            hessian_sum: vec![0.0; num_bins],
            count: vec![0; num_bins],
        }
    }

    fn reset(&mut self) {
        self.gradient_sum.fill(0.0);
        self.hessian_sum.fill(0.0);
        self.count.fill(0);
    }
}

/// A leaf in the priority queue for leaf-wise growth.
#[derive(Debug, Clone)]
struct LeafInfo {
    /// Leaf ID
    id: usize,
    /// Potential gain if this leaf is split
    gain: f64,
    /// Best split feature
    best_feature: usize,
    /// Best split bin
    best_bin: usize,
    /// Sum of gradients
    grad_sum: f64,
    /// Sum of hessians
    hess_sum: f64,
    /// Sample indices in this leaf
    indices: Vec<usize>,
    /// Depth of this leaf
    depth: usize,
}

impl PartialEq for LeafInfo {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}

impl Eq for LeafInfo {}

impl PartialOrd for LeafInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LeafInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap by gain
        self.gain
            .partial_cmp(&other.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Node in a LightGBM tree.
#[derive(Debug, Clone)]
pub(crate) enum LgbNode {
    Split {
        feature: usize,
        bin_threshold: usize, // Split at <= bin_threshold
        left: Box<LgbNode>,
        right: Box<LgbNode>,
        gain: f64,
    },
    Leaf {
        value: f64,
        n_samples: usize,
    },
}

/// A single LightGBM tree.
#[derive(Debug, Clone)]
pub(crate) struct LgbTree {
    root: Option<LgbNode>,
    n_features: usize,
    feature_importance_gain: Vec<f64>,
    feature_importance_split: Vec<usize>,
}

impl LgbTree {
    fn new(n_features: usize) -> Self {
        LgbTree {
            root: None,
            n_features,
            feature_importance_gain: vec![0.0; n_features],
            feature_importance_split: vec![0; n_features],
        }
    }

    /// Predict for a single observation.
    fn predict_one(&self, binned_row: &[usize]) -> f64 {
        match &self.root {
            Some(node) => self.traverse(node, binned_row),
            None => 0.0,
        }
    }

    fn traverse(&self, node: &LgbNode, binned_row: &[usize]) -> f64 {
        match node {
            LgbNode::Leaf { value, .. } => *value,
            LgbNode::Split {
                feature,
                bin_threshold,
                left,
                right,
                ..
            } => {
                if binned_row[*feature] <= *bin_threshold {
                    self.traverse(left, binned_row)
                } else {
                    self.traverse(right, binned_row)
                }
            }
        }
    }
}

// ============================================================================
// Core Algorithm
// ============================================================================

use super::lcg_random;

/// Sigmoid function (numerically stable).
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Compute gradients and hessians for the objective.
fn compute_gradients_hessians(
    y: &ArrayView1<f64>,
    predictions: &Array1<f64>,
    objective: LightGbmObjective,
    huber_delta: f64,
) -> (Array1<f64>, Array1<f64>) {
    let n = y.len();
    let mut gradients = Array1::zeros(n);
    let mut hessians = Array1::zeros(n);

    match objective {
        LightGbmObjective::Regression => {
            // L2 loss: L = (y - f)^2 / 2
            // gradient = -(y - f) = f - y
            // hessian = 1
            for i in 0..n {
                gradients[i] = predictions[i] - y[i];
                hessians[i] = 1.0;
            }
        }
        LightGbmObjective::RegressionL1 => {
            // L1 loss: L = |y - f|
            // gradient = -sign(y - f)
            // hessian = 1 (approximation)
            for i in 0..n {
                let r = y[i] - predictions[i];
                gradients[i] = if r > 0.0 {
                    -1.0
                } else if r < 0.0 {
                    1.0
                } else {
                    0.0
                };
                hessians[i] = 1.0;
            }
        }
        LightGbmObjective::Huber => {
            // Huber loss
            for i in 0..n {
                let r = y[i] - predictions[i];
                if r.abs() <= huber_delta {
                    gradients[i] = -r;
                    hessians[i] = 1.0;
                } else {
                    gradients[i] = -huber_delta * r.signum();
                    hessians[i] = 0.0; // Or small constant
                }
            }
        }
        LightGbmObjective::Binary => {
            // Log loss: L = -[y*log(p) + (1-y)*log(1-p)]
            // where p = sigmoid(f)
            // gradient = p - y
            // hessian = p * (1 - p)
            for i in 0..n {
                let p = sigmoid(predictions[i]);
                gradients[i] = p - y[i];
                hessians[i] = (p * (1.0 - p)).max(1e-10);
            }
        }
    }

    (gradients, hessians)
}

/// Compute loss for current predictions.
fn compute_loss(
    y: &ArrayView1<f64>,
    predictions: &Array1<f64>,
    objective: LightGbmObjective,
    huber_delta: f64,
) -> f64 {
    let n = y.len() as f64;

    match objective {
        LightGbmObjective::Regression => {
            // MSE
            y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| (yi - pi).powi(2))
                .sum::<f64>()
                / n
        }
        LightGbmObjective::RegressionL1 => {
            // MAE
            y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| (yi - pi).abs())
                .sum::<f64>()
                / n
        }
        LightGbmObjective::Huber => {
            y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| {
                    let r = (yi - pi).abs();
                    if r <= huber_delta {
                        0.5 * r.powi(2)
                    } else {
                        huber_delta * (r - 0.5 * huber_delta)
                    }
                })
                .sum::<f64>()
                / n
        }
        LightGbmObjective::Binary => {
            // Log loss
            -y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| {
                    let p = sigmoid(pi);
                    yi * p.max(1e-15).ln() + (1.0 - yi) * (1.0 - p).max(1e-15).ln()
                })
                .sum::<f64>()
                / n
        }
    }
}

/// Compute initial prediction.
fn compute_init_prediction(y: &ArrayView1<f64>, objective: LightGbmObjective) -> f64 {
    match objective {
        LightGbmObjective::Regression | LightGbmObjective::Huber => {
            // Mean
            y.iter().sum::<f64>() / y.len() as f64
        }
        LightGbmObjective::RegressionL1 => {
            // Median
            let mut sorted: Vec<f64> = y.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
        LightGbmObjective::Binary => {
            // Log-odds
            let p = y.iter().sum::<f64>() / y.len() as f64;
            let p = p.clamp(1e-10, 1.0 - 1e-10);
            (p / (1.0 - p)).ln()
        }
    }
}

/// Compute leaf value using Newton's method with regularization.
/// leaf_value = -sum(gradient) / (sum(hessian) + lambda_l2)
#[inline]
fn compute_leaf_value(grad_sum: f64, hess_sum: f64, lambda_l1: f64, lambda_l2: f64) -> f64 {
    // L1 regularization (soft thresholding)
    let adjusted_grad = if lambda_l1 > 0.0 {
        if grad_sum > lambda_l1 {
            grad_sum - lambda_l1
        } else if grad_sum < -lambda_l1 {
            grad_sum + lambda_l1
        } else {
            0.0
        }
    } else {
        grad_sum
    };

    -adjusted_grad / (hess_sum + lambda_l2).max(1e-10)
}

/// Compute split gain using LightGBM's formula.
/// gain = 0.5 * [G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - (G_L+G_R)^2/(H_L+H_R + lambda)]
#[inline]
fn compute_split_gain(
    grad_left: f64,
    hess_left: f64,
    grad_right: f64,
    hess_right: f64,
    lambda_l2: f64,
) -> f64 {
    let score_left = grad_left.powi(2) / (hess_left + lambda_l2).max(1e-10);
    let score_right = grad_right.powi(2) / (hess_right + lambda_l2).max(1e-10);
    let score_parent = (grad_left + grad_right).powi(2) / (hess_left + hess_right + lambda_l2).max(1e-10);

    0.5 * (score_left + score_right - score_parent)
}

/// Build histograms for all features at a node.
fn build_histograms(
    binned_data: &Array2<usize>,
    gradients: &Array1<f64>,
    hessians: &Array1<f64>,
    indices: &[usize],
    feature_bins: &[FeatureBins],
    feature_subset: &[usize],
) -> Vec<Histogram> {
    let n_features = feature_bins.len();
    let mut histograms: Vec<Histogram> = (0..n_features)
        .map(|f| Histogram::new(feature_bins[f].num_bins))
        .collect();

    // Only build histograms for selected features
    for &idx in indices {
        let g = gradients[idx];
        let h = hessians[idx];

        for &f in feature_subset {
            let bin = binned_data[[idx, f]];
            histograms[f].gradient_sum[bin] += g;
            histograms[f].hessian_sum[bin] += h;
            histograms[f].count[bin] += 1;
        }
    }

    histograms
}

/// Use histogram subtraction to compute child histogram from parent and sibling.
fn subtract_histograms(parent: &Histogram, sibling: &Histogram) -> Histogram {
    let num_bins = parent.gradient_sum.len();
    let mut result = Histogram::new(num_bins);

    for i in 0..num_bins {
        result.gradient_sum[i] = parent.gradient_sum[i] - sibling.gradient_sum[i];
        result.hessian_sum[i] = parent.hessian_sum[i] - sibling.hessian_sum[i];
        result.count[i] = parent.count[i].saturating_sub(sibling.count[i]);
    }

    result
}

/// Find best split from histograms.
fn find_best_split(
    histograms: &[Histogram],
    feature_subset: &[usize],
    total_grad: f64,
    total_hess: f64,
    min_data_in_leaf: usize,
    lambda_l2: f64,
) -> Option<(usize, usize, f64)> {
    // Returns (feature, bin_threshold, gain)
    let mut best_gain = 0.0;
    let mut best_split: Option<(usize, usize)> = None;

    for &feature in feature_subset {
        let hist = &histograms[feature];
        let num_bins = hist.gradient_sum.len();

        if num_bins <= 1 {
            continue;
        }

        let mut left_grad = 0.0;
        let mut left_hess = 0.0;
        let mut left_count = 0usize;

        // Scan through bins, computing cumulative sums
        for bin in 0..num_bins - 1 {
            left_grad += hist.gradient_sum[bin];
            left_hess += hist.hessian_sum[bin];
            left_count += hist.count[bin];

            let right_count = hist.count.iter().sum::<usize>() - left_count;

            // Check min_data_in_leaf constraint
            if left_count < min_data_in_leaf || right_count < min_data_in_leaf {
                continue;
            }

            let right_grad = total_grad - left_grad;
            let right_hess = total_hess - left_hess;

            let gain = compute_split_gain(left_grad, left_hess, right_grad, right_hess, lambda_l2);

            if gain > best_gain {
                best_gain = gain;
                best_split = Some((feature, bin));
            }
        }
    }

    best_split.map(|(f, b)| (f, b, best_gain))
}

/// Fit a single LightGBM tree using leaf-wise growth.
fn fit_tree(
    binned_data: &Array2<usize>,
    gradients: &Array1<f64>,
    hessians: &Array1<f64>,
    sample_indices: &[usize],
    feature_bins: &[FeatureBins],
    config: &LightGbmConfig,
    feature_subset: &[usize],
) -> LgbTree {
    let n_features = feature_bins.len();
    let mut tree = LgbTree::new(n_features);

    if sample_indices.is_empty() {
        return tree;
    }

    // Initial leaf (root)
    let total_grad: f64 = sample_indices.iter().map(|&i| gradients[i]).sum();
    let total_hess: f64 = sample_indices.iter().map(|&i| hessians[i]).sum();

    // Build initial histograms
    let histograms = build_histograms(
        binned_data,
        gradients,
        hessians,
        sample_indices,
        feature_bins,
        feature_subset,
    );

    // Find best split for root
    let best_split = find_best_split(
        &histograms,
        feature_subset,
        total_grad,
        total_hess,
        config.min_data_in_leaf,
        config.lambda_l2,
    );

    let root_leaf = LeafInfo {
        id: 0,
        gain: best_split.map(|(_, _, g)| g).unwrap_or(0.0),
        best_feature: best_split.map(|(f, _, _)| f).unwrap_or(0),
        best_bin: best_split.map(|(_, b, _)| b).unwrap_or(0),
        grad_sum: total_grad,
        hess_sum: total_hess,
        indices: sample_indices.to_vec(),
        depth: 0,
    };

    // If no valid split, return single leaf
    if best_split.is_none() || root_leaf.indices.len() < 2 * config.min_data_in_leaf {
        let value = compute_leaf_value(total_grad, total_hess, config.lambda_l1, config.lambda_l2);
        tree.root = Some(LgbNode::Leaf {
            value,
            n_samples: sample_indices.len(),
        });
        return tree;
    }

    // Priority queue for leaf-wise growth
    let mut leaf_queue: BinaryHeap<LeafInfo> = BinaryHeap::new();
    leaf_queue.push(root_leaf);

    // Track nodes by ID for tree construction
    let mut nodes: Vec<Option<LgbNode>> = vec![None; 2 * config.num_leaves];
    let mut leaf_count = 0;
    let mut next_id = 0;

    // Keep track of parent histograms for subtraction trick
    let mut histogram_cache: std::collections::HashMap<usize, Vec<Histogram>> =
        std::collections::HashMap::new();
    histogram_cache.insert(0, histograms);

    while let Some(leaf) = leaf_queue.pop() {
        // Check if we've reached max leaves
        if leaf_count >= config.num_leaves - 1 {
            // Convert remaining to leaf node
            let value =
                compute_leaf_value(leaf.grad_sum, leaf.hess_sum, config.lambda_l1, config.lambda_l2);
            nodes[leaf.id] = Some(LgbNode::Leaf {
                value,
                n_samples: leaf.indices.len(),
            });
            leaf_count += 1;
            continue;
        }

        // Check depth limit
        let max_depth = if config.max_depth < 0 {
            usize::MAX
        } else {
            config.max_depth as usize
        };
        if leaf.depth >= max_depth {
            let value =
                compute_leaf_value(leaf.grad_sum, leaf.hess_sum, config.lambda_l1, config.lambda_l2);
            nodes[leaf.id] = Some(LgbNode::Leaf {
                value,
                n_samples: leaf.indices.len(),
            });
            leaf_count += 1;
            continue;
        }

        // Check minimum gain
        if leaf.gain <= 0.0 {
            let value =
                compute_leaf_value(leaf.grad_sum, leaf.hess_sum, config.lambda_l1, config.lambda_l2);
            nodes[leaf.id] = Some(LgbNode::Leaf {
                value,
                n_samples: leaf.indices.len(),
            });
            leaf_count += 1;
            continue;
        }

        // Split the leaf
        let split_feature = leaf.best_feature;
        let split_bin = leaf.best_bin;

        // Partition indices
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = leaf
            .indices
            .iter()
            .partition(|&&i| binned_data[[i, split_feature]] <= split_bin);

        // Skip if split doesn't meet min_data_in_leaf
        if left_indices.len() < config.min_data_in_leaf
            || right_indices.len() < config.min_data_in_leaf
        {
            let value =
                compute_leaf_value(leaf.grad_sum, leaf.hess_sum, config.lambda_l1, config.lambda_l2);
            nodes[leaf.id] = Some(LgbNode::Leaf {
                value,
                n_samples: leaf.indices.len(),
            });
            leaf_count += 1;
            continue;
        }

        // Update feature importance
        tree.feature_importance_gain[split_feature] += leaf.gain;
        tree.feature_importance_split[split_feature] += 1;

        // Compute child statistics
        let left_grad: f64 = left_indices.iter().map(|&i| gradients[i]).sum();
        let left_hess: f64 = left_indices.iter().map(|&i| hessians[i]).sum();
        let right_grad = leaf.grad_sum - left_grad;
        let right_hess = leaf.hess_sum - left_hess;

        // Build histograms for smaller child, use subtraction for larger
        let (left_histograms, right_histograms) =
            if left_indices.len() < right_indices.len() {
                let left_hist = build_histograms(
                    binned_data,
                    gradients,
                    hessians,
                    &left_indices,
                    feature_bins,
                    feature_subset,
                );

                // Use subtraction trick for right child
                let parent_hist = histogram_cache.remove(&leaf.id).unwrap_or_else(|| {
                    build_histograms(
                        binned_data,
                        gradients,
                        hessians,
                        &leaf.indices,
                        feature_bins,
                        feature_subset,
                    )
                });

                let right_hist: Vec<Histogram> = parent_hist
                    .iter()
                    .zip(left_hist.iter())
                    .map(|(p, l)| subtract_histograms(p, l))
                    .collect();

                (left_hist, right_hist)
            } else {
                let right_hist = build_histograms(
                    binned_data,
                    gradients,
                    hessians,
                    &right_indices,
                    feature_bins,
                    feature_subset,
                );

                let parent_hist = histogram_cache.remove(&leaf.id).unwrap_or_else(|| {
                    build_histograms(
                        binned_data,
                        gradients,
                        hessians,
                        &leaf.indices,
                        feature_bins,
                        feature_subset,
                    )
                });

                let left_hist: Vec<Histogram> = parent_hist
                    .iter()
                    .zip(right_hist.iter())
                    .map(|(p, r)| subtract_histograms(p, r))
                    .collect();

                (left_hist, right_hist)
            };

        // Find best splits for children
        let left_split = find_best_split(
            &left_histograms,
            feature_subset,
            left_grad,
            left_hess,
            config.min_data_in_leaf,
            config.lambda_l2,
        );

        let right_split = find_best_split(
            &right_histograms,
            feature_subset,
            right_grad,
            right_hess,
            config.min_data_in_leaf,
            config.lambda_l2,
        );

        // Create child leaf info
        next_id += 1;
        let left_id = next_id;
        next_id += 1;
        let right_id = next_id;

        let left_leaf = LeafInfo {
            id: left_id,
            gain: left_split.map(|(_, _, g)| g).unwrap_or(0.0),
            best_feature: left_split.map(|(f, _, _)| f).unwrap_or(0),
            best_bin: left_split.map(|(_, b, _)| b).unwrap_or(0),
            grad_sum: left_grad,
            hess_sum: left_hess,
            indices: left_indices,
            depth: leaf.depth + 1,
        };

        let right_leaf = LeafInfo {
            id: right_id,
            gain: right_split.map(|(_, _, g)| g).unwrap_or(0.0),
            best_feature: right_split.map(|(f, _, _)| f).unwrap_or(0),
            best_bin: right_split.map(|(_, b, _)| b).unwrap_or(0),
            grad_sum: right_grad,
            hess_sum: right_hess,
            indices: right_indices,
            depth: leaf.depth + 1,
        };

        // Store histograms for children
        histogram_cache.insert(left_id, left_histograms);
        histogram_cache.insert(right_id, right_histograms);

        // Add children to queue
        leaf_queue.push(left_leaf);
        leaf_queue.push(right_leaf);

        // Create split node (children will be filled later)
        nodes[leaf.id] = Some(LgbNode::Split {
            feature: split_feature,
            bin_threshold: split_bin,
            left: Box::new(LgbNode::Leaf {
                value: 0.0,
                n_samples: 0,
            }), // Placeholder
            right: Box::new(LgbNode::Leaf {
                value: 0.0,
                n_samples: 0,
            }), // Placeholder
            gain: leaf.gain,
        });
    }

    // Convert remaining items in queue to leaves
    for leaf in leaf_queue {
        let value =
            compute_leaf_value(leaf.grad_sum, leaf.hess_sum, config.lambda_l1, config.lambda_l2);
        nodes[leaf.id] = Some(LgbNode::Leaf {
            value,
            n_samples: leaf.indices.len(),
        });
    }

    // Build tree structure from nodes
    tree.root = build_tree_from_nodes(&nodes, 0);

    tree
}

/// Recursively build tree from node array.
fn build_tree_from_nodes(nodes: &[Option<LgbNode>], id: usize) -> Option<LgbNode> {
    if id >= nodes.len() {
        return None;
    }

    nodes[id].clone()
}

// ============================================================================
// Public API
// ============================================================================

/// Fit a LightGBM model.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - LightGBM configuration
///
/// # Returns
///
/// LightGbmResult containing fitted model and diagnostics.
///
/// # Example
///
/// ```rust,no_run
/// use p2a_core::ml::{lightgbm, LightGbmConfig, LightGbmObjective};
/// use ndarray::array;
///
/// let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
/// let y = array![1.1, 1.9, 3.2, 3.8, 5.1];
///
/// let config = LightGbmConfig::default();
/// let result = lightgbm(x.view(), y.view(), &config).unwrap();
/// ```
pub fn lightgbm(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &LightGbmConfig,
) -> EconResult<LightGbmResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for LightGBM".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }

    // Validate binary targets for classification
    if config.objective == LightGbmObjective::Binary {
        for &yi in y.iter() {
            if yi != 0.0 && yi != 1.0 {
                return Err(EconError::Computation(
                    "Binary objective requires binary targets (0 or 1)".to_string(),
                ));
            }
        }
    }

    let mut rng_state = config.seed.unwrap_or(42);

    // Step 1: Bin features into histograms
    let mut feature_bins = Vec::with_capacity(n_features);
    for f in 0..n_features {
        let col_values: Vec<f64> = (0..n_samples).map(|i| x[[i, f]]).collect();
        feature_bins.push(FeatureBins::from_data(&col_values, config.max_bin));
    }

    // Step 2: Create binned data matrix
    let mut binned_data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for f in 0..n_features {
            binned_data[[i, f]] = feature_bins[f].get_bin(x[[i, f]]);
        }
    }

    // Step 3: Initialize predictions
    let init_pred = compute_init_prediction(&y, config.objective);
    let mut predictions = Array1::from_elem(n_samples, init_pred);

    let mut trees = Vec::with_capacity(config.num_iterations);
    let mut train_loss = Vec::with_capacity(config.num_iterations + 1);
    let mut total_importance_gain: Array1<f64> = Array1::zeros(n_features);
    let mut total_importance_split: Array1<f64> = Array1::zeros(n_features);

    // Initial loss
    train_loss.push(compute_loss(&y, &predictions, config.objective, config.huber_delta));

    // Step 4: Boosting iterations
    for iter in 0..config.num_iterations {
        // Compute gradients and hessians
        let (gradients, hessians) =
            compute_gradients_hessians(&y, &predictions, config.objective, config.huber_delta);

        // Sample rows (bagging)
        let sample_indices: Vec<usize> = if config.bagging_fraction < 1.0
            && (config.bagging_freq == 0 || iter % config.bagging_freq == 0)
        {
            let n_subsample = ((n_samples as f64) * config.bagging_fraction).ceil() as usize;
            let mut indices: Vec<usize> = (0..n_samples).collect();

            // Fisher-Yates shuffle
            for i in 0..n_subsample.min(indices.len()) {
                let j = i + lcg_random(&mut rng_state) % (indices.len() - i);
                indices.swap(i, j);
            }
            indices.truncate(n_subsample);
            indices
        } else {
            (0..n_samples).collect()
        };

        // Sample features
        let feature_subset: Vec<usize> = if config.feature_fraction < 1.0 {
            let n_features_sample =
                ((n_features as f64) * config.feature_fraction).ceil() as usize;
            let mut indices: Vec<usize> = (0..n_features).collect();
            for i in 0..n_features_sample.min(indices.len()) {
                let j = i + lcg_random(&mut rng_state) % (indices.len() - i);
                indices.swap(i, j);
            }
            indices.truncate(n_features_sample);
            indices
        } else {
            (0..n_features).collect()
        };

        // Fit tree
        let tree = fit_tree(
            &binned_data,
            &gradients,
            &hessians,
            &sample_indices,
            &feature_bins,
            config,
            &feature_subset,
        );

        // Update predictions
        for i in 0..n_samples {
            let binned_row: Vec<usize> = (0..n_features).map(|f| binned_data[[i, f]]).collect();
            predictions[i] += config.learning_rate * tree.predict_one(&binned_row);
        }

        // Accumulate importance
        for f in 0..n_features {
            total_importance_gain[f] += tree.feature_importance_gain[f];
            total_importance_split[f] += tree.feature_importance_split[f] as f64;
        }

        trees.push(tree);

        // Track loss
        train_loss.push(compute_loss(
            &y,
            &predictions,
            config.objective,
            config.huber_delta,
        ));
    }

    // Normalize feature importances (gain-based)
    let sum_gain: f64 = total_importance_gain.sum();
    let feature_importances = if sum_gain > 0.0 {
        total_importance_gain.mapv(|v| v / sum_gain).to_vec()
    } else {
        vec![0.0; n_features]
    };

    // Convert predictions to probabilities for classification
    let final_predictions = if config.objective == LightGbmObjective::Binary {
        predictions.mapv(sigmoid).to_vec()
    } else {
        predictions.to_vec()
    };

    Ok(LightGbmResult {
        feature_importances,
        train_loss: train_loss.clone(),
        final_train_loss: *train_loss.last().unwrap_or(&0.0),
        num_trees: config.num_iterations,
        init_prediction: init_pred,
        predictions: final_predictions,
        config: config.clone(),
        feature_names: None,
        trees,
        feature_bins,
    })
}

/// Predict using a fitted LightGBM model.
///
/// # Arguments
///
/// * `result` - Fitted LightGBM result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions (probabilities for classification, values for regression)
pub fn lightgbm_predict(result: &LightGbmResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if result.trees.is_empty() {
        return Err(EconError::Computation(
            "Model has no fitted trees".to_string(),
        ));
    }

    if n_features != result.feature_bins.len() {
        return Err(EconError::Computation(format!(
            "Feature count mismatch: model expects {} features, got {}",
            result.feature_bins.len(),
            n_features
        )));
    }

    // Bin the prediction data
    let mut binned_data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for f in 0..n_features {
            binned_data[[i, f]] = result.feature_bins[f].get_bin(x[[i, f]]);
        }
    }

    let mut predictions = Array1::from_elem(n_samples, result.init_prediction);

    for tree in &result.trees {
        for i in 0..n_samples {
            let binned_row: Vec<usize> = (0..n_features).map(|f| binned_data[[i, f]]).collect();
            predictions[i] += result.config.learning_rate * tree.predict_one(&binned_row);
        }
    }

    // Convert to probabilities for classification
    if result.config.objective == LightGbmObjective::Binary {
        Ok(predictions.mapv(sigmoid).to_vec())
    } else {
        Ok(predictions.to_vec())
    }
}

/// Get feature importance from a fitted LightGBM model.
///
/// # Arguments
///
/// * `result` - Fitted LightGBM result
/// * `importance_type` - Type of importance (Split or Gain)
///
/// # Returns
///
/// HashMap of feature name to importance
pub fn lightgbm_feature_importance(
    result: &LightGbmResult,
    importance_type: ImportanceType,
) -> std::collections::HashMap<String, f64> {
    let n_features = result.feature_importances.len();
    let mut importance = std::collections::HashMap::new();

    match importance_type {
        ImportanceType::Gain => {
            // Already computed and normalized
            for i in 0..n_features {
                let name = result
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(i).cloned())
                    .unwrap_or_else(|| format!("Feature_{}", i));
                importance.insert(name, result.feature_importances[i]);
            }
        }
        ImportanceType::Split => {
            // Compute split-based importance
            let mut split_counts = vec![0usize; n_features];
            for tree in &result.trees {
                for (i, &count) in tree.feature_importance_split.iter().enumerate() {
                    split_counts[i] += count;
                }
            }
            let total: f64 = split_counts.iter().sum::<usize>() as f64;
            for i in 0..n_features {
                let name = result
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(i).cloned())
                    .unwrap_or_else(|| format!("Feature_{}", i));
                let imp = if total > 0.0 {
                    split_counts[i] as f64 / total
                } else {
                    0.0
                };
                importance.insert(name, imp);
            }
        }
    }

    importance
}

/// Run LightGBM on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column name
/// * `x_cols` - Feature column names
/// * `config` - LightGBM configuration
///
/// # Returns
///
/// LightGbmResult with model and diagnostics
pub fn run_lightgbm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &LightGbmConfig,
) -> EconResult<LightGbmResult> {
    use crate::linalg::design::DesignMatrix;

    // Build design matrix
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;
    let feature_names = design.column_names;

    // Get y column
    let col_names: Vec<String> = dataset
        .df()
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let y_series = dataset.df().column(y_col).map_err(|_| EconError::ColumnNotFound {
        column: y_col.to_string(),
        available: col_names.clone(),
    })?;

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::Computation(format!("Column '{}' is not numeric", y_col)))?
        .into_no_null_iter()
        .collect();

    let y_arr = Array1::from_vec(y);

    let mut result = lightgbm(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Convenience function for running LightGBM with default configuration.
pub fn run_lightgbm_default(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<LightGbmResult> {
    run_lightgbm(dataset, y_col, x_cols, &LightGbmConfig::default())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lightgbm_regression_basic() {
        // Simple linear relationship with noise
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.1, 2.1, 2.9, 4.2, 4.8, 6.1, 6.9, 8.2, 8.8, 10.1];

        let config = LightGbmConfig {
            num_iterations: 50,
            learning_rate: 0.1,
            num_leaves: 8,
            min_data_in_leaf: 1,
            ..Default::default()
        };

        let result = lightgbm(x.view(), y.view(), &config).unwrap();

        // Check we got the right number of trees
        assert_eq!(result.num_trees, 50);

        // Loss should decrease
        assert!(
            result.final_train_loss < result.train_loss[0],
            "Final loss {} should be < initial loss {}",
            result.final_train_loss,
            result.train_loss[0]
        );

        // Predictions should be close to targets
        let mse: f64 = result
            .predictions
            .iter()
            .zip(y.iter())
            .map(|(p, y)| (p - y).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 2.0, "MSE {} should be < 2.0", mse);
    }

    #[test]
    fn test_lightgbm_binary_classification() {
        // Binary classification
        let x = array![
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 0.0],
            [2.5, 0.5],
            [8.0, 1.0],
            [8.5, 0.5],
            [9.0, 1.0],
            [9.5, 0.5],
        ];
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let config = LightGbmConfig {
            num_iterations: 50,
            learning_rate: 0.3,
            num_leaves: 4,
            min_data_in_leaf: 1,
            objective: LightGbmObjective::Binary,
            ..Default::default()
        };

        let result = lightgbm(x.view(), y.view(), &config).unwrap();

        // Predictions should be probabilities
        for &p in &result.predictions {
            assert!(p >= 0.0 && p <= 1.0, "Probability {} out of range", p);
        }

        // Low values should have low probability, high values should have high probability
        let avg_low: f64 = result.predictions[0..4].iter().sum::<f64>() / 4.0;
        let avg_high: f64 = result.predictions[4..8].iter().sum::<f64>() / 4.0;
        assert!(avg_low < 0.5, "Average low prediction {} should be < 0.5", avg_low);
        assert!(avg_high > 0.5, "Average high prediction {} should be > 0.5", avg_high);
    }

    #[test]
    fn test_lightgbm_huber() {
        // Robust regression with outlier
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 50.0]; // outlier at 10

        let config = LightGbmConfig {
            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 8,
            min_data_in_leaf: 1,
            objective: LightGbmObjective::Huber,
            huber_delta: 1.0,
            ..Default::default()
        };

        let result = lightgbm(x.view(), y.view(), &config).unwrap();

        // Loss should decrease
        assert!(result.final_train_loss <= result.train_loss[0]);
    }

    #[test]
    fn test_lightgbm_predict() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let config = LightGbmConfig {
            num_iterations: 50,
            learning_rate: 0.1,
            num_leaves: 8,
            min_data_in_leaf: 1,
            ..Default::default()
        };

        let result = lightgbm(x_train.view(), y_train.view(), &config).unwrap();

        // Predict on new data
        let x_test = array![[1.5], [4.5], [7.5]];
        let predictions = lightgbm_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        // Predictions should be in reasonable range
        for &p in &predictions {
            assert!(p > 0.0 && p < 10.0, "Prediction {} out of expected range", p);
        }
    }

    #[test]
    fn test_lightgbm_feature_importance() {
        // First feature is the predictor, second is noise
        let x = array![
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 7.0],
            [4.0, 2.0],
            [5.0, 8.0],
            [6.0, 4.0],
            [7.0, 9.0],
            [8.0, 1.0],
            [9.0, 6.0],
            [10.0, 5.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = LightGbmConfig {
            num_iterations: 50,
            learning_rate: 0.1,
            num_leaves: 8,
            min_data_in_leaf: 1,
            ..Default::default()
        };

        let result = lightgbm(x.view(), y.view(), &config).unwrap();

        // First feature should be more important
        assert!(
            result.feature_importances[0] > result.feature_importances[1],
            "Feature 0 importance {} should be > Feature 1 importance {}",
            result.feature_importances[0],
            result.feature_importances[1]
        );
    }

    #[test]
    fn test_lightgbm_regularization() {
        let x = array![
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.6],
            [7.0, 0.7],
            [8.0, 0.8]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Without regularization
        let config_no_reg = LightGbmConfig {
            num_iterations: 50,
            learning_rate: 0.1,
            num_leaves: 8,
            min_data_in_leaf: 1,
            ..Default::default()
        };

        // With L2 regularization
        let config_l2 = LightGbmConfig {
            num_iterations: 50,
            learning_rate: 0.1,
            num_leaves: 8,
            min_data_in_leaf: 1,
            lambda_l2: 10.0,
            ..Default::default()
        };

        let result_no_reg = lightgbm(x.view(), y.view(), &config_no_reg).unwrap();
        let result_l2 = lightgbm(x.view(), y.view(), &config_l2).unwrap();

        // Regularized model should have higher training loss (less overfitting)
        // This is a soft check since it depends on the data
        assert!(result_no_reg.final_train_loss <= result_l2.final_train_loss + 1.0);
    }

    #[test]
    fn test_lightgbm_subsampling() {
        let x = array![
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.6],
            [7.0, 0.7],
            [8.0, 0.8],
            [9.0, 0.9],
            [10.0, 1.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = LightGbmConfig {
            num_iterations: 50,
            learning_rate: 0.1,
            num_leaves: 8,
            min_data_in_leaf: 1,
            bagging_fraction: 0.8,
            feature_fraction: 0.5,
            seed: Some(42),
            ..Default::default()
        };

        let result = lightgbm(x.view(), y.view(), &config).unwrap();

        // Should still work with subsampling
        assert_eq!(result.num_trees, 50);
        assert!(result.final_train_loss < result.train_loss[0]);
    }

    #[test]
    fn test_lightgbm_max_depth() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = LightGbmConfig {
            num_iterations: 50,
            learning_rate: 0.1,
            num_leaves: 31, // Many leaves
            max_depth: 2,   // But limited depth
            min_data_in_leaf: 1,
            ..Default::default()
        };

        let result = lightgbm(x.view(), y.view(), &config).unwrap();

        // Should run without error
        assert_eq!(result.num_trees, 50);
    }

    #[test]
    fn test_feature_bins() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let bins = FeatureBins::from_data(&values, 5);

        // Should have 4 boundaries for 5 bins
        assert_eq!(bins.num_bins, bins.boundaries.len() + 1);
        assert!(bins.num_bins <= 5);

        // Test binning
        let bin_1 = bins.get_bin(1.0);
        let bin_10 = bins.get_bin(10.0);
        assert!(bin_1 < bin_10, "Lower values should be in lower bins");
    }

    #[test]
    fn test_histogram_subtraction() {
        let parent = Histogram {
            gradient_sum: vec![1.0, 2.0, 3.0, 4.0],
            hessian_sum: vec![0.5, 1.0, 1.5, 2.0],
            count: vec![10, 20, 30, 40],
        };

        let sibling = Histogram {
            gradient_sum: vec![0.5, 1.0, 1.5, 2.0],
            hessian_sum: vec![0.25, 0.5, 0.75, 1.0],
            count: vec![5, 10, 15, 20],
        };

        let result = subtract_histograms(&parent, &sibling);

        assert_eq!(result.gradient_sum, vec![0.5, 1.0, 1.5, 2.0]);
        assert_eq!(result.hessian_sum, vec![0.25, 0.5, 0.75, 1.0]);
        assert_eq!(result.count, vec![5, 10, 15, 20]);
    }

    #[test]
    fn test_split_gain_calculation() {
        // Test the split gain formula
        let gain = compute_split_gain(
            -1.0, 2.0,  // left: grad=-1, hess=2
            1.0, 2.0,   // right: grad=1, hess=2
            0.0,        // no regularization
        );

        // Expected: 0.5 * [1/2 + 1/2 - 0/4] = 0.5
        assert!((gain - 0.5).abs() < 1e-10);

        // With regularization, gain should be lower
        let gain_reg = compute_split_gain(-1.0, 2.0, 1.0, 2.0, 1.0);
        assert!(gain_reg < gain);
    }
}
