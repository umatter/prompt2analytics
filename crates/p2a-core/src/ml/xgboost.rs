//! XGBoost (eXtreme Gradient Boosting) implementation.
//!
//! Pure Rust implementation of XGBoost based on Chen & Guestrin (2016).
//! Uses second-order Taylor approximation for optimization and supports
//! L1 (alpha) and L2 (lambda) regularization on leaf weights.
//!
//! ## Key Differences from GBM
//!
//! - **Second-order optimization**: Uses both gradient (g) and hessian (h) for tree construction
//! - **Regularized objective**: Includes gamma (split penalty) and lambda (L2 weight penalty)
//! - **Optimal leaf weights**: Analytically computed as w* = -G / (H + lambda)
//! - **Split gain formula**: Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G+H)^2/(H+lambda)] - gamma,
//!   with G replaced by ThresholdL1(G, alpha) when alpha > 0 (XGBoost CalcGain)
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{xgboost, XGBoostConfig, XGBoostObjective};
//! use ndarray::array;
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let y = array![1.1, 1.9, 3.2, 3.8, 5.1];
//!
//! let config = XGBoostConfig {
//!     n_estimators: 100,
//!     learning_rate: 0.3,
//!     max_depth: 6,
//!     lambda: 1.0,  // L2 regularization
//!     gamma: 0.0,   // min split loss
//!     ..Default::default()
//! };
//!
//! let result = xgboost(x.view(), y.view(), &config).unwrap();
//! println!("Training RMSE: {:.4}", result.final_train_rmse);
//! ```
//!
//! ## References
//!
//! - Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System".
//!   Proceedings of the 22nd ACM SIGKDD, 785-794. <https://doi.org/10.1145/2939672.2939785>
//! - XGBoost Documentation: <https://xgboost.readthedocs.io/en/stable/tutorials/model.html>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::trees::MAX_TREE_DEPTH;
use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::resource;

/// Objective function for XGBoost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum XGBoostObjective {
    /// Squared error for regression (default)
    #[default]
    RegSquaredError,
    /// Logistic loss for binary classification
    BinaryLogistic,
    /// Squared error for classification probabilities
    RegLogistic,
}

impl std::str::FromStr for XGBoostObjective {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace([':', '_', '-'], "").as_str() {
            "regsquarederror" | "squarederror" | "mse" | "regression" => {
                Ok(XGBoostObjective::RegSquaredError)
            }
            "binarylogistic" | "logistic" | "binary" | "classification" => {
                Ok(XGBoostObjective::BinaryLogistic)
            }
            "reglogistic" => Ok(XGBoostObjective::RegLogistic),
            _ => Err(format!("Unknown XGBoost objective: {}", s)),
        }
    }
}

impl std::fmt::Display for XGBoostObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XGBoostObjective::RegSquaredError => write!(f, "reg:squarederror"),
            XGBoostObjective::BinaryLogistic => write!(f, "binary:logistic"),
            XGBoostObjective::RegLogistic => write!(f, "reg:logistic"),
        }
    }
}

/// Configuration for XGBoost.
///
/// # Parameters
///
/// The key regularization parameters follow XGBoost's naming conventions:
/// - `lambda` (L2 regularization on leaf weights): Higher values make the model more conservative
/// - `alpha` (L1 regularization on leaf weights): Can produce sparse leaf weights
/// - `gamma` (minimum loss reduction for splits): Acts as pruning threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostConfig {
    /// Number of boosting rounds (trees)
    pub n_estimators: usize,
    /// Learning rate (shrinkage factor, eta in original paper)
    pub learning_rate: f64,
    /// Maximum depth of trees (0 = no limit, but NOT recommended); every tree stops at depth 512
    pub max_depth: usize,
    /// L2 regularization on leaf weights (reg_lambda in sklearn API)
    pub lambda: f64,
    /// L1 regularization on leaf weights (reg_alpha in sklearn API)
    pub alpha: f64,
    /// Minimum loss reduction required to make a split (gamma)
    pub gamma: f64,
    /// Minimum sum of instance weight (hessian) in a leaf (min_child_weight)
    pub min_child_weight: f64,
    /// Fraction of samples to use for each tree (subsample)
    pub subsample: f64,
    /// Fraction of columns sampled once per tree (colsample_bytree): ceil(fraction * p), at least 1
    pub colsample_bytree: f64,
    /// Fraction of the tree's columns sampled once per depth level, shared by every node at that
    /// level (XGBoost colsample_bylevel): ceil(fraction * columns), at least 1 (XGBoost uses
    /// floor, at least 1)
    pub colsample_bylevel: f64,
    /// Objective function
    pub objective: XGBoostObjective,
    /// Base score (initial prediction)
    pub base_score: Option<f64>,
    /// Random seed
    pub seed: Option<u64>,
    /// Early stopping rounds (0 = disabled): with `k > 0`, `validation_fraction` of the rows are
    /// held out (a seeded draw), the trees are grown on the rest, and boosting stops after `k`
    /// rounds without an improvement of the held-out loss; the model keeps the trees up to the
    /// best round
    pub early_stopping_rounds: usize,
    /// Share of rows held out for early stopping, in (0, 1): `ceil(n * validation_fraction)`
    /// rows, leaving at least 2 to train on. Read only when `early_stopping_rounds > 0`
    #[serde(default = "default_validation_fraction")]
    pub validation_fraction: f64,
    /// Verbosity (0 = silent, 1 = print metrics)
    pub verbosity: usize,
}

impl Default for XGBoostConfig {
    fn default() -> Self {
        XGBoostConfig {
            n_estimators: 100,
            learning_rate: 0.3, // XGBoost default eta
            max_depth: 6,       // XGBoost default
            lambda: 1.0,        // XGBoost default
            alpha: 0.0,
            gamma: 0.0,
            min_child_weight: 1.0, // XGBoost default
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            objective: XGBoostObjective::RegSquaredError,
            base_score: None,
            seed: None,
            early_stopping_rounds: 0,
            validation_fraction: default_validation_fraction(),
            verbosity: 0,
        }
    }
}

/// Default of [`XGBoostConfig::validation_fraction`].
fn default_validation_fraction() -> f64 {
    0.1
}

/// A node in an XGBoost tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum XGBoostNode {
    /// Leaf node with weight (prediction value)
    Leaf {
        weight: f64,
        /// Cover (sum of hessians)
        cover: f64,
    },
    /// Split node
    Split {
        /// Feature index for split
        feature: usize,
        /// Split threshold
        threshold: f64,
        /// Left child (values <= threshold)
        left: Box<XGBoostNode>,
        /// Right child (values > threshold)
        right: Box<XGBoostNode>,
        /// Gain from this split
        gain: f64,
        /// Cover (sum of hessians)
        cover: f64,
    },
}

/// A single tree in the XGBoost ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostTree {
    root: XGBoostNode,
    n_features: usize,
}

impl XGBoostTree {
    /// Predict for a single sample.
    pub fn predict_one(&self, x: &ArrayView1<f64>) -> f64 {
        Self::traverse(&self.root, x)
    }

    fn traverse(node: &XGBoostNode, x: &ArrayView1<f64>) -> f64 {
        match node {
            XGBoostNode::Leaf { weight, .. } => *weight,
            XGBoostNode::Split {
                feature,
                threshold,
                left,
                right,
                ..
            } => {
                if x[*feature] <= *threshold {
                    Self::traverse(left, x)
                } else {
                    Self::traverse(right, x)
                }
            }
        }
    }

    /// Get feature importances (gain-based).
    pub fn feature_importances(&self, gains: &mut HashMap<usize, f64>) {
        Self::collect_gains(&self.root, gains);
    }

    fn collect_gains(node: &XGBoostNode, gains: &mut HashMap<usize, f64>) {
        if let XGBoostNode::Split {
            feature,
            gain,
            left,
            right,
            ..
        } = node
        {
            *gains.entry(*feature).or_insert(0.0) += *gain;
            Self::collect_gains(left, gains);
            Self::collect_gains(right, gains);
        }
    }
}

/// Result from XGBoost fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostResult {
    /// Number of trees fitted
    pub n_estimators: usize,
    /// Feature importances (gain-based, normalized)
    pub feature_importances: Vec<f64>,
    /// Training loss at each iteration
    pub train_loss: Vec<f64>,
    /// Final training RMSE (for regression) or log-loss (for classification)
    pub final_train_rmse: f64,
    /// Base score used for initial prediction
    pub base_score: f64,
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Configuration used
    pub config: XGBoostConfig,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// With early stopping: the number of trees kept, the round (1-based, as R xgboost) with the
    /// lowest held-out loss, so at least 1 (a fit whose held-out loss is never finite is
    /// refused); `None` without early stopping
    pub best_iteration: Option<usize>,
    /// Rows (ascending) held out for early stopping
    #[serde(default)]
    pub validation_rows: Option<Vec<usize>>,
    /// Held-out loss after each round that ran (early stopping only; empty otherwise)
    #[serde(default)]
    pub validation_loss: Vec<f64>,
    /// Internal: trees (not serialized for large models)
    #[serde(skip)]
    pub(crate) trees: Vec<XGBoostTree>,
}

impl std::fmt::Display for XGBoostResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "XGBoost Results")?;
        writeln!(f, "===============")?;
        writeln!(f, "Objective: {}", self.config.objective)?;
        writeln!(f, "Number of estimators: {}", self.n_estimators)?;
        writeln!(f, "Learning rate: {:.4}", self.config.learning_rate)?;
        writeln!(f, "Max depth: {}", self.config.max_depth)?;
        writeln!(f, "Lambda (L2): {:.4}", self.config.lambda)?;
        writeln!(f, "Alpha (L1): {:.4}", self.config.alpha)?;
        writeln!(f, "Gamma (min split gain): {:.4}", self.config.gamma)?;

        if self.config.subsample < 1.0 {
            writeln!(f, "Subsample: {:.2}", self.config.subsample)?;
        }
        if self.config.colsample_bylevel < 1.0 {
            writeln!(
                f,
                "Colsample by level: {:.2}",
                self.config.colsample_bylevel
            )?;
        }
        if self.config.colsample_bytree < 1.0 {
            writeln!(f, "Colsample by tree: {:.2}", self.config.colsample_bytree)?;
        }

        writeln!(f)?;
        match self.config.objective {
            XGBoostObjective::RegSquaredError => {
                writeln!(f, "Final training RMSE: {:.6}", self.final_train_rmse)?;
            }
            XGBoostObjective::BinaryLogistic | XGBoostObjective::RegLogistic => {
                writeln!(f, "Final training log-loss: {:.6}", self.final_train_rmse)?;
            }
        }

        if let Some(best_iter) = self.best_iteration {
            let held_out = self.validation_rows.as_ref().map_or(0, Vec::len);
            writeln!(
                f,
                "Best iteration (early stopping on {held_out} held-out rows): {best_iter} of {} rounds",
                self.validation_loss.len()
            )?;
        }

        if self.train_loss.len() > 1 {
            writeln!(f, "Initial loss: {:.6}", self.train_loss[0])?;
            writeln!(
                f,
                "Loss reduction: {:.2}%",
                (1.0 - self.final_train_rmse / self.train_loss[0]) * 100.0
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Feature Importances (gain):")?;

        let mut indexed: Vec<(usize, f64)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names.get(*i).cloned().unwrap_or_else(|| format!("f{}", i)),
                None => format!("f{}", i),
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

/// Gradient and hessian for squared error loss.
/// Loss = 0.5 * (y - pred)^2
/// g = pred - y (negative gradient)
/// h = 1
#[inline]
fn squared_error_gh(y: f64, pred: f64) -> (f64, f64) {
    (pred - y, 1.0)
}

/// Gradient and hessian for logistic loss.
/// Loss = -y*log(p) - (1-y)*log(1-p)
/// g = p - y
/// h = p * (1 - p)
#[inline]
fn logistic_gh(y: f64, pred: f64) -> (f64, f64) {
    let p = sigmoid(pred);
    let g = p - y;
    let h = (p * (1.0 - p)).max(1e-16); // Avoid zero hessian
    (g, h)
}

/// Sigmoid function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Compute gradient and hessian for all samples.
fn compute_gradients(
    y: &ArrayView1<f64>,
    predictions: &Array1<f64>,
    objective: XGBoostObjective,
) -> (Array1<f64>, Array1<f64>) {
    let n = y.len();
    let mut g = Array1::zeros(n);
    let mut h = Array1::zeros(n);

    let gh_fn: fn(f64, f64) -> (f64, f64) = match objective {
        XGBoostObjective::RegSquaredError => squared_error_gh,
        XGBoostObjective::BinaryLogistic | XGBoostObjective::RegLogistic => logistic_gh,
    };

    for i in 0..n {
        let (gi, hi) = gh_fn(y[i], predictions[i]);
        g[i] = gi;
        h[i] = hi;
    }

    (g, h)
}

/// Compute training loss.
fn compute_loss(
    y: &ArrayView1<f64>,
    predictions: &Array1<f64>,
    objective: XGBoostObjective,
) -> f64 {
    let n = y.len() as f64;

    match objective {
        XGBoostObjective::RegSquaredError => {
            // RMSE
            let mse: f64 = y
                .iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| (yi - pi).powi(2))
                .sum::<f64>()
                / n;
            mse.sqrt()
        }
        XGBoostObjective::BinaryLogistic | XGBoostObjective::RegLogistic => {
            // Log-loss
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

/// Compute initial prediction (base score).
fn compute_base_score(y: &ArrayView1<f64>, objective: XGBoostObjective) -> f64 {
    match objective {
        XGBoostObjective::RegSquaredError => {
            // Mean of y
            y.iter().sum::<f64>() / y.len() as f64
        }
        XGBoostObjective::BinaryLogistic | XGBoostObjective::RegLogistic => {
            // Log-odds of mean probability
            let p = y.iter().sum::<f64>() / y.len() as f64;
            let p = p.clamp(1e-10, 1.0 - 1e-10);
            (p / (1.0 - p)).ln()
        }
    }
}

/// Refuse a non-finite cell of the features or the target, naming it. Shared by the tree
/// boosters of this module family, whose sorts need NaN-free data.
pub(super) fn check_finite_xy(x: ArrayView2<f64>, y: ArrayView1<f64>) -> EconResult<()> {
    if let Some(((i, j), v)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("x must contain only finite values (row {i}, column {j} is {v})"),
        });
    }
    if let Some((i, v)) = y.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("y must contain only finite values (row {i} is {v})"),
        });
    }
    Ok(())
}

/// XGBoost's ThresholdL1: the sum of gradients soft-thresholded by alpha.
#[inline]
fn threshold_l1(g: f64, alpha: f64) -> f64 {
    if g > alpha {
        g - alpha
    } else if g < -alpha {
        g + alpha
    } else {
        0.0
    }
}

/// Optimal leaf weight with L1 and L2 regularization.
/// w* = -ThresholdL1(G, alpha) / (H + lambda)
/// Reference: Chen & Guestrin (2016), Equation 5; XGBoost CalcWeight
#[inline]
fn optimal_weight(g_sum: f64, h_sum: f64, lambda: f64, alpha: f64) -> f64 {
    if h_sum < 1e-10 {
        return 0.0;
    }
    -threshold_l1(g_sum, alpha) / (h_sum + lambda)
}

/// Compute split gain on the thresholded sums, T = ThresholdL1(., alpha).
/// Gain = 0.5 * [T(G_L)^2/(H_L+lambda) + T(G_R)^2/(H_R+lambda) - T(G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
/// Reference: Chen & Guestrin (2016), Equation 7; XGBoost CalcGain
#[inline]
fn split_gain(
    g_left: f64,
    h_left: f64,
    g_right: f64,
    h_right: f64,
    lambda: f64,
    alpha: f64,
    gamma: f64,
) -> f64 {
    let score = |g: f64, h: f64| {
        if h > 0.0 {
            let t = threshold_l1(g, alpha);
            t * t / (h + lambda)
        } else {
            0.0
        }
    };
    0.5 * (score(g_left, h_left) + score(g_right, h_right)
        - score(g_left + g_right, h_left + h_right))
        - gamma
}

/// `split_gain` at alpha = 0, where ThresholdL1 is the identity on finite sums: the same
/// arithmetic without the three soft-thresholds per candidate.
#[inline]
fn split_gain_l2(
    g_left: f64,
    h_left: f64,
    g_right: f64,
    h_right: f64,
    lambda: f64,
    gamma: f64,
) -> f64 {
    let score = |g: f64, h: f64| if h > 0.0 { g * g / (h + lambda) } else { 0.0 };
    0.5 * (score(g_left, h_left) + score(g_right, h_right)
        - score(g_left + g_right, h_left + h_right))
        - gamma
}

/// `f64::total_cmp`'s order as an integer key: sorting by it is sorting by `total_cmp`, one
/// integer comparison per step instead of two bit transforms.
#[inline]
fn total_order_key(v: f64) -> i64 {
    let b = v.to_bits() as i64;
    b ^ (((b >> 63) as u64) >> 1) as i64
}

/// Inverse of `total_order_key` (the transform is its own inverse).
#[inline]
fn from_total_order_key(k: i64) -> f64 {
    f64::from_bits((k ^ (((k >> 63) as u64) >> 1) as i64) as u64)
}

/// Build an XGBoost tree using exact greedy algorithm.
fn build_tree(
    x: &ArrayView2<f64>,
    g: &ArrayView1<f64>,
    h: &ArrayView1<f64>,
    indices: &[usize],
    feature_indices: &[usize],
    levels: &mut LevelColumns,
    rng_state: &mut u64,
    config: &XGBoostConfig,
    depth: usize,
) -> XGBoostNode {
    let n = indices.len();

    // Compute sums for this node
    let g_sum: f64 = indices.iter().map(|&i| g[i]).sum();
    let h_sum: f64 = indices.iter().map(|&i| h[i]).sum();

    // Check stopping conditions
    let max_depth = if config.max_depth == 0 {
        MAX_TREE_DEPTH
    } else {
        config.max_depth.min(MAX_TREE_DEPTH)
    };

    if depth >= max_depth || n <= 1 || h_sum < config.min_child_weight {
        let weight = optimal_weight(g_sum, h_sum, config.lambda, config.alpha);
        return XGBoostNode::Leaf {
            weight,
            cover: h_sum,
        };
    }

    // Find best split using exact greedy algorithm
    // Reference: Algorithm 1 in Chen & Guestrin (2016)
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_split: Option<(usize, f64)> = None;
    let mut best_left_indices = Vec::new();
    let mut best_right_indices = Vec::new();
    let l1 = config.alpha != 0.0;

    for &feature in levels.columns(depth, feature_indices, rng_state) {
        // Sort indices by feature value (stable, total_cmp order, via integer keys)
        let mut keyed: Vec<(i64, usize)> = indices
            .iter()
            .map(|&i| (total_order_key(x[[i, feature]]), i))
            .collect();
        keyed.sort_by_key(|p| p.0);
        let sorted: Vec<(f64, usize)> = keyed
            .into_iter()
            .map(|(k, i)| (from_total_order_key(k), i))
            .collect();

        // Scan with incremental sums - O(n) after sorting
        let mut g_left = 0.0;
        let mut h_left = 0.0;

        for i in 0..sorted.len() - 1 {
            let (x_val, idx) = sorted[i];
            g_left += g[idx];
            h_left += h[idx];

            // Skip if next value is the same (no valid split)
            let next_x = sorted[i + 1].0;
            if (next_x - x_val).abs() < 1e-10 {
                continue;
            }

            let g_right = g_sum - g_left;
            let h_right = h_sum - h_left;

            // Check min_child_weight constraint
            if h_left < config.min_child_weight || h_right < config.min_child_weight {
                continue;
            }

            let gain = if l1 {
                split_gain(
                    g_left,
                    h_left,
                    g_right,
                    h_right,
                    config.lambda,
                    config.alpha,
                    config.gamma,
                )
            } else {
                split_gain_l2(
                    g_left,
                    h_left,
                    g_right,
                    h_right,
                    config.lambda,
                    config.gamma,
                )
            };

            if gain > best_gain {
                best_gain = gain;
                let threshold = (x_val + next_x) / 2.0;
                best_split = Some((feature, threshold));

                // Store partition
                best_left_indices = sorted[..=i].iter().map(|(_, idx)| *idx).collect();
                best_right_indices = sorted[i + 1..].iter().map(|(_, idx)| *idx).collect();
            }
        }
    }

    // If no good split found (gain <= 0), return leaf
    if best_gain <= 0.0 || best_split.is_none() {
        let weight = optimal_weight(g_sum, h_sum, config.lambda, config.alpha);
        return XGBoostNode::Leaf {
            weight,
            cover: h_sum,
        };
    }

    let (feature, threshold) = best_split.unwrap();

    // Build children recursively
    let left = build_tree(
        x,
        g,
        h,
        &best_left_indices,
        feature_indices,
        levels,
        rng_state,
        config,
        depth + 1,
    );
    let right = build_tree(
        x,
        g,
        h,
        &best_right_indices,
        feature_indices,
        levels,
        rng_state,
        config,
        depth + 1,
    );

    XGBoostNode::Split {
        feature,
        threshold,
        left: Box::new(left),
        right: Box::new(right),
        gain: best_gain,
        cover: h_sum,
    }
}

use super::lcg_random;

/// Select a random subset of indices.
fn random_subsample(n: usize, fraction: f64, rng_state: &mut u64) -> Vec<usize> {
    if fraction >= 1.0 {
        return (0..n).collect();
    }

    let n_select = ((n as f64) * fraction).ceil() as usize;
    let mut indices: Vec<usize> = (0..n).collect();

    // Fisher-Yates shuffle and take first n_select
    for i in 0..n_select.min(n) {
        let j = i + lcg_random(rng_state) % (n - i);
        indices.swap(i, j);
    }

    indices.truncate(n_select);
    indices
}

/// A random subset of `base` of ceil(fraction * len) entries, at least 1 (all of it, with no
/// draw, when fraction >= 1).
fn random_subset(base: &[usize], fraction: f64, rng_state: &mut u64) -> Vec<usize> {
    if fraction >= 1.0 {
        return base.to_vec();
    }

    let n = base.len();
    let n_select = (((n as f64) * fraction).ceil() as usize).clamp(1, n);
    let mut indices = base.to_vec();

    for i in 0..n_select {
        let j = i + lcg_random(rng_state) % (n - i);
        indices.swap(i, j);
    }

    indices.truncate(n_select);
    indices
}

/// Select a random subset of feature indices.
fn random_features(n_features: usize, fraction: f64, rng_state: &mut u64) -> Vec<usize> {
    random_subset(&(0..n_features).collect::<Vec<_>>(), fraction, rng_state)
}

/// The columns of one tree's depth levels: each level's subset is drawn from the tree's columns
/// the first time a node at that depth is split, and reused by every other node at that depth.
struct LevelColumns {
    fraction: f64,
    sets: Vec<Vec<usize>>,
}

impl LevelColumns {
    fn new(fraction: f64) -> Self {
        Self {
            fraction,
            sets: Vec::new(),
        }
    }

    fn columns<'a>(
        &'a mut self,
        depth: usize,
        tree_columns: &'a [usize],
        rng_state: &mut u64,
    ) -> &'a [usize] {
        if self.fraction >= 1.0 {
            return tree_columns;
        }
        while self.sets.len() <= depth {
            self.sets
                .push(random_subset(tree_columns, self.fraction, rng_state));
        }
        &self.sets[depth]
    }
}

/// Fit an XGBoost model.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - XGBoost configuration
///
/// # Returns
///
/// XGBoostResult containing fitted model and diagnostics.
///
/// # References
///
/// Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System".
/// Proceedings of the 22nd ACM SIGKDD, 785-794.
pub fn xgboost(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &XGBoostConfig,
) -> EconResult<XGBoostResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for XGBoost".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }

    if !(config.colsample_bylevel > 0.0 && config.colsample_bylevel <= 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "colsample_bylevel must be in (0, 1] (got {})",
                config.colsample_bylevel
            ),
        });
    }
    if !config.alpha.is_finite() || config.alpha < 0.0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "alpha must be finite and non-negative (got {})",
                config.alpha
            ),
        });
    }
    check_finite_xy(x, y)?;

    // Validate binary targets for classification
    if config.objective == XGBoostObjective::BinaryLogistic {
        for &yi in y.iter() {
            if yi != 0.0 && yi != 1.0 {
                return Err(EconError::Computation(
                    "binary:logistic requires binary targets (0 or 1)".to_string(),
                ));
            }
        }
    }

    if config.early_stopping_rounds > 0 {
        return xgboost_early_stopping(x, y, config);
    }

    let fit = boost(x, y, config, None)?;
    let importances = normalized_gains(fit.feature_gains, n_features);

    // Convert predictions to probabilities for classification
    let final_predictions = match config.objective {
        XGBoostObjective::BinaryLogistic | XGBoostObjective::RegLogistic => {
            fit.predictions.mapv(sigmoid).to_vec()
        }
        XGBoostObjective::RegSquaredError => fit.predictions.to_vec(),
    };

    Ok(XGBoostResult {
        n_estimators: fit.trees.len(),
        feature_importances: importances,
        final_train_rmse: *fit.train_loss.last().unwrap_or(&0.0),
        train_loss: fit.train_loss,
        base_score: fit.base_score,
        predictions: final_predictions,
        config: config.clone(),
        feature_names: None,
        best_iteration: None,
        validation_rows: None,
        validation_loss: Vec::new(),
        trees: fit.trees,
    })
}

/// Mixed into the seed of the held-out draw, so that the training rounds draw exactly what a fit
/// on the training rows alone would draw.
const VALIDATION_STREAM: u64 = 0x9E37_79B9_7F4A_7C15;

/// [`xgboost`] with early stopping: hold out `ceil(n * validation_fraction)` rows (a seeded
/// Fisher-Yates draw), boost on the rest while watching the held-out loss, and keep the trees up
/// to the best round. The training predictions cover every row.
fn xgboost_early_stopping(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &XGBoostConfig,
) -> EconResult<XGBoostResult> {
    let n = x.nrows();
    let f = config.validation_fraction;
    if !(f > 0.0 && f < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "validation_fraction must be in (0, 1) when early_stopping_rounds > 0 (got {f})"
            ),
        });
    }
    let n_val = (f * n as f64).ceil() as usize;
    if n_val < 1 || n_val + 2 > n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "validation_fraction {f} holds out {n_val} of {n} rows; early stopping needs at \
                 least 1 held-out row and 2 training rows"
            ),
        });
    }
    let mut rows: Vec<usize> = (0..n).collect();
    let mut state = config.seed.unwrap_or(42) ^ VALIDATION_STREAM;
    for i in (1..n).rev() {
        let j = lcg_random(&mut state) % (i + 1);
        rows.swap(i, j);
    }
    let mut validation = rows[..n_val].to_vec();
    let mut train = rows[n_val..].to_vec();
    validation.sort_unstable();
    train.sort_unstable();

    let x_train = x.select(Axis(0), &train);
    let y_train = y.select(Axis(0), &train);
    let x_val = x.select(Axis(0), &validation);
    let y_val = y.select(Axis(0), &validation);
    let mut fit = boost(
        x_train.view(),
        y_train.view(),
        config,
        Some((x_val.view(), y_val.view())),
    )?;

    // best_iteration counts the kept trees and is at least 1: a fit in which no round ran, or
    // no held-out loss was ever finite, has no best round and is refused
    let best = fit.best_round;
    if best == 0 {
        return Err(EconError::Computation(if fit.validation_loss.is_empty() {
            "early stopping needs at least one boosting round (n_estimators >= 1)".to_string()
        } else {
            format!(
                "early stopping: the held-out loss was never finite in {} rounds; rescale y",
                fit.validation_loss.len()
            )
        }));
    }
    fit.trees.truncate(best);
    fit.train_loss.truncate(best + 1);
    let mut gains: HashMap<usize, f64> = HashMap::new();
    for tree in &fit.trees {
        tree.feature_importances(&mut gains);
    }
    let importances = normalized_gains(gains, x.ncols());

    let mut result = XGBoostResult {
        n_estimators: fit.trees.len(),
        feature_importances: importances,
        final_train_rmse: *fit.train_loss.last().unwrap_or(&0.0),
        train_loss: fit.train_loss,
        base_score: fit.base_score,
        predictions: Vec::new(),
        config: config.clone(),
        feature_names: None,
        best_iteration: Some(best),
        validation_rows: Some(validation),
        validation_loss: fit.validation_loss,
        trees: fit.trees,
    };
    result.predictions = xgboost_predict(&result, x)?;
    Ok(result)
}

/// Gain shares per feature (zeros if no tree split).
fn normalized_gains(feature_gains: HashMap<usize, f64>, n_features: usize) -> Vec<f64> {
    let mut importances = vec![0.0; n_features];
    let total_gain: f64 = feature_gains.values().sum();
    if total_gain > 0.0 {
        for (feature, gain) in feature_gains {
            importances[feature] = gain / total_gain;
        }
    }
    importances
}

/// The boosting rounds of one fit.
struct Boosted {
    trees: Vec<XGBoostTree>,
    /// Loss of the base score, then one entry per round
    train_loss: Vec<f64>,
    base_score: f64,
    /// Raw scores of the training rows
    predictions: Array1<f64>,
    feature_gains: HashMap<usize, f64>,
    /// Held-out loss after each round (early stopping only)
    validation_loss: Vec<f64>,
    /// 1-based round with the lowest held-out loss (early stopping only)
    best_round: usize,
}

/// Grow the trees on `x` and `y`. With `validation`, record the held-out loss after each round
/// and stop after `early_stopping_rounds` rounds without an improvement greater than 1e-10.
fn boost(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &XGBoostConfig,
    validation: Option<(ArrayView2<f64>, ArrayView1<f64>)>,
) -> EconResult<Boosted> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut rng_state = config.seed.unwrap_or(42);

    // Initialize predictions with base score
    let base_score = config
        .base_score
        .unwrap_or_else(|| compute_base_score(&y, config.objective));
    let mut predictions = Array1::from_elem(n_samples, base_score);

    let mut trees = resource::try_with_capacity(config.n_estimators, "XGBoost trees")?;
    let mut train_loss = resource::try_with_capacity(
        config.n_estimators.saturating_add(1),
        "XGBoost training loss",
    )?;
    let mut feature_gains: HashMap<usize, f64> = HashMap::new();

    // Initial loss
    train_loss.push(compute_loss(&y, &predictions, config.objective));

    let mut val_predictions = validation.map(|(xv, _)| Array1::from_elem(xv.nrows(), base_score));
    let mut validation_loss = Vec::new();
    let mut best_loss = f64::INFINITY;
    let mut best_round = 0usize;
    let mut rounds_without_improvement = 0;

    // Boosting iterations
    for iter in 0..config.n_estimators {
        resource::checkpoint()?;
        // Compute gradients and hessians
        let (g, h) = compute_gradients(&y, &predictions, config.objective);

        // Row subsampling
        let sample_indices = random_subsample(n_samples, config.subsample, &mut rng_state);

        // Column subsampling (per tree)
        let feature_indices = random_features(n_features, config.colsample_bytree, &mut rng_state);

        // Build tree
        let root = build_tree(
            &x,
            &g.view(),
            &h.view(),
            &sample_indices,
            &feature_indices,
            &mut LevelColumns::new(config.colsample_bylevel),
            &mut rng_state,
            config,
            0,
        );

        let tree = XGBoostTree { root, n_features };

        // Update predictions for all samples
        for i in 0..n_samples {
            let row = x.row(i);
            predictions[i] += config.learning_rate * tree.predict_one(&row);
        }

        // Accumulate feature importance
        tree.feature_importances(&mut feature_gains);

        // Track loss
        let loss = compute_loss(&y, &predictions, config.objective);
        train_loss.push(loss);

        if config.verbosity > 0 && (iter + 1) % 10 == 0 {
            eprintln!("[{}]\ttrain-loss:{:.6}", iter + 1, loss);
        }

        // Early stopping on the held-out rows
        if let (Some((xv, yv)), Some(vp)) = (validation, val_predictions.as_mut()) {
            for i in 0..xv.nrows() {
                vp[i] += config.learning_rate * tree.predict_one(&xv.row(i));
            }
            let val_loss = compute_loss(&yv, vp, config.objective);
            validation_loss.push(val_loss);
            trees.push(tree);
            if val_loss < best_loss - 1e-10 {
                best_loss = val_loss;
                best_round = iter + 1;
                rounds_without_improvement = 0;
            } else {
                rounds_without_improvement += 1;
                if rounds_without_improvement >= config.early_stopping_rounds {
                    if config.verbosity > 0 {
                        eprintln!(
                            "Early stopping at iteration {} (best: {best_round})",
                            iter + 1
                        );
                    }
                    break;
                }
            }
        } else {
            trees.push(tree);
        }
    }

    Ok(Boosted {
        trees,
        train_loss,
        base_score,
        predictions,
        feature_gains,
        validation_loss,
        best_round,
    })
}

/// Predict using a fitted XGBoost model.
///
/// # Arguments
///
/// * `result` - Fitted XGBoost result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions (probabilities for classification, values for regression)
pub fn xgboost_predict(result: &XGBoostResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    let n_samples = x.nrows();

    if result.trees.is_empty() {
        return Err(EconError::Computation(
            "Model has no fitted trees".to_string(),
        ));
    }

    let mut predictions = Array1::from_elem(n_samples, result.base_score);

    for tree in &result.trees {
        for i in 0..n_samples {
            predictions[i] += result.config.learning_rate * tree.predict_one(&x.row(i));
        }
    }

    // Convert to probabilities for classification
    match result.config.objective {
        XGBoostObjective::BinaryLogistic | XGBoostObjective::RegLogistic => {
            Ok(predictions.mapv(sigmoid).to_vec())
        }
        XGBoostObjective::RegSquaredError => Ok(predictions.to_vec()),
    }
}

/// Predict class labels for binary classification.
pub fn xgboost_predict_class(
    result: &XGBoostResult,
    x: ArrayView2<f64>,
    threshold: f64,
) -> EconResult<Vec<i32>> {
    let probs = xgboost_predict(result, x)?;
    Ok(probs
        .iter()
        .map(|&p| if p >= threshold { 1 } else { 0 })
        .collect())
}

/// Run XGBoost on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Numeric target column. Integer and boolean (0/1) columns are accepted;
///   a missing value (null or NaN) in the target or a feature is an error naming the
///   column (`EconError::NullValues`)
/// * `x_cols` - Feature column names
/// * `config` - XGBoost configuration
///
/// # Returns
///
/// XGBoostResult with model and diagnostics
pub fn run_xgboost(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &XGBoostConfig,
) -> EconResult<XGBoostResult> {
    let (x, y_arr, _missing) =
        super::extract::extract_xy(dataset, y_col, x_cols, crate::data::columns::NaPolicy::Fail)?;
    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    let mut result = xgboost(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Convenience function for running XGBoost with default configuration.
pub fn run_xgboost_default(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<XGBoostResult> {
    run_xgboost(dataset, y_col, x_cols, &XGBoostConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::{NaPolicy, numeric_column};
    use ndarray::array;
    use p2a_validation::{Check, Tol};

    fn case_matrix(ds: &crate::data::Dataset, cols: &[&str]) -> Array2<f64> {
        let mut m = Array2::zeros((ds.nrows(), cols.len()));
        for (j, c) in cols.iter().enumerate() {
            let (v, _) = numeric_column(ds.df(), c, NaPolicy::Fail).unwrap();
            m.column_mut(j).assign(&Array1::from_vec(v));
        }
        m
    }
    fn case_vector(ds: &crate::data::Dataset, col: &str) -> Array1<f64> {
        Array1::from_vec(numeric_column(ds.df(), col, NaPolicy::Fail).unwrap().0)
    }

    fn four_signal_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut u = || (lcg_random(&mut s) % 100_000) as f64 / 100_000.0;
        let x = Array2::from_shape_fn((n, 4), |_| u());
        let noise: Vec<f64> = (0..n).map(|_| 0.1 * (u() - 0.5)).collect();
        let y = Array1::from_shape_fn(n, |i| x.row(i).sum() + noise[i]);
        (x, y)
    }
    fn features_by_depth(
        node: &XGBoostNode,
        depth: usize,
        out: &mut Vec<std::collections::BTreeSet<usize>>,
    ) {
        if let XGBoostNode::Split {
            feature,
            left,
            right,
            ..
        } = node
        {
            if out.len() <= depth {
                out.resize_with(depth + 1, Default::default);
            }
            out[depth].insert(*feature);
            features_by_depth(left, depth + 1, out);
            features_by_depth(right, depth + 1, out);
        }
    }
    fn bylevel_fit(fraction: f64) -> XGBoostResult {
        let (x, y) = four_signal_data(200, 13);
        let mut c = XGBoostConfig::default();
        c.n_estimators = 20;
        c.max_depth = 3;
        c.learning_rate = 0.3;
        c.colsample_bylevel = fraction;
        c.seed = Some(7);
        xgboost(x.view(), y.view(), &c).unwrap()
    }

    #[test]
    fn test_colsample_bylevel_one_feature_per_level() {
        // ceil(0.25 * 4) = 1 column per level: every split at one depth of one tree uses the same column
        let r = bylevel_fit(0.25);
        let mut roots = std::collections::BTreeSet::new();
        for t in &r.trees {
            let mut levels = Vec::new();
            features_by_depth(&t.root, 0, &mut levels);
            assert!(levels.iter().all(|s| s.len() <= 1), "{levels:?}");
            roots.extend(levels.first().into_iter().flatten().copied());
        }
        assert!(
            roots.len() >= 2,
            "the level draws vary across trees: {roots:?}"
        );
        // not vacuous: without per-level sampling some depth mixes columns
        let full = bylevel_fit(1.0);
        assert!(full.trees.iter().any(|t| {
            let mut l = Vec::new();
            features_by_depth(&t.root, 0, &mut l);
            l.iter().any(|s| s.len() >= 2)
        }));
    }

    #[test]
    fn test_colsample_bylevel_changes_the_fit() {
        assert_ne!(bylevel_fit(1.0).predictions, bylevel_fit(0.25).predictions);
    }

    #[test]
    fn test_xgboost_rejects_bad_colsample_bylevel() {
        let (x, y) = four_signal_data(20, 1);
        for v in [0.0, -0.5, 1.5, f64::NAN] {
            let mut c = XGBoostConfig::default();
            c.colsample_bylevel = v;
            let e = xgboost(x.view(), y.view(), &c).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.starts_with("colsample_bylevel")),
                "{e}"
            );
        }
    }

    #[test]
    fn test_xgboost_regression_basic() {
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

        let config = XGBoostConfig {
            n_estimators: 50,
            learning_rate: 0.3,
            max_depth: 3,
            lambda: 1.0,
            gamma: 0.0,
            ..Default::default()
        };

        let result = xgboost(x.view(), y.view(), &config).unwrap();

        // Check we got the right number of trees
        assert_eq!(result.n_estimators, 50);

        // Loss should decrease
        assert!(
            result.train_loss.last().unwrap() < result.train_loss.first().unwrap(),
            "Loss did not decrease: {:?}",
            result.train_loss
        );

        // RMSE should be reasonable
        assert!(
            result.final_train_rmse < 1.0,
            "RMSE {} should be < 1.0",
            result.final_train_rmse
        );
    }

    #[test]
    fn test_xgboost_binary_classification() {
        // Binary classification - two separable groups
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

        let config = XGBoostConfig {
            n_estimators: 50,
            learning_rate: 0.3,
            max_depth: 3,
            objective: XGBoostObjective::BinaryLogistic,
            ..Default::default()
        };

        let result = xgboost(x.view(), y.view(), &config).unwrap();

        // Predictions should be probabilities
        for &p in &result.predictions {
            assert!((0.0..=1.0).contains(&p), "Probability {} out of range", p);
        }

        // Low x values should have low probability, high values should have high probability
        assert!(result.predictions[0] < 0.5, "Sample 0 should be class 0");
        assert!(result.predictions[7] > 0.5, "Sample 7 should be class 1");
    }

    #[test]
    fn test_xgboost_regularization() {
        // y = x = 1..10, base 5.5, one depth-1 round, eta 0.3. The best split is x <= 5.5 for every
        // setting below (score k(10 - k)/2 per side); G_L = 12.5 over 5 rows, G_R = -12.5.
        let x = Array2::from_shape_fn((10, 1), |(i, _)| (i + 1) as f64);
        let y = Array1::from_shape_fn(10, |i| (i + 1) as f64);
        for (lambda, alpha, gamma, w) in [
            (0.0, 0.0, 0.0, 2.5),          // 12.5 / 5
            (10.0, 0.0, 1.0, 12.5 / 15.0), // 12.5 / (5 + 10); gain 0.5 * 20.83 - 1 > 0
            (0.0, 10.0, 0.0, 0.5),         // ThresholdL1(12.5, 10) / 5
        ] {
            let mut c = XGBoostConfig::default();
            c.n_estimators = 1;
            c.max_depth = 1;
            c.learning_rate = 0.3;
            c.lambda = lambda;
            c.alpha = alpha;
            c.gamma = gamma;
            c.min_child_weight = 0.0;
            let r = xgboost(x.view(), y.view(), &c).unwrap();
            for i in 0..10 {
                let want = if i < 5 { 5.5 - 0.3 * w } else { 5.5 + 0.3 * w };
                assert!(
                    (r.predictions[i] - want).abs() < 1e-12,
                    "lambda {lambda} alpha {alpha}: {} vs {want}",
                    r.predictions[i]
                );
            }
        }
    }

    #[test]
    fn test_validate_xgboost_exact_l1() {
        let (case, ds) = crate::test_support::load_case("ml_xgboost_l1");
        let x = case_matrix(&ds, &["x1", "x2"]);
        let y = case_vector(&ds, "y");
        let mut chk = Check::new(&case);
        for (alpha, pred, root) in [
            (0.0, "exact_a0_pred", "exact_a0_root"),
            (3.0, "exact_a3_pred", "exact_a3_root"),
        ] {
            let mut c = XGBoostConfig::default();
            c.n_estimators = 3;
            c.max_depth = 2;
            c.learning_rate = 0.3;
            c.lambda = 1.0;
            c.alpha = alpha;
            c.gamma = 0.0;
            c.min_child_weight = 1.0;
            let r = xgboost(x.view(), y.view(), &c).unwrap();
            let XGBoostNode::Split {
                feature,
                threshold,
                gain,
                ..
            } = &r.trees[0].root
            else {
                panic!("root is a leaf")
            };
            chk.vector(pred, &r.predictions, Tol::coef(16)).vector(
                root,
                &[*feature as f64, *threshold, *gain],
                Tol::coef(16),
            );
        }
        chk.finish();
    }

    /// The xgboost package itself, in single precision, hence the absolute tolerance.
    #[test]
    fn test_validate_xgboost_exact_package() {
        let (case, ds) = crate::test_support::load_case("ml_xgboost_pkg");
        let x = case_matrix(&ds, &["x1", "x2"]);
        let y = case_vector(&ds, "y");
        let single_precision = Tol::new(0.0, 1e-6);
        let mut chk = Check::new(&case);
        for (alpha, pred, root) in [(0.0, "a0_pred", "a0_root"), (3.0, "a3_pred", "a3_root")] {
            let mut c = XGBoostConfig::default();
            c.n_estimators = 3;
            c.max_depth = 2;
            c.learning_rate = 0.3;
            c.lambda = 1.0;
            c.alpha = alpha;
            c.gamma = 0.0;
            c.min_child_weight = 1.0;
            let r = xgboost(x.view(), y.view(), &c).unwrap();
            let XGBoostNode::Split {
                feature,
                threshold,
                gain,
                ..
            } = &r.trees[0].root
            else {
                panic!("root is a leaf")
            };
            chk.vector(pred, &r.predictions, single_precision).vector(
                root,
                &[*feature as f64, *threshold, *gain],
                single_precision,
            );
        }
        chk.finish();
    }

    #[test]
    fn test_split_gain_on_thresholded_sums() {
        // 0.5 * [T(GL)^2/(HL+l) + T(GR)^2/(HR+l) - T(G)^2/(H+l)] - gamma, T = ThresholdL1(., alpha)
        // The terms are near 21 (half an ulp is 1.8e-15 each), so a gain of 4/3 carries about
        // 1e-15 of rounding (measured 1.1e-15): 1e-14 is a few ulps of the terms.
        assert!((split_gain(-8.0, 2.0, -2.0, 2.0, 1.0, 0.0, 0.0) - 4.0 / 3.0).abs() < 1e-14);
        assert!((split_gain(-8.0, 2.0, -2.0, 2.0, 1.0, 1.0, 0.0) - 7.0 / 30.0).abs() < 1e-14);
        assert!((split_gain(-5.0, 2.0, -5.0, 2.0, 1.0, 0.0, 0.0) - (-5.0 / 3.0)).abs() < 1e-14);
        assert!(
            (split_gain(-8.0, 2.0, -2.0, 2.0, 1.0, 0.0, 0.5) - (4.0 / 3.0 - 0.5)).abs() < 1e-14
        );
    }

    #[test]
    fn split_gain_l2_is_split_gain_at_alpha_zero_bitwise() {
        let gs = [
            0.0, -0.0, 1.0, -1.0, 3.7, -8.25, 1e-300, -1e-300, 5e-324, 1e150, -1e150, 0.1, -0.3,
        ];
        let hs = [0.0, 1e-12, 0.5, 2.0, 17.3, 1e200];
        for &gl in &gs {
            for &gr in &gs {
                for &hl in &hs {
                    for &hr in &hs {
                        for (lambda, gamma) in [(0.0, 0.0), (1.0, 0.0), (0.7, 0.25)] {
                            let a = split_gain(gl, hl, gr, hr, lambda, 0.0, gamma);
                            let b = split_gain_l2(gl, hl, gr, hr, lambda, gamma);
                            assert!(
                                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                                "{gl} {hl} {gr} {hr} {lambda} {gamma}: {a} vs {b}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn total_order_key_sorts_as_total_cmp() {
        let vals = [
            3.0,
            -0.0,
            0.0,
            -1.5,
            f64::INFINITY,
            5e-324,
            -5e-324,
            f64::NEG_INFINITY,
            0.0,
            2.0,
            -0.0,
            -1.5,
            f64::MAX,
            f64::MIN,
            1e-310,
        ];
        let mut by_cmp: Vec<(f64, usize)> = vals.iter().copied().zip(0..).collect();
        by_cmp.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut keyed: Vec<(i64, usize)> =
            vals.iter().map(|&v| total_order_key(v)).zip(0..).collect();
        keyed.sort_by_key(|p| p.0);
        let by_key: Vec<(f64, usize)> = keyed
            .into_iter()
            .map(|(k, i)| (from_total_order_key(k), i))
            .collect();
        assert_eq!(by_cmp.len(), by_key.len());
        for (a, b) in by_cmp.iter().zip(&by_key) {
            assert_eq!((a.0.to_bits(), a.1), (b.0.to_bits(), b.1));
        }
    }

    #[test]
    fn test_xgboost_predict() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = XGBoostConfig {
            n_estimators: 50,
            learning_rate: 0.3,
            max_depth: 3,
            ..Default::default()
        };

        let result = xgboost(x_train.view(), y_train.view(), &config).unwrap();

        // Predict on new data
        let x_test = array![[1.5], [3.5], [5.5]];
        let predictions = xgboost_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        // Predictions should be in reasonable range
        for &p in &predictions {
            assert!(p > 0.0 && p < 7.0, "Prediction {} out of expected range", p);
        }
    }

    #[test]
    fn test_xgboost_feature_importance() {
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

        let config = XGBoostConfig {
            n_estimators: 50,
            learning_rate: 0.3,
            max_depth: 3,
            ..Default::default()
        };

        let result = xgboost(x.view(), y.view(), &config).unwrap();

        // First feature should be more important
        assert!(
            result.feature_importances[0] > result.feature_importances[1],
            "Feature 0 importance {} should be > Feature 1 importance {}",
            result.feature_importances[0],
            result.feature_importances[1]
        );
    }

    #[test]
    fn test_xgboost_subsampling() {
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

        let config = XGBoostConfig {
            n_estimators: 50,
            learning_rate: 0.3,
            max_depth: 3,
            subsample: 0.5,
            colsample_bytree: 0.5,
            seed: Some(42),
            ..Default::default()
        };

        let result = xgboost(x.view(), y.view(), &config).unwrap();

        // Should still work with subsampling
        assert_eq!(result.n_estimators, 50);
        assert!(
            result.train_loss.last().unwrap() < result.train_loss.first().unwrap(),
            "Loss should decrease with subsampling"
        );
    }

    #[test]
    fn test_xgboost_early_stopping() {
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

        let config = XGBoostConfig {
            n_estimators: 1000, // Large number
            learning_rate: 0.3,
            max_depth: 6,
            early_stopping_rounds: 10,
            ..Default::default()
        };

        let result = xgboost(x.view(), y.view(), &config).unwrap();

        // Should stop early (far fewer than 1000 trees)
        assert!(
            result.n_estimators < 1000,
            "Early stopping should trigger before 1000 iterations, got {}",
            result.n_estimators
        );
    }

    #[test]
    fn test_optimal_weight() {
        // Test the optimal weight calculation
        // w* = -G / (H + lambda)
        let w = optimal_weight(-10.0, 5.0, 1.0, 0.0);
        // w = -(-10) / (5 + 1) = 10/6 = 1.6667
        assert!((w - 10.0 / 6.0).abs() < 1e-10);

        // With L1 regularization (alpha = 5), g = -10 becomes -10 + 5 = -5
        let w_l1 = optimal_weight(-10.0, 5.0, 1.0, 5.0);
        // w = -(-10 + 5) / (5 + 1) = 5/6 = 0.8333
        assert!((w_l1 - 5.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_gain() {
        // Test the split gain calculation
        // Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G+H)^2/(H+lambda)] - gamma
        let gain = split_gain(-5.0, 2.0, -5.0, 2.0, 1.0, 0.0, 0.0);
        // Left: (-5)^2 / (2+1) = 25/3
        // Right: (-5)^2 / (2+1) = 25/3
        // Parent: (-10)^2 / (4+1) = 100/5 = 20
        // Gain = 0.5 * (25/3 + 25/3 - 20) - 0 = 0.5 * (50/3 - 20) = 0.5 * (-10/3) < 0

        // A good split should have positive gain
        let good_gain = split_gain(-8.0, 2.0, -2.0, 2.0, 1.0, 0.0, 0.0);
        // This represents a split where one side has much stronger gradient
        // Should have positive gain
        assert!(good_gain > 0.0, "Good split should have positive gain");
    }

    #[test]
    fn test_xgboost_display() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = XGBoostConfig {
            n_estimators: 10,
            ..Default::default()
        };

        let result = xgboost(x.view(), y.view(), &config).unwrap();
        let display = format!("{}", result);

        assert!(display.contains("XGBoost Results"));
        assert!(display.contains("Number of estimators: 10"));
        assert!(display.contains("Lambda (L2)"));
    }

    /// n rows, `y = x0 + N(0, 1)`-like noise from an LCG, three features.
    fn early_stop_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let mut s = 20_261_115u64;
        let mut u = || (crate::ml::lcg_random(&mut s) % 1_000_000) as f64 / 1_000_000.0;
        let x = Array2::from_shape_fn((n, 3), |_| u() * 4.0);
        let y = Array1::from_shape_fn(n, |i| {
            // sum of four uniforms: mean 2, variance 1/3; scaled to unit variance
            let e = (u() + u() + u() + u() - 2.0) * 3f64.sqrt();
            x[[i, 0]] + e
        });
        (x, y)
    }

    #[test]
    fn test_xgboost_early_stopping_keeps_the_best_trees() {
        // ML-143. The model is cut back to the best round, and best_iteration is the number of
        // trees kept (1-based, as R xgboost). Before: every tree stayed, and best_iteration was the
        // 0-based round of the lowest training loss.
        let (x, y) = early_stop_data(200);
        let mut c = XGBoostConfig::default();
        c.n_estimators = 1000;
        c.early_stopping_rounds = 10;
        c.subsample = 0.8;
        c.seed = Some(3);
        let r = xgboost(x.view(), y.view(), &c).unwrap();
        let best = r.best_iteration.expect("early stopping ran");
        assert_eq!(r.n_estimators, best);
        assert_eq!(r.trees.len(), best);
        let again = xgboost_predict(&r, x.view()).unwrap();
        assert_eq!(again, r.predictions);
    }

    #[test]
    fn test_xgboost_early_stopping_refuses_a_never_finite_held_out_loss() {
        // Finite y whose squared errors overflow: the held-out RMSE is +inf from the first round,
        // so no round improves on it. Before: zero trees were kept, best_iteration was Some(0)
        // and the final prediction failed with "Model has no fitted trees".
        let (x, y) = early_stop_data(60);
        let y = y.mapv(|v| (v + 10.0) * 1e200);
        let mut c = XGBoostConfig::default();
        c.n_estimators = 20;
        c.early_stopping_rounds = 3;
        c.seed = Some(3);
        let e = xgboost(x.view(), y.view(), &c).unwrap_err();
        let m = e.to_string();
        assert!(m.contains("held-out loss was never finite"), "{m}");
        // a fit with no rounds at all is refused the same way, not with a prediction error
        c.n_estimators = 0;
        let (x, y) = early_stop_data(60);
        let m = xgboost(x.view(), y.view(), &c).unwrap_err().to_string();
        assert!(m.contains("early stopping"), "{m}");
    }

    #[test]
    fn test_xgboost_early_stopping_watches_held_out_rows() {
        // ML-143. Early stopping watches held-out rows, and the model is the plain fit on the
        // training rows cut at the best round. Before: it watched the training loss.
        let (x, y) = early_stop_data(200);
        let mut c = XGBoostConfig::default();
        c.n_estimators = 1000;
        c.early_stopping_rounds = 10;
        c.validation_fraction = 0.2;
        c.subsample = 0.8;
        c.seed = Some(3);
        let r = xgboost(x.view(), y.view(), &c).unwrap();
        let rows = r.validation_rows.clone().unwrap();
        assert_eq!(rows.len(), 40);
        assert!(rows.windows(2).all(|w| w[0] < w[1]));
        let best = r.best_iteration.unwrap();
        let ran = r.validation_loss.len();
        assert!(ran > best && ran < 1000, "ran {ran}, best {best}");
        let min = r
            .validation_loss
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        assert_eq!(r.validation_loss[best - 1], min);
        assert_eq!(r.n_estimators, best);
        assert_eq!(r.train_loss.len(), best + 1);
        assert_eq!(r.final_train_rmse, r.train_loss[best]);

        // The same trees as a fit on the training rows alone with `best` rounds.
        let train: Vec<usize> = (0..200).filter(|i| !rows.contains(i)).collect();
        let mut plain = c.clone();
        plain.early_stopping_rounds = 0;
        plain.n_estimators = best;
        let p = xgboost(
            x.select(Axis(0), &train).view(),
            y.select(Axis(0), &train).view(),
            &plain,
        )
        .unwrap();
        assert_eq!(xgboost_predict(&p, x.view()).unwrap(), r.predictions);
        assert!(
            r.to_string().contains(&format!(
                "early stopping on 40 held-out rows): {best} of {ran} rounds"
            )),
            "{r}"
        );
    }

    #[test]
    fn test_xgboost_early_stopping_refuses_bad_validation_fraction() {
        let (x, y) = early_stop_data(20);
        for f in [0.0, 1.0, f64::NAN, 0.95] {
            let mut c = XGBoostConfig::default();
            c.n_estimators = 5;
            c.early_stopping_rounds = 2;
            c.validation_fraction = f;
            let e = xgboost(x.view(), y.view(), &c).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("validation_fraction")),
                "{f}: {e}"
            );
        }
    }

    #[test]
    fn test_xgboost_best_iteration_only_with_early_stopping() {
        let (x, y) = early_stop_data(100);
        let mut c = XGBoostConfig::default();
        c.n_estimators = 20;
        c.validation_fraction = f64::NAN; // not read without early stopping
        let r = xgboost(x.view(), y.view(), &c).unwrap();
        assert_eq!(r.best_iteration, None);
        assert_eq!(r.validation_rows, None);
        assert!(r.validation_loss.is_empty());
        assert_eq!(r.n_estimators, 20);
        assert_eq!(xgboost_predict(&r, x.view()).unwrap(), r.predictions);
    }

    /// x = i in one column, y = i % 2: every split isolates one row, so an uncapped tree is a
    /// chain as deep as the rows.
    fn chain_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        (
            Array2::from_shape_fn((n, 1), |(i, _)| i as f64),
            Array1::from_shape_fn(n, |i| (i % 2) as f64),
        )
    }

    /// Runs `f` on a thread with a 2 MiB stack, the size of tokio's blocking threads and rayon's workers.
    fn on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap()
    }

    #[test]
    fn test_xgboost_tree_depth_is_capped() {
        // max_depth 0 (no limit) stops at depth 512. Before: a chain as deep as the rows.
        let depth = on_small_stack(|| {
            let (x, y) = chain_data(2_000);
            let mut c = XGBoostConfig::default();
            c.n_estimators = 1;
            c.max_depth = 0;
            c.lambda = 0.0;
            c.min_child_weight = 0.0;
            c.gamma = 0.0;
            let r = xgboost(x.view(), y.view(), &c).unwrap();
            let (mut max, mut stack) = (0, vec![(&r.trees[0].root, 0usize)]);
            while let Some((node, d)) = stack.pop() {
                max = max.max(d);
                if let XGBoostNode::Split { left, right, .. } = node {
                    stack.push((left, d + 1));
                    stack.push((right, d + 1));
                }
            }
            max
        });
        assert_eq!(depth, MAX_TREE_DEPTH);
    }
}
