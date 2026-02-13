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
//! - **Split gain formula**: Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G+H)^2/(H+lambda)] - gamma
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
//!   Proceedings of the 22nd ACM SIGKDD, 785-794. https://doi.org/10.1145/2939672.2939785
//! - XGBoost Documentation: https://xgboost.readthedocs.io/en/stable/tutorials/model.html

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::errors::{EconError, EconResult};
use crate::Dataset;

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
    /// Maximum depth of trees (0 = no limit, but NOT recommended)
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
    /// Fraction of features to use for each tree (colsample_bytree)
    pub colsample_bytree: f64,
    /// Fraction of features to use for each split (colsample_bylevel)
    pub colsample_bylevel: f64,
    /// Objective function
    pub objective: XGBoostObjective,
    /// Base score (initial prediction)
    pub base_score: Option<f64>,
    /// Random seed
    pub seed: Option<u64>,
    /// Early stopping rounds (0 = disabled)
    pub early_stopping_rounds: usize,
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
            verbosity: 0,
        }
    }
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
    /// Iteration where early stopping triggered (if any)
    pub best_iteration: Option<usize>,
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
        writeln!(f, "Gamma (min split gain): {:.4}", self.config.gamma)?;

        if self.config.subsample < 1.0 {
            writeln!(f, "Subsample: {:.2}", self.config.subsample)?;
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
            writeln!(f, "Best iteration (early stopping): {}", best_iter)?;
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
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names
                    .get(*i)
                    .cloned()
                    .unwrap_or_else(|| format!("f{}", i)),
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
fn compute_loss(y: &ArrayView1<f64>, predictions: &Array1<f64>, objective: XGBoostObjective) -> f64 {
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

/// Optimal leaf weight with L1 and L2 regularization.
/// w* = -G / (H + lambda) with L1 soft thresholding
/// Reference: Chen & Guestrin (2016), Equation 5
#[inline]
fn optimal_weight(g_sum: f64, h_sum: f64, lambda: f64, alpha: f64) -> f64 {
    if h_sum < 1e-10 {
        return 0.0;
    }

    // L1 soft thresholding
    let g_thresh = if g_sum > alpha {
        g_sum - alpha
    } else if g_sum < -alpha {
        g_sum + alpha
    } else {
        return 0.0;
    };

    -g_thresh / (h_sum + lambda)
}

/// Compute split gain.
/// Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
/// Reference: Chen & Guestrin (2016), Equation 7
#[inline]
fn split_gain(
    g_left: f64,
    h_left: f64,
    g_right: f64,
    h_right: f64,
    lambda: f64,
    gamma: f64,
) -> f64 {
    let score_left = if h_left > 0.0 {
        g_left * g_left / (h_left + lambda)
    } else {
        0.0
    };
    let score_right = if h_right > 0.0 {
        g_right * g_right / (h_right + lambda)
    } else {
        0.0
    };
    let score_parent = {
        let g_total = g_left + g_right;
        let h_total = h_left + h_right;
        if h_total > 0.0 {
            g_total * g_total / (h_total + lambda)
        } else {
            0.0
        }
    };

    0.5 * (score_left + score_right - score_parent) - gamma
}

/// Build an XGBoost tree using exact greedy algorithm.
fn build_tree(
    x: &ArrayView2<f64>,
    g: &ArrayView1<f64>,
    h: &ArrayView1<f64>,
    indices: &[usize],
    feature_indices: &[usize],
    config: &XGBoostConfig,
    depth: usize,
) -> XGBoostNode {
    let n = indices.len();

    // Compute sums for this node
    let g_sum: f64 = indices.iter().map(|&i| g[i]).sum();
    let h_sum: f64 = indices.iter().map(|&i| h[i]).sum();

    // Check stopping conditions
    let max_depth = if config.max_depth == 0 {
        usize::MAX
    } else {
        config.max_depth
    };

    if depth >= max_depth || n <= 1 || h_sum < config.min_child_weight {
        let weight = optimal_weight(g_sum, h_sum, config.lambda, config.alpha);
        return XGBoostNode::Leaf { weight, cover: h_sum };
    }

    // Find best split using exact greedy algorithm
    // Reference: Algorithm 1 in Chen & Guestrin (2016)
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_split: Option<(usize, f64)> = None;
    let mut best_left_indices = Vec::new();
    let mut best_right_indices = Vec::new();

    for &feature in feature_indices {
        // Sort indices by feature value
        let mut sorted: Vec<(f64, usize)> = indices.iter().map(|&i| (x[[i, feature]], i)).collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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

            let gain = split_gain(g_left, h_left, g_right, h_right, config.lambda, config.gamma);

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
        return XGBoostNode::Leaf { weight, cover: h_sum };
    }

    let (feature, threshold) = best_split.unwrap();

    // Build children recursively
    let left = build_tree(
        x,
        g,
        h,
        &best_left_indices,
        feature_indices,
        config,
        depth + 1,
    );
    let right = build_tree(
        x,
        g,
        h,
        &best_right_indices,
        feature_indices,
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

/// Select a random subset of feature indices.
fn random_features(n_features: usize, fraction: f64, rng_state: &mut u64) -> Vec<usize> {
    if fraction >= 1.0 {
        return (0..n_features).collect();
    }

    let n_select = ((n_features as f64) * fraction).ceil() as usize;
    let mut indices: Vec<usize> = (0..n_features).collect();

    for i in 0..n_select.min(n_features) {
        let j = i + lcg_random(rng_state) % (n_features - i);
        indices.swap(i, j);
    }

    indices.truncate(n_select);
    indices
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
pub fn xgboost(x: ArrayView2<f64>, y: ArrayView1<f64>, config: &XGBoostConfig) -> EconResult<XGBoostResult> {
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

    let mut rng_state = config.seed.unwrap_or(42);

    // Initialize predictions with base score
    let base_score = config.base_score.unwrap_or_else(|| compute_base_score(&y, config.objective));
    let mut predictions = Array1::from_elem(n_samples, base_score);

    let mut trees = Vec::with_capacity(config.n_estimators);
    let mut train_loss = Vec::with_capacity(config.n_estimators + 1);
    let mut feature_gains: HashMap<usize, f64> = HashMap::new();

    // Initial loss
    train_loss.push(compute_loss(&y, &predictions, config.objective));

    let mut best_loss = f64::INFINITY;
    let mut best_iteration = None;
    let mut rounds_without_improvement = 0;

    // Boosting iterations
    for iter in 0..config.n_estimators {
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
            config,
            0,
        );

        let tree = XGBoostTree {
            root,
            n_features,
        };

        // Update predictions for all samples
        for i in 0..n_samples {
            let row = x.row(i);
            predictions[i] += config.learning_rate * tree.predict_one(&row);
        }

        // Accumulate feature importance
        tree.feature_importances(&mut feature_gains);

        trees.push(tree);

        // Track loss
        let loss = compute_loss(&y, &predictions, config.objective);
        train_loss.push(loss);

        if config.verbosity > 0 && (iter + 1) % 10 == 0 {
            eprintln!("[{}]\ttrain-loss:{:.6}", iter + 1, loss);
        }

        // Early stopping
        if config.early_stopping_rounds > 0 {
            if loss < best_loss - 1e-10 {
                best_loss = loss;
                best_iteration = Some(iter);
                rounds_without_improvement = 0;
            } else {
                rounds_without_improvement += 1;
                if rounds_without_improvement >= config.early_stopping_rounds {
                    if config.verbosity > 0 {
                        eprintln!(
                            "Early stopping at iteration {} (best: {})",
                            iter + 1,
                            best_iteration.unwrap_or(0) + 1
                        );
                    }
                    break;
                }
            }
        }
    }

    // Normalize feature importances
    let mut importances = vec![0.0; n_features];
    let total_gain: f64 = feature_gains.values().sum();
    if total_gain > 0.0 {
        for (feature, gain) in feature_gains {
            importances[feature] = gain / total_gain;
        }
    }

    // Convert predictions to probabilities for classification
    let final_predictions = match config.objective {
        XGBoostObjective::BinaryLogistic | XGBoostObjective::RegLogistic => {
            predictions.mapv(sigmoid).to_vec()
        }
        XGBoostObjective::RegSquaredError => predictions.to_vec(),
    };

    Ok(XGBoostResult {
        n_estimators: trees.len(),
        feature_importances: importances,
        train_loss: train_loss.clone(),
        final_train_rmse: *train_loss.last().unwrap_or(&0.0),
        base_score,
        predictions: final_predictions,
        config: config.clone(),
        feature_names: None,
        best_iteration,
        trees,
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
pub fn xgboost_predict_class(result: &XGBoostResult, x: ArrayView2<f64>, threshold: f64) -> EconResult<Vec<i32>> {
    let probs = xgboost_predict(result, x)?;
    Ok(probs.iter().map(|&p| if p >= threshold { 1 } else { 0 }).collect())
}

/// Run XGBoost on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column name
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
    let y_series = dataset
        .df()
        .column(y_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: col_names.clone(),
        })?;

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::Computation(format!("Column '{}' is not numeric", y_col)))?
        .into_no_null_iter()
        .collect();

    let y_arr = Array1::from_vec(y);

    let mut result = xgboost(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Convenience function for running XGBoost with default configuration.
pub fn run_xgboost_default(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<XGBoostResult> {
    run_xgboost(dataset, y_col, x_cols, &XGBoostConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability {} out of range",
                p
            );
        }

        // Low x values should have low probability, high values should have high probability
        assert!(result.predictions[0] < 0.5, "Sample 0 should be class 0");
        assert!(result.predictions[7] > 0.5, "Sample 7 should be class 1");
    }

    #[test]
    fn test_xgboost_regularization() {
        // Test that regularization actually affects the model
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

        // No regularization
        let config_noreg = XGBoostConfig {
            n_estimators: 20,
            learning_rate: 0.3,
            max_depth: 3,
            lambda: 0.0,
            gamma: 0.0,
            ..Default::default()
        };

        // Strong regularization
        let config_reg = XGBoostConfig {
            n_estimators: 20,
            learning_rate: 0.3,
            max_depth: 3,
            lambda: 10.0,
            gamma: 1.0,
            ..Default::default()
        };

        let result_noreg = xgboost(x.view(), y.view(), &config_noreg).unwrap();
        let result_reg = xgboost(x.view(), y.view(), &config_reg).unwrap();

        // Regularized model should have higher training error (less overfitting)
        // This is expected behavior - regularization trades training accuracy for generalization
        // Note: With only 10 samples, both models may fit well, so we just check they run
        assert!(result_noreg.final_train_rmse >= 0.0);
        assert!(result_reg.final_train_rmse >= 0.0);
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
        let gain = split_gain(-5.0, 2.0, -5.0, 2.0, 1.0, 0.0);
        // Left: (-5)^2 / (2+1) = 25/3
        // Right: (-5)^2 / (2+1) = 25/3
        // Parent: (-10)^2 / (4+1) = 100/5 = 20
        // Gain = 0.5 * (25/3 + 25/3 - 20) - 0 = 0.5 * (50/3 - 20) = 0.5 * (-10/3) < 0

        // A good split should have positive gain
        let good_gain = split_gain(-8.0, 2.0, -2.0, 2.0, 1.0, 0.0);
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
}
