//! Gradient Boosting Machine (GBM) implementation.
//!
//! Pure Rust implementation of gradient boosting for regression and classification.
//! Based on Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine".
//!
//! ## Features
//!
//! - **Regression**: Gaussian loss (MSE), Huber loss for robustness
//! - **Classification**: Binomial deviance (logistic loss)
//! - **Regularization**: Learning rate (shrinkage), subsampling, max depth
//! - **Stochastic gradient boosting**: Row and column subsampling
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{gbm, GbmConfig, GbmFamily};
//! use ndarray::array;
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let y = array![1.1, 1.9, 3.2, 3.8, 5.1];
//!
//! let config = GbmConfig {
//!     n_trees: 100,
//!     learning_rate: 0.1,
//!     max_depth: 3,
//!     min_samples_split: 2,
//!     subsample: 0.8,
//!     family: GbmFamily::Gaussian,
//!     ..Default::default()
//! };
//!
//! let result = gbm(x.view(), y.view(), &config).unwrap();
//! println!("Training MSE: {:.4}", result.train_loss);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};

use crate::Dataset;
use crate::errors::{EconError, EconResult};

/// Loss function family for GBM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GbmFamily {
    /// Gaussian (squared error loss) for regression
    #[default]
    Gaussian,
    /// Huber loss for robust regression
    Huber,
    /// Binomial deviance for binary classification
    Binomial,
    /// Laplace (absolute error) for median regression
    Laplace,
}

impl std::str::FromStr for GbmFamily {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gaussian" | "mse" | "squared_error" => Ok(GbmFamily::Gaussian),
            "huber" => Ok(GbmFamily::Huber),
            "binomial" | "logistic" | "bernoulli" => Ok(GbmFamily::Binomial),
            "laplace" | "lad" | "absolute_error" => Ok(GbmFamily::Laplace),
            _ => Err(format!("Unknown GBM family: {}", s)),
        }
    }
}

/// Configuration for Gradient Boosting Machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbmConfig {
    /// Number of boosting iterations (trees)
    pub n_trees: usize,
    /// Learning rate (shrinkage factor)
    pub learning_rate: f64,
    /// Maximum depth of individual trees
    pub max_depth: usize,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Fraction of samples to use for each tree (stochastic GB)
    pub subsample: f64,
    /// Fraction of features to use for each tree
    pub colsample_bytree: f64,
    /// Loss function family
    pub family: GbmFamily,
    /// Huber delta parameter (for Huber loss)
    pub huber_delta: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for GbmConfig {
    fn default() -> Self {
        GbmConfig {
            n_trees: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            subsample: 1.0,
            colsample_bytree: 1.0,
            family: GbmFamily::Gaussian,
            huber_delta: 1.0,
            seed: None,
        }
    }
}

/// Result from GBM fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbmResult {
    /// Feature importances (mean decrease in impurity)
    pub feature_importances: Vec<f64>,
    /// Training loss at each iteration
    pub train_loss: Vec<f64>,
    /// Final training loss
    pub final_train_loss: f64,
    /// Number of trees fitted
    pub n_trees: usize,
    /// Initial prediction (mean for regression, log-odds for classification)
    pub init_prediction: f64,
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Configuration used
    pub config: GbmConfig,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Internal: trees (serialized for prediction)
    #[serde(skip)]
    pub(crate) trees: Vec<GbmTree>,
}

impl std::fmt::Display for GbmResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Gradient Boosting Machine Results")?;
        writeln!(f, "==================================")?;
        writeln!(f, "Family: {:?}", self.config.family)?;
        writeln!(f, "Number of trees: {}", self.n_trees)?;
        writeln!(f, "Learning rate: {:.4}", self.config.learning_rate)?;
        writeln!(f, "Max depth: {}", self.config.max_depth)?;

        if self.config.subsample < 1.0 {
            writeln!(f, "Subsample ratio: {:.2}", self.config.subsample)?;
        }

        writeln!(f)?;
        writeln!(f, "Final training loss: {:.6}", self.final_train_loss)?;

        if self.train_loss.len() > 1 {
            writeln!(f, "Initial loss: {:.6}", self.train_loss[0])?;
            writeln!(
                f,
                "Loss reduction: {:.2}%",
                (1.0 - self.final_train_loss / self.train_loss[0]) * 100.0
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Feature Importances:")?;

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

/// Internal tree node for GBM.
#[derive(Debug, Clone)]
pub(crate) enum GbmNode {
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<GbmNode>,
        right: Box<GbmNode>,
    },
    Leaf {
        value: f64,
        n_samples: usize,
    },
}

/// Internal tree structure for GBM.
#[derive(Debug, Clone)]
pub(crate) struct GbmTree {
    root: Option<GbmNode>,
    n_features: usize,
}

impl GbmTree {
    fn new() -> Self {
        GbmTree {
            root: None,
            n_features: 0,
        }
    }

    /// Fit tree to pseudo-residuals.
    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        residuals: &ArrayView1<f64>,
        max_depth: usize,
        min_samples_split: usize,
        feature_indices: &[usize],
        rng_state: &mut u64,
    ) {
        self.n_features = x.ncols();
        let indices: Vec<usize> = (0..x.nrows()).collect();
        self.root = Some(self.build_tree(
            x,
            residuals,
            &indices,
            0,
            max_depth,
            min_samples_split,
            feature_indices,
            rng_state,
        ));
    }

    fn build_tree(
        &self,
        x: &ArrayView2<f64>,
        residuals: &ArrayView1<f64>,
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        min_samples_split: usize,
        feature_indices: &[usize],
        rng_state: &mut u64,
    ) -> GbmNode {
        let n_samples = indices.len();

        // Check stopping conditions
        if depth >= max_depth || n_samples < min_samples_split || n_samples <= 1 {
            return self.create_leaf(residuals, indices);
        }

        // Check if all residuals are the same
        let first_val = residuals[indices[0]];
        if indices
            .iter()
            .all(|&i| (residuals[i] - first_val).abs() < 1e-10)
        {
            return self.create_leaf(residuals, indices);
        }

        // Find best split
        if let Some((best_feature, best_threshold, left_indices, right_indices)) =
            self.find_best_split(x, residuals, indices, feature_indices)
        {
            if left_indices.is_empty() || right_indices.is_empty() {
                return self.create_leaf(residuals, indices);
            }

            let mut new_rng = *rng_state;
            let left = self.build_tree(
                x,
                residuals,
                &left_indices,
                depth + 1,
                max_depth,
                min_samples_split,
                feature_indices,
                &mut new_rng,
            );
            let right = self.build_tree(
                x,
                residuals,
                &right_indices,
                depth + 1,
                max_depth,
                min_samples_split,
                feature_indices,
                rng_state,
            );

            GbmNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            self.create_leaf(residuals, indices)
        }
    }

    fn create_leaf(&self, residuals: &ArrayView1<f64>, indices: &[usize]) -> GbmNode {
        let sum: f64 = indices.iter().map(|&i| residuals[i]).sum();
        let value = sum / indices.len() as f64;
        GbmNode::Leaf {
            value,
            n_samples: indices.len(),
        }
    }

    fn find_best_split(
        &self,
        x: &ArrayView2<f64>,
        residuals: &ArrayView1<f64>,
        indices: &[usize],
        feature_indices: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let mut best_gain = 0.0f64;
        let mut best_split: Option<(usize, f64)> = None; // (feature, threshold)
        let n = indices.len();

        if n < 2 {
            return None;
        }

        // OPTIMIZED: O(n log n) per feature using incremental sums
        // Compute total sum and sum of squares
        let total_sum: f64 = indices.iter().map(|&i| residuals[i]).sum();
        let total_ss: f64 = indices.iter().map(|&i| residuals[i] * residuals[i]).sum();
        // Node MSE = (SS - sum²/n) / n
        let node_mse_scaled = total_ss - total_sum * total_sum / n as f64;

        for &feature in feature_indices {
            // Sort indices by feature value - O(n log n)
            let mut sorted: Vec<(f64, f64, usize)> = indices
                .iter()
                .map(|&i| (x[[i, feature]], residuals[i], i))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Scan through sorted values with running sums - O(n)
            let mut left_sum = 0.0;
            let mut left_ss = 0.0;
            let mut left_n = 0usize;

            for i in 0..sorted.len() - 1 {
                let (x_val, r_val, _) = sorted[i];
                left_sum += r_val;
                left_ss += r_val * r_val;
                left_n += 1;

                // Skip if next value is the same (no valid split point)
                let next_x = sorted[i + 1].0;
                if (next_x - x_val).abs() < 1e-10 {
                    continue;
                }

                let right_n = n - left_n;
                if left_n == 0 || right_n == 0 {
                    continue;
                }

                // Incremental MSE calculation
                let left_mse_scaled = left_ss - left_sum * left_sum / left_n as f64;
                let right_sum = total_sum - left_sum;
                let right_ss = total_ss - left_ss;
                let right_mse_scaled = right_ss - right_sum * right_sum / right_n as f64;

                // Gain = node_mse - weighted child mse (all scaled by n)
                let gain = node_mse_scaled - left_mse_scaled - right_mse_scaled;

                if gain > best_gain {
                    best_gain = gain;
                    let threshold = (x_val + next_x) / 2.0;
                    best_split = Some((feature, threshold));
                }
            }
        }

        // Now partition indices based on best split
        best_split.map(|(feature, threshold)| {
            let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| x[[i, feature]] <= threshold);
            (feature, threshold, left_indices, right_indices)
        })
    }

    fn predict_one(&self, x: &ArrayView1<f64>) -> f64 {
        match &self.root {
            Some(node) => self.traverse(node, x),
            None => 0.0,
        }
    }

    fn traverse(&self, node: &GbmNode, x: &ArrayView1<f64>) -> f64 {
        match node {
            GbmNode::Leaf { value, .. } => *value,
            GbmNode::Split {
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

    fn accumulate_importances(&self, node: &GbmNode, importances: &mut Array1<f64>, weight: f64) {
        if let GbmNode::Split {
            feature_index,
            left,
            right,
            ..
        } = node
        {
            importances[*feature_index] += weight;

            let (left_weight, right_weight) = match (left.as_ref(), right.as_ref()) {
                (GbmNode::Leaf { n_samples: ln, .. }, GbmNode::Leaf { n_samples: rn, .. }) => {
                    let total = (*ln + *rn) as f64;
                    (*ln as f64 / total, *rn as f64 / total)
                }
                _ => (0.5, 0.5),
            };

            self.accumulate_importances(left, importances, weight * left_weight);
            self.accumulate_importances(right, importances, weight * right_weight);
        }
    }
}

/// Compute MSE for a set of indices.
fn compute_mse_indices(y: &ArrayView1<f64>, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }

    let sum: f64 = indices.iter().map(|&i| y[i]).sum();
    let mean = sum / indices.len() as f64;

    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / indices.len() as f64
}

/// Simple LCG random number generator.
fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
}

/// Compute negative gradient (pseudo-residuals) for the loss function.
fn compute_negative_gradient(
    y: &ArrayView1<f64>,
    predictions: &Array1<f64>,
    family: GbmFamily,
    huber_delta: f64,
) -> Array1<f64> {
    let n = y.len();
    let mut gradient = Array1::zeros(n);

    match family {
        GbmFamily::Gaussian => {
            // Negative gradient of MSE: y - f(x)
            for i in 0..n {
                gradient[i] = y[i] - predictions[i];
            }
        }
        GbmFamily::Huber => {
            // Huber loss negative gradient
            for i in 0..n {
                let r = y[i] - predictions[i];
                if r.abs() <= huber_delta {
                    gradient[i] = r;
                } else {
                    gradient[i] = huber_delta * r.signum();
                }
            }
        }
        GbmFamily::Binomial => {
            // Negative gradient of log loss: y - p
            // Where p = sigmoid(f(x))
            for i in 0..n {
                let p = sigmoid(predictions[i]);
                gradient[i] = y[i] - p;
            }
        }
        GbmFamily::Laplace => {
            // Negative gradient of LAD: sign(y - f(x))
            for i in 0..n {
                let r = y[i] - predictions[i];
                gradient[i] = if r > 0.0 {
                    1.0
                } else if r < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
    }

    gradient
}

/// Compute loss for current predictions.
fn compute_loss(
    y: &ArrayView1<f64>,
    predictions: &Array1<f64>,
    family: GbmFamily,
    huber_delta: f64,
) -> f64 {
    let n = y.len() as f64;

    match family {
        GbmFamily::Gaussian => {
            // MSE
            y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| (yi - pi).powi(2))
                .sum::<f64>()
                / n
        }
        GbmFamily::Huber => {
            // Huber loss
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
        GbmFamily::Binomial => {
            // Log loss (binary cross-entropy)
            -y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| {
                    let p = sigmoid(pi);
                    yi * p.max(1e-15).ln() + (1.0 - yi) * (1.0 - p).max(1e-15).ln()
                })
                .sum::<f64>()
                / n
        }
        GbmFamily::Laplace => {
            // Mean absolute error
            y.iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| (yi - pi).abs())
                .sum::<f64>()
                / n
        }
    }
}

/// Sigmoid function.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Compute initial prediction (F0).
fn compute_init_prediction(y: &ArrayView1<f64>, family: GbmFamily) -> f64 {
    match family {
        GbmFamily::Gaussian | GbmFamily::Huber => {
            // Mean of y
            y.iter().sum::<f64>() / y.len() as f64
        }
        GbmFamily::Binomial => {
            // Log-odds of mean probability
            let p = y.iter().sum::<f64>() / y.len() as f64;
            let p = p.clamp(1e-10, 1.0 - 1e-10);
            (p / (1.0 - p)).ln()
        }
        GbmFamily::Laplace => {
            // Median of y
            let mut sorted: Vec<f64> = y.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
    }
}

/// Fit a Gradient Boosting Machine.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
/// * `config` - GBM configuration
///
/// # Returns
///
/// GbmResult containing fitted model and diagnostics.
pub fn gbm(x: ArrayView2<f64>, y: ArrayView1<f64>, config: &GbmConfig) -> EconResult<GbmResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for GBM".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }

    // Validate binary targets for classification
    if config.family == GbmFamily::Binomial {
        for &yi in y.iter() {
            if yi != 0.0 && yi != 1.0 {
                return Err(EconError::Computation(
                    "Binomial family requires binary targets (0 or 1)".to_string(),
                ));
            }
        }
    }

    let mut rng_state = config.seed.unwrap_or(42);

    // Initialize predictions with F0
    let init_pred = compute_init_prediction(&y, config.family);
    let mut predictions = Array1::from_elem(n_samples, init_pred);

    let mut trees = Vec::with_capacity(config.n_trees);
    let mut train_loss = Vec::with_capacity(config.n_trees);
    let mut total_importances = Array1::zeros(n_features);

    // Initial loss
    train_loss.push(compute_loss(
        &y,
        &predictions,
        config.family,
        config.huber_delta,
    ));

    // Boosting iterations
    for _ in 0..config.n_trees {
        // Compute negative gradient (pseudo-residuals)
        let residuals =
            compute_negative_gradient(&y, &predictions, config.family, config.huber_delta);

        // Subsample rows if needed
        let (sample_indices, sample_x, sample_residuals) = if config.subsample < 1.0 {
            let n_subsample = ((n_samples as f64) * config.subsample).ceil() as usize;
            let mut indices: Vec<usize> = (0..n_samples).collect();

            // Fisher-Yates shuffle and take first n_subsample
            for i in 0..n_subsample.min(indices.len()) {
                let j = i + lcg_random(&mut rng_state) % (indices.len() - i);
                indices.swap(i, j);
            }
            indices.truncate(n_subsample);

            let sub_x = select_rows(&x, &indices);
            let sub_r: Array1<f64> = indices.iter().map(|&i| residuals[i]).collect();

            (indices, sub_x, sub_r)
        } else {
            let indices: Vec<usize> = (0..n_samples).collect();
            (indices, x.to_owned(), residuals.clone())
        };

        // Subsample features if needed
        let feature_indices: Vec<usize> = if config.colsample_bytree < 1.0 {
            let n_features_sample = ((n_features as f64) * config.colsample_bytree).ceil() as usize;
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

        // Fit tree to pseudo-residuals
        let mut tree = GbmTree::new();
        tree.fit(
            &sample_x.view(),
            &sample_residuals.view(),
            config.max_depth,
            config.min_samples_split,
            &feature_indices,
            &mut rng_state,
        );

        // Update predictions for all samples
        for i in 0..n_samples {
            let row = x.row(i);
            predictions[i] += config.learning_rate * tree.predict_one(&row);
        }

        // Accumulate feature importances
        let tree_importances = tree.feature_importances();
        total_importances = total_importances + tree_importances;

        trees.push(tree);

        // Track loss
        train_loss.push(compute_loss(
            &y,
            &predictions,
            config.family,
            config.huber_delta,
        ));
    }

    // Normalize feature importances
    let sum: f64 = total_importances.sum();
    if sum > 0.0 {
        total_importances /= sum;
    }

    // Convert predictions to probabilities for classification
    let final_predictions = if config.family == GbmFamily::Binomial {
        predictions.mapv(sigmoid).to_vec()
    } else {
        predictions.to_vec()
    };

    Ok(GbmResult {
        feature_importances: total_importances.to_vec(),
        train_loss: train_loss.clone(),
        final_train_loss: *train_loss.last().unwrap_or(&0.0),
        n_trees: config.n_trees,
        init_prediction: init_pred,
        predictions: final_predictions,
        config: config.clone(),
        feature_names: None,
        trees,
    })
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

/// Predict using a fitted GBM model.
///
/// # Arguments
///
/// * `result` - Fitted GBM result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions (probabilities for classification, values for regression)
pub fn gbm_predict(result: &GbmResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    let n_samples = x.nrows();

    if result.trees.is_empty() {
        return Err(EconError::Computation(
            "Model has no fitted trees".to_string(),
        ));
    }

    let mut predictions = Array1::from_elem(n_samples, result.init_prediction);

    for tree in &result.trees {
        for i in 0..n_samples {
            predictions[i] += result.config.learning_rate * tree.predict_one(&x.row(i));
        }
    }

    // Convert to probabilities for classification
    if result.config.family == GbmFamily::Binomial {
        Ok(predictions.mapv(sigmoid).to_vec())
    } else {
        Ok(predictions.to_vec())
    }
}

/// Run GBM on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column name
/// * `x_cols` - Feature column names
/// * `config` - GBM configuration
///
/// # Returns
///
/// GbmResult with model and diagnostics
pub fn run_gbm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &GbmConfig,
) -> EconResult<GbmResult> {
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

    let mut result = gbm(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Convenience function for running GBM with default configuration.
pub fn run_gbm_default(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<GbmResult> {
    run_gbm(dataset, y_col, x_cols, &GbmConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gbm_gaussian_basic() {
        // Simple linear relationship
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

        let config = GbmConfig {
            n_trees: 50,
            learning_rate: 0.1,
            max_depth: 3,
            ..Default::default()
        };

        let result = gbm(x.view(), y.view(), &config).unwrap();

        // Check we got the right number of trees
        assert_eq!(result.n_trees, 50);

        // Loss should decrease
        assert!(result.train_loss.last().unwrap() < result.train_loss.first().unwrap());

        // Predictions should be close to targets
        let mse: f64 = result
            .predictions
            .iter()
            .zip(y.iter())
            .map(|(p, y)| (p - y).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 1.0, "MSE {} should be < 1.0", mse);
    }

    #[test]
    fn test_gbm_binomial() {
        // Binary classification
        let x = array![
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 0.0],
            [8.0, 1.0],
            [8.5, 0.5],
            [9.0, 1.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let config = GbmConfig {
            n_trees: 50,
            learning_rate: 0.3,
            max_depth: 2,
            family: GbmFamily::Binomial,
            ..Default::default()
        };

        let result = gbm(x.view(), y.view(), &config).unwrap();

        // Predictions should be probabilities
        for &p in &result.predictions {
            assert!((0.0..=1.0).contains(&p), "Probability {} out of range", p);
        }

        // Low values should have low probability, high values should have high probability
        assert!(result.predictions[0] < 0.5);
        assert!(result.predictions[5] > 0.5);
    }

    #[test]
    fn test_gbm_huber() {
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
        let y = array![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0, // outlier
        ];

        let config = GbmConfig {
            n_trees: 100,
            learning_rate: 0.1,
            max_depth: 3,
            family: GbmFamily::Huber,
            huber_delta: 1.0,
            ..Default::default()
        };

        let result = gbm(x.view(), y.view(), &config).unwrap();

        // Loss should decrease
        assert!(result.train_loss.last().unwrap() < result.train_loss.first().unwrap());
    }

    #[test]
    fn test_gbm_subsample() {
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

        let config = GbmConfig {
            n_trees: 50,
            learning_rate: 0.1,
            max_depth: 3,
            subsample: 0.5,
            colsample_bytree: 0.5,
            seed: Some(42),
            ..Default::default()
        };

        let result = gbm(x.view(), y.view(), &config).unwrap();

        // Should still work with subsampling
        assert_eq!(result.n_trees, 50);
        assert!(result.train_loss.last().unwrap() < result.train_loss.first().unwrap());
    }

    #[test]
    fn test_gbm_predict() {
        let x_train = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = GbmConfig {
            n_trees: 50,
            learning_rate: 0.1,
            max_depth: 3,
            ..Default::default()
        };

        let result = gbm(x_train.view(), y_train.view(), &config).unwrap();

        // Predict on new data
        let x_test = array![[1.5], [3.5], [5.5]];
        let predictions = gbm_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 3);
        // Predictions should be in reasonable range
        for &p in &predictions {
            assert!(p > 0.0 && p < 7.0);
        }
    }

    #[test]
    fn test_gbm_feature_importance() {
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

        let config = GbmConfig {
            n_trees: 50,
            learning_rate: 0.1,
            max_depth: 3,
            ..Default::default()
        };

        let result = gbm(x.view(), y.view(), &config).unwrap();

        // First feature should be more important
        assert!(
            result.feature_importances[0] > result.feature_importances[1],
            "Feature 0 importance {} should be > Feature 1 importance {}",
            result.feature_importances[0],
            result.feature_importances[1]
        );
    }

    #[test]
    fn test_gbm_laplace() {
        // Median regression
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

        let config = GbmConfig {
            n_trees: 50,
            learning_rate: 0.1,
            max_depth: 3,
            family: GbmFamily::Laplace,
            ..Default::default()
        };

        let result = gbm(x.view(), y.view(), &config).unwrap();

        // Loss should decrease
        assert!(result.train_loss.last().unwrap() <= result.train_loss.first().unwrap());
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-10);
        assert!(sigmoid(-100.0) < 1e-10);
    }
}
