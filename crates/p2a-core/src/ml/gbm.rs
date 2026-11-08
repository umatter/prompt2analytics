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
//! ```rust
//! use p2a_core::ml::{gbm, gbm_predict, GbmConfig, GbmFamily};
//! use ndarray::array;
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
//! let y = array![1.1, 1.9, 3.2, 3.8, 5.1];
//!
//! let config = GbmConfig {
//!     n_trees: 100,
//!     learning_rate: 0.1,   // shrinkage
//!     max_depth: 3,
//!     min_samples_split: 2,
//!     subsample: 0.8,       // stochastic gradient boosting: 80% of rows per tree
//!     family: GbmFamily::Gaussian,
//!     ..Default::default()
//! };
//!
//! let result = gbm(x.view(), y.view(), &config).unwrap();
//!
//! // `train_loss` is the loss *trajectory*: the loss of the constant starting
//! // model followed by one entry per boosting iteration. `final_train_loss`
//! // is the scalar at the end of that trajectory.
//! assert_eq!(result.train_loss.len(), config.n_trees + 1);
//! assert_eq!(result.final_train_loss, *result.train_loss.last().unwrap());
//!
//! // The starting model is F0 = mean(y) = 3.02, so the first entry is the
//! // variance of y: sum((y - 3.02)^2) / 5 = 9.908 / 5 = 1.9816.
//! assert!((result.train_loss[0] - 1.9816).abs() < 1e-9);
//!
//! // Boosting drives the Gaussian (MSE) loss below that starting point.
//! assert!(result.final_train_loss < result.train_loss[0]);
//! println!("Training MSE: {:.4}", result.final_train_loss);
//!
//! // The fitted ensemble scores new points.
//! let x_new = array![[1.5], [4.5]];
//! let preds = gbm_predict(&result, x_new.view()).unwrap();
//! assert_eq!(preds.len(), 2);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};

use super::trees::add_split_gain;
use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::resource;

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

crate::option_enum!(
    GbmFamily,
    crate::options::Choices {
        option: "family",
        values: &[
            (&["gaussian", "mse", "squared_error"], GbmFamily::Gaussian),
            (&["huber"], GbmFamily::Huber),
            (&["bernoulli", "binomial", "logistic"], GbmFamily::Binomial),
            (&["laplace", "lad", "absolute_error"], GbmFamily::Laplace),
        ],
    }
);

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
    /// Share of `relative_influence` per feature (gbm `summary()` divided by 100; zeros if no tree
    /// split).
    pub feature_importances: Vec<f64>,
    /// Summed SSE decrease of the splits on each feature, over all trees (gbm `relative.influence`,
    /// unscaled).
    #[serde(default)]
    pub relative_influence: Vec<f64>,
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
        writeln!(f, "Family: {}", self.config.family)?;
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
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

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
    /// Summed SSE decrease of this tree's splits per feature.
    gain: Vec<f64>,
}

impl GbmTree {
    fn new() -> Self {
        GbmTree {
            root: None,
            n_features: 0,
            gain: Vec::new(),
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
        let mut gain = vec![0.0; x.ncols()];
        self.root = Some(self.build_tree(
            x,
            residuals,
            &indices,
            0,
            max_depth,
            min_samples_split,
            feature_indices,
            rng_state,
            &mut gain,
        ));
        self.gain = gain;
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
        gain: &mut [f64],
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
            add_split_gain(gain, best_feature, residuals, &left_indices, &right_indices);

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
                gain,
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
                gain,
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
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

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

use super::lcg_random;

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
            sorted.sort_by(f64::total_cmp);
            let mid = sorted.len() / 2;
            if sorted.len().is_multiple_of(2) {
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
    validate_config(config)?;
    for j in 0..n_features {
        let k = x.column(j).iter().filter(|v| !v.is_finite()).count();
        if k > 0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "GBM: feature column {j} (0-based) has {k} non-finite values (NaN or infinite); remove or impute them first"
                ),
            });
        }
    }
    let k = y.iter().filter(|v| !v.is_finite()).count();
    if k > 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "GBM: the target has {k} non-finite values (NaN or infinite); remove them first"
            ),
        });
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

    let mut trees = resource::try_with_capacity(config.n_trees, "GBM trees")?;
    let mut train_loss =
        resource::try_with_capacity(config.n_trees.saturating_add(1), "GBM training loss")?;
    let mut relative_influence = vec![0.0f64; n_features];

    // Initial loss
    train_loss.push(compute_loss(
        &y,
        &predictions,
        config.family,
        config.huber_delta,
    ));

    // Boosting iterations
    for _ in 0..config.n_trees {
        resource::checkpoint()?;
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

        // Accumulate the relative influence
        for (sum, &g) in relative_influence.iter_mut().zip(&tree.gain) {
            *sum += g;
        }

        trees.push(tree);

        // Track loss
        train_loss.push(compute_loss(
            &y,
            &predictions,
            config.family,
            config.huber_delta,
        ));
    }

    // Shares of the relative influence
    let sum: f64 = relative_influence.iter().sum();
    let feature_importances: Vec<f64> = relative_influence
        .iter()
        .map(|&v| if sum > 0.0 { v / sum } else { 0.0 })
        .collect();

    // Convert predictions to probabilities for classification
    let final_predictions = if config.family == GbmFamily::Binomial {
        predictions.mapv(sigmoid).to_vec()
    } else {
        predictions.to_vec()
    };

    Ok(GbmResult {
        feature_importances,
        relative_influence,
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

/// Refuse settings no fit can honour.
fn validate_config(config: &GbmConfig) -> EconResult<()> {
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    if config.n_trees < 1 {
        return invalid("GBM: n_trees must be at least 1, got 0".to_string());
    }
    if config.max_depth < 1 {
        return invalid("GBM: max_depth must be at least 1, got 0".to_string());
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return invalid(format!(
            "GBM: learning_rate must be finite and above 0, got {}",
            config.learning_rate
        ));
    }
    if !(config.subsample > 0.0 && config.subsample <= 1.0) {
        return invalid(format!(
            "GBM: subsample must be in (0, 1], got {}",
            config.subsample
        ));
    }
    if !(config.colsample_bytree > 0.0 && config.colsample_bytree <= 1.0) {
        return invalid(format!(
            "GBM: colsample_bytree must be in (0, 1], got {}",
            config.colsample_bytree
        ));
    }
    if config.family == GbmFamily::Huber
        && (!config.huber_delta.is_finite() || config.huber_delta <= 0.0)
    {
        return invalid(format!(
            "GBM: huber_delta must be finite and above 0, got {}",
            config.huber_delta
        ));
    }
    Ok(())
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
/// * `y_col` - Numeric target column. Integer and boolean (0/1) columns are accepted;
///   a missing value (null or NaN) in the target or a feature is an error naming the
///   column (`EconError::NullValues`)
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
    let (x, y_arr, _missing) =
        super::extract::extract_xy(dataset, y_col, x_cols, crate::data::columns::NaPolicy::Fail)?;
    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

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

    #[test]
    fn test_validate_gbm_stump_influence() {
        // WT-5. 50 Gaussian stumps equal gbm(interaction.depth = 1, n.minobsinnode = 1, bag.fraction = 1):
        // fitted values, initF, split variables, and relative.influence (summed SSE decrease). Before:
        // feature_importances = split shares 0.68, 0.12, 0.20 (R 0.99494, 0.00159, 0.00348).
        use crate::data::columns::NaPolicy;
        use p2a_validation::{Check, Tol};
        let names = ["x0", "x1", "x2"];
        let (case, ds) = crate::test_support::load_case("gbm_stump_influence");
        let (x, y, _) = crate::ml::extract_xy(&ds, "y", &names, NaPolicy::Fail).unwrap();
        let mut config = GbmConfig::default();
        config.n_trees = 50;
        config.learning_rate = 0.1;
        config.max_depth = 1;
        config.min_samples_split = 2;
        config.subsample = 1.0;
        config.colsample_bytree = 1.0;
        config.family = GbmFamily::Gaussian;
        let r = gbm(x.view(), y.view(), &config).unwrap();
        let split_var: Vec<f64> = r
            .trees
            .iter()
            .map(|t| match &t.root {
                Some(GbmNode::Split { feature_index, .. }) => *feature_index as f64,
                _ => -1.0,
            })
            .collect();
        let tol = Tol::coef(200);
        Check::new(&case)
            .named("influence", &names, &r.relative_influence, tol)
            .named("influence_share", &names, &r.feature_importances, tol)
            .scalar("init_f", r.init_prediction, tol)
            .vector("fitted", &r.predictions, tol)
            .vector("split_var", &split_var, Tol::exact())
            .finish();
    }

    #[test]
    fn test_gbm_rejects_bad_arguments() {
        // ML-105: n_trees = 0 returned a model without trees whose gbm_predict then failed; NaN went into the sorts.
        let x = Array2::from_shape_fn((30, 2), |(i, j)| (i * (j + 3) % 17) as f64 + 0.1 * j as f64);
        let y = Array1::from_shape_fn(30, |i| (i % 7) as f64 + 0.3 * (i % 3) as f64);
        assert!(gbm(x.view(), y.view(), &GbmConfig::default()).is_ok());
        let bad = |f: &dyn Fn(&mut GbmConfig), what: &str| {
            let mut c = GbmConfig::default();
            f(&mut c);
            let e = gbm(x.view(), y.view(), &c).unwrap_err().to_string();
            assert!(e.contains(what), "{what}: {e}");
        };
        bad(&|c| c.n_trees = 0, "n_trees");
        bad(&|c| c.max_depth = 0, "max_depth");
        bad(&|c| c.learning_rate = 0.0, "learning_rate");
        bad(&|c| c.learning_rate = f64::NAN, "learning_rate");
        bad(&|c| c.subsample = 0.0, "subsample");
        bad(&|c| c.subsample = 1.5, "subsample");
        bad(&|c| c.colsample_bytree = 0.0, "colsample_bytree");
        bad(
            &|c| {
                c.family = GbmFamily::Huber;
                c.huber_delta = -1.0;
            },
            "huber_delta",
        );
        // Only the Huber family reads huber_delta.
        let unread = GbmConfig {
            huber_delta: 0.0,
            n_trees: 2,
            ..GbmConfig::default()
        };
        assert!(gbm(x.view(), y.view(), &unread).is_ok());
        // An impossible tree count is an error, not a failed allocation.
        bad(&|c| c.n_trees = usize::MAX, "GBM");
        let mut xb = x.clone();
        xb[[3, 1]] = f64::NAN;
        assert!(
            gbm(xb.view(), y.view(), &GbmConfig::default())
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
        let mut yb = y.clone();
        yb[2] = f64::INFINITY;
        assert!(
            gbm(x.view(), yb.view(), &GbmConfig::default())
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
    }

    #[test]
    fn test_gbm_stops_at_a_cancelled_budget() {
        let x = Array2::from_shape_fn((40, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64);
        let y = Array1::from_shape_fn(40, |i| 0.5 * ((i * 7) % 11) as f64 + ((i * 13) % 5) as f64);
        let token = crate::resource::CancelToken::new();
        token.cancel();
        let budget = crate::resource::Budget::new(None, token);
        let err =
            crate::resource::with_budget(budget, || gbm(x.view(), y.view(), &GbmConfig::default()))
                .unwrap_err();
        assert!(matches!(err, EconError::Cancelled(_)), "{err}");
    }

    #[test]
    fn test_gbm_family_parses_aliases_and_rejects_unknown() {
        GbmFamily::choices().validate().unwrap();
        for (s, f) in [
            ("gaussian", GbmFamily::Gaussian),
            ("squared-error", GbmFamily::Gaussian),
            ("bernoulli", GbmFamily::Binomial),
            ("Binomial", GbmFamily::Binomial),
            ("LAD", GbmFamily::Laplace),
            ("huber", GbmFamily::Huber),
        ] {
            assert_eq!(s.parse::<GbmFamily>().unwrap(), f, "{s}");
        }
        assert!(
            "poisson"
                .parse::<GbmFamily>()
                .unwrap_err()
                .to_string()
                .contains("gaussian")
        );
    }
}
