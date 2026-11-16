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

use super::trees::{MAX_TREE_DEPTH, add_split_gain};
use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::robust::{QuantileType, quantile_sorted};

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
    /// Maximum depth of individual trees (at least 1; trees stop at depth 512 whatever it says)
    pub max_depth: usize,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Fraction of samples to use for each tree (stochastic GB)
    pub subsample: f64,
    /// Fraction of features to use for each tree
    pub colsample_bytree: f64,
    /// Loss function family
    pub family: GbmFamily,
    /// The fixed Huber threshold, used only when `huber_alpha` is `None`; must be finite and above
    /// 0 for the Huber family, ignored by the others
    pub huber_delta: f64,
    /// Huber family: each iteration's threshold is the type-7 `huber_alpha`-quantile of
    /// `|y - F|` over that iteration's in-bag rows (Friedman 2001, Algorithm 4; default 0.9), so
    /// the fit is scale-equivariant. In (0, 1); `None` uses the fixed `huber_delta`.
    #[serde(default = "default_huber_alpha")]
    pub huber_alpha: Option<f64>,
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
            huber_alpha: default_huber_alpha(),
            seed: None,
        }
    }
}

/// Default of [`GbmConfig::huber_alpha`]: the 0.9-quantile of the absolute residuals.
fn default_huber_alpha() -> Option<f64> {
    Some(0.9)
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
        leaf: &Leaf<'_>,
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
            leaf,
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
        leaf: &Leaf<'_>,
    ) -> GbmNode {
        let n_samples = indices.len();

        // Check stopping conditions
        if depth >= max_depth.min(MAX_TREE_DEPTH) || n_samples < min_samples_split || n_samples <= 1
        {
            return self.create_leaf(residuals, indices, leaf);
        }

        // Check if all residuals are the same
        let first_val = residuals[indices[0]];
        if indices
            .iter()
            .all(|&i| (residuals[i] - first_val).abs() < 1e-10)
        {
            return self.create_leaf(residuals, indices, leaf);
        }

        // Find best split
        if let Some((best_feature, best_threshold, left_indices, right_indices)) =
            self.find_best_split(x, residuals, indices, feature_indices)
        {
            if left_indices.is_empty() || right_indices.is_empty() {
                return self.create_leaf(residuals, indices, leaf);
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
                leaf,
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
                leaf,
            );

            GbmNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            self.create_leaf(residuals, indices, leaf)
        }
    }

    fn create_leaf(
        &self,
        residuals: &ArrayView1<f64>,
        indices: &[usize],
        leaf: &Leaf<'_>,
    ) -> GbmNode {
        let value = match leaf {
            Leaf::Mean => {
                let sum: f64 = indices.iter().map(|&i| residuals[i]).sum();
                sum / indices.len() as f64
            }
            Leaf::Median { r } => {
                let mut scratch: Vec<f64> = indices.iter().map(|&i| r[i]).collect();
                median_in_place(&mut scratch)
            }
            Leaf::Newton { p } => {
                // gbm's CBernoulli::FitBestConstant: sum(y - p) / sum(p (1 - p)), 0 when the
                // denominator vanishes. The pseudo-residuals are y - p.
                let num: f64 = indices.iter().map(|&i| residuals[i]).sum();
                let den: f64 = indices.iter().map(|&i| p[i] * (1.0 - p[i])).sum();
                if den == 0.0 { 0.0 } else { num / den }
            }
            Leaf::Huber { r, delta } => {
                let mut scratch: Vec<f64> = indices.iter().map(|&i| r[i]).collect();
                let med = median_in_place(&mut scratch);
                let step: f64 = indices
                    .iter()
                    .map(|&i| {
                        let d = r[i] - med;
                        if d == 0.0 {
                            0.0
                        } else {
                            d.signum() * delta.min(d.abs())
                        }
                    })
                    .sum();
                med + step / indices.len() as f64
            }
        };
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

/// Terminal-node estimate of a family (Friedman 2001, Algorithms 2, 3 and 5; R gbm's
/// `FitBestConstant`), over the rows the tree was grown on. Slices are indexed by those rows'
/// positions in the tree's sample.
pub(crate) enum Leaf<'a> {
    /// Mean pseudo-residual (Gaussian: the least-squares step).
    Mean,
    /// Median of the residuals `y - F` (Laplace).
    Median { r: &'a [f64] },
    /// One Newton step of the binomial deviance, from the fitted probabilities `p` (Bernoulli).
    Newton { p: &'a [f64] },
    /// The median of the residuals `r = y - F` plus the mean of their deviations from it, clipped
    /// at `delta` (Huber).
    Huber { r: &'a [f64], delta: f64 },
}

/// Median of `v` (the mean of the two middle values for an even count, as R's `median`).
/// Reorders `v`; `v` must not be empty.
fn median_in_place(v: &mut [f64]) -> f64 {
    let n = v.len();
    let mid = n / 2;
    let (lower, upper, _) = v.select_nth_unstable_by(mid, f64::total_cmp);
    let upper = *upper;
    if !n.is_multiple_of(2) {
        upper
    } else {
        let lower_max = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lower_max + upper) / 2.0
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
        GbmFamily::Gaussian => {
            // Mean of y
            y.iter().sum::<f64>() / y.len() as f64
        }
        GbmFamily::Binomial => {
            // Log-odds of mean probability
            let p = y.iter().sum::<f64>() / y.len() as f64;
            let p = p.clamp(1e-10, 1.0 - 1e-10);
            (p / (1.0 - p)).ln()
        }
        GbmFamily::Laplace | GbmFamily::Huber => {
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
/// GbmResult containing fitted model and diagnostics. Each terminal node is the family's own
/// estimate over the leaf's in-bag rows (see [`GbmFamily`] and `validation/ml/gbm.md`). For the
/// Huber family, `train_loss` uses each iteration's threshold.
///
/// # Errors
///
/// - [`EconError::InvalidSpecification`] for the settings `validate_config` refuses (`n_trees`
///   or `max_depth` 0, a `learning_rate` that is not finite and positive, `subsample` or
///   `colsample_bytree` outside (0, 1], a `huber_alpha` outside (0, 1) or, with `huber_alpha =
///   None`, a `huber_delta` that is not finite and positive), non-finite features or target, and a
///   target whose sum of squares overflows.
/// - [`EconError::Computation`] for fewer than 2 rows, `x` and `y` of different lengths, and a
///   Binomial target that is not 0/1.
/// - [`EconError::ResourceLimit`] when the tree or loss buffers exceed the allocation ceiling.
/// - [`EconError::Cancelled`] when the computation budget is cancelled or expires.
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

    super::trees::check_target_scale(&y, "GBM")
        .map_err(|message| EconError::InvalidSpecification { message })?;

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

    // Initial loss (Huber: at the threshold of the residuals of F0 over all rows)
    let all_rows: Vec<usize> = (0..n_samples).collect();
    let delta0 = huber_threshold(&y, &predictions, &all_rows, config);
    train_loss.push(compute_loss(&y, &predictions, config.family, delta0));

    // Boosting iterations
    for _ in 0..config.n_trees {
        resource::checkpoint()?;
        // Subsample rows if needed (drawn first: the Huber threshold is taken over them)
        let sample_indices: Vec<usize> = if config.subsample < 1.0 {
            let n_subsample = ((n_samples as f64) * config.subsample).ceil() as usize;
            let mut indices: Vec<usize> = (0..n_samples).collect();

            // Fisher-Yates shuffle and take first n_subsample
            for i in 0..n_subsample.min(indices.len()) {
                let j = i + lcg_random(&mut rng_state) % (indices.len() - i);
                indices.swap(i, j);
            }
            indices.truncate(n_subsample);
            indices
        } else {
            all_rows.clone()
        };

        // Compute negative gradient (pseudo-residuals)
        let delta = huber_threshold(&y, &predictions, &sample_indices, config);
        let residuals = compute_negative_gradient(&y, &predictions, config.family, delta);
        let (sample_x, sample_residuals) = if config.subsample < 1.0 {
            let sub_r: Array1<f64> = sample_indices.iter().map(|&i| residuals[i]).collect();
            (select_rows(&x, &sample_indices), sub_r)
        } else {
            (x.to_owned(), residuals)
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

        // The family's terminal-node inputs over the sampled rows
        let leaf_input: Vec<f64> = match config.family {
            GbmFamily::Gaussian => Vec::new(),
            GbmFamily::Laplace | GbmFamily::Huber => sample_indices
                .iter()
                .map(|&i| y[i] - predictions[i])
                .collect(),
            GbmFamily::Binomial => sample_indices
                .iter()
                .map(|&i| sigmoid(predictions[i]))
                .collect(),
        };
        let leaf = match config.family {
            GbmFamily::Gaussian => Leaf::Mean,
            GbmFamily::Laplace => Leaf::Median { r: &leaf_input },
            GbmFamily::Huber => Leaf::Huber {
                r: &leaf_input,
                delta,
            },
            GbmFamily::Binomial => Leaf::Newton { p: &leaf_input },
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
            &leaf,
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

        // Track loss (Huber: at this iteration's threshold)
        train_loss.push(compute_loss(&y, &predictions, config.family, delta));
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

/// The Huber threshold of an iteration: the type-7 `huber_alpha`-quantile of `|y - F|` over
/// `rows`, or the fixed `huber_delta` (also returned, unread, for the other families).
fn huber_threshold(
    y: &ArrayView1<f64>,
    predictions: &Array1<f64>,
    rows: &[usize],
    config: &GbmConfig,
) -> f64 {
    match (config.family, config.huber_alpha) {
        (GbmFamily::Huber, Some(alpha)) => {
            let mut abs_r: Vec<f64> = rows
                .iter()
                .map(|&i| (y[i] - predictions[i]).abs())
                .collect();
            abs_r.sort_unstable_by(f64::total_cmp);
            quantile_sorted(&abs_r, alpha, QuantileType::T7)
        }
        _ => config.huber_delta,
    }
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
    if config.family == GbmFamily::Huber {
        match config.huber_alpha {
            Some(a) if !(a > 0.0 && a < 1.0) => {
                return invalid(format!(
                    "GBM: huber_alpha must be in (0, 1), got {a}; None uses the fixed huber_delta"
                ));
            }
            None if !config.huber_delta.is_finite() || config.huber_delta <= 0.0 => {
                return invalid(format!(
                    "GBM: huber_delta must be finite and above 0, got {}",
                    config.huber_delta
                ));
            }
            _ => {}
        }
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

    /// A target whose centred sum of squares overflows is refused; a large finite one fits with
    /// finite importance shares.
    #[test]
    fn test_gbm_refuses_overflowing_target() {
        let x = Array2::from_shape_fn((40, 2), |(i, j)| (i * (j + 3) % 17) as f64 + 0.1 * j as f64);
        let y = Array1::from_shape_fn(40, |i| (i % 7) as f64 + 0.3 * (i % 3) as f64);
        let cfg = GbmConfig {
            n_trees: 10,
            seed: Some(3),
            ..Default::default()
        };
        let huge = y.mapv(|v| v * 1e200);
        let err = gbm(x.view(), huge.view(), &cfg).unwrap_err();
        assert!(
            matches!(&err, EconError::InvalidSpecification { message } if message.contains("rescale the target")),
            "{err}"
        );
        let big = y.mapv(|v| v * 1e70);
        let r = gbm(x.view(), big.view(), &cfg).unwrap();
        assert!(r.feature_importances.iter().all(|v| v.is_finite()));
        assert!((r.feature_importances.iter().sum::<f64>() - 1.0).abs() < 1e-12);
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
                c.huber_alpha = None;
                c.huber_delta = -1.0;
            },
            "huber_delta",
        );
        for a in [0.0, 1.0, f64::NAN] {
            bad(
                &|c| {
                    c.family = GbmFamily::Huber;
                    c.huber_alpha = Some(a);
                },
                "huber_alpha",
            );
        }
        // With the adaptive threshold, huber_delta is not read.
        let adaptive = GbmConfig {
            family: GbmFamily::Huber,
            huber_delta: -1.0,
            n_trees: 2,
            ..GbmConfig::default()
        };
        assert!(gbm(x.view(), y.view(), &adaptive).is_ok());
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

    /// 50 stumps on the `gbm_leaf_t4` design with gbm's `n.minobsinnode = 1, bag.fraction = 1`.
    fn leaf_t4_stumps(
        y_col: &str,
        family: GbmFamily,
    ) -> (p2a_validation::Case, Array1<f64>, GbmResult) {
        use crate::data::columns::NaPolicy;
        let (case, ds) = crate::test_support::load_case(match family {
            GbmFamily::Binomial => "gbm_bernoulli_stumps",
            _ => "gbm_laplace_stumps",
        });
        let (x, y, _) =
            crate::ml::extract_xy(&ds, y_col, &["x0", "x1", "x2"], NaPolicy::Fail).unwrap();
        let mut config = GbmConfig::default();
        config.n_trees = 50;
        config.learning_rate = 0.1;
        config.max_depth = 1;
        config.min_samples_split = 2;
        config.subsample = 1.0;
        config.colsample_bytree = 1.0;
        config.family = family;
        let r = gbm(x.view(), y.view(), &config).unwrap();
        (case, y, r)
    }

    fn root_features(r: &GbmResult) -> Vec<f64> {
        r.trees
            .iter()
            .map(|t| match &t.root {
                Some(GbmNode::Split { feature_index, .. }) => *feature_index as f64,
                _ => -1.0,
            })
            .collect()
    }

    #[test]
    fn test_validate_gbm_laplace_stumps() {
        // ML-94/WT-1. Each Laplace leaf is the median of y - F over its rows (gbm's terminal-node
        // estimate). Before: the leaves were the mean sign of the residuals, so 50 trees at rate
        // 0.1 could not move the fit more than 5 from F0 = 704.88: fitted range [700.17, 707.14],
        // MAE 276.94 (R: [198.18, 1139.49], MAE 52.97).
        use p2a_validation::{Check, Tol};
        let (case, y, r) = leaf_t4_stumps("y_lad", GbmFamily::Laplace);
        let mae = y
            .iter()
            .zip(&r.predictions)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / y.len() as f64;
        let tol = Tol::coef(200);
        Check::new(&case)
            .scalar("init_f", r.init_prediction, tol)
            .vector("fitted", &r.predictions, tol)
            .vector("split_var", &root_features(&r), Tol::exact())
            .named("influence", &["x0", "x1", "x2"], &r.relative_influence, tol)
            .scalar("mae", mae, tol)
            .finish();
    }

    #[test]
    fn test_validate_gbm_bernoulli_stumps() {
        // ML-94. Each Bernoulli leaf is the Newton step sum(y - p) / sum(p (1 - p)) over its rows.
        // Before: the mean of y - p, a gradient step: probability range [0.442, 0.789]
        // (R: [0.190, 0.911]).
        use p2a_validation::{Check, Tol};
        let (case, _, r) = leaf_t4_stumps("y_bin", GbmFamily::Binomial);
        let tol = Tol::coef(200);
        Check::new(&case)
            .scalar("init_f", r.init_prediction, tol)
            .vector("fitted_prob", &r.predictions, tol)
            .vector("split_var", &root_features(&r), Tol::exact())
            .named("influence", &["x0", "x1", "x2"], &r.relative_influence, tol)
            .finish();
    }

    #[test]
    fn test_gbm_predict_matches_training_fit_every_family() {
        // The leaves are estimated on the in-bag rows; gbm_predict routes every row through the
        // same trees, so it reproduces the training predictions exactly.
        let n = 120;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| {
            ((i * (2 * j + 5) + 3 * j) % 37) as f64 / 7.0 + 0.01 * j as f64
        });
        let y_reg = Array1::from_shape_fn(n, |i| {
            3.0 * x[[i, 0]] - x[[i, 1]] + ((i * 29) % 13) as f64 / 4.0 - 1.5
        });
        let y_bin = Array1::from_shape_fn(n, |i| {
            if x[[i, 0]] + ((i * 17) % 11) as f64 / 3.0 > 4.5 {
                1.0
            } else {
                0.0
            }
        });
        for family in [
            GbmFamily::Gaussian,
            GbmFamily::Binomial,
            GbmFamily::Laplace,
            GbmFamily::Huber,
        ] {
            let y = if family == GbmFamily::Binomial {
                &y_bin
            } else {
                &y_reg
            };
            let mut config = GbmConfig::default();
            config.n_trees = 30;
            config.max_depth = 3;
            config.subsample = 0.7;
            config.family = family;
            config.seed = Some(11);
            let r = gbm(x.view(), y.view(), &config).unwrap();
            let p = gbm_predict(&r, x.view()).unwrap();
            assert_eq!(p, r.predictions, "{family}");
        }
    }

    /// The `gbm_leaf_t4` design: x (200 x 3) and the large-scale target `y_lad`.
    fn leaf_t4_design() -> (Array2<f64>, Array1<f64>) {
        use crate::data::columns::NaPolicy;
        let (_, ds) = crate::test_support::load_case("gbm_laplace_stumps");
        let (x, y, _) =
            crate::ml::extract_xy(&ds, "y_lad", &["x0", "x1", "x2"], NaPolicy::Fail).unwrap();
        (x, y)
    }

    fn stumps(family: GbmFamily) -> GbmConfig {
        let mut c = GbmConfig::default();
        c.n_trees = 50;
        c.learning_rate = 0.1;
        c.max_depth = 1;
        c.family = family;
        c
    }

    /// Bits of the left/right path that leads `row` to its leaf.
    fn leaf_path(node: &GbmNode, row: ArrayView1<f64>) -> u64 {
        let mut path = 1u64;
        let mut n = node;
        while let GbmNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } = n
        {
            let go_left = row[*feature_index] <= *threshold;
            path = path * 2 + u64::from(!go_left);
            n = if go_left { left } else { right };
        }
        path
    }

    #[test]
    fn test_gbm_huber_leaf_is_friedman_algorithm_4() {
        // WT-1. Friedman (2001), Algorithm 4: F0 = median(y), delta = the 0.9-quantile of
        // |y - F0|, and each leaf is median(r) + mean(sign(r - med) min(delta, |r - med|)) over
        // its rows, r = y - F0. Before: F0 = mean(y) and the leaf was the mean clipped residual.
        let n = 150;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            ((i * (3 * j + 7) + 5 * j) % 41) as f64 / 4.0 + 0.003 * i as f64
        });
        let y = Array1::from_shape_fn(n, |i| {
            let noise = ((i * 37) % 23) as f64 - 11.0;
            let outlier = if i % 17 == 0 { 300.0 } else { 0.0 };
            20.0 * x[[i, 0]] - 5.0 * x[[i, 1]] + noise + outlier
        });
        let mut config = GbmConfig::default();
        config.n_trees = 1;
        config.learning_rate = 1.0;
        config.max_depth = 2;
        config.family = GbmFamily::Huber;
        let r = gbm(x.view(), y.view(), &config).unwrap();

        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            let m = v.len();
            if m % 2 == 1 {
                v[m / 2]
            } else {
                (v[m / 2 - 1] + v[m / 2]) / 2.0
            }
        };
        let f0 = median(&mut y.to_vec());
        assert_eq!(r.init_prediction, f0);
        let resid: Vec<f64> = y.iter().map(|v| v - f0).collect();
        // Type-7 0.9-quantile of |r|, by its definition.
        let mut abs_r: Vec<f64> = resid.iter().map(|v| v.abs()).collect();
        abs_r.sort_by(f64::total_cmp);
        let h = (n - 1) as f64 * 0.9;
        let lo = h.floor() as usize;
        let delta = abs_r[lo] + (h - lo as f64) * (abs_r[(lo + 1).min(n - 1)] - abs_r[lo]);

        let root = r.trees[0].root.as_ref().unwrap();
        let paths: Vec<u64> = (0..n).map(|i| leaf_path(root, x.row(i))).collect();
        let mut leaves: Vec<u64> = paths.clone();
        leaves.sort_unstable();
        leaves.dedup();
        assert!(leaves.len() >= 3, "{} leaves", leaves.len());
        for leaf in leaves {
            let rows: Vec<usize> = (0..n).filter(|&i| paths[i] == leaf).collect();
            let mut rl: Vec<f64> = rows.iter().map(|&i| resid[i]).collect();
            let med = median(&mut rl.clone());
            let step = rl
                .iter_mut()
                .map(|v| (*v - med).signum() * delta.min((*v - med).abs()))
                .sum::<f64>()
                / rows.len() as f64;
            let want = f0 + med + step;
            for &i in &rows {
                let got = r.predictions[i];
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "row {i}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn test_gbm_huber_and_laplace_are_scale_equivariant() {
        // WT-1. A fit on 1000 y is 1000 times the fit on y. Before: Huber clipped at a fixed
        // delta of 1 in the units of y, and Laplace leaves were mean signs.
        let (x, y) = leaf_t4_design();
        let big = y.mapv(|v| 1000.0 * v);
        for family in [GbmFamily::Huber, GbmFamily::Laplace] {
            let c = stumps(family);
            let a = gbm(x.view(), y.view(), &c).unwrap();
            let b = gbm(x.view(), big.view(), &c).unwrap();
            for (p, q) in a.predictions.iter().zip(&b.predictions) {
                assert!(
                    (1000.0 * p - q).abs() <= 1e-9 * q.abs(),
                    "{family}: {} vs {q}",
                    1000.0 * p
                );
            }
        }
    }

    #[test]
    fn test_gbm_huber_fits_a_large_scale_target() {
        // WT-1. Friedman's Huber M-TreeBoost (emulated in R: MAE 42.0, range [176.9, 1217.5]).
        // Before: the fit stayed within 5 of F0 (MAE about 277).
        let (x, y) = leaf_t4_design();
        let r = gbm(x.view(), y.view(), &stumps(GbmFamily::Huber)).unwrap();
        let mae = y
            .iter()
            .zip(&r.predictions)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / y.len() as f64;
        assert!(mae <= 50.0, "MAE {mae}");
        let lo = r.predictions.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = r
            .predictions
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(lo <= 200.0 && hi >= 1100.0, "range [{lo}, {hi}]");
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
    fn test_gbm_tree_depth_is_capped() {
        // Trees stop at depth 512 whatever max_depth says. Before: a chain as deep as the rows.
        let depth = on_small_stack(|| {
            let (x, y) = chain_data(2_000);
            let mut c = GbmConfig::default();
            c.n_trees = 1;
            c.max_depth = 10_000;
            c.min_samples_split = 2;
            let r = gbm(x.view(), y.view(), &c).unwrap();
            let (mut max, mut stack) = (0, vec![(r.trees[0].root.clone().unwrap(), 0usize)]);
            while let Some((node, d)) = stack.pop() {
                max = max.max(d);
                if let GbmNode::Split { left, right, .. } = node {
                    stack.push((*left, d + 1));
                    stack.push((*right, d + 1));
                }
            }
            max
        });
        assert_eq!(depth, MAX_TREE_DEPTH);
    }
}
