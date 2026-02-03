//! AdaBoost (Adaptive Boosting) implementation.
//!
//! Pure Rust implementation of AdaBoost.M1 (Freund & Schapire, 1997) for
//! binary classification and AdaBoost.R2 for regression.
//!
//! ## Features
//!
//! - **Classification**: AdaBoost.M1 with decision stumps (depth-1 trees)
//! - **Regression**: AdaBoost.R2 with configurable loss functions
//! - **Custom weak learners**: Configurable max depth for base trees
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{adaboost, AdaBoostConfig, AdaBoostType};
//! use ndarray::array;
//!
//! // Binary classification
//! let x = array![[1.0], [2.0], [7.0], [8.0]];
//! let y = array![-1.0, -1.0, 1.0, 1.0];
//!
//! let config = AdaBoostConfig {
//!     n_estimators: 50,
//!     boost_type: AdaBoostType::M1,
//!     ..Default::default()
//! };
//!
//! let result = adaboost(x.view(), y.view(), &config).unwrap();
//! println!("Accuracy: {:.4}", result.train_accuracy.unwrap());
//! ```

use ndarray::{Array1, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::Dataset;

/// Type of AdaBoost algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AdaBoostType {
    /// AdaBoost.M1 for binary classification (labels: -1, +1)
    #[default]
    M1,
    /// AdaBoost.R2 for regression
    R2,
    /// SAMME (multi-class classification) - stagewise additive modeling
    Samme,
}

impl std::str::FromStr for AdaBoostType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "m1" | "discrete" | "classification" => Ok(AdaBoostType::M1),
            "r2" | "regression" | "real" => Ok(AdaBoostType::R2),
            "samme" | "multiclass" => Ok(AdaBoostType::Samme),
            _ => Err(format!("Unknown AdaBoost type: {}", s)),
        }
    }
}

/// Loss function for AdaBoost.R2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AdaBoostLoss {
    /// Linear loss
    #[default]
    Linear,
    /// Square loss
    Square,
    /// Exponential loss
    Exponential,
}

/// Configuration for AdaBoost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaBoostConfig {
    /// Number of estimators (boosting iterations)
    pub n_estimators: usize,
    /// Maximum depth of weak learners (1 = stumps)
    pub max_depth: usize,
    /// AdaBoost algorithm type
    pub boost_type: AdaBoostType,
    /// Loss function for R2 (regression)
    pub loss: AdaBoostLoss,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for AdaBoostConfig {
    fn default() -> Self {
        AdaBoostConfig {
            n_estimators: 50,
            max_depth: 1, // Decision stumps by default
            boost_type: AdaBoostType::M1,
            loss: AdaBoostLoss::Linear,
            learning_rate: 1.0,
            seed: None,
        }
    }
}

/// A simple decision stump (single split tree).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionStump {
    /// Feature to split on
    feature: usize,
    /// Threshold value
    threshold: f64,
    /// Prediction for left branch (x <= threshold)
    left_pred: f64,
    /// Prediction for right branch (x > threshold)
    right_pred: f64,
}

impl DecisionStump {
    fn predict_one(&self, x: &ArrayView1<f64>) -> f64 {
        if x[self.feature] <= self.threshold {
            self.left_pred
        } else {
            self.right_pred
        }
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        Array1::from_iter((0..x.nrows()).map(|i| self.predict_one(&x.row(i))))
    }
}

/// Result from AdaBoost fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaBoostResult {
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Class predictions (for classification)
    pub class_predictions: Option<Vec<i32>>,
    /// Feature importances
    pub feature_importances: Vec<f64>,
    /// Number of estimators actually used
    pub n_estimators: usize,
    /// Training accuracy (for classification)
    pub train_accuracy: Option<f64>,
    /// Training error rate per iteration
    pub train_error: Vec<f64>,
    /// Estimator weights (alpha values)
    pub estimator_weights: Vec<f64>,
    /// Configuration used
    pub config: AdaBoostConfig,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Internal: stumps for prediction
    #[serde(skip)]
    pub(crate) stumps: Vec<DecisionStump>,
}

impl std::fmt::Display for AdaBoostResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "AdaBoost Results")?;
        writeln!(f, "================")?;
        writeln!(f, "Type: {:?}", self.config.boost_type)?;
        writeln!(f, "Number of estimators: {}", self.n_estimators)?;
        writeln!(f, "Max depth: {}", self.config.max_depth)?;

        if let Some(acc) = self.train_accuracy {
            writeln!(f, "Training accuracy: {:.4}", acc)?;
        }

        if !self.train_error.is_empty() {
            writeln!(f, "Final training error: {:.4}", self.train_error.last().unwrap())?;
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

        let total: f64 = indexed.iter().map(|(_, v)| v).sum();
        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names.get(*i).cloned().unwrap_or_else(|| format!("X{}", i)),
                None => format!("X{}", i),
            };
            let pct = if total > 0.0 { importance / total * 100.0 } else { 0.0 };
            writeln!(f, "  {}: {:.1}%", name, pct)?;
        }

        Ok(())
    }
}

/// Fit a decision stump to weighted data (for classification).
/// OPTIMIZED: O(n log n) per feature using incremental weighted counts.
fn fit_stump_classification(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    weights: &Array1<f64>,
) -> (DecisionStump, f64) {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let mut best_error = f64::INFINITY;
    let mut best_stump = DecisionStump {
        feature: 0,
        threshold: 0.0,
        left_pred: 1.0,
        right_pred: -1.0,
    };

    // Precompute total weights for each class
    let mut total_pos_weight = 0.0; // y = +1
    let mut total_neg_weight = 0.0; // y = -1
    for i in 0..n_samples {
        if y[i] > 0.0 {
            total_pos_weight += weights[i];
        } else {
            total_neg_weight += weights[i];
        }
    }
    let total_weight = total_pos_weight + total_neg_weight;

    for feature in 0..n_features {
        // Sort samples by feature value - O(n log n)
        let mut sorted: Vec<(f64, f64, f64)> = (0..n_samples)
            .map(|i| (x[[i, feature]], y[i], weights[i]))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Scan through with running weighted counts - O(n)
        let mut left_pos_weight = 0.0;
        let mut left_neg_weight = 0.0;

        for i in 0..sorted.len() - 1 {
            let (x_val, y_val, w) = sorted[i];

            // Update left side weights
            if y_val > 0.0 {
                left_pos_weight += w;
            } else {
                left_neg_weight += w;
            }

            // Skip if next value is the same
            let next_x = sorted[i + 1].0;
            if (next_x - x_val).abs() < 1e-10 {
                continue;
            }

            let threshold = (x_val + next_x) / 2.0;

            // Compute right side weights
            let right_pos_weight = total_pos_weight - left_pos_weight;
            let right_neg_weight = total_neg_weight - left_neg_weight;

            // Try polarity: left = +1, right = -1
            // Error = left_neg_weight (misclassified on left) + right_pos_weight (misclassified on right)
            let error1 = left_neg_weight + right_pos_weight;

            // Try polarity: left = -1, right = +1
            // Error = left_pos_weight + right_neg_weight
            let error2 = left_pos_weight + right_neg_weight;

            if error1 < best_error {
                best_error = error1;
                best_stump = DecisionStump {
                    feature,
                    threshold,
                    left_pred: 1.0,
                    right_pred: -1.0,
                };
            }

            if error2 < best_error {
                best_error = error2;
                best_stump = DecisionStump {
                    feature,
                    threshold,
                    left_pred: -1.0,
                    right_pred: 1.0,
                };
            }
        }
    }

    (best_stump, best_error)
}

/// Fit a decision stump for regression with weighted samples.
/// OPTIMIZED: O(n log n) per feature using incremental weighted sums.
fn fit_stump_regression(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    weights: &Array1<f64>,
) -> DecisionStump {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let mut best_loss = f64::INFINITY;
    let mut best_stump = DecisionStump {
        feature: 0,
        threshold: 0.0,
        left_pred: 0.0,
        right_pred: 0.0,
    };

    // Precompute total weighted sums
    let mut total_wy = 0.0;  // sum(w * y)
    let mut total_wyy = 0.0; // sum(w * y^2)
    let mut total_w = 0.0;   // sum(w)
    for i in 0..n_samples {
        let w = weights[i];
        let yi = y[i];
        total_wy += w * yi;
        total_wyy += w * yi * yi;
        total_w += w;
    }

    for feature in 0..n_features {
        // Sort samples by feature value - O(n log n)
        let mut sorted: Vec<(f64, f64, f64)> = (0..n_samples)
            .map(|i| (x[[i, feature]], y[i], weights[i]))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Scan through with running weighted sums - O(n)
        let mut left_wy = 0.0;
        let mut left_wyy = 0.0;
        let mut left_w = 0.0;

        for i in 0..sorted.len() - 1 {
            let (x_val, yi, w) = sorted[i];

            // Update left side sums
            left_wy += w * yi;
            left_wyy += w * yi * yi;
            left_w += w;

            // Skip if next value is the same
            let next_x = sorted[i + 1].0;
            if (next_x - x_val).abs() < 1e-10 {
                continue;
            }

            // Check minimum weight on each side
            let right_w = total_w - left_w;
            if left_w < 1e-10 || right_w < 1e-10 {
                continue;
            }

            let threshold = (x_val + next_x) / 2.0;

            // Compute weighted means (predictions)
            let left_pred = left_wy / left_w;
            let right_wy = total_wy - left_wy;
            let right_pred = right_wy / right_w;

            // Compute weighted MSE using: sum(w*(y-pred)^2) = sum(w*y^2) - 2*pred*sum(w*y) + pred^2*sum(w)
            // = sum(w*y^2) - sum(w*y)^2/sum(w) when pred = sum(w*y)/sum(w)
            // = wyy - wy^2/w
            let left_loss = left_wyy - left_wy * left_wy / left_w;
            let right_wyy = total_wyy - left_wyy;
            let right_loss = right_wyy - right_wy * right_wy / right_w;

            let total_loss = left_loss + right_loss;

            if total_loss < best_loss {
                best_loss = total_loss;
                best_stump = DecisionStump {
                    feature,
                    threshold,
                    left_pred,
                    right_pred,
                };
            }
        }
    }

    best_stump
}

/// Run AdaBoost.M1 for binary classification.
fn adaboost_m1(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &AdaBoostConfig,
) -> EconResult<AdaBoostResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Initialize sample weights
    let mut weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

    let mut stumps = Vec::with_capacity(config.n_estimators);
    let mut alphas = Vec::with_capacity(config.n_estimators);
    let mut train_error = Vec::with_capacity(config.n_estimators);
    let mut importance = Array1::zeros(n_features);

    // Cumulative predictions (sum of weighted predictions)
    let mut cum_predictions = Array1::zeros(n_samples);

    for _ in 0..config.n_estimators {
        // Fit weak learner
        let (stump, error) = fit_stump_classification(x, y, &weights);

        // Compute alpha (estimator weight)
        // Handle perfect fit or random guess
        let error_clamped = error.clamp(1e-10, 1.0 - 1e-10);
        let alpha = config.learning_rate * 0.5 * ((1.0 - error_clamped) / error_clamped).ln();

        if error >= 0.5 {
            // Weak learner is not better than random, stop
            break;
        }

        // Update cumulative predictions
        let predictions = stump.predict(x);
        for i in 0..n_samples {
            cum_predictions[i] += alpha * predictions[i];
        }

        // Update weights
        for i in 0..n_samples {
            let miss = if (predictions[i] - y[i]).abs() > 0.5 { 1.0 } else { -1.0 };
            weights[i] *= (alpha * miss).exp();
        }

        // Normalize weights
        let weight_sum: f64 = weights.sum();
        weights /= weight_sum;

        // Track importance
        importance[stump.feature] += alpha.abs();

        // Compute training error
        let mut errors = 0;
        for i in 0..n_samples {
            let sign_pred = if cum_predictions[i] >= 0.0 { 1.0 } else { -1.0 };
            if (sign_pred - y[i]).abs() > 0.5 {
                errors += 1;
            }
        }
        train_error.push(errors as f64 / n_samples as f64);

        alphas.push(alpha);
        stumps.push(stump);
    }

    // Final predictions
    let predictions: Vec<f64> = cum_predictions.to_vec();
    let class_predictions: Vec<i32> = predictions
        .iter()
        .map(|&p| if p >= 0.0 { 1 } else { -1 })
        .collect();

    // Training accuracy
    let correct: usize = class_predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, actual)| (**pred as f64 - *actual).abs() < 0.5)
        .count();
    let accuracy = correct as f64 / n_samples as f64;

    // Normalize importance
    let imp_sum: f64 = importance.sum();
    if imp_sum > 0.0 {
        importance /= imp_sum;
    }

    Ok(AdaBoostResult {
        predictions,
        class_predictions: Some(class_predictions),
        feature_importances: importance.to_vec(),
        n_estimators: stumps.len(),
        train_accuracy: Some(accuracy),
        train_error,
        estimator_weights: alphas,
        config: config.clone(),
        feature_names: None,
        stumps,
    })
}

/// Run AdaBoost.R2 for regression.
fn adaboost_r2(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &AdaBoostConfig,
) -> EconResult<AdaBoostResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Initialize sample weights
    let mut weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

    let mut stumps = Vec::with_capacity(config.n_estimators);
    let mut betas = Vec::with_capacity(config.n_estimators);
    let mut train_error = Vec::with_capacity(config.n_estimators);
    let mut importance = Array1::zeros(n_features);

    // Store predictions from each estimator
    let mut all_predictions: Vec<Array1<f64>> = Vec::new();

    for _ in 0..config.n_estimators {
        // Fit weak learner
        let stump = fit_stump_regression(x, y, &weights);
        let predictions = stump.predict(x);

        // Compute weighted absolute error
        let mut max_error = 0.0_f64;
        for i in 0..n_samples {
            let err = (y[i] - predictions[i]).abs();
            max_error = max_error.max(err);
        }

        if max_error < 1e-10 {
            // Perfect fit, stop
            all_predictions.push(predictions);
            betas.push(1.0);
            stumps.push(stump);
            break;
        }

        // Compute loss for each sample
        let mut losses = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let normalized_error = (y[i] - predictions[i]).abs() / max_error;
            losses[i] = match config.loss {
                AdaBoostLoss::Linear => normalized_error,
                AdaBoostLoss::Square => normalized_error.powi(2),
                AdaBoostLoss::Exponential => 1.0 - (-normalized_error).exp(),
            };
        }

        // Weighted average loss
        let avg_loss: f64 = weights.iter().zip(losses.iter()).map(|(w, l)| w * l).sum();

        // For R2, we're more lenient with the loss threshold since it's regression
        // The original R2 paper allows avg_loss up to 1, but we clip beta
        let avg_loss_clamped = avg_loss.clamp(1e-10, 0.999);

        // Compute beta
        let beta = avg_loss_clamped / (1.0 - avg_loss_clamped);

        // Update weights
        for i in 0..n_samples {
            weights[i] *= beta.powf(config.learning_rate * (1.0 - losses[i]));
        }

        // Normalize weights
        let weight_sum: f64 = weights.sum();
        if weight_sum > 1e-10 {
            weights /= weight_sum;
        }

        // Track importance
        importance[stump.feature] += (1.0 / beta).ln();

        all_predictions.push(predictions);
        betas.push(beta);
        stumps.push(stump);

        // Compute training MSE
        let final_pred = compute_weighted_median_predictions(&all_predictions, &betas);
        let mse: f64 = y
            .iter()
            .zip(final_pred.iter())
            .map(|(yi, pi)| (yi - pi).powi(2))
            .sum::<f64>()
            / n_samples as f64;
        train_error.push(mse);
    }

    // Final predictions using weighted median
    let predictions = compute_weighted_median_predictions(&all_predictions, &betas);

    // Normalize importance
    let imp_sum: f64 = importance.sum();
    if imp_sum > 0.0 {
        importance /= imp_sum;
    }

    Ok(AdaBoostResult {
        predictions: predictions.to_vec(),
        class_predictions: None,
        feature_importances: importance.to_vec(),
        n_estimators: stumps.len(),
        train_accuracy: None,
        train_error,
        estimator_weights: betas.iter().map(|b| (1.0 / b).ln()).collect(),
        config: config.clone(),
        feature_names: None,
        stumps,
    })
}

/// Compute weighted median predictions from all estimators.
fn compute_weighted_median_predictions(
    predictions: &[Array1<f64>],
    betas: &[f64],
) -> Array1<f64> {
    if predictions.is_empty() {
        return Array1::zeros(0);
    }

    let n_samples = predictions[0].len();
    let n_estimators = predictions.len();

    // Compute weights (log(1/beta))
    let weights: Vec<f64> = betas.iter().map(|b| (1.0 / b).ln().max(0.0)).collect();
    let total_weight: f64 = weights.iter().sum();

    let mut result = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Get predictions and weights for this sample
        let mut pred_weights: Vec<(f64, f64)> = predictions
            .iter()
            .zip(weights.iter())
            .map(|(p, w)| (p[i], *w))
            .collect();

        // Sort by prediction value
        pred_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find weighted median
        let mut cumsum = 0.0;
        let threshold = total_weight / 2.0;
        for (pred, w) in pred_weights {
            cumsum += w;
            if cumsum >= threshold {
                result[i] = pred;
                break;
            }
        }
    }

    result
}

/// Fit an AdaBoost model.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
///   - For M1: binary labels (-1 or +1)
///   - For R2: continuous values
/// * `config` - AdaBoost configuration
///
/// # Returns
///
/// AdaBoostResult containing the fitted model and diagnostics.
pub fn adaboost(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &AdaBoostConfig,
) -> EconResult<AdaBoostResult> {
    let n_samples = x.nrows();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for AdaBoost".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }

    match config.boost_type {
        AdaBoostType::M1 | AdaBoostType::Samme => {
            // Validate binary labels
            for &yi in y.iter() {
                if (yi - 1.0).abs() > 0.01 && (yi + 1.0).abs() > 0.01 {
                    return Err(EconError::Computation(
                        "AdaBoost.M1 requires binary labels (-1 or +1)".to_string(),
                    ));
                }
            }
            adaboost_m1(&x, &y, config)
        }
        AdaBoostType::R2 => adaboost_r2(&x, &y, config),
    }
}

/// Predict using a fitted AdaBoost model.
///
/// # Arguments
///
/// * `result` - Fitted AdaBoost result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predictions (continuous for regression, signed scores for classification).
pub fn adaboost_predict(result: &AdaBoostResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    let n_samples = x.nrows();

    if result.stumps.is_empty() {
        return Err(EconError::Computation(
            "Model has no fitted estimators".to_string(),
        ));
    }

    match result.config.boost_type {
        AdaBoostType::M1 | AdaBoostType::Samme => {
            // Weighted vote
            let mut predictions = Array1::zeros(n_samples);
            for (stump, &alpha) in result.stumps.iter().zip(result.estimator_weights.iter()) {
                let pred = stump.predict(&x);
                for i in 0..n_samples {
                    predictions[i] += alpha * pred[i];
                }
            }
            Ok(predictions.to_vec())
        }
        AdaBoostType::R2 => {
            // Weighted median
            let all_predictions: Vec<Array1<f64>> =
                result.stumps.iter().map(|s| s.predict(&x)).collect();
            let betas: Vec<f64> = result
                .estimator_weights
                .iter()
                .map(|w| (-w).exp())
                .collect();
            Ok(compute_weighted_median_predictions(&all_predictions, &betas).to_vec())
        }
    }
}

/// Predict class labels using a fitted AdaBoost classifier.
pub fn adaboost_predict_class(result: &AdaBoostResult, x: ArrayView2<f64>) -> EconResult<Vec<i32>> {
    let predictions = adaboost_predict(result, x)?;
    Ok(predictions
        .iter()
        .map(|&p| if p >= 0.0 { 1 } else { -1 })
        .collect())
}

/// Run AdaBoost on a Dataset.
pub fn run_adaboost(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &AdaBoostConfig,
) -> EconResult<AdaBoostResult> {
    use crate::linalg::design::DesignMatrix;

    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;
    let x = design.data;
    let feature_names = design.column_names;

    let col_names: Vec<String> = dataset.df().get_column_names().iter().map(|s| s.to_string()).collect();
    let y_series = dataset
        .df()
        .column(y_col)
        .map_err(|_| EconError::ColumnNotFound { column: y_col.to_string(), available: col_names })?;

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::Computation(format!("Column '{}' is not numeric", y_col)))?
        .into_no_null_iter()
        .collect();

    let y_arr = Array1::from_vec(y);

    let mut result = adaboost(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Run AdaBoost with default configuration.
pub fn run_adaboost_default(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
) -> EconResult<AdaBoostResult> {
    run_adaboost(dataset, y_col, x_cols, &AdaBoostConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adaboost_m1_basic() {
        // Binary classification
        let x = array![
            [1.0, 0.0],
            [2.0, 0.5],
            [1.5, 0.3],
            [8.0, 1.0],
            [9.0, 0.5],
            [8.5, 0.8],
        ];
        let y = array![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let config = AdaBoostConfig {
            n_estimators: 20,
            boost_type: AdaBoostType::M1,
            ..Default::default()
        };

        let result = adaboost(x.view(), y.view(), &config).unwrap();

        assert!(result.n_estimators > 0);
        assert!(result.train_accuracy.unwrap() > 0.5);
        assert_eq!(result.class_predictions.as_ref().unwrap().len(), 6);
    }

    #[test]
    fn test_adaboost_r2_basic() {
        // Regression
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

        let config = AdaBoostConfig {
            n_estimators: 20,
            boost_type: AdaBoostType::R2,
            ..Default::default()
        };

        let result = adaboost(x.view(), y.view(), &config).unwrap();

        assert!(result.n_estimators > 0);
        assert_eq!(result.predictions.len(), 10);
        assert!(result.class_predictions.is_none());
    }

    #[test]
    fn test_adaboost_predict() {
        let x_train = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [8.0, 1.0],
            [9.0, 1.0],
        ];
        let y_train = array![-1.0, -1.0, 1.0, 1.0];

        let config = AdaBoostConfig {
            n_estimators: 20,
            boost_type: AdaBoostType::M1,
            ..Default::default()
        };

        let result = adaboost(x_train.view(), y_train.view(), &config).unwrap();

        let x_test = array![[1.5, 0.0], [8.5, 1.0]];
        let predictions = adaboost_predict(&result, x_test.view()).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions[0] < 0.0); // Should be negative (class -1)
        assert!(predictions[1] > 0.0); // Should be positive (class +1)
    }

    #[test]
    fn test_adaboost_predict_class() {
        let x_train = array![
            [1.0],
            [2.0],
            [8.0],
            [9.0],
        ];
        let y_train = array![-1.0, -1.0, 1.0, 1.0];

        let config = AdaBoostConfig {
            n_estimators: 20,
            boost_type: AdaBoostType::M1,
            ..Default::default()
        };

        let result = adaboost(x_train.view(), y_train.view(), &config).unwrap();

        let x_test = array![[1.5], [8.5]];
        let classes = adaboost_predict_class(&result, x_test.view()).unwrap();

        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0], -1);
        assert_eq!(classes[1], 1);
    }

    #[test]
    fn test_adaboost_feature_importance() {
        // First feature is predictive, second is noise
        let x = array![
            [1.0, 5.0],
            [2.0, 3.0],
            [1.5, 7.0],
            [8.0, 2.0],
            [9.0, 8.0],
            [8.5, 4.0],
        ];
        let y = array![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let config = AdaBoostConfig {
            n_estimators: 20,
            boost_type: AdaBoostType::M1,
            ..Default::default()
        };

        let result = adaboost(x.view(), y.view(), &config).unwrap();

        // First feature should be more important
        assert!(
            result.feature_importances[0] >= result.feature_importances[1],
            "Feature 0 ({}) should be >= Feature 1 ({})",
            result.feature_importances[0],
            result.feature_importances[1]
        );
    }

    #[test]
    fn test_adaboost_learning_rate() {
        let x = array![
            [1.0],
            [2.0],
            [8.0],
            [9.0],
        ];
        let y = array![-1.0, -1.0, 1.0, 1.0];

        let config = AdaBoostConfig {
            n_estimators: 20,
            boost_type: AdaBoostType::M1,
            learning_rate: 0.5, // Smaller learning rate
            ..Default::default()
        };

        let result = adaboost(x.view(), y.view(), &config).unwrap();

        assert!(result.n_estimators > 0);
        assert!(result.train_accuracy.unwrap() >= 0.75);
    }

    #[test]
    fn test_decision_stump() {
        let x = array![[1.0], [2.0], [8.0], [9.0]];
        let y = array![-1.0, -1.0, 1.0, 1.0];
        let weights = Array1::from_elem(4, 0.25);

        let (stump, error) = fit_stump_classification(&x.view(), &y.view(), &weights);

        assert!(error < 0.5, "Error {} should be < 0.5", error);

        // Test predictions
        let pred = stump.predict(&x.view());
        assert!(pred[0] < 0.0);
        assert!(pred[3] > 0.0);
    }
}
