//! Fast MBoost implementation with parallel feature evaluation and correct algorithms.
//!
//! Key optimizations:
//! 1. Parallel base learner evaluation across features
//! 2. Efficient componentwise linear with proper coefficient accumulation
//! 3. Vectorized residual updates
//! 4. Early stopping on a seeded holdout split
//!
//! Reference: Bühlmann & Hothorn (2007), "Boosting Algorithms: Regularization,
//! Prediction and Model Fitting"

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::lcg_random;
use super::mboost::check_step_size;
use crate::errors::{EconError, EconResult};
use crate::resource::{checkpoint, try_with_capacity};

/// Seed of the validation split when [`FastMboostConfig::seed`] is `None`.
pub const FAST_MBOOST_DEFAULT_SEED: u64 = 42;

/// Fast MBoost configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastMboostConfig {
    /// Number of boosting iterations
    pub m_stop: usize,
    /// Learning rate (nu) - shrinkage parameter. Must be finite and in
    /// (0, 1], as for [`mboost`](fn@crate::ml::mboost); [`fast_mboost`] refuses
    /// other values.
    pub nu: f64,
    /// Base learner type
    pub base_learner: FastMboostLearner,
    /// Loss function family
    pub family: FastMboostFamily,
    /// Stop when the validation risk has not improved for this many iterations; the model is
    /// truncated to the best iteration (0 = off)
    pub early_stopping_rounds: usize,
    /// Share of rows held out when early stopping is on (ceil, at least 1 row, at least 2
    /// training rows)
    pub validation_fraction: f64,
    /// Draws the validation split (default [`FAST_MBOOST_DEFAULT_SEED`]); unused without early
    /// stopping
    pub seed: Option<u64>,
}

impl Default for FastMboostConfig {
    fn default() -> Self {
        Self {
            m_stop: 100,
            nu: 0.1,
            base_learner: FastMboostLearner::ComponentwiseLinear,
            family: FastMboostFamily::Gaussian,
            early_stopping_rounds: 0,
            validation_fraction: 0.1,
            seed: None,
        }
    }
}

/// Base learner types
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FastMboostLearner {
    /// Componentwise linear (one feature at a time)
    #[default]
    ComponentwiseLinear,
    /// Small decision stumps
    Stump,
    /// Decision trees with limited depth
    Tree { max_depth: usize },
}

/// Loss function families
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FastMboostFamily {
    /// Gaussian (L2 loss)
    #[default]
    Gaussian,
    /// Binomial (logistic)
    Binomial,
    /// Huber (robust)
    Huber,
}

/// Internal learner representation
#[derive(Debug, Clone)]
enum FittedLearner {
    Linear {
        feature: usize,
        coef: f64,
        intercept: f64,
    },
    Stump {
        feature: usize,
        threshold: f64,
        left_val: f64,
        right_val: f64,
    },
    Tree {
        nodes: Vec<TreeNode>,
    },
}

#[derive(Debug, Clone)]
struct TreeNode {
    feature: Option<usize>,
    threshold: f64,
    left: Option<usize>,
    right: Option<usize>,
    value: f64,
}

/// Fast MBoost result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastMboostResult {
    /// Predictions on training data
    pub predictions: Vec<f64>,
    /// Feature selection frequency (normalized)
    pub feature_importances: Vec<f64>,
    /// Number of iterations used
    pub m_stop: usize,
    /// Training risk per iteration
    pub risk: Vec<f64>,
    /// Risk on the validation rows after each iteration that ran (early stopping only)
    pub val_risk: Option<Vec<f64>>,
    /// Rows (ascending) held out for early stopping
    #[serde(default)]
    pub validation_rows: Option<Vec<usize>>,
    /// 1-based iteration with the smallest validation risk; the model is truncated to it
    #[serde(default)]
    pub best_iteration: Option<usize>,
    /// Coefficients per feature (for linear learners)
    pub coefficients: Option<Vec<f64>>,
    /// Offset (intercept)
    pub offset: f64,
    /// Configuration used
    config: FastMboostConfig,
    #[serde(skip)]
    learners: Vec<FittedLearner>,
    /// Training means of the features, which the linear learners centre at
    #[serde(default)]
    x_means: Vec<f64>,
}

impl std::fmt::Display for FastMboostResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fast MBoost Model")?;
        writeln!(f, "=================")?;
        writeln!(f, "Iterations: {}", self.m_stop)?;
        writeln!(f, "Learning rate: {}", self.config.nu)?;
        writeln!(f, "Family: {:?}", self.config.family)?;
        if let Some(ref coefs) = self.coefficients {
            writeln!(f, "Offset: {:.4}", self.offset)?;
            writeln!(f, "Coefficients:")?;
            for (i, c) in coefs.iter().enumerate() {
                if c.abs() > 1e-10 {
                    writeln!(f, "  x{}: {:.4}", i, c)?;
                }
            }
        }
        Ok(())
    }
}

/// Train a fast MBoost model.
///
/// # Errors
///
/// - [`EconError::InsufficientData`] when `y.len() != x.nrows()`.
/// - [`EconError::InvalidSpecification`] for fewer than 2 rows, no predictor
///   columns, non-finite values in `x` or `y`, or a step size `nu` that is
///   not finite or lies outside (0, 1].
/// - [`EconError::ResourceLimit`] / [`EconError::Cancelled`] from the
///   resource guards: the `m_stop`-sized buffers exceed the allocation
///   ceiling, or the budget is cancelled or expires.
pub fn fast_mboost(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &FastMboostConfig,
) -> EconResult<FastMboostResult> {
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(EconError::InsufficientData {
            required: n,
            provided: y.len(),
            context: "X and y must have same number of rows".to_string(),
        });
    }
    if n < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!("x must have at least 2 rows (got {n})"),
        });
    }
    if p == 0 {
        return Err(EconError::InvalidSpecification {
            message: "x must have at least one predictor column".to_string(),
        });
    }
    // NaN would not panic (the sorts use total_cmp) but would boost into NaN
    // predictions; refuse it as mboost does
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
    check_step_size(config.nu)?;

    if config.early_stopping_rounds == 0 {
        // A strided y (say, a column of a row-major matrix) has no slice; copy it
        // once instead
        let y_owned;
        let y_slice: &[f64] = match y.as_slice() {
            Some(s) => s,
            None => {
                y_owned = y.to_vec();
                &y_owned
            }
        };
        let fit = boost_rows(x, y_slice, config, None)?;
        return Ok(fit.into_result(config, p, None));
    }

    let (train, validation) = validation_split(n, config)?;
    let x_train = x.select(Axis(0), &train);
    let y_train: Vec<f64> = train.iter().map(|&i| y[i]).collect();
    let x_val = x.select(Axis(0), &validation);
    let y_val: Vec<f64> = validation.iter().map(|&i| y[i]).collect();
    let fit = boost_rows(
        x_train.view(),
        &y_train,
        config,
        Some((x_val.view(), y_val.as_slice())),
    )?;
    let predictions = apply_learners(&fit.learners, config.nu, fit.offset, &fit.x_means, x);
    let mut result = fit.into_result(config, p, Some(validation));
    result.predictions = predictions;
    Ok(result)
}

/// Training and validation rows (each ascending) of an early-stopping fit: the rows are permuted
/// by a seeded Fisher-Yates loop and the first ceil(validation_fraction * n) positions are held out.
fn validation_split(n: usize, config: &FastMboostConfig) -> EconResult<(Vec<usize>, Vec<usize>)> {
    let f = config.validation_fraction;
    if !(f > 0.0 && f < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("validation_fraction must be in (0, 1) (got {f})"),
        });
    }
    let n_val = (f * n as f64).ceil() as usize;
    if n_val + 2 > n {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "validation_fraction {f} holds out {n_val} of {n} rows; at least 2 training rows are needed"
            ),
        });
    }
    let mut rows: Vec<usize> = (0..n).collect();
    let mut state = config.seed.unwrap_or(FAST_MBOOST_DEFAULT_SEED);
    for i in (1..n).rev() {
        let j = lcg_random(&mut state) % (i + 1);
        rows.swap(i, j);
    }
    let mut validation = rows[..n_val].to_vec();
    let mut train = rows[n_val..].to_vec();
    validation.sort_unstable();
    train.sort_unstable();
    Ok((train, validation))
}

/// One boosting run on the rows it is given, cut back to the best iteration when validation rows
/// are supplied.
struct BoostedRows {
    predictions: Vec<f64>,
    offset: f64,
    x_means: Vec<f64>,
    learners: Vec<FittedLearner>,
    risk: Vec<f64>,
    val_risk: Option<Vec<f64>>,
    best_iteration: Option<usize>,
}

impl BoostedRows {
    fn into_result(
        self,
        config: &FastMboostConfig,
        p: usize,
        validation_rows: Option<Vec<usize>>,
    ) -> FastMboostResult {
        // Accumulated coefficients for componentwise linear and selection counts, from the kept
        // learners
        let mut accumulated_coefs = vec![0.0; p];
        let mut feature_selection_count = vec![0usize; p];
        for learner in &self.learners {
            match learner {
                FittedLearner::Linear { feature, coef, .. } => {
                    feature_selection_count[*feature] += 1;
                    accumulated_coefs[*feature] += config.nu * coef;
                }
                FittedLearner::Stump { feature, .. } => {
                    feature_selection_count[*feature] += 1;
                }
                FittedLearner::Tree { nodes } => {
                    for node in nodes {
                        if let Some(feat) = node.feature {
                            feature_selection_count[feat] += 1;
                        }
                    }
                }
            }
        }

        // Normalize feature importances
        let total_sel: usize = feature_selection_count.iter().sum();
        let feature_importances: Vec<f64> = if total_sel > 0 {
            feature_selection_count
                .iter()
                .map(|&c| c as f64 / total_sel as f64)
                .collect()
        } else {
            vec![0.0; p]
        };

        // For linear learners, return accumulated coefficients (already in original scale)
        let coefficients = if matches!(config.base_learner, FastMboostLearner::ComponentwiseLinear)
        {
            Some(accumulated_coefs)
        } else {
            None
        };

        FastMboostResult {
            predictions: self.predictions,
            feature_importances,
            m_stop: self.learners.len(),
            risk: self.risk,
            val_risk: self.val_risk,
            validation_rows,
            best_iteration: self.best_iteration,
            coefficients,
            offset: self.offset,
            config: config.clone(),
            learners: self.learners,
            x_means: self.x_means,
        }
    }
}

/// The boosting loop on `x` and `y`. With `validation` (rows the fit never sees) the risk on those
/// rows is recorded after every iteration, the loop stops after `early_stopping_rounds` iterations
/// without improvement, and the learners and training risk are cut back to the best iteration.
fn boost_rows(
    x: ArrayView2<f64>,
    y: &[f64],
    config: &FastMboostConfig,
    validation: Option<(ArrayView2<f64>, &[f64])>,
) -> EconResult<BoostedRows> {
    let n = x.nrows();
    let p = x.ncols();

    // Initialize offset and predictions
    let offset = compute_initial_offset(y, config.family);
    let mut predictions = vec![offset; n];

    // Pre-compute feature statistics for centering
    let x_means: Vec<f64> = (0..p).map(|j| x.column(j).mean().unwrap_or(0.0)).collect();
    let x_stds: Vec<f64> = (0..p)
        .map(|j| {
            let mean = x_means[j];
            let var = x.column(j).iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            var.sqrt().max(1e-10)
        })
        .collect();

    let mut learners = try_with_capacity(config.m_stop, "fast mboost learners")?;
    let mut risk = try_with_capacity(config.m_stop, "fast mboost risk path")?;

    let mut val_predictions = validation.map(|(x_val, _)| vec![offset; x_val.nrows()]);
    let mut val_risk: Vec<f64> = Vec::new();
    let mut best = 0usize; // 1-based; 0 before the first iteration
    let mut best_risk = f64::INFINITY;

    for m in 0..config.m_stop {
        checkpoint()?;

        // Compute negative gradient (pseudo-residuals)
        let residuals = compute_negative_gradient(&predictions, y, config.family);

        // Fit base learner and get update
        let (learner, update) = match &config.base_learner {
            FastMboostLearner::ComponentwiseLinear => {
                fit_componentwise_linear_parallel(x, &residuals, &x_means, &x_stds, p)
            }
            FastMboostLearner::Stump => fit_stump_parallel(x, &residuals, p),
            FastMboostLearner::Tree { max_depth } => fit_tree(x, &residuals, *max_depth),
        };

        // Update predictions with learning rate
        for i in 0..n {
            predictions[i] += config.nu * update[i];
        }

        // Compute training risk
        let train_risk = compute_risk(&predictions, y, config.family);
        risk.push(train_risk);

        if let (Some((x_val, y_val)), Some(val_pred)) = (validation, val_predictions.as_mut()) {
            apply_learner(&learner, config.nu, &x_means, x_val, val_pred);
            let current = compute_risk(val_pred, y_val, config.family);
            val_risk.push(current);
            if current < best_risk {
                best_risk = current;
                best = m + 1;
            }
            learners.push(learner);
            if m + 1 - best >= config.early_stopping_rounds {
                break;
            }
        } else {
            learners.push(learner);
        }
    }

    let (val_risk, best_iteration) = if validation.is_some() {
        learners.truncate(best);
        risk.truncate(best);
        (Some(val_risk), (best > 0).then_some(best))
    } else {
        (None, None)
    };

    Ok(BoostedRows {
        predictions,
        offset,
        x_means,
        learners,
        risk,
        val_risk,
        best_iteration,
    })
}

/// Add one learner's update, `nu` times its fit, to `predictions` (the arithmetic of the training
/// loop, with the features centred at the training means).
fn apply_learner(
    learner: &FittedLearner,
    nu: f64,
    x_means: &[f64],
    x: ArrayView2<f64>,
    predictions: &mut [f64],
) {
    let n = x.nrows();
    match learner {
        FittedLearner::Linear { feature, coef, .. } => {
            let x_mean = x_means[*feature];
            for i in 0..n {
                predictions[i] += nu * (coef * (x[[i, *feature]] - x_mean));
            }
        }
        FittedLearner::Stump {
            feature,
            threshold,
            left_val,
            right_val,
        } => {
            for i in 0..n {
                let val = if x[[i, *feature]] <= *threshold {
                    *left_val
                } else {
                    *right_val
                };
                predictions[i] += nu * val;
            }
        }
        FittedLearner::Tree { nodes } => {
            for i in 0..n {
                predictions[i] += nu * predict_tree(nodes, x.row(i));
            }
        }
    }
}

/// `offset + sum nu * update` over the learners, in learner order.
fn apply_learners(
    learners: &[FittedLearner],
    nu: f64,
    offset: f64,
    x_means: &[f64],
    x: ArrayView2<f64>,
) -> Vec<f64> {
    let mut predictions = vec![offset; x.nrows()];
    for learner in learners {
        apply_learner(learner, nu, x_means, x, &mut predictions);
    }
    predictions
}

/// Compute initial offset based on family
fn compute_initial_offset(y: &[f64], family: FastMboostFamily) -> f64 {
    match family {
        FastMboostFamily::Gaussian => y.iter().sum::<f64>() / y.len() as f64,
        FastMboostFamily::Binomial => {
            let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
            let neg = y.len() as f64 - pos;
            if neg > 0.0 && pos > 0.0 {
                (pos / neg).ln().clamp(-5.0, 5.0)
            } else {
                0.0
            }
        }
        FastMboostFamily::Huber => {
            // Median as robust starting point
            let mut sorted: Vec<f64> = y.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            sorted[sorted.len() / 2]
        }
    }
}

/// Compute negative gradient (pseudo-residuals)
fn compute_negative_gradient(predictions: &[f64], y: &[f64], family: FastMboostFamily) -> Vec<f64> {
    match family {
        FastMboostFamily::Gaussian => {
            // Negative gradient of L2 loss: y - f(x)
            predictions
                .par_iter()
                .zip(y.par_iter())
                .map(|(&p, &yi)| yi - p)
                .collect()
        }
        FastMboostFamily::Binomial => {
            // Negative gradient of log-loss
            predictions
                .par_iter()
                .zip(y.par_iter())
                .map(|(&p, &yi)| {
                    let prob = 1.0 / (1.0 + (-p).exp());
                    yi - prob
                })
                .collect()
        }
        FastMboostFamily::Huber => {
            // Huber loss with delta = 1.345 (for 95% efficiency)
            let delta = 1.345;
            predictions
                .par_iter()
                .zip(y.par_iter())
                .map(|(&p, &yi)| {
                    let r = yi - p;
                    if r.abs() <= delta {
                        r
                    } else {
                        delta * r.signum()
                    }
                })
                .collect()
        }
    }
}

/// Fit componentwise linear learner in parallel
/// This is the key optimization: evaluate all features simultaneously
/// Uses non-standardized (just centered) features to match R's mboost behavior
fn fit_componentwise_linear_parallel(
    x: ArrayView2<f64>,
    residuals: &[f64],
    x_means: &[f64],
    _x_stds: &[f64], // Unused - kept for API compatibility
    p: usize,
) -> (FittedLearner, Vec<f64>) {
    let n = residuals.len();

    // Evaluate all features in parallel - select by R² (like R's mboost)
    let results: Vec<(usize, f64, f64)> = (0..p)
        .into_par_iter()
        .map(|j| {
            let col = x.column(j);
            let x_mean = x_means[j];

            // Compute OLS coefficient: β = Σ(x - x̄)(r) / Σ(x - x̄)²
            // Note: We don't center residuals since we're doing componentwise selection
            let mut ss_xy = 0.0;
            let mut ss_xx = 0.0;

            for (i, &xi) in col.iter().enumerate() {
                let x_centered = xi - x_mean;
                ss_xy += x_centered * residuals[i];
                ss_xx += x_centered * x_centered;
            }

            let coef = if ss_xx > 1e-10 { ss_xy / ss_xx } else { 0.0 };

            // Compute R² for selection (higher is better)
            // R² = (ss_xy)² / (ss_xx * ss_yy)
            let mut ss_yy = 0.0;
            let r_mean: f64 = residuals.iter().sum::<f64>() / n as f64;
            for &r in residuals {
                ss_yy += (r - r_mean).powi(2);
            }

            let r_squared = if ss_xx > 1e-10 && ss_yy > 1e-10 {
                (ss_xy * ss_xy) / (ss_xx * ss_yy)
            } else {
                0.0
            };

            (j, coef, r_squared)
        })
        .collect();

    // Find best feature (maximum R²)
    let (best_feat, best_coef, _) = results
        .into_iter()
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();

    // Compute update values: fitted = coef * (x - x_mean)
    // Note: The update is centered, matching R's bols behavior
    let x_mean = x_means[best_feat];
    let update: Vec<f64> = x
        .column(best_feat)
        .iter()
        .map(|&xi| best_coef * (xi - x_mean))
        .collect();

    let learner = FittedLearner::Linear {
        feature: best_feat,
        coef: best_coef,
        intercept: 0.0, // No intercept in componentwise linear
    };

    (learner, update)
}

/// Fit decision stump in parallel
fn fit_stump_parallel(
    x: ArrayView2<f64>,
    residuals: &[f64],
    p: usize,
) -> (FittedLearner, Vec<f64>) {
    let n = residuals.len();

    // Evaluate all features in parallel
    let results: Vec<(usize, f64, f64, f64, f64)> = (0..p)
        .into_par_iter()
        .map(|j| {
            let col = x.column(j);

            // Sort by feature value
            let mut indexed: Vec<(usize, f64)> =
                col.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

            // Find best split point in one sweep: minimising the RSS of a
            // split is maximising left_sum^2 / left_n + right_sum^2 / right_n.
            // That criterion is accumulated in sort order, so the sweep picks
            // the quadratic search's split except where two candidates have
            // equal RSS, exactly or to within rounding: the two searches may
            // break such a tie either way. Pseudo-residuals with repeated
            // values (a binary or count target) can give candidates exactly
            // equal RSS, so on such data a different, equally good stump can
            // be chosen.
            let mut left_sum = 0.0;
            let mut left_count = 0;
            let total_sum: f64 = residuals.iter().sum();

            let mut best_pos: Option<usize> = None;
            let mut best_gain = f64::NEG_INFINITY;
            let mut best_threshold = indexed[0].1;
            let mut best_left_val = 0.0;
            let mut best_right_val = total_sum / n as f64;

            for i in 0..n - 1 {
                let (orig_idx, val) = indexed[i];
                left_sum += residuals[orig_idx];
                left_count += 1;

                // Only consider split if values differ
                if i + 1 < n && (indexed[i + 1].1 - val).abs() < 1e-10 {
                    continue;
                }

                let right_sum = total_sum - left_sum;
                let right_count = n - left_count;

                if left_count == 0 || right_count == 0 {
                    continue;
                }

                let gain = left_sum * left_sum / left_count as f64
                    + right_sum * right_sum / right_count as f64;

                if gain > best_gain {
                    best_gain = gain;
                    best_pos = Some(i);
                    best_threshold = (val + indexed[i + 1].1) / 2.0;
                    best_left_val = left_sum / left_count as f64;
                    best_right_val = right_sum / right_count as f64;
                }
            }

            // RSS of the chosen split, recomputed exactly as the full search
            // summed it, so for the same split the cross-feature comparison
            // below sees the same values
            let best_rss = match best_pos {
                Some(i) => {
                    let mut rss = 0.0;
                    for k in 0..=i {
                        let resid = residuals[indexed[k].0] - best_left_val;
                        rss += resid * resid;
                    }
                    for k in i + 1..n {
                        let resid = residuals[indexed[k].0] - best_right_val;
                        rss += resid * resid;
                    }
                    rss
                }
                None => f64::INFINITY,
            };

            (j, best_threshold, best_left_val, best_right_val, best_rss)
        })
        .collect();

    // Find best feature
    let (best_feat, threshold, left_val, right_val, _) = results
        .into_iter()
        .min_by(|a, b| a.4.total_cmp(&b.4))
        .unwrap();

    // Compute update values
    let update: Vec<f64> = x
        .column(best_feat)
        .iter()
        .map(|&xi| if xi <= threshold { left_val } else { right_val })
        .collect();

    let learner = FittedLearner::Stump {
        feature: best_feat,
        threshold,
        left_val,
        right_val,
    };

    (learner, update)
}

/// Fit tree with limited depth
fn fit_tree(x: ArrayView2<f64>, residuals: &[f64], max_depth: usize) -> (FittedLearner, Vec<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut nodes = Vec::new();
    let mut update = vec![0.0; n];

    // Build tree using stack
    let mut stack: Vec<(Vec<usize>, usize, Option<usize>, bool)> = vec![];
    stack.push(((0..n).collect(), 0, None, false));

    while let Some((samples, depth, parent_idx, is_left)) = stack.pop() {
        let n_samples = samples.len();
        let sample_residuals: Vec<f64> = samples.iter().map(|&i| residuals[i]).collect();
        let mean_residual = sample_residuals.iter().sum::<f64>() / n_samples as f64;

        if depth >= max_depth || n_samples < 10 {
            // Create leaf
            let node_idx = nodes.len();
            nodes.push(TreeNode {
                feature: None,
                threshold: 0.0,
                left: None,
                right: None,
                value: mean_residual,
            });

            // Set update values for samples
            for &i in &samples {
                update[i] = mean_residual;
            }

            // Link to parent
            if let Some(parent) = parent_idx {
                if is_left {
                    nodes[parent].left = Some(node_idx);
                } else {
                    nodes[parent].right = Some(node_idx);
                }
            }
            continue;
        }

        // Find best split (simplified - uses first improvement found)
        let mut best_split: Option<(usize, f64, f64)> = None; // (feature, threshold, gain)

        for j in 0..p {
            // Get values for this feature
            let mut vals: Vec<(f64, f64)> =
                samples.iter().map(|&i| (x[[i, j]], residuals[i])).collect();
            vals.sort_by(|a, b| a.0.total_cmp(&b.0));

            let total_sum: f64 = sample_residuals.iter().sum();
            let mut left_sum = 0.0;
            let mut left_count = 0;

            for k in 0..vals.len() - 1 {
                left_sum += vals[k].1;
                left_count += 1;

                if (vals[k + 1].0 - vals[k].0).abs() < 1e-10 {
                    continue;
                }

                let right_sum = total_sum - left_sum;
                let right_count = vals.len() - left_count;

                // Gain = variance reduction
                let left_mean = left_sum / left_count as f64;
                let right_mean = right_sum / right_count as f64;
                let total_mean = total_sum / vals.len() as f64;

                let gain = (left_count as f64 * (left_mean - total_mean).powi(2)
                    + right_count as f64 * (right_mean - total_mean).powi(2))
                    / vals.len() as f64;

                let threshold = (vals[k].0 + vals[k + 1].0) / 2.0;

                if best_split.is_none() || gain > best_split.as_ref().unwrap().2 {
                    best_split = Some((j, threshold, gain));
                }
            }
        }

        let node_idx = nodes.len();

        if let Some((feat, threshold, _)) = best_split {
            nodes.push(TreeNode {
                feature: Some(feat),
                threshold,
                left: None,
                right: None,
                value: mean_residual,
            });

            if let Some(parent) = parent_idx {
                if is_left {
                    nodes[parent].left = Some(node_idx);
                } else {
                    nodes[parent].right = Some(node_idx);
                }
            }

            // Partition samples
            let (left_samples, right_samples): (Vec<_>, Vec<_>) = samples
                .into_iter()
                .partition(|&i| x[[i, feat]] <= threshold);

            if !right_samples.is_empty() {
                stack.push((right_samples, depth + 1, Some(node_idx), false));
            }
            if !left_samples.is_empty() {
                stack.push((left_samples, depth + 1, Some(node_idx), true));
            }
        } else {
            // No valid split, make leaf
            nodes.push(TreeNode {
                feature: None,
                threshold: 0.0,
                left: None,
                right: None,
                value: mean_residual,
            });

            for &i in &samples {
                update[i] = mean_residual;
            }

            if let Some(parent) = parent_idx {
                if is_left {
                    nodes[parent].left = Some(node_idx);
                } else {
                    nodes[parent].right = Some(node_idx);
                }
            }
        }
    }

    (FittedLearner::Tree { nodes }, update)
}

/// Compute risk (loss) on predictions
fn compute_risk(predictions: &[f64], y: &[f64], family: FastMboostFamily) -> f64 {
    let n = predictions.len() as f64;
    match family {
        FastMboostFamily::Gaussian => {
            predictions
                .iter()
                .zip(y.iter())
                .map(|(&p, &yi)| (p - yi).powi(2))
                .sum::<f64>()
                / n
        }
        FastMboostFamily::Binomial => {
            predictions
                .iter()
                .zip(y.iter())
                .map(|(&p, &yi)| {
                    let prob = 1.0 / (1.0 + (-p).exp());
                    -(yi * prob.max(1e-10).ln() + (1.0 - yi) * (1.0 - prob).max(1e-10).ln())
                })
                .sum::<f64>()
                / n
        }
        FastMboostFamily::Huber => {
            let delta = 1.345;
            predictions
                .iter()
                .zip(y.iter())
                .map(|(&p, &yi)| {
                    let r = yi - p;
                    if r.abs() <= delta {
                        0.5 * r * r
                    } else {
                        delta * (r.abs() - 0.5 * delta)
                    }
                })
                .sum::<f64>()
                / n
        }
    }
}

/// Make predictions with trained model. Linear learners centre at the training means; a result
/// deserialised without them falls back to the means of the rows given here.
pub fn fast_mboost_predict(result: &FastMboostResult, x: ArrayView2<f64>) -> Vec<f64> {
    let batch_means: Vec<f64>;
    let x_means: &[f64] = if result.x_means.is_empty() {
        batch_means = (0..x.ncols())
            .map(|j| x.column(j).mean().unwrap_or(0.0))
            .collect();
        &batch_means
    } else {
        &result.x_means
    };
    apply_learners(
        &result.learners,
        result.config.nu,
        result.offset,
        x_means,
        x,
    )
}

fn predict_tree(nodes: &[TreeNode], x: ArrayView1<f64>) -> f64 {
    if nodes.is_empty() {
        return 0.0;
    }

    let mut idx = 0;
    loop {
        let node = &nodes[idx];
        if node.feature.is_none() {
            return node.value;
        }

        let feat = node.feature.unwrap();
        if x[feat] <= node.threshold {
            idx = node.left.unwrap_or(idx);
        } else {
            idx = node.right.unwrap_or(idx);
        }

        if idx >= nodes.len() {
            return node.value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_alloc_ceiling, with_budget};
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};
    use std::time::Instant;

    fn noise_xy(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>());
        let y = Array1::from_shape_fn(n, |_| normal.sample(&mut rng));
        (x, y)
    }
    fn es_config(learner: FastMboostLearner, seed: u64) -> FastMboostConfig {
        let mut c = FastMboostConfig::default();
        c.m_stop = 300;
        c.nu = 0.3;
        c.base_learner = learner;
        c.early_stopping_rounds = 10;
        c.validation_fraction = 0.25;
        c.seed = Some(seed);
        c
    }

    #[test]
    fn fast_mboost_early_stopping_stops_at_the_best_validation_risk() {
        let (x, y) = noise_xy(200, 3, 11);
        let r = fast_mboost(x.view(), y.view(), &es_config(FastMboostLearner::Stump, 5)).unwrap();
        let val = r.val_risk.as_ref().expect("validation risk path");
        let best = r.best_iteration.expect("best iteration");
        let argmin = val
            .iter()
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |(bi, bv), (i, &v)| {
                    if v < bv { (i, v) } else { (bi, bv) }
                },
            )
            .0
            + 1;
        assert_eq!(best, argmin);
        assert_eq!(val.len(), best + 10, "stopped by the patience rule");
        assert!(best < 300);
        assert_eq!((r.m_stop, r.risk.len()), (best, best));
        let rows = r.validation_rows.as_ref().unwrap();
        assert_eq!(rows.len(), 50); // ceil(0.25 * 200)
        assert!(rows.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(r.predictions.len(), 200);
    }

    #[test]
    fn fast_mboost_early_stopping_equals_refit_on_training_rows() {
        let (x, y) = noise_xy(200, 3, 12);
        for learner in [
            FastMboostLearner::Stump,
            FastMboostLearner::ComponentwiseLinear,
        ] {
            let cfg = es_config(learner.clone(), 8);
            let r = fast_mboost(x.view(), y.view(), &cfg).unwrap();
            let rows = r.validation_rows.clone().unwrap();
            let train: Vec<usize> = (0..200)
                .filter(|i| rows.binary_search(i).is_err())
                .collect();
            let mut c2 = cfg.clone();
            c2.early_stopping_rounds = 0;
            c2.m_stop = r.best_iteration.unwrap();
            let xt = x.select(Axis(0), &train);
            let yt = y.select(Axis(0), &train);
            let r2 = fast_mboost(xt.view(), yt.view(), &c2).unwrap();
            assert_eq!(r.risk, r2.risk);
            assert_eq!(r.coefficients, r2.coefficients);
            for (k, &i) in train.iter().enumerate() {
                assert_eq!(
                    r.predictions[i].to_bits(),
                    r2.predictions[k].to_bits(),
                    "row {i}"
                );
            }
        }
    }

    #[test]
    fn fast_mboost_seed_draws_the_validation_split() {
        let (x, y) = noise_xy(120, 2, 13);
        let a = fast_mboost(x.view(), y.view(), &es_config(FastMboostLearner::Stump, 1)).unwrap();
        let b = fast_mboost(x.view(), y.view(), &es_config(FastMboostLearner::Stump, 1)).unwrap();
        let c = fast_mboost(x.view(), y.view(), &es_config(FastMboostLearner::Stump, 2)).unwrap();
        assert_eq!(a.validation_rows, b.validation_rows);
        assert_eq!(a.predictions, b.predictions);
        assert_ne!(a.validation_rows, c.validation_rows);
    }

    #[test]
    fn fast_mboost_rejects_bad_validation_fraction() {
        let (x, y) = noise_xy(40, 2, 14);
        // 0.99: ceil(39.6) = 40 leaves no training row
        for f in [0.0, 1.0, -0.1, f64::NAN, 0.99] {
            let mut c = es_config(FastMboostLearner::Stump, 1);
            c.validation_fraction = f;
            let e = fast_mboost(x.view(), y.view(), &c).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.starts_with("validation_fraction")),
                "{f}: {e}"
            );
        }
    }

    #[test]
    fn fast_mboost_predict_uses_training_means() {
        let mut rng = StdRng::seed_from_u64(15);
        let normal = Normal::new(0.0, 0.3).unwrap();
        let x = Array2::from_shape_fn((150, 3), |_| rng.r#gen::<f64>());
        let y = Array1::from_shape_fn(150, |i| {
            2.0 * x[[i, 0]] - x[[i, 2]] + normal.sample(&mut rng)
        });
        let r = fast_mboost(x.view(), y.view(), &FastMboostConfig::default()).unwrap();
        let one = fast_mboost_predict(&r, x.slice(ndarray::s![7..8, ..]));
        assert!(
            (one[0] - r.predictions[7]).abs() <= 1e-12 * r.predictions[7].abs().max(1.0),
            "{} vs {}",
            one[0],
            r.predictions[7]
        );
        let all = fast_mboost_predict(&r, x.view());
        assert!(
            all.iter()
                .zip(&r.predictions)
                .all(|(a, b)| (a - b).abs() <= 1e-12 * b.abs().max(1.0))
        );
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    #[test]
    fn fast_mboost_bounds_cancellation_and_empty_design() {
        let mut rng = StdRng::seed_from_u64(5);
        let x = Array2::from_shape_fn((60, 3), |_| rng.r#gen::<f64>());
        let y: Array1<f64> =
            x.column(0).mapv(|v| 2.0 * v) + Array1::from_shape_fn(60, |_| rng.r#gen::<f64>() * 0.1);
        let huge = FastMboostConfig {
            m_stop: 1usize << 44,
            ..Default::default()
        };
        let r = with_alloc_ceiling(Some(1 << 30), || fast_mboost(x.view(), y.view(), &huge));
        assert!(
            matches!(r, Err(EconError::ResourceLimit(_))),
            "{:?}",
            r.err()
        );
        let r = with_budget(cancelled(), || {
            fast_mboost(x.view(), y.view(), &FastMboostConfig::default())
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{:?}", r.err());
        let empty = Array2::<f64>::zeros((60, 0));
        let stump = FastMboostConfig {
            base_learner: FastMboostLearner::Stump,
            ..Default::default()
        };
        assert!(matches!(
            fast_mboost(empty.view(), y.view(), &stump),
            Err(EconError::InvalidSpecification { .. })
        ));
        // Non-finite data is refused, not boosted into NaN predictions
        let mut bad_x = x.clone();
        bad_x[[7, 2]] = f64::NAN;
        let mut bad_y = y.clone();
        bad_y[11] = f64::INFINITY;
        for (xv, yv) in [(bad_x.view(), y.view()), (x.view(), bad_y.view())] {
            let e = fast_mboost(xv, yv, &stump).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("finite")),
                "{e:?}"
            );
        }
    }

    #[test]
    fn fast_mboost_refuses_a_step_size_outside_zero_one() {
        let mut rng = StdRng::seed_from_u64(13);
        let x = Array2::from_shape_fn((40, 2), |_| rng.r#gen::<f64>());
        let y: Array1<f64> =
            x.column(0).mapv(|v| 2.0 * v) + Array1::from_shape_fn(40, |_| rng.r#gen::<f64>() * 0.1);
        let learners = [
            FastMboostLearner::ComponentwiseLinear,
            FastMboostLearner::Stump,
            FastMboostLearner::Tree { max_depth: 2 },
        ];
        for learner in learners {
            let bad = [
                0.0,
                -0.0,
                -1.0,
                1.0 + 1e-9,
                2.0,
                95.0,
                1e300,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ];
            for nu in bad {
                let config = FastMboostConfig {
                    m_stop: 5,
                    nu,
                    base_learner: learner.clone(),
                    ..Default::default()
                };
                let e = fast_mboost(x.view(), y.view(), &config).unwrap_err();
                assert!(
                    matches!(&e, EconError::InvalidSpecification { message }
                        if message.contains("nu") && message.contains("(0, 1]")),
                    "{learner:?}, nu = {nu:?}: {e}"
                );
            }
            for nu in [1e-6, 0.5, 1.0] {
                let config = FastMboostConfig {
                    m_stop: 5,
                    nu,
                    base_learner: learner.clone(),
                    ..Default::default()
                };
                assert!(
                    fast_mboost(x.view(), y.view(), &config).is_ok(),
                    "nu = {nu}"
                );
            }
        }
    }

    #[test]
    fn fast_mboost_accepts_a_strided_y() {
        // y as a column of a row-major matrix: a valid view whose as_slice()
        // is None, which used to panic
        let mut rng = StdRng::seed_from_u64(9);
        let x = Array2::from_shape_fn((50, 2), |_| rng.r#gen::<f64>());
        let xy = Array2::from_shape_fn((50, 2), |(i, j)| {
            if j == 0 {
                1.5 * x[[i, 0]] + rng.r#gen::<f64>() * 0.1
            } else {
                0.0
            }
        });
        let y_strided = xy.column(0);
        assert!(y_strided.as_slice().is_none());
        let y_owned = y_strided.to_owned();
        for learner in [
            FastMboostLearner::ComponentwiseLinear,
            FastMboostLearner::Stump,
        ] {
            let config = FastMboostConfig {
                m_stop: 20,
                base_learner: learner,
                ..Default::default()
            };
            let strided = fast_mboost(x.view(), y_strided, &config).unwrap();
            let contiguous = fast_mboost(x.view(), y_owned.view(), &config).unwrap();
            assert_eq!(strided.predictions, contiguous.predictions);
            assert_eq!(strided.risk, contiguous.risk);
        }
    }

    /// Two stumps with exactly equal RSS, 2/3: on y = 0, 1, 0, 0 centred by
    /// its mean, x <= 0.5 (1 | 3) and x <= 2 (3 | 1) are mirror images.
    /// Rounding decides which one a search sees as smaller, so either is a
    /// correct answer; the sweep must return one of them, with that RSS.
    #[test]
    fn fast_stump_tie_takes_one_of_the_tied_splits() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 1.0, 3.0, 0.0]).unwrap();
        let y = [0.0, 1.0, 0.0, 0.0];
        let mean = y.iter().sum::<f64>() / 4.0;
        let r: Vec<f64> = y.iter().map(|v| v - mean).collect();
        let (learner, update) = fit_stump_parallel(x.view(), &r, 1);
        let FittedLearner::Stump {
            feature, threshold, ..
        } = &learner
        else {
            panic!("not a stump: {learner:?}");
        };
        assert_eq!(*feature, 0);
        assert!(
            *threshold == 0.5 || *threshold == 2.0,
            "threshold {threshold}"
        );
        let rss: f64 = r.iter().zip(&update).map(|(a, b)| (a - b).powi(2)).sum();
        assert!((rss - 2.0 / 3.0).abs() < 1e-12, "rss {rss}");
    }

    /// Golden (characterization) test: captured on the code before the
    /// stump sweep, on continuous data where no RSS tie breaks differently;
    /// it must pass before and after that change.
    #[test]
    fn fast_mboost_stump_unchanged_by_the_sweep() {
        const GOLDEN: [u64; 3] = [
            4608131849867851993,
            4609280476697468353,
            4591465190771073947,
        ];
        let n = 300;
        let p = 5;
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.3).unwrap();
        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>());
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v)
            + x.column(1).mapv(|v| 0.5 * v)
            + Array1::from_shape_fn(n, |_| normal.sample(&mut rng));
        let config = FastMboostConfig {
            m_stop: 40,
            base_learner: FastMboostLearner::Stump,
            ..Default::default()
        };
        let r = fast_mboost(x.view(), y.view(), &config).unwrap();
        let got = [
            r.predictions[0],
            r.predictions[299],
            *r.risk.last().unwrap(),
        ];
        let bits = got.map(f64::to_bits);
        assert_eq!(bits, GOLDEN, "got {bits:?} = {got:?}");
    }

    #[test]
    fn test_fast_mboost_componentwise_speed() {
        let n = 1000;
        let p = 20;
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.3).unwrap();

        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>());
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v)
            + x.column(1).mapv(|v| 0.5 * v)
            + Array1::from_shape_fn(n, |_| normal.sample(&mut rng));

        let config = FastMboostConfig {
            m_stop: 100,
            nu: 0.1,
            base_learner: FastMboostLearner::ComponentwiseLinear,
            ..Default::default()
        };

        let start = Instant::now();
        let result = fast_mboost(x.view(), y.view(), &config).unwrap();
        let elapsed = start.elapsed();

        println!(
            "Fast MBoost (componentwise): n={}, p={}, m_stop={}",
            n, p, config.m_stop
        );
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Final risk: {:.6}", result.risk.last().unwrap());

        // Compute R²
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(result.predictions.iter())
            .map(|(&yt, &yp)| (yt - yp).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        println!("  R²: {:.4}", r2);
        println!(
            "  Feature importances: x0={:.3}, x1={:.3}",
            result.feature_importances[0], result.feature_importances[1]
        );

        // Performance-regression guard. Asserted only in an optimized build:
        // an unoptimized build of this routine is an order of magnitude slower,
        // so a 1s wall-clock budget fails on an ordinary `cargo test` run and
        // on any loaded or slower machine. The timing is always printed above,
        // and correctness (the R^2 check below) is asserted unconditionally.
        if !cfg!(debug_assertions) {
            assert!(
                elapsed.as_secs_f64() < 1.0,
                "Too slow: {:.2}s",
                elapsed.as_secs_f64()
            );
        }
        // With noise std=0.3 on uniform [0,1], theoretical max R² ≈ 0.80
        // Signal variance: 4*(1/12) + 0.25*(1/12) ≈ 0.354
        // Noise variance: 0.09
        // R² max = 1 - 0.09/0.444 ≈ 0.80
        assert!(r2 > 0.75, "R² too low: {:.4}", r2);
    }

    #[test]
    fn test_fast_mboost_tree_speed() {
        let n = 500;
        let p = 10;
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 0.3).unwrap();

        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>());
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v)
            + x.column(1).mapv(|v| 0.5 * v)
            + Array1::from_shape_fn(n, |_| normal.sample(&mut rng));

        let config = FastMboostConfig {
            m_stop: 100,
            nu: 0.1,
            base_learner: FastMboostLearner::Tree { max_depth: 3 },
            ..Default::default()
        };

        let start = Instant::now();
        let result = fast_mboost(x.view(), y.view(), &config).unwrap();
        let elapsed = start.elapsed();

        println!("Fast MBoost (tree): n={}, p={}", n, p);
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        // Compute R²
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(result.predictions.iter())
            .map(|(&yt, &yp)| (yt - yp).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        println!("  R²: {:.4}", r2);

        assert!(r2 > 0.80, "R² too low: {:.4}", r2);
    }
}
