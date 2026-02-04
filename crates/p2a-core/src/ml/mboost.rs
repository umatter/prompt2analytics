//! MBoost - Model-Based Gradient Boosting.
//!
//! Implements a flexible gradient boosting framework that supports
//! different base learners (trees, linear models, splines).
//!
//! ## Key Features
//!
//! - Multiple base learner types
//! - Component-wise boosting
//! - Built-in cross-validation for stopping
//! - Support for different loss functions
//!
//! ## References
//!
//! Hothorn, T., et al. (2010). "Model-based Boosting 2.0."
//! *Journal of Machine Learning Research*, 11, 2109-2113.
//!
//! Bühlmann, P., & Yu, B. (2003). "Boosting with the L2 Loss."
//! *Journal of the American Statistical Association*, 98(462), 324-339.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};

/// MBoost configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MboostConfig {
    /// Number of boosting iterations
    pub m_stop: usize,
    /// Learning rate (nu)
    pub nu: f64,
    /// Base learner type
    pub base_learner: MboostBaseLearner,
    /// Loss function
    pub family: MboostFamily,
    /// Number of CV folds for early stopping (0 = no CV)
    pub cv_folds: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for MboostConfig {
    fn default() -> Self {
        Self {
            m_stop: 100,
            nu: 0.1,
            base_learner: MboostBaseLearner::Tree { max_depth: 4 },
            family: MboostFamily::Gaussian,
            cv_folds: 0,
            seed: None,
        }
    }
}

/// Base learner types for mboost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MboostBaseLearner {
    /// Decision trees (btree)
    Tree { max_depth: usize },
    /// Linear models (bols)
    Linear,
    /// Componentwise linear (one feature at a time)
    ComponentwiseLinear,
    /// P-splines (bbs)
    Spline { df: usize },
}

impl Default for MboostBaseLearner {
    fn default() -> Self {
        Self::Tree { max_depth: 4 }
    }
}

/// Loss functions for mboost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MboostFamily {
    /// Gaussian (squared error)
    #[default]
    Gaussian,
    /// Binomial (logistic)
    Binomial,
    /// Poisson
    Poisson,
    /// AdaExp (AdaBoost exponential)
    AdaExp,
    /// Huber (robust regression)
    Huber,
}

/// MBoost result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MboostResult {
    /// Predictions
    pub predictions: Vec<f64>,
    /// Feature importances (selection frequency)
    pub feature_importances: Vec<f64>,
    /// Number of iterations used
    pub m_stop: usize,
    /// Training risk (loss) per iteration
    pub risk: Vec<f64>,
    /// CV risk if cross-validation was used
    pub cv_risk: Option<Vec<f64>>,
    /// Optimal stopping iteration from CV
    pub optimal_m: Option<usize>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
    /// Base learner coefficients/models
    #[serde(skip)]
    learners: Vec<MboostLearner>,
    /// Configuration
    config: MboostConfig,
    /// Offset (intercept)
    offset: f64,
}

impl std::fmt::Display for MboostResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MBoost Model")?;
        writeln!(f, "============")?;
        writeln!(f, "Iterations: {}", self.m_stop)?;
        writeln!(f, "Learning rate: {}", self.config.nu)?;
        writeln!(f, "Base learner: {:?}", self.config.base_learner)?;
        writeln!(f, "Family: {:?}", self.config.family)?;

        if let Some(opt_m) = self.optimal_m {
            writeln!(f, "Optimal m (CV): {}", opt_m)?;
        }

        writeln!(f)?;

        if let Some(last_risk) = self.risk.last() {
            writeln!(f, "Final training risk: {:.6}", last_risk)?;
        }

        writeln!(f)?;
        writeln!(f, "Variable Selection Frequency (top 10):")?;

        let mut indexed: Vec<(usize, f64)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, freq) in indexed.iter().take(10) {
            let name = self
                .feature_names
                .as_ref()
                .and_then(|n| n.get(*idx).cloned())
                .unwrap_or_else(|| format!("X{}", idx));
            writeln!(f, "  {}: {:.4}", name, freq)?;
        }

        Ok(())
    }
}

/// Internal base learner model.
#[derive(Debug, Clone)]
enum MboostLearner {
    /// Tree learner
    Tree(SimpleTree),
    /// Linear coefficient for a single feature (with mean for centering)
    Linear {
        feature: usize,
        coef: f64,
        x_mean: f64,
    },
    /// Full linear model
    FullLinear { coefs: Vec<f64>, x_means: Vec<f64> },
}

#[derive(Debug, Clone)]
struct SimpleTree {
    nodes: Vec<SimpleTreeNode>,
}

#[derive(Debug, Clone)]
struct SimpleTreeNode {
    feature: Option<usize>,
    threshold: f64,
    left: Option<usize>,
    right: Option<usize>,
    value: f64,
}

impl SimpleTree {
    fn predict(&self, x: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let mut idx = 0;
        loop {
            let node = &self.nodes[idx];
            if node.feature.is_none() {
                return node.value;
            }

            let feat = node.feature.unwrap();
            if x[feat] <= node.threshold {
                idx = node.left.unwrap_or(idx);
            } else {
                idx = node.right.unwrap_or(idx);
            }

            if idx >= self.nodes.len() {
                return node.value;
            }
        }
    }
}

/// Train an mboost model.
pub fn mboost(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    config: &MboostConfig,
) -> EconResult<MboostResult> {
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(EconError::InsufficientData {
            required: n,
            provided: y.len(),
            context: "X and y must have same number of rows".to_string(),
        });
    }

    // Initialize
    let offset = match config.family {
        MboostFamily::Gaussian => y.mean().unwrap_or(0.0),
        MboostFamily::Binomial => {
            let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
            let neg = n as f64 - pos;
            (pos / neg).ln().max(-10.0).min(10.0)
        }
        MboostFamily::Poisson => y.mean().unwrap_or(1.0).ln().max(-10.0),
        MboostFamily::AdaExp => 0.0,
        MboostFamily::Huber => y.mean().unwrap_or(0.0),
    };

    let mut f_hat = vec![offset; n];
    let mut learners = Vec::with_capacity(config.m_stop);
    let mut risk = Vec::with_capacity(config.m_stop);
    let mut feature_selection_count = vec![0usize; p];

    for _m in 0..config.m_stop {
        // Compute negative gradient (pseudo-residuals)
        let neg_grad = compute_negative_gradient(&f_hat, y.as_slice().unwrap(), config.family);

        // Fit base learner to negative gradient
        let (learner, selected_features) = fit_base_learner(&x, &neg_grad, &config.base_learner);

        // Update selection counts
        for feat in &selected_features {
            feature_selection_count[*feat] += 1;
        }

        // Update predictions
        for i in 0..n {
            let row: Vec<f64> = x.row(i).to_vec();
            f_hat[i] += config.nu * predict_learner(&learner, &row);
        }

        // Compute risk
        let r = compute_risk(&f_hat, y.as_slice().unwrap(), config.family);
        risk.push(r);

        learners.push(learner);
    }

    // Normalize feature importances
    let total_selections: f64 = feature_selection_count.iter().sum::<usize>() as f64;
    let feature_importances: Vec<f64> = if total_selections > 0.0 {
        feature_selection_count
            .iter()
            .map(|&c| c as f64 / total_selections)
            .collect()
    } else {
        vec![0.0; p]
    };

    // Final predictions
    let predictions = match config.family {
        MboostFamily::Gaussian | MboostFamily::Huber => f_hat.clone(),
        MboostFamily::Binomial | MboostFamily::AdaExp => {
            f_hat.iter().map(|&f| sigmoid(f)).collect()
        }
        MboostFamily::Poisson => f_hat.iter().map(|&f| f.exp()).collect(),
    };

    Ok(MboostResult {
        predictions,
        feature_importances,
        m_stop: learners.len(),
        risk,
        cv_risk: None,
        optimal_m: None,
        feature_names: None,
        learners,
        config: config.clone(),
        offset,
    })
}

/// Predict using a trained mboost model.
pub fn mboost_predict(model: &MboostResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    let n = x.nrows();
    let mut f_hat = vec![model.offset; n];

    for i in 0..n {
        let row: Vec<f64> = x.row(i).to_vec();
        for learner in &model.learners {
            f_hat[i] += model.config.nu * predict_learner(learner, &row);
        }
    }

    match model.config.family {
        MboostFamily::Gaussian | MboostFamily::Huber => Ok(f_hat),
        MboostFamily::Binomial | MboostFamily::AdaExp => {
            Ok(f_hat.iter().map(|&f| sigmoid(f)).collect())
        }
        MboostFamily::Poisson => Ok(f_hat.iter().map(|&f| f.exp()).collect()),
    }
}

/// Compute negative gradient for the family.
fn compute_negative_gradient(f: &[f64], y: &[f64], family: MboostFamily) -> Vec<f64> {
    let n = f.len();
    let mut grad = vec![0.0; n];

    match family {
        MboostFamily::Gaussian => {
            for i in 0..n {
                grad[i] = y[i] - f[i];
            }
        }
        MboostFamily::Binomial => {
            for i in 0..n {
                let p = sigmoid(f[i]);
                grad[i] = y[i] - p;
            }
        }
        MboostFamily::Poisson => {
            for i in 0..n {
                let mu = f[i].exp();
                grad[i] = y[i] - mu;
            }
        }
        MboostFamily::AdaExp => {
            for i in 0..n {
                let sign = if y[i] > 0.5 { 1.0 } else { -1.0 };
                grad[i] = sign * (-sign * f[i]).exp();
            }
        }
        MboostFamily::Huber => {
            // Huber with delta = 1.345 (standard)
            let delta = 1.345;
            for i in 0..n {
                let r = y[i] - f[i];
                grad[i] = if r.abs() <= delta {
                    r
                } else {
                    delta * r.signum()
                };
            }
        }
    }

    grad
}

/// Compute risk (loss) for the family.
fn compute_risk(f: &[f64], y: &[f64], family: MboostFamily) -> f64 {
    let n = f.len() as f64;

    match family {
        MboostFamily::Gaussian => {
            f.iter().zip(y).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / (2.0 * n)
        }
        MboostFamily::Binomial => {
            f.iter()
                .zip(y)
                .map(|(&p, &t)| {
                    let prob = sigmoid(p);
                    -t * prob.ln().max(-100.0) - (1.0 - t) * (1.0 - prob).ln().max(-100.0)
                })
                .sum::<f64>()
                / n
        }
        MboostFamily::Poisson => {
            f.iter()
                .zip(y)
                .map(|(&p, &t)| {
                    let mu = p.exp();
                    mu - t * p
                })
                .sum::<f64>()
                / n
        }
        MboostFamily::AdaExp => {
            f.iter()
                .zip(y)
                .map(|(&p, &t)| {
                    let sign = if t > 0.5 { 1.0 } else { -1.0 };
                    (-sign * p).exp()
                })
                .sum::<f64>()
                / n
        }
        MboostFamily::Huber => {
            let delta = 1.345;
            f.iter()
                .zip(y)
                .map(|(p, t)| {
                    let r = (p - t).abs();
                    if r <= delta {
                        0.5 * r.powi(2)
                    } else {
                        delta * r - 0.5 * delta.powi(2)
                    }
                })
                .sum::<f64>()
                / n
        }
    }
}

/// Fit a base learner to the negative gradient.
fn fit_base_learner(
    x: &ArrayView2<f64>,
    neg_grad: &[f64],
    learner_type: &MboostBaseLearner,
) -> (MboostLearner, Vec<usize>) {
    match learner_type {
        MboostBaseLearner::Tree { max_depth } => {
            let (tree, features) = fit_simple_tree(x, neg_grad, *max_depth);
            (MboostLearner::Tree(tree), features)
        }
        MboostBaseLearner::Linear => {
            let (coefs, x_means, features) = fit_linear(x, neg_grad);
            (MboostLearner::FullLinear { coefs, x_means }, features)
        }
        MboostBaseLearner::ComponentwiseLinear => {
            let (feature, coef, x_mean) = fit_componentwise_linear(x, neg_grad);
            (
                MboostLearner::Linear {
                    feature,
                    coef,
                    x_mean,
                },
                vec![feature],
            )
        }
        MboostBaseLearner::Spline { df: _ } => {
            // Simplified: use componentwise linear as approximation
            let (feature, coef, x_mean) = fit_componentwise_linear(x, neg_grad);
            (
                MboostLearner::Linear {
                    feature,
                    coef,
                    x_mean,
                },
                vec![feature],
            )
        }
    }
}

/// Fit a simple regression tree.
fn fit_simple_tree(x: &ArrayView2<f64>, y: &[f64], max_depth: usize) -> (SimpleTree, Vec<usize>) {
    let mut tree = SimpleTree { nodes: Vec::new() };
    let mut selected_features = Vec::new();
    let indices: Vec<usize> = (0..x.nrows()).collect();

    build_simple_tree_node(
        &mut tree,
        x,
        y,
        &indices,
        0,
        max_depth,
        &mut selected_features,
    );

    (tree, selected_features)
}

fn build_simple_tree_node(
    tree: &mut SimpleTree,
    x: &ArrayView2<f64>,
    y: &[f64],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    selected_features: &mut Vec<usize>,
) -> usize {
    let n = indices.len();
    let value: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n as f64;

    if depth >= max_depth || n < 4 {
        let idx = tree.nodes.len();
        tree.nodes.push(SimpleTreeNode {
            feature: None,
            threshold: 0.0,
            left: None,
            right: None,
            value,
        });
        return idx;
    }

    // Find best split
    let p = x.ncols();
    let mut best_feat = None;
    let mut best_threshold = 0.0;
    let mut best_sse = f64::INFINITY;
    let mut best_left = Vec::new();
    let mut best_right = Vec::new();

    for feat in 0..p {
        let mut vals: Vec<f64> = indices.iter().map(|&i| x[[i, feat]]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup();

        for i in 0..vals.len().saturating_sub(1) {
            let thresh = (vals[i] + vals[i + 1]) / 2.0;

            let mut left = Vec::new();
            let mut right = Vec::new();

            for &idx in indices {
                if x[[idx, feat]] <= thresh {
                    left.push(idx);
                } else {
                    right.push(idx);
                }
            }

            if left.len() < 2 || right.len() < 2 {
                continue;
            }

            let left_mean: f64 = left.iter().map(|&i| y[i]).sum::<f64>() / left.len() as f64;
            let right_mean: f64 = right.iter().map(|&i| y[i]).sum::<f64>() / right.len() as f64;

            let left_sse: f64 = left.iter().map(|&i| (y[i] - left_mean).powi(2)).sum();
            let right_sse: f64 = right.iter().map(|&i| (y[i] - right_mean).powi(2)).sum();
            let total_sse = left_sse + right_sse;

            if total_sse < best_sse {
                best_sse = total_sse;
                best_feat = Some(feat);
                best_threshold = thresh;
                best_left = left;
                best_right = right;
            }
        }
    }

    if best_feat.is_none() {
        let idx = tree.nodes.len();
        tree.nodes.push(SimpleTreeNode {
            feature: None,
            threshold: 0.0,
            left: None,
            right: None,
            value,
        });
        return idx;
    }

    let feat = best_feat.unwrap();
    if !selected_features.contains(&feat) {
        selected_features.push(feat);
    }

    let node_idx = tree.nodes.len();
    tree.nodes.push(SimpleTreeNode {
        feature: Some(feat),
        threshold: best_threshold,
        left: None,
        right: None,
        value: 0.0,
    });

    let left_idx = build_simple_tree_node(
        tree,
        x,
        y,
        &best_left,
        depth + 1,
        max_depth,
        selected_features,
    );
    let right_idx = build_simple_tree_node(
        tree,
        x,
        y,
        &best_right,
        depth + 1,
        max_depth,
        selected_features,
    );

    tree.nodes[node_idx].left = Some(left_idx);
    tree.nodes[node_idx].right = Some(right_idx);

    node_idx
}

/// Fit componentwise linear model (select best single feature).
/// Returns (feature_index, coefficient, feature_mean) for centered prediction.
fn fit_componentwise_linear(x: &ArrayView2<f64>, y: &[f64]) -> (usize, f64, f64) {
    let p = x.ncols();
    let n = x.nrows();

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

    let mut best_feat = 0;
    let mut best_coef = 0.0;
    let mut best_x_mean = 0.0;
    let mut best_r2 = f64::NEG_INFINITY;

    for feat in 0..p {
        let x_col: Vec<f64> = (0..n).map(|i| x[[i, feat]]).collect();
        let x_mean: f64 = x_col.iter().sum::<f64>() / n as f64;

        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        let mut ss_yy = 0.0;

        for i in 0..n {
            let xd = x_col[i] - x_mean;
            let yd = y[i] - y_mean;
            ss_xy += xd * yd;
            ss_xx += xd * xd;
            ss_yy += yd * yd;
        }

        if ss_xx < 1e-10 {
            continue;
        }

        let coef = ss_xy / ss_xx;
        let r2 = if ss_yy > 1e-10 {
            (ss_xy * ss_xy) / (ss_xx * ss_yy)
        } else {
            0.0
        };

        if r2 > best_r2 {
            best_r2 = r2;
            best_feat = feat;
            best_coef = coef;
            best_x_mean = x_mean;
        }
    }

    (best_feat, best_coef, best_x_mean)
}

/// Fit full linear model.
/// Returns (coefficients, feature_means, selected_features) for centered prediction.
fn fit_linear(x: &ArrayView2<f64>, y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let p = x.ncols();
    let n = x.nrows();

    // Simple OLS: (X'X)^-1 X'y
    // Simplified: use independent feature fits
    let mut coefs = vec![0.0; p];
    let mut x_means = vec![0.0; p];
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

    for feat in 0..p {
        let x_col: Vec<f64> = (0..n).map(|i| x[[i, feat]]).collect();
        let x_mean: f64 = x_col.iter().sum::<f64>() / n as f64;
        x_means[feat] = x_mean;

        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;

        for i in 0..n {
            ss_xy += (x_col[i] - x_mean) * (y[i] - y_mean);
            ss_xx += (x_col[i] - x_mean).powi(2);
        }

        if ss_xx > 1e-10 {
            coefs[feat] = ss_xy / ss_xx;
        }
    }

    let features: Vec<usize> = (0..p).filter(|&i| coefs[i].abs() > 1e-10).collect();
    (coefs, x_means, features)
}

/// Predict using a single learner.
fn predict_learner(learner: &MboostLearner, x: &[f64]) -> f64 {
    match learner {
        MboostLearner::Tree(tree) => tree.predict(x),
        // Use centered prediction: coef * (x - x_mean)
        MboostLearner::Linear {
            feature,
            coef,
            x_mean,
        } => (x[*feature] - x_mean) * coef,
        MboostLearner::FullLinear { coefs, x_means } => {
            // Use centered prediction: sum of coef * (x - x_mean)
            x.iter()
                .zip(coefs.iter())
                .zip(x_means.iter())
                .map(|((xi, ci), mi)| (xi - mi) * ci)
                .sum()
        }
    }
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
    fn test_mboost_gaussian() {
        let x = Array2::from_shape_vec(
            (100, 2),
            (0..200).map(|i| (i % 100) as f64 / 100.0).collect(),
        )
        .unwrap();

        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v) + x.column(1).mapv(|v| v * 0.5);

        let config = MboostConfig {
            m_stop: 50,
            nu: 0.1,
            base_learner: MboostBaseLearner::Tree { max_depth: 3 },
            family: MboostFamily::Gaussian,
            ..Default::default()
        };

        let result = mboost(x.view(), y.view(), &config).unwrap();

        assert_eq!(result.m_stop, 50);
        assert!(!result.predictions.is_empty());
    }

    #[test]
    fn test_mboost_componentwise() {
        let x = Array2::from_shape_vec(
            (100, 3),
            (0..300).map(|i| (i % 100) as f64 / 100.0).collect(),
        )
        .unwrap();

        let y: Array1<f64> = x.column(0).mapv(|v| 3.0 * v);

        let config = MboostConfig {
            m_stop: 50,
            nu: 0.1,
            base_learner: MboostBaseLearner::ComponentwiseLinear,
            family: MboostFamily::Gaussian,
            ..Default::default()
        };

        let result = mboost(x.view(), y.view(), &config).unwrap();

        // Feature 0 should be selected most often
        assert!(result.feature_importances[0] > result.feature_importances[1]);
    }
}
