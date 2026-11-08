//! AdaBoost (Adaptive Boosting) implementation.
//!
//! Pure Rust implementation of AdaBoost.M1 (Freund & Schapire, 1997) for
//! binary classification, SAMME (Zhu, Zou, Rosset & Hastie, 2009) for K classes
//! and AdaBoost.R2 (Drucker, 1997) for regression.
//!
//! ## Features
//!
//! - **Classification**: AdaBoost.M1 (two classes) and SAMME (K classes) with CART weak learners of
//!   depth `max_depth` (rpart's Gini rule on case weights)
//! - **Regression**: AdaBoost.R2 with least-squares CART weak learners and configurable loss functions
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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::resource::{checkpoint, try_with_capacity, try_zeros2};
use crate::traits::convergence::{ResultNotes, notes_text};

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

crate::option_enum!(
    AdaBoostType,
    crate::options::Choices {
        option: "boost_type",
        values: &[
            (
                &["m1", "adaboost_m1", "discrete", "classification"],
                AdaBoostType::M1
            ),
            (&["r2", "adaboost_r2", "regression"], AdaBoostType::R2),
            (&["samme", "multiclass", "zhu"], AdaBoostType::Samme),
        ],
    }
);

/// Deepest weak learner: rpart.control(maxdepth)'s limit.
pub const ADABOOST_MAX_DEPTH: usize = 30;

/// Most classes SAMME takes: each split candidate costs one operation per class.
const SAMME_MAX_CLASSES: usize = 100;

/// A weighted error within this of the chance level counts as chance: the level itself is
/// reached only up to rounding (K equal class weights).
const CHANCE_MARGIN: f64 = 1e-12;

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
    /// Unused: boosting reweights (adabag boos = FALSE) and is deterministic
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

/// A node of a weak learner.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WeakNode {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<WeakNode>,
        right: Box<WeakNode>,
    },
}

/// A weak learner: a CART tree on case weights, with each feature's share of the improvement.
#[derive(Debug, Clone)]
pub(crate) struct WeakTree {
    root: WeakNode,
    /// Per-feature improvement share (sums to 1, or all 0 for a root leaf)
    importance: Vec<f64>,
}

impl WeakTree {
    fn predict_one(&self, x: &ArrayView1<f64>) -> f64 {
        let mut node = &self.root;
        loop {
            match node {
                WeakNode::Leaf { value } => return *value,
                WeakNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    node = if x[*feature] <= *threshold {
                        left
                    } else {
                        right
                    }
                }
            }
        }
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        Array1::from_iter((0..x.nrows()).map(|i| self.predict_one(&x.row(i))))
    }
}

/// What a weak learner fits: class codes `0..labels.len()` with the label each code stands for
/// (weighted Gini), or a numeric response (weighted least squares).
pub(crate) enum WeakTarget<'a> {
    Class {
        codes: &'a [usize],
        labels: &'a [f64],
    },
    Regression(&'a [f64]),
}

/// Stable ascending row orders of every column, computed once per fit. `x` must be NaN-free.
fn feature_orders(x: &ArrayView2<f64>) -> EconResult<Vec<Vec<usize>>> {
    let n = x.nrows();
    let mut orders = try_with_capacity(x.ncols(), "AdaBoost feature orders")?;
    for j in 0..x.ncols() {
        let col = x.column(j);
        let mut order = try_with_capacity(n, "AdaBoost feature orders")?;
        order.extend(0..n);
        order.sort_by(|&a, &b| col[a].total_cmp(&col[b]));
        orders.push(order);
    }
    Ok(orders)
}

/// The midpoint of two distinct ordered values, kept below the larger (the rows at it go right).
fn split_threshold(a: f64, b: f64) -> f64 {
    let mid = (a + b) / 2.0;
    let mid = if mid.is_finite() {
        mid
    } else {
        a / 2.0 + b / 2.0
    };
    if mid < b { mid } else { a }
}

/// Grows one weak tree. A node owns the segment `[lo, hi)` of every feature's order array, so a
/// level costs O(p n): after a split each feature's segment is stably partitioned into left rows
/// then right rows.
struct WeakGrower<'a, 't> {
    x: ArrayView2<'a, f64>,
    target: &'t WeakTarget<'t>,
    weights: &'t [f64],
    max_depth: usize,
    orders: Vec<Vec<usize>>,
    goes_left: Vec<bool>,
    scratch: Vec<usize>,
    importance: Vec<f64>,
}

impl WeakGrower<'_, '_> {
    /// Node weight, impurity and leaf value, with the class weights of a classification node.
    fn node_stats(&self, lo: usize, hi: usize) -> (f64, f64, f64, Vec<f64>) {
        let rows = &self.orders[0][lo..hi];
        match self.target {
            WeakTarget::Class { codes, labels } => {
                let mut class_w = vec![0.0; labels.len()];
                let mut total = 0.0;
                for &r in rows {
                    class_w[codes[r]] += self.weights[r];
                    total += self.weights[r];
                }
                let squares: f64 = class_w.iter().map(|w| w * w).sum();
                let impurity = if total > 0.0 {
                    total - squares / total
                } else {
                    0.0
                };
                // The class with the largest weight; ties take the smallest class
                let mut best = 0;
                for (k, &w) in class_w.iter().enumerate() {
                    if w > class_w[best] {
                        best = k;
                    }
                }
                (total, impurity, labels[best], class_w)
            }
            WeakTarget::Regression(y) => {
                let (mut total, mut sum) = (0.0, 0.0);
                for &r in rows {
                    total += self.weights[r];
                    sum += self.weights[r] * y[r];
                }
                let mean = if total > 0.0 { sum / total } else { 0.0 };
                let impurity = rows
                    .iter()
                    .fold(0.0, |ss, &r| ss + self.weights[r] * (y[r] - mean).powi(2));
                (total, impurity, mean, vec![sum])
            }
        }
    }

    /// The best split of a node: the first candidate in feature and position order whose
    /// improvement exceeds the running best, started just above zero (rpart's cp = 0: only positive
    /// improvements split; the floor ignores rounding noise).
    fn best_split(
        &self,
        lo: usize,
        hi: usize,
        total: f64,
        impurity: f64,
        totals: &[f64],
    ) -> Option<(usize, usize, f64, f64)> {
        let mut best_improvement = 1e-12 * impurity;
        let mut best = None;
        let mut left = vec![0.0; totals.len()];
        for (feature, order) in self.orders.iter().enumerate() {
            let rows = &order[lo..hi];
            left.fill(0.0);
            let mut left_w = 0.0;
            for pos in 0..rows.len() - 1 {
                let r = rows[pos];
                let w = self.weights[r];
                left_w += w;
                match self.target {
                    WeakTarget::Class { codes, .. } => left[codes[r]] += w,
                    WeakTarget::Regression(y) => left[0] += w * y[r],
                }
                let (a, b) = (self.x[[r, feature]], self.x[[rows[pos + 1], feature]]);
                if a >= b {
                    continue;
                }
                let right_w = total - left_w;
                if left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }
                let improvement = match self.target {
                    WeakTarget::Class { .. } => {
                        let (mut sq_l, mut sq_r, mut sq) = (0.0, 0.0, 0.0);
                        for (l, t) in left.iter().zip(totals) {
                            sq_l += l * l;
                            sq_r += (t - l) * (t - l);
                            sq += t * t;
                        }
                        sq_l / left_w + sq_r / right_w - sq / total
                    }
                    WeakTarget::Regression(_) => {
                        let (s_l, s) = (left[0], totals[0]);
                        let s_r = s - s_l;
                        s_l * s_l / left_w + s_r * s_r / right_w - s * s / total
                    }
                };
                if improvement > best_improvement {
                    best_improvement = improvement;
                    best = Some((feature, pos, split_threshold(a, b), improvement));
                }
            }
        }
        best
    }

    /// Stably partition every feature's segment into the left rows `[lo, mid)` then the right rows.
    fn partition(&mut self, feature: usize, lo: usize, mid: usize, hi: usize) {
        for &r in &self.orders[feature][lo..mid] {
            self.goes_left[r] = true;
        }
        for (f, order) in self.orders.iter_mut().enumerate() {
            if f == feature {
                continue;
            }
            let mut to_left = lo;
            let mut to_right = 0;
            for pos in lo..hi {
                let r = order[pos];
                if self.goes_left[r] {
                    order[to_left] = r;
                    to_left += 1;
                } else {
                    self.scratch[to_right] = r;
                    to_right += 1;
                }
            }
            order[mid..hi].copy_from_slice(&self.scratch[..to_right]);
        }
        for &r in &self.orders[feature][lo..mid] {
            self.goes_left[r] = false;
        }
    }

    fn grow(&mut self, lo: usize, hi: usize, depth: usize) -> WeakNode {
        let (total, impurity, value, totals) = self.node_stats(lo, hi);
        if depth >= self.max_depth || hi - lo < 2 || impurity <= 0.0 {
            return WeakNode::Leaf { value };
        }
        let Some((feature, pos, threshold, improvement)) =
            self.best_split(lo, hi, total, impurity, &totals)
        else {
            return WeakNode::Leaf { value };
        };
        self.importance[feature] += improvement;
        let mid = lo + pos + 1;
        self.partition(feature, lo, mid, hi);
        let left = self.grow(lo, mid, depth + 1);
        let right = self.grow(mid, hi, depth + 1);
        WeakNode::Split {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

/// Fit a weak learner as rpart(weights = w, control = rpart.control(maxdepth = max_depth,
/// minsplit = 2, minbucket = 1, cp = 0)): weighted Gini for classes, weighted least squares for a
/// numeric response.
pub(crate) fn fit_weak_tree(
    x: &ArrayView2<f64>,
    orders: &[Vec<usize>],
    target: &WeakTarget<'_>,
    weights: &[f64],
    max_depth: usize,
) -> WeakTree {
    let n = x.nrows();
    let p = x.ncols();
    if p == 0 {
        // No column to split or to read the rows from: a root leaf over all rows
        let value = match target {
            WeakTarget::Class { codes, labels } => {
                let mut class_w = vec![0.0; labels.len()];
                for (&c, &w) in codes.iter().zip(weights) {
                    class_w[c] += w;
                }
                let mut best = 0;
                for (k, &w) in class_w.iter().enumerate() {
                    if w > class_w[best] {
                        best = k;
                    }
                }
                labels[best]
            }
            WeakTarget::Regression(y) => {
                let total: f64 = weights.iter().sum();
                y.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / total
            }
        };
        return WeakTree {
            root: WeakNode::Leaf { value },
            importance: Vec::new(),
        };
    }
    let mut grower = WeakGrower {
        x: x.view(),
        target,
        weights,
        max_depth,
        orders: orders.to_vec(),
        goes_left: vec![false; n],
        scratch: vec![0; n],
        importance: vec![0.0; p],
    };
    let root = grower.grow(0, n, 0);
    let mut importance = grower.importance;
    let sum: f64 = importance.iter().sum();
    if sum > 0.0 {
        for v in &mut importance {
            *v /= sum;
        }
    }
    WeakTree { root, importance }
}

/// One SAMME round on the current weights: the tree, its weighted error, its alpha and the
/// normalised new weights; `Err(e)` when the error is no better than chance (1 - 1/K).
fn samme_round(
    x: &ArrayView2<f64>,
    orders: &[Vec<usize>],
    codes: &[usize],
    labels: &[f64],
    weights: &[f64],
    max_depth: usize,
    learning_rate: f64,
) -> Result<(WeakTree, f64, f64, Vec<f64>), f64> {
    let k = labels.len() as f64;
    let target = WeakTarget::Class { codes, labels };
    let tree = fit_weak_tree(x, orders, &target, weights, max_depth);
    let pred = tree.predict(x);
    let miss: Vec<bool> = pred
        .iter()
        .zip(codes)
        .map(|(&h, &c)| h != labels[c])
        .collect();
    let error = weighted_error(&miss, weights);
    if error >= 1.0 - 1.0 / k - CHANCE_MARGIN {
        return Err(error);
    }
    let ec = error.clamp(1e-10, 1.0 - 1e-10);
    let alpha = learning_rate * (((1.0 - ec) / ec).ln() + (k - 1.0).ln());
    let mut new_weights: Vec<f64> = weights
        .iter()
        .zip(&miss)
        .map(|(&w, &m)| if m { w * alpha.exp() } else { w })
        .collect();
    let sum: f64 = new_weights.iter().sum();
    for w in &mut new_weights {
        *w /= sum;
    }
    Ok((tree, error, alpha, new_weights))
}

/// `sum w 1{miss} / sum w`.
fn weighted_error(miss: &[bool], weights: &[f64]) -> f64 {
    let (mut wrong, mut total) = (0.0, 0.0);
    for (&m, &w) in miss.iter().zip(weights) {
        total += w;
        if m {
            wrong += w;
        }
    }
    wrong / total
}

/// The refusal of a first weak learner no better than chance.
fn no_better_than_chance(error: f64, bound: f64) -> EconError {
    EconError::InvalidSpecification {
        message: format!(
            "AdaBoost: the first weak learner has weighted error {error:.4} >= {bound:.4}, \
             no better than chance; nothing to boost"
        ),
    }
}

/// The note of a fit that stopped early.
fn stopped_early(done: usize, rounds: usize, error: f64, bound: f64) -> String {
    format!(
        "AdaBoost stopped after {done} of {rounds} rounds: weak learner {} has weighted error \
         {error:.4} >= {bound:.4}, no better than chance",
        done + 1
    )
}

/// `sum_m alpha_m * importance_m`, normalised.
fn add_importance(total: &mut [f64], tree: &WeakTree, weight: f64) {
    for (t, v) in total.iter_mut().zip(&tree.importance) {
        *t += weight * v;
    }
}

fn normalized(mut importance: Vec<f64>) -> Vec<f64> {
    let sum: f64 = importance.iter().sum();
    if sum > 0.0 {
        for v in &mut importance {
            *v /= sum;
        }
    }
    importance
}

/// Result from AdaBoost fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaBoostResult {
    /// Predictions on training data: M1 scores, SAMME predicted labels, R2 values
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
    /// Weighted error of each weak learner (M1, SAMME) or its weighted average loss (R2), in
    /// round order.
    #[serde(default)]
    pub learner_errors: Vec<f64>,
    /// Sorted distinct class labels: [-1, 1] for M1, the class codes for SAMME, None for R2.
    #[serde(default)]
    pub classes: Option<Vec<f64>>,
    /// Why boosting stopped before n_estimators rounds, if it did.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Internal: the weak learners for prediction
    #[serde(skip)]
    pub(crate) learners: Vec<WeakTree>,
    /// Internal: the number of training columns, which prediction requires
    #[serde(skip)]
    pub(crate) n_features: usize,
}

impl ResultNotes for AdaBoostResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl std::fmt::Display for AdaBoostResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "AdaBoost Results")?;
        writeln!(f, "================")?;
        writeln!(f, "Type: {:?}", self.config.boost_type)?;
        writeln!(f, "Number of estimators: {}", self.n_estimators)?;
        if self.config.boost_type == AdaBoostType::Samme {
            if let Some(classes) = &self.classes {
                writeln!(f, "Classes: {}", classes.len())?;
            }
        }
        writeln!(
            f,
            "Weak learner: CART ({}), max depth {}",
            if self.config.boost_type == AdaBoostType::R2 {
                "least squares"
            } else {
                "Gini"
            },
            self.config.max_depth
        )?;

        if let Some(acc) = self.train_accuracy {
            writeln!(f, "Training accuracy: {:.4}", acc)?;
        }

        if !self.train_error.is_empty() {
            writeln!(
                f,
                "Final training error: {:.4}",
                self.train_error.last().unwrap()
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

        let total: f64 = indexed.iter().map(|(_, v)| v).sum();
        for (i, importance) in indexed.iter().take(10) {
            let name = match &self.feature_names {
                Some(names) => names.get(*i).cloned().unwrap_or_else(|| format!("X{}", i)),
                None => format!("X{}", i),
            };
            let pct = if total > 0.0 {
                importance / total * 100.0
            } else {
                0.0
            };
            writeln!(f, "  {}: {:.1}%", name, pct)?;
        }

        write!(f, "{}", notes_text(self))
    }
}

/// Run AdaBoost.M1 for binary classification.
fn adaboost_m1(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &AdaBoostConfig,
) -> EconResult<AdaBoostResult> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let labels = [-1.0, 1.0];
    let codes: Vec<usize> = y.iter().map(|&v| usize::from(v > 0.0)).collect();
    let target = WeakTarget::Class {
        codes: &codes,
        labels: &labels,
    };
    let orders = feature_orders(x)?;

    // Initialize sample weights
    let mut weights = vec![1.0 / n_samples as f64; n_samples];

    let mut learners = try_with_capacity(config.n_estimators, "AdaBoost learners")?;
    let mut alphas = try_with_capacity(config.n_estimators, "AdaBoost weights")?;
    let mut train_error = try_with_capacity(config.n_estimators, "AdaBoost error path")?;
    let mut learner_errors = try_with_capacity(config.n_estimators, "AdaBoost learner errors")?;
    let mut importance = vec![0.0; n_features];
    let mut warnings = Vec::new();

    // Cumulative predictions (sum of weighted predictions)
    let mut cum_predictions = Array1::zeros(n_samples);

    for round in 0..config.n_estimators {
        checkpoint()?;

        // Fit weak learner
        let tree = fit_weak_tree(x, &orders, &target, &weights, config.max_depth);
        let predictions = tree.predict(x);
        let miss: Vec<bool> = (0..n_samples)
            .map(|i| predictions[i] != labels[codes[i]])
            .collect();
        let error = weighted_error(&miss, &weights);

        if error >= 0.5 - CHANCE_MARGIN {
            // Weak learner is not better than random, stop
            if round == 0 {
                return Err(no_better_than_chance(error, 0.5));
            }
            warnings.push(stopped_early(round, config.n_estimators, error, 0.5));
            break;
        }

        // Compute alpha (estimator weight); a perfect fit is clamped
        let error_clamped = error.clamp(1e-10, 1.0 - 1e-10);
        let alpha = config.learning_rate * 0.5 * ((1.0 - error_clamped) / error_clamped).ln();

        // Update cumulative predictions
        for i in 0..n_samples {
            cum_predictions[i] += alpha * predictions[i];
        }

        // Update weights
        for i in 0..n_samples {
            weights[i] *= (if miss[i] { alpha } else { -alpha }).exp();
        }

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= weight_sum;
        }

        // Track importance
        add_importance(&mut importance, &tree, alpha);

        // Compute training error
        let mut errors = 0;
        for i in 0..n_samples {
            let sign_pred = if cum_predictions[i] >= 0.0 { 1.0 } else { -1.0 };
            if (sign_pred - labels[codes[i]]).abs() > 0.5 {
                errors += 1;
            }
        }
        train_error.push(errors as f64 / n_samples as f64);

        alphas.push(alpha);
        learner_errors.push(error);
        learners.push(tree);
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

    Ok(AdaBoostResult {
        predictions,
        class_predictions: Some(class_predictions),
        feature_importances: normalized(importance),
        n_estimators: learners.len(),
        train_accuracy: Some(accuracy),
        train_error,
        estimator_weights: alphas,
        config: config.clone(),
        feature_names: None,
        learner_errors,
        classes: Some(labels.to_vec()),
        warnings,
        learners,
        n_features: x.ncols(),
    })
}

/// The argmax of each row's class scores (ties: the smallest label), as labels.
fn scores_to_labels(scores: &Array2<f64>, labels: &[f64]) -> Vec<f64> {
    scores
        .rows()
        .into_iter()
        .map(|row| {
            let mut best = 0;
            for (k, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = k;
                }
            }
            labels[best]
        })
        .collect()
}

/// Add `alpha` to each row's score of the class its predicted label stands for.
fn add_votes(scores: &mut Array2<f64>, pred: &Array1<f64>, labels: &[f64], alpha: f64) {
    for (i, &h) in pred.iter().enumerate() {
        if let Ok(k) = labels.binary_search_by(|l| l.total_cmp(&h)) {
            scores[[i, k]] += alpha;
        }
    }
}

/// Run SAMME for K classes (Zhu et al. 2009, Algorithm 2).
fn adaboost_samme(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &AdaBoostConfig,
) -> EconResult<AdaBoostResult> {
    let n_samples = x.nrows();
    if let Some((i, v)) = y
        .iter()
        .enumerate()
        .find(|(_, v)| v.fract() != 0.0 || v.abs() > f64::from(i32::MAX))
    {
        return Err(EconError::InvalidSpecification {
            message: format!("SAMME needs integer class codes (row {i} is {v})"),
        });
    }
    let mut labels: Vec<f64> = y.to_vec();
    labels.sort_by(f64::total_cmp);
    labels.dedup();
    if labels.len() < 2 {
        return Err(EconError::InvalidSpecification {
            message: "SAMME needs at least two classes".to_string(),
        });
    }
    if labels.len() > SAMME_MAX_CLASSES {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "SAMME supports at most {SAMME_MAX_CLASSES} classes (got {})",
                labels.len()
            ),
        });
    }
    let codes: Vec<usize> = y
        .iter()
        .map(|v| labels.partition_point(|l| l < v))
        .collect();
    let bound = 1.0 - 1.0 / labels.len() as f64;
    let orders = feature_orders(x)?;

    let mut weights = vec![1.0 / n_samples as f64; n_samples];
    let mut learners = try_with_capacity(config.n_estimators, "AdaBoost learners")?;
    let mut alphas = try_with_capacity(config.n_estimators, "AdaBoost weights")?;
    let mut train_error = try_with_capacity(config.n_estimators, "AdaBoost error path")?;
    let mut learner_errors = try_with_capacity(config.n_estimators, "AdaBoost learner errors")?;
    let mut importance = vec![0.0; x.ncols()];
    let mut warnings = Vec::new();
    let mut scores = try_zeros2(n_samples, labels.len(), "SAMME class scores")?;

    for round in 0..config.n_estimators {
        checkpoint()?;

        let (tree, error, alpha, new_weights) = match samme_round(
            x,
            &orders,
            &codes,
            &labels,
            &weights,
            config.max_depth,
            config.learning_rate,
        ) {
            Ok(fit) => fit,
            Err(error) => {
                if round == 0 {
                    return Err(no_better_than_chance(error, bound));
                }
                warnings.push(stopped_early(round, config.n_estimators, error, bound));
                break;
            }
        };
        weights = new_weights;

        add_votes(&mut scores, &tree.predict(x), &labels, alpha);
        add_importance(&mut importance, &tree, alpha);

        let voted = scores_to_labels(&scores, &labels);
        let wrong = voted.iter().zip(y.iter()).filter(|(a, b)| a != b).count();
        train_error.push(wrong as f64 / n_samples as f64);

        alphas.push(alpha);
        learner_errors.push(error);
        learners.push(tree);
    }

    let predictions = scores_to_labels(&scores, &labels);
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| a == b)
        .count();

    Ok(AdaBoostResult {
        class_predictions: Some(predictions.iter().map(|&p| p as i32).collect()),
        predictions,
        feature_importances: normalized(importance),
        n_estimators: learners.len(),
        train_accuracy: Some(correct as f64 / n_samples as f64),
        train_error,
        estimator_weights: alphas,
        config: config.clone(),
        feature_names: None,
        learner_errors,
        classes: Some(labels),
        warnings,
        learners,
        n_features: x.ncols(),
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
    let y_vec: Vec<f64> = y.to_vec();
    let target = WeakTarget::Regression(&y_vec);
    let orders = feature_orders(x)?;

    // Initialize sample weights
    let mut weights = vec![1.0 / n_samples as f64; n_samples];

    let mut learners = try_with_capacity(config.n_estimators, "AdaBoost learners")?;
    let mut betas = try_with_capacity(config.n_estimators, "AdaBoost betas")?;
    let mut train_error = try_with_capacity(config.n_estimators, "AdaBoost error path")?;
    let mut learner_errors = try_with_capacity(config.n_estimators, "AdaBoost learner errors")?;
    let mut importance = vec![0.0; n_features];

    // Store predictions from each estimator
    let mut all_predictions: Vec<Array1<f64>> = Vec::new();

    for _ in 0..config.n_estimators {
        checkpoint()?;

        // Fit weak learner
        let tree = fit_weak_tree(x, &orders, &target, &weights, config.max_depth);
        let predictions = tree.predict(x);

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
            learner_errors.push(0.0);
            learners.push(tree);
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
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 1e-10 {
            for w in &mut weights {
                *w /= weight_sum;
            }
        }

        // Track importance
        add_importance(&mut importance, &tree, (1.0 / beta).ln());

        all_predictions.push(predictions);
        betas.push(beta);
        learner_errors.push(avg_loss);
        learners.push(tree);

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

    Ok(AdaBoostResult {
        predictions: predictions.to_vec(),
        class_predictions: None,
        feature_importances: normalized(importance),
        n_estimators: learners.len(),
        train_accuracy: None,
        train_error,
        estimator_weights: betas.iter().map(|b: &f64| (1.0 / b).ln()).collect(),
        config: config.clone(),
        feature_names: None,
        learner_errors,
        classes: None,
        warnings: Vec::new(),
        learners,
        n_features: x.ncols(),
    })
}

/// Compute weighted median predictions from all estimators.
fn compute_weighted_median_predictions(predictions: &[Array1<f64>], betas: &[f64]) -> Array1<f64> {
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
        pred_weights.sort_by(|a, b| a.0.total_cmp(&b.0));

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

/// Refuse settings the fit cannot honour.
fn validate_adaboost_config(c: &AdaBoostConfig) -> EconResult<()> {
    let invalid = |message: String| Err(EconError::InvalidSpecification { message });
    if c.n_estimators == 0 {
        return invalid("n_estimators must be at least 1 (got 0)".to_string());
    }
    if !(1..=ADABOOST_MAX_DEPTH).contains(&c.max_depth) {
        return invalid(format!(
            "max_depth must be between 1 and {ADABOOST_MAX_DEPTH} (got {}); rpart's maxdepth limit",
            c.max_depth
        ));
    }
    if !c.learning_rate.is_finite() || c.learning_rate <= 0.0 {
        return invalid(format!(
            "learning_rate must be finite and positive (got {})",
            c.learning_rate
        ));
    }
    Ok(())
}

/// Fit an AdaBoost model.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target values (n_samples)
///   - For M1: binary labels (-1 or +1)
///   - For SAMME: integer class codes
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

    validate_adaboost_config(config)?;
    super::xgboost::check_finite_xy(x, y)?;

    match config.boost_type {
        AdaBoostType::M1 => {
            // Validate binary labels
            if let Some((i, v)) = y.iter().enumerate().find(|&(_, &v)| v != 1.0 && v != -1.0) {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "AdaBoost.M1 needs labels -1 and +1 (row {i} is {v}); use SAMME for other class codes"
                    ),
                });
            }
            adaboost_m1(&x, &y, config)
        }
        AdaBoostType::Samme => adaboost_samme(&x, &y, config),
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
/// Predictions: signed scores for M1, predicted labels for SAMME, continuous values for R2.
pub fn adaboost_predict(result: &AdaBoostResult, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
    let n_samples = x.nrows();

    if result.learners.is_empty() {
        return Err(EconError::Computation(
            "Model has no fitted estimators".to_string(),
        ));
    }
    if x.ncols() != result.n_features {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "AdaBoost was trained on {} columns; the prediction matrix has {}",
                result.n_features,
                x.ncols()
            ),
        });
    }

    match result.config.boost_type {
        AdaBoostType::M1 => {
            // Weighted vote
            let mut predictions = Array1::zeros(n_samples);
            for (tree, &alpha) in result.learners.iter().zip(result.estimator_weights.iter()) {
                let pred = tree.predict(&x);
                for i in 0..n_samples {
                    predictions[i] += alpha * pred[i];
                }
            }
            Ok(predictions.to_vec())
        }
        AdaBoostType::Samme => {
            let labels = result.classes.as_deref().ok_or_else(|| {
                EconError::Computation("SAMME model has no class labels".to_string())
            })?;
            let mut scores = try_zeros2(n_samples, labels.len(), "SAMME class scores")?;
            for (tree, &alpha) in result.learners.iter().zip(result.estimator_weights.iter()) {
                add_votes(&mut scores, &tree.predict(&x), labels, alpha);
            }
            Ok(scores_to_labels(&scores, labels))
        }
        AdaBoostType::R2 => {
            // Weighted median
            let all_predictions: Vec<Array1<f64>> =
                result.learners.iter().map(|s| s.predict(&x)).collect();
            let betas: Vec<f64> = result
                .estimator_weights
                .iter()
                .map(|w| (-w).exp())
                .collect();
            Ok(compute_weighted_median_predictions(&all_predictions, &betas).to_vec())
        }
    }
}

/// Predict class labels using a fitted AdaBoost classifier: the sign for M1, the labels for SAMME.
pub fn adaboost_predict_class(result: &AdaBoostResult, x: ArrayView2<f64>) -> EconResult<Vec<i32>> {
    let predictions = adaboost_predict(result, x)?;
    match result.config.boost_type {
        AdaBoostType::M1 => Ok(predictions
            .iter()
            .map(|&p| if p >= 0.0 { 1 } else { -1 })
            .collect()),
        AdaBoostType::Samme => Ok(predictions.iter().map(|&p| p as i32).collect()),
        AdaBoostType::R2 => Err(EconError::InvalidSpecification {
            message: "class predictions need M1 or SAMME".to_string(),
        }),
    }
}

/// Run AdaBoost on a Dataset.
///
/// `y_col` is a numeric target (M1: -1/+1; SAMME: integer class codes; R2: numeric). Integer and boolean
/// (0/1) columns are accepted; a missing value (null or NaN) in the target or a feature
/// is an error naming the column (`EconError::NullValues`).
pub fn run_adaboost(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &AdaBoostConfig,
) -> EconResult<AdaBoostResult> {
    let (x, y_arr, _missing) =
        super::extract::extract_xy(dataset, y_col, x_cols, crate::data::columns::NaPolicy::Fail)?;
    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

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

    use crate::data::columns::{NaPolicy, numeric_column};
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
    /// The case's CSV for tests that use its data but none of its reference values.
    fn adaboost_sim() -> crate::data::Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/adaboost_sim.csv");
        crate::data::Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }
    fn ada(
        ds: &crate::data::Dataset,
        y: &str,
        kind: AdaBoostType,
        depth: usize,
        lr: f64,
    ) -> AdaBoostResult {
        let x = case_matrix(ds, &["x1", "x2", "x3"]);
        let yv = case_vector(ds, y);
        let mut cfg = AdaBoostConfig::default();
        cfg.boost_type = kind;
        cfg.n_estimators = 8;
        cfg.max_depth = depth;
        cfg.learning_rate = lr;
        adaboost(x.view(), yv.view(), &cfg).unwrap()
    }

    #[test]
    fn test_validate_adaboost_samme_depth1() {
        let (case, ds) = crate::test_support::load_case("ml_adaboost_samme");
        let r = ada(&ds, "y3", AdaBoostType::Samme, 1, 1.0);
        assert_eq!(r.classes.as_deref(), Some(&[0.0, 1.0, 2.0][..]));
        Check::new(&case)
            .vector("samme_d1_alpha", &r.estimator_weights, Tol::coef(120))
            .vector("samme_d1_err", &r.learner_errors, Tol::coef(120))
            .vector("samme_d1_pred", &r.predictions, Tol::exact())
            .scalar("samme_d1_acc", r.train_accuracy.unwrap(), Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_adaboost_samme_depth2() {
        let (case, ds) = crate::test_support::load_case("ml_adaboost_samme");
        let r = ada(&ds, "y3", AdaBoostType::Samme, 2, 1.0);
        Check::new(&case)
            .vector("samme_d2_alpha", &r.estimator_weights, Tol::coef(120))
            .vector("samme_d2_err", &r.learner_errors, Tol::coef(120))
            .vector("samme_d2_pred", &r.predictions, Tol::exact())
            .scalar("samme_d2_acc", r.train_accuracy.unwrap(), Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_adaboost_samme_learning_rate() {
        let (case, ds) = crate::test_support::load_case("ml_adaboost_samme");
        let r = ada(&ds, "y3", AdaBoostType::Samme, 2, 0.5);
        Check::new(&case)
            .vector("samme_d2_lr05_alpha", &r.estimator_weights, Tol::coef(120))
            .vector("samme_d2_lr05_pred", &r.predictions, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_adaboost_samme_first_round_weights() {
        let (case, ds) = crate::test_support::load_case("ml_adaboost_samme");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        let codes: Vec<usize> = case_vector(&ds, "y3").iter().map(|&v| v as usize).collect();
        let orders = feature_orders(&x.view()).unwrap();
        let w0 = vec![1.0 / 120.0; 120];
        let (_, e, alpha, w1) =
            samme_round(&x.view(), &orders, &codes, &[0.0, 1.0, 2.0], &w0, 1, 1.0).unwrap();
        assert!((e - 47.0 / 120.0).abs() < 1e-15); // 47 rows misclassified
        assert!((alpha - (146.0f64 / 47.0).ln()).abs() < 1e-14); // ln(73/47) + ln 2
        assert_eq!(
            w1.iter()
                .filter(|&&w| (w - 1.0 / 219.0).abs() < 1e-15)
                .count(),
            73
        );
        assert_eq!(
            w1.iter()
                .filter(|&&w| (w - 2.0 / 141.0).abs() < 1e-15)
                .count(),
            47
        );
        Check::new(&case)
            .vector("samme_d1_w1", &w1, Tol::coef(120))
            .finish();
    }

    #[test]
    fn test_validate_adaboost_m1_depth1() {
        let (case, ds) = crate::test_support::load_case("ml_adaboost_samme");
        let r = ada(&ds, "yb", AdaBoostType::M1, 1, 1.0);
        Check::new(&case)
            .vector("m1_d1_alpha", &r.estimator_weights, Tol::coef(120))
            .vector("m1_d1_score", &r.predictions, Tol::coef(120))
            .finish();
    }

    #[test]
    fn test_validate_adaboost_m1_depth3() {
        let (case, ds) = crate::test_support::load_case("ml_adaboost_samme");
        let r = ada(&ds, "yb", AdaBoostType::M1, 3, 1.0);
        Check::new(&case)
            .vector("m1_d3_alpha", &r.estimator_weights, Tol::coef(120))
            .vector("m1_d3_score", &r.predictions, Tol::coef(120))
            .finish();
    }

    #[test]
    fn test_validate_adaboost_r2_first_learner() {
        let (case, ds) = crate::test_support::load_case("ml_adaboost_samme");
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        let y = case_vector(&ds, "yr");
        let orders = feature_orders(&x.view()).unwrap();
        let w0 = vec![1.0 / 120.0; 120];
        let mut chk = Check::new(&case);
        for (depth, key) in [(1, "r2_tree1_d1"), (2, "r2_tree1_d2")] {
            let t = fit_weak_tree(
                &x.view(),
                &orders,
                &WeakTarget::Regression(y.as_slice().unwrap()),
                &w0,
                depth,
            );
            chk.vector(key, &t.predict(&x.view()).to_vec(), Tol::coef(120));
        }
        chk.finish();
    }

    #[test]
    fn test_samme_with_two_classes_is_m1() {
        let ds = adaboost_sim();
        let m1 = ada(&ds, "yb", AdaBoostType::M1, 1, 1.0);
        let s2 = ada(&ds, "yb", AdaBoostType::Samme, 1, 1.0);
        for (a, b) in s2.estimator_weights.iter().zip(&m1.estimator_weights) {
            assert!((a - 2.0 * b).abs() <= 1e-14 * a.abs());
        }
        assert_eq!(s2.class_predictions, m1.class_predictions);
    }

    #[test]
    fn test_adaboost_r2_max_depth_changes_the_fit() {
        let ds = adaboost_sim();
        let d1 = ada(&ds, "yr", AdaBoostType::R2, 1, 1.0);
        let d3 = ada(&ds, "yr", AdaBoostType::R2, 3, 1.0);
        assert_ne!(d1.predictions, d3.predictions);
    }

    #[test]
    fn test_adaboost_type_parses_aliases_and_rejects_unknown() {
        AdaBoostType::choices().validate().unwrap();
        assert_eq!(
            "AdaBoost.M1".parse::<AdaBoostType>().unwrap(),
            AdaBoostType::M1
        );
        assert_eq!(
            "SAMME".parse::<AdaBoostType>().unwrap(),
            AdaBoostType::Samme
        );
        assert_eq!(
            "adaboost-r2".parse::<AdaBoostType>().unwrap(),
            AdaBoostType::R2
        );
        let e = "real".parse::<AdaBoostType>().unwrap_err();
        assert!(
            e.to_string()
                .contains("unknown boost_type 'real'; expected one of: m1, r2, samme"),
            "{e}"
        );
    }

    #[test]
    fn test_adaboost_rejects_bad_parameters_and_labels() {
        let ds = adaboost_sim();
        let x = case_matrix(&ds, &["x1", "x2", "x3"]);
        let y3 = case_vector(&ds, "y3");
        let with = |f: &dyn Fn(&mut AdaBoostConfig)| {
            let mut c = AdaBoostConfig::default();
            f(&mut c);
            c
        };
        for (field, cfg) in [
            ("max_depth", with(&|c| c.max_depth = 0)),
            ("max_depth", with(&|c| c.max_depth = ADABOOST_MAX_DEPTH + 1)),
            ("n_estimators", with(&|c| c.n_estimators = 0)),
            ("learning_rate", with(&|c| c.learning_rate = f64::NAN)),
        ] {
            let e = adaboost(x.view(), y3.view(), &cfg).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.starts_with(field)),
                "{e}"
            );
        }
        let mut samme = AdaBoostConfig::default();
        samme.boost_type = AdaBoostType::Samme;
        let half = y3.mapv(|v| v + 0.5);
        let e = adaboost(x.view(), half.view(), &samme).unwrap_err();
        assert!(e.to_string().contains("integer class codes"), "{e}");
        let e = adaboost(x.view(), y3.view(), &AdaBoostConfig::default()).unwrap_err(); // M1 on 3 codes
        assert!(e.to_string().contains("-1 and +1"), "{e}");
        let many = Array1::from_shape_fn(120, |i| i as f64);
        let e = adaboost(x.view(), many.view(), &samme).unwrap_err();
        assert!(e.to_string().contains("at most 100 classes"), "{e}");
        let mut bad = x.clone();
        bad[[2, 1]] = f64::NAN;
        assert!(adaboost(bad.view(), y3.view(), &samme).is_err());
        // constant x: the first SAMME learner is a root leaf with error 2/3 = 1 - 1/K
        let flat = Array2::from_elem((120, 3), 1.0);
        let e = adaboost(flat.view(), y3.view(), &samme).unwrap_err();
        assert!(e.to_string().contains("no better than chance"), "{e}");
    }

    /// Test-only reference: regrow the tree re-sorting each node's rows from scratch.
    fn naive_node(
        x: &ArrayView2<f64>,
        target: &WeakTarget<'_>,
        w: &[f64],
        rows: &[usize],
        depth: usize,
        max_depth: usize,
    ) -> WeakNode {
        let sorted = |f: usize| {
            let mut r = rows.to_vec();
            r.sort(); // ties keep row order, as the stable partition does
            r.sort_by(|&a, &b| x[[a, f]].total_cmp(&x[[b, f]]));
            r
        };
        let by_first = sorted(0);
        let (mut total, mut totals) = (0.0, Vec::new());
        let (impurity, value) = match target {
            WeakTarget::Class { codes, labels } => {
                totals = vec![0.0; labels.len()];
                for &r in &by_first {
                    totals[codes[r]] += w[r];
                    total += w[r];
                }
                let sq: f64 = totals.iter().map(|t| t * t).sum();
                let mut best = 0;
                for (k, &t) in totals.iter().enumerate() {
                    if t > totals[best] {
                        best = k;
                    }
                }
                (total - sq / total, labels[best])
            }
            WeakTarget::Regression(y) => {
                let mut sum = 0.0;
                for &r in &by_first {
                    total += w[r];
                    sum += w[r] * y[r];
                }
                let mean = sum / total;
                totals.push(sum);
                let ss = by_first
                    .iter()
                    .fold(0.0, |ss, &r| ss + w[r] * (y[r] - mean).powi(2));
                (ss, mean)
            }
        };
        if depth >= max_depth || rows.len() < 2 || impurity <= 0.0 {
            return WeakNode::Leaf { value };
        }
        let mut best_imp = 1e-12 * impurity;
        let mut best: Option<(usize, usize, f64, Vec<usize>)> = None;
        for f in 0..x.ncols() {
            let order = sorted(f);
            let mut left = vec![0.0; totals.len()];
            let mut left_w = 0.0;
            for pos in 0..order.len() - 1 {
                let r = order[pos];
                left_w += w[r];
                match target {
                    WeakTarget::Class { codes, .. } => left[codes[r]] += w[r],
                    WeakTarget::Regression(y) => left[0] += w[r] * y[r],
                }
                let (a, b) = (x[[r, f]], x[[order[pos + 1], f]]);
                let right_w = total - left_w;
                if a >= b || left_w <= 0.0 || right_w <= 0.0 {
                    continue;
                }
                let imp = match target {
                    WeakTarget::Class { .. } => {
                        let (mut sq_l, mut sq_r, mut sq) = (0.0, 0.0, 0.0);
                        for (l, t) in left.iter().zip(&totals) {
                            sq_l += l * l;
                            sq_r += (t - l) * (t - l);
                            sq += t * t;
                        }
                        sq_l / left_w + sq_r / right_w - sq / total
                    }
                    WeakTarget::Regression(_) => {
                        let s_r = totals[0] - left[0];
                        left[0] * left[0] / left_w + s_r * s_r / right_w
                            - totals[0] * totals[0] / total
                    }
                };
                if imp > best_imp {
                    best_imp = imp;
                    best = Some((f, pos, split_threshold(a, b), order.clone()));
                }
            }
        }
        let Some((feature, pos, threshold, order)) = best else {
            return WeakNode::Leaf { value };
        };
        WeakNode::Split {
            feature,
            threshold,
            left: Box::new(naive_node(
                x,
                target,
                w,
                &order[..=pos],
                depth + 1,
                max_depth,
            )),
            right: Box::new(naive_node(
                x,
                target,
                w,
                &order[pos + 1..],
                depth + 1,
                max_depth,
            )),
        }
    }

    #[test]
    fn test_weak_tree_partition_matches_naive_regrowth() {
        let mut s = 46u64;
        let mut u = || (crate::ml::lcg_random(&mut s) % 1_000_000) as f64 / 1_000_000.0;
        let n = 150;
        // x3 has ties, so the partition's stability matters
        let x = Array2::from_shape_fn((n, 3), |(_, j)| {
            let v = u();
            if j == 2 { (v * 7.0).floor() } else { v }
        });
        let noise: Vec<f64> = (0..n).map(|_| 0.4 * (u() - 0.5)).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| x[[i, 0]] - 0.5 * x[[i, 1]] + 0.1 * x[[i, 2]] + noise[i])
            .collect();
        let codes: Vec<usize> = y
            .iter()
            .map(|&v| {
                if v < 0.1 {
                    0
                } else if v < 0.5 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let raw: Vec<f64> = (0..n).map(|_| 0.05 + u()).collect();
        let sum: f64 = raw.iter().sum();
        let w: Vec<f64> = raw.iter().map(|v| v / sum).collect();
        let labels = [0.0, 1.0, 2.0];
        let rows: Vec<usize> = (0..n).collect();
        let xv = x.view();
        let orders = feature_orders(&xv).unwrap();
        for target in [
            WeakTarget::Class {
                codes: &codes,
                labels: &labels,
            },
            WeakTarget::Regression(&y),
        ] {
            for depth in 1..=5 {
                let tree = fit_weak_tree(&xv, &orders, &target, &w, depth);
                assert_eq!(
                    tree.root,
                    naive_node(&xv, &target, &w, &rows, 0, depth),
                    "depth {depth}"
                );
            }
        }
    }

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
        let x_train = array![[1.0, 0.0], [2.0, 0.0], [8.0, 1.0], [9.0, 1.0],];
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
    fn test_adaboost_predict_refuses_wrong_column_count() {
        let x_train = array![[1.0, 0.0], [2.0, 0.0], [8.0, 1.0], [9.0, 1.0],];
        let y_train = array![-1.0, -1.0, 1.0, 1.0];
        for boost_type in [AdaBoostType::M1, AdaBoostType::Samme, AdaBoostType::R2] {
            let config = AdaBoostConfig {
                n_estimators: 5,
                boost_type,
                ..Default::default()
            };
            let result = adaboost(x_train.view(), y_train.view(), &config).unwrap();
            for x_test in [array![[1.5]], array![[1.5, 0.0, 2.0]]] {
                let err = adaboost_predict(&result, x_test.view()).unwrap_err();
                assert!(
                    matches!(err, EconError::InvalidSpecification { ref message } if message.contains("2 columns")),
                    "{boost_type:?}: {err}"
                );
            }
            assert!(adaboost_predict(&result, array![[1.5, 0.0]].view()).is_ok());
        }
    }

    #[test]
    fn test_adaboost_predict_class() {
        let x_train = array![[1.0], [2.0], [8.0], [9.0],];
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
        let x = array![[1.0], [2.0], [8.0], [9.0],];
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
    fn test_weak_tree_stump() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![-1.0, -1.0, 1.0, 1.0];
        let codes: Vec<usize> = y.iter().map(|&v| usize::from(v > 0.0)).collect();
        let labels = [-1.0, 1.0];
        let weights = vec![0.25; 4];
        let orders = feature_orders(&x.view()).unwrap();
        let target = WeakTarget::Class {
            codes: &codes,
            labels: &labels,
        };
        let tree = fit_weak_tree(&x.view(), &orders, &target, &weights, 1);
        let pred = tree.predict(&x.view());
        assert_eq!(pred[0], -1.0);
        assert_eq!(pred[1], -1.0);
        assert_eq!(pred[2], 1.0);
        assert_eq!(pred[3], 1.0);
    }
}
