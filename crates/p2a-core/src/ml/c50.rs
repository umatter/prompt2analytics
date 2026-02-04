//! C5.0 Decision Trees and Rule-Based Models.
//!
//! Pure Rust implementation of C5.0 based on Quinlan's algorithm.
//! Similar to R's C50 package.
//!
//! ## Features
//!
//! - **Information gain ratio** for attribute selection (handles bias toward many-valued attributes)
//! - **Global pruning** using pessimistic error estimates
//! - **Winnowing** for automatic feature selection
//! - **Rule extraction** from decision trees
//! - **Boosting** with adaptive reweighting (AdaBoost variant)
//!
//! ## Example
//!
//! ```rust,no_run
//! use p2a_core::ml::{c50, C50Config};
//! use ndarray::array;
//!
//! // Classification with C5.0
//! let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
//! let y = array![0.0, 0.0, 1.0, 1.0];
//!
//! let config = C50Config {
//!     trials: 10,     // Boosting iterations
//!     winnow: true,   // Feature selection
//!     rules: false,   // Output tree, not rules
//!     min_cases: 2,
//!     ..Default::default()
//! };
//!
//! let result = c50(x.view(), y.view(), &config).unwrap();
//! println!("Accuracy: {:.2}%", result.accuracy * 100.0);
//! ```
//!
//! ## References
//!
//! - Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
//! - Quinlan, J. R. (1996). "Improved Use of Continuous Attributes in C4.5".
//!   *Journal of Artificial Intelligence Research*, 4, 77-90.
//! - Freund, Y., & Schapire, R. E. (1997). "A Decision-Theoretic Generalization of
//!   On-Line Learning". *Journal of Computer and System Sciences*, 55(1), 119-139.
//! - R package C50: Kuhn, M., Weston, S., Coulter, N., & Quinlan, R. (2023).
//!   https://cran.r-project.org/package=C50

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Dataset;
use crate::errors::{EconError, EconResult};

/// Configuration for C5.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50Config {
    /// Number of boosting iterations (trials). Default: 1 (no boosting)
    pub trials: usize,
    /// Whether to extract rules from the tree. Default: false
    pub rules: bool,
    /// Whether to use winnowing for feature selection. Default: false
    pub winnow: bool,
    /// Minimum cases in a terminal node. Default: 2
    pub min_cases: usize,
    /// Confidence factor for pessimistic pruning (0-1). Lower = more pruning. Default: 0.25
    pub cf: f64,
    /// Disable global pruning. Default: false
    pub no_global_pruning: bool,
    /// Enable fuzzy thresholds for continuous attributes. Default: false
    pub fuzzy_threshold: bool,
    /// Fraction of training data to sample (0-1). Default: 0.0 (use all data)
    pub sample: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Enable early stopping for boosting. Default: true
    pub early_stopping: bool,
}

impl Default for C50Config {
    fn default() -> Self {
        C50Config {
            trials: 1,
            rules: false,
            winnow: false,
            min_cases: 2,
            cf: 0.25,
            no_global_pruning: false,
            fuzzy_threshold: false,
            sample: 0.0,
            seed: None,
            early_stopping: true,
        }
    }
}

/// A rule extracted from the C5.0 tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50Rule {
    /// Rule ID
    pub id: usize,
    /// Conditions for the rule (feature index, operator, threshold)
    pub conditions: Vec<C50RuleCondition>,
    /// Predicted class
    pub predicted_class: usize,
    /// Confidence (proportion of training cases correctly classified)
    pub confidence: f64,
    /// Support (number of training cases covered)
    pub support: usize,
    /// Lift (ratio of rule confidence to base rate)
    pub lift: f64,
}

/// A condition in a C5.0 rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50RuleCondition {
    /// Feature index
    pub feature: usize,
    /// Comparison operator
    pub operator: ComparisonOp,
    /// Threshold value
    pub threshold: f64,
}

/// Comparison operators for rule conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    LessEqual,
    Greater,
    Equal,
    NotEqual,
}

impl std::fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonOp::LessEqual => write!(f, "<="),
            ComparisonOp::Greater => write!(f, ">"),
            ComparisonOp::Equal => write!(f, "=="),
            ComparisonOp::NotEqual => write!(f, "!="),
        }
    }
}

impl std::fmt::Display for C50RuleCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "X[{}] {} {:.4}",
            self.feature, self.operator, self.threshold
        )
    }
}

impl std::fmt::Display for C50Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let conditions_str: Vec<String> = self.conditions.iter().map(|c| c.to_string()).collect();
        write!(
            f,
            "Rule {}: IF {} THEN class={} (conf={:.3}, support={}, lift={:.2})",
            self.id,
            conditions_str.join(" AND "),
            self.predicted_class,
            self.confidence,
            self.support,
            self.lift
        )
    }
}

/// A node in the C5.0 tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50Node {
    /// Node ID
    pub id: usize,
    /// Number of observations
    pub n: usize,
    /// Class distribution at this node
    pub class_counts: Vec<usize>,
    /// Predicted class (majority class)
    pub predicted_class: usize,
    /// Entropy at this node
    pub entropy: f64,
    /// Split information (if internal node)
    pub split: Option<C50Split>,
    /// Left child (values <= threshold)
    pub left: Option<Box<C50Node>>,
    /// Right child (values > threshold)
    pub right: Option<Box<C50Node>>,
    /// Pessimistic error estimate (for pruning)
    pub error_estimate: f64,
}

/// A split in the C5.0 tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50Split {
    /// Feature index
    pub feature: usize,
    /// Split threshold
    pub threshold: f64,
    /// Information gain
    pub gain: f64,
    /// Gain ratio (gain / split info)
    pub gain_ratio: f64,
}

/// Result from C5.0 fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50Result {
    /// Root nodes for each boosting trial
    #[serde(skip)]
    pub trees: Vec<C50Node>,
    /// Boosting weights for each tree
    pub boosting_weights: Vec<f64>,
    /// Number of boosting trials actually performed
    pub actual_trials: usize,
    /// Extracted rules (if rules=true)
    pub rules: Option<Vec<C50Rule>>,
    /// Variable importance (sum of gain ratios)
    pub variable_importance: Vec<f64>,
    /// Training predictions
    pub predictions: Vec<usize>,
    /// Training accuracy
    pub accuracy: f64,
    /// Number of classes
    pub n_classes: usize,
    /// Class labels (as integers)
    pub class_labels: Vec<usize>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Winnowed features (indices of selected features)
    pub selected_features: Option<Vec<usize>>,
    /// Configuration used
    pub config: C50Config,
}

impl std::fmt::Display for C50Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "C5.0 Classification Model")?;
        writeln!(f, "=========================")?;
        writeln!(
            f,
            "Boosting trials: {} (requested {})",
            self.actual_trials, self.config.trials
        )?;
        writeln!(f, "Classes: {}", self.n_classes)?;
        writeln!(f, "Accuracy: {:.2}%", self.accuracy * 100.0)?;

        if let Some(ref selected) = self.selected_features {
            writeln!(f)?;
            writeln!(f, "Selected features after winnowing: {:?}", selected)?;
        }

        writeln!(f)?;
        writeln!(f, "Variable Importance:")?;
        let mut indexed: Vec<(usize, f64)> = self
            .variable_importance
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .filter(|(_, v)| *v > 0.0)
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

        if let Some(ref rules) = self.rules {
            writeln!(f)?;
            writeln!(f, "Extracted Rules ({}):", rules.len())?;
            for rule in rules.iter().take(10) {
                writeln!(f, "  {}", rule)?;
            }
            if rules.len() > 10 {
                writeln!(f, "  ... ({} more rules)", rules.len() - 10)?;
            }
        }

        if self.actual_trials > 1 {
            writeln!(f)?;
            writeln!(
                f,
                "Boosting weights: {:?}",
                self.boosting_weights
                    .iter()
                    .take(10)
                    .map(|w| format!("{:.3}", w))
                    .collect::<Vec<_>>()
            )?;
        }

        Ok(())
    }
}

/// Simple LCG random number generator.
fn lcg_random(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) ^ *state) as usize
}

/// Compute entropy of class distribution.
fn entropy(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        })
        .sum()
}

/// Compute information gain for a split.
/// Gain(S, A) = Entropy(S) - sum((|Sv|/|S|) * Entropy(Sv))
fn information_gain(parent_counts: &[usize], left_counts: &[usize], right_counts: &[usize]) -> f64 {
    let parent_total: usize = parent_counts.iter().sum();
    let left_total: usize = left_counts.iter().sum();
    let right_total: usize = right_counts.iter().sum();

    if parent_total == 0 || left_total == 0 || right_total == 0 {
        return 0.0;
    }

    let parent_entropy = entropy(parent_counts, parent_total);
    let left_entropy = entropy(left_counts, left_total);
    let right_entropy = entropy(right_counts, right_total);

    let n = parent_total as f64;
    parent_entropy
        - (left_total as f64 / n) * left_entropy
        - (right_total as f64 / n) * right_entropy
}

/// Compute split information for gain ratio calculation.
/// SplitInfo(S, A) = -sum((|Sv|/|S|) * log2(|Sv|/|S|))
fn split_info(left_count: usize, right_count: usize) -> f64 {
    let total = (left_count + right_count) as f64;
    if total == 0.0 {
        return 1.0; // Avoid division by zero
    }

    let mut info = 0.0;
    if left_count > 0 {
        let p_left = left_count as f64 / total;
        info -= p_left * p_left.ln();
    }
    if right_count > 0 {
        let p_right = right_count as f64 / total;
        info -= p_right * p_right.ln();
    }

    if info < 1e-10 {
        1e-10 // Avoid division by zero in gain ratio
    } else {
        info
    }
}

/// Compute gain ratio = information_gain / split_info
/// This corrects for bias toward attributes with many values (Quinlan 1986).
fn gain_ratio(
    parent_counts: &[usize],
    left_counts: &[usize],
    right_counts: &[usize],
) -> (f64, f64) {
    let gain = information_gain(parent_counts, left_counts, right_counts);
    let left_total: usize = left_counts.iter().sum();
    let right_total: usize = right_counts.iter().sum();
    let split = split_info(left_total, right_total);
    (gain, gain / split)
}

/// Pessimistic error estimate for pruning (Quinlan 1993).
/// Uses a continuity correction based on the confidence factor.
fn pessimistic_error(n: usize, errors: usize, cf: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }

    // Use normal approximation for large samples
    // z value for confidence level (cf = 0.25 corresponds to z ~ 0.69)
    let z = 0.69 * (1.0 - cf) / cf;

    let f = errors as f64 / n as f64;
    let n_f = n as f64;

    // Wilson score interval upper bound
    let numerator =
        f + z * z / (2.0 * n_f) + z * ((f * (1.0 - f) / n_f + z * z / (4.0 * n_f * n_f)).sqrt());
    let denominator = 1.0 + z * z / n_f;

    (numerator / denominator) * n_f
}

/// Build a C5.0 tree recursively.
fn build_c50_tree(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    weights: &Array1<f64>,
    indices: &[usize],
    config: &C50Config,
    n_classes: usize,
    depth: usize,
    node_id: &mut usize,
    importance: &mut Array1<f64>,
    selected_features: Option<&[usize]>,
) -> C50Node {
    let id = *node_id;
    *node_id += 1;

    let n = indices.len();

    // Compute weighted class counts
    let mut class_counts = vec![0usize; n_classes];
    let mut weighted_counts = vec![0.0f64; n_classes];
    for &i in indices {
        let class = y[i] as usize;
        class_counts[class] += 1;
        weighted_counts[class] += weights[i];
    }

    // Find majority class (using weighted counts)
    let (predicted_class, _) = weighted_counts
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    let current_entropy = entropy(&class_counts, n);

    // Count errors at this node
    let errors: usize = class_counts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != predicted_class)
        .map(|(_, &c)| c)
        .sum();

    let error_estimate = pessimistic_error(n, errors, config.cf);

    // Check stopping conditions
    if n < config.min_cases * 2 || current_entropy < 1e-10 {
        return C50Node {
            id,
            n,
            class_counts,
            predicted_class,
            entropy: current_entropy,
            split: None,
            left: None,
            right: None,
            error_estimate,
        };
    }

    // Find best split using gain ratio
    let features_to_try: Vec<usize> = match selected_features {
        Some(f) => f.to_vec(),
        None => (0..x.ncols()).collect(),
    };

    let mut best_gain_ratio = 0.0;
    let mut best_split: Option<(usize, f64, f64, f64)> = None;

    // Compute average gain to determine threshold (C4.5 heuristic)
    let mut all_gains: Vec<f64> = Vec::new();

    for &feature in &features_to_try {
        // Get sorted values for this feature
        let mut values: Vec<(f64, usize)> = indices.iter().map(|&i| (x[[i, feature]], i)).collect();
        values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Scan through possible thresholds
        let mut left_counts = vec![0usize; n_classes];

        for j in 0..values.len() - 1 {
            let (x_val, idx) = values[j];
            let class = y[idx] as usize;
            left_counts[class] += 1;

            // Skip if next value is the same
            let next_x = values[j + 1].0;
            if (next_x - x_val).abs() < 1e-10 {
                continue;
            }

            // Check minimum cases constraint
            let left_total: usize = left_counts.iter().sum();
            let right_total = n - left_total;
            if left_total < config.min_cases || right_total < config.min_cases {
                continue;
            }

            // Compute right counts
            let right_counts: Vec<usize> = class_counts
                .iter()
                .zip(left_counts.iter())
                .map(|(&t, &l)| t - l)
                .collect();

            let (gain, gr) = gain_ratio(&class_counts, &left_counts, &right_counts);
            all_gains.push(gain);

            if gr > best_gain_ratio {
                best_gain_ratio = gr;
                let threshold = (x_val + next_x) / 2.0;
                best_split = Some((feature, threshold, gain, gr));
            }
        }
    }

    // C4.5 heuristic: only consider splits with gain >= average gain
    let avg_gain = if all_gains.is_empty() {
        0.0
    } else {
        all_gains.iter().sum::<f64>() / all_gains.len() as f64
    };

    if let Some((feature, threshold, gain, gr)) = best_split {
        // Only split if gain is at least average
        if gain >= avg_gain * 0.99 {
            // Update importance
            importance[feature] += gr;

            // Partition indices
            let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| x[[i, feature]] <= threshold);

            if left_indices.is_empty() || right_indices.is_empty() {
                return C50Node {
                    id,
                    n,
                    class_counts,
                    predicted_class,
                    entropy: current_entropy,
                    split: None,
                    left: None,
                    right: None,
                    error_estimate,
                };
            }

            // Build children
            let left_child = build_c50_tree(
                x,
                y,
                weights,
                &left_indices,
                config,
                n_classes,
                depth + 1,
                node_id,
                importance,
                selected_features,
            );
            let right_child = build_c50_tree(
                x,
                y,
                weights,
                &right_indices,
                config,
                n_classes,
                depth + 1,
                node_id,
                importance,
                selected_features,
            );

            // Pruning decision: compare subtree error vs leaf error
            let subtree_error = left_child.error_estimate + right_child.error_estimate;

            if !config.no_global_pruning && subtree_error >= error_estimate + 0.5 {
                // Prune: return leaf node
                return C50Node {
                    id,
                    n,
                    class_counts,
                    predicted_class,
                    entropy: current_entropy,
                    split: None,
                    left: None,
                    right: None,
                    error_estimate,
                };
            }

            return C50Node {
                id,
                n,
                class_counts,
                predicted_class,
                entropy: current_entropy,
                split: Some(C50Split {
                    feature,
                    threshold,
                    gain,
                    gain_ratio: gr,
                }),
                left: Some(Box::new(left_child)),
                right: Some(Box::new(right_child)),
                error_estimate: subtree_error,
            };
        }
    }

    // No good split found - return leaf
    C50Node {
        id,
        n,
        class_counts,
        predicted_class,
        entropy: current_entropy,
        split: None,
        left: None,
        right: None,
        error_estimate,
    }
}

/// Predict class for a single observation using one tree.
fn predict_one_tree(node: &C50Node, x: &ArrayView1<f64>) -> usize {
    match &node.split {
        None => node.predicted_class,
        Some(split) => {
            if x[split.feature] <= split.threshold {
                predict_one_tree(node.left.as_ref().unwrap(), x)
            } else {
                predict_one_tree(node.right.as_ref().unwrap(), x)
            }
        }
    }
}

/// Predict class probabilities for a single observation using one tree.
fn predict_proba_one_tree(node: &C50Node, x: &ArrayView1<f64>, n_classes: usize) -> Vec<f64> {
    let leaf = match &node.split {
        None => node,
        Some(split) => {
            if x[split.feature] <= split.threshold {
                return predict_proba_one_tree(node.left.as_ref().unwrap(), x, n_classes);
            } else {
                return predict_proba_one_tree(node.right.as_ref().unwrap(), x, n_classes);
            }
        }
    };

    // Return normalized class counts
    let total: usize = leaf.class_counts.iter().sum();
    if total == 0 {
        return vec![1.0 / n_classes as f64; n_classes];
    }
    leaf.class_counts
        .iter()
        .map(|&c| c as f64 / total as f64)
        .collect()
}

/// Predict using boosted ensemble with weighted voting.
fn predict_boosted(
    trees: &[C50Node],
    weights: &[f64],
    x: &ArrayView1<f64>,
    n_classes: usize,
) -> usize {
    let mut class_votes = vec![0.0f64; n_classes];

    for (tree, &weight) in trees.iter().zip(weights.iter()) {
        let proba = predict_proba_one_tree(tree, x, n_classes);
        for (i, p) in proba.iter().enumerate() {
            class_votes[i] += weight * p;
        }
    }

    class_votes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Winnow features based on their importance.
/// Returns indices of selected features.
fn winnow_features(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &C50Config,
    n_classes: usize,
    rng_state: &mut u64,
) -> Vec<usize> {
    let n_features = x.ncols();
    let n_samples = x.nrows();

    // Build initial tree to compute importance
    let weights = Array1::ones(n_samples);
    let indices: Vec<usize> = (0..n_samples).collect();
    let mut importance = Array1::zeros(n_features);
    let mut node_id = 0;

    let _ = build_c50_tree(
        x,
        y,
        &weights,
        &indices,
        config,
        n_classes,
        0,
        &mut node_id,
        &mut importance,
        None,
    );

    // Compute threshold for selection
    // C5.0 uses a heuristic: keep features with importance > median importance
    let mut sorted_importance: Vec<f64> = importance.to_vec();
    sorted_importance.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Use 25th percentile as threshold
    let threshold_idx = n_features / 4;
    let threshold = sorted_importance.get(threshold_idx).copied().unwrap_or(0.0);

    // Select features above threshold (but keep at least 1)
    let mut selected: Vec<usize> = importance
        .iter()
        .enumerate()
        .filter(|&(_, imp)| *imp > threshold)
        .map(|(i, _)| i)
        .collect();

    // If no features selected, keep top 50%
    if selected.is_empty() {
        let median_idx = n_features / 2;
        let median = sorted_importance.get(median_idx).copied().unwrap_or(0.0);
        selected = importance
            .iter()
            .enumerate()
            .filter(|&(_, imp)| *imp >= median)
            .map(|(i, _)| i)
            .collect();
    }

    // Ensure at least one feature
    if selected.is_empty() {
        selected = vec![lcg_random(rng_state) % n_features];
    }

    selected
}

/// Extract rules from a C5.0 tree.
fn extract_rules(
    node: &C50Node,
    conditions: Vec<C50RuleCondition>,
    class_base_rates: &[f64],
    rules: &mut Vec<C50Rule>,
    rule_id: &mut usize,
) {
    match &node.split {
        None => {
            // Leaf node - create rule
            if !conditions.is_empty() {
                let total: usize = node.class_counts.iter().sum();
                let confidence = if total > 0 {
                    node.class_counts[node.predicted_class] as f64 / total as f64
                } else {
                    0.0
                };

                let base_rate = class_base_rates
                    .get(node.predicted_class)
                    .copied()
                    .unwrap_or(0.5);
                let lift = if base_rate > 0.0 {
                    confidence / base_rate
                } else {
                    1.0
                };

                rules.push(C50Rule {
                    id: *rule_id,
                    conditions: conditions.clone(),
                    predicted_class: node.predicted_class,
                    confidence,
                    support: total,
                    lift,
                });
                *rule_id += 1;
            }
        }
        Some(split) => {
            // Internal node - recurse to children
            let mut left_conditions = conditions.clone();
            left_conditions.push(C50RuleCondition {
                feature: split.feature,
                operator: ComparisonOp::LessEqual,
                threshold: split.threshold,
            });
            extract_rules(
                node.left.as_ref().unwrap(),
                left_conditions,
                class_base_rates,
                rules,
                rule_id,
            );

            let mut right_conditions = conditions;
            right_conditions.push(C50RuleCondition {
                feature: split.feature,
                operator: ComparisonOp::Greater,
                threshold: split.threshold,
            });
            extract_rules(
                node.right.as_ref().unwrap(),
                right_conditions,
                class_base_rates,
                rules,
                rule_id,
            );
        }
    }
}

/// Fit a C5.0 decision tree classifier.
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features)
/// * `y` - Target class labels (as floating point, will be converted to integers)
/// * `config` - C5.0 configuration
///
/// # Returns
///
/// C50Result containing the fitted model(s) and diagnostics.
///
/// # References
///
/// - Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.
/// - Implementation follows R package C50 by Kuhn et al.
pub fn c50(x: ArrayView2<f64>, y: ArrayView1<f64>, config: &C50Config) -> EconResult<C50Result> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(EconError::Computation(
            "Need at least 2 samples for C5.0".to_string(),
        ));
    }
    if n_samples != y.len() {
        return Err(EconError::Computation(
            "X and y must have same number of samples".to_string(),
        ));
    }
    if config.min_cases < 1 {
        return Err(EconError::Computation(
            "min_cases must be at least 1".to_string(),
        ));
    }

    let mut rng_state = config.seed.unwrap_or(42);

    // Determine number of classes
    let y_int: Vec<usize> = y.iter().map(|&v| v.round() as usize).collect();
    let max_class = y_int.iter().max().copied().unwrap_or(0);
    let n_classes = max_class + 1;

    if n_classes < 2 {
        return Err(EconError::Computation(
            "Need at least 2 classes for classification".to_string(),
        ));
    }

    // Compute class base rates for lift calculation
    let mut class_counts = vec![0usize; n_classes];
    for &c in &y_int {
        class_counts[c] += 1;
    }
    let class_base_rates: Vec<f64> = class_counts
        .iter()
        .map(|&c| c as f64 / n_samples as f64)
        .collect();

    // Winnowing: select features
    let selected_features = if config.winnow {
        Some(winnow_features(&x, &y, config, n_classes, &mut rng_state))
    } else {
        None
    };

    // Initialize sample weights
    let mut weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);
    // Convert y_int to f64 for the build_c50_tree function
    let y_f64: Vec<f64> = y_int.iter().map(|&c| c as f64).collect();
    let y_array = Array1::from_vec(y_f64);

    // Build boosted ensemble
    let mut trees: Vec<C50Node> = Vec::with_capacity(config.trials);
    let mut boosting_weights: Vec<f64> = Vec::with_capacity(config.trials);
    let mut importance = Array1::zeros(n_features);

    for trial in 0..config.trials {
        // Sample indices if configured
        let indices: Vec<usize> = if config.sample > 0.0 && config.sample < 1.0 {
            let sample_size = ((n_samples as f64 * config.sample).round() as usize).max(2);
            (0..sample_size)
                .map(|_| lcg_random(&mut rng_state) % n_samples)
                .collect()
        } else {
            (0..n_samples).collect()
        };

        let mut node_id = 0;
        let tree = build_c50_tree(
            &x,
            &y_array.view(),
            &weights,
            &indices,
            config,
            n_classes,
            0,
            &mut node_id,
            &mut importance,
            selected_features.as_deref(),
        );

        // Compute weighted error rate
        let mut weighted_error = 0.0;
        let mut total_weight = 0.0;
        for i in 0..n_samples {
            let pred = predict_one_tree(&tree, &x.row(i));
            total_weight += weights[i];
            if pred != y_int[i] {
                weighted_error += weights[i];
            }
        }

        let error_rate = weighted_error / total_weight;

        // Check for early stopping
        if config.early_stopping {
            if error_rate > 0.5 - 1e-10 {
                // Classifier is no better than random - stop boosting
                if trees.is_empty() {
                    // Keep at least one tree
                    trees.push(tree);
                    boosting_weights.push(1.0);
                }
                break;
            }
            if error_rate < 1e-10 {
                // Perfect classifier - no need to boost further
                trees.push(tree);
                boosting_weights.push(1.0);
                break;
            }
        }

        // Compute boosting weight (AdaBoost.M1 formula)
        // alpha = 0.5 * ln((1 - error) / error)
        let alpha = if error_rate > 0.0 && error_rate < 0.5 {
            0.5 * ((1.0 - error_rate) / error_rate).ln()
        } else if error_rate < 1e-10 {
            1.0 // Perfect classifier
        } else {
            0.0 // Very bad classifier
        };

        trees.push(tree);
        boosting_weights.push(alpha);

        // Update sample weights for next iteration
        if trial < config.trials - 1 && alpha > 0.0 {
            let tree = trees.last().unwrap();
            for i in 0..n_samples {
                let pred = predict_one_tree(tree, &x.row(i));
                if pred != y_int[i] {
                    weights[i] *= (alpha).exp();
                } else {
                    weights[i] *= (-alpha).exp();
                }
            }
            // Normalize weights
            let weight_sum: f64 = weights.sum();
            if weight_sum > 0.0 {
                weights /= weight_sum;
            }
        }
    }

    let actual_trials = trees.len();

    // Normalize boosting weights
    let weight_sum: f64 = boosting_weights.iter().sum();
    if weight_sum > 0.0 {
        for w in &mut boosting_weights {
            *w /= weight_sum;
        }
    }

    // Make predictions
    let predictions: Vec<usize> = (0..n_samples)
        .map(|i| predict_boosted(&trees, &boosting_weights, &x.row(i), n_classes))
        .collect();

    // Compute accuracy
    let correct: usize = predictions
        .iter()
        .zip(y_int.iter())
        .filter(|&(p, a)| *p == *a)
        .count();
    let accuracy = correct as f64 / n_samples as f64;

    // Extract rules if requested
    let rules = if config.rules {
        let mut all_rules = Vec::new();
        let mut rule_id = 0;
        for tree in &trees {
            extract_rules(
                tree,
                Vec::new(),
                &class_base_rates,
                &mut all_rules,
                &mut rule_id,
            );
        }
        // Sort rules by support and confidence
        all_rules.sort_by(|a, b| {
            b.support.cmp(&a.support).then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        Some(all_rules)
    } else {
        None
    };

    // Normalize importance
    let imp_sum: f64 = importance.sum();
    if imp_sum > 0.0 {
        importance /= imp_sum;
    }

    Ok(C50Result {
        trees,
        boosting_weights,
        actual_trials,
        rules,
        variable_importance: importance.to_vec(),
        predictions,
        accuracy,
        n_classes,
        class_labels: (0..n_classes).collect(),
        feature_names: None,
        selected_features,
        config: config.clone(),
    })
}

/// Predict using a fitted C5.0 model.
///
/// # Arguments
///
/// * `result` - Fitted C5.0 result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Predicted class labels for each observation.
pub fn c50_predict(result: &C50Result, x: ArrayView2<f64>) -> EconResult<Vec<usize>> {
    if result.trees.is_empty() {
        return Err(EconError::Computation("No trees in the model".to_string()));
    }

    let predictions: Vec<usize> = (0..x.nrows())
        .map(|i| {
            predict_boosted(
                &result.trees,
                &result.boosting_weights,
                &x.row(i),
                result.n_classes,
            )
        })
        .collect();

    Ok(predictions)
}

/// Predict class probabilities using a fitted C5.0 model.
///
/// # Arguments
///
/// * `result` - Fitted C5.0 result
/// * `x` - Feature matrix for prediction
///
/// # Returns
///
/// Class probabilities for each observation (n_samples x n_classes).
pub fn c50_predict_proba(result: &C50Result, x: ArrayView2<f64>) -> EconResult<Vec<Vec<f64>>> {
    if result.trees.is_empty() {
        return Err(EconError::Computation("No trees in the model".to_string()));
    }

    let predictions: Vec<Vec<f64>> = (0..x.nrows())
        .map(|i| {
            let mut class_probs = vec![0.0f64; result.n_classes];
            for (tree, &weight) in result.trees.iter().zip(result.boosting_weights.iter()) {
                let proba = predict_proba_one_tree(tree, &x.row(i), result.n_classes);
                for (j, p) in proba.iter().enumerate() {
                    class_probs[j] += weight * p;
                }
            }
            class_probs
        })
        .collect();

    Ok(predictions)
}

/// Run C5.0 on a Dataset.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `y_col` - Target column name (must be integer class labels)
/// * `x_cols` - Feature column names
/// * `config` - C5.0 configuration
///
/// # Returns
///
/// C50Result with fitted model and diagnostics.
pub fn run_c50(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &C50Config,
) -> EconResult<C50Result> {
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
            available: col_names,
        })?;

    let y: Vec<f64> = y_series
        .f64()
        .map_err(|_| EconError::Computation(format!("Column '{}' is not numeric", y_col)))?
        .into_no_null_iter()
        .collect();

    let y_arr = Array1::from_vec(y);

    let mut result = c50(x.view(), y_arr.view(), config)?;
    result.feature_names = Some(feature_names);

    Ok(result)
}

/// Run C5.0 with default configuration.
pub fn run_c50_default(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<C50Result> {
    run_c50(dataset, y_col, x_cols, &C50Config::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_c50_basic_classification() {
        // Simple linearly separable data
        let x = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [8.0, 8.0],
            [8.5, 8.5],
            [9.0, 9.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let config = C50Config {
            min_cases: 1,
            ..Default::default()
        };

        let result = c50(x.view(), y.view(), &config).unwrap();

        assert_eq!(result.n_classes, 2);
        assert_eq!(result.predictions.len(), 6);
        // Should achieve high accuracy on this simple data
        assert!(
            result.accuracy > 0.8,
            "Accuracy {} should be > 0.8",
            result.accuracy
        );
    }

    #[test]
    fn test_c50_boosting() {
        // XOR-like problem that benefits from boosting
        let x = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.9],
        ];
        let y = array![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];

        let config = C50Config {
            trials: 10,
            min_cases: 1,
            ..Default::default()
        };

        let result = c50(x.view(), y.view(), &config).unwrap();

        assert!(result.actual_trials >= 1);
        assert!(result.boosting_weights.len() == result.actual_trials);
    }

    #[test]
    fn test_c50_rule_extraction() {
        let x = array![
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [7.0, 3.0],
            [8.0, 4.0],
            [9.0, 5.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let config = C50Config {
            rules: true,
            min_cases: 1,
            ..Default::default()
        };

        let result = c50(x.view(), y.view(), &config).unwrap();

        assert!(result.rules.is_some());
        let rules = result.rules.as_ref().unwrap();
        assert!(!rules.is_empty(), "Should extract at least one rule");

        // Check rule structure
        for rule in rules {
            assert!(!rule.conditions.is_empty());
            assert!(rule.confidence >= 0.0 && rule.confidence <= 1.0);
        }
    }

    #[test]
    fn test_c50_winnowing() {
        // First feature is predictive, second is noise
        let x = array![
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 7.0],
            [7.0, 2.0],
            [8.0, 8.0],
            [9.0, 4.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let config = C50Config {
            winnow: true,
            min_cases: 1,
            ..Default::default()
        };

        let result = c50(x.view(), y.view(), &config).unwrap();

        assert!(result.selected_features.is_some());
        let selected = result.selected_features.as_ref().unwrap();
        assert!(!selected.is_empty(), "Should select at least one feature");
    }

    #[test]
    fn test_c50_predict() {
        let x_train = array![[1.0, 1.0], [2.0, 2.0], [8.0, 8.0], [9.0, 9.0],];
        let y_train = array![0.0, 0.0, 1.0, 1.0];

        let config = C50Config {
            min_cases: 1,
            ..Default::default()
        };

        let result = c50(x_train.view(), y_train.view(), &config).unwrap();

        let x_test = array![
            [1.5, 1.5], // Should be class 0
            [8.5, 8.5], // Should be class 1
        ];

        let predictions = c50_predict(&result, x_test.view()).unwrap();
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_c50_multiclass() {
        // 3-class problem
        let x = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [5.0, 5.0],
            [5.5, 5.5],
            [9.0, 9.0],
            [9.5, 9.5],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let config = C50Config {
            min_cases: 1,
            ..Default::default()
        };

        let result = c50(x.view(), y.view(), &config).unwrap();

        assert_eq!(result.n_classes, 3);
        assert!(result.accuracy > 0.5);
    }

    #[test]
    fn test_entropy_calculation() {
        // Pure node (all same class)
        let pure_counts = vec![5, 0];
        assert!((entropy(&pure_counts, 5) - 0.0).abs() < 1e-10);

        // Balanced binary
        let balanced_counts = vec![5, 5];
        let expected = -2.0 * (0.5 * 0.5f64.ln());
        assert!((entropy(&balanced_counts, 10) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gain_ratio_calculation() {
        let parent = vec![5, 5];
        let left = vec![5, 0];
        let right = vec![0, 5];

        let (gain, gr) = gain_ratio(&parent, &left, &right);

        // Perfect split should have high gain
        assert!(gain > 0.5);
        assert!(gr > 0.5);
    }

    #[test]
    fn test_c50_variable_importance() {
        // First feature is the only predictor
        let x = array![
            [1.0, 5.0],
            [2.0, 3.0],
            [3.0, 7.0],
            [7.0, 2.0],
            [8.0, 8.0],
            [9.0, 4.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let config = C50Config {
            min_cases: 1,
            ..Default::default()
        };

        let result = c50(x.view(), y.view(), &config).unwrap();

        // First feature should have higher importance
        assert!(
            result.variable_importance[0] >= result.variable_importance[1],
            "Feature 0 ({}) should have >= importance than Feature 1 ({})",
            result.variable_importance[0],
            result.variable_importance[1]
        );
    }
}
