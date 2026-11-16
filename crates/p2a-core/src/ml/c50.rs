//! C5.0 Decision Trees and Rule-Based Models.
//!
//! A C5.0-style classifier in pure Rust. Follows Quinlan (1993, 1996); not checked against R's
//! C50 package.
//!
//! ## Features
//!
//! - **Thresholds** by information gain with Quinlan's (1996) MDL correction, **attributes** by
//!   gain ratio among those with at least the average penalised gain (C4.5 Release 8)
//! - **Pruning**: pessimistic-error pruning of each subtree at confidence `cf` (C4.5's added
//!   errors; no subtree raising)
//! - **Winnowing**: keeps the features above the 25th percentile of a first tree's importance
//!   (a heuristic, not C5.0's winnowing)
//! - **Rule extraction** from decision trees
//! - **Boosting** with AdaBoost.M1 case weights (not C5.0's own boosting scheme), which enter
//!   every count of the next tree
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
//!   <https://cran.r-project.org/package=C50>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::extract::class_labels_line;
use crate::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::estimator::normal_quantile;

/// Configuration for C5.0.
///
/// Trees stop at depth 512 (root = 0): a node at that depth becomes a leaf
/// whatever the data, so no input can recurse deeper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C50Config {
    /// Number of boosting iterations (trials), 1 to 100 (the limit of R's C5.0).
    /// Default: 1 (no boosting)
    pub trials: usize,
    /// Whether to extract rules from the tree. Default: false
    pub rules: bool,
    /// Whether to winnow: keep the features above the 25th percentile of a first tree's
    /// importance (a heuristic, not C5.0's winnowing). Default: false
    pub winnow: bool,
    /// Minimum cases in a terminal node. Default: 2
    pub min_cases: usize,
    /// Confidence factor of the pessimistic-error pruning, 0 < cf < 1. Lower = more pruning.
    /// Default: 0.25
    pub cf: f64,
    /// Disable global pruning. Default: false
    pub no_global_pruning: bool,
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
    /// Predicted class, as the original class code (an entry of
    /// [`C50Result::class_labels`])
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
    /// Class distribution at this node; entry `k` counts class
    /// [`C50Result::class_labels`]`[k]`
    pub class_counts: Vec<usize>,
    /// Case weights of each class at this node (boosting reweights the cases; in the first
    /// trial every case weighs 1, so these equal `class_counts`). They drive the majority
    /// class, the entropy, the error estimate and the class probabilities.
    pub class_weights: Vec<f64>,
    /// Predicted class (majority class by weight), as an index into
    /// [`C50Result::class_labels`]
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
    /// Variable importance: the sum of the gain ratios of the splits on each feature in the
    /// final (pruned) trees of every trial, normalised to sum 1 (not C5.0's attribute usage)
    pub variable_importance: Vec<f64>,
    /// Training predictions, as class codes (entries of `class_labels`)
    pub predictions: Vec<usize>,
    /// Training accuracy
    pub accuracy: f64,
    /// Number of classes
    pub n_classes: usize,
    /// The distinct class codes of the target, in ascending order. Node
    /// `class_counts[k]`, node `predicted_class` and column `k` of
    /// [`c50_predict_proba`] refer to `class_labels[k]`; `predictions`,
    /// [`c50_predict`] and rule classes report the codes themselves.
    pub class_labels: Vec<usize>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Labels of the class codes when the target column was text, categorical or
    /// boolean; code k is `class_names[k]` ([`run_c50`] only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_names: Option<Vec<String>>,
    /// Winnowed features (indices of selected features)
    pub selected_features: Option<Vec<usize>>,
    /// Configuration used
    pub config: C50Config,
}

impl C50Result {
    /// The name of feature `j`: its column name when known, else `X{j}`.
    fn feature_text(&self, j: usize) -> String {
        self.feature_names
            .as_ref()
            .and_then(|names| names.get(j))
            .cloned()
            .unwrap_or_else(|| format!("X{j}"))
    }

    /// A rule condition with the feature's name, e.g. `x1 <= 3.2500`.
    pub fn condition_text(&self, condition: &C50RuleCondition) -> String {
        format!(
            "{} {} {:.4}",
            self.feature_text(condition.feature),
            condition.operator,
            condition.threshold
        )
    }

    /// The label of class code `code` when the target had labels ([`run_c50`] on a text,
    /// categorical or boolean column), else the code itself.
    pub fn class_text(&self, code: usize) -> String {
        self.class_names
            .as_ref()
            .and_then(|names| names.get(code))
            .cloned()
            .unwrap_or_else(|| code.to_string())
    }

    /// A rule with feature and class names.
    fn rule_text(&self, rule: &C50Rule) -> String {
        let conditions: Vec<String> = rule
            .conditions
            .iter()
            .map(|c| self.condition_text(c))
            .collect();
        format!(
            "Rule {}: IF {} THEN class={} (conf={:.3}, support={}, lift={:.2})",
            rule.id,
            conditions.join(" AND "),
            self.class_text(rule.predicted_class),
            rule.confidence,
            rule.support,
            rule.lift
        )
    }
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
        if let Some(names) = &self.class_names {
            writeln!(f, "{}", class_labels_line(names))?;
        }
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
        indexed.sort_by(|a, b| a.1.is_nan().cmp(&b.1.is_nan()).then(b.1.total_cmp(&a.1)));

        let total: f64 = indexed.iter().map(|(_, v)| v).sum();
        for (i, importance) in indexed.iter().take(10) {
            let name = self.feature_text(*i);
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
                writeln!(f, "  {}", self.rule_text(rule))?;
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

use super::lcg_random;

/// Entropy (natural log) of a class distribution given as case weights summing to `total`.
fn entropy(counts: &[f64], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|&&c| c > 0.0)
        .map(|&c| {
            let p = c / total;
            -p * p.ln()
        })
        .sum()
}

/// Information gain of a binary split, in nats, from the case weights of each class.
/// Gain(S, A) = Entropy(S) - sum((|Sv|/|S|) * Entropy(Sv))
fn information_gain(parent_counts: &[f64], left_counts: &[f64], right_counts: &[f64]) -> f64 {
    let parent_total: f64 = parent_counts.iter().sum();
    let left_total: f64 = left_counts.iter().sum();
    let right_total: f64 = right_counts.iter().sum();

    if parent_total <= 0.0 || left_total <= 0.0 || right_total <= 0.0 {
        return 0.0;
    }

    let parent_entropy = entropy(parent_counts, parent_total);
    let left_entropy = entropy(left_counts, left_total);
    let right_entropy = entropy(right_counts, right_total);

    parent_entropy
        - (left_total / parent_total) * left_entropy
        - (right_total / parent_total) * right_entropy
}

/// Split information of a binary partition with side weights `left` and `right`, in nats.
/// SplitInfo(S, A) = -sum((|Sv|/|S|) * ln(|Sv|/|S|))
fn split_info(left: f64, right: f64) -> f64 {
    let total = left + right;
    if total <= 0.0 {
        return 1.0; // Avoid division by zero
    }

    let mut info = 0.0;
    if left > 0.0 {
        let p_left = left / total;
        info -= p_left * p_left.ln();
    }
    if right > 0.0 {
        let p_right = right / total;
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
fn gain_ratio(parent_counts: &[f64], left_counts: &[f64], right_counts: &[f64]) -> (f64, f64) {
    let gain = information_gain(parent_counts, left_counts, right_counts);
    let left_total: f64 = left_counts.iter().sum();
    let right_total: f64 = right_counts.iter().sum();
    let split = split_info(left_total, right_total);
    (gain, gain / split)
}

/// The confidence level of C4.5's pessimistic error: `cf` and the squared normal deviate
/// `z^2` with `z = Phi^-1(1 - cf)`, computed once per fit.
#[derive(Debug, Clone, Copy)]
struct PruneLevel {
    cf: f64,
    coeff: f64,
}

impl PruneLevel {
    /// `cf` must lie in (0, 1), which [`c50`] checks before any tree is grown.
    fn new(cf: f64) -> EconResult<Self> {
        let z = normal_quantile(1.0 - cf)?;
        Ok(PruneLevel { cf, coeff: z * z })
    }
}

/// C4.5's added errors (`AddErrs`, Quinlan 1993, ch. 4; Release 8 `stats.c`): the errors to
/// add to `e` observed errors among `n` cases so that the total is the upper limit of the
/// binomial error at confidence `cf`.
///
/// - `e < 1e-6`: `n (1 - cf^(1/n))`, the exact limit for no errors;
/// - `e < 0.9999`: interpolated linearly between that and the value at `e = 1`;
/// - `e + 0.5 >= n`: `0.67 (n - e)`;
/// - otherwise the normal upper limit with continuity correction,
///   `n Pr - e` with `Pr = (e + 0.5 + z^2/2 + sqrt(z^2 ((e + 0.5)(1 - (e + 0.5)/n) + z^2/4))) / (n + z^2)`.
///
/// `n` and `e` are case weights.
fn added_errors(n: f64, e: f64, level: PruneLevel) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    if e < 1e-6 {
        n * (1.0 - (level.cf.ln() / n).exp())
    } else if e < 0.9999 {
        let v0 = n * (1.0 - (level.cf.ln() / n).exp());
        v0 + e * (added_errors(n, 1.0, level) - v0)
    } else if e + 0.5 >= n {
        0.67 * (n - e)
    } else {
        let c = level.coeff;
        let pr = (e + 0.5 + c / 2.0 + (c * ((e + 0.5) * (1.0 - (e + 0.5) / n) + c / 4.0)).sqrt())
            / (n + c);
        n * pr - e
    }
}

/// The largest number of distinct class codes C5.0 accepts for `n_samples`
/// rows: `max(2, min(1000, n/2))`.
///
/// This is a method-level cap in the sense of `nnet::multinom`'s `MaxNWts`:
/// a target with more distinct values than this is an
/// identifier or a continuous variable, not a class label.
pub(super) fn max_classes(n_samples: usize) -> usize {
    2.max(1000.min(n_samples / 2))
}

/// The largest class code C5.0 accepts: 2^53, beyond which an `f64` no
/// longer represents every integer exactly.
const MAX_EXACT_CODE: f64 = (1u64 << 53) as f64;

/// Validate the target's class codes and map them to dense indices.
///
/// Returns `(class_labels, dense_index_per_row)`: the distinct codes in
/// ascending order, and for each row the position of its code in that list.
/// Codes must be finite, non-negative integers no larger than
/// [`MAX_EXACT_CODE`].
fn class_codes(y: &ArrayView1<f64>) -> EconResult<(Vec<usize>, Vec<usize>)> {
    let mut codes = Vec::with_capacity(y.len());
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "the target has a non-finite value (NaN or infinite) at row {i}; \
                     C5.0 needs class codes 0, 1, 2, ..."
                ),
            });
        }
        if v < 0.0 || v.fract() != 0.0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "C5.0 needs non-negative integer class codes; the target holds {v} at \
                     row {i}. Recode the classes as 0, 1, 2, ..."
                ),
            });
        }
        if v > MAX_EXACT_CODE {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "class code {v} at row {i} is larger than 2^53 ({MAX_EXACT_CODE}) and \
                     cannot be represented exactly"
                ),
            });
        }
        codes.push(v as usize);
    }
    let mut class_labels = codes.clone();
    class_labels.sort_unstable();
    class_labels.dedup();
    let dense = codes
        .iter()
        .map(|code| match class_labels.binary_search(code) {
            Ok(k) | Err(k) => k,
        })
        .collect();
    Ok((class_labels, dense))
}

/// Deepest tree level (root = 0) the builder grows. A node at this depth
/// becomes a leaf in every mode, so no input can recurse deep enough to
/// overflow a 2 MiB worker stack (user decision, Tier 0). Trees that stay
/// shallower are unchanged.
const MAX_TREE_DEPTH: usize = 512;

/// One feature's best cut: `(threshold, gain, ratio)` with the gain MDL-penalised.
type FeatureCut = (f64, f64, f64);

/// The best cut of `feature` over the rows in `indices` as C4.5 Release 8 chooses it
/// (Quinlan 1996): the threshold with the largest weighted information gain among the cuts
/// leaving at least `min_cases` case weight on each side (the first on ties, at the midpoint
/// between the two values), its gain minus the MDL cost `ln(N_f - 1) / W` (N_f distinct values
/// of the feature in the node, W the node weight; nats, as the gain), and the penalised gain
/// divided by the split information of that partition. `None` when no cut is admissible.
fn best_feature_cut(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    weights: &Array1<f64>,
    indices: &[usize],
    class_weights: &[f64],
    min_cases: usize,
    feature: usize,
) -> Option<FeatureCut> {
    let n_classes = class_weights.len();
    let node_weight: f64 = class_weights.iter().sum();
    let min_w = min_cases as f64;
    let mut values: Vec<(f64, usize)> = indices.iter().map(|&i| (x[[i, feature]], i)).collect();
    values.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut left_counts = vec![0.0f64; n_classes];
    let mut right_counts = vec![0.0f64; n_classes];
    let mut left_total = 0.0f64;
    let mut distinct = 1usize;
    // (threshold, gain, left weight)
    let mut best: Option<(f64, f64, f64)> = None;
    for j in 0..values.len().saturating_sub(1) {
        let (x_val, idx) = values[j];
        left_counts[y[idx] as usize] += weights[idx];
        left_total += weights[idx];
        let next_x = values[j + 1].0;
        if (next_x - x_val).abs() < 1e-10 {
            continue;
        }
        distinct += 1;
        let right_total = node_weight - left_total;
        if left_total < min_w || right_total < min_w {
            continue;
        }
        for (r, (&t, &l)) in right_counts
            .iter_mut()
            .zip(class_weights.iter().zip(left_counts.iter()))
        {
            *r = t - l;
        }
        let gain = information_gain(class_weights, &left_counts, &right_counts);
        if best.is_none_or(|(_, g, _)| gain > g) {
            best = Some(((x_val + next_x) / 2.0, gain, left_total));
        }
    }
    let (threshold, gain, left) = best?;
    let penalised = gain - ((distinct - 1) as f64).ln() / node_weight;
    let ratio = penalised / split_info(left, node_weight - left);
    Some((threshold, penalised, ratio))
}

/// The split of the rows in `indices` as C4.5 Release 8 selects it, as
/// `(feature, threshold, gain, gain_ratio)` (gain MDL-penalised), or `None` when no feature
/// has a positive penalised gain.
///
/// Each feature's cut is [`best_feature_cut`]. Among the features with positive penalised
/// gain, those whose penalised gain is at least the average over them compete on gain ratio;
/// the largest ratio wins, the first feature on ties.
///
/// Kept out of [`build_c50_tree`], and never inlined into it, so the recursive
/// frame holds only node bookkeeping: unoptimised (opt-level 0) builds give
/// every local of the scan its own stack slot, and up to 513 of those frames
/// are live at once on a deep tree.
#[inline(never)]
fn best_c50_split(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    weights: &Array1<f64>,
    indices: &[usize],
    class_weights: &[f64],
    min_cases: usize,
    selected_features: Option<&[usize]>,
) -> Option<(usize, f64, f64, f64)> {
    let features_to_try: Vec<usize> = match selected_features {
        Some(f) => f.to_vec(),
        None => (0..x.ncols()).collect(),
    };
    let cuts: Vec<(usize, FeatureCut)> = features_to_try
        .iter()
        .filter_map(|&f| {
            best_feature_cut(x, y, weights, indices, class_weights, min_cases, f)
                .filter(|&(_, gain, _)| gain > 0.0)
                .map(|cut| (f, cut))
        })
        .collect();
    if cuts.is_empty() {
        return None;
    }
    let average = cuts.iter().map(|(_, (_, g, _))| g).sum::<f64>() / cuts.len() as f64;
    let mut best: Option<(usize, f64, f64, f64)> = None;
    for &(feature, (threshold, gain, ratio)) in &cuts {
        if gain >= average - 1e-12 && best.is_none_or(|(_, _, _, r)| ratio > r) {
            best = Some((feature, threshold, gain, ratio));
        }
    }
    best
}

/// Class counts and leaf statistics of one node.
struct C50NodeStats {
    class_counts: Vec<usize>,
    class_weights: Vec<f64>,
    predicted_class: usize,
    entropy: f64,
    error_estimate: f64,
}

/// Class counts, case weights per class, majority class (by weight, the first on ties),
/// entropy and pessimistic error estimate (both from the weights) of the rows in `indices`.
/// Never inlined, for the reason given at [`best_c50_split`].
#[inline(never)]
fn c50_node_stats(
    y: &ArrayView1<f64>,
    weights: &Array1<f64>,
    indices: &[usize],
    n_classes: usize,
    level: PruneLevel,
) -> C50NodeStats {
    let mut class_counts = vec![0usize; n_classes];
    let mut class_weights = vec![0.0f64; n_classes];
    for &i in indices {
        let class = y[i] as usize;
        class_counts[class] += 1;
        class_weights[class] += weights[i];
    }
    let mut predicted_class = 0;
    for (k, &w) in class_weights.iter().enumerate() {
        if w > class_weights[predicted_class] {
            predicted_class = k;
        }
    }
    let total: f64 = class_weights.iter().sum();
    let node_entropy = entropy(&class_weights, total);
    let errors = total - class_weights[predicted_class];

    C50NodeStats {
        class_counts,
        class_weights,
        predicted_class,
        entropy: node_entropy,
        error_estimate: errors + added_errors(total, errors, level),
    }
}

/// A leaf of `n` rows with statistics `stats`.
#[inline(never)]
fn c50_leaf(id: usize, n: usize, stats: C50NodeStats) -> Box<C50Node> {
    Box::new(C50Node {
        id,
        n,
        class_counts: stats.class_counts,
        class_weights: stats.class_weights,
        predicted_class: stats.predicted_class,
        entropy: stats.entropy,
        split: None,
        left: None,
        right: None,
        error_estimate: stats.error_estimate,
    })
}

/// The internal node for `split` over its grown children, or a leaf when
/// pruning prefers one, unless global pruning is off.
///
/// C4.5's replace-if-no-worse test (Release 8 `prune.c`): the node becomes a leaf when its
/// estimated errors as a leaf are at most the sum of its children's estimates plus 0.1. Subtree
/// raising (replacing the node by its largest branch) is not implemented.
#[inline(never)]
fn c50_join(
    id: usize,
    n: usize,
    stats: C50NodeStats,
    split: (usize, f64, f64, f64),
    left: Box<C50Node>,
    right: Box<C50Node>,
    no_global_pruning: bool,
) -> Box<C50Node> {
    // Pruning decision: compare subtree error vs leaf error
    let subtree_error = left.error_estimate + right.error_estimate;
    if !no_global_pruning && stats.error_estimate <= subtree_error + 0.1 {
        return c50_leaf(id, n, stats);
    }
    let (feature, threshold, gain, gain_ratio) = split;
    Box::new(C50Node {
        id,
        n,
        class_counts: stats.class_counts,
        class_weights: stats.class_weights,
        predicted_class: stats.predicted_class,
        entropy: stats.entropy,
        split: Some(C50Split {
            feature,
            threshold,
            gain,
            gain_ratio,
        }),
        left: Some(left),
        right: Some(right),
        error_estimate: subtree_error,
    })
}

/// Build a C5.0 tree recursively.
///
/// Up to `MAX_TREE_DEPTH + 1` frames of this function are live at once, so
/// it holds only node bookkeeping and boxed children: the statistics, the
/// split scan and node construction run in helpers that are never inlined
/// into it. That keeps 513 levels well inside a 2 MiB worker stack also in an
/// unoptimised (opt-level 0) build.
fn build_c50_tree(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    weights: &Array1<f64>,
    indices: &[usize],
    config: &C50Config,
    n_classes: usize,
    depth: usize,
    node_id: &mut usize,
    selected_features: Option<&[usize]>,
) -> EconResult<Box<C50Node>> {
    crate::resource::checkpoint()?;
    let id = *node_id;
    *node_id += 1;

    let n = indices.len();
    let stats = c50_node_stats(y, weights, indices, n_classes, PruneLevel::new(config.cf)?);
    let node_weight: f64 = stats.class_weights.iter().sum();

    // Check stopping conditions (the node weight against 2 * min_cases, as C4.5), then find
    // the best split using gain ratio
    let stop = depth >= MAX_TREE_DEPTH
        || node_weight < 2.0 * config.min_cases as f64
        || stats.entropy < 1e-10;
    let best = if stop {
        None
    } else {
        best_c50_split(
            x,
            y,
            weights,
            indices,
            &stats.class_weights,
            config.min_cases,
            selected_features,
        )
    };
    let Some(split) = best else {
        return Ok(c50_leaf(id, n, stats));
    };
    let (feature, threshold, _, _) = split;

    // Partition indices
    let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
        indices.iter().partition(|&&i| x[[i, feature]] <= threshold);
    if left_indices.is_empty() || right_indices.is_empty() {
        return Ok(c50_leaf(id, n, stats));
    }

    // Build children
    let left = build_c50_tree(
        x,
        y,
        weights,
        &left_indices,
        config,
        n_classes,
        depth + 1,
        node_id,
        selected_features,
    )?;
    let right = build_c50_tree(
        x,
        y,
        weights,
        &right_indices,
        config,
        n_classes,
        depth + 1,
        node_id,
        selected_features,
    )?;
    Ok(c50_join(
        id,
        n,
        stats,
        split,
        left,
        right,
        config.no_global_pruning,
    ))
}

/// Rescale case weights to mean 1 (sum n), the scale on which `min_cases` counts cases.
fn normalise_case_weights(weights: &mut Array1<f64>) {
    let n = weights.len() as f64;
    let total = weights.sum();
    if total > 0.0 && total.is_finite() {
        weights.mapv_inplace(|w| w * (n / total));
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

    // Return the normalized class weights
    let total: f64 = leaf.class_weights.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / n_classes as f64; n_classes];
    }
    leaf.class_weights.iter().map(|&c| c / total).collect()
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
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Adds the gain ratio of every split of the tree under `node` to `importance[feature]`.
/// Walked over the final trees, after pruning, so a pruned split counts nothing.
fn add_split_importance(node: &C50Node, importance: &mut Array1<f64>) {
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        if let Some(split) = &node.split {
            importance[split.feature] += split.gain_ratio;
        }
        for child in [&node.left, &node.right].into_iter().flatten() {
            stack.push(child.as_ref());
        }
    }
}

/// Winnow features based on their importance.
/// Returns indices of selected features.
fn winnow_features(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    config: &C50Config,
    n_classes: usize,
    rng_state: &mut u64,
) -> EconResult<Vec<usize>> {
    let n_features = x.ncols();
    let n_samples = x.nrows();

    // Build initial tree to compute importance
    let weights = Array1::ones(n_samples);
    let indices: Vec<usize> = (0..n_samples).collect();
    let mut node_id = 0;

    let first = build_c50_tree(
        x,
        y,
        &weights,
        &indices,
        config,
        n_classes,
        0,
        &mut node_id,
        None,
    )?;
    let mut importance = Array1::zeros(n_features);
    add_split_importance(&first, &mut importance);

    // Keep the features above the 25th percentile of the first tree's importance (a heuristic,
    // not C5.0's winnowing).
    let mut sorted_importance: Vec<f64> = importance.to_vec();
    sorted_importance.sort_by(f64::total_cmp);

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

    Ok(selected)
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

/// Fit a C5.0-style decision tree classifier: thresholds by information gain with the MDL
/// correction, attributes by gain ratio, pessimistic-error pruning of each subtree (no subtree
/// raising), optional AdaBoost.M1 boosting and winnowing (see the module docs).
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples x n_features); every value must be finite
/// * `y` - Target class codes: finite, non-negative integers (stored as `f64`)
///   no larger than 2^53. They need not be contiguous; predictions and rules
///   report the same codes. At most `max(2, min(1000, n/2))` distinct codes.
/// * `config` - C5.0 configuration (`trials` 1 to 100)
///
/// # Returns
///
/// C50Result containing the fitted model(s) and diagnostics.
///
/// # Errors
///
/// `InvalidSpecification` for invalid codes, too many classes, non-finite
/// features, `trials` outside 1..=100 or `cf` outside (0, 1); `Cancelled` once the installed
/// computation budget expires (checked per node and per trial).
///
/// # References
///
/// - Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.
/// - Quinlan, J. R. (1996). Improved Use of Continuous Attributes in C4.5. JAIR 4, 77-90.
/// - Follows Quinlan (1993, 1996); not checked against R's C50 package.
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
    if n_features == 0 {
        return Err(EconError::InvalidSpecification {
            message: "C5.0 needs at least one feature column".to_string(),
        });
    }
    if !(config.cf > 0.0 && config.cf < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the pruning confidence cf must satisfy 0 < cf < 1, got {}",
                config.cf
            ),
        });
    }
    if !(1..=100).contains(&config.trials) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "trials must be between 1 and 100 (the boosting limit of R's C5.0), got {}; \
                 use fewer boosting trials",
                config.trials
            ),
        });
    }
    for (j, column) in x.axis_iter(Axis(1)).enumerate() {
        let k = column.iter().filter(|v| !v.is_finite()).count();
        if k > 0 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "feature column {j} (0-based) has {k} non-finite values (NaN or infinite); \
                     remove or impute them first"
                ),
            });
        }
    }

    let mut rng_state = config.seed.unwrap_or(42);

    // Map the class codes to dense indices 0..K-1; everything below works on
    // the dense indices, and results are mapped back to the codes at the end.
    let (class_labels, y_int) = class_codes(&y)?;
    let n_classes = class_labels.len();

    if n_classes < 2 {
        return Err(EconError::Computation(
            "Need at least 2 classes for classification".to_string(),
        ));
    }
    let cap = max_classes(n_samples);
    if n_classes > cap {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "C5.0 supports at most {cap} classes for {n_samples} rows, but the target has \
                 {n_classes} distinct values; it looks like an identifier or a continuous \
                 variable. Use a regression method or recode the target"
            ),
        });
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

    // Convert y_int to f64 for the build_c50_tree function
    let y_f64: Vec<f64> = y_int.iter().map(|&c| c as f64).collect();
    let y_array = Array1::from_vec(y_f64);

    // Winnowing: select features
    let selected_features = if config.winnow {
        Some(winnow_features(
            &x,
            &y_array.view(),
            config,
            n_classes,
            &mut rng_state,
        )?)
    } else {
        None
    };

    // Case weights: 1 per case; boosting reweights them, and each trial grows its tree on
    // weights rescaled to mean 1 (C4.5 case weights), which every count of the tree uses.
    let mut weights = Array1::from_elem(n_samples, 1.0);

    // Build boosted ensemble
    let mut trees: Vec<C50Node> = Vec::with_capacity(config.trials);
    let mut boosting_weights: Vec<f64> = Vec::with_capacity(config.trials);

    for trial in 0..config.trials {
        crate::resource::checkpoint()?;
        // Sample indices if configured
        let indices: Vec<usize> = if config.sample > 0.0 && config.sample < 1.0 {
            let sample_size = ((n_samples as f64 * config.sample).round() as usize).max(2);
            (0..sample_size)
                .map(|_| lcg_random(&mut rng_state) % n_samples)
                .collect()
        } else {
            (0..n_samples).collect()
        };

        normalise_case_weights(&mut weights);
        let mut node_id = 0;
        let tree = *build_c50_tree(
            &x,
            &y_array.view(),
            &weights,
            &indices,
            config,
            n_classes,
            0,
            &mut node_id,
            selected_features.as_deref(),
        )?;

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
    // Report the original class codes, not the dense indices.
    let predictions: Vec<usize> = predictions.iter().map(|&k| class_labels[k]).collect();

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
        // Lift was computed on dense indices; report the class codes.
        for rule in &mut all_rules {
            rule.predicted_class = class_labels[rule.predicted_class];
        }
        // Sort rules by support and confidence
        all_rules.sort_by(|a, b| {
            b.support
                .cmp(&a.support)
                .then(b.confidence.total_cmp(&a.confidence))
        });
        Some(all_rules)
    } else {
        None
    };

    // Importance: the gain ratios of the splits of the final trees, normalised to sum 1.
    let mut importance = Array1::zeros(n_features);
    for tree in &trees {
        add_split_importance(tree, &mut importance);
    }
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
        class_labels,
        feature_names: None,
        class_names: None,
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
/// Predicted class codes (entries of `result.class_labels`) for each observation.
///
/// # Errors
///
/// `Computation` when the model has no trees; `InvalidSpecification` when
/// `result.class_labels` does not hold exactly `result.n_classes` codes (at
/// least one), since a prediction could then not be reported as a code.
pub fn c50_predict(result: &C50Result, x: ArrayView2<f64>) -> EconResult<Vec<usize>> {
    if result.trees.is_empty() {
        return Err(EconError::Computation("No trees in the model".to_string()));
    }
    if result.class_labels.len() != result.n_classes || result.n_classes == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the C5.0 model lists {} class labels for {} classes; it needs one label per \
                 class and at least one class",
                result.class_labels.len(),
                result.n_classes
            ),
        });
    }

    let predictions: Vec<usize> = (0..x.nrows())
        .map(|i| {
            let k = predict_boosted(
                &result.trees,
                &result.boosting_weights,
                &x.row(i),
                result.n_classes,
            );
            result.class_labels[k]
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
/// Class probabilities for each observation (n_samples x n_classes); column
/// `k` is the probability of class code `result.class_labels[k]`.
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
/// * `y_col` - Target column: non-negative integer class codes (integer or decimal
///   column), or a text, categorical or boolean column whose labels become the codes
///   0..K-1 in R `factor()` order (text in byte order, FALSE < TRUE), reported in
///   [`C50Result::class_names`]. A missing value (null or NaN) in the target or a feature
///   is an error naming the column (`EconError::NullValues`)
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
    let (x, target, _missing) = super::extract::extract_x_classes(
        dataset,
        y_col,
        x_cols,
        crate::data::columns::NaPolicy::Fail,
    )?;
    let feature_names: Vec<String> = x_cols.iter().map(|s| s.to_string()).collect();

    let mut result = c50(x.view(), target.y.view(), config)?;
    result.feature_names = Some(feature_names);
    result.class_names = target.labels;

    Ok(result)
}

/// Run C5.0 with default configuration.
pub fn run_c50_default(dataset: &Dataset, y_col: &str, x_cols: &[&str]) -> EconResult<C50Result> {
    run_c50(dataset, y_col, x_cols, &C50Config::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::array;

    fn cancelled_budget() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn two_cluster_x(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, 2), |(i, j)| {
            (if i < n / 2 { 1.0 } else { 8.0 }) + ((i * 7 + j * 3) % 5) as f64 * 0.1
        })
    }

    /// Uniform draws in [0, 1) from a fixed LCG.
    fn lcg_uniforms(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A tree's splits in preorder as (feature, threshold bits).
    fn split_signature(root: &C50Node) -> Vec<(usize, u64)> {
        let (mut out, mut stack) = (Vec::new(), vec![root]);
        while let Some(node) = stack.pop() {
            if let Some(s) = &node.split {
                out.push((s.feature, s.threshold.to_bits()));
            }
            for child in [&node.right, &node.left].into_iter().flatten() {
                stack.push(child.as_ref());
            }
        }
        out
    }

    #[test]
    fn test_c50_boosting_grows_different_trees() {
        // ML-17. The boosting weights reached only the leaves' majority vote, so every trial grew the
        // first tree again.
        let mut u = lcg_uniforms(17);
        let x = Array2::from_shape_fn((200, 3), |_| u());
        let y = Array1::from_shape_fn(200, |i| {
            let label = usize::from(x[[i, 0]] + x[[i, 1]] > 1.0);
            (if i % 7 == 0 { 1 - label } else { label }) as f64
        });
        let cfg = C50Config {
            trials: 10,
            early_stopping: false,
            no_global_pruning: true,
            ..Default::default()
        };
        let r = c50(x.view(), y.view(), &cfg).unwrap();
        let mut signatures: Vec<Vec<(usize, u64)>> = r.trees.iter().map(split_signature).collect();
        signatures.sort();
        signatures.dedup();
        assert!(
            signatures.len() >= 2,
            "{} trials grew {} distinct trees",
            r.trees.len(),
            signatures.len()
        );
    }

    #[test]
    fn test_c50_threshold_is_chosen_by_gain() {
        // ML-18. C4.5 R8 picks a continuous feature's threshold by information gain; the gain ratio
        // favours cuts near the edges (here 19.5, ratio 0.4356, against 12.5, ratio 0.3673).
        let x = Array2::from_shape_fn((24, 1), |(i, _)| (i + 1) as f64);
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let cfg = C50Config {
            no_global_pruning: true,
            ..Default::default()
        };
        let r = c50(x.view(), y.view(), &cfg).unwrap();
        let split = r.trees[0].split.as_ref().expect("root split");
        assert_eq!(split.threshold, 12.5);
    }

    #[test]
    fn test_c50_mdl_penalty_and_feature_average() {
        // ML-18. Before: the largest gain ratio over all cuts of all features, x2 at 2.5 (an edge peel).
        let (x, y) = attribute_example();
        let cfg = C50Config {
            no_global_pruning: true,
            ..Default::default()
        };
        let r = c50(x.view(), y.view(), &cfg).unwrap();
        let split = r.trees[0].split.as_ref().expect("root split");
        assert_eq!((split.feature, split.threshold), (0, 23.5));
        // x2 alone: its penalised gain is negative, so the root is a leaf.
        let x2 = x.slice(ndarray::s![.., 1..2]).to_owned();
        let r = c50(x2.view(), y.view(), &cfg).unwrap();
        assert!(r.trees[0].split.is_none());
    }

    /// The cut of [`best_feature_cut`] by a naive pass that sums the class weights on each side
    /// of every candidate threshold from scratch.
    fn naive_feature_cut(
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: &Array1<f64>,
        feature: usize,
        n_classes: usize,
        min_cases: usize,
    ) -> Option<FeatureCut> {
        let n = x.nrows();
        let mut distinct: Vec<f64> = (0..n).map(|i| x[[i, feature]]).collect();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();
        let total: Vec<f64> = (0..n_classes)
            .map(|k| (0..n).filter(|&i| y[i] as usize == k).map(|i| w[i]).sum())
            .collect();
        let node_weight: f64 = w.sum();
        let mut best: Option<(f64, f64, f64)> = None;
        for pair in distinct.windows(2) {
            let t = (pair[0] + pair[1]) / 2.0;
            let mut left = vec![0.0; n_classes];
            let mut right = vec![0.0; n_classes];
            for i in 0..n {
                let side = if x[[i, feature]] <= pair[0] {
                    &mut left
                } else {
                    &mut right
                };
                side[y[i] as usize] += w[i];
            }
            let (lw, rw): (f64, f64) = (left.iter().sum(), right.iter().sum());
            if lw < min_cases as f64 || rw < min_cases as f64 {
                continue;
            }
            let gain = information_gain(&total, &left, &right);
            if best.is_none_or(|(_, g, _)| gain > g) {
                best = Some((t, gain, lw));
            }
        }
        let (t, gain, lw) = best?;
        let penalised = gain - ((distinct.len() - 1) as f64).ln() / node_weight;
        Some((t, penalised, penalised / split_info(lw, node_weight - lw)))
    }

    #[test]
    fn test_c50_split_scan_matches_naive_reference() {
        // The running-count scan against the naive per-threshold pass, on noisy data with tied
        // values and unequal weights: same threshold, gain and ratio to 1e-12 relative (the scan
        // forms the right-hand weights as total minus left, so the last bits may differ).
        let rel = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-300);
        for seed in 0..20u64 {
            let mut u = lcg_uniforms(1000 + seed);
            let n = 40 + (seed as usize) * 3;
            let x = Array2::from_shape_fn((n, 2), |_| (u() * 12.0).round() / 2.0);
            let y = Array1::from_shape_fn(n, |i| {
                let noisy = u() < 0.2;
                let k =
                    usize::from(x[[i, 0]] + 0.5 * x[[i, 1]] > 4.5) + usize::from(x[[i, 1]] > 4.0);
                (if noisy { (k + 1) % 3 } else { k }) as f64
            });
            let w = Array1::from_shape_fn(n, |_| 0.2 + 2.0 * u());
            let indices: Vec<usize> = (0..n).collect();
            let class_weights: Vec<f64> = (0..3)
                .map(|k| (0..n).filter(|&i| y[i] as usize == k).map(|i| w[i]).sum())
                .collect();
            for feature in 0..2 {
                let scan = best_feature_cut(
                    &x.view(),
                    &y.view(),
                    &w,
                    &indices,
                    &class_weights,
                    2,
                    feature,
                );
                let naive = naive_feature_cut(&x, &y, &w, feature, 3, 2);
                match (scan, naive) {
                    (Some(a), Some(b)) => {
                        assert_eq!(a.0, b.0, "seed {seed} feature {feature}");
                        assert!(rel(a.1, b.1) && rel(a.2, b.2), "seed {seed}: {a:?} {b:?}");
                    }
                    (a, b) => assert_eq!(a.is_none(), b.is_none(), "seed {seed}: {a:?} {b:?}"),
                }
            }
        }
    }

    #[test]
    fn test_c50_cf_must_be_inside_the_unit_interval() {
        // ML-68. cf = 0 made the old coefficient infinite and turned pruning off silently.
        let x = two_cluster_x(20);
        let y = Array1::from_shape_fn(20, |i| if i < 10 { 0.0 } else { 1.0 });
        for cf in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
            let cfg = C50Config {
                cf,
                min_cases: 1,
                ..Default::default()
            };
            let err = c50(x.view(), y.view(), &cfg).unwrap_err();
            assert!(
                matches!(err, EconError::InvalidSpecification { .. })
                    && err.to_string().contains("0 < cf < 1"),
                "{cf}: {err}"
            );
        }
    }

    #[test]
    fn test_c50_added_errors_match_c45() {
        // C4.5 Release 8's AddErrs (stats.c), evaluated in C for these (cf, N, E).
        let cases: [(f64, [f64; 6]); 3] = [
            (
                0.25,
                [
                    1.3393401692638518,
                    1.4182126240518167,
                    1.7870684052862202,
                    1.3035065579682557,
                    2.9019070285606272,
                    0.26800000000000024,
                ],
            ),
            (
                0.1,
                [
                    2.1749812373250887,
                    2.4540410409088818,
                    3.1440288945007175,
                    2.0691566129356174,
                    5.3533344745619615,
                    0.26800000000000024,
                ],
            ),
            (
                0.5,
                [
                    0.68127342150308801,
                    0.590636710751544,
                    0.5,
                    0.5,
                    0.5,
                    0.26800000000000024,
                ],
            ),
        ];
        let ne = [
            (20.0, 0.0),
            (20.0, 0.5),
            (20.0, 3.0),
            (6.0, 1.0),
            (100.0, 12.0),
            (10.0, 9.6),
        ];
        for (cf, expected) in cases {
            let level = PruneLevel::new(cf).unwrap();
            for (&(n, e), &want) in ne.iter().zip(expected.iter()) {
                let got = added_errors(n, e, level);
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs(),
                    "cf {cf} N {n} E {e}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn test_c50_prunes_when_the_leaf_is_no_worse() {
        // A node whose children's estimates sum to 3.0: as a leaf it estimates 2.7 (pruned), 3.05
        // (pruned, within C4.5's 0.1) or 3.3 (kept).
        let leaf = |id: usize, estimate: f64| {
            Box::new(C50Node {
                id,
                n: 5,
                class_counts: vec![4, 1],
                class_weights: vec![4.0, 1.0],
                predicted_class: 0,
                entropy: 0.5,
                split: None,
                left: None,
                right: None,
                error_estimate: estimate,
            })
        };
        let stats = |estimate: f64| C50NodeStats {
            class_counts: vec![8, 2],
            class_weights: vec![8.0, 2.0],
            predicted_class: 0,
            entropy: 0.5,
            error_estimate: estimate,
        };
        for (estimate, pruned) in [(2.7, true), (3.05, true), (3.3, false)] {
            let node = c50_join(
                0,
                10,
                stats(estimate),
                (0, 1.5, 0.1, 0.2),
                leaf(1, 1.5),
                leaf(2, 1.5),
                false,
            );
            assert_eq!(node.split.is_none(), pruned, "leaf estimate {estimate}");
            let kept = c50_join(
                0,
                10,
                stats(estimate),
                (0, 1.5, 0.1, 0.2),
                leaf(1, 1.5),
                leaf(2, 1.5),
                true,
            );
            assert!(kept.split.is_some(), "no_global_pruning keeps the split");
        }
    }

    #[test]
    fn test_c50_importance_counts_only_kept_splits() {
        // ML-124. Importance was added when a split was found, before its subtree could be pruned.
        let mut u = lcg_uniforms(5);
        let x = Array2::from_shape_fn((120, 3), |_| u());
        let y = Array1::from_shape_fn(120, |i| {
            let label = usize::from(x[[i, 0]] > 0.5);
            (if i % 6 == 0 { 1 - label } else { label }) as f64
        });
        let cfg = C50Config {
            cf: 0.01,
            ..Default::default()
        };
        let r = c50(x.view(), y.view(), &cfg).unwrap();
        // Every split keeps class 0 the majority on both sides, so pruning makes the root a leaf; the
        // root's split counted as importance 1 all the same.
        let x1 = Array2::from_shape_fn((20, 1), |(i, _)| (i + 1) as f64);
        let y1 = Array1::from_shape_fn(20, |i| {
            f64::from(u8::from([3, 8, 13, 18].contains(&(i + 1))))
        });
        let pruned = c50(x1.view(), y1.view(), &C50Config::default()).unwrap();
        assert!(pruned.trees[0].split.is_none());
        assert_eq!(pruned.variable_importance, vec![0.0]);
        for j in 0..3 {
            let used = r.trees.iter().any(|t| uses_feature(t, j));
            assert_eq!(
                r.variable_importance[j] > 0.0,
                used,
                "feature {j}: importance {} but used in the final trees: {used}",
                r.variable_importance[j]
            );
        }
    }

    #[test]
    fn test_c50_rules_print_names() {
        // ML-144. Rules printed feature indices (X[0]) and dense class codes.
        let df = polars::prelude::df! {
            "x1" => [1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 7.0, 7.5, 8.0, 8.5, 9.0, 9.4],
            "x2" => [0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 0.0, 0.9, 0.1],
            "lab" => ["no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
        }
        .unwrap();
        let cfg = C50Config {
            rules: true,
            min_cases: 1,
            ..Default::default()
        };
        let r = run_c50(&crate::Dataset::new(df), "lab", &["x1", "x2"], &cfg).unwrap();
        let text = r.to_string();
        assert!(text.contains("x1 > ") || text.contains("x1 <= "), "{text}");
        assert!(
            text.contains("class=yes") && text.contains("class=no"),
            "{text}"
        );
        assert!(!text.contains("X[0]"), "{text}");
    }

    #[test]
    fn test_c50_first_trial_weights_are_case_counts() {
        // Every case weighs 1 in the first trial, so the case weights per class are the counts, and a
        // tree grown from any constant weights (rescaled to mean 1 per trial) is the same tree.
        let mut u = lcg_uniforms(23);
        let x = Array2::from_shape_fn((80, 2), |_| u());
        let y = Array1::from_shape_fn(80, |i| {
            let label = usize::from(x[[i, 0]] + 0.3 * x[[i, 1]] > 0.6);
            (if i % 9 == 0 { 1 - label } else { label }) as f64
        });
        let cfg = C50Config {
            no_global_pruning: true,
            ..Default::default()
        };
        let r = c50(x.view(), y.view(), &cfg).unwrap();
        let mut stack = vec![&r.trees[0]];
        while let Some(node) = stack.pop() {
            let counts: Vec<f64> = node.class_counts.iter().map(|&c| c as f64).collect();
            assert_eq!(node.class_weights, counts, "node {}", node.id);
            for child in [&node.left, &node.right].into_iter().flatten() {
                stack.push(child.as_ref());
            }
        }
        let mut w = Array1::from_elem(80, 3.7);
        normalise_case_weights(&mut w);
        let indices: Vec<usize> = (0..80).collect();
        let mut node_id = 0;
        let tree = build_c50_tree(
            &x.view(),
            &y.view(),
            &w,
            &indices,
            &cfg,
            2,
            0,
            &mut node_id,
            None,
        )
        .unwrap();
        assert_eq!(split_signature(&tree), split_signature(&r.trees[0]));
    }

    /// The attribute-selection example: x1 has the larger MDL-penalised gain; x2 has a positive
    /// gain that the MDL cost of its 30 distinct values outweighs.
    fn attribute_example() -> (Array2<f64>, Array1<f64>) {
        let x1 = [
            22, 15, 24, 18, 3, 8, 13, 4, 9, 20, 23, 5, 27, 17, 21, 26, 29, 10, 16, 14, 12, 25, 28,
            19, 11, 2, 1, 30, 7, 6,
        ];
        let x2 = [
            9, 18, 16, 21, 6, 25, 8, 20, 17, 7, 2, 24, 27, 19, 13, 22, 29, 5, 4, 23, 14, 30, 10,
            12, 26, 28, 11, 3, 15, 1,
        ];
        let y = [
            1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
            0,
        ];
        let x = Array2::from_shape_fn((30, 2), |(i, j)| [x1[i], x2[i]][j] as f64);
        (x, y.iter().map(|&v| v as f64).collect())
    }

    /// Whether `feature` splits some node of `root`.
    fn uses_feature(root: &C50Node, feature: usize) -> bool {
        split_signature(root).iter().any(|&(f, _)| f == feature)
    }

    #[test]
    fn test_c50_large_class_codes_are_dense() {
        // ML-1. Before the fix this aborts the whole test binary
        // ("memory allocation of 80000000000008 bytes failed").
        let x = two_cluster_x(40);
        let y = Array1::from_shape_fn(40, |i| if i < 20 { 0.0 } else { 1e13 });
        let r = c50(
            x.view(),
            y.view(),
            &C50Config {
                min_cases: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.n_classes, 2);
        assert_eq!(r.class_labels, vec![0, 10_000_000_000_000]);
        assert_eq!(r.accuracy, 1.0);
        assert_eq!(r.predictions[0], 0);
        assert_eq!(r.predictions[39], 10_000_000_000_000);
        assert_eq!(c50_predict(&r, x.view()).unwrap()[39], 10_000_000_000_000);
        assert_eq!(c50_predict_proba(&r, x.view()).unwrap()[0].len(), 2);
    }

    #[test]
    fn test_c50_noncontiguous_codes_have_no_phantom_class() {
        let x = two_cluster_x(20);
        let y = Array1::from_shape_fn(20, |i| if i < 10 { 1.0 } else { 2.0 });
        let cfg = C50Config {
            min_cases: 1,
            rules: true,
            ..Default::default()
        };
        let r = c50(x.view(), y.view(), &cfg).unwrap();
        assert_eq!(r.n_classes, 2); // was 3: an empty class 0
        assert_eq!(r.class_labels, vec![1, 2]);
        assert_eq!((r.predictions[0], r.predictions[19]), (1, 2));
        assert!(
            r.rules
                .as_ref()
                .unwrap()
                .iter()
                .all(|rule| rule.predicted_class == 1 || rule.predicted_class == 2)
        );
    }

    #[test]
    fn test_c50_rejects_invalid_class_codes() {
        let x = two_cluster_x(10);
        let cfg = C50Config {
            min_cases: 1,
            ..Default::default()
        };
        let cases: [(f64, &str); 5] = [
            (f64::NAN, "non-finite"),
            (f64::INFINITY, "non-finite"),
            (-1.0, "non-negative integer class codes"),
            (0.5, "non-negative integer class codes"),
            (1e19, "2^53"),
        ];
        for (bad, needle) in cases {
            let mut y = Array1::from_shape_fn(10, |i| if i < 5 { 0.0 } else { 1.0 });
            y[7] = bad;
            let err = c50(x.view(), y.view(), &cfg).unwrap_err().to_string();
            assert!(err.contains(needle), "{bad}: {err}");
        }
    }

    #[test]
    fn test_c50_class_count_is_bounded() {
        let cfg = C50Config {
            min_cases: 1,
            ..Default::default()
        };
        let x = Array2::from_shape_fn((12, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(12, |i| (i % 7) as f64); // 7 classes on 12 rows
        let err = c50(x.view(), y.view(), &cfg).unwrap_err().to_string();
        assert!(err.contains("at most 6 classes"), "{err}");
        let x = Array2::from_shape_fn((3, 1), |(i, _)| i as f64);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        assert!(
            c50(x.view(), y.view(), &cfg)
                .unwrap_err()
                .to_string()
                .contains("at most 2 classes")
        );
        let x = Array2::from_shape_fn((2004, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(2004, |i| (i % 1001) as f64);
        assert!(
            c50(x.view(), y.view(), &cfg)
                .unwrap_err()
                .to_string()
                .contains("at most 1000 classes")
        );
    }

    #[test]
    fn test_c50_trials_must_be_between_1_and_100() {
        let x = two_cluster_x(20);
        let y = Array1::from_shape_fn(20, |i| if i < 10 { 0.0 } else { 1.0 });
        for trials in [0usize, 101, 10_000_000_000] {
            let cfg = C50Config {
                trials,
                min_cases: 1,
                ..Default::default()
            };
            let err = c50(x.view(), y.view(), &cfg).unwrap_err().to_string();
            assert!(
                err.contains("trials must be between 1 and 100"),
                "{trials}: {err}"
            );
        }
        let cfg = C50Config {
            trials: 100,
            min_cases: 1,
            ..Default::default()
        };
        assert!(c50(x.view(), y.view(), &cfg).is_ok());
    }

    /// x = 0, 1, 2, ...; y alternates 0/1: every best split peels one row off an end.
    /// `blocks` blocks of 12 cases with one x value each and alternating classes. Peeling the
    /// first block is the largest gain at every node, and its gain, about 12 ln 2 / W, beats the
    /// MDL cost ln(blocks - 1) / W, so the tree is a chain of `blocks - 1` splits.
    fn chain_data(blocks: usize) -> (Array2<f64>, Array1<f64>) {
        (
            Array2::from_shape_fn((12 * blocks, 1), |(i, _)| (i / 12) as f64),
            Array1::from_shape_fn(12 * blocks, |i| ((i / 12) % 2) as f64),
        )
    }

    /// Runs `f` on a thread with a 2 MiB stack, the size of tokio's blocking
    /// threads and rayon's workers.
    fn on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap()
    }

    fn c50_depth(root: &C50Node) -> usize {
        // iterative, so the test itself never recurses
        let (mut max, mut stack) = (0, vec![(root, 0usize)]);
        while let Some((node, d)) = stack.pop() {
            max = max.max(d);
            for child in [&node.left, &node.right].into_iter().flatten() {
                stack.push((child.as_ref(), d + 1));
            }
        }
        max
    }

    #[test]
    fn test_c50_depth_is_capped_on_a_chain_input() {
        // User decision 2026-10-21 (8). The data grow a chain of 599 splits: without the depth guard
        // the build recurses past 512 levels (before the guard, deep chains overflowed the stack).
        assert_eq!(MAX_TREE_DEPTH, 512);
        let depth = on_small_stack(|| {
            let (x, y) = chain_data(600);
            let cfg = C50Config {
                min_cases: 1,
                no_global_pruning: true,
                ..Default::default()
            };
            let r = c50(x.view(), y.view(), &cfg).unwrap();
            c50_depth(&r.trees[0])
        });
        assert_eq!(depth, MAX_TREE_DEPTH); // exactly 512: the guard, not the data, stopped the chain
    }

    #[test]
    fn test_c50_rejects_non_finite_features_and_honours_cancellation() {
        let x = two_cluster_x(20);
        let y = Array1::from_shape_fn(20, |i| if i < 10 { 0.0 } else { 1.0 });
        let cfg = C50Config {
            min_cases: 1,
            ..Default::default()
        };
        let mut xb = x.clone();
        xb[[3, 1]] = f64::NAN;
        assert!(
            c50(xb.view(), y.view(), &cfg)
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
        let err = with_budget(cancelled_budget(), || c50(x.view(), y.view(), &cfg)).unwrap_err();
        assert!(matches!(err, EconError::Cancelled(_)), "{err}");
    }

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
    fn test_c50_predict_rejects_labels_that_do_not_match_the_classes() {
        // Before the fix a `class_labels` shorter than `n_classes` made
        // c50_predict report the dense index (1) as if it were a class code.
        let x = two_cluster_x(20);
        let y = Array1::from_shape_fn(20, |i| if i < 10 { 5.0 } else { 7.0 });
        let cfg = C50Config {
            min_cases: 1,
            ..Default::default()
        };
        let mut r = c50(x.view(), y.view(), &cfg).unwrap();
        assert_eq!(c50_predict(&r, x.view()).unwrap()[19], 7);

        r.class_labels.truncate(1);
        let err = c50_predict(&r, x.view()).unwrap_err();
        assert!(
            matches!(err, EconError::InvalidSpecification { .. }),
            "{err}"
        );
        assert!(
            err.to_string().contains("1 class labels for 2 classes"),
            "{err}"
        );

        // No classes at all: an error, not an out-of-bounds index.
        r.class_labels.clear();
        r.n_classes = 0;
        let err = c50_predict(&r, x.view()).unwrap_err();
        assert!(
            matches!(err, EconError::InvalidSpecification { .. }),
            "{err}"
        );
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
        let pure_counts = vec![5.0, 0.0];
        assert!((entropy(&pure_counts, 5.0) - 0.0).abs() < 1e-10);

        // Balanced binary
        let balanced_counts = vec![5.0, 5.0];
        let expected = -2.0 * (0.5 * 0.5f64.ln());
        assert!((entropy(&balanced_counts, 10.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gain_ratio_calculation() {
        let parent = vec![5.0, 5.0];
        let left = vec![5.0, 0.0];
        let right = vec![0.0, 5.0];

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
