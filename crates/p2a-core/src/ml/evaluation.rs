//! Model evaluation metrics for classification.
//!
//! Implements ROC curves, AUC, confusion matrix, variable importance,
//! and partial dependence through the engine in [`crate::ml::pdp`].

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use super::pdp::{PdpConfig, PdpResult};
use crate::errors::EconResult;

/// Result of ROC curve and AUC calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocAucResult {
    /// True positive rates (sensitivity) of the rule `score >= threshold` at each threshold
    pub tpr: Vec<f64>,
    /// False positive rates (1 - specificity) at each threshold
    pub fpr: Vec<f64>,
    /// Thresholds in decreasing order: `+inf` (the point (0, 0)), then the distinct scores
    pub thresholds: Vec<f64>,
    /// Area under the ROC curve (Mann-Whitney, ties counted half)
    pub auc: f64,
    /// Optimal threshold (Youden's J; the lowest of tied optima): the observed score starting the rule
    pub optimal_threshold: f64,
    /// Metrics at optimal threshold
    pub optimal_metrics: ClassificationMetrics,
    /// Number of positive samples
    pub n_positive: usize,
    /// Number of negative samples
    pub n_negative: usize,
}

impl std::fmt::Display for RocAucResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ROC/AUC Analysis")?;
        writeln!(f, "================")?;
        writeln!(f, "AUC: {:.4}", self.auc)?;
        writeln!(f)?;
        writeln!(
            f,
            "Samples: {} positive, {} negative",
            self.n_positive, self.n_negative
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Optimal Threshold: {:.4} (Youden's J)",
            self.optimal_threshold
        )?;
        writeln!(
            f,
            "  Sensitivity (TPR): {:.4}",
            self.optimal_metrics.sensitivity
        )?;
        writeln!(
            f,
            "  Specificity (TNR): {:.4}",
            self.optimal_metrics.specificity
        )?;
        writeln!(
            f,
            "  Precision (PPV):   {:.4}",
            self.optimal_metrics.precision
        )?;
        writeln!(
            f,
            "  F1 Score:          {:.4}",
            self.optimal_metrics.f1_score
        )?;
        writeln!(f)?;
        writeln!(f, "ROC curve points: {}", self.thresholds.len())?;
        Ok(())
    }
}

/// Classification metrics at a specific threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// True positives
    pub tp: usize,
    /// True negatives
    pub tn: usize,
    /// False positives
    pub fp: usize,
    /// False negatives
    pub fn_count: usize,
    /// Sensitivity (true positive rate, recall)
    pub sensitivity: f64,
    /// Specificity (true negative rate)
    pub specificity: f64,
    /// Precision (positive predictive value)
    pub precision: f64,
    /// F1 score (harmonic mean of precision and recall)
    pub f1_score: f64,
    /// Accuracy
    pub accuracy: f64,
    /// Youden's J statistic (sensitivity + specificity - 1)
    pub youden_j: f64,
}

/// Confusion matrix result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrixResult {
    /// True positives
    pub tp: usize,
    /// True negatives
    pub tn: usize,
    /// False positives
    pub fp: usize,
    /// False negatives
    pub fn_count: usize,
    /// Sensitivity (recall, true positive rate)
    pub sensitivity: f64,
    /// Specificity (true negative rate)
    pub specificity: f64,
    /// Precision (positive predictive value)
    pub precision: f64,
    /// Negative predictive value
    pub npv: f64,
    /// F1 score
    pub f1_score: f64,
    /// Accuracy
    pub accuracy: f64,
    /// Matthews correlation coefficient
    pub mcc: f64,
    /// Cohen's kappa
    pub kappa: f64,
}

impl std::fmt::Display for ConfusionMatrixResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Confusion Matrix")?;
        writeln!(f, "================")?;
        writeln!(f)?;
        writeln!(f, "                  Predicted")?;
        writeln!(f, "                Neg     Pos")?;
        writeln!(f, "Actual Neg   {:5}   {:5}", self.tn, self.fp)?;
        writeln!(f, "       Pos   {:5}   {:5}", self.fn_count, self.tp)?;
        writeln!(f)?;
        writeln!(f, "Metrics:")?;
        writeln!(f, "  Accuracy:    {:.4}", self.accuracy)?;
        writeln!(f, "  Sensitivity: {:.4}", self.sensitivity)?;
        writeln!(f, "  Specificity: {:.4}", self.specificity)?;
        writeln!(f, "  Precision:   {:.4}", self.precision)?;
        writeln!(f, "  NPV:         {:.4}", self.npv)?;
        writeln!(f, "  F1 Score:    {:.4}", self.f1_score)?;
        writeln!(f, "  MCC:         {:.4}", self.mcc)?;
        writeln!(f, "  Kappa:       {:.4}", self.kappa)?;
        Ok(())
    }
}

/// The checked inputs of a ROC computation: `(score, is_positive)` sorted by score in decreasing order
/// (`f64::total_cmp`), with the class counts. Refuses unequal lengths, an empty input, NaN scores or labels
/// (missing values), infinite scores (as pROC's `roc()`) or labels, and anything but exactly two label values.
/// The positive class is the larger label; labels are matched exactly.
pub(crate) fn roc_inputs(
    predictions: &[f64],
    actual: &[f64],
) -> Result<(Vec<(f64, bool)>, usize, usize), String> {
    let (pos_class, n_pos, n_neg) = roc_classes(predictions, actual)?;
    let mut pairs: Vec<(f64, bool)> = predictions
        .iter()
        .zip(actual)
        .map(|(&p, &a)| (p, a == pos_class))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok((pairs, n_pos, n_neg))
}

/// The checks of [`roc_inputs`] without the sort: the positive label and the class counts.
pub(crate) fn roc_classes(
    predictions: &[f64],
    actual: &[f64],
) -> Result<(f64, usize, usize), String> {
    if predictions.len() != actual.len() {
        return Err("Predictions and actual values must have the same length".to_string());
    }
    if predictions.is_empty() {
        return Err("Need at least one sample".to_string());
    }
    // One pass over each input on the common path; the error path says what is wrong.
    if !(predictions.iter().all(|p| p.is_finite()) && actual.iter().all(|v| v.is_finite())) {
        return Err(non_finite_message(predictions, actual));
    }
    let first = actual[0];
    let mut n_first = 0usize;
    let mut other: Option<f64> = None;
    for &v in actual {
        if v == first {
            n_first += 1;
            continue;
        }
        match other {
            None => other = Some(v),
            Some(o) if o == v => {}
            Some(_) => {
                let mut unique: Vec<f64> = actual.to_vec();
                unique.sort_by(f64::total_cmp);
                unique.dedup();
                return Err(format!(
                    "ROC requires exactly 2 classes, found {}",
                    unique.len()
                ));
            }
        }
    }
    let Some(other) = other else {
        return Err("ROC requires exactly 2 classes, found 1".to_string());
    };
    let pos_class = first.max(other);
    let n_pos = if pos_class == first {
        n_first
    } else {
        actual.len() - n_first
    };
    Ok((pos_class, n_pos, actual.len() - n_pos))
}

/// The refusal for NaN or infinite scores or labels, in this order: NaN scores, NaN labels, infinite labels,
/// infinite scores.
fn non_finite_message(predictions: &[f64], actual: &[f64]) -> String {
    let k = predictions.iter().filter(|p| p.is_nan()).count();
    if k > 0 {
        return format!(
            "predictions contain {k} missing value(s) (NaN); ROC/AUC needs a score for every observation"
        );
    }
    let k = actual.iter().filter(|v| v.is_nan()).count();
    if k > 0 {
        return format!("actual labels contain {k} missing value(s) (NaN)");
    }
    if actual.iter().any(|v| v.is_infinite()) {
        return "actual labels must be finite (two classes)".to_string();
    }
    let k = predictions.iter().filter(|p| p.is_infinite()).count();
    format!(
        "predictions must be finite (pROC refuses infinite scores); found {k} infinite value(s)"
    )
}

/// One pass over scores sorted in decreasing order ([`roc_inputs`]): the ROC curve with the rule
/// `score >= threshold`, the AUC and Youden's optimum.
#[derive(Debug, Clone)]
pub(crate) struct RocSweep {
    /// `+inf` (nothing positive), then each distinct score in decreasing order.
    pub thresholds: Vec<f64>,
    /// Positives with `score >= threshold`.
    pub tp: Vec<usize>,
    /// Negatives with `score >= threshold`.
    pub fp: Vec<usize>,
    /// Area under the curve: the Mann-Whitney statistic with half credit for ties, over `n_pos n_neg`.
    pub auc: f64,
    /// Index of Youden's optimum; the lowest threshold among tied optima.
    pub best: usize,
}

/// Tied groups of a decreasing sort: `(score, positives, negatives)` per distinct score (exact equality, so
/// `-0.0` and `0.0` are one group).
fn tie_groups(sorted_desc: &[(f64, bool)]) -> impl Iterator<Item = (f64, usize, usize)> + '_ {
    let mut i = 0;
    std::iter::from_fn(move || {
        let &(score, _) = sorted_desc.get(i)?;
        let (mut gp, mut gn) = (0usize, 0usize);
        while let Some(&(s, is_pos)) = sorted_desc.get(i) {
            if s != score {
                break;
            }
            if is_pos {
                gp += 1;
            } else {
                gn += 1;
            }
            i += 1;
        }
        Some((score, gp, gn))
    })
}

/// The AUC numerator's contribution of one tie group: `gn (2 tp_above + gp)`, so that the sum over groups is
/// twice the Mann-Whitney count of (positive, negative) pairs ranked correctly, ties counted half.
fn auc_term(gn: usize, tp_above: usize, gp: usize) -> u128 {
    gn as u128 * (2 * tp_above as u128 + gp as u128)
}

fn auc_from(numerator: u128, n_pos: usize, n_neg: usize) -> f64 {
    numerator as f64 / (2.0 * n_pos as f64 * n_neg as f64)
}

/// The ROC sweep of [`RocSweep`] in O(n) over a decreasing sort.
pub(crate) fn roc_sweep(sorted_desc: &[(f64, bool)], n_pos: usize, n_neg: usize) -> RocSweep {
    let mut thresholds = vec![f64::INFINITY];
    let mut tps = vec![0usize];
    let mut fps = vec![0usize];
    let (mut tp, mut fp) = (0usize, 0usize);
    let mut numerator: u128 = 0;
    let mut best = 0usize;
    let mut best_j: i128 = 0;
    for (score, gp, gn) in tie_groups(sorted_desc) {
        numerator += auc_term(gn, tp, gp);
        tp += gp;
        fp += gn;
        thresholds.push(score);
        tps.push(tp);
        fps.push(fp);
        // Youden's J times n_pos n_neg, in integers so equal values tie exactly; `>=` in the decreasing walk
        // keeps the lowest threshold among tied optima.
        let j = tp as i128 * n_neg as i128 - fp as i128 * n_pos as i128;
        if j >= best_j {
            best_j = j;
            best = thresholds.len() - 1;
        }
    }
    RocSweep {
        thresholds,
        tp: tps,
        fp: fps,
        auc: auc_from(numerator, n_pos, n_neg),
        best,
    }
}

/// The AUC of [`roc_sweep`] without the curve.
pub(crate) fn auc_sorted_desc(sorted_desc: &[(f64, bool)], n_pos: usize, n_neg: usize) -> f64 {
    // The numerator is at most 2 n_pos n_neg; in u64 whenever that fits (any n below about 6e9).
    let fits_u64 = (n_pos as u128) * (n_neg as u128) * 2 <= u64::MAX as u128;
    if !fits_u64 {
        let mut tp = 0usize;
        let mut numerator: u128 = 0;
        for (_, gp, gn) in tie_groups(sorted_desc) {
            numerator += auc_term(gn, tp, gp);
            tp += gp;
        }
        return auc_from(numerator, n_pos, n_neg);
    }
    let n = sorted_desc.len();
    let (mut tp, mut numerator) = (0u64, 0u64);
    let mut i = 0;
    while i < n {
        let score = sorted_desc[i].0;
        let mut gp = u64::from(sorted_desc[i].1);
        let mut j = i + 1;
        while j < n && sorted_desc[j].0 == score {
            gp += u64::from(sorted_desc[j].1);
            j += 1;
        }
        let gn = (j - i) as u64 - gp;
        numerator += gn * (2 * tp + gp);
        tp += gp;
        i = j;
    }
    auc_from(u128::from(numerator), n_pos, n_neg)
}

/// Indices of the curve points kept under a cap of `m + 1` points: the first, the last and `best`, the rest
/// evenly spaced by index. `m >= 2`; all points when there are at most `m + 1`.
fn capped_indices(n_points: usize, best: usize, m: usize) -> Vec<usize> {
    if n_points <= m + 1 {
        return (0..n_points).collect();
    }
    let last = n_points - 1;
    let mut idx: Vec<usize> = (0..=m)
        .map(|i| ((i as u128 * last as u128 + m as u128 / 2) / m as u128) as usize)
        .collect();
    if let Err(pos) = idx.binary_search(&best) {
        // best lies strictly between idx[pos - 1] and idx[pos]; replace the nearer interior neighbour
        // (one of the two is interior because m >= 2).
        let (lo, hi) = (pos - 1, pos);
        let lo_ok = lo > 0;
        let hi_ok = hi < m;
        let pick = match (lo_ok, hi_ok) {
            (true, true) => {
                if best - idx[lo] <= idx[hi] - best {
                    lo
                } else {
                    hi
                }
            }
            (true, false) => lo,
            _ => hi,
        };
        idx[pick] = best;
    }
    idx
}

/// Calculate the ROC curve, the AUC and Youden's optimal threshold, as pROC.
///
/// The rule is `score >= threshold`. The curve starts at `+inf` (nothing classified positive, the point (0, 0))
/// and has one point per distinct score in decreasing order, each after its whole tie group is counted
/// (ties are exact equality); k distinct scores give k + 1 points, the (fpr, tpr) pairs of pROC's `roc()`. The
/// AUC is the Mann-Whitney statistic with half credit for ties over `n_pos n_neg` (pROC's `auc()`,
/// `wilcox.test`'s W / (n1 n0)), computed from integer counts. Youden's J is compared in integers; among tied
/// optima the lowest threshold is reported. `optimal_threshold` is the observed score that starts the optimal
/// rule (pROC reports the midpoint below it). One sort, O(n log n).
///
/// # Arguments
/// * `predictions` - Scores, larger meaning more likely positive; any finite scale
/// * `actual` - Labels with exactly two distinct values; the larger is the positive class
/// * `n_thresholds` - Cap on the returned curve: `None` returns every point; `Some(m)` (m >= 2) at most m + 1
///   points, always keeping the `+inf` point, the last point and the optimum, the rest evenly spaced by index.
///   The AUC and the optimum always use every point.
///
/// # Errors
/// Unequal lengths, an empty input, NaN or infinite scores, NaN or infinite labels, anything but two label
/// values, and `n_thresholds` below 2.
pub fn roc_auc(
    predictions: &[f64],
    actual: &[f64],
    n_thresholds: Option<usize>,
) -> Result<RocAucResult, String> {
    if let Some(m) = n_thresholds
        && m < 2
    {
        return Err(format!(
            "n_thresholds must be at least 2 (the curve keeps its first point, its last point and the optimum), got {m}"
        ));
    }
    let (pairs, n_positive, n_negative) = roc_inputs(predictions, actual)?;
    let sweep = roc_sweep(&pairs, n_positive, n_negative);
    let keep = match n_thresholds {
        Some(m) => capped_indices(sweep.thresholds.len(), sweep.best, m),
        None => (0..sweep.thresholds.len()).collect(),
    };
    let tpr = keep
        .iter()
        .map(|&i| sweep.tp[i] as f64 / n_positive as f64)
        .collect();
    let fpr = keep
        .iter()
        .map(|&i| sweep.fp[i] as f64 / n_negative as f64)
        .collect();
    let thresholds = keep.iter().map(|&i| sweep.thresholds[i]).collect();

    let (tp, fp) = (sweep.tp[sweep.best], sweep.fp[sweep.best]);
    let (tn, fn_count) = (n_negative - fp, n_positive - tp);
    let sensitivity = tp as f64 / n_positive as f64;
    let specificity = tn as f64 / n_negative as f64;
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let f1_score = if precision + sensitivity > 0.0 {
        2.0 * precision * sensitivity / (precision + sensitivity)
    } else {
        0.0
    };
    let accuracy = (tp + tn) as f64 / (n_positive + n_negative) as f64;

    Ok(RocAucResult {
        tpr,
        fpr,
        thresholds,
        auc: sweep.auc,
        optimal_threshold: sweep.thresholds[sweep.best],
        optimal_metrics: ClassificationMetrics {
            tp,
            tn,
            fp,
            fn_count,
            sensitivity,
            specificity,
            precision,
            f1_score,
            accuracy,
            youden_j: sensitivity + specificity - 1.0,
        },
        n_positive,
        n_negative,
    })
}

/// Calculate confusion matrix and derived metrics.
///
/// # Arguments
/// * `predictions` - Predicted labels
/// * `actual` - Actual labels
pub fn confusion_matrix(
    predictions: &[i32],
    actual: &[i32],
) -> Result<ConfusionMatrixResult, String> {
    if predictions.len() != actual.len() {
        return Err("Predictions and actual values must have the same length".to_string());
    }

    if predictions.is_empty() {
        return Err("Need at least one sample".to_string());
    }

    // Determine positive class (assume higher value is positive)
    let mut unique_actual: Vec<i32> = actual.to_vec();
    unique_actual.sort();
    unique_actual.dedup();

    if unique_actual.len() != 2 {
        return Err(format!(
            "Confusion matrix requires exactly 2 classes, found {}",
            unique_actual.len()
        ));
    }

    let pos_class = unique_actual[1];

    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (&pred, &actual_val) in predictions.iter().zip(actual.iter()) {
        let is_positive = actual_val == pos_class;
        let pred_positive = pred == pos_class;

        match (is_positive, pred_positive) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (false, true) => fp += 1,
            (true, false) => fn_count += 1,
        }
    }

    let n = (tp + tn + fp + fn_count) as f64;
    let sensitivity = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let specificity = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let npv = if tn + fn_count > 0 {
        tn as f64 / (tn + fn_count) as f64
    } else {
        0.0
    };
    let f1_score = if precision + sensitivity > 0.0 {
        2.0 * precision * sensitivity / (precision + sensitivity)
    } else {
        0.0
    };
    let accuracy = (tp + tn) as f64 / n;

    // Matthews Correlation Coefficient
    let mcc_denom =
        ((tp + fp) as f64 * (tp + fn_count) as f64 * (tn + fp) as f64 * (tn + fn_count) as f64)
            .sqrt();
    let mcc = if mcc_denom > 0.0 {
        (tp as f64 * tn as f64 - fp as f64 * fn_count as f64) / mcc_denom
    } else {
        0.0
    };

    // Cohen's Kappa
    let po = accuracy; // observed agreement
    let pe = ((tp + fp) as f64 / n) * ((tp + fn_count) as f64 / n)
        + ((tn + fn_count) as f64 / n) * ((tn + fp) as f64 / n);
    let kappa = if (1.0 - pe).abs() > 1e-10 {
        (po - pe) / (1.0 - pe)
    } else {
        0.0
    };

    Ok(ConfusionMatrixResult {
        tp,
        tn,
        fp,
        fn_count,
        sensitivity,
        specificity,
        precision,
        npv,
        f1_score,
        accuracy,
        mcc,
        kappa,
    })
}

/// Result of variable importance analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableImportanceResult {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Mean importance score for each feature
    pub importance: Vec<f64>,
    /// Standard deviation of importance scores
    pub importance_std: Vec<f64>,
    /// Rank of each feature (1 = most important)
    pub ranks: Vec<usize>,
    /// A model score printed with the importances, named by `baseline_metric`: the out-of-bag R² of a
    /// random forest, the final training loss of a GBM; `None` when the model has none (CART, a forest
    /// without OOB rows). For permutation importances, the score before permutation.
    pub baseline_score: Option<f64>,
    /// What `baseline_score` is, e.g. "Out-of-bag R²" or "Final training loss".
    #[serde(default)]
    pub baseline_metric: Option<String>,
    /// Number of permutations used; 0 for impurity (MDI) importances
    pub n_permutations: usize,
    /// Model type used
    pub model_type: String,
}

impl std::fmt::Display for VariableImportanceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = if self.n_permutations == 0 {
            "Variable Importance (impurity-based, MDI)"
        } else {
            "Variable Importance (Permutation-based)"
        };
        writeln!(f, "{header}")?;
        writeln!(f, "{:=<1$}", "", header.chars().count())?;
        writeln!(f, "Model: {}", self.model_type)?;
        if let (Some(metric), Some(value)) = (&self.baseline_metric, self.baseline_score) {
            writeln!(f, "{metric}: {value:.4}")?;
        }
        if self.n_permutations > 0 {
            writeln!(f, "Permutations: {}", self.n_permutations)?;
        }
        writeln!(f)?;

        // Sort by importance
        let mut indices: Vec<usize> = (0..self.feature_names.len()).collect();
        indices.sort_by(|&a, &b| desc_nan_last(self.importance[a], self.importance[b]));

        writeln!(
            f,
            "{:<20} {:>12} {:>10} {:>6}",
            "Feature", "Importance", "Std", "Rank"
        )?;
        writeln!(f, "{:-<50}", "")?;

        for &idx in &indices {
            writeln!(
                f,
                "{:<20} {:>12.4} {:>10.4} {:>6}",
                self.feature_names[idx],
                self.importance[idx],
                self.importance_std[idx],
                self.ranks[idx]
            )?;
        }

        Ok(())
    }
}

/// Deprecated name of [`PdpResult`]. The GBM and CART helpers return the engine's result
/// since the PDP/ICE engine was unified: `grid_values`, `pdp_values`, `feature_name`, `model` replace
/// `grid_values[0]`, `pd_values`, `feature_names`, `model_type`.
#[deprecated(
    note = "use `PdpResult`: gbm_partial_dependence and cart_partial_dependence return the PDP engine's result"
)]
pub type PartialDependenceResult = PdpResult;

/// Descending order with NaN last.
fn desc_nan_last(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_nan(), b.is_nan()) {
        (false, false) => b.total_cmp(&a),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (true, true) => Ordering::Equal,
    }
}

/// Ranks from 1, largest value first; ties keep column order, NaN comes last.
fn rank_descending(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| desc_nan_last(values[a], values[b]));
    let mut ranks = vec![0; values.len()];
    for (rank, &idx) in order.iter().enumerate() {
        ranks[idx] = rank + 1;
    }
    ranks
}

/// Extract variable importance from Random Forest result.
///
/// Uses the built-in mean decrease impurity (MDI) importance.
pub fn rf_variable_importance(
    result: &super::RandomForestResult,
    feature_names: Option<&[String]>,
) -> VariableImportanceResult {
    let n_features = result.feature_importances.len();
    let names: Vec<String> = feature_names
        .map(|n| n.to_vec())
        .or_else(|| result.feature_names.clone())
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    let importance = result.feature_importances.clone();

    let ranks = rank_descending(&importance);

    VariableImportanceResult {
        feature_names: names,
        importance,
        importance_std: vec![0.0; n_features], // MDI doesn't have std
        ranks,
        baseline_score: result.oob_score,
        baseline_metric: result.oob_score.map(|_| "Out-of-bag R²".to_string()),
        n_permutations: 0, // MDI-based, not permutation
        model_type: "Random Forest (MDI)".to_string(),
    }
}

/// Extract variable importance from GBM result.
pub fn gbm_variable_importance(
    result: &super::GbmResult,
    feature_names: Option<&[String]>,
) -> VariableImportanceResult {
    let n_features = result.feature_importances.len();
    let names: Vec<String> = feature_names
        .map(|n| n.to_vec())
        .or_else(|| result.feature_names.clone())
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    let importance = result.feature_importances.clone();

    let ranks = rank_descending(&importance);

    VariableImportanceResult {
        feature_names: names,
        importance,
        importance_std: vec![0.0; n_features],
        ranks,
        baseline_score: Some(result.final_train_loss),
        baseline_metric: Some("Final training loss".to_string()),
        n_permutations: 0,
        model_type: "Gradient Boosting (MDI)".to_string(),
    }
}

/// Extract variable importance from CART result.
pub fn cart_variable_importance(
    result: &super::CartResult,
    feature_names: Option<&[String]>,
) -> VariableImportanceResult {
    let n_features = result.variable_importance.len();
    let names: Vec<String> = feature_names
        .map(|n| n.to_vec())
        .or_else(|| result.feature_names.clone())
        .unwrap_or_else(|| (0..n_features).map(|i| format!("Feature_{}", i)).collect());

    let importance = result.variable_importance.clone();

    let ranks = rank_descending(&importance);

    VariableImportanceResult {
        feature_names: names,
        importance,
        importance_std: vec![0.0; n_features],
        ranks,
        baseline_score: None,
        baseline_metric: None,
        n_permutations: 0,
        model_type: "CART (MDI)".to_string(),
    }
}

/// The GBM and CART helpers: the engine on a uniform grid, with the feature and model named in the result.
fn model_partial_dependence<F>(
    data: ArrayView2<'_, f64>,
    predict: F,
    feature_idx: usize,
    feature_name: &str,
    grid_resolution: usize,
    model: &str,
) -> EconResult<PdpResult>
where
    F: Fn(&Array2<f64>) -> EconResult<Vec<f64>>,
{
    let config = PdpConfig {
        use_quantiles: false,
        grid_resolution: Some(grid_resolution),
        ..Default::default()
    };
    let mut result = super::pdp::pd_1d(data, feature_idx, predict, &config)?;
    result.feature_name = Some(feature_name.to_string());
    result.model = Some(model.to_string());
    Ok(result)
}

/// Compute partial dependence for GBM model.
///
/// Uniform grid `seq(min, max, length.out = grid_resolution)` (`pdp::partial(grid.resolution)`), as
/// [`crate::ml::partial_dependence`] with this `grid_resolution` and `use_quantiles = false`; errors as
/// [`crate::ml::pdp::feature_grid`]. The GBM's predictions are probabilities for the Binomial family, as
/// `gbm_predict` returns them.
///
/// # Arguments
/// * `data` - Original feature matrix (n_samples x n_features), with the columns the model was fitted on
/// * `result` - Fitted GBM result
/// * `feature_idx` - Index of feature for partial dependence
/// * `feature_name` - Name of the feature
/// * `grid_resolution` - Number of grid points, at least 1
pub fn gbm_partial_dependence(
    data: ArrayView2<f64>,
    result: &super::GbmResult,
    feature_idx: usize,
    feature_name: &str,
    grid_resolution: usize,
) -> EconResult<PdpResult> {
    model_partial_dependence(
        data,
        super::pdp::gbm_predictor(result, data.ncols())?,
        feature_idx,
        feature_name,
        grid_resolution,
        "GBM",
    )
}

/// Compute partial dependence for CART model. Grid and errors as [`gbm_partial_dependence`].
///
/// # Arguments
/// * `data` - Original feature matrix (n_samples x n_features), with the columns the model was fitted on
/// * `result` - Fitted CART result
/// * `feature_idx` - Index of feature for partial dependence
/// * `feature_name` - Name of the feature
/// * `grid_resolution` - Number of grid points, at least 1
pub fn cart_partial_dependence(
    data: ArrayView2<f64>,
    result: &super::CartResult,
    feature_idx: usize,
    feature_name: &str,
    grid_resolution: usize,
) -> EconResult<PdpResult> {
    model_partial_dependence(
        data,
        super::pdp::cart_predictor(result, data.ncols())?,
        feature_idx,
        feature_name,
        grid_resolution,
        "CART",
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn test_roc_auc_perfect() {
        // Perfect separation
        let predictions = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let actual = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let result = roc_auc(&predictions, &actual, Some(100)).unwrap();

        // AUC should be 1.0 for perfect separation
        assert!((result.auc - 1.0).abs() < 0.01);
        assert_eq!(result.n_positive, 3);
        assert_eq!(result.n_negative, 3);
    }

    #[test]
    fn test_roc_auc_random() {
        // Random predictions (AUC should be ~0.5)
        let predictions = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let actual = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let result = roc_auc(&predictions, &actual, Some(100)).unwrap();

        // With all same predictions, AUC should be 0.5
        assert!((result.auc - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = vec![1, 1, 0, 0, 1, 0];
        let actual = vec![1, 0, 0, 1, 1, 0];

        let result = confusion_matrix(&predictions, &actual).unwrap();

        assert_eq!(result.tp, 2); // predicted 1, actual 1
        assert_eq!(result.tn, 2); // predicted 0, actual 0
        assert_eq!(result.fp, 1); // predicted 1, actual 0
        assert_eq!(result.fn_count, 1); // predicted 0, actual 1

        assert!((result.accuracy - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_perfect() {
        let predictions = vec![1, 1, 0, 0];
        let actual = vec![1, 1, 0, 0];

        let result = confusion_matrix(&predictions, &actual).unwrap();

        assert_eq!(result.tp, 2);
        assert_eq!(result.tn, 2);
        assert_eq!(result.fp, 0);
        assert_eq!(result.fn_count, 0);
        assert!((result.accuracy - 1.0).abs() < 1e-10);
        assert!((result.mcc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rf_variable_importance() {
        use super::super::random_forest;
        use ndarray::{Array1, Array2};

        // Create simple dataset where feature 0 is most important
        let data = Array2::from_shape_vec(
            (100, 3),
            (0..300)
                .map(|i| (i % 100) as f64 + (i / 100) as f64 * 0.1)
                .collect(),
        )
        .unwrap();

        // Target is strongly correlated with feature 0
        let target: Array1<f64> = data.column(0).to_owned()
            + data.column(1).mapv(|x| x * 0.1)
            + Array1::from_iter((0..100).map(|i| (i as f64 * 0.01).sin() * 0.1));

        let rf_result = random_forest(
            data.view(),
            target.view(),
            Some(10),
            Some(5),
            Some(2),
            Some("sqrt"),
            Some(42),
            Some(vec!["x0".to_string(), "x1".to_string(), "x2".to_string()]),
        )
        .unwrap();

        let importance = rf_variable_importance(&rf_result, None);

        assert_eq!(importance.feature_names.len(), 3);
        assert_eq!(importance.importance.len(), 3);
        assert_eq!(importance.ranks.len(), 3);
        assert_eq!(importance.model_type, "Random Forest (MDI)");

        // Most important feature should have rank 1
        let best_idx = importance
            .importance
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        assert_eq!(importance.ranks[best_idx], 1);
    }

    #[test]
    fn test_gbm_partial_dependence() {
        use super::super::{GbmConfig, GbmFamily, gbm};
        use ndarray::{Array1, Array2};

        // Create dataset with linear relationship to x0
        let n = 100;
        let data = Array2::from_shape_fn((n, 3), |(i, j)| {
            match j {
                0 => i as f64 / n as f64,        // x0: 0 to 1
                1 => (i % 10) as f64 / 10.0,     // x1: noise
                _ => (i * 7 % 13) as f64 / 13.0, // x2: noise
            }
        });

        // Target = 2 * x0 + noise
        let target: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| 2.0 * (i as f64 / n as f64) + (i as f64 * 0.01).sin() * 0.1),
        );

        let config = GbmConfig {
            n_trees: 50,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 5,
            family: GbmFamily::Gaussian,
            seed: Some(42),
            ..Default::default()
        };

        let result = gbm(data.view(), target.view(), &config).unwrap();

        let pd = gbm_partial_dependence(
            data.view(),
            &result,
            0, // x0
            "x0",
            10, // 10 grid points
        )
        .unwrap();

        assert_eq!(pd.feature_name.as_deref(), Some("x0"));
        assert_eq!(pd.grid_values.len(), 10);
        assert_eq!(pd.pdp_values.len(), 10);
        assert_eq!(pd.model.as_deref(), Some("GBM"));

        // PD should be increasing (since target = 2*x0 + noise)
        let first = pd.pdp_values[0];
        let last = pd.pdp_values[pd.pdp_values.len() - 1];
        assert!(last > first, "PD should increase with x0");
    }

    /// The GBM/CART helpers use the same uniform grid rule as every other entry point (pdp's seq, exact) and the same
    /// averaging (identical to try_partial_dependence on the model's predictions).
    #[test]
    fn test_validate_pdp_lm_model_helpers_uniform_grid() {
        use super::super::pdp::{
            test_fixtures::{column, matrix},
            try_partial_dependence,
        };
        use super::super::{CartConfig, GbmConfig, cart, cart_predict, gbm, gbm_predict};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("pdp_lm");
        let x = matrix(&ds, &["x1", "x2", "x3"]);
        let y = column(&ds, "y");
        let gc = GbmConfig {
            n_trees: 30,
            seed: Some(3),
            ..Default::default()
        };
        let gfit = gbm(x.view(), y.view(), &gc).unwrap();
        let cfit = cart(x.view(), y.view(), &CartConfig::default()).unwrap();
        let gpd = gbm_partial_dependence(x.view(), &gfit, 0, "x1", 25).unwrap();
        let cpd = cart_partial_dependence(x.view(), &cfit, 0, "x1", 25).unwrap();
        Check::new(&case)
            .vector("pd_uniform_grid", &gpd.grid_values, Tol::exact())
            .finish();
        Check::new(&case)
            .vector("pd_uniform_grid", &cpd.grid_values, Tol::exact())
            .finish();
        let uniform = PdpConfig {
            use_quantiles: false,
            grid_resolution: Some(25),
            ..Default::default()
        };
        let gref =
            try_partial_dependence(&x, 0, |m| gbm_predict(&gfit, m.view()), &uniform).unwrap();
        let cref =
            try_partial_dependence(&x, 0, |m| cart_predict(&cfit, m.view()), &uniform).unwrap();
        assert_eq!(gpd.pdp_values, gref.pdp_values);
        assert_eq!(cpd.pdp_values, cref.pdp_values);
        assert_eq!(
            (gpd.model.as_deref(), cpd.model.as_deref()),
            (Some("GBM"), Some("CART"))
        );
        assert_eq!(gpd.feature_name.as_deref(), Some("x1"));
    }

    /// grid_resolution 0 is an error (was an empty result); a narrower matrix is an error, not an out-of-bounds
    /// index in the trees' traversal.
    #[test]
    fn test_model_pd_rejects_zero_grid_and_column_mismatch() {
        use super::super::{CartConfig, GbmConfig, cart, gbm};
        let (x, y) = super::super::pdp::test_fixtures::two_feature_data();
        let gc = GbmConfig {
            n_trees: 5,
            seed: Some(7),
            ..Default::default()
        };
        let gfit = gbm(x.view(), y.view(), &gc).unwrap();
        let cfit = cart(x.view(), y.view(), &CartConfig::default()).unwrap();
        let e = gbm_partial_dependence(x.view(), &gfit, 0, "x0", 0)
            .unwrap_err()
            .to_string();
        assert!(e.contains("grid_resolution must be at least 1"), "{e}");
        let e = cart_partial_dependence(x.view(), &cfit, 0, "x0", 0)
            .unwrap_err()
            .to_string();
        assert!(e.contains("grid_resolution must be at least 1"), "{e}");
        let narrow = x.slice(ndarray::s![.., 0..1]);
        let e = gbm_partial_dependence(narrow, &gfit, 0, "x0", 5)
            .unwrap_err()
            .to_string();
        assert!(e.contains("was fitted on 2 features"), "{e}");
        let e = cart_partial_dependence(narrow, &cfit, 0, "x0", 5)
            .unwrap_err()
            .to_string();
        assert!(e.contains("was fitted on 2 features"), "{e}");
    }

    /// One partial-dependence result type: the alias, and the helpers' signatures.
    #[test]
    fn test_partial_dependence_result_is_pdp_result() {
        #[allow(deprecated)]
        fn alias_is_pdp_result(r: PdpResult) -> PartialDependenceResult {
            r
        }
        let _ = alias_is_pdp_result;
        let _: fn(
            ArrayView2<f64>,
            &super::super::GbmResult,
            usize,
            &str,
            usize,
        ) -> EconResult<PdpResult> = gbm_partial_dependence;
        let _: fn(
            ArrayView2<f64>,
            &super::super::CartResult,
            usize,
            &str,
            usize,
        ) -> EconResult<PdpResult> = cart_partial_dependence;
    }

    /// NaN scores or labels are refused (they were sorted as equal to everything and counted as negative
    /// predictions).
    #[test]
    fn test_roc_auc_refuses_nan() {
        let e = roc_auc(&[0.1, f64::NAN, 0.8, 0.4], &[0.0, 0.0, 1.0, 1.0], Some(10)).unwrap_err();
        assert!(e.contains("NaN"), "{e}");
        let e = roc_auc(&[0.1, 0.3, 0.8, 0.4], &[0.0, f64::NAN, 1.0, 1.0], Some(10)).unwrap_err();
        assert!(e.contains("labels"), "{e}");
        let r = roc_auc(&[0.1, 0.3, 0.8, 0.4], &[0.0, 0.0, 1.0, 1.0], Some(10)).unwrap();
        assert!((r.auc - 1.0).abs() < 1e-12);
    }

    /// The four score vectors of `ml_roc_small` (set 1-4: ml91, ml91u, ml92, ml93) from the case's dataset.
    pub(crate) fn roc_small_sets(
        ds: &crate::data::Dataset,
    ) -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
        use crate::data::columns::{ColumnRequest, NaPolicy};
        let c = ColumnRequest::new(ds.df())
            .numerics(&["set", "pred", "label"])
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        let (set, pred, label) = (c.numeric("set"), c.numeric("pred"), c.numeric("label"));
        ["ml91", "ml91u", "ml92", "ml93"]
            .iter()
            .enumerate()
            .map(|(k, &name)| {
                let rows: Vec<usize> = (0..set.len())
                    .filter(|&i| set[i] == (k + 1) as f64)
                    .collect();
                (
                    name,
                    rows.iter().map(|&i| pred[i]).collect(),
                    rows.iter().map(|&i| label[i]).collect(),
                )
            })
            .collect()
    }

    /// The `ml_roc_ties` columns: class (3/7, loaded as Int64), tied probabilities, integer scores.
    pub(crate) fn roc_ties_columns(ds: &crate::data::Dataset) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        use crate::data::columns::{ColumnRequest, NaPolicy};
        let c = ColumnRequest::new(ds.df())
            .numerics(&["cls", "s_prob", "s_lin"])
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        (
            c.numeric("cls").to_vec(),
            c.numeric("s_prob").to_vec(),
            c.numeric("s_lin").to_vec(),
        )
    }

    fn check_roc(c: &mut p2a_validation::Check<'_>, key: &str, r: &RocAucResult, n: usize) {
        use p2a_validation::Tol;
        c.scalar(&format!("{key}_auc"), r.auc, Tol::coef(n))
            .vector(&format!("{key}_tpr"), &r.tpr, Tol::coef(n))
            .vector(&format!("{key}_fpr"), &r.fpr, Tol::coef(n))
            .scalar(
                &format!("{key}_best_sensitivity"),
                r.optimal_metrics.sensitivity,
                Tol::coef(n),
            )
            .scalar(
                &format!("{key}_best_specificity"),
                r.optimal_metrics.specificity,
                Tol::coef(n),
            )
            .scalar(
                &format!("{key}_best_threshold"),
                r.optimal_threshold,
                Tol::exact(),
            );
    }

    /// ML-91. pROC on the findings' score vectors. Before: ml91 AUC 0.6667 (the tie at 1.0 lost the first
    /// triangle), curves ascending with the 0.01 grid added.
    #[test]
    fn test_validate_roc_auc_small() {
        let (case, ds) = crate::test_support::load_case("ml_roc_small");
        let mut c = p2a_validation::Check::new(&case);
        for (key, pred, label) in roc_small_sets(&ds) {
            let r = roc_auc(&pred, &label, Some(100)).unwrap();
            check_roc(&mut c, key, &r, pred.len());
        }
        c.finish();
    }

    /// ML-91. 240 rows, classes 3/7; 54 rows tie at 1.0 across both classes. Before: prob AUC about 0.7688.
    #[test]
    fn test_validate_roc_auc_ties() {
        let (case, ds) = crate::test_support::load_case("ml_roc_ties");
        let (cls, s_prob, s_lin) = roc_ties_columns(&ds);
        let mut c = p2a_validation::Check::new(&case);
        check_roc(&mut c, "prob", &roc_auc(&s_prob, &cls, None).unwrap(), 240);
        check_roc(&mut c, "lin", &roc_auc(&s_lin, &cls, None).unwrap(), 240);
        c.finish();
    }

    /// Seeded score/label draws with heavy ties: n in {2, 7, 60, 500}, scores from a 5-value grid or a
    /// rounded normal.
    pub(crate) fn tied_draws() -> Vec<(Vec<f64>, Vec<f64>)> {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::seed_from_u64(20261115);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut out = Vec::new();
        for draw in 0..50 {
            let n = [2, 7, 60, 500][draw % 4];
            loop {
                let label: Vec<f64> = (0..n)
                    .map(|_| if rng.gen_bool(0.4) { 1.0 } else { 0.0 })
                    .collect();
                if label.iter().all(|&l| l == label[0]) {
                    continue;
                }
                let pred: Vec<f64> = label
                    .iter()
                    .map(|&l| {
                        if draw % 2 == 0 {
                            [0.0, 0.25, 0.5, 0.75, 1.0][rng.gen_range(0..5usize)]
                        } else {
                            ((normal.sample(&mut rng) + 0.7 * l) * 10.0).round() / 10.0
                        }
                    })
                    .collect();
                out.push((pred, label));
                break;
            }
        }
        out
    }

    /// The sweep replaces a quadratic threshold loop: its AUC equals the naive pair count, and every curve
    /// point's counts equal a direct count of `score >= threshold`.
    #[test]
    fn roc_auc_equals_the_pair_count_on_tied_scores() {
        for (pred, label) in tied_draws() {
            let r = roc_auc(&pred, &label, None).unwrap();
            let pos: Vec<f64> = (0..pred.len())
                .filter(|&i| label[i] == 1.0)
                .map(|i| pred[i])
                .collect();
            let neg: Vec<f64> = (0..pred.len())
                .filter(|&i| label[i] == 0.0)
                .map(|i| pred[i])
                .collect();
            let mut twice = 0usize;
            for &p in &pos {
                for &q in &neg {
                    twice += if p > q { 2 } else { usize::from(p == q) };
                }
            }
            let naive = twice as f64 / (2.0 * pos.len() as f64 * neg.len() as f64);
            assert!((r.auc - naive).abs() <= 1e-15, "{} vs {naive}", r.auc);
            assert_eq!(r.thresholds[0], f64::INFINITY);
            assert!(r.thresholds.windows(2).all(|w| w[0] > w[1]));
            for (k, &t) in r.thresholds.iter().enumerate() {
                let tp = pos.iter().filter(|&&p| p >= t).count();
                let fp = neg.iter().filter(|&&q| q >= t).count();
                assert_eq!(r.tpr[k], tp as f64 / pos.len() as f64);
                assert_eq!(r.fpr[k], fp as f64 / neg.len() as f64);
            }
            assert_eq!(*r.tpr.last().unwrap(), 1.0);
            assert_eq!(*r.fpr.last().unwrap(), 1.0);
        }
    }

    /// pROC refuses infinite scores; before, they were accepted.
    #[test]
    fn roc_auc_refuses_infinite_scores() {
        let e = roc_auc(&[0.1, f64::INFINITY, 0.3], &[0.0, 1.0, 1.0], None).unwrap_err();
        assert!(e.contains("finite"), "{e}");
        let e = roc_auc(&[0.1, f64::NEG_INFINITY, 0.3], &[0.0, 1.0, 1.0], None).unwrap_err();
        assert!(e.contains("finite"), "{e}");
    }

    /// -0.0 and 0.0 are one tie group.
    #[test]
    fn roc_auc_groups_signed_zeros() {
        let r = roc_auc(&[-0.0, 0.0, 0.5], &[1.0, 0.0, 1.0], None).unwrap();
        assert_eq!(r.thresholds.len(), 3);
        assert_eq!(r.tpr, vec![0.0, 0.5, 1.0]);
        assert_eq!(r.fpr, vec![0.0, 0.0, 1.0]);
        assert!((r.auc - 0.75).abs() < 1e-15);
    }

    /// ml93 has two tied Youden optima (thresholds 0.9 and 0.5); pROC lists 0.5 first, and so does p2a.
    #[test]
    fn roc_auc_reports_the_lowest_of_tied_optima() {
        let pred = [0.5, 0.5, 0.5, 0.5, 0.2, 0.9];
        let label = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = roc_auc(&pred, &label, None).unwrap();
        assert_eq!(r.optimal_threshold, 0.5);
        assert_eq!(
            (r.optimal_metrics.tp, r.optimal_metrics.fp),
            (3, 2),
            "the rule score >= 0.5"
        );
    }

    /// `n_thresholds` caps the returned curve; the AUC and the optimum use every point. Before, `Some(10)`
    /// returned the grid plus every score.
    #[test]
    fn roc_auc_caps_the_returned_points() {
        let n = 1000;
        let pred: Vec<f64> = (0..n)
            .map(|i| ((i * 7919) % n) as f64 / n as f64 + 0.3 * (i % 2) as f64)
            .collect();
        let label: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let full = roc_auc(&pred, &label, None).unwrap();
        assert!(full.thresholds.len() > 500);
        for m in [2, 3, 10, 100] {
            let r = roc_auc(&pred, &label, Some(m)).unwrap();
            assert!(r.thresholds.len() <= m + 1, "{m}: {}", r.thresholds.len());
            assert_eq!(r.thresholds[0], f64::INFINITY);
            assert_eq!(
                *r.thresholds.last().unwrap(),
                *full.thresholds.last().unwrap()
            );
            assert_eq!((*r.tpr.last().unwrap(), *r.fpr.last().unwrap()), (1.0, 1.0));
            assert!(r.thresholds.contains(&full.optimal_threshold));
            assert!(r.thresholds.windows(2).all(|w| w[0] > w[1]));
            assert_eq!(r.auc, full.auc);
            assert_eq!(r.optimal_threshold, full.optimal_threshold);
            assert_eq!(r.optimal_metrics.tp, full.optimal_metrics.tp);
            assert_eq!(r.optimal_metrics.fp, full.optimal_metrics.fp);
            // every returned point is a point of the full curve
            for (k, t) in r.thresholds.iter().enumerate() {
                let j = full.thresholds.iter().position(|u| u == t).unwrap();
                assert_eq!((r.tpr[k], r.fpr[k]), (full.tpr[j], full.fpr[j]));
            }
        }
        let small = roc_auc(&[0.1, 0.4, 0.35, 0.8], &[0.0, 0.0, 1.0, 1.0], Some(100)).unwrap();
        assert_eq!(small.thresholds.len(), 5);
        for m in [0, 1] {
            let e = roc_auc(&pred, &label, Some(m)).unwrap_err();
            assert!(e.contains("n_thresholds"), "{e}");
        }
    }

    /// ML-132: the impurity importances were headed "Permutation-based" with a "Baseline R²" that was a
    /// training loss (GBM) or 0 (CART, a forest without OOB).
    #[test]
    fn importance_display_names_the_method_and_baseline() {
        use super::super::{GbmConfig, gbm, random_forest};
        let (x, y) = super::super::pdp::test_fixtures::two_feature_data();
        let gc = GbmConfig {
            n_trees: 10,
            seed: Some(3),
            ..Default::default()
        };
        let g = gbm_variable_importance(&gbm(x.view(), y.view(), &gc).unwrap(), None).to_string();
        assert!(
            g.contains("impurity") && g.contains("Final training loss:"),
            "{g}"
        );
        assert!(
            !g.contains("Permutation") && !g.contains("R²") && !g.contains("Permutations"),
            "{g}"
        );
        let rf = random_forest(
            x.view(),
            y.view(),
            Some(20),
            None,
            None,
            None,
            Some(4),
            None,
        )
        .unwrap();
        let shown = rf_variable_importance(&rf, None).to_string();
        // this forest reports an out-of-bag score, so the out-of-bag branch is the one shown
        assert!(rf.oob_score.is_some());
        assert!(shown.contains("Out-of-bag R²:"), "{shown}");
        assert!(
            !shown.contains("Baseline") && !shown.contains("Permutation"),
            "{shown}"
        );
        let mut no_oob = rf.clone();
        no_oob.oob_score = None;
        let shown = rf_variable_importance(&no_oob, None).to_string();
        assert!(!shown.contains("R²") && !shown.contains("loss"), "{shown}");
        let fit =
            super::super::cart(x.view(), y.view(), &super::super::CartConfig::default()).unwrap();
        let c = cart_variable_importance(&fit, None);
        assert_eq!(c.baseline_score, None);
        let shown = c.to_string();
        assert!(
            shown.contains("impurity") && !shown.contains("R²") && !shown.contains("Baseline"),
            "{shown}"
        );
        let perm = VariableImportanceResult {
            n_permutations: 5,
            baseline_score: Some(0.8),
            baseline_metric: Some("R²".to_string()),
            ..c
        };
        let shown = perm.to_string();
        assert!(
            shown.contains("Permutation-based")
                && shown.contains("Permutations: 5")
                && shown.contains("R²: 0.8000"),
            "{shown}"
        );
    }

    /// Importance ranks put NaN last; ties keep column order. Before, NaN compared equal: [0.2, NaN, 0.5, 0.1]
    /// ranked [1, 2, 3, 4], and CART importances [NaN, b] ranked NaN first.
    #[test]
    fn test_importance_ranks_put_nan_last() {
        assert_eq!(
            rank_descending(&[0.2, f64::NAN, 0.5, 0.1]),
            vec![2, 4, 1, 3]
        );
        assert_eq!(rank_descending(&[0.3, 0.3, 0.1]), vec![1, 2, 3]);
        let (x, y) = super::super::pdp::test_fixtures::two_feature_data();
        let mut fit =
            super::super::cart(x.view(), y.view(), &super::super::CartConfig::default()).unwrap();
        fit.variable_importance[0] = f64::NAN;
        let r = cart_variable_importance(&fit, None);
        assert_eq!(r.ranks, vec![2, 1]);
        let shown = format!("{r}");
        assert!(shown.find("Feature_1").unwrap() < shown.find("Feature_0").unwrap());
    }
}
