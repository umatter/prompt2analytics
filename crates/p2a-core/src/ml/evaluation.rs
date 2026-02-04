//! Model evaluation metrics for classification.
//!
//! Implements ROC curves, AUC, confusion matrix, variable importance,
//! and partial dependence plots.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Result of ROC curve and AUC calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocAucResult {
    /// True positive rates (sensitivity) at each threshold
    pub tpr: Vec<f64>,
    /// False positive rates (1 - specificity) at each threshold
    pub fpr: Vec<f64>,
    /// Thresholds used for ROC curve
    pub thresholds: Vec<f64>,
    /// Area under the ROC curve
    pub auc: f64,
    /// Optimal threshold (Youden's J statistic)
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

/// Calculate ROC curve and AUC.
///
/// # Arguments
/// * `predictions` - Predicted probabilities (0-1 scale)
/// * `actual` - Actual binary labels (0/1 or any two distinct values)
/// * `n_thresholds` - Number of threshold points for ROC curve (default: 100)
pub fn roc_auc(
    predictions: &[f64],
    actual: &[f64],
    n_thresholds: Option<usize>,
) -> Result<RocAucResult, String> {
    if predictions.len() != actual.len() {
        return Err("Predictions and actual values must have the same length".to_string());
    }

    if predictions.is_empty() {
        return Err("Need at least one sample".to_string());
    }

    // Determine positive and negative class values
    let mut unique_values: Vec<f64> = actual.iter().cloned().collect();
    unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_values.dedup();

    if unique_values.len() != 2 {
        return Err(format!(
            "ROC requires exactly 2 classes, found {}",
            unique_values.len()
        ));
    }

    let neg_class = unique_values[0];
    let pos_class = unique_values[1];

    // Convert actual to binary (0/1)
    let y_true: Vec<i32> = actual
        .iter()
        .map(|&v| if (v - pos_class).abs() < 1e-10 { 1 } else { 0 })
        .collect();

    let n_positive = y_true.iter().filter(|&&y| y == 1).count();
    let n_negative = y_true.len() - n_positive;

    if n_positive == 0 || n_negative == 0 {
        return Err("Need at least one positive and one negative sample".to_string());
    }

    // Generate thresholds
    let n_thresh = n_thresholds.unwrap_or(100);
    let mut thresholds: Vec<f64> = (0..=n_thresh).map(|i| i as f64 / n_thresh as f64).collect();

    // Also add unique prediction values as thresholds for more accurate curve
    let mut pred_thresholds: Vec<f64> = predictions.iter().cloned().collect();
    pred_thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.extend(pred_thresholds);
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup();

    let mut tpr = Vec::with_capacity(thresholds.len());
    let mut fpr = Vec::with_capacity(thresholds.len());
    let mut best_j = f64::NEG_INFINITY;
    let mut best_threshold = 0.5;
    let mut best_metrics = None;

    for &threshold in &thresholds {
        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (pred, &actual_label) in predictions.iter().zip(y_true.iter()) {
            let pred_label = if *pred >= threshold { 1 } else { 0 };

            match (actual_label, pred_label) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fp += 1,
                (1, 0) => fn_count += 1,
                _ => {}
            }
        }

        let sensitivity = if n_positive > 0 {
            tp as f64 / n_positive as f64
        } else {
            0.0
        };
        let specificity = if n_negative > 0 {
            tn as f64 / n_negative as f64
        } else {
            0.0
        };
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let f1 = if precision + sensitivity > 0.0 {
            2.0 * precision * sensitivity / (precision + sensitivity)
        } else {
            0.0
        };
        let accuracy = (tp + tn) as f64 / (tp + tn + fp + fn_count) as f64;
        let youden_j = sensitivity + specificity - 1.0;

        tpr.push(sensitivity);
        fpr.push(1.0 - specificity);

        if youden_j > best_j {
            best_j = youden_j;
            best_threshold = threshold;
            best_metrics = Some(ClassificationMetrics {
                tp,
                tn,
                fp,
                fn_count,
                sensitivity,
                specificity,
                precision,
                f1_score: f1,
                accuracy,
                youden_j,
            });
        }
    }

    // Calculate AUC using trapezoidal rule
    // Sort by FPR to ensure proper integration
    let mut points: Vec<(f64, f64)> = fpr.iter().cloned().zip(tpr.iter().cloned()).collect();
    points.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10 && (a.1 - b.1).abs() < 1e-10);

    let mut auc = 0.0;
    for i in 1..points.len() {
        let dx = points[i].0 - points[i - 1].0;
        let avg_y = (points[i].1 + points[i - 1].1) / 2.0;
        auc += dx * avg_y;
    }

    Ok(RocAucResult {
        tpr,
        fpr,
        thresholds,
        auc,
        optimal_threshold: best_threshold,
        optimal_metrics: best_metrics.unwrap_or(ClassificationMetrics {
            tp: 0,
            tn: 0,
            fp: 0,
            fn_count: 0,
            sensitivity: 0.0,
            specificity: 0.0,
            precision: 0.0,
            f1_score: 0.0,
            accuracy: 0.0,
            youden_j: 0.0,
        }),
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
    let mut unique_actual: Vec<i32> = actual.iter().cloned().collect();
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
    /// Baseline model performance (before permutation)
    pub baseline_score: f64,
    /// Number of permutations used
    pub n_permutations: usize,
    /// Model type used
    pub model_type: String,
}

impl std::fmt::Display for VariableImportanceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Variable Importance (Permutation-based)")?;
        writeln!(f, "========================================")?;
        writeln!(f, "Model: {}", self.model_type)?;
        writeln!(f, "Baseline R²: {:.4}", self.baseline_score)?;
        writeln!(f, "Permutations: {}", self.n_permutations)?;
        writeln!(f)?;

        // Sort by importance
        let mut indices: Vec<usize> = (0..self.feature_names.len()).collect();
        indices.sort_by(|&a, &b| {
            self.importance[b]
                .partial_cmp(&self.importance[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

/// Result of partial dependence analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialDependenceResult {
    /// Feature name(s) for PD
    pub feature_names: Vec<String>,
    /// Grid values for each feature
    pub grid_values: Vec<Vec<f64>>,
    /// Partial dependence values (flattened if 2D)
    pub pd_values: Vec<f64>,
    /// Model type used
    pub model_type: String,
    /// Is this a 2D partial dependence?
    pub is_2d: bool,
}

impl std::fmt::Display for PartialDependenceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Partial Dependence")?;
        writeln!(f, "==================")?;
        writeln!(f, "Model: {}", self.model_type)?;
        writeln!(f, "Features: {}", self.feature_names.join(", "))?;
        writeln!(f)?;

        if !self.is_2d {
            writeln!(f, "{:<12} {:>12}", "Value", "PD")?;
            writeln!(f, "{:-<26}", "")?;

            for (val, pd) in self.grid_values[0].iter().zip(self.pd_values.iter()) {
                writeln!(f, "{:<12.4} {:>12.4}", val, pd)?;
            }
        } else {
            writeln!(
                f,
                "2D partial dependence with {} x {} grid points",
                self.grid_values[0].len(),
                self.grid_values[1].len()
            )?;
        }

        Ok(())
    }
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

    // Calculate ranks
    let mut indexed: Vec<(usize, f64)> = importance.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0; n_features];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank + 1;
    }

    VariableImportanceResult {
        feature_names: names,
        importance,
        importance_std: vec![0.0; n_features], // MDI doesn't have std
        ranks,
        baseline_score: result.oob_score.unwrap_or(0.0),
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

    // Calculate ranks
    let mut indexed: Vec<(usize, f64)> = importance.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0; n_features];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank + 1;
    }

    VariableImportanceResult {
        feature_names: names,
        importance,
        importance_std: vec![0.0; n_features],
        ranks,
        baseline_score: result.final_train_loss,
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

    // Calculate ranks
    let mut indexed: Vec<(usize, f64)> = importance.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0; n_features];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank + 1;
    }

    VariableImportanceResult {
        feature_names: names,
        importance,
        importance_std: vec![0.0; n_features],
        ranks,
        baseline_score: 0.0,
        n_permutations: 0,
        model_type: "CART (MDI)".to_string(),
    }
}

/// Compute partial dependence for GBM model.
///
/// # Arguments
/// * `data` - Original feature matrix (n_samples x n_features)
/// * `result` - Fitted GBM result
/// * `feature_idx` - Index of feature for partial dependence
/// * `feature_name` - Name of the feature
/// * `grid_resolution` - Number of grid points
pub fn gbm_partial_dependence(
    data: ArrayView2<f64>,
    result: &super::GbmResult,
    feature_idx: usize,
    feature_name: &str,
    grid_resolution: usize,
) -> Result<PartialDependenceResult, String> {
    use super::gbm_predict;

    let n_samples = data.nrows();
    let n_features = data.ncols();

    if feature_idx >= n_features {
        return Err(format!(
            "Feature index {} out of bounds ({})",
            feature_idx, n_features
        ));
    }

    if result.trees.is_empty() {
        return Err(
            "GBM model has no fitted trees - cannot compute partial dependence".to_string(),
        );
    }

    // Generate grid values for the feature
    let col = data.column(feature_idx);
    let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let step = if grid_resolution > 1 {
        (max_val - min_val) / (grid_resolution - 1) as f64
    } else {
        0.0
    };

    let grid: Vec<f64> = (0..grid_resolution)
        .map(|i| min_val + step * i as f64)
        .collect();

    // Compute partial dependence at each grid point
    let mut pd_values = Vec::with_capacity(grid_resolution);

    for &grid_val in &grid {
        // Create modified data with feature set to grid value
        let mut modified = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                modified[[i, j]] = if j == feature_idx {
                    grid_val
                } else {
                    data[[i, j]]
                };
            }
        }

        // Predict and average
        let predictions = gbm_predict(result, modified.view())
            .map_err(|e| format!("GBM prediction failed: {}", e))?;
        let avg = predictions.iter().sum::<f64>() / n_samples as f64;
        pd_values.push(avg);
    }

    Ok(PartialDependenceResult {
        feature_names: vec![feature_name.to_string()],
        grid_values: vec![grid],
        pd_values,
        model_type: "GBM".to_string(),
        is_2d: false,
    })
}

/// Compute partial dependence for CART model.
///
/// # Arguments
/// * `data` - Original feature matrix (n_samples x n_features)
/// * `result` - Fitted CART result
/// * `feature_idx` - Index of feature for partial dependence
/// * `feature_name` - Name of the feature
/// * `grid_resolution` - Number of grid points
pub fn cart_partial_dependence(
    data: ArrayView2<f64>,
    result: &super::CartResult,
    feature_idx: usize,
    feature_name: &str,
    grid_resolution: usize,
) -> Result<PartialDependenceResult, String> {
    use super::cart_predict;

    let n_samples = data.nrows();
    let n_features = data.ncols();

    if feature_idx >= n_features {
        return Err(format!(
            "Feature index {} out of bounds ({})",
            feature_idx, n_features
        ));
    }

    // Generate grid values for the feature
    let col = data.column(feature_idx);
    let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let step = if grid_resolution > 1 {
        (max_val - min_val) / (grid_resolution - 1) as f64
    } else {
        0.0
    };

    let grid: Vec<f64> = (0..grid_resolution)
        .map(|i| min_val + step * i as f64)
        .collect();

    // Compute partial dependence at each grid point
    let mut pd_values = Vec::with_capacity(grid_resolution);

    for &grid_val in &grid {
        // Create modified data with feature set to grid value
        let mut modified = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                modified[[i, j]] = if j == feature_idx {
                    grid_val
                } else {
                    data[[i, j]]
                };
            }
        }

        // Predict and average
        let predictions = cart_predict(result, modified.view())
            .map_err(|e| format!("CART prediction failed: {}", e))?;
        let avg = predictions.iter().sum::<f64>() / n_samples as f64;
        pd_values.push(avg);
    }

    Ok(PartialDependenceResult {
        feature_names: vec![feature_name.to_string()],
        grid_values: vec![grid],
        pd_values,
        model_type: "CART".to_string(),
        is_2d: false,
    })
}

#[cfg(test)]
mod tests {
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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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

        assert_eq!(pd.feature_names, vec!["x0"]);
        assert_eq!(pd.grid_values.len(), 1);
        assert_eq!(pd.grid_values[0].len(), 10);
        assert_eq!(pd.pd_values.len(), 10);
        assert!(!pd.is_2d);

        // PD should be increasing (since target = 2*x0 + noise)
        let first = pd.pd_values[0];
        let last = pd.pd_values[pd.pd_values.len() - 1];
        assert!(last > first, "PD should increase with x0");
    }
}
