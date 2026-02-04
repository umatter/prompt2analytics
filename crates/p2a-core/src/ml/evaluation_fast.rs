//! Fast model evaluation metrics using optimized algorithms.
//!
//! ROC/AUC: O(n log n) using Mann-Whitney U statistic instead of O(n × thresholds)

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Fast ROC/AUC result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastRocAucResult {
    /// Area under the ROC curve (computed via Mann-Whitney U)
    pub auc: f64,
    /// True positive rates at each threshold
    pub tpr: Vec<f64>,
    /// False positive rates at each threshold
    pub fpr: Vec<f64>,
    /// Thresholds (sorted prediction values)
    pub thresholds: Vec<f64>,
    /// Optimal threshold (Youden's J)
    pub optimal_threshold: f64,
    /// Sensitivity at optimal threshold
    pub optimal_sensitivity: f64,
    /// Specificity at optimal threshold
    pub optimal_specificity: f64,
    /// Number of positive samples
    pub n_positive: usize,
    /// Number of negative samples
    pub n_negative: usize,
}

/// Compute ROC/AUC using the fast Mann-Whitney U statistic method.
///
/// Complexity: O(n log n) for sorting + O(n) for AUC calculation
/// This is much faster than the threshold-based O(n × t) approach.
///
/// # Algorithm
/// 1. Sort predictions with their labels
/// 2. Compute AUC = (sum of positive ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
/// 3. Generate ROC curve in a single pass through sorted data
pub fn fast_roc_auc(predictions: &[f64], actual: &[f64]) -> Result<FastRocAucResult, String> {
    let n = predictions.len();
    if n != actual.len() {
        return Err("Predictions and actual must have same length".to_string());
    }
    if n == 0 {
        return Err("Need at least one sample".to_string());
    }

    // Determine positive/negative class
    let mut unique: Vec<f64> = actual.iter().cloned().collect();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup();

    if unique.len() != 2 {
        return Err(format!("Need exactly 2 classes, found {}", unique.len()));
    }

    let pos_class = unique[1];

    // Create (prediction, is_positive) pairs and sort by prediction descending
    let mut pairs: Vec<(f64, bool, usize)> = predictions
        .iter()
        .zip(actual.iter())
        .enumerate()
        .map(|(i, (&pred, &act))| (pred, (act - pos_class).abs() < 1e-10, i))
        .collect();

    // Sort by prediction descending (for ROC curve generation)
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = pairs.iter().filter(|(_, is_pos, _)| *is_pos).count();
    let n_neg = n - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return Err("Need at least one positive and one negative sample".to_string());
    }

    // === Fast AUC via Mann-Whitney U statistic ===
    // Sort by prediction ascending for rank calculation
    let mut ranked: Vec<(f64, bool)> = pairs.iter().map(|(p, is_pos, _)| (*p, *is_pos)).collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (handle ties by averaging)
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all elements with same prediction value
        while j < n && (ranked[j].0 - ranked[i].0).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j + 1) as f64 / 2.0 + (j - i - 1) as f64 / 2.0;
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // Sum ranks of positive samples
    let rank_sum_pos: f64 = ranked
        .iter()
        .zip(ranks.iter())
        .filter(|((_, is_pos), _)| *is_pos)
        .map(|(_, rank)| *rank)
        .sum();

    // Mann-Whitney U statistic
    // U = rank_sum_pos - n_pos * (n_pos + 1) / 2
    // AUC = U / (n_pos * n_neg)
    let u = rank_sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0;
    let auc = u / (n_pos as f64 * n_neg as f64);

    // === Generate ROC curve in single pass ===
    // pairs is already sorted by prediction descending
    let mut tpr = Vec::with_capacity(n + 1);
    let mut fpr = Vec::with_capacity(n + 1);
    let mut thresholds = Vec::with_capacity(n + 1);

    // Start at (0, 0) - threshold above all predictions
    tpr.push(0.0);
    fpr.push(0.0);
    thresholds.push(f64::INFINITY);

    let mut tp = 0;
    let mut fp = 0;
    let mut best_j = f64::NEG_INFINITY;
    let mut best_threshold = 0.5;
    let mut best_sens = 0.0;
    let mut best_spec = 0.0;

    let mut prev_pred = f64::INFINITY;

    for (pred, is_pos, _) in &pairs {
        // Only add point when prediction changes (avoid duplicate points)
        if (*pred - prev_pred).abs() > 1e-15 {
            let current_tpr = tp as f64 / n_pos as f64;
            let current_fpr = fp as f64 / n_neg as f64;

            tpr.push(current_tpr);
            fpr.push(current_fpr);
            thresholds.push(*pred);

            // Check Youden's J
            let spec = 1.0 - current_fpr;
            let j = current_tpr + spec - 1.0;
            if j > best_j {
                best_j = j;
                best_threshold = *pred;
                best_sens = current_tpr;
                best_spec = spec;
            }

            prev_pred = *pred;
        }

        if *is_pos {
            tp += 1;
        } else {
            fp += 1;
        }
    }

    // End at (1, 1) - threshold below all predictions
    tpr.push(1.0);
    fpr.push(1.0);
    thresholds.push(f64::NEG_INFINITY);

    Ok(FastRocAucResult {
        auc,
        tpr,
        fpr,
        thresholds,
        optimal_threshold: best_threshold,
        optimal_sensitivity: best_sens,
        optimal_specificity: best_spec,
        n_positive: n_pos,
        n_negative: n_neg,
    })
}

/// Parallel AUC computation for very large datasets.
/// Uses parallel sorting and parallel rank summation.
pub fn fast_roc_auc_parallel(predictions: &[f64], actual: &[f64]) -> Result<f64, String> {
    let n = predictions.len();
    if n != actual.len() {
        return Err("Predictions and actual must have same length".to_string());
    }

    // Determine positive class
    let pos_class = actual.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Create pairs
    let mut pairs: Vec<(f64, bool)> = predictions
        .par_iter()
        .zip(actual.par_iter())
        .map(|(&pred, &act)| (pred, (act - pos_class).abs() < 1e-10))
        .collect();

    // Parallel sort
    pairs.par_sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos: usize = pairs.par_iter().filter(|(_, is_pos)| *is_pos).count();
    let n_neg = n - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return Err("Need at least one positive and one negative".to_string());
    }

    // Compute rank sum of positives (simplified - ignoring ties for speed)
    let rank_sum_pos: f64 = pairs
        .par_iter()
        .enumerate()
        .filter(|(_, (_, is_pos))| *is_pos)
        .map(|(i, _)| (i + 1) as f64)
        .sum();

    let u = rank_sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0;
    let auc = u / (n_pos as f64 * n_neg as f64);

    Ok(auc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_roc_auc_perfect() {
        let predictions = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let actual = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let result = fast_roc_auc(&predictions, &actual).unwrap();

        assert!(
            (result.auc - 1.0).abs() < 0.01,
            "AUC should be ~1.0: {}",
            result.auc
        );
    }

    #[test]
    fn test_fast_roc_auc_random() {
        let predictions = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let actual = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let result = fast_roc_auc(&predictions, &actual).unwrap();

        assert!(
            (result.auc - 0.5).abs() < 0.1,
            "AUC should be ~0.5: {}",
            result.auc
        );
    }

    #[test]
    fn test_fast_roc_auc_large() {
        // Test with larger dataset
        let n = 10000;
        let predictions: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let actual: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();

        let result = fast_roc_auc(&predictions, &actual).unwrap();

        // With this ordering, AUC should be 1.0
        assert!((result.auc - 1.0).abs() < 0.01);
    }
}
