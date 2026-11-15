//! Fast model evaluation metrics using optimized algorithms.
//!
//! ROC/AUC in one sort (O(n log n)), on the tie-aware sweep of [`crate::ml::roc_auc`].

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::evaluation::{auc_sorted_desc, roc_classes, roc_inputs, roc_sweep};

/// Fast ROC/AUC result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastRocAucResult {
    /// Area under the ROC curve (Mann-Whitney, ties counted half)
    pub auc: f64,
    /// True positive rates of the rule `score >= threshold` at each threshold
    pub tpr: Vec<f64>,
    /// False positive rates at each threshold
    pub fpr: Vec<f64>,
    /// Thresholds in decreasing order: `+inf` (the point (0, 0)), then the distinct scores
    pub thresholds: Vec<f64>,
    /// Optimal threshold (Youden's J; the lowest of tied optima): the observed score starting the rule
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

/// Compute the ROC curve, the AUC and Youden's optimum in one sort.
///
/// Same results as [`crate::ml::roc_auc`] with `n_thresholds = None` (pROC's convention): the rule
/// `score >= threshold`, a first point at `+inf`, one point per distinct score in decreasing order, each after
/// its whole tie group is counted (ties are exact equality), the AUC as the Mann-Whitney statistic with half
/// credit for ties, and the lowest threshold among tied Youden optima. O(n log n).
///
/// # Errors
/// As `roc_auc`: unequal lengths, an empty input, NaN or infinite scores, NaN or infinite labels, anything but
/// two label values.
pub fn fast_roc_auc(predictions: &[f64], actual: &[f64]) -> Result<FastRocAucResult, String> {
    let (pairs, n_pos, n_neg) = roc_inputs(predictions, actual)?;
    let sweep = roc_sweep(&pairs, n_pos, n_neg);
    let tpr: Vec<f64> = sweep.tp.iter().map(|&t| t as f64 / n_pos as f64).collect();
    let fpr: Vec<f64> = sweep.fp.iter().map(|&f| f as f64 / n_neg as f64).collect();
    Ok(FastRocAucResult {
        auc: sweep.auc,
        optimal_threshold: sweep.thresholds[sweep.best],
        optimal_sensitivity: tpr[sweep.best],
        optimal_specificity: (n_neg - sweep.fp[sweep.best]) as f64 / n_neg as f64,
        tpr,
        fpr,
        thresholds: sweep.thresholds,
        n_positive: n_pos,
        n_negative: n_neg,
    })
}

/// The AUC alone, with a parallel sort, for very large datasets.
///
/// Equal to the AUC of [`fast_roc_auc`] and [`crate::ml::roc_auc`]: tied scores get half credit, so the
/// result does not depend on the input order. Use it when only the AUC is needed.
///
/// # Errors
/// As [`fast_roc_auc`].
pub fn fast_roc_auc_parallel(predictions: &[f64], actual: &[f64]) -> Result<f64, String> {
    let (pos_class, n_pos, n_neg) = roc_classes(predictions, actual)?;
    let mut pairs: Vec<(f64, bool)> = predictions
        .par_iter()
        .zip(actual.par_iter())
        .map(|(&pred, &act)| (pred, act == pos_class))
        .collect();
    pairs.par_sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(auc_sorted_desc(&pairs, n_pos, n_neg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_roc_auc_refuses_nan_predictions() {
        // X-6: a NaN prediction entered a sort with no total order and gave a NaN-dependent AUC.
        let predictions = [0.1, f64::NAN, 0.8, 0.4, 0.65, 0.3];
        let actual = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let e = fast_roc_auc(&predictions, &actual).unwrap_err();
        assert!(e.contains("missing"), "{e}");
        let e = fast_roc_auc_parallel(&predictions, &actual).unwrap_err();
        assert!(e.contains("missing"), "{e}");
        let labels = [0.0, f64::NAN, 1.0, 0.0, 1.0, 0.0];
        let e = fast_roc_auc(&[0.1, 0.2, 0.8, 0.4, 0.65, 0.3], &labels).unwrap_err();
        assert!(e.contains("missing"), "{e}");
        // infinite scores are refused as in roc_auc (pROC)
        let inf = [0.1, f64::INFINITY, 0.8, 0.4, 0.65, 0.3];
        let e = fast_roc_auc(&inf, &actual).unwrap_err();
        assert!(e.contains("finite"), "{e}");
        let e = fast_roc_auc_parallel(&inf, &actual).unwrap_err();
        assert!(e.contains("finite"), "{e}");
    }

    use crate::ml::evaluation::tests::{roc_small_sets, roc_ties_columns, tied_draws};

    fn check_fast(
        c: &mut p2a_validation::Check<'_>,
        key: &str,
        r: &FastRocAucResult,
        par: f64,
        n: usize,
    ) {
        use p2a_validation::Tol;
        c.scalar(&format!("{key}_auc"), r.auc, Tol::coef(n))
            .scalar(&format!("{key}_auc"), par, Tol::coef(n))
            .vector(&format!("{key}_tpr"), &r.tpr, Tol::coef(n))
            .vector(&format!("{key}_fpr"), &r.fpr, Tol::coef(n))
            .scalar(
                &format!("{key}_best_sensitivity"),
                r.optimal_sensitivity,
                Tol::coef(n),
            )
            .scalar(
                &format!("{key}_best_specificity"),
                r.optimal_specificity,
                Tol::coef(n),
            )
            .scalar(
                &format!("{key}_best_threshold"),
                r.optimal_threshold,
                Tol::exact(),
            );
    }

    /// ML-92, ML-93. Before: ml92 optimum 0.3 (with the rates of score > 0.3); ml93 parallel AUC 0.5556.
    #[test]
    fn test_validate_fast_roc_auc_small() {
        let (case, ds) = crate::test_support::load_case("ml_roc_small");
        let mut c = p2a_validation::Check::new(&case);
        for (key, pred, label) in roc_small_sets(&ds) {
            let r = fast_roc_auc(&pred, &label).unwrap();
            let par = fast_roc_auc_parallel(&pred, &label).unwrap();
            check_fast(&mut c, key, &r, par, pred.len());
        }
        c.finish();
    }

    /// ML-92, ML-93 on 240 tied rows, classes 3/7.
    #[test]
    fn test_validate_fast_roc_auc_ties() {
        let (case, ds) = crate::test_support::load_case("ml_roc_ties");
        let (cls, s_prob, s_lin) = roc_ties_columns(&ds);
        let mut c = p2a_validation::Check::new(&case);
        for (key, s) in [("prob", &s_prob), ("lin", &s_lin)] {
            let r = fast_roc_auc(s, &cls).unwrap();
            let par = fast_roc_auc_parallel(s, &cls).unwrap();
            check_fast(&mut c, key, &r, par, 240);
        }
        c.finish();
    }

    /// ML-93: the parallel AUC averaged no ranks over ties, so reordering tied rows changed it (0.5556 and
    /// 1.0 here, 0.7778 both ways now).
    #[test]
    fn fast_roc_auc_parallel_does_not_depend_on_the_order_of_ties() {
        let pred = [0.5, 0.5, 0.5, 0.5, 0.2, 0.9];
        for label in [
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        ] {
            let a = fast_roc_auc_parallel(&pred, &label).unwrap();
            assert!((a - 7.0 / 9.0).abs() < 1e-15, "{a}");
        }
        let (_, ds) = crate::test_support::load_case("ml_roc_ties");
        let (cls, s_prob, _) = roc_ties_columns(&ds);
        let reference = fast_roc_auc_parallel(&s_prob, &cls).unwrap();
        use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
        let mut rng = StdRng::seed_from_u64(7);
        let mut idx: Vec<usize> = (0..cls.len()).collect();
        for _ in 0..20 {
            idx.shuffle(&mut rng);
            let p: Vec<f64> = idx.iter().map(|&i| s_prob[i]).collect();
            let y: Vec<f64> = idx.iter().map(|&i| cls[i]).collect();
            assert_eq!(fast_roc_auc_parallel(&p, &y).unwrap(), reference);
        }
    }

    /// ML-93: three label values were taken as max-vs-rest (`Ok(0.75)`).
    #[test]
    fn fast_roc_auc_parallel_refuses_three_classes() {
        let e = fast_roc_auc_parallel(&[0.1, 0.4, 0.6, 0.9], &[0.0, 1.0, 2.0, 2.0]).unwrap_err();
        assert!(e.contains("exactly 2 classes"), "{e}");
    }

    /// The three functions share one convention: field by field equal on tied draws.
    #[test]
    fn fast_and_exact_agree_on_tied_scores() {
        for (pred, label) in tied_draws() {
            let e = crate::ml::roc_auc(&pred, &label, None).unwrap();
            let f = fast_roc_auc(&pred, &label).unwrap();
            let p = fast_roc_auc_parallel(&pred, &label).unwrap();
            assert_eq!((f.auc, p), (e.auc, e.auc));
            assert_eq!(f.tpr, e.tpr);
            assert_eq!(f.fpr, e.fpr);
            assert_eq!(f.thresholds, e.thresholds);
            assert_eq!(f.optimal_threshold, e.optimal_threshold);
            assert_eq!(f.optimal_sensitivity, e.optimal_metrics.sensitivity);
            assert_eq!(f.optimal_specificity, e.optimal_metrics.specificity);
        }
    }

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
