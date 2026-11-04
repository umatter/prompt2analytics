//! Property tests for kernel SVM, ROC/AUC and confusion matrices on deterministic data. No R
//! output is compared here. CART is pinned to rpart in `ml::cart::tests` (T3-WP44); the SVM pins
//! to e1071 live in `ml::svm`/`ml::svm_fast` (T3-WP45).
//!
//! Run with: cargo test -p p2a-core --test validation_ml_new_methods

use ndarray::{Array1, Array2};
use p2a_core::{
    KernelSvmConfig,
    SvmKernel,
    confusion_matrix,
    // Kernel SVM
    kernel_svm,
    // ROC/AUC
    roc_auc,
};

/// Generate test data with deterministic values.
/// y_reg = 2*x1 + 0.5*x2 + noise
/// y_class = based on x1 + x2 > 1
fn generate_test_data(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut x = Array2::zeros((n, 3));
    let mut y_reg = Array1::zeros(n);
    let mut y_class = Array1::zeros(n);

    for i in 0..n {
        // Deterministic pseudo-random using LCG
        let seed = ((i * 1103515245 + 12345) % (1 << 31)) as f64 / (1 << 31) as f64;
        let x1 = ((i * 48271) % 1000) as f64 / 1000.0;
        let x2 = ((i * 16807 + 500) % 1000) as f64 / 1000.0;
        let x3 = (seed - 0.5) * 0.5; // noise

        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        x[[i, 2]] = x3;

        // Regression target: y = 2*x1 + 0.5*x2 + small noise
        let noise = (seed - 0.5) * 0.3;
        y_reg[i] = 2.0 * x1 + 0.5 * x2 + noise;

        // Classification target: based on x1 + x2 > 1
        y_class[i] = if x1 + x2 > 1.0 { 1.0 } else { -1.0 };
    }

    (x, y_reg, y_class)
}

/// Training accuracy on 200 deterministic rows whose classes are separated by x1 + x2 = 1 (x3 is
/// noise). `e1071::svm(X, factor(y), cost = 1, gamma = 1/3, scale = FALSE, tolerance = 1e-3)`
/// reaches 0.995 on these exact rows for both kernels (nSV 83 linear, 97 radial; computed for
/// T3-WP48); the bound allows two more misclassifications (197 of 200).
#[test]
fn test_kernel_svm_separable_training_accuracy() {
    let (x, _, y_class) = generate_test_data(200);
    let accuracy = |predictions: &[i32]| {
        predictions
            .iter()
            .zip(y_class.iter())
            .filter(|(p, l)| (**p as f64 - **l).abs() < 0.5)
            .count() as f64
            / y_class.len() as f64
    };

    let mut config_rbf = KernelSvmConfig::default();
    config_rbf.kernel = SvmKernel::Rbf;
    config_rbf.c = 1.0;
    config_rbf.gamma = Some(1.0 / 3.0);
    config_rbf.max_iter = 1000;
    config_rbf.tolerance = 1e-3;
    let result_rbf =
        kernel_svm(x.view(), y_class.view(), &config_rbf, None).expect("RBF SVM should succeed");
    let acc_rbf = accuracy(&result_rbf.predictions);
    assert!(acc_rbf >= 0.985, "RBF SVM training accuracy {acc_rbf}");

    let mut config_linear = KernelSvmConfig::default();
    config_linear.kernel = SvmKernel::Linear;
    config_linear.c = 1.0;
    config_linear.max_iter = 1000;
    config_linear.tolerance = 1e-3;
    let result_linear = kernel_svm(x.view(), y_class.view(), &config_linear, None)
        .expect("Linear SVM should succeed");
    let acc_linear = accuracy(&result_linear.predictions);
    assert!(
        acc_linear >= 0.985,
        "linear SVM training accuracy {acc_linear}"
    );
    println!("training accuracy: rbf {acc_rbf}, linear {acc_linear}");
}

/// Negatives 0.2 + 0.01 i, positives 0.6 + 0.008 i (i = 0..49). AUC = Mann-Whitney U / (n1 n0) =
/// 2438/2500 (R: `wilcox.test(pos, neg)$statistic / 2500`; 2437 pairs pos > neg, 2 ties). The
/// first maximal Youden threshold is 0.6 (sensitivity 1, specificity 0.8).
#[test]
fn test_roc_auc_calculation() {
    let n = 100;
    let mut predictions = Vec::with_capacity(n);
    let mut actual = Vec::with_capacity(n);
    for i in 0..50 {
        predictions.push(0.2 + (i as f64) * 0.01);
        actual.push(0.0);
    }
    for i in 0..50 {
        predictions.push(0.6 + (i as f64) * 0.008);
        actual.push(1.0);
    }

    let result = roc_auc(&predictions, &actual, Some(100)).expect("ROC/AUC should succeed");

    assert!((result.auc - 0.9752).abs() < 1e-12, "AUC {}", result.auc);
    assert_eq!(result.optimal_threshold, 0.6);
    assert_eq!(result.optimal_metrics.sensitivity, 1.0);
    assert!(
        (result.optimal_metrics.specificity - 0.8).abs() < 1e-12,
        "specificity {}",
        result.optimal_metrics.specificity
    );
}

/// Test confusion matrix metrics
#[test]
fn test_confusion_matrix_metrics() {
    let predictions = vec![1, 1, 0, 0, 1, 0, 1, 0];
    let actual = vec![1, 0, 0, 1, 1, 0, 1, 1];

    let result = confusion_matrix(&predictions, &actual).expect("Confusion matrix should succeed");

    println!("Confusion Matrix:");
    println!(
        "  TP={}, TN={}, FP={}, FN={}",
        result.tp, result.tn, result.fp, result.fn_count
    );
    println!("  Accuracy: {:.4}", result.accuracy);
    println!("  Precision: {:.4}", result.precision);
    println!("  Recall: {:.4}", result.sensitivity);
    println!("  F1: {:.4}", result.f1_score);

    // Manual verification
    assert_eq!(result.tp, 3);
    assert_eq!(result.tn, 2);
    assert_eq!(result.fp, 1);
    assert_eq!(result.fn_count, 2);
    assert!((result.accuracy - 5.0 / 8.0).abs() < 1e-10);
}
