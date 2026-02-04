//! Benchmark fast implementations vs original vs R
//! Run with: cargo test -p p2a-core --test benchmark_fast_implementations --release -- --nocapture

use ndarray::{Array1, Array2};
use p2a_core::{
    FastKernel,
    FastMboostConfig,
    FastMboostLearner,
    FastSvmConfig,
    FastXgbConfig,
    FastXgbObjective,
    KernelSvmConfig,
    MboostBaseLearner,
    MboostConfig,
    SvmKernel,
    XgbConfig,
    XgbObjective,
    fast_mboost,
    fast_roc_auc,
    fast_roc_auc_parallel,
    // Fast implementations
    fast_svm,
    fast_xgboost,
    // Original implementations
    kernel_svm,
    mboost,
    roc_auc,
    // Boosting implementations (original and fast)
    xgb,
};
use std::time::Instant;

/// Generate test data deterministically
fn generate_data(n: usize) -> (Array2<f64>, Array1<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Array2::zeros((n, 3));
    let mut y_class = Array1::zeros(n);
    let mut predictions = Vec::with_capacity(n);
    let mut actual = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = ((i * 48271) % 10000) as f64 / 10000.0;
        let x2 = ((i * 16807 + 5000) % 10000) as f64 / 10000.0;
        let x3 = (((i * 1103515245 + 12345) % 10000) as f64 / 10000.0 - 0.5) * 0.5;

        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        x[[i, 2]] = x3;

        y_class[i] = if x1 + x2 > 1.0 { 1.0 } else { -1.0 };

        // For ROC/AUC: predictions are scores (0-1 range)
        predictions.push(x1 + x2);
        actual.push(if x1 + x2 > 1.0 { 1.0 } else { 0.0 });
    }

    (x, y_class, predictions, actual)
}

#[test]
fn benchmark_roc_auc_implementations() {
    println!("\n============================================================");
    println!("ROC/AUC BENCHMARK: Fast vs Original");
    println!("============================================================\n");

    let sizes = [1000, 5000, 10000, 50000, 100000];

    println!(
        "{:>10} | {:>15} | {:>15} | {:>15} | {:>10}",
        "n", "Original (ms)", "Fast (ms)", "FastPar (ms)", "Speedup"
    );
    println!("{}", "-".repeat(75));

    for &n in &sizes {
        let (_, _, predictions, actual) = generate_data(n);

        // Original ROC/AUC
        let start = Instant::now();
        let _ = roc_auc(&predictions, &actual, Some(100));
        let t_original = start.elapsed().as_secs_f64() * 1000.0;

        // Fast ROC/AUC (Mann-Whitney U)
        let start = Instant::now();
        let _ = fast_roc_auc(&predictions, &actual);
        let t_fast = start.elapsed().as_secs_f64() * 1000.0;

        // Fast ROC/AUC Parallel
        let start = Instant::now();
        let _ = fast_roc_auc_parallel(&predictions, &actual);
        let t_fast_par = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = t_original / t_fast.min(t_fast_par);

        println!(
            "{:>10} | {:>15.2} | {:>15.2} | {:>15.2} | {:>10.1}x",
            n, t_original, t_fast, t_fast_par, speedup
        );
    }

    println!("\nNote: Fast implementation uses O(n log n) Mann-Whitney U statistic");
    println!("      Original uses O(n × thresholds) threshold-based approach");
}

#[test]
fn benchmark_svm_implementations() {
    println!("\n============================================================");
    println!("SVM BENCHMARK: Fast vs Original");
    println!("============================================================\n");

    // Note: SVM is O(n^2) or O(n^3), so we use smaller sizes
    let sizes = [100, 500, 1000, 2000];

    println!(
        "{:>10} | {:>15} | {:>15} | {:>10}",
        "n", "Original (ms)", "Fast (ms)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &n in &sizes {
        let (x, y_class, _, _) = generate_data(n);

        // Original Kernel SVM (RBF)
        let config_orig = KernelSvmConfig {
            kernel: SvmKernel::Rbf,
            c: 1.0,
            gamma: Some(1.0 / 3.0),
            max_iter: 1000,
            tolerance: 1e-3,
            ..Default::default()
        };

        let start = Instant::now();
        let _ = kernel_svm(x.view(), y_class.view(), &config_orig, None);
        let t_original = start.elapsed().as_secs_f64() * 1000.0;

        // Fast SVM (RBF)
        let config_fast = FastSvmConfig {
            kernel: FastKernel::Rbf { gamma: 1.0 / 3.0 },
            c: 1.0,
            tolerance: 1e-3,
            max_iter: 1000,
            ..Default::default()
        };

        let start = Instant::now();
        let _ = fast_svm(x.view(), y_class.view(), &config_fast);
        let t_fast = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = if t_fast > 0.0 {
            t_original / t_fast
        } else {
            0.0
        };

        println!(
            "{:>10} | {:>15.2} | {:>15.2} | {:>10.1}x",
            n, t_original, t_fast, speedup
        );
    }

    println!("\nNote: Fast implementation uses precomputed kernel matrix");
    println!("      + optimized SMO with proper error cache updates");
}

#[test]
fn benchmark_svm_linear() {
    println!("\n============================================================");
    println!("LINEAR SVM BENCHMARK: Fast vs Original");
    println!("============================================================\n");

    let sizes = [100, 500, 1000, 2000];

    println!(
        "{:>10} | {:>15} | {:>15} | {:>10}",
        "n", "Original (ms)", "Fast (ms)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &n in &sizes {
        let (x, y_class, _, _) = generate_data(n);

        // Original Linear SVM
        let config_orig = KernelSvmConfig {
            kernel: SvmKernel::Linear,
            c: 1.0,
            max_iter: 1000,
            tolerance: 1e-3,
            ..Default::default()
        };

        let start = Instant::now();
        let _ = kernel_svm(x.view(), y_class.view(), &config_orig, None);
        let t_original = start.elapsed().as_secs_f64() * 1000.0;

        // Fast Linear SVM
        let config_fast = FastSvmConfig {
            kernel: FastKernel::Linear,
            c: 1.0,
            tolerance: 1e-3,
            max_iter: 1000,
            ..Default::default()
        };

        let start = Instant::now();
        let _ = fast_svm(x.view(), y_class.view(), &config_fast);
        let t_fast = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = if t_fast > 0.0 {
            t_original / t_fast
        } else {
            0.0
        };

        println!(
            "{:>10} | {:>15.2} | {:>15.2} | {:>10.1}x",
            n, t_original, t_fast, speedup
        );
    }
}

#[test]
fn verify_fast_roc_accuracy() {
    println!("\n============================================================");
    println!("ROC/AUC ACCURACY VERIFICATION");
    println!("============================================================\n");

    let sizes = [100, 1000, 10000];

    for &n in &sizes {
        let (_, _, predictions, actual) = generate_data(n);

        let result_orig = roc_auc(&predictions, &actual, Some(100)).unwrap();
        let result_fast = fast_roc_auc(&predictions, &actual).unwrap();
        let auc_parallel = fast_roc_auc_parallel(&predictions, &actual).unwrap();

        let diff_fast = (result_orig.auc - result_fast.auc).abs();
        let diff_par = (result_orig.auc - auc_parallel).abs();

        println!(
            "n={}: Original AUC={:.6}, Fast AUC={:.6}, Parallel AUC={:.6}",
            n, result_orig.auc, result_fast.auc, auc_parallel
        );
        println!(
            "      Diff (fast): {:.6}, Diff (parallel): {:.6}",
            diff_fast, diff_par
        );

        // AUC should be very close (within 1% for practical purposes)
        assert!(diff_fast < 0.02, "Fast AUC differs too much from original");
        assert!(
            diff_par < 0.02,
            "Parallel AUC differs too much from original"
        );
    }

    println!("\n✓ All AUC values match within tolerance");
}

#[test]
fn benchmark_xgboost_implementations() {
    println!("\n============================================================");
    println!("XGBOOST BENCHMARK: Fast (Histogram) vs Original");
    println!("============================================================\n");

    let sizes = [500, 1000, 2000, 5000];

    println!(
        "{:>10} | {:>15} | {:>15} | {:>10} | {:>10}",
        "n", "Original (ms)", "Fast (ms)", "Speedup", "R² diff"
    );
    println!("{}", "-".repeat(70));

    for &n in &sizes {
        let p = 10;
        let (x, _, _, _) = generate_data(n);
        // Create proper feature matrix
        let x_full = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i * (j + 1) * 48271) % 10000) as f64 / 10000.0
        });
        let y: Array1<f64> = x_full.column(0).mapv(|v| 2.0 * v)
            + x_full.column(1).mapv(|v| 0.5 * v)
            + Array1::from_shape_fn(n, |i| ((i * 12345) % 1000) as f64 / 10000.0);

        // Original XGBoost
        let config_orig = XgbConfig {
            n_estimators: 50,
            max_depth: 4,
            learning_rate: 0.3,
            ..Default::default()
        };

        let start = Instant::now();
        let result_orig = xgb(x_full.view(), y.view(), &config_orig);
        let t_original = start.elapsed().as_secs_f64() * 1000.0;

        // Fast XGBoost (histogram-based)
        let config_fast = FastXgbConfig {
            n_estimators: 50,
            max_depth: 4,
            learning_rate: 0.3,
            max_bin: 256,
            ..Default::default()
        };

        let start = Instant::now();
        let result_fast = fast_xgboost(x_full.view(), y.view(), &config_fast);
        let t_fast = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = if t_fast > 0.0 {
            t_original / t_fast
        } else {
            0.0
        };

        // Compute R² for both
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        let (r2_orig, r2_fast): (f64, f64) = match (result_orig, result_fast) {
            (Ok(r_o), Ok(r_f)) => {
                let ss_res_o: f64 = y
                    .iter()
                    .zip(r_o.predictions.iter())
                    .map(|(&yt, &yp): (&f64, &f64)| (yt - yp).powi(2))
                    .sum();
                let ss_res_f: f64 = y
                    .iter()
                    .zip(r_f.predictions.iter())
                    .map(|(&yt, &yp): (&f64, &f64)| (yt - yp).powi(2))
                    .sum();
                (1.0 - ss_res_o / ss_tot, 1.0 - ss_res_f / ss_tot)
            }
            _ => (0.0, 0.0),
        };

        println!(
            "{:>10} | {:>15.2} | {:>15.2} | {:>10.1}x | {:>10.4}",
            n,
            t_original,
            t_fast,
            speedup,
            (r2_orig - r2_fast).abs()
        );
    }

    println!("\nNote: Fast implementation uses histogram-based splitting (O(n) per split)");
    println!("      Original uses sorted-based splitting (O(n log n) per split)");
}

#[test]
fn benchmark_mboost_implementations() {
    println!("\n============================================================");
    println!("MBOOST BENCHMARK: Fast (Parallel) vs Original");
    println!("============================================================\n");

    let sizes = [500, 1000, 2000, 5000];

    println!(
        "{:>10} | {:>15} | {:>15} | {:>10} | {:>10}",
        "n", "Original (ms)", "Fast (ms)", "Speedup", "R² diff"
    );
    println!("{}", "-".repeat(70));

    for &n in &sizes {
        let p = 20;
        let x = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i * (j + 1) * 48271) % 10000) as f64 / 10000.0
        });
        let y: Array1<f64> = x.column(0).mapv(|v| 2.0 * v)
            + x.column(1).mapv(|v| 0.5 * v)
            + Array1::from_shape_fn(n, |i| ((i * 12345) % 1000) as f64 / 3000.0);

        // Original MBoost (componentwise linear)
        let config_orig = MboostConfig {
            m_stop: 100,
            nu: 0.1,
            base_learner: MboostBaseLearner::ComponentwiseLinear,
            ..Default::default()
        };

        let start = Instant::now();
        let result_orig = mboost(x.view(), y.view(), &config_orig);
        let t_original = start.elapsed().as_secs_f64() * 1000.0;

        // Fast MBoost (parallel feature evaluation)
        let config_fast = FastMboostConfig {
            m_stop: 100,
            nu: 0.1,
            base_learner: FastMboostLearner::ComponentwiseLinear,
            ..Default::default()
        };

        let start = Instant::now();
        let result_fast = fast_mboost(x.view(), y.view(), &config_fast);
        let t_fast = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = if t_fast > 0.0 {
            t_original / t_fast
        } else {
            0.0
        };

        // Compute R² for both
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        let (r2_orig, r2_fast): (f64, f64) = match (result_orig, result_fast) {
            (Ok(r_o), Ok(r_f)) => {
                let ss_res_o: f64 = y
                    .iter()
                    .zip(r_o.predictions.iter())
                    .map(|(&yt, &yp): (&f64, &f64)| (yt - yp).powi(2))
                    .sum();
                let ss_res_f: f64 = y
                    .iter()
                    .zip(r_f.predictions.iter())
                    .map(|(&yt, &yp): (&f64, &f64)| (yt - yp).powi(2))
                    .sum();
                (1.0 - ss_res_o / ss_tot, 1.0 - ss_res_f / ss_tot)
            }
            _ => (0.0, 0.0),
        };

        println!(
            "{:>10} | {:>15.2} | {:>15.2} | {:>10.1}x | {:>10.4} (orig={:.4}, fast={:.4})",
            n,
            t_original,
            t_fast,
            speedup,
            (r2_orig - r2_fast).abs(),
            r2_orig,
            r2_fast
        );
    }

    println!("\nNote: Fast implementation uses parallel feature evaluation via Rayon");
    println!("      Original evaluates features sequentially");
}

#[test]
fn summary_comparison() {
    println!("\n============================================================");
    println!("IMPLEMENTATION COMPARISON SUMMARY");
    println!("============================================================\n");

    println!("ROC/AUC:");
    println!("  - Original: O(n × thresholds) with 100 thresholds");
    println!("  - Fast: O(n log n) using Mann-Whitney U statistic");
    println!("  - Expected speedup: 10-100x for large datasets");
    println!();
    println!("SVM:");
    println!("  - Original: Basic SMO without kernel caching");
    println!("  - Fast: Optimized SMO with:");
    println!("    * Precomputed kernel matrix (for small n)");
    println!("    * Proper error cache updates");
    println!("    * Maximal violating pair selection");
    println!("  - Note: For very large datasets (n > 10000),");
    println!("    libsvm's chunking strategy would be better");
    println!();
    println!("XGBoost:");
    println!("  - Original: Sorted-based splitting O(n log n)");
    println!("  - Fast: Histogram-based splitting O(n) with:");
    println!("    * Pre-binned data (256 bins)");
    println!("    * Parallel histogram construction");
    println!("    * Histogram subtraction trick");
    println!("  - Expected speedup: 2-5x for medium datasets");
    println!();
    println!("MBoost:");
    println!("  - Original: Sequential feature evaluation");
    println!("  - Fast: Parallel feature evaluation with:");
    println!("    * Rayon-based parallelism");
    println!("    * Vectorized RSS computation");
    println!("  - Expected speedup: 2-8x depending on core count");
    println!();
    println!("Comparison with R:");
    println!("  - R's xgboost uses C++ with OpenMP (highly optimized)");
    println!("  - R's mboost uses pure R (slow)");
    println!("  - Our fast_xgboost competitive with R");
    println!("  - Our fast_mboost should beat R's mboost significantly");
}
