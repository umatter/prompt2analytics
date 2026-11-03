//! Benchmark Rust ML methods for comparison with R
//!
//! The test prints timings and asserts nothing, and an unoptimised build spends tens of minutes
//! in it, so it is `#[ignore]`d and `cargo test` (CI included) reports it as ignored. Run it in a
//! release build:
//!
//! ```text
//! cargo test -p p2a-core --release --no-default-features --features ci \
//!     --test benchmark_ml_rust -- --ignored --nocapture
//! ```

use ndarray::{Array1, Array2};
use p2a_core::{CartConfig, CartMethod, KernelSvmConfig, SvmKernel, cart, kernel_svm, roc_auc};
use std::time::Instant;

/// Generate test data deterministically
fn generate_data(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut x = Array2::zeros((n, 3));
    let mut y_reg = Array1::zeros(n);
    let mut y_class = Array1::zeros(n);

    for i in 0..n {
        let x1 = ((i * 48271) % 10000) as f64 / 10000.0;
        let x2 = ((i * 16807 + 5000) % 10000) as f64 / 10000.0;
        let x3 = (((i * 1103515245 + 12345) % 10000) as f64 / 10000.0 - 0.5) * 0.5;

        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        x[[i, 2]] = x3;

        let noise = (((i * 7919 + 1) % 1000) as f64 / 1000.0 - 0.5) * 0.3;
        y_reg[i] = 2.0 * x1 + 0.5 * x2 + noise;
        y_class[i] = if x1 + x2 > 1.0 { 1.0 } else { -1.0 };
    }

    (x, y_reg, y_class)
}

#[test]
#[ignore = "benchmark; run with --release -- --ignored"]
fn benchmark_rust_ml() {
    println!("\n============================================================");
    println!("RUST ML BENCHMARKS");
    println!("============================================================\n");

    let sizes = [1000, 5000, 10000, 20000];

    for &n in &sizes {
        println!("\n--- n = {} ---", n);

        let (x, y_reg, y_class) = generate_data(n);
        let y_class_01: Array1<f64> = y_class.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        // SVM RBF
        let config_rbf = KernelSvmConfig {
            kernel: SvmKernel::Rbf,
            c: 1.0,
            gamma: Some(1.0 / 3.0),
            max_iter: 1000,
            tolerance: 1e-3,
            ..Default::default()
        };

        let start = Instant::now();
        let _ = kernel_svm(x.view(), y_class.view(), &config_rbf, None);
        let t_svm_rbf = start.elapsed().as_secs_f64();
        println!("SVM RBF:    {:.4} sec", t_svm_rbf);

        // SVM Linear
        let config_linear = KernelSvmConfig {
            kernel: SvmKernel::Linear,
            c: 1.0,
            max_iter: 1000,
            tolerance: 1e-3,
            ..Default::default()
        };

        let start = Instant::now();
        let _ = kernel_svm(x.view(), y_class.view(), &config_linear, None);
        let t_svm_lin = start.elapsed().as_secs_f64();
        println!("SVM Linear: {:.4} sec", t_svm_lin);

        // CART Regression
        let mut config_cart = CartConfig::default();
        config_cart.method = CartMethod::Anova;
        config_cart.max_depth = 5;
        config_cart.min_split = 10;
        config_cart.min_bucket = 5;
        config_cart.cp = 0.01;
        config_cart.xval = 0;
        config_cart.max_surrogate = 0;
        config_cart.use_surrogate = false;
        config_cart.seed = Some(42);

        let start = Instant::now();
        let cart_result = cart(x.view(), y_reg.view(), &config_cart).unwrap();
        let t_cart_reg = start.elapsed().as_secs_f64();
        println!("CART Reg:   {:.4} sec", t_cart_reg);

        // CART Classification
        let mut config_cart_class = CartConfig::default();
        config_cart_class.method = CartMethod::Gini;
        config_cart_class.max_depth = 5;
        config_cart_class.min_split = 10;
        config_cart_class.min_bucket = 5;
        config_cart_class.cp = 0.01;
        config_cart_class.xval = 0;
        config_cart_class.max_surrogate = 0;
        config_cart_class.use_surrogate = false;
        config_cart_class.seed = Some(42);

        let start = Instant::now();
        let cart_class_result = cart(x.view(), y_class_01.view(), &config_cart_class).unwrap();
        let t_cart_class = start.elapsed().as_secs_f64();
        println!("CART Class: {:.4} sec", t_cart_class);

        // ROC/AUC
        let predictions: Vec<f64> = cart_class_result.predictions.clone();
        let actual: Vec<f64> = y_class_01.to_vec();

        let start = Instant::now();
        let _ = roc_auc(&predictions, &actual, Some(100));
        let t_auc = start.elapsed().as_secs_f64();
        println!("ROC/AUC:    {:.4} sec", t_auc);
    }

    println!("\n============================================================");
    println!("BENCHMARK COMPLETE");
    println!("============================================================\n");
}
