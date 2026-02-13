//! Performance benchmark: Rust implementations of ML methods
//!
//! Run with: cargo run --release --example ml_benchmark -p p2a-core

use ndarray::{Array1, Array2};
use p2a_core::ml::{
    AdaBoostConfig, AdaBoostType, CartConfig, CartMethod, GbmConfig, adaboost, cart, gbm,
};
use p2a_core::regression::{GlmnetConfig, cv_glmnet, glmnet};
use std::time::Instant;

/// Simple LCG random number generator
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Generate a standard normal random number using Box-Muller transform
    fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }
}

fn generate_data(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = SimpleRng::new(seed);

    // Generate X matrix
    let mut x_data = Vec::with_capacity(n * p);
    for _ in 0..(n * p) {
        x_data.push(rng.next_normal(0.0, 1.0));
    }
    let x = Array2::from_shape_vec((n, p), x_data).unwrap();

    // y_reg = 3*x1 - 1.5*x2 + noise
    let y_reg = Array1::from_iter(
        (0..n).map(|i| 3.0 * x[[i, 0]] - 1.5 * x[[i, 1]] + rng.next_normal(0.0, 0.5)),
    );

    // y_class = sign(x1 + x2)
    let y_class = Array1::from_iter((0..n).map(|i| {
        if x[[i, 0]] + x[[i, 1]] > 0.0 {
            1.0
        } else {
            -1.0
        }
    }));

    (x, y_reg, y_class)
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("ML Methods Performance Benchmark: Rust");
    println!("{}", "=".repeat(60));
    println!();

    let sizes = [1000, 5000, 10000, 50000];

    for &n in &sizes {
        println!("\n--- n = {} observations ---", n);

        let p = 10;
        let (x, y_reg, y_class) = generate_data(n, p, 42);

        // 1. GLMNET Ridge
        println!("\nglmnet Ridge (alpha=0, single lambda):");
        let config = GlmnetConfig {
            alpha: 0.0,
            lambda: Some(vec![0.1]),
            standardize: true,
            ..Default::default()
        };
        let start = Instant::now();
        for _ in 0..10 {
            let _ = glmnet(x.view(), y_reg.view(), &config).unwrap();
        }
        let elapsed = start.elapsed();
        println!("  Total (10 runs): {:.3} sec", elapsed.as_secs_f64());
        println!("  Per run:         {:.4} sec", elapsed.as_secs_f64() / 10.0);

        // 2. GLMNET Lasso with CV (skip for large n)
        if n <= 10000 {
            println!("\nglmnet Lasso CV (5-fold):");
            let cv_config = GlmnetConfig {
                alpha: 1.0,
                nlambda: 50,
                standardize: true,
                ..Default::default()
            };
            let start = Instant::now();
            let _ = cv_glmnet(x.view(), y_reg.view(), &cv_config, 5, None).unwrap();
            let elapsed = start.elapsed();
            println!("  Time: {:.3} sec", elapsed.as_secs_f64());
        }

        // 3. GLMNET path (100 lambdas)
        println!("\nglmnet path (100 lambdas):");
        let path_config = GlmnetConfig {
            alpha: 0.5,
            nlambda: 100,
            standardize: true,
            ..Default::default()
        };
        let start = Instant::now();
        for _ in 0..5 {
            let _ = glmnet(x.view(), y_reg.view(), &path_config).unwrap();
        }
        let elapsed = start.elapsed();
        println!("  Total (5 runs): {:.3} sec", elapsed.as_secs_f64());
        println!("  Per run:        {:.4} sec", elapsed.as_secs_f64() / 5.0);

        // 4. CART Regression
        println!("\nCART regression (depth=5):");
        let cart_config = CartConfig {
            method: CartMethod::Anova,
            max_depth: 5,
            min_split: 20,
            min_bucket: 7,
            cp: 0.01,
            ..Default::default()
        };
        let start = Instant::now();
        for _ in 0..10 {
            let _ = cart(x.view(), y_reg.view(), &cart_config).unwrap();
        }
        let elapsed = start.elapsed();
        println!("  Total (10 runs): {:.3} sec", elapsed.as_secs_f64());
        println!("  Per run:         {:.4} sec", elapsed.as_secs_f64() / 10.0);

        // 5. CART Classification
        let y_class_01: Array1<f64> = Array1::from_iter((0..n).map(|i| {
            if x[[i, 0]] + x[[i, 1]] > 0.0 {
                1.0
            } else {
                0.0
            }
        }));
        println!("\nCART classification (depth=5):");
        let cart_class_config = CartConfig {
            method: CartMethod::Gini,
            max_depth: 5,
            min_split: 20,
            min_bucket: 7,
            cp: 0.01,
            ..Default::default()
        };
        let start = Instant::now();
        for _ in 0..10 {
            let _ = cart(x.view(), y_class_01.view(), &cart_class_config).unwrap();
        }
        let elapsed = start.elapsed();
        println!("  Total (10 runs): {:.3} sec", elapsed.as_secs_f64());
        println!("  Per run:         {:.4} sec", elapsed.as_secs_f64() / 10.0);

        // 6. GBM (only for smaller sizes - slow)
        if n <= 10000 {
            println!("\nGBM (50 trees, depth=3):");
            let gbm_config = GbmConfig {
                n_trees: 50,
                learning_rate: 0.1,
                max_depth: 3,
                min_samples_split: 10,
                seed: Some(42),
                ..Default::default()
            };
            let start = Instant::now();
            for _ in 0..3 {
                let _ = gbm(x.view(), y_reg.view(), &gbm_config).unwrap();
            }
            let elapsed = start.elapsed();
            println!("  Total (3 runs): {:.3} sec", elapsed.as_secs_f64());
            println!("  Per run:        {:.4} sec", elapsed.as_secs_f64() / 3.0);
        }

        // 7. AdaBoost (only for smaller sizes)
        if n <= 10000 {
            println!("\nAdaBoost (50 iterations):");
            let ada_config = AdaBoostConfig {
                n_estimators: 50,
                boost_type: AdaBoostType::M1,
                max_depth: 1,
                ..Default::default()
            };
            let start = Instant::now();
            for _ in 0..3 {
                let _ = adaboost(x.view(), y_class.view(), &ada_config).unwrap();
            }
            let elapsed = start.elapsed();
            println!("  Total (3 runs): {:.3} sec", elapsed.as_secs_f64());
            println!("  Per run:        {:.4} sec", elapsed.as_secs_f64() / 3.0);
        }
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("Benchmark complete");
    println!("{}", "=".repeat(60));
}
