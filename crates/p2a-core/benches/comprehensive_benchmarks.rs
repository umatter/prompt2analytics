//! Comprehensive benchmarks with distribution statistics and memory tracking
//!
//! Run with: `cargo bench -p p2a-core --bench comprehensive_benchmarks`
//!
//! This generates output similar to R's `bench::mark()` with:
//! - Full distribution statistics (min, p25, median, p75, max, mean, std)
//! - Memory allocation tracking
//! - Iterations per second
//! - Raw timing data for detailed analysis

mod bench_utils;

use bench_utils::{
    BenchConfig, BenchmarkResult, print_header, print_result, run_benchmark, save_results,
};
use p2a_core::regression::CovarianceType;
use p2a_core::{
    Dataset, kmeans, pca, run_arima, run_fixed_effects, run_hdfe, run_logit, run_mstl, run_ols,
    run_probit, run_random_effects,
};
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic regression data
fn generate_regression_data(n: usize, k: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut columns: Vec<Column> = Vec::new();

    for i in 1..=k {
        let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        columns.push(Column::new(format!("x{}", i).into(), x));
    }

    let y: Vec<f64> = (0..n)
        .map(|row| {
            let mut sum = 0.0;
            for col in &columns {
                if let Ok(val) = col.get(row) {
                    if let Ok(v) = val.try_extract::<f64>() {
                        sum += v;
                    }
                }
            }
            sum + rng.gen_range(0.0..0.5)
        })
        .collect();

    columns.insert(0, Column::new("y".into(), y));

    let df = DataFrame::new(columns).expect("Failed to create DataFrame");
    Dataset::new(df)
}

/// Generate synthetic panel data
fn generate_panel_data(n_entities: usize, n_periods: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_entities * n_periods;

    let entity: Vec<i64> = (0..n).map(|i| (i / n_periods) as i64).collect();
    let time: Vec<i64> = (0..n).map(|i| (i % n_periods) as i64).collect();
    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let entity_effect = (entity[i] as f64) * 0.1;
            entity_effect + x1[i] * 0.5 + x2[i] * 0.3 + rng.gen_range(0.0..0.5)
        })
        .collect();

    let df = df! {
        "entity" => entity,
        "time" => time,
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate binary outcome data for logit/probit
fn generate_binary_data(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let linear = -1.0 + 0.5 * x1[i] + 0.3 * x2[i];
            let prob = 1.0 / (1.0 + (-linear).exp());
            if rng.gen_range(0.0..1.0) < prob {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let df = df! {
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate time series data
fn generate_time_series(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let y: Vec<f64> = (0..n)
        .map(|t| {
            let trend = 0.01 * t as f64;
            let seasonal = (t as f64 * std::f64::consts::PI / 6.0).sin() * 2.0;
            let noise = rng.gen_range(0.0..0.5);
            trend + seasonal + noise
        })
        .collect();

    let df = df! {
        "y" => y,
    }
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate clustering data
fn generate_cluster_data(n: usize, k: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = ndarray::Array2::zeros((n, k));

    for i in 0..n {
        let cluster = i % 3;
        let center = match cluster {
            0 => 0.0,
            1 => 3.0,
            _ => 6.0,
        };
        for j in 0..k {
            data[[i, j]] = center + rng.gen_range(-0.5..0.5);
        }
    }

    data
}

fn main() {
    let config = BenchConfig {
        warmup_iterations: 10,
        measurement_iterations: 100,
        capture_raw_times: true,
    };

    let mut results: Vec<BenchmarkResult> = Vec::new();

    println!("\n=== p2a Comprehensive Benchmarks ===\n");
    println!(
        "Configuration: {} warmup, {} measurement iterations\n",
        config.warmup_iterations, config.measurement_iterations
    );

    // ============================================
    // Regression Benchmarks
    // ============================================
    println!("\n--- Regression ---");
    print_header();

    for n in [100, 1000, 10000] {
        let dataset = generate_regression_data(n, 5, 42);
        let x_cols = vec!["x1", "x2", "x3", "x4", "x5"];

        // OLS Standard
        let result = run_benchmark("OLS", "standard", n, &config, || {
            run_ols(&dataset, "y", &x_cols, true, CovarianceType::Standard)
        });
        print_result(&result);
        results.push(result);

        // OLS HC1
        let result = run_benchmark("OLS", "HC1", n, &config, || {
            run_ols(&dataset, "y", &x_cols, true, CovarianceType::HC1)
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Panel Data Benchmarks
    // ============================================
    println!("\n--- Panel Data ---");
    print_header();

    for (n_ent, n_per) in [(10, 10), (50, 20), (100, 50)] {
        let n = n_ent * n_per;
        let dataset = generate_panel_data(n_ent, n_per, 42);

        // Fixed Effects
        let result = run_benchmark("FixedEffects", "within", n, &config, || {
            run_fixed_effects(&dataset, "y", &["x1", "x2"], "entity")
        });
        print_result(&result);
        results.push(result);

        // Random Effects
        let result = run_benchmark("RandomEffects", "GLS", n, &config, || {
            run_random_effects(&dataset, "y", &["x1", "x2"], "entity")
        });
        print_result(&result);
        results.push(result);

        // HDFE
        let result = run_benchmark("HDFE", "2-way", n, &config, || {
            run_hdfe(
                &dataset,
                "y",
                &["x1", "x2"],
                &["entity", "time"],
                None,
                CovarianceType::Standard,
            )
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Discrete Choice Benchmarks
    // ============================================
    println!("\n--- Discrete Choice ---");
    print_header();

    for n in [100, 500, 1000] {
        let dataset = generate_binary_data(n, 42);

        // Logit
        let result = run_benchmark("Logit", "MLE", n, &config, || {
            run_logit(&dataset, "y", &["x1", "x2"])
        });
        print_result(&result);
        results.push(result);

        // Probit
        let result = run_benchmark("Probit", "MLE", n, &config, || {
            run_probit(&dataset, "y", &["x1", "x2"])
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Time Series Benchmarks
    // ============================================
    println!("\n--- Time Series ---");
    print_header();

    for n in [100, 200, 500] {
        let dataset = generate_time_series(n, 42);

        // ARIMA
        let result = run_benchmark("ARIMA", "(1,1,1)", n, &config, || {
            run_arima(&dataset, "y", 1, 1, 1)
        });
        print_result(&result);
        results.push(result);

        // MSTL
        let result = run_benchmark("MSTL", "period=12", n, &config, || {
            run_mstl(&dataset, "y", &[12])
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // ML Benchmarks
    // ============================================
    println!("\n--- Machine Learning ---");
    print_header();

    for n in [100, 1000, 5000] {
        let data = generate_cluster_data(n, 5, 42);

        // K-Means
        let result = run_benchmark("K-Means", "k=3", n, &config, || {
            kmeans(data.view(), 3, Some(100), Some(1e-4), Some(5), Some(42))
        });
        print_result(&result);
        results.push(result);

        // PCA
        let result = run_benchmark("PCA", "k=3", n, &config, || {
            pca(data.view(), Some(3), false)
        });
        print_result(&result);
        results.push(result);
    }

    // ============================================
    // Save Results
    // ============================================
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let results_path = format!("performance/results/rust_comprehensive_{}.json", timestamp);

    // Try to save, but don't fail if directory doesn't exist
    if let Err(e) = save_results(&results, &results_path) {
        eprintln!("\nNote: Could not save results to {}: {}", results_path, e);
        // Try current directory as fallback
        let fallback_path = format!("rust_comprehensive_{}.json", timestamp);
        if let Ok(()) = save_results(&results, &fallback_path) {
            println!("\nResults saved to: {}", fallback_path);
        }
    } else {
        println!("\nResults saved to: {}", results_path);
    }

    // Print summary
    println!("\n=== Summary ===");
    println!("Total benchmarks: {}", results.len());
    println!(
        "Iterations per benchmark: {}",
        config.measurement_iterations
    );

    // Print distribution example
    if let Some(r) = results.first() {
        println!("\nExample distribution ({} {}):", r.method, r.variant);
        println!("  Min:    {:>10.1} µs", r.time_min_us);
        println!("  P25:    {:>10.1} µs", r.time_p25_us);
        println!("  Median: {:>10.1} µs", r.time_median_us);
        println!("  P75:    {:>10.1} µs", r.time_p75_us);
        println!("  Max:    {:>10.1} µs", r.time_max_us);
        println!("  Mean:   {:>10.1} µs", r.time_mean_us);
        println!("  Std:    {:>10.1} µs", r.time_std_us);
    }
}
