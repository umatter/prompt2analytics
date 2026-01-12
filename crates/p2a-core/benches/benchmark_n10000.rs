//! N=10,000 benchmarks for paper main text
//!
//! Run with: `cargo bench -p p2a-core --bench benchmark_n10000`

mod bench_utils;

use bench_utils::{run_benchmark, BenchConfig, BenchmarkResult, print_header, print_result, save_results};
use p2a_core::{
    Dataset, run_ols, run_fixed_effects,
    run_logit, run_arima, run_mstl,
    kmeans, pca,
};
use p2a_core::regression::CovarianceType;
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const N: usize = 10000;

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
                    if let Some(v) = val.try_extract::<f64>().ok() {
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
    }.expect("Failed to create DataFrame");

    Dataset::new(df)
}

fn generate_binary_data(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let linear = -1.0 + 0.5 * x1[i] + 0.3 * x2[i];
            let prob = 1.0 / (1.0 + (-linear).exp());
            if rng.gen_range(0.0..1.0) < prob { 1.0 } else { 0.0 }
        })
        .collect();

    let df = df! {
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }.expect("Failed to create DataFrame");

    Dataset::new(df)
}

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
    }.expect("Failed to create DataFrame");

    Dataset::new(df)
}

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

    println!("\n=== p2a Benchmarks at N=10,000 ===\n");

    // Regression
    println!("--- Regression ---");
    print_header();

    let reg_dataset = generate_regression_data(N, 5, 42);
    let x_cols = vec!["x1", "x2", "x3", "x4", "x5"];

    let result = run_benchmark("OLS", "standard", N, &config, || {
        run_ols(&reg_dataset, "y", &x_cols, true, CovarianceType::Standard)
    });
    print_result(&result);
    results.push(result);

    let result = run_benchmark("OLS", "HC1", N, &config, || {
        run_ols(&reg_dataset, "y", &x_cols, true, CovarianceType::HC1)
    });
    print_result(&result);
    results.push(result);

    // Panel (100 entities x 100 periods = 10,000)
    println!("\n--- Panel Data ---");
    print_header();

    let panel_dataset = generate_panel_data(100, 100, 42);

    let result = run_benchmark("FixedEffects", "within", N, &config, || {
        run_fixed_effects(&panel_dataset, "y", &["x1", "x2"], "entity")
    });
    print_result(&result);
    results.push(result);

    // Discrete Choice
    println!("\n--- Discrete Choice ---");
    print_header();

    let binary_dataset = generate_binary_data(N, 42);

    let result = run_benchmark("Logit", "MLE", N, &config, || {
        run_logit(&binary_dataset, "y", &["x1", "x2"])
    });
    print_result(&result);
    results.push(result);

    // Time Series
    println!("\n--- Time Series ---");
    print_header();

    let ts_dataset = generate_time_series(N, 42);

    let result = run_benchmark("ARIMA", "(1,1,1)", N, &config, || {
        run_arima(&ts_dataset, "y", 1, 1, 1)
    });
    print_result(&result);
    results.push(result);

    let result = run_benchmark("MSTL", "period=12", N, &config, || {
        run_mstl(&ts_dataset, "y", &[12])
    });
    print_result(&result);
    results.push(result);

    // ML
    println!("\n--- Machine Learning ---");
    print_header();

    let ml_data = generate_cluster_data(N, 5, 42);

    let result = run_benchmark("K-Means", "k=3", N, &config, || {
        kmeans(ml_data.view(), 3, Some(100), Some(1e-4), Some(5), Some(42))
    });
    print_result(&result);
    results.push(result);

    let result = run_benchmark("PCA", "k=3", N, &config, || {
        pca(ml_data.view(), Some(3), false)
    });
    print_result(&result);
    results.push(result);

    // Save Results
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let results_path = format!("performance/results/rust_n10000_{}.json", timestamp);

    if let Err(e) = save_results(&results, &results_path) {
        eprintln!("\nNote: Could not save results to {}: {}", results_path, e);
        let fallback_path = format!("rust_n10000_{}.json", timestamp);
        if let Ok(()) = save_results(&results, &fallback_path) {
            println!("\nResults saved to: {}", fallback_path);
        }
    } else {
        println!("\nResults saved to: {}", results_path);
    }

    println!("\n=== Summary ===");
    println!("Total benchmarks: {} methods at N={}", results.len(), N);
}
