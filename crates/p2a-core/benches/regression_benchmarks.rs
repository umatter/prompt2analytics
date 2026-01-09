//! Regression method benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- regression`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::{Dataset, run_ols, run_diagnostics};
use p2a_core::regression::CovarianceType;
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic regression data with known DGP
fn generate_regression_data(n: usize, k: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate predictors
    let mut columns: Vec<Column> = Vec::new();

    for i in 1..=k {
        let x: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0) * 2.0 - 1.0).collect();
        columns.push(Column::new(format!("x{}", i).into(), x));
    }

    // Generate y = sum(x_i) + noise
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
            sum + rng.gen_range(0.0..1.0) * 0.5 // Add noise
        })
        .collect();

    columns.insert(0, Column::new("y".into(), y));

    let df = DataFrame::new(columns).expect("Failed to create DataFrame");
    Dataset::new(df)
}

fn ols_standard_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("OLS_Standard");

    for n in [100, 1000, 10000] {
        let dataset = generate_regression_data(n, 5, 42);
        let x_cols = vec!["x1", "x2", "x3", "x4", "x5"];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                run_ols(&dataset, "y", &x_cols, true, CovarianceType::Standard)
            });
        });
    }

    group.finish();
}

fn ols_robust_se_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("OLS_RobustSE");

    let dataset = generate_regression_data(1000, 5, 42);
    let x_cols = vec!["x1", "x2", "x3", "x4", "x5"];

    for (name, cov_type) in [
        ("HC0", CovarianceType::HC0),
        ("HC1", CovarianceType::HC1),
        ("HC2", CovarianceType::HC2),
        ("HC3", CovarianceType::HC3),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &cov_type, |b, cov| {
            b.iter(|| {
                run_ols(&dataset, "y", &x_cols, true, cov.clone())
            });
        });
    }

    group.finish();
}

fn diagnostics_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Diagnostics");

    for n in [100, 1000, 10000] {
        let dataset = generate_regression_data(n, 5, 42);
        let x_cols: Vec<&str> = vec!["x1", "x2", "x3", "x4", "x5"];

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_diagnostics(data, "y", &x_cols)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    ols_standard_benchmark,
    ols_robust_se_benchmark,
    diagnostics_benchmark
);
criterion_main!(benches);
