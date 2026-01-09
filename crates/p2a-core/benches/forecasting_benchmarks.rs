//! Forecasting method benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- forecasting`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::{Dataset, run_arima, run_mstl, run_changepoint, CostFunction};
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic time series with trend and seasonality as a Dataset
fn generate_time_series_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let y: Vec<f64> = (0..n)
        .map(|t| {
            let trend = 0.01 * t as f64;
            let seasonal = (t as f64 * std::f64::consts::PI / 6.0).sin() * 2.0;
            let noise = rng.gen_range(0.0..1.0) * 0.5;
            trend + seasonal + noise
        })
        .collect();

    let df = df! {
        "y" => y,
    }.expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate time series with changepoints as a Dataset
fn generate_changepoint_dataset(n: usize, n_changes: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let segment_size = n / (n_changes + 1);

    let mut y = Vec::with_capacity(n);
    let mut level = 0.0;

    for i in 0..n {
        if i > 0 && i % segment_size == 0 {
            level += rng.gen_range(0.0..1.0) * 5.0 - 2.5; // Random level shift
        }
        y.push(level + rng.gen_range(0.0..1.0) * 0.3);
    }

    let df = df! {
        "y" => y,
    }.expect("Failed to create DataFrame");

    Dataset::new(df)
}

fn arima_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ARIMA");

    for n in [100, 200, 500] {
        let dataset = generate_time_series_dataset(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_arima(data, "y", 1, 1, 1)
            });
        });
    }

    group.finish();
}

fn mstl_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MSTL");

    for n in [100, 200, 500] {
        let dataset = generate_time_series_dataset(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_mstl(data, "y", &[12]) // Monthly seasonality
            });
        });
    }

    group.finish();
}

fn changepoint_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Changepoint");

    for n in [100, 500, 1000] {
        let dataset = generate_changepoint_dataset(n, 3, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_changepoint(data, "y", Some(1.0), Some(5), CostFunction::MeanChange)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    arima_benchmark,
    mstl_benchmark,
    changepoint_benchmark
);
criterion_main!(benches);
