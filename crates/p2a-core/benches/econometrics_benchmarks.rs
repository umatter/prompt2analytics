//! Econometrics method benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- econometrics`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::{Dataset, run_fixed_effects, run_hdfe, run_logit, run_probit};
use p2a_core::regression::CovarianceType;
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic panel data
fn generate_panel_data(n_entities: usize, n_periods: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_entities * n_periods;

    let entity: Vec<i64> = (1..=n_entities as i64)
        .flat_map(|e| std::iter::repeat(e).take(n_periods))
        .collect();

    let time: Vec<i64> = (1..=n_entities)
        .flat_map(|_| 1..=n_periods as i64)
        .collect();

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0) * 2.0 - 1.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0) * 2.0 - 1.0).collect();

    // y = 1.0*x1 + 0.5*x2 + entity_effect + noise
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let entity_effect = (entity[i] as f64) * 0.5;
            1.0 * x1[i] + 0.5 * x2[i] + entity_effect + rng.gen_range(0.0..1.0) * 0.3
        })
        .collect();

    let df = df! {
        "y" => y,
        "x1" => x1,
        "x2" => x2,
        "entity" => entity,
        "time" => time,
    }.expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate synthetic binary outcome data
fn generate_binary_data(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0) * 2.0 - 1.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0) * 2.0 - 1.0).collect();

    // Probability via logit
    let y: Vec<i64> = (0..n)
        .map(|i| {
            let latent = 0.5 * x1[i] + 0.3 * x2[i];
            let prob = 1.0 / (1.0 + (-latent).exp());
            if rng.gen_range(0.0..1.0) < prob { 1 } else { 0 }
        })
        .collect();

    let df = df! {
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }.expect("Failed to create DataFrame");

    Dataset::new(df)
}

fn fixed_effects_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FixedEffects");

    for (n_entities, n_periods) in [(10, 10), (50, 20), (100, 50)] {
        let dataset = generate_panel_data(n_entities, n_periods, 42);
        let n = n_entities * n_periods;

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_fixed_effects(data, "y", &["x1", "x2"], "entity")
            });
        });
    }

    group.finish();
}

fn hdfe_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDFE");

    for (n_entities, n_periods) in [(10, 10), (50, 20), (100, 50)] {
        let dataset = generate_panel_data(n_entities, n_periods, 42);
        let n = n_entities * n_periods;

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_hdfe(data, "y", &["x1", "x2"], &["entity", "time"], None, CovarianceType::Standard)
            });
        });
    }

    group.finish();
}

fn logit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Logit");

    for n in [100, 500, 1000] {
        let dataset = generate_binary_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_logit(data, "y", &["x1", "x2"])
            });
        });
    }

    group.finish();
}

fn probit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Probit");

    for n in [100, 500, 1000] {
        let dataset = generate_binary_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_probit(data, "y", &["x1", "x2"])
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    fixed_effects_benchmark,
    hdfe_benchmark,
    logit_benchmark,
    probit_benchmark
);
criterion_main!(benches);
