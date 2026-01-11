//! Econometrics method benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- econometrics`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::{
    Dataset, run_fixed_effects, run_hdfe, run_logit, run_probit,
    run_ipw_treatment, run_doubly_robust, run_mediation_analysis,
    IpwConfig, DoublyRobustConfig, MediationConfig, Estimand, DRMethod,
};
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

/// Generate synthetic treatment effects data
fn generate_treatment_data(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Propensity score model
    let treatment: Vec<f64> = (0..n)
        .map(|i| {
            let ps = 1.0 / (1.0 + (-0.5 - 0.3 * x1[i] - 0.2 * x2[i]).exp());
            if rng.gen_range(0.0..1.0) < ps { 1.0 } else { 0.0 }
        })
        .collect();

    // Outcome with treatment effect = 2.0
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            2.0 * treatment[i] + 1.0 * x1[i] + 0.5 * x2[i] + rng.gen_range(-0.5..0.5)
        })
        .collect();

    let df = df! {
        "outcome" => outcome,
        "treatment" => treatment,
        "x1" => x1,
        "x2" => x2,
    }.expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate synthetic mediation data
fn generate_mediation_data(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Random treatment assignment (50/50)
    let treatment: Vec<f64> = (0..n)
        .map(|_| if rng.gen_range(0.0..1.0) < 0.5 { 1.0 } else { 0.0 })
        .collect();

    // Mediator: M = 0.5*D + 0.3*X + noise
    let mediator: Vec<f64> = (0..n)
        .map(|i| 0.5 * treatment[i] + 0.3 * x[i] + rng.gen_range(-0.3..0.3))
        .collect();

    // Outcome: Y = 0.4*D + 0.6*M + 0.2*X + noise
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            0.4 * treatment[i] + 0.6 * mediator[i] + 0.2 * x[i] + rng.gen_range(-0.5..0.5)
        })
        .collect();

    let df = df! {
        "outcome" => outcome,
        "treatment" => treatment,
        "mediator" => mediator,
        "x" => x,
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

fn ipw_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("IPW_TreatmentEffects");

    for n in [200, 500, 1000, 2000] {
        let dataset = generate_treatment_data(n, 42);

        // Use minimal bootstrap for benchmarking (bootstrap is slow)
        let config = IpwConfig {
            trim: 0.05,
            estimand: Estimand::ATE,
            bootstrap: 99,  // Reduced for speed
            normalized: true,
            seed: Some(42),
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_ipw_treatment(data, "outcome", "treatment", &["x1", "x2"], config.clone())
            });
        });
    }

    group.finish();
}

fn doubly_robust_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("DoublyRobust_AIPW");

    for n in [200, 500, 1000, 2000] {
        let dataset = generate_treatment_data(n, 42);

        let config = DoublyRobustConfig {
            method: DRMethod::AIPW,
            trim: 0.05,
            estimand: Estimand::ATE,
            bootstrap: 99,  // Reduced for speed
            seed: Some(42),
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_doubly_robust(data, "outcome", "treatment", &["x1", "x2"], config.clone())
            });
        });
    }

    group.finish();
}

fn mediation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MediationAnalysis");

    for n in [200, 500, 1000] {
        let dataset = generate_mediation_data(n, 42);

        let config = MediationConfig {
            bootstrap: 99,  // Reduced for speed
            trim: 0.05,
            seed: Some(42),
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| {
                run_mediation_analysis(data, "outcome", "treatment", "mediator", &["x"], config.clone())
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
    probit_benchmark,
    ipw_benchmark,
    doubly_robust_benchmark,
    mediation_benchmark
);
criterion_main!(benches);
