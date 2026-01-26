//! Benchmarks for new causal inference methods

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use p2a_core::data::Dataset;
use polars::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;

fn create_causal_dataset(n: usize) -> Dataset {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x1: Vec<f64> = (0..n).map(|_| rng.sample(normal)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.sample(normal)).collect();
    let x3: Vec<f64> = (0..n).map(|_| rng.sample(normal)).collect();

    // Propensity score based treatment
    let treatment: Vec<f64> = x1.iter().zip(x2.iter()).zip(x3.iter())
        .map(|((&a, &b), &c)| {
            let ps = 1.0 / (1.0 + (-0.5 - 0.3*a - 0.2*b + 0.1*c).exp());
            if rng.r#gen::<f64>() < ps { 1.0 } else { 0.0 }
        })
        .collect();

    // Outcome with treatment effect
    let y: Vec<f64> = treatment.iter().zip(x1.iter()).zip(x2.iter()).zip(x3.iter())
        .map(|(((&t, &a), &b), &c)| {
            2.0 + 0.5*t + 0.3*a + 0.2*b + 0.1*c + rng.sample(normal)
        })
        .collect();

    let df = df! {
        "y" => y,
        "treatment" => treatment,
        "x1" => x1,
        "x2" => x2,
        "x3" => x3,
    }.unwrap();

    Dataset::new(df)
}

fn create_panel_dataset(n_units: usize, n_periods: usize) -> Dataset {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 0.5).unwrap();

    let mut unit = Vec::new();
    let mut time = Vec::new();
    let mut treated = Vec::new();
    let mut y = Vec::new();

    for u in 0..n_units {
        let treat_time = if u < n_units / 3 { 4 }
                        else if u < 2 * n_units / 3 { 6 }
                        else { 100 }; // Never treated

        for t in 1..=n_periods {
            unit.push(u as f64);
            time.push(t as f64);
            let is_treated = if t >= treat_time { 1.0 } else { 0.0 };
            treated.push(is_treated);
            y.push(1.0 + 0.5 * is_treated + 0.1 * (t as f64) + rng.sample(normal));
        }
    }

    let df = df! {
        "unit" => unit,
        "time" => time,
        "treated" => treated,
        "y" => y,
    }.unwrap();

    Dataset::new(df)
}

fn bench_matching(c: &mut Criterion) {
    use p2a_core::econometrics::{match_it, MatchMethod};

    let dataset = create_causal_dataset(1000);
    let covariates = vec!["x1", "x2", "x3"];

    let mut group = c.benchmark_group("matching");

    group.bench_function("nearest_neighbor_n1000", |b| {
        b.iter(|| {
            match_it(
                black_box(&dataset),
                "treatment",
                &covariates,
                MatchMethod::NearestNeighbor {
                    ratio: 1,
                    caliper: None,
                    replace: false,
                },
                None,
            )
        })
    });

    group.bench_function("cem_n1000", |b| {
        b.iter(|| {
            match_it(
                black_box(&dataset),
                "treatment",
                &covariates,
                MatchMethod::CoarsenedExact {
                    cutpoints: None,
                    n_bins: Some(4),
                },
                None,
            )
        })
    });

    group.finish();
}

fn bench_weighting(c: &mut Criterion) {
    use p2a_core::econometrics::{weightit, WeightMethod, WeightItConfig};

    let dataset = create_causal_dataset(1000);
    let covariates = vec!["x1", "x2", "x3"];

    let mut group = c.benchmark_group("weighting");

    group.bench_function("logistic_n1000", |b| {
        b.iter(|| {
            let config = WeightItConfig {
                method: WeightMethod::Logistic,
                ..Default::default()
            };
            weightit(black_box(&dataset), "treatment", &covariates, config)
        })
    });

    group.bench_function("entropy_n1000", |b| {
        b.iter(|| {
            let config = WeightItConfig {
                method: WeightMethod::Entropy,
                ..Default::default()
            };
            weightit(black_box(&dataset), "treatment", &covariates, config)
        })
    });

    group.finish();
}

fn bench_cbps(c: &mut Criterion) {
    use p2a_core::econometrics::{run_cbps, CbpsConfig, CbpsMethod};

    let dataset = create_causal_dataset(1000);
    let covariates = vec!["x1", "x2", "x3"];

    let mut group = c.benchmark_group("cbps");

    group.bench_function("exact_n1000", |b| {
        b.iter(|| {
            let config = CbpsConfig {
                method: CbpsMethod::ExactBalance,
                ..Default::default()
            };
            run_cbps(black_box(&dataset), "treatment", &covariates, Some(config))
        })
    });

    group.finish();
}

fn bench_sensemakr(c: &mut Criterion) {
    use p2a_core::regression::run_sensemakr;

    let dataset = create_causal_dataset(1000);
    let covariates = vec!["x1", "x2", "x3"];

    let mut group = c.benchmark_group("sensemakr");

    group.bench_function("sensitivity_n1000", |b| {
        b.iter(|| {
            run_sensemakr(
                black_box(&dataset),
                "y",
                "treatment",
                &covariates,
                Some(&["x1"]),
                None, None, 1.0, 0.05
            )
        })
    });

    group.finish();
}

fn bench_tmle(c: &mut Criterion) {
    use p2a_core::econometrics::{tmle, TmleConfig};

    let dataset = create_causal_dataset(500); // Smaller for TMLE
    let covariates = vec!["x1", "x2", "x3"];

    let mut group = c.benchmark_group("tmle");
    group.sample_size(10); // TMLE is slow

    group.bench_function("tmle_n500", |b| {
        b.iter(|| {
            tmle(
                black_box(&dataset),
                "y",
                "treatment",
                &covariates,
                TmleConfig::default()
            )
        })
    });

    group.finish();
}

fn bench_bacon(c: &mut Criterion) {
    use p2a_core::econometrics::bacon_decomp;

    let dataset = create_panel_dataset(50, 10);

    let mut group = c.benchmark_group("bacon");

    group.bench_function("decomp_50x10", |b| {
        b.iter(|| {
            bacon_decomp(
                black_box(&dataset),
                "y",
                "unit",
                "time",
                "treated"
            )
        })
    });

    group.finish();
}

fn bench_marginal_effects(c: &mut Criterion) {
    use p2a_core::regression::{marginal_effects, ModelType};

    let dataset = create_causal_dataset(1000);
    let x_cols = vec!["treatment", "x1", "x2", "x3"];

    let mut group = c.benchmark_group("marginal_effects");

    group.bench_function("ols_n1000", |b| {
        b.iter(|| {
            marginal_effects(
                black_box(&dataset),
                "y",
                &x_cols,
                ModelType::Ols
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_matching,
    bench_weighting,
    bench_cbps,
    bench_sensemakr,
    bench_tmle,
    bench_bacon,
    bench_marginal_effects
);

criterion_main!(benches);
