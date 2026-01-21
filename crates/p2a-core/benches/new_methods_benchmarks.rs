//! Benchmarks for newly implemented methods
//!
//! Run with: `cargo bench -p p2a-core -- new_methods`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::{
    // cor.test
    cor_test, CorrelationMethod, Alternative,
    // power analysis
    power_t_test, power_prop_test, power_anova_test,
    TTestType, PowerAlternative,
    // robust stats
    fivenum, iqr, mad, ecdf, density, DensityKernel,
    // spline/approx
    spline, approx, SplineMethod, ApproxMethod, ApproxRule,
    // smooth.spline
    smooth_spline, SmoothSplineConfig,
    // prop.trend.test
    prop_trend_test,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn generate_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.r#gen::<f64>() * 10.0).collect()
}

fn generate_correlated_pair(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 10.0).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 0.7 + rng.r#gen::<f64>() * 3.0).collect();
    (x, y)
}

fn cor_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cor_test");

    for size in [100, 1000, 10000].iter() {
        let (x, y) = generate_correlated_pair(*size, 42);

        group.bench_with_input(BenchmarkId::new("pearson", size), size, |b, _| {
            b.iter(|| cor_test(&x, &y, CorrelationMethod::Pearson, Alternative::TwoSided, 0.95))
        });

        group.bench_with_input(BenchmarkId::new("spearman", size), size, |b, _| {
            b.iter(|| cor_test(&x, &y, CorrelationMethod::Spearman, Alternative::TwoSided, 0.95))
        });
    }
    group.finish();
}

fn power_analysis_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_analysis");

    group.bench_function("power_t_test", |b| {
        b.iter(|| power_t_test(
            Some(30.0), Some(0.5), Some(1.0), Some(0.05), None,
            TTestType::TwoSample, PowerAlternative::TwoSided
        ))
    });

    group.bench_function("power_prop_test", |b| {
        b.iter(|| power_prop_test(
            Some(100.0), 0.5, Some(0.6), Some(0.05), None,
            PowerAlternative::TwoSided
        ))
    });

    group.bench_function("power_anova_test", |b| {
        b.iter(|| power_anova_test(
            4, Some(20.0), 1.0, 3.0, Some(0.05), None
        ))
    });

    group.finish();
}

fn robust_stats_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("robust_stats");

    for size in [100, 1000, 10000].iter() {
        let x = generate_data(*size, 42);

        group.bench_with_input(BenchmarkId::new("fivenum", size), size, |b, _| {
            b.iter(|| fivenum(&x))
        });

        group.bench_with_input(BenchmarkId::new("iqr", size), size, |b, _| {
            b.iter(|| iqr(&x, None))
        });

        group.bench_with_input(BenchmarkId::new("mad", size), size, |b, _| {
            b.iter(|| mad(&x, None, None))
        });

        group.bench_with_input(BenchmarkId::new("ecdf", size), size, |b, _| {
            b.iter(|| ecdf(&x))
        });

        group.bench_with_input(BenchmarkId::new("density", size), size, |b, _| {
            b.iter(|| density(&x, None, DensityKernel::Gaussian, None, None, None))
        });
    }
    group.finish();
}

fn spline_approx_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("spline_approx");

    for size in [10, 50, 100].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut x: Vec<f64> = (0..*size).map(|_| rng.r#gen::<f64>() * 10.0).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        group.bench_with_input(BenchmarkId::new("spline_natural", size), size, |b, _| {
            b.iter(|| spline(&x, &y, None, Some(100), SplineMethod::Natural))
        });

        group.bench_with_input(BenchmarkId::new("approx_linear", size), size, |b, _| {
            b.iter(|| approx(&x, &y, None, Some(100), ApproxMethod::Linear, ApproxRule::Na, 0.5))
        });
    }
    group.finish();
}

fn smooth_spline_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("smooth_spline");

    for size in [50, 100, 200].iter() {
        let x: Vec<f64> = (0..*size).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        group.bench_with_input(BenchmarkId::new("df10", size), size, |b, _| {
            b.iter(|| smooth_spline(&x, &y, SmoothSplineConfig::with_df(10.0)))
        });
    }
    group.finish();
}

fn prop_trend_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("prop_trend_test");

    for k in [3, 5, 10, 20].iter() {
        let x: Vec<usize> = (0..*k).map(|i| 50 + i * 5).collect();
        let n: Vec<usize> = vec![100; *k];

        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, _| {
            b.iter(|| prop_trend_test(&x, &n, None))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    cor_test_benchmark,
    power_analysis_benchmark,
    robust_stats_benchmark,
    spline_approx_benchmark,
    smooth_spline_benchmark,
    prop_trend_test_benchmark,
);
criterion_main!(benches);
