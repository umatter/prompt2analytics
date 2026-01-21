//! Hypothesis testing benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- hypothesis`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::{
    var_test, VarTestResult,
    prop_test_one, prop_test_two, PropTestResult,
    binom_test, BinomTestResult,
    fligner_test, FlignerResult,
    ansari_test, AnsariBradleyResult,
    mood_test, MoodTestResult,
    kruskal_test, KruskalWallisResult,
    friedman_test, FriedmanResult,
    quade_test, QuadeResult,
    mantelhaen_test, Table2x2, CmhAlternative, MantelHaenszelResult,
    oneway_test, OnewayTestResult,
    mcnemar_test, McnemarResult,
    pairwise_t_test, pairwise_wilcox_test, PValueAdjustMethod, PairwiseTTestResult,
    Alternative,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate two samples for variance/scale tests
fn generate_two_samples(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 2.0 + 4.0).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 3.0 + 4.0).collect();

    (x, y)
}

/// Generate three groups for Fligner test
fn generate_three_groups(n: usize, seed: u64) -> Vec<(String, Vec<f64>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let group_size = n / 3;

    vec![
        ("A".to_string(), (0..group_size).map(|_| rng.r#gen::<f64>() * 2.0 + 5.0).collect()),
        ("B".to_string(), (0..group_size).map(|_| rng.r#gen::<f64>() * 3.0 + 5.0).collect()),
        ("C".to_string(), (0..group_size).map(|_| rng.r#gen::<f64>() * 4.0 + 5.0).collect()),
    ]
}

fn var_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("var_test");

    for size in [100, 1000, 10000].iter() {
        let (x, y) = generate_two_samples(*size, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| var_test(&x, &y, 1.0, Alternative::TwoSided, 0.95))
            },
        );
    }
    group.finish();
}

fn prop_test_one_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("prop_test_one");

    for size in [100, 1000, 10000].iter() {
        let successes = (*size as f64 * 0.3) as u64;
        let trials = *size as u64;

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| prop_test_one(successes, trials, 0.5, Alternative::TwoSided, 0.95, true))
            },
        );
    }
    group.finish();
}

fn prop_test_two_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("prop_test_two");

    for size in [100, 1000, 10000].iter() {
        let s1 = (*size as f64 * 0.3) as u64;
        let s2 = (*size as f64 * 0.4) as u64;
        let n = *size as u64;

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| prop_test_two(s1, n, s2, n, Alternative::TwoSided, 0.95, true))
            },
        );
    }
    group.finish();
}

fn binom_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("binom_test");

    for size in [100, 1000, 10000].iter() {
        let successes = (*size as f64 * 0.3) as u64;
        let trials = *size as u64;

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| binom_test(successes, trials, 0.5, Alternative::TwoSided, 0.95))
            },
        );
    }
    group.finish();
}

fn fligner_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("fligner_test");

    for size in [100, 1000, 10000].iter() {
        let groups = generate_three_groups(*size, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| fligner_test(&groups))
            },
        );
    }
    group.finish();
}

fn ansari_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ansari_test");

    // Ansari test with exact computation is expensive for large n
    // Use fixed size of 50 for exact test comparison with R
    let (x, y) = generate_two_samples(50, 42);

    group.bench_function("n=50_exact", |b| {
        b.iter(|| ansari_test(&x, &y, Alternative::TwoSided, true, None))
    });

    // Also benchmark with approximation for larger sizes
    for size in [100, 1000].iter() {
        let (x, y) = generate_two_samples(*size, 42);

        group.bench_with_input(
            BenchmarkId::new("approx", size),
            size,
            |b, _| {
                b.iter(|| ansari_test(&x, &y, Alternative::TwoSided, false, None))
            },
        );
    }
    group.finish();
}

fn mood_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mood_test");

    for size in [100, 1000, 10000, 100000].iter() {
        let (x, y) = generate_two_samples(*size, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| mood_test(&x, &y, Alternative::TwoSided))
            },
        );
    }
    group.finish();
}

/// Generate grouped data for Kruskal-Wallis test
fn generate_kruskal_groups(n_per_group: usize, n_groups: usize, seed: u64) -> Vec<(String, Vec<f64>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..n_groups)
        .map(|i| {
            let group_name = format!("G{}", i);
            let values: Vec<f64> = (0..n_per_group)
                .map(|_| rng.r#gen::<f64>() * 10.0 + (i as f64) * 2.0)
                .collect();
            (group_name, values)
        })
        .collect()
}

fn kruskal_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("kruskal_test");

    // Test with 3 groups of varying sizes
    for size in [100, 1000, 10000, 100000].iter() {
        let groups = generate_kruskal_groups(*size / 3, 3, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| kruskal_test(&groups))
            },
        );
    }
    group.finish();
}

/// Generate blocked data for Friedman test (n_blocks x n_treatments matrix)
fn generate_friedman_data(n_blocks: usize, n_treatments: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<String>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let data: Vec<Vec<f64>> = (0..n_blocks)
        .map(|_| {
            (0..n_treatments)
                .map(|t| rng.r#gen::<f64>() * 10.0 + (t as f64) * 2.0)
                .collect()
        })
        .collect();

    let names: Vec<String> = (0..n_treatments).map(|i| format!("T{}", i)).collect();

    (data, names)
}

fn friedman_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("friedman_test");

    // Test with 3 treatments and varying number of blocks
    for n_blocks in [30, 100, 300, 1000].iter() {
        let (data, names) = generate_friedman_data(*n_blocks, 3, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(n_blocks),
            n_blocks,
            |b, _| {
                b.iter(|| friedman_test(&data, &names))
            },
        );
    }
    group.finish();
}

fn quade_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("quade_test");

    // Test with varying number of blocks and treatments
    // (blocks, treatments)
    let configs = vec![
        (10, 3),    // Small: 10 blocks, 3 treatments
        (50, 4),    // Medium: 50 blocks, 4 treatments
        (100, 5),   // Large: 100 blocks, 5 treatments
        (500, 3),   // Very large: 500 blocks, 3 treatments
    ];

    for (n_blocks, n_treatments) in configs {
        let (data, names) = generate_friedman_data(n_blocks, n_treatments, 42);
        let label = format!("b{}_t{}", n_blocks, n_treatments);

        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &label,
            |b, _| {
                b.iter(|| quade_test(&data, &names))
            },
        );
    }
    group.finish();
}

fn oneway_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("oneway_test");

    // Test with 3 groups of varying sizes (Welch's ANOVA)
    for size in [100, 1000, 10000, 100000].iter() {
        let groups = generate_kruskal_groups(*size / 3, 3, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| oneway_test(&groups, false))  // Welch's ANOVA (var.equal = FALSE)
            },
        );
    }
    group.finish();
}

fn mcnemar_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcnemar_test");

    // McNemar test operates on summary counts, so we benchmark repeated invocations
    // Using the classic R documentation example: b=150, c=86
    let b = 150u64;
    let c = 86u64;

    for n_tests in [1, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_tests),
            n_tests,
            |bench, &n| {
                bench.iter(|| {
                    for _ in 0..n {
                        let _ = mcnemar_test(b, c, true);
                    }
                })
            },
        );
    }
    group.finish();
}

/// Generate pairwise test data: k groups with n_per_group observations each
fn generate_pairwise_groups(k_groups: usize, n_per_group: usize, seed: u64) -> (Vec<f64>, Vec<String>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut values = Vec::with_capacity(k_groups * n_per_group);
    let mut groups = Vec::with_capacity(k_groups * n_per_group);

    for g in 0..k_groups {
        let group_mean = (g + 1) as f64 * 10.0;  // Groups have different means
        for _ in 0..n_per_group {
            values.push(group_mean + rng.r#gen::<f64>() * 4.0 - 2.0);  // Mean ± 2
            groups.push(format!("G{}", g + 1));
        }
    }

    (values, groups)
}

fn pairwise_t_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_t_test");

    // Test configurations: (k_groups, n_per_group)
    let configs = vec![
        (3, 10),    // 3 comparisons
        (5, 50),    // 10 comparisons
        (10, 100),  // 45 comparisons
        (20, 500),  // 190 comparisons
    ];

    for (k, n) in configs {
        let (values, groups) = generate_pairwise_groups(k, n, 42);
        let label = format!("k{}_n{}", k, n);

        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(values.clone(), groups.clone()),
            |b, (vals, grps)| {
                b.iter(|| pairwise_t_test(vals, grps, true, Alternative::TwoSided, PValueAdjustMethod::Holm))
            },
        );
    }
    group.finish();
}

fn pairwise_wilcox_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_wilcox_test");

    // Test configurations: (k_groups, n_per_group)
    // Use smaller configs for Wilcoxon (ranking is more expensive)
    let configs = vec![
        (3, 10),    // 3 comparisons
        (5, 50),    // 10 comparisons
        (10, 100),  // 45 comparisons
        (20, 200),  // 190 comparisons (smaller n per group due to ranking cost)
    ];

    for (k, n) in configs {
        let (values, groups) = generate_pairwise_groups(k, n, 42);
        let label = format!("k{}_n{}", k, n);

        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(values.clone(), groups.clone()),
            |b, (vals, grps)| {
                b.iter(|| pairwise_wilcox_test(vals, grps, Alternative::TwoSided, PValueAdjustMethod::Holm, Some(false)))
            },
        );
    }
    group.finish();
}

/// Generate stratified 2×2 table data for CMH test
fn generate_cmh_tables(n_strata: usize, seed: u64) -> Vec<Table2x2> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..n_strata)
        .map(|_| {
            // Generate realistic cell counts with some variation
            let base = (rng.r#gen::<f64>() * 50.0 + 10.0) as f64;
            Table2x2::new(
                base * (0.5 + rng.r#gen::<f64>()),
                base * (0.8 + rng.r#gen::<f64>()),
                base * (0.3 + rng.r#gen::<f64>()),
                base * (1.0 + rng.r#gen::<f64>()),
            )
        })
        .collect()
}

fn mantelhaen_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mantelhaen_test");

    // Test with varying number of strata
    for n_strata in [5, 10, 50, 100].iter() {
        let tables = generate_cmh_tables(*n_strata, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(n_strata),
            n_strata,
            |b, _| {
                b.iter(|| mantelhaen_test(&tables, None, true, CmhAlternative::TwoSided))
            },
        );
    }
    group.finish();
}

fn poisson_test_benchmark(c: &mut Criterion) {
    use p2a_core::{poisson_test, PoissonAlternative, PoissonTestResult};

    let mut group = c.benchmark_group("poisson_test");

    // One-sample test benchmark
    for size in [1u64, 10, 100, 1000].iter() {
        let x = *size * 10;  // events scale with size
        let t = *size as f64;  // time base

        group.bench_with_input(
            BenchmarkId::new("one_sample", size),
            size,
            |b, _| {
                b.iter(|| poisson_test(&[x], &[t], 1.0, PoissonAlternative::TwoSided, 0.95))
            },
        );
    }

    // Two-sample test benchmark
    for size in [1u64, 10, 100, 1000].iter() {
        let x1 = *size * 5;
        let x2 = *size * 10;
        let t1 = *size as f64;
        let t2 = *size as f64 * 2.0;

        group.bench_with_input(
            BenchmarkId::new("two_sample", size),
            size,
            |b, _| {
                b.iter(|| poisson_test(&[x1, x2], &[t1, t2], 1.0, PoissonAlternative::TwoSided, 0.95))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    var_test_benchmark,
    prop_test_one_benchmark,
    prop_test_two_benchmark,
    binom_test_benchmark,
    fligner_test_benchmark,
    ansari_test_benchmark,
    mood_test_benchmark,
    kruskal_test_benchmark,
    friedman_test_benchmark,
    quade_test_benchmark,
    mantelhaen_test_benchmark,
    oneway_test_benchmark,
    mcnemar_test_benchmark,
    pairwise_t_test_benchmark,
    pairwise_wilcox_test_benchmark,
    poisson_test_benchmark,
);

criterion_main!(benches);
