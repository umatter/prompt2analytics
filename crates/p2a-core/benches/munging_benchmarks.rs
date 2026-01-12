//! Data munging benchmarks for p2a-core
//!
//! Benchmarks filter, select, join, group_by, pivot/melt, and lag/lead operations.
//! Run with: `cargo bench -p p2a-core -- munging`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p2a_core::data::munging::{
    filter, select, sort, left_join, group_by, pivot, melt, lag, fill_na,
    AggFn, AggSpec, FillStrategy,
};
use p2a_core::Dataset;
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic data for munging benchmarks
/// Creates: id, group (100 groups), x1, x2, x3 (numeric), category (10 categories)
fn generate_munging_data(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let id: Vec<i64> = (1..=n as i64).collect();
    let group: Vec<i64> = (0..n).map(|_| rng.gen_range(1..=100)).collect();
    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x3: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let category: Vec<String> = (0..n)
        .map(|_| {
            let idx = rng.gen_range(0..10);
            ((b'a' + idx) as char).to_string()
        })
        .collect();

    let df = DataFrame::new(vec![
        Column::new("id".into(), id),
        Column::new("group".into(), group),
        Column::new("x1".into(), x1),
        Column::new("x2".into(), x2),
        Column::new("x3".into(), x3),
        Column::new("category".into(), category),
    ])
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate data for join benchmarks (right table with 1/10 the rows)
fn generate_join_data(n: usize, seed: u64) -> (Dataset, Dataset) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Left table: main data
    let left = generate_munging_data(n, seed);

    // Right table: lookup table with 1/10 the rows
    let right_n = n / 10;
    let key: Vec<i64> = (1..=right_n as i64).collect();
    let value: Vec<f64> = (0..right_n).map(|_| rng.gen_range(0.0..100.0)).collect();
    let label: Vec<String> = (0..right_n)
        .map(|i| format!("label_{}", i % 100))
        .collect();

    let right_df = DataFrame::new(vec![
        Column::new("group".into(), key),
        Column::new("lookup_value".into(), value),
        Column::new("lookup_label".into(), label),
    ])
    .expect("Failed to create right DataFrame");

    (left, Dataset::new(right_df))
}

/// Generate panel data for lag/lead benchmarks
fn generate_panel_data(n_entities: usize, n_periods: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_entities * n_periods;

    let entity: Vec<i64> = (0..n)
        .map(|i| (i / n_periods + 1) as i64)
        .collect();
    let period: Vec<i64> = (0..n)
        .map(|i| (i % n_periods + 1) as i64)
        .collect();
    let value: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let df = DataFrame::new(vec![
        Column::new("entity".into(), entity),
        Column::new("period".into(), period),
        Column::new("value".into(), value),
    ])
    .expect("Failed to create panel DataFrame");

    Dataset::new(df)
}

/// Generate data for pivot benchmarks (long format)
fn generate_pivot_data(n_ids: usize, n_vars: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_ids * n_vars;

    let id: Vec<i64> = (0..n).map(|i| (i / n_vars + 1) as i64).collect();
    let variable: Vec<String> = (0..n)
        .map(|i| format!("var_{}", i % n_vars + 1))
        .collect();
    let value: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..100.0)).collect();

    let df = DataFrame::new(vec![
        Column::new("id".into(), id),
        Column::new("variable".into(), variable),
        Column::new("value".into(), value),
    ])
    .expect("Failed to create pivot DataFrame");

    Dataset::new(df)
}

/// Generate data with missing values for fill_na benchmarks
fn generate_data_with_na(n: usize, na_fraction: f64, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let id: Vec<i64> = (1..=n as i64).collect();
    let x1: Vec<Option<f64>> = (0..n)
        .map(|_| {
            if rng.gen_range(0.0..1.0) < na_fraction {
                None
            } else {
                Some(rng.gen_range(-1.0..1.0))
            }
        })
        .collect();
    let x2: Vec<Option<f64>> = (0..n)
        .map(|_| {
            if rng.gen_range(0.0..1.0) < na_fraction {
                None
            } else {
                Some(rng.gen_range(-1.0..1.0))
            }
        })
        .collect();

    let df = DataFrame::new(vec![
        Column::new("id".into(), id),
        Column::new("x1".into(), x1),
        Column::new("x2".into(), x2),
    ])
    .expect("Failed to create DataFrame with NA");

    Dataset::new(df)
}

// =============================================================================
// FILTER BENCHMARKS
// =============================================================================

fn filter_numeric_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_numeric");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| filter(ds, "x1", "gt", "0.0"));
        });
    }

    group.finish();
}

fn filter_string_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_string");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| filter(ds, "category", "eq", "a"));
        });
    }

    group.finish();
}

// =============================================================================
// SELECT BENCHMARKS
// =============================================================================

fn select_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("select");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| select(ds, &["id", "x1", "x2"]));
        });
    }

    group.finish();
}

// =============================================================================
// SORT BENCHMARKS
// =============================================================================

fn sort_single_column_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_single");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| sort(ds, &["x1"], &[false]));
        });
    }

    group.finish();
}

fn sort_multi_column_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_multi");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| sort(ds, &["group", "x1"], &[false, true]));
        });
    }

    group.finish();
}

// =============================================================================
// JOIN BENCHMARKS
// =============================================================================

fn left_join_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("left_join");

    for n in [10_000, 100_000, 1_000_000] {
        let (left, right) = generate_join_data(n, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(left.clone(), right.clone()),
            |b, (l, r)| {
                b.iter(|| left_join(l, r, &["group"], None, None));
            },
        );
    }

    group.finish();
}

// =============================================================================
// GROUP BY BENCHMARKS
// =============================================================================

fn group_by_single_agg_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_by_single_agg");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| {
                group_by(
                    ds,
                    &["group"],
                    &[AggSpec::new("x1", AggFn::Sum)],
                )
            });
        });
    }

    group.finish();
}

fn group_by_multi_agg_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_by_multi_agg");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| {
                group_by(
                    ds,
                    &["group"],
                    &[
                        AggSpec::new("x1", AggFn::Sum),
                        AggSpec::new("x2", AggFn::Mean),
                        AggSpec::new("x3", AggFn::Max),
                    ],
                )
            });
        });
    }

    group.finish();
}

fn group_by_multi_key_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_by_multi_key");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| {
                group_by(
                    ds,
                    &["group", "category"],
                    &[
                        AggSpec::new("x1", AggFn::Sum),
                        AggSpec::new("x2", AggFn::Mean),
                    ],
                )
            });
        });
    }

    group.finish();
}

// =============================================================================
// PIVOT/MELT BENCHMARKS
// =============================================================================

fn pivot_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pivot");
    // Use smaller sizes for pivot as it creates many columns
    for n_ids in [1_000, 10_000, 100_000] {
        let dataset = generate_pivot_data(n_ids, 5, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n_ids), &dataset, |b, ds| {
            b.iter(|| pivot(ds, &["id"], "variable", "value"));
        });
    }

    group.finish();
}

fn melt_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("melt");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| melt(ds, &["id", "group"], &["x1", "x2", "x3"], "variable", "value"));
        });
    }

    group.finish();
}

// =============================================================================
// LAG/LEAD BENCHMARKS
// =============================================================================

fn lag_simple_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("lag_simple");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_munging_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| lag(ds, "x1", 1, None));
        });
    }

    group.finish();
}

fn lag_grouped_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("lag_grouped");

    // Panel data: entities x periods
    for (n_entities, n_periods) in [(100, 100), (500, 200), (1000, 1000)] {
        let dataset = generate_panel_data(n_entities, n_periods, 42);
        let label = format!("{}x{}", n_entities, n_periods);

        group.bench_with_input(BenchmarkId::from_parameter(label), &dataset, |b, ds| {
            b.iter(|| lag(ds, "value", 1, Some(&["entity"])));
        });
    }

    group.finish();
}

// =============================================================================
// FILL NA BENCHMARKS
// =============================================================================

fn fill_na_forward_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill_na_forward");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_data_with_na(n, 0.1, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| fill_na(ds, Some(&["x1", "x2"]), FillStrategy::Forward));
        });
    }

    group.finish();
}

fn fill_na_mean_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill_na_mean");

    for n in [10_000, 100_000, 1_000_000] {
        let dataset = generate_data_with_na(n, 0.1, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, ds| {
            b.iter(|| fill_na(ds, Some(&["x1", "x2"]), FillStrategy::Mean));
        });
    }

    group.finish();
}

// =============================================================================
// CRITERION SETUP
// =============================================================================

criterion_group!(
    filter_benches,
    filter_numeric_benchmark,
    filter_string_benchmark
);

criterion_group!(
    select_benches,
    select_benchmark
);

criterion_group!(
    sort_benches,
    sort_single_column_benchmark,
    sort_multi_column_benchmark
);

criterion_group!(
    join_benches,
    left_join_benchmark
);

criterion_group!(
    group_by_benches,
    group_by_single_agg_benchmark,
    group_by_multi_agg_benchmark,
    group_by_multi_key_benchmark
);

criterion_group!(
    reshape_benches,
    pivot_benchmark,
    melt_benchmark
);

criterion_group!(
    lag_benches,
    lag_simple_benchmark,
    lag_grouped_benchmark
);

criterion_group!(
    fill_na_benches,
    fill_na_forward_benchmark,
    fill_na_mean_benchmark
);

criterion_main!(
    filter_benches,
    select_benches,
    sort_benches,
    join_benches,
    group_by_benches,
    reshape_benches,
    lag_benches,
    fill_na_benches
);
