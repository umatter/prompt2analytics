//! Regression method benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- regression`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::{
    Dataset, run_ols, run_diagnostics, run_one_way_anova, run_two_way_anova,
    one_sample_t_test, two_sample_t_test, Alternative,
    acf, pacf, ccf, AcfType,
    chisq_test_gof, chisq_test_independence,
    fisher_exact_test, FisherAlternative,
    nls, NlsConfig, NlsAlgorithm,
    model_exponential_decay, model_michaelis_menten,
    loess, LoessConfig,
    wilcoxon_rank_sum, wilcoxon_signed_rank, WilcoxonConfig,
    shapiro_wilk_test,
    ks_test_two_sample, ks_test_one_sample, TheoreticalDistribution,
    manova_one_way, run_tukey_hsd, bartlett_test,
    spectrum, spectrum_ar, SpectrumConfig,
    box_test, BoxTestType,
    pp_test,
};
use ndarray::Array1;
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

/// Generate synthetic ANOVA data with known group means
fn generate_anova_data(n_per_group: usize, n_groups: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut y_values: Vec<f64> = Vec::new();
    let mut group_labels: Vec<String> = Vec::new();

    // Generate data for each group with different means
    for g in 0..n_groups {
        let group_mean = (g + 1) as f64 * 5.0; // Means: 5, 10, 15, ...
        for _ in 0..n_per_group {
            y_values.push(group_mean + rng.gen_range(-1.0..1.0));
            group_labels.push(format!("Group{}", g));
        }
    }

    let df = DataFrame::new(vec![
        Column::new("y".into(), y_values),
        Column::new("group".into(), group_labels),
    ]).expect("Failed to create DataFrame");

    Dataset::new(df)
}

/// Generate synthetic two-way ANOVA data
fn generate_two_way_anova_data(n_per_cell: usize, levels_a: usize, levels_b: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut y_values: Vec<f64> = Vec::new();
    let mut factor_a: Vec<String> = Vec::new();
    let mut factor_b: Vec<String> = Vec::new();

    for a in 0..levels_a {
        for b in 0..levels_b {
            let cell_mean = (a + 1) as f64 * 5.0 + (b + 1) as f64 * 10.0;
            for _ in 0..n_per_cell {
                y_values.push(cell_mean + rng.gen_range(-1.0..1.0));
                factor_a.push(format!("A{}", a));
                factor_b.push(format!("B{}", b));
            }
        }
    }

    let df = DataFrame::new(vec![
        Column::new("y".into(), y_values),
        Column::new("factor_a".into(), factor_a),
        Column::new("factor_b".into(), factor_b),
    ]).expect("Failed to create DataFrame");

    Dataset::new(df)
}

fn anova_one_way_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ANOVA_OneWay");

    // Vary total sample size (groups × n_per_group)
    for (n_per_group, n_groups) in [(20, 3), (100, 5), (500, 10)] {
        let dataset = generate_anova_data(n_per_group, n_groups, 42);
        let total_n = n_per_group * n_groups;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{}g{}", total_n, n_groups)),
            &dataset,
            |b, data| {
                b.iter(|| run_one_way_anova(data, "y", "group"));
            },
        );
    }

    group.finish();
}

fn anova_two_way_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ANOVA_TwoWay");

    // 2x2 factorial with varying cell sizes
    for n_per_cell in [10, 50, 100] {
        let dataset = generate_two_way_anova_data(n_per_cell, 2, 2, 42);
        let total_n = n_per_cell * 4;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{}_2x2", total_n)),
            &dataset,
            |b, data| {
                b.iter(|| run_two_way_anova(data, "y", "factor_a", "factor_b", true));
            },
        );
    }

    // 3x4 factorial
    let dataset = generate_two_way_anova_data(20, 3, 4, 42);
    group.bench_with_input(
        BenchmarkId::from_parameter("n240_3x4"),
        &dataset,
        |b, data| {
            b.iter(|| run_two_way_anova(data, "y", "factor_a", "factor_b", true));
        },
    );

    group.finish();
}

/// Generate synthetic t-test data
fn generate_ttest_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Two groups with different means
    let x: Vec<f64> = (0..n)
        .map(|_| 5.0 + rng.gen_range(-1.0..1.0))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|_| 6.0 + rng.gen_range(-1.0..1.0) * 1.2)  // Different mean and variance
        .collect();

    (x, y)
}

fn ttest_one_sample_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("TTest_OneSample");

    for n in [100, 1000, 10000, 100000] {
        let (x, _) = generate_ttest_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| one_sample_t_test(&x, 5.0, Alternative::TwoSided, 0.95));
        });
    }

    group.finish();
}

fn ttest_two_sample_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("TTest_TwoSample");

    for n in [100, 1000, 10000, 100000] {
        let (x, y) = generate_ttest_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| two_sample_t_test(&x, &y, 0.0, Alternative::TwoSided, false, 0.95));
        });
    }

    group.finish();
}

/// Generate synthetic time series data for ACF benchmarks
fn generate_time_series(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate AR(1) process with phi = 0.7
    let mut x = vec![0.0; n];
    x[0] = rng.gen_range(-1.0..1.0);
    for t in 1..n {
        x[t] = 0.7 * x[t - 1] + rng.gen_range(-0.5..0.5);
    }
    x
}

fn acf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ACF");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_time_series(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| acf(&x, None, AcfType::Correlation, true, false));
        });
    }

    group.finish();
}

fn pacf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("PACF");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_time_series(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| pacf(&x, None));
        });
    }

    group.finish();
}

fn ccf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("CCF");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_time_series(n, 42);
        let y = generate_time_series(n, 123);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| ccf(&x, &y, None, p2a_core::CcfType::Correlation));
        });
    }

    group.finish();
}

/// Generate synthetic categorical data for chi-squared benchmarks
fn generate_categorical_data(n_categories: usize, total_count: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate random counts for each category
    let mut counts: Vec<f64> = (0..n_categories)
        .map(|_| rng.gen_range(1.0..100.0))
        .collect();

    // Scale to total_count
    let sum: f64 = counts.iter().sum();
    counts.iter_mut().for_each(|c| *c = (*c / sum) * total_count as f64);

    counts
}

/// Generate synthetic contingency table for independence test
fn generate_contingency_table(n_rows: usize, n_cols: usize, total_count: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate random counts for each cell
    let mut table: Vec<Vec<f64>> = (0..n_rows)
        .map(|_| {
            (0..n_cols)
                .map(|_| rng.gen_range(1.0..100.0))
                .collect()
        })
        .collect();

    // Scale to total_count
    let sum: f64 = table.iter().flat_map(|r| r.iter()).sum();
    table.iter_mut().for_each(|row| {
        row.iter_mut().for_each(|c| *c = (*c / sum) * total_count as f64);
    });

    table
}

fn chisq_gof_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ChiSquared_GOF");

    // Benchmark varying number of categories
    for n_categories in [5, 10, 20, 50, 100] {
        let observed = generate_categorical_data(n_categories, 10000, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("k{}", n_categories)),
            &n_categories,
            |b, _| {
                b.iter(|| chisq_test_gof(&observed, None, false));
            },
        );
    }

    group.finish();
}

fn chisq_independence_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ChiSquared_Independence");

    // Benchmark varying table sizes
    for (n_rows, n_cols) in [(2, 2), (3, 3), (5, 5), (10, 10), (20, 20)] {
        let table = generate_contingency_table(n_rows, n_cols, 10000, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n_rows, n_cols)),
            &(n_rows, n_cols),
            |b, _| {
                b.iter(|| chisq_test_independence(&table, false));
            },
        );
    }

    group.finish();
}

fn chisq_yates_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ChiSquared_Yates");

    // 2x2 table with and without Yates' correction
    let table = generate_contingency_table(2, 2, 100, 42);

    group.bench_function("without_yates", |b| {
        b.iter(|| chisq_test_independence(&table, false));
    });

    group.bench_function("with_yates", |b| {
        b.iter(|| chisq_test_independence(&table, true));
    });

    group.finish();
}

/// Generate synthetic 2×2 table for Fisher's exact test
fn generate_2x2_table(total: usize, imbalance: f64, seed: u64) -> [[f64; 2]; 2] {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate counts with some imbalance for interesting odds ratios
    let base = total as f64 / 4.0;
    let a = (base * (1.0 + imbalance) + rng.gen_range(-5.0..5.0)).max(1.0);
    let b = (base * (1.0 - imbalance * 0.5) + rng.gen_range(-5.0..5.0)).max(1.0);
    let c = (base * (1.0 - imbalance * 0.5) + rng.gen_range(-5.0..5.0)).max(1.0);
    let d = (base * (1.0 + imbalance) + rng.gen_range(-5.0..5.0)).max(1.0);

    [[a, b], [c, d]]
}

fn fisher_exact_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FisherExact");

    // Benchmark with varying table sizes (total counts)
    for total in [20, 100, 500, 1000] {
        let table = generate_2x2_table(total, 0.3, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{}", total)),
            &total,
            |b, _| {
                b.iter(|| fisher_exact_test(&table, FisherAlternative::TwoSided, None));
            },
        );
    }

    group.finish();
}

fn fisher_exact_ci_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FisherExact_CI");

    // Benchmark CI computation (more expensive due to binary search)
    for total in [20, 100, 500] {
        let table = generate_2x2_table(total, 0.3, 42);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{}", total)),
            &total,
            |b, _| {
                b.iter(|| fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)));
            },
        );
    }

    group.finish();
}

fn fisher_exact_alternatives_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FisherExact_Alternatives");

    let table = generate_2x2_table(100, 0.3, 42);

    group.bench_function("two_sided", |b| {
        b.iter(|| fisher_exact_test(&table, FisherAlternative::TwoSided, None));
    });

    group.bench_function("greater", |b| {
        b.iter(|| fisher_exact_test(&table, FisherAlternative::Greater, None));
    });

    group.bench_function("less", |b| {
        b.iter(|| fisher_exact_test(&table, FisherAlternative::Less, None));
    });

    group.finish();
}

// ============================================================================
// NLS Benchmarks
// ============================================================================

/// Generate synthetic exponential decay data for NLS benchmarks
fn generate_exponential_decay_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // True model: y = 10 * exp(-0.5 * x) + 2 + noise
    let x: Array1<f64> = (0..n)
        .map(|i| i as f64 * 5.0 / n as f64)
        .collect();

    let y: Array1<f64> = x.iter()
        .map(|&xi| {
            let noise: f64 = rng.gen_range(-0.2..0.2);
            10.0 * (-0.5 * xi).exp() + 2.0 + noise
        })
        .collect();

    (x, y)
}

/// Generate synthetic Michaelis-Menten data for NLS benchmarks
fn generate_michaelis_menten_data(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // True model: V = 200 * S / (0.1 + S) + noise
    // Generate log-spaced substrate concentrations
    let x: Array1<f64> = (0..n)
        .map(|i| 0.01 * (10.0_f64).powf(i as f64 * 3.0 / n as f64))
        .collect();

    let y: Array1<f64> = x.iter()
        .map(|&xi| {
            let noise: f64 = rng.gen_range(-2.0..2.0);
            200.0 * xi / (0.1 + xi) + noise
        })
        .collect();

    (x, y)
}

fn nls_exponential_decay_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("NLS_ExponentialDecay");

    for n in [10, 50, 100, 500, 1000] {
        let (x, y) = generate_exponential_decay_data(n, 42);
        let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                nls(
                    &x, &y,
                    model_exponential_decay,
                    &start,
                    &["a", "b", "c"],
                    NlsConfig::default()
                )
            });
        });
    }

    group.finish();
}

fn nls_michaelis_menten_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("NLS_MichaelisMenten");

    for n in [8, 20, 50, 100, 500] {
        let (x, y) = generate_michaelis_menten_data(n, 42);
        let start = Array1::from_vec(vec![150.0, 0.05]);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                nls(
                    &x, &y,
                    model_michaelis_menten,
                    &start,
                    &["Vmax", "Km"],
                    NlsConfig::default()
                )
            });
        });
    }

    group.finish();
}

fn nls_algorithm_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("NLS_Algorithm");

    let (x, y) = generate_exponential_decay_data(100, 42);
    let start = Array1::from_vec(vec![8.0, 0.3, 1.0]);

    // Levenberg-Marquardt (default)
    group.bench_function("LevenbergMarquardt", |b| {
        b.iter(|| {
            nls(
                &x, &y,
                model_exponential_decay,
                &start,
                &["a", "b", "c"],
                NlsConfig::default()
            )
        });
    });

    // Gauss-Newton
    group.bench_function("GaussNewton", |b| {
        b.iter(|| {
            nls(
                &x, &y,
                model_exponential_decay,
                &start,
                &["a", "b", "c"],
                NlsConfig {
                    algorithm: NlsAlgorithm::GaussNewton,
                    ..Default::default()
                }
            )
        });
    });

    group.finish();
}

// ============================================================================
// Wilcoxon Test Benchmarks
// ============================================================================

/// Generate two independent samples for rank sum test
fn generate_two_sample_data(n1: usize, n2: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n1).map(|_| rng.gen_range(0.0..1.0) * 10.0 + 5.0).collect();
    let y: Vec<f64> = (0..n2).map(|_| rng.gen_range(0.0..1.0) * 10.0 + 6.0).collect();
    (x, y)
}

/// Generate paired sample data for signed rank test
fn generate_paired_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0) * 30.0 + 100.0).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi + rng.gen_range(-1.0..1.0) * 20.0 - 5.0).collect();
    (x, y)
}

fn wilcoxon_rank_sum_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wilcoxon_RankSum");

    for n in [10, 50, 100, 500, 1000] {
        let (x, y) = generate_two_sample_data(n, n, 42);
        let config = WilcoxonConfig {
            exact: Some(false),
            correct: true,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &config)
            });
        });
    }

    group.finish();
}

fn wilcoxon_signed_rank_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wilcoxon_SignedRank");

    for n in [10, 50, 100, 500, 1000] {
        let (x, y) = generate_paired_data(n, 42);
        let config = WilcoxonConfig {
            exact: Some(false),
            correct: true,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                wilcoxon_signed_rank(&x, Some(&y), 0.0, Alternative::TwoSided, &config)
            });
        });
    }

    group.finish();
}

fn wilcoxon_exact_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Wilcoxon_Exact");

    for n in [5, 10, 15, 20] {
        let (x, y) = generate_two_sample_data(n, n, 42);
        let config = WilcoxonConfig {
            exact: Some(true),
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &config)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Shapiro-Wilk Test Benchmarks
// ============================================================================

/// Generate data for Shapiro-Wilk normality test
fn generate_normal_data_for_shapiro(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect()
}

fn shapiro_wilk_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ShapiroWilk");

    for n in [10, 50, 100, 500, 1000, 2000, 5000] {
        let x = generate_normal_data_for_shapiro(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                shapiro_wilk_test(&x)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Kolmogorov-Smirnov Test Benchmarks
// ============================================================================

/// Generate two samples for KS two-sample test
fn generate_ks_two_sample_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Sample from slightly different distributions
    let x: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0) * 2.0 - 1.0).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0) * 2.0 - 0.9).collect(); // Slightly shifted
    (x, y)
}

/// Generate normal-like data for one-sample KS test
fn generate_ks_normal_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Generate approximately normal data using sum of uniform randoms
    (0..n)
        .map(|_| {
            let sum: f64 = (0..12).map(|_| rng.gen_range(0.0..1.0)).sum();
            sum - 6.0 // Approximately N(0,1)
        })
        .collect()
}

fn ks_two_sample_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("KS_TwoSample");

    for n in [100, 1000, 10000, 100000] {
        let (x, y) = generate_ks_two_sample_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                ks_test_two_sample(&x, &y, Alternative::TwoSided)
            });
        });
    }

    group.finish();
}

fn ks_one_sample_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("KS_OneSample");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_ks_normal_data(n, 42);
        let dist = TheoreticalDistribution::Normal;

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                ks_test_one_sample(&x, dist, Alternative::TwoSided)
            });
        });
    }

    group.finish();
}

// ============================================================================
// MANOVA Benchmarks
// ============================================================================

/// Generate multivariate data with group structure for MANOVA
fn generate_manova_data(n_per_group: usize, n_groups: usize, n_vars: usize, seed: u64) -> (ndarray::Array2<f64>, Vec<String>) {
    use ndarray::Array2;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_total = n_per_group * n_groups;

    let mut y = Array2::<f64>::zeros((n_total, n_vars));
    let mut groups = Vec::with_capacity(n_total);

    for g in 0..n_groups {
        let group_label = format!("G{}", g + 1);
        for i in 0..n_per_group {
            let row = g * n_per_group + i;
            groups.push(group_label.clone());

            // Add group effect and noise
            for v in 0..n_vars {
                let group_effect = (g as f64) * 2.0 + (v as f64) * 0.5;
                let noise = rng.gen_range(-1.0..1.0);
                // Add independent variation in each variable
                let var_noise = rng.gen_range(-0.5..0.5);
                y[[row, v]] = group_effect + noise + var_noise * (v as f64 + 1.0);
            }
        }
    }

    (y, groups)
}

fn manova_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MANOVA");

    // Different configurations: (n_per_group, n_groups, n_vars)
    let configs = [
        (33, 3, 2),    // n=100, p=2, g=3
        (250, 4, 3),   // n=1000, p=3, g=4
        (2000, 5, 5),  // n=10000, p=5, g=5
    ];

    for (n_per_group, n_groups, n_vars) in configs {
        let (y, groups) = generate_manova_data(n_per_group, n_groups, n_vars, 42);
        let label = format!("n{}_p{}_g{}", n_per_group * n_groups, n_vars, n_groups);

        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| {
                manova_one_way(&y, &groups)
            });
        });
    }

    group.finish();
}

/// Generate grouped data for Tukey HSD benchmarks
fn generate_tukey_data(n_per_group: usize, n_groups: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_total = n_per_group * n_groups;

    let mut y = Vec::with_capacity(n_total);
    let mut group = Vec::with_capacity(n_total);

    for g in 0..n_groups {
        let group_effect = (g as f64) * 2.0; // 2 units shift per group
        for _ in 0..n_per_group {
            let noise: f64 = rng.gen_range(0.0..1.0) - 0.5;
            y.push(group_effect + noise);
            group.push(format!("G{}", g + 1));
        }
    }

    let df = df! {
        "y" => y,
        "group" => group
    }.unwrap();

    Dataset::new(df)
}

fn tukey_hsd_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("TukeyHSD");

    // Different configurations: (n_per_group, n_groups)
    let configs = [
        (9, 3),     // n=27, k=3
        (20, 5),    // n=100, k=5
        (100, 10),  // n=1000, k=10
    ];

    for (n_per_group, n_groups) in configs {
        let dataset = generate_tukey_data(n_per_group, n_groups, 42);
        let label = format!("n{}_k{}", n_per_group * n_groups, n_groups);

        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| {
                run_tukey_hsd(&dataset, "y", "group", 0.95)
            });
        });
    }

    group.finish();
}

/// Generate grouped data for Bartlett test benchmarks
/// Creates groups with different variances to test the algorithm
fn generate_bartlett_data(n_per_group: usize, n_groups: usize, seed: u64) -> Vec<(String, Vec<f64>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut groups = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        // Each group has a different variance: var ~ (g+1)^2
        let group_std = (g + 1) as f64;
        let mut group_vals = Vec::with_capacity(n_per_group);
        for _ in 0..n_per_group {
            let base = 10.0 + (g as f64) * 5.0;
            let noise: f64 = (rng.gen_range(0.0..1.0) - 0.5) * 2.0 * group_std;
            group_vals.push(base + noise);
        }
        groups.push((format!("G{}", g + 1), group_vals));
    }

    groups
}

fn bartlett_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("BartlettTest");

    // Different configurations: (n_per_group, n_groups)
    let configs = [
        (5, 3),      // n=15, k=3
        (20, 5),     // n=100, k=5
        (100, 10),   // n=1000, k=10
        (500, 20),   // n=10000, k=20
    ];

    for (n_per_group, n_groups) in configs {
        let data = generate_bartlett_data(n_per_group, n_groups, 42);
        let label = format!("n{}_k{}", n_per_group * n_groups, n_groups);

        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| {
                bartlett_test(&data)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Spectral Density Estimation Benchmarks
// ============================================================================

/// Generate time series data with known spectral properties for spectrum benchmarks
fn generate_spectrum_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate a series with two frequency components + noise
    // f1 = 0.1 (period 10), f2 = 0.25 (period 4)
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            let signal1 = (2.0 * std::f64::consts::PI * 0.1 * t).sin();
            let signal2 = 0.5 * (2.0 * std::f64::consts::PI * 0.25 * t).sin();
            let noise: f64 = rng.gen_range(-0.3..0.3);
            signal1 + signal2 + noise
        })
        .collect();
    x
}

fn spectrum_periodogram_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spectrum_Periodogram");

    // FFT-based: O(n log n), can handle large datasets
    for n in [100, 1000, 10000, 100000] {
        let x = generate_spectrum_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                spectrum(&x, SpectrumConfig::default())
            });
        });
    }

    group.finish();
}

fn spectrum_smoothed_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spectrum_Smoothed");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_spectrum_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                spectrum(&x, SpectrumConfig {
                    spans: Some(vec![3, 3]),
                    taper: 0.1,
                    detrend: true,
                    demean: true,
                    pad_ratio: 1.0,
                })
            });
        });
    }

    group.finish();
}

fn spectrum_ar_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spectrum_AR");

    for n in [100, 1000, 10000] {
        let x = generate_spectrum_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                spectrum_ar(&x, None, None)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Box-Pierce and Ljung-Box Test Benchmarks
// ============================================================================

fn generate_boxtest_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn box_test_ljung_box_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("BoxTest_LjungBox");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_boxtest_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                box_test(&x, Some(10), BoxTestType::LjungBox, 0)
            });
        });
    }

    group.finish();
}

fn box_test_box_pierce_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("BoxTest_BoxPierce");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_boxtest_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                box_test(&x, Some(10), BoxTestType::BoxPierce, 0)
            });
        });
    }

    group.finish();
}

/// Generate data for Phillips-Perron test benchmark (random walk)
// ============================================================================
// LOESS Benchmarks
// ============================================================================

/// Generate smooth data with noise for LOESS
fn generate_loess_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 4.0 * std::f64::consts::PI).collect();
    let y: Vec<f64> = x.iter()
        .map(|&xi| xi.sin() + 0.3 * xi + rng.gen_range(-0.5..0.5))
        .collect();
    (x, y)
}

fn loess_gaussian_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LOESS_Gaussian");

    for n in [100, 1000, 10000] {
        let (x, y) = generate_loess_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                loess(&x, &y, LoessConfig {
                    span: 0.5,
                    degree: 2,
                    robust: false,
                    ..Default::default()
                })
            });
        });
    }

    group.finish();
}

fn loess_robust_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LOESS_Robust");

    for n in [100, 1000, 5000] {
        let (x, y) = generate_loess_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                loess(&x, &y, LoessConfig {
                    span: 0.5,
                    degree: 2,
                    robust: true,
                    ..Default::default()
                })
            });
        });
    }

    group.finish();
}

fn loess_span_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LOESS_SpanComparison");

    let (x, y) = generate_loess_data(1000, 42);

    for span in [0.3, 0.5, 0.75, 0.9] {
        let span_str = format!("{:.2}", span);
        group.bench_with_input(BenchmarkId::from_parameter(&span_str), &span, |b, &s| {
            b.iter(|| {
                loess(&x, &y, LoessConfig {
                    span: s,
                    degree: 2,
                    robust: false,
                    ..Default::default()
                })
            });
        });
    }

    group.finish();
}

// ============================================================================
// PP Test Benchmarks
// ============================================================================

fn generate_pptest_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = vec![0.0f64; n];
    for i in 1..n {
        x[i] = x[i - 1] + rng.gen_range(-1.0..1.0);
    }
    x
}

fn pp_test_lshort_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("PPTest_lshort");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_pptest_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                pp_test(&x, true)
            });
        });
    }

    group.finish();
}

fn pp_test_llong_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("PPTest_llong");

    for n in [100, 1000, 10000, 100000] {
        let x = generate_pptest_data(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                pp_test(&x, false)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    ols_standard_benchmark,
    ols_robust_se_benchmark,
    diagnostics_benchmark,
    anova_one_way_benchmark,
    anova_two_way_benchmark,
    ttest_one_sample_benchmark,
    ttest_two_sample_benchmark,
    acf_benchmark,
    pacf_benchmark,
    ccf_benchmark,
    chisq_gof_benchmark,
    chisq_independence_benchmark,
    chisq_yates_benchmark,
    fisher_exact_benchmark,
    fisher_exact_ci_benchmark,
    fisher_exact_alternatives_benchmark,
    nls_exponential_decay_benchmark,
    nls_michaelis_menten_benchmark,
    nls_algorithm_comparison_benchmark,
    loess_gaussian_benchmark,
    loess_robust_benchmark,
    loess_span_comparison_benchmark,
    wilcoxon_rank_sum_benchmark,
    wilcoxon_signed_rank_benchmark,
    wilcoxon_exact_benchmark,
    shapiro_wilk_benchmark,
    ks_two_sample_benchmark,
    ks_one_sample_benchmark,
    manova_benchmark,
    tukey_hsd_benchmark,
    bartlett_test_benchmark,
    spectrum_periodogram_benchmark,
    spectrum_smoothed_benchmark,
    spectrum_ar_benchmark,
    box_test_ljung_box_benchmark,
    box_test_box_pierce_benchmark,
    pp_test_lshort_benchmark,
    pp_test_llong_benchmark
);
criterion_main!(benches);
