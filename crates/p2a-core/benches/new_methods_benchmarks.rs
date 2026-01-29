//! Benchmarks for newly implemented methods
//!
//! Run with: `cargo bench -p p2a-core -- new_methods`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use p2a_core::linalg::design::DesignMatrix;
use p2a_core::{
    Alternative,
    ApproxMethod,
    ApproxRule,
    BgTestType,
    CorrelationMethod,
    DensityKernel,
    GmmConfig,
    GmmStep,
    GmmTransform,
    GsynthConfig,
    GsynthForce,
    HacKernel,
    HurdleType,
    Linkage,
    MixedLogitConfig,
    PanelGlsModel,
    PowerAlternative,
    RandomDistribution,
    RandomParameterSpec,
    ResetType,
    SmoothSplineConfig,
    SplineMethod,
    TTestType,
    approx,
    cmdscale,
    // cor.test
    cor_test,
    cutree,
    data::Dataset,
    density,
    ecdf,
    // robust stats
    fivenum,
    forecasting::cpgram::{cpgram, white_noise_test},
    granger_test,
    hierarchical,
    iqr,
    isoreg,
    // batch 2: toeplitz, line, cpgram, supsmu, constrOptim, ppr, se.contrast, model.tables
    linalg::toeplitz::{toeplitz, toeplitz_asymmetric},
    loglin,
    mad,
    // batch 1: medpolish, cmdscale, cutree, isoreg, loglin
    medpolish,
    ml::ppr::{PprConfig, ppr},
    power_anova_test,
    power_prop_test,
    // power analysis
    power_t_test,
    // prop.trend.test
    prop_trend_test,
    regression::CovarianceType,
    regression::{line, supsmu},
    reset_test_from_ols,
    // Extended TWFE
    run_etwfe,
    // Panel data methods
    run_gmm,
    // Generalized synthetic control
    run_gsynth,
    // Hurdle models
    run_hurdle,
    // Mixed logit (random parameters logit)
    run_mixed_logit,
    // McFadden conditional logit
    run_mlogit,
    // Discrete choice models (compare with R's nnet, MASS, pscl packages)
    run_multinom,
    run_negbin,
    run_ols,
    run_ordered_logit,
    run_panel_gls,
    run_zinb,
    run_zip,
    // smooth.spline
    smooth_spline,
    // spline/approx
    spline,
    stats::{
        constroptim::{ConstrOptimConfig, OptimMethod, constr_optim},
        modeltables::{TableType, model_tables, model_tables_two_way},
        run_one_way_anova,
        secontrast::{ContrastType, generate_contrasts, se_contrast},
    },
    vcov_hac,
    wald_test_from_ols,
};
use polars::prelude::*;
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
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi * 0.7 + rng.r#gen::<f64>() * 3.0)
        .collect();
    (x, y)
}

fn cor_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cor_test");

    for size in [100, 1000, 10000].iter() {
        let (x, y) = generate_correlated_pair(*size, 42);

        group.bench_with_input(BenchmarkId::new("pearson", size), size, |b, _| {
            b.iter(|| {
                cor_test(
                    &x,
                    &y,
                    CorrelationMethod::Pearson,
                    Alternative::TwoSided,
                    0.95,
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("spearman", size), size, |b, _| {
            b.iter(|| {
                cor_test(
                    &x,
                    &y,
                    CorrelationMethod::Spearman,
                    Alternative::TwoSided,
                    0.95,
                )
            })
        });
    }
    group.finish();
}

fn power_analysis_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_analysis");

    group.bench_function("power_t_test", |b| {
        b.iter(|| {
            power_t_test(
                Some(30.0),
                Some(0.5),
                Some(1.0),
                Some(0.05),
                None,
                TTestType::TwoSample,
                PowerAlternative::TwoSided,
            )
        })
    });

    group.bench_function("power_prop_test", |b| {
        b.iter(|| {
            power_prop_test(
                Some(100.0),
                0.5,
                Some(0.6),
                Some(0.05),
                None,
                PowerAlternative::TwoSided,
            )
        })
    });

    group.bench_function("power_anova_test", |b| {
        b.iter(|| power_anova_test(4, Some(20.0), 1.0, 3.0, Some(0.05), None))
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
            b.iter(|| {
                approx(
                    &x,
                    &y,
                    None,
                    Some(100),
                    ApproxMethod::Linear,
                    ApproxRule::Na,
                    0.5,
                )
            })
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

fn medpolish_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("medpolish");

    for nrow in [10, 20, 50].iter() {
        let ncol = *nrow;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data: Vec<Vec<f64>> = (0..*nrow)
            .map(|_| (0..ncol).map(|_| rng.r#gen::<f64>() * 100.0).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", nrow, ncol)),
            nrow,
            |b, _| b.iter(|| medpolish(&data, None, None, false)),
        );
    }
    group.finish();
}

fn isoreg_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("isoreg");

    for size in [100, 1000, 10000].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| xi * 0.5 + rng.r#gen::<f64>() * 10.0)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| isoreg(&x, &y))
        });
    }
    group.finish();
}

fn loglin_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("loglin");

    // 2x2 table
    group.bench_function("2x2", |b| {
        let table = vec![10.0, 15.0, 20.0, 25.0];
        let dims = vec![2, 2];
        let margins = vec![vec![0], vec![1]]; // independence model
        b.iter(|| loglin(&table, &dims, &margins, None, None))
    });

    // 2x3 table
    group.bench_function("2x3", |b| {
        let table = vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0];
        let dims = vec![2, 3];
        let margins = vec![vec![0], vec![1]];
        b.iter(|| loglin(&table, &dims, &margins, None, None))
    });

    // 2x2x2 table
    group.bench_function("2x2x2", |b| {
        let table = vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0];
        let dims = vec![2, 2, 2];
        let margins = vec![vec![0, 1], vec![0, 2], vec![1, 2]];
        b.iter(|| loglin(&table, &dims, &margins, None, None))
    });

    group.finish();
}

fn cmdscale_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmdscale");

    for size in [20, 50, 100].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // Generate random distance matrix
        let mut d: Vec<Vec<f64>> = vec![vec![0.0; *size]; *size];
        for i in 0..*size {
            for j in (i + 1)..*size {
                let dist = rng.r#gen::<f64>() * 10.0;
                d[i][j] = dist;
                d[j][i] = dist;
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| cmdscale(&d, Some(2), Some(true), Some(false)))
        });
    }
    group.finish();
}

fn cutree_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cutree");

    for size in [50, 100, 200].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data_vec: Vec<f64> = (0..(*size * 2))
            .map(|_| rng.r#gen::<f64>() * 10.0)
            .collect();
        let data = Array2::from_shape_vec((*size, 2), data_vec).unwrap();
        let hc = hierarchical(data.view(), None, Linkage::Complete, None).unwrap();

        group.bench_with_input(BenchmarkId::new("k5", size), size, |b, _| {
            b.iter(|| cutree(&hc, Some(5), None))
        });
    }
    group.finish();
}

// ============================================================================
// Batch 2 Benchmarks: toeplitz, line, cpgram, supsmu, constrOptim, ppr,
//                     se.contrast, model.tables
// ============================================================================

fn toeplitz_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("toeplitz");

    for size in [10, 50, 100, 500].iter() {
        let x: Vec<f64> = (0..*size).map(|i| 1.0 / (i + 1) as f64).collect();

        group.bench_with_input(BenchmarkId::new("symmetric", size), size, |b, _| {
            b.iter(|| toeplitz(&x))
        });

        let col: Vec<f64> = (0..*size).map(|i| (i + 1) as f64).collect();
        let row: Vec<f64> = (0..*size).map(|i| -(i as f64)).collect();

        group.bench_with_input(BenchmarkId::new("asymmetric", size), size, |b, _| {
            b.iter(|| toeplitz_asymmetric(&col, &row))
        });
    }
    group.finish();
}

fn line_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("line");

    for size in [20, 100, 500, 1000].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 * xi + 5.0 + rng.r#gen::<f64>() * 10.0)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| line::line(&x, &y, None))
        });
    }
    group.finish();
}

fn cpgram_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpgram");

    for size in [64, 256, 1024, 4096].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x: Vec<f64> = (0..*size).map(|_| rng.r#gen::<f64>()).collect();

        group.bench_with_input(BenchmarkId::new("cpgram", size), size, |b, _| {
            b.iter(|| cpgram(&x, None))
        });

        group.bench_with_input(BenchmarkId::new("white_noise_test", size), size, |b, _| {
            b.iter(|| white_noise_test(&x, None))
        });
    }
    group.finish();
}

fn supsmu_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("supsmu");

    for size in [50, 200, 1000].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x: Vec<f64> = (0..*size).map(|i| i as f64 / *size as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (xi * 6.28).sin() + rng.r#gen::<f64>() * 0.5)
            .collect();

        group.bench_with_input(BenchmarkId::new("auto_span", size), size, |b, _| {
            b.iter(|| supsmu::supsmu(&x, &y, None, None, false, 0.0))
        });

        group.bench_with_input(BenchmarkId::new("fixed_span", size), size, |b, _| {
            b.iter(|| supsmu::supsmu(&x, &y, None, Some(0.1), false, 0.0))
        });
    }
    group.finish();
}

fn constr_optim_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("constr_optim");

    // Simple quadratic: minimize (x-2)^2 + (y-3)^2 subject to x + y >= 1
    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
    let grad = |x: &[f64]| vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)];

    group.bench_function("2d_quadratic_nelder_mead", |b| {
        let ui = vec![vec![1.0, 1.0]]; // x + y >= 1
        let ci = vec![1.0];
        let config = ConstrOptimConfig {
            method: OptimMethod::NelderMead,
            ..Default::default()
        };
        b.iter(|| {
            constr_optim(
                &[0.0, 0.0],
                f,
                None::<fn(&[f64]) -> Vec<f64>>,
                &ui,
                &ci,
                config.clone(),
            )
        })
    });

    group.bench_function("2d_quadratic_bfgs", |b| {
        let ui = vec![vec![1.0, 1.0]];
        let ci = vec![1.0];
        let config = ConstrOptimConfig {
            method: OptimMethod::BFGS,
            ..Default::default()
        };
        b.iter(|| constr_optim(&[0.0, 0.0], f, Some(&grad), &ui, &ci, config.clone()))
    });

    // Higher dimensional problem
    group.bench_function("10d_quadratic", |b| {
        let f10 = |x: &[f64]| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| (xi - i as f64).powi(2))
                .sum::<f64>()
        };
        let ui: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let mut row = vec![0.0; 10];
                row[i] = 1.0;
                row
            })
            .collect();
        let ci = vec![0.0; 10]; // x_i >= 0
        let config = ConstrOptimConfig::default();
        b.iter(|| {
            constr_optim(
                &[5.0; 10],
                f10,
                None::<fn(&[f64]) -> Vec<f64>>,
                &ui,
                &ci,
                config.clone(),
            )
        })
    });

    group.finish();
}

fn ppr_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppr");

    for (n, p) in [(100, 5), (500, 10), (1000, 5)].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Generate X matrix
        let x_data: Vec<f64> = (0..(*n * *p)).map(|_| rng.r#gen::<f64>()).collect();
        let x = Array2::from_shape_vec((*n, *p), x_data.clone()).unwrap();

        // Generate y = sum of first 2 projections + noise
        let y: Vec<f64> = (0..*n)
            .map(|i| {
                let row_start = i * *p;
                let proj1: f64 = (0..*p)
                    .map(|j| x_data[row_start + j] * (j as f64 + 1.0) / *p as f64)
                    .sum();
                proj1.sin() + rng.r#gen::<f64>() * 0.1
            })
            .collect();

        let label = format!("n{}_p{}", n, p);
        group.bench_with_input(BenchmarkId::new("nterms1", &label), &label, |b, _| {
            let config = PprConfig {
                nterms: 1,
                ..Default::default()
            };
            b.iter(|| ppr(x.view(), &y, None, config.clone()))
        });

        group.bench_with_input(BenchmarkId::new("nterms3", &label), &label, |b, _| {
            let config = PprConfig {
                nterms: 3,
                ..Default::default()
            };
            b.iter(|| ppr(x.view(), &y, None, config.clone()))
        });
    }
    group.finish();
}

fn se_contrast_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("se_contrast");

    for n_per_group in [10, 50, 100].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let k = 4; // 4 groups
        let n = n_per_group * k;

        // Generate data
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let group = i / n_per_group;
                (group as f64 * 5.0) + rng.r#gen::<f64>() * 2.0
            })
            .collect();
        let groups: Vec<String> = (0..n).map(|i| format!("G{}", i / n_per_group)).collect();

        let df = df! {
            "value" => &values,
            "group" => &groups
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let label = format!("k{}_n{}", k, n_per_group);
        group.bench_with_input(
            BenchmarkId::new("treatment_contrasts", &label),
            &label,
            |b, _| {
                let anova = run_one_way_anova(&dataset, "value", "group").unwrap();
                let contrasts = generate_contrasts(k, ContrastType::Treatment);
                b.iter(|| se_contrast(&anova, &contrasts))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("helmert_contrasts", &label),
            &label,
            |b, _| {
                let anova = run_one_way_anova(&dataset, "value", "group").unwrap();
                let contrasts = generate_contrasts(k, ContrastType::Helmert);
                b.iter(|| se_contrast(&anova, &contrasts))
            },
        );
    }
    group.finish();
}

fn model_tables_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_tables");

    for n_per_group in [10, 50, 100].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let k = 5; // 5 groups
        let n = n_per_group * k;

        let values: Vec<f64> = (0..n)
            .map(|i| {
                let grp = i / n_per_group;
                (grp as f64 * 3.0) + rng.r#gen::<f64>() * 2.0
            })
            .collect();
        let groups: Vec<String> = (0..n).map(|i| format!("G{}", i / n_per_group)).collect();

        let df = df! {
            "value" => &values,
            "group" => &groups
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let label = format!("k{}_n{}", k, n_per_group);
        group.bench_with_input(BenchmarkId::new("means", &label), &label, |b, _| {
            let anova = run_one_way_anova(&dataset, "value", "group").unwrap();
            b.iter(|| model_tables(&anova, TableType::Means, true))
        });

        group.bench_with_input(BenchmarkId::new("effects", &label), &label, |b, _| {
            let anova = run_one_way_anova(&dataset, "value", "group").unwrap();
            b.iter(|| model_tables(&anova, TableType::Effects, true))
        });
    }

    // Two-way model tables
    for size in [3, 5, 10].iter() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data: Vec<Vec<Vec<f64>>> = (0..*size)
            .map(|i| {
                (0..*size)
                    .map(|j| {
                        (0..5)
                            .map(|_| (i + j) as f64 * 2.0 + rng.r#gen::<f64>())
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let factor_a: Vec<String> = (0..*size).map(|i| format!("A{}", i)).collect();
        let factor_b: Vec<String> = (0..*size).map(|i| format!("B{}", i)).collect();

        let label = format!("{}x{}", size, size);
        group.bench_with_input(BenchmarkId::new("two_way", &label), &label, |b, _| {
            b.iter(|| model_tables_two_way(&data, &factor_a, &factor_b, TableType::Means, true))
        });
    }

    group.finish();
}

// ============================================================================
// Batch 3 Benchmarks: Econometric tests and discrete choice models
// For comparison with R's lmtest, sandwich, nnet, MASS, pscl packages
// Sample sizes: 100, 500, 1000, 5000 (matching R benchmark)
// ============================================================================

fn create_regression_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // AR(1) errors for serial correlation
    let mut e = vec![0.0; n];
    e[0] = rng.r#gen::<f64>() - 0.5;
    for i in 1..n {
        e[i] = 0.5 * e[i - 1] + (rng.r#gen::<f64>() - 0.5);
    }
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(e.iter())
        .map(|(&xi, &ei)| 2.0 + 0.5 * xi + ei)
        .collect();

    let df = df! {
        "y" => &y,
        "x" => &x
    }
    .unwrap();
    Dataset::new(df)
}

fn create_multivar_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x1: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 10.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 10.0).collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&xi1, &xi2)| 1.0 + 2.0 * xi1 + 0.5 * xi2 + (rng.r#gen::<f64>() - 0.5) * 2.0)
        .collect();

    let df = df! {
        "y" => &y,
        "x1" => &x1,
        "x2" => &x2
    }
    .unwrap();
    Dataset::new(df)
}

fn create_multinomial_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 10.0 - 5.0).collect();

    // Generate categorical outcome based on x
    let y: Vec<&str> = x
        .iter()
        .map(|&xi| {
            let p_a = 1.0 / (1.0 + (0.5 + 1.0 * xi).exp() + (1.0 + 2.0 * xi).exp());
            let p_b =
                (0.5 + 1.0 * xi).exp() / (1.0 + (0.5 + 1.0 * xi).exp() + (1.0 + 2.0 * xi).exp());
            let u: f64 = rng.r#gen();
            if u < p_a {
                "A"
            } else if u < p_a + p_b {
                "B"
            } else {
                "C"
            }
        })
        .collect();

    let df = df! {
        "y" => &y,
        "x" => &x
    }
    .unwrap();
    Dataset::new(df)
}

fn create_ordered_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 6.0 - 3.0).collect();

    // Latent variable model with logistic errors
    let y: Vec<&str> = x
        .iter()
        .map(|&xi| {
            let latent = 1.5 * xi + (rng.r#gen::<f64>().ln() - (1.0 - rng.r#gen::<f64>()).ln()); // logistic noise
            if latent < -1.0 {
                "Low"
            } else if latent < 1.0 {
                "Med"
            } else {
                "High"
            }
        })
        .collect();

    let df = df! {
        "y" => &y,
        "x" => &x
    }
    .unwrap();
    Dataset::new(df)
}

fn create_count_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 3.0).collect();

    // Negative binomial-like counts
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (0.5 + 0.8 * xi).exp();
            // Simple overdispersed count simulation

            (mu + rng.r#gen::<f64>() * mu.sqrt() * 2.0).max(0.0).floor()
        })
        .collect();

    let df = df! {
        "y" => &y,
        "x" => &x
    }
    .unwrap();
    Dataset::new(df)
}

fn create_zeroinfl_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 7.0).collect();

    // Zero-inflated counts
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let zero_prob = 1.0 / (1.0 + (xi - 2.0).exp()); // Higher prob of zero for low x
            if rng.r#gen::<f64>() < zero_prob {
                0.0
            } else {
                let mu = (0.5 + 0.5 * xi).exp();
                (mu + rng.r#gen::<f64>() * mu.sqrt()).max(0.0).floor()
            }
        })
        .collect();

    let df = df! {
        "y" => &y,
        "x" => &x
    }
    .unwrap();
    Dataset::new(df)
}

fn create_granger_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Random walk x
    let mut x = vec![0.0; n];
    for i in 1..n {
        x[i] = x[i - 1] + (rng.r#gen::<f64>() - 0.5);
    }
    // y depends on lagged x
    let mut y = vec![0.0; n];
    y[0] = rng.r#gen::<f64>() - 0.5;
    for i in 1..n {
        y[i] = 0.3 * x[i - 1] + 0.5 * y[i - 1] + (rng.r#gen::<f64>() - 0.5);
    }

    let df = df! {
        "y" => &y,
        "x" => &x
    }
    .unwrap();
    Dataset::new(df)
}

// Breusch-Godfrey test benchmark
// Use _from_ols variant for fair comparison with R (which pre-computes lm())
fn bg_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bgtest");

    for &n in &[100, 500, 1000, 5000] {
        let dataset = create_regression_dataset(n, 123);
        // Pre-compute OLS and design matrix (like R's model <- lm(y ~ x))
        let ols_result = run_ols(&dataset, "y", &["x"], true, CovarianceType::Standard).unwrap();
        let df = dataset.df();
        let design = DesignMatrix::from_dataframe(df, &["x"], true).unwrap();
        let x_matrix = design.data;

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                p2a_core::bg_test_from_ols(&ols_result, &x_matrix.view(), 1, BgTestType::Chisq, 0.0)
            })
        });
    }
    group.finish();
}

// RESET test benchmark
// Use _from_ols variant for fair comparison with R (which pre-computes lm())
fn reset_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("resettest");

    for &n in &[100, 500, 1000, 5000] {
        let dataset = create_regression_dataset(n, 456);
        // Pre-compute OLS and design matrix
        let ols_result = run_ols(&dataset, "y", &["x"], true, CovarianceType::Standard).unwrap();
        let df = dataset.df();
        let design = DesignMatrix::from_dataframe(df, &["x"], true).unwrap();
        let x_matrix = design.data;
        let y_vec = DesignMatrix::extract_column(df, "y").unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                reset_test_from_ols(
                    &ols_result,
                    &x_matrix.view(),
                    &y_vec,
                    &[2, 3],
                    ResetType::Fitted,
                )
            })
        });
    }
    group.finish();
}

// Wald test benchmark
// Use _from_ols variant for fair comparison with R (which pre-computes lm())
fn wald_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("waldtest");

    for &n in &[100, 500, 1000, 5000] {
        let dataset = create_multivar_dataset(n, 789);
        // Pre-compute both OLS models
        let ols_unrestricted =
            run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::Standard).unwrap();
        let ols_restricted =
            run_ols(&dataset, "y", &["x1"], true, CovarianceType::Standard).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| wald_test_from_ols(&ols_unrestricted, &ols_restricted, true))
        });
    }
    group.finish();
}

// HAC (Newey-West) standard errors benchmark
fn vcov_hac_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vcovHAC");

    for &n in &[100, 500, 1000, 5000] {
        let dataset = create_regression_dataset(n, 101);
        let ols_result = run_ols(&dataset, "y", &["x"], true, CovarianceType::Standard).unwrap();

        // Build X matrix
        let df = dataset.df();
        let design = DesignMatrix::from_dataframe(df, &["x"], true).unwrap();
        let x_matrix = design.data.clone();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| vcov_hac(&ols_result, &x_matrix, None, HacKernel::Bartlett, false))
        });
    }
    group.finish();
}

// Granger causality test benchmark
fn granger_test_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("grangertest");

    for &n in &[100, 500, 1000, 5000] {
        let dataset = create_granger_dataset(n, 202);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| granger_test(&dataset, "y", "x", 2))
        });
    }
    group.finish();
}

// Multinomial logit benchmark
fn multinom_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("multinom");

    for &n in &[100, 500, 1000, 5000] {
        let dataset = create_multinomial_dataset(n, 404);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| run_multinom(&dataset, "y", &["x"], None))
        });
    }
    group.finish();
}

// Ordered logit benchmark
fn ordered_logit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("polr");

    for &n in &[100, 500, 1000, 5000] {
        let dataset = create_ordered_dataset(n, 505);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| run_ordered_logit(&dataset, "y", &["x"]))
        });
    }
    group.finish();
}

// Negative binomial regression benchmark
fn negbin_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("glm_nb");

    for &n in &[100, 500, 1000, 5000] {
        let dataset = create_count_dataset(n, 606);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| run_negbin(&dataset, "y", &["x"], None))
        });
    }
    group.finish();
}

// Zero-inflated Poisson benchmark
fn zip_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip");

    for &n in &[100, 500, 1000] {
        let dataset = create_zeroinfl_dataset(n, 707);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| run_zip(&dataset, "y", &["x"], None))
        });
    }
    group.finish();
}

// Zero-inflated negative binomial benchmark
fn zinb_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("zinb");

    for &n in &[100, 500, 1000] {
        let dataset = create_zeroinfl_dataset(n, 808);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| run_zinb(&dataset, "y", &["x"], None))
        });
    }
    group.finish();
}

// Helper: Create balanced panel dataset
fn create_panel_dataset(n_entities: usize, n_periods: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_obs = n_entities * n_periods;

    let entity_ids: Vec<i64> = (0..n_entities)
        .flat_map(|e| std::iter::repeat_n(e as i64 + 1, n_periods))
        .collect();
    let time_ids: Vec<i64> = (0..n_entities).flat_map(|_| 1..=n_periods as i64).collect();

    // Generate data with entity fixed effects
    let entity_effects: Vec<f64> = (0..n_entities)
        .map(|_| (rng.r#gen::<f64>() - 0.5) * 2.0)
        .collect();

    let x: Vec<f64> = (0..n_obs)
        .map(|_| (rng.r#gen::<f64>() - 0.5) * 4.0)
        .collect();

    let y: Vec<f64> = (0..n_obs)
        .enumerate()
        .map(|(i, _)| {
            let entity_idx = i / n_periods;
            5.0 + 2.0 * x[i] + entity_effects[entity_idx] + (rng.r#gen::<f64>() - 0.5) * 1.0
        })
        .collect();

    let df = DataFrame::new(vec![
        Column::new("entity".into(), entity_ids),
        Column::new("time".into(), time_ids),
        Column::new("y".into(), y),
        Column::new("x".into(), x),
    ])
    .unwrap();

    Dataset::new(df)
}

// Hurdle model benchmark
fn create_hurdle_dataset(n: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 5.0 - 2.5).collect();

    // Hurdle data: binary part then count part
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let prob_positive = 1.0 / (1.0 + (-0.5 - 0.8 * xi).exp());
            if rng.r#gen::<f64>() < prob_positive {
                // Positive: truncated Poisson-like
                let mu = (0.8 + 0.5 * xi).exp();

                (mu + rng.r#gen::<f64>() * mu.sqrt()).max(1.0).floor()
            } else {
                0.0
            }
        })
        .collect();

    let df = df! {
        "y" => &y,
        "x" => &x
    }
    .unwrap();
    Dataset::new(df)
}

fn hurdle_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("hurdle");

    for &n in &[100, 500, 1000, 2000] {
        let dataset = create_hurdle_dataset(n, 42);

        group.bench_with_input(BenchmarkId::new("poisson", n), &n, |b, _| {
            b.iter(|| run_hurdle(&dataset, "y", &["x"], None, HurdleType::Poisson))
        });

        group.bench_with_input(BenchmarkId::new("negbin", n), &n, |b, _| {
            b.iter(|| run_hurdle(&dataset, "y", &["x"], None, HurdleType::NegBin))
        });
    }
    group.finish();
}

// Harvey-Collier test benchmark
fn harvey_collier_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("harvtest");

    for &n in &[50, 100, 500, 1000] {
        let dataset = create_regression_dataset(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| p2a_core::harvey_collier_test(&dataset, "y", &["x"]))
        });
    }
    group.finish();
}

// Panel GLS benchmark
fn panel_gls_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pggls");

    // (n_entities, n_periods) combinations to achieve target sizes
    for &(n_e, n_t, label) in &[(10, 10, 100), (50, 20, 1000), (100, 100, 10000)] {
        let dataset = create_panel_dataset(n_e, n_t, 42);

        group.bench_with_input(BenchmarkId::new("fe", label), &label, |b, _| {
            b.iter(|| {
                run_panel_gls(
                    &dataset,
                    "y",
                    &["x"],
                    "entity",
                    "time",
                    Some(PanelGlsModel::FixedEffects),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("pooling", label), &label, |b, _| {
            b.iter(|| {
                run_panel_gls(
                    &dataset,
                    "y",
                    &["x"],
                    "entity",
                    "time",
                    Some(PanelGlsModel::Pooling),
                )
            })
        });
    }
    group.finish();
}

// GMM benchmark
fn gmm_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pgmm");

    // GMM is computationally intensive, use smaller sizes
    for &(n_e, n_t, label) in &[(20, 10, 200), (50, 10, 500), (100, 10, 1000)] {
        let dataset = create_panel_dataset(n_e, n_t, 42);

        group.bench_with_input(BenchmarkId::from_parameter(label), &label, |b, _| {
            b.iter(|| {
                run_gmm(
                    &dataset,
                    "y",
                    &["x"],
                    "entity",
                    "time",
                    1, // lags
                    Some(GmmConfig {
                        transform: GmmTransform::Difference,
                        step: GmmStep::OneStep, // Use one-step for benchmarking
                        max_lag: Some(3),
                        min_lag: 2,
                        collapse: true,
                        robust: false,
                    }),
                )
            })
        });
    }
    group.finish();
}

// Helper: Create McFadden conditional logit dataset
fn create_mlogit_dataset(n_choosers: usize, n_alts: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_obs = n_choosers * n_alts;

    let mut chooser_ids: Vec<i64> = Vec::with_capacity(n_obs);
    let mut choice_ids: Vec<i64> = Vec::with_capacity(n_obs);
    let mut alt_price: Vec<f64> = Vec::with_capacity(n_obs);
    let mut chosen: Vec<f64> = Vec::with_capacity(n_obs);

    for c in 0..n_choosers {
        // Generate utilities for each alternative
        let utilities: Vec<f64> = (0..n_alts)
            .map(|_a| {
                let price = 10.0 + rng.r#gen::<f64>() * 20.0;
                -0.1 * price + rng.r#gen::<f64>() * 2.0 // Utility decreases with price
            })
            .collect();

        // Softmax to find chosen alternative
        let max_u = utilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = utilities.iter().map(|u| (u - max_u).exp()).sum();
        let probs: Vec<f64> = utilities
            .iter()
            .map(|u| (u - max_u).exp() / exp_sum)
            .collect();

        // Sample chosen based on probabilities
        let rand_val: f64 = rng.r#gen();
        let mut cum_prob = 0.0;
        let mut chosen_alt = 0;
        for (a, &p) in probs.iter().enumerate() {
            cum_prob += p;
            if rand_val <= cum_prob {
                chosen_alt = a;
                break;
            }
        }

        for a in 0..n_alts {
            chooser_ids.push(c as i64);
            choice_ids.push(a as i64);
            alt_price.push(10.0 + rng.r#gen::<f64>() * 20.0);
            chosen.push(if a == chosen_alt { 1.0 } else { 0.0 });
        }
    }

    let df = DataFrame::new(vec![
        Column::new("chooser".into(), chooser_ids),
        Column::new("alt".into(), choice_ids),
        Column::new("price".into(), alt_price),
        Column::new("chosen".into(), chosen),
    ])
    .unwrap();

    Dataset::new(df)
}

// McFadden conditional logit benchmark
fn mlogit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlogit");

    for &(n_choosers, n_alts, label) in &[
        (50, 3, 150),
        (100, 3, 300),
        (200, 4, 800),
        (500, 3, 1500),
        (1000, 3, 3000),
        (2000, 5, 10000),
    ] {
        let dataset = create_mlogit_dataset(n_choosers, n_alts, 42);

        group.bench_with_input(BenchmarkId::from_parameter(label), &label, |b, _| {
            b.iter(|| run_mlogit(&dataset, "chosen", "chooser", "alt", &["price"], &[], None))
        });
    }
    group.finish();
}

// Helper: Create gsynth panel dataset
fn create_gsynth_dataset(n_control: usize, n_treated: usize, n_times: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_units = n_control + n_treated;
    let n_obs = n_units * n_times;

    let mut unit_ids: Vec<String> = Vec::with_capacity(n_obs);
    let mut time_ids: Vec<i64> = Vec::with_capacity(n_obs);
    let mut outcomes: Vec<f64> = Vec::with_capacity(n_obs);
    let mut treatment: Vec<f64> = Vec::with_capacity(n_obs);

    // Treatment starts in second half for treated units
    let treatment_start = n_times / 2 + 1;

    // Control units
    for i in 0..n_control {
        let unit_fe = (rng.r#gen::<f64>() - 0.5) * 2.0;
        for t in 1..=n_times {
            unit_ids.push(format!("C{}", i + 1));
            time_ids.push(t as i64);
            outcomes.push(10.0 + unit_fe + t as f64 * 0.5 + rng.r#gen::<f64>() * 0.5);
            treatment.push(0.0);
        }
    }

    // Treated units
    for i in 0..n_treated {
        let unit_fe = (rng.r#gen::<f64>() - 0.5) * 2.0;
        let treat_effect = 3.0 + rng.r#gen::<f64>() * 2.0;
        for t in 1..=n_times {
            unit_ids.push(format!("T{}", i + 1));
            time_ids.push(t as i64);
            let base = 10.0 + unit_fe + t as f64 * 0.5 + rng.r#gen::<f64>() * 0.5;
            let effect = if t >= treatment_start {
                treat_effect
            } else {
                0.0
            };
            outcomes.push(base + effect);
            treatment.push(if t >= treatment_start { 1.0 } else { 0.0 });
        }
    }

    let df = DataFrame::new(vec![
        Column::new("unit".into(), unit_ids),
        Column::new("time".into(), time_ids),
        Column::new("outcome".into(), outcomes),
        Column::new("treated".into(), treatment),
    ])
    .unwrap();

    Dataset::new(df)
}

// Generalized synthetic control benchmark
fn gsynth_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gsynth");

    // gsynth is computationally intensive, use reasonable sizes
    for &(n_ctrl, n_treat, n_times, label) in
        &[(10, 2, 15, 180), (20, 5, 20, 500), (50, 10, 25, 1500)]
    {
        let dataset = create_gsynth_dataset(n_ctrl, n_treat, n_times, 42);

        group.bench_with_input(BenchmarkId::from_parameter(label), &label, |b, _| {
            b.iter(|| {
                run_gsynth(
                    &dataset,
                    "outcome",
                    "treated",
                    "unit",
                    "time",
                    &[],
                    GsynthConfig {
                        n_factors: 1,
                        cross_validate: false,
                        force: GsynthForce::Unit,
                        bootstrap_se: false,
                        min_pre_periods: 3,
                        ..Default::default()
                    },
                )
            })
        });
    }
    group.finish();
}

// ETWFE (Extended Two-Way Fixed Effects) benchmark
fn create_etwfe_dataset(n_units: usize, n_periods: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let n_obs = n_units * n_periods;
    let mut unit_ids: Vec<i64> = Vec::with_capacity(n_obs);
    let mut time_ids: Vec<i64> = Vec::with_capacity(n_obs);
    let mut first_treat: Vec<i64> = Vec::with_capacity(n_obs);
    let mut treat: Vec<i64> = Vec::with_capacity(n_obs);
    let mut y: Vec<f64> = Vec::with_capacity(n_obs);

    // Assign treatment timing to half the units
    let n_treated = n_units / 2;
    let mut treatment_times: Vec<i64> = (0..n_units)
        .map(|i| {
            if i < n_treated {
                3 + (rng.r#gen::<u64>() % (n_periods as u64 - 4)) as i64
            } else {
                0 // Never treated
            }
        })
        .collect();

    // Shuffle
    for i in (1..n_units).rev() {
        let j = rng.r#gen::<usize>() % (i + 1);
        treatment_times.swap(i, j);
    }

    // Generate unit and time fixed effects
    let unit_fes: Vec<f64> = (0..n_units)
        .map(|_| rng.r#gen::<f64>() * 4.0 - 2.0)
        .collect();
    let time_fes: Vec<f64> = (0..n_periods)
        .map(|_| rng.r#gen::<f64>() * 2.0 - 1.0)
        .collect();

    for u in 0..n_units {
        for t in 0..n_periods {
            unit_ids.push(u as i64);
            time_ids.push(t as i64);
            first_treat.push(treatment_times[u]);

            let is_treated = treatment_times[u] > 0 && (t as i64) >= treatment_times[u];
            treat.push(if is_treated { 1 } else { 0 });

            // Cohort-specific treatment effect
            let att = if is_treated {
                5.0 - 0.3 * treatment_times[u] as f64
            } else {
                0.0
            };

            y.push(unit_fes[u] + time_fes[t] + att + rng.r#gen::<f64>() * 2.0 - 1.0);
        }
    }

    let df = df! {
        "unit" => unit_ids,
        "time" => time_ids,
        "y" => y,
        "treat" => treat,
        "first_treat" => first_treat,
    }
    .unwrap();

    Dataset::new(df)
}

fn etwfe_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("etwfe");
    group.sample_size(10);

    for &(units, periods) in &[(50, 10), (100, 10), (100, 20), (200, 15)] {
        let dataset = create_etwfe_dataset(units, periods, 42);
        let n_obs = units * periods;

        group.bench_with_input(
            BenchmarkId::new(format!("{}x{}", units, periods), n_obs),
            &n_obs,
            |b, _| {
                b.iter(|| {
                    run_etwfe(
                        &dataset,
                        "y",
                        "unit",
                        "time",
                        "treat",
                        "first_treat",
                        None,
                        None,
                    )
                })
            },
        );
    }
    group.finish();
}

// Mixed Logit (Random Parameters Logit) benchmark
fn create_mixedlogit_dataset(n_choosers: usize, n_alts: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let n_obs = n_choosers * n_alts;
    let mut chooser_ids: Vec<i64> = Vec::with_capacity(n_obs);
    let mut alt_ids: Vec<i64> = Vec::with_capacity(n_obs);
    let mut prices: Vec<f64> = Vec::with_capacity(n_obs);
    let mut times: Vec<f64> = Vec::with_capacity(n_obs);
    let mut chosen: Vec<i64> = Vec::with_capacity(n_obs);

    // Individual-specific price sensitivity (heterogeneous)
    let beta_prices: Vec<f64> = (0..n_choosers)
        .map(|_| -0.1 + rng.r#gen::<f64>() * 0.06 - 0.03)
        .collect();
    let beta_time = -0.02;

    for c in 0..n_choosers {
        let mut utilities: Vec<f64> = Vec::with_capacity(n_alts);

        for a in 0..n_alts {
            chooser_ids.push(c as i64);
            alt_ids.push(a as i64);

            let price = 10.0 + rng.r#gen::<f64>() * 20.0;
            let time_val = 5.0 + rng.r#gen::<f64>() * 55.0;

            prices.push(price);
            times.push(time_val);

            // Utility with Gumbel error
            let gumbel_error = -(-rng.r#gen::<f64>().ln()).ln();
            let utility = beta_prices[c] * price + beta_time * time_val + gumbel_error;
            utilities.push(utility);
        }

        // Find chosen alternative
        let chosen_alt = utilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        for a in 0..n_alts {
            chosen.push(if a == chosen_alt { 1 } else { 0 });
        }
    }

    let df = df! {
        "chooser" => chooser_ids,
        "alt" => alt_ids,
        "price" => prices,
        "time" => times,
        "chosen" => chosen,
    }
    .unwrap();

    Dataset::new(df)
}

fn mixed_logit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_logit");
    group.sample_size(10);

    for &(n_choosers, n_alts, n_draws) in &[(50, 3, 50), (50, 3, 100), (100, 3, 50), (100, 3, 100)]
    {
        let dataset = create_mixedlogit_dataset(n_choosers, n_alts, 42);
        let label = format!("n{}_alts{}_draws{}", n_choosers, n_alts, n_draws);

        group.bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
            b.iter(|| {
                run_mixed_logit(
                    &dataset,
                    "chooser",
                    "alt",
                    "chosen",
                    &["price", "time"],
                    &[RandomParameterSpec {
                        name: "price".to_string(),
                        distribution: RandomDistribution::Normal,
                    }],
                    Some(MixedLogitConfig {
                        n_draws,
                        halton: true,
                        max_iter: 100,
                        tolerance: 1e-5,
                        seed: Some(42),
                    }),
                )
            })
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
    medpolish_benchmark,
    isoreg_benchmark,
    loglin_benchmark,
    cmdscale_benchmark,
    cutree_benchmark,
    // Batch 2 benchmarks
    toeplitz_benchmark,
    line_benchmark,
    cpgram_benchmark,
    supsmu_benchmark,
    constr_optim_benchmark,
    ppr_benchmark,
    se_contrast_benchmark,
    model_tables_benchmark,
    // Batch 3 benchmarks: econometric tests and discrete choice models
    bg_test_benchmark,
    reset_test_benchmark,
    wald_test_benchmark,
    vcov_hac_benchmark,
    granger_test_benchmark,
    multinom_benchmark,
    ordered_logit_benchmark,
    negbin_benchmark,
    zip_benchmark,
    zinb_benchmark,
    hurdle_benchmark,
    harvey_collier_benchmark,
    // Panel data benchmarks
    panel_gls_benchmark,
    gmm_benchmark,
    // New choice model and causal inference
    mlogit_benchmark,
    gsynth_benchmark,
    // Latest batch: ETWFE and mixed logit
    etwfe_benchmark,
    mixed_logit_benchmark,
);
criterion_main!(benches);
