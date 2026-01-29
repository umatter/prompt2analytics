//! Forecasting method benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- forecasting`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use p2a_core::{
    ArConfig, ArMethod, CostFunction, Dataset, DecomposeConfig, DecomposeType, SeasonalType,
    StateSpaceModel, StructTsConfig, StructTsType, ar, decompose, kalman_filter, kalman_forecast,
    kalman_smoother, run_arima, run_changepoint, run_holt_winters, run_mstl, struct_ts,
};
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
    }
    .expect("Failed to create DataFrame");

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
    }
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

fn arima_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ARIMA");

    for n in [100, 200, 500] {
        let dataset = generate_time_series_dataset(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &dataset, |b, data| {
            b.iter(|| run_arima(data, "y", 1, 1, 1));
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
            b.iter(|| run_changepoint(data, "y", Some(1.0), Some(5), CostFunction::MeanChange));
        });
    }

    group.finish();
}

/// Generate synthetic time series with trend and multiplicative seasonality
fn generate_seasonal_dataset(n: usize, period: usize, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Create seasonal pattern
    let seasonal_pattern: Vec<f64> = (0..period)
        .map(|i| 0.8 + 0.4 * (i as f64 * 2.0 * std::f64::consts::PI / period as f64).sin())
        .collect();

    let y: Vec<f64> = (0..n)
        .map(|t| {
            let trend = 100.0 + 0.5 * t as f64;
            let seasonal = seasonal_pattern[t % period];
            let noise = 1.0 + rng.gen_range(0.0..1.0) * 0.05 - 0.025; // ±2.5% noise
            trend * seasonal * noise
        })
        .collect();

    let df = df! {
        "y" => y,
    }
    .expect("Failed to create DataFrame");

    Dataset::new(df)
}

fn holt_winters_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("HoltWinters");

    // Test different dataset sizes with period=12 (monthly data)
    for n in [48, 120, 240, 480] {
        let dataset = generate_seasonal_dataset(n, 12, 42);

        group.bench_with_input(
            BenchmarkId::new("multiplicative", n),
            &dataset,
            |b, data| {
                b.iter(|| {
                    run_holt_winters(
                        data,
                        "y",
                        12,
                        SeasonalType::Multiplicative,
                        None, // optimize alpha
                        None, // optimize beta
                        None, // optimize gamma
                    )
                });
            },
        );
    }

    // Test with fixed parameters (no optimization)
    for n in [48, 120, 240, 480] {
        let dataset = generate_seasonal_dataset(n, 12, 42);

        group.bench_with_input(BenchmarkId::new("fixed_params", n), &dataset, |b, data| {
            b.iter(|| {
                run_holt_winters(
                    data,
                    "y",
                    12,
                    SeasonalType::Multiplicative,
                    Some(0.2), // fixed alpha
                    Some(0.1), // fixed beta
                    Some(0.3), // fixed gamma
                )
            });
        });
    }

    group.finish();
}

fn holt_winters_period_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("HoltWinters_Period");

    // Test different seasonal periods with n=240
    for period in [4, 12, 24, 52] {
        let n = period * 5; // 5 full cycles
        let dataset = generate_seasonal_dataset(n, period, 42);

        group.bench_with_input(BenchmarkId::from_parameter(period), &dataset, |b, data| {
            b.iter(|| {
                run_holt_winters(
                    data,
                    "y",
                    period,
                    SeasonalType::Multiplicative,
                    None,
                    None,
                    None,
                )
            });
        });
    }

    group.finish();
}

/// Generate AR(2) time series for benchmarking
fn generate_ar2_series(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let phi1 = 0.7;
    let phi2 = -0.2;

    let mut x = vec![0.0; n];
    x[0] = rng.r#gen::<f64>() - 0.5;
    x[1] = phi1 * x[0] + rng.r#gen::<f64>() - 0.5;

    for t in 2..n {
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + (rng.r#gen::<f64>() - 0.5) * 0.5;
    }
    x
}

fn ar_yule_walker_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("AR_YuleWalker");

    for n in [50, 100, 500, 1000] {
        let x = generate_ar2_series(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &x, |b, data| {
            b.iter(|| {
                ar(
                    data,
                    ArConfig {
                        method: ArMethod::YuleWalker,
                        ..Default::default()
                    },
                )
            });
        });
    }

    group.finish();
}

fn ar_burg_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("AR_Burg");

    for n in [50, 100, 500, 1000] {
        let x = generate_ar2_series(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &x, |b, data| {
            b.iter(|| {
                ar(
                    data,
                    ArConfig {
                        method: ArMethod::Burg,
                        ..Default::default()
                    },
                )
            });
        });
    }

    group.finish();
}

fn ar_ols_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("AR_OLS");

    for n in [50, 100, 500, 1000] {
        let x = generate_ar2_series(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &x, |b, data| {
            b.iter(|| {
                ar(
                    data,
                    ArConfig {
                        method: ArMethod::Ols,
                        ..Default::default()
                    },
                )
            });
        });
    }

    group.finish();
}

/// Generate seasonal time series for decompose benchmarking
fn generate_decompose_series(n: usize, period: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..n)
        .map(|t| {
            let trend = 100.0 + 0.5 * t as f64;
            let seasonal = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
            let noise = (rng.r#gen::<f64>() - 0.5) * 2.0;
            trend + seasonal + noise
        })
        .collect()
}

fn decompose_additive_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decompose_Additive");

    for n in [48, 120, 240, 480, 1200] {
        let x = generate_decompose_series(n, 12, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &x, |b, data| {
            b.iter(|| decompose(data, 12, DecomposeConfig::default()));
        });
    }

    group.finish();
}

fn decompose_multiplicative_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decompose_Multiplicative");

    // Generate multiplicative data
    fn generate_mult_series(n: usize, period: usize, seed: u64) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|t| {
                let trend = 100.0 + 0.5 * t as f64;
                let seasonal =
                    1.0 + 0.2 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
                let noise = 1.0 + (rng.r#gen::<f64>() - 0.5) * 0.02;
                trend * seasonal * noise
            })
            .collect()
    }

    for n in [48, 120, 240, 480, 1200] {
        let x = generate_mult_series(n, 12, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &x, |b, data| {
            b.iter(|| {
                decompose(
                    data,
                    12,
                    DecomposeConfig {
                        decompose_type: DecomposeType::Multiplicative,
                        filter: None,
                    },
                )
            });
        });
    }

    group.finish();
}

/// Generate random walk with drift for StructTS benchmarking
fn generate_local_level_series(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut level = 100.0;
    let mut y = Vec::with_capacity(n);

    for _ in 0..n {
        level += (rng.r#gen::<f64>() - 0.5) * 2.0; // Level noise
        let obs = level + (rng.r#gen::<f64>() - 0.5) * 5.0; // Observation noise
        y.push(obs);
    }
    y
}

/// Generate trend + seasonal series for BSM benchmarking
fn generate_bsm_series(n: usize, period: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut level = 100.0;
    let mut slope = 0.5;

    let mut y = Vec::with_capacity(n);

    for t in 0..n {
        level += slope + (rng.r#gen::<f64>() - 0.5) * 0.5;
        slope += (rng.r#gen::<f64>() - 0.5) * 0.05;
        let seasonal = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin();
        let obs = level + seasonal + (rng.r#gen::<f64>() - 0.5) * 3.0;
        y.push(obs);
    }
    y
}

fn struct_ts_level_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("StructTS_Level");

    for n in [50, 100, 200, 500] {
        let y = generate_local_level_series(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &y, |b, data| {
            b.iter(|| {
                struct_ts(
                    data,
                    StructTsConfig {
                        model_type: StructTsType::Level,
                        ..Default::default()
                    },
                )
            });
        });
    }

    group.finish();
}

fn struct_ts_trend_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("StructTS_Trend");

    for n in [50, 100, 200, 500] {
        let y = generate_local_level_series(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &y, |b, data| {
            b.iter(|| {
                struct_ts(
                    data,
                    StructTsConfig {
                        model_type: StructTsType::Trend,
                        ..Default::default()
                    },
                )
            });
        });
    }

    group.finish();
}

fn struct_ts_bsm_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("StructTS_BSM");

    // Shorter series for BSM because it's more complex
    for n in [48, 96, 144, 240] {
        let y = generate_bsm_series(n, 12, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &y, |b, data| {
            b.iter(|| {
                struct_ts(
                    data,
                    StructTsConfig {
                        model_type: StructTsType::BSM,
                        period: Some(12),
                        ..Default::default()
                    },
                )
            });
        });
    }

    group.finish();
}

/// Build a simple local level state-space model for Kalman filter benchmarking
fn build_local_level_model() -> StateSpaceModel {
    // State-space representation of local level model:
    // y_t = level_t + epsilon_t (observation)
    // level_{t+1} = level_t + eta_t (state)
    StateSpaceModel::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(), // Transition T (1x1)
        Array1::from_vec(vec![1.0]),                        // Observation Z (1x1)
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(), // Selection R (1x1)
        Array2::from_shape_vec((1, 1), vec![0.5]).unwrap(), // State cov Q (1x1)
        1.0,                                                // Observation var H
    )
    .expect("Failed to create model")
}

fn kalman_filter_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("KalmanFilter");

    let model = build_local_level_model();
    let init_state = Array1::from_vec(vec![0.0]);
    let init_cov = Array2::from_shape_vec((1, 1), vec![10000.0]).unwrap();

    for n in [100, 500, 1000, 5000] {
        let y = generate_local_level_series(n, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &y, |b, data| {
            b.iter(|| kalman_filter(data, &model, init_state.view(), init_cov.view()));
        });
    }

    group.finish();
}

fn kalman_smoother_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("KalmanSmoother");

    let model = build_local_level_model();
    let init_state = Array1::from_vec(vec![0.0]);
    let init_cov = Array2::from_shape_vec((1, 1), vec![10000.0]).unwrap();

    for n in [100, 500, 1000, 5000] {
        let y = generate_local_level_series(n, 42);
        // Pre-compute filter result for smoother benchmark
        let filter_result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &filter_result, |b, data| {
            b.iter(|| kalman_smoother(data, &model));
        });
    }

    group.finish();
}

fn kalman_forecast_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("KalmanForecast");

    let model = build_local_level_model();
    let init_state = Array1::from_vec(vec![0.0]);
    let init_cov = Array2::from_shape_vec((1, 1), vec![10000.0]).unwrap();
    let horizon = 24;

    for n in [100, 500, 1000, 5000] {
        let y = generate_local_level_series(n, 42);
        // Pre-compute filter result for forecast benchmark
        let filter_result = kalman_filter(&y, &model, init_state.view(), init_cov.view()).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &filter_result, |b, data| {
            b.iter(|| kalman_forecast(data, &model, horizon));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    arima_benchmark,
    mstl_benchmark,
    changepoint_benchmark,
    holt_winters_benchmark,
    holt_winters_period_benchmark,
    ar_yule_walker_benchmark,
    ar_burg_benchmark,
    ar_ols_benchmark,
    decompose_additive_benchmark,
    decompose_multiplicative_benchmark,
    struct_ts_level_benchmark,
    struct_ts_trend_benchmark,
    struct_ts_bsm_benchmark,
    kalman_filter_benchmark,
    kalman_smoother_benchmark,
    kalman_forecast_benchmark,
);
criterion_main!(benches);
