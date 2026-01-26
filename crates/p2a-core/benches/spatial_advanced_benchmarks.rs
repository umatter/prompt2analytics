//! Benchmarks for advanced spatial methods: sphet, splm, spatialprobit
//!
//! Run with: cargo bench -p p2a-core -- spatial_advanced

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use p2a_core::data::Dataset;
use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
use polars::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

// Generate spatial data for benchmarking
fn generate_spatial_data(n: usize, rho: f64) -> (Dataset, SpatialWeights) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create grid coordinates
    let side = (n as f64).sqrt().ceil() as usize;
    let mut coords = Vec::with_capacity(n);
    for i in 0..side {
        for j in 0..side {
            if coords.len() < n {
                coords.push((i as f64, j as f64));
            }
        }
    }

    // Create k-nearest neighbors
    let nb = Neighbors::from_knn(&coords, 4);
    let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

    // Generate X using gen_range
    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // True parameters
    let beta = [2.0, 0.5, -0.3];

    // Generate Xβ
    let xb: Vec<f64> = (0..n)
        .map(|i| beta[0] + beta[1] * x1[i] + beta[2] * x2[i])
        .collect();
    let xb_arr = Array1::from_vec(xb);

    // Add noise
    let epsilon: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let epsilon_arr = Array1::from_vec(epsilon);

    // y = (I - rho*W)^{-1}(Xβ + ε) - simplified: y ≈ Xβ + rho*W*Xβ + ε
    let w_xb = listw.lag(&xb_arr);
    let y: Vec<f64> = (0..n)
        .map(|i| xb_arr[i] + rho * w_xb[i] + epsilon_arr[i])
        .collect();

    let df = df! {
        "y" => &y,
        "x1" => &x1,
        "x2" => &x2,
    }
    .unwrap();

    (Dataset::new(df), listw)
}

// Generate panel data
fn generate_panel_data(n_units: usize, n_time: usize) -> (Dataset, SpatialWeights) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n_total = n_units * n_time;

    // Create spatial structure
    let side = (n_units as f64).sqrt().ceil() as usize;
    let mut coords = Vec::with_capacity(n_units);
    for i in 0..side {
        for j in 0..side {
            if coords.len() < n_units {
                coords.push((i as f64, j as f64));
            }
        }
    }

    let nb = Neighbors::from_knn(&coords, 4);
    let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

    // Generate panel data
    let mut id = Vec::with_capacity(n_total);
    let mut time = Vec::with_capacity(n_total);
    let mut y = Vec::with_capacity(n_total);
    let mut x1 = Vec::with_capacity(n_total);
    let mut x2 = Vec::with_capacity(n_total);

    // Fixed effects
    let alpha_i: Vec<f64> = (0..n_units).map(|_| rng.gen_range(-1.0..1.0)).collect();

    for i in 0..n_units {
        for t in 0..n_time {
            id.push((i + 1) as f64);
            time.push((t + 1) as f64);

            let x1_val = rng.gen_range(-1.0..1.0);
            let x2_val = rng.gen_range(-1.0..1.0);
            x1.push(x1_val);
            x2.push(x2_val);

            let y_val = 2.0 + 0.5 * x1_val - 0.3 * x2_val + alpha_i[i] + rng.gen_range(-0.25..0.25);
            y.push(y_val);
        }
    }

    let df = df! {
        "id" => &id,
        "time" => &time,
        "y" => &y,
        "x1" => &x1,
        "x2" => &x2,
    }
    .unwrap();

    (Dataset::new(df), listw)
}

// Generate binary spatial data
fn generate_binary_spatial_data(n: usize, rho: f64) -> (Dataset, SpatialWeights) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create grid coordinates
    let side = (n as f64).sqrt().ceil() as usize;
    let mut coords = Vec::with_capacity(n);
    for i in 0..side {
        for j in 0..side {
            if coords.len() < n {
                coords.push((i as f64, j as f64));
            }
        }
    }

    let nb = Neighbors::from_knn(&coords, 4);
    let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

    // Generate X
    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Latent variable
    let xb: Vec<f64> = (0..n).map(|i| 0.5 * x1[i] - 0.3 * x2[i]).collect();
    let xb_arr = Array1::from_vec(xb);
    let w_xb = listw.lag(&xb_arr);

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let y_star = xb_arr[i] + rho * w_xb[i] + rng.gen_range(-1.0..1.0);
            if y_star > 0.0 { 1.0 } else { 0.0 }
        })
        .collect();

    let df = df! {
        "y" => &y,
        "x1" => &x1,
        "x2" => &x2,
    }
    .unwrap();

    (Dataset::new(df), listw)
}

fn bench_sphet(c: &mut Criterion) {
    use p2a_core::econometrics::{run_sphet, SphetConfig, SphetModel, SphetSE};

    let mut group = c.benchmark_group("sphet");
    group.sample_size(10);

    for n in [100, 400, 900] {
        let (dataset, listw) = generate_spatial_data(n, 0.4);

        // SAR model
        group.bench_with_input(BenchmarkId::new("SAR", n), &n, |b, _| {
            b.iter(|| {
                let config = SphetConfig {
                    model: SphetModel::SpatialLag,
                    se_type: SphetSE::Robust,
                    ..Default::default()
                };
                run_sphet(
                    black_box(&dataset),
                    black_box("y"),
                    black_box(&["x1", "x2"]),
                    black_box(&mut listw.clone()),
                    black_box(config),
                )
            })
        });

        // SEM model
        group.bench_with_input(BenchmarkId::new("SEM", n), &n, |b, _| {
            b.iter(|| {
                let config = SphetConfig {
                    model: SphetModel::SpatialError,
                    se_type: SphetSE::Robust,
                    ..Default::default()
                };
                run_sphet(
                    black_box(&dataset),
                    black_box("y"),
                    black_box(&["x1", "x2"]),
                    black_box(&mut listw.clone()),
                    black_box(config),
                )
            })
        });
    }

    group.finish();
}

fn bench_splm(c: &mut Criterion) {
    use p2a_core::econometrics::{run_spml, SpmlConfig, SpatialPanelEffect, SpatialPanelModel};

    let mut group = c.benchmark_group("splm");
    group.sample_size(10);

    for (n_units, n_time) in [(25, 10), (49, 10), (100, 10)] {
        let n_total = n_units * n_time;
        let (dataset, listw) = generate_panel_data(n_units, n_time);

        // Fixed effects with spatial lag
        group.bench_with_input(BenchmarkId::new("FE_lag", n_total), &n_total, |b, _| {
            b.iter(|| {
                let config = SpmlConfig {
                    model: SpatialPanelModel::Within,
                    effect: SpatialPanelEffect::Individual,
                    lag: true,
                    ..Default::default()
                };
                run_spml(
                    black_box(&dataset),
                    black_box("y"),
                    black_box(&["x1", "x2"]),
                    black_box("id"),
                    black_box("time"),
                    black_box(&mut listw.clone()),
                    black_box(config),
                )
            })
        });
    }

    group.finish();
}

fn bench_spatialprobit(c: &mut Criterion) {
    use p2a_core::econometrics::{run_sar_probit, SpatialProbitConfig};

    let mut group = c.benchmark_group("spatialprobit");
    group.sample_size(10);

    for n in [100, 225, 400] {
        let (dataset, listw) = generate_binary_spatial_data(n, 0.3);

        // SAR probit with 500 draws (matching R benchmark)
        group.bench_with_input(BenchmarkId::new("SAR_probit", n), &n, |b, _| {
            b.iter(|| {
                let config = SpatialProbitConfig {
                    n_draws: 500,
                    burn_in: 100,
                    seed: Some(42),
                    ..Default::default()
                };
                run_sar_probit(
                    black_box(&dataset),
                    black_box("y"),
                    black_box(&["x1", "x2"]),
                    black_box(&mut listw.clone()),
                    black_box(config),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sphet, bench_splm, bench_spatialprobit);
criterion_main!(benches);
