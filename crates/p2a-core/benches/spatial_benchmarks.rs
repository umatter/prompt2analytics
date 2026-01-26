//! Benchmarks for spatial econometrics methods
//!
//! Run with: `cargo bench -p p2a-core -- spatial`
//!
//! Compares performance of Rust implementations against R's spdep/spatialreg packages.
//! R benchmarks are in: performance/comparisons/r_comparison/benchmark_spatial.R

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use p2a_core::data::Dataset;
use p2a_core::regression::CovarianceType;
use p2a_core::spatial::{
    localmoran, moran_test, spatial_lm_tests, MoranAlternative, Neighbors, SpatialWeights,
    WeightStyle,
};
use p2a_core::{
    run_ols, run_sac_dataset, run_sar_dataset, run_sem_dataset, SacConfig, SarConfig, SemConfig,
};
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Create grid coordinates for a n_side x n_side grid
fn create_grid_coords(n_side: usize) -> Vec<(f64, f64)> {
    let mut coords = Vec::with_capacity(n_side * n_side);
    for y in 0..n_side {
        for x in 0..n_side {
            coords.push((x as f64, y as f64));
        }
    }
    coords
}

/// Generate data for benchmarking (simplified, with spatial patterns)
fn create_spatial_data(
    n_side: usize,
    seed: u64,
) -> (Vec<(f64, f64)>, Vec<f64>, Vec<f64>, Neighbors, SpatialWeights) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_side * n_side;

    // Create grid coordinates
    let coords = create_grid_coords(n_side);

    // Create k-nearest neighbors (k=4)
    let nb = Neighbors::from_knn(&coords, 4);
    let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);

    // Generate x variable
    let x: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() * 2.0 - 1.0).collect();

    // Generate y with spatial pattern based on coordinates and x
    // y = 2 + 0.7*x + 0.3*(x_coord + y_coord) + error
    // This creates spatial dependence without needing matrix inversion
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let (cx, cy) = coords[i];
            let error = rng.r#gen::<f64>() * 0.5 - 0.25;
            2.0 + 0.7 * x[i] + 0.3 * (cx + cy) / (n_side as f64) + error
        })
        .collect();

    (coords, x, y, nb, listw)
}

/// Benchmark neighbor creation (knearneigh + nb2listw equivalent)
fn neighbors_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_neighbors");

    for n_side in [10, 20, 32, 50].iter() {
        let n = n_side * n_side;
        let coords = create_grid_coords(*n_side);

        group.bench_with_input(BenchmarkId::new("knn_k4", n), &n, |b, _| {
            b.iter(|| {
                let nb = Neighbors::from_knn(&coords, 4);
                let _listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
            })
        });
    }
    group.finish();
}

/// Benchmark Moran's I test (moran.test equivalent)
fn moran_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("moran_test");

    for n_side in [10, 20, 32, 50].iter() {
        let n = n_side * n_side;
        let (_, _, y, _, listw) = create_spatial_data(*n_side, 42);
        let y_arr = Array1::from_vec(y);

        group.bench_with_input(BenchmarkId::new("moran_i", n), &n, |b, _| {
            b.iter(|| moran_test(&y_arr, &listw, MoranAlternative::Greater))
        });
    }
    group.finish();
}

/// Benchmark LM tests (lm.LMtests equivalent)
fn lm_tests_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_lm_tests");

    for n_side in [10, 20, 32, 50].iter() {
        let n = n_side * n_side;
        let (_, x, y, _, listw) = create_spatial_data(*n_side, 42);

        // Fit OLS model first (needed for LM tests)
        let df = df! {
            "y" => &y,
            "x" => &x,
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let ols_result =
            run_ols(&dataset, "y", &["x"], true, CovarianceType::Standard).expect("OLS failed");

        // Get residuals
        use p2a_core::traits::LinearEstimator;
        let residuals = ols_result.residuals();

        // Build X matrix with intercept
        let mut x_mat = Array2::zeros((n, 2));
        for i in 0..n {
            x_mat[[i, 0]] = 1.0; // intercept
            x_mat[[i, 1]] = x[i];
        }

        // Clone listw for benchmarking (spatial_lm_tests takes mutable reference)
        let listw_clone = listw.clone();

        group.bench_with_input(BenchmarkId::new("lm_tests", n), &n, |b, _| {
            b.iter(|| {
                let mut lw = listw_clone.clone();
                spatial_lm_tests(&residuals, &x_mat, &mut lw)
            })
        });
    }
    group.finish();
}

/// Benchmark SAR model (lagsarlm equivalent)
fn sar_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sar_model");
    group.sample_size(10); // Reduce sample size for slower tests

    for n_side in [10, 20, 32].iter() {
        let n = n_side * n_side;
        let (_, x, y, _, mut listw) = create_spatial_data(*n_side, 42);

        // Pre-compute eigenvalues so clones include them
        let _ = listw.eigenvalues();

        let df = df! {
            "y" => &y,
            "x" => &x,
        }
        .unwrap();
        let dataset = Dataset::new(df);
        // Disable impact computation for fair comparison with R's basic lagsarlm
        let config = SarConfig {
            compute_impacts: false,
            ..Default::default()
        };
        let listw_clone = listw.clone();

        group.bench_with_input(BenchmarkId::new("lagsarlm", n), &n, |b, _| {
            b.iter(|| {
                let mut lw = listw_clone.clone();
                run_sar_dataset(&dataset, "y", &["x"], &mut lw, config.clone())
            })
        });
    }
    group.finish();
}

/// Benchmark SEM model (errorsarlm equivalent)
fn sem_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sem_model");
    group.sample_size(10); // Reduce sample size for slower tests

    for n_side in [10, 20, 32].iter() {
        let n = n_side * n_side;
        let (_, x, y, _, mut listw) = create_spatial_data(*n_side, 42);

        // Pre-compute eigenvalues so clones include them
        let _ = listw.eigenvalues();

        let df = df! {
            "y" => &y,
            "x" => &x,
        }
        .unwrap();
        let dataset = Dataset::new(df);
        let config = SemConfig::default();
        let listw_clone = listw.clone();

        group.bench_with_input(BenchmarkId::new("errorsarlm", n), &n, |b, _| {
            b.iter(|| {
                let mut lw = listw_clone.clone();
                run_sem_dataset(&dataset, "y", &["x"], &mut lw, config.clone())
            })
        });
    }
    group.finish();
}

/// Benchmark Local Moran's I (localmoran equivalent)
fn localmoran_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("localmoran");

    for n_side in [10, 20, 32].iter() {
        let n = n_side * n_side;
        let (_, _, y, _, listw) = create_spatial_data(*n_side, 42);
        let y_arr = Array1::from_vec(y);

        // Analytical p-values (n_perm = 0)
        group.bench_with_input(BenchmarkId::new("analytical", n), &n, |b, _| {
            b.iter(|| localmoran(&y_arr, &listw, 0.05, 0))
        });
    }

    // Also benchmark permutation-based (only for small n)
    for n_side in [10, 20].iter() {
        let n = n_side * n_side;
        let (_, _, y, _, listw) = create_spatial_data(*n_side, 42);
        let y_arr = Array1::from_vec(y);

        // Permutation-based (99 permutations)
        group.bench_with_input(BenchmarkId::new("perm_99", n), &n, |b, _| {
            b.iter(|| localmoran(&y_arr, &listw, 0.05, 99))
        });
    }

    group.finish();
}

/// Benchmark SAC model (sacsarlm equivalent)
fn sac_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sac_model");
    group.sample_size(10); // Reduce sample size for slower tests

    for n_side in [10, 20, 32].iter() {
        let n = n_side * n_side;
        let (_, x, y, _, mut listw) = create_spatial_data(*n_side, 42);

        // Pre-compute eigenvalues so clones include them
        let _ = listw.eigenvalues();

        let df = df! {
            "y" => &y,
            "x" => &x,
        }
        .unwrap();
        let dataset = Dataset::new(df);
        let config = SacConfig::default();
        let listw_clone = listw.clone();

        group.bench_with_input(BenchmarkId::new("sacsarlm", n), &n, |b, _| {
            b.iter(|| {
                let mut lw = listw_clone.clone();
                run_sac_dataset(&dataset, "y", &["x"], &mut lw, config.clone())
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    neighbors_benchmark,
    moran_benchmark,
    localmoran_benchmark,
    lm_tests_benchmark,
    sar_benchmark,
    sem_benchmark,
    sac_benchmark,
);
criterion_main!(benches);
