//! Machine learning method benchmarks for p2a-core
//!
//! Run with: `cargo bench -p p2a-core -- ml`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use p2a_core::{Linkage, dbscan, hierarchical, kmeans, pca};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate synthetic cluster data
fn generate_cluster_data(n: usize, k: usize, n_clusters: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut data = Array2::zeros((n, k));

    for i in 0..n {
        let cluster = i % n_clusters;
        let center = cluster as f64 * 3.0; // Cluster centers at 0, 3, 6, ...

        for j in 0..k {
            data[[i, j]] = center + rng.gen_range(0.0..1.0) * 0.5;
        }
    }

    data
}

fn kmeans_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("KMeans");

    for n in [100, 1000, 5000] {
        let data = generate_cluster_data(n, 5, 3, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, d| {
            b.iter(|| kmeans(d.view(), 3, Some(100), Some(1e-4), Some(5), Some(42)));
        });
    }

    group.finish();
}

fn dbscan_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("DBSCAN");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, d| {
            b.iter(|| dbscan(d.view(), 0.5, 5));
        });
    }

    group.finish();
}

fn hierarchical_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hierarchical");

    for n in [50, 100, 200] {
        let data = generate_cluster_data(n, 5, 3, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, d| {
            b.iter(|| hierarchical(d.view(), Some(3), Linkage::Ward, None));
        });
    }

    group.finish();
}

fn pca_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCA");

    for n in [100, 1000, 5000] {
        let data = generate_cluster_data(n, 10, 3, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, d| {
            b.iter(|| pca(d.view(), Some(5), false));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    kmeans_benchmark,
    dbscan_benchmark,
    hierarchical_benchmark,
    pca_benchmark
);
criterion_main!(benches);
