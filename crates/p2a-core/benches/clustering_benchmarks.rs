//! Clustering benchmarks for comparison with R.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array2;
use p2a_core::ml::{
    affinity_propagation,
    affinity_propagation_optimized,
    calinski_harabasz,
    davies_bouldin,
    dunn_index,
    gap_statistic,
    gaussian_mixture,
    hdbscan,
    hdbscan_optimized,
    kmedoids,
    kmedoids_optimized,
    nmi,
    optics,
    optics_optimized,
    rand_index,
    silhouette,
    // Optimized versions
    silhouette_optimized,
    spectral_clustering,
};

/// Generate cluster data with clear separation (matches R benchmark).
fn generate_cluster_data(n: usize, k: usize, n_clusters: usize, separation: f64) -> Array2<f64> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.5).unwrap();

    let mut data = Array2::zeros((n, k));

    for i in 0..n {
        let cluster = i % n_clusters;
        let center = cluster as f64 * separation;

        for j in 0..k {
            data[[i, j]] = center + rng.sample(normal);
        }
    }

    data
}

/// Generate labels for validation metrics.
fn generate_labels(n: usize, n_clusters: usize) -> Vec<usize> {
    (0..n).map(|i| i % n_clusters).collect()
}

fn benchmark_silhouette(c: &mut Criterion) {
    let mut group = c.benchmark_group("silhouette");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);
        let labels = generate_labels(n, 3);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| silhouette(black_box(data.view()), black_box(&labels)))
        });
    }

    group.finish();
}

fn benchmark_calinski_harabasz(c: &mut Criterion) {
    let mut group = c.benchmark_group("calinski_harabasz");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);
        let labels = generate_labels(n, 3);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| calinski_harabasz(black_box(data.view()), black_box(&labels)))
        });
    }

    group.finish();
}

fn benchmark_davies_bouldin(c: &mut Criterion) {
    let mut group = c.benchmark_group("davies_bouldin");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);
        let labels = generate_labels(n, 3);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| davies_bouldin(black_box(data.view()), black_box(&labels)))
        });
    }

    group.finish();
}

fn benchmark_dunn_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("dunn_index");

    for n in [100, 500] {
        // Skip 1000 due to O(n^2) complexity
        let data = generate_cluster_data(n, 5, 3, 5.0);
        let labels = generate_labels(n, 3);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| dunn_index(black_box(data.view()), black_box(&labels)))
        });
    }

    group.finish();
}

fn benchmark_gap_statistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("gap_statistic");
    group.sample_size(10); // Reduce samples due to bootstrap cost

    for n in [100, 200] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| gap_statistic(black_box(data.view()), 5, Some(20), Some(42)))
        });
    }

    group.finish();
}

fn benchmark_kmedoids(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmedoids");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| kmedoids(black_box(data.view()), 3, Some(100), Some(42)))
        });
    }

    group.finish();
}

fn benchmark_spectral_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_clustering");

    for n in [100, 200, 500] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| spectral_clustering(black_box(data.view()), 3, None, Some(42)))
        });
    }

    group.finish();
}

fn benchmark_affinity_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("affinity_propagation");
    group.sample_size(10);

    for n in [100, 200] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                affinity_propagation(black_box(data.view()), None, Some(0.9), Some(100), Some(10))
            })
        });
    }

    group.finish();
}

fn benchmark_hdbscan(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdbscan");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| hdbscan(black_box(data.view()), Some(5), Some(5)))
        });
    }

    group.finish();
}

fn benchmark_optics(c: &mut Criterion) {
    let mut group = c.benchmark_group("optics");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| optics(black_box(data.view()), 5, None, None))
        });
    }

    group.finish();
}

fn benchmark_gaussian_mixture(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_mixture");
    group.sample_size(10);

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| gaussian_mixture(black_box(data.view()), 3, None, Some(50), None, Some(42)))
        });
    }

    group.finish();
}

fn benchmark_rand_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_index");

    for n in [100, 500, 1000] {
        let true_labels = generate_labels(n, 3);
        let pred_labels: Vec<usize> = (0..n).map(|i| (i * 7) % 3).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| rand_index(black_box(&true_labels), black_box(&pred_labels)))
        });
    }

    group.finish();
}

fn benchmark_nmi(c: &mut Criterion) {
    let mut group = c.benchmark_group("nmi");

    for n in [100, 500, 1000] {
        let true_labels = generate_labels(n, 3);
        let pred_labels: Vec<usize> = (0..n).map(|i| (i * 7) % 3).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| nmi(black_box(&true_labels), black_box(&pred_labels), None))
        });
    }

    group.finish();
}

// =============================================================================
// OPTIMIZED VERSIONS - Comparison benchmarks
// =============================================================================

fn benchmark_silhouette_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("silhouette_optimized");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);
        let labels = generate_labels(n, 3);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| silhouette_optimized(black_box(data.view()), black_box(&labels)))
        });
    }

    group.finish();
}

fn benchmark_kmedoids_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmedoids_optimized");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| kmedoids_optimized(black_box(data.view()), 3, Some(100), Some(42)))
        });
    }

    group.finish();
}

fn benchmark_hdbscan_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdbscan_optimized");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| hdbscan_optimized(black_box(data.view()), Some(5), Some(5)))
        });
    }

    group.finish();
}

fn benchmark_optics_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("optics_optimized");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| optics_optimized(black_box(data.view()), 5, None))
        });
    }

    group.finish();
}

fn benchmark_affinity_propagation_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("affinity_propagation_optimized");
    group.sample_size(10);

    for n in [100, 200] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                affinity_propagation_optimized(
                    black_box(data.view()),
                    None,
                    Some(0.9),
                    Some(100),
                    Some(10),
                )
            })
        });
    }

    group.finish();
}

// Comparison benchmark - original vs optimized side by side
fn benchmark_silhouette_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("silhouette_comparison");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);
        let labels = generate_labels(n, 3);

        group.bench_with_input(BenchmarkId::new("original", n), &n, |b, _| {
            b.iter(|| silhouette(black_box(data.view()), black_box(&labels)))
        });

        group.bench_with_input(BenchmarkId::new("optimized", n), &n, |b, _| {
            b.iter(|| silhouette_optimized(black_box(data.view()), black_box(&labels)))
        });
    }

    group.finish();
}

fn benchmark_kmedoids_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmedoids_comparison");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::new("original", n), &n, |b, _| {
            b.iter(|| kmedoids(black_box(data.view()), 3, Some(100), Some(42)))
        });

        group.bench_with_input(BenchmarkId::new("optimized", n), &n, |b, _| {
            b.iter(|| kmedoids_optimized(black_box(data.view()), 3, Some(100), Some(42)))
        });
    }

    group.finish();
}

fn benchmark_hdbscan_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdbscan_comparison");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::new("original", n), &n, |b, _| {
            b.iter(|| hdbscan(black_box(data.view()), Some(5), Some(5)))
        });

        group.bench_with_input(BenchmarkId::new("optimized", n), &n, |b, _| {
            b.iter(|| hdbscan_optimized(black_box(data.view()), Some(5), Some(5)))
        });
    }

    group.finish();
}

fn benchmark_optics_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optics_comparison");

    for n in [100, 500, 1000] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::new("original", n), &n, |b, _| {
            b.iter(|| optics(black_box(data.view()), 5, None, None))
        });

        group.bench_with_input(BenchmarkId::new("optimized", n), &n, |b, _| {
            b.iter(|| optics_optimized(black_box(data.view()), 5, None))
        });
    }

    group.finish();
}

fn benchmark_affinity_propagation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("affinity_propagation_comparison");
    group.sample_size(10);

    for n in [100, 200] {
        let data = generate_cluster_data(n, 5, 3, 5.0);

        group.bench_with_input(BenchmarkId::new("original", n), &n, |b, _| {
            b.iter(|| {
                affinity_propagation(black_box(data.view()), None, Some(0.9), Some(100), Some(10))
            })
        });

        group.bench_with_input(BenchmarkId::new("optimized", n), &n, |b, _| {
            b.iter(|| {
                affinity_propagation_optimized(
                    black_box(data.view()),
                    None,
                    Some(0.9),
                    Some(100),
                    Some(10),
                )
            })
        });
    }

    group.finish();
}

// =============================================================================
// LARGE-N BENCHMARKS - Testing KD-Tree and Dual-Tree algorithms
// =============================================================================

fn benchmark_hdbscan_large_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdbscan_large_n");
    group.sample_size(10); // Reduce samples for large datasets

    // Test the automatic algorithm selection
    for n in [1000, 2000, 5000, 10000] {
        let data = generate_cluster_data(n, 5, 5, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| hdbscan(black_box(data.view()), Some(10), Some(10)))
        });
    }

    group.finish();
}

fn benchmark_optics_large_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("optics_large_n");
    group.sample_size(10);

    for n in [1000, 2000, 5000, 10000] {
        let data = generate_cluster_data(n, 5, 5, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| optics(black_box(data.view()), 10, None, None))
        });
    }

    group.finish();
}

fn benchmark_hdbscan_dimensionality(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdbscan_dimensionality");
    group.sample_size(10);

    let n = 2000;

    // Test with different dimensions to see when KD-tree becomes ineffective
    for d in [2, 5, 10, 15, 20, 30] {
        let data = generate_cluster_data(n, d, 5, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| hdbscan(black_box(data.view()), Some(10), Some(10)))
        });
    }

    group.finish();
}

fn benchmark_optics_dimensionality(c: &mut Criterion) {
    let mut group = c.benchmark_group("optics_dimensionality");
    group.sample_size(10);

    let n = 2000;

    for d in [2, 5, 10, 15, 20, 30] {
        let data = generate_cluster_data(n, d, 5, 5.0);

        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| optics(black_box(data.view()), 10, None, None))
        });
    }

    group.finish();
}

// Benchmark KD-tree core distance computation
fn benchmark_kdtree_core_distances(c: &mut Criterion) {
    use p2a_core::ml::KdTree;

    let mut group = c.benchmark_group("kdtree_core_distances");
    group.sample_size(10);

    for n in [1000, 5000, 10000, 20000] {
        let data = generate_cluster_data(n, 5, 5, 5.0);
        let data_vecs: Vec<Vec<f64>> = (0..n).map(|i| data.row(i).to_vec()).collect();
        let tree = KdTree::new(data_vecs);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| tree.compute_core_distances(10))
        });
    }

    group.finish();
}

// Benchmark MST construction algorithms
fn benchmark_mst_algorithms(c: &mut Criterion) {
    use p2a_core::ml::{KdTree, build_connected_mst, dual_tree_boruvka_mst, kdtree_prim_mst};

    let mut group = c.benchmark_group("mst_algorithms");
    group.sample_size(10);

    for n in [1000, 2000, 5000] {
        let data = generate_cluster_data(n, 5, 5, 5.0);
        let data_vecs: Vec<Vec<f64>> = (0..n).map(|i| data.row(i).to_vec()).collect();
        let tree = KdTree::new(data_vecs);
        let core_dists = tree.compute_core_distances(10);

        group.bench_with_input(BenchmarkId::new("kdtree_prim", n), &n, |b, _| {
            b.iter(|| kdtree_prim_mst(&tree, &core_dists))
        });

        group.bench_with_input(BenchmarkId::new("dual_tree_boruvka", n), &n, |b, _| {
            b.iter(|| dual_tree_boruvka_mst(&tree, &core_dists))
        });

        group.bench_with_input(BenchmarkId::new("connected_mst", n), &n, |b, _| {
            b.iter(|| build_connected_mst(&tree, &core_dists))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_silhouette,
    benchmark_calinski_harabasz,
    benchmark_davies_bouldin,
    benchmark_dunn_index,
    benchmark_gap_statistic,
    benchmark_kmedoids,
    benchmark_spectral_clustering,
    benchmark_affinity_propagation,
    benchmark_hdbscan,
    benchmark_optics,
    benchmark_gaussian_mixture,
    benchmark_rand_index,
    benchmark_nmi,
);

criterion_group!(
    optimized_benches,
    benchmark_silhouette_optimized,
    benchmark_kmedoids_optimized,
    benchmark_hdbscan_optimized,
    benchmark_optics_optimized,
    benchmark_affinity_propagation_optimized,
);

criterion_group!(
    comparison_benches,
    benchmark_silhouette_comparison,
    benchmark_kmedoids_comparison,
    benchmark_hdbscan_comparison,
    benchmark_optics_comparison,
    benchmark_affinity_propagation_comparison,
);

criterion_group!(
    large_n_benches,
    benchmark_hdbscan_large_n,
    benchmark_optics_large_n,
    benchmark_hdbscan_dimensionality,
    benchmark_optics_dimensionality,
    benchmark_kdtree_core_distances,
    benchmark_mst_algorithms,
);

criterion_main!(
    benches,
    optimized_benches,
    comparison_benches,
    large_n_benches
);
