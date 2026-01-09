# Machine Learning Performance Report

## Overview

This report documents the performance of p2a machine learning methods compared to reference implementations.

## Methods Benchmarked

| Method | p2a Function | R Reference | Python Reference |
|--------|--------------|-------------|------------------|
| K-Means | `kmeans` | `stats::kmeans()` | `sklearn.cluster.KMeans` |
| DBSCAN | `dbscan` | `dbscan::dbscan()` | `sklearn.cluster.DBSCAN` |
| Hierarchical | `hierarchical` | `stats::hclust()` | `scipy.cluster.hierarchy` |
| PCA | `pca` | `stats::prcomp()` | `sklearn.decomposition.PCA` |

## Benchmark Configuration

- **Rust**: Criterion with 50 measurement iterations
- **R**: microbenchmark with 50 iterations
- **Data**: Synthetic clusters with k=5 features, 3 clusters

## Results Summary

### K-Means Clustering

| n | p2a Rust (μs) | R kmeans (μs) | Speedup |
|---|---------------|---------------|---------|
| 100 | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD |
| 5000 | TBD | TBD | TBD |

### DBSCAN

| n | p2a Rust (μs) | R dbscan (μs) | Speedup |
|---|---------------|---------------|---------|
| 100 | TBD | TBD | TBD |
| 500 | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD |

### Hierarchical Clustering

| n | p2a Rust (μs) | R hclust (μs) | Speedup |
|---|---------------|---------------|---------|
| 50 | TBD | TBD | TBD |
| 100 | TBD | TBD | TBD |
| 200 | TBD | TBD | TBD |

### PCA

| n | p2a Rust (μs) | R prcomp (μs) | Speedup |
|---|---------------|---------------|---------|
| 100 | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD |
| 5000 | TBD | TBD | TBD |

## Running Benchmarks

### Rust Benchmarks

```bash
cargo bench -p p2a-core -- ml
```

### R Benchmarks

```bash
cd performance/comparisons/r_comparison
Rscript benchmark_ml.R
```

## Scaling Behavior

Expected complexity:
- K-Means: O(n × k × i) where k = clusters, i = iterations
- DBSCAN: O(n²) worst case, O(n log n) with spatial indexing
- Hierarchical: O(n² log n) with efficient algorithm
- PCA: O(n × d²) where d = dimensions

## Notes

- Results marked "TBD" will be populated after running the benchmark suite
- Hierarchical clustering limited to n ≤ 200 due to O(n²) memory requirement
- K-Means uses n_init=5 for multiple random initializations

## Hardware Configuration

See `performance/hardware_profiles.md` for benchmark hardware specifications.
