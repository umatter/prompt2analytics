# Validation: Silhouette widths (cluster::silhouette)

## Method Overview

One core (`silhouette_core` in `crates/p2a-core/src/ml/cluster_validation.rs`) serves `silhouette`,
`silhouette_from_dist`, `silhouette_optimized` and the PAM, CLARA and cluster-statistics helpers in
`advanced_clustering_mod/common.rs`. It follows `cluster::silhouette`'s C routine `sildist`: a(i) and b(i) are
means of distances summed in index order; b(i) takes the first minimising cluster; s(i) = (b - a) / max(a, b),
and s(i) = 0 for a singleton (Rousseeuw 1987) and when a(i) = b(i). Per-cluster averages run over all members.
Labellings without 2 to n - 1 non-empty clusters (R returns `NA`) and non-finite data or dissimilarities are refused.

## Reference

`cluster::silhouette(x, dist)` (R 4.3.3, cluster 2.1.6, C routine `sildist`).

## Where

R code `validation/r/cases/ml_density_clustering.R`, case `silhouette_singletons`, dataset
`validation/datasets/generated/blobs_noise.csv` (115 x 2), labels with one singleton cluster.

## Values

| Key | Value | Tolerance |
|---|---|---|
| `sil_width` (115) | first 0.4848189098035294, 0.7431798836529211, 0.7664693729980926; the singleton (row 107) 0 | `Tol::coef` |
| `avg_width` | 0.7044989909620206 (was 0.71319464313593373 with singleton = 1) | `Tol::stat` |
| `cluster_avg` | 0.6789283619445188, 0.6404714049517972, 0.8302594161514012, 0.7072284309790935, 0 (the singleton), 0.8646829635399542 | `Tol::stat` |
| `neighbor` | R's neighbour cluster (1-based) | exact |
| `n_singletons` | 1 | exact |

## Tolerances

`validation/README.md`, n = 115: widths `Tol::coef(115)` (1e-8 relative), averages `Tol::stat(115)` (1e-6 relative); neighbours exact.

## Known differences

None.

## Tests

R-pinned (`validation/methods/ml_density_clustering.json`, method `silhouette`):
`ml::cluster_validation::tests::test_validate_silhouette_singletons`,
`ml::cluster_validation::tests::test_validate_silhouette_from_dist_singletons`,
`ml::cluster_optimized::tests::test_validate_silhouette_optimized_singletons`,
`ml::advanced_clustering_mod::common::tests::test_validate_silhouette_helpers_singletons`. Refusals:
`test_silhouette_needs_two_to_n_minus_1_clusters`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- silhouette
```
