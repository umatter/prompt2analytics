# Validation: cluster indices (CH, DB, Dunn, ARI, NMI, silhouette count, gap statistic, cluster_stats)

## Method Overview

`crates/p2a-core/src/ml/cluster_validation.rs` and `cluster_stats` in
`crates/p2a-core/src/ml/advanced_clustering_mod/common.rs`.

- **Labels.** Every index works on the distinct labels, whatever their values: gapped labels (0, 2, 7) give the same
  result as (0, 1, 2), and a huge label value allocates nothing by value (labels are compacted by a dense remap when
  the largest label is below 2n, otherwise by sorting). CH, DB and Dunn need at least 2 distinct clusters.
  `silhouette`'s `n_clusters` counts the distinct labels.
- **Calinski-Harabasz, Davies-Bouldin, Dunn** as `clusterCrit::intCriteria`: DB's scatter is the mean distance to
  the centroid; Dunn is the minimum between-cluster distance over the maximum diameter.
- **ARI and NMI** from the contingency table of non-empty cells (dense when `k_true * k_pred <= 4n`, sorted pairs
  otherwise). NMI's variants (`max`, `min`, `sqrt`, `sum`) as `aricode::NMI`; degenerate partitions follow
  scikit-learn: both partitions a single cluster give 1, a mutual information of 0 gives 0.
- **Gap statistic**: Tibshirani et al. (2001) as `cluster::clusGap(x, kmeans, K.max = k_max, B = n_boot,
  d.power = 2, spaceH0 = "original")` with `maxSE(method = "Tibs2001SEmax")`: uniform references over the
  bounding box, the inner k-means stopping on the relative 1e-4 rule (to convergence cost 1.45x at n = 10,000),
  `SE = sd * sqrt(1 + 1/B)` with the sample standard deviation
  (divisor B - 1), `n_boot >= 2`. `log_w` is the log of the within-cluster sum of squares; clusGap's `logW` is
  `log_w - log 2` (it sums each pair once), so gap and SE are equal. clusGap's own defaults differ (`d.power = 1`,
  `spaceH0 = "scaledPCA"`, `B = 100`, `firstSEmax`). The reference draws are p2a's own, so the gap values are not
  pinned; `logW` and the choice rule are.
- **`cluster_stats`** as `fpc::cluster.stats(dist(x), labels)`: `average_within` is the cluster-size-weighted mean of
  each cluster's mean pairwise distance (singletons left out), `average_between` the mean distance over pairs in
  different clusters, separation, diameters and per-cluster silhouettes as fpc; Davies-Bouldin is the one definition
  above. One pass over the pairs; the distance matrix is resource-guarded.

## Reference

`clusterCrit::intCriteria` (clusterCrit 1.3.0), `fpc::cluster.stats` (fpc 2.2.15), `aricode::ARI`, `aricode::NMI`
(aricode 1.1.0), `cluster::clusGap` and `cluster::maxSE` (cluster 2.1.6), R 4.3.3.

## Where

R code `validation/r/cases/ml_cluster_t4.R`, cases `cluster_stats_blobs`, `partition_agreement` and
`gap_logw_blobs`, dataset `validation/datasets/generated/ap_blobs.csv` (n = 115, the PAM k = 3 labels). Registry
`validation/methods/ml_cluster_t4.json`, method `ml_cluster_indices_t4`.

## Values

| Key | Value |
|---|---|
| `ch` | 244.05642758710653 |
| `db` | 0.40597252367402986 |
| `dunn` | 0.12453710235345887 |
| `avg_silwidth` | 0.7034059835844825 |
| `fpc_average_within`, `fpc_average_between` | 0.887401571811688, 3.1596990320089553 |
| `fpc_separation`, `fpc_max_diameter` | 0.5306567910688311, 4.261033708354085 |
| `ari` | 0.3188622754491018 |
| `nmi_max`, `nmi_min`, `nmi_sqrt`, `nmi_sum` | 0.5171367888025904, 0.5534047654433435, 0.5349635158675976, 0.5346564309892753 |
| `logW` (clusGap's, k = 1..5) | 5.377114329893122, 4.780133761015491, 3.697860217214868, 3.3522723554479215, 2.952199465764463 |
| `tibs_k1`, `tibs_k2` | 3, 5 (maxSE `Tibs2001SEmax` on the case's gap and SE vectors) |

## Tolerances

`Tol::stat(n)` for the indices and `cluster_stats`; partition agreement on 15 labels `Tol::stat(15)`; `log W`
`Tol::coef(115)`; chosen k exact.

## Known differences

- The gap statistic's reference draws are p2a's (seeded), not R's; `spaceH0 = "scaledPCA"` and `d.power = 1` are not offered.

## Tests

R-pinned: `test_validate_cluster_indices_blobs`, `test_validate_partition_agreement`, `test_validate_gap_logw_and_rule`
(`ml::cluster_validation`), `test_validate_cluster_stats_blobs` (`advanced_clustering_mod::common`). Properties:
`test_nmi_degenerate_partitions`, `test_indices_refuse_one_distinct_label`, `test_huge_labels_do_not_allocate_by_value`,
`test_silhouette_reports_distinct_cluster_count`, `test_compact_labels_dense_and_sparse_agree`,
`test_gap_refuses_fewer_than_two_references`, `test_cluster_stats_gapped_labels`.

## Timing

Quiet integration re-time, 2026-11-16: release builds of v0.5.0 (`f80b0131`) and of the integrated branch,
`--no-default-features --features ci`; one core (`taskset -c 2`), `RAYON_NUM_THREADS=1` (one thread for polars and
BLAS too), cwd `/dev/shm`; per row five processes per side, interleaved with the order alternating (nine for rows
within 1.05-1.15x), each the median of its calls after a warm-up; cells are the median over the processes, the ratio
of the minima in brackets. 1-minute load average 0.7-2.7 on 12 cores, nothing else building. Data: noisy three blobs
(centres (0, 0), (4, 4), (8, 0), unit normal noise; 2 columns unless stated); gapped labels are 0, 4, 9; the pair
indices compare the labels with a copy in which 10% are moved to the next cluster. Rows marked † are the lane's (load
23-53, three processes per side) and were not repeated. Rand index and NMI build the contingency table from the raw
labels when it is small, sized first by the labels OR-ed together (one vectorised pass instead of the maxima), and
the Davies-Bouldin index groups the rows without copying labels that are already compact and reuses one difference
buffer for the dispersions (the same products and sums). Their rows were re-timed on 2026-11-17 after these changes,
with the same protocol and nine processes per side (load 1.4-2.2); the first quiet re-time had measured 1.12-1.54x.
Peak memory of `silhouette_optimized` at n = 10,000 (quiet re-time, median of three): 3,057 MiB at v0.5.0, 767 MiB now.

| Call | n | v0.5.0 | this change | ratio [min] |
|---|---|---|---|---|
| cluster_stats | 100 | 92.7 µs | 75 µs | 0.81 [0.81] |
| cluster_stats | 1,000 | 8.51 ms | 6.83 ms | 0.80 [0.81] |
| calinski_harabasz | 100 | 10.1 µs | 10.7 µs | 1.06 [1.08] |
| calinski_harabasz | 1,000 | 86.5 µs | 92.7 µs | 1.07 [1.06] |
| calinski_harabasz | 10,000 | 869 µs | 931 µs | 1.07 [1.07] |
| calinski_harabasz, gapped labels | 100 | 10.4 µs | 10.8 µs | 1.03 [1.05] |
| calinski_harabasz, gapped labels | 10,000 | 870 µs | 930 µs | 1.07 [1.06] |
| davies_bouldin | 100 | 6.98 µs | 3.73 µs | 0.53 [0.52] |
| davies_bouldin | 1,000 | 55.3 µs | 19.5 µs | 0.35 [0.35] |
| davies_bouldin | 10,000 | 542 µs | 173 µs | 0.32 [0.32] |
| davies_bouldin, gapped | 100 | 7.63 µs | 3.71 µs | 0.49 [0.48] |
| davies_bouldin, gapped | 1,000 | 56.3 µs | 19.6 µs | 0.35 [0.35] |
| davies_bouldin, gapped | 10,000 | 543 µs | 176 µs | 0.32 [0.32] |
| rand_index | 100 | 0.578 µs | 0.484 µs | 0.84 [0.87] |
| rand_index | 1,000 | 3.04 µs | 1.85 µs | 0.61 [0.60] |
| rand_index | 10,000 | 29.1 µs | 17.2 µs | 0.59 [0.57] |
| rand_index, gapped | 100 | 0.998 µs | 0.851 µs | 0.85 [0.79] |
| rand_index, gapped | 1,000 | 3.95 µs | 2.2 µs | 0.56 [0.62] |
| rand_index, gapped | 10,000 | 30.6 µs | 18.9 µs | 0.62 [0.57] |
| nmi | 100 | 0.708 µs | 0.617 µs | 0.87 [0.90] |
| nmi | 1,000 | 3.21 µs | 1.98 µs | 0.62 [0.64] |
| nmi | 10,000 | 29.2 µs | 17.7 µs | 0.61 [0.59] |
| nmi, gapped | 100 | 1.17 µs | 0.993 µs | 0.85 [0.80] |
| nmi, gapped | 1,000 | 4.24 µs | 2.34 µs | 0.55 [0.65] |
| nmi, gapped | 10,000 | 35.2 µs | 18.5 µs | 0.53 [0.58] |
| silhouette | 100 | 43.8 µs | 42.9 µs | 0.98 [0.97] |
| silhouette | 1,000 | 4.79 ms | 4.73 ms | 0.99 [0.98] |
| silhouette, gapped | 100 | 44.2 µs | 42.9 µs | 0.97 [0.97] |
| silhouette, gapped | 1,000 | 4.82 ms | 4.75 ms | 0.98 [0.98] |
| silhouette_optimized | 100 | 154 µs | 74.2 µs | 0.48 [0.43] |
| silhouette_optimized | 1,000 | 12.1 ms | 5.32 ms | 0.44 [0.46] |
| gap_statistic, k_max 5, B 20 | 100 | 4.59 ms | 4.58 ms | 1.00 [1.00] |
| gap_statistic, k_max 5, B 20 | 1,000 | 69.4 ms | 69.5 ms | 1.00 [1.00] |
| cluster_stats † | 10,000 | 3.64 s | 3.36 s | 0.92 |
| calinski_harabasz, gapped labels † | 1,000 | 107.0 µs | 111.1 µs | 1.04 |
| silhouette † | 10,000 | 1.39 s | 1.40 s | 1.01 |
| silhouette, gapped † | 10,000 | 1.44 s | 1.44 s | 1.00 |
| silhouette_optimized † | 10,000 | 4.89 s | 1.42 s | 0.29 |
| gap_statistic, k_max 5, B 20 † | 10,000 | 1.56 s | 1.35 s | 0.87 |

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::cluster_validation advanced_clustering_mod::common
```
