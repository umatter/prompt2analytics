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

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::cluster_validation advanced_clustering_mod::common
```
