# Validation: HDBSCAN (dbscan::hdbscan)

## Method Overview

`p2a_core::ml::hdbscan()` and `hdbscan_with()` (`crates/p2a-core/src/ml/advanced_clustering_mod/hdbscan.rs`)
follow R's `dbscan::hdbscan` 1.2.6 step by step:

1. Core distances counting the point itself: the distance to the `(min_samples - 1)`-th nearest other point
   (`kNNdist(x, minPts - 1)`); `min_samples = 1` gives core distances 0. The k-d tree path (n >= 500, d <= 15) and
   the brute-force path give the same values bit for bit.
2. The minimum spanning tree of the mutual-reachability distances `max(d(i, j), core_i, core_j)` by Prim's
   algorithm on the dense graph (dbscan's `mst.cpp`): start at point 0, strict `<` for weight updates, the next
   vertex is the smallest weight with the lowest index among equal weights. O(n^2 d) time, O(n) memory.
3. Single linkage of the tree in `hclust` encoding (`hclustMergeOrder`, edges in stable weight order).
4. The cluster tree (`computeStability`): walking the merges top-down, a merge whose two sides both hold at least
   `min_cluster_size` points splits the current node into the next two ids.
5. Excess-of-mass selection (FOSC, `prune_unstable = FALSE`), computed iteratively.
6. Labels (`getSalientAssignments`), dbscan's membership probabilities and GLOSH outlier scores.

`min_samples` counts the point itself, as R's `minPts` and sklearn's `min_samples` (the pre-Tier-3 value `m`
corresponds to the new `m + 1`). Labels: the selected clusters in dbscan's post-order; R label = p2a label + 1,
noise is -1 in p2a and 0 in R.

## Reference

`dbscan::hdbscan(x, minPts)` (R 4.3.3, dbscan 1.2.6), `dbscan::kNNdist(x, k = minPts - 1)` for the core
distances, and `stats::hclust(dist(x), "single")$height` for the merge heights at `min_samples = 1`.

## Where

R code `validation/r/cases/ml_density_clustering.R`, case `hdbscan_blobs`, dataset
`validation/datasets/generated/blobs_noise.csv` (115 x 2: three Gaussian blobs of 40, 35 and 30 points plus 10
uniform points, seed 20261109).

## Values

| Key | Value | Rust test | Tolerance |
|---|---|---|---|
| `mp5_cluster` (0 = noise) | 0: 6, 1: 31, 2: 37, 3: 41; noise rows 106, 107, 109, 111, 112, 115 | `test_validate_hdbscan_blobs_mp5` | exact |
| `mp5_membership_prob[1..4]` | 0.34428795026288833, 0.7410663803834842, 0.7009940704369092, 0.36507502957144083 | `test_validate_hdbscan_blobs_mp5` | `Tol::stat` |
| `mp5_outlier_scores[1..4]` | 0.6051107803793943, 0, 0, 0.5921823338563266 | `test_validate_hdbscan_blobs_mp5` | `Tol::stat` |
| `mp5_cluster_scores` | 118.8922670135319, 85.1205848510343, 116.00428316319727 | `test_validate_hdbscan_blobs_tree` | `Tol::stat` |
| `mp5_coredist[1..4]` (= `knndist_k4`) | 0.395884341186166, 0.15633045855102273, 0.180524005138077, 0.38333419939671715 | `test_validate_hdbscan_blobs_tree` | `Tol::coef` |
| `mp5_hc_height` | first 0.1353551061227736, last 3.4280615947254875 | `test_validate_hdbscan_blobs_tree` | `Tol::coef` |
| `mp5_tree_*` | ids 0..6, parents NA, 0, 0, 2, 2, 4, 4; eps birth/death, stabilities, points at split and leaving | `test_validate_hdbscan_blobs_tree` | exact / `Tol::coef` / `Tol::stat` |
| `mp5_salient` | 1, 3, 4 | `test_validate_hdbscan_blobs_tree` | exact |
| `mp10_*` | the same table of labels; scores 72.61130425551904, 51.783749888103934, 72.33543923943381 | `test_validate_hdbscan_blobs_mp10`, `_tree` | as above |
| `mp5_x1000_*` | the same labels; scores 0.11889226701353191, ...; probabilities and GLOSH as at unit scale | `test_validate_hdbscan_scale_free` | exact / `Tol::stat` |
| `nosplit_cluster` (rows 1-9) | all 0 (noise) | `test_validate_hdbscan_no_split_and_boundary` | exact |
| `boundary_cluster`, `boundary_cluster_scores` (rows 1-5, 41-45) | 2, 2, 2, 2, 2, 1, 1, 1, 1, 1; 2.1270587375909638, 2.5158888914032342 | `test_validate_hdbscan_no_split_and_boundary` | exact / `Tol::stat` |
| `single_linkage_height` | first 0.005268237961375066, last 2.932250407406957 | `test_validate_hdbscan_min_samples_one_is_single_linkage` | `Tol::coef` |

## Tolerances

`validation/README.md`, n = 115: labels, ids and counts exact; core distances, merge heights and eps birth/death
`Tol::coef(115)` (1e-8 relative); stabilities, probabilities and GLOSH `Tol::stat(115)` (1e-6 relative). R sums the
stabilities with Rcpp sugar; the planning port agreed to 4.3e-14.

## Known differences

- **Membership probabilities when no cluster is found** (registry deviation, ML-49): dbscan reports
  `(max - coredist) / max` over the root for every point although all are noise (case value
  `nosplit_membership_prob_dbscan`, not asserted); p2a reports 0, the value dbscan gives noise points whenever a
  cluster exists.
- **Membership probabilities when every member has core distance 0** (registry deviation, ML-49): dbscan's
  `(max - coredist) / max` is 0/0 = NaN (`min_samples = 1`, which dbscan does not allow, or duplicate points); p2a
  reports 1.
- **GLOSH when no split exists** (registry deviation, ML-49): dbscan leaves the root's `eps_death` at its
  `unordered_map` default 0, so every point scores 1 (case value `nosplit_outlier_scores`, not asserted). p2a
  computes GLOSH by its definition (Campello et al. 2015): `1 - eps_min / eps_p`, with `eps_min` the smallest merge
  height and `eps_p` the height at which p first merges. Not R-pinned; asserted as a property.
- `min_cluster_size != min_samples` has no dbscan analogue (dbscan has one `minPts`), so it is checked by property
  only.
- dbscan's stability is not Campello's canonical condensed tree: a point leaves at the height where it becomes a
  singleton, or at its cluster's death if that is higher. p2a follows dbscan.

## Tests

- R-pinned (`validation/methods/ml_density_clustering.json`, method `hdbscan`):
  `ml::advanced_clustering_mod::hdbscan::tests::test_validate_hdbscan_blobs_mp5`, `_blobs_mp10`, `_blobs_tree`,
  `_scale_free`, `_no_split_and_boundary`, `_min_samples_one_is_single_linkage`, and
  `ml::cluster_optimized::tests::test_validate_hdbscan_optimized_matches_hdbscan` (the delegation).
- Parameters and paths: `test_hdbscan_min_samples_0_is_error`, `test_hdbscan_rejects_bad_parameters`,
  `test_hdbscan_kdtree_core_distances_match_brute_force`, `test_hdbscan_parallel_prim_matches_sequential`,
  `test_hdbscan_cluster_consistency`, `test_mst_correctness_*`, `test_hdbscan_perf_comparison`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::advanced_clustering_mod::hdbscan
```
