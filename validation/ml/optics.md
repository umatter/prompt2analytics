# Validation: OPTICS (dbscan::optics, extractDBSCAN, extractXi)

## Method Overview

`p2a_core::ml::optics()` and `optics_with()` (`crates/p2a-core/src/ml/advanced_clustering_mod/optics.rs`) follow
R's `dbscan::optics` 1.2.6 (`optics_int`): squared distances throughout, the eps-neighbourhood inclusive of the
point itself, the core distance the `minPts`-th smallest distance in that neighbourhood (`INFINITY`, R `NA`, when
fewer than `minPts` points lie within `eps`), the seeds an indexed min-heap with decrease-key in which the higher
id wins equal reachabilities (dbscan/ELKI), and dbscan's predecessor bookkeeping. The k-d tree (n >= 500, d <= 15)
only proposes neighbour candidates; every kept distance is computed by the same kernel, so both paths give the
same ordering bit for bit. Memory is O(n) besides one neighbourhood.

Flat clusterings:

- `OpticsResult::extract_dbscan(eps_cl)` is `extractDBSCAN(object, eps_cl)`: walking the ordering, a point whose
  reachability exceeds `eps_cl` starts a cluster if its core distance is at most `eps_cl`, else it is noise.
- `OpticsResult::extract_xi(xi, correct_predecessors)` is `extractXi(object, xi, minimum = FALSE,
  correctPredecessors)`: steep areas, the mib bookkeeping, the `minPts` rule, the predecessor correction; each point
  takes the smallest xi-cluster containing it.
- The legacy `optics(data, min_samples, max_eps, xi)`: with `xi`, `extractXi(xi)` with the correction; without `xi`
  and with a finite `max_eps`, `extractDBSCAN(eps_cl = max_eps)`; otherwise no flat clustering (all labels -1 and a
  warning), as R's `optics()` alone. The pre-Tier-3 extraction (a threshold at the mean reachability) is gone.

`min_samples` counts the point itself (R `minPts`). `reachability` is in ordering order; `reachability_by_point()`
gives R's `reachdist`. Ordering positions are 0-based (R positions + 1); labels: R = p2a + 1, noise -1 / 0.

## Reference

`dbscan::optics(x, eps, minPts)`, `dbscan::extractDBSCAN(object, eps_cl)` and
`dbscan::extractXi(object, xi, minimum = FALSE, correctPredecessors)` (R 4.3.3, dbscan 1.2.6).

## Where

R code `validation/r/cases/ml_density_clustering.R`, case `optics_blobs`, dataset
`validation/datasets/generated/blobs_noise.csv` (115 x 2).

## Values

| Key | Call | Rust test | Tolerance |
|---|---|---|---|
| `inf_order`, `inf_reachdist`, `inf_coredist` | `optics(x, minPts = 5)` | `test_validate_optics_inf` | exact / `Tol::coef` |
| `eps05_order`, `eps05_reachdist`, `eps05_coredist` (14 NA) | `optics(x, eps = 0.5, minPts = 5)` | `test_validate_optics_eps05` | exact / `Tol::coef` |
| `inf_predecessor`, `eps05_predecessor` | `$predecessor` | `test_validate_optics_predecessor` | exact |
| `eps035_order`, `eps035_dbscan_cluster`, `inf_dbscan_cluster`, `eps05_dbscan_cluster` | `extractDBSCAN(..., eps_cl)` | `test_validate_optics_extract_dbscan`, `test_validate_optics_optimized_matches_optics` | exact |
| `inf_xi_*`, `inf_xi_nocorrect_*`, `eps05_xi_*` (cluster, clusters start/end) | `extractXi(..., xi = 0.05)` | `test_validate_optics_extract_xi` | exact |

## Tolerances

`validation/README.md`, n = 115: orderings, predecessors and labels exact; distances `Tol::coef(115)`.

## Known differences

- `extractXi(..., minimum = TRUE)` is not offered: R's own code drops the first element of
  `sort(unique(cluster))`, assuming noise is present.
- Undefined distances are `INFINITY` in p2a and `NA` in R.

## Tests

- R-pinned (`validation/methods/ml_density_clustering.json`, method `optics`): the six tests above.
- Paths and parameters: `test_optics_kdtree_neighbourhoods_match_brute_force`, `test_optics_rejects_bad_parameters`,
  `test_optics_extraction_rejects_bad_parameters`, `test_optics_basic`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::advanced_clustering_mod::optics
```
