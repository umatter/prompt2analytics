# Validation: PAM, CLARA, kmedoids_optimized, fuzzy c-means, FANNY

## Method Overview

`kmedoids` and `clara` (`advanced_clustering_mod/kmedoids.rs`) and `fuzzy_cmeans` and `fanny`
(`advanced_clustering_mod/fuzzy.rs`).

- **`kmedoids`** is PAM. BUILD and SWAP are deterministic, as in `cluster::pam`, so its `seed` is documented as
  ignored.
- **`clara`** runs PAM on seeded samples (`CLARA_DEFAULT_SEED` for `None`).
- **`kmedoids_optimized`** (`ml/cluster_optimized.rs`) is PAM too: BUILD and best-improvement SWAP with the gains
  computed in parallel, O(k n^2) per SWAP iteration. A swap is made while it lowers the total cost by more than
  `1e-12` of the cost (scale-free); the reported cost is recomputed after each swap. `seed` is ignored. Its distance
  matrix, like `silhouette_optimized`'s, is allocated through the resource guards and checks the budget per row.
- **`fuzzy_cmeans`** is `e1071::cmeans`: memberships `u_ij = 1 / sum_l (d_ij / d_il)^(2/(m-1))` on distances (a
  point on one or more centres shares membership equally among them), Euclidean centres the `u^m`-weighted means and
  Manhattan centres the `u^m`-weighted medians (the lower one on a tie of the weights), objective `withinerror`
  `sum u^m d^2 / n` (Manhattan: `d`, not squared, as e1071). It stops when the objective changes by at most
  `tolerance` relative. `fuzzy_cmeans_from_centers` starts from given centres (finite, distinct, the data's columns),
  as `cmeans(x, centers = <matrix>)`; `fuzzy_cmeans` starts from `FUZZY_DEFAULT_SEED` for `None`.
- **`fanny`** is `cluster::fanny`'s Kaufman-Rousseeuw iteration: a Gauss-Seidel sweep over the points on the
  dissimilarities, R's starting memberships when `seed` is `None` (a seeded random start for `Some`), `tol` 1e-15
  relative and `maxit` 500 (R's defaults), `k` in `1..=n/2 - 1` (R's message). The normalised Dunn coefficient is
  `(k F - 1) / (k - 1)`; the clustering takes the first maximal membership.

## Reference

`cluster::pam`, `cluster::clara(samples = 1, sampsize = n, pamLike = TRUE)`, `cluster::fanny` (cluster 2.1.6),
`e1071::cmeans` (e1071 1.7.17) on R 4.3.3.

## Where

R code `validation/r/cases/ml_partition_t3.R`, cases `pam_clara_blobs` and `fuzzy_reference_blobs`, and
`validation/r/cases/ml_cluster_t4.R`, cases `fcm_cmeans_from_start` and `fanny_blobs_more`; dataset
`validation/datasets/generated/ap_blobs.csv` (n = 115). Registry `validation/methods/ml_mixtures.json`, methods
`ml_pam_clara` and `ml_fuzzy_partitioning`, and `validation/methods/ml_cluster_t4.json`, methods `ml_fcm_t4`,
`ml_fanny_t4` and `ml_kmedoids_optimized_t4`.

`e1071::cmeans` returns reals that differ between runs of one R process (centres by 1.5e-15, memberships by 1.5e-13,
the objective by 8.5e-16, relative), below `run.R --check`'s 1e-12. The case's JSON is therefore not byte-reproducible;
it is never edited by hand.

## Values

| | k = 2 | k = 3 | k = 4 |
|---|---|---|---|
| medoids (1-based) | 30, 51 | 18, 51, 83 | 18, 51, 83, 106 |
| `pamk_total` | 141.6754178801239 | 66.9503538526157 | 60.916331588022736 |
| `pamk_avg_sil` | 0.48941093083270004 | 0.7034059835844825 | 0.7245248130297822 |

`clara3`: medoids 18, 51, 83, `objective` 0.5821769900227454 (= 66.9503538526157 / 115), `avg_sil` 0.7034059835844825.

`kmedoids_optimized` reaches the same medoids and totals for k = 2, 3, 4.

FANNY (k = 3, m = 2): `fanny_coeff` 0.6625798715638848, 0.4938698073458272; `fanny_objective` 33.28123392479843;
`fanny_blobs_more` adds other k and memberships.

c-means from (0,0), (3,0), (1.5,2.5), m = 2: centres (-0.031927131974105315, -0.026196593233908008),
(1.4172334737645318, 2.5575435255668837), (2.9935543325501768, 0.05565880401835589); `withinerror`
0.45453513290186937. `fcm_cmeans_from_start` holds m = 1.5 and the Manhattan runs too.

## Tolerances

Medoids, medoid assignments and hard labels exact; totals and silhouettes 1e-12 (arithmetic at fixed medoids);
`kmedoids_optimized`'s total `Tol::coef(115)`. c-means centres, memberships and `withinerror` and FANNY memberships
and coefficients are iterative fixed points (`Tol::iterative`, the value in each test); FANNY's objective
`Tol::stat(115)`.

## Known differences

- **PAM SWAP.** p2a takes the first improving swap and restarts; `cluster::pam` takes the best swap per step. On the
  case's row order both reach the same medoids for k = 2, 3, 4 (`stopifnot` in the case). On permuted rows the
  written-out p2a SWAP stopped at other medoids in 30 of 100 permutations for k = 2 and 44 of 100 for k = 4, never for
  k = 3 (0 of 200). CLARA is PAM on a shuffled sample, so it is pinned at k = 3 only. No claim of PAM equivalence holds
  beyond the pinned data.
- **Fuzzy c-means random start.** `fuzzy_cmeans` draws its start from its own seeded generator, not R's
  `cmeans(x, k)` draw; the fixed point is pinned from given centres (`fuzzy_cmeans_from_centers`).
- **FANNY with a seed.** `Some(seed)` gives a random start, which R does not have; `None` is R's start.

## Not R-pinned

| Method | Reason |
|---|---|
| k-prototypes | `clustMixType::kproto` is installed; no finding names a pin. Seeded runs are reproducible, and mode ties go to the smallest category. |
| Spherical k-means | `skmeans` is installed; a pin needs a directional dataset. |
| Mini-batch and trimmed k-means | No R implementation is installed (`ClusterR`, `tclust`, `trimcluster` absent). Property tests only. |

## Tests

R-pinned: `test_validate_pam_blobs`, `test_validate_clara_full_sample_is_pam`, `test_validate_fcm_cmeans_blobs_labels`,
`test_validate_fcm_cmeans_blobs`, `test_validate_fcm_cmeans_from_start`, `test_validate_fanny_blobs`,
`test_validate_fanny_blobs_more`, `test_validate_kmedoids_optimized_pam`.

Properties: `test_kmedoids_result_does_not_depend_on_the_seed`, `test_clara_default_seed_is_reproducible`,
`test_kmedoids_rejects_bad_parameters`, `test_fuzzy_cmeans_stopping_rule_is_scale_free`,
`test_fcm_distance_parses_aliases_and_rejects_unknown`,
`test_fuzzy_methods_reject_bad_parameters_and_default_seeds_are_fixed`,
`test_fcm_manhattan_centres_are_weighted_medians`, `test_fanny_is_scale_free`, `test_fanny_refuses_k_above_half_n`,
`test_kmedoids_optimized_reported_cost_is_true_cost`, `test_kmedoids_optimized_scale_free`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- advanced_clustering_mod::kmedoids::tests::test_validate advanced_clustering_mod::fuzzy::tests::test_validate cluster_optimized::tests::test_validate
```
