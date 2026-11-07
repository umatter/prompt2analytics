# Validation: PAM, CLARA, fuzzy c-means, FANNY

## Method Overview

`kmedoids` and `clara` (`advanced_clustering_mod/kmedoids.rs`) and `fuzzy_cmeans` and `fanny`
(`advanced_clustering_mod/fuzzy.rs`).

- **`kmedoids`** is PAM. BUILD and SWAP are deterministic, as in `cluster::pam`, so its `seed` is documented as
  ignored.
- **`clara`** runs PAM on seeded samples (`CLARA_DEFAULT_SEED` for `None`).
- **`fuzzy_cmeans`** stops on `relative_change` of the objective, with `FUZZY_DEFAULT_SEED` for `None`. A point exactly
  on a centre takes full membership there.
- **`fanny`** uses the same seed rule and argmax over memberships with the first maximum winning.

## Reference

`cluster::pam`, `cluster::clara(samples = 1, sampsize = n, pamLike = TRUE)`, `cluster::fanny` (cluster 2.1.6),
`e1071::cmeans` (e1071 1.7.17) on R 4.3.3.

## Where

R code `validation/r/cases/ml_partition_t3.R`, cases `pam_clara_blobs` and `fuzzy_reference_blobs`, dataset
`validation/datasets/generated/ap_blobs.csv` (n = 115). Registry `validation/methods/ml_mixtures.json`, methods
`ml_pam_clara` and `ml_fuzzy_partitioning`.

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

Recorded, not pinned (Known differences):

- FANNY: `fanny_coeff` 0.6625798715638848, 0.4938698073458272; `fanny_objective` 33.28123392479843.
- c-means from (0,0), (3,0), (1.5,2.5): `cmeans_centers` (-0.03192713197410528, -0.026196593233907987),
  (1.4172334737645325, 2.5575435255668855), (2.993554332550177, 0.055658804018355906); `cmeans_withinerror`
  0.45453513290186937, which is e1071's `sum u^2 d^2 / n`.

## Tolerances

Medoids, medoid assignments and hard labels exact; totals and silhouettes 1e-12 (arithmetic at fixed medoids).

## Known differences

- **PAM SWAP.** p2a takes the first improving swap and restarts; `cluster::pam` takes the best swap per step. On the
  case's row order both reach the same medoids for k = 2, 3, 4 (`stopifnot` in the case). On permuted rows the
  written-out p2a SWAP stopped at other medoids in 30 of 100 permutations for k = 2 and 44 of 100 for k = 4, never for
  k = 3 (0 of 200). CLARA is PAM on a shuffled sample, so it is pinned at k = 3 only. No claim of PAM equivalence holds
  beyond the pinned data.
- **Fuzzy c-means (ML-9).** The exponent `2/(m-1)` is applied to squared distances, so centres, memberships and the
  objective differ from `e1071::cmeans`. The hard partition agrees and is pinned (20 of 20 random starts reach it).
- **FANNY (ML-11).** Not `cluster::fanny`'s Kaufman-Rousseeuw iteration: memberships, coefficients, objective and
  clustering differ. `fuzzy_reference_blobs` holds the references for the fix.

## Not R-pinned

| Method | Reason |
|---|---|
| Spectral clustering | ML-44 (power iteration) and ML-112 (eigenvalue label) will change the algorithm; `kernlab::specc` is installed for that fix. |
| k-prototypes | `clustMixType::kproto` is installed; no finding names a pin. Seeded runs are reproducible, and mode ties go to the smallest category. |
| Spherical k-means | `skmeans` is installed; a pin needs a directional dataset. |
| Mini-batch and trimmed k-means | No R implementation is installed (`ClusterR`, `tclust`, `trimcluster` absent). Property tests only. |

## Tests

R-pinned: `test_validate_pam_blobs`, `test_validate_clara_full_sample_is_pam`, `test_validate_fcm_cmeans_blobs_labels`.

Properties: `test_kmedoids_result_does_not_depend_on_the_seed`, `test_clara_default_seed_is_reproducible`,
`test_kmedoids_rejects_bad_parameters`, `test_fuzzy_cmeans_stopping_rule_is_scale_free`,
`test_fcm_distance_parses_aliases_and_rejects_unknown`,
`test_fuzzy_methods_reject_bad_parameters_and_default_seeds_are_fixed`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- advanced_clustering_mod::kmedoids::tests::test_validate advanced_clustering_mod::fuzzy::tests::test_validate
```
