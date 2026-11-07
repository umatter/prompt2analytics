# Validation: Gaussian mixtures (mclust::Mclust)

## Method Overview

`p2a_core::ml::gaussian_mixture()` and `advanced_clustering_mod::gmm::gaussian_mixture_with()`
(`crates/p2a-core/src/ml/advanced_clustering_mod/gmm.rs`) fit a Gaussian mixture by EM. The four covariance types are
mclust's models:

| `CovarianceType` | mclust | Parameters at d = 3, k = 3 |
|---|---|---|
| `Full` | VVV | 29 |
| `Tied` | EEE (one covariance, pooled over the components) | 17 |
| `Diagonal` | VVI | 20 |
| `Spherical` | VII | 14 |

Every density goes through a Cholesky factor, so the log-determinant and the Mahalanobis term are exact in any
dimension. `Tied` factors its one covariance once per M-step.

**Start.** `GmmInit::KMeans` (the default), `Classification` or `Parameters`. The k-means start runs on the data
divided by the root mean column variance, with its own tolerance `1e-4`, so it does not depend on the units. mclust's
default start is hierarchical; both reach the pinned optimum (`stopifnot` over 10 k-means seeds in the case).

**Stopping rule.** `relative_change(l, l_prev) < tolerance`, default `1e-8` (`GMM_DEFAULT_TOLERANCE`). mclust stops at
`|dl| / (1 + |l|) < 1e-5`, which leaves the parameters about 7e-3 from the optimum on these data; the former p2a rule
`|dl| < 1e-4` is relative 7e-8 at `|l| = 1442`. A fit that reaches `max_iterations` reports `converged = false` and a
warning. The default `max_iterations` stays 100 unless a timing row says otherwise; see the docs of `gaussian_mixture`.

**Singular covariance.** A component's covariance counts as singular when a squared pivot of its Cholesky factor falls
below `SYM_RANK_TOL * d` times its diagonal entry. It then gets a ridge of `GMM_SINGULAR_RIDGE = 1e-10` times each
column's variance (the mean variance for a constant column) and a warning; mclust returns `NA` for the whole model.
Both the check and the ridge are per column, so neither depends on the units of a column.

**Scale equivariance.** Multiplying the data by c multiplies means by c, covariances by c squared, shifts the
log-likelihood by `-n d ln c` and leaves the proportions and the classification in place. The former absolute ridge and
thresholds did not (`test_gmm_is_scale_equivariant`).

## Reference

`mclust::Mclust(x, G = 3, modelNames = ...)`, with `mclust::mstep` / `estep` iterated to a tight fixed point (R 4.3.3,
mclust 6.1.3).

## Where

R code `validation/r/cases/ml_mixtures.R`, case `gmm_mclust_3d`, dataset `validation/datasets/generated/mixture_3d.csv`
(n = 300, d = 3). Components are ordered by the first mean. Registry `validation/methods/ml_mixtures.json`, method
`ml_gmm`.

## Values

| | VVV (Full) | EEE (Tied) | VVI (Diagonal) | VII (Spherical) |
|---|---|---|---|---|
| `*_df` | 29 | 17 | 20 | 14 |
| `*_tight_loglik` | -1442.6147212071528 | -1482.1551353150792 | -1481.113071174032 | -1489.8173644464891 |
| `*_tight_bic` (mclust's sign) | -3050.6391341793355 | -3061.274572699314 | -3076.301791841188 | -3059.487683538165 |
| `*_loglik` (Mclust, tol 1e-12) | -1442.6147212073045 | -1482.1551353151042 | -1481.1130711746373 | -1489.817364446648 |

- `VVV_tight_pro` 0.3950870265949487, 0.2671793245795005, 0.33773364882555096.
- `EEE_tight_sigma`, the same matrix for all three components: 0.741180167413247, 0.248001431191691,
  0.10686952831976484, 0.8765022057649102, 0.2017208813581348, 0.8444233775252769 (upper triangle, row-wise).

## Tolerances

EM stops on a log-likelihood rule, about 1e-6 from the fixed point in the parameters, so the tests compare with the
tight values:

- `EM_PARAM`: parameters 1e-5 relative. With p2a's rule at 1e-13 and mclust's steps the parameters end within 2.2e-7
  (VVV), 2.1e-7 (EEE), 1.64e-6 (VVI) and 3.3e-7 (VII) of the tight values.
- `EM_LOGLIK`: log-likelihood 1e-12 relative (measured 6.3e-11 absolute).
- `FIXED_POINT`: 1e-10 relative, started at the tight values (`test_validate_gmm_mclust_fixed_point`).
- `df` and classifications exact.

## Known differences

- The start (k-means, not `hc`) and the stopping rule above. Iteration counts are not pinned.
- Singular covariance: ridge and warning where mclust reports `NA`.

## Tests

R-pinned: `test_validate_gmm_mclust_vvv`, `test_validate_gmm_mclust_eee`, `test_validate_gmm_mclust_vvi_vii`,
`test_validate_gmm_mclust_fixed_point`.

Properties: `test_gmm_component_log_pdf_matches_the_closed_form`, `test_gmm_tied_pools_one_covariance`,
`test_gmm_is_scale_equivariant`, `test_gmm_singular_component_gets_a_relative_ridge`,
`test_gmm_singularity_check_does_not_depend_on_column_units`, `test_gmm_rejects_bad_parameters`,
`test_covariance_type_parses_aliases_and_rejects_unknown`, `test_gmm_display_names_the_model_and_notes`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- advanced_clustering_mod::gmm::tests::test_validate
```
