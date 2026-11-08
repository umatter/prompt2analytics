# Validation: affinity propagation (apcluster::apcluster)

## Method Overview

`p2a_core::ml::affinity_propagation()` (`crates/p2a-core/src/ml/advanced_clustering_mod/affinity.rs`) is a port of
`apcluster::apcluster(negDistMat(r = 2), q = 0.5)`: similarities are negative squared distances, the preference
defaults to the median of the off-diagonal similarities (found by selection), and responsibilities and availabilities
are damped as in Dueck and Frey.

**Convergence.** The exemplar set is kept in a ring buffer of `convergence_iterations` entries. The fit converges when
the set has been unchanged over the whole buffer and is not empty. The former loop also counted an empty set as stable
and stopped there with no exemplars.

**Refinement.** After the last iteration each point joins its most similar exemplar, each cluster's exemplar becomes
the member with the largest summed similarity, and points are assigned again.

**No exemplar.** A run that ends with no exemplars returns an error naming the limits. apcluster returns an empty
result (it does so at `maxits = 20` on these data). `Display` of an empty result does not fail.

**Defaults.** p2a keeps 0.5 / 200 / 15 (sklearn's) against apcluster's 0.9 / 1000 / 100, which would cost at least 100
iterations per run. This is a convention, recorded here and in the registry. The damping must lie in [0.5, 1).

## Reference

`apcluster::apcluster` (R 4.3.3, apcluster 1.4.14) with `nonoise = TRUE`.

## Where

R code `validation/r/cases/ml_mixtures.R`, case `apcluster_blobs`, dataset
`validation/datasets/generated/ap_blobs.csv` (n = 115). Registry `validation/methods/ml_mixtures.json`, method
`ml_affinity_propagation`.

## Values

| | `lam9` (0.9; 2000/100) | `lam9_default` (0.9; 200/15) |
|---|---|---|
| exemplars (1-based) | 18, 48, 65, 83, 106, 109, 111 | 18, 53, 83, 106, 109, 111 |
| `n_clusters` | 7 | 6 |
| `netsim` | -84.37889891843788 | -82.99997087288662 |
| `converged` | 1 | 1 |

`preference` is -7.4802872266721785. Iteration counts, the iteration of a removal and non-converged states are not
R-pinned. The test's own 6-point data (`p = -100`, 0.9, 200/15) gives exemplars 2 and 5 in `apcluster`.

**`lam5` is recorded, not pinned.** `apcluster(lam = 0.5, maxits = 2000, convits = 100)` does not converge on these data
and stops at `maxits` (even at 6000). That oscillation depends on the last bit of the preference: apcluster's value and
the value one ulp below it (-7.480287226672179, what direct squared distances give) lead to oscillation and to
convergence after 151 passes, with the similarities of either route. p2a takes the lower value, so it converges here
where apcluster does not. Non-convergence is
tested as a property: `test_affinity_propagation_reports_the_iteration_limit` uses a window longer than the limit,
which no rounding can fill.

## Tolerances

Exemplars, `exemplar_of` and `n_clusters` exact; `netsim` and `preference` 1e-12 relative, since they are arithmetic on
the input at a fixed exemplar set.

## Known differences

- `nonoise`: p2a adds no noise to the similarities, so the case sets `nonoise = TRUE`.
- No exemplars: an error where apcluster returns an empty result.
- Defaults, above.
- Squared distances are summed directly; `negDistMat(r = 2)` squares a `sqrt`. The results at a converged fixed point
  agree (pinned); `lam5` above is where the last bit shows.

## Tests

R-pinned: `test_validate_apcluster_lam9`, `test_validate_apcluster_lam9_defaults`.

Properties: `test_affinity_propagation_basic`, `test_affinity_propagation_reports_the_iteration_limit`, `test_affinity_propagation_no_exemplar_is_an_error`,
`test_affinity_propagation_display_without_clusters_does_not_panic`,
`test_median_similarity_equals_the_sorted_type_7_median`, `test_affinity_propagation_rejects_bad_parameters`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- advanced_clustering_mod::affinity::tests::test_validate
```
