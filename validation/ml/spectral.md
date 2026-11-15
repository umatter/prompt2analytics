# Validation: spectral clustering (Ng-Jordan-Weiss, kernlab::specc's embedding)

## Method Overview

`spectral_clustering` (`crates/p2a-core/src/ml/advanced_clustering_mod/spectral.rs`):

1. The affinity A: RBF `exp(-gamma |x_i - x_j|^2)` (default `gamma = 1 / (p * mean column variance)`), the
   symmetrised k-nearest-neighbour graph with weight 1, or a precomputed matrix (finite, non-negative and
   symmetric, otherwise refused); zero diagonal for RBF and k-NN.
2. M = D^(-1/2) A D^(-1/2) with D the degrees; a point of degree 0 has D^(-1/2) = 0.
3. The k largest eigenvalues of M and their eigenvectors, exactly to solver tolerance. The graph's connected
   components are solved separately: dense (faer's self-adjoint eigendecomposition) for up to 64 points, otherwise
   Krylov-Schur (faer's partial eigensolver) on (I + M) / 2, whose spectrum lies in [0, 1], at tolerance
   128 eps with a seeded start, falling back to the dense solver when it returns fewer pairs. Each component
   contributes its eigenvalue 1 first; the rest are the largest remaining eigenvalues over all components.
4. The rows of the eigenvector matrix are normalised and clustered by k-means (Lloyd to convergence, 10 starts).

`eigenvalues` are those of L_sym = I - M, ascending (`1 - mu`).

## Reference

`kernlab::specc` (kernlab 0.9.33) was read for the algorithm; the case computes the NJW embedding in R with
`eigen(symmetric = TRUE)` (R 4.3.3), since `specc` itself clusters with a random k-means start.

## Where

R code `validation/r/cases/ml_cluster_t4.R`, case `spectral_njw`, datasets
`validation/datasets/generated/ap_blobs.csv` (n = 115) and `validation/datasets/generated/spectral_pair.csv`
(n = 42, two disconnected groups at `gamma = 0.5`). Registry `validation/methods/ml_cluster_t4.json`, method
`ml_spectral_t4`.

## Values

| Key | Value |
|---|---|
| `blobs_gamma` | 0.2657154227720211 |
| `blobs_lsym` | 1.2212453270876722e-15, 0.2856190337112784, 0.32132742193641406 |
| `pair_k2_mu` | 1.0000000000000002, 0.9999999999999989 (two components) |
| `pair_k3_mu` | 1.0000000000000002, 0.9999999999999989, 0.9969535540526746 |
| `blobs_cluster`, `pair_k2_cluster`, `pair_k3_cluster` | the k-means labels of the embedding (first-appearance order) |

## Tolerances

`gamma` and the L_sym eigenvalues `Tol::coef(n)`; the pair's eigenvalues 1e-6 relative plus 1e-12 absolute (an
eigenvalue 0 of L_sym has no relative scale); labels exact.

## Known differences

- A precomputed affinity that is negative or asymmetric is refused; `specc` would use it.
- A point of degree 0 gets D^(-1/2) = 0 (R gives NaN and fails).

## Tests

R-pinned: `test_validate_spectral_njw_blobs`, `test_validate_spectral_njw_pair`. Properties:
`test_spectral_knn_weights_are_binary`, `test_spectral_partial_and_dense_agree`,
`test_spectral_precomputed_must_be_symmetric_and_nonnegative`, `test_spectral_clustering_is_cancellable`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- advanced_clustering_mod::spectral
```
