# Validation: K-Means Clustering (Lloyd)

## Method

`p2a_core::ml::kmeans` and `kmeans_from_centers` (`crates/p2a-core/src/ml/clustering.rs`) run Lloyd
iterations, R's `kmeans(x, centers, algorithm = "Lloyd")` (not R's default Hartigan-Wong):

1. assign each point to its nearest centroid (the first on ties, R's `dd < best`);
2. stop when no point changed cluster (the first pass always counts as a change);
3. move each centroid to the mean of its points; repeat, at most `max_iterations` passes (default 300).

An explicit `tolerance` > 0 stops earlier, when the sum of the squared centroid shifts is at most
`tolerance` times the mean column variance (scikit-learn's relative `tol`), and is followed by a final
assignment pass, so the labels are always those of the returned centroids (that pass is not counted in
`n_iterations`, which never exceeds `max_iterations`). Both rules are scale-free.
`tolerance = None` (the MCP and CLI default) is 0: until no point is reassigned. A run that reaches the
limit has `convergence.converged == false` and a warning ("k-means (Lloyd) did not converge after ...").

`kmeans` starts each of `n_init` runs from k-means++ and keeps the smallest inertia (the first on ties).
`kmeans_from_centers` starts from given centres, as R's `centers =` matrix: the centres must be finite,
have the data's columns and be distinct ("initial centers are not distinct"), and a cluster that empties is
an error ("empty cluster: try a better set of initial centers"), both R's texts.

Iteration counts and the state at the iteration limit (an intermediate value) are not R-pinned (re-scope
rule 2): the case records them (`km_unit_iter`, `km_small_iter`, `km_iter2_*`) but no test compares them.

## R code

`validation/r/cases/ml_hierarchical.R`, case `kmeans_lloyd_pca_mds` (R 4.3.3), on
`validation/datasets/generated/hclust_60.csv`: `kmeans(X, centers = X[c(1, 21, 41), ], algorithm = "Lloyd")`,
the same on `X * 1e-5`, and with `iter.max = 2`.

## Values

| Key | R |
|---|---|
| `km_unit_cluster` | three groups of 20 |
| `km_unit_tot_withinss` | 152.68291992468085 |
| `km_small_cluster` | three groups of 20 (the same partition) |
| `km_small_tot_withinss` | 1.5268291992468086e-08 (= 152.68291992468085 x 1e-10) |
| `km_unit_centers` | 3 x 3, column-major |

## Tests and tolerances

| Test (`crates/p2a-core/src/ml/clustering.rs`) | Keys | Tolerance |
|---|---|---|
| `test_validate_kmeans_unit_scale` | `km_unit_cluster`, `km_unit_centers`, `km_unit_tot_withinss` | labels exact; centres `Tol::coef(60)`; tot.withinss `Tol::stat(60)` |
| `test_validate_kmeans_small_scale` | `km_small_cluster`, `km_small_tot_withinss` | labels exact; `Tol::stat(60)` |
| `test_validate_kmeans_not_converged` | (iteration limit 2: not converged, warning, inertia at least the converged run's) | property (data from case `kmeans_lloyd_pca_mds`), not an R comparison |
| `test_validate_kmeans_scale_invariant` | (x 2^-20 and x 2^20: same labels and passes, with and without a tolerance) | property (data from case `kmeans_lloyd_pca_mds`), not an R comparison; exact |
| `test_validate_kmeans_labels_match_final_centroids` | (every label is a nearest returned centroid) | property (data from case `kmeans_lloyd_pca_mds`), not an R comparison; exact |
| `test_kmeans_from_centers_validates`, `test_kmeans_tolerance_stop_counts_at_most_max_iterations`, `test_kmeans_plusplus_start_feeds_lloyd`, `test_kmeans_separates_three_blobs`, `test_kmeans_inertia_decreases_with_k`, `test_kmeans_same_seed_same_result` | p2a properties | |

## Missing and non-finite values

- **Core.** `kmeans` refuses NaN and ±Inf before it initialises and names the first bad
  cell in row-major order: "k-means: data must contain only finite values (row 4, column 0 is
  NaN); remove or impute missing values first". It used to report "K-means failed to
  converge" (every inertia was NaN). R stops as well: `kmeans(m, 2)` on a matrix with an
  `NA` gives "NA/NaN/Inf in foreign function call (arg 1)" (R 4.3.3).
- **MCP and CLI.** `ml_kmeans` and `p2a ml kmeans` read their columns through the shared
  extractor with the Fail policy: a missing value (null or NaN) is an error naming the
  column ("Column 'x1_na' contains 1 missing value(s) (null or NaN) ..."; the MCP text adds
  that `munge_drop_na` or `munge_fill_na` can remove or impute the rows), a text column is
  refused by name, and integer and boolean columns are numeric.
- **Tests.** `test_na_kmeans_refuses_non_finite_input` (`crates/p2a-core/src/ml/clustering.rs`),
  `test_na_mcp_ml_matrix_tools_name_the_column` and
  `test_mcp_ml_matrix_tools_int64_equals_float64` (`crates/p2a-mcp/src/tools/handlers/ml.rs`),
  `test_cli_ml_integer_columns_accepted` (`crates/p2a-cli/tests/tier2/wp17_ml.rs`). They
  compare p2a with p2a and pin no R number.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::clustering::tests::test_validate_kmeans ml::clustering::tests::test_kmeans
```

## References

- Lloyd, S.P. (1982). "Least Squares Quantization in PCM". *IEEE Transactions on Information Theory*, 28(2), 129-137.
- R `stats::kmeans`; scikit-learn `KMeans` (`tol`).
