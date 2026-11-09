# Validation: Principal Component Analysis (PCA)

## Method

`pca` (`crates/p2a-core/src/ml/reduction.rs`) centres the data and takes the eigendecomposition of the
covariance matrix (n - 1 denominator) or, for n <= 4p, the SVD of the centred data; components are sorted by
decreasing variance. `explained_variance` is R's `prcomp()$sdev^2`; `components` (rows = PCs) is R's
`rotation` transposed. Loadings are unique up to sign: compare absolute values.

## R code

`validation/r/cases/ml_hierarchical.R`, case `kmeans_lloyd_pca_mds` (R 4.3.3):
`prcomp(X, scale. = TRUE)` and `prcomp(X)` on `validation/datasets/generated/hclust_60.csv` (60 x 3). The
scaled run passes the standardised data to `pca`.

## Values

| Key | R |
|---|---|
| `pca_sdev` | 1.3221524880813766, 0.9819157249713941, 0.5364273551135557 |
| `pca_var_ratio` | 0.5826957339132586, 0.3213861636486997, 0.09591810243804165 |
| `pca_unscaled_sdev` | 2.6485861925208094, 1.9008767308107948, 0.8828277799830404 |
| `pca_unscaled_var_ratio` | 0.6149348946467286, 0.316744312461122, 0.06832079289214951 |
| `pca_rotation_abs`, `pca_unscaled_rotation_abs` | `abs(rotation)`, column-major |

**Deviation ML-35** (Tier 4): with `n_components < p` on the SVD path (n <= 4p), `explained_variance_ratio` is
relative to the kept components only. The pins use all components on the covariance path (n = 60 > 4p).

## Tests and tolerances

| Test (`crates/p2a-core/src/ml/reduction.rs`) | Keys | Tolerance |
|---|---|---|
| `test_validate_prcomp` | `pca_*`, `pca_unscaled_*` | sdev, ratio `Tol::stat(60)`; rotation `Tol::coef(60)` |
| `test_pca_n_components_shapes`, `test_pca_components_orthonormal`, `test_pca_transformation_matches_projection`, `test_pca_n_components_bound`, `test_pca_deterministic`, `test_pca_centering` | p2a properties | |

## Missing and non-finite values

- **Core.** `pca` refuses NaN and ±Inf and names the first bad cell in row-major order:
  "PCA: data must contain only finite values (row 4, column 0 is NaN); remove or impute
  missing values first". It used to return NaN variances (or an SVD failure). R stops as
  well: `prcomp(m)` on a matrix with an `NA` gives "infinite or missing values in 'x'"
  (R 4.3.3).
- **MCP and CLI.** `ml_pca` and `p2a ml pca` read their columns with the Fail policy: a
  missing value is an error naming the column, a text column is refused by name ("Column
  'lab' contains non-numeric values ..."), and integer columns give the same result as the
  same values stored as decimals.
- **Tests.** `test_na_pca_refuses_non_finite_input` (`crates/p2a-core/src/ml/reduction.rs`),
  `test_na_mcp_ml_matrix_tools_name_the_column` and
  `test_mcp_ml_matrix_tools_int64_equals_float64` (`crates/p2a-mcp/src/tools/handlers/ml.rs`),
  `test_na_cli_ml_missing_values_refused` (`crates/p2a-cli/tests/tier2/wp17_ml.rs`). They
  compare p2a with p2a and pin no R number.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::reduction::tests::test_validate_prcomp ml::reduction::tests::test_pca
```

## MCP tools

`ml_pca` on `hclust_60` with `{columns a, b, c, output_column: PC}`, and with `{n_components: 2,
output_column: Q}`, case `mcp_pca_scores` (`validation/r/cases/mcp_ml_spatial.R`). Keys
`pca_x_pc1` to `pca_x_pc3` (one sign per component) and `pca_sdev` (the explained variance is
`sdev^2`), `Tol::coef(60)`. Test: `test_validate_mcp_pca_output_column`
(`crates/p2a-mcp/tests/tier3/wp57_ml_spatial.rs`).
MCP text is compared at print precision (`Tol::printed`: 6 decimals, z 4 decimals), JSON blocks at the
[validation/README.md](../README.md) table for the case's n (`Tol::coef(n)`), labels exactly. Registry:
`validation/methods/mcp_ml_spatial.json`.

## References

- Jolliffe, I.T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
