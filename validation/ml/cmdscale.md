# Validation: Classical Multidimensional Scaling (cmdscale)

## Method

Classical (metric) multidimensional scaling, also known as Principal Coordinates Analysis (PCoA)
(`cmdscale`, `cmdscale_from_data`, `crates/p2a-core/src/ml/reduction.rs`):

1. square the distance matrix;
2. double-centre: B = -0.5 * H * D² * H with H = I - (1/n)11';
3. eigendecomposition of B (faer's self-adjoint solver since T0-WP16; eigenvalues sorted algebraically, as R);
4. the top k eigenvectors scaled by sqrt(eigenvalue) are the points.

Point coordinates are unique up to sign: compare absolute values.

## R code

`validation/r/cases/ml_hierarchical.R`, case `kmeans_lloyd_pca_mds` (R 4.3.3): `cmdscale(eurodist, k = 2,
eig = TRUE)` (the 21 x 21 matrix is `validation/datasets/generated/eurodist_matrix.csv`), and
`cmdscale(dist(c(0, -1, 1)), k = 1, eig = TRUE)`, three points on a line whose middle point is the centroid.

## Values

| Key | R |
|---|---|
| `mds_eig` | 19538377.089542832, 11856555.334001094, 1528844.4679873697, ... (21 values, one null eigenvalue -3.7e-09) |
| `mds_points_abs` | `abs(points)`, column-major (first coordinate of Athens 2290.2746796314523, of Barcelona 825.3827903533329) |
| `mds_line_points_abs` | 0, 1, 1 |
| `mds_line_eig` | 2, 0, 0 |
| `mds_gof` | 0.7537543155079838, 0.8679134296478231 (recorded, not compared) |

Before T0-WP16 the power iteration gave eurodist eigenvalues 19538377, 11856555, -2251844, 1528844, ... (not
sorted algebraically), an all-zero configuration for the line, and collapsed at small scale (ML-36).

**Deviation ML-98** (Tier 4): p2a's `gof` is `[sum_pos / sum_abs, sum_k / sum_abs]`; R's GOF is
`sum_k / c(sum(abs(ev)), sum(pmax(ev, 0)))`. GOF is not compared.

## Tests and tolerances

| Test (`crates/p2a-core/src/ml/reduction.rs`) | Keys | Tolerance |
|---|---|---|
| `test_validate_cmdscale_eurodist` | `mds_points_abs`, `mds_eig` | points `Tol::coef(21)`; eigenvalues 1e-9 relative, 1e-6 absolute (the null eigenvalue is roundoff) |
| `test_validate_cmdscale_line_configuration` | `mds_line_points_abs`, `mds_line_eig` | 1e-12 |
| `test_validate_cmdscale_is_scale_equivariant` | (data x 2^-17: eigenvalues and points rescale) | property (data from case `kmeans_lloyd_pca_mds`), not an R comparison; 1e-9 relative |
| `test_validate_cmdscale_matches_r` (T0-WP16) | R 4.3.3 values typed into the test: 8 x 2 points and eight eigenvalues, Euclidean and Manhattan | 1e-9 absolute; a quoted-value check |
| `test_cmdscale_basic`, `test_cmdscale_from_data`, `test_cmdscale_preserves_euclidean_distances`, `test_cmdscale_gof`, `test_cmdscale_validation_non_square` and the other `test_cmdscale_validation_` tests | p2a properties | |

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::reduction::tests::test_validate_cmdscale ml::reduction::tests::test_cmdscale
```

## References

- Torgerson, W. S. (1952). "Multidimensional scaling: I. Theory and method". *Psychometrika*, 17, 401-419.
- Gower, J. C. (1966). "Some distance properties of latent root and vector methods used in multivariate analysis". *Biometrika*, 53, 325-338.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cmdscale.html
