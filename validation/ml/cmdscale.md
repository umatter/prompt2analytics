# Validation: Classical Multidimensional Scaling (cmdscale)

## Method

Classical (metric) multidimensional scaling, also known as Principal Coordinates Analysis (PCoA)
(`cmdscale`, `cmdscale_from_data`, `crates/p2a-core/src/ml/reduction.rs`):

1. square the distance matrix;
2. double-centre: B = -0.5 * H * D² * H with H = I - (1/n)11';
3. eigendecomposition of B (faer's self-adjoint solver since T0-WP16; eigenvalues sorted algebraically, as R);
4. the top k eigenvectors scaled by sqrt(eigenvalue) are the points.

`gof` is R's `GOF`: the sum of the positive eigenvalues among the first k over the sum of the absolute
eigenvalues, and over the sum of the positive eigenvalues. `add = Some(true)` is R's `add = TRUE`: Cailliez's
constant `ac`, the largest real eigenvalue of the 2n x 2n matrix `[[0, -DC(D^2)], [-I, DC(2D)]]` (`DC` the double
centring, faer's general eigensolver where R uses LAPACK `dgeev`), is added to every off-diagonal distance before
steps 1-4; the matrix is pre-flighted against the allocation ceiling (O((2n)^3) work, library only: MCP passes no
`add`). `eig` is accepted for R compatibility; all n eigenvalues are always returned. Unlike R, which stops,
`k` is clamped to n - 1.

Point coordinates are unique up to sign: compare absolute values.

## R code

`validation/r/cases/ml_hierarchical.R`, case `kmeans_lloyd_pca_mds` (R 4.3.3): `cmdscale(eurodist, k = 2,
eig = TRUE)` (the 21 x 21 matrix is `validation/datasets/generated/eurodist_matrix.csv`), and
`cmdscale(dist(c(0, -1, 1)), k = 1, eig = TRUE)`, three points on a line whose middle point is the centroid.

`validation/r/cases/ml_reduction_t4.R`, case `cmdscale_gof_add` (R 4.3.3): `cmdscale(eurodist, k = 2, eig = TRUE)`
with `add = FALSE` and `TRUE`, and `cmdscale(dist(mds_points8, "manhattan"), k = 2 | 3, eig = TRUE[, add = TRUE])`
(the eight points of `test_validate_cmdscale_matches_r`, `generated/mds_points8.csv`; the Manhattan distances
are not Euclidean, so the two GOF measures differ). Keys `<tag>_gof`, `<tag>_ac`, `<tag>_points_abs`,
`<tag>_eig` for the tags `euro_k2`, `euro_k2_add`, `manh_k2`, `manh_k3_add`.

## Values

| Key | R |
|---|---|
| `mds_eig` | 19538377.089542832, 11856555.334001094, 1528844.4679873697, ... (21 values, one null eigenvalue -3.7e-09) |
| `mds_points_abs` | `abs(points)`, column-major (first coordinate of Athens 2290.2746796314523, of Barcelona 825.3827903533329) |
| `mds_line_points_abs` | 0, 1, 1 |
| `mds_line_eig` | 2, 0, 0 |
| `mds_gof` | 0.7537543155079838, 0.8679134296478231 |
| `euro_k2_add_ac`, `euro_k2_add_gof` | 2132.6784951979516; 0.5115564107430031, 0.5115564107430032 |
| `manh_k2_gof` | 0.8401393234736658, 0.9188986857234781 |
| `manh_k3_add_ac`, `manh_k3_add_gof` | 2.5126651550583388; 0.8905833583379315, 0.8905833583379316 |

Before T0-WP16 the power iteration gave eurodist eigenvalues 19538377, 11856555, -2251844, 1528844, ... (not
sorted algebraically), an all-zero configuration for the line, and collapsed at small scale (ML-36).

ML-98 (fixed in Tier 4): `gof` was `[sum_pos / sum_abs, sum_k / sum_abs]` (eurodist 0.8685, 0.7538), and `add`
was accepted and ignored.

## Tests and tolerances

| Test (`crates/p2a-core/src/ml/reduction.rs`) | Keys | Tolerance |
|---|---|---|
| `test_validate_cmdscale_eurodist` | `mds_points_abs`, `mds_eig`, `mds_gof` | points and GOF `Tol::coef(21)`; eigenvalues 1e-9 relative, 1e-6 absolute (the null eigenvalue is roundoff) |
| `test_validate_cmdscale_gof_add` | `cmdscale_gof_add`: every tag's GOF, `ac`, points, eigenvalues | GOF, `ac`, points `Tol::coef(n)` (n = 21, 8); eigenvalues 1e-9 relative with 1e-6 absolute (eurodist), 1e-5 absolute (eurodist with `add`: the null eigenvalue moves with the roundoff of `ac`, 6.7e-10 here), 1e-9 absolute (Manhattan) |
| `test_cmdscale_add_preflights_its_matrix` | the 2n x 2n matrix of `add` under an allocation ceiling; cancellation | property |
| `test_validate_cmdscale_line_configuration` | `mds_line_points_abs`, `mds_line_eig` | 1e-12 |
| `test_validate_cmdscale_is_scale_equivariant` | (data x 2^-17: eigenvalues and points rescale) | property (data from case `kmeans_lloyd_pca_mds`), not an R comparison; 1e-9 relative |
| `test_validate_cmdscale_matches_r` (T0-WP16) | R 4.3.3 values typed into the test: 8 x 2 points and eight eigenvalues, Euclidean and Manhattan | 1e-9 absolute; a quoted-value check |
| `test_cmdscale_basic`, `test_cmdscale_from_data`, `test_cmdscale_preserves_euclidean_distances`, `test_cmdscale_gof`, `test_cmdscale_validation_non_square` and the other `test_cmdscale_validation_` tests | p2a properties | |

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::reduction::tests::test_validate_cmdscale ml::reduction::tests::test_cmdscale
```

## MCP tool

`ml_cmdscale` with `is_distance_matrix: true` on the 21 eurodist columns, `k: 2`: the JSON `gof.gof1`, `gof.gof2`
equal `euro_k2_gof` (`Tol::coef(21)`). Test: `test_validate_mcp_cmdscale_gof`
(`crates/p2a-mcp/tests/tier3/wp57_ml_spatial.rs`). The JSON keys were `gof1_positive_eigenvalues` and
`gof2_k_eigenvalues` before ML-98.

## Timing

Release builds of v0.5.0 (`f80b0131`) and of this change with `--no-default-features --features ci`, one core
(`taskset -c 2`, `RAYON_NUM_THREADS=1`), base and head processes interleaved, each process the median of 51 / 9 / 3
calls at n = 100 / 1,000 / 10,000 after a warm-up, the table the median over 5 processes (3 for the single-call
rows); peak RSS from `/usr/bin/time -v`, one fit per process, the smallest of 3. Intel Core 7 150U, machine load
10-20 from other jobs (preliminary; the integration run repeats it).

`cmdscale_from_data` (k = 2) on standard-normal data in 3 dimensions; `add = TRUE` is new and is timed against
`add = FALSE` on the same Manhattan distances (head only).

| Run | n | v0.5.0 | head | head / v0.5.0 |
|---|---|---|---|---|
| default | 100 | 1.96 ms | 1.96 ms | 1.00 |
| default | 1,000 | 0.317 s | 0.324 s | 1.02 |
| `add = FALSE` | 100 | | 1.9 ms | |
| `add = TRUE` | 100 | | 31.5 ms | |
| `add = FALSE` | 1,000 | | 0.44 s | |
| `add = TRUE` | 1,000 | | 8.2 s | |

Peak RSS at n = 1,000: 41.3 MB against 41.3 MB (default); 52.9 MB (`add = FALSE`) and 93.8 MB (`add = TRUE`, the
2n x 2n matrix of the additive constant).

## References

- Torgerson, W. S. (1952). "Multidimensional scaling: I. Theory and method". *Psychometrika*, 17, 401-419.
- Gower, J. C. (1966). "Some distance properties of latent root and vector methods used in multivariate analysis". *Biometrika*, 53, 325-338.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cmdscale.html
