# Spatial weight styles and the spectrum of W

`crates/p2a-core/src/spatial/weights.rs`, `spatial/neighbors.rs`. Registry entries `spatial_weights` and
`spatial_logdet` in `validation/methods/spatial_ml_weights.json`.

R-pinned to the formulas of `spdep::nb2listw(style, zero.policy = TRUE)` and `spatialreg` method
"eigen", written out in base R because neither package is installed: cases `spatial_weights_moran`
and `spatial_sar_sdm_sem` of `validation/r/cases/spatial_ml_weights.R`, dataset
`validation/datasets/generated/spatial_sim.csv` (n = 80).

## Styles

g_ij is the binary or given weight; eff.n = n - #islands under `zero.policy = TRUE`.

| p2a | spdep style | Weight | Sum on the (0, 0.12] band |
|---|---|---|---|
| `RowStd` | W | g_ij / Σ_j g_ij | 80 |
| `Binary` | B | g_ij | 672 |
| `GlobalStd` | C | g_ij eff.n / Σ g | 80 |
| `UnitSum` | U | g_ij / Σ g | 1 |
| `MinMax` | minmax | g_ij / min(max row sum, max column sum) | 42 |
| `VarStab` | S | q_ij eff.n / Σ q, q_ij = g_ij / sqrt(Σ_j g_ij²) | 80 |

On the band with its four islands, C and S sum to 76 (eff.n) and U sums to 1. Weights from a decay
function have style `Custom`: they are neither binary nor row-standardised.

`WeightStyle` parses spdep's letters and the long names (`WeightStyle::choices()`); an unknown
spelling is `InvalidSpecification`.

## Neighbours

k-nearest-neighbour ties are broken by the lower index. That equals
`spdep::knearneigh(x, k, use_kd_tree = FALSE)`. The default kd-tree search (dbscan) can order exact
distance ties differently. Case key `knn_grid_ties` pins the 5 x 5 integer grid at k = 4.

## Spectrum, log-determinant and rho interval

- `try_spectrum` computes the eigenvalues of W once. Both complex parts are kept.
- log|I - ρW| = Re Σ log(1 - ρλ_i) over the complex eigenvalues. The former sum over real parts gave
  -2.35577 at ρ = 0.5 where the determinant is -2.33829.
- `try_rho_range` is `1/range(Re(eig))`, as spatialreg's `eig.range`: (-2.009, 1) on the case W.
- `WeightsSpectrum` also gives tr((I - ρW)^-1), tr(W (I - ρW)^-1), tr((W (I - ρW)^-1)²) and
  min |1 - ρλ|.
- Non-finite weights or a failed decomposition are `EconError::Computation` from the `try_*` methods
  and NaN from the infallible ones; they are never zeros.

## Pinned values

| Quantity | R |
|---|---|
| log|I - ρW| at ρ = -0.8, 0.5, 0.9 | -3.94802, -2.33829, -12.53161 |
| Complex eigenvalues (|Im| > 1e-4) | 20 |
| tr((I - 0.5 W)^-1) | 85.506543 |
| tr(W (I - 0.5 W)^-1) | 11.013086 |
| Moran's I, styles W, B, C, U, minmax, S | -0.014286, 0.011544, 0.011544, 0.011544, 0.011544, 0.002011 |

The log-determinant is also checked against `base::determinant` inside the case.

Tolerances from `validation/README.md` at n = 80: weights 1e-6, statistics 1e-5 relative, ρ interval
1e-6, counts and neighbour ids exact.

## Tests

- `test_validate_weight_styles`, `test_validate_knn_ties_match_knearneigh_without_kd_tree`
- `test_validate_logdet_complex_eigenvalues`

## MCP tools

`spatial_neighbors` and `moran_test` on `spatial_sim`, case `spatial_weights_moran`: the requests
`{method: distance, d_min: 0, d_max: 0.2, style: W | B | C | U | minmax | S}` and then
`{variable: x1}`; `{method: knn, k: 4}` for the alternatives; `{method: distance, d_max: 0.12}` for
the islands. Keys `style_<S>_sum` and `style_<S>_moran_I`, `style_C_sum` and `style_U_sum` (U is C
divided by n), `moran` (I, E, variance, z and p for the greater, less and two-sided alternatives),
`moran_islands`, `n_islands`. Tests (`crates/p2a-mcp/tests/tier3/wp57_ml_spatial.rs`):
`test_validate_mcp_spatial_styles`, `test_validate_mcp_style_u_is_c_over_n`,
`test_validate_mcp_moran_alternative`, `test_validate_mcp_moran_islands`.
MCP text is compared at print precision (`Tol::printed`: 6 decimals, z 4 decimals), JSON blocks at the
[validation/README.md](../README.md) table for the case's n (`Tol::coef(n)`), labels exactly. Registry:
`validation/methods/mcp_ml_spatial.json`.
