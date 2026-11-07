# Validation: Spatial GMM (sphet)

## Method

`p2a_core::econometrics::sphet` estimates three spatial models by the Kelejian-Prucha
procedures. It is **R-pinned** to a written-out reference, not to the sphet package: see
"Reference".

1. **Spatial lag (SAR)**, `y = lambda W y + X beta + e`: spatial 2SLS (Kelejian-Prucha 1998,
   step 1).
2. **Spatial error (SEM)**, `y = X beta + u`, `u = rho W u + e`: OLS, GM for rho, then OLS of
   `(I - rho W) y` on `(I - rho W) X` (feasible GLS).
3. **SARAR**: 2SLS, GM for rho on its residuals, then 2SLS of `(I - rho W) y` on
   `(I - rho W) Z` with the same instruments (GS2SLS, Kelejian-Prucha 1998).

**Instruments.** `H = [1, X, W X, ..., W^order X]` over the non-constant columns of X. For a
row-standardised W, `W 1 = 1`, so the constant is never lagged. Instruments that a pivoted QR
finds collinear are dropped with a warning; fewer instruments than regressors is an error.

**GM moments for rho.**

- `het = true`: the two moments of Kelejian-Prucha (2010), with `eb = W e`,
  `m1 = eb'eb - e' D e` and `m2 = eb'e`, where `D = diag(W'W)` (the column sums of `W^2`).
  They hold under heteroskedasticity of unknown form.
- `het = false`: the three moments of Kelejian-Prucha (1999), with `sigma^2` concentrated out.

Both objectives are quartics in rho. They are minimised exactly on [-0.99, 0.99]: the interior
roots of the derivative cubic, by a safeguarded Newton method, and the two bounds. A solution at
a bound is reported as not converged, with a warning and a NaN SE of rho. A Newton solve that
reaches its iteration limit is reported as not converged too.

**Standard errors.**

- Classical: `s2 (Zh'Zh)^-1`, `s2 = u'u / (n - p)`.
- White (HC0): `(Zh'Zh)^-1 sum u_i^2 zh_i zh_i' (Zh'Zh)^-1`.
- Spatial HAC (Kelejian-Prucha 2007): `(Zh'Zh)^-1 Zh' Omega Zh (Zh'Zh)^-1`,
  `Omega_ij = K(d_ij / d) u_i u_j`, with sphet's seven distance kernels (triangular,
  Epanechnikov, rectangular, bisquare, Parzen, Tukey-Hanning, quadratic spectral), distances from
  unit coordinates or a distance matrix, and a fixed bandwidth or each unit's farthest
  W-neighbour as its bandwidth. A kernel that is not positive semi-definite can give a negative
  variance; the result then carries a warning.
- SE of rho: the GMM sandwich `(J'J)^-1 J' Psi J (J'J)^-1 / n` of the identity-weighted GM
  estimator, with `Psi` the variance of the moments of Kelejian-Prucha (2010, Theorem 2) and its
  homoskedastic analogue with the third and fourth moments for `het = false`. The first-step
  term is `(I - rho W')^-1 T alpha_r` for SARAR, computed by a Neumann series on `W'` (a series
  that does not converge gives a NaN SE with a warning), and 0 for the error model.

### Data requirements

- One dataset row per unit of W: row i of the data is unit i of the weights, so no row can
  be dropped.
- Missing values (null or NaN) in y or X are refused with `NullValues` naming the column,
  as spdep/spatialreg do (`na.action = na.fail`); remove the incomplete rows and rebuild W
  first. `±Inf` is refused as well.
- Integer and Boolean (0/1) columns are accepted and give the same estimates as their
  Float64 copies.
- Tests (T2-WP14): `test_na_sphet_int64_equals_float64_and_refuses_missing`
  (`econometrics::sphet::tests`; Int64 and Float64 give identical results for the lag,
  error and SARAR models, and a null y is `NullValues`).

## Reference

Written out in base R in `validation/r/cases/spatial_sphet.R` (case `sphet_gmm_hac`), because
sphet and spdep are not installed. The kernels, the HAC formula, the heteroskedastic GM moments
and the `Psi` of the rho variance follow the sources of CRAN's sphet 2.1-1 (`R/kernelsfun.R`,
`R/twostagels.R`, `R/gm_functions.R`). The reference minimiser is exact (the real roots of the
derivative cubic and the two bounds), with `optim` and `optimize` as cross-checks in the case.

## Case values

Dataset `validation/datasets/generated/sphet_sim.csv` (n = 150, seed 20261208); W is the 5
nearest neighbours on the coordinates, row-standardised. Instruments: 7 columns of rank 7 (the
old set with the lagged constant had 9 columns of rank 7).

| Quantity | Key | Value |
|---|---|---|
| SAR lambda | `sar_coef[3]` | 0.326000605909 |
| SAR x1, x2 | `sar_coef[1:2]` | 0.761709395561, -0.598082457536 |
| SAR SE classical, lambda | `sar_se_classical[3]` | 0.210131003624 |
| SAR SE White, lambda | `sar_se_white[3]` | 0.217600260783 |
| SAR SE HAC triangular d = 0.15, lambda | `sar_se_hac_tri_015[3]` | 0.245773633675 |
| SAR SE HAC Epanechnikov, variable bandwidth, lambda | `sar_se_hac_epa_var[3]` | 0.252037712452 |
| SEM rho, hom / het | `sem_hom_rho`, `sem_het_rho` | 0.478736169234, 0.481592734349 |
| SEM SE of rho, hom / het | `sem_hom_rho_se`, `sem_het_rho_se` | 0.081743091533, 0.077491950815 |
| SEM SE White, hom | `sem_hom_se_white` | 0.158183029233, 0.116372251634, 0.082801232028 |
| SARAR rho, hom / het | `sarar_rho`, `sarar_het_rho` | 0.417169197783, 0.425202346393 |
| SARAR SE of rho, hom / het | `sarar_rho_se`, `sarar_het_rho_se` | 0.250748553362, 0.246484644673 |
| SARAR lambda, hom / het | `sarar_coef[3]`, `sarar_het_coef[3]` | -0.089265315534, -0.093313177577 |

## Tests

All in `econometrics::sphet::tests`. Tolerances: closed-form 2SLS coefficients 1e-8 and SEs 1e-6
relative; GM rho and everything after it `Tol::iterative(1e-8)`; SE of rho 1e-6.

Pinned to the case (registered in `validation/methods/spatial_sphet.json`):

- `test_validate_sphet_instruments`: instrument names and rank.
- `test_validate_sphet_sar_classical`, `test_validate_sphet_sar_white`: SAR coefficients and SEs.
- `test_validate_sphet_sar_hac`: triangular and Epanechnikov at the case's fixed bandwidths.
- `test_validate_sphet_sar_hac_kernels`: the other five kernels at d = 0.15.
- `test_validate_sphet_sar_hac_variable`: the variable bandwidth.
- `test_validate_sphet_sar_p_values`: upper-tail p-values.
- `test_validate_sphet_sem_gm_hom`, `test_validate_sphet_sem_gm_het`: rho, objective, sigma^2.
- `test_validate_sphet_sem_se_hom`, `test_validate_sphet_sem_se_het`: FGLS coefficients and SEs.
- `test_validate_sphet_rho_se`: SE of rho for the error and SARAR models.
- `test_validate_sphet_sarar`, `test_validate_sphet_sarar_het`: GS2SLS coefficients and SEs.

Properties, which read the same dataset but pin no R value (not registered):

- HAC: `test_sphet_hac_row_permutation_invariant` (1e-11), `test_sphet_hac_tiny_bandwidth_equals_white`
  (a bandwidth below the smallest pair distance is White), `test_sphet_hac_distance_matrix_equals_coords`
  (1e-12), `test_sphet_hac_rejects_bad_inputs`, `test_spatial_hac_kernel_parses_aliases_and_rejects_unknown`,
  `test_sphet_hac_warns_when_a_variance_is_negative`.
- GM: `test_sphet_het_option_changes_rho`, `test_sphet_gm_flags_boundary_solution`,
  `test_sphet_gm_reports_iteration_limit`, `test_gm_quartic_minimiser_interior_boundary_and_flat`,
  `test_sphet_gm_scale_invariant`, `test_sphet_rejects_self_neighbours_for_gm`.
- SEs: `test_sphet_sem_white_se_is_scale_equivariant`, `test_sphet_rho_se_scale_invariant`,
  `test_sphet_rho_se_shrinks_with_n`, `test_sphet_rho_se_is_nan_at_the_boundary_and_for_a_divergent_series`.
- Inputs and output: `test_sphet_rejects_unidentified_lag_and_collinear_regressors`,
  `test_sphet_drops_collinear_instruments_with_a_warning`,
  `test_sphet_display_reports_se_method_moments_and_notes`.

The SE of rho was also checked by Monte Carlo when the formula was derived (n = 600: SD of the
estimates 0.0525 against a mean SE of 0.0523; SARAR `Psi` with and without the first-step term).
That was a one-off check, not rerun since; it is not a committed test.

## Differences from sphet

- rho is the step-1 identity-weighted GM estimate; sphet's `gstslshet` adds the efficient step
  with `Psi^-1` weights. The SE of rho is the sandwich valid for the identity weighting.
- Homoskedastic GM: the Kelejian-Prucha (1999) three moments; sphet's `spreg(het = FALSE)` uses
  the Drukker et al. two-moment version.
- Search interval [-0.99, 0.99]; sphet's `nlminb` bounds are (-0.9, 0.9).
- The error model's SE of rho uses `alpha_r = 0`, its value under exogenous X; sphet's `Psi`
  plugs in the sample `alpha_r`.
- The variable bandwidth takes W's neighbour sets as the distance object (each unit's farthest
  W-neighbour); sphet takes a separate `sphet::distance` object.
- The spatial HAC is also available for the error and SARAR models.

## What the rewrite changed

Numbers on the case's dataset.

- GM rho: 0.8995 before, 0.4787 now (`sem_hom_rho`).
- SE of rho: about 1.70 before, independent of n and scaling with y; 0.0817 now.
- SEM "robust" SEs: 0.0485, 0.0063, 0.0079 before; White on the transformed data now,
  0.1582, 0.1164, 0.0828.
- HAC: the old lag-based form gave 0.3946 or 0.4161 for the same data in a different row order; it
  now works on distances and does not depend on the row order.
- Instruments: 9 columns of rank 7 before (the lagged constant), 7 of rank 7 now.
- SARAR lambda: -0.1067 with transformed instruments before, -0.0893 now.

## Key differences from ML estimation

1. GMM does not require normal errors.
2. With `het = true` it is robust to heteroskedasticity of unknown form.
3. It does not need the eigenvalues of W.
4. It is less efficient than ML under homoskedasticity.

## References

1. Kelejian, H.H. & Prucha, I.R. (1998). "A Generalized Spatial Two-Stage Least Squares Procedure for Estimating a Spatial Autoregressive Model with Autoregressive Disturbances." Journal of Real Estate Finance and Economics, 17(1), 99-121.

2. Kelejian, H.H. & Prucha, I.R. (1999). "A Generalized Moments Estimator for the Autoregressive Parameter in a Spatial Model." International Economic Review, 40(2), 509-533.

3. Kelejian, H.H. & Prucha, I.R. (2007). "HAC Estimation in a Spatial Framework." Journal of Econometrics, 140(1), 131-154.

4. Kelejian, H.H. & Prucha, I.R. (2010). "Specification and Estimation of Spatial Autoregressive Models with Autoregressive and Heteroskedastic Disturbances." Journal of Econometrics, 157(1), 53-67.

5. Arraiz, I., Drukker, D.M., Kelejian, H.H. & Prucha, I.R. (2010). "A Spatial Cliff-Ord-type Model with Heteroskedastic Innovations: Small and Large Sample Results." Journal of Regional Science, 50(2), 592-614.

6. Piras, G. (2010). "sphet: Spatial Models with Heteroskedastic Innovations in R." Journal of Statistical Software, 35(1), 1-21. https://www.jstatsoft.org/article/view/v035i01
