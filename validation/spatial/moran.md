# Moran's I, Geary's C, regression diagnostics and local Moran

`crates/p2a-core/src/spatial/diagnostics.rs`: `moran_test`, `geary_test`, `moran_test_residuals`,
`spatial_lm_tests`, `localmoran` / `localmoran_with`. Registry entries `spatial_moran` in
`validation/methods/spatial_ml_weights.json`; `spatial_lm_morantest`, `spatial_lm_rstests` and
`spatial_localmoran` in `validation/methods/spatial_diag_t4.json`.

R-pinned to the formulas of `spdep::moran.test(randomisation = TRUE, zero.policy = TRUE, adjust.n = TRUE)`
and `spdep::geary()`, written out in base R because spdep is not installed: case
`spatial_weights_moran` of `validation/r/cases/spatial_ml_weights.R`, dataset
`validation/datasets/generated/spatial_sim.csv` (n = 80).

## Moran's I

With nn = n - #islands: I = (nn / S0) z'Wz / z'z and E[I] = -1 / (nn - 1). The variance is the
randomisation variance with nn in place of n. The kurtosis K = n Σz⁴ / (Σz²)² stays over all n, as
spdep's `moran()` uses `length(x)`. `var_norm` is not offered.

`MoranAlternative` is `two.sided`, `greater` or `less`: two.sided p = 2 Φ(-|z|), greater the upper tail
and less the lower tail.

## Geary's C

C = ((nn - 1) / (2 S0)) Σ w_ij (x_i - x_j)² / z'z and E[C] = 1. The variance is the Cliff-Ord
randomisation variance:

    VC = [(nn - 1) S1 (nn² - 3 nn + 3 - (nn - 1) K)
          - (1/4)(nn - 1) S2 (nn² + 3 nn - 6 - K (nn² - nn + 2))
          + S0² (nn² - 3 - K (nn - 1)²)] / (nn (nn - 2) (nn - 3) S0²)

z = (C - 1) / sqrt(VC), with two-sided p. The case checks this formula against an exact permutation
variance (n = 7, all 5,040 permutations, 1e-10). Whether `spdep::geary.test` takes K over n or over
nn, and its sign of z, was not checked (spdep is not installed).

## Errors

| Input | Error | spdep |
|---|---|---|
| Non-finite x | `InvalidSpecification` | |
| Constant x | `InvalidSpecification` | NaN |
| No links (S0 = 0) | `InvalidSpecification` | NaN |
| Fewer than four units with neighbours | `InsufficientData` | NaN |
| Moran variance at or below 1e-8 E[I]² | `Computation` | NaN with a warning |
| Geary variance at or below 1e-8 S1 / S0² | `Computation` | |

## Residual Moran's I (`lm.morantest`)

`moran_test_residuals(residuals, x, listw, alternative)` takes the OLS residuals and the design X (with
its intercept column) and follows `spdep::lm.morantest(zero.policy = TRUE)`. W enters through its
symmetric part U = (W + W') / 2, and with Z = U X, C1 = X'Z, C3 = (X'X)⁻¹ C1, tr A = tr C3:

    I = (N / S0) u'Wu / u'u
    E[I] = -N tr A / ((N - p) S0)
    Var[I] = N² / (S0² (N - p)(N - p + 2)) (S1 + 2 tr C3² - 4 tr((X'X)⁻¹ Z'Z) - 2 (tr A)² / (N - p))

N is the number of units with neighbours and p the rank of X (pivoted QR, tolerance 1e-7; aliased
columns are dropped as `lm` drops them). N - p below 1 is `InsufficientData`; a non-positive variance
is `Computation`. spdep's default alternative is "greater".

## Rao score tests (`lm.RStests`)

`spatial_lm_tests(y, x, &listw)` fits the OLS of y on X itself (pivoted QR) and returns spdep's five
tests. With u the residuals, ŷ the fitted values, s² = u'u / n, T = tr(W'W + W²) and M the residual maker
of X:

    RSerr    = (u'Wu / s²)² / T
    RSlag    = (u'Wy / s²)² / nJ,   nJ = (||M Wŷ||² + T s²) / s²
    adjRSerr = (u'Wu/s² - T/nJ u'Wy/s²)² / (T (1 - T/nJ))
    adjRSlag = (u'Wy/s² - u'Wu/s²)² / (nJ - T)
    SARMA    = adjRSlag + RSerr   (df 2)

All-zero residuals, a W without links (T = 0) or a denominator that is not positive is `Computation`
naming the test. Like spdep, the tests assume a row-standardised W and warn (`warnings`) otherwise.

## Local Moran (`localmoran`)

I_i = (z_i / m2) Σ_j w_ij z_j with m2 = Σz² / n, as `spdep::localmoran(mlvar = TRUE)`. The moments
default to spdep's conditional randomisation (`LocalMoranConfig.conditional = true`); `false` gives the
total-randomisation moments. The p-value is two-sided normal, or with `permutations > 0` spdep's folded
conditional-permutation p-value ("Pr(folded) Sim") with each unit's neighbour values drawn without
repeats (Anselin's scheme, spdep's `no_repeat_in_row = TRUE`), seeded per permutation. A zero variance
(an island, or z_i = 0 under conditional moments) gives NaN z and p and the cluster "not significant".
`global_i` is Moran's I, Σ I_i / S0. Non-finite or constant x and `alpha` outside [0, 1] are
`InvalidSpecification`.

## Reference values

| | 4 nearest neighbours | (0, 0.12] band, 4 islands |
|---|---|---|
| I | -0.017872 | 0.019277 |
| E[I] | -0.012658 | -0.013333 |
| Randomisation variance | 0.005076 | 0.009330 |
| z | -0.073181 | 0.337598 |
| p greater | 0.529169 | 0.367833 |
| p two-sided | 0.941662 | 0.735666 |
| C | 1.012246 | 0.916113 |
| Var C | 0.006046 | 0.010152 |
| z (C) | 0.157492 | -0.832549 |
| p (C) | 0.874857 | 0.405099 |

Tolerances from `validation/README.md` at n = 80: statistics 1e-5 relative, p-values 1e-5 relative or
1e-12 absolute, the island count exact.

Residual Moran (case `spatial_lm_diag_t4` of `validation/r/cases/spatial_diag_t4.R`, y on x1, x2):

| | kNN-4, W | band, W | band, B | band with 4 islands, W |
|---|---|---|---|---|
| I | 0.384001 | 0.363220 | 0.232057 | 0.430584 |
| E[I] | -0.011689 | -0.012639 | -0.013283 | -0.011026 |
| Var[I] | 0.005062 | 0.003417 | 0.002516 | 0.009307 |
| z | 5.561722 | 6.429717 | 4.891528 | 4.577650 |
| p greater | 1.3356e-08 | 6.3921e-11 | 5.0028e-07 | 2.3511e-06 |

Rao score tests (same case):

| | kNN-4, W | band, B |
|---|---|---|
| RSerr | 27.255506 | 18.093790 |
| RSlag | 44.391145 | 18.695085 |
| adjRSerr | 0.796227 | 6.799108 |
| adjRSlag | 17.931866 | 7.400403 |
| SARMA | 45.187372 | 25.494193 |

Local Moran (case `spatial_localmoran_t4`, kNN-4, W): the 80 values of I_i, E, Var, z and p under
conditional and total randomisation; global I = 0.370321.

## Timing against the previous implementation (Tier 4)

Quiet integration re-time, 2026-11-16: release builds of v0.5.0 (`f80b0131`) and of the integrated branch,
`--no-default-features --features ci`; one core (`taskset -c 2`), `RAYON_NUM_THREADS=1` (one thread for polars and
BLAS too), cwd `/dev/shm`; per row five processes per side, interleaved with the order alternating (nine for rows
within 1.05-1.15x), each the median of its calls after a warm-up; cells are the median over the processes, the ratio
of the minima in brackets. 1-minute load average 0.7-2.7 on 12 cores, nothing else building. Data: points on a
jittered square grid, 4-nearest-neighbour weights, row-standardised;
regressions with an intercept and two regressors.

| Method | n | f80b0131 (µs) | this change (µs) | ratio [min] |
|---|---|---|---|---|
| `moran_test_residuals` | 100 | 14.49 | 27.62 | 1.91 [1.91] |
| `moran_test_residuals` | 1,000 | 148.9 | 252.7 | 1.70 [1.68] |
| `moran_test_residuals` | 10,000 | 1,600 | 3,075 | 1.92 [1.93] |
| `spatial_lm_tests` | 100 | 31.39 | 27.69 | 0.88 [0.88] |
| `spatial_lm_tests` | 1,000 | 248.0 | 235.8 | 0.95 [0.94] |
| `spatial_lm_tests` | 10,000 | 2,552 | 3,028 | 1.19 [1.21] |
| `spatial_lm_tests`, the MCP path (OLS fit included) | 100 | 55.53 | 36.96 | 0.67 [0.68] |
| `spatial_lm_tests`, the MCP path (OLS fit included) | 1,000 | 334.5 | 330.3 | 0.99 [0.97] |
| `spatial_lm_tests`, the MCP path (OLS fit included) | 10,000 | 3,905 | 3,858 | 0.99 [0.96] |
| `localmoran`, normal p-values | 100 | 17.02 | 17.87 | 1.05 [1.06] |
| `localmoran`, normal p-values | 1,000 | 173.0 | 179.4 | 1.04 [1.05] |
| `localmoran`, normal p-values | 10,000 | 2,037 | 2,180 | 1.07 [1.07] |
| `localmoran`, 99 permutations | 100 | 338.1 | 871.2 | 2.58 [2.83] |
| `localmoran`, 99 permutations | 1,000 | 3,731 | 5,993 | 1.61 [1.59] |
| `localmoran`, 99 permutations | 10,000 | 39,972 | 123,207 | 3.08 [3.08] |

The residual Moran test now forms `lm.morantest`'s moments through the regression's projection (solves
with the k-column design and its trace terms), and the score tests their trace term and projected lag;
the permutations of `localmoran` draw each unit's neighbours without repeats from the other n - 1
values, from a generator per permutation. The ratios are within the estimates signed off 2026-11-15
(2-6x, 1.2-2x and 1.5-3x), except `localmoran` with 99 permutations at n = 10,000: 3.08x by the median (39.97 -> 123.2 ms),
above the 2.99x signed off 2026-11-16 and within its 3.18x by the minima; that row has gone to the user.

## Known differences

spdep returns NaN where p2a refuses the input (table above).

## Not covered

The permutation p-values of `localmoran` are a property test (99,999 permutations converge to the
enumerated limits of a 7-unit case), not a pin: the draws differ from spdep's. spdep is not installed,
so the residual Moran, score test and local Moran references are spdep's formulas written out in base R
(`validation/r/cases/spatial_diag_t4.R`).

## Tests

- `test_validate_moran_islands`
- `test_validate_lm_morantest_knn_spatial_sim`, `test_validate_lm_morantest_band_styles_spatial_sim`,
  `test_validate_lm_morantest_islands_spatial_sim`, `lm_morantest_under_h0_has_unit_variance_z`
- `test_validate_lm_rstests_knn_spatial_sim`, `test_validate_lm_rstests_band_binary_spatial_sim`,
  `lm_rstests_are_invariant_to_regressor_scale`, `lm_rstests_sarma_identities`,
  `test_validate_mcp_lm_rstests_knn_spatial_sim` (p2a-mcp)
- `test_validate_localmoran_conditional_spatial_sim`, `test_validate_localmoran_total_spatial_sim`,
  `localmoran_global_i_is_morans_i`, `localmoran_conditional_permutation_converges_to_exact`,
  `localmoran_seed_changes_permutation_p_only`, `localmoran_zero_variance_is_nan_not_clamped`
- `geary_refuses_a_degenerate_permutation_distribution`, `geary_variance_guard_is_relative_to_s1_over_s0_squared`
- `test_moran_and_geary_refuse_degenerate_inputs`, `test_moran_alternative_parses_aliases_and_rejects_unknown`,
  `test_moran_test_residuals_refuses_df_below_one`, `moran_matches_r_reference`
