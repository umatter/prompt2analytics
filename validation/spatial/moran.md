# Moran's I and Geary's C

`crates/p2a-core/src/spatial/diagnostics.rs`: `moran_test`, `geary_test`. Registry entry `spatial_moran` in
`validation/methods/spatial_ml_weights.json`.

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
| Geary variance at or below 1e-8 | `Computation` | |

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

## Known differences

spdep returns NaN where p2a refuses the input (table above).

## Not covered

Not pinned to R: residual Moran moments (the `lm.morantest` formulas), the LM tests, `localmoran`.

## Tests

- `test_validate_moran_islands`
- `test_moran_and_geary_refuse_degenerate_inputs`, `test_moran_alternative_parses_aliases_and_rejects_unknown`,
  `test_moran_test_residuals_refuses_df_below_one`, `moran_matches_r_reference`
