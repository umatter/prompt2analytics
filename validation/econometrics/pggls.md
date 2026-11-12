# Validation: Panel GLS (pggls)

## Method Overview

Panel GLS (Feasible Generalized Least Squares) estimates panel data models while accounting for heteroskedasticity across periods and correlation between the periods of an entity (an unrestricted T × T error covariance, as plm's `pggls`; no N × N cross-sectional correlation). It transforms the data to achieve efficient estimation when the standard OLS assumptions are violated.

**Key Parameters:**
- `model`: Type of transformation - "within" (fixed effects), "pooling", or "fd" (first difference)
- Entity and time identifiers for panel structure

**Models Supported:**
1. **Fixed Effects GLS (FEGLS)**: within transformation, GLS on the demeaned model
2. **Pooled GLS**: GLS on the levels with an intercept
3. **First Difference GLS**: GLS on the first differences with an intercept column of ones

**Status**: R-pinned (methods `panel_gls` in `validation/methods/econ_panel_t3.json` and
`panel_gls_unbalanced` in `validation/methods/econ_panel_t4.json`): coefficients, SEs, RSS,
R² and df of all three models on the balanced simulated panel `pggls_sim` (N = 40, T = 5),
and coefficients, SEs, z, p-values, RSS, R², df and observations of all three models on the
unbalanced panel `pggls_unbalanced` (N = 40, T = 6). Since T0-WP14 p2a refuses panels with more distinct
periods than entities (T > N; ECON-32): the T × T error covariance is estimated from N
residual vectors and has no inverse there, so full Grunfeld (N = 10, T = 20) is refused,
while plm's `pggls` returns a generalised-inverse fit. That is why the pins use a simulated
panel with N > T, and Grunfeld appears only in the refusal tests and the 1935-1944 check.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `pggls()` | 2.6.7 (R 4.3.3) |
| linearmodels | Python | `PanelOLS` | N/A |

## Test Cases

Both use `inv ~ value + capital` on `validation/datasets/grunfeld.csv`. Values are printed
with `formatC(digits = 16, format = "g")`.

### `pggls_sim` (N = 40, T = 5): the registered pin

`validation/r/cases/econ_panel_t3.R`, R 4.3.3, plm 2.6.7;
`validation/datasets/generated/pggls_panel.csv` (`id`, `t`, `x1`, `x2`, `y`; n = 200):

```r
d <- pdata.frame(read.csv("validation/datasets/generated/pggls_panel.csv"), index = c("id", "t"))
pggls(y ~ x1 + x2, data = d, model = "pooling")   # also "within", "fd"
```

| Model | Quantity | R |
|---|---|---|
| pooling | coef (Intercept), x1, x2 | 1.150322778, 0.7532237028, -0.3709417407 |
| | SE | 0.1937637705, 0.09185105405, 0.07790031066 |
| | RSS, R² | 551.2630702, 0.1111792491 |
| | df_resid | 197 |
| within | coef x1, x2 | 0.7651582127, -0.3652536682 |
| | df_resid | 198 |
| fd | coef (Intercept), x1, x2 | 0.007046468406, 0.7822919666, -0.3756904277 |
| | df_resid | 157 (160 differenced rows less 3 coefficients) |

Every quantity of the table is pinned, with the within and FD SEs, RSS and R² of the case
file. R² is `1 - RSS / TSS` with the RSS of the transformed model and the TSS of the raw
outcome, for all three models (pggls's `summary()$rsqr`).

Tests (method `panel_gls`):
- `econometrics::panel::gls_models::tests::test_validate_pggls_pooling_sim`,
  `econometrics::panel::gls_models::tests::test_validate_pggls_within_sim` and
  `econometrics::panel::gls_models::tests::test_validate_pggls_fd_sim`: coefficients (README
  tolerance for n = 200, 1e-8), SEs, RSS and R² (1e-6), df exact; the z statistics are
  `coef / se` and the p-values normal. The FD fit has 160 observations and no first-step
  aliasing warning (its intercept is a column of ones).
- `econometrics::panel::gls_models::tests::test_validate_panel_gls_within_reports_singular_error_covariance`:
  the within residuals sum to zero in every entity of a balanced panel, so Ω has rank T - 1
  (eigenvalues 1.99, 1.47, 1.02, 0.60 and -5.6e-17 in pggls's estimate). The fit warns "panel
  GLS error covariance (T x T) is singular (rank 4 of 5); a generalized inverse was used".

### `pggls_unbalanced` (N = 40, T = 6): per-entity blocks

`validation/r/cases/econ_panel_t4.R`, R 4.3.3, plm 2.6.7;
`validation/datasets/generated/pggls_unbalanced.csv` (212 rows): entities 1-8 lack t = 1,
9-16 lack t = 6, 17-22 lack t = 3 and 23-25 lack t = 1 and 2.

| model | coef | se | df | rss | R² | nobs |
|---|---|---|---|---|---|---|
| pooling | 0.85241, 0.97740, -0.55794 | 0.19365, 0.055888, 0.048389 | 209 | 608.20 | 0.23709 | 212 |
| within | 0.95487, -0.53433 | 0.064265, 0.056078 | 210 | 174.57 | 0.78102 | 212 |
| fd | 0.0074459, 0.97772, -0.55463 | 0.049115, 0.066720, 0.058884 | 169 | 253.61 | 0.68188 | 172 |

Tests (method `panel_gls_unbalanced`): `econometrics::panel::gls_models::tests::test_validate_pggls_unbalanced_pooling`,
`econometrics::panel::gls_models::tests::test_validate_pggls_unbalanced_within` and
`econometrics::panel::gls_models::tests::test_validate_pggls_unbalanced_fd`: coefficients,
SEs, z, p-values (`Tol::p_value`), df, RSS, R² and observations at the README tolerances for
n = 212. The within test also asserts the indefinite-covariance warning.

## Unbalanced panels and the generalized Cholesky

Each entity is weighted by the generalized inverse of its own observed-period block
`Ω_ii` of the estimated Ω (not by a submatrix of the inverse of the full Ω). pggls inverts
these blocks with `bdsmatrix`'s generalized Cholesky, and p2a does the same: LDL' in period
order with no pivoting; a pivot below `1e-10` times the largest diagonal element of the whole
Ω is set to zero together with its column of L, and `G = L⁻ᵀ D⁺ L⁻¹`. On a positive-definite
block this is the inverse; on the balanced within model it drops the last period (Ω has
rank T - 1 there).

Each cell of Ω averages over the entities observed in both periods. On an unbalanced panel
that cell-averaged within Ω is indefinite (smallest eigenvalue -0.134 on `pggls_unbalanced`;
this held for every unbalanced pattern tried). The generalized Cholesky then zeroes the
non-positive pivots of the indefinite blocks (the last pivot of the full 6-period block is
-1.21), much as Wooldridge's FEGLS drops one period, and p2a warns: "panel GLS: the error
covariance averaged over the entities observed in each pair of periods is not positive
semidefinite (an unbalanced panel); the non-positive pivots of {m} entities' blocks were
dropped (generalized Cholesky, as plm::pggls)". A plain inverse, an eigen-clipped Ω or a
per-block clip give other numbers (0.95021 or 0.94799 for x1 instead of 0.95487). Pooling and
first-difference Ω are positive definite on this panel (smallest eigenvalues 0.119 and 0.287).

Two rows of one entity in one period are refused (`InvalidSpecification` naming the entity
and the period).

### Grunfeld 1935-1944 (N = T = 10)

The panel p2a's pooled-GLS test fits.

**R Code:**
```r
library(plm)
d <- pdata.frame(subset(read.csv("validation/datasets/grunfeld.csv"), year <= 1944), index = c("firm", "year"))
for (m in c("within", "pooling", "fd")) {
  fit <- pggls(inv ~ value + capital, data = d, model = m)
  print(formatC(coef(fit), digits = 16, format = "g")); print(formatC(sqrt(diag(vcov(fit))), digits = 16, format = "g"))
}
```

**R results** (R 4.3.3, plm 2.6.7):

| | within (n 100) | pooling (n 100) | fd (n 90) |
|---|---|---|---|
| (Intercept) | none | -13.74653219816151 (SE 1.539090790420873) | 10.03656990989108 (SE 1.338068135647484) |
| value | 0.06888727256843909 (SE 0.0007405206690987179) | 0.09591408245586269 (SE 0.0008283909896617362) | 0.06039987296684579 (SE 0.001458872991019236) |
| capital | 0.1544609271562051 (SE 0.006065620653202037) | 0.1596389387764905 (SE 0.004552539460846831) | -0.4829444560383797 (SE 0.01675773016581451) |

`validation/scripts/validate_panel_full.R` tests 6-7 write the within and pooling columns to
`validation/expected/fegls_1935_1944.csv` and `validation/expected/pggls_pooling_1935_1944.csv`.

### Full Grunfeld (N = 10, T = 20): refused by p2a

plm's generalised-inverse fit; p2a returns `InvalidSpecification` for this input. With
T > N the 20 × 20 error covariance has rank at most 10.

**R Code:**
```r
library(plm)
d <- pdata.frame(read.csv("validation/datasets/grunfeld.csv"), index = c("firm", "year"))
pggls(inv ~ value + capital, data = d, model = "within")   # also "pooling", "fd"
```

**R results** (R 4.3.3, plm 2.6.7):

| | within (n 200) | pooling (n 200) | fd (n 190) |
|---|---|---|---|
| (Intercept) | none | -40.32087035321853 (SE 3.606576997901137) | -1.76627310074304 (SE 0.07410199508316578) |
| value | 0.1096888240695295 (SE 0.0009197911431547233) | 0.115395750204538 (SE 0.0008612843838058983) | 0.08973990829009358 (SE 0.0001667942832044628) |
| capital | 0.3086506488152131 (SE 0.003522505893393277) | 0.2310418000883722 (SE 0.004088210706544555) | 0.2919898520737104 (SE 0.002313382314341975) |

## Rust tests

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ECON-36):
- `econometrics::panel::gls_models::tests::test_validate_pooled_gls_square_grunfeld_vs_plm` (`crates/p2a-core/src/econometrics/panel/gls_models.rs`): pooled coefficients on Grunfeld 1935-1944 within 1e-8 absolute of the pggls values typed into the test (-13.7465321982, 0.0959140825, 0.1596389388), and the SEs within 1e-8 relative of pggls's (1.539090790420873, 0.0008283909896617362, 0.004552539460846831). This is a quoted-value check.
- `econometrics::panel::tests::test_panel_gls_fe_full_grunfeld_is_refused` and `econometrics::panel::tests::test_pooled_gls_full_grunfeld_is_refused` (`crates/p2a-core/src/econometrics/panel/mod.rs`): full Grunfeld returns `InvalidSpecification`, and the message names N = 10 and T = 20. Their doc comments record pggls's values.
- `econometrics::panel::gls_models::tests::test_panel_gls_refuses_more_periods_than_entities` (`crates/p2a-core/src/econometrics/panel/gls_models.rs`): the T > N refusal on a 3 × 6 synthetic panel and on Grunfeld up to 1954, and acceptance of a 5 × 5 panel.
- `econometrics::panel::gls_models::tests::test_panel_gls_refuses_duplicate_entity_period` and `econometrics::panel::gls_models::tests::test_panel_gls_block_inverse_matches_dense_on_pd`: the duplicate-period refusal, and the generalized Cholesky against the dense inverse (positive definite), as a generalized inverse (rank 3 of 4) and with a negative pivot.
- `econometrics::panel::tests::test_panel_gls_fe` and `econometrics::panel::tests::test_panel_gls_pooling` (`crates/p2a-core/src/econometrics/panel/mod.rs`): their own 25-row panel (5 entities, periods 1-5). FE checks n_obs 25, 5 groups, 5 periods and the coefficient within 0.5 of the DGP's 2.0; pooling checks structure only.
- The FE and first-difference R pins are the `pggls_sim` and `pggls_unbalanced` tests above.

## Numerical Precision Summary

- **Coefficients, SEs, RSS, R², df**: pinned for all three models on `pggls_sim` and
  `pggls_unbalanced` at the README tolerances (coefficients 1e-8, SEs and statistics 1e-6
  relative). An R replica of the algorithm reproduces pggls to within 8e-15 relative.
- **Inference**: z statistics with normal p-values, as `summary.pggls`.

## Missing values

`run_panel_gls` reads y, the regressors, the entity and the time with one NA-aware request
and drops the rows with a missing value (null or NaN) in any of them, as pggls does
(`na.omit`); the count is in `PanelGlsResult.missing`, its first warning and the last line
of the output. Outcomes and times may be integers of any width (Int32 from parquet or a
cast, Int64, unsigned) or whole-number floats, and the entity text or numeric; a fractional
time is refused (`InvalidSpecification`, "must hold whole numbers") instead of being
truncated, and a missing id or time drops its row instead of becoming "" or period 0.
Tests: `econometrics::panel::gls_models::tests::test_validate_na_panel_gls_omit_equals_complete_rows`
(on `panel_grunfeld_na` up to 1944, 4 of 101 rows dropped, all three models equal to the fit
on the complete rows) and
`econometrics::panel::gls_models::tests::test_validate_na_panel_gls_integer_outcome_and_time`.
These compare fits with each other; the R pins are the cases above.

## Known Differences

1. **T > N is refused**, where pggls uses a generalised inverse (ECON-32, done in T0-WP14).
   p2a estimates the unrestricted T × T Ω from the first-step residuals, as pggls does
   (`gls_models.rs`, `Omega[s,t] = (1/N) * sum_i(e_is * e_it)`).
2. **Text output**: p2a's core text labels the statistics `z` and `P>|z|`; the CLI and MCP
   tables still print `t` headers over the same numbers (to be relabelled there).

## Timing

Release builds of v0.5.0 (`f80b0131`) and this version, on one core (`taskset -c 2`,
`RAYON_NUM_THREADS=1`), nine rounds interleaving the two builds; each round reports the median
of at least 41 calls after ten warm-up calls, and the table gives the median of the nine. The
machine was shared with other builds (load average 12-50 on 12 logical CPUs), so ratios
within about 0.05 of 1 are noise. Simulated panels of n / 5 entities over 5 periods with two regressors, AR(1)
errors and entity effects; the unbalanced rows drop every tenth row.

| Fit | n | v0.5.0 µs | new µs | new / v0.5.0 |
|---|---|---|---|---|
| pooling, balanced | 100 | 77.8 | 59.4 | 0.76 |
| pooling, balanced | 1,000 | 509.6 | 495.2 | 0.97 |
| pooling, balanced | 10,000 | 6,053 | 4,699 | 0.78 |
| pooling, unbalanced | 100 | 73.9 | 57.9 | 0.78 |
| pooling, unbalanced | 1,000 | 491.3 | 364.2 | 0.74 |
| pooling, unbalanced | 10,000 | 5,766 | 3,594 | 0.62 |
| within, balanced | 100 | 93.3 | 58.9 | 0.63 |
| within, balanced | 1,000 | 540.7 | 485.3 | 0.90 |
| within, balanced | 10,000 | 6,282 | 4,512 | 0.72 |
| within, unbalanced | 100 | 89.5 | 57.2 | 0.64 |
| within, unbalanced | 1,000 | 520.1 | 351.3 | 0.68 |
| within, unbalanced | 10,000 | 5,974 | 3,572 | 0.60 |
| first differences, balanced | 100 | 98.5 | 58.4 | 0.59 |
| first differences, balanced | 1,000 | 675.3 | 381.0 | 0.56 |
| first differences, balanced | 10,000 | 8,166 | 4,298 | 0.53 |
| first differences, unbalanced | 100 | 91.3 | 55.7 | 0.61 |
| first differences, unbalanced | 1,000 | 634.6 | 357.4 | 0.56 |
| first differences, unbalanced | 10,000 | 7,700 | 3,974 | 0.52 |

The transformed-model fit is faster than v0.5.0's in every row: the within model reuses the
demeaned arrays, and Omega is factorised once per pattern of observed periods instead of
once per entity. Peak resident memory of a process fitting the three models at n = 10,000:
11.1 MB balanced and 11.3 MB unbalanced, against 11.8 MB and 12.0 MB for v0.5.0.

## Performance Comparison

No speed comparison with R is claimed here. Speed comparisons with R come from the benchmark pipeline in the
paper repository (`performance/` in `../prompt2analytics-paper`), which runs the Rust
benchmarks of `crates/p2a-core/benches/` against R.

## References

- Baltagi, B. H. (2013). *Econometric Analysis of Panel Data* (5th ed.). Wiley.
- Croissant, Y., & Millo, G. (2008). "Panel Data Econometrics in R: The plm Package." *Journal of Statistical Software*, 27(2).
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

## Implementation Notes

`run_panel_gls` (`crates/p2a-core/src/econometrics/panel/gls_models.rs`):
1. The rows of each entity in time order (two rows in one period are refused), and the
   transformed model: the levels with an intercept (pooling), the entity-demeaned y and X
   (within), or the differences of consecutive observed rows of an entity with an intercept
   column of ones (FD; a gap is differenced across, and each differenced row takes the period
   of its later row). A first-step OLS (pivoted QR) on the transformed model gives the
   residuals; a genuinely aliased column is named in a warning, as `lm` does.
2. The T × T error covariance from those residuals, `Omega[s,t]` the mean of `e_is * e_it`
   over the entities observed in both periods, as pggls estimates it. When the full Ω is
   singular but has no negative pivot, the warning "panel GLS error covariance (T x T) is
   singular (rank r of T); a generalized inverse was used" is given.
3. GLS on the transformed model, β_GLS = (Σ_i X_i'G_i X_i)⁻¹ Σ_i X_i'G_i y_i with G_i the
   generalized-Cholesky inverse of the entity's own block Ω_ii (one factorisation per distinct
   period pattern). A singular X'Ω⁻¹X is an error; a reciprocal condition number below 1e-10
   adds "High condition number detected, results may be numerically unstable".
4. The variance (X'Ω⁻¹X)⁻¹, unscaled, as pggls; df = n - k with n the rows of the transformed
   model; z statistics and normal p-values; R² = 1 - RSS / TSS(raw y).
