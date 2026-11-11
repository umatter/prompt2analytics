# Validation: Robust Standard Errors (HC0-HC3)

## Method Overview

Heteroskedasticity-consistent (HC) standard errors give valid inference when the
homoskedasticity assumption fails. `run_ols` and `run_ols_raw` compute them with
`cov_type = HC0 | HC1 | HC2 | HC3`; coefficients, R² and the other fit statistics do not
depend on the choice (see [ols.md](ols.md)).

**Formula**:
```
V̂_HC = (X'X)⁻¹ X' diag(ûᵢ² ωᵢ) X (X'X)⁻¹
```

with n observations, k coefficients and leverage hᵢᵢ = xᵢ'(X'X)⁻¹xᵢ:

| Type | ωᵢ | Reference |
|---|---|---|
| HC0 | 1 | White (1980) |
| HC1 | n/(n − k) | the CLI and MCP default; Stata's `robust` |
| HC2 | 1/(1 − hᵢᵢ) | MacKinnon and White (1985) |
| HC3 | 1/(1 − hᵢᵢ)² | MacKinnon and White (1985); `sandwich`'s default |

t, p and the 95% CI use the HC standard error with n − k degrees of freedom, as
`lmtest::coeftest(fit, vcov = vcovHC(fit, type = ...))` does.

## Reference Implementation

`sandwich::vcovHC(fit, type = "HC0" | "HC1" | "HC2" | "HC3")`, sandwich 3.1.3 (R 4.3.3),
in `validation/r/cases/regression.R` ([source](../r/cases/regression.R)):

```r
V  <- vcovHC(m, type = t); se <- sqrt(diag(V))
p  <- 2 * pt(abs(coef(m) / se), m$df.residual, lower.tail = FALSE)
ci <- coef(m) + c(-1, 1) * qt(0.975, m$df.residual) * se
```

The cases `ols_grunfeld` (`lm(inv ~ value + capital)` on `validation/datasets/grunfeld.csv`,
n = 200) and `ols_mtcars` (`lm(mpg ~ wt + hp + disp)` on all 32 rows of
`datasets::mtcars`, `validation/datasets/generated/mtcars.csv`) hold the SE, t, p and CI
bounds for each type. `Rscript validation/r/run.R --check` recomputes them.

### `ols_grunfeld`

| SE | (Intercept) | value | capital |
|---|---|---|---|
| HC0 | 11.487562855581498 | 0.006759679290053672 | 0.048497663239300846 |
| HC1 | 11.574701117099652 | 0.006810954456871954 | 0.04886553953434222 |
| HC2 | 12.667874294420187 | 0.0069550258013054 | 0.0531650538310572 |
| HC3 | 14.01349546658532 | 0.007162666244380056 | 0.05850986620531393 |

HC1 p-values (the `p2a reg ols --robust hc1` exhibit): 0.00028980093362191437,
2.165754515305109e-40, 4.453595026390839e-06.

### `ols_mtcars`

| SE | (Intercept) | wt | hp | disp |
|---|---|---|---|---|
| HC0 | 2.2617298610195706 | 0.9062358470344324 | 0.008660583993070133 | 0.007842552849312815 |
| HC1 | 2.4178909261060477 | 0.9688068717759363 | 0.00925855373469901 | 0.008384041657084677 |
| HC2 | 2.4663339455531585 | 0.9887621746138576 | 0.01096541857387077 | 0.008797659447440553 |
| HC3 | 2.71026981081267 | 1.0808620677984107 | 0.014374570077587197 | 0.010121375444370815 |

## Tests

In `crates/p2a-core/src/regression/ols.rs` (registry entry `ols_robust_se` in
`validation/methods/regression.json`):

| Test | Case | Pins |
|---|---|---|
| `test_validate_ols_robust_se_grunfeld` | `ols_grunfeld` | for HC0, HC1, HC2 and HC3: coefficients, SE, t, p and both CI bounds |
| `test_validate_ols_robust_se_mtcars` | `ols_mtcars` | the same |
| `test_validate_ols_raw_grunfeld` | `ols_grunfeld` | HC1 through `run_ols_raw` |

Tolerances (`validation/README.md`, relative): SE, t and CI 1e-6 for n = 200 and 1e-5 for
n = 32; coefficients 1e-8 and 1e-6; p-values the same relative tolerance or 1e-12
absolute.

Measured agreement: SEs within **8.1e-15** relative on Grunfeld and **3.6e-14** on
mtcars; t within 8.8e-15 and 3.9e-12; CI bounds within 5.4e-13 and 3.1e-13; p-values
within 6e-16 and 3.8e-13 absolute.

A deliberate perturbation of the HC1 factor (n/(n − k − 1) instead of n/(n − k)) makes the
three tests fail (for example `se_hc1[value] rust=0.006828307235204594 R=0.006810954456871954`).

## Aliased designs (T3-WP09)

On a design with aliased columns (see [ols.md](ols.md), "Rank-deficient and
ill-conditioned designs") every variance estimator works on the r identified columns, as
`sandwich` does, and reports NaN for the aliased ones: HC0-HC3 (HC1 factor n/(n − r)),
the clustered CR1 SEs ((G/(G − 1))·((n − 1)/(n − r))), Newey-West/HAC, Driscoll-Kraay and
the bootstrap. They take the identified block of the fit's own (X'X)⁻¹, from whichever
solve the fit used, so no second decomposition is made.

Pinned to `ols_aliased` (`y ~ x1 + x2 + x3`, x2 = 2 x1): `dup_hc0_se`, `dup_hc1_se`,
`dup_hc3_se` (`test_validate_ols_aliased_duplicate`), `dup_cl_hc1_se`
(`vcovCL(cluster = ~cl, type = "HC1")`) and `dup_nw_lag4_se`
(`NeweyWest(lag = 4, prewhite = FALSE)`) in `test_validate_ols_aliased_sandwich`; SE
1e-6 relative (registry entry `ols_sandwich_aliased` in
`validation/methods/reg_rank_qr.json`).

**Bootstrap.** Replicates resample around y − residuals. The residual and wild bootstraps
solve every replicate with one factor of the fixed identified design
(`test_validate_bootstrap_fixed_design_solve_matches_refit`: equal to a least-squares refit
within 1e-12); the pairs bootstrap refits each replicate and skips one whose resampled
design is rank deficient, reported as "{s} of {B} bootstrap replicates had a rank-deficient
design and were skipped". Each pairs replicate takes `solve_ols`'s solve decision and
coefficients and forms no residuals (timing in `ols.md`). `test_validate_bootstrap_aliased_design_uses_identified_columns`
and `test_validate_bootstrap_wild_variants_near_hc0` (Rademacher, Mammen and Webb weights,
B = 2000, seeded, within 10% of HC0) are Monte Carlo property tests: the bootstrap SEs are
**not R-pinned**. The wild weights follow `sandwich::vcovBS`: Rademacher ±1, Mammen's
two-point law (−(√5 − 1)/2 with probability (√5 + 1)/(2√5), (√5 + 1)/2 otherwise), Webb's
six points ±√(1/2), ±1, ±√(3/2).

**Option names.** `kernel`: bartlett (newey_west, nw), parzen, quadratic_spectral (qs),
truncated, tukey_hanning; `bootstrap_type`: pairs (xy), residual, wild (rademacher),
mammen, webb. Case and separators are ignored; an unknown name is an error listing these
(`sandwich`'s norm, jackknife and fractional are not offered).

**Driscoll-Kraay.** Needs more time periods than identified coefficients (T > r);
otherwise `InsufficientData`. Its t and p use T − r degrees of freedom; R's
`coeftest(lm, vcov = vcovPL)` uses n − r. p2a keeps T − r because the estimator is
asymptotic in T. The meat sums the scores within each period and weighs lag j by
k(j / (L + 1)) over the lags with a non-zero weight, with no further scaling: lag L equals
`sandwich::vcovPL(m, cluster = ~time, lag = L, kernel = ..., adjust = FALSE)` and
`plm::vcovSCC` for every kernel. The default lag is floor(T^(1/4))
(`vcovPL(lag = "NW1987")`). The Driscoll-Kraay SEs are R-pinned
(`test_validate_driscoll_kraay_panel`, and the five kernels on Grunfeld in the
`reg_vcov_t4` cases).

**Negative variances.** The Truncated and Tukey-Hanning kernels do not guarantee a positive
semi-definite covariance. A negative diagonal element of a HAC or Driscoll-Kraay covariance
gives a NaN standard error (and NaN t and p) and a warning naming the coefficient, as
`sqrt` of a negative variance does in R; it is no longer clamped to 0. The bootstrap
covariance is a sample covariance and is not clamped either.

**Wald F.** With a robust covariance (HC0-HC3, one- and two-way clustered) the result also
carries `wald_f`: the overall Wald test of the identified slopes, F = b's V_ss⁻¹ b_s / q on
q and n − r degrees of freedom, as `lmtest::waldtest(m, vcov = ...)` with `test = "F"`
(case `ols_wald_f_grunfeld`, `test_validate_ols_wald_f_grunfeld`). The text output prints
it first, "F-statistic (Wald, HC1): ...", followed by the classical F, labelled
"F-statistic (classical): ..."; the JSON keeps `f_statistic` (classical) and adds `wald_f`.
Aliased slopes are left out; a singular slope covariance gives no Wald F and a warning. The
classical covariance has no `wald_f`.

## Known Differences

1. **Default type**: `sandwich::vcovHC` defaults to HC3; the p2a CLI and MCP default to
   HC1. The tests name the type explicitly.
2. **Leverage clamp**: `compute_leverage` clamps hᵢᵢ below 1 − 1e-10 so that the HC2 and
   HC3 weights stay finite at hᵢᵢ = 1; `sandwich` applies no clamp (its ωᵢ is 0/0 or
   infinite there). The clamp never binds on these data.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- regression::ols::tests::test_validate_ols_robust_se
P2A_VALIDATION_VERBOSE=1 cargo test -p p2a-core --lib --no-default-features --features ci -- test_validate_ols_robust_se --nocapture
```

## References

- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity". *Econometrica*, 48(4), 817-838.
- MacKinnon, J.G. & White, H. (1985). "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties". *Journal of Econometrics*, 29(3), 305-325.
- Zeileis, A. (2004). "Econometric Computing with HC and HAC Covariance Matrix Estimators". *Journal of Statistical Software*, 11(10), 1-17.
