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
