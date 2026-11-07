# Validation: Phillips-Perron Unit Root Test

## Method Overview

The Phillips-Perron test examines the null hypothesis that a time series has a unit root (non-stationary) against the alternative that it is stationary. Unlike the ADF test which uses lagged differences to handle serial correlation, the PP test makes a non-parametric correction to the t-statistic using the Newey-West estimator.

**Regression model:**
```
Δyₜ = α + βt + γyₜ₋₁ + uₜ
```

**Test statistic**, R's `PP.test` "Dickey-Fuller", the corrected t statistic Z(τ) (R does not return Z(α)):
```
Z(τ) = τ̂ × √(σ̂²/λ²) - n³ / (4 √3 √D √λ²) × (λ² - σ̂²)
```

Where, with n = length(x) - 1:
- τ̂ = t-statistic of γ̂ from the OLS regression
- σ̂² = RSS / n
- λ² = Newey-West long-run variance estimate with Bartlett weights
- D = det(X'X) for X = [1, t, yₜ₋₁], t = 1..n; p2a computes it from centred sums, R by a four-term difference with the same value

The sign of λ² - σ̂² is kept.

**Truncation lag:**
- lshort=TRUE: trunc(4*((length(x)-1)/100)^0.25)
- lshort=FALSE: trunc(12*((length(x)-1)/100)^0.25)

**P-value:** R's table (Banerjee et al. 1993, Table 4.2, 6 x 8 as tabulated in `PP.test`), interpolated linearly in n and then in the statistic, both clamped at the table's ends (`approx(rule = 2)`), so p lies in [0.01, 0.99].

**Key parameters:**
- `x`: Time series data
- `lshort`: Whether to use short truncation lag formula (default: TRUE)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `PP.test` | R 4.3.x |

## Test Cases

### Case `stat_pp_t3` (R-pinned)

Case in `validation/r/cases/stats_misc_t3.R`, dataset `validation/datasets/generated/pp_t3.csv`. The
internals are recomputed with `PP.test`'s own code.

| n | series | lag rule | statistic | p | lag | γ̂ | t | σ̂² | λ² |
|---|---|---|---|---|---|---|---|---|---|
| 200 | ar | short | -7.590885410540011 | 0.01 | 4 | -0.45717790651442436 | -7.617386158111387 | 1.1623414925199784 | 1.147806794887778 |
| 200 | ar | long | -7.504043458195605 | 0.01 | 14 | -0.45717790651442436 | -7.617386158111387 | 1.1623414925199784 | 1.1002914390310594 |
| 200 | rw | short | -2.6942210465302874 | 0.28535397485905667 | 4 | -0.06461527829701241 | -2.6317013704465775 | 1.076833429706771 | 1.1382538505659412 |
| 200 | rw | long | -2.618227278491557 | 0.31718049272681104 | 14 | -0.06461527829701241 | -2.6317013704465775 | 1.076833429706771 | 1.0637787669273304 |
| 100 | rw100 | short | -2.3122134056118275 | 0.4473018203762106 | 3 | -0.11617646781492785 | -2.259073884064882 | 0.9290416704446076 | 0.9652933956613979 |

**Rust tests:**
- `stats::pptest::tests::test_validate_pp_test_t3_internals`, `stats::pptest::tests::test_validate_pp_test_t3_statistic`
- `stats::pptest::tests::test_pp_correction_keeps_sign` (`ar`, long lag: λ² < σ̂²; not R-pinned)
- `stats::pptest::tests::test_pp_p_value_clamps_and_is_monotone` (the table's clamps; not R-pinned)
- `stats::pptest::tests::test_pp_test_needs_positive_residual_df`: five observations at least, since four leave a residual df of 0

### Case `stat_pp_gdp_t3` (R-pinned)

`PP.test(gdp_growth)` on `validation/datasets/gdp_quarterly.csv` (n = 280):

| Lag rule | statistic | lag | p |
|---|---|---|---|
| short | -12.018455837825698 | 5 | 0.01 |
| long | -11.686391180032054 | 15 | 0.01 |

`tseries::pp.test` gives the same short-lag statistic (-12.018455837825698). Before this
correction p2a reported -211.82 here. **Rust test:** `stats::pptest::tests::test_validate_pp_test_t3_gdp`.

### Truncation lag

`compute_truncation_lag(n, ..)` is the formula; `pp_test` passes length(x) - 1.

| n passed | short | long |
|---|---|---|
| 50 | 3 | 10 |
| 100 | 4 | 12 |
| 1000 | 7 | 21 |

**Rust tests:** `stats::pptest::tests::test_truncation_lag_short`, `stats::pptest::tests::test_truncation_lag_long`.

## Numerical Precision Summary

| Computation | Tolerance | Notes |
|-------------|-----------|-------|
| γ̂ | `Tol::coef(n)` | |
| Statistic, t, σ̂², λ² | `Tol::stat(n)` | |
| P-value | `Tol::p_value(n)` | |
| Truncation lag | exact | |

**Status:** R-pinned (registry `validation/methods/stats_misc_t3.json`, id `pp_test_t3`).

## Missing values

`run_pp_test` refuses a column with missing (empty) cells, naming the column and the number of
missing cells (R's `sum(is.na(x))`); the count is pinned with the ACF readers'
([acf.md](../stats/acf.md)). R's `PP.test` does not refuse them; its statistic is `NA`.

## Known Differences from R

1. **Exact fit** (registry deviation STAT-6): when the regression fits exactly, as for a quadratic or a geometric series, the residuals are rounding error next to the differences or to the level of the series. p2a refuses (`InvalidSpecification`). `PP.test` warns of an essentially perfect fit and returns a statistic made of that rounding error.

2. **Singular regression**: when the lagged series is constant or a linear trend (for example 1, 2, 3, ..., n), the design matrix is singular. p2a returns `SingularMatrix`, with a check on centred sums that does not depend on the scale or the level of the series; R stops with "singularities in regression".

3. **Near-linear series** (registry deviation STAT-6): p2a's bound is `1 - r^2(t, y_{t-1}) <= 1e-12`, or a remainder after the trend within rounding at the level of the series. It is stricter than R's QR tolerance: the remainder comes from a subtraction that loses about 1e-15 of the sum of squares of `y_{t-1}`. `PP.test` still returns a statistic down to about `1 - r^2 = 1e-14` (for n = 101, a step of 1e5 and noise of sd 1, `1 - r^2` is about 1e-13 and the statistic is about -9.85); p2a refuses these series. **Rust test:** `stats::pptest::tests::test_pp_test_refuses_near_linear_series`.

4. **Range**: p2a refuses a series whose squares do not fit a double (`InvalidSpecification`).

Tests: `stats::pptest::tests::test_pp_test_refuses_singular_regression`, `stats::pptest::tests::test_pp_test_refuses_exact_fit`, `stats::pptest::tests::test_pp_test_is_scale_and_level_free`, `stats::pptest::tests::test_pp_test_refuses_values_whose_squares_leave_the_range`.

## Performance Comparison

### Phillips-Perron Test (lshort=TRUE)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 2.14      | 2,140  | **~1,000x faster** |
| n=1,000      | 27.95     | 1,860  | **~67x faster** |
| n=10,000     | 343       | 6,140  | **~18x faster** |
| n=100,000    | 4,659     | 67,680 | **~15x faster** |

### Phillips-Perron Test (lshort=FALSE)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 2.71      | 1,760  | **~650x faster** |
| n=1,000      | 43.31     | 1,900  | **~44x faster** |
| n=10,000     | 652       | 6,300  | **~10x faster** |
| n=100,000    | 10,268    | 71,500 | **~7x faster** |

**Implementation Notes:**
- Uses explicit OLS regression with 3x3 matrix inverse (constant, trend, lagged value)
- Newey-West variance with Bartlett weights computed directly
- P-values interpolated in R's table
- Rust is faster than R at **all** dataset sizes (7x to 1,000x speedup)

*Benchmarks run on 2026-01-19. Rust benchmarks via Criterion, R benchmarks via system.time().*

## References

- Phillips, P. C. B. & Perron, P. (1988). "Testing for a Unit Root in Time Series Regression." *Biometrika*, 75(2), 335-346.
- Banerjee, A., Dolado, J. J., Galbraith, J. W., & Hendry, D. (1993). *Co-integration, Error Correction, and the Econometric Analysis of Non-Stationary Data*. Oxford University Press.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/PP.test.html
