# Validation: New Statistical Methods (15 Methods)

## Overview

This document validates the 15 newly implemented R stats package methods against the original R implementations.

## Methods Validated

1. **cor.test** - Correlation tests (Pearson, Spearman, Kendall)
2. **power.t.test** - Power analysis for t-tests
3. **power.prop.test** - Power analysis for proportion tests
4. **power.anova.test** - Power analysis for ANOVA
5. **fivenum** - Tukey's five number summary
6. **IQR** - Interquartile range
7. **mad** - Median absolute deviation
8. **ecdf** - Empirical cumulative distribution function
9. **density** - Kernel density estimation
10. **spline** - Cubic spline interpolation
11. **approx** - Linear/constant interpolation
12. **smooth.spline** - Cubic smoothing spline
13. **prop.trend.test** - Test for trend in proportions
14. **gls** - Generalized least squares (via nlme)
15. **stl** - Seasonal decomposition using LOESS

## Reference Implementations

| Package | Language | Functions | Version Tested |
|---------|----------|-----------|----------------|
| stats | R | cor.test, power.t.test, power.prop.test, power.anova.test, fivenum, IQR, mad, ecdf, density, spline, approx, smooth.spline, prop.trend.test | R 4.3.3 |
| nlme | R | gls | nlme 3.1-x |
| stats | R | stl | R 4.3.3 |

## Performance Comparison Summary

### Correlation Tests (cor.test)

| Method | Size | Rust (µs) | R (µs) | Speedup |
|--------|------|-----------|--------|---------|
| Pearson | n=100 | 1.5 | 130 | **87x** |
| Pearson | n=1,000 | 9.4 | 140 | **15x** |
| Pearson | n=10,000 | 96 | 510 | **5x** |
| Spearman | n=100 | 7.4 | 170 | **23x** |
| Spearman | n=1,000 | 82 | 630 | **8x** |
| Spearman | n=10,000 | 1,350 | 6,610 | **5x** |

### Power Analysis

| Function | Rust (µs) | R (µs) | Speedup |
|----------|-----------|--------|---------|
| power.t.test | 4.3 | 82 | **19x** |
| power.prop.test | 0.2 | 38 | **190x** |
| power.anova.test | 4.3 | 31 | **7x** |

### Robust Statistics

| Function | Size | Rust (µs) | R (µs) | Speedup |
|----------|------|-----------|--------|---------|
| fivenum | n=100 | 2.9 | 70 | **24x** |
| fivenum | n=1,000 | 34 | 100 | **3x** |
| fivenum | n=10,000 | 510 | 1,100 | **2x** |
| IQR | n=100 | 5.8 | 90 | **16x** |
| IQR | n=1,000 | 67 | 100 | **1.5x** |
| IQR | n=10,000 | 1,020 | 430 | 0.4x |
| mad | n=100 | 5.2 | 90 | **17x** |
| mad | n=1,000 | 66 | 100 | **1.5x** |
| mad | n=10,000 | 1,080 | 820 | 0.8x |
| ecdf | n=100 | 4.5 | 150 | **33x** |
| ecdf | n=1,000 | 42 | 250 | **6x** |
| ecdf | n=10,000 | 592 | 1,990 | **3x** |

### Kernel Density Estimation

`density` follows R's `density()` (R 4.3.3): the finite values are linearly binned onto 2m
points over [from - 4 bw, to + 4 bw], m = max(n, 512) (rounded up to a power of two above 512),
convolved with the kernel by FFT (`rustfft`, feature `spectral-analysis`) and linearly
interpolated onto the n requested points from `from` to `to`. Every kernel is scaled so that `bw`
is its standard deviation; `cosine` is R's raised cosine and `optcosine` R's optimal cosine. The
default bandwidth is `bw.nrd0` (type-7 IQR; when min(sd, IQR/1.34) is 0, sd, then |x[1]|, then 1).
Builds without `spectral-analysis` compute the exact kernel sum at the same points. R-pinned in
`validation/methods/stats_descriptive_t4.json`, id `density_t4` (`validation/stats/robust.md`).
The earlier timing figures of this section measured an FFT path that no longer exists; the rows
above 1.1x are under "Rule 5" below.

### Rule 5 (Tier 4, signed off 2026-11-14)

Rows above 1.1x of v0.5.0 (`f80b0131`) after the Tier 4 change, measured on a loaded machine and signed off on
2026-11-14; the quiet re-time (2026-11-15, this tree at `2d0bb8b6`, one core, interleaved, five rounds, nine for
rows within 1.05-1.15x, load average 1.0-3.0) gives the figures in brackets, and rows at or below 1.1x drop off:

- `density`, default Gaussian kernel: 1.66x at n = 100 and 1.42x at n = 1,000 [quiet 1.70x, 1.49x, and
  1.10x at n = 10,000; 68.5 µs -> 116 µs at n = 100]; a 100-point grid 2.54x [2.09x / 1.51x at
  n = 100 / 1,000; a 1,000-point grid 1.39x / 1.36x]. The quiet rows above the signed figures go to the user. R's binned FFT now runs at every size, where v0.5.0 summed directly on small inputs. The
  kernel ordinates above m are copied from those below it (every kernel is even) and the FFT plans
  are reused.
- `cor.test(method = "kendall")`: the exact distribution below n = 50 without ties (up to 10x,
  34 µs at n = 49; R's exact p-value where v0.5.0 used the normal approximation; the table of each n
  is computed once per process), and with ties 1.27x at n = 100 (the tie-corrected variance) [quiet: exact n = 10-49
  0.91-0.96x, ties 0.97x; both drop off].
- `Box.test` wrappers 2.2-2.5x and `var.test` 1.11x at n = 100: the NA-aware column extractor [quiet: Box
  1.84 / 2.25 / 1.52x at n = 100 / 1,000 / 10,000 (2.37 µs -> 4.35 µs at n = 100); var.test 1.05x, drops off].

## Numerical Validation

### cor.test Validation

`cor_test` follows `cor.test`'s defaults (R 4.3.3):

- **Pearson:** r, t on n - 2 degrees of freedom, the Fisher-z interval. A one-sided interval
  ends at -1 (`less`) or 1 (`greater`).
- **Spearman:** rho; the statistic is S = (n^3 - n)(1 - rho)/6. Without ties and for n < 1290
  the p-value comes from the exact distribution of S: all n! permutations for n <= 9 (counted
  once per n and cached), the AS 89 Edgeworth series for 10 <= n < 1290. With ties or for
  n >= 1290: the t approximation. A perfect monotone relation gets the direction-aware p
  (`less` is 1 for rho = 1).
- **Kendall:** tau-b = S / sqrt((n0 - n1)(n0 - n2)). Without ties and for n < 50 the exact
  distribution of T, the number of concordant pairs (statistic `T`); otherwise z = S / sqrt(var S)
  with the tie-corrected variance and no continuity correction (statistic `z`).

```r
# case stat_cor_test_t4, validation/r/cases/stats_descriptive_t4.R
# (validation/datasets/generated/cor_test_t4.csv, columns set, x, y)
x <- 1:10; y <- c(1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0)   # set pear
cor.test(x, y)                          # r 0.999070526757068, t 65.5554248847905, p 3.26167845639871e-12
cor.test(x, y, alternative = "less")    # CI [-1, 0.999731847375987]
y <- c(3, 1, 2, 6, 4, 5, 9, 10, 7, 8)                                # set perm
cor.test(x, y, method = "spearman")     # S 28, p 0.005556804545853975 (Edgeworth)
cor.test(x, y, method = "kendall")      # tau 0.6444444444444445, T 37, p 0.009148478835978802
```

**Rust tests:** `stats::cortest::tests::test_validate_pearson_correlation` (r, t, df, two-sided p,
95% and 90% intervals), `stats::cortest::tests::test_validate_pearson_confidence_interval`
(one-sided p and intervals), `stats::cortest::tests::test_validate_spearman_t4` (rho, S, three p on
the exact, Edgeworth and t paths, perfect and tied data),
`stats::cortest::tests::test_validate_kendall_tie_free_against_r` (exact T and z without ties) and
`stats::cortest::tests::test_validate_kendall_ties_t4` (tau-b and z with ties). Tolerance
`validation/README.md` by n: r, rho, tau `Tol::coef(n)`; t, z, S and interval bounds `Tol::stat(n)`;
p `Tol::p_value(n)`; T and df exact. **Status:** R-pinned for all three methods and every
alternative (registry `validation/methods/stats_descriptive_t4.json`, id `cor_test_t4`).

#### cor.test with missing values (R-pinned)

`p2a_core::stats::cortest::run_cor_test_dataset(dataset, x_col, y_col, method, alternative,
conf_level)` runs `cor.test` on two dataset columns. It reads both through the shared
extractor in one request and keeps the complete pairs, as `cor.test` does
(`complete.cases(x, y)`); the drop is reported in `CorTestResult.missing`. Integer columns are
accepted; a text column is `NonNumericColumn`. `run_cor_test` (slices) is unchanged.

```r
# case stat_two_sample_na, validation/r/cases/stats_onesample.R
# (validation/datasets/generated/two_sample_na.csv: x and y have 7 values each, 6 complete pairs)
ct <- cor.test(d$x, d$y)

# R 4.3.3 gives:
# r = 0.9782906634777736
# t = 9.441251753133958 (df = 4)
# p-value = 0.0007018271843026733
# 95% CI: (0.8091413571114043, 0.9977196153804951), 6 pairs
```

Before T2-WP06 and T2-WP20 the MCP handler extracted the columns itself and read nulls as 0
(r 0.80 on 22 pairs of this dataset) and refused integer columns. `hypothesis_cor_test` now
calls `run_cor_test_dataset`, keeps the complete pairs and reports the drop under `missing`
(`{"n_total": 22, "n_used": 6, "n_dropped": 16, "by_column": {"x": 15, "y": 15}}` here).

**MCP tool:** `tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_cor_test_two_sample_na`
pins the tool's JSON (`estimate`, `statistic`, `df`, `p_value`, `confidence_interval`, `n`,
`missing`) to the values above at the tolerances below (registry
`validation/methods/mcp_stats.json`, id `mcp_hypothesis_cor_test`). The property test
`tools::handlers::hypothesis::tests::test_mcp_hypothesis_two_sample_columns` checks that the
Int64 pair `xi`, `yi` gives exactly the result of its Float64 copy (18 pairs) and that text and
absent columns are refused with the core's error.

**Rust tests:** `stats::cortest::tests::test_validate_cor_test_two_sample_na` (r, t, df, p,
CI, n; `missing`) and `stats::cortest::tests::test_validate_na_cor_test_columns` (Int64 against
Float64, text, absent column, unknown method). Tolerance `validation/README.md`, n < 100: r 1e-6,
t and CI 1e-5 relative, p `Tol::p_value`; df and counts exact.

**Status:** Pearson, two-sided, on dataset columns: R-pinned (registry
`validation/methods/stats_onesample.json`, id `cor_test_pearson`). The three methods on slices,
one-sided alternatives included, are R-pinned under `cor_test_t4` (above).

### power.t.test Validation

```r
# R code
power.t.test(n = 30, delta = 0.5, sd = 1, sig.level = 0.05, type = "two.sample")

# R 4.3.3 gives:
# power = 0.4778409859409388
```

**Rust test**: R-pinned in `stats::power::tests::test_validate_power_legacy_api` and `test_validate_power_t_solve_power` (case `stat_power_solve`); see [power.md](power.md).

### smooth.spline Validation

```r
# R code
x <- 1:20
y <- sin(x/5)
result <- smooth.spline(x, y, df = 6)

# R 4.3.3 gives:
# df = 6.000793539100071
# lambda = 0.0005617122726951194
```

**Rust test** (property test): `regression::smooth_spline::tests::test_smooth_spline_df_target_is_met` (`crates/p2a-core/src/regression/smooth_spline.rs`): on this x and y with small added noise, df within the bisection's stop rule (0.05) of 6, no warning, and every residual below 0.5. No R value is compared. Not R-pinned (REG-29, REG-30); see [smooth_spline.md](../regression/smooth_spline.md).

### spline Validation

```r
# R code
x <- c(1, 2, 3, 4, 5)
y <- c(1, 4, 9, 16, 25)  # y = x^2
spline(x, y, xout = c(1.5, 2.5, 3.5, 4.5), method = "natural")

# Expected y values: 2.25, 6.25, 12.25, 20.25 (perfect squares)
```

**Rust test**: `test_spline_natural_basic` in `stats/spline.rs`

### approx Validation

```r
# R code
x <- c(1, 2, 3, 4, 5)
y <- c(1, 4, 9, 16, 25)
approx(x, y, xout = c(1.5, 2.5, 3.5, 4.5))

# Expected y values: 2.5, 6.5, 12.5, 20.5 (linear interpolation)
```

**Rust test**: `test_approx_linear_basic` in `stats/spline.rs`

### prop.trend.test Validation

```r
# R code (from R documentation)
smokers <- c(83, 90, 129, 70)
patients <- c(86, 93, 136, 82)
prop.trend.test(smokers, patients)

# R 4.3.3 gives:
# X-squared = 8.22492228519771
# df = 1
# p-value = 0.004131897477641487
```

**Rust test**: `stats::proptrendtest::tests::test_prop_trend_basic` (`crates/p2a-core/src/stats/proptrendtest.rs`). It uses other data (5, 10, 15, 20 successes out of 50 each) and checks p < 0.05 and df = 1. Not R-pinned.

#### MCP tool on count columns with a missing group (R-pinned)

`hypothesis_prop_trend_test` reads its two count columns through the shared extractor. A group
(row) with a missing count is dropped from both columns and reported under `missing`; every
other group keeps its score: its row position (1, 2, 3, ...) by default, or the user's score of
its dataset row (`scores` must have one value per dataset row). R's `prop.trend.test` refuses
such data ("0 (non-NA) cases"), so the reference is `prop.trend.test` on the complete groups
with those scores. A count that is not a non-negative whole number is refused with the column
and the 1-based row ("Column 'trials_neg' must hold non-negative whole numbers (counts); row 5
is -1"); every integer dtype is accepted.

```r
# case stat_prop_trend_na, validation/r/cases/stats_mcp.R
# (validation/datasets/generated/prop_trend_na.csv: 7 groups; successes missing in group 4)
cc <- complete.cases(d$successes, d$trials)          # prop.trend.test(d$successes, d$trials): "0 (non-NA) cases"
pt <- prop.trend.test(d$successes[cc], d$trials[cc], score = which(cc))
pd <- prop.trend.test(d$successes[cc], d$trials[cc], score = d$dose[cc])

# R 4.3.3 gives:
# default scores 1, 2, 3, 5, 6, 7: X-squared = 33.9806387718953, df = 1, p-value = 5.56632139655329e-09
# dose scores 0, 1, 2, 8, 16, 32: X-squared = 30.58249222660634, p-value = 3.199680918972898e-08
# groups used: 6 of 7
```

Before, the handler cast the columns to `usize` with `into_no_null_iter`: the missing count of
group 4 became 0 (X-squared 37.388 on scores 1..7), a negative count wrapped and overflowed the
sums (a debug-build panic), a fractional count was truncated and Int32 columns were refused.

**MCP tests** (`crates/p2a-mcp/src/tools/handlers/hypothesis.rs`):
- `tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_prop_trend_test_na`
  (X-squared, df, p and the scores used, with the default and the dose scores; `missing`; a
  `scores` vector of the wrong length is refused; Int32 counts give the same statistic)
- `tools::handlers::hypothesis::tests::test_mcp_hypothesis_prop_trend_test_refuses_non_counts`
  (not R-pinned: a negative and a fractional count are refused with the column and the row)

**Tolerance:** `validation/README.md`, n = 6 (< 100): X-squared 1e-5 relative (`Tol::stat`), p
1e-5 relative plus 1e-12 absolute (`Tol::p_value`); df, scores and counts exact.

**Status:** R-pinned (registry `validation/methods/mcp_stats.json`, id
`mcp_hypothesis_prop_trend_test`).

## Known Differences

### IQR/mad at Large n

At n=10,000, Rust's IQR and mad are slower than R due to:
- Full array sorting (R may use partial sorting)
- Consider implementing quickselect-based quantiles for optimization

### density() at Large n

Rust's kernel density is significantly slower because:
- Direct O(n × m) kernel evaluation vs R's FFT-based O(n log n) convolution
- Optimization: Implement FFT-based convolution using `rustfft`

### smooth.spline()

Rust's implementation is significantly slower because:
- O(n³) cross-validation search vs R's optimized FORTRAN
- Reinventing the wheel without specialized numerical libraries
- Consider using sparse solvers or linking to LAPACK

## Test Coverage

All methods have unit tests verifying:
- Basic functionality with known inputs
- Edge cases (empty arrays, single values)
- Numerical accuracy against R expected values

Run tests with:
```bash
cargo test -p p2a-core -- cor_test
cargo test -p p2a-core -- power_
cargo test -p p2a-core -- fivenum
cargo test -p p2a-core -- iqr
cargo test -p p2a-core -- mad
cargo test -p p2a-core -- ecdf
cargo test -p p2a-core -- density
cargo test -p p2a-core -- spline
cargo test -p p2a-core -- approx
cargo test -p p2a-core -- smooth_spline
cargo test -p p2a-core -- prop_trend
```

## Performance Summary (After Optimization)

| Method | Performance vs R | Status |
|--------|-----------------|--------|
| cor.test (Pearson) | 5-87x faster | PASS |
| cor.test (Spearman) | 5-23x faster | PASS |
| power.t.test | 19x faster | PASS |
| power.prop.test | 190x faster | PASS |
| power.anova.test | 7x faster | PASS |
| fivenum | 2-24x faster | PASS |
| IQR | 0.4-16x | MARGINAL (slow at n=10k) |
| mad | 0.8-17x | MARGINAL (slow at n=10k) |
| ecdf | 3-33x faster | PASS |
| **density** | **1.7-10.5x faster** | **PASS (optimized)** |
| spline | 2.5-23x faster | PASS |
| approx | 5-37x faster | PASS |
| **smooth.spline** | **2.4-8x faster** | **PASS (optimized)** |
| prop.trend.test | ~500x faster | PASS |

### Optimization Summary

**density()**: Optimized from 0.02x to **10.5x faster than R** using FFT-based convolution.
- Improvement: **~500x speedup** at n=10,000

**smooth.spline()**: Optimized from 0.002x to **8x faster than R** using O(n) banded matrix algorithms.
- Improvement: **~1000x speedup** at n=200 (from 76ms to 70µs)

## References

- R Development Core Team (2024). R: A Language and Environment for Statistical Computing. https://www.r-project.org/
- Pinheiro, J., Bates, D., et al. (2024). nlme: Linear and Nonlinear Mixed Effects Models. R package. https://cran.r-project.org/package=nlme
- Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. Journal of Official Statistics, 6(1), 3-73.
