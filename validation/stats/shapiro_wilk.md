# Validation: Shapiro-Wilk Normality Test

## Method Overview

This document validates the p2a-core implementation of the Shapiro-Wilk normality test against R's `stats::shapiro.test()`.

**Functions Implemented:**
- `shapiro_wilk_test()` - test on a slice of values
- `run_shapiro_wilk()` - dataset wrapper for one column

**Key Parameters:**
- Input: numeric vector x (3 ≤ n ≤ 5000 after dropping missing values)
- Output: W statistic, p-value, n, `reject_normality` (p < 0.05), `missing` (dataset wrapper)

**Null Hypothesis:** H₀: The data is normally distributed
**Alternative Hypothesis:** H₁: The data is not normally distributed

**Status:** R-pinned (registry `validation/methods/stats_onesample.json`, id `shapiro_wilk`).

## Reference Implementation

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `shapiro.test()` (C code `src/library/stats/src/swilk.c`, AS R94) | R 4.3.3 |

## Algorithm

`shapiro_wilk_test` is a port of R's `swilk.c` (Royston 1995, Algorithm AS R94) and of the
checks `shapiro.test` makes before calling it:

1. Missing values (NaN) are dropped. An infinite value is refused. A sample with fewer than 3
   or more than 5000 values is refused ("sample size must be between 3 and 5000").
2. The sample is sorted. A range below 1e-10 is refused ("all 'x' values are identical"); a
   range below 1 divides every value by the range (R PR#1768), so W and p do not depend on
   the scale. A range that overflows (finite values such as -1e308 and 1e308) is refused
   with "the range of x (max - min) overflows; rescale the values", where R's `shapiro.test`
   returns W = NaN and p = NA (`test_shapiro_wilk_refuses_overflowing_range`).
3. Coefficients: n = 3 uses a = (-√½, 0, √½). Otherwise the Blom scores
   mᵢ = Φ⁻¹((i - 3/8) / (n + 1/4)) are normalised, the largest coefficient aₙ (and for n > 5
   also aₙ₋₁) is replaced by Royston's polynomial correction in 1/√n (swilk.c `c1`, `c2`), and
   the other coefficients are rescaled so that Σ aᵢ² = 1.
4. W is the squared correlation of the data and the coefficients, computed through 1 - W in
   swilk.c's order of operations.
5. p-value:
   - n = 3: the exact distribution, p = (6/π)(asin √W - π/3);
   - 4 ≤ n ≤ 11: y = -ln(γ - ln(1 - W)) with γ = -2.273 + 0.459 n, standardised with mean
     `poly(c3, n)` and standard deviation `exp(poly(c4, n))`; p is the upper normal tail;
   - 12 ≤ n ≤ 5000: ln(1 - W) standardised with mean `poly(c5, ln n)` and standard deviation
     `exp(poly(c6, ln n))`; p is the upper normal tail.

## Missing values and column types

`run_shapiro_wilk` reads its column through the shared extractor (`p2a_core::data::columns`)
with the `Omit` policy, as `shapiro.test` does (`x[complete.cases(x)]`): null and NaN cells are
dropped and counted in `ShapiroWilkResult.missing`, and the Display output ends with a line
`Missing values: <k> of <n> rows dropped because of missing values (<column>: <k>)` when any
were dropped. Integer, decimal and boolean columns are accepted; a text column is refused with
`NonNumericColumn`, an infinite value with `NonFiniteValues`.

## R Reference

Case `stat_shapiro_wilk` in `validation/r/cases/stats_onesample.R` (dataset
`validation/datasets/generated/shapiro_cases.csv`: 17 sample columns, each padded with missing
values to 5000 rows; `n25_int` is an Int64 column):

```r
set.seed(42)   # columns as in the case file
s <- shapiro.test(d[[col]])   # statistic, p.value and sum(!is.na(d[[col]])) for each column
```

## Results

| Column | n | W (R) | p (R) |
|---|---|---|---|
| n3_124 (1, 2, 4) | 3 | 0.96428571428571419 | 0.6368868450289632 |
| n3_123 (1, 2, 3) | 3 | 1 | 0.99999999999999334 |
| n4 | 4 | 0.87603183393998441 | 0.32196480117232057 |
| n5 (1.2, 2.3, 3.1, 4.5, 5.2) | 5 | 0.96961715622637146 | 0.87279076733163286 |
| n8_rnorm | 8 | 0.94017272477637004 | 0.61278059351291658 |
| n10_unif ((1:10)/10) | 10 | 0.97016461123066544 | 0.89236730752391935 |
| n11_rnorm | 11 | 0.94630500428750619 | 0.59717576421735885 |
| n12_rnorm | 12 | 0.95121761971182228 | 0.654849716063022 |
| n15_exp | 15 | 0.68846396016921318 | 0.00018757005337063294 |
| n20_rnorm | 20 | 0.89558965209753827 | 0.034116318620098424 |
| n30_exp | 30 | 0.87618064676822194 | 0.0023064625443275488 |
| n50_rnorm | 50 | 0.97525514085910248 | 0.37320315265855797 |
| n100_qnorm (`qnorm(ppoints(100))`) | 100 | 0.99956292252754664 | 0.99999999999991407 |
| n200_exp | 200 | 0.90025966178453187 | 2.610474881347321e-10 |
| n40_ties | 40 | 0.94072174591887936 | 0.036573998564684469 |
| n25_int (Int64) | 25 | 0.95657428069779105 | 0.35047858533073967 |
| n5000_rnorm | 5000 | 0.99965571348085192 | 0.5723573129367866 |

All 17 samples match R within the tolerances below, through the library, the dataset wrapper
(NA-padded Float64 and Int64 columns) and the MCP tool `hypothesis_shapiro_wilk`.

**Rust tests:**
- `stats::shapiro::tests::test_validate_shapiro_wilk_swilk` (all 17 samples, `shapiro_wilk_test`)
- `stats::shapiro::tests::test_validate_shapiro_wilk_columns` (all 17 columns, `run_shapiro_wilk`, and `missing`)
- `stats::shapiro::tests::test_validate_shapiro_wilk_rescales_small_range` (range below 1 and the 1e-10 bound)
- `stats::shapiro::tests::test_validate_na_shapiro_wilk_columns` (a NaN cell, the Display line, text and absent columns)
- `tier2::wp06_onesample::test_validate_mcp_hypothesis_shapiro_wilk` (p2a-mcp, four columns)

**Tolerance:** `validation/README.md` by n: W `Tol::stat(n)`, p `Tol::p_value(n)`, n exact.

## Corrections to the previous version of this document (STAT-67)

The previous version quoted R values that R does not produce, and its tests checked ranges only:

| Sample | Quoted before | R 4.3.3 `shapiro.test` |
|---|---|---|
| `set.seed(42); round(rnorm(20), 4)` (1.3710, -0.5647, ...) | W 0.9341, p 0.1836 | W 0.9440295936, p 0.2853909932 |
| c(1.2, 2.3, 3.1, 4.5, 5.2) | W 0.9856, p 0.9609 | W 0.9696171562, p 0.8727907673 |
| c(0.05, 0.12, ..., 12.96) (n = 15) | W ~0.75 | W 0.7648546811, p 0.00134718194 |

Before the port, p2a's W lacked Royston's coefficient corrections and each p-value branch used
a wrong formula or tail (STAT-1): n = 5 gave W 0.9866, p 0 against R's 0.9696, 0.8728.

## Performance Comparison

*Measured on 2026-01-19 before the T2-WP06 port (`cargo bench -p p2a-core -- ShapiroWilk`); not
re-measured. The port evaluates n/2 normal quantiles with statrs instead of a truncated
rational approximation, so the Rust timings can change.*

| Dataset Size | Rust (µs) | R (µs)* | Speedup |
|--------------|-----------|---------|---------|
| n=10         | 0.61      | ~30     | ~49x    |
| n=50         | 1.93      | ~45     | ~23x    |
| n=100        | 3.72      | ~55     | ~15x    |
| n=500        | 16.3      | ~100    | ~6x     |
| n=1,000      | 34.4      | ~150    | ~4x     |
| n=2,000      | 72.5      | ~250    | ~3x     |
| n=5,000      | 200.5     | ~500    | ~2.5x   |

*Note: R timings estimated from the microbenchmark package (run with `Rscript benchmark_shapiro.R` in the companion paper repo).*

## References

- Shapiro, S. S. & Wilk, M. B. (1965). "An analysis of variance test for normality
  (complete samples)". *Biometrika*, 52(3-4), 591-611.
- Royston, J. P. (1982). "An extension of Shapiro and Wilk's W test for normality
  to large samples". *Journal of the Royal Statistical Society Series C*, 31(2), 115-124.
- Royston, P. (1995). "Remark AS R94: A remark on Algorithm AS 181: The W-test
  for normality". *Journal of the Royal Statistical Society Series C*, 44(4), 547-551.
- R source: `src/library/stats/src/swilk.c` and `src/library/stats/R/shapiro.test.R`.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/shapiro.test.html
