# Validation: Robust summaries and interpolation tools (fivenum, IQR, mad, ecdf, density, spline, approx)

## Method Overview

Seven MCP tools in `crates/p2a-mcp/src/tools/handlers/stats.rs` read one or two dataset columns
and call the core functions of `p2a_core::stats::robust` and `p2a_core::stats::spline`:

| Tool | Core function | R function |
|---|---|---|
| `stats_fivenum` | `fivenum` | `fivenum` (Tukey's hinges) |
| `stats_iqr` | `iqr` | `IQR(type = 7)` |
| `stats_mad` | `run_mad` | `mad(center = median or mean, constant = 1.4826)` |
| `stats_ecdf` | `run_ecdf` | `ecdf(x)(at)` |
| `stats_density` | `run_density` | `density` (only the number of observations is pinned) |
| `stats_spline` | `spline` | `spline(method = "natural")` |
| `stats_approx` | `approx` | `approx(method = "linear")` |

The tools read their columns through the shared column extractor
(`crate::tools::common::numeric_column` / `numeric_columns`), so integer columns are accepted.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `fivenum`, `IQR`, `mad`, `ecdf`, `approx`, `spline` | R 4.3.3 |

## R case: `stat_descriptive_na`

`validation/r/cases/stats_mcp.R`, case `stat_descriptive_na`, dataset
`validation/datasets/generated/descriptive_na.csv` (15 rows: `id, v, vi, s, xs, ys`). `v` is
missing in rows 2, 9 and 13 (1-based), `vi` is an Int64 column missing in rows 4 and 11, `s` is
text, `xs` is missing in row 5 and `ys` in row 10 (13 complete pairs, no tied `xs`).

```r
fivenum(d$v); IQR(d$v, na.rm = TRUE); mad(d$v, na.rm = TRUE)
mad(d$v, center = mean(d$v, na.rm = TRUE), na.rm = TRUE)
ecdf(d$v[!is.na(d$v)])(c(15, 20, 25))
approx(d$xs, d$ys, xout = c(1.5, 4, 6.5, 8))                       # drops the incomplete pairs
spline(d$xs, d$ys, method = "natural", xout = c(1.5, 4, 6.5, 8))
```

| Quantity | R value |
|---|---|
| `fivenum(v)` | 10.954, 18.241, 21.0075, 25.595999999999997, 30.132 |
| `IQR(v)` / `mad(v)` / `mad(v, center = mean)` | 6.390499999999996 / 4.776937199999999 / 4.776937199999999 |
| `mean(v)` (the centre used) | 21.259916666666665 |
| `ecdf(v)` at 15, 20, 25 | 0.08333333333333333, 0.4166666666666667, 0.75 |
| values of `v` used | 12 |
| `fivenum(vi)` | 68, 85, 92, 101, 124 |
| `IQR(vi)` / `mad(vi)` / values used | 16 / 13.343399999999999 / 13 |
| `approx` at 1.5, 4, 6.5, 8 | 0.8082820053715308, -0.7521932489451476, 0.3538291876750702, 0.9312373477672532 |
| natural `spline` at 1.5, 4, 6.5, 8 | 0.7478985384585353, -0.7661760653678952, 0.3201682303488883, 1.128381748164739 |
| complete (x, y) pairs | 13 |

## Missing values

| Tool | Policy |
|---|---|
| `stats_fivenum`, `stats_iqr`, `stats_mad`, `stats_ecdf`, `stats_density` | the column's missing values (null or NaN) are dropped, as `na.rm = TRUE`, and reported under `missing`; `n` (and `data_n` for the density) counts the values used |
| `stats_spline`, `stats_approx` | rows with a missing x or y are dropped together, as R's `regularize.values` drops them, and reported; `n_knots` counts the pairs used |

`±Inf` is refused by the extractor (`NonFiniteValues`), a text column with "Column 's' contains
non-numeric values", an absent column with "Column 'nope' not found in dataset. ...". Nothing
is added to the JSON when no value was dropped.

## Tests

- `tools::handlers::stats::tests::test_validate_mcp_stats_robust_summaries_na` (p2a-mcp: the
  five-number summary, IQR, both MADs and the mean centre, the ECDF and `n` on `v` and on the
  Int64 `vi`; the density's `data_n`; `missing` of every call; the text and absent columns)
- `tools::handlers::stats::tests::test_validate_mcp_stats_spline_approx_na` (p2a-mcp: `approx`
  and the natural `spline` at `xout`, `n_knots`, `missing`)

**Tolerance:** `validation/README.md`, n = 12 to 15 (< 100): order statistics, centres and ECDF
values 1e-6 (`Tol::coef`), IQR, MAD and interpolated values 1e-5 relative (`Tol::stat`), counts
exact.

**Status:** R-pinned with known deviations (registry `validation/methods/mcp_stats.json`, id
`mcp_stats_robust_and_interpolation`).

## Known Differences

`stats_iqr` computes R's quantile types 1-9 since T3-WP02 (STAT-36 is fixed in the core
`iqr`; see the core section below); only its default type 7 is pinned through the MCP tool.

- **STAT-37:** the density's FFT grid is mislabelled when the number of grid points is not a
  power of two; the density values are not pinned.
- **STAT-65:** the default density bandwidth is not `bw.nrd0`, and the non-Gaussian kernels
  scale `bw` differently from R; the bandwidth is not pinned.
- **STAT-69:** `stats_spline`'s default method `fmm` is a clamped spline, not R's
  Forsythe-Malcolm-Moler end conditions; only `method = "natural"` is pinned.
- **STAT-87:** `stats_fivenum`'s `iqr` field is the spread of the hinges, not R's `IQR`; it is
  not pinned.

---

## Core: quantile types, box-plot statistics and the distribution helpers

### Overview

`p2a_core::stats::robust` computes sample quantiles of R's nine types (`quantile`, `quantiles`,
`quantile_sorted`, `QuantileType`), `IQR(type)` (`iqr`), Tukey's five-number summary (`fivenum`),
`grDevices::boxplot.stats` (`boxplot_stats`) and the bootstrap percentile interval
(`percentile_interval`). `p2a_core::traits::estimator` holds the shared distribution helpers:
the upper-tail p-values `t_test_p_value` (two-sided), `chi_squared_p_value`, `f_test_p_value` and
`normal_p_value_two_sided`, and the quantiles `normal_quantile`, `z_critical`, `t_quantile`,
`t_critical` and `chi_squared_quantile`, with `validate_level` for levels and probabilities.

### Algorithm

- **Quantiles.** R's `quantile.default` definitions (Hyndman and Fan 1996). Types 1-3 are
  discontinuous: `nppm = n p` (type 3: `n p - 1/2`), `j = floor(nppm)`, and `h` is 1 when
  `nppm > j` (type 1), the average of the two order statistics when `nppm = j` (type 2), or the
  round-half-even rule (type 3). Types 4-9 interpolate at `nppm = a + p (n + 1 - a - b)` with
  (a, b) = (0, 1), (.5, .5), (0, 0), (1, 1), (1/3, 1/3), (3/8, 3/8); like R,
  `j = floor(nppm + 4 eps)` and `h` below `4 eps` counts as 0 (the fuzz rule), so a probability
  that lands on an order statistic up to rounding returns that order statistic. Type 7 uses
  `1 + (n - 1) p`. A type outside 1..9 is an `InvalidSpecification` (it used to fall back to
  type 7, STAT-36). `iqr` sorts once and takes both quartiles from the sorted copy.
- **fivenum** uses R's index formula `0.5 (x[floor(d)] + x[ceiling(d)])` at
  `d = 1, n4, (n + 1)/2, n + 1 - n4, n`, `n4 = floor((n + 3)/2)/2`. **boxplot_stats** takes the
  hinges from it, marks as outliers the values beyond `coef` times the hinge spread (none for
  `coef = 0`), reports them in data order, and moves the whiskers to the most extreme values
  that are not outliers. **percentile_interval** is type 7 at `(1 - level)/2` and `(1 + level)/2`.
- **Distribution helpers.** The tails are computed directly, in log space, not as `1 - cdf`:
  the incomplete beta continued fraction (t and F) and the incomplete gamma series and continued
  fraction (chi-square) that statrs 0.18 uses, with the prefactor kept in logs so no tail
  underflows above about 1e-308. For t with df >= 1000 the four-term Cornish-Fisher expansion
  (Fisher 1925; Abramowitz and Stegun 26.7.5) is used where `z^2 <= 0.005 df`, where its
  truncation error is below double precision. Quantiles are a safeguarded Newton iteration on
  the log tail in `u = ln |x|`. The normal quantile is statrs's (Boost `erf_inv`); the
  two-sided normal tail is the chi-square(1) upper tail of `z^2`, because statrs's `erfc` is only
  accurate to about 5e-11 relative.
  These replace statrs calls with these defects (measured against R 4.3.3):
  1. `StudentsT::inverse_cdf` does not terminate for df >= 1.5e7 (and at q = 1e-100, df = 1000):
     an OLS confidence interval on 15 million rows hung.
  2. It panics for q < 1e-154.
  3. It is wrong at large df: 1.959847719928278 at (0.975, 1e6) against R's
     1.9599663568141066 (6.1e-5 relative), 6.8% off at 1e7.
  4. `ChiSquared::sf` returns 1 for x < 1.1e-15 at any df (R: 0.16873527823624623 at x = 1e-16,
     df = 0.01); `ChiSquared::inverse_cdf` returns a coarse bisection point when the density
     overflows (STAT-60's cause, e.g. `qchisq(0.025, 318)`) and is wrong in the lower tail for
     df <= 1.5.
  5. `FisherSnedecor::sf` loses accuracy far in the tail (8.3e-9 relative at F = 1e8, df 3 and 40).
  And `1 - cdf` returned exactly 0 for every p-value below about 1e-16 (X-17).

### Missing values

- `quantile`, `iqr`, `fivenum` and `boxplot_stats` drop NaN (`na.rm = TRUE`); `boxplot_stats`'s
  `n` counts the values kept.
- `quantiles` and `percentile_interval` refuse NaN (R's `quantile` without `na.rm`); drop failed
  bootstrap replicates first and report their number.
- `±Inf` is kept, as in R: `quantiles(c(1, Inf), c(0, .5, 1))` is `1, Inf, Inf`.

### R reference

`validation/r/cases/stats_quantile.R`, case `stat_quantile_types`, dataset
`validation/datasets/generated/quantile_sample.csv` (28 rows: `x` with ties, `y` with 7 values and
21 NA).

Quantiles at p = 0.25 (before this change p2a returned 8.04575, type 7, for every type except 1):

| type | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| `quantile(x, .25)` | 7.511 | 7.8675 | 7.511 | 7.511 | 7.8675 | 7.68925 | 8.04575 | 7.808083333333334 | 7.8229375 |
| `IQR(x, type)` | 4.489 | 4.1325 | 4.489 | 4.489 | 4.1325 | 4.31075 | 3.95425 | 4.191916666666666 | 4.1770625 |
| `quantile(1:10, .25)` | 3 | 3 | 2 | 2.5 | 3 | 2.75 | 3.25 | 2.9166666666666665 | 2.9375 |

Further keys: `fivenum(x)` = 5.284, 7.8675, 9.3155, 12, 15.197; `boxplot.stats(x, coef = 0.5)`
stats 7.207, 7.8675, 9.3155, 12, 13.476 with outliers 5.722, 5.704, 5.284, 15.197, 14.129, 15.044;
DATA-62's example `boxplot.stats(c(1, 7, 12, 15, 15, 16, 18, 18))` = 1, 9.5, 15, 17, 18; the
type 9 gap case `quantile(c(rep(-1e20, 7), 1:21), 0.26991150442477874, type = 9)` = 1 (without the
fuzz it would be -88816.84); percentile intervals of x at 0.95: 5.567500000000001, 15.093725.

Distribution helpers (inputs and R values):

| key | inputs | R |
|---|---|---|
| `2*pt(-abs(t), df)` | (16.967101614364047, 197), (13.5, 197), (2, 1e6), (10, 4e5) | 2.165754515305109e-40, 7.793084732270165e-30, 0.045500533851319185, 1.5337144421476894e-23 |
| `pchisq(x, df, lower = FALSE)` | (100, 1), (1e-16, 0.01), (1000, 100) | 1.5239706048321054e-23, 0.16873527823624623, 2.306076738035398e-148 |
| `pf(F, d1, d2, lower = FALSE)` | (1e8; 3, 40), (60; 10, 1e6) | 1.6208895078118348e-137, 1.9221028081786625e-122 |
| `qt(p, df)` | (0.975, 1e6), (0.975, 2e7), (1e-10, 0.5) | 1.9599663568141066, 1.9599641031536221, -1.0284911563163566e+19 |
| `qchisq(p, df)` | (0.025, 318), (1e-10, 0.5), (0.6, 0.01) | 270.492276865318, 1.3499395786223513e-40, 4.8126039477452196e-45 |

The case holds 15 t, 8 chi-square and 7 F p-values, 5 normal p-values, 7 normal quantiles,
4 normal critical values, 17 t quantiles and 12 chi-square quantiles.

### Rust tests

- `stats::robust::tests::test_validate_quantile_legacy_api_types`, `test_validate_iqr`,
  `test_quantile_rejects_type_0_and_12`, `test_validate_quantile_types_1_9`,
  `test_validate_quantile_fuzz_rule`, `test_quantiles_refuses_nan_and_bad_probabilities`,
  `test_validate_percentile_interval` (method `quantile_types`)
- `stats::robust::tests::test_validate_fivenum`, `test_validate_boxplot_stats` (method
  `boxplot_stats`)
- `traits::estimator::tests::test_t_p_value_tail_is_not_zero`,
  `test_validate_p_value_helpers_stat_quantile_types`,
  `test_validate_normal_helpers_stat_quantile_types`,
  `test_validate_quantile_helpers_stat_quantile_types`, `test_t_critical_large_df_matches_r`,
  `test_t_critical_terminates_for_huge_df`, `test_default_confidence_intervals_large_df`
  (method `distribution_helpers`)

**Tolerance:**

- Quantiles, IQR and percentile intervals: the `validation/README.md` closed-form tolerance
  (`Tol::coef(n)`: n = 28, 7, 10); the fuzz and gap cases exact (they test which order statistic
  is chosen).
- `fivenum` and `boxplot.stats`: `Tol::coef(n)` for the stats; `n`, the number of outliers and
  the outliers themselves exact.
- Distribution helpers: relative 1e-11 with no absolute floor (`Tol::new(1e-11, 0.0)`), because
  X-17 concerns p-values that `Tol::p_value`'s 1e-12 floor would accept as 0.
  Measured maxima: 1.5e-13 (pt), 1.1e-13 (pchisq), 6.1e-13 (pf), 3.2e-15 (two-sided normal),
  3.6e-13 (qt), 8.0e-13 (qchisq), 5.4e-16 (qnorm). The quantile, box-plot and percentile keys
  agree exactly.

**Status:** R-pinned (registry `validation/methods/stats_quantile.json`, ids `quantile_types`,
`boxplot_stats`, `distribution_helpers`).

### Known deviations

None for the three methods of `stats_quantile.json`. Regions where R's own `qt` is inaccurate
(df < 1 with 1 - p < 1e-6, and p = 1e-300, where R's `pt` of R's `qt` misses p by up to 1.5%) are
not pinned. `silverman_bandwidth` keeps its own index rule for the IQR (STAT-65, Tier 4) until the
default bandwidth becomes `bw.nrd0`.

### Corrections to earlier statements

- The `robust.rs` documentation claimed quantile types 1-9 while types 2-6, 8 and 9 were
  computed as type 7 (STAT-36).
- The Tier 3 design paired `t_test_p_value(13.5, 197)` with 2.165754515305109e-40; R gives
  7.793084732270165e-30 for t = 13.5, and 2.165754515305109e-40 is the p-value of the HC1 t of
  `value` in `ols_grunfeld` (16.967101614364047).
