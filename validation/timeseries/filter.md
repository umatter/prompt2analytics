# Linear filter, discrete integration, theoretical ARMA ACF

`filter`, `diffinv` and `arma_acf` in `crates/p2a-core/src/forecasting/tsutils.rs`.
R reference case: `ts_filter_ar_cpt` (`validation/r/cases/ts_utils.R`), series `x` of
`validation/datasets/generated/ts_utils.csv` (a 60-value random walk).

## filter

`filter(x, f, method, sides, init)` follows `stats::filter`.

| Rule | p2a | R |
|---|---|---|
| Convolution, `sides = 1` | y[i] = sum_j f[j] x[i - j] | the same |
| Convolution, `sides = 2` | y[i] = sum_j f[j] x[i + p/2 - j], so the window is x[i + p/2 - p + 1 ..= i + p/2] | C `cfilter`, `nshift = length(filter) %/% 2` |
| Outside the series | NaN | NA |
| Filter longer than the series (convolution) | `InvalidSpecification` | "'filter' is longer than time series" |
| NaN coefficient | `InvalidSpecification` | "missing values in 'filter'" |
| Recursive | y[i] = x[i] + sum_j f[j] y[i - 1 - j] | C `rfilter`, no length limit |

For an even length the window has one more value after position i than before it, as in R. The finding's
example on x = 1, 4, 2, 8, 5, 7, 3, 9, 6, 10: `filter(x, rep(.25, 4), sides = 2)` is
NA 3.75 4.75 5.5 5.75 6 6.25 7 NA NA.

**Convention kept as p2a's.** `init` holds the p values before `x[0]` oldest first; R's `init` is newest first. p2a
also pads or truncates `init` where R requires `length(init) == length(filter)`. The MCP tool never passes `init`.

**Tests.** `test_validate_filter_even_sides2`, `test_validate_filter_odd_sides2`,
`test_validate_filter_even_sides1`, `test_validate_filter_odd_sides1`, `test_validate_filter_recursive`
at `Tol::coef(60)`; `test_filter_even_two_sided_matches_r_cfilter`, `test_filter_longer_than_series_is_an_error` and
`test_filter_rejects_missing_coefficients` hold the example and the refusals.

**Wall-clock (signed off 2026-11-08).** Convolution with p = 3, medians of the round medians over five interleaved
rounds against `ff8e7559`, `release`, one P core, `RAYON_NUM_THREADS=1`, on a quiet machine (load average 1.1-2.0,
2026-11-08). Ratio is this tree over the base.

| Row | n | base (µs) | this tree (µs) | ratio (median) | ratio (min) |
|---|---|---:|---:|---:|---:|
| `sides = 1` | 100 | 0.203 | 0.244 | 1.20 | 1.23 |
| `sides = 1` | 1,000 | 1.577 | 2.003 | 1.27 | 1.27 |
| `sides = 1` | 10,000 | 15.73 | 19.82 | 1.26 | 1.26 |
| `sides = 2` | 100 | 0.210 | 0.248 | 1.18 | 1.18 |
| `sides = 2` | 1,000 | 1.707 | 2.001 | 1.17 | 1.17 |
| `sides = 2` | 10,000 | 17.06 | 19.78 | 1.16 | 1.16 |

The rewrite follows R's `cfilter` at the edges (NaN wherever the window leaves the series or holds a NaN) and checks
the call's budget once per block of outputs. About 0.4 ns more per output value; the cause was not isolated (variants
of the coefficient loop and an unblocked loop with a single check measured the same). Signed off by the user.

## diffinv

Pinned: `diffinv(x[1:12], xi = 1.5)`, `diffinv(x[1:12], lag = 2, xi = c(1, -2))` and
`diffinv(x[1:12], differences = 2)`, in `test_validate_diffinv_ts_filter_ar_cpt` at `Tol::coef(12)`.

**Deviation TS-28.** With `differences >= 2` and non-zero `xi`, p2a splits `xi` per level and R does not. The case
records `diffinv_d2_xi` (R 1, 3, 5.3503169975, ...; p2a is off by up to 11); the test does not compare it.

## arma_acf

Pinned: `ARMAacf(ar = 0.8)`, `ARMAacf(ar = c(0.5, -0.3))` with its PACF (lags 1..10; R omits lag 0) and
`ARMAacf(ma = c(0.4, 0.25))`, in `test_validate_arma_acf_ts_filter_ar_cpt` at 1e-10 relative plus 1e-12 absolute: the
AR recursion stops at a change in gamma(0) below 1e-12.

**Deviation TS-14.** The fixed-point code is off for a mixed model and a persistent one. The case records
`armaacf_arma11` (off by 8.1e-2) and `armaacf_ar2_persistent` (off by 3.6e-5); the test does not compare them.
