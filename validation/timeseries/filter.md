# Linear filter, discrete integration, running median, ARIMA simulation, theoretical ARMA ACF

`filter`, `diffinv`, `runmed`, `arima_sim` and `arma_acf` in `crates/p2a-core/src/forecasting/tsutils.rs`.
R reference cases: `ts_filter_ar_cpt` (`validation/r/cases/ts_utils.R`), series `x` of
`validation/datasets/generated/ts_utils.csv` (a 60-value random walk), and `ts_utils_t4`
(`validation/r/cases/ts_utils_t4.R`, `validation/datasets/generated/ts_utils_t4.csv`).

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

Pinned: `diffinv(x[1:12], xi = 1.5)`, `diffinv(x[1:12], lag = 2, xi = c(1, -2))`, `diffinv(x[1:12], differences = 2)`
and `diffinv(x[1:12], differences = 2, xi = c(1, 3))`, in `test_validate_diffinv_ts_filter_ar_cpt` at `Tol::coef(12)`.
Case `ts_utils_t4` adds differences 2 and 3 with start values at lags 1, 2 and 3 (`dinv_*`), in
`test_validate_diffinv_xi_ts_utils_t4` at 1e-12 absolute (dyadic inputs: the sums are exact).

**Algorithm.** R's recursion `diffinv(x, lag, d, xi) = diffinv(diffinv(x, lag, d - 1, diff(xi, lag)), lag, 1,
xi[1:lag])`, unrolled: with D^k the k-th lag difference of `xi`, the series is integrated for k = d - 1 down to 0 with
the first `lag` values of D^k as start values. `xi` holds the first `lag * differences` values of the undifferenced
series; any other length is `InvalidSpecification` (R: "'xi' does not have the right length"), and `None` means zeros.
`test_diffinv_refuses_wrong_xi_length` holds the refusal and the round trip `diffinv(diff(x, 2), xi = x[1:2]) == x`.

## runmed

`runmed(x, k, endrule)` follows `stats::runmed`. The body is the running median of k values; `Keep` keeps the k / 2
values at each end, `Constant` repeats the outermost running median, and `Median` is R's default
`smoothEnds(runmed(x, k, endrule = "keep"), k)`: for i = 2..=k / 2 (while 2 i <= n) the i-th value from each end is the
median of the 2 i - 1 values of the kept series at that end, and the first and last values follow Tukey's end-point
rule med3(y[1], sm[2], 3 sm[2] - 2 sm[3]).

Pinned: case `ts_utils_t4` (`rm_*`: k = 3, 5, 7, 9 and k = n), in `test_validate_runmed_median_ts_utils_t4` at 1e-12
absolute; the MCP tool in `test_validate_mcp_runmed_median` (`mcp_series_t3`, `runmed_median_head`).

## arima_sim

`arima_sim` and `arima_sim_with_sd` follow `stats::arima.sim`: innovations `sd * N(0, 1)` (the `n_start` burn-in values
first, then `n`; supplied innovations are used as given), the MA filter with its first q values set to 0, the AR
recursion from zeros, the burn-in dropped, then d integrations. R returns `n + d` values for `d > 0`, the first d of
them 0; p2a returns the rest.

| Rule | p2a and R |
|---|---|
| Stationarity | a root of 1 - phi_1 z - ... - phi_p z^p on or inside the unit circle is refused ("'ar' part of model is not stationary") |
| Default burn-in | `p + q + ceiling(6 / log(minroots))` (`p + q` for a pure MA); minroots in closed form for p <= 2 (after dropping trailing zero coefficients, as `polyroot` does), from the companion matrix's eigenvalues above |
| Short burn-in | `n_start < p + q` is refused ("burn-in 'n.start' must be as long as 'ar + ma'") |
| `sd` | finite and positive |

Pinned: `sim_arma21` (ARMA(2, 1), `n.start = 13`) and `sim_arima111` (ARIMA(1, 1, 1), `n.start = 11`) from given
innovations, and R's default `n.start` for seven models (10, 598, 118, 13, 2, 29, 11), in
`test_validate_arima_sim_given_innovations_ts_utils_t4`. Random draws use p2a's RNG, not R's, and are not compared.
The burn-in grows like 6 / log(minroots) as the AR part nears a unit root (AR(0.99): 598); it is allocated through the
allocation ceiling and runs under the call's budget.

## arma_acf

Pinned: `ARMAacf(ar = 0.8)`, `ARMAacf(ar = c(0.5, -0.3))` with its PACF (lags 1..10; R omits lag 0),
`ARMAacf(ma = c(0.4, 0.25))`, `ARMAacf(ar = 0.5, ma = 0.3)` and the persistent `ARMAacf(ar = c(1.5, -0.56))`, in
`test_validate_arma_acf_ts_filter_ar_cpt`. Case `ts_utils_t4` (`validation/r/cases/ts_utils_t4.R`) adds mixed
ARMA(1, 2), ARMA(2, 1) and ARMA(3, 2) with their PACF, AR(0.99), and `lag.max` below p and below q + 1, in
`test_validate_arma_acf_mixed_ts_utils_t4`. Both at 1e-10 relative plus 1e-12 absolute (closed form).

**Algorithm.** As R's `ARMAacf`: with r = max(p, q + 1), the AR part padded with zeros to r terms and sigma^2 = 1, the
autocovariances gamma(0..r) solve the r + 1 equations
gamma(k) - sum_{i=1..r} phi_i gamma(|k - i|) = sum_{j=k..q} theta_j psi_{j-k} (theta_0 = psi_0 = 1), by Gaussian
elimination with partial pivoting; rho(k) = gamma(k) / gamma(0), extended by rho(k) = sum_i phi_i rho(k - i) beyond r.
AR(1) gives phi^k, a pure MA the closed form. The PACF is the Durbin-Levinson recursion on that ACF (R's C `pacf1`).

**Differences from R (not pinned).**
- A non-stationary AR part (a root of 1 - phi_1 z - ... - phi_p z^p on or inside the unit circle, found by the
  step-down recursion) is refused with `InvalidSpecification`. R does not check: `ARMAacf(ar = 1.2, lag.max = 3)`
  returns 1, 1.2, 1.44, 1.728 and `ar = c(1.5, -0.5)` fails in `solve`.
- The result always has `lag_max + 1` values. R returns q + 1 values for a pure MA with `lag.max < q`
  (`acf_ma3_lag5` records it), two values for an AR model with `lag.max = 0`, and crashes on the latter with
  `pacf = TRUE`.
