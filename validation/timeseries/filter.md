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

## Timing against v0.5.0

Release builds of a scratch harness against f80b0131 (v0.5.0) and against this change, `--no-default-features
--features ci`; one core (`taskset -c 10`), `RAYON_NUM_THREADS=1`, cwd `/dev/shm`; per row nine processes per side,
interleaved, each the median of its calls after ten warm-ups; cells are the median over the processes, the ratio of the
minima in brackets. The machine was heavily loaded (1-minute load average 11-36 on 12 cores), so the absolute times are
preliminary and ratios within about 5% are noise. Measured 2026-11-16.
The AR(0.99) and AR(0.5) `arima_sim` rows are the median of 15 processes per side.

| Method | n | f80b0131 (µs) | this change (µs) | ratio [min] |
|---|---|---|---|---|
| `diffinv`, lag 1, differences 2, with xi | 100 | 2.4 | 2.5 | 1.05 [1.05] |
| `diffinv`, lag 1, differences 2, with xi | 1,000 | 22.0 | 22.2 | 1.01 [1.01] |
| `diffinv`, lag 1, differences 2, with xi | 10,000 | 312.9 | 312.6 | 1.00 [1.00] |
| `runmed`, k = 7, median end rule | 100 | 22.0 | 20.7 | 0.94 [0.93] |
| `runmed`, k = 7, median end rule | 1,000 | 229.9 | 216.4 | 0.94 [0.94] |
| `runmed`, k = 7, median end rule | 10,000 | 2,342.2 | 2,229.5 | 0.95 [0.95] |
| `runmed`, k = 2 floor(n / 20) + 1, median end rule | 100 | 46.8 | 33.4 | 0.71 [0.72] |
| `runmed`, k = 2 floor(n / 20) + 1, median end rule | 1,000 | 3,866.1 | 3,705.0 | 0.96 [0.96] |
| `runmed`, k = 2 floor(n / 20) + 1, median end rule | 10,000 | 462,245 | 464,937 | 1.01 [0.95] |
| `arima_sim`, AR(0.5) | 100 | 4.8 | 3.0 | 0.62 [0.62] |
| `arima_sim`, AR(0.5) | 1,000 | 24.7 | 23.3 | 0.94 [0.94] |
| `arima_sim`, AR(0.5) | 10,000 | 254.8 | 259.8 | 1.02 [1.02] |
| `arima_sim`, AR(0.95) | 100 | 5.1 | 6.0 | 1.18 [1.17] |
| `arima_sim`, AR(0.95) | 1,000 | 26.1 | 27.1 | 1.04 [1.02] |
| `arima_sim`, AR(0.95) | 10,000 | 281.5 | 286.3 | 1.02 [1.01] |
| `arima_sim`, AR(0.99) | 100 | 4.9 | 16.1 | 3.30 [3.35] |
| `arima_sim`, AR(0.99) | 1,000 | 24.7 | 36.7 | 1.49 [1.49] |
| `arima_sim`, AR(0.99) | 10,000 | 253.9 | 276.3 | 1.09 [1.09] |
| `arima_sim`, ARMA(2, 1) (0.5, -0.3; 0.4) | 100 | 6.0 | 3.6 | 0.60 [0.60] |
| `arima_sim`, ARMA(2, 1) (0.5, -0.3; 0.4) | 1,000 | 30.4 | 27.5 | 0.90 [0.90] |
| `arima_sim`, ARMA(2, 1) (0.5, -0.3; 0.4) | 10,000 | 328.0 | 316.3 | 0.96 [0.95] |
| `arima_sim`, MA(2) (0.4, 0.2) | 100 | 5.2 | 2.9 | 0.55 [0.55] |
| `arima_sim`, MA(2) (0.4, 0.2) | 1,000 | 26.5 | 25.1 | 0.95 [0.94] |
| `arima_sim`, MA(2) (0.4, 0.2) | 10,000 | 285.8 | 292.2 | 1.02 [1.03] |
| `arma_acf`, AR(2) (0.5, -0.3), lag_max = n | 100 | 2.1 | 2.3 | 1.10 [1.09] |
| `arma_acf`, AR(2) (0.5, -0.3), lag_max = n | 1,000 | 15.9 | 17.0 | 1.07 [1.06] |
| `arma_acf`, AR(2) (0.5, -0.3), lag_max = n | 10,000 | 204.9 | 205.6 | 1.00 [0.99] |
| `arma_acf`, ARMA(2, 1) (0.5, 0.3; 0.4), lag_max = n | 100 | 51.6 | 2.5 | 0.05 [0.05] |
| `arma_acf`, ARMA(2, 1) (0.5, 0.3; 0.4), lag_max = n | 1,000 | 515.5 | 17.0 | 0.03 [0.03] |
| `arma_acf`, ARMA(2, 1) (0.5, 0.3; 0.4), lag_max = n | 10,000 | 193,873 | 205.3 | 0.00 [0.00] |
| `arma_acf` PACF, ARMA(1, 1) (0.5; 0.3), lag_max = n | 100 | 92.6 | 43.7 | 0.47 [0.47] |
| `arma_acf` PACF, ARMA(1, 1) (0.5; 0.3), lag_max = n | 1,000 | 9,931.2 | 1,350.5 | 0.14 [0.14] |
| `arma_acf` PACF, ARMA(1, 1) (0.5; 0.3), lag_max = n | 10,000 | 2,777,366 | 116,039 | 0.04 [0.04] |

`arima_sim` follows its burn-in: R's p + q + ceiling(6 / log(min root modulus)) replaces a fixed 100 steps, so
fast-decaying models run fewer steps and persistent ones more (AR(0.95): 118, AR(0.99): 598). The AR(0.99) rows
were estimated from the step count at 3.49x / 1.45x / 1.05x; the measured 1.49x and 1.09x are about 3% above, the
same per-step cost the AR(0.5) row shows at n = 10,000 (1.02x where the step count gives 0.99x). `arma_acf` solves
R's linear system once instead of iterating, and its PACF no longer works on subnormal values.
