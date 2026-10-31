# Validation: ARIMA

Registry: [`validation/methods/forecasting_arima.json`](../methods/forecasting_arima.json)
(methods `arima_css` and `arima_forecast`). R cases: `arima_css_forecast`,
`arima_fixed_forecast` and `arima_misspecified_seasonal` in
[`validation/r/cases/forecasting_arima.R`](../r/cases/forecasting_arima.R), on
`validation/datasets/generated/arima_series.csv` (n = 300) and
`validation/datasets/generated/arima_misspecified_seasonal.csv` (n = 1000), both written by
the case file.

Functions: `p2a_core::forecasting::{run_arima_with, forecast_arima_with, ArimaSpec}`;
`run_arima` and `forecast_arima` are the same with R's defaults (`ArimaSpec::new(p, d, q)`).
MCP tools `ts_arima_fit`, `ts_arima_forecast`; CLI `p2a ts arima`.

## Method

R's parameterisation, with the MA sign `+theta` and the mean and drift in the levels:

```
phi(B) (1 - B)^d (x_t - mu - delta t) = theta(B) e_t,   t = 1, ..., n
phi(B)   = 1 - phi_1 B - ... - phi_p B^p
theta(B) = 1 + theta_1 B + ... + theta_q B^q
```

`mu` is the series mean (R's `intercept`) and `delta` the drift (forecast::Arima's
`drift`). Coefficients are reported in R's order and names: `ar1..arp`, `ma1..maq`,
`intercept`, `drift`.

## Estimation

Conditional sum of squares, as `stats::arima(method = "CSS")` and
`forecast::Arima(method = "CSS")`. R's default method is CSS-ML, which p2a does not
implement (known difference 1).

- `w = (1 - B)^d (x - mu - delta t)`, length n - d.
- `e_t = w_t - sum_i phi_i w_{t-i} - sum_j theta_j e_{t-j}` for the observations after the
  first d + p, which condition the fit (their residuals are 0, and the MA sum reaches back
  only to residuals after them), as R's `ARIMA_CSS`.
- The coefficients minimise SSR = sum e_t^2. Start values are R's: AR and MA coefficients
  0, mean and drift by least squares of the differenced series on their regressors.
- Levenberg-Marquardt with the analytic Jacobian of the recursion (accumulated into J'J
  and J'e row by row). An iteration takes the Gauss-Newton step when the largest
  `|step_k| / (1 + |coef_k|)` is at most 1e-8 (converged, after a final Gauss-Newton step
  that is kept if it does not raise SSR), and otherwise damped steps
  `(J'J + lambda diag(J'J)) step = -J'e` with lambda from 1e-3, under Nielsen's (1999)
  gain-ratio rule: rho is the actual decrease of SSR over the decrease the linearised model
  predicts, `step' (lambda diag(J'J) step - J'e)`. A step with rho > 0 is accepted and
  lambda multiplied by max(1/3, 1 - (2 rho - 1)^3); otherwise lambda is multiplied by nu,
  which then doubles (nu = 2 again after an accepted step). Above lambda = 1e16 the rounding
  floor is reached and the fit counts as converged when the scaled Gauss-Newton step is at
  most 1e-6; the Gauss-Newton step is then tried as at the 1e-8 test (a step that leaves SSR
  unchanged has rho = 0 and is rejected, so the floor can come a little before that test).
  At most 200 iterations; a fit that stops short is reported with `converged = false` and a
  warning, not as an error.
- Why the gain ratio: where the model leaves autocorrelated residuals, J'J underestimates
  the curvature of SSR and the Gauss-Newton map can have an eigenvalue near -1, so the MA
  coefficient alternates around the optimum while every step still lowers SSR. The earlier
  rule (accept any step that does not raise SSR, then divide lambda by 10) never damped that
  2-cycle: ARIMA(1,1,1) with drift on a seasonal series (`arima_misspecified_seasonal`,
  eigenvalue -1.01 at n = 100 and -0.997 at n = 1000) stopped at the 200-iteration cap with
  coefficients up to 8e-3 (relative) from the optimum. A step that achieves little of its
  predicted decrease (rho near 0) now raises lambda, which damps the oscillation: the fit
  converges in 33 (n = 100) and 32 (n = 1000) iterations, within 1.7e-7 (relative) of the
  optimum. The 12 fits of `arima_css_forecast` that estimate coefficients take 4 to 26
  iterations (3 to 28 under the earlier rule).
- Precision: the stopping rule bounds the last scaled Gauss-Newton step, not the distance to
  the optimum. Where the Gauss-Newton model is adequate the two are of the same order; the
  R-pinned fits below agree with the exact optimum to `Tol::iterative(1e-6)` or better.
- A coefficient whose derivative is identically zero (for example an AR term on a series
  that is constant after differencing) is an error naming it ("not identified").

## Options

| Option | Default | Rule |
|---|---|---|
| `include_mean` | true when d = 0 (R) | `true` with d > 0 is an error: after differencing a mean has no effect (R ignores it) |
| `include_drift` | false | d = 0 (mean plus a linear trend) or d = 1; d >= 2 is an error (forecast::Arima drops the drift with a warning) |

## Reported quantities

| Quantity | Formula |
|---|---|
| `coefficients`, `coef_names` | the optimum, R order and names; `ar_coeffs`, `ma_coeffs`, `intercept` (mean, 0 when none), `drift` sliced from them |
| `constant` | mean x (1 - sum(ar)) (d = 0), drift x (1 - sum(ar)) (d = 1 with drift), 0 without either, none for d = 0 with drift |
| `residuals` | n values, the first `n_cond = d + p` are 0 (R's `residuals()`) |
| `sigma2` | SSR / (n - d - k), k the number of coefficients (forecast::Arima) |
| `loglik` | -0.5 [(n - d) ln(SSR / (n - d - p)) + (n - d)(1 + ln 2 pi)] (stats::arima, CSS) |
| `aic` | -2 loglik + 2 (k + 1), labelled "CSS likelihood": R reports NA for CSS fits |
| `converged`, `iterations`, `warnings` | non-convergence, "MA part of model is not invertible", "AR part is not stationary" |

## Forecasts

As `stats::predict.Arima` computes them for a CSS fit: the state-space form of
`stats::makeARIMA` (r = max(p, q + 1) ARMA states with the stationary covariance as
prior, obtained by doubling; d differencing states with prior variance 1e6), a Kalman filter
over `x_t - mu - delta t`, then `KalmanForecast`. The point forecast adds
`mu + delta (n + h)`; the standard error is `sqrt(sigma2 Z'P Z)` with forecast::Arima's
`sigma2`; the 80% and 95% limits are `mean -/+ qnorm(0.9 | 0.975) se`, as
`forecast::forecast`.

Why not the CSS recursion with future errors 0: it conditions on zero pre-sample
innovations, R on the Kalman state, and the two differ whenever |theta|^n is not
negligible. At the same coefficients the recursion differs from `predict.Arima` by
1.0e-4 relative (0.011 absolute) for ARIMA(1,2,1) with theta -0.98, by 2.0e-5 for
ARMA(1,1) with theta -0.8 and n = 40 (h = 1: 18.6047685874 against 18.6044047523), and by
3.3e-5 for ARIMA(0,1,1) with theta -0.9 and n = 60. The state-space forecaster matched
`predict.Arima` to 7.9e-15 (means) and 2.0e-15 (standard errors) on 15 models in the
planning check.

## Reference cases

- `arima_css_forecast`: 13 fits on the generated series (a: AR(1) around 100; c: random
  walk; m: ARMA(1,1) around 20; b: AR(2); dr: drift plus MA(1) in differences; ai: `a` x 10
  rounded, an integer column). forecast::Arima's own CSS optimum comes from optim's BFGS with
  numerical gradients and sits up to 2.3e-4 (relative) from the exact optimum (dr_121, where
  its gradient is 0.43). The case therefore re-solves every fit by Gauss-Newton on
  `stats::arima(fixed = ...)` residuals with `numDeriv::jacobian` (keys `*_tight`), and
  computes forecasts, standard errors and limits at that optimum with `forecast::forecast`.
  The package values (untagged keys) are cross-checks.
- `arima_fixed_forecast`: `predict(arima(x, order, fixed = ..., method = "CSS"))` for five
  fixed coefficient vectors, the forecaster alone. `m40_101` (40 observations, MA -0.8) is
  the case the CSS recursion fails.
- `arima_misspecified_seasonal`: ARIMA(1,1,1) with drift,
  `forecast::Arima(order = c(1, 1, 1), include.drift = TRUE, method = "CSS")`, on a seasonal
  series it misspecifies: `validation/datasets/generated/arima_misspecified_seasonal.csv`,
  y = 0.01 t + 2 sin(pi t / 6) + U(0, 0.5), t = 0..999, seed 42 (the unified benchmark's
  time-series series), fitted at n = 1000 and on its first 100 values. The residuals stay
  autocorrelated (0.34 at lag 2, 0.41 / 0.51 at lag 12) and the Gauss-Newton map has an
  eigenvalue of -1.0099 (n = 100) and -0.9973 (n = 1000) at the optimum; the case asserts
  it is below -0.99. Gauss-Newton (as for `*_tight` above) does not contract there, and
  numDeriv's gradient in the drift is off by 2.5e-8 at n = 100, so the exact optimum comes from
  Newton's method on the analytic gradient of the CSS residuals (their derivatives follow the
  residual recursion), each evaluation checked against `stats::arima(fixed = ...)`'s
  residuals to 1e-12; the Hessian is `numDeriv::jacobian` of that gradient (max |gradient| at
  the solution 1.9e-15 and 3.6e-14). The package values (optim's default `reltol`, the call
  the benchmark makes) are cross-checks: their drift is 2.8e-5 (n = 100) and 2.4e-6
  (n = 1000) relative from the optimum.

### Values (exact CSS optimum)

| Fit | Coefficients |
|---|---|
| a_100 = ARIMA(1,0,0) on a | ar1 0.5717663508709422, intercept 100.06020605096334 |
| a_101 = ARIMA(1,0,1) on a | ar1 0.652639363096857, ma1 -0.12251527325936847, intercept 100.06142321306237 |
| c_011 = ARIMA(0,1,1) on c | ma1 -0.04896666928757075 |
| dr_011_drift = ARIMA(0,1,1) with drift on dr | ma1 -0.3452781569914636, drift 0.3393262262518742 |
| dr_121 = ARIMA(1,2,1) on dr | ar1 -0.3413696463168867, ma1 -0.983643280892541 (forecast::Arima: -0.34129109649782435, -0.9836056642026672) |

Forecasts at h = 1:

| Fit | Mean | SE | Lo 95 | Hi 95 |
|---|---|---|---|---|
| a_100 | 99.33962576264658 | 0.9289959465451472 | 97.51882716563439 | 101.16042435965876 |
| a_101 | 99.28833037497282 | 0.9288173913533515 | 97.4678817397058 | 101.10877901023983 |
| m_002 | 19.610367125066006 | 1.0125056527549094 | 17.625892511523165 | 21.594841738608846 |
| c_011 | 46.80509914469101 | 0.9841503411312719 | 44.87619992070091 | 48.733998368681114 |
| c_010 | 46.7367599 (the last value) | 0.9838218227523156 | 44.808504560200916 | 48.66501523979909 |
| dr_011_drift | 108.6716568738075 | 0.9592587040633475 | 106.79154436198678 | 110.55176938562823 |
| dr_121 | 108.46864704795638 | 0.9813802606630233 | 106.54517708191833 | 110.39211701399444 |
| dr_100_drift | 108.96909398222695 | 0.9750281000806958 | 107.05807402215427 | 110.88011394229963 |
| m_100_nomean | 18.808518444986415 | 1.0608668826049783 | 16.729257562689376 | 20.887779327283454 |
| ai_101 | 992.8988709944479 | 9.27657796206508 | 974.7171122890223 | 1011.0806296998734 |

The full vectors (h = 1..5, 80% limits, residual heads and tails, SSR, sigma2,
log-likelihood, AIC) are in `validation/expected/arima_css_forecast.json`.

### Values (misspecified seasonal series, exact CSS optimum)

| Fit | ar1 | ma1 | drift |
|---|---|---|---|
| n = 100 | 0.69020288306856947 | 0.16463016415297885 | -0.0040354185164835224 |
| n = 1000 | 0.70228272647770884 | 0.19138619050272601 | 0.0074334979545297242 |
| n = 100, forecast::Arima | 0.69020281862898891 | 0.16463018844377031 | -0.0040353049113718798 |
| n = 1000, forecast::Arima | 0.70228210289857906 | 0.19138623804441671 | 0.0074335159952327885 |

SSR 22.847783951680395 (n = 100) and 222.16078373044002 (n = 1000); the sigma2,
log-likelihood and AIC at the optimum are in
`validation/expected/arima_misspecified_seasonal.json`.

## Tests and tolerances

| Test | Pins | Tolerance |
|---|---|---|
| `test_validate_arima_css_fit` | 13 fits: coefficients (tight and package), SSR, sigma2, log-likelihood, AIC, n_cond, number of coefficients, residual head and tail | coefficients `Tol::iterative(1e-6)` against the exact optimum (both sides solved to 1e-8 scaled steps), `Tol::iterative(1e-3)` against forecast::Arima's optimum; SSR, sigma2, loglik, AIC `Tol::stat(300)` = 1e-6; residuals 1e-6 relative + 1e-7 absolute |
| `test_validate_arima_mean_parameterisation` | `run_arima`: the intercept is the mean, d > 0 fits no drift, the AIC formula | as above |
| `test_validate_arima_misspecified_seasonal` | ARIMA(1,1,1) with drift on the misspecified seasonal series, n = 100 and 1000: `converged`, no warning, coefficients (tight and package), SSR, sigma2, log-likelihood, AIC | coefficients `Tol::iterative(1e-6)` against the exact optimum, `Tol::iterative(1e-3)` against forecast::Arima's; SSR, sigma2, loglik, AIC `Tol::stat(n)` |
| `test_validate_arima_css_forecast` | 13 fits: forecasts, SE, 80/95% limits (h = 1..5) | `Tol::stat(300)`; forecast::Arima's means at `Tol::iterative(1e-5)` |
| `test_validate_arima_forecast_default_spec` | `forecast_arima` (R's defaults) on a_100, c_011, dr_121 | `Tol::stat(300)` |
| `test_validate_arima_fixed_forecast` | `predict.Arima` for fixed coefficients (n = 300 and 40), CSS residual tails | `Tol::coef(n)`, `Tol::se(n)` |
| `test_validate_mcp_ts_arima_forecast_intervals`, `test_validate_mcp_ts_arima_fit_drift` | MCP text output | print precision (`Tol::printed(4)`, `Tol::printed(6)`) |
| `test_validate_cli_ts_arima_json`, `test_validate_cli_ts_arima_text_drift` | CLI JSON (full tolerance) and text (print precision) | as above |

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- forecasting::arima_model
```

## Input data

- Integer and Boolean columns are accepted (`ai` is Int64: `ai_101` is pinned to R).
- A missing value (empty cell, NA or NaN) is an error naming the column and the count
  (`EconError::NullValues`). R's default CSS-ML switches to the exact likelihood, whose
  Kalman filter skips NA; CSS cannot skip a gap, and p2a implements neither. The MCP and CLI
  messages add how to fill the gap.
- Text columns are refused (`NonNumericColumn`), an absent column is `ColumnNotFound`.

Tests: `test_na_arima_int64_and_missing` (and, for the other time-series entry points,
`test_na_holt_winters_int64_and_missing`, `test_na_stl_int64_and_missing`,
`test_na_decompose_int64_and_missing`, `test_na_mstl_int64_and_missing`,
`test_na_struct_ts_int64_and_missing`, `test_na_causal_impact_int64_and_missing`,
`test_na_changepoint_int64_and_missing`, `test_na_granger_var_missing_and_text`).

## Known differences from R

1. CSS only: no CSS-ML or ML estimation.
2. No coefficient standard errors. R's `var.coef` for CSS comes from optim's numerical
   Hessian.
3. A non-stationary AR part (CSS does not constrain it) starts the Kalman state of the ARMA
   block from a diffuse prior (variance 1e6), with a warning. R uses Gardner's Q0, which is
   undefined there; pure AR forecasts coincide. At an exact unit root R returns NaN.
4. No seasonal terms.
5. `include_mean = true` with d > 0 and a drift with d >= 2 are errors; R ignores them, the
   latter with a warning.

## References

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C. & Ljung, G.M. (2015). *Time Series Analysis:
  Forecasting and Control* (5th ed.). Wiley.
- Gardner, G., Harvey, A.C. & Phillips, G.D.A. (1980). Algorithm AS 154. *Applied
  Statistics*, 29(3), 311-322.
- Hyndman, R.J. & Khandakar, Y. (2008). "Automatic Time Series Forecasting: The forecast
  Package for R". *Journal of Statistical Software*, 27(3).
- Nielsen, H.B. (1999). *Damping Parameter in Marquardt's Method*. Report IMM-REP-1999-05,
  Technical University of Denmark.
