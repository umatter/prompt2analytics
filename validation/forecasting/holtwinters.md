# Holt-Winters exponential smoothing

`holt_winters`, `run_holt_winters`, `holt_winters_forecast` and `holt_winters_predict` in
`crates/p2a-core/src/forecasting/holtwinters.rs`. R reference case: `ts_holtwinters_forecast`
(`validation/r/cases/ts_holtwinters.R`), `validation/datasets/generated/holtwinters_series.csv`.

## Model starts

As `stats::HoltWinters`, with t 1-based:

| Model | Filter starts at | l.start | b.start |
|---|---|---|---|
| Holt (trend, no season) | t = 3 | x[2] | x[2] - x[1] |
| Simple exponential smoothing | t = 2 | x[1] | none |
| Seasonal | t = period + 1 | intercept of the regression of `decompose`'s trend on time, with or without a trend | its slope (0 without a trend) |

`level`, `fitted`, `residuals` and `trend` are NaN before the start.

## Parameter rules

`InvalidSpecification`, naming the parameter, for:

- `period` 0, and below 2 for a seasonal model;
- `alpha` outside (0, 1] (R: "cannot fit models without level");
- `beta` and `gamma` outside [0, 1], when used (R: "must be within the unit interval");
- `start_periods` below 2 for a seasonal model (R: "need at least 2 periods to compute seasonal start values");
- `max_iter` 0, and a `tolerance` that is not finite and positive, whether or not an optimiser runs;
- a seasonal size that overflows.

In `run_holt_winters`, `beta = 0` and `gamma = 0` turn the component off, which R's `FALSE` does; an out-of-range value is
refused. Through `holt_winters`, `Some(0.0)` with `use_trend` or `use_seasonal` is R's 0: a component that is not updated.

## Prediction intervals

`holt_winters_predict(result, horizon, level)` gives forecasts and limits. For additive and non-seasonal fits this is
`predict.HoltWinters(prediction.interval = TRUE)` (Yar and Chatfield): the variance at horizon h is
var(residuals) (1 + sum_{j=1}^{h-1} psi_j^2), with psi_j = alpha (1 + j beta) + 1{j mod f = 0} gamma (1 - alpha), and the limits
are forecast -/+ qnorm((1 + level) / 2) sqrt(variance). A fit without a seasonal component uses this formula whatever its label.

**Multiplicative seasonal intervals are refused, and are not R-pinned.** `predict.HoltWinters`' multiplicative variance reads
the trend b where s_f belongs when (rel - j) mod f = 0, uses alpha + gamma (1 - alpha) for the one-step term, where the term is
1, and shifts the seasonal indices without a trend. On `airpass_mult`, its h = 1 interval implies a standard error of 7.61
against a residual SD of 13.12, narrower than the one-step error. The point forecasts of those fits are pinned. Deviation TS-22
records `airpass_mult_upr` and `airpass_mult_lwr`, which the test does not compare.

## Pins

| Test | R fit | Pinned |
|---|---|---|
| `test_validate_holtwinters_forecast_additive` | co2 to 1969, alpha 0.5, beta 0.01, gamma 0.3, additive | SSE, coefficients, error count, residual variance, fit and 95 % limits to h = 12; 80 % upper limit to h = 24 |
| `test_validate_holtwinters_forecast_multiplicative` | AirPassengers, alpha 0.3, beta 0.05, gamma 0.4, multiplicative | SSE, coefficients, error count, residual variance, fit; refusal of intervals |
| `test_validate_holtwinters_forecast_holt` | AirPassengers, alpha 0.6, beta 0.1, gamma = FALSE | as additive (142 one-step errors) |
| `test_validate_holtwinters_notrend_and_ses` | beta = FALSE, gamma 0.2; alpha 0.4 with beta = gamma = FALSE | as additive (132 and 143 errors) |
| `test_validate_holtwinters_at_r_optimum` | the filter at R's optimised parameters: AirPassengers multiplicative and additive, the legacy quarterly and non-seasonal series | SSE, coefficients; fit for `opt_mult` |

Coefficients and forecasts are compared at `Tol::coef(n)`, SSE, limits and variance at `Tol::stat(n)`, counts exactly.
`test_holtwinters_rejects_period_0` and `test_holtwinters_rejects_out_of_range_parameters` hold the refusals and compare no R
number.

The fits at R's optimum supersede `validation/expected/holtwinters_*.csv`: R 4.3.3 reproduces those seven files' values to
their 15 digits, with the legacy series regenerated from the legacy seeds.

**Deviation TS-24.** The optimiser is not pinned. p2a runs Nelder-Mead on [0.01, 0.99], R L-BFGS-B from its own starts. The
filter is pinned at R's optimum.

## Input data (column types and missing values)

`run_holt_winters` reads its columns through the shared extractor with `NaPolicy::Fail`. Integer and Boolean columns are accepted (an `Int64` column gives the result of its `Float64` copy); a missing value (empty cell, NA or NaN) is an error naming the column and the count (`EconError::NullValues`); a text column is refused (`NonNumericColumn`) and an absent one is `ColumnNotFound`.

R's `HoltWinters` errors on a missing value ("NA/NaN/Inf in foreign function call"), so p2a refusing it matches R.

Test: `test_na_holt_winters_int64_and_missing` (on `validation/datasets/generated/arima_series.csv`, the dataset of the ARIMA case `arima_css_forecast`: columns `ai` (Int64), `a_na` and `ai_na` (two missing cells), `lab` (text)).

## References

- Holt, C. C. (1957). "Forecasting Trends and Seasonal by Exponentially Weighted Averages". ONR Memorandum 52/1957, Carnegie Institute of Technology.
- Holt, C. C. (2004). "Forecasting seasonals and trends by exponentially weighted moving averages". International Journal of Forecasting, 20(1), 5-10. (Reprint of 1957 paper)
- Winters, P. R. (1960). "Forecasting Sales by Exponentially Weighted Moving Averages". Management Science, 6(3), 324-342.
- Hyndman, R. J. & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice" (3rd ed). OTexts. https://otexts.com/fpp3/
- R Documentation: `stats::HoltWinters()` - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/HoltWinters.html
