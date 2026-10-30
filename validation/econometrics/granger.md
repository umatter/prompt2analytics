# Validation: Granger causality F test

Registry: [`validation/methods/forecasting_arima.json`](../methods/forecasting_arima.json)
(method `granger_test`). R case: `ts_granger_int` in
[`validation/r/cases/forecasting_arima.R`](../r/cases/forecasting_arima.R).

## Method

`granger_test(dataset, dependent, cause, lags)` asks whether the lags of `cause` improve
the prediction of `dependent` beyond its own lags. With L lags and the rows that have every
lag available (n - L of them):

- restricted model: `y_t ~ 1 + y_{t-1} + ... + y_{t-L}`;
- unrestricted model: the same plus `x_{t-1} + ... + x_{t-L}`;
- `F = ((RSS_r - RSS_u) / L) / (RSS_u / (n - L - 2L - 1))` with `df1 = L`,
  `df2 = n - L - 2L - 1`, and the upper-tail F p-value.

This is `lmtest::grangertest(x = cause, y = dependent, order = L)` (default method).

## Reference case

`ts_granger_int`: `grangertest(d$m, d$ai, order = 2 | 3)` on
`validation/datasets/generated/arima_series.csv` (n = 300). The dependent series `ai` holds
whole numbers, so polars reads it as `Int64`; the cause `m` is an ARMA(1,1) series.

| Lags | F | p | df1 | df2 | rows used |
|---|---|---|---|---|---|
| 2 | 1.3015337730001175 | 0.27368259598049266 | 2 | 293 | 298 |
| 3 | 2.4420318454610483 | 0.064413303851786688 | 3 | 290 | 297 |
