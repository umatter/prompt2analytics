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

## Tests and tolerances

`test_validate_granger_int_series` (`crates/p2a-core/src/econometrics/timeseries/granger.rs`)
pins F at `Tol::stat(300)` (1e-6 relative), the p-value at `Tol::p_value(300)`, and the
degrees of freedom and the rows used exactly, for both lag orders.

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::timeseries::granger::tests::test_validate_granger
```

## Input data and missing values

Both series are read through the shared extractor with `NaPolicy::Fail`: integer and
Boolean columns are accepted, text is `NonNumericColumn`, and a missing value (empty cell,
NA or NaN) in either series is `EconError::NullValues` naming the column and the count.

R's `grangertest` drops rows instead, but its two interfaces disagree on which: on a
120-row series with three NA cells, the formula method (`grangertest(y ~ x, data)`) lets
`model.frame` drop the NA rows and then lags the spliced series (115 rows, F 4.99915905),
while the default method (`grangertest(x, y)`) lags first and drops every lagged row that
touches an NA (111 rows, F 4.35551641). There is no single R behaviour to copy, so p2a
refuses the gap and the MCP and CLI messages say how to fill it. The VAR family (`run_var`,
`run_varma`, `run_vecm`, `run_var_irf`) follows the same rule, as `vars::VAR` refuses NA.
Test: `test_na_granger_var_missing_and_text`.
