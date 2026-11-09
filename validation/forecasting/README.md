# Forecasting Validation

This directory contains validation documentation for forecasting methods.

## Methods

| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| ARIMA | [arima.md](arima.md) | `run_arima()`, `forecast_arima()` | R `forecast::Arima()` |
| MSTL | [mstl.md](mstl.md) | `run_mstl()` | R `forecast::mstl()` |
| GARCH(p, q) | [garch.md](garch.md) | `garch()`, `run_garch()` | conditional likelihood written out in R (fGarch and rugarch not installed) |
| Holt-Winters | [holtwinters.md](holtwinters.md) | `run_holt_winters()` | R `stats::HoltWinters()` |
| Kalman filter, smoother and forecast | [kalman.md](kalman.md) | `kalman_filter()` | R `stats::KalmanRun()`, `KalmanSmooth()`, `KalmanForecast()` |
| Structural time series | [structts.md](structts.md) | `run_struct_ts()` | R `stats::StructTS()` |
| CausalImpact | [causal_impact.md](causal_impact.md) | `run_causal_impact()` | R `stats::StructTS(type = "level")` and `KalmanForecast()` |
| Changepoint | [../timeseries/changepoint.md](../timeseries/changepoint.md) | `run_changepoint()` | R `changepoint::cpt.mean()`, `cpt.var()`, `cpt.meanvar()` |

Whether each is pinned to R is in the method index of [../README.md](../README.md).

## Key Test Cases

- **AirPassengers**: Classic seasonal time series (n=144)
- **Synthetic changepoints**: Known change locations
- **Multiple seasonality**: For MSTL validation

## Running Tests

```bash
validation/run_validation.sh --rust-only --category forecasting
```
