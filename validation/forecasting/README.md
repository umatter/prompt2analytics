# Forecasting Validation

This directory contains validation documentation for forecasting methods.

## Methods

| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| ARIMA | [arima.md](arima.md) | `run_arima()`, `forecast_arima()` | R `forecast::Arima()` |
| MSTL | [mstl.md](mstl.md) | `run_mstl()` | R `forecast::mstl()` |
| Changepoint | [changepoint.md](changepoint.md) | `run_changepoint()` | R `changepoint`, Python `ruptures` |

## Key Test Cases

- **AirPassengers**: Classic seasonal time series (n=144)
- **Synthetic changepoints**: Known change locations
- **Multiple seasonality**: For MSTL validation

## Running Tests

```bash
cargo test -p p2a-core -- forecasting::tests::test_validate
```
