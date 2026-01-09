# Forecasting Performance Report

## Overview

This report documents the performance of p2a forecasting methods compared to reference implementations.

## Methods Benchmarked

| Method | p2a Function | R Reference | Notes |
|--------|--------------|-------------|-------|
| ARIMA | `run_arima` | `forecast::Arima()` | ARIMA(1,1,1) |
| MSTL | `run_mstl` | `forecast::mstl()` | Multiple seasonal decomposition |
| Changepoint | `run_changepoint` | `changepoint::cpt.mean()` | Binary segmentation |

## Benchmark Configuration

- **Rust**: Criterion with 20 measurement iterations
- **R**: microbenchmark with 20 iterations
- **Time series**: Synthetic with trend and seasonality

## Results Summary

### ARIMA

| n | p2a Rust (μs) | R forecast (μs) | Speedup |
|---|---------------|-----------------|---------|
| 100 | TBD | TBD | TBD |
| 200 | TBD | TBD | TBD |
| 500 | TBD | TBD | TBD |

### MSTL Decomposition

| n | p2a Rust (μs) | R forecast (μs) | Speedup |
|---|---------------|-----------------|---------|
| 100 | TBD | TBD | TBD |
| 200 | TBD | TBD | TBD |
| 500 | TBD | TBD | TBD |

### Changepoint Detection

| n | p2a Rust (μs) | R changepoint (μs) | Speedup |
|---|---------------|-------------------|---------|
| 100 | TBD | TBD | TBD |
| 500 | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD |

## Running Benchmarks

### Rust Benchmarks

```bash
cargo bench -p p2a-core -- forecasting
```

### R Benchmarks

```bash
cd performance/comparisons/r_comparison
Rscript benchmark_forecasting.R
```

## Algorithm Notes

### ARIMA
- Uses augurs crate for ARIMA fitting
- Kalman filter based estimation
- Automatic differencing for stationarity

### MSTL
- Uses augurs crate for MSTL decomposition
- LOESS-based trend extraction
- Multiple seasonal periods supported

### Changepoint Detection
- Binary segmentation algorithm
- Cost functions: MeanChange, VarianceChange, MeanAndVariance
- Penalty-based model selection

## Notes

- Results marked "TBD" will be populated after running the benchmark suite
- Forecasting methods may have significant variance due to optimization convergence
- Fewer iterations (20) due to longer execution times

## Hardware Configuration

See `performance/hardware_profiles.md` for benchmark hardware specifications.
