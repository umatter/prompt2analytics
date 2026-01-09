# Regression Performance Report

## Overview

This report documents the performance of p2a regression methods compared to reference implementations.

## Methods Benchmarked

| Method | p2a Function | R Reference | Python Reference |
|--------|--------------|-------------|------------------|
| OLS | `run_ols` | `stats::lm()` | `statsmodels.OLS()` |
| Robust SE (HC0-HC3) | `run_ols` | `sandwich::vcovHC()` | `statsmodels.get_robustcov_results()` |
| Clustered SE | `run_ols_clustered` | `sandwich::vcovCL()` | `linearmodels.fit(cov_type='clustered')` |

## Benchmark Configuration

- **Rust**: Criterion with 100 measurement iterations
- **R**: microbenchmark with 100 iterations
- **Sample sizes**: n ∈ {100, 1000, 10000}
- **Predictors**: k = 5

## Results Summary

### OLS Standard

| n | p2a Rust (μs) | R lm() (μs) | Speedup |
|---|---------------|-------------|---------|
| 100 | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD |
| 10000 | TBD | TBD | TBD |

### Robust Standard Errors (n=1000)

| Type | p2a Rust (μs) | R sandwich (μs) | Speedup |
|------|---------------|-----------------|---------|
| HC0 | TBD | TBD | TBD |
| HC1 | TBD | TBD | TBD |
| HC2 | TBD | TBD | TBD |
| HC3 | TBD | TBD | TBD |

## Running Benchmarks

### Rust Benchmarks

```bash
cargo bench -p p2a-core -- regression
```

### R Benchmarks

```bash
cd performance/comparisons/r_comparison
Rscript benchmark_regression.R
```

## Notes

- Results marked "TBD" will be populated after running the benchmark suite
- Timings are median values to reduce impact of outliers
- Speedup = R time / Rust time

## Hardware Configuration

See `performance/hardware_profiles.md` for benchmark hardware specifications.
