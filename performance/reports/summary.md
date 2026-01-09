# Performance Summary

## Overview

This document summarizes the performance of p2a Rust implementations compared to R reference implementations across all method categories.

## Benchmark Categories

| Category | Methods | Rust Benchmark | R Benchmark |
|----------|---------|----------------|-------------|
| Regression | OLS, Robust SEs, Clustered SEs | `regression_benchmarks.rs` | `benchmark_regression.R` |
| Econometrics | FE, RE, HDFE, IV, Logit, Probit | `econometrics_benchmarks.rs` | `benchmark_econometrics.R` |
| Machine Learning | K-Means, DBSCAN, Hierarchical, PCA | `ml_benchmarks.rs` | `benchmark_ml.R` |
| Forecasting | ARIMA, MSTL, Changepoint | `forecasting_benchmarks.rs` | `benchmark_forecasting.R` |

## Running All Benchmarks

### Rust

```bash
# Run all benchmarks
cargo bench -p p2a-core

# Run specific category
cargo bench -p p2a-core -- regression
cargo bench -p p2a-core -- econometrics
cargo bench -p p2a-core -- ml
cargo bench -p p2a-core -- forecasting
```

### R

```bash
cd performance/comparisons/r_comparison
Rscript run_all_benchmarks.R
```

## Performance Highlights

*To be populated after running benchmarks*

### Expected Advantages

1. **Compiled Language**: Rust's zero-cost abstractions provide native performance
2. **Memory Efficiency**: No garbage collection pauses
3. **Parallel Processing**: Rayon-based parallelism for large datasets
4. **SIMD**: faer library uses SIMD for linear algebra operations

### Expected Comparable Performance

1. **ARIMA**: Both use similar Kalman filter implementations
2. **MLE Methods**: Similar Newton-Raphson convergence behavior

## Detailed Reports

- [Regression Performance](regression_performance.md)
- [Econometrics Performance](econometrics_performance.md)
- [ML Performance](ml_performance.md)
- [Forecasting Performance](forecasting_performance.md)

## Methodology

### Benchmark Protocol

1. **Warmup**: Both Criterion and microbenchmark handle warmup automatically
2. **Iterations**: 20-100 depending on method complexity
3. **Statistics**: Median used for comparison (robust to outliers)
4. **Same Data**: Identical data generation with same seeds

### Comparison Metrics

- **Speedup**: R median time / Rust median time
- **Absolute Time**: Median execution time in microseconds
- **Scaling**: Performance ratio at different sample sizes

## Hardware

Benchmarks should be run on the same hardware for fair comparison. See `hardware_profiles.md` for recommended specifications.

## Reproducibility

All benchmarks use:
- Fixed random seeds (42)
- Identical data generating processes
- Same sample sizes

## Updates

This report will be updated when:
1. New benchmarks are run
2. Implementations are optimized
3. New methods are added

---

*Last updated: 2026-01-09*
