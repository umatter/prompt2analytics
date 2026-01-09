# Python Comparison Benchmarks

This directory contains Python benchmark scripts for cross-language performance comparison with p2a Rust implementations.

## Prerequisites

```bash
pip install numpy pandas scikit-learn statsmodels linearmodels
```

## Status

Python benchmarks are planned but not yet implemented. Priority is given to R comparisons as R is the primary reference implementation for econometric methods.

## Planned Benchmarks

| Category | Library | Methods |
|----------|---------|---------|
| Regression | statsmodels | OLS, Robust SEs |
| Panel | linearmodels | PanelOLS, RandomEffects |
| IV | linearmodels | IV2SLS |
| Discrete Choice | statsmodels | Logit, Probit |
| ML | scikit-learn | KMeans, PCA, DBSCAN |
| Time Series | statsmodels | VAR, ARIMA |

## Contributing

To add Python benchmarks:

1. Create `benchmark_[category].py`
2. Use `timeit` or `pytest-benchmark` for timing
3. Use same data generation patterns as Rust/R
4. Save results to `results/` directory
