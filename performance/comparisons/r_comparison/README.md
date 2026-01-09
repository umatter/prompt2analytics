# R Comparison Benchmarks

This directory contains R benchmark scripts for cross-language performance comparison with p2a Rust implementations.

## Prerequisites

### Required Packages

```r
install.packages(c("microbenchmark", "sandwich", "plm", "lfe"))
```

### Optional Packages (for full benchmark suite)

```r
install.packages(c("forecast", "dbscan", "changepoint"))
```

## Running Benchmarks

### Run All Benchmarks

```bash
cd performance/comparisons/r_comparison
Rscript run_all_benchmarks.R
```

### Run Individual Categories

```bash
# Regression (OLS, Robust SEs)
Rscript benchmark_regression.R

# Econometrics (Panel, HDFE, Discrete Choice)
Rscript benchmark_econometrics.R

# Machine Learning (K-means, PCA, Clustering)
Rscript benchmark_ml.R

# Forecasting (ARIMA, MSTL, Changepoint)
Rscript benchmark_forecasting.R
```

## Output

Results are saved to the `results/` directory:

| File | Description |
|------|-------------|
| `regression_ols.csv` | OLS timing results |
| `regression_robust_se.csv` | HC0-HC3 timing results |
| `econometrics_fe.csv` | Fixed Effects timing |
| `econometrics_hdfe.csv` | HDFE timing |
| `econometrics_logit.csv` | Logit timing |
| `econometrics_probit.csv` | Probit timing |
| `ml_kmeans.csv` | K-means timing |
| `ml_pca.csv` | PCA timing |
| `forecasting_arima.csv` | ARIMA timing |
| `combined_results.csv` | All results merged |

## Data Generation

All benchmark scripts use the **same data generation patterns** as the Rust benchmarks to ensure fair comparison:

- Same random seeds (`set.seed(42)`)
- Same sample sizes
- Same data generating processes (DGP)

## Benchmark Methodology

- **Iterations**: 20-100 per method (fewer for slow methods)
- **Unit**: Microseconds
- **Statistics**: Mean, median, min, max
- **Warmup**: Handled by microbenchmark package

## Comparing with Rust Results

After running both R and Rust benchmarks:

```bash
# Run Rust benchmarks
cargo bench -p p2a-core

# Run R benchmarks
cd performance/comparisons/r_comparison
Rscript run_all_benchmarks.R
```

Results can be compared using the summary reports in `performance/reports/`.

## Reference Packages

| Method | R Package | Function |
|--------|-----------|----------|
| OLS | stats | `lm()` |
| Robust SE | sandwich | `vcovHC()` |
| Fixed Effects | plm | `plm(..., model="within")` |
| HDFE | lfe | `felm()` |
| Logit/Probit | stats | `glm(..., family=binomial)` |
| K-means | stats | `kmeans()` |
| Hierarchical | stats | `hclust()` |
| PCA | stats | `prcomp()` |
| ARIMA | forecast | `Arima()` |
| MSTL | forecast | `mstl()` |
| DBSCAN | dbscan | `dbscan()` |
| Changepoint | changepoint | `cpt.mean()` |
