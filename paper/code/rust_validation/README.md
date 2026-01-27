# Rust vs R Validation and Benchmarking Framework

Systematic validation of p2a-core Rust implementations against R reference packages, plus performance benchmarking.

## Quick Start

```bash
cd paper/code/rust_validation

# Run everything (validation + benchmarks)
./scripts/run_all.sh

# Or run individually:
./scripts/run_validation.sh    # Accuracy validation
./scripts/run_benchmark.sh     # Performance benchmarks
./scripts/generate_report.sh   # Generate summary reports
```

## Prerequisites

### R Packages
```r
install.packages(c("optparse", "jsonlite", "bench", "sandwich", "lmtest",
                   "plm", "fixest", "AER", "forecast", "MASS"))
```

### System Tools
- `p2a` CLI: `cargo build --release -p p2a-cli`
- `jq`: `apt install jq`
- `hyperfine` (optional, for better benchmarks): `cargo install hyperfine`
- Python 3 with standard library

## Directory Structure

```
rust_validation/
├── config/
│   ├── methods.json       # Method registry (R ↔ p2a mapping)
│   └── tolerances.json    # Numerical tolerances by method
├── datasets/
│   ├── longley.csv        # Classic econometric dataset
│   ├── grunfeld.csv       # Panel data
│   └── generate_synthetic.R  # Generates benchmark datasets
├── r_scripts/
│   ├── run_method.R       # Generic R method runner
│   ├── benchmark_method.R # R benchmark wrapper
│   └── methods/           # Method-specific implementations
├── scripts/
│   ├── run_validation.sh  # Main validation runner
│   ├── run_benchmark.sh   # Main benchmark runner
│   ├── compare_results.sh # JSON comparison with tolerance
│   └── generate_report.sh # Generate summary reports
├── results/
│   ├── validation/        # R vs Rust comparison results
│   ├── benchmarks/        # Timing results
│   └── summaries/         # Aggregated reports
└── figures/               # Generated plots
```

## Methods Covered

| Method | p2a Command | R Reference |
|--------|-------------|-------------|
| OLS | `regression ols` | `stats::lm` |
| OLS + HC1 | `regression ols --robust hc1` | `sandwich::vcovHC` |
| Clustered SE | `regression clustered` | `sandwich::vcovCL` |
| Panel FE | `panel fe` | `plm::plm(model="within")` |
| Panel RE | `panel re` | `plm::plm(model="random")` |
| HDFE | `panel hdfe` | `fixest::feols` |
| Logit | `discrete logit` | `stats::glm(binomial)` |
| Probit | `discrete probit` | `stats::glm(binomial(probit))` |
| 2SLS | `causal iv` | `AER::ivreg` |
| ARIMA | `timeseries arima` | `forecast::Arima` |
| K-means | `ml kmeans` | `stats::kmeans` |
| PCA | `ml pca` | `stats::prcomp` |

## Validation

Validates numerical accuracy by comparing results with configurable tolerances:

```bash
# Run all validations
./scripts/run_validation.sh

# Run specific method
./scripts/run_validation.sh ols

# Manual comparison
./scripts/compare_results.sh results/validation/r_ols_longley.json \
                             results/validation/rust_ols_longley.json
```

### Tolerances

Default tolerance: `1e-6` for coefficients/standard errors.

Method-specific overrides in `config/tolerances.json`:
- MLE methods (logit, probit): `1e-4`
- ARIMA: `1e-3`
- K-means: `1e-3` (initialization-dependent)

## Benchmarking

Measures performance at multiple sample sizes (n=100, 1000, 10000, 100000):

```bash
# Run all benchmarks
./scripts/run_benchmark.sh

# Run specific method
./scripts/run_benchmark.sh ols

# Run specific method at specific size
./scripts/run_benchmark.sh ols 10000
```

### Configuration

Environment variables:
- `ITERATIONS`: Number of iterations (default: 100)
- `WARMUP`: Warmup iterations (default: 5)

## Output Formats

### Validation JSON
```json
{
  "method": "ols",
  "dataset": "longley.csv",
  "n": 16,
  "results": {
    "coefficients": {"(Intercept)": -3482.26, "GNP": 15.06, ...},
    "std_errors": {"(Intercept)": 890.42, "GNP": 84.91, ...},
    "r_squared": 0.9954
  }
}
```

### Benchmark Summary
```json
{
  "methods": {
    "ols": [
      {"n": 100, "r_median_us": 1200, "rust_median_us": 450, "speedup": 2.7},
      {"n": 1000, "r_median_us": 2100, "rust_median_us": 620, "speedup": 3.4}
    ]
  },
  "overall_stats": {
    "mean_speedup": 3.2,
    "min_speedup": 1.8,
    "max_speedup": 5.1
  }
}
```

## Adding New Methods

1. Create R implementation in `r_scripts/methods/method_name.R`:
   ```r
   run_method <- function(data, dep_var, indep_vars, ...) {
     # Implementation
     list(coefficients = ..., std_errors = ..., ...)
   }
   ```

2. Add method mapping to `config/methods.json`

3. Add tolerance override to `config/tolerances.json` if needed

4. Add validation/benchmark calls to shell scripts

## Integration with Paper

Results are used to generate:
- Table 1: Performance comparison (from `benchmark_summary.json`)
- Validation methodology section (from validation results)

```bash
# After running benchmarks, copy summary for paper
cp results/summaries/benchmark_summary.json ../llm_eval/
```
