# Plan: Rust vs R Validation and Benchmarking Framework

## Overview

A unified, script-based evaluation framework to systematically validate p2a-core Rust implementations against R reference packages and benchmark performance. This framework lives in the paper repository (not p2a-core) and produces publication-ready results with full reproducibility.

## Goals

1. **Numerical Accuracy**: Verify Rust implementations produce identical (within tolerance) results to canonical R implementations
2. **Performance Benchmarking**: Measure execution time for standardized sample sizes (n = 100, 1,000, 10,000, 100,000)
3. **Reproducibility**: All results generated from scripts with fixed seeds, documented dependencies
4. **Publication Quality**: Output suitable for direct inclusion in the paper's comparison section

## Directory Structure

```
paper/code/rust_validation/
├── README.md                      # Documentation and usage
├── PLAN.md                        # This file
│
├── config/
│   ├── methods.json               # Method registry with R/Rust mappings
│   ├── tolerances.json            # Numerical tolerance by method/sample size
│   └── sample_sizes.json          # Standard N values for benchmarks
│
├── datasets/
│   ├── longley.csv                # Classic multicollinearity dataset
│   ├── grunfeld.csv               # Panel data (firm/year)
│   ├── generate_synthetic.R       # Script to generate synthetic data
│   └── synthetic/                 # Generated synthetic datasets
│       ├── regression_n100.csv
│       ├── regression_n1000.csv
│       ├── regression_n10000.csv
│       └── ...
│
├── r_scripts/
│   ├── run_method.R               # Generic R method runner
│   ├── methods/
│   │   ├── ols.R                  # OLS via lm()
│   │   ├── ols_robust.R           # OLS + vcovHC()
│   │   ├── ols_clustered.R        # OLS + vcovCL()
│   │   ├── panel_fe.R             # plm() with model="within"
│   │   ├── panel_re.R             # plm() with model="random"
│   │   ├── hdfe.R                 # lfe::felm() or fixest::feols()
│   │   ├── iv_2sls.R              # AER::ivreg()
│   │   ├── did.R                  # Difference-in-differences
│   │   ├── logit.R                # glm(family=binomial(link="logit"))
│   │   ├── probit.R               # glm(family=binomial(link="probit"))
│   │   ├── arima.R                # forecast::Arima()
│   │   ├── mstl.R                 # forecast::mstl()
│   │   ├── var.R                  # vars::VAR()
│   │   ├── kmeans.R               # stats::kmeans()
│   │   └── pca.R                  # stats::prcomp()
│   └── benchmark_method.R         # Benchmark wrapper using bench package
│
├── rust_runner/
│   ├── Cargo.toml                 # Minimal crate depending on p2a-core
│   ├── src/
│   │   ├── main.rs                # CLI: rust_runner <method> <dataset> [--benchmark]
│   │   └── methods/
│   │       ├── mod.rs
│   │       ├── ols.rs
│   │       ├── panel.rs
│   │       ├── discrete.rs
│   │       ├── timeseries.rs
│   │       └── ml.rs
│   └── build.sh                   # Build release binary
│
├── scripts/
│   ├── run_validation.sh          # Main entry point for accuracy tests
│   ├── run_benchmark.sh           # Main entry point for performance tests
│   ├── run_all.sh                 # Run both validation and benchmark
│   ├── compare_results.sh         # Compare R vs Rust JSON outputs
│   ├── generate_report.sh         # Generate markdown/LaTeX report
│   └── generate_figures.R         # Generate benchmark figures
│
├── results/                       # Generated outputs (gitignored except summaries)
│   ├── validation/
│   │   ├── ols_longley.json
│   │   ├── ols_synthetic_n1000.json
│   │   └── ...
│   ├── benchmarks/
│   │   ├── ols_n100.json
│   │   ├── ols_n1000.json
│   │   └── ...
│   └── summaries/
│       ├── validation_summary.json
│       ├── benchmark_summary.json
│       └── comparison_report.md
│
└── expected/                      # Reference values for regression testing
    ├── ols_longley_expected.json
    ├── panel_grunfeld_expected.json
    └── ...
```

## Method Registry (config/methods.json)

```json
{
  "methods": [
    {
      "id": "ols",
      "name": "Ordinary Least Squares",
      "category": "regression",
      "r_package": "stats",
      "r_function": "lm",
      "rust_module": "regression::ols",
      "outputs": ["coefficients", "std_errors", "t_values", "p_values", "r_squared", "adj_r_squared", "f_statistic"],
      "datasets": ["longley", "synthetic"],
      "sample_sizes": [100, 1000, 10000, 100000]
    },
    {
      "id": "ols_hc1",
      "name": "OLS with HC1 Robust SEs",
      "category": "regression",
      "r_package": "sandwich",
      "r_function": "vcovHC(..., type='HC1')",
      "rust_module": "regression::ols",
      "outputs": ["coefficients", "std_errors", "t_values", "p_values"],
      "datasets": ["synthetic"],
      "sample_sizes": [100, 1000, 10000]
    },
    {
      "id": "panel_fe",
      "name": "Panel Fixed Effects",
      "category": "panel",
      "r_package": "plm",
      "r_function": "plm(..., model='within')",
      "rust_module": "econometrics::panel",
      "outputs": ["coefficients", "std_errors", "r_squared_within"],
      "datasets": ["grunfeld", "synthetic_panel"],
      "sample_sizes": [100, 1000, 5000]
    }
    // ... additional methods
  ]
}
```

## Validation Protocol

### Phase 1: Generate Reference Data

```bash
# Generate synthetic datasets with fixed seed
Rscript datasets/generate_synthetic.R --seed 42 --sizes 100,1000,10000,100000
```

### Phase 2: Run R Reference Implementation

```bash
# Run R method and capture output as JSON
Rscript r_scripts/run_method.R \
  --method ols \
  --dataset datasets/longley.csv \
  --output results/validation/r_ols_longley.json
```

Output format:
```json
{
  "method": "ols",
  "dataset": "longley",
  "n": 16,
  "k": 6,
  "timestamp": "2026-01-24T10:00:00Z",
  "r_version": "4.5.2",
  "packages": {"stats": "4.5.2"},
  "results": {
    "coefficients": {"(Intercept)": -3482258.63, "GNP": 15.0619, ...},
    "std_errors": {"(Intercept)": 890420.38, "GNP": 84.9149, ...},
    "t_values": {...},
    "p_values": {...},
    "r_squared": 0.9954,
    "adj_r_squared": 0.9925,
    "f_statistic": 330.3
  }
}
```

### Phase 3: Run Rust Implementation

```bash
# Run Rust method via CLI runner
./rust_runner/target/release/rust_runner \
  --method ols \
  --dataset datasets/longley.csv \
  --output results/validation/rust_ols_longley.json
```

### Phase 4: Compare Results

```bash
# Compare R and Rust outputs
./scripts/compare_results.sh \
  results/validation/r_ols_longley.json \
  results/validation/rust_ols_longley.json \
  --tolerance 1e-6 \
  --output results/validation/ols_longley_comparison.json
```

Comparison output:
```json
{
  "method": "ols",
  "dataset": "longley",
  "status": "PASS",
  "tolerance": 1e-6,
  "comparisons": [
    {"field": "coefficients.GNP", "r": 15.0619, "rust": 15.0619, "diff": 1.2e-10, "pass": true},
    {"field": "std_errors.GNP", "r": 84.9149, "rust": 84.9148, "diff": 3.4e-5, "pass": true}
  ],
  "max_diff": 3.4e-5,
  "all_passed": true
}
```

## Benchmarking Protocol

### Phase 1: R Benchmarks

```bash
Rscript r_scripts/benchmark_method.R \
  --method ols \
  --dataset datasets/synthetic/regression_n10000.csv \
  --iterations 100 \
  --warmup 10 \
  --output results/benchmarks/r_ols_n10000.json
```

Uses `bench::mark()` for precise timing:
```json
{
  "method": "ols",
  "n": 10000,
  "iterations": 100,
  "timing": {
    "min_us": 2050,
    "q25_us": 2120,
    "median_us": 2181,
    "q75_us": 2290,
    "max_us": 3450,
    "mean_us": 2215,
    "sd_us": 180
  },
  "memory_mb": 12.5
}
```

### Phase 2: Rust Benchmarks

```bash
./rust_runner/target/release/rust_runner \
  --method ols \
  --dataset datasets/synthetic/regression_n10000.csv \
  --benchmark \
  --iterations 100 \
  --warmup 10 \
  --output results/benchmarks/rust_ols_n10000.json
```

Uses `std::time::Instant` for timing (Criterion-style):
```json
{
  "method": "ols",
  "n": 10000,
  "iterations": 100,
  "timing": {
    "min_us": 890,
    "q25_us": 905,
    "median_us": 924,
    "q75_us": 950,
    "max_us": 1120,
    "mean_us": 932,
    "sd_us": 45
  },
  "memory_mb": 8.2
}
```

### Phase 3: Generate Comparison

```bash
./scripts/generate_report.sh \
  --validation-dir results/validation \
  --benchmark-dir results/benchmarks \
  --output results/summaries/
```

## Scripts Detail

### run_validation.sh

```bash
#!/usr/bin/env bash
# Run validation for specified methods or all

METHODS=${1:-"all"}
DATASETS=${2:-"all"}

# Parse config
METHODS_LIST=$(jq -r '.methods[].id' config/methods.json)

for method in $METHODS_LIST; do
  for dataset in $(get_datasets_for_method $method); do
    echo "Validating $method on $dataset..."

    # Run R
    Rscript r_scripts/run_method.R --method $method --dataset $dataset \
      --output results/validation/r_${method}_${dataset}.json

    # Run Rust
    ./rust_runner/target/release/rust_runner --method $method --dataset $dataset \
      --output results/validation/rust_${method}_${dataset}.json

    # Compare
    ./scripts/compare_results.sh \
      results/validation/r_${method}_${dataset}.json \
      results/validation/rust_${method}_${dataset}.json \
      --tolerance $(get_tolerance $method) \
      --output results/validation/${method}_${dataset}_comparison.json
  done
done

# Generate summary
./scripts/summarize_validation.sh results/validation/
```

### run_benchmark.sh

```bash
#!/usr/bin/env bash
# Run benchmarks for specified methods and sample sizes

METHODS=${1:-"all"}
SAMPLE_SIZES="100 1000 10000 100000"
ITERATIONS=100
WARMUP=10

for method in $METHODS_LIST; do
  for n in $SAMPLE_SIZES; do
    dataset="synthetic/regression_n${n}.csv"

    echo "Benchmarking $method at n=$n..."

    # R benchmark
    Rscript r_scripts/benchmark_method.R \
      --method $method --dataset datasets/$dataset \
      --iterations $ITERATIONS --warmup $WARMUP \
      --output results/benchmarks/r_${method}_n${n}.json

    # Rust benchmark
    ./rust_runner/target/release/rust_runner \
      --method $method --dataset datasets/$dataset \
      --benchmark --iterations $ITERATIONS --warmup $WARMUP \
      --output results/benchmarks/rust_${method}_n${n}.json
  done
done

# Generate summary with speedup calculations
./scripts/summarize_benchmarks.sh results/benchmarks/
```

## Output Reports

### validation_summary.json

```json
{
  "timestamp": "2026-01-24T12:00:00Z",
  "total_tests": 45,
  "passed": 45,
  "failed": 0,
  "by_category": {
    "regression": {"total": 12, "passed": 12},
    "panel": {"total": 8, "passed": 8},
    "discrete": {"total": 6, "passed": 6},
    "timeseries": {"total": 10, "passed": 10},
    "ml": {"total": 9, "passed": 9}
  },
  "max_differences": {
    "ols": 1.2e-10,
    "ols_hc1": 3.4e-8,
    "panel_fe": 2.1e-9
  }
}
```

### benchmark_summary.json

```json
{
  "timestamp": "2026-01-24T12:00:00Z",
  "hardware": {
    "cpu": "Intel Core i7-1260P",
    "ram_gb": 64,
    "os": "Linux 6.x"
  },
  "software": {
    "r_version": "4.5.2",
    "rust_version": "1.92.0",
    "p2a_core_version": "0.1.0"
  },
  "results": [
    {
      "method": "ols",
      "n": 10000,
      "r_median_us": 2181,
      "rust_median_us": 924,
      "speedup": 2.4,
      "r_memory_mb": 12.5,
      "rust_memory_mb": 8.2
    }
  ]
}
```

## Methods to Validate

Based on the paper's comparison section and p2a-core implementation:

### Tier 1: Core Methods (Required)
| Method | R Package | Priority |
|--------|-----------|----------|
| OLS | stats::lm | High |
| OLS + HC0-HC3 | sandwich::vcovHC | High |
| OLS + Clustered SE | sandwich::vcovCL | High |
| Panel FE | plm::plm | High |
| Panel RE | plm::plm | High |
| Hausman Test | plm::phtest | High |
| HDFE | fixest::feols / lfe::felm | High |
| 2SLS/IV | AER::ivreg | High |
| Logit | stats::glm | High |
| Probit | stats::glm | High |

### Tier 2: Extended Methods (Important)
| Method | R Package | Priority |
|--------|-----------|----------|
| DiD | Manual / did::att_gt | Medium |
| IPW | causalweight::treatweight | Medium |
| AIPW | causalweight::drlate | Medium |
| ARIMA | forecast::Arima | Medium |
| MSTL | forecast::mstl | Medium |
| VAR | vars::VAR | Medium |
| K-means | stats::kmeans | Medium |
| PCA | stats::prcomp | Medium |

### Tier 3: Additional Methods (Nice to Have)
| Method | R Package | Priority |
|--------|-----------|----------|
| NLS | stats::nls | Low |
| VECM | vars::vec2var | Low |
| t-SNE | Rtsne::Rtsne | Low |
| DBSCAN | dbscan::dbscan | Low |

## Implementation Phases

### Phase 1: Infrastructure (Day 1-2)
- [ ] Create directory structure
- [ ] Set up rust_runner crate with p2a-core dependency
- [ ] Create config files (methods.json, tolerances.json)
- [ ] Write synthetic data generator
- [ ] Implement generic R method runner

### Phase 2: Core Validation (Day 3-5)
- [ ] Implement OLS validation (both R and Rust sides)
- [ ] Implement robust SE validation
- [ ] Implement panel data validation
- [ ] Implement discrete choice validation
- [ ] Create comparison scripts

### Phase 3: Benchmarking (Day 6-7)
- [ ] Implement R benchmark wrapper
- [ ] Implement Rust benchmark mode
- [ ] Create benchmark aggregation scripts
- [ ] Generate benchmark figures

### Phase 4: Extended Methods (Day 8-10)
- [ ] Add remaining Tier 1 methods
- [ ] Add Tier 2 methods
- [ ] Add Tier 3 methods as time permits

### Phase 5: Reporting (Day 11-12)
- [ ] Generate validation summary report
- [ ] Generate benchmark summary report
- [ ] Create LaTeX tables for paper
- [ ] Generate figures (boxplots, speedup charts)

## Usage Examples

```bash
# Full validation and benchmark run
cd paper/code/rust_validation
./scripts/run_all.sh

# Validate only OLS methods
./scripts/run_validation.sh ols

# Benchmark panel methods at n=10000
./scripts/run_benchmark.sh panel 10000

# Generate report from existing results
./scripts/generate_report.sh

# Quick check: validate single method on single dataset
Rscript r_scripts/run_method.R --method ols --dataset datasets/longley.csv
./rust_runner/target/release/rust_runner --method ols --dataset datasets/longley.csv
```

## Dependencies

### R Packages
```r
install.packages(c(
  "bench",           # Benchmarking
  "jsonlite",        # JSON output
  "sandwich",        # Robust SEs
  "plm",             # Panel data
  "lfe",             # HDFE (alternative)
  "fixest",          # HDFE (preferred)
  "AER",             # IV regression
  "forecast",        # Time series
  "vars",            # VAR models
  "causalweight",    # Treatment effects
  "Rtsne",           # t-SNE
  "dbscan"           # DBSCAN clustering
))
```

### Rust
- p2a-core (local path dependency)
- clap (CLI parsing)
- serde_json (JSON output)
- polars (data loading)

## Success Criteria

1. **All Tier 1 methods pass validation** with documented tolerances
2. **Benchmark results reproducible** across runs (< 5% variance)
3. **Paper comparison table** can be generated directly from results
4. **Full run completes** in < 30 minutes on standard hardware
5. **Scripts work** on Linux and macOS

## Notes

- This framework is for the paper, not for p2a-core's own test suite
- Focus on methods actually discussed in the comparison section
- Use the same datasets and sample sizes mentioned in the paper
- Results feed directly into Table 1 (performance) and validation claims
