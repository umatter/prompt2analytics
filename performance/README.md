# Performance Benchmarking Framework

This directory contains performance benchmarks and cross-language comparisons for prompt2analytics.

## Purpose

Performance benchmarking serves multiple goals:
1. **Establish baselines**: Know how fast our implementations are
2. **Track regressions**: Detect performance degradation over time
3. **Compare with alternatives**: Benchmark against R, Python, Julia
4. **Guide optimization**: Identify bottlenecks for improvement

## Structure

```
performance/
├── README.md                 # This file
├── methodology.md            # Benchmarking methodology
├── hardware_profiles.md      # Test hardware documentation
│
├── benchmarks/               # Rust Criterion benchmarks
│   ├── regression_benchmarks.rs
│   ├── econometrics_benchmarks.rs
│   ├── ml_benchmarks.rs
│   └── forecasting_benchmarks.rs
│
├── results/                  # Raw benchmark data
│   └── YYYY-MM-DD/           # Date-stamped results
│       ├── hardware_info.json
│       └── *.csv
│
├── comparisons/              # Cross-language comparisons
│   ├── r_comparison/
│   │   └── benchmark_runner.R
│   ├── python_comparison/
│   │   └── benchmark_runner.py
│   └── combined_results.csv
│
└── reports/                  # Summary reports
    ├── summary.md
    ├── gpu_performance.md
    ├── regression_performance.md
    ├── econometrics_performance.md
    ├── ml_performance.md
    └── forecasting_performance.md
```

## Running Benchmarks

### Rust (Criterion)

```bash
# Run all benchmarks
cargo bench -p p2a-core

# Run specific benchmark group
cargo bench -p p2a-core -- regression
cargo bench -p p2a-core -- econometrics
cargo bench -p p2a-core -- ml

# Run specific benchmark
cargo bench -p p2a-core -- ols_standard

# Generate HTML report
# (automatically generated in target/criterion/)
open target/criterion/report/index.html
```

### GPU vs CPU

```bash
# GPU-enabled (requires --features cuda and NVIDIA GPU)
cargo bench -p p2a-core --bench gpu_benchmarks --features cuda

# CPU-only baseline (force all thresholds to infinity)
P2A_GPU_XTX_MIN_NKK=999999999999 P2A_GPU_XTX_MIN_K=999999 \
  cargo bench -p p2a-core --bench gpu_benchmarks --features cuda
```

Results and analysis: `reports/gpu_performance.md`

### R Comparisons

```bash
cd performance/comparisons/r_comparison
Rscript benchmark_runner.R
```

### Python Comparisons

```bash
cd performance/comparisons/python_comparison
python benchmark_runner.py
```

## Benchmark Output Format

All benchmark results are saved in CSV format:

```csv
method,variant,n,k,mean_us,median_us,std_us,ci_lower,ci_upper,memory_kb,timestamp
run_ols,standard,1000,5,245.3,242.1,12.5,240.1,250.5,1024,2026-01-09T10:30:00Z
run_ols,HC1,1000,5,312.7,308.2,15.2,305.1,320.3,1024,2026-01-09T10:30:05Z
```

## Key Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| mean_us | Mean execution time | microseconds |
| median_us | Median execution time | microseconds |
| std_us | Standard deviation | microseconds |
| ci_lower/ci_upper | 95% confidence interval | microseconds |
| memory_kb | Peak memory usage | kilobytes |

## Scaling Tests

Each method is tested at multiple sample sizes to understand scaling behavior:

| n | Use Case |
|---|----------|
| 100 | Small sample baseline |
| 1,000 | Typical analysis |
| 10,000 | Medium-scale data |
| 100,000 | Large-scale performance |

## Cross-Language Comparison

Performance is compared against equivalent implementations:

| p2a Method | R Package | Python Package |
|------------|-----------|----------------|
| `run_ols` | `lm()` | `statsmodels.OLS` |
| `run_hdfe` | `lfe::felm` | - |
| `kmeans` | `stats::kmeans` | `sklearn.cluster.KMeans` |
| ... | ... | ... |

Results are compiled in `comparisons/combined_results.csv`.

## Interpretation Guidelines

1. **Faster is better**: Lower execution time indicates better performance
2. **Consistency matters**: Lower std deviation indicates stable performance
3. **Scaling behavior**: Linear scaling (O(n)) is typical; check for worse cases
4. **Memory efficiency**: Watch for memory bloat with large n

## Adding New Benchmarks

When implementing a new method, add benchmarks:

1. Add to appropriate `benchmarks/*.rs` file
2. Test multiple sample sizes
3. Run and save results
4. Update category report

See `methodology.md` for detailed protocol.
