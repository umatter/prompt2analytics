# Benchmarking Methodology

This document describes the methodology for performance benchmarking in prompt2analytics.

## Principles

1. **Reproducibility**: All benchmarks must be reproducible on any compatible hardware
2. **Statistical rigor**: Use proper statistical methods for timing measurements
3. **Fair comparisons**: Ensure equivalent workloads across languages
4. **Distribution-based reporting**: Report full timing distributions, not just point estimates
5. **Memory tracking**: Track memory allocation alongside timing
6. **Documentation**: Record all relevant environmental factors

## Quick Start

### Rust Comprehensive Benchmarks

```bash
# Run comprehensive benchmarks with distribution statistics and memory tracking
cargo bench -p p2a-core --bench comprehensive_benchmarks

# Results saved to: performance/results/rust_comprehensive_YYYYMMDD_HHMMSS.json
```

### R Comprehensive Benchmarks

```bash
# Install required packages
R -e 'install.packages(c("bench", "sandwich", "plm", "lfe", "forecast"))'

# Run comprehensive benchmarks
cd performance/comparisons/r_comparison
Rscript benchmark_comprehensive.R

# Results saved to: results/r_comprehensive_YYYYMMDD_HHMMSS.csv
```

## Output Format

Both Rust and R benchmarks produce consistent output with:

| Statistic | Description |
|-----------|-------------|
| `time_min_us` | Minimum time (µs) |
| `time_p25_us` | 25th percentile (µs) |
| `time_median_us` | Median time (µs) |
| `time_p75_us` | 75th percentile (µs) |
| `time_max_us` | Maximum time (µs) |
| `time_mean_us` | Mean time (µs) |
| `time_std_us` | Standard deviation (µs) |
| `itr_per_sec` | Iterations per second |
| `mem_alloc_bytes` | Memory allocated (bytes) |

This matches the output format of R's `bench::mark()` function

## Benchmarking Protocol

### 1. Environment Preparation

Before running benchmarks:

```bash
# Close other applications to minimize interference
# Disable power management / frequency scaling if possible
# Run on AC power (not battery)

# Document environment
lscpu > hardware_info.txt
cat /proc/meminfo >> hardware_info.txt
rustc --version >> hardware_info.txt
R --version >> hardware_info.txt
python --version >> hardware_info.txt
```

### 2. Warmup Phase

Allow JIT compilation and cache warming:
- **Rust/Criterion**: Automatic (configurable warmup time)
- **R**: Run function 10 times before measuring
- **Python**: Run function 10 times before measuring

### 3. Measurement Phase

Statistical requirements:
- **Minimum iterations**: 100 (or until CI is stable)
- **Confidence level**: 95%
- **Outlier handling**: Report but don't exclude

### 4. Data Generation

Use consistent random seeds across languages:

```rust
// Rust
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha8Rng;
let mut rng = ChaCha8Rng::seed_from_u64(42);
```

```r
# R
set.seed(42)
```

```python
# Python
import numpy as np
np.random.seed(42)
```

## Criterion Configuration

Standard Criterion configuration for p2a-core:

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, SamplingMode};
use std::time::Duration;

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(3))
        .sampling_mode(SamplingMode::Auto)
}
```

## Benchmark Structure

### Standard Benchmark Template

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use p2a_core::{Dataset, run_ols, CovarianceType};

fn generate_data(n: usize, k: usize) -> Dataset {
    // Use fixed seed for reproducibility
    // Generate synthetic regression data
    // Return Dataset
}

fn ols_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("OLS");

    for n in [100, 1000, 10000, 100000] {
        let dataset = generate_data(n, 5);
        let x_cols: Vec<&str> = (1..=5).map(|i| format!("x{}", i)).collect();

        group.bench_with_input(
            BenchmarkId::new("standard", n),
            &(&dataset, &x_cols),
            |b, (d, cols)| {
                b.iter(|| run_ols(d, "y", cols, true, CovarianceType::Standard))
            }
        );

        group.bench_with_input(
            BenchmarkId::new("HC1", n),
            &(&dataset, &x_cols),
            |b, (d, cols)| {
                b.iter(|| run_ols(d, "y", cols, true, CovarianceType::HC1))
            }
        );
    }

    group.finish();
}

criterion_group!(benches, ols_benchmarks);
criterion_main!(benches);
```

## R Benchmark Template

```r
library(microbenchmark)

generate_data <- function(n, k) {
  set.seed(42)
  data <- data.frame(
    y = rnorm(n),
    matrix(rnorm(n * k), ncol = k, dimnames = list(NULL, paste0("x", 1:k)))
  )
  return(data)
}

run_benchmarks <- function(n_values = c(100, 1000, 10000, 100000)) {
  results <- data.frame()

  for (n in n_values) {
    data <- generate_data(n, 5)

    mb <- microbenchmark(
      lm(y ~ x1 + x2 + x3 + x4 + x5, data = data),
      times = 100,
      unit = "us"
    )

    result <- data.frame(
      method = "lm",
      variant = "standard",
      n = n,
      k = 5,
      mean_us = mean(mb$time) / 1000,
      median_us = median(mb$time) / 1000,
      std_us = sd(mb$time) / 1000,
      timestamp = Sys.time()
    )

    results <- rbind(results, result)
  }

  return(results)
}

# Save results
results <- run_benchmarks()
write.csv(results, "results.csv", row.names = FALSE)
```

## Python Benchmark Template

```python
import timeit
import numpy as np
import pandas as pd
from datetime import datetime

def generate_data(n, k, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, k)
    y = np.random.randn(n)
    return pd.DataFrame(
        np.column_stack([y, X]),
        columns=['y'] + [f'x{i}' for i in range(1, k+1)]
    )

def benchmark_function(func, data, n_runs=100):
    times = timeit.repeat(
        lambda: func(data),
        number=1,
        repeat=n_runs
    )
    return {
        'mean_us': np.mean(times) * 1e6,
        'median_us': np.median(times) * 1e6,
        'std_us': np.std(times) * 1e6,
    }

def run_benchmarks(n_values=[100, 1000, 10000, 100000]):
    import statsmodels.api as sm

    results = []

    for n in n_values:
        data = generate_data(n, 5)

        def run_ols(d):
            X = sm.add_constant(d[['x1', 'x2', 'x3', 'x4', 'x5']])
            return sm.OLS(d['y'], X).fit()

        timing = benchmark_function(run_ols, data)
        timing.update({
            'method': 'statsmodels.OLS',
            'variant': 'standard',
            'n': n,
            'k': 5,
            'timestamp': datetime.now().isoformat()
        })
        results.append(timing)

    return pd.DataFrame(results)

if __name__ == "__main__":
    results = run_benchmarks()
    results.to_csv('results.csv', index=False)
```

## Scaling Analysis

For each method, analyze scaling behavior:

1. **Fit complexity model**: O(n), O(n²), O(n³)?
2. **Memory scaling**: Linear, superlinear?
3. **Identify bottlenecks**: Which operations dominate?

### Complexity Categories

| Method | Expected Time | Expected Space |
|--------|---------------|----------------|
| OLS (n×k matrix) | O(nk² + k³) | O(nk) |
| HDFE (iterative) | O(iter × n × fe) | O(n) |
| K-means | O(iter × n × k × K) | O(nk + Kk) |
| PCA | O(min(n,k)²) | O(nk) |

## Memory Profiling

### Rust Memory Profiling

```bash
# Using peak RSS
/usr/bin/time -v cargo run --release --example benchmark_ols 2>&1 | grep "Maximum resident"
```

### Heaptrack (Linux)

```bash
cargo build --release
heaptrack target/release/examples/benchmark_ols
heaptrack_gui heaptrack.benchmark_ols.*.zst
```

## Reporting Results

### Summary Statistics Table

| Method | n | Mean (μs) | Median (μs) | 95% CI | Memory (KB) |
|--------|---|-----------|-------------|--------|-------------|
| OLS | 1000 | 245.3 | 242.1 | [240.1, 250.5] | 1024 |

### Cross-Language Comparison

| Method | p2a (μs) | R (μs) | Python (μs) | p2a vs R | p2a vs Python |
|--------|----------|--------|-------------|----------|---------------|
| OLS n=1000 | 245 | 1200 | 890 | 4.9x | 3.6x |

### Scaling Plot

Generate plots showing execution time vs sample size:

```r
library(ggplot2)
results <- read.csv("combined_results.csv")
ggplot(results, aes(x = n, y = mean_us, color = language)) +
  geom_line() +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "OLS Performance Scaling",
       x = "Sample Size (n)",
       y = "Execution Time (μs)")
```

## Version Tracking

Record software versions for each benchmark run:

```json
{
  "timestamp": "2026-01-09T10:00:00Z",
  "rust": "1.75.0",
  "p2a_core": "0.2.0",
  "r": "4.3.2",
  "python": "3.11.5",
  "statsmodels": "0.14.0",
  "sklearn": "1.3.0"
}
```

## Best Practices

1. **Isolate benchmarks**: Run on dedicated hardware when possible
2. **Multiple runs**: Run full benchmark suite multiple times
3. **Control for variance**: Close background processes
4. **Document anomalies**: Note any unusual results
5. **Compare fairly**: Use equivalent algorithms, not just equivalent results
