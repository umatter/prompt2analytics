# Reproducibility Guide

This document provides detailed instructions for reproducing the benchmarks, validation tests, and LLM evaluation results presented in the paper.

## Prerequisites

### R Environment

R version 4.5.0 or later with the following packages:

```r
install.packages(c(
  "plm",           # Panel data models
  "lfe",           # High-dimensional fixed effects
  "sandwich",      # Robust standard errors
  "forecast",      # Time series (ARIMA, MSTL)
  "microbenchmark", # Benchmark timing
  "bench",         # Alternative benchmark package
  "dplyr",         # Data manipulation
  "ggplot2",       # Plotting
  "data.table"     # Fast data operations
))
```

### Rust Environment

Rust 1.85 or later (edition 2024):

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build in release mode with LTO
cargo build --release -p p2a-cli
```

### Hardware

Benchmarks were run on:
- Intel Core i7-1260P (12th Gen)
- 64 GB RAM
- Linux 6.17.9
- R 4.5.2 with OpenBLAS
- Rust 1.92.0 with LTO enabled

Results may vary on different hardware but relative performance should be similar.

## Benchmark Protocol

### Random Seeds

Both R and Rust use seed = 42 for reproducibility:

```r
# R
set.seed(42)
```

```rust
// Rust
use rand::SeedableRng;
let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
```

### Warmup Protocol

Before measurement:
- **Fast methods** (OLS, robust SEs, PCA): 10 warmup iterations, then 100 measured
- **Moderate methods** (panel FE, clustering): 10 warmup iterations, then 50 measured
- **Slow methods** (ARIMA, MSTL): 5 warmup iterations, then 20 measured

Warmup allows R's JIT compiler to optimize and stabilizes Rust's Criterion benchmarks.

### Timing Measurement

R benchmarks use `microbenchmark::microbenchmark()` which:
- Forces garbage collection before each iteration
- Reports median, mean, min, max, and quartiles
- Handles warmup automatically

Rust benchmarks use Criterion which:
- Provides automatic warmup
- Detects and excludes outliers
- Reports confidence intervals

## Running Benchmarks

### Full Benchmark Suite

```bash
cd paper/code
make benchmarks
```

This runs both R and Rust benchmarks and generates:
- `data/benchmark_results_r.csv`
- `data/benchmark_results_rust.csv`
- `data/benchmark_combined.csv`

### Individual Benchmarks

```bash
# R benchmarks only
Rscript run_r_benchmarks.R

# Rust benchmarks only
cargo bench -p p2a-core --features bench
```

### Generating Figures

```bash
# After running benchmarks:
Rscript fig_benchmark_histogram.R
Rscript fig_benchmark_speedup.R
Rscript fig_benchmark_execution.R
```

## Validation Protocol

### Numerical Tolerances

Tolerances are sample-size dependent:

| Sample Size | Tolerance | Rationale |
|------------|-----------|-----------|
| n < 100 | 1e-6 | Ill-conditioned matrices amplify differences |
| 100 <= n < 1000 | 1e-8 | Standard precision |
| n >= 1000 | 1e-10 | Near machine precision |

### Running Validation Tests

```bash
# All validation tests
cargo test -p p2a-core --features validation

# Specific method
cargo test -p p2a-core test_ols_longley
cargo test -p p2a-core test_panel_grunfeld
```

### Comparing Against R

The `rust_validation/` directory contains scripts for cross-language validation:

```bash
cd paper/code/rust_validation
Rscript generate_r_reference.R  # Generate R reference values
cargo run --release              # Compare Rust output
```

## LLM Evaluation

### Test Cases

The 87 test cases are in `llm_eval/test_cases/`:
- `regression.json` - OLS, robust SEs, diagnostics
- `panel.json` - Fixed effects, random effects, Hausman
- `discrete.json` - Logit, probit
- `timeseries.json` - ARIMA, VAR, VECM
- `causal.json` - DiD, IV, treatment effects

### Running Evaluation

```bash
cd paper/code/llm_eval

# Single model evaluation
python evaluate_model.py --model gpt-4o-mini --output results/

# Full evaluation suite
python run_all_evaluations.py
```

### API Keys

Set environment variables for API access:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"  # For Mistral, Llama models
```

### Local Model Evaluation (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.1:8b
ollama pull ministral:3b

# Run evaluation
python evaluate_model.py --model ollama/llama3.1:8b --local
```

## Output Files

After running all scripts:

```
paper/
├── figures/
│   ├── benchmark_histogram.pdf
│   ├── benchmark_speedup.pdf
│   ├── benchmark_boxplots.pdf
│   └── fig_latency_comparison.pdf
├── tables/
│   ├── tab_benchmark_summary.tex
│   ├── tab_model_summary.tex
│   └── tab_category_accuracy.tex
└── data/
    ├── benchmark_combined.csv
    └── llm_eval_results.csv
```

## Troubleshooting

### R Package Version Conflicts

If `plm` or `lfe` fail to install:
```r
# Install specific versions
remotes::install_version("plm", version = "2.6-4")
remotes::install_version("lfe", version = "2.9-0")
```

### Rust Build Failures

```bash
# Clear build cache
cargo clean

# Rebuild with verbose output
cargo build --release -p p2a-cli -v
```

### Benchmark Variance

If results show high variance:
1. Close other applications
2. Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance`
3. Increase iteration count in benchmark scripts

## Contact

For reproducibility issues, please open a GitHub issue at:
https://github.com/prompt2analytics/prompt2analytics/issues
