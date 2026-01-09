# Econometrics Performance Report

## Overview

This report documents the performance of p2a econometric methods compared to reference implementations.

## Methods Benchmarked

| Method | p2a Function | R Reference | Notes |
|--------|--------------|-------------|-------|
| Fixed Effects | `run_fixed_effects` | `plm::plm()`, `lfe::felm()` | Within transformation |
| Random Effects | `run_random_effects` | `plm::plm()` | Swamy-Arora |
| HDFE | `run_hdfe` | `lfe::felm()` | Method of Alternating Projections |
| Hausman Test | `run_hausman_test` | `plm::phtest()` | FE vs RE |
| IV/2SLS | `run_iv2sls` | `AER::ivreg()` | Two-stage least squares |
| Logit | `run_logit` | `stats::glm()` | Newton-Raphson MLE |
| Probit | `run_probit` | `stats::glm()` | Newton-Raphson MLE |

## Benchmark Configuration

- **Rust**: Criterion with 50 measurement iterations
- **R**: microbenchmark with 50 iterations
- **Panel sizes**: (entities × periods) ∈ {(10×10), (50×20), (100×50)}

## Results Summary

### Fixed Effects

| n | p2a Rust (μs) | R plm (μs) | R lfe (μs) | Speedup vs plm |
|---|---------------|------------|------------|----------------|
| 100 | TBD | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD | TBD |
| 5000 | TBD | TBD | TBD | TBD |

### HDFE (Two-way Fixed Effects)

| n | p2a Rust (μs) | R lfe::felm (μs) | Speedup |
|---|---------------|------------------|---------|
| 100 | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD |
| 5000 | TBD | TBD | TBD |

### Discrete Choice Models

| Method | n | p2a Rust (μs) | R glm (μs) | Speedup |
|--------|---|---------------|------------|---------|
| Logit | 100 | TBD | TBD | TBD |
| Logit | 500 | TBD | TBD | TBD |
| Logit | 1000 | TBD | TBD | TBD |
| Probit | 100 | TBD | TBD | TBD |
| Probit | 500 | TBD | TBD | TBD |
| Probit | 1000 | TBD | TBD | TBD |

## Running Benchmarks

### Rust Benchmarks

```bash
cargo bench -p p2a-core -- econometrics
```

### R Benchmarks

```bash
cd performance/comparisons/r_comparison
Rscript benchmark_econometrics.R
```

## HDFE Convergence

The p2a HDFE implementation uses the Method of Alternating Projections (MAP):
- Default tolerance: 1e-8
- Default max iterations: 1000
- Typical convergence: 5-20 iterations

## Notes

- Results marked "TBD" will be populated after running the benchmark suite
- Timings are median values to reduce impact of outliers
- MLE methods (Logit, Probit) use Newton-Raphson which may require multiple iterations

## Hardware Configuration

See `performance/hardware_profiles.md` for benchmark hardware specifications.
