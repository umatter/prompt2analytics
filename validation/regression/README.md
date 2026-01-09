# Regression Validation

This directory contains validation documentation for regression methods.

## Methods

| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| OLS | [ols.md](ols.md) | `run_ols()` | R `lm()`, statsmodels |
| Robust SEs | [ols_robust_se.md](ols_robust_se.md) | `run_ols()` with HC0-HC3 | R `sandwich::vcovHC()` |
| Clustered SEs | [ols_clustered.md](ols_clustered.md) | `run_ols_clustered()` | R `sandwich::vcovCL()` |

## Key Test Datasets

- **Longley (1967)**: Classic dataset for testing collinearity (n=16, k=6)
- **Synthetic data**: Known DGP for exact coefficient recovery

## Running Tests

```bash
cargo test -p p2a-core -- regression::ols::tests::test_validate
```
