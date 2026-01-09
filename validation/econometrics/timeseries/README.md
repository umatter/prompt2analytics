# Time Series Validation

This directory contains validation documentation for time series econometric methods.

## Methods

| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| VAR | [var.md](var.md) | `run_var()` | R `vars::VAR()` |
| VARMA | [varma.md](varma.md) | `run_varma()` | R custom |
| VECM | [vecm.md](vecm.md) | `run_vecm()` | R `vars::vec2var()`, `urca::ca.jo()` |
| IRF | [irf.md](irf.md) | `run_var_irf()` | R `vars::irf()` |

## Key Test Cases

- **Bivariate VAR**: Two-variable system for basic validation
- **Cointegration tests**: Johansen procedure for VECM
- **Orthogonalized vs structural IRF**: Different identification schemes

## Running Tests

```bash
cargo test -p p2a-core -- timeseries::tests::test_validate
```
