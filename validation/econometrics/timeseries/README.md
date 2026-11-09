# Time Series Validation

Validation documents for the multivariate time-series methods in
`crates/p2a-core/src/econometrics/timeseries/`. Whether each is pinned to R, and at what tolerance,
is in the method index of [../../README.md](../../README.md) (derived from
`validation/methods/*.json`) and on `validation/VALIDATION_STATUS.md`.

## Methods

| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| VAR | [var.md](var.md) | `run_var()` | R `vars::VAR()` |
| IRF | [irf.md](irf.md) | `run_var_irf()`, `compute_irf()` | R `vars::irf()` |
| Granger causality | [granger.md](granger.md) | `granger_test()`, `run_granger_test()` | R `lmtest::grangertest()` |
| VARMA | [varma.md](varma.md) | `run_varma()`, `run_varma_with()` | Hannan-Rissanen (1982) written out in R; q = 0 checked against `vars::VAR()`. q > 0: coefficients, Sigma_u and information criteria; no SEs (step-2 OLS SEs are not valid) |
| VECM | [vecm.md](vecm.md) | `run_vecm_with()`, `run_vecm()` | R `urca::ca.jo()`, `urca::cajorls()` |

Integer series and missing values in Granger tests: [../granger.md](../granger.md).

## Running Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::timeseries::
validation/run_validation.sh --rust-only --category econometrics
```
