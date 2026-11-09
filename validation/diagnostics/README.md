# Regression Diagnostics Validation

This directory contains validation documentation for regression diagnostic tests.

## Methods

| Diagnostic | File | p2a Function | Reference |
|------------|------|--------------|-----------|
| Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF, condition number, RESET | [regression_diagnostics.md](regression_diagnostics.md) | `run_diagnostics()`, `reset_test()` | R `tseries::jarque.bera.test()`, `lmtest::bptest()`, `lmtest::dwtest()`, `car::vif()`, `kappa()`, `lmtest::resettest()` |
| Breusch-Godfrey | [bgtest.md](bgtest.md) | `bg_test()` | R `lmtest::bgtest()` |
| Harvey-Collier | [harvtest.md](harvtest.md) | `harvey_collier_test()` | R `lmtest::harvtest()` |

Whether each is pinned to R is in the method index of [../README.md](../README.md).

## Validation Approach

Each diagnostic test validates:
1. **Test statistic**: The computed statistic value
2. **P-value**: For hypothesis tests
3. **Critical values**: Where applicable

## Key Test Cases

- **Normal residuals**: For Jarque-Bera validation
- **Heteroskedastic data**: For Breusch-Pagan validation
- **Autocorrelated data**: For Durbin-Watson validation
- **Collinear data**: For VIF/condition number validation

## Running Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- regression::diagnostics::tests::test_validate
```
