# Regression Diagnostics Validation

This directory contains validation documentation for regression diagnostic tests.

## Methods

| Diagnostic | p2a Function | Reference |
|------------|--------------|-----------|
| Jarque-Bera | `run_diagnostics()` | R `tseries::jarque.bera.test()` |
| Breusch-Pagan | `run_diagnostics()` | R `lmtest::bptest()` |
| Durbin-Watson | `run_diagnostics()` | R `lmtest::dwtest()` |
| VIF | `run_diagnostics()` | R `car::vif()` |
| Condition Number | `run_diagnostics()` | R `kappa()` |

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
cargo test -p p2a-core -- regression::diagnostics::tests::test_validate
```
