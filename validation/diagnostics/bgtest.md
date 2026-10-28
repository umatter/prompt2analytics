# Breusch-Godfrey Test Validation

## Method
**Name:** Breusch-Godfrey Test (bgtest)
**R Package:** lmtest
**Category:** Diagnostics
**Priority Score:** 95

## Description

The Breusch-Godfrey test is a Lagrange Multiplier test for higher-order serial correlation in regression residuals. It is more general than the Durbin-Watson test because:

1. It tests for serial correlation of order p (AR(p) or MA(p))
2. It is valid when lagged dependent variables appear as regressors
3. It is asymptotically valid regardless of regressor stochasticity

## Mathematical Formulation

Given the original regression:
```
y = Xβ + ε
```

The test procedure:
1. Estimate the model and compute residuals: ê = y - Xβ̂
2. Run auxiliary regression: êₜ = Xβ + ρ₁êₜ₋₁ + ... + ρₚêₜ₋ₚ + v
3. Compute test statistic:
   - **Chi-squared:** LM = n × R² ~ χ²(p)
   - **F-test:** F = (R²/p) / ((1-R²)/(n-k-p)) ~ F(p, n-k-p)

## Implementation

**Rust Functions:**
- `bg_test(dataset, y_col, x_cols, order, test_type, fill)` - Main function
- `run_bg_test(dataset, y_col, x_cols)` - Convenience wrapper (order=1, chisq)
- `bg_test_from_ols(ols_result, x, order, test_type, fill)` - From existing OLS

**MCP Tool:** `regression_bgtest`

**Location:** `crates/p2a-core/src/regression/diagnostics.rs`

## R reference (pinned)

Two R reference cases pin the test: the AR(1) DGP of the legacy script
`validation/scripts/validate_regression_diag.R` (Test 4), ported with the same seed and draws,
and the Grunfeld regression of the `p2a reg diagnostics` exhibit.

| Case | Data | R code |
|---|---|---|
| `diag_bgtest_ar1` | `validation/datasets/generated/bgtest_ar1.csv` (n = 100; `y = 2 + 3 x + e`, `e` AR(1) with rho = 0.5, `set.seed(42)`) | `validation/r/cases/regression_legacy.R` |
| `ols_diagnostics_grunfeld` | `validation/datasets/grunfeld.csv` (n = 200; `inv ~ value + capital`) | `validation/r/cases/regression.R` |

R call: `bgtest(lm(y ~ x), order = 1 | 4, type = "Chisq" | "F")` (lmtest 0.9.40, `fill = 0`).
The auxiliary R-squared comes from `stats::lm` on the zero-filled lagged residuals.

| Case | Quantity | R value |
|---|---|---|
| `diag_bgtest_ar1` | LM, order 1 (df, p) | 29.41725627 (1, 5.835570525e-08) |
| | LM, order 4 (df, p) | 29.62679143 (4, 5.829739137e-06) |
| | F, order 1 (df1, df2, p) | 40.42735812 (1, 97, 6.684182268e-09) |
| | F, order 4 (df1, df2, p) | 9.893390009 (4, 94, 1.004726915e-06) |
| | lagged-residual coefficient, order 1 | 0.5430943053 |
| | lagged-residual coefficients, order 4 | 0.5615690953, -0.04353347514, 0.04124071549, -0.04081137602 |
| | auxiliary R-squared, orders 1 and 4 | 0.2941725627, 0.2962679143 |
| `ols_diagnostics_grunfeld` | LM, order 1 (p) | 135.2510384 (2.909552493e-31) |
| | F, order 1 (p) | 409.4151148 (6.902946376e-50); df (1, 196) |
| | LM, order 4 (p) | 135.4159814 (2.702716519e-28) |

Tests: `test_validate_bgtest_ar1` (every value above for `diag_bgtest_ar1`, and n) and
`test_validate_bgtest_grunfeld` (the Grunfeld rows; the F df are asserted directly because
the case does not store them).

Tolerance (`validation/README.md`, n = 100 and 200): statistics and R-squared 1e-6 relative,
coefficients 1e-8 relative, p-values 1e-6 relative or 1e-12 absolute; df and n exact. The
methods registry entry is `bgtest` in `validation/methods/regression_diagnostics.json`; it
records no deviation. These cases supersede the legacy `validation/expected/bgtest_test.csv`,
whose values they reproduce to the 15 digits it prints.

## Property tests

- `test_bg_test_basic`
- `test_bg_test_higher_order`
- `test_bg_test_f_type`
- `test_bg_test_with_autocorrelation`

## Validation Status

R-pinned.

- [x] Core implementation complete
- [x] Unit tests passing
- [x] MCP tool exposed
- [x] R reference cases pinned (`diag_bgtest_ar1`, `ols_diagnostics_grunfeld`)
- [ ] Performance benchmarks

## References

- Breusch, T.S. (1979). Testing for autocorrelation in dynamic linear models. *Australian Economic Papers*, 17, 334-355.
- Godfrey, L.G. (1978). Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables. *Econometrica*, 46, 1293-1302.
