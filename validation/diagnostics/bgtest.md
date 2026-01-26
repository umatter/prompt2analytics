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

## R Comparison

```r
library(lmtest)

# Generate data with serial correlation
set.seed(123)
n <- 100
x <- 1:n
e <- numeric(n)
e[1] <- rnorm(1)
for(i in 2:n) {
  e[i] <- 0.7 * e[i-1] + rnorm(1)  # AR(1) errors
}
y <- 2 + 0.5 * x + e

model <- lm(y ~ x)

# Breusch-Godfrey test (order 1)
bg1 <- bgtest(model, order = 1)
print(bg1)
#   Breusch-Godfrey test for serial correlation of order up to 1
#
# data:  model
# LM test = 51.837, df = 1, p-value = 5.929e-13

# Breusch-Godfrey test (order 4)
bg4 <- bgtest(model, order = 4)
print(bg4)

# F-test version
bgf <- bgtest(model, order = 1, type = "F")
print(bgf)
```

## Test Cases

| Test | Order | Type | Expected |
|------|-------|------|----------|
| No autocorrelation | 1 | Chisq | p > 0.05 |
| Strong AR(1) | 1 | Chisq | p < 0.05 |
| Higher order | 4 | Chisq | Valid statistic |
| F-test variant | 1 | F | Valid F-statistic |

## Validation Status

- [x] Core implementation complete
- [x] Unit tests passing
- [x] MCP tool exposed
- [ ] R comparison benchmarks
- [ ] Performance benchmarks

## References

- Breusch, T.S. (1979). Testing for autocorrelation in dynamic linear models. *Australian Economic Papers*, 17, 334-355.
- Godfrey, L.G. (1978). Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables. *Econometrica*, 46, 1293-1302.
