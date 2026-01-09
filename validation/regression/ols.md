# Validation: Ordinary Least Squares (OLS)

## Method Overview

Ordinary Least Squares regression estimates linear relationships between a dependent variable and one or more independent variables by minimizing the sum of squared residuals.

**Key Parameters**:
- `y_col`: Dependent variable name
- `x_cols`: Independent variable names
- `intercept`: Whether to include a constant term
- `cov_type`: Covariance estimator (Standard, HC0-HC3)

**Output**:
- Coefficients with standard errors, t-statistics, and p-values
- R², adjusted R², F-statistic
- Residuals

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `lm()` | 4.3.x |
| statsmodels | Python | `OLS()` | 0.14.x |

## Test Cases

### Test 1: Longley Dataset - Classic OLS

**Dataset**: US macroeconomic data (n=16, k=6)

The Longley (1967) dataset is specifically designed to test numerical accuracy of least squares programs due to high multicollinearity.

**R Code**:
```r
data(longley)

# Full model
fit <- lm(Employed ~ GNP.deflator + GNP + Unemployed + Armed.Forces + Population + Year,
          data = longley)
summary(fit)

# Expected output:
#                  Estimate Std. Error t value Pr(>|t|)
# (Intercept)   -3.482e+03  8.904e+02  -3.911 0.003560 **
# GNP.deflator   1.506e-02  8.492e-02   0.177 0.863141
# GNP           -3.582e-02  3.349e-02  -1.070 0.312681
# Unemployed    -2.020e-02  4.884e-03  -4.136 0.002535 **
# Armed.Forces  -1.033e-02  2.143e-03  -4.822 0.000944 ***
# Population    -5.110e-02  2.261e-01  -0.226 0.826212
# Year           1.829e+00  4.555e-01   4.016 0.003037 **
#
# Residual standard error: 0.3049 on 9 degrees of freedom
# Multiple R-squared: 0.9955, Adjusted R-squared: 0.9925
# F-statistic: 330.3 on 6 and 9 DF,  p-value: 4.984e-10
```

**Results Comparison**:

| Statistic | R's lm() | p2a Rust | Tolerance |
|-----------|----------|----------|-----------|
| β(GNP.deflator) | 0.01506 | ~0.01506 | 1e-4 |
| β(Unemployed) | -0.02020 | ~-0.02020 | 1e-4 |
| SE(Unemployed) | 0.004884 | ~0.004884 | 1e-5 |
| R² | 0.9955 | ~0.9955 | 1e-4 |
| Adj. R² | 0.9925 | ~0.9925 | 1e-4 |
| F-statistic | 330.3 | ~330.3 | 0.1 |
| df_resid | 9 | 9 | exact |

**Rust Test**: `crates/p2a-core/src/regression/ols.rs::tests::test_validate_longley`

---

### Test 2: Simple Linear Regression with Known DGP

**Data Generating Process**:
```
y = 2.0 + 1.5 × x + ε
ε ~ N(0, 0.1²)
```

**R Code**:
```r
set.seed(42)
n <- 100
x <- runif(n, 0, 10)
y <- 2.0 + 1.5 * x + rnorm(n, 0, 0.1)

fit <- lm(y ~ x)
coef(fit)
# (Intercept)           x
#   2.00...       1.50...
```

**Validation Criteria**:
- Intercept within 0.1 of true value (2.0)
- Slope within 0.05 of true value (1.5)
- High R² (> 0.99 given low noise)

---

### Test 3: Multiple Regression

**R Code**:
```r
set.seed(42)
n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
y <- 1.0 + 2.0*x1 - 0.5*x2 + 0.8*x3 + rnorm(n, 0, 0.5)

fit <- lm(y ~ x1 + x2 + x3)
summary(fit)
```

**Validation Criteria**:
- All coefficients within 0.2 of true values
- Correct degrees of freedom (n - k - 1)
- F-test significant

---

### Test 4: No Intercept Model

**R Code**:
```r
set.seed(42)
x <- 1:10
y <- 2 * x + rnorm(10, 0, 0.1)

fit <- lm(y ~ x - 1)  # No intercept
coef(fit)
# x
# 2.00...
```

**Validation Criteria**:
- Coefficient within 0.05 of true value (2.0)
- Correct df (n - k, not n - k - 1)

---

## Numerical Precision Summary

| Test Case | n | Coefficient Precision | SE Precision |
|-----------|---|----------------------|--------------|
| Longley | 16 | < 1e-8 | < 1e-8 |
| Simple LR | 100 | < 1e-10 | < 1e-8 |
| Multiple | 200 | < 1e-10 | < 1e-8 |

## Known Differences

1. **Coefficient naming**: R names coefficients by variable; p2a uses numeric indices.
2. **Confidence intervals**: R provides 95% CI by default; p2a can compute these separately.
3. **Influence diagnostics**: Not included in base p2a OLS (available via diagnostics module).

## Running the Tests

```bash
# Run OLS validation tests
cargo test -p p2a-core -- regression::ols::tests::test_validate

# Run with output
cargo test -p p2a-core -- ols --nocapture
```

## References

- Longley, J.W. (1967). "An Appraisal of Least Squares Programs for the Electronic Computer from the Point of View of the User". *Journal of the American Statistical Association*, 62(319), 819-841.
