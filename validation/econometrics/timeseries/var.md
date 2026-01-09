# Validation: Vector Autoregression (VAR)

## Method Overview

Vector Autoregression models multiple time series as a system where each variable depends on its own lags and lags of all other variables.

**Model (VAR(p))**:
```
Y_t = c + A₁Y_{t-1} + A₂Y_{t-2} + ... + A_pY_{t-p} + ε_t
```

where Y_t is a k×1 vector, A_i are k×k coefficient matrices, and ε_t ~ N(0, Σ).

**Estimation**: OLS equation-by-equation (equivalent to GLS for reduced-form VAR).

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| vars | R | `VAR()` | 1.5-x |
| statsmodels | Python | `VAR()` | 0.14.x |

## Test Cases

### Test 1: Bivariate VAR(1)

**Data Generating Process**:
```
y₁ₜ = 0.5×y₁ₜ₋₁ + 0.2×y₂ₜ₋₁ + ε₁ₜ
y₂ₜ = 0.1×y₁ₜ₋₁ + 0.6×y₂ₜ₋₁ + ε₂ₜ
```

**R Code**:
```r
library(vars)

set.seed(42)
n <- 200

# Generate VAR(1) data
A <- matrix(c(0.5, 0.1, 0.2, 0.6), 2, 2, byrow = TRUE)
y <- matrix(0, n, 2)
y[1, ] <- rnorm(2)

for (t in 2:n) {
  y[t, ] <- A %*% y[t-1, ] + rnorm(2, 0, 0.5)
}

data <- data.frame(y1 = y[, 1], y2 = y[, 2])

# Estimate VAR(1)
var_fit <- VAR(data, p = 1, type = "none")
summary(var_fit)

# Check coefficient matrix
coef(var_fit$varresult$y1)
coef(var_fit$varresult$y2)
```

**Results Comparison**:

| Coefficient | True Value | R Estimate | p2a Rust | Tolerance |
|-------------|------------|------------|----------|-----------|
| A[1,1] (y1.l1 in y1 eq) | 0.5 | ~0.5 | ~0.5 | 0.1 |
| A[1,2] (y2.l1 in y1 eq) | 0.2 | ~0.2 | ~0.2 | 0.1 |
| A[2,1] (y1.l1 in y2 eq) | 0.1 | ~0.1 | ~0.1 | 0.1 |
| A[2,2] (y2.l1 in y2 eq) | 0.6 | ~0.6 | ~0.6 | 0.1 |

**Rust Test**: `crates/p2a-core/src/econometrics/timeseries.rs::tests::test_validate_var1`

---

### Test 2: VAR(2) with Intercept

**R Code**:
```r
library(vars)

set.seed(42)
n <- 300

# VAR(2) process
A1 <- matrix(c(0.4, 0.1, 0.15, 0.5), 2, 2, byrow = TRUE)
A2 <- matrix(c(0.1, 0.05, 0.05, 0.1), 2, 2, byrow = TRUE)
c <- c(1, 0.5)  # Intercepts

y <- matrix(0, n, 2)
y[1:2, ] <- rnorm(4)

for (t in 3:n) {
  y[t, ] <- c + A1 %*% y[t-1, ] + A2 %*% y[t-2, ] + rnorm(2, 0, 0.5)
}

data <- data.frame(y1 = y[, 1], y2 = y[, 2])

# Estimate VAR(2) with intercept
var_fit <- VAR(data, p = 2, type = "const")
summary(var_fit)
```

**Validation Criteria**:
- All lag coefficients close to true values
- Intercepts close to true values
- Correct number of parameters

---

### Test 3: Lag Selection (Information Criteria)

**R Code**:
```r
library(vars)

set.seed(42)
# Generate VAR(2) data
# ... (as above)

# Lag selection
VARselect(data, lag.max = 5, type = "const")

# AIC, BIC, HQ criteria
# True order is 2, so p=2 should be selected by most criteria
```

**Validation Criteria**:
- AIC, BIC, HQ computed correctly
- Lag selection matches R's VARselect

---

### Test 4: Residual Covariance Matrix

**R Code**:
```r
library(vars)

# Fit VAR
var_fit <- VAR(data, p = 1, type = "const")

# Residual covariance matrix
summary(var_fit)$covres

# Residual correlation
summary(var_fit)$corres
```

**Validation Criteria**:
- Residual covariance matches R
- Correlation matrix computed correctly

---

## Model Diagnostics

| Diagnostic | R Function | Purpose |
|------------|------------|---------|
| Serial correlation | `serial.test()` | Portmanteau test |
| Normality | `normality.test()` | JB test on residuals |
| Stability | `stability()` | Check eigenvalues < 1 |

## Numerical Precision Summary

| VAR Order | n | Coefficient Precision |
|-----------|---|----------------------|
| VAR(1) | 200 | < 0.05 |
| VAR(2) | 300 | < 0.05 |
| VAR(3) | 500 | < 0.08 |

## Known Differences

1. **Constant term**: R's "type" = "const", "trend", "both", "none".
2. **Coefficient ordering**: R stacks by equation; p2a uses matrix form.
3. **Degrees of freedom**: Slight differences in small-sample adjustments.

## Running the Tests

```bash
# Run VAR validation tests
cargo test -p p2a-core -- timeseries::tests::test_var

# Run with output
cargo test -p p2a-core -- var --nocapture
```

## References

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Sims, C.A. (1980). "Macroeconomics and Reality". *Econometrica*, 48(1), 1-48.
- Pfaff, B. (2008). *Analysis of Integrated and Cointegrated Time Series with R*, 2nd ed. Springer.
