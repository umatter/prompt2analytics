# Validation: Regression Diagnostics

## Method Overview

Regression diagnostics test assumptions of the linear model:
- **Jarque-Bera**: Normality of residuals
- **Breusch-Pagan**: Homoskedasticity
- **Durbin-Watson**: No autocorrelation
- **VIF**: No multicollinearity
- **Condition Number**: Matrix conditioning

## Reference Implementations

| Test | R Package | Function |
|------|-----------|----------|
| Jarque-Bera | tseries | `jarque.bera.test()` |
| Breusch-Pagan | lmtest | `bptest()` |
| Durbin-Watson | lmtest | `dwtest()` |
| VIF | car | `vif()` |
| Condition Number | base | `kappa()` |

## Test Cases

### Jarque-Bera Test

**R Code**:
```r
library(tseries)

set.seed(42)
# Normal residuals
resid_normal <- rnorm(100)
jarque.bera.test(resid_normal)
# p-value should be > 0.05

# Non-normal residuals
resid_skewed <- rexp(100)
jarque.bera.test(resid_skewed)
# p-value should be < 0.05
```

**Formula**:
```
JB = (n/6) × (S² + (K-3)²/4)
```
where S = skewness, K = kurtosis.

---

### Breusch-Pagan Test

**R Code**:
```r
library(lmtest)

set.seed(42)
x <- rnorm(100)
y <- 1 + 2*x + rnorm(100, 0, abs(x))  # Heteroskedastic

fit <- lm(y ~ x)
bptest(fit)
# p-value should be < 0.05 (heteroskedasticity detected)
```

---

### Durbin-Watson Test

**R Code**:
```r
library(lmtest)

# Autocorrelated errors
set.seed(42)
e <- arima.sim(model = list(ar = 0.7), n = 100)
x <- 1:100
y <- 1 + 2*x + e

fit <- lm(y ~ x)
dwtest(fit)
# DW < 2 indicates positive autocorrelation
```

**Interpretation**:
- DW ≈ 2: No autocorrelation
- DW < 2: Positive autocorrelation
- DW > 2: Negative autocorrelation

---

### VIF

**R Code**:
```r
library(car)

set.seed(42)
x1 <- rnorm(100)
x2 <- 0.9 * x1 + rnorm(100, 0, 0.3)  # Highly correlated
x3 <- rnorm(100)
y <- 1 + x1 + x2 + x3 + rnorm(100)

fit <- lm(y ~ x1 + x2 + x3)
vif(fit)
# x1 and x2 should have high VIF (> 5)
```

## Numerical Precision

| Diagnostic | Test Stat Precision | p-value Precision |
|------------|--------------------|--------------------|
| JB | < 0.1 | < 0.01 |
| BP | < 0.1 | < 0.01 |
| DW | < 0.01 | < 0.01 |
| VIF | < 0.1 | - |

## Running the Tests

```bash
cargo test -p p2a-core -- diagnostics
```

## References

- Jarque, C.M. & Bera, A.K. (1980). "Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals". *Economics Letters*, 6(3), 255-259.
- Breusch, T.S. & Pagan, A.R. (1979). "A Simple Test for Heteroscedasticity and Random Coefficient Variation". *Econometrica*, 47(5), 1287-1294.
