# Validation: Instrumental Variables / Two-Stage Least Squares (2SLS)

## Method Overview

Instrumental Variables (IV) estimation using Two-Stage Least Squares (2SLS) addresses endogeneity when E(ε|X) ≠ 0. It requires instruments Z that are correlated with the endogenous regressors but uncorrelated with the error term.

**Two Stages**:
1. **First Stage**: Regress endogenous X on instruments Z (and exogenous variables)
2. **Second Stage**: Regress Y on predicted X̂ (and exogenous variables)

**Key Parameters**:
- `y_col`: Dependent variable
- `endog_cols`: Endogenous regressors
- `exog_cols`: Exogenous regressors
- `instrument_cols`: Instrumental variables

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| AER | R | `ivreg()` | 1.2-x |
| estimatr | R | `iv_robust()` | 1.0-x |
| linearmodels | Python | `IV2SLS()` | 5.x |

## Test Cases

### Test 1: Classic Supply-Demand Example

**Data Generating Process** (Simultaneous Equations):
```
Quantity = α + β×Price + ε_d  (Demand)
Quantity = γ + δ×Price + ε_s  (Supply)
```

Use weather (Z) as instrument for Price in demand equation.

**R Code**:
```r
library(AER)

set.seed(42)
n <- 500

# Instrument (weather affects supply, not demand directly)
weather <- rnorm(n)

# Structural errors (correlated with price)
eps_d <- rnorm(n, 0, 1)
eps_s <- rnorm(n, 0, 1)

# Supply: price increases with good weather
price <- 5 + 2*weather + eps_s

# Demand: quantity decreases with price
# True elasticity = -0.8
quantity <- 100 - 0.8*price + eps_d

data <- data.frame(quantity = quantity, price = price, weather = weather)

# OLS (biased due to simultaneity)
ols_fit <- lm(quantity ~ price, data = data)
coef(ols_fit)["price"]  # Biased toward 0

# 2SLS using weather as instrument
iv_fit <- ivreg(quantity ~ price | weather, data = data)
coef(iv_fit)["price"]  # Should be close to -0.8
summary(iv_fit)
```

**Validation Criteria**:
- IV coefficient closer to true value (-0.8) than OLS
- First-stage F-statistic significant

---

### Test 2: Multiple Instruments (Overidentified)

**R Code**:
```r
library(AER)

set.seed(42)
n <- 1000

# Two instruments
z1 <- rnorm(n)
z2 <- rnorm(n)

# Endogenous regressor
x <- 1 + 0.5*z1 + 0.3*z2 + rnorm(n, 0, 0.5)

# Outcome
# True coefficient = 2.0
y <- 3 + 2*x + rnorm(n, 0, 1)

data <- data.frame(y = y, x = x, z1 = z1, z2 = z2)

# 2SLS with two instruments (overidentified)
iv_fit <- ivreg(y ~ x | z1 + z2, data = data)
summary(iv_fit, diagnostics = TRUE)

# Check Sargan test for overidentification
```

**Results Comparison**:

| Statistic | R's ivreg | p2a Rust | Tolerance |
|-----------|-----------|----------|-----------|
| β(x) | ~2.0 | ~2.0 | 0.1 |
| First-stage F | > 10 | > 10 | - |
| Sargan statistic | varies | varies | 0.1 |

---

### Test 3: First-Stage Diagnostics

**R Code**:
```r
library(AER)

set.seed(42)
n <- 500

# Strong instrument
z_strong <- rnorm(n)
x_strong <- 2*z_strong + rnorm(n, 0, 0.5)

# Weak instrument
z_weak <- rnorm(n)
x_weak <- 0.1*z_weak + rnorm(n, 0, 1)

# First stage regression
first_stage_strong <- lm(x_strong ~ z_strong)
summary(first_stage_strong)$fstatistic[1]  # Should be >> 10

first_stage_weak <- lm(x_weak ~ z_weak)
summary(first_stage_weak)$fstatistic[1]  # May be < 10 (weak)
```

**Validation Criteria**:
- F-statistic > 10 indicates strong instruments (Staiger-Stock rule)
- Partial R² measures instrument strength
- Report weak instrument warning when F < 10

---

### Test 4: With Exogenous Controls

**R Code**:
```r
library(AER)

set.seed(42)
n <- 500

z <- rnorm(n)
w <- rnorm(n)  # Exogenous control
x <- 1 + 0.5*z + 0.3*w + rnorm(n, 0, 0.5)
y <- 2 + 1.5*x + 0.8*w + rnorm(n, 0, 1)

data <- data.frame(y = y, x = x, z = z, w = w)

# x is endogenous, w is exogenous control
iv_fit <- ivreg(y ~ x + w | z + w, data = data)
summary(iv_fit)
```

**Validation Criteria**:
- Coefficient on x close to 1.5
- Coefficient on w close to 0.8
- w included in both stages

---

## First-Stage Diagnostics

| Diagnostic | Rule of Thumb | Reference |
|------------|---------------|-----------|
| F-statistic | > 10 | Staiger & Stock (1997) |
| Partial R² | Higher is better | - |
| Sanderson-Windmeijer F | > 10 per endogenous var | Sanderson & Windmeijer (2016) |

## Numerical Precision Summary

| Test Case | n | Coefficient Precision |
|-----------|---|----------------------|
| Supply-demand | 500 | < 0.05 |
| Overidentified | 1000 | < 0.03 |

## Known Differences

1. **Standard errors**: Default SEs may differ (robust vs. standard).
2. **Weak instrument tests**: Different implementations of weak ID test.
3. **Overidentification test**: Sargan vs. Hansen J-test for clustered data.

## Running the Tests

```bash
# Run IV validation tests
cargo test -p p2a-core -- iv::tests::test_validate

# Run with output
cargo test -p p2a-core -- iv2sls --nocapture
```

## References

- Angrist, J.D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 4.
- Staiger, D. & Stock, J.H. (1997). "Instrumental Variables Regression with Weak Instruments". *Econometrica*, 65(3), 557-586.
- Stock, J.H. & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV Regression". In *Identification and Inference for Econometric Models*.
