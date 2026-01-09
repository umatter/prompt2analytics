# Validation: ARIMA

## Method Overview

ARIMA (AutoRegressive Integrated Moving Average) models time series as a combination of autoregressive (AR), differencing (I), and moving average (MA) components.

**Model ARIMA(p,d,q)**:
```
(1 - φ₁B - ... - φ_pB^p)(1-B)^d Y_t = (1 + θ₁B + ... + θ_qB^q)ε_t
```

where B is the backshift operator.

**Estimation**: Maximum Likelihood.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| forecast | R | `Arima()`, `auto.arima()` | 8.21-x |
| statsmodels | Python | `ARIMA()` | 0.14.x |

## Test Cases

### Test 1: AR(1) Process

**R Code**:
```r
library(forecast)

set.seed(42)
n <- 200

# AR(1) with φ = 0.7
y <- arima.sim(model = list(ar = 0.7), n = n)

# Fit ARIMA(1,0,0)
fit <- Arima(y, order = c(1, 0, 0))
summary(fit)

# Coefficient should be ≈ 0.7
```

**Results Comparison**:

| Parameter | True Value | R Estimate | p2a Rust | Tolerance |
|-----------|------------|------------|----------|-----------|
| AR(1) | 0.7 | ~0.7 | ~0.7 | 0.1 |
| σ² | 1.0 | ~1.0 | ~1.0 | 0.2 |
| AIC | - | varies | varies | 1.0 |

---

### Test 2: ARMA(1,1) Process

**R Code**:
```r
library(forecast)

set.seed(42)
y <- arima.sim(model = list(ar = 0.5, ma = 0.3), n = 300)

fit <- Arima(y, order = c(1, 0, 1))
summary(fit)

# ar1 ≈ 0.5, ma1 ≈ 0.3
```

---

### Test 3: ARIMA(1,1,1) - With Differencing

**R Code**:
```r
library(forecast)

set.seed(42)
# Random walk with ARMA errors
y <- cumsum(arima.sim(model = list(ar = 0.5, ma = 0.3), n = 200))

fit <- Arima(y, order = c(1, 1, 1))
summary(fit)
```

---

### Test 4: Forecasting

**R Code**:
```r
library(forecast)

set.seed(42)
y <- arima.sim(model = list(ar = 0.7), n = 100)

fit <- Arima(y, order = c(1, 0, 0))
fc <- forecast(fit, h = 10)

# Point forecasts
print(fc$mean)

# Prediction intervals
print(fc$lower)
print(fc$upper)
```

## Numerical Precision

| Model | Coefficient Precision | Forecast Precision |
|-------|----------------------|-------------------|
| AR(1) | < 0.05 | < 0.1 |
| ARMA(1,1) | < 0.08 | < 0.15 |
| ARIMA(1,1,1) | < 0.10 | < 0.2 |

## Running the Tests

```bash
cargo test -p p2a-core -- arima
```

## References

- Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Hyndman, R.J. & Khandakar, Y. (2008). "Automatic Time Series Forecasting: The forecast Package for R". *Journal of Statistical Software*, 27(3).
