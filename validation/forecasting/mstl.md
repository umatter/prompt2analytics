# Validation: MSTL Decomposition

## Method Overview

MSTL (Multiple Seasonal-Trend decomposition using LOESS) decomposes time series with multiple seasonal patterns.

**Decomposition**:
```
Y_t = T_t + S₁_t + S₂_t + ... + R_t
```

where T is trend, S_i are seasonal components, and R is remainder.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| forecast | R | `mstl()` | 8.21-x |

## Test Cases

### Test 1: Monthly Data with Annual Seasonality

**R Code**:
```r
library(forecast)

# AirPassengers dataset (monthly, period = 12)
data(AirPassengers)
decomp <- mstl(AirPassengers)
plot(decomp)

# Extract components
trend <- decomp[, "Trend"]
seasonal <- decomp[, "Seasonal12"]
remainder <- decomp[, "Remainder"]
```

**Validation Criteria**:
- Trend captures long-term movement
- Seasonal pattern repeats with period 12
- Remainder is stationary

---

### Test 2: Multiple Seasonality

**R Code**:
```r
library(forecast)

# Create data with daily and weekly seasonality
set.seed(42)
n <- 365 * 2
t <- 1:n

# Daily (period 7) and annual (period 365) patterns
daily <- sin(2 * pi * t / 7)
annual <- sin(2 * pi * t / 365)
trend <- 0.01 * t
y <- trend + 2*daily + 3*annual + rnorm(n, 0, 0.5)

y_ts <- ts(y, frequency = 7)
decomp <- mstl(y_ts, s.window = "periodic")
```

## Numerical Precision

Trend and seasonal components should match R within 1e-4.

## Running the Tests

```bash
cargo test -p p2a-core -- mstl
```

## References

- Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess". *Journal of Official Statistics*, 6(1), 3-73.
