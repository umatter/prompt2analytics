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

## Input data (column types and missing values)

`run_mstl` reads its columns through the shared extractor with `NaPolicy::Fail`. Integer and Boolean columns are accepted (an `Int64` column gives the result of its `Float64` copy); a missing value (empty cell, NA or NaN) is an error naming the column and the count (`EconError::NullValues`); a text column is refused (`NonNumericColumn`) and an absent one is `ColumnNotFound`.

`forecast::mstl` interpolates missing values (`na.interp`) before it decomposes; p2a does not interpolate and refuses them, so fill the gaps first (for example `munge_fill_na`).

Test: `test_na_mstl_int64_and_missing` (on `validation/datasets/generated/arima_series.csv`, the dataset of the ARIMA case `arima_css_forecast`: columns `ai` (Int64), `a_na` and `ai_na` (two missing cells), `lab` (text)).

## References

- Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess". *Journal of Official Statistics*, 6(1), 3-73.
