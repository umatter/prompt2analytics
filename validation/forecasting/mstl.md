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

`augurs-mstl` computes in f32. `run_mstl` centres the series by its mean and scales it by its standard deviation
before the fit, restores the components afterwards (T = s T' + c, S_i = s S_i') and computes the residuals in f64 as
x - T - sum S, so the components carry about 1e-7 of the series' spread per operation whatever its level.
`test_mstl_level_invariance` holds that y and y + 1e8 decompose alike to 1e-5 of the spread and that the components
add back to the series. The components are not pinned against `forecast::mstl` (not R-pinned; the MCP registry entry
`mcp_t3_remaining_handlers` records it as deviation TS-26).

## Running the Tests

```bash
cargo test -p p2a-core -- mstl
```

## Input data (column types and missing values)

`run_mstl` reads its columns through the shared extractor with `NaPolicy::Fail`. Integer and Boolean columns are accepted (an `Int64` column gives the result of its `Float64` copy); a missing value (empty cell, NA or NaN) is an error naming the column and the count (`EconError::NullValues`); a text column is refused (`NonNumericColumn`) and an absent one is `ColumnNotFound`.

`forecast::mstl` interpolates missing values (`na.interp`) before it decomposes; p2a does not interpolate and refuses them, so fill the gaps first (for example `munge_fill_na`).

Test: `test_na_mstl_int64_and_missing` (on `validation/datasets/generated/arima_series.csv`, the dataset of the ARIMA case `arima_css_forecast`: columns `ai` (Int64), `a_na` and `ai_na` (two missing cells), `lab` (text)).

## Timing against v0.5.0

Quiet integration re-time, 2026-11-16: release builds of v0.5.0 (`f80b0131`) and of the integrated branch,
`--no-default-features --features ci`; one core (`taskset -c 2`), `RAYON_NUM_THREADS=1` (one thread for polars and
BLAS too), cwd `/dev/shm`; per row five processes per side, interleaved with the order alternating (nine for rows
within 1.05-1.15x), each the median of its calls after a warm-up; cells are the median over the processes, the ratio
of the minima in brackets. 1-minute load average 0.7-2.7 on 12 cores, nothing else building.

| Method | n | f80b0131 (µs) | this change (µs) | ratio [min] |
|---|---|---|---|---|
| `run_mstl`, periods [12] | 100 | 46.73 | 47.29 | 1.01 [1.01] |
| `run_mstl`, periods [12] | 1,000 | 407.9 | 413.1 | 1.01 [1.01] |
| `run_mstl`, periods [12] | 10,000 | 3,797 | 3,897 | 1.03 [1.03] |

The centring and rescaling add three O(n) passes around the f32 fit.

## References

- Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess". *Journal of Official Statistics*, 6(1), 3-73.
