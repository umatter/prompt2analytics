# Validation: Holt-Winters Exponential Smoothing

## Method Overview

Holt-Winters exponential smoothing (triple exponential smoothing) is a forecasting method for time series data with trend and seasonality. It uses three smoothing equations:

1. **Level**: a[t] = α(Y[t] - s[t-p]) + (1-α)(a[t-1] + b[t-1])  (additive)
2. **Trend**: b[t] = β(a[t] - a[t-1]) + (1-β)b[t-1]
3. **Seasonal**: s[t] = γ(Y[t] - a[t]) + (1-γ)s[t-p]  (additive)

For multiplicative seasonality, the seasonal component multiplies rather than adds.

### Key Parameters
- **alpha (α)**: Level smoothing parameter (0-1)
- **beta (β)**: Trend smoothing parameter (0-1)
- **gamma (γ)**: Seasonal smoothing parameter (0-1)
- **period**: Seasonal period (e.g., 12 for monthly data with yearly seasonality)
- **seasonal**: "additive" or "multiplicative"

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | HoltWinters() | R 4.x |

### R HoltWinters() Details
- Uses optimization to find α, β, γ that minimize SSE
- Supports both additive and multiplicative seasonality
- Initialization uses decomposition-based method
- Returns fitted values, residuals, coefficients, and SSE

## Test Cases

### Test 1: AirPassengers Subset - Fixed Parameters (n=24, period=12)

**Data**: First 24 months of AirPassengers dataset

**R Code**:
```r
y <- c(112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
       115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140)
hw <- HoltWinters(ts(y, frequency=12), alpha=0.2, beta=0.1, gamma=0.3, seasonal="additive")
```

**Results Comparison**:
| Metric | R HoltWinters | Rust p2a | Tolerance | Status |
|--------|---------------|----------|-----------|--------|
| SSE | 369.5158 | 369.5158 | 0.01 | ✅ PASS |
| Final level (a) | 147.9523 | 147.9523 | 0.01 | ✅ PASS |
| Final trend (b) | 1.66295 | 1.66295 | 0.01 | ✅ PASS |

**Rust Test**: `crates/p2a-core/src/forecasting/holtwinters.rs::tests::test_validate_holt_winters_against_r`

### Test 2: AirPassengers - Optimized Additive Seasonality

**Purpose**: Verify optimized parameters on additive seasonal model.

**R Code**:
```r
hw_add <- HoltWinters(ts(y, frequency = 12), seasonal = "additive")
```

**Notes**: Optimization may converge to different local optima between R and Rust due to different algorithms (R: L-BFGS-B, Rust: Nelder-Mead). The key is that both produce reasonable SSE values.

### Test 3: Optimized Multiplicative Seasonality

**Purpose**: Verify multiplicative seasonal model.

**R Code**:
```r
hw_mult <- HoltWinters(ts(y, frequency = 12), seasonal = "multiplicative")
```

### Test 4: Non-Seasonal Model (Holt's Linear)

**Purpose**: Verify Holt's linear method (no seasonality, gamma=FALSE).

**Rust Test**: `test_holt_linear`

## Numerical Precision Summary

| Component | Expected Tolerance | Notes |
|-----------|-------------------|-------|
| Smoothing parameters (α,β,γ) | 0.01 | Optimization may find different local optima |
| SSE | 5% | Different initialization can affect SSE |
| Fitted values | 1% relative | |
| Forecasts | 2% relative | Error compounds with forecast horizon |

## Known Differences

1. **Initialization Method**: Our implementation now exactly matches R's initialization:
   - Decomposes the first `start_periods * period` observations using centered moving average
   - Fits linear regression on the extracted trend component (excluding NAs at edges)
   - Uses regression intercept as `l.start` and slope as `b.start`
   - Seasonal indices from decomposition, normalized to sum to 0 (additive) or average to 1 (multiplicative)

2. **Optimization Algorithm**: R uses `optim()` with L-BFGS-B. Our implementation uses Nelder-Mead simplex. With fixed parameters, results are **identical**. With optimization, both algorithms typically find similar local optima but may differ slightly due to:
   - Different convergence criteria
   - Different starting simplex configurations
   - Numerical precision in gradient-free vs gradient-based methods

3. **Output Format**: R returns only `n-period` fitted values for seasonal models. Our implementation returns all `n` values with `NaN` for the first `period` observations (used for initialization).

## Performance Comparison

### With Parameter Optimization (Nelder-Mead)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=48 | 51 | 7,020 | **~138x** |
| n=120 | 665 | 12,380 | **~19x** |
| n=240 | 1,082 | 7,460 | **~7x** |
| n=480 | 1,226 | 8,720 | **~7x** |

### With Fixed Parameters (no optimization)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=48 | 2.7 | 3,900 | **~1,444x** |
| n=120 | 6.4 | 3,760 | **~588x** |
| n=240 | 12.7 | 4,140 | **~326x** |
| n=480 | 24.9 | 4,720 | **~190x** |

### Performance Notes

1. **Fixed parameters**: Rust is dramatically faster (100-1000x) when parameters are pre-specified because it only needs to run the filtering algorithm without optimization.

2. **Parameter optimization**: Rust uses Nelder-Mead simplex optimization which:
   - Converges in ~50-150 function evaluations (vs ~200+ for coordinate descent)
   - Scales much better with dataset size
   - Is now **7-138x faster than R** across all tested sizes

3. **Optimization algorithm comparison**:
   - Old approach (coordinate descent): O(n × iterations × 75 evaluations)
   - New approach (Nelder-Mead): O(n × 50-150 evaluations)
   - R's L-BFGS-B: efficient but has higher per-iteration overhead

*Benchmarks run on: Linux 6.17, Intel CPU*
*R version: 4.x with stats package*
*Rust: release build with Nelder-Mead optimization*

## References

- Holt, C. C. (1957). "Forecasting Trends and Seasonal by Exponentially Weighted Averages". ONR Memorandum 52/1957, Carnegie Institute of Technology.
- Holt, C. C. (2004). "Forecasting seasonals and trends by exponentially weighted moving averages". International Journal of Forecasting, 20(1), 5-10. (Reprint of 1957 paper)
- Winters, P. R. (1960). "Forecasting Sales by Exponentially Weighted Moving Averages". Management Science, 6(3), 324-342.
- Hyndman, R. J. & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice" (3rd ed). OTexts. https://otexts.com/fpp3/
- R Documentation: `stats::HoltWinters()` - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/HoltWinters.html
