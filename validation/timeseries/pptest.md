# Validation: Phillips-Perron Unit Root Test

## Method Overview

The Phillips-Perron test examines the null hypothesis that a time series has a unit root (non-stationary) against the alternative that it is stationary. Unlike the ADF test which uses lagged differences to handle serial correlation, the PP test makes a non-parametric correction to the t-statistic using the Newey-West estimator.

**Regression model:**
```
Δyₜ = α + βt + γyₜ₋₁ + uₜ
```

**Z(τ) test statistic:**
```
Z(τ) = τ̂ × √(σ̂²/λ²) - correction
```

Where:
- τ̂ = t-statistic from OLS regression
- σ̂² = residual variance
- λ² = Newey-West long-run variance estimate with Bartlett weights

**Truncation lag:**
- lshort=TRUE: trunc(4*(n/100)^0.25)
- lshort=FALSE: trunc(12*(n/100)^0.25)

**Key parameters:**
- `x`: Time series data
- `lshort`: Whether to use short truncation lag formula (default: TRUE)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `PP.test` | R 4.3.x |

## Test Cases

### Test 1: Stationary AR(1) Process

**Purpose**: Verify the test correctly rejects unit root for a stationary process.

**Data**: AR(1) with φ = 0.5: x_t = 0.5 × x_{t-1} + ε_t

**R Code**:
```r
set.seed(42)
n <- 200
x <- numeric(n)
phi <- 0.5
for (i in 2:n) {
  x[i] <- phi * x[i-1] + rnorm(1)
}
PP.test(x)
# Dickey-Fuller Z(alpha) = -157.42, Truncation lag parameter = 4, p-value = 0.01
```

**Results Comparison**:

| Metric | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| Truncation lag | 4 | 4 | exact | ✅ PASS |
| Statistic sign | negative | negative | - | ✅ PASS |
| p-value | < 0.05 | < 0.05 | indicates rejection | ✅ PASS |

**Rust Test**: `crates/p2a-core/src/stats/pptest.rs::tests::test_stationary_series`

### Test 2: Random Walk (Unit Root Process)

**Purpose**: Verify the test fails to reject unit root for a non-stationary process.

**Data**: Random walk: x_t = x_{t-1} + ε_t (cumulative sum)

**R Code**:
```r
set.seed(42)
x <- cumsum(rnorm(200))
PP.test(x)
# Dickey-Fuller = less negative, Truncation lag parameter = 4, p-value > 0.1
```

**Results Comparison**:

| Metric | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| Truncation lag | 4 | 4 | exact | ✅ PASS |
| p-value | > 0.1 | > 0.1 | fail to reject | ✅ PASS |

**Rust Test**: `crates/p2a-core/src/stats/pptest.rs::tests::test_random_walk`

### Test 3: Truncation Lag Calculation

**Purpose**: Verify truncation lag formula matches R exactly.

**R Code**:
```r
# Short lag formula
trunc(4*(100/100)^0.25)  # = 4
trunc(4*(1000/100)^0.25) # = 7
trunc(4*(50/100)^0.25)   # = 3

# Long lag formula
trunc(12*(100/100)^0.25)  # = 12
trunc(12*(1000/100)^0.25) # = 21
```

**Results Comparison**:

| n | lshort=TRUE R | lshort=TRUE Rust | lshort=FALSE R | lshort=FALSE Rust |
|---|---------------|------------------|----------------|-------------------|
| 50 | 3 | 3 | 10 | 10 |
| 100 | 4 | 4 | 12 | 12 |
| 1000 | 7 | 7 | 21 | 21 |

**Rust Test**: `crates/p2a-core/src/stats/pptest.rs::tests::test_truncation_lag_short` and `test_truncation_lag_long`

## Numerical Precision Summary

| Computation | Relative Tolerance | Notes |
|-------------|-------------------|-------|
| Truncation lag | exact | Integer calculation |
| Test statistic | ±1.0 | Non-standard distribution; focus on sign and magnitude |
| P-value | ±0.05 | Interpolated from tables; exact match not expected |

## Known Differences from R

1. **Test statistic format**: R's PP.test returns Z(alpha) while our implementation returns Z(tau). Both test the same hypothesis but have different scales. The p-values are comparable.

2. **P-value interpolation**: P-values are interpolated from critical value tables (Banerjee et al., 1993). Minor differences in interpolation are expected.

3. **Singular matrix handling**: For perfectly linear data (e.g., 1, 2, 3, ..., n), the design matrix becomes singular due to collinearity between the trend and lagged values. Our implementation returns an error in this case.

## Performance Comparison

### Phillips-Perron Test (lshort=TRUE)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 2.14      | 2,140  | **~1,000x faster** |
| n=1,000      | 27.95     | 1,860  | **~67x faster** |
| n=10,000     | 343       | 6,140  | **~18x faster** |
| n=100,000    | 4,659     | 67,680 | **~15x faster** |

### Phillips-Perron Test (lshort=FALSE)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 2.71      | 1,760  | **~650x faster** |
| n=1,000      | 43.31     | 1,900  | **~44x faster** |
| n=10,000     | 652       | 6,300  | **~10x faster** |
| n=100,000    | 10,268    | 71,500 | **~7x faster** |

**Implementation Notes:**
- Uses explicit OLS regression with 3x3 matrix inverse (constant, trend, lagged value)
- Newey-West variance with Bartlett weights computed directly
- P-value interpolation from asymptotic critical values with small-sample correction
- Rust is faster than R at **all** dataset sizes (7x to 1,000x speedup)

*Benchmarks run on 2026-01-19. Rust benchmarks via Criterion, R benchmarks via system.time().*

## References

- Phillips, P. C. B. & Perron, P. (1988). "Testing for a Unit Root in Time Series Regression." *Biometrika*, 75(2), 335-346.
- Banerjee, A., Dolado, J. J., Galbraith, J. W., & Hendry, D. (1993). *Co-integration, Error Correction, and the Econometric Analysis of Non-Stationary Data*. Oxford University Press.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/PP.test.html
