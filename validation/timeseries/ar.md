# Validation: Autoregressive Model Fitting (ar)

## Method Overview

The `ar()` function fits autoregressive models to time series data with automatic order selection via AIC. It supports multiple estimation methods and returns coefficients, prediction variance, and diagnostic information.

**Key Features:**
- Automatic order selection using AIC
- Multiple fitting methods: Yule-Walker (default), Burg, OLS
- Returns partial autocorrelations as byproduct
- Computes residuals for model diagnostics

**Mathematical Background:**

The AR(p) model:
```
x_t - μ = φ₁(x_{t-1} - μ) + φ₂(x_{t-2} - μ) + ... + φₚ(x_{t-p} - μ) + ε_t
```

where ε_t ~ WN(0, σ²).

**Estimation Methods:**

1. **Yule-Walker**: Solves Yule-Walker equations via Durbin-Levinson algorithm
2. **Burg**: Minimizes forward and backward prediction error
3. **OLS**: Ordinary least squares regression on lagged values

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | ar | R 4.3+ |

## Test Cases

### Test 1: AR(2) Process with AIC Selection

**R Code:**
```r
set.seed(42)
x <- arima.sim(n=100, model=list(ar=c(0.7, -0.2)))
result <- ar(x, method="yule-walker")
result$order  # Order selected
result$ar     # Coefficients
```

**Expected Behavior:**
- AIC should select an order close to the true order (2)
- Coefficients should be close to true values (0.7, -0.2)

**Rust Test:** `crates/p2a-core/src/forecasting/ar.rs::tests::test_validate_ar_yule_walker_against_r`

### Test 2: Fixed Order Fitting

**Test:** Fit AR(3) with fixed order (no AIC selection).

**Rust Test:** `crates/p2a-core/src/forecasting/ar.rs::tests::test_ar_fixed_order`

### Test 3: Method Comparison

**Test:** Compare Yule-Walker, Burg, and OLS methods on same data.

**Expected:** All methods should produce reasonable coefficients, with slight differences.

**Rust Test:** `crates/p2a-core/src/forecasting/ar.rs::tests::test_ar_methods`

## Numerical Precision Summary

| Component | Tolerance | Notes |
|-----------|-----------|-------|
| AR coefficients | 0.1 | Estimation variability |
| Prediction variance | 0.1 | Scale-dependent |
| Order selection | exact or ±1 | AIC-based |

## Known Differences

1. **AIC formula**: R uses `n * log(var) + 2*(p+1)`. Our implementation uses the same formula.

2. **Default max order**: R uses `min(n-1, 10*log10(n))`. We match this default.

3. **OLS effective sample size**: OLS method uses `n-p` observations after lagging, which affects variance estimation slightly.

## Performance Comparison

| Series Length | Method | Rust (µs) | R (µs) | Speedup |
|--------------|--------|-----------|--------|---------|
| n=50 | Yule-Walker | 10.8 | 600 | ~56x |
| n=100 | Yule-Walker | 17.6 | 520 | ~30x |
| n=500 | Yule-Walker | 67.0 | 740 | ~11x |
| n=1000 | Yule-Walker | 148.2 | 760 | ~5x |
| n=50 | Burg | ~12 | 390 | ~32x |
| n=100 | Burg | ~20 | 380 | ~19x |
| n=500 | Burg | ~70 | 470 | ~7x |
| n=1000 | Burg | ~150 | 750 | ~5x |
| n=50 | OLS | ~20 | 3970 | ~199x |
| n=100 | OLS | ~35 | 5300 | ~151x |

**Notes:**
- Rust consistently outperforms R by 5-200x depending on method and series length
- Yule-Walker and Burg have similar performance in Rust
- OLS in R is significantly slower due to matrix operations overhead
- Rust benchmarks from Criterion (median); R benchmarks from system.time (mean of 100 iterations)

## References

- Brockwell, P. J. & Davis, R. A. (1991). "Time Series: Theory and Methods". Springer.
- Burg, J. P. (1967). "Maximum Entropy Spectral Analysis". 37th Meeting of the Society of Exploration Geophysicists.
- R Core Team. `stats::ar()` function. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ar.html
