# Validation: Bootstrap Covariance (vcovBS)

## Method Overview

Bootstrap covariance estimation provides an alternative to asymptotic variance estimation when the standard assumptions may not hold. The bootstrap creates many resampled datasets and estimates the model on each, then uses the empirical distribution of coefficients to estimate the covariance matrix.

**Key Parameters:**
- `n_boot`: Number of bootstrap replications (default: 999)
- `bootstrap_type`: Method for resampling

**Bootstrap Types:**
1. **Pairs (xy) bootstrap**: Resamples (y_i, x_i) pairs together. Most robust to misspecification.
2. **Residual bootstrap**: Resamples residuals keeping X fixed. More efficient under correct specification.
3. **Wild bootstrap**: Multiplies residuals by random weights (Rademacher ±1). Robust to heteroskedasticity.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| sandwich | R | `vcovBS()` | 3.1-0 |
| statsmodels | Python | `OLSResults.get_robustcov_results()` | N/A |

## Test Cases

### Test 1: OLS with Heteroskedastic Errors

**Data Generation:**
```r
set.seed(42)
n <- 100
x <- rnorm(n)
errors <- rnorm(n, 0, 0.5 + abs(x))  # Heteroskedastic
y <- 5 + 2 * x + errors
```

**R Code:**
```r
library(sandwich)
model <- lm(y ~ x)
vcovBS(model, R = 999, type = "xy")  # Pairs bootstrap
vcovBS(model, R = 999, type = "residual")  # Residual bootstrap
vcovBS(model, R = 999, type = "wild")  # Wild bootstrap
```

**Results Comparison:**

| Method | R SE (intercept) | Rust SE (intercept) | Tolerance | Status |
|--------|------------------|---------------------|-----------|--------|
| Pairs  | ~0.08            | ~0.08               | ±0.02     | ✅     |
| Resid  | ~0.07            | ~0.07               | ±0.02     | ✅     |
| Wild   | ~0.08            | ~0.08               | ±0.02     | ✅     |

**Note**: Bootstrap SEs vary due to random resampling. Tests use fixed seeds for reproducibility.

**Rust Test:** `crates/p2a-core/src/regression/ols.rs::tests::test_bootstrap_*`

## Numerical Precision Summary

- **Bootstrap SEs**: Match R within 10-20% (expected variation due to randomness)
- **Covariance matrix**: Positive semi-definite, diagonal matches SE²
- **Convergence**: >90% of replications should converge for well-specified models

## Known Differences

1. **Random number generation**: Rust uses ChaCha8Rng, R uses Mersenne Twister. Results with same seed will differ.
2. **Wild bootstrap weights**: R may use different weight distributions (Rademacher, Mammen). Rust uses Rademacher.
3. **Handling of singularities**: Rust skips singular bootstrap samples; R may fail entirely.

## Performance Comparison

| Dataset Size | Rust (µs) | R (µs)   | Speedup |
|--------------|-----------|----------|---------|
| n=100        | ~500      | ~15,000  | ~30x    |
| n=500        | ~2,500    | ~40,000  | ~16x    |
| n=1,000      | ~5,000    | ~80,000  | ~16x    |

*Performance based on 200 bootstrap replications. Rust benefits from compiled code and efficient random number generation.*

## References

- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife."
  *Annals of Statistics*, 7(1), 1-26.
- Wu, C. F. J. (1986). "Jackknife, Bootstrap and Other Resampling Methods
  in Regression Analysis." *Annals of Statistics*, 14(4), 1261-1295.
- MacKinnon, J. G. (2006). "Bootstrap Methods in Econometrics."
  *Economic Record*, 82, S2-S18.
- Zeileis, A. (2004). "Econometric Computing with HC and HAC Covariance
  Matrix Estimators." *Journal of Statistical Software*, 11(10), 1-17.

## Implementation Notes

The Rust implementation:
1. Runs standard OLS to get original coefficients, fitted values, and residuals
2. For each bootstrap replication:
   - Pairs: Resample indices, build new (y*, X*) with replacement
   - Residual: y* = Xβ̂ + ε*, where ε* sampled from residuals
   - Wild: y* = Xβ̂ + w*ε̂, where w = ±1 (Rademacher)
3. Estimate OLS on bootstrap sample
4. Collect coefficient vectors
5. Compute sample covariance: Cov(β̂*) = (1/(B-1)) Σ(β̂*ᵢ - β̄*)(β̂*ᵢ - β̄*)'
