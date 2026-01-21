# Validation: Canonical Correlation Analysis (CCA)

## Method Overview

Canonical Correlation Analysis finds linear combinations of two sets of variables that have maximum correlation with each other. Given X (n × p) and Y (n × q), it computes:
- Canonical correlations ρ₁ ≥ ρ₂ ≥ ... ≥ ρᵣ where r = min(p, q)
- X coefficients (xcoef): p × r matrix for linear combinations of X
- Y coefficients (ycoef): q × r matrix for linear combinations of Y

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | cancor() | R 4.3+ |

## Algorithm

The implementation uses a numerically stable approach:

1. Center X and Y by subtracting column means
2. Compute covariance matrices: Σxx, Σyy, Σxy
3. Cholesky decompose: Lx = chol(Σxx), Ly = chol(Σyy)
4. Form: M = Lx⁻ᵀ Σxy Ly⁻¹
5. SVD: M = U Σ Vᵀ
6. Canonical correlations = singular values
7. Coefficients: xcoef = Lx⁻¹ U, ycoef = Ly⁻¹ V

## Test Cases

### Test 1: LifeCycleSavings Dataset (R classic example)

**R Code**:
```r
data(LifeCycleSavings)
pop <- LifeCycleSavings[, 2:3]  # pop15, pop75
oec <- LifeCycleSavings[, -(2:3)]  # sr, dpi, ddpi
result <- cancor(pop, oec)
print(result$cor)
```

**Expected Results**:
| Canonical Correlation | R Value | Tolerance |
|-----------------------|---------|-----------|
| ρ₁ | 0.8247... | 1e-4 |
| ρ₂ | 0.3652... | 1e-4 |

**Rust Test**: `crates/p2a-core/src/stats/cancor.rs::tests::test_validate_cancor_against_r`

### Test 2: Synthetic Correlated Data

**R Code**:
```r
set.seed(42)
n <- 100
z <- rnorm(n)
X <- cbind(z + rnorm(n, 0, 0.5), 0.8*z + rnorm(n, 0, 0.5), 0.5*z + rnorm(n, 0, 0.7))
Y <- cbind(0.9*z + rnorm(n, 0, 0.4), 0.7*z + rnorm(n, 0, 0.6))
result <- cancor(X, Y)
```

**Results Comparison**: Canonical correlations match within 1e-4 tolerance.

**Rust Test**: See unit tests in `cancor.rs`

### Test 3: Simple 2×2 Case

Tests with minimal dimensions to verify basic correctness.

## Numerical Precision Summary

| Statistic | Tolerance | Notes |
|-----------|-----------|-------|
| Canonical correlations | 1e-4 | May differ slightly due to Cholesky vs QR approach |
| Coefficients | 1e-3 | Sign may be flipped (both valid solutions) |
| Canonical scores | 1e-4 | Depends on coefficient accuracy |

## Known Differences

1. **Coefficient signs**: The sign of canonical coefficients may be flipped compared to R. This is mathematically equivalent since both (a, b) and (-a, -b) produce the same correlation.

2. **Near-singular matrices**: Our implementation uses Cholesky decomposition which may fail for singular covariance matrices. R's implementation uses QR decomposition which can handle some rank-deficient cases.

3. **Centering**: Our implementation centers data by default (xcenter=true, ycenter=true), matching R's default behavior.

## Performance Comparison

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 25        | 196    | **7.9x** ✅ |
| n=1,000      | 20,135    | 595    | 0.03x ⚠️ |
| n=10,000     | 126,680   | 2,312  | 0.02x ⚠️ |
| n=100,000    | 580,740   | 29,986 | 0.05x ⚠️ |

*Benchmarked 2026-01-20. Rust performs well for small samples but R's BLAS/LAPACK backend dominates for large n.*

### Performance Notes

For **small samples (n ≤ 100)**, the Rust implementation is significantly faster than R due to reduced overhead. This covers many practical use cases in canonical correlation analysis where sample sizes are typically small to moderate.

For **large samples (n > 1000)**, R's implementation benefits from highly optimized BLAS/LAPACK routines for the X'X matrix multiplication which dominates the computation time. The Rust implementation uses faer's matrix operations which, while correct, do not achieve the same level of optimization for this particular operation pattern.

**Recommendations:**
- For typical CCA use cases with n < 500, use the Rust implementation
- For large-scale analyses (n > 10,000), consider the performance tradeoff

## References

- Hotelling, H. (1936). "Relations Between Two Sets of Variates". *Biometrika*, 28(3/4), 321-377.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cancor.html
- Gundersen, G. (2018). "Canonical Correlation Analysis". https://gregorygundersen.com/blog/2018/07/17/cca/
