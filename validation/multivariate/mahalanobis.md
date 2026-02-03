# Validation: Mahalanobis Distance

## Method Overview

The `mahalanobis()` function computes the squared Mahalanobis distance of observations from a center point, accounting for the covariance structure of the data.

**Key Features:**
- Measures multivariate distance accounting for correlations
- Useful for outlier detection
- Under multivariate normality, D² ~ χ²(p)
- Supports pre-computed inverse covariance for efficiency

**Mathematical Background:**

The squared Mahalanobis distance is defined as:

```
D² = (x - μ)' Σ⁻¹ (x - μ)
```

where:
- x is the observation vector
- μ is the center (typically the mean)
- Σ is the covariance matrix

When Σ = I (identity), Mahalanobis distance reduces to Euclidean distance.

**Implementation:**

The Rust implementation uses Cholesky decomposition for efficient computation:
1. Decompose Σ = LL' (Cholesky)
2. Solve Lz = (x - μ) via forward substitution
3. D² = z'z = ||z||²

This avoids explicit matrix inverse and is more numerically stable. Falls back to pseudoinverse for singular/near-singular covariance matrices.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | mahalanobis | R 4.3+ |

## Test Cases

### Test 1: Identity Covariance (Euclidean Distance)

**R Code:**
```r
x <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), ncol = 2, byrow = TRUE)
center <- c(0.5, 0.5)
cov_mat <- diag(2)
mahalanobis(x, center, cov_mat)
# Expected: [0.5, 0.5, 0.5, 0.5] (all equidistant from center)
```

**Rust Test:** `crates/p2a-core/src/stats/mahalanobis.rs::tests::test_mahalanobis_basic`

### Test 2: With Correlation

**Test:** Verify distances with correlated variables differ from uncorrelated case.

**Rust Test:** `crates/p2a-core/src/stats/mahalanobis.rs::tests::test_mahalanobis_with_correlation`

### Test 3: Auto-compute Center and Covariance

**Test:** Let function compute center and covariance from data.

**R Code:**
```r
x <- matrix(c(1, 2, 3, 5, 2, 4, 5, 3), ncol = 2, byrow = TRUE)
center <- colMeans(x)  # [2.75, 3.5]
cov_mat <- cov(x)
mahalanobis(x, center, cov_mat)
```

**Rust Test:** `crates/p2a-core/src/stats/mahalanobis.rs::tests::test_validate_mahalanobis_against_r`

### Test 4: Pre-inverted Covariance

**Test:** Verify same results when providing inverse covariance directly.

**Rust Test:** `crates/p2a-core/src/stats/mahalanobis.rs::tests::test_mahalanobis_inverted`

## Numerical Precision Summary

| Component | Tolerance | Notes |
|-----------|-----------|-------|
| Squared distances | 1e-10 | High precision |
| Center computation | 1e-10 | Mean computation |
| Covariance | 1e-10 | Sample covariance |

## Known Differences

1. **Covariance formula**: Both use n-1 denominator (sample covariance).

2. **Matrix inverse**: R uses `solve()`, Rust uses LU decomposition via faer.

3. **Singular matrices**: Both will fail on singular covariance matrices.

## Performance Comparison

### Sample Size Scaling (p=5)

| n | Rust (µs) | R (µs) | Speedup |
|---|-----------|--------|---------|
| 100 | 23 | 150 | 6.5x |
| 1,000 | 215 | 250 | 1.2x |
| 10,000 | 2,149 | 1,300 | 0.6x |
| 100,000 | 21,451 | 11,810 | 0.55x |

### Variable Scaling (n=1,000)

| p | Rust (µs) | R (µs) | Speedup |
|---|-----------|--------|---------|
| 2 | 180 | 100 | 0.56x |
| 5 | 210 | 220 | 1.05x |
| 10 | 321 | 410 | 1.28x |
| 20 | 568 | 940 | 1.65x |
| 50 | 1,590 | 3,900 | 2.45x |

**Notes:**
- Rust benchmarks from Criterion (median), R from microbenchmark (mean of 100 iterations)
- Rust uses Cholesky decomposition + forward substitution (optimized Jan 2026)
- R uses highly optimized BLAS routines via `solve()` for matrix operations
- **Rust excels for high-dimensional data (p >= 10)**: 1.3-2.5x faster
- **R excels for large n with small p**: 1.5-2x faster due to BLAS vectorization
- For typical use cases (n < 10,000, p >= 5), Rust performance is comparable or better
- Cholesky optimization improved high-p performance by ~90% (p=50: 15ms → 1.6ms)

## References

- Mahalanobis, P. C. (1936). "On the generalized distance in statistics".
  Proceedings of the National Institute of Sciences (Calcutta), 2, 49–55.
- R Core Team. `stats::mahalanobis()` function.
  https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mahalanobis.html
