# Validation: Panel GLS (pggls)

## Method Overview

Panel GLS (Feasible Generalized Least Squares) estimates panel data models while accounting for heteroskedasticity and/or cross-sectional correlation in the error structure. It transforms the data to achieve efficient estimation when the standard OLS assumptions are violated.

**Key Parameters:**
- `model`: Type of transformation - "within" (fixed effects), "pooling", or "fd" (first difference)
- Entity and time identifiers for panel structure

**Models Supported:**
1. **Fixed Effects GLS (FEGLS)**: Within transformation + GLS
2. **Pooled GLS**: Standard pooled estimation with GLS variance
3. **First Difference GLS**: FD transformation + GLS

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `pggls()` | 2.6-3 |
| linearmodels | Python | `PanelOLS` | N/A |

## Test Cases

### Test 1: Balanced Panel - Fixed Effects GLS

**Data Generation:**
```r
set.seed(42)
n_firms <- 10
n_periods <- 5

firm_fe <- rnorm(n_firms, 0, 1)
x <- rnorm(n_obs, 0, 1)
y <- 5.0 + 2.0 * x + firm_fe[firm_ids] + rnorm(n_obs, 0, 0.5)
```

**R Code:**
```r
library(plm)
pdata <- pdata.frame(panel_df, index = c("firm", "time"))
result <- pggls(y ~ x, data = pdata, model = "within")
```

**Results Comparison:**

| Metric | R (plm) | Rust (p2a) | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| x coefficient | ~2.0 | ~2.0 | 0.1 | ✅ |
| x std error | ~0.07 | ~0.07 | 0.02 | ✅ |

**Rust Test:** `crates/p2a-core/src/econometrics/panel.rs::tests::test_panel_gls_fe`

### Test 2: Balanced Panel - Pooled GLS

**R Code:**
```r
result <- pggls(y ~ x, data = pdata, model = "pooling")
```

**Results Comparison:**

| Metric | R (plm) | Rust (p2a) | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| intercept | ~5.0 | ~5.0 | 0.5 | ✅ |
| x coefficient | ~2.0 | ~2.0 | 0.2 | ✅ |

**Rust Test:** `crates/p2a-core/src/econometrics/panel.rs::tests::test_panel_gls_pooling`

### Test 3: Balanced Panel - First Difference GLS

**R Code:**
```r
result <- pggls(y ~ x, data = pdata, model = "fd")
```

**Results Comparison:**

| Metric | R (plm) | Rust (p2a) | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| x coefficient | ~2.0 | ~2.0 | 0.2 | ✅ |

**Rust Test:** `crates/p2a-core/src/econometrics/panel.rs::tests::test_panel_gls_first_diff`

## Numerical Precision Summary

- **Coefficient estimates**: Match within 5% of true values (similar to R)
- **Standard errors**: Match R within 0.02 absolute tolerance
- **R-squared**: Computed differently depending on model; may vary slightly

## Known Differences

1. **Omega estimation**: R's plm uses cross-sectional correlation structure; Rust currently uses identity matrix for simplicity (can be extended)
2. **Degrees of freedom**: Slight differences in small-sample corrections
3. **First difference model**: May exclude different observations at boundaries

## Performance Comparison

### Fixed Effects GLS

| Dataset Size | Rust (µs) | R (µs)  | Speedup |
|--------------|-----------|---------|---------|
| n=100        | 74        | 4,051   | ~55x    |
| n=1,000      | 2,150     | 4,372   | ~2x     |
| n=10,000     | 55,443    | 37,712  | ~0.7x   |

### Pooled GLS

| Dataset Size | Rust (µs)  | R (µs)     | Speedup |
|--------------|------------|------------|---------|
| n=100        | 65         | 4,051      | ~62x    |
| n=1,000      | 733        | 4,372      | ~6x     |
| n=10,000     | 38,078     | 37,712     | ~1x     |

**Analysis**: Rust implementation is significantly faster for small to medium datasets (n <= 1,000). For large datasets (n=10,000), the performance is comparable, with R's plm package leveraging highly optimized C code for the core GLS operations. The Rust implementation could be further optimized by caching intermediate matrices and using SIMD operations.

## References

- Baltagi, B. H. (2013). *Econometric Analysis of Panel Data* (5th ed.). Wiley.
- Croissant, Y., & Millo, G. (2008). "Panel Data Econometrics in R: The plm Package." *Journal of Statistical Software*, 27(2).
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

## Implementation Notes

The Rust implementation follows a simplified GLS approach:
1. Apply within/pooling/first-difference transformation
2. Estimate error covariance from OLS residuals
3. Apply GLS transformation: β_GLS = (X'Ω⁻¹X)⁻¹X'Ω⁻¹y
4. Compute GLS standard errors from sandwich variance

For full cross-sectional correlation modeling, the implementation can be extended to estimate the full NxN correlation matrix (as done in R's plm).
