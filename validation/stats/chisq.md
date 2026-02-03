# Validation: Chi-Squared Tests

## Method Overview

This document validates the p2a-core implementation of Pearson's chi-squared tests against R's `stats::chisq.test`.

**Functions Implemented:**
- `chisq_test_gof()` - Chi-squared goodness-of-fit test
- `chisq_test_independence()` - Chi-squared test of independence
- `run_chisq_gof()` - Dataset wrapper for goodness-of-fit test
- `run_chisq_independence()` - Dataset wrapper for independence test

**Key Parameters:**
- `observed` - Observed frequency counts
- `probs` - Expected probabilities (optional, uniform if not specified)
- `table` - Contingency table for independence test
- `correct` - Apply Yates' continuity correction for 2×2 tables (default: true)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `chisq.test()` | R 4.3+ |
| scipy | Python | `scipy.stats.chisquare()`, `scipy.stats.chi2_contingency()` | 1.11+ |

## Mathematical Formulas

### Chi-Squared Test Statistic
```
χ² = Σ (O_i - E_i)² / E_i
```

### Goodness-of-Fit Test

Tests H₀: The population probabilities equal specified values p_i.
```
E_i = n × p_i
df = k - 1  (where k = number of categories)
```

### Test of Independence

Tests H₀: Row and column variables are independent.
```
E_ij = (row_i_total × col_j_total) / grand_total
df = (r - 1)(c - 1)  (where r = rows, c = columns)
```

### Yates' Continuity Correction (2×2 tables only)
```
χ² = Σ (|O_ij - E_ij| - 0.5)² / E_ij
```

### Pearson Residuals
```
r_ij = (O_ij - E_ij) / √E_ij
```

### Standardized Residuals (Independence Test)
```
std_r_ij = (O_ij - E_ij) / √(E_ij × (1 - p_i.) × (1 - p_.j))
```

## Test Cases

### Test 1: Goodness-of-Fit (Uniform Distribution)

**Data:**
```r
x <- c(89, 37, 30, 28, 2)
chisq.test(x)
```

**Results Comparison:**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| χ² statistic | 109.1075 | 109.11 | 1e-2 | ✅ |
| df | 4 | 4 | exact | ✅ |
| p-value | < 2.2e-16 | < 2.2e-16 | 1e-10 | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/chisq.rs::tests::test_validate_gof_against_r`

### Test 2: Goodness-of-Fit (Specified Probabilities)

**Data:**
```r
x <- c(10, 20, 30, 40)
probs <- c(0.1, 0.2, 0.3, 0.4)
chisq.test(x, p = probs)
```

**Results Comparison:**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| χ² statistic | 0.0 | 0 | 1e-10 | ✅ |
| df | 3 | 3 | exact | ✅ |
| p-value | 1.0 | 1 | 1e-10 | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/chisq.rs::tests::test_gof_with_probs`

### Test 3: Test of Independence (2×3 Table)

**Data:**
```r
M <- as.table(rbind(
  c(762, 327, 468),  # Female: Democrat, Independent, Republican
  c(484, 239, 477)   # Male: Democrat, Independent, Republican
))
chisq.test(M)
```

**Results Comparison:**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| χ² statistic | 30.07 | 30.07 | 1e-2 | ✅ |
| df | 2 | 2 | exact | ✅ |
| p-value | 2.954e-07 | 2.954e-07 | 1e-9 | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/chisq.rs::tests::test_validate_independence_against_r`

### Test 4: 2×2 Table with Yates' Correction

**Data:**
```r
M <- matrix(c(12, 7, 5, 16), nrow = 2, byrow = FALSE)
chisq.test(M, correct = TRUE)   # With Yates'
chisq.test(M, correct = FALSE)  # Without Yates'
```

**Results Comparison (With Yates'):**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| χ² statistic | 4.8123 | 4.8123 | 1e-2 | ✅ |
| df | 1 | 1 | exact | ✅ |
| p-value | 0.02826 | 0.02826 | 1e-4 | ✅ |

**Results Comparison (Without Yates'):**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| χ² statistic | 6.3199 | 6.3199 | 1e-2 | ✅ |
| df | 1 | 1 | exact | ✅ |
| p-value | 0.01194 | 0.01194 | 1e-4 | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/chisq.rs::tests::test_validate_2x2_yates_against_r`

### Test 5: Fair Die Test

**Data:**
```r
die_rolls <- c(16, 18, 22, 14, 15, 15)  # 100 rolls
chisq.test(die_rolls)
```

**Results Comparison:**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| χ² statistic | 2.6 | 2.6 | 1e-6 | ✅ |
| df | 5 | 5 | exact | ✅ |
| p-value | 0.7614 | 0.7614 | 1e-3 | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/chisq.rs::tests::test_gof_uniform`

## Numerical Precision Summary

| Statistic | Typical Tolerance | Notes |
|-----------|-------------------|-------|
| χ² statistic | 1e-2 | Small rounding differences |
| df | Exact | Integer values |
| p-value | 1e-4 to 1e-10 | Depends on tail probability |
| Pearson residuals | 1e-4 | √E term |
| Standardized residuals | 5e-2 | More complex formula |

## Known Differences

1. **Yates' Correction**: Only applied for 2×2 tables. R applies by default; our `correct` parameter defaults to `true` for consistency.

2. **Standardized Residuals**: R uses a more complex formula involving estimated marginal proportions. Our implementation may show small differences at higher decimal places.

3. **Warning for Small Expected Values**: R warns when expected values are < 5. Our implementation doesn't automatically warn but this is documented.

## Performance Comparison

### Goodness-of-Fit Test

| Categories (k) | Rust (µs) | R (µs) | Speedup |
|----------------|-----------|--------|---------|
| k=5 | 0.17 | 100 | ~588x |
| k=10 | 0.21 | 40 | ~190x |
| k=20 | 0.27 | 40 | ~148x |
| k=50 | 0.43 | 40 | ~93x |
| k=100 | 0.63 | 40 | ~63x |

### Test of Independence

| Table Size | Rust (µs) | R (µs) | Speedup |
|------------|-----------|--------|---------|
| 2×2 | 0.39 | 100 | ~256x |
| 3×3 | 0.65 | 80 | ~123x |
| 5×5 | 1.05 | 80 | ~76x |
| 10×10 | 2.80 | 80 | ~29x |
| 20×20 | 9.60 | 100 | ~10x |

### Yates' Correction (2×2 table)

| Correction | Rust (µs) | R (µs) | Speedup |
|------------|-----------|--------|---------|
| Without Yates | 0.39 | 60 | ~154x |
| With Yates | 0.29 | 100 | ~345x |

**Benchmark Notes:**
- Rust benchmarks: Criterion with 100 samples, median times reported
- R benchmarks: system.time() with 50 replications, mean times reported
- Environment: Rust release build, R 4.3+

**Performance Analysis:**
- Rust implementation is significantly faster across all test sizes (10x-588x speedup)
- The speedup is most pronounced for smaller tables and category counts
- For larger tables, R's vectorized C backend closes the gap somewhat
- Yates' correction in Rust is faster than without (uses simpler abs subtraction)

## References

- Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling". *Philosophical Magazine*, Series 5, 50(302), 157-175.
- Yates, F. (1934). "Contingency tables involving small numbers and the χ² test". *Supplement to the Journal of the Royal Statistical Society*, 1(2), 217-235.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/chisq.test.html
