# Validation: Fisher's Exact Test

## Method Overview

This document validates the p2a-core implementation of Fisher's exact test against R's `stats::fisher.test`.

**Functions Implemented:**
- `fisher_exact_test()` - Fisher's exact test for 2×2 tables (f64 input)
- `fisher_exact_test_int()` - Convenience wrapper for integer input
- `run_fisher_test()` - Dataset wrapper for two categorical columns

**Key Parameters:**
- `table` - 2×2 contingency table [[a, b], [c, d]]
- `alternative` - TwoSided (default), Greater, or Less
- `conf_level` - Confidence level for odds ratio CI (optional)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `fisher.test()` | R 4.3+ |
| scipy | Python | `scipy.stats.fisher_exact()` | 1.11+ |

## Mathematical Formulas

### Hypergeometric Distribution

For a 2×2 table:
```
         | Col 1 | Col 2 | Total
---------+-------+-------+-------
Row 1    |   a   |   b   | a + b
Row 2    |   c   |   d   | c + d
---------+-------+-------+-------
Total    | a + c | b + d |   n
```

The probability of observing exactly `a` in the top-left cell (given fixed marginals):
```
P(X = a) = C(a+b, a) × C(c+d, c) / C(n, a+c)
         = (a+b)! × (c+d)! × (a+c)! × (b+d)! / (a! × b! × c! × d! × n!)
```

### Sample Odds Ratio
```
OR = (a × d) / (b × c)
```

### P-Value Calculation

- **Two-sided**: Sum probabilities of all tables with P(X=x) ≤ P(X=observed)
- **Greater**: P(X ≥ a) = 1 - CDF(a-1)
- **Less**: P(X ≤ a) = CDF(a)

## Test Cases

### Test 1: Lady Tasting Tea (Classic Example)

**Data:**
```r
M <- matrix(c(3, 1, 1, 3), nrow = 2, byrow = TRUE)
fisher.test(M)
```

**Results Comparison:**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| p-value | 0.4857 | 0.4857 | 1e-4 | ✅ |
| sample OR | 9.0 | (CML: 6.408) | exact | ✅ |

**Note:** R reports CML (Conditional Maximum Likelihood) odds ratio estimate; we report sample OR.

**Rust Test:** `crates/p2a-core/src/stats/fisher.rs::tests::test_validate_against_r_lady_tea`

### Test 2: Significant Association

**Data:**
```r
M <- matrix(c(1, 11, 9, 3), nrow = 2, byrow = TRUE)
fisher.test(M)
```

**Results Comparison:**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| p-value | 0.002759 | 0.002759 | 1e-4 | ✅ |
| sample OR | 0.0303 | (CML: 0.0372) | 1e-4 | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/fisher.rs::tests::test_validate_against_r_basic`

### Test 3: One-Sided Tests

**Data:**
```r
M <- matrix(c(6, 2, 1, 7), nrow = 2, byrow = TRUE)
fisher.test(M, alternative = "greater")
fisher.test(M, alternative = "less")
fisher.test(M, alternative = "two.sided")
```

**Results Comparison:**

| Alternative | Rust (p2a) | R | Tolerance | Status |
|-------------|------------|---|-----------|--------|
| greater | 0.02028 | 0.02028 | 5e-3 | ✅ |
| less | 0.9993 | 0.9993 | 1e-2 | ✅ |
| two.sided | 0.04056 | 0.04056 | 5e-3 | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/fisher.rs::tests::test_validate_against_r_one_sided`

### Test 4: Zero Cell

**Data:**
```r
M <- matrix(c(0, 5, 5, 10), nrow = 2, byrow = TRUE)
fisher.test(M)
```

**Results Comparison:**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| p-value | 0.2663 | 0.2663 | 1e-4 | ✅ |
| sample OR | 0.0 | (CML: 0.0) | exact | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/fisher.rs::tests::test_zero_cells`

### Test 5: Extreme Table (High Significance)

**Data:**
```r
M <- matrix(c(50, 1, 1, 50), nrow = 2, byrow = TRUE)
fisher.test(M)
```

**Results Comparison:**

| Statistic | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| p-value | <1e-20 | 1.30e-26 | 1e-15 | ✅ |
| sample OR | 2500.0 | (CML: 1530.9) | exact | ✅ |

**Rust Test:** `crates/p2a-core/src/stats/fisher.rs::tests::test_extreme_table`

## Numerical Precision Summary

| Statistic | Typical Tolerance | Notes |
|-----------|-------------------|-------|
| p-value | 1e-4 to 1e-10 | Depends on table extremity |
| sample odds ratio | exact | Integer arithmetic |

## Known Differences

1. **Odds Ratio Estimation**:
   - R's `fisher.test()` returns the Conditional Maximum Likelihood (CML) estimate
   - Our implementation returns the sample odds ratio: `(a×d)/(b×c)`
   - Both are valid; CML is asymptotically more efficient but requires iterative computation

2. **Confidence Intervals**:
   - R uses exact Cornfield intervals via CML
   - Our implementation uses binary search with the non-central hypergeometric distribution
   - May show small differences at boundary cases

3. **r×c Tables**:
   - R supports arbitrary r×c contingency tables using network algorithms
   - Our implementation currently only supports 2×2 tables (use chi-squared test for larger tables)

## Performance Comparison

### Fisher's Exact Test (Two-Sided, no CI)

| Total Count | Rust (µs) | R (µs) | Speedup |
|-------------|-----------|--------|---------|
| n=20        | 1.78      | 580    | ~326x   |
| n=100       | 7.43      | 600    | ~81x    |
| n=500       | 80.1      | 1360   | ~17x    |
| n=1000      | 182.5     | 1900   | ~10x    |

### Fisher's Exact Test with 95% Confidence Interval

| Total Count | Rust (µs) | R (µs) | Speedup |
|-------------|-----------|--------|---------|
| n=20        | 185       | 660    | ~3.6x   |
| n=100       | 968       | 640    | 0.7x    |
| n=500       | 5138      | 920    | 0.2x    |

### Alternative Hypotheses (n=100)

| Alternative | Rust (µs) | R (µs) | Speedup |
|-------------|-----------|--------|---------|
| two.sided   | 7.41      | 740    | ~100x   |
| greater     | 1.60      | 500    | ~313x   |
| less        | 2.48      | 480    | ~194x   |

**Benchmark Notes:**
- Rust benchmarks: Criterion with 100 samples, median times reported
- R benchmarks: system.time() with 50 replications, mean times reported
- Environment: Rust release build, R 4.3+

**Performance Analysis:**
- For basic p-value computation (no CI), Rust is 10-326x faster depending on table size
- One-sided tests are significantly faster than two-sided (no need to enumerate all tables)
- CI computation is slower in Rust due to binary search with non-central hypergeometric
- R's exact CI method (Cornfield) is more optimized for this specific calculation
- For p-values only (the most common use case), Rust provides substantial speedups

## References

- Fisher, R. A. (1935). "The logic of inductive inference". *Journal of the Royal Statistical Society*, 98(1), 39-82.
- Fisher, R. A. (1922). "On the interpretation of χ² from contingency tables, and the calculation of P". *Journal of the Royal Statistical Society*, 85(1), 87-94.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html
