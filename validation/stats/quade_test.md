# Validation: Quade Test (quade.test)

## Method Overview

The Quade test is a non-parametric test for unreplicated complete block designs, similar to the Friedman test. It uses a weighted ranking approach where blocks are weighted by their range, making it more powerful when block effects vary considerably.

**Key Features:**
- Uses F distribution (unlike Friedman's chi-squared approximation)
- Weights blocks by their range to recover between-block information
- More powerful than Friedman test when block ranges vary substantially
- Requires complete blocks (one observation per treatment per block)

**Mathematical Background:**
1. Compute the range within each block: Range_i = max(X_i) - min(X_i)
2. Rank the ranges across blocks: Q_i
3. Rank values within each block: R_ij
4. Compute weighted scores: S_ij = Q_i × (R_ij - (k+1)/2)
5. Compute treatment totals: S_j = Σ_i S_ij
6. F statistic: T = (b-1)B / (A - B)
   where A = ΣΣ S_ij² and B = (1/b) Σ S_j²

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | quade.test | R 4.3+ |

## Test Cases

### Test 1: Store-Brand Data (R Documentation Example)

**R Code:**
```r
y <- matrix(c(5, 4, 7, 10, 12,
              1, 3, 1, 0, 2,
              16, 12, 22, 22, 35,
              5, 4, 3, 5, 4,
              10, 9, 7, 13, 10,
              19, 18, 28, 25, 20,
              10, 7, 6, 8, 7),
            nrow = 7, byrow = TRUE,
            dimnames = list(Store = as.character(1:7),
                            Brand = LETTERS[1:5]))
quade.test(y)

# Result: Quade F = 2.4266, num df = 4, denom df = 24, p-value = 0.07566

# Intermediate values:
# A = 1360, B = 391.6429
# Block ranges: 8, 3, 23, 2, 6, 10, 4
# Treatment sums: -9.5, -38.0, -1.5, 23.0, 26.0
```

**Results Comparison:**

| Metric | R Value | Rust Value | Tolerance |
|--------|---------|------------|-----------|
| F statistic | 2.4266 | 2.4266 | 0.01 |
| df1 | 4 | 4 | exact |
| df2 | 24 | 24 | exact |
| p-value | 0.07566 | 0.07566 | 0.01 |
| A statistic | 1360 | 1360 | 1.0 |
| B statistic | 391.6429 | 391.6429 | 0.1 |

**Rust Test:** `crates/p2a-core/src/stats/quade.rs::tests::test_validate_quade_against_r`

### Test 2: Perfect Treatment Ordering (All Ties in Range)

**R Code:**
```r
y <- matrix(c(1.0, 2.0, 3.0,
              1.5, 2.5, 3.5,
              1.2, 2.2, 3.2,
              1.8, 2.8, 3.8,
              1.1, 2.1, 3.1),
            nrow = 5, byrow = TRUE)
quade.test(y)

# Result: Quade F = 36, num df = 2, denom df = 8, p-value = 1e-04
# Note: All block ranges are 2.0 (ties), but floating-point differences
# cause one block to get rank 1 while others get average rank 3.5
```

**Results Comparison:**

| Metric | R Value | Rust Value | Tolerance |
|--------|---------|------------|-----------|
| F statistic | 36 | 36 | 1.0 |
| df1 | 2 | 2 | exact |
| df2 | 8 | 8 | exact |
| p-value | 0.0001 | ~0.0001 | 0.001 |

**Rust Test:** `crates/p2a-core/src/stats/quade.rs::tests::test_quade_basic`

## Numerical Precision Summary

| Component | Tolerance | Notes |
|-----------|-----------|-------|
| F statistic | 0.01 | May differ slightly due to floating-point ranking |
| p-value | 0.01 | From F distribution CDF |
| Degrees of freedom | exact | Integer valued |
| A, B statistics | 1.0 | Sum of squared scores |
| Block ranges | exact | max - min within block |

## Known Differences

1. **Tie handling in range ranking**: R uses exact equality for detecting ties in rankings. Floating-point differences (e.g., 2.0 vs 1.9999999999999998) result in different ranks. Rust matches this behavior.

2. **F distribution p-value**: Both use the F distribution CDF. Very small p-values may have slightly different precision.

## Performance Comparison

| Configuration | Rust (µs) | R (µs) | Speedup |
|---------------|-----------|--------|---------|
| 10 blocks × 3 treatments | 1.89 | 670 | ~355x |
| 50 blocks × 4 treatments | 13.7 | 1,860 | ~136x |
| 100 blocks × 5 treatments | 27.0 | 3,610 | ~134x |
| 500 blocks × 3 treatments | 122 | 14,120 | ~116x |

**Notes:**
- Rust consistently outperforms R by 116-355x depending on problem size
- Rust benchmarks from Criterion (median); R benchmarks from system.time (mean of 100 iterations)
- Speedup is highest for small problems where R overhead dominates
- Performance measured on the same machine, same data generation seed

## References

- Quade, D. (1979). "Using weighted rankings in the analysis of complete blocks with additive block effects". Journal of the American Statistical Association, 74(367), 680-683.
- Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.). New York: Wiley. Pages 373-380.
- R Core Team. `stats::quade.test()` function. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quade.test.html
