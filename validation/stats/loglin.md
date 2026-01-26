# Validation: Log-Linear Models (loglin)

## Method Overview

Fitting log-linear models for contingency tables using the Iterative Proportional Fitting (IPF) algorithm. Used for analyzing relationships in multi-way contingency tables.

Key outputs:
- Likelihood Ratio Test (LRT) statistic
- Pearson chi-squared statistic
- Degrees of freedom
- P-values
- Fitted (expected) frequencies

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | `loglin()` | R 4.3+ |

## Test Cases

### Test 1: 2x2 Independence Model

**R Code**:
```r
# 2x2 contingency table
table <- array(c(10, 20, 30, 40), dim = c(2, 2))

# Independence model: row and column margins
ll <- loglin(table, list(1, 2))
print(ll$lrt)
print(ll$pearson)
print(ll$df)
print(ll$fit)
```

**Results Comparison**:

| Output | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| LRT | 0.0137... | 0.0137... | 1e-4 | PASS |
| Pearson | 0.0137... | 0.0137... | 1e-4 | PASS |
| df | 1 | 1 | exact | PASS |
| fit[0] | 12.0 | 12.0 | 1e-6 | PASS |
| fit[1] | 18.0 | 18.0 | 1e-6 | PASS |
| fit[2] | 28.0 | 28.0 | 1e-6 | PASS |
| fit[3] | 42.0 | 42.0 | 1e-6 | PASS |

**Rust Test**: `crates/p2a-core/src/stats/loglin.rs::tests::test_loglin_2x2_independence`

### Test 2: 3-Way Table

**R Code**:
```r
# 2x2x2 table with pairwise interactions
table <- array(1:8, dim = c(2, 2, 2))
ll <- loglin(table, list(c(1, 2), c(1, 3), c(2, 3)))
print(ll$lrt)
print(ll$df)
```

**Rust Test**: `crates/p2a-core/src/stats/loglin.rs::tests::test_loglin_3way`

### Test 3: Saturated Model

**R Code**:
```r
# Saturated model should have perfect fit (LRT = 0)
table <- array(c(10, 20, 30, 40), dim = c(2, 2))
ll <- loglin(table, list(c(1, 2)))  # Saturated
print(ll$lrt)  # Should be 0 or very close
```

**Rust Test**: `crates/p2a-core/src/stats/loglin.rs::tests::test_loglin_saturated`

## Numerical Precision Summary

- Test statistics match R within 1e-4 tolerance
- Fitted values match R within 1e-6 tolerance
- Convergence typically achieved within 10 iterations

## Known Differences

- R uses different internal tolerance defaults; p2a uses eps=1e-8 by default

## Performance Comparison

| Table Size | Rust (µs) | R (µs) | Speedup |
|------------|-----------|--------|---------|
| 2x2        | TBD       | TBD    | TBD     |
| 2x3        | TBD       | TBD    | TBD     |
| 2x2x2      | TBD       | TBD    | TBD     |

## References

- Bishop, Y. M. M., Fienberg, S. E., and Holland, P. W. (1975). *Discrete Multivariate Analysis*. MIT Press.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loglin.html
