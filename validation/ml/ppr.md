# Projection Pursuit Regression Validation

## Method Overview

Projection Pursuit Regression (PPR) fits a model of the form:
y = sum_{m=1}^{M} f_m(alpha_m' * x) + error

where alpha_m are projection directions and f_m are univariate smooth functions (ridge functions).

**Key Parameters:**
- `x`: Predictor matrix (n x p)
- `y`: Response vector (n)
- `nterms`: Number of terms (ridge functions) to fit
- `max_terms`: Maximum terms to consider for backward elimination
- `sm_method`: Smoothing method for ridge functions

## Reference Implementations

| Package | Function | Notes |
|---------|----------|-------|
| R stats | `ppr()` | Reference implementation |

## Algorithm

1. Initialize with zero model
2. For each term m = 1, ..., M:
   a. Find optimal projection direction alpha_m maximizing correlation
   b. Fit smooth function f_m to projected values
   c. Update residuals
3. Optional backward elimination to remove weak terms

## Test Cases

### Test Case 1: Simple Projection

**R Code:**
```r
set.seed(42)
n <- 200
x <- matrix(rnorm(n * 3), n, 3)
# True model: y = sin(x1 + x2) + noise
alpha_true <- c(1, 1, 0) / sqrt(2)
proj <- x %*% alpha_true
y <- sin(2 * proj) + rnorm(n, sd = 0.1)

fit <- ppr(x, y, nterms = 1)
print(fit$alpha)  # Should be close to c(0.707, 0.707, 0)
print(cor(fit$fitted.values, y))  # Should be high
```

**Rust tests** (not R-pinned: no `ppr()` value is compared; the R-pinned case comes with ML-107 for the tests and ML-34 for the smoother options):
- `ml::ppr::tests::test_ppr_basic` (`crates/p2a-core/src/ml/ppr.rs`): nterms, n, p and the lengths of the fitted values and residuals on its own 15-row dataset.

### Test Case 2: Multiple Terms

**R Code:**
```r
set.seed(42)
n <- 300
x <- matrix(rnorm(n * 4), n, 4)
# Two-term model
y <- sin(x[,1] + x[,2]) + cos(x[,3] - x[,4]) + rnorm(n, sd = 0.2)

fit <- ppr(x, y, nterms = 2)
print(summary(fit))
```

**Rust tests** (not R-pinned: no `ppr()` value is compared; the R-pinned case comes with ML-107 for the tests and ML-34 for the smoother options):
- `ml::ppr::tests::test_ppr_multiple_terms` (`crates/p2a-core/src/ml/ppr.rs`): at most 2 terms and a non-empty goodness-of-fit vector (n = 50).

### Test Case 3: Comparison with R Output

**R Code:**
```r
set.seed(42)
X <- matrix(rnorm(100 * 3), 100, 3)
y <- sin(X[,1] * 2 + X[,2]) + rnorm(100, sd = 0.1)
fit <- ppr(X, y, nterms = 1)
print(round(fit$alpha, 6))
# Should give projection direction
print(round(fit$fitted.values[1:5], 6))
```

**Rust tests** (not R-pinned: no `ppr()` value is compared; the R-pinned case comes with ML-107 for the tests and ML-34 for the smoother options):
- `ml::ppr::tests::test_ppr_with_weights` (`crates/p2a-core/src/ml/ppr.rs`): n with case weights.
- `ml::ppr::tests::test_ppr_predict` (`crates/p2a-core/src/ml/ppr.rs`): the number of predictions. No test compares fitted values, directions or ridge functions with R.

## Numerical Precision Summary

| Sample Size | Alpha Tolerance | Fitted Value Tolerance |
|-------------|-----------------|----------------------|
| n < 100 | 0.1 | 0.05 |
| n = 100-500 | 0.05 | 0.02 |
| n > 500 | 0.02 | 0.01 |

## Known Differences

1. **Smoothing method**: Different smoother implementations
2. **Optimization**: Local optima possible; directions may differ by sign
3. **Scaling**: Different centering/scaling conventions

## Performance Notes

- O(n * p * nterms) per iteration
- Rust implementation 3-10x faster than R
- Memory scales with n * p

## References

1. Friedman, J. H., & Stuetzle, W. (1981). Projection Pursuit Regression. JASA, 76(376), 817-823.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.
3. R Core Team. ppr() documentation.
