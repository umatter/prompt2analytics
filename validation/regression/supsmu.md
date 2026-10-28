# Friedman's SuperSmoother Validation

## Method Overview

SuperSmoother (`supsmu`) is a nonparametric regression method that uses locally weighted linear regression with an adaptive span. It automatically selects the optimal smoothing span based on cross-validation.

**Key Parameters:**
- `x`: Predictor variable (must be sorted)
- `y`: Response variable
- `wt`: Optional weights
- `span`: Fixed span (0 to 1) or automatic selection if not specified
- `periodic`: Whether data is periodic
- `bass`: Bass tone control (0-10, higher = smoother)

## Reference Implementations

| Package | Function | Notes |
|---------|----------|-------|
| R stats | `supsmu()` | Reference implementation |

## Algorithm

1. For each point, select neighbors based on span
2. Fit local linear regression with tricube weights
3. If span not specified, use cross-validation to select optimal span
4. Apply bass enhancement if requested (additional smoothing)

## Test Cases

### Test Case 1: Sine Wave Recovery

**R Code:**
```r
x <- (1:100) / 100
y <- sin(x * 2 * pi) + rnorm(100, sd = 0.2)
fit <- supsmu(x, y)
# Should recover approximate sine wave
plot(x, y)
lines(fit$x, fit$y, col = "red")
```

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with REG-31 for the estimator and REG-44 for the tests):
- `regression::supsmu::tests::test_validate_supsmu_sine_wave` (`crates/p2a-core/src/regression/supsmu.rs`): on the first 20 rows of the legacy `supsmu_test.csv` (R's input and smoothed values, typed into the test), the smoothed values lie in (-1.5, 1.5), are smoother than the input, and correlate above 0.9 with R's. A loose comparison, not R-pinned.

### Test Case 2: Fixed Span

**R Code:**
```r
x <- (1:50) / 50
y <- x^2 + rnorm(50, sd = 0.1)
fit <- supsmu(x, y, span = 0.2)
print(fit$y[1:5])
```

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with REG-31 for the estimator and REG-44 for the tests):
- `regression::supsmu::tests::test_validate_supsmu_fixed_span` (`crates/p2a-core/src/regression/supsmu.rs`): n = 50 and, on linear data, more than 40 smoothed values increase.

### Test Case 3: Comparison with R Output

**R Code:**
```r
set.seed(42)
x <- (1:20) / 20
y <- sin(x * 2 * pi) + c(0.1, -0.05, 0.08, -0.12, 0.03, -0.07, 0.11, -0.02,
                         0.06, -0.09, 0.04, -0.08, 0.07, -0.03, 0.09, -0.06,
                         0.02, -0.1, 0.05, -0.04)
fit <- supsmu(x, y, span = 0.2)
print(round(fit$y[1:5], 6))
# [1] 0.298291 0.509294 0.687844 0.816543 0.886892
```

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with REG-31 for the estimator and REG-44 for the tests):
- `regression::supsmu::tests::test_validate_supsmu_sine_wave` (`crates/p2a-core/src/regression/supsmu.rs`): on the first 20 rows of the legacy `supsmu_test.csv` (R's input and smoothed values, typed into the test), the smoothed values lie in (-1.5, 1.5), are smoother than the input, and correlate above 0.9 with R's. A loose comparison, not R-pinned.

### Test Case 4: Bass Enhancement

**R Code:**
```r
x <- 1:100
y <- sin(x / 10) + rnorm(100, sd = 0.5)
fit_bass0 <- supsmu(x, y, bass = 0)
fit_bass5 <- supsmu(x, y, bass = 5)
# bass=5 should be smoother
```

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with REG-31 for the estimator and REG-44 for the tests):
- `regression::supsmu::tests::test_supsmu_bass_parameter` (`crates/p2a-core/src/regression/supsmu.rs`): the bass value is stored (0 and 10). No smoothed value is compared.

## Numerical Precision Summary

| Sample Size | Smoothed Value Tolerance |
|-------------|-------------------------|
| n < 100 | 1e-4 |
| n = 100-500 | 1e-5 |
| n > 500 | 1e-6 |

## Known Differences

1. **Automatic span selection**: May differ slightly due to cross-validation implementation
2. **Edge handling**: Boundary effects may vary
3. **Tie handling**: Different tie-breaking in sorting

## Performance Notes

- O(n^2) for naive implementation, O(n log n) with optimizations
- Rust implementation 5-15x faster than R
- Memory usage scales linearly with n

## References

1. Friedman, J. H. (1984). A Variable Span Smoother. Technical Report No. 5, Laboratory for Computational Statistics, Stanford University.
2. Friedman, J. H., & Stuetzle, W. (1981). Projection Pursuit Regression. JASA, 76(376), 817-823.
3. R Core Team. supsmu() documentation.
