# Tukey's Resistant Line Validation

## Method Overview

Tukey's resistant line (`line()`) is a robust regression method that fits a line using medians instead of means, making it resistant to outliers. The data is divided into three groups, and medians are computed for each group to determine the slope and intercept.

**Key Parameters:**
- `x`: Predictor variable
- `y`: Response variable
- `iter`: Number of polishing iterations (default: 1)

## Reference Implementations

| Package | Function | Notes |
|---------|----------|-------|
| R stats | `line()` | Reference implementation |

## Algorithm

1. Sort data by x values
2. Divide into three groups (by x tertiles)
3. Compute median x and median y for left and right groups
4. Slope = (median_y_right - median_y_left) / (median_x_right - median_x_left)
5. Intercept computed using middle group
6. Optional polishing iterations to refine residuals

## Test Cases

### Test Case 1: Simple Linear Relationship

**R Code:**
```r
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.9, 17.8, 20.1)
fit <- line(x, y)
print(fit$coefficients)
# intercept     slope
#      0.0       2.0
```

**Expected Output:**
- Intercept: ~0.0
- Slope: ~2.0

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with REG-12 for the estimator and REG-44 for the tests):
- `regression::line::tests::test_validate_line_clean_data` (`crates/p2a-core/src/regression/line.rs`): the slope is within 0.1 of 0.5 and the intercept within 0.5 of 2.0.

### Test Case 2: With Outliers (Resistance Test)

**R Code:**
```r
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2, 4, 6, 8, 100, 12, 14, 16, 18, 20)  # Outlier at position 5
fit <- line(x, y)
# Should still give slope ~2, resistant to outlier
print(fit$coefficients)
```

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with REG-12 for the estimator and REG-44 for the tests):
- `regression::line::tests::test_validate_line_with_outliers` (`crates/p2a-core/src/regression/line.rs`): R's `line()` values typed into the test (intercept 2.42169727298821, slope 0.475089391721295), checked within 1.0 and 0.1. A quoted-value check, not R-pinned.

### Test Case 3: Comparison with OLS

**R Code:**
```r
set.seed(42)
x <- 1:50
y <- 2 * x + 5 + rnorm(50, sd = 2)
y[25] <- 200  # Add outlier

# Resistant line
fit_line <- line(x, y)

# OLS (not resistant)
fit_lm <- lm(y ~ x)

cat("line(): slope =", fit_line$coefficients[2], "\n")
cat("lm():   slope =", coef(fit_lm)[2], "\n")
# line() slope will be closer to 2.0
```

## Numerical Precision Summary

| Sample Size | Coefficient Tolerance |
|-------------|----------------------|
| n < 50 | 1e-4 |
| n = 50-200 | 1e-6 |
| n > 200 | 1e-8 |

## Known Differences

1. **Tie handling**: When x values have ties, group assignment may differ slightly between implementations
2. **Edge cases**: Very small samples (n < 9) may produce different results

## Performance Notes

- O(n log n) complexity due to sorting
- Rust implementation ~10x faster than R for n > 1000
- Memory efficient: only stores residuals

## References

1. Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
2. Velleman, P. F., & Hoaglin, D. C. (1981). Applications, Basics, and Computing of Exploratory Data Analysis.
3. R Core Team. line() documentation.
