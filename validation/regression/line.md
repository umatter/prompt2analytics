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

## Missing and Non-finite Values (REG-13)

`line()` deletes the pairs with a missing (NaN) or infinite x or y before the fit and
reports them in `LineResult::missing` (`by_column` names `x` and `y`); `x`, `y`, `fitted`
and `residuals` hold the kept pairs in input order. With fewer than 3 complete pairs it is
an error. Sorting and medians use `f64::total_cmp`. Before, a NaN made the sort comparator
inconsistent, which can panic since Rust 1.81, or gave meaningless coefficients. MCP
`regression_line` and `p2a reg line` read x and y with the
incomplete pairs dropped and report them (`missing` in JSON, a `Note:` line in the CLI
text); the dataset paths refuse `±Inf` (`NonFiniteValues`) before `line()` is reached.
With fewer than 3 complete pairs they fail before `line()` too, with `InsufficientData`
naming the method and the columns and counting the dataset's rows ("Tukey's resistant
line of 'y' on 'x': 2 of 6 rows have no missing value").

R: case `reg_smoothers_na` of `validation/r/cases/regression_na.R`
([source](../r/cases/regression_na.R)), 60 points (`set.seed(8)`) in
`validation/datasets/generated/smoothers_na.csv` with x missing in rows 5 and 33 and y in
rows 12 and 48. R's `line(x, y)` uses the 56 complete pairs (`n_complete`), and its
coefficients on the NA data equal those on the complete pairs.

Tests (registry entry `line_missing` in `validation/methods/regression_na.json`), exact:

- `test_validate_reg_smoothers_na_fixture`: the dtypes and missing cells polars reads, 56
  complete pairs, no tied x among them;
- `test_validate_line_na`: `n` is R's `n_complete`; the fit on the NA data equals the fit
  on the complete pairs; an infinite x is deleted too; too few complete pairs is an error;
- `test_validate_mcp_regression_line_na`, `test_validate_cli_reg_line_na`: the same
  through MCP and the CLI.

Property tests, not registered (no R value):
`test_mcp_regression_line_supsmu_too_few_complete_rows` and
`test_cli_reg_line_supsmu_too_few_complete_rows` check the too-few-rows error of line and
supsmu through MCP and the CLI.

### Known difference (REG-12)

p2a's resistant line is a different variant of Tukey's line (the iteration count is off by
one and the intercept comes from the middle group). On the 56 complete pairs R gives
intercept 0.441594195008563728 and slope 0.053890384144849528; p2a gives
-0.514777229630535849 and 0.062911296831234101 (equal to an R transcription of `line.rs`).
The coefficients are a registry deviation until REG-12 (Tier 4) ports `tukeyline`.

### Known difference: infinite values

R's `line()` keeps `±Inf` (it deletes only `NA` pairs, `complete.cases`); p2a's array API
deletes them, and the dataset paths refuse them.

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
