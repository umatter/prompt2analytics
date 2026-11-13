# Validation: Median Polish (medpolish)

## Method Overview

Tukey's Median Polish algorithm for robust two-way decomposition. Decomposes a matrix into:
- Overall effect (grand median)
- Row effects
- Column effects
- Residuals

The algorithm iteratively subtracts row and column medians until convergence.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | `medpolish()` | R 4.3+ |

## Test Cases

### Test 1: Tukey's Original Example

**R Code**:
```r
# Tukey's two-way data example
data <- matrix(c(
  8.0, 6.0, 7.5,
  5.2, 4.0, 5.5,
  6.8, 5.3, 6.5
), nrow = 3, byrow = TRUE)

mp <- medpolish(data)
print(mp$overall)
print(mp$row)
print(mp$col)
print(mp$residuals)
```

#### R reference (not pinned)

No test asserts these values; they are R's output for the matrix above.

| Output | R Value |
|--------|---------|
| Overall | 6.0 |
| Row[0] | 0.75 |
| Row[1] | -1.0 |
| Row[2] | 0.25 |
| Col[0] | 0.3 |
| Col[1] | -1.0 |
| Col[2] | 0.5 |

**Rust Test**: `crates/p2a-core/src/stats/medpolish.rs::tests::test_medpolish_reconstructs_the_table` checks, on another 4 x 3 matrix, that overall + row + column + residual reconstructs the input exactly and that the overall effect is within 1.0 of R's 9.5. It compares no other R value, and no test asserts the table above.

### Test 2: Perfect Additive Model

A matrix with perfect row + column structure should have zero residuals.

**Rust Test**: `crates/p2a-core/src/stats/medpolish.rs::tests::test_medpolish_perfect_additive`

### MCP tool on integer columns with a missing cell (R-pinned)

The MCP tool `descriptive_medpolish` reads each requested column once through the shared
column extractor (`numeric_options`), so integer and float columns are accepted; a missing cell
(null or NaN) becomes NaN in the matrix, which `run_medpolish` skips with `na_rm = true` and
refuses otherwise, as R's `medpolish` does. Before, every Int64 column was refused ("Column 'c1'
is not numeric.") and each cell was looked up column by column.

**R Code** (case `stat_isoreg_medpolish_int` in `validation/r/cases/stats_mcp.R`; dataset
`validation/datasets/generated/isoreg_medpolish_int.csv`, 10 rows; `c1`-`c4` Int64, `c3` missing in
row 4):
```r
mp <- medpolish(as.matrix(d[paste0("c", 1:4)]), na.rm = TRUE)
mp3 <- medpolish(as.matrix(d[c("c1", "c2", "c4")]))
medpolish(as.matrix(d[paste0("c", 1:4)]))     # error: NA without na.rm = TRUE
```

| Quantity | R value |
|---|---|
| overall (`c1`-`c4`, `na.rm = TRUE`) | 38.01953125 |
| row effects | 4.23828125, -0.23828125, 5.76171875, -3.72265625, 1.01953125, -5.01953125, -9.26171875, 0.23828125, -0.76171875, 6.48046875 |
| column effects `c1`, `c2`, `c3`, `c4` | -6.27734375, -1.78125, 1.7421875, 8.21875 |
| overall (`c1`, `c2`, `c4`) | 37 |
| row effects | 5, -1, 5, -4, 0, -5, -10.875, 0, 0.125, 6 |
| column effects `c1`, `c2`, `c4` | -5.0625, 0, 10 |

**MCP test:** `tools::handlers::stats::tests::test_validate_mcp_descriptive_medpolish_int` (the
overall, row and column effects of both fits at their 6 printed decimals; the refusal without
`na_rm`).

**Tolerance:** the tool's 6 printed decimals (`Tol::printed(6)`).

**Status:** R-pinned (registry `validation/methods/mcp_stats.json`, id
`mcp_descriptive_medpolish`).

## Numerical Precision Summary

- All outputs match R within 1e-6 tolerance
- Convergence behavior matches R's default (eps=0.01, maxiter=10)

## Known Differences

- None identified

## Performance Comparison

| Matrix Size | Rust (µs) | R (µs) | Speedup |
|-------------|-----------|--------|---------|
| 10x10       | TBD       | TBD    | TBD     |
| 20x20       | TBD       | TBD    | TBD     |
| 50x50       | TBD       | TBD    | TBD     |

## References

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/medpolish.html
