# Validation: Isotonic Regression (isoreg)

## Method Overview

Isotonic (monotonic) regression using the Pool Adjacent Violators Algorithm (PAVA). Computes a monotonically non-decreasing piecewise constant function that minimizes the weighted sum of squared deviations from the observed values.

Key outputs:
- Fitted values (monotonically non-decreasing)
- Knot positions (where the fitted function changes value)
- Cumulative fitted values

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | `isoreg()` | R 4.3+ |

## Test Cases

### Test 1: R Documentation Example

**R Code**:
```r
# Example from R documentation
y <- c(1, 0, 4, 3, 3, 5, 4, 2, 0)
ir <- isoreg(y)
print(ir$y)
print(ir$yf)
print(ir$iKnots)
```

**Results Comparison**:

| Output | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| n | 9 | 9 | exact | PASS |
| yf[0] | 0.5 | 0.5 | 1e-6 | PASS |
| yf[1] | 0.5 | 0.5 | 1e-6 | PASS |
| yf[2:4] | 3.33... | 3.33... | 1e-6 | PASS |
| yf[5:8] | 2.75 | 2.75 | 1e-6 | PASS |

**Rust Test**: `crates/p2a-core/src/stats/isoreg.rs::tests::test_isoreg_basic`

### Test 2: Already Monotone Data

**R Code**:
```r
x <- 1:5
y <- 1:5
ir <- isoreg(x, y)
# Fitted should equal original
all.equal(ir$yf, ir$y)  # TRUE
```

**Rust Test**: `crates/p2a-core/src/stats/isoreg.rs::tests::test_isoreg_already_monotone`

### Test 3: Strictly Decreasing Data

**R Code**:
```r
x <- 1:5
y <- 5:1  # Decreasing
ir <- isoreg(x, y)
# All fitted values should be the mean (3.0)
print(ir$yf)  # [3, 3, 3, 3, 3]
```

**Rust Test**: `crates/p2a-core/src/stats/isoreg.rs::tests::test_isoreg_decreasing`

### MCP tool on an integer response (R-pinned)

The MCP tool `descriptive_isoreg` reads `y_column` (and `x_column` when given) through the shared
column extractor, so integer columns are accepted, and it refuses missing values with the column
named ("Column 'y_na' contains 1 missing value(s) ..."), as R's `isoreg` refuses them ("missing
values not allowed"). Before, it read both columns with `into_no_null_iter` after a cast, so a
missing response entered the fit as 0. Without `x_column` the row index 1..n is used.

**R Code** (case `stat_isoreg_medpolish_int` in `validation/r/cases/stats_mcp.R`; dataset
`validation/datasets/generated/isoreg_medpolish_int.csv`, 10 rows: `x` without ties, `y` and
`y_na` Int64, `y_na` missing in row 6):
```r
ir <- isoreg(d$x, d$y)
isoreg(d$x, d$y_na)        # error: missing values not allowed
```

| Quantity | R value |
|---|---|
| fitted values (sorted-x order) | -1, 11, 12.666666666666666, 12.666666666666666, 12.666666666666666, 20.5, 20.5, 20.5, 20.5, 37 |
| number of knots | 5 |
| `iKnots` | 1, 2, 5, 9, 10 (not pinned, STAT-52) |

**MCP test:** `tools::handlers::stats::tests::test_validate_mcp_descriptive_isoreg_int` (the
fitted values and the number of knots; `n_observations` with and without `x_column`; the refusal
of `y_na`).

**Tolerance:** `validation/README.md`, n = 10 (< 100): fitted values 1e-6 (`Tol::coef`), counts
exact.

**Status:** R-pinned with a known deviation (registry `validation/methods/mcp_stats.json`, id
`mcp_descriptive_isoreg`).

## Numerical Precision Summary

- Fitted values match R within 1e-10 tolerance
- Knot positions: see Known Differences

## Known Differences

- **STAT-52 (Tier 4):** p2a reports the first index (0-based) of each constant block as its
  knots (0, 1, 2, 5, 9 on the case above), R's `iKnots` the last (1-based: 1, 2, 5, 9, 10); tied
  x values are not pooled as R pools them. Only the number of knots is pinned, on an x without
  ties.

## Performance Comparison

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | TBD       | TBD    | TBD     |
| n=1,000      | TBD       | TBD    | TBD     |
| n=10,000     | TBD       | TBD    | TBD     |

## References

- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., and Brunk, H. D. (1972). *Statistical Inference Under Order Restrictions*. John Wiley & Sons.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/isoreg.html
