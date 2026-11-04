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

All values below are R 4.3.3 from case `stat_rank_isoreg` (`validation/r/cases/stats_rank_t3.R`;
dataset `validation/datasets/generated/rank_isoreg.csv`, one row per point, column `set` naming
the example). Rust test: `stats::isoreg::tests::test_validate_isoreg_examples` (registry
`validation/methods/stats_rank_t3.json`, id `isoreg`). Tolerances are the `validation/README.md`
rows for n < 100 (fitted values `Tol::coef`, 1e-6 relative); sorted x, order and knot counts
exact.

### Test 1: R documentation example (doc)

```r
ir <- isoreg(c(1, 0, 4, 3, 3, 5, 4, 2, 0))
```

| Quantity | R value | p2a | Status |
|---|---|---|---|
| `yf` | 0.5, 0.5, 3, 3, 3, 3, 3, 3, 3 | the same | pinned |
| number of knots | 2 | 2 | pinned |
| `iKnots` | 2, 9 (1-based block ends) | `i_knots` 0, 2 (0-based block starts) | deviation STAT-52 (definitions) |
| `yc` | 0, 1, 1, 5, 8, 11, 16, 20, 22, 22 (`cumsum(c(0, y))`, n + 1 values) | cumsum(yf), n values | deviation STAT-52 (definitions) |

The previous version of this table listed `yf[2:4]` = 3.33 and `yf[5:8]` = 2.75, a decreasing
fit that neither R nor p2a produces; both give 3 for the last seven values.

### Test 2: last value changed (doc3)

`isoreg(c(1, 0, 4, 3, 3, 5, 4, 2, 3))`: `yf` 0.5, 0.5, 3.3333333333333335 (x3), 3.5 (x4), pinned.

### Test 3: with x (ir4, the `?isoreg` example)

`isoreg(1:10, c(5, 9, 1:2, 5:8, 3, 8))`: `yf` 4.25 (x4), 5, 6 (x4), 8, pinned. R's `iKnots` are
4, 5, 6, 9, 10: R keeps the two adjacent pooled blocks with the fitted value 6 apart, p2a merges
them into one (4 knots); deviation STAT-52, not compared.

### Test 4: already monotone and strictly decreasing

`isoreg(1:5)`: `yf` 1, 2, 3, 4, 5. `isoreg(5:1)`: `yf` 3 (x5), one knot. Both pinned.

### Test 5: unsorted, tie-free x

15 points (seed 20261026; x a permutation of 0.641 ... 14.889):

| Quantity | R value |
|---|---|
| `yf` | -0.055, 0.52, 1.627, 1.9435, 1.9435, 2.0379999999999994, 2.2249999999999996, 2.2249999999999996, 2.3665000000000003, 2.3665000000000003, 2.496666666666666 (x3), 2.7875000000000014 (x2) |
| sorted x | 0.641, 1.522, 2.569, 3.757, 4.827, 5.631, 6.729, 7.667, 8.753, 9.8, 10.722, 11.736, 12.749, 13.888, 14.889 |
| `ord` (1-based) | 9, 5, 6, 8, 11, 14, 3, 10, 4, 1, 2, 13, 12, 15, 7 |

All three are pinned (p2a's `ord` is 0-based; the test adds 1).

### Known difference: tied x (STAT-52)

`isoreg(c(3, 1, 2, 2, 1), c(5, 1, 2, 4, 3))`: R orders by (x, -y) and pools tied x, `yf` 2, 2, 3,
3, 5 (`iKnots` 2, 4, 5); p2a gives 1, 2.5, 2.5, 4, 5. Kept in the case as `yf_tiedx` and
`iknots_tiedx`, not compared.

### Input checks

A NaN in x or a non-finite y is refused ("missing values not allowed: x[1] is NaN", as R's
"missing values not allowed"); x is sorted with `f64::total_cmp`. Test:
`stats::isoreg::tests::test_isoreg_rejects_nan_x`.

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

- Fitted values agree with R to 6.4e-16 relative on every tie-free set (planning port) and are
  pinned at the `validation/README.md` rows (`Tol::coef`).
- Sorted x, order and the knot counts of doc and decr are exact.

## Known Differences

- **STAT-52 (Tier 4):** p2a reports the first index (0-based) of each constant block as its
  knots (0, 1, 2, 5, 9 on the case above), R's `iKnots` the last (1-based: 1, 2, 5, 9, 10); tied
  x values are not pooled as R pools them; adjacent pooled blocks with equal fitted values are
  one block in p2a (ir4: 4 knots, R 5). Only the number of knots is pinned, where it agrees.

## Performance Comparison

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | TBD       | TBD    | TBD     |
| n=1,000      | TBD       | TBD    | TBD     |
| n=10,000     | TBD       | TBD    | TBD     |

## References

- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., and Brunk, H. D. (1972). *Statistical Inference Under Order Restrictions*. John Wiley & Sons.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/isoreg.html
