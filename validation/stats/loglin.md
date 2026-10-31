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

### MCP tool on long-form data with missing values (R-pinned)

The MCP tool `stats_loglin` takes long-form data (one row per cell or observation: a count
column and the factor columns) and builds the table before calling `loglin`. It reads the count
and every factor through the shared column extractor in one request, so a row with a missing
count or a missing factor label is dropped, as `xtabs` drops it, and reported under `missing`.
The levels of each factor follow R's `factor()` order of the kept rows (numeric codes
numerically: 2 before 10), and duplicated cells are summed. G2, X2 and both p-values are JSON
numbers. Before, a missing label became a level `""` of its own, levels came in hash order and
the statistics were 4-decimal strings (on the case below: a 4 x 3 x 2 table, independence df 17,
G2 68.31).

**R Code** (case `stat_loglin_na` in `validation/r/cases/stats_mcp.R`; dataset
`validation/datasets/generated/loglin_na.csv`, 16 rows: a 3 x 2 x 2 table in long form, rows 11
and 13 the same cell (b, 10, y), row 14 with a missing `f1`, row 15 a missing `f2`, row 16 a
missing `cnt`; `f2` holds the Int64 codes 2 and 10):
```r
tab <- xtabs(cnt ~ f1 + f2 + f3, data = d)            # 13 rows used; dimnames f2 = c("2", "10")
loglin(tab, list(1, 2, 3)); loglin(tab, list(c(1, 2), 3)); loglin(tab, list(c(1, 2), c(1, 3)))
# p-values: pchisq(statistic, df, lower.tail = FALSE) (loglin itself reports none)
```

| Model (MCP `margins`) | G2 (lrt) | Pearson X2 | df | p (G2) | p (X2) |
|---|---|---|---|---|---|
| independence (omitted) | 15.959234383672948 | 15.642276135815438 | 7 | 0.02549142201482758 | 0.0285932605922053 |
| [12][3] (`[[0, 1], [2]]`) | 5.251990050660357 | 5.249591014734913 | 5 | 0.3859073684210251 | 0.38618531921891985 |
| [12][13] (`[[0, 1], [0, 2]]`) | 3.371328778805629 | 3.367170036253498 | 3 | 0.33783775176599284 | 0.3384027088963117 |

Table total 448, dimensions 3 x 2 x 2, levels `a, b, c` / `2, 10` / `x, y`; 13 of 16 rows used
(`cnt`, `f1` and `f2` each missing once).

**MCP test:** `tools::handlers::stats::tests::test_validate_mcp_stats_loglin_na` (both statistics,
df and p-values of the three models, total, dimensions, levels and `missing`; a text count column
is refused with "Column 'f1' contains non-numeric values").

**Tolerance:** `validation/README.md`, n = 448 counts: statistics 1e-6 relative (`Tol::stat`), p
1e-6 relative plus 1e-12 absolute (`Tol::p_value`); df, dimensions, total and counts exact. All
three models are decomposable, so the one-cycle IPF fit is exact; for non-decomposable models
the convergence finding STAT-28 (Tier 3) is open.

**Status:** R-pinned (registry `validation/methods/mcp_stats.json`, id `mcp_stats_loglin`).

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
