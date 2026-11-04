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

## Algorithm

Iterative proportional fitting as Haberman's AS 51 and R's `loglin`: the fit starts at 1 in
every cell and the observed margins are computed once. A cycle visits each margin in turn,
records the largest absolute deviation of the fitted margin from the observed one **before**
rescaling the fit to it (a cell whose fitted margin is 0 becomes 0), and the fit has converged
after a cycle whose largest deviation is below `eps`. Defaults are R's: `eps = 0.1`, `iter = 20`.
A fit that stops at `max_iter` keeps `converged = false`, a `Convergence` record and the warning
"loglin did not converge after N iterations (...)" (R: "algorithm did not converge"). p-values
are the chi-square upper tail (`traits::estimator::chi_squared_p_value`); a statistic at or below
0, or `df = 0`, has p = 1.

Before the STAT-28 fix the deviation was measured right after each rescaling, so it was zero by
construction: every model stopped after one cycle and reported convergence, which is exact only
for decomposable models.

## Test Cases

### Test 1: 2x2 independence

```r
loglin(array(c(10, 20, 30, 40), c(2, 2)), list(1, 2), fit = TRUE)
# lrt 0.80434864609648349, pearson 0.79365079365079361, df 1, fit 12, 18, 28, 42 (2 iterations)
```

(An earlier version of this page quoted 0.0137 here, which no run produces.) Decomposable fits
are pinned in `test_validate_loglin_decomposable_against_r` (G2, X2, p, df and the fitted table
of the 2x3x4 model `list(c(1,2), c(2,3))`, 1e-10) and the degrees of freedom of five models in
`test_validate_loglin_df_against_r`.

### Test 2: no three-way interaction on a 2x3x4 table (R-pinned)

Case `stat_loglin_nodecomp` (`validation/r/cases/stats_power_loglin.R`, dataset
`validation/datasets/generated/loglin_2x3x4.csv`, 652 counts in long form):

```r
loglin(A, list(c(1, 2), c(1, 3), c(2, 3)), eps = 1e-12, iter = 1000, fit = TRUE)
```

| Quantity | R | Tolerance |
|---|---|---|
| G2 (`lrt`) | 12.728242021076264 | `Tol::stat(652)` (1e-6 relative) |
| X2 (`pearson`) | 12.673307054444779 | `Tol::stat(652)` |
| df | 6 | exact |
| p (G2), p (X2) | 0.047560301442266605, 0.048527369139224945 | `Tol::p_value(652)` |
| fitted table (24 cells) | `fit_rowmajor` | `Tol::stat(652)` |
| decomposable `list(c(1,2), c(1,3))`: G2, df | 17.48379813637173, 12 | `Tol::stat(652)`, exact |

R's output, not R-pinned: 8 cycles at `eps = 1e-12`; 3 cycles at the default `eps = 0.1` (G2
12.72824202137702); one cycle (`iter = 1`) G2 12.72946180036028, which is what the pre-fix code
returned as converged. The test asserts convergence at R's defaults and non-convergence after one
cycle.

**Rust tests:** `stats::loglin::tests::test_validate_loglin_nodecomp`,
`stats::loglin::tests::test_loglin_reports_nonconvergence` (at `iter = 3`: `converged = false`,
the warning and the `Converged: no (3 iterations ...)` line).

### Test 3: STAT-28's 2x2x2 table (R-pinned)

```r
A <- array(0, c(2, 2, 2)); obs <- c(10, 20, 30, 40, 15, 25, 35, 5)   # row-major in (2, 2, 2)
loglin(A, list(c(1, 2), c(1, 3), c(2, 3)), eps = 1e-10, iter = 1000, fit = TRUE)
```

Pinned: G2 8.147900084041238, X2 8.211542795886288, their p-values 0.004311105401133119 and
0.004162474480305096, the fitted table and df 1 (`Tol::stat(180)`, `Tol::p_value(180)`).
R's output, not R-pinned: 24 cycles at `eps = 1e-10`, 5 at the default (G2 8.14808241013826), and
one cycle G2 10.018549340799476 (p 0.0015, against 0.0043 at convergence).

**Rust test:** `stats::loglin::tests::test_validate_loglin_nodecomp_2x2x2`.

### R example without a Rust test

`loglin(array(1:8, c(2, 2, 2)), list(c(1, 2), c(1, 3), c(2, 3)))`: G2 0.047291730689550837 after
2 cycles at `eps = 0.1`; 0.047284830446495341 after 9 cycles at `eps = 1e-12`.

### Saturated model

`loglin(array(c(10, 20, 30, 40), c(2, 2)), list(c(1, 2)))`: G2 0, df 0, the fit equals the table
(`test_loglin_saturated`).

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
three models are decomposable; non-decomposable models are pinned in Tests 2 and 3 above.

**Status:** R-pinned (registry `validation/methods/mcp_stats.json`, id `mcp_stats_loglin`).

## Numerical Precision Summary

- G2, X2 and the fitted table: 1e-6 relative (`Tol::stat`) at `eps = 1e-12` (Test 2) and
  `eps = 1e-10` (Test 3); p-values `Tol::p_value`; df exact.
- Iteration counts are not pinned: the fit is pinned at the solution.

## Known Differences

- The defaults are R's (`eps = 0.1`, `iter = 20`); an earlier version of this page said p2a used
  `eps = 1e-8`, which it never did.
- Pearson's X2 skips cells with a zero fitted value, where R's sum gives NaN.

## Performance Comparison

| Table Size | Rust (µs) | R (µs) | Speedup |
|------------|-----------|--------|---------|
| 2x2        | TBD       | TBD    | TBD     |
| 2x3        | TBD       | TBD    | TBD     |
| 2x2x2      | TBD       | TBD    | TBD     |

## References

- Bishop, Y. M. M., Fienberg, S. E., and Holland, P. W. (1975). *Discrete Multivariate Analysis*. MIT Press.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loglin.html
