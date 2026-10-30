# Validation: Cochran-Mantel-Haenszel Test (mantelhaen.test)

## Method Overview

The Cochran-Mantel-Haenszel (CMH) test evaluates the null hypothesis that two nominal variables are conditionally independent in each stratum, assuming no three-way interaction. It is commonly used in epidemiological studies to control for confounding when analyzing stratified 2×2 tables.

**Key Features:**
- Tests conditional independence in stratified 2×2 tables
- Provides common odds ratio estimate (Mantel-Haenszel estimator)
- Supports Yates' continuity correction
- Returns 95% confidence interval for the odds ratio

**Functions:** `p2a_core::stats::mantelhaen::{mantelhaen_test, run_mantelhaen_test, Table2x2}`;
MCP tool `hypothesis_mantelhaen`.

**Mathematical Background:**
```
CMH = (|Σ(a_k - E[a_k])| - 0.5)² / Σ Var(a_k)

where:
  E[a_k] = n1_k × m1_k / n_k     (expected count for cell a in stratum k)
  Var(a_k) = n1_k × n2_k × m1_k × m2_k / (n_k² × (n_k - 1))

Common OR (Mantel-Haenszel) = Σ(a_k × d_k / n_k) / Σ(b_k × c_k / n_k)
```

## Data, missing values and level order

`run_mantelhaen_test(dataset, row, col, stratum, ..)` builds the 2 × 2 × K tables with
`p2a_core::stats::stratified_tables` (`crates/p2a-core/src/stats/contingency.rs`), as R's
`mantelhaen.test(table(row, col, stratum))`:

- Rows with a missing value (null, or NaN in a float column) in any of the three columns are
  dropped, as `table()` drops `NA`, and reported in `MantelHaenszelResult.missing` (Display:
  a `Missing values:` line; MCP: `missing`). A missing stratum label is not a stratum.
- Row, column and stratum levels follow R `factor()` order (numbers by value, text in
  C-locale order), so the first level of each variable is the first row or column of every
  stratum's table and the orientation of the common odds ratio is R's. The row and column
  levels are common to all strata; `row_levels` and `col_levels` carry them, and
  `stratum_stats` are in stratum order.
- After dropping missing rows the row and column variables must have exactly 2 levels each;
  otherwise the error names the levels.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | mantelhaen.test | R 4.3.3 |

## R reference

Case `stat_mantelhaen_na`, defined in `validation/r/cases/stats_contingency.R`, dataset
`validation/datasets/generated/mantelhaen_na.csv` (240 rows of `treat` ctl/trt, `outcome`
no/yes and `stratum` s1/s2/s3, with 3 missing strata; seed 11):

```r
m  <- mantelhaen.test(table(d$treat, d$outcome, d$stratum))
mc <- mantelhaen.test(table(d$treat, d$outcome, d$stratum), correct = FALSE)
```

| Quantity | R value |
|---|---|
| Mantel-Haenszel X² (continuity correction) | 9.464008299682503 |
| p-value | 0.0020954268679205345 |
| X² without correction | 10.292389179376896 |
| p-value without correction | 0.0013358001984906988 |
| common odds ratio | 2.397932948724719 |
| 95% CI | 1.4029852155370948, 4.098462594545848 |
| rows used | 237 |

## Test Cases

### Test 1: Missing strata (R-pinned)

**Rust Tests:** `p2a-core` `stats::mantelhaen::tests::test_validate_mantelhaen_na` (statistic
with and without correction, p-values, common odds ratio and CI, rows used, strata s1 s2 s3,
levels, the missing-value summary) and `p2a-mcp`
`tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_mantelhaen_na` (the MCP
JSON). Before the fix a missing stratum formatted as "null" and formed a fourth stratum (240
rows); its three rows are all "no", so the statistic happened to equal R's.

### Test 2: Rabbits Data (R Documentation Example; values quoted)

**R Code:**
```r
Rabbits <- array(c(
  0, 0, 6, 5,
  3, 0, 3, 6,
  6, 2, 0, 4,
  5, 6, 1, 0,
  2, 5, 0, 0
), dim = c(2, 2, 5),
dimnames = list(
  Delay = c("None", "1.5h"),
  Response = c("Cured", "Died"),
  Penicillin.Level = c("1/8", "1/4", "1/2", "1", "4")
))
mantelhaen.test(Rabbits)

# Result:
# Mantel-Haenszel X-squared = 3.9286, df = 1, p-value = 0.04747
# common odds ratio estimate: 7
# 95 percent confidence interval: 1.026713 47.725133
```

**Rust Test:** `stats::mantelhaen::tests::test_validate_mantelhaen_against_r` (array input, the
values above at the printed precision; not registered).

### Test 3: Single Stratum

A single 2×2 table should produce results similar to a standard chi-squared test.

**Rust Test:** `stats::mantelhaen::tests::test_mantelhaen_single_stratum`

## Tolerance

`stat_mantelhaen_na` (n = 237): the statistic and p-value rows of `validation/README.md`
(`Tol::stat(237)`, `Tol::p_value(237)`: 1e-6 relative), the same for the common odds ratio
and its CI; rows used and strata exact. Registry: `validation/methods/stats_contingency.json`,
method `mantelhaen`.

## Known Differences

1. **Continuity correction**: Both R and Rust apply Yates' correction when |DELTA| >= 0.5 (default: true).

2. **Odds ratio CI**: Both use Robins, Breslow, Greenland (1986) variance formula for the log
   odds ratio, two-sided at 95%. Open: the one-sided intervals of `alternative = "less"` /
   `"greater"` and a `conf_level` other than 0.95 (STAT-59), and the method label when
   |DELTA| < 0.5 (STAT-81).

3. **Small strata**: a stratum with one observation contributes nothing, where R refuses the
   table ("sample size in each stratum must be > 1").

4. **Array layout**: R fills arrays column-major, so input tables need careful indexing:
   - R: c(a, c, b, d) fills to [[a,b], [c,d]]
   - Rust: Table2x2::new(a, b, c, d)

## Performance Comparison

Measured before Tier 2 (array input; the algorithm is unchanged):

| Number of Strata | Rust (µs) | R (µs) | Speedup |
|------------------|-----------|--------|---------|
| 5 | 0.89 | 360 | ~404x |
| 10 | 1.24 | 340 | ~274x |
| 50 | 6.5 | 860 | ~132x |
| 100 | 12.9 | 1,620 | ~126x |

**Notes:**
- Rust benchmarks from Criterion (median); R benchmarks from system.time (mean of 100 iterations)

## References

- Cochran, W. G. (1954). "Some Methods for Strengthening the Common χ² Tests". Biometrics, 10(4), 417-451.
- Mantel, N. & Haenszel, W. (1959). "Statistical Aspects of the Analysis of Data from Retrospective Studies of Disease". JNCI, 22(4), 719-748.
- Robins, J., Breslow, N., & Greenland, S. (1986). "Estimators of the Mantel-Haenszel variance consistent in both sparse data and large-strata limiting models". Biometrics, 42(2), 311-323.
- R Core Team. `stats::mantelhaen.test()` function. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mantelhaen.test.html
