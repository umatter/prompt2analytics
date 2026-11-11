# Validation: Mood Two-Sample Test of Scale

## Method Overview

Mood's two-sample test is a non-parametric test for comparing scale parameters of two distributions. It tests whether two independent samples have the same dispersion (scale), using squared deviations of ranks from the mean rank as the test statistic.

**Key Parameters:**
- `x`, `y`: Two numeric samples
- `alternative`: "two.sided", "greater", or "less"

**Underlying Model:**
- Sample 1: f(x - l)
- Sample 2: f((x - l)/s)/s
- H₀: s = 1 (equal scales)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | mood.test | R 4.3+ |
| scipy.stats | Python | mood | 1.11+ |

## Test Cases

All values in Tests 1-3 are R 4.3.3 from cases `stat_rank_two_sample` and
`stat_rank_small_groups` (`validation/r/cases/stats_rank_t3.R`; datasets
`validation/datasets/generated/rank_two_sample.csv` and `rank_small_groups.csv`), registry
`validation/methods/stats_rank_t3.json`, id `mood_scale_t3`. Tolerances: `validation/README.md`
rows for n < 100 (z `Tol::stat`, p `Tol::p_value`); T exact.

### Test 1: Four tie-free two-sample sets

```r
mood.test(x, y, alternative = "two.sided" | "less" | "greater")
```

| Set | x, y | z | p | p (less) | p (greater) | T |
|---|---|---|---|---|---|---|
| basic | 1:5, 10 * (1:5) | 0 | 1 | 0.5 | 0.5 | 41.25 |
| scale | 10 values near 5, 10 spread over 0.8-9.2 | -3.236694374850748 | 0.0012092284281239386 | 0.0006046142140619693 | 0.999395385785938 | 112.5 |
| manual | 1:3, 4:6 | 0 | 1 | 0.5 | 0.5 | 8.75 |
| sim | 20 + 20 normal draws, sd 1 and 3 (seed 20261024) | -2.5699457794613076 | 0.01017144333403517 | 0.005085721667017585 | 0.9949142783329824 | 1685 |

T is the sum of the squared deviations of the x ranks from (N + 1)/2 (for "manual": (1 - 3.5)² +
(2 - 3.5)² + (3 - 3.5)² = 8.75 = E[T], so z = 0). p2a ranks the pooled sample with
`stats::ranks::midranks` (R's `rank()`).

**Rust test:** `stats::mood::tests::test_validate_mood_two_sample`

### Test 2: Values of order 1e-12

x and y are the tie-free column `ynt` of groups a and b of `stat_rank_small_groups` (7 + 7 values,
all within 1e-10 of each other): z -2.2857142857142856, p 0.0222709789592328, T 49.75.

**Rust test:** `stats::mood::tests::test_validate_mood_small_scale`

### Test 3: Data with ties (R-pinned)

| Set | x, y | R z | R p |
|---|---|---|---|
| ties | (1, 2, 2, 3, 4), (2, 3, 3, 4, 5) | 0.3293662746201299 | 0.7418788559147744 |
| serum | Hollander & Wolfe serum data (`?ansari.test`), 20 + 20 | 1.0371275614960966 | 0.2996764118570592 |

A tied observation is scored as R does (Mielke 1967): the average of the squared rank deviations
over the positions its tie group occupies, `(r - (N + 1)/2)² + (t² - 1)/12` for midrank r and
group size t. `mood_z_ties` and `mood_p_ties` are pinned and both sets are flagged as tied.
Case `stat_rank_scale_t4` adds a heavily tied pair (`mood_heavy_*`: T, z, p for every
alternative; `stats::mood::tests::test_validate_mood_heavy_ties_t4`, registry
`validation/methods/stats_rank_t4.json`, id `mood_heavy_t4`).

### Test 4: Missing values and integer columns (R-pinned)

**R Code** (case `stat_two_sample_na` in `validation/r/cases/stats_onesample.R`; `mi`, `mj` are
tie-free Int64 columns of `validation/datasets/generated/two_sample_na.csv` with 15 and 12
values, the rest missing):
```r
mt_nt <- mood.test(d$mi, d$mj)   # each sample drops its own NA
```

| Quantity | R | Tolerance |
|---|---|---|
| Z | 3.074157339784952 | `Tol::stat(27)` |
| p-value | 0.002110980682689112 | `Tol::p_value(27)` |
| T (sum of squared rank deviations of `mi`, cross-check) | 1348 | `Tol::stat(27)` |
| n_x, n_y | 15, 12 | exact |

**Missing values:** `run_mood_test` reads each column through the shared extractor and drops its
own missing values (null or NaN), as `mood.test` does; the drop is reported in
`MoodTestResult.missing` and a last Display line `Missing values: ...`. Integer columns are
accepted; a text column is `NonNumericColumn`. Before T2-WP06, nulls were read as 0 and
integer columns were refused.

The tied integer pair `xi`/`yi` of the same case is pinned too: z -2.667468897704627,
p 0.007642496253585789.

**Rust tests:**
- `stats::mood::tests::test_validate_mood_test_two_sample_na` (Z, p, T, n; z and p on `xi`/`yi`)
- `stats::mood::tests::test_validate_na_mood_test_missing_rows_and_errors` (`missing`, Display line, text and absent columns)

**Status:** R-pinned (registry `validation/methods/stats_onesample.json`,
id `mood_test`).

**MCP tool: missing values and integer columns.** `hypothesis_mood_test` calls
`run_mood_test`, so the tool drops each sample's own missing values, accepts integer columns and
reports the drop under `missing` (the two samples' summaries summed; `by_column` is
`{"mi": 7, "mj": 10}` on this case). Before, the handler read its columns with
`into_no_null_iter` and refused Int64 columns ("Column 'mi' must be numeric.").
`tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_mood_test_two_sample_na` pins
the tool's `statistic` (T), `z_score`, `p_value`, `n_x` and `n_y` to the tie-free pair `mi`, `mj`
above at the same tolerances and checks `has_ties` and `missing` (registry
`validation/methods/mcp_stats.json`, id `mcp_hypothesis_mood_test`).

## Numerical Precision Summary

- On tie-free data z and the three p-values are pinned at the `validation/README.md` rows
  (`Tol::stat`, `Tol::p_value`); T is exact.
- With ties z and p are pinned too (R's averaged scores).

## Known Differences

1. **Exact vs Asymptotic:** R uses asymptotic (normal) approximation for all sample sizes. Our implementation also uses asymptotic approximation.

2. **Tie handling:** as R, each tied observation is scored by Mielke's (1967) average of the
   squared rank deviations over the positions the tie occupies, and the variance carries R's
   tie correction.

## Performance Comparison

Benchmarked on 2026-01-20.

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 13.7      | 340    | ~25x    |
| n=1,000      | 185.8     | 960    | ~5x     |
| n=10,000     | 2,824     | 8,920  | ~3x     |
| n=100,000    | 35,080    | 102,240| ~3x     |

**Performance Evaluation:**
- All sizes show Rust faster than R: PASS
- At least 2 sizes >= 2x speedup: PASS (all 4)
- n=10,000 >= 1.5x speedup: PASS (~3x)

## References

- Conover, W. J. (1971). *Practical Nonparametric Statistics*. New York: John Wiley & Sons. Pages 234-235.
- Mielke, P. W. (1967). "Note on Some Squared Rank Tests with Existing Ties." *Technometrics*, 9(2), 312-314.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mood.test.html
