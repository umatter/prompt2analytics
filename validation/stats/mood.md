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

### Test 1: Equal Scales - Basic
**Description:** Two samples with identical dispersion patterns, different locations.

**R Code:**
```r
x <- c(1, 2, 3, 4, 5)
y <- c(10, 20, 30, 40, 50)
mood.test(x, y)

# Expected output:
# Mood two-sample test of scale
# Z = 0, p-value = 1
```

**Rust Test:** `crates/p2a-core/src/stats/mood.rs::tests::test_validate_mood_against_r_basic`

**Results Comparison:**

| Metric | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| Z-score | 0.0 | ~0.0 | 0.5 | PASS |
| p-value | 1.0 | ~1.0 | 0.1 | PASS |

### Test 2: Manual Calculation Verification
**Description:** Small example for manual verification of test statistic.

**Data:**
- x = [1, 2, 3]
- y = [4, 5, 6]
- Combined ranks: 1, 2, 3, 4, 5, 6
- Mean rank = 3.5
- T for x = (1-3.5)² + (2-3.5)² + (3-3.5)² = 6.25 + 2.25 + 0.25 = 8.75
- Expected E[T] = m(N²-1)/12 = 3(35)/12 = 8.75
- z ≈ 0 (T = E[T])

**Rust Test:** `crates/p2a-core/src/stats/mood.rs::tests::test_validate_mood_statistic_calculation`

**Results Comparison:**

| Metric | Expected | Rust Value | Tolerance | Status |
|--------|----------|------------|-----------|--------|
| T statistic | 8.75 | 8.75 | 0.01 | PASS |
| Z-score | ~0 | ~0 | 0.5 | PASS |

### Test 3: Different Scales
**Description:** Two samples with clearly different dispersions.

**R Code:**
```r
# Sample with small variance
x <- c(4.5, 4.8, 5.0, 5.2, 5.5)  # range = 1

# Sample with large variance
y <- c(1.0, 3.0, 5.0, 7.0, 9.0)  # range = 8

mood.test(x, y)
```

**Rust Test:** `crates/p2a-core/src/stats/mood.rs::tests::test_mood_test_different_scales`

### Test 4: Data with Ties
**Description:** Verifies the Mielke (1967) tie correction is applied.

**R Code:**
```r
x <- c(1, 2, 2, 3, 4)
y <- c(2, 3, 3, 4, 5)
mood.test(x, y)
```

**Rust Test:** `crates/p2a-core/src/stats/mood.rs::tests::test_mood_test_with_ties`

**Results Comparison:**

| Metric | Expected | Rust Value | Status |
|--------|----------|------------|--------|
| has_ties | TRUE | TRUE | PASS |
| p-value in [0,1] | TRUE | TRUE | PASS |

### Test 5: Missing values and integer columns (R-pinned)

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

The tied integer pair `xi`/`yi` of the same case is not pinned: see STAT-29 under Known
Differences.

**Rust tests:**
- `stats::mood::tests::test_validate_mood_test_two_sample_na` (Z, p, T, n)
- `stats::mood::tests::test_validate_na_mood_test_missing_rows_and_errors` (`missing`, Display line, text and absent columns)

**Status:** R-pinned with a known deviation (registry `validation/methods/stats_onesample.json`,
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

- Z-scores match R within 0.5 for equal-scale data
- Test statistic T matches manual calculation within 0.01
- P-values match R's asymptotic approximation

## Known Differences

1. **Exact vs Asymptotic:** R uses asymptotic (normal) approximation for all sample sizes. Our implementation also uses asymptotic approximation.

2. **Tie handling (known deviation STAT-29):** R scores each tied observation by Mielke's
   (1967) average of the squared rank deviations over the positions the tie occupies; p2a
   squares the deviation of the midrank. Only the variance correction for ties is shared. On
   the tied integer pair `xi`/`yi` of case `stat_two_sample_na`, R gives z -2.667468897704627
   (p 0.007642496253585789) and p2a -2.698344723655288 (p 0.006968523851116348). The tied
   values are left out of the test's `Check` until STAT-29 is fixed; the tie-free pair
   `mi`/`mj` is pinned.

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
