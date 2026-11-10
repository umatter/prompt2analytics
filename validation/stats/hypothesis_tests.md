# Validation: t-tests, and missing values and integer columns through MCP and the CLI

ANOVA (one-way; two-way with Type I, II and III sums of squares) is documented in [anova.md](anova.md),
Welch's one-way test in [oneway.md](oneway.md).

## Method Overview

This document validates the implementation of:
- **t-test**: One-sample, two-sample (Welch's), paired

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `t.test()` | R 4.x |

---

## Test Cases

### Test 1: One-Sample t-test

**R Code:**
```r
x <- c(2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7)
t.test(x, mu = 2.0)
```

**R Output:**
```
t = 5.3316, df = 6, p-value = 0.001775
95% CI: (2.262799, 2.708629)
mean of x = 2.485714
```

**Rust Test:** `crates/p2a-core/src/stats/ttest.rs::tests::test_validate_one_sample_against_r`

**Status:** not R-pinned (inline quoted values).

**Results Comparison:**
| Metric | R | Rust | Tolerance | Match |
|--------|---|------|-----------|-------|
| t-statistic | 5.3316 | 5.3316 | 1e-3 | ✅ |
| df | 6 | 6 | exact | ✅ |
| p-value | 0.001775 | 0.001775 | 1e-4 | ✅ |
| CI lower | 2.2628 | 2.2628 | 0.01 | ✅ |
| CI upper | 2.7086 | 2.7086 | 0.01 | ✅ |

### Test 2: Two-Sample Welch t-test

**R Code:**
```r
x <- c(2.1, 2.5, 2.3, 2.8, 2.6)
y <- c(3.2, 3.5, 3.1, 3.8, 3.4)
t.test(x, y)
```

**R Output:**
```
t = -5.4636, df = 7.9985, p-value = 0.0005993
95% CI: (-1.3367526, -0.5432474)
mean of x = 2.46, mean of y = 3.40
```

**Rust Test:** `crates/p2a-core/src/stats/ttest.rs::tests::test_validate_welch_against_r`

**Status:** not R-pinned (inline quoted values).

**Results Comparison:**
| Metric | R | Rust | Tolerance | Match |
|--------|---|------|-----------|-------|
| t-statistic | -5.4636 | -5.4636 | 1e-3 | ✅ |
| df (Welch) | 7.9985 | 7.9985 | 0.01 | ✅ |
| p-value | 0.0005993 | 0.0005993 | 1e-4 | ✅ |

### Test 3: Paired t-test

**R Code:**
```r
before <- c(200, 190, 210, 180, 195)
after <- c(195, 185, 202, 175, 188)
t.test(before, after, paired = TRUE)
```

**R Output:**
```
t = 9.4868, df = 4, p-value = 0.0006889
95% CI: (4.244022, 7.755978)
mean difference = 6
```

**Rust Test:** `crates/p2a-core/src/stats/ttest.rs::tests::test_validate_paired_against_r`

**Status:** not R-pinned (inline quoted values).

**Results Comparison:**
| Metric | R | Rust | Tolerance | Match |
|--------|---|------|-----------|-------|
| t-statistic | 9.4868 | 9.4868 | 1e-3 | ✅ |
| df | 4 | 4 | exact | ✅ |
| p-value | 0.0006889 | 0.0006889 | 1e-4 | ✅ |
| mean diff | 6.0 | 6.0 | 1e-3 | ✅ |

### Test 3b: t-tests with missing values and integer columns (R-pinned)

**R Code** (case `stat_two_sample_na` in `validation/r/cases/stats_onesample.R`; dataset
`validation/datasets/generated/two_sample_na.csv`, 22 rows: `x` and `y` with 7 values each and
6 complete pairs, `xi`/`yi` Int64 columns with 20 and 18 values):
```r
one <- t.test(d$x); two <- t.test(d$x, d$y); two_eq <- t.test(d$x, d$y, var.equal = TRUE)
pr <- t.test(d$x, d$y, paired = TRUE); ti <- t.test(d$xi, d$yi)     # NA removed as t.test does
```

**R values** (the full set, with the standard errors and estimates, is in
`validation/expected/stat_two_sample_na.json`):

| Test | t | df | p | 95% CI |
|---|---|---|---|---|
| one-sample `x` (n = 7) | 19.224879776549294 | 6 | 1.281621618642191e-06 | (5.622820623444335, 7.26289366226995) |
| Welch `x`, `y` (7, 7) | 2.4211420066066545 | 11.986295516630276 | 0.03227121954167002 | (0.11282854757136428, 2.1443143095714943) |
| pooled `x`, `y` | 2.4211420066066545 | 12 | 0.032250124224961374 | (0.11295735014996908, 2.144185506992889) |
| paired `x`, `y` (6 pairs) | 14.696938456699081 | 5 | 2.635511420589e-05 | (0.9901128720208033, 1.4098871279791965) |
| Welch `xi`, `yi` (Int64; 20, 18) | 0.9936932426636085 | 27.038848651726983 | 0.32918223786045675 | (-3.0521808116098357, 8.78551414494317) |

**Missing values** (`t_test`, the dataset wrapper):

| Test | Policy |
|---|---|
| one-sample | the column's missing values (null or NaN) are dropped (`x[!is.na(x)]`) |
| two-sample | each column drops its own missing values (`t.test(x, y)`); `missing` sums both |
| paired | a pair with a missing value in either column is dropped (`complete.cases(x, y)`) |

The drop is reported in `TTestResult.missing` and a last Display line `Missing values: ...`.
Integer, decimal and boolean columns are accepted; a text column is `NonNumericColumn`, an
infinite value `NonFiniteValues`. Before T2-WP06, a null was read as 0 (one-sample t 3.09 on
21 df) and an integer column was refused.

**Rust tests:**
- `stats::ttest::tests::test_validate_t_test_two_sample_na` (every number above, the SEs and estimates)
- `stats::ttest::tests::test_validate_na_t_test_missing_rows` (`missing` per test, the Display line, a NaN cell)
- `stats::ttest::tests::test_validate_na_t_test_errors` (text, absent and infinite columns)

**Tolerance:** `validation/README.md`, n < 100: estimates `Tol::coef` (1e-6), SE, t, Welch df
and CI `Tol::stat`/`Tol::se` (1e-5 relative), p `Tol::p_value`; counts and integer df exact.

**Status:** R-pinned (registry `validation/methods/stats_onesample.json`, id `t_test`).

### MCP tool: missing values and integer columns

`hypothesis_t_test` calls the dataset wrapper `t_test`, so the MCP tool drops missing values
as above (each sample on its own; complete pairs when paired), reports the drop in the last line
of its text (`Missing values: 15 of 22 rows dropped because of missing values (x: 15)` for the
one-sample test of `x`) and accepts integer columns. Before, the handler read its columns with
`into_no_null_iter`, so every missing cell entered the test as 0 (one-sample t 3.09 on 21 df
where R gives 19.22 on 6), and an Int64 column was refused.

The tool's text is pinned to the same case, `stat_two_sample_na`, at its printed precision: t,
p, the CI and the means to 6 decimals, df to 2 (one-sample `x`, Welch `x`, `y`, paired `x`, `y`,
Welch on the Int64 `xi`, `yi`; the values are in the table above).

**MCP tests** (`crates/p2a-mcp/src/tools/handlers/hypothesis.rs`):
- `tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_t_test_two_sample_na` (t,
  df, p, CI and means of the four tests; the missing-value lines)
- `tools::handlers::hypothesis::tests::test_mcp_hypothesis_two_sample_columns` (not R-pinned, a
  property test of the t-test, cor.test, KS and Mood tools: Int64 columns give the results of
  their Float64 copies; a text column, in either role, is refused with "Column 'g' contains
  non-numeric values", an absent one with "Column 'nope' not found in dataset."; a paired test
  without `y` keeps its own error)

**Tolerance:** `Tol::printed(6)` (`Tol::printed(2)` for df).

**Status:** R-pinned (registry `validation/methods/mcp_stats.json`, id `mcp_hypothesis_t_test`).

Every JSON hypothesis tool reports dropped rows under `missing` (`n_total`, `n_used`,
`n_dropped`, `by_column`), and has no such key when nothing was dropped.
`tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_handlers_report_missing_rows`
checks the report of Bartlett, Kruskal-Wallis, oneway, Friedman, Quade, Wilcoxon, the pairwise t
and Wilcoxon tests and Shapiro-Wilk against R's complete cases on the T2-WP06 and T2-WP07 cases
(`stat_two_sample_na`, `stat_oneway_na_labels`, `stat_friedman_quade`, `stat_wilcoxon_na`,
`stat_pairwise_mahalanobis_na`, `stat_shapiro_wilk`): rows used, group sizes and blocks exactly
(registry id `mcp_hypothesis_missing_report`). Friedman and Quade drop the whole block of a
missing response.

## Missing cells and integer columns through MCP and the CLI (X-2, X-8)

The reviewer's repro of findings X-2 and X-8 is a committed R case: `validation/r/cases/na_acceptance.R`,
case `na_x2_x8_reviewer`, dataset `validation/datasets/generated/na_reviewer.csv` (8 rows: the
reviewer's `x` and `y` with one empty cell each, the integer group `g` and score `i`, plus the integer
survival columns `time` and `event`). The rule: empty cells are missing and are dropped as R drops NA,
with a note; integer columns are numeric. The MCP tests load the CSV with `create_dataset`, as the
reviewer did; the CLI tests with `data load`.

| Quantity | R | p2a before Tier 2 |
|---|---|---|
| one-sample t, df (`t.test(x)`) | 19.224879776549294, 6 | 6.585420425150713, 7 (the empty cell read as 0) |
| mean, 95% CI | 6.4428571428571431; [5.6228206234443352, 7.26289366226995] | mean 5.6375 |
| paired t, df, p, mean difference | 14.696938456699081, 5, 2.6355114205889999e-05, 1.2 | t 0.8849631314399664, df 7 |
| Welch t, df, p | 2.4211420066066545, 11.986295516630276, 0.032271219541670018 | t 0.8822470125695193, df 13.60729227134347 |
| `cor.test(x, y)` r, t, df, p | 0.9782906634777736, 9.4412517531339581, 4, 0.0007018271843026733 | r 0.006219360234733851 on 8 pairs |
| `aov(y ~ factor(g))` F, df, p, rows | 0.085835141593650022, (1, 5), 0.7813098377313259, 7 | F 0.5994334277620387 on (1, 6), N = 8 |
| `t.test(i)` t, df, p, mean (Int64) | 7.5055534994651349, 7, 0.00013666401542355999, 6.5 | refused: "Column 'i' must be numeric." (MCP), "expected `Float64`, got `i64`" (CLI) |
| `aov(i ~ factor(g))` F, df, p | 0.30000000000000027, (1, 6), 0.60364505651013611 | refused as non-numeric |
| `wilcox.test(i)` V, exact p | 36, 0.0078125000000000017 | refused as non-numeric |
| `cor.test(i, y)` r, df, p | 0.45289724238230478, 5, 0.30750148504153196 | refused: "expected `Float64`, got `i64`" |
| Kaplan-Meier S(t) at 3, 5, 7, 9, 15 (Int64 time and event) | 0.875, 0.72916666666666674, 0.58333333333333337, 0.3888888888888889, 0; median 9 | refused: "Column 'time' contains non-numeric values" |

**Tests:**
- `p2a-mcp::tier2::wp22_acceptance::test_validate_mcp_x2_reviewer_missing_cells` (one-sample,
  paired and Welch t and df, `cor.test`, one-way ANOVA, rows used, the `Missing values:` lines)
- `p2a-mcp::tier2::wp22_acceptance::test_validate_mcp_x8_reviewer_integer_columns` (t, ANOVA,
  Wilcoxon, `cor.test` and the Kaplan-Meier median on the Int64 columns)
- `p2a-cli::tier2::wp22_acceptance::test_validate_cli_x2_reviewer_missing_cells` (`stats t-test-one`,
  `stats t-test-paired`: JSON values, `missing` and the `Note:` line)
- `p2a-cli::tier2::wp22_acceptance::test_validate_cli_x8_reviewer_integer_columns` (`stats
  t-test-one` on `i`, `survival km` with Int64 `time` and `event`: S(t), SE, numbers at risk and
  events, median)

**Tolerance:** the MCP text at its printed precision (t to 6 decimals, df to 2, the ANOVA table to
4, the Kaplan-Meier median to 3); the CLI JSON at `validation/README.md`'s n < 100 tolerances
(means `Tol::coef`, statistics and SEs `Tol::stat`/`Tol::se`, p-values `Tol::p_value`); df and
counts exact.

**Status:** R-pinned (registry `validation/methods/na_acceptance.json`, id `na_acceptance`).

---

## Confidence levels

`one_sample_t_test`, `two_sample_t_test`, `paired_t_test`, `t_test`, `cor_test`, `prop_test_one`,
`prop_test_two`, `prop_test_k` and `poisson_test` refuse a `conf_level` outside (0, 1), NaN included,
with `InvalidSpecification` ("conf_level must be strictly between 0 and 1, got 95"). R's `t.test`
stops too ("'conf.level' must be a single number between 0 and 1"). Before, 95 for "95%" reached the
distribution's quantile function and failed there.

`cor_test` refuses a constant x or y and infinite values before any distribution is built; R's
`cor.test` returns NA with the warning 'the standard deviation is zero'. This is a deviation of the
registry entry `cor_test_pearson`.

**R Code:** case `stat_ttest_levels_t3` in `validation/r/cases/stats_misc_t3.R`, dataset
`validation/datasets/generated/ttest_levels_t3.csv` (x has 12 values, y 15; 12 complete pairs).

| Quantity | R |
|---|---|
| one sample, level 0.90, two-sided CI | 4.870807801387869, 5.772025531945465 |
| one sample, level 0.90, one-sided bounds | 5.66351755314406 (less), 4.979315780189274 (greater) |
| Welch, level 0.99, two-sided CI | -0.40016154497221085, 1.746861544972211 |
| paired, level 0.90, two-sided CI | -0.16986609694703195, 1.6856994302803654 |
| t (one sample), Welch df | 1.2809941317687727, 24.975739555092265 |

Every p-value of the three tests under the three alternatives is in the case.

**Rust tests:**
- `stats::ttest::tests::test_validate_t_test_t3_levels` (R-pinned; registry `validation/methods/stats_misc_t3.json`, id `t_test_levels_t3`)
- `stats::ttest::tests::test_t_test_rejects_level_95`
- `stats::cortest::tests::test_cor_test_constant_column`, `stats::cortest::tests::test_cor_test_rejects_invalid_level`
- `stats::proptest::tests::test_prop_test_rejects_invalid_level`
- `stats::poissontest::tests::test_poisson_refuses_non_finite_parameters`

## Numerical Precision Summary

| Method | Metric | Typical Tolerance | Notes |
|--------|--------|-------------------|-------|
| t-test | t-statistic | 1e-3 | Exact numerical match |
| t-test | p-value | 1e-4 | Using same t-distribution (statrs) |
| t-test | df (Welch) | 0.01 | Welch-Satterthwaite approximation |

---

## Performance Comparison

### Benchmark Environment
- **Rust**: Criterion.rs, release mode (`cargo bench`)
- **R**: `system.time()` with 50 iterations, median reported
- **Hardware**: Same machine for both

### T-Test Performance

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n = 100 | 5.5 | <1000* | >100x |
| n = 1,000 | 10.7 | <1000* | >90x |
| n = 10,000 | 52 | 1000 | ~19x |
| n = 100,000 (one-sample) | 249 | 2000 | **8x** |
| n = 100,000 (two-sample) | 478 | 5000 | **10x** |

*R's `system.time()` has millisecond resolution, so sub-ms times appear as 0.

### Performance Analysis

The Rust implementation shows significant speedups across all methods:

1. **Small datasets**: 10-100x faster due to R's interpreter overhead
2. **Large datasets**: 3-10x faster due to efficient memory access and no GC
3. **Scalability**: Rust maintains sub-millisecond performance up to n=100,000

The larger relative speedups for small datasets are expected because R's function call overhead and object creation dominate for small inputs, while Rust has minimal overhead.

### Wall-clock against the previous tree (signed off 2026-11-08)

Medians of the round medians over five interleaved rounds against `ff8e7559`, `release`, one P core,
`RAYON_NUM_THREADS=1`, on a quiet machine (load average 1.1-2.0, 2026-11-08). Ratio is this tree over the base.

| Row | n | base (µs) | this tree (µs) | ratio (median) |
|---|---|---:|---:|---:|
| one-sample t-test | 100 | 2.120 | 2.393 | 1.13 |
| one-sample t-test | 1,000 | 3.679 | 4.519 | 1.23 |
| one-sample t-test | 10,000 | 13.00 | 10.86 | 0.84 |
| two-sample t-test | 100 | 2.311 | 2.560 | 1.11 |
| two-sample t-test | 1,000 | 4.517 | 2.307 | 0.51 |
| two-sample t-test | 10,000 | 23.45 | 21.32 | 0.91 |

The p-value and the interval now come from the shared accurate t-tail helpers (`t_test_p_value`, `t_quantile`)
instead of statrs `StudentsT`. Their cost depends on the t statistic, not on n, so the change is a fixed amount per
call and the ratio varies with the data. Signed off by the user.

---

## Known Differences

None identified. Both implementations produce numerically identical results within floating-point precision.

---

## References

- Student (W. S. Gosset) (1908). "The probable error of a mean". *Biometrika*, 6(1), 1-25.
- Welch, B. L. (1947). "The generalization of 'Student's' problem". *Biometrika*, 34(1-2), 28-35.
- Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver & Boyd.
- R Core Team (2024). R: A Language and Environment for Statistical Computing.
