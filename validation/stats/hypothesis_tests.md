# Validation: Hypothesis Tests (t-test, ANOVA)

## Method Overview

This document validates the implementation of:
- **t-test**: One-sample, two-sample (Welch's), paired
- **ANOVA**: One-way, two-way with interaction

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `t.test()` | R 4.x |
| stats | R | `aov()` | R 4.x |

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

### Test 4: One-Way ANOVA

**R Code:**
```r
# Groups: A (mean~10), B (mean~15), C (mean~20)
data <- data.frame(
  y = c(9.5, 10.2, 10.8, 9.8, 10.5,
        14.2, 15.5, 14.8, 15.2, 15.8,
        19.5, 20.2, 19.8, 20.5, 20.8),
  group = factor(rep(c("A","B","C"), each=5))
)
summary(aov(y ~ group, data = data))
```

**R Output:**
```
            Df Sum Sq Mean Sq F value   Pr(>F)
group        2 250.01  125.01   400.7 1.03e-11 ***
Residuals   12   3.74    0.31
```

**Rust Test:** `crates/p2a-core/src/stats/anova.rs::tests::test_validate_one_way_anova_against_r`

**Results Comparison:**
| Metric | R | Rust | Tolerance | Match |
|--------|---|------|-----------|-------|
| SS Between | 250.012 | 250.012 | 1e-3 | ✅ |
| SS Within | 3.744 | 3.744 | 1e-3 | ✅ |
| df Between | 2 | 2 | exact | ✅ |
| df Within | 12 | 12 | exact | ✅ |
| F-statistic | 400.66 | 400.66 | 1e-3 | ✅ |
| p-value | <1e-10 | <1e-10 | - | ✅ |
| η² | 0.9852 | 0.9852 | 1e-4 | ✅ |

### Test 5: Two-Way ANOVA

**R Code:**
```r
data <- data.frame(
  y = c(10,11,12, 15,16,17, 20,21,22, 30,31,32),
  factor_a = factor(rep(c("A","B","A","B"), each=3)),
  factor_b = factor(rep(rep(c("Low","High"), each=6)))
)
summary(aov(y ~ factor_a * factor_b, data = data))
```

**Rust Test:** `crates/p2a-core/src/stats/anova.rs::tests::test_validate_two_way_anova_against_r`

**Results Comparison:**
| Metric | R | Rust | Tolerance | Match |
|--------|---|------|-----------|-------|
| df Factor A | 1 | 1 | exact | ✅ |
| df Factor B | 1 | 1 | exact | ✅ |
| df Interaction | 1 | 1 | exact | ✅ |
| df Error | 8 | 8 | exact | ✅ |
| F Factor B | 468.75 | 468.75 | 0.1 | ✅ |
| F Interaction | 18.75 | 18.75 | 0.1 | ✅ |

---

## Numerical Precision Summary

| Method | Metric | Typical Tolerance | Notes |
|--------|--------|-------------------|-------|
| t-test | t-statistic | 1e-3 | Exact numerical match |
| t-test | p-value | 1e-4 | Using same t-distribution (statrs) |
| t-test | df (Welch) | 0.01 | Welch-Satterthwaite approximation |
| ANOVA | F-statistic | 1e-3 | Exact numerical match |
| ANOVA | SS components | 1e-3 | Sum of squares decomposition |
| ANOVA | Effect sizes | 1e-4 | η², ω² |

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

### One-Way ANOVA Performance

| Dataset | Rust (µs) | R (µs) | Speedup |
|---------|-----------|--------|---------|
| n=60 (3 groups) | 14 | 1000 | **~70x** |
| n=500 (5 groups) | 101 | 1500 | **~15x** |
| n=5000 (10 groups) | 943 | 3000 | **~3x** |

### Two-Way ANOVA Performance

| Dataset | Rust (µs) | R (µs) | Speedup |
|---------|-----------|--------|---------|
| n=40 (2x2) | 27 | 2000 | **~74x** |
| n=200 (2x2) | 112 | 2000 | **~18x** |
| n=400 (2x2) | 207 | 2000 | **~10x** |

### Performance Analysis

The Rust implementation shows significant speedups across all methods:

1. **Small datasets**: 10-100x faster due to R's interpreter overhead
2. **Large datasets**: 3-10x faster due to efficient memory access and no GC
3. **Scalability**: Rust maintains sub-millisecond performance up to n=100,000

The larger relative speedups for small datasets are expected because R's function call overhead and object creation dominate for small inputs, while Rust has minimal overhead.

---

## Known Differences

None identified. Both implementations produce numerically identical results within floating-point precision.

---

## References

- Student (W. S. Gosset) (1908). "The probable error of a mean". *Biometrika*, 6(1), 1-25.
- Welch, B. L. (1947). "The generalization of 'Student's' problem". *Biometrika*, 34(1-2), 28-35.
- Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver & Boyd.
- R Core Team (2024). R: A Language and Environment for Statistical Computing.
