# Validation: Bartlett, Fligner-Killeen and F tests of variances

## Method Overview

Bartlett's test tests the null hypothesis that all k population variances are equal against the alternative that at least two are different. It is commonly used as a preliminary test before ANOVA to check the homoscedasticity assumption.

**Key Parameters:**
- `response`: Response variable (numeric)
- `factor`: Grouping factor (categorical)

**Test Statistic:**
```
T = [(N-k) ln(s²_p) - Σ(nᵢ-1)ln(s²ᵢ)] / C

where:
- s²_p = Σ(nᵢ-1)s²ᵢ / (N-k)  (pooled variance)
- s²ᵢ = sample variance of group i
- nᵢ = sample size of group i
- N = total sample size
- k = number of groups
- C = 1 + [1/(3(k-1))] × [Σ(1/(nᵢ-1)) - 1/(N-k)]  (correction factor)
```

Under H₀, T ~ χ²(k-1).

**Outputs:**
- K-squared test statistic
- Degrees of freedom (k-1)
- P-value from chi-squared distribution
- Group-wise variance estimates

**Important Note:** Bartlett's test is sensitive to departures from normality. If samples are non-normal, consider using Levene's test instead.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `bartlett.test()` | R 4.3.2 |

## Test Cases

### Test 1: Equal Variances (Groups with Same Variance)

Three groups with equal variance - should fail to reject H₀.

**R Code:**
```r
y <- c(1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7)
group <- factor(c(rep("A", 5), rep("B", 5), rep("C", 5)))
bartlett.test(y ~ group)
```

**Expected Results:**
- All groups have variance = 2.5
- P-value > 0.05 (fail to reject H₀)

**Rust Test:** `crates/p2a-core/src/stats/bartlett.rs::tests::test_bartlett_basic`

### Test 2: Unequal Variances (Significant Difference)

Three groups with clearly different variances - should reject H₀.

**R Code:**
```r
x <- c(1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 5, 10, 15, 20, 25)
g <- factor(c(rep("A", 5), rep("B", 5), rep("C", 5)))
bartlett.test(x ~ g)

# Output:
#   Bartlett test of homogeneity of variances
# data:  x by g
# Bartlett's K-squared = 12.142, df = 2, p-value = 0.002309

tapply(x, g, var)
# A    B    C
# 2.5  2.5 62.5
```

**Results Comparison:**

| Metric | R | Rust (p2a) | Tolerance |
|--------|---|------------|-----------|
| K-squared | 12.142 | 12.142 | < 0.01 |
| df | 2 | 2 | exact |
| p-value | 0.002309 | 0.002309 | < 0.001 |
| Var(A) | 2.5 | 2.5 | < 0.001 |
| Var(B) | 2.5 | 2.5 | < 0.001 |
| Var(C) | 62.5 | 62.5 | < 0.001 |

**Rust Test:** `crates/p2a-core/src/stats/bartlett.rs::tests::test_validate_bartlett_against_r`

### Test 3: Two Groups Only

Minimal case with only 2 groups.

**Results Comparison:**

| Metric | Expected | Rust (p2a) | Tolerance |
|--------|----------|------------|-----------|
| df | 1 | 1 | exact |
| n_groups | 2 | 2 | exact |

**Rust Test:** `crates/p2a-core/src/stats/bartlett.rs::tests::test_bartlett_two_groups`

### Test 4: Unequal Sample Sizes

Test with groups of different sizes (n = 3, 5, 4).

**Results Comparison:**

| Metric | Expected | Rust (p2a) | Tolerance |
|--------|----------|------------|-----------|
| n_obs | 12 | 12 | exact |
| n_groups | 3 | 3 | exact |
| Group sizes correct | Yes | Yes | - |

**Rust Test:** `crates/p2a-core/src/stats/bartlett.rs::tests::test_bartlett_unequal_sample_sizes`

### Test 5: Large Variance Difference

Groups with very different variances - tests numerical stability.

**Setup:**
- Group "Low": very small variance (data: 1.0, 1.1, 1.2, 0.9, 1.0)
- Group "Med": medium variance (data: 1.0, 2.0, 3.0, 0.0, 2.0)
- Group "High": large variance (data: 1.0, 10.0, 20.0, -5.0, 4.0)

**Results Comparison:**

| Metric | Expected | Rust (p2a) | Tolerance |
|--------|----------|------------|-----------|
| p-value | < 0.05 | < 0.05 | - |
| Conclusion | Reject H₀ | Reject H₀ | - |

**Rust Test:** `crates/p2a-core/src/stats/bartlett.rs::tests::test_bartlett_unequal_variances`

## R-pinned tests: missing values and integer columns

All three tests below run on case `stat_two_sample_na` in `validation/r/cases/stats_onesample.R`
(dataset `validation/datasets/generated/two_sample_na.csv`, 22 rows). `xi` and `yi` are Int64
columns with missing values; the group column `g` (levels a, b, c) is missing in row 22, and
`xi` in rows 21 and 22, so `xi ~ g` has 20 complete rows (groups of 7, 7 and 6).

### Bartlett's test, `xi ~ g`

```r
bt <- bartlett.test(xi ~ g, data = d)   # model.frame: na.omit
okg <- complete.cases(d$xi, d$g)
gvar <- tapply(d$xi[okg], d$g[okg], var); gn <- tapply(d$xi[okg], d$g[okg], length)
```

| Quantity | R |
|---|---|
| K-squared | 1.0870055705502384 |
| df | 2 |
| p-value | 0.5807105812221599 |
| group variances a, b, c (cross-check) | 63.95238095238095, 32.80952380952381, 27.5 |
| pooled variance (cross-check) | 42.239495798319325 |
| group sizes | 7, 7, 6 (N = 20) |

### Fligner-Killeen test, `xi ~ g`

```r
fl <- fligner.test(xi ~ g, data = d)    # model.frame: na.omit
```

| Quantity | R |
|---|---|
| Fligner-Killeen:med chi-squared | 1.1621429648018324 |
| df | 2 |
| p-value | 0.5592987666070923 |
| group sizes | 7, 7, 6 |

### F test to compare two variances, `var.test(xi, yi)`

```r
vt <- var.test(d$xi, d$yi)              # each sample drops its own NA
```

| Quantity | R |
|---|---|
| F | 0.35160452177225177 |
| num df, denom df | 19, 17 |
| p-value | 0.030340988035820703 |
| 95% CI of the ratio | (0.13353100158915807, 0.9025664881231058) |
| ratio of variances | 0.35160452177225177 |
| var(xi), var(yi) | 40.01052631578948, 113.79411764705883 |
| n_x, n_y | 20, 18 |

### Missing values and column types

- `run_bartlett_test` and `run_fligner_test` read the value and group columns in one request
  through the shared extractor: a row with a missing value or a missing label is dropped from
  both (R's formula interface, `na.omit`), and the drop is reported in `missing` and a last
  Display line `Missing values: ...`. Groups are listed in R's factor order (numeric labels
  numerically, text in byte order).
- `run_var_test` drops each sample's own missing values (`var.test(x, y)`), as `t.test` does.
- Integer, decimal and boolean responses are accepted; the group column may have any label
  type. A text response is `NonNumericColumn`.

Before T2-WP06, the Fligner wrapper read null values as 0 and required a string group column,
the F test read nulls as 0, and all three refused integer responses.

**Rust tests:**
- `stats::bartlett::tests::test_validate_bartlett_test_two_sample_na`
- `stats::bartlett::tests::test_validate_na_bartlett_test_errors`
- `tier2::wp06_onesample::test_validate_mcp_hypothesis_bartlett_test` (p2a-mcp, the tool `hypothesis_bartlett_test`)
- `stats::fligner::tests::test_validate_fligner_test_two_sample_na`
- `stats::fligner::tests::test_validate_na_fligner_test_errors`
- `stats::vartest::tests::test_validate_var_test_two_sample_na`
- `stats::vartest::tests::test_validate_na_var_test_missing_rows_and_errors`

**Tolerance:** `validation/README.md`, n < 100: statistics, variances and CI 1e-5 relative
(`Tol::stat`), the variance ratio 1e-6 (`Tol::coef`), p `Tol::p_value`; df and counts exact.

**Status:** R-pinned (registry `validation/methods/stats_onesample.json`, ids `bartlett_test`,
`fligner_test`, `var_test`).

## Numerical Precision Summary

| Test Case | Statistic Match | P-value Match |
|-----------|-----------------|---------------|
| Equal variances | Exact | Exact |
| Unequal variances | < 0.01 | < 0.001 |
| Two groups | Exact | Exact |
| Unequal sizes | Exact | Exact |

## Known Differences

1. **Normality Sensitivity**: Like R, our implementation assumes normality. Both will produce incorrect results for non-normal data.

2. **Numerical Stability**: Our implementation uses the same correction factor formula as R's `bartlett.test()` for consistent results.

3. **Zero Variance Handling**: Unlike some implementations, we explicitly reject groups with zero variance as invalid input.

## Performance Comparison

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=15, k=3    | 0.33      | 580    | ~1758x  |
| n=100, k=5   | 0.54      | 740    | ~1370x  |
| n=1,000, k=10| 2.7       | 880    | ~326x   |
| n=10,000, k=20| 23.3     | 3040   | ~130x   |

*Benchmarks run on Linux with Rust Criterion (100 samples) and R system.time (50 iterations).*

**Note:** The extreme speedup is due to:
1. Rust's zero-allocation design for small datasets
2. No R interpreter overhead
3. Direct computation without data frame manipulation

## MCP Tool Usage

```json
{
  "tool": "hypothesis_bartlett_test",
  "dataset": "my_data",
  "response": "score",
  "factor": "treatment"
}
```

## References

- Bartlett, M. S. (1937). "Properties of Sufficiency and Statistical Tests". *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences*, 160(901), 268-282.
- R Core Team. `stats::bartlett.test()` function. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/bartlett.test.html
- NIST/SEMATECH e-Handbook of Statistical Methods. https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.htm
