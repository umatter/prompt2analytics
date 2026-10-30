# Validation: Pairwise t-test (pairwise.t.test)

## Method Overview

Pairwise t-tests with p-value adjustment for multiple comparisons. Performs all pairwise comparisons between group levels using t-tests, then applies correction for family-wise error rate (FWER) or false discovery rate (FDR).

**Key Parameters:**
- `pool_sd`: If true, use pooled SD from all groups (like ANOVA MSE); if false, use Welch's t-test for each pair
- `p_adjust_method`: Correction method (Holm, Bonferroni, Hochberg, Hommel, BH, BY, none)
- `alternative`: Direction of alternative hypothesis

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | pairwise.t.test | R 4.3+ |
| stats | R | p.adjust | R 4.3+ |

## Test Cases

### Test 1: Three Groups with Clear Differences (Pooled SD)

**R Code:**
```r
x <- c(1.0, 2.0, 3.0, 2.5, 1.5, 10.0, 11.0, 12.0, 10.5, 11.5, 20.0, 21.0, 22.0, 20.5, 21.5)
g <- factor(c(rep("A", 5), rep("B", 5), rep("C", 5)))

# Pooled SD, no adjustment
pairwise.t.test(x, g, pool.sd = TRUE, p.adjust.method = "none")

#         A           B
# B  1.4e-10       -
# C  7.1e-14  1.4e-10

# With Holm adjustment
pairwise.t.test(x, g, pool.sd = TRUE, p.adjust.method = "holm")

#         A           B
# B  2.7e-10       -
# C  2.1e-13  2.7e-10
```

**Results Comparison:**

| Comparison | R Raw p-value | Rust Raw p-value | Tolerance |
|------------|---------------|------------------|-----------|
| B vs A | 1.4e-10 | ~1.4e-10 | 1e-12 |
| C vs A | 7.1e-14 | ~7.1e-14 | 1e-15 |
| C vs B | 1.4e-10 | ~1.4e-10 | 1e-12 |

**Rust Test:** `crates/p2a-core/src/stats/pairwise.rs::tests::test_validate_pairwise_t_test_pooled_against_r`

### Test 2: P-value Adjustment Methods

**R Code:**
```r
p <- c(0.001, 0.01, 0.05, 0.1)

# Holm
p.adjust(p, method = "holm")
# [1] 0.004 0.030 0.100 0.100

# BH (FDR)
p.adjust(p, method = "BH")
# [1] 0.004 0.020 0.0667 0.100

# Bonferroni
p.adjust(p, method = "bonferroni")
# [1] 0.004 0.040 0.200 0.400
```

**Results Comparison:**

| Input p | Holm (R) | Holm (Rust) | BH (R) | BH (Rust) | Tolerance |
|---------|----------|-------------|--------|-----------|-----------|
| 0.001 | 0.004 | 0.004 | 0.004 | 0.004 | 1e-6 |
| 0.01 | 0.030 | 0.030 | 0.020 | 0.020 | 1e-6 |
| 0.05 | 0.100 | 0.100 | 0.0667 | 0.0667 | 1e-4 |
| 0.1 | 0.100 | 0.100 | 0.100 | 0.100 | 1e-6 |

**Rust Tests:**
- `crates/p2a-core/src/stats/pairwise.rs::tests::test_validate_p_adjust_holm_against_r`
- `crates/p2a-core/src/stats/pairwise.rs::tests::test_validate_p_adjust_bh_against_r`

### Test 3: Missing values and an integer response (R-pinned)

**R Code** (case `stat_pairwise_mahalanobis_na` in `validation/r/cases/stats_onesample.R`;
dataset `validation/datasets/generated/pairwise_mahalanobis_na.csv`, 30 rows: `y` is an Int64
column missing in rows 4 and 18, `g` has the levels g1, g2, g3 of 10 rows each):
```r
set.seed(9)
pw <- pairwise.t.test(d$y, d$g, p.adjust.method = "holm")       # pool.sd = TRUE (R's default)
pw_raw <- pairwise.t.test(d$y, d$g, p.adjust.method = "none")
```

| Comparison | Holm p | raw p | t (cross-check) |
|---|---|---|---|
| g2 vs g1 | 0.0035213461082837507 | 0.0035213461082837507 | 3.221930381127964 |
| g3 vs g1 | 3.0116645128644064e-07 | 1.0038881709548021e-07 | 7.3740995281476955 |
| g3 vs g2 | 0.000830954221002286 | 0.000415477110501143 | 4.0684687051849355 |

Group sizes 9, 9, 10; means 9.555555555555555, 12.444444444444445, 16; pooled SD
1.9020456823582805; df 25 for every comparison. The order (g2, g1), (g3, g1), (g3, g2) is both
R's `lower.tri` order and p2a's storage order.

**Missing values:** `run_pairwise_t_test` and `run_pairwise_wilcox_test` read the response and
the factor in one request through the shared extractor: a row with a missing response or label
is dropped from both (`pairwise.t.test` does the same), and the drop is reported in `missing` and
a last Display line `Missing values: ...`. Integer responses are accepted (before T2-WP06 they
failed as non-numeric) and labels come from the column as stored, so a text label "NA" is a
level like any other (the CSV loader reads the token NA as missing).

**Rust tests:**
- `stats::pairwise::tests::test_validate_pairwise_t_test_na` (adjusted and raw p, t, df, sizes, means, pooled SD)
- `stats::pairwise::tests::test_validate_na_pairwise_tests_missing_rows_and_errors` (both wrappers: `missing`, Display line, text and absent responses; the Wilcoxon p-values are not pinned, STAT-42)
- `tier2::wp06_onesample::test_validate_mcp_hypothesis_pairwise_t_test` (p2a-mcp, the tool with `pool_sd: true`)

**Tolerance:** `validation/README.md`, n < 100: means 1e-6, t and pooled SD 1e-5 relative,
p `Tol::p_value`; df and counts exact.

**Status:** R-pinned (registry `validation/methods/stats_onesample.json`, id
`pairwise_t_test_pooled`).

## Numerical Precision Summary

| Component | Tolerance | Notes |
|-----------|-----------|-------|
| Raw p-values | 1e-12 | Relative tolerance for very small p-values |
| Adjusted p-values | 1e-4 | Some methods involve sorting and cumulative operations |
| t-statistics | 1e-6 | Same as underlying t-test |
| Degrees of freedom | 1e-6 | Integer for pooled SD, Welch-Satterthwaite otherwise |

## Known Differences

1. **Welch's t-test default**: R uses `pool.sd = !paired` (TRUE for unpaired by default), our default is FALSE (Welch's by default), matching modern best practices.

2. **Sorting ties**: When p-values are identical, the order may differ slightly between R and Rust implementations, but final adjusted values are equivalent.

3. **Group order**: the core function orders the groups by their label as text, so numeric labels 2 and 10 come out as 10, 2; R orders the levels of a numeric factor numerically. The p-values per pair are the same; only the order of the rows and columns differs.

## Performance Comparison

| Dataset Size (k groups, n per group) | Comparisons | Rust (µs) | R (µs) | Speedup |
|--------------------------------------|-------------|-----------|--------|---------|
| k=3, n=10 | 3 | 4.2 | 590 | ~140x |
| k=5, n=50 | 10 | 23 | 740 | ~32x |
| k=10, n=100 | 45 | 91 | 1420 | ~16x |
| k=20, n=500 | 190 | 765 | 4870 | ~6x |

**Notes:**
- Rust consistently outperforms R by 6-140x depending on problem size
- Both using pooled SD with Holm adjustment
- R overhead is proportionally larger for smaller problems
- Rust benchmarks from Criterion (median); R benchmarks from system.time (mean of 100 iterations)

## References

- R Core Team. `stats::pairwise.t.test()` function. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/pairwise.t.test.html
- Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure". Scandinavian Journal of Statistics, 6(2), 65-70.
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate". JRSS-B, 57(1), 289-300.
- Benjamini, Y. & Yekutieli, D. (2001). "The control of the false discovery rate in multiple testing under dependency". Annals of Statistics, 29(4), 1165-1188.
- Hochberg, Y. (1988). "A Sharper Bonferroni Procedure for Multiple Tests of Significance". Biometrika, 75(4), 800-802.
- Hommel, G. (1988). "A Stagewise Rejective Multiple Test Procedure Based on a Modified Bonferroni Test". Biometrika, 75(2), 383-386.
