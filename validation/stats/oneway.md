# Validation: Welch's One-Way ANOVA (oneway.test)

## Method Overview

Welch's one-way ANOVA tests the equality of means in multiple groups without assuming equal variances. It is the generalization of Welch's t-test to more than two groups.

**Key Parameters:**
- `groups`: Vector of (group_name, values) tuples
- `var_equal`: If true, use standard ANOVA assuming equal variances

**Welch's F Statistic:**
```
F = Σ(w_i * (m_i - m̄)²) / ((k-1) * (1 + 2*(k-2)*tmp))
```

Where:
- w_i = n_i / v_i (weight for group i)
- m_i = mean of group i
- m̄ = weighted grand mean
- tmp = Σ((1 - w_i/Σw_j)² / (n_i - 1)) / (k² - 1)

**Degrees of Freedom:**
- df1 (numerator) = k - 1
- df2 (denominator) = 1 / (3 * tmp)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | oneway.test | R 4.3+ |

## Test Cases

### Test 1: Welch's ANOVA (var.equal = FALSE)
**Description:** Three groups with unequal sample sizes and variances.

**R Code:**
```r
x <- c(1, 2, 3, 4, 5)
y <- c(10, 11, 12, 13, 14, 15)
z <- c(3, 4, 5)
g <- factor(c(rep("A", 5), rep("B", 6), rep("C", 3)))
df <- data.frame(value = c(x, y, z), group = g)
oneway.test(value ~ group, data = df, var.equal = FALSE)

# Expected output:
# F = 46.645, num df = 2, denom df = 6.8877, p-value = 9.905e-05
```

**Rust Test:** `crates/p2a-core/src/stats/oneway.rs::tests::test_validate_oneway_against_r_welch`

**Results Comparison:**

| Metric | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| F statistic | 46.645 | 46.645 | 0.1 | PASS |
| df_num | 2 | 2 | exact | PASS |
| df_denom | 6.8877 | 6.8877 | 0.01 | PASS |
| p-value | 9.905e-05 | 9.905e-05 | 0.00005 | PASS |

### Test 2: Standard ANOVA (var.equal = TRUE)
**Description:** Same data with equal variance assumption.

**R Code:**
```r
oneway.test(value ~ group, data = df, var.equal = TRUE)

# Expected output:
# F = 53.575, num df = 2, denom df = 11, p-value = 2.134e-06
```

**Rust Test:** `crates/p2a-core/src/stats/oneway.rs::tests::test_validate_oneway_against_r_standard`

**Results Comparison:**

| Metric | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| F statistic | 53.575 | 53.575 | 0.1 | PASS |
| df_num | 2 | 2 | exact | PASS |
| df_denom | 11 | 11 | exact | PASS |
| p-value | 2.134e-06 | 2.134e-06 | 0.000001 | PASS |

## Missing values, labels and the R case

`run_oneway_test(dataset, value, group, var_equal)` reads the value and the group in one
request through the shared extractor (`p2a_core::data::columns`):

- **NA policy:** a row whose value or group label is missing is dropped (joint `na.omit`, as
  `oneway.test(value ~ group)`), and the drop is reported in `missing`. A missing label is never
  a level of its own (before T2-WP07 it became a group named `null`).
- **Labels and order:** groups follow R `factor()` order: text by byte order, numeric labels by
  value (an Int64 group column with labels 2 and 10 gives the groups "2", "10"). Text labels
  are printed as stored, without quotes (STAT-25).
- **Dtypes:** integer values are accepted; a text value column is `NonNumericColumn`, an absent
  column `ColumnNotFound`.

**R case** `stat_oneway_na_labels` (`validation/r/cases/stats_anova.R`, dataset
`validation/datasets/generated/oneway_na_labels.csv`, 9 rows; `g9` is missing in 3 of them):

```r
oneway.test(v9 ~ g9, data = d)                     # Welch
oneway.test(v9 ~ g9, data = d, var.equal = TRUE)   # classical F
```

| Quantity | Welch | classical |
|---|---|---|
| F | 13.5 | 13.5 |
| df1 / df2 | 1 / 4 | 1 / 4 |
| p-value | 0.021311641128756727 | 0.021311641128756727 |

(Both groups have variance 1 and 3 rows, so the two tests coincide here.)

**Rust tests:**
- `stats::oneway::tests::test_validate_oneway_na_labels_welch` (F and p at `Tol::stat(6)` /
  `Tol::p_value(6)`, Welch df at `Tol::stat(6)` since Satterthwaite's df is real-valued,
  classical df exact; group labels and rows dropped)
- `stats::oneway::tests::test_validate_na_oneway_dtypes_and_errors` (Int64 labels in numeric
  order, the text and absent-column errors)
- MCP: `test_validate_mcp_hypothesis_oneway_na_labels` (`crates/p2a-mcp/tests/tier2/wp07_anova.rs`)

## Numerical Precision Summary

- F statistics match R within 0.01
- Degrees of freedom match exactly (or within 0.01 for fractional df)
- P-values match R within 0.00001

## Performance Comparison

Benchmarked on 2026-01-20.

| Dataset Size | Rust (µs) | R (µs)  | Speedup |
|--------------|-----------|---------|---------|
| n=100        | ~8        | 740     | ~92x    |
| n=1,000      | ~20       | 1,360   | ~68x    |
| n=10,000     | ~100      | 4,080   | ~41x    |
| n=100,000    | ~800      | 36,480  | ~46x    |

**Performance Evaluation:**
- All sizes show Rust significantly faster than R: PASS
- At least 2 sizes >= 2x speedup: PASS (all 4 sizes show 40-90x)
- n=10,000 >= 1.5x speedup: PASS (~41x)

## References

- Welch, B. L. (1951). "On the Comparison of Several Mean Values: An Alternative Approach". Biometrika, 38(3/4), 330-336.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/oneway.test.html

## Tail accuracy

Both the equal-variance and the Welch p-values are F upper tails computed directly (`f_test_p_value`), not
`1 - cdf`, so tails below 1e-16 are reported as R reports them (6.39e-23 and 5.69e-18 on case `tail_oneway`,
not 0). See [upper_tails.md](upper_tails.md).
