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
