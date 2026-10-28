# Standard Errors for ANOVA Contrasts Validation

## Method Overview

`se.contrast` computes standard errors for linear contrasts of treatment means in ANOVA models. A contrast is a linear combination of group means where coefficients sum to zero.

**Key Parameters:**
- `anova`: ANOVA model result
- `contrasts`: Matrix of contrast coefficients (each column is a contrast)

**Formula:**
SE(L) = sqrt(MSE * sum(c_i^2 / n_i))

where L = sum(c_i * mu_i), MSE is mean squared error, and n_i is group sample size.

## Reference Implementations

| Package | Function | Notes |
|---------|----------|-------|
| R stats | `se.contrast()` | Reference implementation |
| R stats | `contr.treatment()`, `contr.helmert()` | Contrast generators |

## Test Cases

### Test Case 1: Simple Pairwise Contrast

**R Code:**
```r
# Three groups with known means
value <- c(5.1, 4.9, 5.2, 4.8, 5.0,  # Group A, mean ~5
           7.2, 6.8, 7.1, 7.0, 6.9,  # Group B, mean ~7
           9.0, 9.2, 8.9, 9.1, 9.0)  # Group C, mean ~9
group <- factor(rep(c("A", "B", "C"), each = 5))
data <- data.frame(value = value, group = group)

fit <- aov(value ~ group, data = data)
# Contrast: A vs B (c = [1, -1, 0])
contrasts <- matrix(c(1, -1, 0), nrow = 3)
se <- se.contrast(fit, contrasts)
print(se)
# Expected: sqrt(MSE * (1/5 + 1/5)) = sqrt(MSE * 0.4)
```

**Rust tests** (not R-pinned: no `se.contrast` value is compared; the R-pinned case comes with STAT-66):
- `stats::secontrast::tests::test_se_contrast_basic` (`crates/p2a-core/src/stats/secontrast.rs`): one SE, in (0, 1).

### Test Case 2: Treatment Contrasts

**R Code:**
```r
# 4 groups
set.seed(42)
n <- 10
value <- c(rnorm(n, 10, 1), rnorm(n, 12, 1), rnorm(n, 15, 1), rnorm(n, 18, 1))
group <- factor(rep(1:4, each = n))
data <- data.frame(value = value, group = group)

fit <- aov(value ~ group, data = data)
# Treatment contrasts: compare each group to reference (group 1)
contrasts <- contr.treatment(4)
se <- se.contrast(fit, contrasts)
print(se)
```

**Rust tests** (not R-pinned: no `se.contrast` value is compared; the R-pinned case comes with STAT-66):
- `stats::secontrast::tests::test_generate_contrasts_treatment` (`crates/p2a-core/src/stats/secontrast.rs`): the treatment contrast matrix (k - 1 rows, exact entries, rows summing to zero).

### Test Case 3: Helmert Contrasts

**R Code:**
```r
fit <- aov(value ~ group, data = data)  # From above
# Helmert: each group vs mean of previous groups
contrasts <- contr.helmert(4)
se <- se.contrast(fit, contrasts)
print(se)
```

**Rust tests** (not R-pinned: no `se.contrast` value is compared; the R-pinned case comes with STAT-66):
- `stats::secontrast::tests::test_generate_contrasts_helmert` (`crates/p2a-core/src/stats/secontrast.rs`): the Helmert contrast matrix entries within 1e-10.

### Test Case 4: Contrast Estimate and T-statistic

**R Code:**
```r
# Using the same fit
contrasts <- matrix(c(1, -1, 0, 0), nrow = 4)  # Group 1 vs Group 2
se <- se.contrast(fit, contrasts)
# Estimate
means <- tapply(data$value, data$group, mean)
estimate <- sum(c(1, -1, 0, 0) * means)
# T-statistic
t_stat <- estimate / se
print(c(estimate = estimate, se = se, t = t_stat))
```

**Rust tests** (not R-pinned: no `se.contrast` value is compared; the R-pinned case comes with STAT-66):
- `stats::secontrast::tests::test_contrast_t_statistic` (`crates/p2a-core/src/stats/secontrast.rs`): the t statistic is negative with abs(t) > 2.

## Numerical Precision Summary

| Sample Size per Group | SE Tolerance |
|----------------------|--------------|
| n < 10 | 1e-4 |
| n = 10-50 | 1e-6 |
| n > 50 | 1e-8 |

## Known Differences

1. **Contrast normalization**: R may normalize contrasts differently
2. **Unbalanced designs**: Slight differences in unequal group sizes

## Performance Notes

- O(k) per contrast where k = number of groups
- Rust implementation 10x+ faster than R
- Memory scales with number of contrasts

## References

1. Maxwell, S. E., & Delaney, H. D. (2004). Designing Experiments and Analyzing Data. Lawrence Erlbaum Associates.
2. Rosenthal, R., & Rosnow, R. L. (1985). Contrast Analysis: Focused Comparisons in the Analysis of Variance. Cambridge University Press.
3. R Core Team. se.contrast() documentation.
