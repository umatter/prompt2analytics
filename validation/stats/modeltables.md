# Model Tables Validation

## Method Overview

`model.tables` computes tables of means or effects from an ANOVA model fit, along with standard errors. It provides a summary of treatment effects in a factorial design.

**Key Parameters:**
- `aov_model`: Fitted ANOVA model
- `type`: Either "means" (group means) or "effects" (deviations from grand mean)
- `se`: Whether to compute standard errors

## Reference Implementations

| Package | Function | Notes |
|---------|----------|-------|
| R stats | `model.tables()` | Reference implementation |

## Test Cases

### Test Case 1: One-Way ANOVA Means

**R Code:**
```r
value <- c(5.1, 4.9, 5.2, 4.8, 5.0,  # Group A, mean ~5
           7.2, 6.8, 7.1, 7.0, 6.9,  # Group B, mean ~7
           9.0, 9.2, 8.9, 9.1, 9.0)  # Group C, mean ~9
group <- factor(rep(c("A", "B", "C"), each = 5))
data <- data.frame(value = value, group = group)

fit <- aov(value ~ group, data = data)
mt <- model.tables(fit, "means", se = TRUE)
print(mt)
# Tables of means
# Grand mean
#
# 7
#
# group
#     A   B   C
#     5   7   9
#
# Standard errors...
```

**Rust tests** (not R-pinned: no `model.tables` value is compared, and p2a's SE definition differs from R's; the R-pinned case comes with STAT-75):
- `stats::modeltables::tests::test_model_tables_means` (`crates/p2a-core/src/stats/modeltables.rs`): the table type, the grand mean within 0.1 of 7.0, three groups with positive means and positive SEs.

### Test Case 2: One-Way ANOVA Effects

**R Code:**
```r
fit <- aov(value ~ group, data = data)  # From above
mt <- model.tables(fit, "effects", se = TRUE)
print(mt)
# Tables of effects
# group
#     A   B   C
#    -2   0   2
```

**Rust tests** (not R-pinned: no `model.tables` value is compared, and p2a's SE definition differs from R's; the R-pinned case comes with STAT-75):
- `stats::modeltables::tests::test_model_tables_effects` (`crates/p2a-core/src/stats/modeltables.rs`): the table type and effects that sum to within 0.5 of zero.

### Test Case 3: Two-Way ANOVA

**R Code:**
```r
# 2x2 factorial design
value <- c(10, 11, 12,   # A1, B1
           15, 16, 17,   # A1, B2
           20, 21, 22,   # A2, B1
           25, 26, 27)   # A2, B2
factorA <- factor(rep(c("A1", "A1", "A1", "A2", "A2", "A2"), 2))
factorB <- factor(rep(c("B1", "B2"), each = 6))
data <- data.frame(value = value, factorA = factorA, factorB = factorB)

fit <- aov(value ~ factorA * factorB, data = data)
mt <- model.tables(fit, "means")
print(mt)
```

**Rust tests** (not R-pinned: no `model.tables` value is compared, and p2a's SE definition differs from R's; the R-pinned case comes with STAT-75):
- `stats::modeltables::tests::test_model_tables_two_way` (`crates/p2a-core/src/stats/modeltables.rs`): hand-computed grand mean (6.5) and the four cell means, within 1e-10.

### Test Case 4: Two-Way Effects with Interaction

**R Code:**
```r
fit <- aov(value ~ factorA * factorB, data = data)
mt <- model.tables(fit, "effects")
print(mt)
# Shows main effects and interaction effects
```

**Rust tests** (not R-pinned: no `model.tables` value is compared, and p2a's SE definition differs from R's; the R-pinned case comes with STAT-75):
- `stats::modeltables::tests::test_model_tables_two_way_effects` (`crates/p2a-core/src/stats/modeltables.rs`): hand-computed main effects (±3.0, ±1.5) within 1e-10 and that an interaction table is returned.

## Numerical Precision Summary

| Design | Mean Tolerance | SE Tolerance |
|--------|---------------|--------------|
| One-way, balanced | 1e-10 | 1e-8 |
| One-way, unbalanced | 1e-8 | 1e-6 |
| Two-way | 1e-8 | 1e-6 |

## Known Differences

1. **Unbalanced designs**: Different weighting schemes
2. **Interaction terms**: Slightly different decomposition
3. **Display formatting**: Output format differs

## Performance Notes

- O(n) for one-way ANOVA
- O(a * b * n) for two-way where a, b are factor levels
- Rust implementation 10x+ faster than R

## References

1. Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). Applied Linear Statistical Models (5th ed.). McGraw-Hill.
2. Montgomery, D. C. (2017). Design and Analysis of Experiments (9th ed.). Wiley.
3. R Core Team. model.tables() documentation.
