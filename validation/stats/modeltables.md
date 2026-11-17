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

**Rust tests** (not R-pinned: hand-computed values; the R-pinned cases are below):
- `stats::modeltables::tests::test_model_tables_means` (`crates/p2a-core/src/stats/modeltables.rs`): the table type, the grand mean within 0.1 of 7.0, three groups with positive means and a positive standard error (one per term).

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

**Rust tests** (not R-pinned: hand-computed values; the R-pinned cases are below):
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

**Rust tests** (not R-pinned: hand-computed values; the R-pinned cases are below):
- `stats::modeltables::tests::test_model_tables_two_way` (`crates/p2a-core/src/stats/modeltables.rs`): hand-computed grand mean (6.5) and the four cell means, within 1e-10.

### Test Case 4: Two-Way Effects with Interaction

**R Code:**
```r
fit <- aov(value ~ factorA * factorB, data = data)
mt <- model.tables(fit, "effects")
print(mt)
# Shows main effects and interaction effects
```

**Rust tests** (not R-pinned: hand-computed values; the R-pinned cases are below):
- `stats::modeltables::tests::test_model_tables_two_way_effects` (`crates/p2a-core/src/stats/modeltables.rs`): hand-computed main effects (±3.0, ±1.5) within 1e-10 and that an interaction table is returned.

### R-pinned values (T2-WP07)

Two tests pin model tables to the R cases of `validation/r/cases/stats_anova.R`:

- `stats::modeltables::tests::test_validate_model_tables_two_way_empty_cell` (case
  `stat_anova_two_way_unbalanced`, the 3 x 2 layout `y3 ~ fa3 * fb3` with the (C, L) cell
  empty): `df_mse` is 8, the residual df of `aov(y3 ~ fa3 * fb3)` (`empty_resid_df`), and
  `mse * df_mse` is its residual SS 5.166173166666655 (`empty_resid_ss`, `Tol::stat(13)`).
- `stats::modeltables::tests::test_validate_model_tables_one_way_na_labels` (case
  `stat_oneway_na_labels`, the one-way ANOVA on the 5 of 9 rows without a missing value): group
  means a 1.5, b 4.5, c 7 (`aov_group_means`), sizes 2, 2, 1, grand mean 3.8, and `mse * df_mse`
  equal to `aov`'s within SS 1.0000000000000007 on 2 df.

**Residual df rule:** `model_tables_two_way` takes the residual df as n minus the number of
non-empty cells, the residual df of `aov(y ~ a * b)` (and of p2a's two-way ANOVA). Before T2-WP07
it was `n - a * b`, one too few per empty cell (7 instead of 8 here), so its MSE disagreed with
the two-way ANOVA. Empty cells are NaN (see "Empty cells and layout").

Both layouts are unbalanced, so neither has a standard error: as R, `se` is `None` and `se_note`
holds R's message "Design is unbalanced - use se.contrast() for se's".

## Standard errors and the two-way projections (T4-WP31, STAT-75, TOOL-22)

R's `model.tables` reports one standard error per term, and only for a balanced design (every
cell, or every group, has the same number of rows r). With r_term the replication of the term
(n divided by its number of cells):

- `type = "means"`: "Standard errors for differences of means", sqrt(2 MSE / r_term);
- `type = "effects"`: "Standard errors of effects", sqrt(MSE / r_term).

An unbalanced design gets none ("Design is unbalanced - use se.contrast() for se's"). p2a holds
the value in `se` (one element; `ModelTablesSE` holds one per term), names it in `se_kind`
("differences of means" / "effects") and sets `se_note` instead on an unbalanced design.

The two-way tables are the projections of the sequential fit `y ~ a + b + a:b` (`aov`'s term
order), averaged over each term's cells: the a effect is fit(a) - grand, the b effect fit(a + b) -
fit(a), the a:b effect the cell mean - fit(a + b), with fit(a + b) the least-squares additive fit
(a weighted fit on the cell means). Means are the grand mean plus the effects of the term and of
its margins: the a and a:b means are the raw level and cell means, the b means are not the raw
margins on an unbalanced layout. A balanced layout takes a fast path through the raw margins. The
MCP tool `stats_model_tables` returns these tables (`tables`, `n`, `se`, `se_note`).

**R Code:** case `stats_mcp_t4` in `validation/r/cases/stats_mcp_t4.R`, datasets
`validation/datasets/generated/modeltables_t4_{oneway,bal,unbal,empty}.csv`.

| Quantity | R |
|---|---|
| one-way balanced means (`ow_means_g`) | 4.876500000000004, 7.0747500000000025, 8.890750000000002 |
| one-way balanced se: means, effects | 0.5596865442171876 = sqrt(2 MSE / 4), 0.3957581507548378 = sqrt(MSE / 4) |
| one-way unbalanced (3, 4, 5 rows) | no se |
| two-way balanced se means (a, b, a:b) | 0.2263442969035022, 0.2772140168012994, 0.3920398222403207 |
| two-way balanced se effects (a, b, a:b) | 0.16004958722336768, 0.19601991112016037, 0.2772140168012995 |
| two-way unbalanced b means (projections) | 11.302047916666652, 13.21761666666666, 14.548787499999992 |
| the same, raw margins (not reported) | 11.59425, 13.0378, 14.503833333333333 |
| two-way empty cell (a2:b3) | mean and a:b effect NA |

**Rust tests:** `stats::modeltables::tests::test_validate_model_tables_one_way_se_stats_mcp_t4`,
`test_validate_model_tables_one_way_unbalanced_stats_mcp_t4`,
`test_validate_model_tables_two_way_stats_mcp_t4` (means and effects at `Tol::coef(n)`, the
empty-cell effects with an absolute floor of 1e-12 for R's -2.8e-17, MSE `Tol::stat(n)`, se
`Tol::se(n)`, cell sizes and df exact), and through the MCP dispatch path
`tier3::wp56_stats_ts::test_validate_mcp_model_tables_two_way` and
`test_validate_mcp_model_tables_one_way`.

**Status:** R-pinned (registry `validation/methods/stats_mcp_t4.json`, id `model_tables_t4`).

## Empty cells and layout

`model_tables_two_way` takes `data[a][b]`, one vector per factor A x factor B cell. A cell with no
observations, and a factor level with none, has a NaN mean (R: `NA`), and its effect and interaction are NaN; such a layout is
unbalanced, so it has no standard error. A row whose length differs from `data[0]` is an error naming the row.

**R Code:** case `stat_modeltables_t3` in `validation/r/cases/stats_misc_t3.R`, dataset
`validation/datasets/generated/modeltables_t3.csv` (3 x 3 layout, A3:B2 empty, 23 rows).

| Quantity | R |
|---|---|
| cell means (A1:B1, A2:B1, ...) | 4.5263333333333335, 6.398, 7.15625, 6.285, 7.515333333333333, 8.366666666666667, 9.380749999999999, NA, 11.500499999999999 |
| cell sizes | 3, 2, 4, 2, 3, 3, 4, 0, 2 |
| factor A means | 6.111111111111112, 7.527, 10.087333333333333 |
| factor B means (raw, `tapply`; not reported) | 7.074666666666666, 7.0684000000000005, 8.52511111111111 |
| factor B means (`model.tables`) | 6.522573268921095, 7.748625120772947, 8.699301663982823 |
| grand mean | 7.6408695652173915 |
| residual df, residual SS | 15, 7.821394000000002 |

**Rust tests:** `stats::modeltables::tests::test_validate_modeltables_t3_empty_cell`,
`stats::modeltables::tests::test_modeltables_ragged_is_error` (not R-pinned: R has no ragged layout).

**Status:** R-pinned (registry `validation/methods/stats_misc_t3.json`, id `model_tables_t3`).

The factor B means are R's `model.tables` margins of the sequential fit (`b_means_model_tables`),
not the raw `tapply` margins (`b_means_raw`, kept in the case for comparison).

## Numerical Precision Summary

| Design | Mean Tolerance | SE Tolerance |
|--------|---------------|--------------|
| One-way, balanced | 1e-10 | 1e-8 |
| One-way, unbalanced | 1e-8 | 1e-6 |
| Two-way | 1e-8 | 1e-6 |

## Known Differences

1. **Display formatting**: Output format differs

## Performance Notes

- O(n) for one-way ANOVA
- O(a * b * n) for two-way where a, b are factor levels
- Rust implementation 10x+ faster than R

## References

1. Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2005). Applied Linear Statistical Models (5th ed.). McGraw-Hill.
2. Montgomery, D. C. (2017). Design and Analysis of Experiments (9th ed.). Wiley.
3. R Core Team. model.tables() documentation.
