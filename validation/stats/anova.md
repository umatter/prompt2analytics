# Validation: One-way and two-way ANOVA (aov, car::Anova)

## Method Overview

`run_one_way_anova(dataset, response, factor)` is `aov(response ~ factor)`: between- and
within-group sums of squares, F on (k - 1, n - k) df, eta-squared and omega-squared, and the
group sizes, means and standard deviations in R `factor()` order (numeric labels numerically:
2 before 10; text labels as stored, without quotes). Tukey's HSD (`run_tukey_hsd`) and the
one-way model tables build on it.

`run_two_way_anova_with(dataset, response, factor_a, factor_b, &TwoWayAnovaConfig { interaction,
ss_type })` fits the two-way model by least squares on a dummy-coded design and reports the
sums of squares of R's three conventions:

- **Type I** (`SsType::I`, the default and what `run_two_way_anova` returns): sequential sums of
  squares, A, then B given A, then A:B given both, as `aov(y ~ a * b)` and `anova(lm(...))`.
  They add up to the total sum of squares. On an unbalanced layout they depend on the order of
  the factors, as in R.
- **Type II** (`SsType::II`): each main effect given the other, the interaction given both, as
  `car::Anova(type = 2)`. With empty cells the terms are still compared as nested models, as car
  does after its note "model has aliased coefficients".
- **Type III** (`SsType::III`): every term given all the others, with sum-to-zero contrasts
  (`contr.sum`), as `car::Anova(lm(..., contrasts = list(a = contr.sum, b = contr.sum)), type = 3)`.
  Type III sums of squares test hypotheses about the parameters of one coding, and only the
  sum-to-zero coding makes them the hypotheses of no main effect averaged over the other factor;
  with treatment contrasts they would test the effect at the baseline level. When empty cells
  leave the model with aliased coefficients, Type III is refused with `InvalidSpecification`, as
  `car::Anova` refuses it ("there are aliased coefficients in the model").

The df of each term are the rank it adds to the design (R's `lm` rule: a column whose residual
on the earlier columns is below 1e-7 of its length is aliased), so empty cells reduce the
interaction df: a 3 x 2 layout with one empty cell has interaction df 1 and residual df
`n - rank`. When empty cells leave the interaction no df, `aov` prints no `a:b` row; p2a sets
`ss_ab`, `df_ab`, `ms_ab`, `f_ab`, `p_ab` and `sig_ab` to `None` (while `with_interaction` stays
`true`) and its Display prints the row as "not estimable". A main effect that enters after the
other factor (B given A in Types I and II, A given B in Type II) loses one df for every
disconnected component of the layout beyond the first: with K components the additive model
has rank `La + Lb - K`, so B given A has `Lb - K` df and A given B `La - K`. It is aliased
entirely, df 0, when it is a function of the other factor: B given A when A is nested in B
(every level of A occurs with a single level of B, so the B indicators are sums of A
indicators), A given B when B is nested in A. A term with df 0 has NaN for MS, F and p; `aov`
drops that row too, and the Display prints it as "not estimable (aliased with the other
factor)".

The fit runs on the non-empty cells (O(n) to build them) and only the additive model is
fitted. The factor with more present levels (A on a tie) is absorbed: its fitted values are the
level means of the cell means weighted by the cell sizes, the same normal equations as the
regression on the rows. The treatment-coded columns of the other factor are residualised on that
projection and on each other, Gram-Schmidt with re-orthogonalisation (one cancellation point per
column), and the four sums of squares (A, B given A, B, A given B) are differences of fitted
vectors. For a subjects x conditions layout (randomised complete blocks, repeated measures) the
cost grows with the number of conditions, not with the number of subjects. The interaction needs
no columns of its own: A + B + A:B spans the indicator of every non-empty cell, so the model with
the interaction is saturated (rank = number of non-empty cells c, residual = within-cell SS), and
A:B given A and B is the additive model's residual between the cells, on `c - rank(A + B)` df.
With the interaction, a Type III main effect is the Wald test of equal unweighted marginal means,
`sum_l (m_l - m_w)^2 / s_l` over the marginal means `m_l` (variance `s_l` sigma^2, `m_w` their
`1 / s_l`-weighted mean), which is what the sum-to-zero coefficients test; it is refused before
any fit when a cell is empty. The cost is O(n + c min(La, Lb)^2) for La and Lb levels.
`stats::anova::tests::test_two_way_anova_shortcuts_match_explicit_interaction_columns`
checks both identities at 1e-10 relative against the fit on explicit interaction columns (cell
indicators, or the products of the sum codes for Type III). The result carries `ss_type` and,
when rows were dropped, `missing`.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `aov`, `summary.aov` | R 4.3.3 |
| car | R | `Anova(type = 2)`, `Anova(type = 3)` | car 3.1.5 |

## Missing values

Rows with a missing response or a missing factor level are dropped, as `aov`'s `na.omit`, and
reported: `missing.n_dropped`, `missing.by_column` and a last Display line `Missing values: 1 of 13
rows dropped because of missing values (y: 1, fa: 1)`. A missing label is never a level of its
own. Integer responses are accepted. Factor levels follow R `factor()` (numeric labels in
numeric order). This holds for the one-way and the two-way ANOVA alike.

## R case: `stat_oneway_na_labels` (one-way)

`validation/r/cases/stats_anova.R`, case `stat_oneway_na_labels`, dataset
`validation/datasets/generated/oneway_na_labels.csv` (9 rows). `y` is an Int64 column missing in
3 rows and `g` is missing in 2, so 5 rows are used (4 dropped). The same case pins Kruskal-Wallis
and `oneway.test` on `v9 ~ g9` (`kruskal.md`, `oneway.md`).

```r
summary(aov(y ~ g, data = d))
```

| Quantity | R value |
|---|---|
| F / p | 21.79999999999999 / 0.043859649122807036 |
| df between / within | 2 / 2 |
| SS between / within | 21.800000000000004 / 1.0000000000000007 |
| rows used, grand mean | 5, 3.8 |
| group sizes | a 2, b 2, c 1 |
| group means | a 1.5, b 4.5, c 7 |

Before T2-WP07 the Int64 response was refused as non-numeric, and with a Float64 response a
missing value was read as 0 and a missing label became a group named `Null`.

**Rust tests** (`crates/p2a-core/src/stats/anova.rs`):
- `stats::anova::tests::test_validate_oneway_na_labels_aov` (F, p, df, SS, rows used, grand
  mean, group sizes and means; labels exactly `a`, `b`, `c`; 4 rows dropped)
- `stats::anova::tests::test_validate_na_one_way_anova_dtypes_and_errors` (Int64 and Float64
  responses give identical results; a text response is `NonNumericColumn`, an absent factor
  `ColumnNotFound`)

**MCP test:** `test_validate_mcp_anova_one_way_na_labels` (`anova_one_way` text at its 4 printed
decimals; group rows `a`, `b`, `c`, no `Null` group, the missing-value line).

## R case: `stat_anova_two_way_unbalanced`

`validation/r/cases/stats_anova.R`, case `stat_anova_two_way_unbalanced`, dataset
`validation/datasets/generated/anova_two_way_unbalanced.csv` (13 rows). The 2 x 2 layout
`y ~ fa * fb` has cells A-L, A-H, B-L, B-H of 2, 4, 4 and 2 rows, and row 13 has `y` and `fa`
missing (12 rows used). The 3 x 2 layout `y3 ~ fa3 * fb3` (13 rows) has the (C, L) cell empty.
"nocell" is the 2 x 2 layout without its (B, H) rows, where `aov` has no `fa:fb` row.

```r
m_int <- aov(y ~ fa * fb, data = d); m_add <- aov(y ~ fa + fb, data = d); m3 <- aov(y3 ~ fa3 * fb3, data = d)
car::Anova(lm(y ~ fa * fb, data = d), type = 2)
car::Anova(lm(y ~ fa * fb, data = d, contrasts = list(fa = contr.sum, fb = contr.sum)), type = 3)
car::Anova(lm(y3 ~ fa3 * fb3, data = d), type = 2)                         # "model has aliased coefficients"
car::Anova(lm(y3 ~ fa3 * fb3, ..., contrasts = contr.sum), type = 3)       # error: aliased coefficients
summary(aov(y ~ fa * fb, data = d[!(d$fa %in% "B" & d$fb == "H"), ]))      # no fa:fb row
```

| Quantity | R value | Quantity | R value |
|---|---|---|---|
| Type I fa SS / df / F / p | 7.707224083333312 / 1 / 10.979337311422938 / 0.01064244904694956 | additive fa SS / F / p | 7.707224083333312 / 2.8431459477118017 / 0.1260425478999001 |
| Type I fb SS / F / p | 124.21955004166654 / 176.95714122675213 / 9.740678499769933e-07 | additive fb SS / F / p | 124.21955004166654 / 45.8238019952318 / 8.183900202554958e-05 |
| Type I fa:fb SS / F / p | 18.78147337500006 / 26.755175295286254 / 0.000850485876591206 | additive residual SS / df | 24.397276125000058 / 9 |
| residual SS / df | 5.615802749999996 / 8 | rows used / dropped | 12 / 1 |
| Type II fa SS / F / p | 40.1011053749999 / 57.12608816255155 / 6.557512579890506e-05 | Type III fa SS / F / p | 40.10110537499995 / 57.126088162551625 / 6.55751257989047e-05 |
| Type II fb SS / F / p | 124.21955004166645 / 176.95714122675201 / 9.74067849976995e-07 | Type III fb SS / F / p | 124.21955004166651 / 176.9571412267521 / 9.740678499769933e-07 |
| Type II fa:fb SS / F / p | 18.78147337500007 / 26.755175295286264 / 0.0008504858765912052 | Type III fa:fb SS / F / p | 18.781473375000047 / 26.755175295286232 / 0.0008504858765912075 |
| Type II residual SS / df | 5.615802749999996 / 8 | additive Type III fa SS / F | 40.10110537499993 / 14.793042736650944 |
| additive Type II fa SS / F / p | 40.101105374999904 / 14.793042736650925 / 0.00392924691676117 | additive Type III fb SS / F | 124.21955004166651 / 45.82380199523182 |
| additive Type II fb SS / F / p | 124.21955004166655 / 45.823801995231804 / 8.183900202554958e-05 | | |
| empty cell: fa3 SS / df / F / p | 38.858623507692336 / 2 / 30.086969409710978 / 0.00018962098657745717 | empty cell Type II fa3 SS / df / F / p | 31.125611091666666 / 2 / 24.099549192424476 / 0.0004106223228395833 |
| empty cell: fb3 SS / df / F / p | 1.1198468166666713 / 1 / 1.7341219979108438 / 0.22435483514379265 | empty cell Type II fb3 SS | 1.1198468166666675 |
| empty cell: fa3:fb3 SS / df / F / p | 0.5787908166666674 / 1 / 0.896277841248001 / 0.3714959790471991 | empty cell Type II fa3:fb3 SS / df | 0.5787908166666655 / 1 |
| empty cell residual SS / df | 5.166173166666655 / 8 | nocell residual SS / df, rows used | 4.35810475 / 7, 10 |
| nocell fa SS / F / p | 5.816952066666669 / 9.34320462734786 / 0.01840962406055401 | nocell fb SS / F / p | 23.199102083333315 / 37.262462446166126 / 0.0004890061588316758 |

Full precision is in `validation/expected/stat_anova_two_way_unbalanced.json`
(`Rscript validation/r/run.R --write stat_anova_two_way_unbalanced`).

**Rust tests** (`crates/p2a-core/src/stats/anova.rs`):
- `stats::anova::tests::test_validate_two_way_anova_unbalanced_type1` (Type I with and without
  the interaction, residuals, rows used and dropped; the Type I parts add up to the total)
- `stats::anova::tests::test_validate_two_way_anova_unbalanced_type2`
- `stats::anova::tests::test_validate_two_way_anova_unbalanced_type3`
- `stats::anova::tests::test_validate_two_way_anova_unbalanced_empty_cell` (the 3 x 2 layout
  in Types I and II, the Type III refusal, and the nocell layout without interaction df)
- `stats::anova::tests::test_validate_na_two_way_anova_dtypes_and_errors` (on the Int64 case
  `stat_friedman_quade`, `y ~ treat + block`: an Int64 response gives exactly the result of its
  Float64 cast; a text response is `NonNumericColumn`, an absent factor `ColumnNotFound`, the
  same factor twice `InvalidSpecification`)
- `stats::anova::tests::test_validate_anova2_rcbd_t4` (case `stat_anova2_rcbd_t4`, registry
  `anova2_rcbd_t4`: an unbalanced 40-subject x 3-condition layout with three empty cells, the
  additive model as `anova(lm(y ~ subj + cond))`, `anova(lm(y ~ cond + subj))` and
  `drop1(test = "F")`; Type I SS, df, F and p in both orders, Type II SS, F and p, residual SS and
  df; this layout is the one where the absorbed factor is the one with many levels)
- `test_two_way_anova_sparse_layout_df_from_rank` (not R-pinned: a 10 x 10 layout with 30
  non-empty cells, df 9 / 9 / 11 / 30 as R `aov` gives them)

**MCP test** (`crates/p2a-mcp/tests/tier2/wp07_anova.rs`):
`test_validate_mcp_anova_two_way_unbalanced` (`anova_two_way` text at its 4 printed decimals).

## MCP tools

- **`anova_two_way` takes `ss_type`:** `"I"` (the default; sequential, as R `aov`), `"II"` or
  `"III"` (as `car::Anova`; also `"1"`, `"2"`, `"3"` and a leading "type"). An unknown value is a
  tool error naming the accepted ones. Before, the field did not exist, so serde ignored it and
  Type I was computed whatever was asked (fa SS 7.7072 on the case above where Type II is
  40.1011). The tool prints the "Sums of squares: Type ..." line of the result.
- **Dropped rows:** `anova_manova`, `anova_tukey_hsd`, `stats_se_contrast` and
  `stats_model_tables` report the rows their core functions drop under `missing` (`n_total`,
  `n_used`, `n_dropped`, `by_column`), with no such key when nothing was dropped; `anova_one_way`
  and `anova_two_way` print the "Missing values" line of their text.
- **Two-way model tables:** `stats_model_tables` with two factors reads the response and both
  factors through the shared extractor in one request, so its cells use exactly the rows of its
  ANOVA, and its levels follow R's `factor()` order (Int64 codes 2 before 10; before, the levels
  were sorted as strings, "10" before "2"). Empty cells keep a mean of 0 (TOOL-22, Tier 4).

The Type II and III values of `stat_anova_two_way_unbalanced` (table above: fa, fb and fa:fb SS,
F and p; the Type II residual SS and df; the additive Type II model) are pinned through
`anova_two_way` at its 4 printed decimals, each df exactly. The `missing` reports are pinned to
R's complete cases: `stat_manova_na` (33 of 36 rows, `y1` missing in 2 and `y2` in 1),
`stat_oneway_na_labels` (5 of 9, `y` 3 and `g` 2) and `stat_loglin_na` for `cnt ~ f2 + f3` (14 of
16, `cnt` 1 and `f2` 1; levels `2, 10` and `x, y`; its model-table cells add up to 14 rows and
its `ss_a` equals the `f2` SS that `anova_two_way` prints for the same columns).

**MCP tests** (`crates/p2a-mcp/src/tools/handlers/stats.rs`):
- `tools::handlers::stats::tests::test_validate_mcp_anova_two_way_ss_type`
- `tools::handlers::stats::tests::test_validate_mcp_anova_handlers_report_missing_rows`

Registry `validation/methods/mcp_stats.json`, ids `mcp_anova_two_way_ss_type` and
`mcp_anova_handlers_missing`.

## Tolerances

`validation/README.md`, n = 5 to 13 (< 100): sums of squares and F 1e-5 relative, p-values
1e-5 relative plus 1e-12 absolute, df and counts exact; MCP text at its 4 printed decimals. The
algorithm reproduces R to about 1e-14 relative on every value above.

## Legacy tests (R-pinned)

Case files in `validation/r/cases/stats_power_loglin.R`; registry `validation/methods/stats_power_loglin.json`,
method `anova_legacy`.

- `stats::anova::tests::test_validate_one_way_anova_against_r`: case `stat_anova_oneway_legacy`
  (`aov(yield ~ fertilizer)`, 15 rows, three levels A, B, C): sums of squares, mean squares, F, eta and
  omega squared at `Tol::stat(15)`, p (1.0316137484534604e-11) at `Tol::p_value(15)`, df and n exact, the
  grand and group means at `Tol::coef(15)`, the group SDs at `Tol::se(15)`, group sizes, minima and maxima
  exact, in level order.
- `stats::anova::tests::test_validate_two_way_anova_against_r` (a balanced 2 x 2 layout, cited in
  `validation/stats/hypothesis_tests.md`): case `stat_anova_twoway_legacy` (`yield` is Int64),
  `aov(yield ~ fertilizer * water)` (`int_*`) and `aov(yield ~ fertilizer + water)` (`add_*`): SS, MS and F
  at `Tol::stat(12)`, p at `Tol::p_value(12)`, df exact for every term, total SS, grand mean
  (`Tol::coef(12)`) and n; the additive fit has no interaction row. On a balanced layout the
  least-squares sums of squares equal the old marginal formulas.

