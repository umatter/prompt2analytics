# Validation: Ordered Logit and Probit (MASS::polr)

Registry: [`validation/methods/discrete_ordered_count.json`](../methods/discrete_ordered_count.json)
(method `discrete_ordered_logit`). R case: `disc_ordered` in
[`validation/r/cases/discrete.R`](../r/cases/discrete.R).

## Method Overview

`run_ordered_model_with(dataset, y, x, &OrderedConfig { link, levels, max_iter, tol })`
fits the proportional odds model of `MASS::polr`. `run_ordered_logit` and
`run_ordered_probit` call it with the default options and the logit or probit link.
For an outcome with categories `1 < 2 < ... < J`:

    P(y <= j | x) = F(zeta_j - x'beta),   j = 1, ..., J - 1

with `F` the logistic (logit) or standard normal (probit) CDF. There is no intercept:
it is absorbed into the cut points `zeta_1 < ... < zeta_{J-1}`, which are reported as
`thresholds` and labelled `categories[i]|categories[i+1]`, polr's names.

## Category order and `levels`

The categories follow the outcome's data type, as R's `factor()` orders them:

- integer and float columns by value (`-2 < -1 < 0 < ... < 10`, not the string order
  `"-1" < "-2"` or `"10" < "2"`); labels are the decimal labels (`"1"`, `"-2"`, `"1.5"`);
- an Enum column in its declared order;
- booleans `FALSE < TRUE`, dates by time.

A text (`String`) or `Categorical` outcome has no order the program could know. Without
`OrderedConfig::levels` the call fails with `InvalidSpecification`, naming the column and
listing the observed labels (at most 20). With `levels`, lowest category first, the
categories are exactly `levels`. `levels` must name every observed label exactly once; a
label without observations is refused, because polr cannot estimate the cut points next to
an empty category. R sorts text labels alphabetically (`factor()`), which for
`low/mid/high` is the order `high < low < mid`, a different model; p2a asks instead of
guessing (ECON-19).

At most 100 categories are accepted (a method-level cap on the dense Hessian).

## Missing values

The outcome and the regressors are read in one request with joint listwise deletion, as
polr's default `na.action = na.omit`. The dropped rows are reported in
`OrderedResult::missing` (JSON key `missing`, omitted when nothing was dropped) and in
the last line of the text output, for example
`Missing values: 7 of 400 rows dropped because of missing values (y_na: 4, x1_na: 3)`.
A missing outcome is never read as 0 or as an empty category. A text regressor is
`NonNumericColumn`, an unknown column `ColumnNotFound`.

## R reference

Case `disc_ordered` (`validation/r/cases/discrete.R`; dataset
`validation/datasets/generated/ordered.csv`, 400 rows, seed 2026): `y` coded 1/2/3
(Int64), `y_label` the same categories as text `low/mid/high`, regressors `x1`, `x2`,
`y_wide` with 13 integer levels -2..10, and `y_na`, `x1_na` with 4 and 3 missing cells
(7 incomplete rows). `Rscript validation/r/run.R --write disc_ordered` writes it.

```r
y <- cut(ystar, c(-Inf, -0.5, 0.8, Inf), labels = FALSE); lab <- c("low", "mid", "high")[y]
y_wide <- cut(ystar, quantile(ystar, seq(0, 1, length.out = 14)), labels = FALSE, include.lowest = TRUE) - 3L
fy <- factor(dd[[yname]]); tab <- as.vector(table(fy))       # counts in factor() order
polr(factor(y_na) ~ x1_na + x2, ...)                          # na.omit: 393 rows
```

| key | R value |
|---|---|
| `logit_counts` (y 1/2/3) | [174, 90, 136] |
| `wide_levels` | [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] |
| `wide_counts` | [31, 31, 31, 30, 31, 31, 30, 31, 31, 30, 31, 31, 31] |
| `na_n_used`, `na_n_dropped` | 393, 7 |
| `na_missing_y`, `na_missing_x1` | 4, 3 |

## Rust tests

R-pinned (`crates/p2a-core/src/econometrics/discrete/ordered_models.rs`):

- `econometrics::discrete::ordered_models::tests::test_validate_ordered_logit_integer_levels_in_numeric_order`:
  `y_wide` gives the categories `-2, ..., 10` in numeric order and `wide_counts`.
- `econometrics::discrete::ordered_models::tests::test_validate_ordered_logit_enum_outcome_uses_declared_order`:
  `y_label` cast to an Enum `low < mid < high` keeps that order; `logit_counts`.
- `econometrics::discrete::ordered_models::tests::test_validate_ordered_logit_missing_rows_dropped`:
  `y_na ~ x1_na + x2` uses 393 rows, drops 7 (`y_na` 4, `x1_na` 3) and prints the note.

Not R-pinned (error paths and fixtures, same data):

- `econometrics::discrete::ordered_models::tests::test_ordered_text_outcome_requires_levels`:
  a text outcome without `levels` is an error that lists `high, low, mid` and asks for
  `levels`.
- `econometrics::discrete::ordered_models::tests::test_ordered_levels_are_validated`:
  a missing, unobserved, repeated or empty `levels` is refused; `[low, mid, high]` gives
  the counts of the integer fit.
- `econometrics::discrete::ordered_models::tests::test_na_ordered_float_outcome_equals_int64`:
  the same codes as Int64 and Float64 give the same fit.
- `econometrics::discrete::ordered_models::tests::test_na_ordered_column_errors`:
  `NonNumericColumn`, `ColumnNotFound` and no regressor.

## Tolerance

Counts and rows used: exact (`Tol::exact()`).

## Status

R-pinned (registry `validation/methods/discrete_ordered_count.json`, id
`discrete_ordered_logit`): category order, counts and rows used. The estimates follow with
ECON-4 and ECON-5 (T2-WP10).

## Known Differences

- Text outcomes need `levels`, where R sorts the labels alphabetically (ECON-19).
- `validation/expected/ordered_logit_test.csv`, `ordered_logit_multi_test.csv` and
  `ordered_probit_test.csv` are legacy files that no test reads.

## References

- McCullagh, P. (1980). Regression models for ordinal data. *Journal of the Royal
  Statistical Society B*, 42(2), 109-142.
- Venables, W. N., & Ripley, B. D. (2002). *Modern Applied Statistics with S* (4th ed.),
  section 7.3. Springer.
