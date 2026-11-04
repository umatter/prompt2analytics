# Tukey's resistant line (`regression::line`)

## Method

The x values are split into thirds; the slope is `(median(y_R) - median(y_L)) / (median(x_R) -
median(x_L))`, polished `iter` times on the residuals, and the intercept is taken from the three
summary points. Related to R's `stats::line`; R >= 3.5's `tukeyline` differs (iteration count, the
intercept as `median(y - b x)`, its third cut points; REG-12, open).

## R reference

Case `line_tukey_outliers` (`validation/r/cases/regression_smoothers_t3.R`): `stats::line(x, y,
iter)` for iter 1, 2, 3 and `stats::lm(y ~ x)` on the old test's 30 points (y = 2 + 0.5 x + noise with
three outliers).

## Values

`iter1_coef` 1.739 / 0.513, `iter2_coef` 1.763 / 0.511, `iter3_coef` 1.763 / 0.511 (awaiting REG-12);
`ols_coef` 2.0836551724137866 / 0.5208609566184653; n = 30. The values the old test quoted
(2.42169727298821 / 0.475089391721295) are R's output on other data (set.seed(42)), not on this y.

## Tolerances

`ols_coef` `Tol::coef(30)`, n exact. The line coefficients are not compared with R (REG-12).

## Tests

- `test_validate_line_resists_outliers` (registry entry `line`): pins `ols_coef` and n through
  `run_ols`, and asserts that the resistant slope is within 0.03 of 0.5 and closer to 0.5 than R's OLS
  slope.
- `test_line_clean_data_recovers_dgp`: the slope within 0.1 of 0.5 and the intercept within 0.5 of
  2.0 on clean data.

## Deviations

REG-12: `iter1_coef`, `iter2_coef`, `iter3_coef`.

## Missing and Non-finite Values (REG-13)

`line()` deletes the pairs with a missing (NaN) or infinite x or y before the fit and
reports them in `LineResult::missing` (`by_column` names `x` and `y`); `x`, `y`, `fitted`
and `residuals` hold the kept pairs in input order. With fewer than 3 complete pairs it is
an error. Sorting and medians use `f64::total_cmp`. Before, a NaN made the sort comparator
inconsistent and the coefficients meaningless. MCP
`regression_line` and `p2a reg line` read x and y with the
incomplete pairs dropped and report them (`missing` in JSON, a `Note:` line in the CLI
text); the dataset paths refuse `±Inf` (`NonFiniteValues`) before `line()` is reached.
With fewer than 3 complete pairs they fail before `line()` too, with `InsufficientData`
naming the method and the columns and counting the dataset's rows ("Tukey's resistant
line of 'y' on 'x': 2 of 6 rows have no missing value").

R: case `reg_smoothers_na` of `validation/r/cases/regression_na.R`
([source](../r/cases/regression_na.R)), 60 points (`set.seed(8)`) in
`validation/datasets/generated/smoothers_na.csv` with x missing in rows 5 and 33 and y in
rows 12 and 48. R's `line(x, y)` uses the 56 complete pairs (`n_complete`), and its
coefficients on the NA data equal those on the complete pairs.

Tests (registry entry `line_missing` in `validation/methods/regression_na.json`), exact:

- `test_validate_reg_smoothers_na_fixture`: the dtypes and missing cells polars reads, 56
  complete pairs, no tied x among them;
- `test_validate_line_na`: `n` is R's `n_complete`; the fit on the NA data equals the fit
  on the complete pairs; an infinite x is deleted too; too few complete pairs is an error;
- `test_validate_mcp_regression_line_na`, `test_validate_cli_reg_line_na`: the same
  through MCP and the CLI.

Property tests, not registered (no R value):
`test_mcp_regression_line_supsmu_too_few_complete_rows` and
`test_cli_reg_line_supsmu_too_few_complete_rows` check the too-few-rows error of line and
supsmu through MCP and the CLI.

### Known difference (REG-12)

p2a's resistant line is a different variant of Tukey's line (the iteration count is off by
one and the intercept comes from the middle group). On the 56 complete pairs R gives
intercept 0.441594195008563728 and slope 0.053890384144849528; p2a gives
-0.514777229630535849 and 0.062911296831234101 (equal to an R transcription of `line.rs`).
The coefficients are a registry deviation until REG-12 (Tier 4) ports `tukeyline`.

### Known difference: infinite values

R's `line()` keeps `±Inf` (it deletes only `NA` pairs, `complete.cases`); p2a's array API
deletes them, and the dataset paths refuse them.

