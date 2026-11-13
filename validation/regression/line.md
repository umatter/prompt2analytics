# Tukey's resistant line (`regression::line`)

## Method

R's `stats::line` (R >= 3.5, `tukeyline`): with `z` the sorted x and `q(p) = (z[floor((n - 1) p)]
+ z[ceil((n - 1) p)]) / 2`, the left third is `x <= q(1/3)` and the right third `x >= q(2/3)`.
From slope 0, each of the `iter` steps (default 1, at least 1) adds `(median(r_R) - median(r_L)) /
(median(x_R) - median(x_L))` with `r = y - slope x`; the intercept is `median(y - slope x)`.
Outer thirds whose x medians are equal leave the line without a slope: an error (R divides by
zero and returns a non-finite slope). At least `MIN_PAIRS` (3) complete pairs.

## R reference

- Case `line_tukey_t4` (`validation/r/cases/regression_smoothers_t4.R`): `line(x, y, iter)` for
  iter 1, 2, 3 on the eleven-point example, on tied x (20 rows on 0..4) and on n = 31, with the
  fitted values and residuals of the n = 31 fit.
- Case `line_tukey_outliers` (`regression_smoothers_t3.R`): `line(x, y, iter)` for iter 1, 2, 3
  and `lm(y ~ x)` on the old test's 30 points (y = 2 + 0.5 x + noise with three outliers).

## Values

`line_tukey_outliers`: `iter1_coef` 1.739 / 0.513, `iter2_coef` 1.763 / 0.511, `iter3_coef` 1.763 /
0.511 (printed to 3 decimals); `ols_coef` 2.0836551724137866 / 0.5208609566184653; n = 30.

## Tolerances

`line_tukey_t4`: the coefficient row of `validation/README.md` (`Tol::coef(n)`).
`line_tukey_outliers`: the line coefficients `Tol::printed(3)` (the case stores R's printed
values), `ols_coef` `Tol::coef(30)`, n exact.

## Tests

- `test_validate_line_tukey_t4` (registry entry `line_tukey`, `regression_smoothers_t4.json`).
- `test_validate_line_resists_outliers` (registry entry `line`): R's coefficients for iter 1-3,
  `ols_coef` and n through `run_ols`, and the resistant slope closer to 0.5 than R's OLS slope.
- `test_line_refuses_equal_third_medians`; `test_line_clean_data_recovers_dgp`.

## Deviations

None. Release note: the line coefficients changed to R's.

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

`test_validate_line_na` also pins R's coefficients on the 56 complete pairs (`line_coef`,
intercept 0.441594195008563728, slope 0.053890384144849528; `Tol::coef(56)`).

### Known difference: infinite values

R's `line()` keeps `±Inf` (it deletes only `NA` pairs, `complete.cases`); p2a's array API
deletes them, and the dataset paths refuse them.

