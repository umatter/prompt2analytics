# Validation: Chi-Squared Tests and Contingency Tables

## Method and options

Pearson's chi-squared tests as R's `stats::chisq.test` (R 4.3.3): goodness of fit of one
categorical variable to given probabilities, and independence of two categorical variables
(Yates' continuity correction for 2 × 2 tables, R's default `correct = TRUE`).

- Core (`crates/p2a-core/src/stats/chisq.rs`): `chisq_test_gof(observed, probs, rescale_p)` and
  `chisq_test_independence(table, correct)` on counts; `run_chisq_gof(dataset, column, probs)`,
  `run_chisq_gof_with(dataset, column, &GofExpected, rescale)` and
  `run_chisq_independence(dataset, row, col, correct)` on data. MCP tools
  `hypothesis_chisq_gof`, `hypothesis_chisq_independence`; CLI `p2a stats chisq-gof`,
  `p2a stats chisq-indep`.
- Contingency tables (`crates/p2a-core/src/stats/contingency.rs`):
  `contingency_table(df, row, col)` (r × c counts with level labels), `level_counts(df, col,
  extra_levels)` (one column's categories and counts) and `stratified_tables(df, row, col,
  stratum)` (one table per stratum, used by Mantel-Haenszel). Fisher's exact test and the
  chi-squared and Mantel-Haenszel tests build their tables with them.
- `ChiSquaredResult` reports `row_labels` (the goodness-of-fit categories, or the table's row
  levels), `col_labels`, `warnings` and `missing`. `observed`, `expected`, `residuals` and
  `std_residuals` are row-major in `row_labels × col_labels` order.

### Missing values and level order

- Rows with a missing value (null, or NaN in a float column) in a used column are dropped, as
  R's `table()` drops `NA`, and reported in `missing` (Display: a `Missing values:` line; MCP:
  `missing`). A missing value is never a category.
- Levels follow R `factor()` order: numbers by value (2 < 10), text in byte order (R under
  `LC_COLLATE=C`; R in a UTF-8 locale sorts text case-insensitively), booleans FALSE < TRUE,
  dates chronologically, `Categorical` columns by label and `Enum` columns in their declared
  order. Int64 and Float64 columns with the same values give the same table and labels.
- The levels are those of the kept rows. R's `table(x, y)` keeps a level of `x` that occurs
  only where `y` is missing as an all-zero row, and `chisq.test` then returns NaN; p2a drops
  that level.
- Datetime, struct, list and array columns are refused as categorical variables.
- The builders count in one O(n) pass into a buffer checked and allocated through
  `p2a_core::resource`, with `resource::checkpoint()` every 65,536 rows; no polars lazy query
  runs, so they work inside the MCP server's tokio runtime.

### Goodness of fit: probabilities

- `GofExpected::Uniform` (default): every observed category equally likely.
- `GofExpected::Probs(p)` (`run_chisq_gof`'s `probs`, MCP `probs`): one probability per
  category in the order of the sorted categories (`row_labels`; MCP `categories`). A length
  mismatch is an error that lists the categories in that order.
- `GofExpected::Named(pairs)` (MCP `expected`, a category → probability map): categories in
  the map that never occur in the data count 0 (a die face never rolled); an observed category
  without a probability is an error, where R's `factor(x, levels = ...)` would silently turn
  its rows into NA. Keys match the categories by label, or by value on a numeric column
  ("2.0" names the level "2"), through a hash of the canonical key, so a map of `e` entries
  over `k` observed categories costs O(k + e). The MCP tool refuses a map with more entries
  than its `array_length` budget (the limit `probs` has as an array).
  The categories are reported as the observed levels in factor order, then the declared categories
  that never occur, in the order given (MCP: the map's keys in byte order, since the request map is
  sorted). This is R's `table(factor(x, levels = ...))` when the declared order is the observed
  order followed by the extras; the statistic does not depend on the order.
- Probabilities must be finite and non-negative and sum to 1 within
  `sqrt(.Machine$double.eps)` (1.49e-8), as R checks. `rescale` divides them by their sum
  (R's `rescale.p = TRUE`), which must then be positive.

### Statistic, p-value and the small-expected warning

`X² = Σ (O - E)² / E`, `E = n p` (goodness of fit) or `E_ij = r_i c_j / n` (independence),
`df = k - 1` or `(r - 1)(c - 1)`; with Yates' correction (2 × 2 only)
`X² = Σ (|O - E| - 0.5)² / E`. The p-value is the chi-squared upper tail (statrs `sf`).
Pearson residuals are `(O - E) / sqrt(E)`; the standardized residuals of the independence
test are R's `stdres`, `(O - E) / sqrt(E (1 - r_i / n)(1 - c_j / n))`. NaN or infinite counts
are refused (R: "all entries of 'x' must be nonnegative and finite").

When any expected count is below 5, `warnings` holds R's warning text, "Chi-squared
approximation may be incorrect", followed by how many expected counts are below 5. Display
prints it as `Warning: ...`; MCP returns `warnings`.

## R reference

Cases in `validation/r/cases/stats_contingency.R`:

| case | dataset (`validation/datasets/generated/`) | R call |
|---|---|---|
| `stat_contingency_na_codes` | `contingency_na_codes.csv` (12 rows; 3 missing `h`, 1 missing `code`) | `chisq.test(table(g, h))` with and without correction; `chisq.test(table(h))` |
| `stat_chisq_gof` | `chisq_gof.csv` (52 rows; `cat` = a ×5, b ×30, c ×15, 2 missing) | `chisq.test(table(cat))`, `p = c(.1, .6, .3)`, `p = c(.2, .5, .3)`, `chisq.test(table(factor(cat, levels = c("a", "b", "c", "d"))), p = c(.1, .5, .3, .1))`; a die with face 6 never rolled |
| `stat_chisq_small_expected` | `chisq_small_expected.csv` (13 rows) | `chisq.test(table(x, y))`, which warns |
| `stat_fisher_rows_2000` | `fisher_rows_2000.csv` (2000 rows) | `chisq.test(table(treat, outcome))` (Yates) |
| `stat_mantelhaen_na` | `mantelhaen_na.csv` (240 rows; 3 missing strata) | `table(treat, outcome, stratum)` (the builder only) |

### Pinned values (R 4.3.3; the case files hold 17 digits)

| test | X² | df | p | observed | warning |
|---|---|---|---|---|---|
| independence `g × h`, Yates | 0.9506250000000002 | 1 | 0.3295602599606207 | [3, 1, 1, 4] | yes |
| independence `g × h`, no correction | 2.7225 | 1 | 0.09894293606729652 | | yes |
| independence `x × y` (3 × 2) | 1.9345238095238098 | 2 | 0.38012242588687084 | [4, 2, 1, 3, 1, 2] | yes (min expected 1.3846153846153846) |
| independence `treat × outcome`, Yates | 3.042 | 1 | 0.08113589702211416 | [520, 480, 480, 520] | no |
| GOF `h` | 0.1111111111111111 | 1 | 0.7388826803635273 | [4, 5] | yes |
| GOF `cat`, uniform | 19.000000000000004 | 2 | 7.485182988770046e-05 | [5, 30, 15] | no |
| GOF `cat`, p = (.1, .6, .3) | 0 | 2 | 1 | | no |
| GOF `cat`, p = (.2, .5, .3) | 3.5 | 2 | 0.17377394345044514 | | no |
| GOF `cat`, named with unobserved `d` | 6 | 3 | 0.11161022509471258 | [5, 30, 15, 0] | no |
| GOF die, face 6 never rolled | 11.200000000000001 | 5 | 0.04755564396470486 | [9, 11, 8, 12, 10, 0] | |

The expected counts, Pearson residuals and standardized residuals of the `g × h` and `x × y`
tables, and the expected counts and residuals of the goodness-of-fit cases, are pinned as
well. The table of `code` by `out` (rows "2", "10") is [4, 2, 1, 4]; the stratified tables of
`stat_mantelhaen_na` keep 237 of 240 rows in strata s1, s2, s3.

## Tests

- Contingency builder (`p2a-core`, method `contingency_table`):
  `stats::contingency::tests::test_validate_contingency_table_na_codes`,
  `stats::contingency::tests::test_validate_na_contingency_int64_equals_float64`,
  `stats::contingency::tests::test_validate_na_contingency_rejects_datetime_and_absent_columns`,
  `stats::contingency::tests::test_validate_level_counts_extra_levels`,
  `stats::contingency::tests::test_validate_stratified_tables_mantelhaen_na`,
  `stats::contingency::tests::test_validate_contingency_table_inside_tokio_runtime`; and,
  without a case, `stats::contingency::tests::test_na_contingency_nan_float_is_missing` (the
  one inline NaN fixture), `stats::contingency::tests::test_same_category_matches_labels_and_numeric_values`,
  `stats::contingency::tests::test_level_counts_matches_many_extras_by_hash`,
  `stats::contingency::tests::test_contingency_table_of_one_column_with_itself_is_diagonal`
  and `stats::contingency::tests::test_contingency_table_honours_cancellation_and_ceiling`.
- Independence (method `chisq_independence`): `stats::chisq::tests::test_validate_chisq_na_codes`,
  `stats::chisq::tests::test_validate_chisq_small_expected`,
  `stats::chisq::tests::test_validate_chisq_rows_2000`; `p2a-mcp`
  `tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_chisq_independence_rows_2000`
  and `tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_chisq_independence_small_expected`.
- Goodness of fit (method `chisq_gof`): `stats::chisq::tests::test_validate_chisq_gof`,
  `stats::chisq::tests::test_validate_chisq_gof_na_codes`,
  `stats::chisq::tests::test_validate_chisq_gof_errors`,
  `stats::chisq::tests::test_validate_run_chisq_gof_inside_tokio_runtime`; `p2a-mcp`
  `tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_chisq_gof_expected`; and,
  without a case, `stats::chisq::tests::test_chisq_gof_named_matches_numeric_categories_by_value`.
- Inline data with values quoted from R (not registered): the Tier 0 tests
  `stats::chisq::tests::test_validate_run_chisq_independence_from_data_against_r`,
  `stats::chisq::tests::test_validate_run_chisq_independence_on_multi_chunk_frame_against_r`,
  `stats::chisq::tests::test_validate_run_chisq_independence_counts_both_signed_zeros` and
  `stats::chisq::tests::test_validate_run_chisq_independence_on_categorical_and_enum_columns_against_r`,
  and the array-level `stats::chisq::tests::test_validate_gof_against_r`,
  `stats::chisq::tests::test_validate_independence_against_r` and
  `stats::chisq::tests::test_validate_2x2_yates_against_r`.
- Registry: `validation/methods/stats_contingency.json`, methods `contingency_table`,
  `chisq_gof` and `chisq_independence`.

## Tolerance

The statistic and p-value rows of `validation/README.md` (`Tol::stat(n)`, `Tol::p_value(n)`;
n = 9 to 60: 1e-5 relative, n = 2000: 1e-8), the same for expected counts and residuals;
counts, df and levels exact.

## Known differences

1. Empty levels are never a table row or column (see "Missing values and level order").
2. A named goodness-of-fit map must list every observed category (R would turn the rows of an
   unlisted category into NA and drop them).
3. A probability of 0 for an unobserved category contributes 0 to X² (R's
   `(0 - 0)^2 / 0` is NaN).
4. R's `simulate.p.value` is not implemented.

## Performance

Measured before Tier 2; the array-level algorithm (`chisq_test_gof`, `chisq_test_independence`)
is unchanged.

### Goodness-of-Fit Test

| Categories (k) | Rust (µs) | R (µs) | Speedup |
|----------------|-----------|--------|---------|
| k=5 | 0.17 | 100 | ~588x |
| k=10 | 0.21 | 40 | ~190x |
| k=20 | 0.27 | 40 | ~148x |
| k=50 | 0.43 | 40 | ~93x |
| k=100 | 0.63 | 40 | ~63x |

### Test of Independence

| Table Size | Rust (µs) | R (µs) | Speedup |
|------------|-----------|--------|---------|
| 2×2 | 0.39 | 100 | ~256x |
| 3×3 | 0.65 | 80 | ~123x |
| 5×5 | 1.05 | 80 | ~76x |
| 10×10 | 2.80 | 80 | ~29x |
| 20×20 | 9.60 | 100 | ~10x |

**Benchmark Notes:**
- Rust benchmarks: Criterion with 100 samples, median times reported
- R benchmarks: system.time() with 50 replications, mean times reported
- Environment: Rust release build, R 4.3+

## References

- Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in
  the case of a correlated system of variables is such that it can be reasonably supposed to
  have arisen from random sampling". *Philosophical Magazine*, Series 5, 50(302), 157-175.
- Yates, F. (1934). "Contingency tables involving small numbers and the χ² test".
  *Supplement to the Journal of the Royal Statistical Society*, 1(2), 217-235.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/chisq.test.html
