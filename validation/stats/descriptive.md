# Validation: correlation matrix and descriptive statistics

## Methods

| Core function | R reference | Registry id |
|---|---|---|
| `stats::correlation::correlation_matrix` | `cor(use = "pairwise.complete.obs")` | `correlation_matrix_t4` |
| `stats::descriptive::DescriptiveStats::compute` | `mean`, `sd`, `min`, `max`, `quantile(type = 7)` over the non-missing values | `describe_t4` |

Both registry entries are in `validation/methods/stats_descriptive_t4.json`.

## Correlation matrix

Every numeric column (integer, unsigned and float types) enters; text and other columns are
skipped. Each column is cast to Float64 and read validity-aware: null and NaN cells are missing.

- **Pairwise complete observations.** Entry (i, j) is the Pearson correlation over the rows
  where both columns are present, as R's `cor(use = "pairwise.complete.obs")`. `pair_n[i][j]`
  reports how many rows that is (`pair_n[i][i]`: the present values of column i) and `n_rows`
  the dataset's rows. The text table adds one line, "Pairwise complete observations: 35 to 38 of
  40 rows", when some pair dropped rows.
- **NaN entries, as R.** A pair with fewer than two complete rows, an infinite value among its
  complete rows, or a constant column is NaN. An infinite cell in a row the other column misses
  does not enter that pair. The diagonal is 1 for a column with at least two present values, all
  finite and not all equal, and NaN otherwise.
- **Why pairwise.** R's own default, `use = "everything"`, makes every pair with a missing cell
  NA. Complete cases over all numeric columns (`use = "complete.obs"`) would let a column the
  caller did not ask about drop rows from the pairs it did ask about.

The CLI `viz heatmap` keeps its own complete-case correlation (`validation/methods/cli_stats.json`),
so on data with missing cells its matrix differs from this one.

## Descriptive statistics

For each numeric column, every statistic is computed over the present values: the mean, the
standard deviation (n - 1), the minimum and maximum, and the quartiles and median of R's type 7
(`quantile()`, `summary()`), through `stats::robust::quantile_sorted`. `count` is the number of
present values, `null_count` the number of missing cells (null, and NaN in a numeric column) and
`null_pct` their share of the rows. A column with no present value gets NaN statistics. Text and
categorical columns report the number of distinct values.

## R cases

- `stat_cor_matrix_t4` (`validation/r/cases/stats_descriptive_t4.R`): 40 rows with double columns
  `a`, `b`, `c`, an Int64 column `i` and a text column `lab`, 2-3 missing cells per numeric column
  in different rows (`validation/datasets/generated/cor_matrix_t4.csv`); the pairwise matrix
  (a-b: 0.7013945439754816 over 35 rows), the pair counts, and a two-column case
  (`cor_matrix_small_t4.csv`, a = 1, 2, 3, 4, 100 and b missing in row 5) whose pairwise
  correlation is 0.6.
- `stat_describe_t4`: 12 rows, a double column without missing values, one with a missing cell, an
  Int64 column with two (`describe_t4.csv`), and the quartiles of 1:4 (1.75, 2.5, 3.25).

## Tests

- `stats::correlation::tests::test_validate_correlation_matrix_pairwise_t4`: the 4 x 4 matrix,
  pair counts, rows, and the two-column case.
- `stats::correlation::tests::test_correlation_matrix_infinite_and_nan_cells`,
  `stats::correlation::tests::test_correlation_matrix_pair_with_one_row_is_nan`,
  `stats::correlation::tests::test_to_string_table_pairwise_note`: properties.
- `stats::descriptive::tests::test_validate_describe_quantiles_t4`: count, mean, sd, min,
  quartiles, median and max of the three columns, and the quartiles of 1:4.
- `stats::descriptive::tests::test_describe_quantiles_skip_nan_cells`: a NaN cell is missing.

## Tolerance

`validation/README.md`, n < 100: coefficients `Tol::coef(n)` (1e-6 relative, abs 1e-12); the
describe test also holds sd at `Tol::coef(n)`, tighter than `Tol::se`; counts exact.

## Status

R-pinned: `correlation_matrix_t4` and `describe_t4` in `validation/methods/stats_descriptive_t4.json`.
The MCP `compute_correlation` and `viz_heatmap` tools call `correlation_matrix`; their own output
is not pinned here.
