# MCP data tools: R references (T3-WP59)

The MCP data and chart tools are pinned to R through the cases below. R code:
`validation/r/cases/mcp_data_t3.R` (cases `data_describe_columns`, `viz_dendrogram_hclust`) and the
cases of T3-WP50 (`validation/r/cases/data_munging_t3.R`) and T3-WP52
(`validation/r/cases/data_viz_t3.R`) that the tools reuse. Registry: `validation/methods/mcp_data.json`.

## Fill strategies (`munge_fill_na`, cleaning `fill_na`)
- Reference: `names(which.max(table(x)))` (ties to the first level in sorted order), `tidyr::replace_na`.
- Case `data_munging_fill_mode`: modes k 1, f 2.5, s "a"; filled k 3 1 1 3 1 2 1 5 4 6.
- Keywords mean, median, mode, forward, backward, zero, constant (case-insensitive; aliases ffill/locf,
  bfill/nocb, most_frequent); the constant comes from `value`; unknown strategies are errors.
- Tests: `test_validate_mcp_fill_na_mode` (exact); behaviour tests `test_mcp_fill_constant_keeps_case`,
  `test_mcp_fill_unknown_strategy_is_error`.
- Cleaning sessions store their result only through `output_dataset` (`cleaning_session_apply`,
  `cleaning_rollback`); no R reference (bookkeeping). Behaviour tests
  `test_mcp_cleaning_session_output_dataset_is_stored`, `test_mcp_cleaning_rollback_output_dataset`.

## Value counts (`munge_value_counts`)
- Reference: `dplyr::count(d, g, sort = TRUE)`, `n / sum(n)`; case `data_munging_value_counts`: values 2, 7,
  NA, 5; counts 4, 3, 2, 1; proportions 0.4, 0.3, 0.2, 0.1.
- Test: `test_validate_mcp_value_counts_normalize` (exact).

## Replacing values with missing (`str_replace_value`)
- Reference: `dplyr::na_if`; case `data_munging_na_if`.
- Test: `test_validate_mcp_str_replace_value_null` (exact).

## describe_dataset
- Reference: `summary()`, `sd()`, `quantile(type = 7)`, `head(sort(table(x), decreasing = TRUE), 5)`
  under `LC_COLLATE = C`; case `data_describe_columns` (15 rows: integer, double and integer columns with
  NA; text with tied counts and more than five levels; logical; all-missing).
- Decisive values: income n 12, missing 3, mean 51604.898333333331, sd 8933.5463495481035, quartiles
  43618.024999999994, 53723.68, 59312.740000000005; age sd 11.347653095210074; region top east 4, west 4,
  north 3, south 3; grade 7 distinct, top A 3, B 3, C 2, D 2, E 2.
- NaN cells are missing (R's `is.na`); Inf is a value (mean Inf, sd NaN), from Rust fixtures because the
  harness cannot write NaN or Inf.
- Tests and tolerances: `test_validate_mcp_describe_dataset_summaries` (counts, min, max and level tables
  exact; mean and quartiles 1e-6, sd 1e-5 relative, the validation/README.md tolerances for n < 100),
  `test_validate_mcp_describe_dataset_nan_inf`, `test_validate_mcp_describe_dataset_statistics` (the MCP
  text, 7 significant digits).

## viz_dendrogram
- Reference: `hclust(dist(x))` (complete); case `viz_dendrogram_hclust`: merge rows (-6, -7), (-1, -2),
  (-4, -5), (-3, 2), (1, 3), (4, 5); maximum height 8.6739999999999995; sizes 2, 2, 2, 3, 4, 7.
- Conversion to the tool's 0-based rows: -j -> j - 1, k -> n + k - 1.
- Test: `test_validate_mcp_dendrogram_hclust` (counts exact, height at 4 printed decimals); behaviour test
  `test_mcp_dendrogram_bad_linkage`.

## Chart replies (`viz_residual_diagnostics`, `viz_histogram`)
- `viz_residual_diag` (T3-WP52): `lm(mpg ~ wt)` without hat values flags row 16 (Cook's D
  0.53190562559976873) and sigma 3.0458821247893773; with the hat values of `lm(mpg ~ wt + hp + qsec)`
  none (max 0.3328494305479045), sigma 2.5777848825208176. Test
  `test_validate_mcp_residual_diagnostics_cooks` (rows exact, sigma 7 significant digits).
- `viz_histogram_bins` (T3-WP52): 100 normal draws, 8 Sturges bins, counts 4 7 9 13 31 17 12 7. Test
  `test_validate_mcp_histogram_counts` (exact).
