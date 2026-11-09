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
