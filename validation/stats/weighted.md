# Validation: Weighted mean and weighted covariance (weighted.mean, cov.wt)

## Method Overview

`p2a_core::stats::weighted_mean(x, w, na_rm)` is R's `weighted.mean(x, w)`:
`sum(w * x) / sum(w)`. `p2a_core::stats::run_cov_wt(data, n_rows, n_cols, weights, compute_cor,
method)` is R's `cov.wt(x, wt, cor, method)`: the weights are normalised to sum to 1, the centre
is the weighted mean of each column, and the covariance divides the weighted cross-products by
`1 - sum(wt^2)` (`method = "unbiased"`, the default) or by 1 (`method = "ML"`).

The MCP tools `stats_weighted_mean` and `stats_cov_wt` read their columns through the shared
column extractor (`crate::tools::common::numeric_columns` / `numeric_matrix`), so integer
columns are accepted and missing values are handled as below.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `weighted.mean`, `cov.wt` | R 4.3.3 |

## R case: `stat_weighted_na`

`validation/r/cases/stats_mcp.R`, case `stat_weighted_na`, dataset
`validation/datasets/generated/weighted_na.csv` (12 rows: `id, v, w, a, b, wc, b_na`). `v` is
missing in row 3 and `w` in row 8 (1-based), so 10 rows have both; `a`, `b` and `b_na` are Int64
columns, and `b_na` is missing in row 5.

```r
cc <- complete.cases(d$v, d$w)
weighted.mean(d$v[cc], d$w[cc]); weighted.mean(d$a, d$wc)
cw <- cov.wt(cbind(a = d$a, b = d$b), wt = d$wc, cor = TRUE)
cw_ml <- cov.wt(cbind(a = d$a, b = d$b), wt = d$wc, cor = TRUE, method = "ML")
cov.wt(cbind(d$a, d$b_na), wt = d$wc)   # error: 'x' must contain finite values only
cov.wt(cbind(d$a, d$b), wt = d$w)       # error: a weight is missing
```

| Quantity | R value |
|---|---|
| weighted mean of `v` on the 10 complete pairs | 10.452435597189695 |
| pairs used, sum of their weights | 10, 17.080000000000002 |
| weighted mean of `a` (Int64) with weights `wc`, sum of `wc` | 52.99383449710296, 13.462 |
| covariance of `a`, `b` (unbiased, row-major) | 89.80989108436671, 31.108720176349422, 31.108720176349422, 32.92152765080039 |
| centre `a`, `b` | 52.99383449710296, 29.287401574803148 |
| correlation (row-major) | 1, 0.5721106993998918, 0.5721106993998918, 1.0000000000000002 |
| covariance (ML) | 80.37100640790817, 27.8392404048959, 27.8392404048959, 29.46152453625662 |
| correlation (ML) | 0.9999999999999998, 0.5721106993998918, 0.5721106993998918, 1 |
| normalised weights `wt` | the 12 values in `validation/expected/stat_weighted_na.json` |
| `n.obs` | 12 |

Two cross-check values are recorded and not pinned: `wmean_na_rm_r`, R's own
`weighted.mean(v, w, na.rm = TRUE)`, is `NA` because `na.rm` drops only missing values and a
kept weight is missing; `wmean_mispaired` (10.440777464788733) is what the MCP tool returned
before it dropped incomplete pairs, when each column's nulls were dropped on their own and the
rest paired by position.

## Missing values

| Tool | Policy |
|---|---|
| `stats_weighted_mean`, `na_rm` true (default) | rows where the value or the weight is missing are dropped together (weighted.mean on `complete.cases(v, w)`) and reported under `missing`; `n` and `sum_weights` count the rows used |
| `stats_weighted_mean`, `na_rm = false` | a missing value or weight is an error naming the column ("Column 'v' contains 1 missing value(s) ..."); R would return `NA` |
| `stats_cov_wt` | a missing value in a column or in the weights is an error naming the column, as `cov.wt` refuses them |

A text column is refused with "Column 'x' contains non-numeric values", an infinite value with
the extractor's `NonFiniteValues` error.

## Tests

- `tools::handlers::stats::tests::test_validate_mcp_stats_weighted_mean_na` (p2a-mcp: the
  weighted mean, `n` and `sum_weights` on the complete pairs, `missing`, the Int64 columns with
  no `missing` key, and the `na_rm = false` refusal)
- `tools::handlers::stats::tests::test_validate_mcp_stats_cov_wt_int` (p2a-mcp: covariance,
  correlation, centre, normalised weights and `n_obs`, unbiased and ML, on Int64 columns; the
  refusals of a missing value and a missing weight)

**Tolerance:** `validation/README.md`, n = 10 to 12 (< 100): weighted means, centres, sums of
weights and normalised weights 1e-6 (`Tol::coef`), covariances and correlations 1e-5 relative
(`Tol::stat`), counts exact.

**Status:** R-pinned with a known deviation (registry `validation/methods/mcp_stats.json`, id
`mcp_stats_weighted`).

## Known Differences

- **TOOL-23:** `stats_cov_wt` uses its `center` field to switch the correlation matrix on or
  off and always centres the data on the weighted means; R's `cov.wt(center = FALSE)` does not
  centre. The tool is pinned only with `center` omitted.
- **R's `NA`:** with `na_rm = true` the tool drops a row whose weight is missing, where R's
  `weighted.mean(na.rm = TRUE)` returns `NA`; with `na_rm = false` the tool refuses the data
  where R returns `NA`.
