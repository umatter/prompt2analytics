# Validation: Mauchly's test of sphericity (mauchly.test)

## Method Overview

`p2a_core::stats::run_mauchly_test(data, n, p)` tests whether the covariance matrix of `p`
repeated measures is spherical after projecting out the mean (R's `mauchly.test(fit, X = ~1)`
for `fit <- lm(cbind(t1, ..., tp) ~ 1)`). It reports Mauchly's W, the chi-squared statistic
`-((n - 1) - (2d^2 + d + 2) / (6d)) log W` for `d = p - 1` contrasts, on `d (d + 1) / 2 - 1`
df, its first-order p-value, and the
Greenhouse-Geisser, Huynh-Feldt and lower-bound epsilon corrections (R's
`stats:::sphericity`).

The MCP tool `stats_mauchly_test` reads the repeated-measures columns through the shared
column extractor (`crate::tools::common::numeric_matrix`), so integer columns are accepted.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `mauchly.test`, `stats:::sphericity` | R 4.3.3 |

## R case: `stat_mauchly_na`

`validation/r/cases/stats_mcp.R`, case `stat_mauchly_na`, dataset
`validation/datasets/generated/mauchly_na.csv` (16 subjects: `id, t1, t2, t3, t4`, all Int64).
`t2` is missing for subject 5 and `t4` for subject 11, so 14 subjects are complete. The
variances grow across the conditions, so sphericity is rejected.

```r
fit <- lm(cbind(t1, t2, t3, t4) ~ 1, data = d)    # drops the 2 incomplete subjects
mauchly.test(fit, X = ~1); stats:::sphericity(SSD(fit), X = ~1)
# z and the first-order p-value are mauchly.test.SSD's own intermediate quantities
```

| Quantity | R value |
|---|---|
| W | 0.09759122928472669 |
| chi-squared (R's `z`) / df | 27.277231938098648 / 5 |
| first-order p-value | 5.0371160244126754e-05 |
| Greenhouse-Geisser / Huynh-Feldt / lower bound | 0.561998278305735 / 0.6364951336453561 / 0.3333333333333333 |
| subjects used / in the data | 14 / 16 |
| R's reported p-value (with the second-order term) | 5.450912248903203e-05 (not pinned, STAT-82) |

## Missing values

A subject (row) with a missing value in any of the repeated-measures columns is dropped, as
`lm(cbind(...) ~ 1)` drops it, and reported under `missing` (`n_total`, `n_used`, `n_dropped`,
`by_column`); `n_subjects` counts the complete subjects. A text column is refused with "Column
'x' contains non-numeric values".

`sphericity_violated` is `p < 0.05` when the p-value is finite and `null` otherwise, so a
p-value that could not be computed is never read as "sphericity holds".

## Tests

- `tools::handlers::stats::tests::test_validate_mcp_stats_mauchly_test_na` (p2a-mcp: W,
  chi-squared, df, first-order p, the three epsilons, the subjects used, the verdict and
  `missing`; the refusal of two columns)
- `tools::handlers::stats::tests::test_mcp_stats_mauchly_verdict_is_null_for_non_finite_p`
  (not R-pinned: the verdict of a NaN or infinite p-value is `null`)

**Tolerance:** `validation/README.md`, n = 14 (< 100): W, chi-squared and the epsilons 1e-5
relative (`Tol::stat`), p 1e-5 relative plus 1e-12 absolute (`Tol::p_value`), df and counts
exact.

**Status:** R-pinned with a known deviation (registry `validation/methods/mcp_stats.json`, id
`mcp_stats_mauchly`).

## Known Differences

- **STAT-82 (Tier 3):** p2a reports the first-order chi-square tail, 5.0371e-05 on this case,
  pinned to R's own intermediate `Pr1`; `mauchly.test` adds Box's second-order term and reports
  5.4509e-05.
- **Two columns:** with 2 repeated measures there is one contrast and sphericity holds
  trivially; R reports W = 1, p = 1. The tool refuses fewer than 3 columns, as its request
  documents (p2a's p-value is NaN there, df 0).
