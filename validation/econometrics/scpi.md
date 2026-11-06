# Synthetic Control with Prediction Intervals (SCPI) Validation

## What p2a computes

`run_scpi` (`crates/p2a-core/src/econometrics/scpi.rs`) fits donor weights on the pre-treatment
outcomes and reports a heuristic interval. **This is not the Cattaneo-Feng-Titiunik (2021)
simulation-based construction (CAUS-32, open); the interval width is constant over post periods.**

- Weights: simplex (`min ||y - X w||^2`, `w >= 0`, `sum w = 1`, the exact simplex QP shared with the
  synthetic control; non-unique optima give the minimum-norm weights and a warning), lasso, ridge or
  lasso-simplex. Penalties must be finite, ridge lambda > 0, lasso lambda >= 0.
- In-sample variance: `SSE / T0`.
- Out-of-sample variance: gaussian `SSE / (T0 - J)` (needs T0 > J), subgaussian
  `1.5 (median|e| / qnorm(0.75))^2`, or the mean squared leave-one-out or k-fold prediction error of
  refitted simplex weights (a failing fold is an error).
- Interval: `effect +/- qt(1 - alpha/2, T0 - 1) sqrt(in + out)`; `alpha` must lie in (0, 1).

## Reference and pins

Case `scpi_quantities` in `validation/r/cases/causal_synth_scpi.R` writes p2a's formulas out in R
(weights by `quadprog` with a KKT polish, `stats::qt`) on `synth_qp_panel.csv` (T0 = 14, J = 8). The
R package `scpi` is not installed and not compared.

| Test | Pins | Tolerance |
|---|---|---|
| `test_validate_scpi_simplex_weights` | weights (D01, D04, D05, D07 positive), synthetic path, effects, in-sample variance, pre-RMSPE, pre-R2 | weights 1e-6; statistics 1e-5 |
| `test_validate_scpi_variances` | gaussian, subgaussian, LOO and 5-fold variances | 1e-5 |
| `test_validate_scpi_critical_value` | `qt` at df 1-100 for alpha 0.10, 0.05, 0.01; crit_95 = 2.160369 at df 13; interval bounds at 95% and 90% | 1e-10 relative; 1e-5 |

Before: Frank-Wolfe weights left 0.0004-0.0006 on four donors whose optimal weight is 0 (CAUS-80),
and the critical value was 2.015 at df 13 (a wrong Cornish-Fisher term, z beyond df 30; CAUS-82).

Property tests (not registered): `test_scpi_interval_width_increases_with_level`,
`test_scpi_rejects_invalid_alpha`, `test_scpi_gaussian_needs_more_periods_than_donors`,
`test_scpi_rejects_invalid_lambda`, `test_scpi_rejects_non_finite_outcomes`,
`test_variance_method_parses_aliases_and_rejects_unknown`, `test_scpi_constraint_from_options`, and
the lasso, ridge and lasso-simplex smoke tests.

## Deviations

1. CAUS-32: the interval is p2a's own formula, pinned to itself; not the CFT construction.

## References

- Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). "Prediction Intervals for Synthetic Control
  Methods." *Journal of the American Statistical Association*, 116(536), 1865-1880.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *JASA*, 105(490), 493-505.
- Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). "Efficient Projections onto the
  l1-Ball for Learning in High Dimensions." *ICML 2008*.

## MCP extraction (T2-WP11)

The MCP tool `scpi` reads the unit column as a factor, the time column as a whole-number time
index and the outcome as a numeric column, in one request to the shared extractor with the
`Fail` policy. Units may be text or integer ids; they are the factor levels in R `factor()`
order (integer ids numerically, text in byte order), and the donors are the other units in that
order (before T2-WP11 it was hash order and changed from run to run). Times are the sorted
distinct values. With fewer rows than units x times cells the panel cannot be balanced, and the
tool error names the counts and the first empty cell (`scpi needs a balanced panel: 10 units x
20 times = 200 cells, but the data has 199 rows (unit 2 has no row for time 1940; cells
without a row: 1)`), found from the sorted cells of the rows. Otherwise the outcome grid is
filled in one pass: a second row for a (unit, time) cell (`unit 1 appears twice at time 1935`)
and a missing outcome are tool errors naming the unit and time or the column. Before, the
handler refused integer unit columns ("must be string type"; finding TOOL-5, scpi site).

Tests (`crates/p2a-mcp/src/tools/handlers/causal.rs`, on `validation/datasets/grunfeld.csv`;
not registered, no R value): `test_mcp_scpi_accepts_integer_unit_ids` and
`test_mcp_scpi_reports_unknown_units_and_missing_cells`.
