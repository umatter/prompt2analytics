# CausalImpact

Status: **R-pinned** for the counterfactual only (`validation/methods/ts_statespace.json`, method
`forecasting_causal_impact`, case `structts_nile_ukgas`, `validation/r/cases/ts_statespace.R`). Recompute with
`Rscript validation/r/run.R --check structts_nile_ukgas`.

The reference is `stats::StructTS(type = "level")` on the pre-period at the tight optimum of its objective plus
`stats::KalmanForecast`, with `qnorm(0.975)` bands. The `CausalImpact` package (bsts) is not installed and is not
the reference: p2a takes the method's idea from it, not its numbers.

## What is computed

1. **Controls.** With control series, OLS of the response on an intercept and the controls over the pre-period
   (`regression::ols::solve_ols`). The residuals feed the model. An aliased control has a NaN coefficient and its
   adjustment uses the identified coefficients only, as `predict.lm` on a rank-deficient fit; two warnings say so.
2. **Model.** The StructTS engine (`validation/forecasting/structts.md`) on the pre-period: R's prior and objective,
   a local level plus an optional slope and an optional dummy seasonal of `seasonal_period` seasons.
3. **Counterfactual.** The Kalman filter over the pre-period and `kalman_forecast` over the gap and the
   post-period, with the regression added back.
4. **Effects.** Point effect = observed − predicted; bands `predicted ± z_critical(1 − alpha) · se`; the summary and
   the tail probabilities (`Normal::sf` for an upper tail).

It is not Bayesian. The intervals are the plug-in Gaussian forecast of the fitted model, not a posterior: the
"credible interval" wording in the MCP text and the `prob_positive` names are carried over from the method's source.

## Period rules

Both periods are inclusive and in the units of the time column.

| Input | Result |
|---|---|
| Start after end | `pre_period start 39 is after its end 0` (and the same for `post_period`) |
| No time value in the period | `post_period [200, 210] contains no time values (the time column runs from 0 to 79)` |
| Time column not strictly increasing | `time column 'time' must be strictly increasing; row 5 has 3 after 4` |
| Pre-period not ending before the post-period | `Pre-period must end before post-period begins (pre_period ends at time .., post_period starts at time ..)` |
| Pre-period starting before the data | clipped, with the warning `pre_period start -5 is before the first time value 0; the pre-period starts at 0` |
| Post-period ending after the data | clipped, with the warning `post_period end 500 is after the last time value 99; the post-period ends at 99` |
| Fewer than 10 pre-period observations | `InsufficientData` |

`alpha` must be strictly between 0 and 1, `max_iter` at least 1 and `tolerance` finite and positive.
Convergence is reported in `convergence` and `warnings` (`ResultNotes`).

## Seasonal period

`seasonal_period` must be at least 2 and at most the length of the pre-period, checked before anything is sized by
it (`causal_impact_refuses_a_seasonal_period_below_two`,
`causal_impact_refuses_a_seasonal_period_longer_than_the_pre_period`).

## The R-pinned counterfactual

`test_validate_causal_impact_counterfactual_nile`: Nile, pre-period 1-90, post-period 91-100.

| Quantity | Value | Tolerance |
|---|---|---|
| Level variance | 1266.84610 | `Tol::iterative(1e-6)` |
| Observation variance | 15537.38597 | `Tol::iterative(1e-6)` |
| Pre-period log-likelihood | -579.50460 | `Tol::stat(90)` |
| Forecast (all 10 steps) | 888.94647 | `Tol::iterative(1e-6)` |
| Upper 95% band, first step | 1170.61181 | `Tol::iterative(1e-6)` |
| Point effect, first step | 131.05353 | `Tol::iterative(1e-6)` |
| Cumulative effect | -143.46468 | `Tol::iterative(1e-6)` |

## Placebo calibration

`test_causal_impact_placebo_calibration` draws 200 seeded local-level series with no intervention
(`sigma_eta = 0.5`, `sigma_eps = 1`, `n_pre = 80`, `n_post = 20`) and asserts that the fit converges, the one-step
95% interval covers 0 at a rate in `[0.88, 0.995]`, the pointwise intervals over all horizons at a rate in
`[0.88, 0.975]`, and the average effect is unbiased within 4 Monte Carlo standard errors. The windows sit 4-5 Monte
Carlo standard errors from the same design run with `stats::StructTS` and `KalmanForecast` in R (six seeds): one-step
0.905-0.965, pointwise 0.918-0.939. p2a's rates at the test's seed are 0.94 and 0.9365. There is no R value to
compare directly, so this is a property test and stays out of the registry.

## Known deviation

The registry's deviation for finding TS-5: the variance of the cumulative effect is the sum of the marginal forecast
variances, which ignores that forecast errors of a random walk are correlated. In a simulation of a placebo design with
`stats::StructTS` and `KalmanForecast` in R, the 95% cumulative interval covered 0.35-0.46 of the time and its
false-positive rate was 0.62-0.71. Do not read the cumulative interval or the p-value as calibrated.

## Tests

- R-pinned: `test_validate_causal_impact_counterfactual_nile`.
- Input rules: `test_causal_impact_rejects_inverted_periods`, `test_causal_impact_rejects_degenerate_config`,
  `test_causal_impact_clipped_periods_warn`, `test_causal_impact_invalid_periods`,
  `test_causal_impact_insufficient_pre_period`.
- Reporting: `test_causal_impact_reports_convergence`, `test_causal_impact_aliased_control_warns`,
  `test_causal_impact_tail_probability_is_not_zero`.
- Properties on simulated data, no R values: `test_causal_impact_placebo_calibration`, `test_causal_impact_positive_effect`,
  `test_causal_impact_with_controls`, `test_causal_impact_with_trend`, `test_causal_impact_model_info`,
  `test_causal_impact_inference`, `test_causal_impact_cumulative_effect_recovers_dgp`,
  `test_causal_impact_point_effects_recover_dgp`, `test_causal_impact_intervals_are_ordered`,
  `test_causal_impact_series_lengths`.
