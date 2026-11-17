# CausalImpact

Status: **R-pinned**: the counterfactual (`validation/methods/ts_statespace.json`, method
`forecasting_causal_impact`, case `structts_nile_ukgas`, `validation/r/cases/ts_statespace.R`) and the intervals,
p-value and control coefficients (`validation/methods/ts_statespace_t4.json`, method
`forecasting_causal_impact_intervals`, case `causal_impact_intervals_t4`, `validation/r/cases/ts_statespace_t4.R`).
Recompute with `Rscript validation/r/run.R --check structts_nile_ukgas causal_impact_intervals_t4`.

The reference is `stats::StructTS(type = "level")` on the pre-period at the tight optimum of its objective plus
`stats::KalmanForecast`, with `qnorm(0.975)` bands. The `CausalImpact` package (bsts) is not installed and is not
the reference: p2a takes the method's idea from it, not its numbers.

## What is computed

1. **Model.** The StructTS engine (`validation/forecasting/structts.md`) on the pre-period: R's prior and objective,
   a local level plus an optional slope and an optional dummy seasonal of `seasonal_period` seasons. With control
   series the variances are fitted to the residuals of an OLS regression of the response on an intercept and the
   controls over the pre-period (`regression::ols::solve_ols`); a regression that leaves no residual variance is
   refused ("the control regression leaves no residual variance in the pre-period; drop controls that reproduce the
   response").
2. **Controls as diffuse regression states** (Durbin and Koopman 2012, sec. 6.2). The response is filtered over the
   pre-period from the fit's prior with its first value as the initial level; each control runs a mean-only pass that
   reuses the response's gains M_t = P_t|t-1 Z and innovation variances F_t (they do not depend on the data). The
   coefficients are the GLS regression of the response's innovations on the controls' (`linalg::solve::least_squares`
   on rows scaled by 1/sqrt(F_t), no intercept: the level absorbs it), Var(beta) = (sum v_t v_t' / F_t)^-1.
   `regression_coefficients` holds one coefficient per control, aligned with `control_names`, and
   `regression_std_errors` their standard errors. An aliased control (one collinear with another, or a constant, whose
   innovations vanish) has NaN coefficient and SE and two warnings.
3. **Counterfactual.** `kalman_forecast` over the gap and the post-period, minus beta' times the controls' forecasts
   T^h a_T(x), plus beta' x_t; in the pre-period the filtered fit, likewise adjusted. The forecast variance gains
   d_t' Var(beta) d_t, d_t the controls' forecast errors; the pre-period bands treat beta as known.
4. **Intervals.** Pointwise: `predicted ± z_critical(1 − alpha) · se`. Cumulative: the variance of the **sum** of the
   post-period forecast errors, which correlate through the state. Cov(e_i, e_j) = Z' T^(j-i) P_i Z for j > i, plus H on
   the diagonal; `cumulative_forecast_variance` sums 1' Sigma 1 in one pass over the horizon (g_h = T g_(h-1) +
   P_h Z, V_h = V_(h-1) + 2 Z'g_h - Z'P_h Z + H over the post-period steps), plus D_h' Var(beta) D_h with D_h the
   running sum of d_t. The running cumulative band uses V at each post step; the average effect's SE is the
   cumulative SE over n_post.
5. **Relative effect.** (Y - P) / P with Y and P the observed and counterfactual post-period sums, and SE
   |Y| / P^2 times the cumulative SE (delta method, Y fixed). NaN, with a warning, when P is 0 or not finite.
6. **p-value.** Two-sided, 2 Phi(-|z|) with z the cumulative effect over its SE (`normal_p_value_two_sided`);
   `significant` is `p_value < alpha`, which holds exactly when the 1 - alpha interval of the cumulative effect excludes
   0. `prob_positive` and `prob_negative` are Phi(z) and Phi(-z).

It is not Bayesian: there is no prior and no posterior. The intervals are the normal approximation under the fitted
model with its maximum-likelihood variances plugged in; they leave out the uncertainty of those variances, which is
why the placebo coverage below sits under the nominal level. The `CausalImpact` package (bsts, a posterior sampler
with spike-and-slab regression) is the method's source, not its algorithm, and its numbers are not reproduced.
Pre-period predictions come from the filtered states, conditioned on y_t, as CausalImpact's do; the summary does not
use them.

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

## The R-pinned intervals

Case `causal_impact_intervals_t4`: the StructTS fit at its tight optimum, `stats::KalmanRun` / `KalmanForecast`
(`nit = -1`), and the joint covariance of the forecast errors written out in R, checked inside the case against the
full H x H matrix (to 1e-9) and, for the local level, against the closed form H^2 P_T + q sum_(i,j) min(i, j) + H h.
The control case writes out the GLS on `KalmanRun`'s innovations (`lm.wfit`) and the d' Var(beta) d terms. All at
`Tol::iterative(1e-6)`.

| Test | Fit | Cumulative SE (R) | Pinned |
|---|---|---|---|
| `test_validate_causal_impact_intervals_nile_level` | level, Nile 1-90, forecast 91-100 | 1013.86729 (marginal sum: sqrt(263,532) = 513.36) | variances, cumulative effect, SE, bounds, average bounds, relative effect and bounds, Phi(z), two-sided p, running band |
| `test_validate_causal_impact_intervals_nile_trend` | local linear trend, same periods | 1060.91921 | the same, slope variance 0 on its bound |
| `test_validate_causal_impact_intervals_ukgas_bsm` | BSM, log10 UKgas 1-92, forecast 93-108 | 1.72003 | the same |
| `test_validate_causal_impact_intervals_control` | level + one control (simulated, effect 3 from t = 81) | 34.06159 | the above, plus beta 2.05990, its SE 0.22234, the counterfactual and its pointwise variance |

## Placebo calibration

`test_causal_impact_placebo_calibration` draws 200 seeded local-level series with no intervention
(`sigma_eta = 0.5`, `sigma_eps = 1`, `n_pre = 80`, `n_post = 20`) and asserts that the fit converges, the one-step
95% interval covers 0 at a rate in `[0.88, 0.995]`, the pointwise intervals over all horizons at a rate in
`[0.88, 0.975]`, the cumulative interval at a rate in `[0.84, 0.96]`, `significant` fires at a rate in `[0.02, 0.18]`
and agrees with the cumulative interval excluding 0, and the average effect is unbiased within 4 Monte Carlo standard
errors. The same design run with `stats::StructTS` and `KalmanForecast` in R (six seeds) gave one-step 0.905-0.965,
pointwise 0.918-0.939, cumulative 0.88-0.92 with the joint covariance (0.35-0.46 with the sum of the marginal variances
used before) and a two-sided false-positive rate of 0.08-0.12. p2a's rates at the test's seed: one-step 0.94,
pointwise 0.934, cumulative 0.93, false positives 0.07. The shortfall from 0.95 is the plug-in of the estimated
variances (R simulation with sigma points over the log-variances: 0.90-0.945); with one random-walk control the R
simulation covered 0.89-0.92 with the diffuse regression states. A property test, outside the registry.

## Tests

- R-pinned: `test_validate_causal_impact_counterfactual_nile`, `test_validate_causal_impact_intervals_nile_level`,
  `test_validate_causal_impact_intervals_nile_trend`, `test_validate_causal_impact_intervals_ukgas_bsm`,
  `test_validate_causal_impact_intervals_control`.
- Naive references: `cumulative_variance_matches_the_full_covariance` (the running variance against the full
  covariance matrix built entry by entry, level, trend, seasonal 5 and trend + seasonal 12, with and without a gap,
  1e-12), `control_gls_solves_the_innovation_normal_equations` (the mean-only passes are the filter, and beta solves the
  GLS normal equations, for level + trend + seasonal 4 with two controls).
- Edge cases: `causal_impact_relative_effect_is_nan_when_the_counterfactual_sums_to_zero`,
  `causal_impact_names_a_control_fit_without_residual_variance`, `causal_impact_constant_control_is_aliased_with_the_level`.
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

## Timing

Quiet integration re-time, 2026-11-16: release builds of v0.5.0 (`f80b0131`) and of the integrated branch,
`--no-default-features --features ci`; one core (`taskset -c 2`), `RAYON_NUM_THREADS=1` (one thread for polars and
BLAS too), cwd `/dev/shm`; per row five processes per side, interleaved with the order alternating (nine for rows
within 1.05-1.15x), each the median of its calls after a warm-up; cells are the median over the processes, the ratio
of the minima in brackets. 1-minute load average 0.7-2.7 on 12 cores, nothing else building. n is the pre-period
length; the post-period has 30 points. Median time per call against v0.5.0:

| Model | n = 100 | n = 1,000 | n = 10,000 | Ratio |
|---|---|---|---|---|
| level | 671 µs / 677 µs | 4.51 ms / 4.46 ms | 38.8 ms / 38.3 ms | 0.99-1.01 |
| level + trend + seasonal 12 | 9.15 ms / 9.1 ms | 165.2 ms / 163.7 ms | 2.35 s / 2.32 s | 0.98-0.99 |
| level, 1 control | 632 µs / 646 µs | 4.55 ms / 4.55 ms | 39.2 ms / 39.6 ms | 1.00-1.02 |
| level, 5 controls | 686 µs / 726 µs | 4.68 ms / 4.84 ms | 40.4 ms / 42.5 ms | 1.03-1.06 |

The controls now cost one mean-only filter pass each and a GLS solve on the n x k innovations, beside the OLS fit the
variances are estimated from. Five controls at n = 100: 1.06x (686 µs / 726 µs, nine rounds), within the
1.15x accepted; signed off 2026-11-16. Peak resident memory of one process with five controls at n = 10,000
(median of three): 14.3 MiB at v0.5.0, 16.2 MiB here (1.13x; signed off 2026-11-17).
