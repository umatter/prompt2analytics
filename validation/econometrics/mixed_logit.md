# Validation: Mixed Logit (mlogit::mlogit with random parameters)

Registry: [`validation/methods/discrete_mixed.json`](../methods/discrete_mixed.json)
(method `mixed_logit`). Reference: [`validation/r/cases/discrete_mixed.R`](../r/cases/discrete_mixed.R),
cases `disc_mixed_logit` (300 choice situations, one per person) and
`disc_mixed_logit_panel` (100 people x 4 situations). The conditional-logit pin on the same
data is in [mlogit.md](mlogit.md).

`run_mixed_logit(dataset, choice_id, alt_id, choice, x_cols, random_specs, config)` fits
the model of `mlogit::mlogit(choice ~ x + w | 0, rpar = c(x = "n"), R = 50, halton = NA)`,
with `panel = TRUE` when `MixedLogitConfig::individual_col` names the decision maker.
`run_gmnl` and `run_mixl` are wrappers that build the random-parameter specs; `gmnl::gmnl`
and `mixl::mixl` fit the same model with other draws.

## Model

Decision maker n in choice situation t chooses among alternatives j with utility
`U_ntj = beta_n' x_ntj + epsilon_ntj`, epsilon i.i.d. type I extreme value. Each
coefficient is fixed or random across decision makers, with z standard normal:

| Distribution | beta | Parameters |
|---|---|---|
| `Fixed` | `mu` | `mu` |
| `Normal` | `mu + sigma z` | `mu`, `sigma` |
| `LogNormal` | `exp(mu + sigma z)` | `mu`, `sigma` (of the log) |
| `Uniform` | `mu + sigma (2 Phi(z) - 1)` | `mu`, half-width `sigma` |
| `Triangular` | `mu + sigma t(Phi(z))`, `t(u) = 2 sqrt(u) - 1` below 1/2, `1 - 2 sqrt(1 - u)` above | `mu`, half-width `sigma` |

The uniform and triangular `sigma` are half-widths, not standard deviations (ECON-65 is
open). `sigma` is estimated as `|s|` of an unconstrained `s`, so `sigma >= 0`; mlogit keeps
it signed. With finitely many draws the simulated likelihoods at `s` and `-s` differ: on
the panel case the signed likelihood has a second local maximum at
`sd.x = -0.78419952920914038` (log-likelihood -339.96441895353917), above mlogit's
`sd.x = +0.7928` optimum (-340.03563169181729). That maximum is a simulation artefact; the
profile in `sigma >= 0` is unimodal on both cases, so p2a reports the `sigma > 0` optimum
that mlogit reports. Where mlogit returns a negative `sd`, p2a's estimate can differ.

## Draws

mlogit's scheme (`mlogit:::make.random.nb`, `mlogit:::halton`):

- random parameter m (in `x_cols` order) uses the m-th prime of mlogit's list 2, 3, 5, ...,
  61, 71, ..., 199 (45 primes, 67 is not in it); more than 45 random parameters with
  Halton draws is an error;
- the radical-inverse sequence starts with point 0 = 0; the first `halton_drop` points (100)
  are dropped, and `halton_drop = 0` is refused (the normal quantile of 0 is -infinity);
- decision maker u (0-based, in sorted id order) gets the points
  `drop + u R .. drop + u R + R - 1`, through the standard normal quantile;
- without `individual_col` every choice situation is a decision maker; with it the
  individual is, every situation must belong to one individual, and the probabilities of
  an individual's situations are multiplied before they are averaged over its draws.

`halton: false` draws standard normal values from `StdRng` seeded with `seed` (from
entropy when `None`), parameter by parameter, unit-major. The result reports the scheme in
words (`draw_scheme`, e.g. "Halton, prime 2, first 100 points dropped, 50 draws per choice
situation") and the number of units and individuals.

## Estimation

The simulated log-likelihood and its analytic gradient are computed in log space (unit u,
draw r: `l_ur = sum_s ln P_s(chosen | beta_ur)`, the unit contributes
`ln((1/R) sum_r exp(l_ur))`).

1. Stage 1 fits the conditional logit (every coefficient fixed) from 0.
2. Stage 2 starts the mixed logit from the stage-1 means and `s = 0.1`.
3. Each stage runs BFGS on the analytic gradient (start inverse Hessian from the
   central-difference Hessian), then at most 10 Newton steps on the central-difference
   Hessian (step `1e-5 max(1, |theta_k|)`), stopping at the first damped step.
   `max_iter` bounds BFGS iterations plus Newton steps per stage; `iterations` reports
   both stages.
4. Standard errors are the square roots of the diagonal of the inverse of minus that
   Hessian at the estimate. If it is not negative definite, every SE is NaN and a warning
   says so.

`converged` means that the largest gradient component at the reported estimate is at most
`tolerance (1 + |LL|)` (default `tolerance` 1e-6) and the Hessian is negative definite; a
failed line search never counts as convergence. The gradient test depends on the units of
a regressor: the score of `mu` for `c x` is `c` times the score for `x`. The badly scaled
test (`x * 100`, `w / 100`) stays well inside the tolerance. Without random parameters the
estimate is the conditional logit, `n_draws` is 0 and `draw_scheme` is "none (no random
parameters: conditional logit)".

## Missing values

The ids, the alternatives, the choice indicator, the attributes and the individual id come
from one request with `NaPolicy::Fail`: a missing cell is refused with `NullValues` naming
the column, as R's mlogit refuses it (`mlogit_refuses` = [1, 1, 1] in `disc_choice_na` for
a missing choice, attribute and id). Int64 and Boolean columns are accepted; a text
attribute is `NonNumericColumn`.

## R reference

The case file reconstructs mlogit's simulated log-likelihood (asserted equal to
`logLik(mlogit(...))` to 1e-8 at mlogit's estimate, for one and for two random
parameters) and re-solves it to full precision: `optim(BFGS, reltol = 1e-15)` from mlogit's
estimate, then Newton steps on numDeriv's gradient and Hessian (max |gradient| 3.5e-9 and
1.9e-9). SEs come from `numDeriv::hessian` at that point. mlogit's own estimate
(`coef_pkg`, about 2.5e-5 relative from the optimum) is recorded and cross-checked at 1e-4;
its BHHH-style SEs (`se_pkg`, 3-7 % different) are recorded but not pinned.

| Case | Quantity | mean_x | mean_w | sd_x | log-likelihood |
|---|---|---|---|---|---|
| cross-section | `coef` | 0.9470512152252945 | -0.7183429709052912 | 1.3070751532911158 | -276.5251074444699 |
| | `se` | 0.20046392776803168 | 0.12021512327601581 | 0.38794252490063635 | |
| | `coef_pkg` | 0.947067133323218 | -0.7183445241198516 | 1.3071074769935906 | -276.5251074444845 |
| panel | `coef` | 1.102177983733925 | -0.6571884176168149 | 0.7927882700155024 | -340.0356316918173 |
| | `se` | 0.15153404370712772 | 0.08810821951847463 | 0.18628003731162757 | |
| | `coef_pkg` | 1.1021775167920584 | -0.6571899044846461 | 0.7927870403905685 | -340.03563169182104 |

`n_units` 300 / 100, `n_situations` 300 / 400, 50 draws per unit.

Draw layout and fixed points (`disc_mixed_logit`):

| Key | Values |
|---|---|
| `halton_p3_drop100` | `mlogit:::halton(3, 6, 100)` |
| `halton_p2_drop100` | `mlogit:::halton(2, 4, 100)` |
| `halton_p7_3000_tail` | last three of `mlogit:::halton(7, 3000, 100)` |
| `first_draws` | unit 0, draws 0-4: -1.043158263318454, 0.38110545476355645, -0.25739352610093824, 1.2726986411905357, -1.3662038163720984 |
| `unit1_first_draws` | unit 1, draws 0-2: -0.22714306250271524, 1.3417178410802537, -1.2950224067058145 |
| `ln_n_loglik` | -285.1704856092805 at (x -0.1, w -0.7, sd.x 1.0, sd.w 0.4), x log-normal (prime 2), w normal (prime 3), 20 draws |
| `ln_n_grad` | -20.479690697699866, -4.975026685778039, 1.2432919784706993, -4.214789946546363 (numDeriv) |

## Tests and tolerances

R-pinned (`crates/p2a-core/src/econometrics/discrete/mixed_logit.rs` unless noted):

| Test | Quantities |
|---|---|
| `econometrics::discrete::mixed_logit::tests::test_validate_mixed_logit_halton_draws` | Halton points bit-exact; normal draws of units 0 and 1 at 1e-13 (statrs vs R's AS241 `qnorm`); the scheme |
| `econometrics::discrete::mixed_logit::tests::test_validate_mixed_logit_two_random_loglik_gradient` | log-likelihood at a fixed point 1e-11; analytic gradient vs numDeriv 1e-7 (abs 1e-8) |
| `econometrics::discrete::mixed_logit::tests::test_validate_mixed_logit_loglik_at_r_optimum` | log-likelihood at R's optimum 1e-11, cross-section and panel; max \|gradient\| < 1e-6 there |
| `econometrics::discrete::mixed_logit::tests::test_validate_mixed_logit_cross_section` | means and SD 1e-6 relative, SEs 1e-5, log-likelihood 1e-6, counts exact, mlogit's estimate 1e-4; convergence, z, p, AIC |
| `econometrics::discrete::mixed_logit::tests::test_validate_mixed_logit_panel` | the same on the panel case, `n_individuals` 100 |
| `econometrics::discrete::mixed_logit::tests::test_validate_mixed_logit_no_random_is_conditional_logit` | no random coefficient: the polished conditional logit (`clogit_*_tight`), and `run_conditional_logit` at 1e-8 |
| `econometrics::discrete::mixed_logit::tests::test_validate_mixed_logit_badly_scaled_regressors` | `x * 100`, `w / 100`: the scaled R estimate, SEs and log-likelihood, and the scaled conditional logit |
| `econometrics::discrete::mixed_logit::tests::test_validate_gmnl_and_mixl_wrappers` | `run_gmnl` and `run_mixl` give `coef` and `se` |
| `p2a-mcp` `tools::handlers::discrete::tests::test_validate_mcp_mixed_logit` | the MCP text at print precision (`Tol::printed(4)`), `Converged: Yes` |
| `p2a-mcp` `tools::handlers::discrete::tests::test_validate_mcp_mixed_logit_panel` | the same with `individual_id` |
| `p2a-cli` `tier2::wp18_discrete::test_validate_cli_mixed_logit` | CLI text at print precision; JSON means/SD 1e-6, SEs 1e-5 |
| `p2a-cli` `tier2::wp18_discrete::test_validate_cli_mixed_logit_panel` | the same with `--individual person` |

Not R-pinned (properties and refusals):

- `econometrics::discrete::mixed_logit::tests::test_mixed_logit_gradient_matches_central_differences`:
  the analytic gradient equals central differences for the normal, log-normal, uniform and
  triangular laws, with `s` on either side of 0.
- `econometrics::discrete::mixed_logit::tests::test_na_mixed_logit_missing_cells_refused`:
  missing choice, attribute, id and individual id, an absent individual column and a text
  attribute.
- `econometrics::discrete::mixed_logit::tests::test_mixed_logit_individual_must_be_constant_within_situation`.
- `econometrics::discrete::mixed_logit::tests::test_mixed_logit_rejects_unknown_random_variable`:
  also `halton_drop = 0`.
- `econometrics::discrete::mixed_logit::tests::test_mixed_logit_reports_non_convergence`:
  `max_iter = 1` gives `converged = false` and a warning with the gradient.
- `econometrics::discrete::mixed_logit::tests::test_mixed_logit_pseudo_random_draws_are_seeded`.
- `econometrics::discrete::mixed_logit::tests::test_simulated_probability_checks_cancellation_within_a_situation`:
  the budget is checked inside one unit's draws.
- The structure tests on a 5-situation fixture (`test_mixed_logit_structure`,
  `test_mixed_logit_multiple_random`, `test_mixed_logit_lognormal`, `test_mixed_logit_display`)
  and the control checks (`test_mixed_logit_rejects_invalid_controls`,
  `test_mixed_logit_impossible_draw_count_is_an_error`, `test_mixed_logit_stops_when_cancelled`).

## Status

R-pinned (registry `validation/methods/discrete_mixed.json`, id `mixed_logit`).
