# LTMLE (Longitudinal Targeted Maximum Likelihood Estimation) Validation

## 1. Method

`run_ltmle` / `ltmle` (`crates/p2a-core/src/econometrics/ltmle.rs`) estimate the mean outcome
`E[Y^d]` under a treatment regime `d = (d_1, ..., d_T)` over `T >= 2` time points with
time-varying confounding (nodes `L_1, A_1, ..., L_T, A_T, Y`), and the contrast between two
regimes, by the longitudinal TMLE of the R package `ltmle` 1.x (Lendle et al. 2017) with its
defaults: `stratify = FALSE`, main-term `glm` models on the full history, `gbounds` on the
cumulative probability of following the regime, and the influence-curve variance.

## 2. Reference implementation

The reference is the estimator written out in R with `glm` and `lm`
(`validation/r/cases/causal_ltmle.R`), so that every intermediate (H, cumulative g, epsilons, Q
and g coefficients) can be pinned. The case cross-checks it against `ltmle::ltmle` 1.3-0
itself: the logistic-Q regime means and EIC standard errors of the two-period binary case
(always, never, dynamic), of `gbounds = c(0.1, 0.5)`, of the rescaled continuous outcome and of
the three-period case (always, never, dynamic, (1, 0, 1)) must agree to 1e-8 relative, or the
case stops (`ltmle(.., SL.library = "glm", stratify = FALSE, variance.method = "ic")`; ltmle
fits at glm's default control, about 1e-11 from the tight fits). R 4.3.3; `packages = {ltmle}`.
The linear Q model is p2a's own variant and has no ltmle counterpart: it is pinned to the
written-out reference only.

## 3. Algorithm as implemented

Nodes are ordered `L_1, A_1, L_2, A_2, ..., L_T, A_T, Y`; time points are 1-based in messages.

1. **History designs.** The g design of node t has the columns `(Intercept)`, then for every
   earlier node its covariates and its treatment, then the covariates of node t (R's
   `reformulate` order: `A2 ~ L0 + A1 + L1`). The Q design appends `A_t`
   (`Y ~ L0 + A1 + L1 + A2`). The designs are built once per call.
2. **g.** For each node a binomial logit GLM (glm.fit's IRLS; see "Nuisance fits" below) of
   `A_t` on its g design, on all subjects. Its fitted values are reported as
   `propensity_scores` (unbounded).
3. **Cumulative g (`CalcCumG`, `SetA`).** g is predicted with every earlier `A_s` set to the
   regime; the probability of the regime's treatment is `p` or `1 - p`; the cumulative product
   `c_t` is bounded to `gbounds` (default `(0.01, 1)`), not each factor. `n_bounded` counts the
   subjects outside the bounds, `n_bounded_following` those among the regime's followers.
4. **Clever covariate.** `H_t = I(Abar_t = dbar_t) / c_t` (0 for non-followers).
5. **Backward pass (`UpdateQ`).** From `t = T` to 1, starting from `Q*_{T+1} = Y`: a
   quasibinomial logit GLM of `Q*_{t+1}` on the Q design (pooled over all subjects), predicted
   on the link scale with every `A_s, s <= t`, set to the regime; the fluctuation is the
   intercept-only quasibinomial fit with offset `logit Q_t(d)` and weights `H_t` (zero weights
   drop the non-followers); `Q*_t = plogis(logit Q_t(d) + eps_t)` for every subject. The last
   node's Q regression has the same response and design for every regime and is fitted once.
6. **Estimate and EIC (`CalcIC`).** `psi = mean(Q*_1)`,
   `EIC = sum_t H_t (Q*_{t+1} - Q*_t) + Q*_1 - psi`.

Aliased columns (a duplicated history column, say) get NaN coefficients and predictions use the
identified columns, as `predict.glm`; the aliasing, convergence and separation warnings of every
fit are reported with the model they come from.

### Nuisance fits

Every logistic g and Q regression first runs `econometrics::glm::lean_logit_refined`: the same
Fisher scoring and deviance rule as `fit_glm`, through the normal equations, with the refinement
step the shared GLM takes when `kappa_1` of the equilibrated `X'WX` is above 100. It computes
only what LTMLE reads. A fit leaves for `fit_glm` from glm.fit's start, which then reports it,
when it would report anything: a linear predictor at the link's clamp, a rank-deficient design
or `kappa_1` above 1e4, no convergence, a binomial response that is not 0/1. A response whose mean
is within 1e-6 of 0 or 1 goes to `fit_glm` directly.

The Q regression of node `t - 1` starts from node t's coefficients on the shared history, with
`L_t`, `A_t` and `eps_t` moved into the intercept at their means (`warm_start`); its response is
node t's targeted prediction. It takes glm.fit's start instead when `max_iter` is below 50: the
limit decides what is reported as not converged, on the step count from glm.fit's start, which a
warm start would undercut. Fits that can take the warm path needed at most 33 steps from
glm.fit's start (T = 10, n = 1,500 to 10,000, 12 data seeds each), and a response mean within
1e-6 of 0 or 1 never takes it.

Two reported things change. `NodeModel::iterations` of a warm-started node counts the steps from
its warm start (2 where glm.fit's start takes 4 or 5), so it is not comparable with glm.fit's.
And each fit stops at the deviance rule on a path that depends on its start, so at the default
tolerance the numbers move by the rule's slack. The tests hold every reported quantity to
`fit_glm` from glm.fit's start. Largest measured gaps over 64 data sets at the default tolerance
and 18 at 1e-14 (T = 2 to 10, saturated, separated, exact-duplicate and near-duplicate designs,
dynamic regimes), each relative to the largest entry of the same vector:

| Quantity | default tolerance (1e-10) | tolerance 1e-14 | near-duplicate at 1e-7 |
|---|---:|---:|---:|
| ATE, regime means | 2e-12 | 3e-13 | 2e-11 |
| SE, z | 5e-12 | 3e-13 | 3e-11 |
| p-value | 1e-9 | 1e-11 | 1e-9 |
| EIC | 1e-11 | 2e-12 | 2e-10 |
| Q* | 4e-11 | 1e-11 | 1e-9 |
| Q coefficients | 1e-10 | 2e-11 | 4e-9 |
| fluctuation `eps` | 4e-10 | 2e-11 | 7e-10 |
| g coefficients, propensities, cumulative g, H | 5e-15 | 2e-15 | 1e-15 |

The tests assert 1e-10 on ATE, SE, z and regime means everywhere, 1e-10 on the rest at 1e-14,
and 1e-9 on the rest at the default tolerance. The last column has its own bound of 1e-8: there
the condition number of about 1e7 amplifies the rounding that reaches the clever covariate from
the g fits, on the near-duplicate's coefficient and the predictions, and the gap is the same with
the warm start off. On a saturated node the likelihood is flat, and those fits run `fit_glm` from
glm.fit's start; relative to the coefficient vector their gap stays in the table.

On the default tolerance the warnings and convergence status were equal on all 64 data sets. At
`max_iter` 2 to 6 and 49 they are equal by construction. At 1e-14, where rounding decides the
deviance rule, they differed on 7 of 18 data sets: on most of them the fast path converges where
`fit_glm` from glm.fit's start runs out of steps, with the numbers within the 1e-14 column.

There is no warm start across regimes. The last node's Q regression is fitted once for all
regimes, and every earlier one already starts from its own regime's later node and takes about
two steps, so a start from the other regime has few steps left to save. The fits that stay
expensive are the saturated ones, which keep glm.fit's start ("Wall-clock against fe13b889").

## 4. Regimes and contrasts

- `Regime::Static(v)`: one value for every node (`Regime::always()`, `Regime::never()`) or one
  value per node (`Static(vec![1, 0])`).
- `Regime::Dynamic(rules)`: exactly one `DynamicRule` per node,
  `d_t = above if L > threshold else 1 - above`, on a covariate observed at or before the node
  (`Regime::threshold_rules(columns, c)` gives `A_t = 1{L_t > c}`; ltmle's `rule`).
- `LtmleConfig.contrast`: the reference regime (default never treat). The result's `ate`, `se`,
  `z_stat`, `p_value` and CI are `psi(intervention) - psi(contrast)` with the differenced EIC;
  `None` estimates the single mean and leaves the contrast fields NaN.
- `RegimeKind` and `LtmleQModel` parse option strings (`"always"`, `"never"`, `"dynamic"`;
  `"logistic"`, `"linear"` and aliases).

## 5. Continuous outcomes

With the logistic Q model an outcome outside [0, 1] is mapped to [0, 1] by its range before the
fits, and psi, the EIC and `targeted_predictions` are scaled back (ltmle's behaviour with
`Yrange = NULL`); the result reports `outcome_range`. `LtmleQModel::Linear` is p2a's Gaussian
variant: least-squares Q fits (OLS's normal equations, the pivoted QR when ill-conditioned) and
`eps_t` the `H_t`-weighted mean residual among the followers, `Q*_t = Q_t(d) + eps_t`.

## 6. Inference

`SE = sd(EIC) / sqrt(n)` (ltmle `variance.method = "ic"`), CIs `psi -/+ qnorm(0.975) SE`, the
p-value `2 Phi(-|z|)`. ltmle's default `variance.method = "tmle"` is a substitution variance
estimator that needs extra targeted fits per node ("takes significantly more time", ltmle NEWS)
and is unavailable for non-binary outcomes; p2a keeps the influence-curve variance as a
documented convention, and the output says so.

## 7. R case

`validation/r/cases/causal_ltmle.R`, case `ltmle_two_period`, dataset
`validation/datasets/generated/ltmle_sim.csv` (600 rows; `L0, A1, L1, A2, Y` the two-period
binary design, `Yc` a continuous outcome, `R1, Lr, R2, Yr` a randomized design with a known
effect of 5.5, `L2, A3, Y3` the third time point). Key values:

| Quantity | Value |
|---|---|
| psi always / never (binary, two periods) | 0.5736132092515959 / 0.3255881665399437 |
| ATE (SE), z, p | 0.24802504271165215 (0.05306179615051192), 4.674267753924484, 2.950042547809292e-06 |
| psi dynamic `A_t = 1{L_{t-1} > 0}` (SE) | 0.45392016719826284 (0.032384516421308296) |
| psi (1, 0) | 0.5956272044613323 |
| gbounds (0.1, 0.5): psi never | 0.32392409968247965 (per-factor bounds would give 0.32635506045819096) |
| rescaled Yc: ATE (SE) | 2.808160267792386 (0.13487370329062567) |
| linear Yc: ATE (SE) | 2.804806711524769 (0.13571364292968557) |
| randomized: ATE (SE), truth 5.5 | 5.647189491841183 (0.14222217207100085) |
| three periods: ATE (SE) | 0.3953563630362491 (0.07074698162059881) |

## 8. Tests and tolerances

The registry entry is `validation/methods/causal_ltmle.json` (method `causal_ltmle`). At
n = 600: coefficients, means, epsilons, H and cumulative g 1e-8 relative; SEs, z and CI bounds
1e-6; p-values 1e-6 relative / 1e-12 absolute; counts exact. p2a runs at `tolerance = 1e-14`,
the case's `glm.control(epsilon = 1e-14)`; the default tolerance 1e-8 is checked against the
same tight values at `Tol::iterative(1e-8)` (measured gaps: 2.8e-13, 1.1e-11, 6.0e-15 and
4.1e-10 for psi always, psi never, the ATE SE and the never epsilons).

## 9. Planning checks (plan T3-WP27)

- Saturated models (n = 2000, discrete L0 and L1, fully interacted g and Q): the TMLE equals the
  hand-computed g-formula to 1.1e-16 for (1, 1), (0, 0) and (1, 0); every epsilon is below 4e-15.
- Monte Carlo (400 replications of the case DGP at n = 600): ATE bias -0.0057 (truth 0.34616),
  mean EIC SE 0.0507 against the Monte Carlo SD 0.0489, 95% coverage 0.948 (always 0.958, never
  0.955, dynamic 0.970).
- Nonparametric bootstrap on the case data (B = 400): ATE SE 0.05278 against the EIC SE 0.05306.

## 10. Deviations from design A.24

1. `Q_t` is predicted with every `A_s, s <= t`, set to the regime (ltmle's `SetA`); A.24 set
   only `A_t`. The design values move (psi always 0.5736124939592598 -> 0.5736132092515959,
   ATE SE 0.0530591757778956 -> 0.05306179615051192).
2. `gbounds` bound the cumulative probability (ltmle's `CalcCumG` + `Bound`), default
   `(0.01, 1)`; A.24 bounded each factor at 0.01. The default bounds bind for no follower on
   the case data; the case pins `(0.1, 0.5)`, which binds.
3. The case keeps its id but also pins a continuous outcome, a randomized design with a known
   effect, three time points and the mixed static regimes (1, 0) and (1, 0, 1).

## MCP extraction (T2-WP11)

The MCP tool `ltmle` reads every node (the outcomes, the treatments and each time point's
comma-separated covariates) in one request to the shared extractor with the `Fail` policy. p2a's
LTMLE has no censoring node, and dropping a row would change the sample of every time point, so
a missing (null or NaN) value is a tool error naming the column. Integer and boolean columns are
numeric; no data problem raises a protocol error. Before T2-WP11 outcomes and treatments were read
with `into_no_null_iter`, so a missing final outcome counted as Y = 0 and a missing treatment as
untreated, with the full row count reported, and missing covariates became NaN and failed as a
singular matrix (finding TOOL-7, the handler part of CAUS-24).

Tests (`crates/p2a-mcp/src/tools/handlers/causal.rs`, on `rd_sim.csv` with arbitrary node roles;
not registered): `test_mcp_ltmle_accepts_int64_treatment` and `test_mcp_ltmle_refuses_missing_values`.

## 12. Limitations

- No censoring (C) nodes: missing values are refused.
- A single outcome node at the last time point; non-zero earlier `outcomes` entries are ignored
  with a warning (no intermediate or survival outcomes).
- No marginal structural models (`ltmleMSM`).
- No SuperLearner: main-term logistic (or linear) models only, no `Qform`/`gform` override.
- No `stratify = TRUE`.
- No `variance.method = "tmle"` (section 6).
- The full-history designs grow with the number of time points: node t has
  `1 + sum_{s <= t} k_s + t` columns, so the cost per regime grows as `T^3` (as in ltmle).
  See "Wall-clock against fe13b889" for the cost when a fluctuation saturates a regime.

## Wall-clock against fe13b889 (T3-PERF6B)

Medians in ms over five interleaved rounds of the wave-2B timing harness (`wp27`, one core, one
rayon thread, `release`), on a shared machine. "Before ratio" is this tree before T3-PERF6B; ratios
are against fe13b889, whose estimator is not comparable: it fitted shorter histories (its T = 2
growth ATE is -0.0029, this tree's 0.2397). The n = 100 rows are near the clock's resolution.

| Row | n | fe13b889 | this tree | before ratio | ratio |
|---|---|---:|---:|---:|---:|
| bench | 100 | 0.22 | 0.11 | 0.91 | 0.50 |
| bench | 1,000 | 1.69 | 0.94 | 0.96 | 0.56 |
| bench | 10,000 | 16.4 | 9.2 | 0.96 | 0.56 |
| linear Q | 100 | 0.06 | 0.06 | 1.32 | 0.89 |
| linear Q | 1,000 | 0.48 | 0.44 | 1.33 | 0.93 |
| linear Q | 10,000 | 4.7 | 4.5 | 1.40 | 0.97 |
| binary Y | 100 | 0.15 | 0.30 | 2.53 | 2.05 |
| binary Y | 1,000 | 1.20 | 1.08 | 1.55 | 0.90 |
| binary Y | 10,000 | 11.6 | 10.5 | 1.53 | 0.91 |
| growth T = 2 | 10,000 | 9.9 | 9.0 | 1.51 | 0.91 |
| growth T = 3 | 10,000 | 13.7 | 15.3 | 1.87 | 1.11 |
| growth T = 5 | 10,000 | 23.0 | 24.7 | 2.41 | 1.07 |
| growth T = 10 | 10,000 | 45.1 | 258.7 | 10.84 | 5.74 |

**Growth T = 10 is above 1.1x and not signed off.** Its dataset has about 20 followers of each
regime at the last node, and the always-treat fluctuation there reaches `eps = 29.2`. Every
earlier Q response of that regime is then within about `exp(-29)` of 1, and each of those nine
fits takes 30 IRLS steps in `fit_glm` as its linear predictor climbs to the clamp, where the
other regime's take 2. Of six data seeds at T = 10, n = 10,000, two saturate this way. Single
runs, medians of five rounds, against fe13b889's row above (45.1 ms, its seed 42):

| Data seed | this tree | ratio |
|---|---:|---:|
| 42 (the row above), saturates | 269 | 5.96 |
| 3, saturates | 259 | 5.75 |
| 1, 2, 4, 5, no saturation | 65-67 | 1.44-1.49 |

Starting `fit_glm` from `warm_start` on those fits (not done; measured in an earlier, busier
session, with fe13b889's row at 62.5 ms) brings seed 42 from 437 ms to 122 ms (1.95x) and seed 3
from 445 ms to 110 ms. On seeds 42 and 3 and a saturated fixture, the ATE, SE, regime means, EIC, Q*
and fluctuation coefficients stay within 5e-11, and the warnings and convergence status are
equal. The reported Q coefficients of the saturated nodes move, by up to 2.7, because their
likelihood is flat, so that start changes a reported number and awaits a decision.

## 13. References

- van der Laan, M.J. & Gruber, S. (2012). "Targeted Minimum Loss Based Estimation of
  Causal Effects of Multiple Time Point Interventions." *The International Journal
  of Biostatistics*, 8(1), Article 9. https://doi.org/10.1515/1557-4679.1370

- Lendle, S.D., Schwab, J., Petersen, M.L., & van der Laan, M.J. (2017). "ltmle:
  An R Package Implementing Targeted Minimum Loss-Based Estimation for Longitudinal
  Data." *Journal of Statistical Software*, 81(1), 1-21. https://doi.org/10.18637/jss.v081.i01

- van der Laan, M.J. & Rose, S. (2011). *Targeted Learning: Causal Inference for
  Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

- Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and
  Causal Inference Models." *Biometrics*, 61(4), 962-973.

- R package `ltmle`: https://cran.r-project.org/package=ltmle (1.3-0 for the cross-check).

## 14. Rust Test Functions

R-pinned (registered under `causal_ltmle`; case `ltmle_two_period`):
`test_validate_ltmle_ate_se`, `test_validate_ltmle_default_control`, `test_validate_ltmle_linear_q`,
`test_validate_ltmle_randomized_known_effect`, `test_validate_ltmle_logistic_rescaled_outcome`,
`test_validate_ltmle_gbounds_cumulative`, `test_validate_ltmle_three_period`,
`test_validate_ltmle_intervention_changes_estimate`, `test_validate_ltmle_always`,
`test_validate_ltmle_never`, `test_validate_ltmle_dynamic`, `test_validate_ltmle_static_mixed`,
`test_validate_ltmle_three_period_regimes`, `test_validate_ltmle_nuisance_models`,
`test_validate_ltmle_duplicated_history_column_is_aliased`.

Property and error tests (on `ltmle_sim.csv` without loading the case, or on inline data; not
registered): `test_ltmle_config_validation_errors`, `test_ltmle_no_follower_is_an_error`,
`test_ltmle_too_few_observations`, `test_ltmle_data_rejects_non_finite`,
`test_ltmle_regime_validation_errors`, `test_regime_kind_parses_aliases_and_rejects_unknown`,
`test_ltmle_q_model_parses_aliases_and_rejects_unknown`,
`test_ltmle_display_reports_regimes_and_notes`, `test_ltmle_covariate_scale_invariance`,
`test_ltmle_intermediate_outcomes_warn`, `test_ltmle_data_validation`,
`test_ltmle_data_validation_errors`, `test_intervention_type_display`,
`test_ltmle_single_time_point_error`, `test_ltmle_earlier_q_fits_start_from_the_later_node`,
`test_ltmle_fast_paths_match_the_shared_glm`, `test_ltmle_fast_paths_match_on_other_regimes`,
`test_ltmle_fast_paths_leave_for_collinear_and_separated_nodes`,
`test_ltmle_linear_q_matches_after_the_row_predictions`.

## 15. Status

R-pinned: the logistic-Q quantities to the written-out ltmle algorithm, cross-checked against
ltmle 1.3-0; the linear-Q quantities to a written-out reference.
