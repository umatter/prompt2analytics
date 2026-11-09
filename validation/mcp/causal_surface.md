# MCP causal surface

What the MCP causal tools map their requests to, and which tests pin that mapping against R.
Registry: `validation/methods/causal_mcp_surface.json`. Status: R-pinned where the registry lists a case;
the tests are in `crates/p2a-mcp/tests/tier3/wp55_causal.rs`.

Every R-pinned test follows one pattern. It computes the library result for the configuration the request must
map to, checks that result against the R case, and requires the tool's text to equal that result's text. A request
that maps to the wrong configuration therefore fails the pin, even when the library itself is correct.

## What the MCP layer maps

Option strings are parsed with the core's option tables (`FromStr`, case and separators `-`, `_`, `.` and ` `
ignored). An unknown value is a tool error raised before the dataset is read, and the tool text is the core message
verbatim (for example `unknown comparison_group 'sometimes'; expected one of: never_treated, not_yet_treated`). No
causal handler has a default arm that maps an unknown string to a default. Six small tables that the core does not
provide yet live in the handler module (twang `stop_method` and `estimand`, scpi pooling weights, the
marginal-effects model, evalue's effect measure, ltmle's contrast). Their aliases and refusals are unit-tested
(`test_local_option_tables_parse_aliases_and_reject_unknown`).

| Tool | Request field | Core configuration |
|---|---|---|
| `staggered_did` | `anticipation`, `comparison_group`, `estimation_method`; bootstrap seeded by the session seed | `StaggeredDidConfig.{anticipation, comparison_group, estimation_method, seed}` |
| `etwfe` | `vcov` (`cluster` default, `iid`), `aggregation`, `tref`, `gref`, `cgroup` | `EtwfeConfig.{vcov, tref, gref, cgroup}`; `aggregation` returns that emfx table |
| `rd_estimate`, `rd_fuzzy`, `rd_bw`, `rd_multi` | `bwselect`, `kernel`, `vce`, `scaleregul` | `RdConfig` / `RdMultiConfig`; `rd_bw` calls `rd_bandwidth_with` |
| `iv_mte` | `x` (covariates), `estimand`, `prte_weights`, `bootstrap_reps` | `IVMTEConfig.{include_covariates, propensity_covariates, estimand, prte_weights, bootstrap_se, n_bootstrap, seed}` |
| `gmm_iv` | `method`, `vcov` (HAC default), `kernel`, `weight_matrix` | `GeneralGmmConfig` |
| `treatment_cbps` | `method` (`over` default), `estimand` (`ATE` default) | `CbpsConfig.{method, estimand}` |
| `treatment_entropy_balance` | `estimand` (`ATT` default) | `EntropyBalanceConfig.estimand` |
| `treatment_ipw`, `treatment_doubly_robust` | `bootstrap` (IPW default 0: the M-estimation SE), seeds | `IpwConfig`, `DoublyRobustConfig` |
| `propensity_matching` | `method`, `distance`, `m_order`, `output_name` | `match_it_with`; the matched sample is registered as `MatchIt::match.data` |
| `natural_effects_mediation`, `mediation_analysis` | `scale`, `family`; `n_bootstrap` (also `bootstrap`) | `MedflexConfig`, `MediationConfig` |
| `ltmle` | `regime`, `contrast`, `dynamic_columns`, `dynamic_threshold`, `q_model`, `ps_lower`, `ps_upper` | `LtmleData::from_dataset`, `LtmleConfig.{intervention, contrast, q_model, gbounds}` |
| `gformula` | `outcome_type`, `time_col`, `event_col`, `intervention` | `GFormulaData::new_survival` for survival, `with_names` always |
| `collaborative_tmle` | `cv_criterion`, `order`, `stopping_rule`, `q_model` | `CTmleConfig.{cv_criterion, order, stopping_rule, q_model}` |
| `treatment_double_ml` | `model` (also `model_type`), `ml_method`, `lambda`, `instrument`, `fold_column`, `seed` | `DoubleMLConfig.{model_type, ml_method, lambda, folds, seed}`; PLIV and IIVM through `run_double_ml_iv` |
| `gsynth` | `cv` (also `cross_validate`), `r_max` (or its former name `max_factors`), `bootstrap_reps` (also `n_bootstrap`), `bootstrap_se`, `estimator`, `force` | `GsynthConfig.{cross_validate, max_factors, bootstrap_se, n_bootstrap, seed, estimator, force}` |
| `sensemakr` | `contour_data` | `run_ols(.., CovarianceType::Standard)` and `generate_contour_data`, appended as one JSON line |

Seeds: where a tool has no `seed` field, its random draws (bootstrap, folds) use the session seed.

## Defaults the MCP chooses

- **MatchIt's defaults.** `cem` uses Sturges bins. `full` is unconstrained (`min.controls = 0`,
  `max.controls = Inf`). Nearest-neighbour matching on a propensity score uses `m.order = 'largest'`. The matched
  sample keeps the units with positive weight, as `match.data`. Options the chosen method does not use are ignored
  with a warning, as MatchIt warns.
- **ltmle's defaults.** The outcome model is logistic (a continuous outcome is rescaled to [0, 1] by its range).
  gbounds are (0.01, 1) on the cumulative probability of following the regime. Only the last entry of `outcomes` is
  the outcome node.
- **gformula survival.** `time_col` is required; `event_col` defaults to `outcome`. Either field without
  `outcome_type = 'survival'` is a tool error.
- **gsynth.** The bootstrap uses 200 replicates when it is on and `bootstrap_reps` is unset: gsynth's `nboots` and
  the core default.

## Conventions the MCP inherits as p2a's

Where the default is p2a's rather than R's, the description says so.

- `etwfe`: standard errors clustered by unit by default (etwfe's default too; `iid` on request).
- `gmm_iv`: HAC with Bartlett weights and floor(4 (n/100)^(2/9)) lags, no prewhitening, by default. gmm's own default
  is the Quadratic Spectral kernel with the Andrews bandwidth and prewhitening. A fixed weighting matrix reports no
  J test.
- `treatment_cbps`: estimand ATE by default; R's CBPS defaults to ATT = 1.
- `treatment_double_ml`: PLR fits m(X) = E[D|X] with the regression learner even for a 0/1 D. DoubleML has no
  default learner. Folds are seeded with the session seed, else 42.
- `ltmle`: the influence-curve variance.
- `collaborative_tmle`: a fixed initial outcome model Q. ctmle's `ctmleDiscrete` also updates Q.
- `synthetic_control`: predictors standardised by their SD, as Synth; `custom_v_weights` act on that scale and are
  normalised.
- `gsynth`: r = 2 without cross-validation by default. gsynth defaults to `CV = TRUE` over r = 0..5. Following R
  would add a leave-one-out fit for each of six ranks to every default call. With `cv = true`, the core
  cross-validates leave-one-out (fect `cv.method = "loo"`). gsynth 1.4's random-fold choice is recorded in the case
  and not pinned.

## Cases

New cases in `validation/r/cases/causal_mcp_surface.R`:

| Case | What | Key values |
|---|---|---|
| `mte_mcp_covariate` | `iv_mte` with instrument z1 and covariate z2: probit P(d \| z1, z2), y on [1, P, P^2, P^3, z2], Heckman-Vytlacil weights, delta-method SEs with the HC1 covariance; and the same call without the covariate (`nox_*`) | probit (tight) -0.17198, 0.88681, 0.64270; ATE 1.41459 (SE 0.16855), ATT 1.85607 (SE 0.20983); without the covariate ATE 1.18586 |
| `matchit_mcp_output` | the matched sample `propensity_matching` registers: MatchIt 4.8.0 nearest 1:1 and 1:2 on the logit PS, weights and subclass | n = 240; 178 units matched at 1:1, 240 at 1:2 |
| `sensemakr_mcp_contour` | `sensemakr::adjusted_estimate` of `directlyharmed` on classical `lm` (darfur without village effects), 20 x 20 grid over [0, 0.5] | estimate 0.0488855 (dof 1268); corner (0.5, 0.5) -0.414210 |

The probit pins of `mte_mcp_covariate` are the R solution: glm at epsilon 1e-14, restarted from its own estimate
until the coefficients stop changing, because a single epsilon-1e-14 fit stops 3e-10 short. The values at
`glm.control()` are recorded and not pinned.

Wave-2 cases the tests read: `did_cs_covariates`, `etwfe_staggered`, `rd_bwselect_sharp`, `rd_bwselect_fuzzy`,
`gmm_linear_iv`, `cbps_sim_methods`, `weights_ebal`, `ipw_aipw_sim`, `matchit_full_cem`, `matchit_nearest`,
`medflex_scales`, `ltmle_two_period`, `ctmle_cv`, `gformula_hazard`, `doubleml_models`, `gsynth_simdata`, and, for
the parameter checks, `mte_local_iv`.

## Tests and tolerances

Tier 1 tolerances (`validation/README.md`) apply unless a row below says otherwise. The iterative rows exist
because the MCP runs the library's default control while the case holds the solution.

| Test | Iterative or special tolerances |
|---|---|
| `test_validate_mcp_staggered_did` | doubly robust ATT Tol::iterative(1e-8) |
| `test_validate_mcp_iv_mte_uses_covariates` | first stage Tol::iterative(1e-10); second-stage coefficients Tol::iterative(1e-8) (the near-collinear cubic in P amplifies the 2e-11 first-stage gap); the default configuration within Tol::iterative(1e-6) of the tight estimands |
| `test_validate_mcp_gmm_iv` | CUE Tol::iterative(1e-8) |
| `test_validate_mcp_treatment_cbps` | Tol::iterative(1e-6) |
| `test_validate_mcp_weighting` | AIPW and IPW effects Tol::iterative(1e-8) (IRLS propensity) |
| `test_validate_mcp_propensity_matching` | full-matching optimum and PS Tol::iterative(1e-8); counts exact |
| `test_validate_mcp_ltmle` | always, never, dynamic and the SE Tol::iterative(1e-8) against the tight values |
| `test_validate_mcp_collaborative_tmle` | ATE Tol::iterative(1e-7), SE Tol::iterative(1e-6) (default glm epsilon 1e-8 against 1e-14) |
| `test_validate_mcp_gformula` | risk curves and hazard coefficients Tol::iterative(1e-8) against the tight fit |
| `test_validate_mcp_treatment_double_ml` | PLR lasso Tol::new(1e-9, 1e-12) against the exact glmnet minimiser |
| `test_validate_mcp_gsynth` | the cross-validated r exact |

Every other pinned test in the registry uses Tier 1 tolerances: `test_validate_mcp_etwfe`,
`test_validate_mcp_rd_bandwidth_selection`, `test_validate_mcp_matching_output_dataset`,
`test_validate_mcp_natural_effects_mediation` and `test_validate_mcp_sensemakr_contour_data`.

The non-numeric tests are:

- `test_mcp_causal_option_aliases_and_unknown`: every option field accepts its aliases and refuses an unknown value,
  naming the option and its canonical spellings.
- `test_mcp_causal_degenerate_parameters_are_tool_errors`: impossible values (negative or zero RD bandwidths, an
  empty cutoff list, a level outside (0, 1), `scaleregul < 0`, `n_grid` below 2, an unknown reference cohort,
  inverted gbounds, a risk-difference E-value without `baseline_risk`) are tool errors naming the parameter.
- `test_mcp_causal_option_descriptions_list_accepted_values`: a drift guard. Every option field's schema description
  lists each canonical value of the core table it is parsed with, quoted as `'name'`.

## No longer R-pinned

- `iv_mte`'s IRLS iteration count and its `glm.control()` (epsilon 1e-8) values. The estimands of the default
  configuration stay pinned against the tight fit at Tol::iterative(1e-6).
- `ltmle`'s default-control keys. The default configuration stays pinned against the tight values at
  Tol::iterative(1e-8).
- `gformula`'s hazard iteration count and R's epsilon-1e-8 risks. The default configuration stays pinned against
  the tight fit at Tol::iterative(1e-8).
- The SBW `balance_penalty` is refused when it is not 1000, rather than reported.

## Not covered

- `heterogeneity_test` (`HetTxRequest`) and `ml_bart_causal` (`BartCausalRequest`): ML-module tools whose options
  are parsed outside this surface; they are not in this registry.
- The sensemakr t-statistic grid: the tool returns adjusted estimates only, because the core's adjusted SE differs
  from sensemakr (REG-27).
