# MCP model tools: regression, discrete, panel and survival

## Overview

These tests pin the MCP surfaces of the regression, discrete, panel and survival tools through
`call_tool_with_session`, the dispatcher every transport uses (request parsing with
`deny_unknown_fields`, request limits, column extraction, the tool's output). The core
estimators are pinned in p2a-core; what the MCP path adds, and what these tests check, is:

- **JSON replies.** `regression_glmnet`, `regression_lasso`, `regression_ridge` and
  `regression_cv_glmnet` return the coefficient path as compact JSON, with the warnings and a
  selection summary (order of entry, the lambda where each variable enters, its estimate at
  the smallest lambda) before the path, so the first 16 KB of a reply, which the chat loop
  keeps, still answer which predictors are selected. By default the glmnet, lasso and ridge
  path entries list lambda, df and the deviance ratio, and the coefficients are given once, at
  the last lambda (`at_last_lambda`), which keeps a 100-lambda reply with 30 predictors under
  16 KB; `path_coefficients: true` adds the intercept and coefficients at every lambda, and the
  coefficient pins read that form. On `glmnet_sim` the default replies are 6,055 (glmnet, lasso)
  and 8,705 bytes (ridge) with 8 predictors and 10,008 and 11,047 bytes with 30. The three tools
  run 0.80-0.97x of the base with 8 and 30 predictors and 100 to 1,000 rows (2026-11-09, against
  fe13b889: release builds, one core, eight interleaved rounds, load average 1.4-2.1); with the
  coefficients at every lambda they had run 1.23-1.81x. `regression_gls` returns the
  coefficients, the correlation parameter and whether it was estimated (REML or ML) or
  user-supplied, the method, group and number of groups, sigma, the log-likelihood, AIC and
  BIC as nlme, and convergence.
- **Request fields.** `regression_gls.method` and `.group`, `regression_quantreg.method` and
  `.se`, `regression_smooth_spline.criterion`, `mlogit.asc` and `.reflevel`,
  `panel_unit_root.lag_selection` and `.pmax`, `cox_ph.robust_se` and `.cluster`.
- **Option parsing.** Every option string goes through the library's option types or a
  handler `Choices` table: case and separators are ignored, R's spellings are accepted, and an
  unknown value is a tool error that names the accepted ones (before, several tools fell back
  to a default silently).
- **Seeds.** `regression_cv_glmnet` folds, `regression_bootstrap_cov`, `regression_quantreg`
  with `se = "boot"` and pseudo-random mixed-logit draws use the request seed (where the tool
  has one), else the session seed (`set_seed`), else 42.
- **Notes.** The text tools append the convergence line and the warnings their table does not
  already show; a note the table shows is not repeated.

## Reference

R cases by area (each in `validation/r/cases/<file>.R`, values in `validation/expected/`):

- regression: `glmnet_paths`, `gls_compsymm`, `gls_ar1`, `rq_br_se`, `ols_aliased`,
  `ols_diagnostics_grunfeld`, `diag_bgtest_ar1`, `reset_princomp`, `diag_wald_sim`,
  `diag_harvtest_nonlin`, `hac_ar1`, `dk_panel`, `step_legacy`, `nls_expdecay`,
  `loess_x10_direct`, `smoothers_gcv_periodic_loess`, `supsmu_sine`;
- discrete: `disc_mlogit_fishing`, `disc_binary_glm`, `disc_multinom_overlap`,
  `disc_hurdle_zero`, `feglm_families`;
- survival: `surv_cox_robust`, `surv_aft`, `mcp_competing_risks`;
- panel: `dpd_sys_balanced`, `purtest_lag_selection`, `pvcm_pmg_sim`, `pggls_sim`,
  `panel_absorbed_regressors`;
- dispatcher (`crates/p2a-mcp/tests/integration_tests.rs`): `ols_grunfeld`,
  `panel_re_grunfeld`, `disc_binary_glm`, `surv_logrank_lung`, `iv_two_endog`,
  `stat_anova_two_way_unbalanced`, `stat_manova_na`, `stat_fisher_rows_2000`,
  `stat_chisq_small_expected`, `stat_chisq_gof`, `cli_stat_acf_int`, `mcp_dispatch_smoke`.

`validation/r/cases/mcp_dispatch.R` holds the two cases of this document:
`mcp_competing_risks` (survfit multi-state Aalen-Johansen CIF of two competing events on 40
rows with integer, tied times; `generated/cr_small.csv`) and `mcp_dispatch_smoke` (a 48-row
dataset with integer and missing cells: kmeans, Moran's I written out, cpt.mean, the lag-0
correlation, string operations; `generated/mcp_dispatch.csv`). Recompute them with
`Rscript validation/r/run.R --check` (all cases) or `--write mcp_competing_risks
mcp_dispatch_smoke`.

## Test cases

Tests of `crates/p2a-mcp/tests/tier3/wp58_models.rs` unless the module is named. "printed" is
`Tol::printed` at the decimals the text shows.

| Test | Tool(s) | Case keys | Tolerance | Finding |
|---|---|---|---|---|
| `test_validate_mcp_glmnet_returns_coefficients` | regression_glmnet, regression_ridge | `a10_std_coef_tight`, `a10_std_dev_ratio_tight`, `a0_std_coef_tight`, `binom_coef_tight` | 1e-3 absolute (dev ratio 1e-5); library equality | REG-8 |
| `test_validate_mcp_lasso_default_path` | regression_lasso | `path_lambda`, `path_df` | `Tol::coef(150)`, exact | REG-8 |
| `test_validate_mcp_gls_compsymm` | regression_gls | `{reml,ml}_profile_{rho,coef,se,sigma,loglik}`, `{reml,ml}_{aic,bic}`, `cs03_reml_*` | rho 1e-6 relative; `Tol::coef/se/stat(n)` | TOOL-19, REG-43 |
| `test_validate_mcp_gls_ar1_reml` | regression_gls | `phi`, `coef`, `se`, `log_lik_reml`, `phi_ml`, `coef_ml`, `log_lik_ml` | rho 1e-6 relative; `Tol::coef/se/stat(n)` | TOOL-19 |
| `test_validate_mcp_quantreg_br_se` | regression_quantreg | `tau{25,50,75}_coef`, `tau25_se_{nid,iid,ker}`, `tau{25,50,75}_se_nid` | printed | TOOL-14, TOOL-32 |
| `test_validate_mcp_regression_ols_aliased` | regression_ols | `trap_coef`, `trap_se` | printed | TOOL-32 |
| `test_validate_mcp_regression_diagnostics_tests` | regression_diagnostics, _bgtest, _resettest, _waldtest, _harvtest | `jarque_bera`, `breusch_pagan`, `durbin_watson`, `condition_number`, `bgtest4_f`, `princomp_stat`, `princomp_df`, `wald_drop_x2_x3_f`, `harvey_collier`, `harvey_collier_df` | printed; df exact | TOOL-32 |
| `test_validate_mcp_regression_hac_dk` | regression_hac, regression_driscoll_kraay | `coef`, `kernhac_parzen_bw4`; `coef`, `n_periods` | printed | TOOL-32 |
| `test_validate_mcp_loess_x10_and_mass_point` | regression_loess | `fit_s075_d2`, `fit_s075_d1`, `fit_s12_d2`, `loess_masspoint_expected` | `Tol::coef(10)`, `Tol::coef(100)` | TOOL-32 |
| `test_validate_mcp_supsmu_span_and_periodic` | regression_supsmu | `cv_x`, `span03_y`, `weighted_y`, `supsmu_periodic_y`, `supsmu_plain_y` | `Tol::coef(100)` | TOOL-32 |
| `test_validate_mcp_regression_step` | regression_step | `forward_aic`, `backward_bic` | `Tol::stat(n)` | TOOL-32 |
| `test_validate_mcp_regression_nls` | regression_nls | `coef`, `se` | printed | TOOL-32 |
| `test_validate_mcp_mlogit_asc` | mlogit | `asc_coef_tight`, `asc_se_tight`, `asc_loglik`, `noasc_*`, `pier_*_tight` | printed | TOOL-32 |
| `test_validate_mcp_logit_probit_binary_glm` | logit, probit | `logit_coef`, `logit_se`, `probit_coef_tight`, `probit_se_tight`, `aliased_coef` | printed | TOOL-32 |
| `test_validate_mcp_multinom_overlap` | multinom | `coef_tight`, `se_tight`, `refB_coef_tight`, `refB_se_tight` | printed | TOOL-32 |
| `test_validate_mcp_hurdle_zero_part` | hurdle_model | `zero_coef`, `zero_se_tight` | printed | TOOL-32 |
| `test_validate_mcp_feglm_gaussian` | feglm | `gauss_coef` | printed | TOOL-32 |
| `test_validate_mcp_cox_robust` | cox_ph | `efron_coef`, `efron_se`, `efron_se_robust`, `efron_rscore_robust`, `efron_*_cluster`, `breslow_*` | printed (robust score 2 decimals) | TOOL-32 |
| `test_validate_mcp_aft` | aft | `{weibull,exponential,lognormal,loglogistic}_{coef,se,loglik,scale}` | printed | TOOL-32 |
| `test_validate_mcp_competing_risks_cif` | competing_risks | `times`, `cif1`, `cif2` | printed | TOOL-32 |
| `test_validate_mcp_panel_gmm_system` | panel_gmm | `twostep_coef`, `twostep_se_robust`, `twostep_n_obs`, `twostep_sargan`, `onestep_coef`, `onestep_se_robust` | printed; n exact | TOOL-32 |
| `test_validate_mcp_panel_unit_root_aic` | panel_unit_root | `ips_aic_lags`, `ips_sic_lags` (min, max, mean) | printed (2 decimals) | TOOL-32 |
| `test_validate_mcp_panel_heterogeneous_and_gls` | panel_pvcm, panel_pmg, panel_gls | `random_coef`, `random_se`, `mg_coef`, `pooling_coef` | printed | TOOL-32, TOOL-4 |
| `test_validate_mcp_panel_fe_drops_absorbed` | panel_fixed_effects, panel_hdfe | `fe_coef`, `fe_se` | printed | TOOL-32 |
| integration_tests, module dispatch: test_validate_dispatch_ + module + _module for 12 modules (regression, panel, discrete, survival, causal, stats, hypothesis, timeseries, ml, spatial, munging, cleaning) | one or more tools per module | the dispatcher cases above | printed; JSON coefficient rows of `validation/README.md` | TOOL-31 |

The dispatcher tests of the viz (`test_dispatch_viz_module`: the reply is a PNG), data, utils
and search modules, the database test (feature `database`, which CI leaves out) and the router
test load no case and are not registered.

### Property and error tests (not R-pinned)

These tests pin no R value: they check option parsing, refusals, seeds and equality with the
library. They read their dataset from `validation/datasets/generated/` without loading a case
(`generated_dataset` in `crates/p2a-mcp/tests/tier3/wp58_text.rs`), carry no `test_validate`
in their names and are not in the registry, so the status page does not count them as
R-pinned.

| Test | Tool(s) | Checks | Finding |
|---|---|---|---|
| `test_mcp_cv_glmnet_returns_coefficients_and_uses_session_seed` | regression_cv_glmnet | library equality for the same seed | REG-8 |
| `test_mcp_glmnet_reply_survives_the_chat_loop_cut` | regression_lasso | entry order equals the library's | REG-8 |
| `test_mcp_glmnet_default_reply_fits_the_chat_budget` | regression_glmnet, _lasso, _ridge | default reply under 16,384 bytes with 8 and 30 predictors; `at_last_lambda` equals the last `path_coefficients` entry | REG-8 |
| `test_mcp_gls_options_and_errors` | regression_gls | refusals | TOOL-19 |
| `test_mcp_quantreg_empty_taus_is_error` | regression_quantreg | refusals | TOOL-14 |
| `test_mcp_regression_options_reject_unknown` | bgtest, resettest, bootstrap_cov, step, nls | refusals | TOOL-32 |
| `test_mcp_bootstrap_cov_is_seeded` | regression_bootstrap_cov | equal text for equal seeds | TOOL-32 |
| `test_mcp_smooth_spline_criterion` | regression_smooth_spline | library equality (not R-pinned) | TOOL-32 |
| `test_mcp_discrete_options_and_boundaries` | mixed_logit, ordered_model, zeroinfl, hurdle_model, feglm | refusals | TOOL-14, TOOL-32 |
| `test_mcp_mixed_logit_uses_session_seed` | mixed_logit | equal text for equal seeds | TOOL-32 |
| `test_mcp_cox_options_are_checked` | cox_ph | refusals | TOOL-32 |
| `test_mcp_panel_pmg_model_spellings` | panel_pmg | refusals list the accepted models | TOOL-32 |

## Tolerances

- Text replies: `Tol::printed(d)`, half a unit of the last of the `d` decimals the text shows.
- JSON replies: the rows of `validation/README.md` (coefficients `Tol::coef(n)`, SEs and
  statistics `Tol::se(n)` / `Tol::stat(n)`); the GLS correlation parameter 1e-6 relative.
- glmnet: 1e-3 absolute on the coefficients. The MCP tools run at glmnet's default
  convergence threshold 1e-7; at that threshold an R port of p2a's coordinate descent lies at
  most 1.66e-4 (lasso), 1.55e-4 (elastic net), 1.41e-4 (ridge) and 1.53e-4 (binomial) from the
  exact minimiser on these fits, against which the `*_tight` keys are computed. Deviance
  ratios 1e-5 absolute.

## Known differences

Registry deviations (quantities not pinned through MCP):

- REG-24: Driscoll-Kraay standard errors.
- REG-29, REG-30: smooth.spline fit, df, lambda and criterion value (the property test
  `test_mcp_smooth_spline_criterion` compares with the library and checks that GCV and CV
  differ).
- ECON-16: the hurdle count part.
- ECON-20: FEGLM standard errors and deviance.
- ECON-33, ECON-34: pooled-GLS and mean-group standard errors.
- ECON-41, ECON-43, ECON-44: the LLC, IPS and Maddala-Wu statistics.
- Smoke tests only: Holt-Winters forecasts (TS-23, TS-24), VAR IRF, factanal (STAT-23, STAT-49).

Conventions where p2a differs from R's defaults (stated in the tool descriptions):

- `logit` / `probit`: IRLS with epsilon 1e-10 and maxit 100 (R's glm.control: 1e-8 and 25).
- `panel_unit_root`: without `lag_selection`, a fixed floor(T_max^(1/3)) lags for every unit
  with dfcor = true (plm::purtest's default: SIC with pmax 10 and dfcor = FALSE).
- `regression_quantreg`: method `auto` ('br' up to 5,000 rows, 'fn' above; rq's default is
  'br' at every n); standard errors `nid` at every n (summary.rq uses 'rank' below n = 1001,
  which p2a does not offer).
- `regression_loess`: a tied predictor value shared by more points than the span gives the
  tie mean with a warning (lowess's rule; R's loess returns 0 there).
- `panel_gmm` system GMM: the level equations have no intercept (plm::pgmm effect =
  'individual'; Stata's xtabond2 adds _cons).
- `aft`: the tolerance is the relative change in the log-likelihood (survreg's rel.tolerance).
- `regression_gls`: the default correlation is 'ar1' (nlme's default is none).
- `panel_pvcm` (model within) and `panel_pmg` print the mean-group coefficients, which
  plm::pvcm prints only per entity.
