# Validation: CLI model commands (`reg`, `discrete`, `panel`, `survival`, `stats`), persisted datasets, notes

## Method Overview

T3-WP60 makes the computing commands of `p2a reg`, `discrete`, `panel`, `survival` and `stats`
report what their core result says about itself (convergence status and every warning, in text and
JSON), parses their options into the cores' enums, and exposes the options the Tier 3 cores added.
The numbers come from the cores; the tests here pin them through the binary against R.

## What is pinned

| Test | Case(s) | Keys | Tolerance |
|---|---|---|---|
| `test_cli_logit_prints_converged_and_warnings` | `disc_binary_glm` | `aliased_coef`; the separation and glm.fit warnings in text and JSON | coefficients `Tol::coef(400)` |
| `test_validate_cli_logit_probit_glm` | `disc_binary_glm` | `logit_coef/se/z/p/loglik/aic/null_loglik`; `probit_coef_tight`, `probit_se_tight`, `probit_loglik_tight`, `probit_aic_tight` | logit `Tol::coef/se/stat/p_value(400)`; probit coefficients and SEs `Tol::iterative(1e-6)` (p2a's default control), log-likelihood and AIC `Tol::stat(400)` |
| `test_validate_cli_mlogit_base` | `disc_multinom_overlap` | `refB_coef_tight`, `refB_se_tight`, `coef_tight`, `loglik` | `Tol::iterative(1e-8)` coefficients, `1e-6` SEs |
| `test_validate_cli_clogit_asc_reflevel` | `disc_mlogit_fishing` | `generic_asc_*_tight`, `generic_noasc_*_tight`, the reflevel-pier reparametrisation | `Tol::iterative(1e-9)` coefficients, `1e-8` SEs |
| `test_validate_cli_discrete_poisson` | `cli_discrete_poisson` | `coef`, `se`, `z`, `p`, `loglik`, `aic`, `deviance`, `null_deviance`, `n`, `na_coef`, `na_se`, `na_n_dropped` | `Tol::coef/se/stat/p_value(400)`; counts exact |
| `test_validate_cli_cox_ties_robust_cluster` | `surv_cox_robust` | `{efron,breslow}_coef/se/z/p/loglik`, `efron_se_robust/z_robust/p_robust/rscore_robust/rscore_p_robust`, `efron_coef_cluster/se_cluster/se_model_cluster/z_cluster/p_cluster/rscore_cluster`, `n`, `n_clusters`, `n_cluster_obs` | `Tol::coef/se/stat/p_value(227)`; counts exact |
| `test_validate_cli_aft_distributions` | `surv_aft` | `{weibull,exponential,lognormal,loglogistic}_coef/se/z/p/scale/loglik/aic/bic`, `*_se_log_scale` (not exponential: scale fixed), `weibull_z_log_scale` in the printed Log(scale) row | `Tol::coef/se/stat/p_value(227)`; printed row `Tol::printed(4)` |
| `test_validate_cli_competing_risks_ci` | `cli_competing_risks` | `times`, `cif_k`, `survfit_se_k`, `survfit_ci95_*_k`, `survfit_ci50_*_k` (null where the CIF is 0) | CIF `Tol::coef(200)`, SEs `Tol::se(200)`, limits `Tol::stat(200)`; first printed row `Tol::printed(4)` |
| `test_validate_cli_unit_root_lag_selection` | `purtest_lag_selection` | `ips_{aic,sic,hall}_dfcor_lags`, `trho_{aic,sic,hall}_dfcor`, `trend_lags_aic_dfcor`, `trend_trho_aic_dfcor`; the `Unit lags:` line | lags exact; statistics `Tol::stat(480)` |
| `test_validate_cli_panel_absorbed_regressors` | `panel_absorbed_regressors` | `fe_coef`, `fe_se`, `hdfe2_coef`; `dropped` names z (fe) and z, w (hdfe) | `Tol::coef/se(180)` |
| `test_validate_cli_quantreg_se` | `rq_br_se` | `tau50_coef`, `tau50_{se,t,p,ci_lower,ci_upper}_{nid,iid,ker}`, `tau50_objective`; the default fit has no rank note and runs br at n = 100 | `Tol::coef/se/stat/p_value(100)` |
| `test_validate_cli_gls_compsymm_group` | `gls_compsymm` | `{reml,ml}_profile_{coef,se,rho,sigma,loglik}`, `{reml,ml}_{coef,aic,bic}`, `n_groups`, `cs03_reml_{coef,se}` | `Tol::coef/se/stat(150)`; rho `Tol::iterative(1e-6)`; nlme's own coefficients (cross-check) `Tol::iterative(1e-6)` |
| `test_validate_cli_loess_masspoint_warns` | `smoothers_gcv_periodic_loess` | `loess_masspoint_expected` (first 10 fitted values); the zero-width-neighbourhood warning in text and JSON | `Tol::coef(100)` |
| `test_validate_cli_hac_aliased_df` | `cli_hac_aliased` | `coef`, `se_nw`, `t_nw`, `p_nw` (identified columns), `df_resid`, `r2` | `Tol::coef/se/stat/p_value(120)`; df exact |
| `test_validate_cli_nls_expdecay` | `nls_expdecay` | `coef`, `se`, `rss` | `Tol::iterative(1e-6)` coefficients, `1e-5` SEs, `1e-8` RSS |

`test_cli_smooth_spline_criterion` loads no case: it checks that `--criterion gcv|cv` gives the core's
own fit (p2a's smoothing spline is not R's yet), so it is not in the registry.
