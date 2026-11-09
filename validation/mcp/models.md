# MCP model tools: regression, discrete, panel and survival

Purpose: the MCP surfaces of the regression, discrete, panel and survival tools, pinned through
`call_tool_with_session` (the dispatcher every transport uses) against the R reference cases of the
methods they call.

Cases: `glmnet_paths`, `gls_compsymm`, `gls_ar1`, `rq_br_se`, `ols_aliased`, `ols_diagnostics_grunfeld`, `diag_bgtest_ar1`, `reset_princomp`, `diag_wald_sim`, `diag_harvtest_nonlin`, `hac_ar1`, `dk_panel`, `step_legacy`, `nls_expdecay`, `ols_mtcars`, `loess_x10_direct`, `smoothers_gcv_periodic_loess`, `supsmu_sine`, `disc_mlogit_fishing`, `disc_binary_glm`, `disc_multinom_overlap`, `disc_hurdle_zero`, `feglm_families`, `disc_mixed_logit`, `disc_zeroinfl`, `surv_cox_robust`, `surv_aft`, `dpd_sys_balanced`, `purtest_lag_selection`, `pvcm_pmg_sim`, `pggls_sim`, `panel_absorbed_regressors`, `mcp_competing_risks`; the dispatcher tests (`crates/p2a-mcp/tests/integration_tests.rs`) use `ols_grunfeld`, `panel_re_grunfeld`, `disc_binary_glm`, `surv_logrank_lung`, `iv_two_endog`, `stat_anova_two_way_unbalanced`, `stat_manova_na`, `stat_fisher_rows_2000`, `stat_chisq_small_expected`, `stat_chisq_gof`, `cli_stat_acf_int` and `mcp_dispatch_smoke`.

Tests: see the registry, `validation/methods/mcp_models.json`.
