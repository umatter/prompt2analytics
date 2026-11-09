# MCP model tools: regression, discrete, panel and survival

Purpose: the MCP surfaces of the regression, discrete, panel and survival tools, pinned through
`call_tool_with_session` (the dispatcher every transport uses) against the R reference cases of the
methods they call.

Cases: `glmnet_paths`, `gls_compsymm`, `gls_ar1`, `rq_br_se`, `ols_aliased`, `ols_diagnostics_grunfeld`, `diag_bgtest_ar1`, `reset_princomp`, `diag_wald_sim`, `diag_harvtest_nonlin`, `hac_ar1`, `dk_panel`, `step_legacy`, `nls_expdecay`, `ols_mtcars`.

Tests: see the registry, `validation/methods/mcp_models.json`.
