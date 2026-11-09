# MCP model tools: regression, discrete, panel and survival

Purpose: the MCP surfaces of the regression, discrete, panel and survival tools, pinned through
`call_tool_with_session` (the dispatcher every transport uses) against the R reference cases of the
methods they call.

Cases: `glmnet_paths`, `gls_compsymm`, `gls_ar1`.

Tests: see the registry, `validation/methods/mcp_models.json`.
