//! Tier 3 MCP-path tests: one module per Tier 3 package with MCP tests (T3 design 2.5, 3.7).
//! Packages edit only their own module; `common` holds the shared helpers (T3-WP05).
#![cfg(feature = "http")]
mod common;
mod wp05_mechanics;
mod wp53_llm_routes;
mod wp54_ui_persistence;
mod wp55_causal;
mod wp56_stats_ts;
mod wp57_ml_spatial;
mod wp58_models;
mod wp59_data;
mod wp62_numeric;
