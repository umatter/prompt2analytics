//! Tier 2 MCP-path tests: one module per package without a handler file of its own.
//! Packages edit only their own module; `common` holds the shared helpers.
#![cfg(feature = "http")]
mod common;
mod wp04_data;
mod wp06_onesample;
mod wp07_anova;
mod wp08_iv_panel;
mod wp10_ordered_count;
mod wp12_did;
mod wp13_causal;
mod wp19_dynamic_panel;
mod wp22_acceptance;
