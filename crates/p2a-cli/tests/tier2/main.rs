//! Tier 2 CLI integration tests.
//!
//! Run with
//! `cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier2`.
//! One module per work package; each package edits only its own module.
//! Shared helpers live in `common.rs`.

mod common;
mod wp03_flags;
mod wp04_data;
mod wp09_survival;
mod wp11_rd;
mod wp14_spatial;
mod wp15_regression;
mod wp16_arima;
mod wp17_ml;
mod wp18_discrete;
mod wp19_dynamic_panel;
mod wp21_stats_viz;
mod wp22_acceptance;
