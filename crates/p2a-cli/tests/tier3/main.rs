//! Tier 3 CLI integration tests.
//!
//! Run with
//! `cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier3`.
//! One module per work package; each package edits only its own module (T3 design section 9).
//! Shared helpers live in `common.rs`.

mod common;
mod wp06_flags;
mod wp60_models;
mod wp61_other;
