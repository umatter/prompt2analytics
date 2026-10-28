//! Test-only access to the R reference cases in `validation/` (schema 1).
//!
//! The flow is R case -> JSON -> [`Case`] -> [`Check`]:
//!
//! 1. An R case in `validation/r/cases/*.R` computes reference values with the named
//!    R function on a committed dataset and `Rscript validation/r/run.R --write <id>`
//!    writes them to `validation/expected/<id>.json` (schema [`SCHEMA`]).
//! 2. A Rust test loads that file with [`Case::load`], loads the case's dataset through
//!    the crate under test (`Case::dataset_path`), runs the estimator, and compares every
//!    reported number with a [`Check`], using the tolerances of [`Tol`].
//! 3. `Rscript validation/r/run.R --check` fails when a committed value no longer
//!    matches what R computes, so an R package upgrade cannot silently move a reference.
//!
//! [`golden::assert_golden`] compares CLI output with committed golden files and
//! [`text`] parses the CLI's regression table, for exhibit tests that check the printed
//! numbers against R at print precision. See `validation/README.md`, section
//! "R reference harness".
//!
//! ```
//! use p2a_validation::{Case, Check, Tol};
//! let case = Case::load("harness_selftest");
//! Check::new(&case).scalar("n", 200.0, Tol::exact()).finish();
//! ```
//!
//! This crate is a dev-dependency only (`publish = false`); it depends on serde and
//! serde_json, so p2a-core, p2a-cli and p2a-mcp can all use it without a cycle.

use std::path::{Path, PathBuf};

pub mod check;
pub mod golden;
pub mod text;

mod case;

pub use case::{Case, CaseError, Value};
pub use check::{Check, Tol};

/// The case-file schema this crate reads (`"schema": 1` in `validation/expected/*.json`).
pub const SCHEMA: u32 = 1;

/// Repository root: `$P2A_REPO_ROOT` if set (and non-empty), else two levels above this
/// crate's manifest directory (`crates/p2a-validation/../..`).
pub fn repo_root() -> PathBuf {
    if let Some(root) = std::env::var_os("P2A_REPO_ROOT")
        && !root.is_empty()
    {
        return PathBuf::from(root);
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest.to_path_buf())
}

/// `<repo>/validation`.
pub fn validation_dir() -> PathBuf {
    repo_root().join("validation")
}

/// Resolve a repository-relative path (as stored in case files) to an absolute path.
pub fn repo_path(rel: &str) -> PathBuf {
    repo_root().join(rel)
}
