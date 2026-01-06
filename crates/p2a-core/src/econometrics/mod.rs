//! Econometrics module powered by greeners.
//!
//! Provides panel data estimators, instrumental variables, and causal inference tools.

mod panel;
mod iv;
mod did;
mod convert;

pub use panel::{PanelResult, run_fixed_effects, run_random_effects};
pub use iv::{IVResult, run_iv2sls};
pub use did::{DiDResult, run_did};
pub use convert::polars_to_greeners;
