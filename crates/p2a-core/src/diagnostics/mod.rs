//! Identification diagnostics for causal inference methods.
//!
//! This module provides data-driven warnings about potential violations of
//! identification assumptions. Warnings are informational and non-blocking;
//! they help users interpret results appropriately but do not prevent execution.
//!
//! # Supported Diagnostics
//!
//! - **Instrumental Variables**: Weak instrument detection (first-stage F-stat),
//!   overidentification tests (Sargan/Hansen J)
//! - **Difference-in-Differences**: Parallel pre-trends tests
//! - **Matching/IPW**: Positivity violations, extreme weights, covariate balance
//! - **Regression Discontinuity**: McCrary manipulation test, bandwidth sensitivity
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::diagnostics::{IdentificationReport, WarningSeverity};
//!
//! let report = result.identification_report();
//! for warning in &report.warnings {
//!     if warning.severity >= WarningSeverity::Warning {
//!         println!("⚠️ {}: {}", warning.title, warning.message);
//!     }
//! }
//! ```

mod did;
mod iv;
mod matching;
mod rd;
mod types;

pub use did::*;
pub use iv::*;
pub use matching::*;
pub use rd::*;
pub use types::*;
