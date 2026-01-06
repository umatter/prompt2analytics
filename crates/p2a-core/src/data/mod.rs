//! Data loading and manipulation module.
//!
//! Provides functionality for loading datasets from various file formats
//! and basic data manipulation operations.

mod loader;
mod dataset;
mod stata;
mod sas;

pub use loader::DataLoader;
pub use dataset::{Dataset, DatasetInfo};
pub use stata::{load_stata, StataError};
pub use sas::{load_sas, SasError};
