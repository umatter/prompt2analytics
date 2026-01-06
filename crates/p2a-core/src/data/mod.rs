//! Data loading and manipulation module.
//!
//! Provides functionality for loading datasets from various file formats
//! and basic data manipulation operations.

mod loader;
mod dataset;

pub use loader::DataLoader;
pub use dataset::{Dataset, DatasetInfo};
