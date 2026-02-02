//! Data simulation and random data generation.
//!
//! This module provides functionality for generating random datasets with various
//! statistical distributions. Useful for testing, simulation studies, and creating
//! example data for analysis.
//!
//! # Supported Distributions
//!
//! - **Uniform**: Values uniformly distributed between min and max
//! - **Normal**: Gaussian distribution with specified mean and standard deviation
//! - **Binomial**: Number of successes in n trials with probability p
//! - **Poisson**: Count data with specified rate (lambda)
//! - **Exponential**: Time between events with specified rate
//! - **Bernoulli**: Binary (0/1) outcomes with specified probability
//! - **Categorical**: Random selection from specified categories
//! - **Sequence**: Sequential integers (useful for IDs)
//!
//! # Example
//!
//! ```
//! use p2a_core::simulation::{generate_random_data, ColumnSpec, Distribution};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let columns = vec![
//!     ColumnSpec::new("id", Distribution::Sequence { start: 1 }),
//!     ColumnSpec::new("x", Distribution::Normal { mean: 0.0, std: 1.0 }),
//!     ColumnSpec::new("y", Distribution::Uniform { min: 0.0, max: 100.0 }),
//!     ColumnSpec::new("group", Distribution::Categorical {
//!         categories: vec!["A".to_string(), "B".to_string(), "C".to_string()],
//!         weights: None,
//!     }),
//! ];
//!
//! let dataset = generate_random_data(1000, columns, Some(42))?;
//! # Ok(())
//! # }
//! ```

mod generator;

pub use generator::{ColumnSpec, Distribution, GenerationError, generate_random_data};
