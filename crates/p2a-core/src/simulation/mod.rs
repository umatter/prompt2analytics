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
//! - **UniformInt**: Integers uniformly distributed between min and max (both inclusive)
//! - **Sequence**: Sequential integers (useful for IDs)
//! - **Constant**: The same number in every row
//! - **ConstantString**: The same text in every row (stored once)
//!
//! # Parameter bounds
//!
//! [`Distribution::validate`] (called by [`generate_random_data`]) requires every
//! floating-point parameter to be finite. Beyond that: a uniform range must have
//! `min < max` and a representable width `max - min`; normal and exponential scale
//! parameters must be positive; probabilities lie in `[0, 1]`; a Poisson mean
//! `lambda` lies in `(0, 1e13]` (larger means are not sampled accurately); a
//! binomial `n` lies in `1..=2^53`; categorical weights are non-negative with a
//! finite, positive sum. `n_rows` lies in `1..=IdxSize::MAX` (polars' row limit),
//! and a sequence's last value `start + n_rows - 1` must fit in `i64`.
//!
//! Column buffers are allocated fallibly through [`crate::resource`], and generation
//! stops with an error when the thread's [`crate::resource::Budget`] is cancelled or
//! past its deadline.
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
