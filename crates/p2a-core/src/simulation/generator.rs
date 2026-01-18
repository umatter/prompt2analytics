//! Random data generation implementation.

use polars::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Exp, Normal, Poisson, Uniform};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;

/// Error type for data generation operations.
#[derive(Debug, Clone)]
pub enum GenerationError {
    /// Invalid distribution parameters
    InvalidParameters(String),
    /// Column creation failed
    ColumnCreation(String),
    /// DataFrame creation failed
    DataFrameCreation(String),
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenerationError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
            GenerationError::ColumnCreation(msg) => {
                write!(f, "Failed to create column: {}", msg)
            }
            GenerationError::DataFrameCreation(msg) => {
                write!(f, "Failed to create DataFrame: {}", msg)
            }
        }
    }
}

impl std::error::Error for GenerationError {}

/// Statistical distribution specification for random data generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Distribution {
    /// Uniform distribution between min and max.
    Uniform {
        /// Minimum value (inclusive)
        min: f64,
        /// Maximum value (exclusive)
        max: f64,
    },

    /// Normal (Gaussian) distribution.
    Normal {
        /// Mean of the distribution
        mean: f64,
        /// Standard deviation (must be positive)
        #[serde(rename = "std")]
        std: f64,
    },

    /// Binomial distribution: number of successes in n trials.
    Binomial {
        /// Number of trials
        n: u64,
        /// Probability of success per trial (0 to 1)
        p: f64,
    },

    /// Poisson distribution for count data.
    Poisson {
        /// Rate parameter (lambda, must be positive)
        lambda: f64,
    },

    /// Exponential distribution for time between events.
    Exponential {
        /// Rate parameter (must be positive)
        rate: f64,
    },

    /// Bernoulli distribution: binary 0/1 outcomes.
    Bernoulli {
        /// Probability of 1 (success)
        p: f64,
    },

    /// Categorical distribution: random selection from categories.
    Categorical {
        /// List of category values
        categories: Vec<String>,
        /// Optional weights (must sum to 1 if provided, or will be normalized)
        #[serde(default)]
        weights: Option<Vec<f64>>,
    },

    /// Integer uniform distribution.
    UniformInt {
        /// Minimum value (inclusive)
        min: i64,
        /// Maximum value (inclusive)
        max: i64,
    },

    /// Sequential integers (useful for IDs).
    Sequence {
        /// Starting value
        start: i64,
    },

    /// Constant value (same value for all rows).
    Constant {
        /// The constant value
        value: f64,
    },

    /// Constant string value (same value for all rows).
    ConstantString {
        /// The constant string value
        value: String,
    },
}

impl Distribution {
    /// Validate distribution parameters.
    pub fn validate(&self) -> Result<(), GenerationError> {
        match self {
            Distribution::Uniform { min, max } => {
                if min >= max {
                    return Err(GenerationError::InvalidParameters(
                        format!("Uniform: min ({}) must be less than max ({})", min, max),
                    ));
                }
            }
            Distribution::Normal { std, .. } => {
                if *std <= 0.0 {
                    return Err(GenerationError::InvalidParameters(
                        format!("Normal: std ({}) must be positive", std),
                    ));
                }
            }
            Distribution::Binomial { n, p } => {
                if *p < 0.0 || *p > 1.0 {
                    return Err(GenerationError::InvalidParameters(
                        format!("Binomial: p ({}) must be between 0 and 1", p),
                    ));
                }
                if *n == 0 {
                    return Err(GenerationError::InvalidParameters(
                        "Binomial: n must be positive".to_string(),
                    ));
                }
            }
            Distribution::Poisson { lambda } => {
                if *lambda <= 0.0 {
                    return Err(GenerationError::InvalidParameters(
                        format!("Poisson: lambda ({}) must be positive", lambda),
                    ));
                }
            }
            Distribution::Exponential { rate } => {
                if *rate <= 0.0 {
                    return Err(GenerationError::InvalidParameters(
                        format!("Exponential: rate ({}) must be positive", rate),
                    ));
                }
            }
            Distribution::Bernoulli { p } => {
                if *p < 0.0 || *p > 1.0 {
                    return Err(GenerationError::InvalidParameters(
                        format!("Bernoulli: p ({}) must be between 0 and 1", p),
                    ));
                }
            }
            Distribution::Categorical { categories, weights } => {
                if categories.is_empty() {
                    return Err(GenerationError::InvalidParameters(
                        "Categorical: categories cannot be empty".to_string(),
                    ));
                }
                if let Some(w) = weights {
                    if w.len() != categories.len() {
                        return Err(GenerationError::InvalidParameters(format!(
                            "Categorical: weights length ({}) must match categories length ({})",
                            w.len(),
                            categories.len()
                        )));
                    }
                    if w.iter().any(|&x| x < 0.0) {
                        return Err(GenerationError::InvalidParameters(
                            "Categorical: weights must be non-negative".to_string(),
                        ));
                    }
                    let sum: f64 = w.iter().sum();
                    if sum <= 0.0 {
                        return Err(GenerationError::InvalidParameters(
                            "Categorical: weights must sum to a positive value".to_string(),
                        ));
                    }
                }
            }
            Distribution::UniformInt { min, max } => {
                if min > max {
                    return Err(GenerationError::InvalidParameters(
                        format!("UniformInt: min ({}) must be <= max ({})", min, max),
                    ));
                }
            }
            Distribution::Sequence { .. } => {}
            Distribution::Constant { .. } => {}
            Distribution::ConstantString { .. } => {}
        }
        Ok(())
    }
}

/// Specification for a column to generate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSpec {
    /// Column name
    pub name: String,
    /// Distribution to sample from
    pub distribution: Distribution,
}

impl ColumnSpec {
    /// Create a new column specification.
    pub fn new(name: &str, distribution: Distribution) -> Self {
        Self {
            name: name.to_string(),
            distribution,
        }
    }
}

/// Generate a random dataset with specified columns and distributions.
///
/// # Arguments
///
/// * `n_rows` - Number of rows to generate
/// * `columns` - Specifications for each column
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` containing the generated data.
///
/// # Example
///
/// ```rust
/// use p2a_core::simulation::{generate_random_data, ColumnSpec, Distribution};
///
/// let columns = vec![
///     ColumnSpec::new("x", Distribution::Normal { mean: 0.0, std: 1.0 }),
///     ColumnSpec::new("y", Distribution::Uniform { min: 0.0, max: 10.0 }),
/// ];
///
/// let dataset = generate_random_data(100, columns, Some(42))?;
/// assert_eq!(dataset.df().height(), 100);
/// # Ok::<(), p2a_core::simulation::GenerationError>(())
/// ```
pub fn generate_random_data(
    n_rows: usize,
    columns: Vec<ColumnSpec>,
    seed: Option<u64>,
) -> Result<Dataset, GenerationError> {
    if n_rows == 0 {
        return Err(GenerationError::InvalidParameters(
            "n_rows must be positive".to_string(),
        ));
    }

    if columns.is_empty() {
        return Err(GenerationError::InvalidParameters(
            "columns cannot be empty".to_string(),
        ));
    }

    // Validate all distributions first
    for col in &columns {
        col.distribution.validate()?;
    }

    // Create RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Generate each column
    let mut series_vec: Vec<Column> = Vec::with_capacity(columns.len());

    for col in columns {
        let series = generate_column(&col.name, &col.distribution, n_rows, &mut rng)?;
        series_vec.push(series);
    }

    // Create DataFrame
    let df = DataFrame::new(series_vec).map_err(|e| {
        GenerationError::DataFrameCreation(format!("Failed to create DataFrame: {}", e))
    })?;

    Ok(Dataset::new(df).with_name("generated"))
}

/// Generate a single column of random data.
fn generate_column(
    name: &str,
    distribution: &Distribution,
    n_rows: usize,
    rng: &mut StdRng,
) -> Result<Column, GenerationError> {
    match distribution {
        Distribution::Uniform { min, max } => {
            let dist = Uniform::new(*min, *max);
            let values: Vec<f64> = (0..n_rows).map(|_| rng.sample(dist)).collect();
            Ok(Column::new(name.into(), values))
        }

        Distribution::Normal { mean, std } => {
            let dist = Normal::new(*mean, *std).map_err(|e| {
                GenerationError::ColumnCreation(format!("Normal distribution error: {}", e))
            })?;
            let values: Vec<f64> = (0..n_rows).map(|_| rng.sample(dist)).collect();
            Ok(Column::new(name.into(), values))
        }

        Distribution::Binomial { n, p } => {
            let dist = Binomial::new(*n, *p).map_err(|e| {
                GenerationError::ColumnCreation(format!("Binomial distribution error: {}", e))
            })?;
            let values: Vec<i64> = (0..n_rows).map(|_| rng.sample(dist) as i64).collect();
            Ok(Column::new(name.into(), values))
        }

        Distribution::Poisson { lambda } => {
            let dist = Poisson::new(*lambda).map_err(|e| {
                GenerationError::ColumnCreation(format!("Poisson distribution error: {}", e))
            })?;
            let values: Vec<i64> = (0..n_rows).map(|_| rng.sample(dist) as i64).collect();
            Ok(Column::new(name.into(), values))
        }

        Distribution::Exponential { rate } => {
            let dist = Exp::new(*rate).map_err(|e| {
                GenerationError::ColumnCreation(format!("Exponential distribution error: {}", e))
            })?;
            let values: Vec<f64> = (0..n_rows).map(|_| rng.sample(dist)).collect();
            Ok(Column::new(name.into(), values))
        }

        Distribution::Bernoulli { p } => {
            let values: Vec<i32> = (0..n_rows)
                .map(|_| if rng.r#gen::<f64>() < *p { 1 } else { 0 })
                .collect();
            Ok(Column::new(name.into(), values))
        }

        Distribution::Categorical { categories, weights } => {
            let normalized_weights: Vec<f64> = match weights {
                Some(w) => {
                    let sum: f64 = w.iter().sum();
                    w.iter().map(|x| x / sum).collect()
                }
                None => {
                    let uniform_weight = 1.0 / categories.len() as f64;
                    vec![uniform_weight; categories.len()]
                }
            };

            // Compute cumulative weights
            let mut cumulative: Vec<f64> = Vec::with_capacity(normalized_weights.len());
            let mut sum = 0.0;
            for w in &normalized_weights {
                sum += w;
                cumulative.push(sum);
            }

            let values: Vec<String> = (0..n_rows)
                .map(|_| {
                    let r: f64 = rng.r#gen();
                    for (i, &c) in cumulative.iter().enumerate() {
                        if r < c {
                            return categories[i].clone();
                        }
                    }
                    categories.last().unwrap().clone()
                })
                .collect();

            Ok(Column::new(name.into(), values))
        }

        Distribution::UniformInt { min, max } => {
            let values: Vec<i64> = (0..n_rows)
                .map(|_| rng.r#gen_range(*min..=*max))
                .collect();
            Ok(Column::new(name.into(), values))
        }

        Distribution::Sequence { start } => {
            let values: Vec<i64> = (0..n_rows as i64).map(|i| start + i).collect();
            Ok(Column::new(name.into(), values))
        }

        Distribution::Constant { value } => {
            let values: Vec<f64> = vec![*value; n_rows];
            Ok(Column::new(name.into(), values))
        }

        Distribution::ConstantString { value } => {
            let values: Vec<String> = vec![value.clone(); n_rows];
            Ok(Column::new(name.into(), values))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_uniform() {
        let columns = vec![ColumnSpec::new("x", Distribution::Uniform { min: 0.0, max: 10.0 })];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        assert_eq!(dataset.df().height(), 100);
        assert_eq!(dataset.df().width(), 1);

        let col = dataset.df().column("x").unwrap();
        let values = col.f64().unwrap();

        for val in values.into_no_null_iter() {
            assert!(val >= 0.0 && val < 10.0);
        }
    }

    #[test]
    fn test_generate_normal() {
        let columns = vec![ColumnSpec::new("x", Distribution::Normal { mean: 100.0, std: 10.0 })];
        let dataset = generate_random_data(1000, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values: Vec<f64> = col.f64().unwrap().into_no_null_iter().collect();

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        // Mean should be close to 100 with 1000 samples
        assert!((mean - 100.0).abs() < 2.0);
    }

    #[test]
    fn test_generate_binomial() {
        let columns = vec![ColumnSpec::new("x", Distribution::Binomial { n: 10, p: 0.5 })];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values = col.i64().unwrap();

        for val in values.into_no_null_iter() {
            assert!(val >= 0 && val <= 10);
        }
    }

    #[test]
    fn test_generate_poisson() {
        let columns = vec![ColumnSpec::new("x", Distribution::Poisson { lambda: 5.0 })];
        let dataset = generate_random_data(1000, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values: Vec<i64> = col.i64().unwrap().into_no_null_iter().collect();

        let mean: f64 = values.iter().sum::<i64>() as f64 / values.len() as f64;
        // Mean should be close to lambda (5.0)
        assert!((mean - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_generate_exponential() {
        let columns = vec![ColumnSpec::new("x", Distribution::Exponential { rate: 1.0 })];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values = col.f64().unwrap();

        for val in values.into_no_null_iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_generate_bernoulli() {
        let columns = vec![ColumnSpec::new("x", Distribution::Bernoulli { p: 0.5 })];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values = col.i32().unwrap();

        for val in values.into_no_null_iter() {
            assert!(val == 0 || val == 1);
        }
    }

    #[test]
    fn test_generate_categorical() {
        let categories = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Categorical {
                categories: categories.clone(),
                weights: None,
            },
        )];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values = col.str().unwrap();

        for val in values.into_iter().flatten() {
            assert!(categories.contains(&val.to_string()));
        }
    }

    #[test]
    fn test_generate_categorical_with_weights() {
        let categories = vec!["A".to_string(), "B".to_string()];
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Categorical {
                categories,
                weights: Some(vec![0.9, 0.1]),
            },
        )];
        let dataset = generate_random_data(1000, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values: Vec<&str> = col.str().unwrap().into_no_null_iter().collect();

        let a_count = values.iter().filter(|&&x| x == "A").count();
        // With weights 0.9/0.1, A should be much more common
        assert!(a_count > 800);
    }

    #[test]
    fn test_generate_uniform_int() {
        let columns = vec![ColumnSpec::new("x", Distribution::UniformInt { min: 1, max: 6 })];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values = col.i64().unwrap();

        for val in values.into_no_null_iter() {
            assert!(val >= 1 && val <= 6);
        }
    }

    #[test]
    fn test_generate_sequence() {
        let columns = vec![ColumnSpec::new("id", Distribution::Sequence { start: 1 })];
        let dataset = generate_random_data(5, columns, Some(42)).unwrap();

        let col = dataset.df().column("id").unwrap();
        let values: Vec<i64> = col.i64().unwrap().into_no_null_iter().collect();

        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_generate_constant() {
        let columns = vec![ColumnSpec::new("x", Distribution::Constant { value: 42.0 })];
        let dataset = generate_random_data(5, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values: Vec<f64> = col.f64().unwrap().into_no_null_iter().collect();

        assert_eq!(values, vec![42.0, 42.0, 42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_generate_constant_string() {
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::ConstantString {
                value: "test".to_string(),
            },
        )];
        let dataset = generate_random_data(3, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values: Vec<&str> = col.str().unwrap().into_no_null_iter().collect();

        assert_eq!(values, vec!["test", "test", "test"]);
    }

    #[test]
    fn test_generate_multiple_columns() {
        let columns = vec![
            ColumnSpec::new("id", Distribution::Sequence { start: 1 }),
            ColumnSpec::new("x", Distribution::Normal { mean: 0.0, std: 1.0 }),
            ColumnSpec::new("y", Distribution::Uniform { min: 0.0, max: 100.0 }),
            ColumnSpec::new(
                "group",
                Distribution::Categorical {
                    categories: vec!["A".to_string(), "B".to_string()],
                    weights: None,
                },
            ),
        ];
        let dataset = generate_random_data(50, columns, Some(42)).unwrap();

        assert_eq!(dataset.df().height(), 50);
        assert_eq!(dataset.df().width(), 4);
        assert!(dataset.df().column("id").is_ok());
        assert!(dataset.df().column("x").is_ok());
        assert!(dataset.df().column("y").is_ok());
        assert!(dataset.df().column("group").is_ok());
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let columns = vec![ColumnSpec::new("x", Distribution::Normal { mean: 0.0, std: 1.0 })];

        let dataset1 = generate_random_data(10, columns.clone(), Some(42)).unwrap();
        let dataset2 = generate_random_data(10, columns, Some(42)).unwrap();

        let values1: Vec<f64> = dataset1
            .df()
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let values2: Vec<f64> = dataset2
            .df()
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        assert_eq!(values1, values2);
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid uniform
        let columns = vec![ColumnSpec::new("x", Distribution::Uniform { min: 10.0, max: 0.0 })];
        assert!(generate_random_data(10, columns, None).is_err());

        // Invalid normal std
        let columns = vec![ColumnSpec::new("x", Distribution::Normal { mean: 0.0, std: -1.0 })];
        assert!(generate_random_data(10, columns, None).is_err());

        // Invalid binomial p
        let columns = vec![ColumnSpec::new("x", Distribution::Binomial { n: 10, p: 1.5 })];
        assert!(generate_random_data(10, columns, None).is_err());

        // Invalid poisson lambda
        let columns = vec![ColumnSpec::new("x", Distribution::Poisson { lambda: 0.0 })];
        assert!(generate_random_data(10, columns, None).is_err());

        // Empty categories
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Categorical {
                categories: vec![],
                weights: None,
            },
        )];
        assert!(generate_random_data(10, columns, None).is_err());

        // Zero rows
        let columns = vec![ColumnSpec::new("x", Distribution::Uniform { min: 0.0, max: 1.0 })];
        assert!(generate_random_data(0, columns, None).is_err());

        // Empty columns
        assert!(generate_random_data(10, vec![], None).is_err());
    }
}
