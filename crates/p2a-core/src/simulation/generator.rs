//! Random data generation implementation.

use polars::prelude::*;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Binomial, Exp, Normal, Poisson, Uniform};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::resource;

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

/// Largest Poisson mean rand_distr 0.4.3 samples accurately. In a line-for-line emulation of
/// its rejection sampler (f64, 20,000 draws per lambda), sd/sqrt(lambda) is 0.999 at 1e13 but
/// 1.058 at 1e14 and 1.174 at 1e15, because cancellation in its log-gamma step inflates the
/// variance; above ~2.6e305, lambda * ln(lambda) overflows and the sampler never returns.
const MAX_POISSON_LAMBDA: f64 = 1e13;
/// Largest binomial n whose draws fit exactly in f64 and i64.
const MAX_BINOMIAL_N: u64 = 1 << 53;

fn invalid(msg: String) -> GenerationError {
    GenerationError::InvalidParameters(msg)
}

/// `Err` naming `what` unless `value` is finite.
fn require_finite(what: &str, value: f64) -> Result<(), GenerationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("{what} ({value}) must be a finite number")))
    }
}

impl Distribution {
    /// Validate distribution parameters.
    ///
    /// Every distribution parameter (other than a constant column's value)
    /// must be finite. In addition: a uniform range must be non-empty and
    /// its width representable, normal and exponential scale parameters
    /// positive, a Poisson mean in `(0, 1e13]`, a binomial `n` in `1..=2^53`,
    /// probabilities in `[0, 1]`, and categorical weights non-negative with
    /// a finite, positive sum.
    pub fn validate(&self) -> Result<(), GenerationError> {
        match self {
            Distribution::Uniform { min, max } => {
                require_finite("Uniform: min", *min)?;
                require_finite("Uniform: max", *max)?;
                if min >= max {
                    return Err(GenerationError::InvalidParameters(format!(
                        "Uniform: min ({}) must be less than max ({})",
                        min, max
                    )));
                }
                if !(max - min).is_finite() {
                    return Err(invalid(format!(
                        "Uniform: the range max - min ({max} - {min}) must be at most {:e}",
                        f64::MAX
                    )));
                }
            }
            Distribution::Normal { mean, std } => {
                require_finite("Normal: mean", *mean)?;
                require_finite("Normal: std", *std)?;
                if *std <= 0.0 {
                    return Err(GenerationError::InvalidParameters(format!(
                        "Normal: std ({}) must be positive",
                        std
                    )));
                }
            }
            Distribution::Binomial { n, p } => {
                if !(0.0..=1.0).contains(p) {
                    return Err(GenerationError::InvalidParameters(format!(
                        "Binomial: p ({}) must be between 0 and 1",
                        p
                    )));
                }
                if *n == 0 {
                    return Err(GenerationError::InvalidParameters(
                        "Binomial: n must be positive".to_string(),
                    ));
                }
                if *n > MAX_BINOMIAL_N {
                    return Err(invalid(format!(
                        "Binomial: n ({n}) must be at most {MAX_BINOMIAL_N} (2^53), \
                         so that every draw is exact"
                    )));
                }
            }
            Distribution::Poisson { lambda } => {
                require_finite("Poisson: lambda", *lambda)?;
                if *lambda <= 0.0 {
                    return Err(GenerationError::InvalidParameters(format!(
                        "Poisson: lambda ({}) must be positive",
                        lambda
                    )));
                }
                if *lambda > MAX_POISSON_LAMBDA {
                    return Err(invalid(format!(
                        "Poisson: lambda ({lambda}) must be at most {MAX_POISSON_LAMBDA:e}; \
                         larger means cannot be sampled accurately (for such counts use a \
                         normal distribution with mean = lambda and std = sqrt(lambda))"
                    )));
                }
            }
            Distribution::Exponential { rate } => {
                require_finite("Exponential: rate", *rate)?;
                if *rate <= 0.0 {
                    return Err(GenerationError::InvalidParameters(format!(
                        "Exponential: rate ({}) must be positive",
                        rate
                    )));
                }
            }
            Distribution::Bernoulli { p } => {
                if !(0.0..=1.0).contains(p) {
                    return Err(GenerationError::InvalidParameters(format!(
                        "Bernoulli: p ({}) must be between 0 and 1",
                        p
                    )));
                }
            }
            Distribution::Categorical {
                categories,
                weights,
            } => {
                if categories.is_empty() {
                    return Err(GenerationError::InvalidParameters(
                        "Categorical: categories cannot be empty".to_string(),
                    ));
                }
                if categories.len() > IdxSize::MAX as usize {
                    return Err(invalid(format!(
                        "Categorical: {} categories exceed the limit of {}",
                        categories.len(),
                        IdxSize::MAX
                    )));
                }
                if let Some(w) = weights {
                    if w.len() != categories.len() {
                        return Err(GenerationError::InvalidParameters(format!(
                            "Categorical: weights length ({}) must match categories length ({})",
                            w.len(),
                            categories.len()
                        )));
                    }
                    if let Some(bad) = w.iter().find(|x| !x.is_finite()) {
                        return Err(invalid(format!(
                            "Categorical: weights must be finite numbers (got {bad})"
                        )));
                    }
                    if w.iter().any(|&x| x < 0.0) {
                        return Err(GenerationError::InvalidParameters(
                            "Categorical: weights must be non-negative".to_string(),
                        ));
                    }
                    let sum: f64 = w.iter().sum();
                    if !sum.is_finite() {
                        return Err(invalid(format!(
                            "Categorical: the weights' sum must be at most {:e}",
                            f64::MAX
                        )));
                    }
                    if sum <= 0.0 {
                        return Err(GenerationError::InvalidParameters(
                            "Categorical: weights must sum to a positive value".to_string(),
                        ));
                    }
                }
            }
            Distribution::UniformInt { min, max } => {
                if min > max {
                    return Err(GenerationError::InvalidParameters(format!(
                        "UniformInt: min ({}) must be <= max ({})",
                        min, max
                    )));
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

    if n_rows > IdxSize::MAX as usize {
        return Err(invalid(format!(
            "n_rows ({n_rows}) must be at most {}",
            IdxSize::MAX
        )));
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
        resource::checkpoint().map_err(resource_err)?;
        let series = generate_column(&col.name, &col.distribution, n_rows, &mut rng)?;
        series_vec.push(series);
    }

    // Create DataFrame
    let df = DataFrame::new(series_vec).map_err(|e| {
        GenerationError::DataFrameCreation(format!("Failed to create DataFrame: {}", e))
    })?;

    Ok(Dataset::new(df).with_name("generated"))
}

/// Rows generated between two cancellation checkpoints.
const CHECKPOINT_ROWS: usize = 1 << 16;

fn resource_err(e: crate::errors::EconError) -> GenerationError {
    GenerationError::InvalidParameters(e.to_string())
}

/// Fallible, cancellable `(0..n_rows).map(|_| draw()).collect()`: the buffer is
/// allocated through [`resource::try_with_capacity`] and the budget is checked
/// before every block of [`CHECKPOINT_ROWS`] rows. `draw` is called exactly
/// `n_rows` times, in order, so seeded output is the same as the plain collect.
fn fill<T>(
    n_rows: usize,
    what: &str,
    mut draw: impl FnMut() -> T,
) -> Result<Vec<T>, GenerationError> {
    let mut out = resource::try_with_capacity::<T>(n_rows, what).map_err(resource_err)?;
    for i in 0..n_rows {
        if i % CHECKPOINT_ROWS == 0 {
            resource::checkpoint().map_err(resource_err)?;
        }
        out.push(draw());
    }
    Ok(out)
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
            let values: Vec<f64> = fill(n_rows, "uniform column", || rng.sample(dist))?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::Normal { mean, std } => {
            let dist = Normal::new(*mean, *std).map_err(|e| {
                GenerationError::ColumnCreation(format!("Normal distribution error: {}", e))
            })?;
            let values: Vec<f64> = fill(n_rows, "normal column", || rng.sample(dist))?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::Binomial { n, p } => {
            let dist = Binomial::new(*n, *p).map_err(|e| {
                GenerationError::ColumnCreation(format!("Binomial distribution error: {}", e))
            })?;
            let values: Vec<i64> = fill(n_rows, "binomial column", || rng.sample(dist) as i64)?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::Poisson { lambda } => {
            let dist = Poisson::new(*lambda).map_err(|e| {
                GenerationError::ColumnCreation(format!("Poisson distribution error: {}", e))
            })?;
            let values: Vec<i64> = fill(n_rows, "Poisson column", || rng.sample(dist) as i64)?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::Exponential { rate } => {
            let dist = Exp::new(*rate).map_err(|e| {
                GenerationError::ColumnCreation(format!("Exponential distribution error: {}", e))
            })?;
            let values: Vec<f64> = fill(n_rows, "exponential column", || rng.sample(dist))?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::Bernoulli { p } => {
            let values: Vec<i32> = fill(n_rows, "Bernoulli column", || {
                if rng.r#gen::<f64>() < *p { 1 } else { 0 }
            })?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::Categorical {
            categories,
            weights,
        } => {
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

            // Each row draws one f64 and takes the first label whose cumulative
            // weight exceeds it (the last label when rounding leaves none), found
            // by binary search. Rows hold indices into a k-row label column, so
            // each label's bytes are stored once.
            let k = categories.len();
            let labels = StringChunked::from_slice(name.into(), categories.as_slice());
            let idx: Vec<IdxSize> = fill(n_rows, "category indices", || {
                let r: f64 = rng.r#gen();
                cumulative.partition_point(|&c| c <= r).min(k - 1) as IdxSize
            })?;
            let column = labels.take(idx.as_slice()).map_err(|e| {
                GenerationError::ColumnCreation(format!("Categorical column error: {}", e))
            })?;
            Ok(column.into_column())
        }

        Distribution::UniformInt { min, max } => {
            let values: Vec<i64> = fill(n_rows, "integer column", || rng.r#gen_range(*min..=*max))?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::Sequence { start } => {
            // n_rows >= 1 (checked by the caller); the last value must fit in i64,
            // after which `start + i` cannot overflow for any row.
            let fits = i64::try_from(n_rows - 1)
                .ok()
                .and_then(|span| start.checked_add(span))
                .is_some();
            if !fits {
                return Err(invalid(format!(
                    "Sequence: start ({start}) plus n_rows - 1 ({}) exceeds the largest \
                     integer {}",
                    n_rows - 1,
                    i64::MAX
                )));
            }
            let mut offset: i64 = 0;
            let values: Vec<i64> = fill(n_rows, "sequence column", || {
                let v = start + offset;
                offset += 1;
                v
            })?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::Constant { value } => {
            let values: Vec<f64> = fill(n_rows, "constant column", || *value)?;
            Ok(Column::new(name.into(), values))
        }

        Distribution::ConstantString { value } => {
            // One stored copy of the text behind a 16-byte view per row.
            resource::check_alloc::<[u8; 16]>(n_rows, "string column").map_err(resource_err)?;
            Ok(StringChunked::full(name.into(), value, n_rows).into_column())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource;

    /// Sum of the string bytes the column physically stores (views of <= 12 bytes are inline).
    fn stored_string_bytes(ds: &Dataset, col: &str) -> usize {
        ds.df()
            .column(col)
            .unwrap()
            .str()
            .unwrap()
            .downcast_iter()
            .map(|arr| arr.total_buffer_len())
            .sum()
    }

    #[test]
    fn test_constant_string_stores_value_once() {
        let value = "x".repeat(1000);
        let ds = generate_random_data(
            10_000,
            vec![ColumnSpec::new(
                "s",
                Distribution::ConstantString {
                    value: value.clone(),
                },
            )],
            Some(1),
        )
        .unwrap();
        assert!(
            stored_string_bytes(&ds, "s") <= 2 * value.len(),
            "{}",
            stored_string_bytes(&ds, "s")
        );
        let ca = ds.df().column("s").unwrap().str().unwrap().clone();
        assert_eq!(ca.len(), 10_000);
        assert!(ca.into_no_null_iter().all(|s| s == value));
    }

    #[test]
    fn test_categorical_labels_stored_once() {
        // 100-byte labels
        let categories: Vec<String> = (0..3).map(|i| format!("{i}{}", "y".repeat(99))).collect();
        let ds = generate_random_data(
            10_000,
            vec![ColumnSpec::new(
                "c",
                Distribution::Categorical {
                    categories: categories.clone(),
                    weights: None,
                },
            )],
            Some(3),
        )
        .unwrap();
        assert!(
            stored_string_bytes(&ds, "c") <= 2 * 300,
            "{}",
            stored_string_bytes(&ds, "c")
        );
        let ca = ds.df().column("c").unwrap().str().unwrap().clone();
        assert!(
            ca.into_no_null_iter()
                .all(|s| categories.iter().any(|c| c == s))
        );
    }

    #[test]
    fn test_categorical_binary_search_matches_linear_scan() {
        // Pin: the pre-change sampler, consuming one f64 per row from the same seeded StdRng.
        let categories: Vec<String> = (0..50).map(|i| format!("c{i}")).collect();
        let mut wrng = StdRng::seed_from_u64(7);
        let mut weights: Vec<f64> = (0..50).map(|_| wrng.r#gen::<f64>()).collect();
        // zero-weight runs, including the last label
        for i in [0, 17, 18, 49] {
            weights[i] = 0.0;
        }
        let n = 20_000;
        let ds = generate_random_data(
            n,
            vec![ColumnSpec::new(
                "x",
                Distribution::Categorical {
                    categories: categories.clone(),
                    weights: Some(weights.clone()),
                },
            )],
            Some(99),
        )
        .unwrap();
        let got: Vec<String> = ds
            .df()
            .column("x")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .map(str::to_string)
            .collect();
        let sum: f64 = weights.iter().sum();
        let mut cumulative = Vec::new();
        let mut acc = 0.0;
        for w in &weights {
            acc += w / sum;
            cumulative.push(acc);
        }
        let mut rng = StdRng::seed_from_u64(99);
        let expected: Vec<String> = (0..n)
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
        assert_eq!(got, expected);
    }

    #[test]
    fn test_numeric_draws_unchanged_by_chunked_fill() {
        // Pin: the chunked, fallible fill must consume the RNG exactly like `(0..n).map(sample)`.
        let n = 200_001; // crosses three 65,536-row chunks
        let ds = generate_random_data(
            n,
            vec![
                ColumnSpec::new(
                    "a",
                    Distribution::Normal {
                        mean: 1.0,
                        std: 2.0,
                    },
                ),
                ColumnSpec::new("b", Distribution::Poisson { lambda: 30.0 }),
            ],
            Some(5),
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        let a: Vec<f64> = (0..n)
            .map(|_| rng.sample(Normal::new(1.0, 2.0).unwrap()))
            .collect();
        let b: Vec<i64> = (0..n)
            .map(|_| rng.sample(Poisson::new(30.0).unwrap()) as i64)
            .collect();
        assert_eq!(
            ds.df()
                .column("a")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            a
        );
        assert_eq!(
            ds.df()
                .column("b")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            b
        );
    }

    #[test]
    fn test_generation_stops_when_cancelled() {
        let token = resource::CancelToken::new();
        token.cancel();
        let out = resource::with_budget(resource::Budget::new(None, token), || {
            generate_random_data(
                200_000,
                vec![ColumnSpec::new(
                    "x",
                    Distribution::Normal {
                        mean: 0.0,
                        std: 1.0,
                    },
                )],
                Some(1),
            )
        });
        // Dataset is not Debug, so no unwrap_err.
        let err = match out {
            Ok(_) => panic!("generation ignored the cancelled budget"),
            Err(e) => e.to_string(),
        };
        // EconError::Cancelled displays "Computation stopped: ..."
        assert!(err.contains("stopped"), "{err}");
    }

    #[test]
    fn test_impossible_row_count_is_an_error() {
        let columns = vec![ColumnSpec::new("x", Distribution::Constant { value: 1.0 })];
        assert!(generate_random_data(usize::MAX / 2, columns, Some(1)).is_err());
    }

    #[test]
    fn test_generate_uniform() {
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Uniform {
                min: 0.0,
                max: 10.0,
            },
        )];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        assert_eq!(dataset.df().height(), 100);
        assert_eq!(dataset.df().width(), 1);

        let col = dataset.df().column("x").unwrap();
        let values = col.f64().unwrap();

        for val in values.into_no_null_iter() {
            assert!((0.0..10.0).contains(&val));
        }
    }

    #[test]
    fn test_generate_normal() {
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Normal {
                mean: 100.0,
                std: 10.0,
            },
        )];
        let dataset = generate_random_data(1000, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values: Vec<f64> = col.f64().unwrap().into_no_null_iter().collect();

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        // Mean should be close to 100 with 1000 samples
        assert!((mean - 100.0).abs() < 2.0);
    }

    #[test]
    fn test_generate_binomial() {
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Binomial { n: 10, p: 0.5 },
        )];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values = col.i64().unwrap();

        for val in values.into_no_null_iter() {
            assert!((0..=10).contains(&val));
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
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Exponential { rate: 1.0 },
        )];
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
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::UniformInt { min: 1, max: 6 },
        )];
        let dataset = generate_random_data(100, columns, Some(42)).unwrap();

        let col = dataset.df().column("x").unwrap();
        let values = col.i64().unwrap();

        for val in values.into_no_null_iter() {
            assert!((1..=6).contains(&val));
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
            ColumnSpec::new(
                "x",
                Distribution::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
            ),
            ColumnSpec::new(
                "y",
                Distribution::Uniform {
                    min: 0.0,
                    max: 100.0,
                },
            ),
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
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Normal {
                mean: 0.0,
                std: 1.0,
            },
        )];

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
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Uniform {
                min: 10.0,
                max: 0.0,
            },
        )];
        assert!(generate_random_data(10, columns, None).is_err());

        // Invalid normal std
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Normal {
                mean: 0.0,
                std: -1.0,
            },
        )];
        assert!(generate_random_data(10, columns, None).is_err());

        // Invalid binomial p
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Binomial { n: 10, p: 1.5 },
        )];
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
        let columns = vec![ColumnSpec::new(
            "x",
            Distribution::Uniform { min: 0.0, max: 1.0 },
        )];
        assert!(generate_random_data(0, columns, None).is_err());

        // Empty columns
        assert!(generate_random_data(10, vec![], None).is_err());
    }

    #[test]
    fn test_poisson_lambda_bound_rejected_before_sampling() {
        // rand_distr 0.4.3's rejection loop never ends once lambda*ln(lambda) overflows (~2.6e305);
        // between 9.2e18 and that, every draw saturates to i64::MAX.
        for lambda in [
            1e308,
            2.6e305,
            1e19,
            1e14,
            1.0000001e13,
            f64::INFINITY,
            f64::NAN,
        ] {
            let err = Distribution::Poisson { lambda }.validate().unwrap_err();
            assert!(
                matches!(err, GenerationError::InvalidParameters(_)),
                "lambda={lambda}"
            );
        }
        let msg = Distribution::Poisson { lambda: 1e308 }
            .validate()
            .unwrap_err()
            .to_string();
        assert!(msg.contains("1e13"), "{msg}");
        assert!(Distribution::Poisson { lambda: 1e13 }.validate().is_ok());
    }

    #[test]
    fn test_non_finite_and_overflowing_parameters_rejected() {
        let two = || vec!["a".to_string(), "b".to_string()];
        let bad = [
            Distribution::Uniform {
                min: f64::NEG_INFINITY,
                max: 0.0,
            },
            // rand's Uniform::new panics on this today
            Distribution::Uniform {
                min: f64::NAN,
                max: 1.0,
            },
            // max - min overflows: "range overflow" panic
            Distribution::Uniform {
                min: -1e308,
                max: 1e308,
            },
            Distribution::Normal {
                mean: f64::INFINITY,
                std: 1.0,
            },
            Distribution::Normal {
                mean: 0.0,
                std: f64::NAN,
            },
            Distribution::Normal {
                mean: 0.0,
                std: f64::INFINITY,
            },
            Distribution::Binomial { n: 10, p: f64::NAN },
            // draws do not fit i64; rand_distr asserts
            Distribution::Binomial {
                n: u64::MAX,
                p: 0.5,
            },
            Distribution::Exponential {
                rate: f64::INFINITY,
            },
            Distribution::Exponential { rate: f64::NAN },
            Distribution::Bernoulli { p: f64::NAN },
            Distribution::Categorical {
                categories: two(),
                weights: Some(vec![f64::NAN, 1.0]),
            },
            Distribution::Categorical {
                categories: two(),
                weights: Some(vec![f64::INFINITY, 1.0]),
            },
            // the sum overflows
            Distribution::Categorical {
                categories: two(),
                weights: Some(vec![1e308, 1e308]),
            },
        ];
        for d in bad {
            assert!(d.validate().is_err(), "{d:?} must be rejected");
        }
        assert!(
            Distribution::Binomial { n: 1 << 53, p: 0.5 }
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn test_poisson_saturating_lambda_is_an_error_not_i64_max() {
        // 1e19 terminates in rand_distr but every draw is i64::MAX; it must now be refused.
        let columns = vec![ColumnSpec::new("x", Distribution::Poisson { lambda: 1e19 })];
        assert!(generate_random_data(3, columns, Some(1)).is_err());
    }

    #[test]
    fn test_poisson_large_lambda_within_bound_is_accurate() {
        let columns = vec![ColumnSpec::new("x", Distribution::Poisson { lambda: 1e13 })];
        let ds = generate_random_data(1000, columns, Some(42)).unwrap();
        let v: Vec<f64> = ds
            .df()
            .column("x")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .map(|x| x as f64)
            .collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
        assert!(((mean - 1e13) / 1e13).abs() < 1e-6, "mean {mean}");
        // sd of the variance ratio at n = 1000 is about 0.045; 0.2 is more than 4 sd
        assert!((var / 1e13 - 1.0).abs() < 0.2, "var/lambda {}", var / 1e13);
    }

    #[test]
    fn test_sequence_overflow_rejected() {
        let cols = |start| vec![ColumnSpec::new("id", Distribution::Sequence { start })];
        assert!(generate_random_data(3, cols(i64::MAX - 1), None).is_err());
        let ds = generate_random_data(3, cols(i64::MAX - 2), None).unwrap();
        let v: Vec<i64> = ds
            .df()
            .column("id")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(v, vec![i64::MAX - 2, i64::MAX - 1, i64::MAX]);
    }
}
