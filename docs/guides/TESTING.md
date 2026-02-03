# Testing Guide

This document describes how to run and write tests for prompt2analytics.

## Running Tests

### Quick Test Commands

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p p2a-core
cargo test -p p2a-mcp
cargo test -p p2a-cli

# Run a specific test by name
cargo test test_ols_basic

# Run tests matching a pattern
cargo test panel
cargo test discrete

# Run tests with output (useful for debugging)
cargo test -- --nocapture

# Run validation tests only
cargo test -p p2a-core -- test_validate
```

### Test Categories

#### Unit Tests

Located in each module's `#[cfg(test)]` block. Run with:
```bash
cargo test -p p2a-core --lib
```

#### Integration Tests

Located in `tests/` directories. Run with:
```bash
cargo test -p p2a-core --test '*'
```

#### Validation Tests

Compare Rust implementations against R reference results. Run with:
```bash
cargo test -p p2a-core -- test_validate
```

## Test Runtime Expectations

Test runtime varies by system resources. Approximate times on a modern laptop:

| Test Suite | Command | Expected Time |
|------------|---------|---------------|
| Core library (unit) | `cargo test -p p2a-core --lib` | 30-60 seconds |
| Panel data tests | `cargo test panel` | 5-10 seconds |
| Discrete choice tests | `cargo test discrete` | 10-15 seconds |
| ML clustering tests | `cargo test clustering` | 15-30 seconds |
| Full test suite | `cargo test` | 2-5 minutes |

**Note**: First run will be slower due to compilation.

## System Requirements

### Memory

Some tests require significant memory:

- **Clustering benchmarks**: ~2GB for large dataset tests
- **PCA tests**: ~1GB for large matrix decomposition
- **Panel data tests**: ~500MB for unbalanced panel with many groups

If tests fail with out-of-memory errors:
```bash
# Limit parallel test threads
cargo test -- --test-threads=2

# Or run specific test suites sequentially
cargo test panel && cargo test discrete && cargo test ml
```

### CPU

Tests benefit from multiple cores:
- Benchmarks use Rayon for parallelization
- Set thread count via environment: `RAYON_NUM_THREADS=4 cargo test`

## Writing Tests

### Test Data Guidelines

Test datasets should have realistic noise to avoid degenerate cases:

```rust
// Good: y has noise
let df = df! {
    "y" => [1.1, 1.9, 3.2, 3.8, 5.1],  // y ≈ x + noise
    "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
}?;

// Bad: perfect fit causes zero residuals
let df = df! {
    "y" => [1.0, 2.0, 3.0, 4.0, 5.0],  // y = x exactly
    "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
}?;
```

### Test Tolerances

For numerical comparisons, use appropriate tolerances:

```rust
use approx::assert_relative_eq;

// For coefficients (usually well-conditioned)
assert_relative_eq!(result.coefficients[0], expected, epsilon = 1e-6);

// For p-values near boundaries
assert_relative_eq!(result.p_value, expected, epsilon = 1e-4);

// For ill-conditioned results
assert_relative_eq!(result.value, expected, max_relative = 0.01);
```

### Test Structure

Follow the existing pattern:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_feature_basic() {
        // 1. Create test data
        let df = df! { ... }?;
        let dataset = Dataset::new(df);

        // 2. Run the method
        let result = run_feature(&dataset, ...)?;

        // 3. Assert expected properties
        assert_eq!(result.n_obs, 100);
        assert!(result.r_squared > 0.0);
        assert!(result.r_squared <= 1.0);
    }

    #[test]
    fn test_feature_edge_case() {
        // Test edge cases: empty data, single observation, etc.
    }

    #[test]
    fn test_feature_error_handling() {
        // Test that appropriate errors are returned
        let result = run_feature(&bad_data, ...);
        assert!(result.is_err());
    }
}
```

## Validation Framework

### Structure

```
validation/
├── README.md                   # Validation methodology
├── reference_implementations.md # R code for reference values
├── R/                          # R scripts for generating reference values
│   ├── generate_ols_reference.R
│   └── ...
└── data/                       # Reference datasets (optional)
```

### Adding a New Validation Test

1. **Create R reference script** in `validation/R/`:
   ```r
   # validation/R/generate_feature_reference.R
   library(lmtest)

   # Create test data
   set.seed(42)
   x <- rnorm(100)
   y <- 2 + 3*x + rnorm(100)

   # Run R implementation
   model <- lm(y ~ x)

   # Print reference values
   cat("Expected coefficients:", coef(model), "\n")
   cat("Expected R-squared:", summary(model)$r.squared, "\n")
   ```

2. **Add Rust test** using the reference values:
   ```rust
   #[test]
   fn test_validate_feature() {
       // Use same data and seed as R script
       let data = generate_test_data(42);
       let result = run_feature(&data)?;

       // Compare to R reference values
       assert_relative_eq!(result.r_squared, 0.8765, epsilon = 1e-4);
   }
   ```

3. **Document** in `validation/reference_implementations.md`

## Benchmarks

### Running Benchmarks

```bash
# All benchmarks
cargo bench -p p2a-core

# Specific benchmark
cargo bench -p p2a-core --bench regression_benchmarks

# With specific function
cargo bench -p p2a-core -- ols
```

### Performance Profiling

```bash
# Generate flamegraph (requires flamegraph crate)
cargo flamegraph --bench regression_benchmarks

# Memory profiling (requires heaptrack)
heaptrack ./target/release/deps/regression_benchmarks-*
```

## Continuous Integration

CI runs automatically on pull requests:

1. **Build check**: `cargo check --all-targets`
2. **Lint**: `cargo clippy --all-targets --all-features`
3. **Format**: `cargo fmt --check`
4. **Tests**: `cargo test`
5. **Coverage**: Uploaded to Codecov

See `.github/workflows/ci.yml` for the full CI configuration.

## Troubleshooting

### Common Issues

**Test hangs on clustering**:
DBSCAN with large datasets can be slow. Use smaller test data or increase timeout.

**Memory allocation failure**:
Reduce parallel test threads: `cargo test -- --test-threads=1`

**Floating point comparison failures**:
Use `approx` crate with appropriate tolerances instead of exact equality.

**Polars schema errors**:
Ensure test data types match expected schema. Use explicit casts if needed.

### Getting Help

- Check existing tests for patterns
- Review `CLAUDE.md` for coding guidelines
- Open an issue on GitHub for persistent problems
