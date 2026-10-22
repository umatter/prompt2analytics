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

The p2a-core `tests/` directory includes `benchmark_ml_rust.rs` and `benchmark_fast_implementations.rs`,
which are benchmarks and run for a long time in a debug build; use `--lib` or a named `--test` for a
fast correctness loop. The p2a-mcp integration tests are listed in
[p2a-mcp integration tests](#p2a-mcp-integration-tests).

#### Validation Tests

Compare Rust implementations against R reference results. Run with:
```bash
cargo test -p p2a-core -- test_validate
```

## p2a-mcp integration tests

Each file in `crates/p2a-mcp/tests/` covers one topic. The servers are built in-process with
`transport::http::build_app` and driven with `tower::ServiceExt::oneshot`, so almost nothing binds a
socket.

| File | What it covers | Compile gate | Command |
|---|---|---|---|
| `integration_tests.rs` | server construction, core methods through the server | none | any feature set |
| `governor.rs` | the tool governor, deployment modes, Host check, CORS, embedded start, `/ws` routing | `#![cfg(feature = "http")]`; the `/ws` test needs `websocket`, a few tests need `llm` | the CI set, plus `--features full` for `/ws` |
| `limits_e2e.rs` | request limits on the HTTP, LLM, websocket and stdio paths, the response-size cap, the audit trail | module `http_paths` needs `http`; module `stdio_path` is ungated | the CI set, and `cargo test -p p2a-mcp --no-default-features --test limits_e2e` for stdio only |
| `sessions.rs` | the session registry, conversation ownership, persistence bounds, the DB-path rule | `#![cfg(all(feature = "http", feature = "db"))]` | `--no-default-features --features "http,llm,auth,db"` (not in CI) |
| `llm_egress.rs` | egress guard, upstream errors and retries, the key rule, the tool allowlist, context budget, chat-loop bounds | `#![cfg(all(feature = "http", feature = "llm"))]`; the conversation test needs `db` | the CI set, plus `--features full` |
| `isolation.rs` | per-session directories, symbolic links, the disk quota, session export and import, the stale-directory purge | `#![cfg(feature = "http")]` | the CI set |

Commands for the feature sets CI does not build:

```bash
cargo test -p p2a-mcp --features full                                          # db + websocket + everything
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,db"      # sessions.rs without websocket
cargo check -p p2a-mcp --no-default-features                                   # stdio-only build
cargo test -p p2a-core --features database data::database                      # SQL lockdown (duckdb-sys)
```

`llm_egress.rs` relies on the Off defaults: run it with `P2A_LLM_ALLOWED_HOSTS`, `P2A_LIMITS` and
`P2A_LIMIT_OVERRIDES` unset.

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

### Resource-governance tests

Tests for the resource guards (core) and the server's limits follow these conventions:

- **Abort-class regressions** call the core function with the absurd value (an overflowing size, a
  count far beyond the data) and assert `Err`. Before the fix such a test aborts the test binary,
  which is the failure it guards against.
- **Loops that used to hang** are tested through the extracted guard (the range or parameter check),
  never through the full path, so a regression cannot hang CI.
- **No wall-clock assertions.** A timing assertion is gated with `if !cfg!(debug_assertions)` so it
  runs only in release builds; prefer none at all.
- **Governor tests** use synthetic closures that loop on `resource::checkpoint()` until a release
  flag is set, with bounded polling and generous margins.
- **Limits and allocation ceilings** are selected per test with the thread-local
  `limits::with_limits` and `resource::with_alloc_ceiling`. Never call `limits::install` or
  `resource::set_alloc_ceiling_bytes` in a test: they are process-wide, and the tests of one binary
  run in parallel.
- **Complexity rewrites** need an equivalence test against a naive reference on noisy random data,
  plus R-pinned values where R has the method.

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

CI (`.github/workflows/ci.yml`) runs on pushes to `main` and on pull requests, on the free GitHub
runner, with feature sets chosen to keep it cheap:

1. **Format**: `cargo fmt --all -- --check`
2. **Tests** (`cargo test --tests`):
   - `cargo test -p p2a-core --tests --no-default-features --features ci`
   - `cargo test -p p2a-mcp --tests --no-default-features --features "http,llm,auth"`
   - `cargo test -p p2a-cli --tests --no-default-features --features spectral-analysis`
3. **Doctests**: p2a-core with `ci`, p2a-mcp with `http,llm,auth`
4. **Clippy** on p2a-core (`ci`), p2a-mcp (`http,llm,auth`), p2a-cli (`spectral-analysis`) and
   p2a-chat; it reports but does not gate
5. **p2a-dioxus**: `cargo check --no-default-features --features web --target wasm32-unknown-unknown`,
   plus the host tests with `web`

CI does not build the `database` feature of p2a-core (duckdb-sys), the `db` and `websocket` features of
p2a-mcp, `cuda`, release builds or the Dioxus desktop target; run those locally (commands above). A
separate Security audit workflow (`.github/workflows/audit.yml`) runs `cargo audit` weekly and on
every change to `Cargo.lock`, any `Cargo.toml` or the workflow itself.

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

**Link failures or doctests that "couldn't compile"**:
A full disk and a distro `rustdoc` that does not match its `libLLVM` both look like broken code.
`CLAUDE.md` ("Two traps on a dev box: disk and rustdoc") gives the checks and the workarounds; run
`df -h /` first.

### Getting Help

- Check existing tests for patterns
- Review `CLAUDE.md` for coding guidelines
- Open an issue on GitHub for persistent problems
