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

Tests named `test_validate_*` come in two kinds. *R-pinned* tests load a committed R reference
case (`validation/expected/<case>.json`) and compare every number the method reports at the
tolerances of `p2a_validation::Tol`; they are registered in `validation/methods/*.json`. The
others check properties, known data-generating processes or quoted values, and compare no
committed R output. `validation/VALIDATION_STATUS.md` (generated) says which methods are
R-pinned. Run them with:
```bash
validation/run_validation.sh                  # every test_validate test (CI feature sets) + Rscript validation/r/run.R --check
validation/run_validation.sh --rust-only --category regression
cargo test -p p2a-core -- test_validate       # p2a-core only, default features
```

## p2a-mcp integration tests

Each file in `crates/p2a-mcp/tests/` covers one topic. The servers are built in-process with
`transport::http::build_app` and driven with `tower::ServiceExt::oneshot`, so almost nothing binds a
socket.

| File | What it covers | Compile gate | Command |
|---|---|---|---|
| `integration_tests.rs` | server construction, core methods through the server | none | any feature set |
| `governor.rs` | the tool governor, deployment modes, Host check, CORS, embedded start, `/ws` routing | `#![cfg(feature = "http")]`; the `/ws` test needs `websocket`, a few tests need `llm` | the CI set |
| `limits_e2e.rs` | request limits on the HTTP, LLM, websocket and stdio paths, the response-size cap, the audit trail | module `http_paths` needs `http`; module `stdio_path` is ungated | the CI set, and `cargo test -p p2a-mcp --no-default-features --test limits_e2e` for stdio only |
| `dataset_quotas.rs` | the governor's dataset quotas against what datasets really cost (wide frames, many small frames, categorical labels, parquet files whose decoded size the footer gives before decoding), and the refusal of parquet files polars cannot read | `#![cfg(feature = "http")]` | the CI set |
| `sessions.rs` | the session registry, conversation ownership, persistence bounds, the DB-path rule | `#![cfg(all(feature = "http", feature = "db"))]` | `--no-default-features --features "http,llm,auth,db"` (not in CI) |
| `orphan_sweep_memory.rs` | what the orphan sweeps of a RocksDB conversation store read and hold, measured with a counting allocator | `#![cfg(feature = "db")]` | `--features full --test orphan_sweep_memory` (not in CI) |
| `llm_egress.rs` | egress guard, upstream errors and retries, the key rule, the tool allowlist, context budget, chat-loop bounds | `#![cfg(all(feature = "http", feature = "llm"))]`; the conversation test needs `db` | the CI set, plus `--features full` |
| `isolation.rs` | per-session directories, symbolic links, the disk quota, session export and import, the stale-directory purge | `#![cfg(feature = "http")]` | the CI set |
| `uploads.rs` | `upload_dataset` decodes an uploaded Parquet file from the request's bytes and stores nothing (not in the session directory, not in the temporary folder) | `#![cfg(feature = "http")]` | the CI set |
| `serve_limits.rs` | `serve` and `start_http_transport` install the request limits of an open public configuration, overrides included, on a bound socket; while it runs, a start of other limits (`serve`, `start_transport` over HTTP or stdio, `start_embedded_server`, which returns the refusal) or of the same limits with another allocation ceiling or session-directory setting is refused | `#![cfg(feature = "http")]` | the CI set |
| `serve_limits_hold.rs` | a server holds its limits until its last connection or websocket session closes: after an aborted `start_http_transport` with a keep-alive connection open, and after a graceful shutdown with a websocket session open, a local start is refused and the old connection keeps the caps; once it closes, the local start is accepted. Its tests run one at a time, since each ends by installing the off profile | `#![cfg(feature = "http")]`; the websocket test needs `websocket` | the CI set |

Commands for the feature sets CI does not build:

```bash
cargo test -p p2a-mcp --features full                                          # db + everything
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,db"      # sessions.rs
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

R-pinned tests use `p2a_validation::Tol` (the table in `validation/README.md`: `Tol::coef(n)`,
`Tol::se(n)`, `Tol::stat(n)`, `Tol::p_value(n)`, `Tol::iterative(rel)`, `Tol::printed(d)`,
`Tol::exact()`); the `approx` examples below apply to other unit tests.

For numerical comparisons in other unit tests, use appropriate tolerances:

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
  `limits::with_limits` and `resource::with_alloc_ceiling`. Do not call `limits::install` or
  `resource::set_alloc_ceiling_bytes` per test: they are process-wide, and the tests of one binary
  run in parallel. The exceptions: a test binary whose tests all share one configuration installs
  it once inside a `OnceLock` (`crates/p2a-mcp/tests/isolation.rs`), and p2a-core's
  `global_ceiling_round_trips` sets only a harmless ceiling (`usize::MAX - 1`) and restores `None`.
  `transport::http::serve` and `start_http_transport` install the limits and policies of the
  configuration they serve, and refuse a start whose limits or policies (allocation ceiling,
  session directories) differ from those of a server still running in the process; a server holds
  them until its last connection or websocket session has closed, which can be after `serve`
  returned. So a test serves a configuration whose resource profile is off, that has no overrides
  and that keeps the default policies (it installs the off profile, the default), or runs in a
  binary whose servers all share one configuration (`crates/p2a-mcp/tests/serve_limits.rs`), or
  whose tests run one at a time (`serve_limits_hold.rs`).
- **Complexity rewrites** need an equivalence test against a naive reference on noisy random data,
  plus R-pinned values where R has the method.

## Validation Framework

`validation/README.md` is the full reference; in short:

### Structure

```
validation/
├── README.md                 # what "validated" means, running, the workflow below, method index
├── VALIDATION_STATUS.md      # generated by tools/validation_status.py render; never edited by hand
├── run_validation.sh         # test_validate tests in the CI feature sets + R drift check
├── r/harness.R, r/run.R      # R case harness; run.R --check | --write | --list
├── r/cases/<area>.R          # R reference cases, one file per area
├── expected/<case>.json      # one R reference case per file (schema 1); expected/*.csv are legacy
├── methods/<area>.json       # registry: method -> cases, tests, quantities, tolerance, deviations
├── datasets/                 # committed reference data; datasets/generated/ is written by R cases
├── tools/                    # validation_status.py (select | collect | lint | render) and its tests
└── <category>/<method>.md    # method documents
```

### Adding an R-pinned validation

1. **Write a case** in `validation/r/cases/<area>.R`: one `case(id, area, method, function(ctx) ...)`
   per reference, calling the R function named as the method's reference; simulated data through
   `set.seed()` and `ctx$gen_dataset()` (written, then read back), with noise.
2. **Write and check it**: `Rscript validation/r/run.R --write <id>`, then `--check`.
3. **Write the test** `test_validate_<method>_<dataset>[_<variant>]` in the module it tests,
   pinning every number the method reports:
   ```rust
   #[test]
   fn test_validate_ols_grunfeld() {
       use p2a_validation::{Check, Tol};
       let (case, ds) = crate::test_support::load_case("ols_grunfeld");
       let r = run_ols(&ds, "inv", &["value", "capital"], true, CovarianceType::HC1).unwrap();
       let n = r.n_obs;
       let names = r.variable_names.as_slice();
       let coef: Vec<f64> = r.coefficients.iter().map(|c| c.estimate).collect();
       let se: Vec<f64> = r.coefficients.iter().map(|c| c.std_error).collect();
       Check::new(&case)
           .named("coef", names, &coef, Tol::coef(n))
           .named("se_hc1", names, &se, Tol::se(n))
           .scalar("adj_r2", r.adj_r_squared, Tol::stat(n))
           .finish();
   }
   ```
   p2a-cli and p2a-mcp tests load cases with `p2a_validation::Case::load`.
4. **Register it** in `validation/methods/<area>.json` (cases, tests, quantities, tolerance,
   deviations).
5. **Update the method document** and the `validation/README.md` index row.
6. **Run** `python3 validation/tools/validation_status.py lint --strict --tests target/*-tests.txt`
   and `validation/run_validation.sh`.
7. **Regenerate the status page.** Adding, removing or renaming a `test_validate` test changes
   the counts CI checks: regenerate `validation/VALIDATION_STATUS.md` with
   `validation/run_validation.sh --write-status` before the change reaches main (during the
   review campaign: at the wave integration, not in a lane).

A mismatch is a defect: fix it, or record a registry deviation and keep the quantity out of
the `Check` with a `// deviation <finding id>` comment. Never widen a tolerance to make it pass.

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
   - `cargo test -p p2a-mcp --tests --no-default-features --features "http,llm,auth,websocket"`
   - `cargo test -p p2a-cli --tests --no-default-features --features spectral-analysis`
3. **Doctests**: p2a-core with `ci`, p2a-mcp with `http,llm,auth,websocket`
4. **Validation harness, registry lint and runner tests** (after the doctests, in the same job):
   - `cargo test -p p2a-validation`
   - the three test lists, `cargo test -p <pkg> --tests <the feature set of item 2> -- --list > target/<pkg>-tests.txt 2>&1`
   - `python3 validation/tools/validation_status.py lint --strict --tests target/p2a-core-tests.txt target/p2a-mcp-tests.txt target/p2a-cli-tests.txt`
   - `P2A_CHECK_STATUS_PAGE=1 python3 -m unittest discover -s validation/tools/tests -v` (includes the
     status-page check: `validation/VALIDATION_STATUS.md`'s `test_validate` counts must equal the lists)
   - `bash validation/tools/tests/test_run_validation.sh`

   CI does not install R; the R drift check (`Rscript validation/r/run.R --check`) runs in
   `validation/run_validation.sh` locally.
5. **Clippy** on p2a-core (`ci`), p2a-mcp (`http,llm,auth,websocket`), p2a-cli (`spectral-analysis`),
   p2a-chat and `cargo clippy -p p2a-validation --all-targets`; it reports but does not gate
6. **p2a-dioxus**: `cargo check --no-default-features --features web --target wasm32-unknown-unknown`,
   plus the host tests with `web`

CI does not build the `database` feature of p2a-core (duckdb-sys), the `db` feature of
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
