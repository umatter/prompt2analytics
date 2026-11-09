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

# NA and numeric source guards, the Tier 2 and Tier 3 MCP-path and CLI tests
cargo test -p p2a-validation --test na_guard
cargo test -p p2a-validation --test numeric_guard
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,websocket" --test tier2
cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier2
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,websocket" --test tier3
cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier3

# Documentation tests (see "Documentation tests")
cargo test -p p2a-cli --no-default-features --features spectral-analysis --test docs_examples
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,websocket" --test tool_examples_doc
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

The p2a-core `tests/` directory includes `benchmark_ml_rust.rs` and `benchmark_fast_implementations.rs`. Their
timing-only tests are `#[ignore]`d, so `cargo test` reports them as ignored (see
[Benchmark tests](#benchmark-tests-ignored-by-default)); use `--lib` or a named `--test` for a fast correctness
loop. The p2a-mcp integration tests are listed in [p2a-mcp integration tests](#p2a-mcp-integration-tests).

The MCP-path and CLI tests of the October 2026 review's Tier 2 packages live in two `tier2` test
binaries, `crates/p2a-mcp/tests/tier2/` and `crates/p2a-cli/tests/tier2/`, with one module per
package (`wp<nn>_<topic>.rs`); a package edits only its own module. The shared helpers are in each
binary's `common.rs`: MCP `session_with(case_id)`, `session_with_datasets`, `call` (the dispatcher the
HTTP API uses) and `text`; CLI `temp_session()`, whose `TempSession` runs the real `p2a` binary with
`ok`, `err`, `json`, `json_docs`, `load` and `load_as`. The MCP binary needs the `http` feature.

#### Tier 3 test binaries

The Tier 3 packages follow the same layout in two `tier3` binaries:

- `crates/p2a-mcp/tests/tier3/` (`#![cfg(feature = "http")]`): `wp05_mechanics`, `wp53_llm_routes`,
  `wp54_ui_persistence`, `wp55_causal`, `wp56_stats_ts`, `wp57_ml_spatial`, `wp58_models`, `wp58_text`,
  `wp59_data`. The `wp53_llm_routes::conversations` and `wp53_llm_routes::chat_conversation` modules need `db`.
- `crates/p2a-cli/tests/tier3/`: `wp06_flags`, `wp21_did`, `t3perf4_vecm`, `wp60_models`, `wp61_other`.

```bash
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,websocket" --test tier3   # the CI set
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,db" --test tier3          # with the db modules
cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier3
```

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
| `tier2/` (`main.rs`, one module per package) | Tier 2 MCP-path tests through `call_tool_with_session`: missing values, integer columns and the R pins of the migrated tools | `#![cfg(feature = "http")]` | the CI set |
| `tier3/` (`main.rs`, one module per package) | Tier 3 MCP-path tests (see [Tier 3 test binaries](#tier-3-test-binaries)) | `#![cfg(feature = "http")]`; two `wp53_llm_routes` modules need `db` | the CI set, plus `http,llm,auth,db` |
| `tool_examples_doc.rs` | the MCP documentation (see [Documentation tests](#documentation-tests)) | none; the `database` tools are checked only in builds that have them | the CI set and the default features |
| `serve_limits_hold.rs` | a server holds its limits until its last connection or websocket session closes: after an aborted `start_http_transport` with a keep-alive connection open, and after a graceful shutdown with a websocket session open, a local start is refused and the old connection keeps the caps; once it closes, the local start is accepted. Its tests run one at a time, since each ends by installing the off profile | `#![cfg(feature = "http")]`; the websocket test needs `websocket` | the CI set |

Commands for the feature sets CI does not build:

```bash
cargo test -p p2a-mcp --features full                                          # db + everything
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,db"      # sessions.rs
cargo check -p p2a-mcp --no-default-features                                   # stdio-only build
cargo test -p p2a-core --features database data::database                      # SQL lockdown (duckdb-sys)
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,database"   # the SQL tool handlers
cargo test -p p2a-mcp --test tool_examples_doc                                 # the database tools' examples, exact tools reference
```

`llm_egress.rs` relies on the Off defaults: run it with `P2A_LLM_ALLOWED_HOSTS`, `P2A_LIMITS` and
`P2A_LIMIT_OVERRIDES` unset.

## Documentation tests

Two test binaries hold the documentation to the code. Both run in CI's test step.

| Binary | What it checks | Gate | Regenerate |
|---|---|---|---|
| `crates/p2a-cli/tests/docs_examples.rs` | every fenced `p2a` example in `README.md`, `docs/guides/CLI_REFERENCE.md`, `docs/cookbook.md` and `CLAUDE.md` runs through the real binary (without its `--session`, in an empty directory) and is not refused by the parser, does not panic or hang, and names a session; no placeholder in a shell fence; no refused option value; every after-help example of the binary parses and names a session; the generated sections are what the binary's help prints; every source path `CLAUDE.md` names exists | `#![cfg(feature = "spectral-analysis")]` (the CI set) | `P2A_UPDATE_DOCS=1 cargo test -p p2a-cli --no-default-features --features spectral-analysis --test docs_examples`, or `cargo run -p p2a-cli --example generate_cli_reference` after a build |
| `crates/p2a-mcp/tests/tool_examples_doc.rs` | every `/tool key:value` example in `docs/guides/MCP_TOOL_EXAMPLES.md` and `README.md` names a registered tool, only its fields and every required field, with values of the field's type, and no refused option value; the guide's headings and parameter bullets; README's generated tool table and the tool names in its MCP sections; `docs/mcp/tools-reference.md`; every "N tools" count in the docs | none (the `database` tools' examples are skipped and the reference compared row by row without that feature) | `cargo run -p p2a-mcp --example generate_tool_docs` (default features; it refuses to write the reference without `database`), or `P2A_UPDATE_DOCS=1` with the test |

- Edit only outside the `BEGIN GENERATED`/`END GENERATED` lines; the text between them is rewritten.
- Run the two regenerations one after the other, not in parallel, and with plain `cargo test` (one test
  binary at a time) rather than a parallel runner: both rewrite `README.md`, and two rewrites at once
  would lose one of the sections.
- The CLI reference is rendered from the `p2a` binary of the CI feature set. A CLI command behind a feature
  other than `spectral-analysis` would make it build-dependent; none exists.

## Test Runtime Expectations

Test runtime varies by system resources. Approximate times on a modern laptop:

| Test Suite | Command | Expected Time |
|------------|---------|---------------|
| Core library (unit) | `cargo test -p p2a-core --lib` | 30-60 seconds |
| Panel data tests | `cargo test panel` | 5-10 seconds |
| Discrete choice tests | `cargo test discrete` | 10-15 seconds |
| ML clustering tests | `cargo test clustering` | 15-30 seconds |
| Full test suite | `cargo test` | 2-5 minutes |
| NA guard | `cargo test -p p2a-validation --test na_guard` | about 1 second (after the build) |
| Tier 2 MCP-path tests | `cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,websocket" --test tier2` | under 1 second (after the build) |
| Tier 2 CLI tests | `cargo test -p p2a-cli --no-default-features --features spectral-analysis --test tier2` | a few seconds (after the build) |
| Numeric guard | `cargo test -p p2a-validation --test numeric_guard` | a few seconds (after the build) |
| CLI documentation tests | `cargo test -p p2a-cli --no-default-features --features spectral-analysis --test docs_examples` | under 10 seconds (after the build; several hundred runs of the debug `p2a` binary) |
| MCP documentation tests | `cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,websocket" --test tool_examples_doc` | under 1 second (after the build) |

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

### Missing values and integer columns

Fixtures for code that reads user columns (`p2a_core::data::columns`, CLAUDE.md "Reading user
columns") follow the R harness's conventions:

- An empty CSV field is missing, and the harness writes R's `NA` as an empty field.
- Whole-number doubles load as `Int64` (assert `is.integer` in the case).
- Every dataset with missing cells keeps a never-missing `id` column.
- NaN and ±Inf exist only in Rust-built fixtures (`f64::NAN`, `f64::INFINITY`).

Every migrated entry point gets four fixture tests that fail before the migration:

- an `Int64` column gives the same result as the same values as `Float64`;
- a null: `NaPolicy::Omit` gives the complete-row result plus `missing.n_dropped`, and `Fail` gives
  `NullValues` naming the column;
- a text column gives `NonNumericColumn`;
- an absent column gives `ColumnNotFound`.

A test is registered in `validation/methods/*.json`, and named `test_validate_*`, exactly when it
compares at least one value of its R case through `p2a_validation::Check` (a count R computes counts).
A test that checks only an error variant, a text or a property is named `test_na_*`, `test_mcp_*` or
`test_cli_*` and is not registered; it loads its CSV through `p2a_validation::repo_path` (CLI:
`TempSession::load_as`), because the registry lint refuses `Case::load` in an unregistered test.

### The numeric guard

`crates/p2a-validation/tests/numeric_guard.rs` walks the product crates' `src` (skipping `#[cfg(test)]`
items) for five forbidden numeric patterns: `partial-cmp-unwrap`
(`partial_cmp(..).unwrap*` comparators), `one-minus-cdf` (`1 - cdf` p-values), `abramowitz-stegun` (local
approximation constants), `critical-value-literal` (`1.96`, `2.576`, `1.645` as literals) and
`discarded-inverse-condition` (a `safe_inverse` result whose condition element is dropped). It prints
`file:line` for each. A deliberate use carries `// numeric-ok: <reason>` or, for an inverse,
`// inverse-ok: <reason>`; the module documentation is the reference. `cargo test -p p2a-validation` runs
it, in CI too.

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

### Benchmark tests (ignored by default)

A few timing programs are written as tests rather than `benches/` targets. They time code and
assert nothing, so they carry `#[ignore = "benchmark; run with --release -- --ignored"]`:
`cargo test` and CI report them as ignored instead of running them in a debug build.

| Test | File | What it times |
|------|------|---------------|
| `benchmark_rust_ml` | `crates/p2a-core/tests/benchmark_ml_rust.rs` | kernel SVM (RBF and linear), CART and ROC/AUC at n = 1,000 to 20,000 |
| `benchmark_roc_auc_implementations`, `benchmark_svm_implementations`, `benchmark_svm_linear`, `benchmark_xgboost_implementations`, `benchmark_mboost_implementations`, `summary_comparison` | `crates/p2a-core/tests/benchmark_fast_implementations.rs` | fast against original ROC/AUC, SVM, XGBoost and mboost; `summary_comparison` prints notes only |
| `test_hdbscan_full_benchmark` | `crates/p2a-core/src/ml/advanced_clustering_mod/hdbscan.rs` | HDBSCAN on large synthetic data |

`verify_fast_roc_accuracy` in `benchmark_fast_implementations.rs` checks the fast AUC against the
original one and runs with the normal tests.

Run the benchmarks in a release build:

```bash
cargo test -p p2a-core --release --no-default-features --features ci --test benchmark_ml_rust -- --ignored --nocapture
cargo test -p p2a-core --release --no-default-features --features ci --test benchmark_fast_implementations -- --ignored --nocapture
cargo test -p p2a-core --release --no-default-features --features ci --lib -- hdbscan_full_benchmark --ignored --nocapture
```

Cost: before they were ignored, CI's debug build spent 1,792.67 s on `benchmark_rust_ml` and 24.59 s on
the seven tests of `benchmark_fast_implementations.rs` (August 2026). The kernel SVM at n = 20,000
needs about 3.1 GB of memory; at opt-level 1 its RBF and linear fits took about 150 s each.

Timing assertions follow the same rule: a test asserts wall-clock time only inside
`if !cfg!(debug_assertions) { .. }` or after `if cfg!(debug_assertions) { return; }`, so a debug-build
`cargo test` never depends on the machine's speed or load.

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
2. **Keep duckdb-sys out**: `cargo tree --locked -p <pkg> <feature set> -i libduckdb-sys` must report no match
   for p2a-core (`ci`), p2a-mcp (`http,llm,auth,websocket`) and p2a-cli (`spectral-analysis`)
3. **Tests** (`RUST_TEST_THREADS=4`):
   - `cargo test -p p2a-core --tests --no-default-features --features ci`
   - `cargo test -p p2a-mcp --tests --no-default-features --features "http,llm,auth,websocket"` (includes
     `tool_examples_doc`)
   - `cargo test -p p2a-cli --tests --no-default-features --features spectral-analysis` (includes
     `docs_examples`)
   - `cargo test -p p2a-text`
   - `cargo test -p p2a-chat --lib`
4. **Doctests**: p2a-core with `ci`, p2a-mcp with `http,llm,auth,websocket`
5. **Validation harness, registry lint and runner tests** (after the doctests, in the same job):
   - `cargo test -p p2a-validation` (includes the NA guard, `tests/na_guard.rs`, and the numeric guard,
     `tests/numeric_guard.rs`)
   - the three test lists, `cargo test -p <pkg> --tests <the feature set of item 3> -- --list > target/<pkg>-tests.txt 2>&1`
   - `python3 validation/tools/validation_status.py lint --strict --tests target/p2a-core-tests.txt target/p2a-mcp-tests.txt target/p2a-cli-tests.txt`
   - `P2A_CHECK_STATUS_PAGE=1 python3 -m unittest discover -s validation/tools/tests -v` (includes the
     status-page check: `validation/VALIDATION_STATUS.md`'s `test_validate` counts must equal the lists)
   - `bash validation/tools/tests/test_run_validation.sh`

   CI does not install R; the R drift check (`Rscript validation/r/run.R --check`) runs in
   `validation/run_validation.sh` locally.
6. **Clippy** on p2a-core (`ci`), p2a-mcp (`http,llm,auth,websocket`), p2a-cli (`spectral-analysis`),
   p2a-chat, p2a-text and `cargo clippy -p p2a-validation --all-targets`; it reports but does not gate
7. **p2a-dioxus**: `cargo check --no-default-features --features web --target wasm32-unknown-unknown`,
   plus the host tests with `web`

CI does not build the `database` feature of p2a-core (duckdb-sys) or of p2a-mcp and p2a-cli, the `db` feature of
p2a-mcp, `cuda`, release builds or the Dioxus desktop target; run those locally (commands above). A
separate Security audit workflow (`.github/workflows/audit.yml`) runs `cargo audit` weekly and on
every change to `Cargo.lock`, any `Cargo.toml` or the workflow itself.

Benchmark-only tests are `#[ignore]`d, so the Test job reports them as ignored (see
[Benchmark tests](#benchmark-tests-ignored-by-default)). No workflow measures code coverage.

## Timing comparisons

A method rewrite is timed against the code it replaces (and against R where R is timed) before it merges;
more than about 1.1x slower than the current code at n = 100, 1,000 and 10,000 needs the user's sign-off. No
timing assertion is committed.

- Build the base commit in a separate worktree (`git worktree add ../p2a-base <commit>`) with `--release` and
  the CI feature set (`--no-default-features --features ci` for p2a-core), so both sides share flags.
- Fix `RAYON_NUM_THREADS` (for example `RAYON_NUM_THREADS=4`) on both sides, and take the median of 5 runs at
  n = 100, 1,000 and 10,000.
- Time with an uncommitted scratch program or a local copy of a bench, not a test. One Criterion group:
  `cargo bench -p p2a-core --bench <bench> -- <group>`; the custom-harness benches (`harness = false`, see
  `bench_utils.rs`) run one bench file with `cargo bench -p p2a-core --bench <bench>`, with the sizes edited
  locally.
- Report the three medians for each side in the commit body or the package report.

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

**`na_guard` reports `file:line: f64-without-cast` (or `into_no_null_iter`)**:
Read the column through `p2a_core::data::columns` (in p2a-mcp: `tools::common::numeric_column(s)` /
`numeric_matrix` with an explicit `NaPolicy`). If the data cannot hold nulls (a column the code built,
a comparison mask), add `// na-ok: <reason>` on the line or the line above.

**`numeric_guard` reports `file:line`**:
Sort NaN-free data with `f64::total_cmp`, take upper tails from the direct-tail helpers of
`traits::estimator`, take quantiles from its quantile helpers instead of literals, invert with
`linalg::solve::sym_inverse` (or use `safe_inverse`'s condition), or add the marker its module documentation
names (`// numeric-ok: <reason>`, `// inverse-ok: <reason>`).

**A documentation test fails**:
`docs_examples` and `tool_examples_doc` print `file:line` and what is wrong. A documented command, flag, tool
or field that no longer exists is fixed in the document; a stale generated section is regenerated (see
[Documentation tests](#documentation-tests)), never edited by hand.

**Link failures or doctests that "couldn't compile"**:
A full disk and a distro `rustdoc` that does not match its `libLLVM` both look like broken code.
`CLAUDE.md` ("Two traps on a dev box: disk and rustdoc") gives the checks and the workarounds; run
`df -h /` first.

### Getting Help

- Check existing tests for patterns
- Review `CLAUDE.md` for coding guidelines
- Open an issue on GitHub for persistent problems
