# /validate-method - Unified R vs Rust Validation Command

Validate and benchmark a p2a-core method against its R equivalent.

## Usage

```
/validate-method <method-name> [options]
```

**Arguments:**
- `method-name`: The method to validate (e.g., `ols`, `t_test`, `kmeans`, `panel_fe`)

**Options (passed after method name):**
- `--mode all|validate|benchmark` - Run mode (default: all)
- `--sizes n1,n2,...` - Custom dataset sizes (default: 100,1000,10000)
- `--format markdown|json|terminal` - Output format (default: terminal)

## Examples

```bash
/validate-method ols                          # Full validation + benchmark
/validate-method t_test --mode validate       # Only numerical validation
/validate-method kmeans --mode benchmark      # Only performance comparison
/validate-method panel_fe --sizes 500,5000    # Custom dataset sizes
```

---

## WORKFLOW

When the user invokes this command with a method name, follow this workflow:

### STEP 1: DISCOVERY

1. Read the method registry at `.claude/tooling/validation/method_registry.json`
2. Look up the requested method by its ID (e.g., "ols", "t_test")
3. If not found by exact match, search for partial matches or suggest similar methods
4. Extract the method's metadata:
   - `display_name`: Human-readable name
   - `r_name`: R function name
   - `r_packages`: Required R packages
   - `category`: Method category
   - `rust_module`: Rust source file path
   - `rust_function`: Rust function name
   - `r_benchmark_script`: Path to R benchmark (may not exist)
   - `criterion_benchmark`: Path to Criterion benchmark file
   - `criterion_group`: Criterion benchmark group name
   - `validation_doc`: Path to validation documentation
   - `test_patterns`: Rust test name patterns

5. Read `validation/methods/*.json` (the methods registry) and find the entries for the method
   (by `id`, `name` or `doc`): their `cases`, `tests` and `deviations`. The registry is
   authoritative for tests; `test_patterns` in `method_registry.json` are search hints, and
   some are stale. A method with no registry entry is not R-pinned.

6. Report the artifact status:
   ```
   === Method: Ordinary Least Squares (ols) ===
   R equivalent: lm() from stats package
   Rust module: crates/p2a-core/src/regression/ols.rs

   Artifacts:
     [X] R benchmark script: ../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_regression.R
     [X] Criterion benchmark: crates/p2a-core/benches/regression_benchmarks.rs (OLS_Standard)
     [X] Validation doc: validation/regression/ols.md
     [X] Registry: validation/methods/regression.json, id ols (cases ols_grunfeld, ols_mtcars, ols_longley)
     [X] Registered tests: p2a-core::regression::ols::tests::test_validate_ols_grunfeld, ...
   ```

### STEP 2: ARTIFACT CHECK

For each artifact type:

1. **R Benchmark Script**
   - If exists: Note the path, will use it in Step 4
   - If missing: Offer to generate from template (`.claude/tooling/validation/templates/r_benchmark.R.template`)

2. **Criterion Benchmark**
   - If exists: Note the benchmark group name
   - If missing: Warn that manual addition is needed (Criterion benchmarks require code)

3. **Validation Documentation**
   - If exists: Read it to understand expected tolerances and test cases
   - If missing: Note that it should be created after validation

4. **Rust Tests**
   - Report the registered tests (`tests` of the registry entries) and whether each is in
     `cargo test -p <pkg> --tests -- --list`
   - Use `test_patterns` only to find unregistered `test_validate_*` tests, and report those as
     not R-pinned

### STEP 3: RUN VALIDATION (if mode includes "validate")

Execute these steps to validate numerical correctness:

1. **Check the R reference cases:**
   ```bash
   Rscript validation/r/run.R --check {{case_ids}}
   ```
   Exit 1 means a recomputed R value differs from the committed case: report it (an R package
   upgrade or a changed case), do not rewrite the case without the user.

2. **Run the registered Rust tests** (the `tests` of the registry entries) with the runner,
   once per category they fall into:
   ```bash
   validation/run_validation.sh --rust-only --category {{category}}
   ```
   A test's category is `cli` for p2a-cli, `mcp` for p2a-mcp, and otherwise the first module
   of its path (`diagnostics` counts as `econometrics`; the `validation_ml*` and
   `validation_boosting_methods` binaries as `ml`). The runner uses each package's CI feature
   set and fails when a test fails, does not compile, or fewer tests run than it selected. It
   selects only tests with `test_validate` in their name, in the categories stats |
   regression | econometrics | forecasting | ml | cli | mcp; run a registered test outside
   them (such as `p2a-core::traits::...`) with the direct form below. Before reporting a
   registered test as passed, find its id in the `tests` map of the run's `results.json`
   (the runner prints its path as `Results: ...`) with the status `ok`; a missing id is a
   failure.

   Direct form, one test or several of one test binary: a registry id is
   `<package>::<libtest path>` for a unit test and `<package>::<binary>::<name>` for an
   integration test (`crates/<package>/tests/<binary>.rs`). Drop the `<package>::` prefix.
   A unit test runs with `--lib` (`--bins` for p2a-cli, which has no library), an
   integration test with `--test <binary>` and the bare name. Use the package's CI feature
   set: p2a-core `--no-default-features --features ci`, p2a-cli `--no-default-features
   --features spectral-analysis`, p2a-mcp `--no-default-features --features
   "http,llm,auth,websocket"`.
   ```bash
   # p2a-core::regression::ols::tests::test_validate_ols_grunfeld
   cargo test -p p2a-core --lib --no-default-features --features ci -- --exact regression::ols::tests::test_validate_ols_grunfeld
   # p2a-cli::exhibits::test_validate_cli_ols
   cargo test -p p2a-cli --test exhibits --no-default-features --features spectral-analysis -- --exact test_validate_cli_ols
   # p2a-mcp::tools::handlers::regression::tests::test_validate_mcp_regression_ols
   cargo test -p p2a-mcp --lib --no-default-features --features "http,llm,auth,websocket" -- --exact tools::handlers::regression::tests::test_validate_mcp_regression_ols
   ```
   `--exact` with a name that matches nothing runs 0 tests and still exits 0. The `N passed`
   of the `test result:` line must equal the number of names given; any other count, 0
   passed included, is a failure, never a pass.

3. **The comparison happens inside the tests** (`p2a_validation::Check` at the `Tol`
   tolerances of `validation/README.md`): a failing test lists every quantity that differs,
   with R's value, Rust's value and the allowed difference. Set `P2A_VALIDATION_VERBOSE=1` to
   print every comparison. Report the registry `deviations` as known differences, with their
   finding ids.

4. **If no case exists** for the method: write one from
   `.claude/tooling/validation/templates/r_validation.R.template` into
   `validation/r/cases/<area>.R`, run `Rscript validation/r/run.R --write <id>`, add a
   `test_validate_*` test that loads it (`crate::test_support::load_case`) and compares with
   `Check`/`Tol`, and add the registry entry. Then run
   `python3 validation/tools/validation_status.py lint --strict --tests target/*-tests.txt`,
   after writing the three lists with `cargo test -p <package> --tests <CI feature set> --
   --list > target/<package>-tests.txt 2>&1` (the commands are in CLAUDE.md's validation
   block).

5. **Generate validation report:**
   ```
   === Validation Results: OLS ===

   R check: 3 case(s) match the committed files (ols_grunfeld, ols_mtcars, ols_longley)
   Registered tests: 4/4 passed (p2a-core, ci features; each id found with status ok)
   Known deviations: none
   Unregistered test_validate tests (not R-pinned): none
   ```

### STEP 4: RUN BENCHMARKS (if mode includes "benchmark")

Execute these steps to compare performance:

1. **Run Rust (Criterion) benchmarks:**
   ```bash
   cargo bench -p p2a-core -- {{criterion_group}}
   ```
   Parse the output for timing information (mean, median, std).

2. **Run R benchmarks:**
   ```bash
   cd ../prompt2analytics-paper/performance/comparisons/r_comparison && Rscript {{r_benchmark_script}}
   ```
   If the R script doesn't exist, offer to generate it from template.

3. **Parse R benchmark results:**
   - Look for CSV or JSON output in `results/` directory
   - Extract mean/median times in microseconds

4. **Calculate speedup factors:**
   For each sample size (100, 1000, 10000):
   ```
   speedup = R_time / Rust_time
   ```

5. **Evaluate against requirements:**
   From registry `performance_requirements`:
   - n=1,000: Rust must be >= 1.5x faster
   - n=10,000: Rust must be >= 2.0x faster
   - n=100,000: Rust should be >= 3.0x faster

6. **Generate performance report:**
   ```
   === Performance Results: OLS ===

   Sample Size | R (us)    | Rust (us) | Speedup | Required | Status
   ------------|-----------|-----------|---------|----------|--------
   n=100       | 45.2      | 28.1      | 1.61x   | N/A      | -
   n=1,000     | 312.5     | 89.3      | 3.50x   | 1.5x     | PASS
   n=10,000    | 8,542.1   | 1,203.4   | 7.10x   | 2.0x     | PASS
   n=100,000   | 245,621   | 32,451    | 7.57x   | 3.0x     | PASS

   Overall: All performance requirements met!
   ```

### STEP 5: GENERATE REPORT

1. **Terminal output (always):**
   - Print combined validation + benchmark results
   - Highlight any failures with actionable suggestions

2. **Save markdown report:**
   - Path: `validation/reports/{{method}}_{{date}}.md` (gitignored; never commit it)
   - Include:
     - Method summary
     - Validation results table
     - Performance comparison table
     - Any warnings or failures
     - Suggestions for improvement

3. **Update method registry (optional):**
   - Add `last_validated` timestamp
   - Update `last_benchmark_date`

4. **Update validation doc (if needed):**
   - If performance numbers are missing or outdated, update them

5. **Do not edit `validation/VALIDATION_STATUS.md`.** It is generated, and only
   `validation/run_validation.sh --write-status` regenerates it (once per wave integration).

---

## HANDLING MISSING ARTIFACTS

### Missing R Reference Case

If the method has no registry entry or its cases are missing:

1. Read the template at `.claude/tooling/validation/templates/r_validation.R.template`
2. Use the R function named as the method's reference (`validation/reference_implementations.md`)
3. Write the case into `validation/r/cases/<area>.R` (ask the user before writing), then
   `Rscript validation/r/run.R --write <id>` and `--check <id>`
4. Add the `test_validate_*` test and the registry entry in `validation/methods/<area>.json`
5. Update the method doc and the `validation/README.md` index row (status from the registry)
6. A new `test_validate` test changes the counts CI checks: `validation/VALIDATION_STATUS.md`
   must be regenerated with `validation/run_validation.sh --write-status` before the change
   reaches main (during the review campaign: at the wave integration, not in a lane)

### Missing R Benchmark Script

If the R benchmark script doesn't exist:

1. Read the template at `.claude/tooling/validation/templates/r_benchmark.R.template`
2. Look at the method's `r_packages`, `r_name`, and similar existing scripts
3. Generate a customized script with proper data generation and benchmark code
4. Ask user for confirmation before writing
5. Write to `../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_{{method}}.R`

### Missing Validation Documentation

If the validation doc doesn't exist:

1. After running validation, offer to create the doc
2. Use the template structure from existing docs in `validation/`
3. Include:
   - Method overview
   - R code examples
   - Results comparison tables with actual numbers
   - Rust test commands

### Missing Criterion Benchmark

This requires code changes:

1. Warn the user that Criterion benchmarks need manual addition
2. Point to the appropriate benchmark file (`criterion_bench_file` from category)
3. Provide a code template for the benchmark function
4. Reference similar benchmarks in the same file

---

## EXAMPLE SESSION

```
User: /validate-method ols

Claude: I'll validate the OLS method against R's lm() function.

=== Method: Ordinary Least Squares (ols) ===
R equivalent: lm() from stats package
Rust module: crates/p2a-core/src/regression/ols.rs

Artifacts:
  [X] R benchmark script: ../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_regression.R
  [X] Criterion benchmark: crates/p2a-core/benches/regression_benchmarks.rs (OLS_Standard)
  [X] Validation doc: validation/regression/ols.md
  [X] Registry: validation/methods/regression.json, id ols (cases ols_grunfeld, ols_mtcars, ols_longley)
  [X] Registered tests: p2a-core::regression::ols::tests::test_validate_ols_grunfeld, ...

Checking R cases and running the registered tests...
[Rscript validation/r/run.R --check output]
[validation/run_validation.sh --rust-only --category regression output]

=== Validation Results: OLS ===

R check: 3 case(s) match the committed files
Registered tests: 4/4 passed (each id found in results.json with status ok)
Known deviations: none

Running benchmarks...
[cargo bench output]
[Rscript output]

=== Performance Results: OLS ===

Sample Size | R (us)    | Rust (us) | Speedup | Required | Status
------------|-----------|-----------|---------|----------|--------
n=100       | 45.2      | 28.1      | 1.61x   | N/A      | -
n=1,000     | 312.5     | 89.3      | 3.50x   | 1.5x     | PASS
n=10,000    | 8,542.1   | 1,203.4   | 7.10x   | 2.0x     | PASS

Overall: All tests passed! Method validated successfully.
Report saved to: validation/reports/ols_2026-01-21.md
```

---

## METHOD REGISTRY REFERENCE

The method registry at `.claude/tooling/validation/method_registry.json` contains:

- **70+ methods** across 12 categories
- R function mappings and required packages
- Paths to existing benchmarks and validation docs
- Test patterns for Rust validation tests
- Tolerance guidelines and performance requirements

**Categories:**
- `regression`: OLS, NLS, LOESS, GLS, smooth splines
- `panel`: Fixed effects, random effects, HDFE, Hausman
- `causal`: IV/2SLS, DiD, synthetic control, RD, treatment effects
- `discrete`: Logit, probit
- `timeseries`: VAR, VECM, IRF, ACF/PACF
- `survival`: Kaplan-Meier, Cox PH, AFT
- `hypothesis`: t-test, ANOVA, chi-squared, Wilcoxon, Shapiro-Wilk, KS
- `multivariate`: MANOVA, factor analysis, canonical correlation
- `correlation`: Correlation tests
- `power`: Power analysis
- `forecasting`: ARIMA, Holt-Winters, STL, Kalman
- `ml`: K-means, DBSCAN, PCA, t-SNE, random forest

---

## QUICK COMMANDS REFERENCE

```bash
# Check the R reference cases (all, or the listed ids)
Rscript validation/r/run.R --check {{case_ids}}

# (Re)write one case after changing it
Rscript validation/r/run.R --write {{case_id}}

# Run the validation tests of a category (CI feature sets)
validation/run_validation.sh --rust-only --category {{category}}

# Run Rust validation tests of a method, printing every comparison
P2A_VALIDATION_VERBOSE=1 cargo test -p p2a-core -- test_validate_{{method}} --nocapture

# Registry, case and doc lint
python3 validation/tools/validation_status.py lint --strict --tests target/*-tests.txt

# Run Rust benchmarks for a method
cargo bench -p p2a-core -- {{criterion_group}}

# Run R benchmarks
Rscript ../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_{{method}}.R

# View Criterion HTML report
open target/criterion/report/index.html
```

---

## PERFORMANCE REQUIREMENTS

The command enforces these speedup factors over R:

| Sample Size | Minimum Speedup | Rationale |
|-------------|-----------------|-----------|
| n=1,000     | 1.5x faster     | Baseline for small datasets |
| n=10,000    | 2.0x faster     | Moderate workloads |
| n=100,000   | 3.0x faster     | Large-scale performance |

**If Rust is slower than R:**
1. Report a warning with specific metrics
2. Suggest optimization areas:
   - Check for unnecessary allocations
   - Review matrix operations (use faer instead of manual loops)
   - Consider parallelization for large n
3. Reference similar optimized implementations

---

## TROUBLESHOOTING

### R benchmark fails to run

1. Check that required packages are installed:
   ```r
   install.packages(c("microbenchmark", "jsonlite"))
   ```
2. Ensure data generation matches Rust's seed (42)
3. Check working directory is project root

### Rust tests not found

1. Search for tests in the module:
   ```bash
   grep -r "fn test_" crates/p2a-core/src/
   ```
2. Tests may use different naming conventions
3. Check both `#[test]` and `#[cfg(test)]` modules

### Results differ significantly

1. R-pinned tests read the case's own dataset (`validation/datasets/` or `datasets/generated/`),
   so R and Rust see the same doubles; check that the test loads the case with `load_case`
2. Verify algorithm parameters match (e.g., max iterations, covariance type)
3. A mismatch is a defect: fix it, or record a registry deviation with its finding id; never
   widen a tolerance to make it pass

### Criterion benchmark not running

1. Ensure the benchmark is registered in `criterion_group!`
2. Check that the benchmark file is in `Cargo.toml` under `[[bench]]`
3. Run with verbose output: `cargo bench -p p2a-core -- --verbose`

---

## SEE ALSO

- `.claude/skills/validation-benchmarking/SKILL.md` - Detailed validation workflow
- `.claude/tooling/validation/README.md` - Template documentation
- `validation/README.md` - Validation docs index, the R-pinned workflow and the `Tol` table
- `validation/methods/README.md` - Methods registry schema
- `validation/r/run.R` - R reference cases (`--check`, `--write`, `--list`)
- `../prompt2analytics-paper/performance/comparisons/r_comparison/run_all_benchmarks.R` - Run all R benchmarks