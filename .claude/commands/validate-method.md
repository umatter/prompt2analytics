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

5. Report the artifact status:
   ```
   === Method: Ordinary Least Squares (ols) ===
   R equivalent: lm() from stats package
   Rust module: crates/p2a-core/src/regression/ols.rs

   Artifacts:
     [X] R benchmark script: ../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_regression.R
     [X] Criterion benchmark: crates/p2a-core/benches/regression_benchmarks.rs (OLS_Standard)
     [X] Validation doc: validation/regression/ols.md
     [X] Rust tests: test_validate_ols, test_validate_longley
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
   - Search for tests matching the patterns in `test_patterns`
   - Report which tests exist

### STEP 3: RUN VALIDATION (if mode includes "validate")

Execute these steps to validate numerical correctness:

1. **Run Rust validation tests:**
   ```bash
   cargo test -p p2a-core -- {{test_pattern}} --nocapture
   ```
   Run for each pattern in `test_patterns`. Capture output.

2. **Parse Rust test output:**
   - Look for coefficient values, standard errors, p-values, R-squared
   - Note any assertion failures

3. **Run R validation script (if exists):**
   - If `r_benchmark_script` exists and contains validation code, run it
   - Otherwise, check for `validation/scripts/{{method}}_validation.R`

4. **Compare results:**
   - Apply tolerances from registry:
     - Coefficients: 1e-6 (small), 1e-8 (medium), 1e-10 (large sample)
     - Standard errors: 1e-4 (small), 1e-6 (medium), 1e-8 (large)
     - P-values: 0.01 (small), 0.001 (medium), 0.0001 (large)
   - Report pass/fail for each metric

5. **Generate validation report:**
   ```
   === Validation Results: OLS ===

   Test: Longley Dataset (n=16)
     Coefficient β₁: R=0.01506, Rust=0.01506, Diff=2.3e-11 [PASS]
     SE(β₁): R=0.08492, Rust=0.08492, Diff=1.1e-10 [PASS]
     R²: R=0.9955, Rust=0.9955, Diff=1e-12 [PASS]

   Test: Synthetic DGP (n=1000)
     All 5 coefficients within tolerance [PASS]
     All 5 standard errors within tolerance [PASS]

   Overall: 12/12 checks passed
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
   - Path: `validation/reports/{{method}}_{{date}}.md`
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

---

## HANDLING MISSING ARTIFACTS

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
  [X] Rust tests: test_validate_ols, test_validate_longley

Running validation tests...
[cargo test output]

=== Validation Results: OLS ===

Test: Longley Dataset (n=16)
  Coefficient (Intercept): R=-3482.26, Rust=-3482.26, Diff=1.2e-10 [PASS]
  Coefficient (GNP.deflator): R=0.01506, Rust=0.01506, Diff=2.3e-11 [PASS]
  R²: R=0.9955, Rust=0.9955, Diff=1e-12 [PASS]

Overall: 12/12 checks passed

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
# Run Rust validation tests
cargo test -p p2a-core -- test_validate_{{method}} --nocapture

# Run all validation tests
cargo test -p p2a-core -- test_validate --nocapture

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

1. Verify same random seed (42) in both R and Rust
2. Check data generation matches (same DGP)
3. Verify algorithm parameters match (e.g., max iterations)
4. Some methods have intentional differences (document in validation doc)

### Criterion benchmark not running

1. Ensure the benchmark is registered in `criterion_group!`
2. Check that the benchmark file is in `Cargo.toml` under `[[bench]]`
3. Run with verbose output: `cargo bench -p p2a-core -- --verbose`

---

## SEE ALSO

- `.claude/skills/validation-benchmarking/SKILL.md` - Detailed validation workflow
- `.claude/tooling/validation/README.md` - Template documentation
- `validation/README.md` - Validation docs index
- `../prompt2analytics-paper/performance/comparisons/r_comparison/run_all_benchmarks.R` - Run all R benchmarks