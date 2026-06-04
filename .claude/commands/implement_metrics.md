---
description: Implement a new econometric method from a URL or file containing method documentation
argument-hint: <url-or-path>
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch, Task
---

# Implement Econometric Method

You are implementing a new econometric method for the prompt2analytics Rust library.

## Source
The user has provided: $ARGUMENTS

## Context
- Current codebase patterns: @CLAUDE.md
- Econometrics guide: @docs/guides/ECONOMETRICS_GUIDE.md
- Existing implementations: @crates/p2a-core/src/econometrics/

---

## Workflow

### Phase 0: Existing Implementation Check (MANDATORY FIRST STEP)

**Before any research or implementation, check what already exists.**

1. **Extract method identifiers** from the source URL/file:
   - Method name (e.g., "Generalized Least Squares", "GLS", "FGLS")
   - Common abbreviations and aliases
   - Related methods (e.g., GLS relates to OLS, WLS)

2. **Search the codebase** for existing implementations:
   ```
   # Search Rust source files
   Grep: method name, abbreviations in crates/p2a-core/src/

   # Check module files
   Read: crates/p2a-core/src/regression/mod.rs
   Read: crates/p2a-core/src/econometrics/mod.rs

   # Check MCP tools
   Grep: method name in crates/p2a-mcp/src/server.rs
   ```

3. **Check documentation**:
   ```
   Read: docs/guides/ECONOMETRICS_GUIDE.md
   Read: docs/guides/MCP_TOOL_EXAMPLES.md
   ```

4. **Identify reusable components**:
   - Matrix operations in `linalg/matrix_ops.rs`
   - Variance estimators in `regression/ols.rs`
   - Panel data utilities in `econometrics/panel.rs`
   - Distribution functions in `traits/estimator.rs`

5. **Report findings to user**:

   **If method is FULLY IMPLEMENTED:**
   ```
   ⚠️ METHOD ALREADY EXISTS

   The requested method "[Method Name]" is already implemented in this repository:

   - Implementation: crates/p2a-core/src/[path].rs
   - MCP Tool: [tool_name]
   - Documentation: docs/guides/ECONOMETRICS_GUIDE.md#[section]

   No implementation needed. Would you like me to:
   1. Show you how to use the existing implementation?
   2. Explain the current implementation details?
   3. Suggest enhancements to the existing method?
   ```
   **STOP HERE** — Do not proceed to Phase 1.

   **If method is PARTIALLY IMPLEMENTED:**
   ```
   ℹ️ PARTIAL IMPLEMENTATION FOUND

   Related functionality exists:
   - [Existing component 1]: [location]
   - [Existing component 2]: [location]

   Missing components for full [Method Name] support:
   - [Missing feature 1]
   - [Missing feature 2]

   Proceeding to implement only the missing parts...
   ```
   Continue to Phase 1, focusing only on gaps.

   **If method is NOT IMPLEMENTED:**
   ```
   ✓ NEW METHOD

   "[Method Name]" is not currently implemented.

   Reusable components identified:
   - [Component 1] from [location]
   - [Component 2] from [location]

   Proceeding with full implementation...
   ```
   Continue to Phase 1.

---

### Phase 1: Research

1. Fetch the provided URL or read the local file
2. Extract:
   - Method name and description
   - Mathematical formulation (equations)
   - Key assumptions
   - Estimator properties (consistency, efficiency)
3. Search for reference implementations in R, Python, or Stata
4. **Collect all sources for citation:**
   - Original paper(s) introducing the method
   - Reference implementations (package name, authors, URL)
   - Any tutorials or documentation used
5. Document findings before proceeding

---

### Phase 2: Planning

1. **Map to existing components** (from Phase 0 findings)
2. Design the API following column-based conventions
3. Identify required dependencies (already in Cargo.toml or new)
4. Plan test strategy with known results
5. **Prepare citation list** for inclusion in code and documentation
6. Write plan to the plan file for user approval

---

### Phase 3: Implementation

1. **Reuse existing components** where possible (don't duplicate code)
2. Implement new functionality in appropriate module under `crates/p2a-core/src/`
3. **Add reference block in module/function doc comments** (see Citation Requirements below)
4. **Add inline comments citing specific equations/algorithms**
5. Use `EconError` and `EconResult<T>` for error handling
6. Implement `LinearEstimator` trait if applicable

#### 3a. MCP Tool Implementation (MANDATORY)

**Every new econometric method MUST have an MCP tool exposure.**

1. Add tool handler in `crates/p2a-mcp/src/server.rs`:
   ```rust
   #[derive(Deserialize, JsonSchema)]
   pub struct MyMethodRequest {
       pub dataset: String,
       // ... method-specific parameters
   }

   #[tool(description = "Method description for LLM consumption")]
   async fn my_method(&self, #[tool(aggr)] request: MyMethodRequest) -> Result<String, McpError> {
       // Implementation calling p2a-core function
   }
   ```

2. Follow existing patterns - examine similar tools in `server.rs`
3. Return structured JSON output suitable for LLM interpretation
4. Update `docs/guides/MCP_TOOL_EXAMPLES.md` with usage example

---

### Phase 4: Testing & Validation Against R/Python (MANDATORY)

**Every implementation MUST be validated against the original R (or Python) implementation.**

#### 4a. Unit Tests

1. Write unit tests in the implementation file
2. Use realistic test data (with noise, not perfect relationships)
3. **Document test data source** if from published paper/package
4. Run `cargo test -p p2a-core` to verify

#### 4b. Validation Tests Against R/Python (MANDATORY)

**Purpose**: Ensure numerical correctness by comparing results with established implementations.

1. **Create R validation script** in `validation/scripts/[method]_validation.R`:
   ```r
   # [Method Name] Validation Script
   # Compares R package results with p2a-core output

   library(package_name)  # Reference implementation

   # Test Case 1: Synthetic data
   set.seed(42)
   n <- 1000
   # Generate test data...

   # Run R implementation
   result_r <- function_name(...)

   # Save expected results for Rust comparison
   write.csv(data.frame(
     coefficient = coef(result_r),
     std_error = sqrt(diag(vcov(result_r))),
     # ... other outputs
   ), "validation/expected/[method]_test1.csv")

   # Print results for documentation
   print(summary(result_r))
   ```

2. **Add Rust validation test** with `test_validate_` prefix:
   ```rust
   #[test]
   fn test_validate_[method]_against_r() {
       // Load same test data used in R script
       let dataset = create_test_dataset();

       // Run Rust implementation
       let result = run_[method](&dataset, ...).unwrap();

       // Compare against R expected values (from CSV or hardcoded)
       // Tolerance: typically 1e-6 for coefficients, 1e-4 for std errors
       assert!((result.coefficients()[0] - expected_coef).abs() < 1e-6,
           "Coefficient mismatch: Rust={}, R={}",
           result.coefficients()[0], expected_coef);
   }
   ```

3. **Document discrepancies**: If results differ beyond tolerance, document why:
   - Different default options (e.g., df adjustment)
   - Numerical precision differences
   - Algorithm variations

4. Run validation: `cargo test -p p2a-core -- test_validate_[method] --nocapture`

---

### Phase 5: Documentation

1. Add method description to `docs/guides/ECONOMETRICS_GUIDE.md`
   - Include **References** section with full citations
   - Link to original papers and reference implementations
2. Add usage example to `docs/guides/MCP_TOOL_EXAMPLES.md` if new MCP tool

---

### Phase 6: Prepare Validation Artifacts (MANDATORY)

**Purpose**: Prepare the artifacts needed for validation, then delegate to `/validate-method`.

#### 6a. Add Method to Registry

**Add the new method to the validation method registry** at `.claude/tooling/validation/method_registry.json`:

```json
"[method_id]": {
  "display_name": "[Method Display Name]",
  "r_name": "[R function name]",
  "r_packages": ["package1", "package2"],
  "category": "[category]",
  "rust_module": "crates/p2a-core/src/[path]/[file].rs",
  "rust_function": "[function_name]",
  "r_benchmark_script": "../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_[method].R",
  "criterion_benchmark": "crates/p2a-core/benches/[category]_benchmarks.rs",
  "criterion_group": "[BenchmarkGroupName]",
  "validation_doc": "validation/[category]/[method].md",
  "test_patterns": ["test_validate_[method]"]
}
```

#### 6b. Ensure Criterion Benchmark Exists

**Check if Criterion benchmark exists** for the method. If not, add it:

1. **Add Rust benchmark** to `crates/p2a-core/benches/[category]_benchmarks.rs`:
   ```rust
   fn [method]_benchmark(c: &mut Criterion) {
       let mut group = c.benchmark_group("[method]");

       for size in [100, 1_000, 10_000, 100_000].iter() {
           let dataset = generate_test_dataset(*size);

           group.bench_with_input(
               BenchmarkId::from_parameter(size),
               size,
               |b, _| {
                   b.iter(|| run_[method](&dataset, ...))
               },
           );
       }
       group.finish();
   }
   ```

2. Register the benchmark in `criterion_group!` macro.

#### 6c. Ensure R Benchmark Script Exists

**Check if R benchmark script exists.** If not, the `/validate-method` command will offer to generate one from the template at `.claude/tooling/validation/templates/r_benchmark.R.template`.

Alternatively, create it manually in `../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_[method].R`.

#### 6d. Create Basic Validation Document Structure

**Create** `validation/[category]/[method].md` with the basic structure:

```markdown
# Validation: [Method Name]

## Method Overview
[Brief description, key parameters]

## Reference Implementations
| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| [pkg]   | R        | [func]() | [version]      |

## Test Cases
### Test 1: Synthetic Data
**R Code**: [to be filled by /validate-method]
**Results Comparison**: [to be filled by /validate-method]
**Rust Test**: `test_validate_[method]`

## Numerical Precision Summary
[To be filled by /validate-method]

## Performance Comparison
[To be filled by /validate-method]

## References
[Citations from Phase 1]
```

---

### Phase 7: Run Validation & Benchmarking via /validate-method (MANDATORY)

**This phase delegates to the `/validate-method` command for unified validation and benchmarking.**

#### 7a. Invoke the Validation Command

**Run the `/validate-method` command** for the newly implemented method:

```
/validate-method [method_id]
```

This command will automatically:

1. **Discovery**: Look up the method in the registry, verify all artifacts exist
2. **Validation**: Run Rust tests (`cargo test -p p2a-core -- test_validate_[method]`)
3. **Benchmarks**: Run Criterion benchmarks and R benchmarks
4. **Comparison**: Calculate speedup factors and evaluate against requirements
5. **Report**: Generate validation report and update documentation

#### 7b. Review Validation Results

The `/validate-method` command will output:

```
=== Validation Results: [Method Name] ===

Test: [Test Case 1]
  Coefficient β₁: R=[value], Rust=[value], Diff=[diff] [PASS/FAIL]
  ...

Overall: X/Y checks passed

=== Performance Results: [Method Name] ===

Sample Size | R (us)    | Rust (us) | Speedup | Required | Status
------------|-----------|-----------|---------|----------|--------
n=1,000     | [time]    | [time]    | [X]x    | 1.5x     | [PASS/FAIL]
n=10,000    | [time]    | [time]    | [X]x    | 2.0x     | [PASS/FAIL]
n=100,000   | [time]    | [time]    | [X]x    | 3.0x     | [PASS/FAIL]
```

#### 7c. Handle Failures

**If validation fails:**
- Fix the implementation based on the error messages
- Re-run `/validate-method [method_id] --mode validate`

**If performance requirements are not met (Rust slower than R):**

1. **Identify bottlenecks** — Common issues:
   - Unnecessary allocations (use `Vec::with_capacity`, reuse buffers)
   - Redundant matrix operations (cache intermediate results)
   - Using `safe_inverse` when Cholesky solve is faster
   - Repeated column extractions from DataFrame

2. **Apply optimizations** — Consider:
   - Using `faer` directly for linear algebra
   - Pre-computing reusable matrices (X'X, X'y)
   - Using `rayon` for data-parallel operations

3. **Re-benchmark** — Run validation again:
   ```
   /validate-method [method_id] --mode benchmark
   ```

4. **Document optimization history** in the validation doc if multiple iterations needed.

#### 7d. Performance Requirements

The `/validate-method` command enforces these speedup requirements:

| Sample Size | Minimum Speedup | Status |
|-------------|-----------------|--------|
| n=1,000     | 1.5x faster     | Required |
| n=10,000    | 2.0x faster     | Required |
| n=100,000   | 3.0x faster     | Expected |

**Exit criteria:**
- No dataset size shows Rust SLOWER than R
- At least 2 of 4 sizes show >= 2x speedup
- n=10,000 shows >= 1.5x speedup

#### 7e. Update Indexes

After successful validation:

1. The `/validate-method` command saves a report to `validation/reports/[method]_[date].md`
2. Manually update `validation/README.md` index (mark as "Complete")
3. Update `validation/reference_implementations.md` if new R package used
4. Update implementation queue status to "completed"

#### 7f. Final Verification

Confirm the `/validate-method` command completed successfully:

```
✅ IMPLEMENTATION COMPLETE

Method: [Method Name]
- Validation: All tests passed
- Performance: Meets speedup requirements
- Report: validation/reports/[method]_[date].md
- Registry: Updated with last_validated timestamp
```

**IMPORTANT**: Implementation is NOT complete until `/validate-method` reports success

---

## Implementation Checklist

After completing all phases, verify ALL items are complete:

### MCP Tool (Phase 3a)
- [ ] MCP tool added to `crates/p2a-mcp/src/server.rs`
- [ ] Tool returns structured JSON output
- [ ] Usage example added to `docs/guides/MCP_TOOL_EXAMPLES.md`

### Validation Tests (Phase 4b)
- [ ] R validation script created in `validation/scripts/[method]_validation.R`
- [ ] Expected results saved to `validation/expected/`
- [ ] Rust `test_validate_*` tests implemented and passing
- [ ] Results match R within documented tolerances
- [ ] Any discrepancies documented with explanation

### Validation Artifacts (Phase 6)
- [ ] Method added to `.claude/tooling/validation/method_registry.json`
- [ ] Criterion benchmark added to `crates/p2a-core/benches/`
- [ ] R benchmark script exists (or will be generated by `/validate-method`)
- [ ] Basic validation document structure created

### /validate-method Execution (Phase 7) - MANDATORY
- [ ] `/validate-method [method_id]` executed successfully
- [ ] All validation tests passed
- [ ] All performance requirements met (see speedup table)
- [ ] Validation report saved to `validation/reports/`
- [ ] If optimizations needed: re-ran `/validate-method --mode benchmark`

### Index Updates (Phase 7e)
- [ ] `validation/README.md` index updated (marked "Complete")
- [ ] `validation/reference_implementations.md` updated if new package
- [ ] Implementation queue status updated to "completed"

---

## Citation Requirements

**CRITICAL**: All sources must be properly cited throughout the implementation:

1. **In Rust source code** — Add a doc comment block at the top of each new function/module:
   ```rust
   /// Generalized Least Squares (GLS) estimator.
   ///
   /// # References
   ///
   /// - Aitken, A. C. (1936). "On Least Squares and Linear Combination of Observations".
   ///   Proceedings of the Royal Society of Edinburgh, 55, 42-48.
   /// - Implementation adapted from R package `nlme` (Pinheiro & Bates, 2000).
   ///   Source: https://cran.r-project.org/package=nlme
   ```

2. **In ECONOMETRICS_GUIDE.md** — Include full bibliographic references:
   ```markdown
   ## References
   - Author, A. (Year). "Title". Journal, Volume(Issue), Pages. DOI/URL
   ```

3. **In code comments** — Reference specific equations or algorithms:
   ```rust
   // Variance estimator from Equation (3.15) in Greene (2018), p. 287
   let variance = ...;
   ```

## Citation Format Examples

| Source Type | Format |
|-------------|--------|
| Journal article | Author (Year). "Title". *Journal*, Vol(Issue), Pages. DOI |
| Book | Author (Year). *Title* (Edition). Publisher. |
| R package | Package: name (Author, Year). URL |
| Python library | Library: name (Author, Year). URL |
| Stata | StataCorp (Year). Stata Statistical Software: Release X. |

---

## Important Guidelines

### Mandatory Deliverables (NO EXCEPTIONS)
- **MCP Tool**: Every method MUST have an MCP tool in `server.rs`
- **R Validation**: Every method MUST be validated against R with matching results
- **Benchmarks**: Every method MUST have Criterion benchmarks
- **R Comparison**: Every method MUST include performance comparison vs R
- **Performance**: Rust MUST be faster than R — if not, optimize until it is
- **/validate-method**: Every implementation MUST end with running `/validate-method [method_id]`

### Implementation Standards
- **Check first**: ALWAYS run Phase 0 before any implementation work
- **Reuse code**: Use existing components; don't duplicate functionality
- **Column-based API**: Use explicit column names, not R-style formulas
- **Error handling**: Use `EconError` and `EconResult<T>`
- **Test data**: Add noise to test data to avoid zero residuals
- **LinearEstimator trait**: Implement for consistent output interface
- **Matrix operations**: Use functions from `linalg/matrix_ops.rs`
- **Citations**: ALWAYS cite original sources in code docs AND user-facing documentation

---

## Summary of Required Outputs

For EVERY new econometric method, the following artifacts MUST be delivered:

| Phase | Required Artifact | Location |
|-------|-------------------|----------|
| 3 | Core implementation | `crates/p2a-core/src/...` |
| 3a | MCP tool | `crates/p2a-mcp/src/server.rs` |
| 4b | R validation script | `validation/scripts/[method]_validation.R` |
| 4b | Rust validation tests | `test_validate_*` in implementation file |
| 6a | Method registry entry | `.claude/tooling/validation/method_registry.json` |
| 6b | Criterion benchmark | `crates/p2a-core/benches/` |
| 6c | R benchmark script | `../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_[method].R` |
| 6d | Validation document | `validation/[category]/[method].md` |
| 7 | **/validate-method execution** | Runs all validation and benchmarks automatically |
| 7 | **Validation report** | `validation/reports/[method]_[date].md` |

**Implementation is NOT complete until:**
1. **`/validate-method [method_id]` executes successfully**
2. **All validation tests pass** (numerical correctness verified against R)
3. **Performance requirements met** (Rust faster than R at required speedups)
4. **If initially slower**, optimizations applied and `/validate-method` re-run

---

**BEGIN by running Phase 0: Check for existing implementations.**

**END by running Phase 7: Execute `/validate-method [method_id]` to validate and benchmark.**
