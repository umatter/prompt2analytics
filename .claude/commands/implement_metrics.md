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
   Read: DEVELOPMENT_REPORT.md
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
3. Update `DEVELOPMENT_REPORT.md` with completed work

---

### Phase 6: Benchmarking & Performance Comparison (MANDATORY)

**Purpose**: Establish performance baselines and compare against R implementations.

#### 6a. Create Validation Document

1. Create `validation/[category]/[method].md` following the template structure
2. Include:
   - **Reference implementations table**: R/Python/Julia packages used for comparison
   - **At least 2 test cases**:
     - Synthetic data with known DGP (verifies coefficient recovery)
     - Real dataset if available (verifies practical accuracy)
   - **R/Python code** for reproduction
   - **Results comparison table** with tolerances
   - **Link to Rust test functions**
   - **Performance comparison section** (see 6c below)

**Template** (see existing files like `validation/econometrics/hdfe.md`):
```markdown
# Validation: [Method Name]

## Method Overview
[Brief description, key parameters]

## Reference Implementations
| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|

## Test Cases
### Test 1: [Dataset] - [Scenario]
**R Code**: [reproducible code]
**Results Comparison**: [table with tolerances]
**Rust Test**: `crates/p2a-core/src/.../tests::test_validate_...`

## Numerical Precision Summary
## Known Differences

## Performance Comparison
| Dataset Size | Rust (p2a) | R (package) | Speedup |
|--------------|------------|-------------|---------|
| n=1,000      |            |             |         |
| n=10,000     |            |             |         |
| n=100,000    |            |             |         |

## References
```

#### 6b. Add Criterion Benchmark (MANDATORY)

**Every new method MUST have performance benchmarks.**

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

   criterion_group!(benches, [method]_benchmark);
   ```

2. Run: `cargo bench -p p2a-core -- [method_name]`
3. Save baseline results to `performance/results/`

#### 6c. R Performance Comparison (MANDATORY)

**Every method MUST include a performance comparison against R.**

1. **Create R benchmark script** in `performance/benchmarks/[method]_r_benchmark.R`:
   ```r
   # [Method Name] R Benchmark
   library(microbenchmark)
   library(package_name)

   # Benchmark at different dataset sizes
   sizes <- c(100, 1000, 10000, 100000)

   results <- data.frame(
     size = integer(),
     time_ms = numeric()
   )

   for (n in sizes) {
     set.seed(42)
     # Generate data of size n...

     timing <- microbenchmark(
       function_name(...),
       times = 10
     )

     results <- rbind(results, data.frame(
       size = n,
       time_ms = median(timing$time) / 1e6  # Convert to ms
     ))
   }

   print(results)
   write.csv(results, "performance/results/[method]_r_times.csv")
   ```

2. **Run R benchmark**: `Rscript performance/benchmarks/[method]_r_benchmark.R`

3. **Document comparison** in validation file:
   - Include table with Rust vs R execution times
   - Calculate speedup factor (R_time / Rust_time)
   - Note any memory usage differences if significant

4. **Expected outcome**: Rust implementation should typically be 5-50x faster than R for compute-intensive methods

#### 6d. Update Indexes

1. Add entry to `validation/README.md` method index (mark as "Complete")
2. Add entry to `validation/reference_implementations.md` if new package used
3. Update `performance/reports/[category]_performance.md` with benchmark results

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

### Validation Document (Phase 6a)
- [ ] `validation/[category]/[method].md` created with complete template
- [ ] At least 2 test cases documented (synthetic + real if available)
- [ ] R/Python reproduction code included
- [ ] Results comparison table with explicit tolerances
- [ ] Performance comparison table included

### Benchmarks (Phase 6b-c)
- [ ] Criterion benchmark added to `crates/p2a-core/benches/`
- [ ] R benchmark script created in `performance/benchmarks/`
- [ ] R benchmark executed and times recorded
- [ ] Speedup factor calculated and documented
- [ ] Results saved to `performance/results/`

### Index Updates (Phase 6d)
- [ ] `validation/README.md` index updated (marked "Complete")
- [ ] `validation/reference_implementations.md` updated if new package
- [ ] `performance/reports/[category]_performance.md` updated with benchmark results

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
| 6a | Validation document | `validation/[category]/[method].md` |
| 6b | Criterion benchmark | `crates/p2a-core/benches/` |
| 6c | R benchmark script | `performance/benchmarks/[method]_r_benchmark.R` |

**Implementation is NOT complete until all items in the checklist are verified.**

---

**BEGIN by running Phase 0: Check for existing implementations.**
