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
7. Add MCP tool in `crates/p2a-mcp/src/server.rs` if user-facing

---

### Phase 4: Testing

1. Write unit tests in the implementation file
2. Use realistic test data (with noise, not perfect relationships)
3. Compare results against reference implementation if available
4. **Document test data source** if from published paper/package
5. Run `cargo test -p p2a-core` to verify

---

### Phase 5: Documentation

1. Add method description to `docs/guides/ECONOMETRICS_GUIDE.md`
   - Include **References** section with full citations
   - Link to original papers and reference implementations
2. Add usage example to `docs/guides/MCP_TOOL_EXAMPLES.md` if new MCP tool
3. Update `DEVELOPMENT_REPORT.md` with completed work

---

### Phase 6: Validation & Benchmarking

**Purpose**: Document validation against reference implementations and establish performance baselines.

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
## References
```

#### 6b. Add Validation Tests

1. Add Rust test functions with `test_validate_` prefix
2. Compare against R/Python results with appropriate tolerances
3. Run: `cargo test -p p2a-core -- test_validate_[method] --nocapture`

#### 6c. Add Criterion Benchmark (Optional for Complex Methods)

1. Add benchmark to `crates/p2a-core/benches/[category]_benchmarks.rs`
2. Test scaling behavior at n = 100, 1000, 10000
3. Run: `cargo bench -p p2a-core -- [method_name]`
4. Save results to `performance/results/[date]/`

#### 6d. Update Indexes

1. Add entry to `validation/README.md` method index (mark as "Complete")
2. Add entry to `validation/reference_implementations.md` if new package used
3. If benchmark added, update `performance/reports/[category]_performance.md`

---

## Validation Checklist

After completing Phase 6, verify:

- [ ] `validation/[category]/[method].md` created with complete template
- [ ] At least 2 test cases documented (synthetic + real if available)
- [ ] R/Python reproduction code included
- [ ] Results comparison table with explicit tolerances
- [ ] Rust `test_validate_*` tests pass
- [ ] `validation/README.md` index updated
- [ ] (Optional) Criterion benchmark added for complex methods

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

- **Check first**: ALWAYS run Phase 0 before any implementation work
- **Reuse code**: Use existing components; don't duplicate functionality
- **Column-based API**: Use explicit column names, not R-style formulas
- **Error handling**: Use `EconError` and `EconResult<T>`
- **Test data**: Add noise to test data to avoid zero residuals
- **LinearEstimator trait**: Implement for consistent output interface
- **Matrix operations**: Use functions from `linalg/matrix_ops.rs`
- **Citations**: ALWAYS cite original sources in code docs AND user-facing documentation

---

**BEGIN by running Phase 0: Check for existing implementations.**
