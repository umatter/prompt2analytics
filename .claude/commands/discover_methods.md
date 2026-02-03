---
description: Discover unimplemented methods from a package index (R, Python, Julia) and prioritize for implementation
argument-hint: <index-url-or-path>
allowed-tools: Read, Bash, Glob, Grep, WebFetch, WebSearch, Task
---

# Discover Unimplemented Methods

You are analyzing a statistical/econometrics package index to identify methods NOT YET implemented in p2a-core, then prioritizing them for implementation.

## Source
The user has provided: $ARGUMENTS

This could be:
- An R package index (e.g., `https://stat.ethz.ch/R-manual/R-devel/library/stats/html/00Index.html`)
- A Python package docs index (e.g., statsmodels, scipy.stats)
- A Julia package index
- A local file listing methods

---

## Phase 1: Parse the Index

### 1a. Fetch and Extract Methods

1. **Fetch the index page/file**
2. **Extract all method/function names** along with:
   - Brief description (if available)
   - Category (if categorized in the source)
   - Link to detailed documentation

3. **Output initial count**:
   ```
   📊 INDEX ANALYSIS: [Package Name]

   Total functions/methods found: [N]
   Source: [URL or file path]
   ```

### 1b. Categorize Methods

Group methods into these categories based on descriptions and names:

| Category | Keywords/Patterns | Priority |
|----------|-------------------|----------|
| **Regression** | lm, glm, ols, regression, fit, coef | HIGH |
| **Panel Data** | panel, fixed.effects, random.effects, plm | HIGH |
| **Time Series** | arima, var, garch, acf, forecast, ts | HIGH |
| **Hypothesis Testing** | test, t.test, chi, anova, wilcox | HIGH |
| **Causal Inference** | iv, 2sls, did, rdd, matching | HIGH |
| **Distributions** | dnorm, pnorm, qnorm, rbinom, distribution | MEDIUM |
| **Descriptive Stats** | mean, var, sd, quantile, summary | MEDIUM |
| **Survival Analysis** | surv, cox, hazard, kaplan | MEDIUM |
| **Clustering/ML** | kmeans, hclust, pca, cluster | MEDIUM |
| **Utilities** | predict, residuals, fitted, update | LOW |
| **Data Manipulation** | reshape, merge, aggregate | LOW |
| **Plotting** | plot, hist, qqnorm | LOW (skip) |
| **Internal/Helper** | .Internal, .Call, print.* | SKIP |

---

## Phase 2: Cross-Reference with Existing Implementations

### 2a. Build Existing Method Inventory

Search the codebase to build a list of what's already implemented:

```bash
# Get all public functions from p2a-core
Grep: "pub fn " in crates/p2a-core/src/ --type rust

# Get all MCP tools
Grep: "#\[tool\(" in crates/p2a-mcp/src/server.rs

# Read module exports
Read: crates/p2a-core/src/lib.rs
Read: crates/p2a-core/src/regression/mod.rs
Read: crates/p2a-core/src/econometrics/mod.rs
Read: crates/p2a-core/src/ml/mod.rs
Read: crates/p2a-core/src/stats/mod.rs
```

### 2b. Create Method Mapping

Build a mapping of R/Python function names to p2a-core equivalents:

| R Function | Likely p2a-core Equivalent | Check Pattern |
|------------|---------------------------|---------------|
| `lm` | `run_ols` | `ols`, `linear` |
| `glm` | `run_glm`, `run_logit`, `run_probit` | `glm`, `logit`, `probit` |
| `t.test` | `t_test` | `t_test`, `ttest` |
| `var` | (time series) `run_var` | `var`, `vector_auto` |
| `arima` | `run_arima` | `arima` |
| `kmeans` | `kmeans` | `kmeans`, `k_means` |
| `pca` | `run_pca` | `pca`, `principal` |
| etc. | | |

### 2c. Match and Classify

For each method from the index:

1. **Direct name match**: Search for exact function name
2. **Alias match**: Search for common aliases (e.g., "lm" → "ols")
3. **Description match**: Search for key terms in description

Classify each method as:
- ✅ **IMPLEMENTED**: Exact or close match found
- 🔶 **PARTIAL**: Related functionality exists but not complete
- ❌ **NOT IMPLEMENTED**: No match found
- ⏭️ **SKIP**: Low priority or out of scope (plotting, internal helpers)

---

## Phase 3: Prioritization

### 3a. Scoring Criteria

Score each unimplemented method (0-100):

| Factor | Weight | Scoring |
|--------|--------|---------|
| **Category Priority** | 40% | HIGH=40, MEDIUM=25, LOW=10 |
| **Econometrics Relevance** | 30% | Core econometrics=30, Stats=20, ML=15, Other=5 |
| **Implementation Complexity** | 20% | Simple=20, Medium=12, Complex=5 |
| **Dependencies Available** | 10% | All deps in p2a-core=10, Some=5, None=0 |

### 3b. Complexity Estimation

Estimate complexity based on:
- **Simple** (score 20): Single formula, no iteration, pure calculation
  - Examples: t-test, correlation, basic descriptive stats
- **Medium** (score 12): Iterative algorithm or matrix operations
  - Examples: OLS variants, basic MLE, clustering
- **Complex** (score 5): Advanced optimization, multiple stages, specialized algorithms
  - Examples: GARCH, state-space models, Bayesian methods

### 3c. Dependency Check

Check if required components exist:
- Matrix operations (`linalg/matrix_ops.rs`)
- Distribution functions (`statrs` crate)
- Optimization routines
- Data structures

---

## Phase 4: Output Report

### 4a. Summary Statistics

```
═══════════════════════════════════════════════════════════════
📊 METHOD DISCOVERY REPORT
   Source: [Package Name] ([URL])
   Date: [Date]
═══════════════════════════════════════════════════════════════

SUMMARY
───────────────────────────────────────────────────────────────
Total methods scanned:     [N]
Already implemented:       [N] (XX%)
Partially implemented:     [N] (XX%)
Not implemented:          [N] (XX%)
Skipped (out of scope):   [N] (XX%)

CATEGORY BREAKDOWN
───────────────────────────────────────────────────────────────
Category          | Total | Implemented | Gaps | Priority
──────────────────|───────|─────────────|──────|──────────
Regression        |   XX  |     XX      |  XX  | HIGH
Panel Data        |   XX  |     XX      |  XX  | HIGH
Time Series       |   XX  |     XX      |  XX  | HIGH
Hypothesis Tests  |   XX  |     XX      |  XX  | HIGH
Distributions     |   XX  |     XX      |  XX  | MEDIUM
...
```

### 4b. Prioritized Implementation List

```
TOP 20 METHODS TO IMPLEMENT (by priority score)
═══════════════════════════════════════════════════════════════

Rank │ Score │ Method          │ Category      │ Complexity │ Doc Link
─────┼───────┼─────────────────┼───────────────┼────────────┼──────────
  1  │  92   │ gls             │ Regression    │ Medium     │ [link]
  2  │  89   │ nls             │ Regression    │ Medium     │ [link]
  3  │  87   │ arima           │ Time Series   │ Complex    │ [link]
  4  │  85   │ garch           │ Time Series   │ Complex    │ [link]
  5  │  82   │ anova           │ Hypothesis    │ Simple     │ [link]
...

QUICK WINS (High priority + Simple complexity)
───────────────────────────────────────────────────────────────
These can be implemented quickly with high impact:

1. [method] - [brief description] - [doc link]
2. [method] - [brief description] - [doc link]
...
```

### 4c. Detailed Gap Analysis

For each HIGH priority unimplemented method:

```
───────────────────────────────────────────────────────────────
METHOD: [Name]
───────────────────────────────────────────────────────────────
Description: [From source docs]
Category: [Category]
Priority Score: [XX/100]
Complexity: [Simple/Medium/Complex]
Documentation: [URL]

Existing Related Components:
  • [Component 1] in [location] - can be reused for [purpose]
  • [Component 2] in [location] - provides [functionality]

Implementation Notes:
  • [Key algorithm or approach needed]
  • [Dependencies required]
  • [Estimated effort: X hours]

Command to implement:
  /implement_metrics [specific-doc-url]
───────────────────────────────────────────────────────────────
```

### 4d. Save Report

Save the full report to: `docs/discovery/[package-name]-[date].md`

### 4e. Generate Implementation Queue (CRITICAL)

**Create the machine-readable queue file** at `docs/discovery/implementation_queue.json`:

```json
{
  "source": "[Package Name]",
  "source_url": "[URL]",
  "generated_at": "[ISO timestamp]",
  "total_methods": [N],
  "completed": 0,
  "in_progress": 0,
  "blocked": 0,
  "pending": [N],
  "methods": [
    {
      "rank": 1,
      "method": "[method_name]",
      "category": "[category]",
      "priority_score": [score],
      "complexity": "[Simple/Medium/Complex]",
      "doc_url": "[full_url_to_method_docs]",
      "description": "[brief description]",
      "status": "pending"
    },
    // ... top 20 methods sorted by priority_score desc
  ]
}
```

**Important:** This file enables the `/implement_next` command to automatically process methods in priority order.

---

## Phase 5: Interactive Options

After presenting the report, offer:

```
NEXT STEPS
═══════════════════════════════════════════════════════════════

What would you like to do?

1. Implement top priority method: [method name]
   → /implement_metrics [url]

2. Implement all "Quick Wins" (N methods)
   → Will run /implement_metrics for each

3. Export full list to CSV for review
   → Saves to docs/discovery/[package]-methods.csv

4. Focus on specific category
   → Re-run analysis for just [category]

5. Compare with another package
   → Run discovery on additional source
```

---

## Example Workflow

For input: `https://stat.ethz.ch/R-manual/R-devel/library/stats/html/00Index.html`

1. **Fetch** → Parse 520+ R stats functions
2. **Categorize** → Group into regression, tests, distributions, etc.
3. **Cross-reference** → Find existing: `run_ols` (lm), `t_test` (t.test), etc.
4. **Score** → Prioritize unimplemented by relevance and complexity
5. **Report** → Show top 20 gaps with implementation commands
6. **Action** → User picks a method → run `/implement_metrics`

---

## Important Notes

- This command does NOT implement anything - it only discovers and prioritizes
- For actual implementation, use `/implement_metrics <specific-method-url>`
- The cross-reference is based on name/description matching - manual review recommended
- Focus on HIGH priority categories first (regression, panel, time series, causal)
- Skip plotting functions and internal helpers entirely
