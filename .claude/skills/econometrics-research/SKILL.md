---
name: econometrics-research
description: Research econometric methods from academic sources. Use when fetching papers, extracting mathematical formulations, or finding reference implementations.
---

# Econometrics Research

## Finding Reference Implementations

Search these sources for existing implementations:

### R Packages
- **CRAN**: https://cran.r-project.org/web/packages/
- **Key packages**:
  - `lmtest` - Linear model tests
  - `sandwich` - Robust covariance estimators
  - `plm` - Panel data models
  - `AER` - Applied econometrics
  - `ivreg` - Instrumental variables
  - `fixest` - Fast fixed effects
  - `did` - Difference-in-differences
  - `lfe` - Linear fixed effects
  - `vars` - Vector autoregressions

### Python
- `statsmodels` - Comprehensive statistics
- `linearmodels` - Panel data, IV
- `econml` - Causal inference
- `arch` - GARCH, volatility models

### Stata
- Official documentation: https://www.stata.com/manuals/
- User-written commands (SSC archive)

## Extracting Mathematical Formulation

When reading a paper or documentation, extract:

1. **Estimator formula**
   - Point estimate (e.g., β̂ = (X'X)⁻¹X'y)
   - Closed-form vs iterative solution

2. **Variance estimator**
   - Standard errors formula
   - Robust/clustered variants
   - Degrees of freedom adjustment

3. **Key assumptions**
   - Exogeneity (E[ε|X] = 0)
   - Homoskedasticity vs heteroskedasticity
   - No autocorrelation
   - Rank conditions

4. **Asymptotic properties**
   - Consistency (plim β̂ → β)
   - Efficiency (minimum variance)
   - Distribution (√n(β̂ - β) → N(0, V))

## Using DeepWiki for GitHub Repos

Use DeepWiki MCP tools to explore R/Python package repositories:

```
mcp__deepwiki__ask_question
- repoName: "cran/fixest" or "statsmodels/statsmodels"
- question: "How is the fixed effects estimator implemented?"

mcp__deepwiki__read_wiki_structure
- repoName: "rstudio/plm"
```

## Key Academic Sources

- **Journal of Statistical Software**: Implementation papers
- **Econometrica**: Theoretical foundations
- **Journal of Econometrics**: Applied methods
- **arXiv (stat.ME, econ.EM)**: Preprints

## Research Checklist

- [ ] Identify the method's full name and common abbreviations
- [ ] Find the original paper/reference
- [ ] Extract mathematical formulation
- [ ] List all required assumptions
- [ ] Find at least one reference implementation (R/Python/Stata)
- [ ] Note any special cases or variants
- [ ] Document test cases with known results
- [ ] **Collect all sources for citation** (see below)

## Citation Collection

**CRITICAL**: During research, collect complete citation information for ALL sources:

### For Academic Papers
Record:
- Author(s) full names
- Year of publication
- Article title
- Journal name
- Volume, issue, pages
- DOI or URL

Example:
```
Aitken, A. C. (1936). "On Least Squares and Linear Combination of Observations".
Proceedings of the Royal Society of Edinburgh, 55, 42-48.
```

### For R Packages
Record:
- Package name
- Author(s)
- Year
- CRAN URL
- Key functions used

Example:
```
Package: fixest (Bergé, 2018). Fast Fixed-Effects Estimations.
https://cran.r-project.org/package=fixest
Functions referenced: feols(), fixef()
```

### For Python Libraries
Record:
- Library name
- Author(s) or organization
- Version used for validation
- Documentation URL
- Key classes/functions

Example:
```
Library: linearmodels (Kevin Sheppard, 2017).
https://bashtage.github.io/linearmodels/
Classes referenced: PanelOLS, RandomEffects
```

### For Stata
Record:
- Command name
- StataCorp reference
- Manual section

Example:
```
Stata command: xtreg
StataCorp. 2023. Stata Statistical Software: Release 18.
Reference: [XT] xtreg - Fixed-, between-, and random-effects and population-averaged linear models
```

### For Textbooks
Record:
- Author(s)
- Year
- Title
- Edition
- Publisher
- Specific chapters/equations referenced

Example:
```
Greene, W. H. (2018). Econometric Analysis (8th ed.). Pearson.
Chapter 11: Models for Panel Data, Equations 11-1 through 11-15.
```
