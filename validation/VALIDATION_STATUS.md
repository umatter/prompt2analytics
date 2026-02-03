# Validation Status Report

**Last Updated:** 2026-01-27
**Commit:** 2955f76
**Branch:** full-rust-migration

## Summary

| Metric | Count | Status |
|--------|-------|--------|
| **Rust Validation Tests** | 117 | All Pass |
| **R Validation Scripts** | 23 | All Pass |
| **Validation Documentation** | 95 | Mixed |

## How to Run Validation

```bash
# Full validation (Rust + R)
./validation/run_validation.sh

# Rust tests only (faster)
./validation/run_validation.sh --rust-only

# R scripts only
./validation/run_validation.sh --r-only

# Run with verbose output
./validation/run_validation.sh --verbose

# Filter by category
./validation/run_validation.sh --category stats
```

## Coverage by Category

### Statistics (stats)
| Tests | Docs | Status |
|-------|------|--------|
| 86 | 27 | **Good** |

**Validated Methods:**
- T-tests (one-sample, two-sample, paired)
- ANOVA (one-way, two-way)
- Chi-squared tests
- Fisher's exact test
- Wilcoxon tests
- Shapiro-Wilk normality
- Kolmogorov-Smirnov
- Bartlett's test
- Kruskal-Wallis
- Friedman test
- Tukey HSD
- ACF/PACF
- Box-Pierce/Ljung-Box
- Factor analysis
- Canonical correlation
- And more...

### Regression
| Tests | Docs | Status |
|-------|------|--------|
| 6 | 12 | **Partial** |

**Validated Methods:**
- Sensemakr (sensitivity analysis)
- E-Value (confounding)
- LOESS
- NLS (nonlinear least squares)
- Smooth splines

**Needs Validation:**
- OLS (basic tests exist, need R comparison)
- Robust SEs (HC0-HC3)
- Clustered SEs
- GLS
- Quantile regression

### Econometrics
| Tests | Docs | Status |
|-------|------|--------|
| 19 | 42 | **Good** |

**Validated Methods:**
- HDFE (high-dimensional fixed effects)
- FEGLM (logit, probit)
- Survival analysis (Kaplan-Meier, Cox PH, AFT)
- Balke-Pearl bounds

**Needs Validation:**
- Panel FE/RE
- IV/2SLS
- DiD
- Treatment effects (IPW, AIPW)
- RD (regression discontinuity)

### Forecasting
| Tests | Docs | Status |
|-------|------|--------|
| 6 | 6 | **Partial** |

**Validated Methods:**
- AR models
- Holt-Winters
- STL decomposition
- Classical decomposition
- Lag/embed utilities

**Needs Validation:**
- ARIMA
- GARCH
- Changepoint detection
- Kalman filter

### Machine Learning (ml)
| Tests | Docs | Status |
|-------|------|--------|
| 0 (validation prefix) | 8 | **Needs Work** |

Note: ML module has 50+ unit tests but no `test_validate_*` named tests for R comparison.

**Needs Validation:**
- K-means
- DBSCAN
- Hierarchical clustering
- PCA
- t-SNE
- Random Forest
- SVM
- Causal Forest

## Priority Actions for JSS Paper

### High Priority (Core Methods)
1. **OLS with Robust SEs** - Foundational method, needs R comparison
2. **Panel Fixed/Random Effects** - Core econometrics
3. **IV/2SLS** - Important causal method
4. **Difference-in-Differences** - Widely used
5. **Logit/Probit** - Binary outcomes

### Medium Priority
6. Regression Discontinuity (Sharp/Fuzzy)
7. Treatment effects (IPW, AIPW)
8. K-means clustering
9. PCA
10. ARIMA forecasting

### Low Priority (Specialized)
11. Spatial models
12. Survival analysis (already validated)
13. Advanced clustering (HDBSCAN, etc.)
14. Mixed logit

## File Locations

- **Runner Script:** `validation/run_validation.sh`
- **R Scripts:** `validation/scripts/`
- **Validation Docs:** `validation/[category]/`
- **Method Registry:** `.claude/tooling/validation/method_registry.json`
- **Reports:** `validation/reports/`

## Adding New Validations

1. Create Rust test with `test_validate_` prefix
2. Create R script in `validation/scripts/`
3. Create documentation in `validation/[category]/method.md`
4. Run `./validation/run_validation.sh` to verify

See `validation/README.md` for detailed template.
