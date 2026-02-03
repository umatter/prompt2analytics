# Method Name Mappings

This file maps common statistical function names from R, Python, and Julia to their p2a-core equivalents.
Used by `/discover_methods` for cross-referencing.

## Regression Methods

| R Function | Python (statsmodels) | Julia | p2a-core | Status |
|------------|---------------------|-------|----------|--------|
| `lm` | `OLS` | `lm` | `run_ols` | ✅ Implemented |
| `glm` | `GLM` | `glm` | `run_logit`, `run_probit` | 🔶 Partial |
| `gls` | `GLS` | `gls` | - | ❌ Not implemented |
| `nls` | - | `nls` | - | ❌ Not implemented |
| `lm.fit` | - | - | (internal) | ⏭️ Skip |
| `rlm` (MASS) | `RLM` | - | - | ❌ Not implemented |
| `lqs` (MASS) | - | - | - | ❌ Not implemented |

## Panel Data Methods

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `plm` (plm) | `PanelOLS` | `FixedEffectModels` | `run_fixed_effects` | ✅ Implemented |
| `pggls` (plm) | - | - | - | ❌ Not implemented |
| `pgmm` (plm) | - | - | - | ❌ Not implemented |
| `phtest` (plm) | - | - | `hausman_test` | ✅ Implemented |

## Time Series Methods

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `arima` | `ARIMA` | - | - | ❌ Not implemented |
| `ar` | `AR` | - | - | ❌ Not implemented |
| `VAR` (vars) | `VAR` | - | `run_var` | ✅ Implemented |
| `acf` | `acf` | - | - | ❌ Not implemented |
| `pacf` | `pacf` | - | - | ❌ Not implemented |
| `stl` | `STL` | - | - | ❌ Not implemented |
| `garch` (rugarch) | `arch_model` | - | - | ❌ Not implemented |

## Hypothesis Tests

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `t.test` | `ttest_ind` | `OneSampleTTest` | `t_test` | ✅ Implemented |
| `chisq.test` | `chi2_contingency` | - | - | ❌ Not implemented |
| `wilcox.test` | `wilcoxon` | - | - | ❌ Not implemented |
| `ks.test` | `kstest` | - | - | ❌ Not implemented |
| `shapiro.test` | `shapiro` | - | - | ❌ Not implemented |
| `var.test` | `levene` | - | - | ❌ Not implemented |
| `anova` | `anova_lm` | - | - | ❌ Not implemented |
| `aov` | - | - | - | ❌ Not implemented |

## Causal Inference / IV

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `ivreg` (AER) | `IV2SLS` | - | `run_2sls` | ✅ Implemented |
| `felm` (lfe) | - | `FixedEffectModels` | `run_hdfe` | ✅ Implemented |
| `did` (did) | - | - | `run_did` | ✅ Implemented |
| `rdrobust` | `rdrobust` | - | - | ❌ Not implemented |
| `matchit` | `CausalModel` | - | - | ❌ Not implemented |
| `synth` | - | - | - | ❌ Not implemented |

## Distributions

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `dnorm/pnorm/qnorm/rnorm` | `scipy.stats.norm` | `Normal` | via `statrs` | ✅ Available |
| `dt/pt/qt/rt` | `scipy.stats.t` | `TDist` | via `statrs` | ✅ Available |
| `dchisq/pchisq/qchisq` | `scipy.stats.chi2` | `Chisq` | via `statrs` | ✅ Available |
| `df/pf/qf` | `scipy.stats.f` | `FDist` | via `statrs` | ✅ Available |
| `dbinom/pbinom/qbinom` | `scipy.stats.binom` | `Binomial` | via `statrs` | ✅ Available |
| `dpois/ppois/qpois` | `scipy.stats.poisson` | `Poisson` | via `statrs` | ✅ Available |

## Descriptive Statistics

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `mean` | `np.mean` | `mean` | polars `.mean()` | ✅ Available |
| `var` | `np.var` | `var` | polars `.var()` | ✅ Available |
| `sd` | `np.std` | `std` | polars `.std()` | ✅ Available |
| `cor` | `np.corrcoef` | `cor` | `correlation_matrix` | ✅ Implemented |
| `cov` | `np.cov` | `cov` | via polars | ✅ Available |
| `quantile` | `np.quantile` | `quantile` | polars `.quantile()` | ✅ Available |
| `summary` | `describe` | `describe` | `summary_stats` | ✅ Implemented |

## Machine Learning

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `kmeans` | `KMeans` | `kmeans` | `kmeans` | ✅ Implemented |
| `hclust` | `AgglomerativeClustering` | - | `hierarchical_clustering` | ✅ Implemented |
| `prcomp/princomp` | `PCA` | `fit(PCA, ...)` | `run_pca` | ✅ Implemented |
| `lda` (MASS) | `LinearDiscriminantAnalysis` | - | - | ❌ Not implemented |
| `qda` (MASS) | `QuadraticDiscriminantAnalysis` | - | - | ❌ Not implemented |
| `randomForest` | `RandomForestClassifier` | - | `random_forest` | ✅ Implemented |
| `glmnet` | - | - | - | ❌ Not implemented |

## Survival Analysis

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `Surv` (survival) | - | - | - | ❌ Not implemented |
| `survfit` | `KaplanMeierFitter` | - | - | ❌ Not implemented |
| `coxph` | `CoxPHFitter` | - | - | ❌ Not implemented |
| `survreg` | `WeibullAFTFitter` | - | - | ❌ Not implemented |

## Diagnostics

| R Function | Python | Julia | p2a-core | Status |
|------------|--------|-------|----------|--------|
| `bptest` (lmtest) | `het_breuschpagan` | - | `breusch_pagan_test` | ✅ Implemented |
| `dwtest` (lmtest) | `durbin_watson` | - | `durbin_watson_test` | ✅ Implemented |
| `jarque.bera.test` | `jarque_bera` | - | `jarque_bera_test` | ✅ Implemented |
| `vif` (car) | `variance_inflation_factor` | - | `vif` | ✅ Implemented |
| `resettest` (lmtest) | - | - | - | ❌ Not implemented |
| `bgtest` (lmtest) | - | - | - | ❌ Not implemented |

---

## Search Patterns for Cross-Reference

When searching for existing implementations, use these patterns:

```
# Regression
ols|linear_regression|lm|ordinary.least

# Panel
fixed.effect|random.effect|panel|within|between|hdfe

# Time Series
arima|var|garch|acf|pacf|forecast|stationary

# Tests
test|hypothesis|p.value|statistic|critical

# IV/Causal
instrumental|2sls|iv|did|difference.in.difference|treatment

# Distributions
distribution|pdf|cdf|quantile|random|sample

# ML
cluster|kmeans|pca|principal|classify|predict
```

---

## Priority Categories for Implementation

### HIGH Priority (Core Econometrics)
- GLS, FGLS (Feasible Generalized Least Squares)
- ARIMA, GARCH (Time Series)
- GMM (Generalized Method of Moments)
- Quantile Regression
- ANOVA / MANOVA

### MEDIUM Priority (Extended Statistics)
- Survival Analysis (Cox, Kaplan-Meier)
- Non-parametric tests
- Robust regression (M-estimation)
- Mixed effects models

### LOW Priority (Utilities)
- Additional distribution functions
- Plotting helpers
- Data manipulation utilities

---

## Updating This File

When implementing a new method:
1. Update the status column (❌ → ✅)
2. Add the p2a-core function name
3. Add any aliases used

When discovering new source packages:
1. Add new rows for unmapped functions
2. Note the source package in parentheses
