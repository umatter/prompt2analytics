# Reference Implementations Catalog

This document catalogs the reference implementations used to validate prompt2analytics methods.

## R Packages

### Core Statistics
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| stats | 4.3.x | `lm()`, `glm()`, `kmeans()`, `hclust()`, `prcomp()` | Base R |

### Econometrics
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| plm | 2.6-x | `plm()`, `pdata.frame()`, `phtest()` | [plm](https://cran.r-project.org/package=plm) |
| lfe | 2.8-x | `felm()` | [lfe](https://cran.r-project.org/package=lfe) |
| alpaca | 0.3-x | `feglm()` | [alpaca](https://cran.r-project.org/package=alpaca) |
| fixest | 0.11-x | `feglm()`, `fepois()` | [fixest](https://cran.r-project.org/package=fixest) |
| AER | 1.2-x | `ivreg()` | [AER](https://cran.r-project.org/package=AER) |
| sandwich | 3.0-x | `vcovHC()`, `vcovCL()` | [sandwich](https://cran.r-project.org/package=sandwich) |
| lmtest | 0.9-x | `bptest()`, `dwtest()` | [lmtest](https://cran.r-project.org/package=lmtest) |

### Time Series
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| vars | 1.5-x | `VAR()`, `VECM()`, `irf()` | [vars](https://cran.r-project.org/package=vars) |
| forecast | 8.21-x | `auto.arima()`, `Arima()`, `mstl()` | [forecast](https://cran.r-project.org/package=forecast) |
| urca | 1.3-x | `ca.jo()`, `ur.df()` | [urca](https://cran.r-project.org/package=urca) |
| changepoint | 2.2-x | `cpt.mean()`, `cpt.var()` | [changepoint](https://cran.r-project.org/package=changepoint) |

### Machine Learning
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| dbscan | 1.1-x | `dbscan()` | [dbscan](https://cran.r-project.org/package=dbscan) |
| Rtsne | 0.16-x | `Rtsne()` | [Rtsne](https://cran.r-project.org/package=Rtsne) |
| randomForest | 4.7-x | `randomForest()` | [randomForest](https://cran.r-project.org/package=randomForest) |
| e1071 | 1.7-x | `svm()` | [e1071](https://cran.r-project.org/package=e1071) |

## Python Packages

### Statistics & Econometrics
| Package | Version | Functions Used | PyPI |
|---------|---------|---------------|------|
| statsmodels | 0.14.x | `OLS()`, `Logit()`, `Probit()`, `VAR()` | [statsmodels](https://pypi.org/project/statsmodels/) |
| linearmodels | 5.3-x | `PanelOLS()`, `IV2SLS()` | [linearmodels](https://pypi.org/project/linearmodels/) |

### Machine Learning
| Package | Version | Functions Used | PyPI |
|---------|---------|---------------|------|
| scikit-learn | 1.3-x | `KMeans()`, `DBSCAN()`, `PCA()`, `TSNE()`, `RandomForestClassifier()`, `SVC()` | [sklearn](https://pypi.org/project/scikit-learn/) |

### Time Series
| Package | Version | Functions Used | PyPI |
|---------|---------|---------------|------|
| arch | 6.x | `arch_model()` | [arch](https://pypi.org/project/arch/) |
| ruptures | 1.1-x | `Binseg()`, `Pelt()` | [ruptures](https://pypi.org/project/ruptures/) |

## Julia Packages

| Package | Functions Used | Registry |
|---------|---------------|----------|
| GLM.jl | `lm()`, `glm()` | General |
| FixedEffectModels.jl | `reg()` | General |

## Version Compatibility

Our validation tests were run against the following environment:

**R Environment**:
```r
R version 4.3.x
Platform: x86_64-pc-linux-gnu
```

**Python Environment**:
```
Python 3.11.x
numpy 1.26.x
scipy 1.11.x
```

## Installation Commands

### R
```r
install.packages(c(
  "plm", "lfe", "alpaca", "fixest", "AER", "sandwich", "lmtest",
  "vars", "forecast", "urca", "changepoint",
  "dbscan", "Rtsne", "randomForest", "e1071"
))
```

### Python
```bash
pip install statsmodels linearmodels scikit-learn arch ruptures
```

## Method-to-Package Mapping

| p2a Method | Primary Reference | Secondary Reference |
|------------|------------------|---------------------|
| `run_ols` | R `lm()` | Python `statsmodels.OLS` |
| `run_ols` (HC0-HC3) | R `sandwich::vcovHC` | Python `statsmodels` |
| `run_ols_clustered` | R `sandwich::vcovCL` | Python `linearmodels` |
| `run_fixed_effects` | R `plm` (within) | Python `linearmodels.PanelOLS` |
| `run_random_effects` | R `plm` (random) | - |
| `run_hausman_test` | R `plm::phtest` | - |
| `run_iv2sls` | R `AER::ivreg` | Python `linearmodels.IV2SLS` |
| `run_did` | Manual calculation | - |
| `run_logit` | R `glm(family=binomial)` | Python `statsmodels.Logit` |
| `run_probit` | R `glm(family=binomial(probit))` | Python `statsmodels.Probit` |
| `run_hdfe` | R `lfe::felm` | - |
| `run_feglm` | R `alpaca::feglm` | R `fixest::feglm` |
| `run_var` | R `vars::VAR` | Python `statsmodels.VAR` |
| `run_varma` | R custom | Python `statsmodels` |
| `run_vecm` | R `vars::vec2var`, `urca::ca.jo` | - |
| `run_var_irf` | R `vars::irf` | - |
| `run_arima` | R `forecast::Arima` | Python `statsmodels.ARIMA` |
| `run_mstl` | R `forecast::mstl` | - |
| `run_changepoint` | R `changepoint` | Python `ruptures` |
| `kmeans` | R `stats::kmeans` | Python `sklearn.cluster.KMeans` |
| `dbscan` | R `dbscan::dbscan` | Python `sklearn.cluster.DBSCAN` |
| `hierarchical` | R `stats::hclust` | Python `scipy.cluster.hierarchy` |
| `pca` | R `stats::prcomp` | Python `sklearn.decomposition.PCA` |
| `tsne` | R `Rtsne::Rtsne` | Python `sklearn.manifold.TSNE` |
| `random_forest` | R `randomForest` | Python `sklearn.ensemble.RandomForestClassifier` |
| `linear_svm` | R `e1071::svm` | Python `sklearn.svm.LinearSVC` |
| `run_diagnostics` | R `lmtest` | - |

## Citation Format

When citing reference implementations in validation documents:

```markdown
**R Package Citation**:
> Author (Year). "Package: Title". R package version X.Y.Z. URL

**Example**:
> Gaure S (2013). "lfe: Linear Group Fixed Effects". R package version 2.8-8.
> https://CRAN.R-project.org/package=lfe
```
