# Reference Implementations Catalog

This document catalogs the reference implementations used to validate prompt2analytics methods.
A method is *pinned* to a reference only when a registered test compares it with a committed R
case (`validation/methods/*.json`, `validation/expected/<case>.json`); see
[README.md](README.md). Versions are those installed on the reference machine (re-checked
2026-10-28), and for the harness packages those recorded in the case files.

## R Packages

### Core Statistics
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| stats | R 4.3.3 | `lm()`, `summary.lm()`, `confint()`, `logLik()`, `AIC()`, `BIC()`, `step()`, `extractAIC()`, `glm()`, `kmeans()`, `hclust()`, `prcomp()` | Base R |
| base | R 4.3.3 | `kappa(exact = TRUE)` | Base R |

### Econometrics
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| plm | 2.6.7 | `plm()`, `pdata.frame()`, `ercomp()`, `pwaldtest()`, `phtest()`, `vcovHC(method = "arellano")`, `pgmm()`, `pggls()` | [plm](https://cran.r-project.org/package=plm) |
| fixest | 0.14.2 | `feols()`, `feglm()`, `fepois()`, `r2()`, `fitstat()` | [fixest](https://cran.r-project.org/package=fixest) |
| lfe | 3.1.1 | `felm()` (legacy reference, no longer pinned) | [lfe](https://cran.r-project.org/package=lfe) |
| alpaca | 0.3.5 | `feglm()` | [alpaca](https://cran.r-project.org/package=alpaca) |
| AER | 1.2.17 | `ivreg()` | [AER](https://cran.r-project.org/package=AER) |
| sandwich | 3.1.3 | `vcovHC()`, `vcovCL(type = "HC1")` one- and two-way, `kernHAC()`, `NeweyWest()`, `vcovPL()` | [sandwich](https://cran.r-project.org/package=sandwich) |
| lmtest | 0.9.40 | `bptest()`, `dwtest()`, `bgtest()`, `resettest()`, `harvtest()`, `waldtest()` | [lmtest](https://cran.r-project.org/package=lmtest) |
| car | 3.1.5 | `vif()` | [car](https://cran.r-project.org/package=car) |
| tseries | 0.10.63 | `jarque.bera.test()` | [tseries](https://cran.r-project.org/package=tseries) |
| quantreg | 6.1 | `rq()`, `summary.rq()` | [quantreg](https://cran.r-project.org/package=quantreg) |
| nlme | 3.1.164 | `gls(correlation = corAR1())` | [nlme](https://cran.r-project.org/package=nlme) |
| strucchange | 1.6.0 | `recresid()` | [strucchange](https://cran.r-project.org/package=strucchange) |
| aod | 1.3.3 | `wald.test()` (legacy cross-check of the Wald Chisq form) | [aod](https://cran.r-project.org/package=aod) |
| nnet | 7.3.19 | `multinom()` | [nnet](https://cran.r-project.org/package=nnet) |
| MASS | 7.3.60.0.1 | `polr()` | [MASS](https://cran.r-project.org/package=MASS) |
| rdrobust | 4.0.0 | `rdrobust()`, `rdbwselect()` | [rdrobust](https://cran.r-project.org/package=rdrobust) |

### Causal Inference
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| MatchIt | 4.8.0 | `matchit(method = NULL, distance = "glm")`, `summary()$sum.all[, "eCDF Max"]`: the quoted values of `econometrics::matching::tests::test_validate_ks_before_matching_vs_matchit` (T0-WP11) | [MatchIt](https://cran.r-project.org/package=MatchIt) |

### Time Series
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| vars | 1.6.1 | `VAR()`, `VECM()`, `irf()` | [vars](https://cran.r-project.org/package=vars) |
| forecast | 9.0.2 | `auto.arima()`, `Arima()`, `mstl()` | [forecast](https://cran.r-project.org/package=forecast) |
| urca | 1.3.4 | `ca.jo()`, `ur.df()` | [urca](https://cran.r-project.org/package=urca) |
| changepoint | 2.3 | `cpt.mean()`, `cpt.var()` | [changepoint](https://cran.r-project.org/package=changepoint) |

### Machine Learning
| Package | Version | Functions Used | CRAN |
|---------|---------|---------------|------|
| dbscan | 1.2.6 | `dbscan()` | [dbscan](https://cran.r-project.org/package=dbscan) |
| Rtsne | not installed on the reference machine | `Rtsne()` | [Rtsne](https://cran.r-project.org/package=Rtsne) |
| randomForest | 4.7.1.2 | `randomForest()` | [randomForest](https://cran.r-project.org/package=randomForest) |
| e1071 | 1.7.17 | `svm()` | [e1071](https://cran.r-project.org/package=e1071) |

## Python Packages (context only, not pinned)

No Python reference is pinned: no test compares p2a with a committed Python result.

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

## Julia Packages (context only, not pinned)

| Package | Functions Used | Registry |
|---------|---------------|----------|
| GLM.jl | `lm()`, `glm()` | General |
| FixedEffectModels.jl | `reg()` | General |

## Version Compatibility

Every R case records `R.version.string` and the package versions it used
(`validation/expected/<case>.json`), and the status page summarises them. Current: R 4.3.3
(2024-02-29). `Rscript validation/r/run.R --check` fails when an installed package version
computes a different reference value.

## Installation Commands

### R
Harness packages (the R cases in `validation/r/cases/*.R`):
```r
install.packages(c("jsonlite", "plm", "fixest", "sandwich", "lmtest", "car", "tseries",
                   "quantreg", "nlme", "aod", "strucchange", "nnet", "MASS"))
```

Legacy scripts (`validation/scripts/*.R`, run only with `validation/run_validation.sh --legacy-r`):
```r
install.packages(c(
  "AER", "aod", "BART", "boot", "car", "e1071", "EValue", "forecast", "gbm", "glmnet",
  "lightgbm", "lmtest", "MASS", "MatchIt", "mboost", "medflex", "mediation",
  "microbenchmark", "nlme", "nnet", "optmatch", "pdp", "plm", "pROC", "pscl", "quantreg",
  "randomForest", "rdrobust", "rpart", "sandwich", "survival", "tmle", "tseries", "vars",
  "WeightIt", "xgboost"
))
```

Other references named in method documents: `lfe`, `alpaca`, `urca`, `changepoint`, `dbscan`,
`Rtsne`.

### Python
```bash
pip install statsmodels linearmodels scikit-learn arch ruptures
```

## Method-to-Package Mapping

"Pinned" marks the methods with a registry entry (`validation/methods/*.json`); "deviation"
names a quantity the registry records as not matching R, with its finding.

| p2a Method | Primary Reference | Secondary Reference |
|------------|------------------|---------------------|
| `run_ols` | R `stats::lm`, `summary.lm`, `confint` (pinned) | Python `statsmodels.OLS` |
| `run_ols` (HC0-HC3) | R `sandwich::vcovHC(type = "HC0"` ... `"HC3")` (pinned) | Python `statsmodels` |
| OLS logLik/AIC/BIC | R `stats::logLik`, `AIC`, `BIC`, with k + 1 parameters (pinned) | - |
| `run_ols_clustered` | R `sandwich::vcovCL(cluster = ~g, type = "HC1")`, two-way `~g1 + g2` (pinned) | Python `linearmodels` |
| `step` | R `stats::step`, `extractAIC` (pinned) | - |
| `run_diagnostics` | R `tseries::jarque.bera.test`, `lmtest::bptest` (studentized), `lmtest::dwtest`, `car::vif`, `base::kappa(exact = TRUE)` (pinned) | - |
| `bg_test`, `reset_test`, `harvey_collier_test`, `wald_test` | R `lmtest::bgtest`, `resettest`, `harvtest` (with `strucchange::recresid`), `waldtest` (pinned; deviations REG-35 RESET, REG-37 Wald Chisq) | R `aod::wald.test` |
| `run_fixed_effects` | R `plm(model = "within")`, `vcovHC(method = "arellano")`, `pwaldtest` (pinned) | Python `linearmodels.PanelOLS` |
| `run_random_effects` | R `plm(model = "random")` Swamy-Arora, `ercomp`, `pwaldtest`; z inference (pinned) | - |
| `run_hausman_test` | R `plm::phtest`, classical covariances (pinned) | - |
| `run_hdfe` | R `fixest::feols`, `fitstat`; adjusted within R² from `plm(effect = "twoways")` (pinned) | R `lfe::felm` |
| HAC (`vcov_hac`) | R `sandwich::NeweyWest(lag =, prewhite = FALSE)`, `kernHAC(kernel =, bw =, prewhite = FALSE, adjust = FALSE)` (pinned; deviations REG-22, REG-50, REG-61) | - |
| Driscoll-Kraay (`vcov_driscoll_kraay`) | R `sandwich::vcovPL` (pinned; deviations REG-24, REG-50) | R `plm::vcovSCC` |
| Quantile regression | R `quantreg::rq(method = "br")`, `summary.rq(se = "iid" / "nid")` (pinned; deviations REG-26, REG-52, REG-64) | - |
| GLS AR(1) | R `nlme::gls(correlation = corAR1())`, REML (pinned; deviations REG-9, REG-43) | - |
| `run_multinom` | R `nnet::multinom` (pinned) | - |
| `run_ordered_logit`, `run_ordered_probit` | R `MASS::polr` (pinned; deviations ECON-4, ECON-5) | - |
| `run_iv2sls` | R `AER::ivreg` | Python `linearmodels.IV2SLS` |
| `run_did` | Manual calculation | - |
| `run_logit` | R `glm(family=binomial)` | Python `statsmodels.Logit` |
| `run_probit` | R `glm(family=binomial(probit))` | Python `statsmodels.Probit` |
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
| `run_rd_estimate` | R `rdrobust::rdrobust` | - |
| `run_rd_bw` | R `rdrobust::rdbwselect` | - |
| `run_rd_fuzzy` | R `rdrobust::rdrobust` (fuzzy) | - |

## Citation Format

When citing reference implementations in validation documents:

```markdown
**R Package Citation**:
> Author (Year). "Package: Title". R package version X.Y.Z. URL

**Example**:
> Gaure S (2013). "lfe: Linear Group Fixed Effects". R package version 2.8-8.
> https://CRAN.R-project.org/package=lfe
```
