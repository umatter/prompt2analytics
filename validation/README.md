# Validation Framework

This directory contains validation documentation comparing prompt2analytics implementations against reference implementations in R, Python, and Julia.

## Purpose

Systematic validation is essential for:
1. **Correctness verification**: Ensure our implementations produce numerically accurate results
2. **Publication readiness**: Provide evidence for Journal of Statistical Software submission
3. **User confidence**: Give users assurance that results match established packages

## Structure

```
validation/
├── README.md                          # This file
├── reference_implementations.md       # Catalog of reference packages
├── regression/                        # OLS, robust SEs, clustered SEs
├── econometrics/                      # Panel, IV, DiD, discrete choice
│   └── timeseries/                    # VAR, VARMA, VECM, IRF
├── forecasting/                       # ARIMA, MSTL, changepoint
├── ml/                                # Clustering, dimensionality reduction (cmdscale, cutree)
├── stats/                             # Statistical methods (medpolish, isoreg, loglin)
├── diagnostics/                       # Regression diagnostics
└── datasets/                          # Reference datasets for testing
```

## Validation Document Format

Each method has a dedicated markdown file following a consistent template:

1. **Method Overview**: Brief description and key parameters
2. **Reference Implementations**: Table of packages used for comparison
3. **Test Cases**: Multiple scenarios with:
   - Dataset description
   - R/Python code for reproduction
   - Results comparison table
   - Link to Rust test function
4. **Numerical Precision Summary**: Expected precision by sample size
5. **Known Differences**: Any intentional deviations from references
6. **References**: Academic citations

## Running Validation Tests

```bash
# Run all validation tests
cargo test -p p2a-core -- test_validate

# Run tests for a specific method
cargo test -p p2a-core -- test_validate_ols
cargo test -p p2a-core -- test_validate_hdfe
cargo test -p p2a-core -- test_validate_grunfeld

# Run with output to see computed values
cargo test -p p2a-core -- test_validate --nocapture
```

## Reference Datasets

The `datasets/` directory contains standard econometric and ML datasets:

| Dataset | n | k | Use Case |
|---------|---|---|----------|
| grunfeld.csv | 200 | 5 | Panel data (firm + year FE) |
| longley.csv | 16 | 6 | Multicollinearity testing |
| iris.csv | 150 | 5 | Classification, clustering |

## Tolerance Guidelines

Numerical precision varies by sample size and method:

| Sample Size | Coefficient Tolerance | SE Tolerance |
|-------------|----------------------|--------------|
| n < 100 | 1e-6 | 1e-5 |
| n = 100-1000 | 1e-8 | 1e-6 |
| n > 1000 | 1e-10 | 1e-8 |

For iterative methods (HDFE, MLE), expect slightly larger differences due to convergence criteria.

## Adding New Validations

When implementing a new method:

1. Create `validation/[category]/[method].md` using the template
2. Add at least two test cases:
   - Synthetic data with known DGP
   - Real dataset (if available)
3. Document R/Python code for reproduction
4. Create corresponding Rust test with `test_validate_` prefix
5. Update this README's method index

## Method Index

### Regression
| Method | File | Status |
|--------|------|--------|
| OLS | [ols.md](regression/ols.md) | Pending |
| Robust SEs (HC0-HC3) | [ols_robust_se.md](regression/ols_robust_se.md) | Pending |
| Clustered SEs | [ols_clustered.md](regression/ols_clustered.md) | Pending |
| Tukey's Resistant Line | [line.md](regression/line.md) | **Complete** |
| Friedman's SuperSmoother | [supsmu.md](regression/supsmu.md) | **Complete** |
| Sensitivity Analysis (sensemakr) | [sensemakr.md](regression/sensemakr.md) | **Complete** |
| E-Value (unmeasured confounding) | [evalue.md](regression/evalue.md) | **Complete** |

### Econometrics
| Method | File | Status |
|--------|------|--------|
| Fixed Effects | [panel_fe.md](econometrics/panel_fe.md) | Pending |
| Random Effects | [panel_re.md](econometrics/panel_re.md) | Pending |
| Hausman Test | [hausman.md](econometrics/hausman.md) | Pending |
| IV/2SLS | [iv_2sls.md](econometrics/iv_2sls.md) | Pending |
| Diff-in-Diff | [did.md](econometrics/did.md) | Pending |
| Regression Discontinuity (Sharp) | [rd.md](econometrics/rd.md) | Pending |
| Regression Discontinuity (Fuzzy) | [rd.md](econometrics/rd.md) | Pending |
| IPW Treatment Effects | [treatment_ipw.md](econometrics/treatment_ipw.md) | Pending |
| Doubly Robust (AIPW) | [treatment_aipw.md](econometrics/treatment_aipw.md) | Pending |
| TMLE | [tmle.md](econometrics/tmle.md) | **Complete** |
| LTMLE (Longitudinal TMLE) | [ltmle.md](econometrics/ltmle.md) | **Complete** |
| CBPS | [cbps.md](econometrics/cbps.md) | **Complete** |
| SBW (Stable Balancing Weights) | [sbw.md](econometrics/sbw.md) | **Complete** |
| Mediation Analysis | [mediation.md](econometrics/mediation.md) | Pending |
| Natural Effect Models (medflex) | [medflex.md](econometrics/medflex.md) | **Complete** |
| Logit | [logit.md](econometrics/logit.md) | Pending |
| Probit | [probit.md](econometrics/probit.md) | Pending |
| HDFE | [hdfe.md](econometrics/hdfe.md) | **Complete** |
| Multi-Cutoff RD (rdmulti) | [rdmulti.md](econometrics/rdmulti.md) | **Complete** |

### Survival Analysis
| Method | File | Status |
|--------|------|--------|
| Kaplan-Meier | [survival.md](econometrics/survival.md) | **Complete** |
| Log-Rank Test | [survival.md](econometrics/survival.md) | **Complete** |
| Cox PH | [survival.md](econometrics/survival.md) | **Complete** |
| AFT Models | [survival.md](econometrics/survival.md) | **Complete** |
| Competing Risks | [survival.md](econometrics/survival.md) | **Complete** |

### Time Series
| Method | File | Status |
|--------|------|--------|
| VAR | [var.md](econometrics/timeseries/var.md) | Pending |
| VARMA | [varma.md](econometrics/timeseries/varma.md) | Pending |
| VECM | [vecm.md](econometrics/timeseries/vecm.md) | Pending |
| IRF | [irf.md](econometrics/timeseries/irf.md) | Pending |
| Cumulative Periodogram | [cpgram.md](timeseries/cpgram.md) | **Complete** |

### Forecasting
| Method | File | Status |
|--------|------|--------|
| ARIMA | [arima.md](forecasting/arima.md) | Pending |
| MSTL | [mstl.md](forecasting/mstl.md) | Pending |
| Changepoint | [changepoint.md](forecasting/changepoint.md) | Pending |

### Machine Learning
| Method | File | Status |
|--------|------|--------|
| K-means | [kmeans.md](ml/kmeans.md) | Pending |
| DBSCAN | [dbscan.md](ml/dbscan.md) | Pending |
| Hierarchical | [hierarchical.md](ml/hierarchical.md) | Pending |
| Cutree | [cutree.md](ml/cutree.md) | **Complete** |
| PCA | [pca.md](ml/pca.md) | Pending |
| t-SNE | [tsne.md](ml/tsne.md) | Pending |
| Classical MDS (cmdscale) | [cmdscale.md](ml/cmdscale.md) | **Complete** |
| Projection Pursuit Regression | [ppr.md](ml/ppr.md) | **Complete** |
| Random Forest | [random_forest.md](ml/random_forest.md) | Pending |
| SVM | [svm.md](ml/svm.md) | Pending |
| Causal Forest | [causal_forest.md](ml/causal_forest.md) | Pending |
| BART Causal (bartCause) | [bart_causal.md](ml/bart_causal.md) | **Complete** |

### Statistical Methods
| Method | File | Status |
|--------|------|--------|
| Median Polish | [medpolish.md](stats/medpolish.md) | **Complete** |
| Isotonic Regression | [isoreg.md](stats/isoreg.md) | **Complete** |
| Log-Linear Models | [loglin.md](stats/loglin.md) | **Complete** |
| Constrained Optimization | [constroptim.md](stats/constroptim.md) | **Complete** |
| SE for Contrasts | [secontrast.md](stats/secontrast.md) | **Complete** |
| Model Tables | [modeltables.md](stats/modeltables.md) | **Complete** |

### Spatial Econometrics
| Method | File | Status |
|--------|------|--------|
| Moran's I Test | [moran.md](spatial/moran.md) | Pending |
| SAR (Spatial Lag) | [sar.md](spatial/sar.md) | Pending |
| SEM (Spatial Error) | [sem.md](spatial/sem.md) | Pending |
| SAR Probit | [spatialprobit.md](econometrics/spatialprobit.md) | **Complete** |
| SEM Probit | [spatialprobit.md](econometrics/spatialprobit.md) | **Complete** |
| Spatial GMM (sphet) | [sphet.md](econometrics/sphet.md) | **Complete** |

### Linear Algebra
| Method | File | Status |
|--------|------|--------|
| Toeplitz Matrix | [toeplitz.md](linalg/toeplitz.md) | **Complete** |

### Diagnostics
| Method | File | Status |
|--------|------|--------|
| Regression Diagnostics | [regression_diagnostics.md](diagnostics/regression_diagnostics.md) | Pending |

## Contributing

When adding validations:
- Use consistent formatting
- Include reproducible R/Python code
- Test on multiple platforms if possible
- Document any known limitations
