# Validation Status Report

**Last Updated:** 2026-03-05
**Branch:** main

## Summary

| Metric | Count | Status |
|--------|-------|--------|
| **Rust Validation Tests** (`test_validate_*`) | 437 | All Pass |
| **R Validation Scripts** | 41 | All Pass |
| **R Expected Value CSVs** | 73 | All Pass |
| **Total Test Functions** | 1,848 | All pass (0 failures) |

### Recent Changes
- Added 15 ML validation tests: C5.0 (3), Cubist (3), CTree (3), MBoost (4), SHAP (2)
- Fixed LightGBM index-out-of-bounds bug and consolidated validation tests
- Added 15 causal/econometrics Rust benchmarks to comprehensive_benchmarks.rs
- Consolidated spatial R benchmarks from 5 to 2 files
- Integrated tracking allocator for per-method heap measurement
- Added validation docs for staggered DiD, ETWFE, Bacon, DoubleML, CTMLE

## How to Run Validation

```bash
# Full validation (Rust + R)
./validation/run_validation.sh

# Rust tests only (faster)
./validation/run_validation.sh --rust-only

# R scripts only
./validation/run_validation.sh --r-only

# Filter by category
./validation/run_validation.sh --category stats

# Run specific validation tests
cargo test -p p2a-core -- test_validate
```

## Coverage by Category

### Statistics (121 validation tests)

**Fully Validated (R cross-validated):**
- T-tests (one-sample, two-sample, paired, Welch)
- ANOVA (one-way, two-way, Welch variant)
- MANOVA (Pillai, Wilks, Hotelling, Roy)
- Chi-squared (goodness-of-fit, independence, Yates)
- Fisher's exact test
- Wilcoxon (rank-sum, signed-rank)
- Shapiro-Wilk normality
- Kolmogorov-Smirnov (one-sample, two-sample)
- Bartlett's test
- Kruskal-Wallis
- Friedman test
- Tukey HSD
- ACF, PACF, CCF
- Box-Pierce, Ljung-Box
- Phillips-Perron
- Factor analysis (MLE with rotation)
- Canonical correlation
- Power analysis (t-test, proportion, ANOVA)
- Robust statistics (fivenum, IQR, MAD, ECDF, density)
- Spline interpolation
- Weighted mean and covariance
- Spectrum/periodogram
- Ansari-Bradley, Fligner-Killeen, Mood, Quade
- McNemar, Mantel-Haenszel
- Binomial test, proportion test, Poisson test
- Correlation tests (Pearson, Spearman, Kendall)
- Pairwise t-test, pairwise Wilcoxon
- p.adjust (Holm, BH)
- Mahalanobis distance
- Median polish
- Isotonic regression
- Constrained optimization
- Classical MDS (cmdscale)

### Regression (32 validation tests)

**Fully Validated (R cross-validated):**
- OLS (Longley dataset, simple regression)
- Robust SEs: HC0, HC1, HC2, HC3
- Clustered SEs (one-way, two-way)
- HAC (Newey-West)
- Driscoll-Kraay
- Bootstrap covariance
- LOESS
- NLS (exponential decay, Michaelis-Menten)
- GLS (AR1, compound symmetry)
- Smooth splines
- Sensemakr (sensitivity analysis)
- E-value (unmeasured confounding)
- Quantile regression
- Stepwise selection
- Super smoother (supsmu)
- Line (Tukey)
- Diagnostics: RESET, Breusch-Godfrey, Harvey-Collier, Wald

### Econometrics (155 validation tests)

**Fully Validated (R cross-validated):**
- Panel FE (Grunfeld, full Grunfeld, synthetic)
- Panel RE (Grunfeld, full Grunfeld)
- Hausman test
- Panel GLS (FEGLS, pooled GLS, PGGLS)
- PVCM (within, random)
- PMG (Pooled Mean Group)
- Arellano-Bond GMM (one-step, two-step)
- HDFE (high-dimensional fixed effects, vs felm)
- FEGLM (logit, probit, vs alpaca)
- IV/2SLS (basic, overidentified, with controls)
- First-stage diagnostics
- DiD (classic 2x2, with covariates, null effect)
- RD (sharp vs rdrobust, bandwidth methods, polynomial orders, CIs, p-values)
- Fuzzy RD (structure validation)
- Treatment: IPW (ATE, ATT, propensity scores, trimming)
- Treatment: AIPW/Doubly robust
- TMLE (vs R tmle package, targeting step, clever covariate, influence curve, counterfactuals, truncation, continuous outcome)
- Matching (propensity score, caliper, balance SMD, ESS, full matching, subclassification, CEM)
- Mediation analysis (decomposition, CIs, proportion mediated)
- Spatial: SAR, SEM, SAC, spatial Durbin, impacts
- Survival: Kaplan-Meier, Cox PH (ties, concordance), AFT, competing risks, log-rank
- Time series: VAR, VARMA, VECM, IRF, Granger causality
- Discrete: Logit, probit, multinomial logit, ordered logit/probit, negative binomial, ZIP, ZINB, hurdle (Poisson, NB), conditional logit, mixed logit
- Balke-Pearl bounds

**DGP-validated (known-parameter recovery, not exact R cross-validation):**
- Staggered DiD (Callaway-Sant'Anna): ATT recovery, pre-trend nulls, event study structure, never-treated vs not-yet-treated, aggregation
- Extended TWFE (Wooldridge): coefficient recovery, cohort effects, event study, pre-trends, standard errors
- Goodman-Bacon decomposition: weight summation, timing variation, 2x2 DD recovery, treated-untreated vs timing groups, within-group decomposition, total effect recovery
- Double/Debiased ML: PLR coefficient recovery, cross-fitting, IRM treatment effect
- Synthetic control: pre-period fit, weight constraints, treatment effect sign
- SCPI: prediction intervals cover truth, width ordering, point estimate consistency, pre-period coverage
- CTMLE: ATE in plausible range, propensity truncation, CI coverage, covariate selection
- LTMLE: single-period equivalence to TMLE, monotone treatment, time-varying confounding, survival outcome, wide CI for small samples
- CBPS: propensity score bounds, positive weights, balance improvement, convergence, overidentification test
- Entropy balancing (WeightIt): positive finite weights, balance improvement, ESS reasonable, ATT estimation, convergence

### Forecasting (54 validation tests)

**Fully Validated (R cross-validated):**
- AR models (Yule-Walker, OLS, Burg, order selection)
- Holt-Winters (additive, multiplicative, non-seasonal, forecasting)
- STL decomposition
- Classical decomposition (additive, multiplicative)
- GARCH (parameters, conditional variance, forecasting, inference, information criteria)
- Changepoint detection (PELT, binary segmentation)
- Time series utilities: lag, embed, cumsum, detrend, filter/diffinv

**DGP-validated (known-parameter recovery):**
- ARIMA: AR(1) coefficient recovery, AIC finite, forecast decay, MA(1) coefficient recovery
- Kalman filter/smoother: local level tracking, smoothed tighter than filtered, log-likelihood finite
- MSTL: components sum to data, seasonal periodicity, trend smoothness
- Structural time series: local level tracking, trend slope, BSM components sum, log-likelihood
- CausalImpact: pre-period fit, post-period detection, cumulative effect, no-effect null

### Machine Learning (69 validation tests)

**Fully Validated (R cross-validated):**
- K-means (centroids, assignment, inertia, reproducibility, vs R)
- DBSCAN (eps sensitivity, noise detection, vs R)
- Hierarchical clustering (Ward, single, complete linkage, cutree, vs R)
- PCA (centering, orthonormality, variance ordering, transformation, vs R)
- t-SNE (local structure preservation)

**DGP-validated (known-parameter recovery):**
- Random Forest: classification accuracy on separable data, feature importance ordering, regression MSE, OOB error
- SVM: linear separability, margin properties, support vector count, prediction accuracy
- C5.0: classification accuracy, boosted trials, prediction consistency
- Cubist: regression R², committee improvement, out-of-sample prediction
- CTree: regression R², classification accuracy, prediction on test data
- MBoost: linear and tree base learners, out-of-sample R², Poisson family
- SHAP: tree ensemble attribution, additivity property, feature importance ordering
- XGBoost: regression R², feature importance, classification accuracy
- LightGBM: regression R² (index-out-of-bounds bug fixed)
- BART: regression R², posterior intervals

## Benchmark Coverage

The comprehensive benchmark file (`crates/p2a-core/benches/comprehensive_benchmarks.rs`) now includes:
- Regression: OLS, OLS+HC1
- Panel: FE, RE, HDFE
- Discrete: Logit, Probit
- Time Series: ARIMA, MSTL
- ML: K-Means, PCA, DBSCAN, Hierarchical, Random Forest
- Spatial: SAR, SEM
- Causal: DiD, Staggered DiD, ETWFE, Bacon, IV/2SLS, RD, TMLE, CTMLE, IPW, CBPS, Matching, WeightIt, DoubleML, Mediation, LTMLE
- Other: LOESS, Doubly Robust, Changepoint, Synthetic Control, Factor Analysis, Fisher Exact, Isotonic Regression, Jarque-Bera

Spatial R benchmarks consolidated from 5 to 2 files. Redundant `ml_benchmarks.rs` removed.

## File Locations

- **Runner Script:** `validation/run_validation.sh`
- **R Scripts:** `validation/scripts/` (41 scripts)
- **Expected Values:** `validation/expected/` (73 CSVs)
- **Validation Docs:** `validation/[category]/`
- **Reports:** `validation/reports/`
- **Pipeline Overview:** `performance/README.md`
- **Coverage Check:** `performance/comparisons/check_coverage.R` (companion paper repo)

## Adding New Validations

1. Create Rust test with `test_validate_` prefix
2. Create R script in `validation/scripts/`
3. Create documentation in `validation/[category]/method.md`
4. Run `./validation/run_validation.sh` to verify

See `validation/README.md` for detailed template.
