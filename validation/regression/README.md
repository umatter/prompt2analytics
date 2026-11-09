# Regression Validation

This directory contains validation documentation for regression methods.

## Methods

| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| OLS | [ols.md](ols.md) | `run_ols()`, `run_ols_raw()` | R `lm()` |
| Robust SEs | [ols_robust_se.md](ols_robust_se.md) | `run_ols()` with HC0-HC3 | R `sandwich::vcovHC()` |
| Clustered SEs | [ols_clustered.md](ols_clustered.md) | `run_ols_clustered()` | R `sandwich::vcovCL()` |
| Bootstrap covariance | [vcovBS.md](vcovBS.md) | `vcov_bootstrap()` | R `sandwich::vcovBS()` (not pinned: R's RNG) |
| Regression through the origin | [no_intercept.md](no_intercept.md) | `run_ols_raw()`, `run_ols()` | R `lm(y ~ 0 + x)`, `summary.lm()` |
| HAC, Driscoll-Kraay, quantile regression, GLS, RESET, Wald (legacy DGPs) | [regression_legacy_cases.md](regression_legacy_cases.md) | `vcov_hac()`, `vcov_driscoll_kraay()`, `run_quantreg()`, `run_gls()` | R `sandwich`, `quantreg`, `nlme`, `lmtest` |
| Quantile regression | [quantreg.md](quantreg.md) | `run_quantreg()`, `quantreg()` | R `quantreg::rq()`, `summary.rq()` |
| Generalized least squares | [gls.md](gls.md) | `run_gls()`, `run_gls_with()` | R `nlme::gls()` |
| glmnet (elastic net, lasso, ridge) | [glmnet.md](glmnet.md) | `run_glmnet()`, `run_cv_glmnet()` | R `glmnet::glmnet()`, `cv.glmnet()` |
| Nonlinear least squares | [nls.md](nls.md) | `run_nls()` | R `stats::nls()` |
| LOESS | [loess.md](loess.md) | `run_loess()` | R `stats::loess()` |
| Smoothing spline | [smooth_spline.md](smooth_spline.md) | `run_smooth_spline()` | R `stats::smooth.spline()` |
| Friedman's SuperSmoother | [supsmu.md](supsmu.md) | `run_supsmu()` | R `stats::supsmu()` |
| Tukey's resistant line | [line.md](line.md) | `run_line()` | R `stats::line()` |
| Average marginal effects | [marginal_effects.md](marginal_effects.md) | `marginal_effects()` | R `marginaleffects` |
| Sensitivity to unmeasured confounding | [sensemakr.md](sensemakr.md) | `run_sensemakr()` | R `sensemakr::sensemakr()` |
| E-values | [evalue.md](evalue.md) | `evalue_rr()`, `evalue_or()` | R `EValue::evalues.RR()`, `evalues.OR()` |

Whether each is pinned to R is in the method index of [../README.md](../README.md).

## Key Test Datasets

- **Longley (1967)**: Classic dataset for testing collinearity (n=16, k=6)
- **Synthetic data**: Known DGP for exact coefficient recovery

## Running Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- regression::ols::tests::test_validate
```
