# Doubly Robust (AIPW) Treatment Effect Validation

## Method Overview

`run_doubly_robust()` (`crates/p2a-core/src/econometrics/treatment.rs`) combines a logistic
propensity score (`glm::fit_propensity`) with OLS outcome models fitted per arm on
(1, covariates) by pivoted QR (`lm.fit`; aliased columns count as 0 in predictions, as
`predict.lm`, with a warning). Three methods:

| Method | Effect | Standard error |
|--------|--------|----------------|
| `AIPW` (default) | ATE: mean of psi_i = m1 - m0 + D (Y - m1)/e - (1-D)(Y - m0)/(1-e); ATT: sum phi / n1 with phi = D (Y - m0) - (1-D) e/(1-e) (Y - m0) | analytic: sd(IF)/sqrt(n), n - 1 denominator, with IF = psi (ATE) or (phi - D tau) n/n1 (ATT); `bootstrap_reps = 0`, `se_method` "analytic influence function" |
| `IPW` | Hajek IPW | bootstrap SD, `se_method` "bootstrap (B replicates)" |
| `Regression` | mean(m1 - m0) | bootstrap SD |

The AIPW SE ignores the estimation of the nuisance models (the usual influence-function SE of
the R AIPW and tmle packages). `bootstrap` must be at least 2 for `IPW` and `Regression`; it is
ignored by `AIPW`. Bootstrap intervals are type-7 percentile intervals (normal interval below
20 successful replicates); an unseeded bootstrap uses `DEFAULT_BOOTSTRAP_SEED` (42). Analytic
intervals use `qnorm(0.975)`, p-values the normal survival function. `trim` must be in
[0, 0.5).

## Reference

- R code: `validation/r/cases/causal_ipw_aipw.R`, case `ipw_aipw_sim`.
- Data: `validation/datasets/generated/weighting_sim.csv` (300 units, 151 treated).
- Registry: `validation/methods/causal_weighting.json`, entry `treatment_aipw`.

| Quantity (trim = 0) | R value |
|---------------------|---------|
| AIPW ATE, SE | 1.4265452600187223, 0.12280740322389656 |
| AIPW ATT, SE | 1.5121321048221326, 0.13667714475644796 |
| Regression adjustment ATE | 1.4307395308757216 |
| Outcome R² treated / control | 0.6566631288592546 / 0.6176147190812733 |

## Tests and tolerances

| Test | Checks | Tolerance |
|------|--------|-----------|
| `test_validate_aipw` | AIPW ATE, SE, outcome R², `bootstrap_reps = 0`, CI/t/p | effect `Tol::coef(n)`, SE `Tol::se(n)`, R² `Tol::stat(n)` |
| `test_validate_aipw_att` | AIPW ATT and SE | as above |
| `test_validate_dr_regression` | regression-adjustment effect; bootstrap SE reported as such (not pinned) | `Tol::coef(n)` |

Property tests (no case): `test_dr_reports_se_method`, `test_dr_ipw_matches_ipw` (DR IPW and
IPW agree to 1e-12), `test_dr_regression_recovers_dgp`,
`test_unseeded_bootstrap_is_reproducible`, `test_ipw_rejects_bad_trim`.

## Known Differences

1. Outcome models are OLS per arm; packages such as AIPW or tmle default to machine-learning
   learners.
2. No cross-fitting (DoubleML cross-fits).
3. The bootstrap resamples units with the fitted propensity scores and outcome predictions
   held fixed.

## Missing values and column types

Missing values (null or NaN) in the outcome, treatment or covariates are refused, with
"Column 'x' contains k missing value(s) (null or NaN). Drop or impute them first."
(`EconError::NullValues`, naming the column and the count). A text column gives
`NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are
read as numbers and give the same result as Float64. Results on complete data are
unchanged.

R: MatchIt 4.8.0 and tmle 2.1.1 refuse missing treatment or covariate values; p2a refuses
missing values in every column the doubly robust estimator reads (checked for T2-WP13).

Tests: `test_validate_na_doubly_robust_columns` (case `causal_mediation_na`, registry
`validation/methods/causal_na.json`, id `weighting_na_refusal`).

## References

- Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994). "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed."
- Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and Causal Inference Models."
- Glynn, A.N. & Quinn, K.M. (2010). "An Introduction to the Augmented Inverse Propensity Weighted Estimator."
