# Doubly Robust (AIPW) Treatment Effect Validation

## Method Overview

`run_doubly_robust()` (`crates/p2a-core/src/econometrics/treatment.rs`) combines a logistic
propensity score (`glm::fit_propensity_lean`) with OLS outcome models fitted per arm on
(1, covariates) by OLS's solve decision: the normal equations on a full-rank,
well-conditioned design, pivoted QR otherwise (`lm.fit`; aliased columns count as 0 in
predictions, as `predict.lm`, with a warning).

Propensity fits go through `glm::fit_propensity_lean`: `fit_propensity`'s checks, then the lean
logit IRLS (`glm::fit_logit_lean`), the same Fisher scoring and deviance rule through the normal
equations. A fit that leaves that path (separation, a rank-deficient or ill-conditioned weighted
design, no convergence, a row with `y < 1` above `eta = 15` or `y > 0` below -15, a last step that no
longer meets the rule once refined) is `fit_glm`'s own fit, warnings included. The lean numbers
differ from `fit_glm`'s by rounding on regular designs; the gaps measured are at `fit_logit_lean`.

Three methods:

| Method | Effect | Standard error |
|--------|--------|----------------|
| `AIPW` (default) | ATE: mean of psi_i = m1 - m0 + D (Y - m1)/e - (1-D)(Y - m0)/(1-e); ATT: sum phi / n1 with phi = D (Y - m0) - (1-D) e/(1-e) (Y - m0) | analytic: sd(IF)/sqrt(n), n - 1 denominator, with IF = psi (ATE) or (phi - D tau) n/n1 (ATT); `bootstrap_reps = 0`, `se_method` "analytic influence function" |
| `IPW` | Hajek IPW | `bootstrap = 0`: the M-estimation SE with the propensity estimated, `se_method` "M-estimation (propensity estimated)"; `bootstrap >= 2` (default 999): bootstrap SD, "bootstrap (B replicates; models refitted)" |
| `Regression` | ATE: mean(m1 - m0) over all kept units; ATT: over the kept treated units | `bootstrap = 0`: the sandwich of the stacked estimating equations of the two arm regressions and the averaged contrast (HC0, the trimmed set fixed), "analytic (stacked estimating equations, trimming fixed)"; `bootstrap >= 2`: bootstrap SD |

The AIPW SE ignores the estimation of the nuisance models (the usual influence-function SE of
the R AIPW and tmle packages). For `IPW` and `Regression`, `bootstrap` is 0 (analytic) or at
least 2; 1 is an error. Each bootstrap replicate resamples every row, refits the propensity model
(when trimming or weighting needs it) and both arm regressions, and re-applies the trimming; a
replicate whose refit fails or leaves an arm empty is dropped and counted. The default stays 999
replicates; `bootstrap` is ignored by `AIPW`. Regression adjustment for the ATT averages the
predicted contrast over the treated (before it averaged over every unit, which is the ATE: on
`weighting_sim` 1.43074 against R's ATT 1.56333). Bootstrap intervals are type-7 percentile intervals (normal interval below
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
| Regression adjustment ATT / ATC (case `ra_ipw_t4`) | 1.5633310471356239 / 1.2963682627197133 |
| Regression adjustment sandwich SE: ATE / ATT (case `ra_ipw_t4`) | 0.12164156911903133 / 0.1294302125002856 |
| Outcome R² treated / control | 0.6566631288592546 / 0.6176147190812733 |

## Tests and tolerances

| Test | Checks | Tolerance |
|------|--------|-----------|
| `test_validate_aipw` | AIPW ATE, SE, outcome R², `bootstrap_reps = 0`, CI/t/p | effect `Tol::coef(n)`, SE `Tol::se(n)`, R² `Tol::stat(n)` |
| `test_validate_aipw_att` | AIPW ATT and SE | as above |
| `test_validate_dr_regression` | regression-adjustment effect; bootstrap SE reported as such (not pinned) | `Tol::coef(n)` |
| `test_validate_dr_regression_att` | regression-adjustment ATT over the treated (registry `causal_effects_t4.json`, entry `treatment_regression_adjustment_t4`) | `Tol::coef(n)` |
| `test_validate_dr_regression_sandwich_se` | regression-adjustment ATE and ATT with the analytic sandwich SE at `bootstrap = 0` | effect `Tol::coef(n)`, SE `Tol::se(n)` |
| `test_validate_dr_ipw_mest` | method `ipw` at `bootstrap = 0`: Hajek effect and M-estimation SE | as above |

Property tests (no case): `test_dr_reports_se_method`, `test_dr_ipw_matches_ipw` (DR IPW and
IPW agree to 1e-12), `test_dr_regression_recovers_dgp`,
`test_unseeded_bootstrap_is_reproducible`, `test_ipw_rejects_bad_trim`,
`test_dr_bootstrap_replicate_refits_models`, `test_dr_regression_bootstrap_se_near_sandwich`,
`test_dr_bootstrap_one_is_refused`.

## Known Differences

1. Outcome models are OLS per arm; packages such as AIPW or tmle default to machine-learning
   learners.
2. No cross-fitting (DoubleML cross-fits).
3. The bootstrap refits every model in each replicate; its draws are not R's, so bootstrap SEs
   are not pinned. The analytic `Regression` SE holds the trimmed set fixed; with `trim = 0` it is
   the fully interacted `lm`'s standardised contrast.

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

## Wall-clock against ff8e7559

Medians in µs over three interleaved rounds of the timing harness (`wp23`, one P core, one rayon
thread, `release`) at 003af73a; ratio is this tree over ff8e7559.

| Row | n | ff8e7559 | this tree | ratio |
|---|---|---:|---:|---:|
| AIPW | 100 | 25.98 | 16.86 | 0.65 |
| AIPW | 1,000 | 162.20 | 103.86 | 0.64 |
| AIPW | 10,000 | 1,863.52 | 1,341.66 | 0.72 |

Against `fit_glm` the shared lean logit fit differs by at most 4.5e-9 on a coefficient, 9.1e-8 on an SE and
2.1e-6 on a covariance entry at the default tolerance (quasi-separated fits of n = 15 to 80 with a rare binary
covariate; 8.2e-11 and 2.7e-11 on heavy-tailed fits). These gaps are accepted.

## References

- Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994). "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed."
- Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and Causal Inference Models."
- Glynn, A.N. & Quinn, K.M. (2010). "An Introduction to the Augmented Inverse Propensity Weighted Estimator."
