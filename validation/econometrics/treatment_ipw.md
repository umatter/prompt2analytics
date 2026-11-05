# IPW Treatment Effect Validation

## Method Overview

`run_ipw_treatment()` (`crates/p2a-core/src/econometrics/treatment.rs`) estimates the ATE or
ATT by inverse probability weighting with a logistic propensity score (the shared GLM,
`glm::fit_propensity`; scores clipped to [1e-10, 1 - 1e-10]) and optional symmetric trimming
(`trim` in [0, 0.5); ATT trims treated units on the upper tail only).

| Estimator | Effect |
|-----------|--------|
| Hajek ATE (`normalized = true`, default) | weighted mean of treated (w = 1/e) minus weighted mean of controls (w = 1/(1-e)) |
| Horvitz-Thompson ATE (`normalized = false`) | mean of D y/e - (1-D) y/(1-e) |
| ATT | treated mean minus the control mean weighted by e/(1-e) |

## Standard errors

Three analytic SEs are computed on every call:

| Field | Definition | R equivalent |
|-------|------------|--------------|
| `se_mest` | M-estimation: the exact sandwich of the stacked logit-score and weighted-mean (HT: mean) estimating equations, so the propensity estimation is accounted for | `WeightIt::lm_weightit(vcov = "asympt")` (Hajek, ATT); the stacked sandwich in the case for HT |
| `se_eif` | efficient influence function (outcome models per arm by OLS), centred at the IPW estimate | written out in the case |
| `se_naive` | propensity treated as known: HC0 of the weighted regression of y on (1, D) (Hajek, ATT); sqrt(sum IF^2)/n with IF = D y/e - (1-D) y/(1-e) - tau (HT) | `lm_weightit(vcov = "HC0")` / `sandwich::vcovHC(type = "HC0")` |

`std_error` is `se_mest` (decision WP23-14 (a)), and `se_method` says
"M-estimation (propensity estimated)". With trimming, the logit score runs over all units and
the weighted-mean equations over the kept units: the trimmed set is held fixed (Crump et al.
2009). That trimmed case is not R-pinned. If the propensity information matrix is singular,
or the M-estimation SE is not finite, `std_error` falls back to `se_eif` with a warning, and to
`se_naive` if no outcome model can be fitted either. A trim that leaves one arm without units
is an `InsufficientData` error naming the arm and the trim. The outcome models of `se_eif`
(the treated arm's for the ATE only; the ATT form uses the control arm) report aliasing as
"outcome model (treated|control): ...". With `bootstrap >= 2`, `std_error` is the bootstrap SD ("bootstrap (B
replicates)") and the interval the type-7 percentile interval (normal interval below 20
successful replicates); an unseeded bootstrap uses `DEFAULT_BOOTSTRAP_SEED` (42), so results
are reproducible (Tier 3 R13). Analytic intervals use `qnorm(0.975)`; p-values the normal
survival function.

## Reference

- R code: `validation/r/cases/causal_ipw_aipw.R`, case `ipw_aipw_sim` (glm with
  `epsilon = 1e-14`; WeightIt 2.0.0; sandwich).
- Data: `validation/datasets/generated/weighting_sim.csv` (300 units, 151 treated; covariates
  x1, x2, x3), written by `validation/r/cases/causal_weighting.R`.
- Registry: `validation/methods/causal_weighting.json`, entry `treatment_ipw`.

| Quantity (trim = 0, bootstrap = 0) | R value |
|------------------------------------|---------|
| Hajek ATE | 1.4382520579328 |
| Hajek SE: M-estimation / WeightIt asympt / EIF / naive | 0.1225765751696695 / 0.12257657511202587 / 0.12260441640780892 / 0.20992212935271706 |
| HT ATE | 1.4177875444032675 |
| HT SE: M-estimation / EIF / naive | 0.15067342510051335 / 0.12260359599504984 / 0.39692785210986226 |
| ATT | 1.5144404674963763 |
| ATT SE: M-estimation / WeightIt asympt / EIF / naive | 0.15513367447104748 / 0.15513367426840752 / 0.13644355245633874 / 0.23137177194254968 |

## Tests and tolerances

| Test | Checks | Tolerance |
|------|--------|-----------|
| `test_validate_ipw_hajek_se` | Hajek effect; `std_error` = `se_mest`, `se_eif`, `se_naive`; CI, t and p from `qnorm(0.975)` | effect `Tol::coef(n)`; SEs `Tol::se(n)`; WeightIt cross-check 1e-6 relative |
| `test_validate_ipw_ht_se` | HT effect and the three SEs | as above |
| `test_validate_ipw_att_se` | ATT effect and the three SEs | as above |

Property tests (no case): `test_ipw_mest_se_with_a_fixed_trimmed_set`,
`test_dr_reports_se_method`, `test_unseeded_bootstrap_is_reproducible`,
`test_ipw_rejects_bad_trim`, `test_seeded_bootstrap_unchanged`,
`test_ipw_propensity_summary_in_range`, `test_ipw_trimming_reduces_sample`,
`test_ipw_t_stat_and_p_value_consistent`.

## Known Differences

1. The M-estimation SE uses glm's fit at `epsilon = 1e-10` (the shared GLM); WeightIt fits at
   glm's default `epsilon = 1e-8` and its Jacobian differs by about 5e-10, hence the 1e-6
   cross-check.
2. The bootstrap resamples units with the propensity scores held fixed (no refit per
   replicate).
3. Under trimming, the SE holds the trimmed set fixed; it is not R-pinned.

## Missing values and column types

Missing values (null or NaN) in the outcome, treatment or covariates are refused, with
"Column 'x' contains k missing value(s) (null or NaN). Drop or impute them first."
(`EconError::NullValues`, naming the column and the count). A text column gives
`NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are
read as numbers and give the same result as Float64. Results on complete data are
unchanged. A NaN cell counts as missing, like a null.

R: MatchIt 4.8.0 and tmle 2.1.1 refuse missing treatment or covariate values, and WeightIt
2.0.0 refuses a missing treatment; p2a refuses missing values in every column the IPW
estimator reads (checked for T2-WP13).

Tests: `test_validate_na_ipw_columns`, `test_validate_na_ipw_nan_outcome_is_missing`,
`test_validate_mcp_treatment_ipw_null_outcome_is_named` (case `causal_mediation_na`, registry
`validation/methods/causal_na.json`, id `weighting_na_refusal`). They pin the refused column
and the missing count; the IPW estimates are pinned by the tests above.

## References

- Horvitz, D.G. & Thompson, D.J. (1952). "A Generalization of Sampling Without Replacement from a Finite Universe."
- Lunceford, J.K. & Davidian, M. (2004). "Stratification and weighting via the propensity score in estimation of causal treatment effects: a comparative study." *Statistics in Medicine*, 23, 2937-2960.
- Crump, R.K., Hotz, V.J., Imbens, G.W. & Mitnik, O.A. (2009). "Dealing with limited overlap in estimation of average treatment effects." *Biometrika*, 96(1), 187-199.
- Greifer, N. (2024). WeightIt: Weighting for Covariate Balance in Observational Studies. R package.
