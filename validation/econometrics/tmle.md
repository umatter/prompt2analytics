# Targeted Maximum Likelihood Estimation (TMLE) Validation

Status: R-pinned (registry `validation/methods/causal_ctmle_stdreg.json`, method `causal_tmle`).

## Method Summary

TMLE is a doubly robust, semiparametric efficient estimator of the average treatment effect.
It differs from AIPW by a targeting step that fluctuates the initial outcome model along the
efficient influence curve of the ATE.

## Algorithm (as implemented)

1. **Initial outcome model Q(A, W).** A logistic glm (`QModel::Logistic`, the default:
   binomial for a 0/1 outcome, quasibinomial for a proportion) or least squares
   (`QModel::Linear`) of Y on W and A (`econometrics::glm`, R's `glm.fit`; `solve_ols`).
2. **Propensity score g(W).** A logistic glm of A on W, bounded to `truncate_ps` (default
   (0.01, 0.99)). A must be coded 0/1.
3. **Targeting step.** Two clever covariates, as tmle 2.1.1: H1 = A/g and
   H0 = (1 - A)/(1 - g), fitted as `glm(Y ~ -1 + H0W + H1W, offset = qlogis(Q),
   quasibinomial)` for the logistic Q (an offset regression on the identity scale for the
   linear Q). Q*(1, W) moves by epsilon1 / g, Q*(0, W) by epsilon0 / (1 - g); both scores are
   solved (`fluctuation_coef`, `fluctuation_coef_control`).
4. **Estimate.** ATE = mean(Q*(1, W) - Q*(0, W)).
5. **Variance.** IC = (H1 - H0)(Y - Q*(A, W)) + Q*(1, W) - Q*(0, W) - ATE;
   SE = sqrt(var(IC) / n); Wald 95% CI and two-sided normal p-value.

**Continuous outcome under the logistic Q.** Y is rescaled to [0, 1] by its range, as R's
tmle does (Gruber & van der Laan 2010); the estimates are reported on the original scale and
a warning names the range (`outcome_range`). A constant outcome is refused.

Every glm reports its convergence and warnings; aliased covariates get the warning
"Coefficients not defined because of singularities: ..." and an aliased treatment in Q is an
error (the effect is not identified).

## R reference

Case `tmle_glm` in `validation/r/cases/causal_stdreg.R` writes the algorithm out with
`stats::glm` (epsilon 1e-14) on `validation/datasets/generated/ctmle_sim.csv` (n = 300;
outcomes `yb` binary and `yc` continuous; covariates w1..w4), `truncate_ps` (0.01, 0.99).

| Key | logistic Q (`yb`) | linear Q (`yc`) |
|---|---|---|
| `*_epsilon1`, `*_epsilon0` | 0.0170969311147531, -0.023075260087028083 | -0.03449000836874782, -0.002787455439970942 |
| `*_ate` | 0.24817356846720856 | 1.9492378228097935 |
| `*_se` | 0.05557509884796119 | 0.1521457919515258 |
| `*_ci_lower`, `*_ci_upper` | 0.1392483762879512, 0.35709876064646595 | 1.651037550185479, 2.247438095434108 |
| `*_z`, `*_p` | 4.465553343344399, 7.986207006467184e-06 | 12.811644658768003, 1.4111159641175093e-37 |
| `*_n_truncated`, `*_ic_mean` | 0, -4.2e-17 | 0, -2.1e-17 |

The case also records the Q and g coefficients and Q, Q*, Q*(1), Q*(0), g and IC at six rows.

**Cross-check with tmle 2.1.1.** `tmle::tmle(cvQinit = FALSE)` with glm libraries gives
`tmle_pkg_ate` 0.2481735684652222 and `tmle_pkg_se` 0.055575098846510025 on `yb`: the
written-out algorithm agrees to 8e-12 (ATE) and 2.6e-11 (SE), so these are pinned too (the
case stops if they differ).

**Rescaled continuous outcome.** The logistic Q on `yc` rescaled from [-5.3358358029,
8.6007654538]: `rescaled_ate` 1.951299785179432, `rescaled_se` 0.15249104635598437, pinned.
tmle 2.1.1 gives 1.9521579422711284 (SE 0.15168759637679835), 4.4e-4 relative away, because
it also bounds Q to 1% beyond the observed range and to alpha = 0.9995 before the
fluctuation; p2a rescales only. The tmle values are recorded, not asserted.

## Differences from R tmle

1. **SuperLearner.** R's tmle fits Q and g by SuperLearner; p2a uses parametric glm / least
   squares only (with glm libraries the two agree, see above).
2. **Q bounding.** tmle bounds Q to 1% beyond the observed range and with alpha = 0.9995;
   p2a bounds Q only to (1e-10, 1 - 1e-10) before taking logits.
3. **Variance.** IC-based only.

## Tests and tolerances

R-pinned (`load_case("tmle_glm")`): `test_validate_tmle_glm_logit`,
`test_validate_tmle_glm_linear`, `test_validate_tmle_glm_default_control` (R's
`glm.control()` defaults: within `Tol::iterative(1e-8)` of the tight values, all fits
converged), `test_validate_tmle_aliased_covariate_and_treatment`,
`test_validate_tmle_logistic_q_rescales_continuous_outcome` and `test_validate_tmle_scale_free`
(covariates scaled by 1e-6 and 1e6).

| Quantity | Tolerance |
|---|---|
| epsilons, ATE, Q, Q*, g and IC at six rows | `Tol::iterative(1e-8)` (glm epsilon 1e-14 on both sides) |
| SE, CI, z | 1e-6 relative |
| p-value | 1e-6 relative / 1e-12 absolute |
| tmle 2.1.1 ATE and SE | 1e-8 |
| rescaling range, `n_truncated` | exact |

Property tests on `ctmle_sim.csv` read directly: separation in Q warns
(`test_tmle_q_separation_warns`), a fit stopped by `max_iter` is reported
(`test_tmle_nonconvergence_is_reported`), a non-0/1 treatment is refused
(`test_tmle_refuses_non_binary_treatment`), bad controls are refused
(`test_tmle_rejects_bad_config`), the bivariate fluctuation solves both scores
(`test_tmle_bivariate_fluctuation_solves_both_scores`), and the option strings parse
(`test_q_model_parses_aliases_and_rejects_unknown`).

## Options

| Option | Strings |
|---|---|
| `q_model` | `logistic` (`logit`, `binomial`, `binary`), `linear` (`gaussian`, `continuous`, `ols`) |
| `g_model` | `logistic` (`logit`, `glm`) |

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::tmle
Rscript validation/r/run.R --check
```

## Missing values and column types

Missing values (null or NaN) in the outcome, treatment or covariates are refused, with
"Column 'x' contains k missing value(s) (null or NaN). Drop or impute them first."
(`EconError::NullValues`, naming the column and the count). A text column gives
`NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are
read as numbers and give the same result as Float64. Results on complete data are
unchanged.

R: tmle 2.1.1 refuses a missing treatment or covariate ("No missing values allowed in 'A' or
'W'") and treats a missing outcome as censored (`Delta`); p2a refuses a missing outcome as well
(checked for T2-WP13).

Tests: `test_validate_na_tmle_columns` (case `causal_mediation_na`, registry
`validation/methods/causal_na.json`, id `weighting_na_refusal`).

## References

1. van der Laan, M.J. & Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

2. van der Laan, M.J. & Rubin, D. (2006). Targeted Maximum Likelihood Learning. *The International Journal of Biostatistics*, 2(1), Article 11. https://doi.org/10.2202/1557-4679.1043

3. Gruber, S. & van der Laan, M.J. (2012). tmle: An R Package for Targeted Maximum Likelihood Estimation. *Journal of Statistical Software*, 51(13), 1-35. https://doi.org/10.18637/jss.v051.i13

4. R package `tmle`: https://cran.r-project.org/package=tmle

## File Locations

- Implementation: `crates/p2a-core/src/econometrics/tmle.rs`
- R case: `validation/r/cases/causal_stdreg.R` (`tmle_glm`)
- This document: `validation/econometrics/tmle.md`
