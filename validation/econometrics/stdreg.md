# Validation: Regression Standardization (G-computation) - stdReg

Status: R-pinned (registry `validation/methods/causal_ctmle_stdreg.json`, method `causal_stdreg`).

## Method Overview

Regression standardization (G-computation, the parametric g-formula for a single time point)
estimates causal effects by fitting an outcome model and averaging its predictions under
A = 1 and A = 0 over the covariate distribution (all rows for the ATE, the treated for the
ATT, the controls for the ATC).

## Mathematical Framework

Under the assumptions of consistency, no unmeasured confounding (conditional exchangeability), and positivity:

```
E[Y(a)] = sum_x E[Y|A=a, X=x] * P(X=x)
        = (1/n) * sum_i E[Y|A=a, X_i]
```

For ATE (Average Treatment Effect):
```
ATE = E[Y(1)] - E[Y(0)]
    = (1/n) * sum_i [Y_hat_i(1) - Y_hat_i(0)]
```

For ATT (Average Treatment Effect on Treated):
```
ATT = E[Y(1) - Y(0) | A=1]
    = (1/n1) * sum_{i:A_i=1} [Y_hat_i(1) - Y_hat_i(0)]
```

For ATC (Average Treatment Effect on Controls):
```
ATC = E[Y(1) - Y(0) | A=0]
    = (1/n0) * sum_{i:A_i=0} [Y_hat_i(1) - Y_hat_i(0)]
```

## Implementation

`crates/p2a-core/src/econometrics/stdreg.rs`.

- **Outcome models.** Linear: least squares (`regression::ols::solve_ols`; the pivoted QR of
  `lm` when X'X is ill-conditioned). Logistic and Poisson: `econometrics::glm` (R's `glm.fit`)
  with `StdRegConfig::tolerance` as `epsilon` and `max_iter` as `maxit` (defaults 1e-8 and
  100). The logistic model refuses an outcome outside [0, 1]. Optional A x W interactions.
- **Convergence.** Logistic and Poisson results carry `convergence` (the "Converged:" line of
  Display and the `ResultNotes` block); a fit stopped by `max_iter` adds a "did not converge"
  warning.
- **Separation.** When fitted probabilities are numerically 0 or 1 (within 10 eps), glm's
  warning "glm.fit: fitted probabilities numerically 0 or 1 occurred" is reported with a
  sentence that counts the observations and says the maximum likelihood estimates do not
  exist.
- **Aliasing.** An aliased covariate gets a `NaN` coefficient (`null` in JSON) and the
  warning "outcome model: Coefficients not defined because of singularities: <names>"; the
  predictions use the identified columns (as `predict.lm` on estimable rows). An aliased
  treatment, or an interaction `A:W_j` aliased while `W_j` is not, is an error: the
  treatment effect is not identified.
- **Standard errors.** The bootstrap (`n_bootstrap` replicates, each refitting the outcome
  model; non-converged, separated and failed replicates are counted in warnings) is the
  reference. A replicate computes coefficients only: the linear model by the normal equations
  without OLS's refinement step (the pivoted QR when the design is rank-deficient or ill
  conditioned), the logistic model by TMLE's lean logit IRLS from the full-sample fit (the
  shared GLM when a replicate separates, is aliased, does not converge or has a converged fit
  with a row beyond the path's bound, `y < 1` above `eta = 15` or `y > 0` below -15, so those
  counts are the shared GLM's, and a replicate past the bound can count as not converged where
  the lean path alone would have converged), the Poisson model by the shared GLM; its
  percentile intervals for the effect, risk ratio and odds ratio use R's type-7 quantiles of
  the finite draws (`stats::robust::percentile_interval`). The delta-method and sandwich SEs
  are approximations on the treatment coefficient (CAUS-39, Tier 4), computed on the identified
  columns with a rank-checked inverse and a checked residual df. Normal intervals and p-values
  use `z_critical` and `normal_p_value_two_sided`.

## R reference

stdReg is not installed. The cases in `validation/r/cases/causal_stdreg.R` fit `stats::glm`
(epsilon 1e-14) or `stats::lm` and standardise the predictions as `stdReg::stdGlm` does, on
`validation/datasets/generated/stdreg_sim.csv` (n = 300; `yc` continuous with an A x x1
interaction, `yb` binary, `yp` Poisson with mean 222.4, `ys` = 1{x1 > 0.3}, `x1dup` =
2 x1 + 1).

| Key | R value |
|---|---|
| `logit_coef` (1, x1, x2, a) | -0.6557248048371712, 0.7907472864446148, -0.34401877351902155, 0.7574552632300408 |
| `logit_ey1`, `logit_ey0`, `logit_ate` | 0.5073753169514722, 0.34583038938994765, 0.16154492756152455 |
| `logit_att`, `logit_atc` | 0.16489988073060807, 0.15856978041158248 |
| `logit_int_ate`, `logit_int_att`, `logit_int_atc` | 0.1635170827729261, 0.16485284794792382, 0.16233253629698485 |
| `pois_coef` | 5.302379996194625, 0.098190581111931, 0.05251967624740585, 0.19888933354127217 |
| `pois_ey1`, `pois_ey0`, `pois_ate` | 245.7233703364667, 201.4048494768568, 44.31852085960989 |
| `pois_att`, `pois_atc` | 44.7201784180356, 43.962333968175784 |
| `logit_r2`, `pois_r2` (1 - deviance / null deviance) | 0.11559061465461717, 0.8278262458731582 |
| `int_coef` (lm, interactions) | 1.0453148804393335, 0.47034163451602723, -0.18182901285405184, 0.7229240607743814, 0.19629512882936156, -0.3982527111947572 |
| `int_ate`, `int_att`, `int_atc`, `int_r2` | 0.6873326323708036, 0.7090998207865995, 0.6680296539643431, 0.35741255922400317 |
| `alias_coef` (1, x1, x2, x1dup, a) | 1.0720912063931451, 0.5573733596703365, -0.39657179117390473, NA, 0.6906491233525095 |
| `alias_ate` | 0.6906491233525096 (R 4.3.3 `predict.lm` does not warn: the rows are estimable) |
| `sep_ey1`, `sep_ey0`, `sep_ate` | 0.3500000000000552, 0.3499999999783898, 2.17e-11 |
| `sep_fitted_extreme`, `sep_converged`, `sep_warn_fitted01`, `sep_warn_noconv` | 294, 1, 1, 0 |

Before the fix, the Poisson model (mean 222) returned E[Y(1)] = E[Y(0)] = 1e10 and ATE 0 (the
clipped mean), and the separated logistic fit returned ATE 0 and SE 0 without a warning.

**Delta-method SEs (recorded, not asserted).** stdReg's stacked estimating equations, written
out in the case: `logit_stdreg_ate_se` 0.05312617119409766, `logit_int_stdreg_ate_se`
0.053540593969082506, `pois_stdreg_ate_se` 1.7444188478832212. p2a's delta and sandwich SEs
are approximations until CAUS-39 (Tier 4).

**Default control.** At `StdRegConfig::default()` (R's `glm.control()` epsilon 1e-8) the logit
and Poisson coefficients and ATEs are within about 6e-10 of the tight values above, and the
fits converge.

## Tests and tolerances

R-pinned: `test_validate_stdreg_glm` (logit with and without interactions and Poisson;
coefficients, E[Y(1)], E[Y(0)], ATE, ATT, ATC, R2), `test_validate_stdreg_glm_default_control`,
`test_validate_stdreg_linear` (interactions; the aliased covariate with its NA coefficient and
warning), `test_validate_stdreg_separation_warns` and `test_validate_stdreg_poisson_scale_free`
(covariates scaled by 1e-6 and 1e6).

| Quantity | Tolerance |
|---|---|
| coefficients and effects | `Tol::coef(300)` = 1e-8 (glm epsilon 1e-14 on both sides) |
| R2 | `Tol::stat(300)` = 1e-6 |
| default-control coefficients and ATEs | `Tol::iterative(1e-8)` of the tight values |
| scaled Poisson ATE | 1e-9 of the unscaled run, `Tol::iterative(1e-8)` of `pois_ate` |
| separation count, convergence, warning flags | exact |

Property tests on `stdreg_sim.csv` read directly: an aliased treatment (or treatment
interaction) is an error (`test_stdreg_aliased_treatment_is_an_error`), an aliased covariate
leaves the sandwich and delta SEs unchanged (`test_stdreg_aliased_covariate_keeps_sandwich_se`),
`confidence_level` is validated (`test_stdreg_rejects_bad_level`), the logistic model refuses an
outcome outside [0, 1] (`test_stdreg_logistic_needs_unit_outcome`), and Display ends with the
notes (`test_stdreg_notes_follow_display`). `test_stdreg_bootstrap_ci_is_type_7` checks the
percentile rule against R's `quantile(1:10, c(.05, .95))`;
`test_stdreg_options_parse_aliases_and_reject_unknown` the option strings.

## Options

| Option | Strings |
|---|---|
| `model_type` | `linear` (`ols`, `gaussian`, `continuous`), `logistic` (`logit`, `binomial`, `binary`), `poisson` (`count`, `counts`, `log_linear`) |
| `estimand` | `ate`, `att`, `atc`, `levels` (`potential_outcomes`) |
| `se_method` | `bootstrap` (`boot`), `delta` (`delta_method`), `sandwich` (`robust`) |

## MCP Tool

**Tool name**: `regression_standardization`

**Parameters**:
- `dataset`: Name or ID of a loaded dataset
- `outcome`: Outcome variable column
- `treatment`: Binary treatment indicator column (0/1)
- `covariates`: Covariate column names
- `model_type`: `linear` (default), `logistic` or `poisson`
- `estimand`: `ate` (default), `att`, `atc` or `levels`
- `se_method`: `bootstrap` (default), `delta` or `sandwich`
- `n_bootstrap`: Number of bootstrap replications (default 999)
- `interactions`: Include treatment-covariate interactions (default false)
- `confidence_level`: Confidence level for intervals (default 0.95)

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::stdreg
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

R: stdReg is not installed here; not verified.

Tests: `test_validate_na_stdreg_columns` (case `causal_mediation_na`, registry
`validation/methods/causal_na.json`, id `weighting_na_refusal`).

## Wall-clock against fe13b889

Medians over three interleaved rounds (`wp6f`, one P core, one rayon thread, `release`) at 811c26bb,
logistic model, two covariates; ratio is this tree over fe13b889.

| Row | n | fe13b889 (ms) | this tree (ms) | ratio |
|---|---|---:|---:|---:|
| normal covariates | 1,000 | 157.6 | 63.2 | 0.40 |
| normal covariates | 10,000 | 1,889.7 | 675.7 | 0.36 |
| heavy-tailed covariate with far misfit rows | 10,000 | 2,383.7 | 2,307.3 | 0.97 |

Replicates with a row past `|eta| = 15` are refitted by the shared GLM, which takes the far-row design
back to about the base's time.

## References

1. Robins, J.M. (1986). "A new approach to causal inference in mortality studies with a sustained exposure period -- application to control of the healthy worker survivor effect." *Mathematical Modelling*, 7(9-12), 1393-1512.

2. Snowden, J.M., Rose, S., & Mortimer, K.M. (2011). "Implementation of G-computation on a simulated dataset: Demonstration of a causal inference technique." *American Journal of Epidemiology*, 173(7), 731-738.

3. Hernan, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. Chapter 13.

4. Sjolander, A. (2016). "Regression standardization with the R package stdReg." *European Journal of Epidemiology*, 31(6), 563-574.
