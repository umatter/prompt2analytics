# Validation: Collaborative Targeted Maximum Likelihood Estimation (C-TMLE)

Status: R-pinned (registry `validation/methods/causal_ctmle_stdreg.json`, method `causal_ctmle`).

## Method Overview

Collaborative TMLE (van der Laan & Gruber, 2010) selects the covariates of the propensity
model g by how much they help the targeting step, judged by cross-validation. p2a implements
greedy selection of g with a fixed initial outcome model Q:

1. **Initial Q.** A logistic glm (`CTmleQModel::Logistic`, the default) or least squares
   (`Linear`) of Y on all candidate covariates and A, fitted on the full data and on the
   training rows of every fold. Folds are deterministic: observation `i` is in fold `i mod V`.
2. **Path of g models.** Forward: start from the intercept-only g and add, per step, the
   candidate with the smallest criterion. Backward: start from all candidates and remove one
   per step. Prespecified: add the covariates in exactly the given order (an empty list means
   the listed order). `max_covariates` caps every order; `stop_factor` ends the path when the
   best criterion of a step exceeds `stop_factor` times the best so far (ctmle's
   `stopFactor`).
3. **V-fold risk of a candidate g.** For every fold v, Q, g and the fluctuation epsilon (one
   clever covariate H = A/g - (1 - A)/(1 - g), logit scale for the logistic Q) are fitted on
   the training folds; the loss is evaluated on fold v.
4. **Selection.** The first minimum of the criterion (`CVMinimum`), the step with the fewest
   covariates within one SE of it (`OneSE`), or the last step (`MaxCovariates(k)`). The
   reported estimate is the full-data TMLE with the selected g; its SE is
   sqrt(var(IC) / n).

A continuous outcome under the logistic Q is rescaled once to [0, 1] by its range (as R's
tmle). The criteria are on that scale; the ATE, SE, CI, influence curve and Q* are reported on
the original scale, with a warning that names the range (`outcome_range`).

## Criteria

| `CVCriterion` | Definition | SE | ctmle name |
|---|---|---|---|
| `RSS` (default) | sum over folds of the validation-fold RSS of Q* | sqrt(V) sd(fold RSS) | likelihood (`cv`) |
| `VarIC` | mean over folds of the validation-fold variance of the influence curve | sd(fold varIC) / sqrt(V) | `varIC.cv` |
| `PenRSS` | RSS + `penalty_factor` ln(n) k, for a g with k covariates | as RSS | penalised likelihood |

Every step also reports `cv_rss` and `cv_var_ic`, whatever the criterion.

## Deviations from ctmle 0.1.2

ctmle is installed but not used as the pin; its source (CRAN tarball 0.1.2,
`functions_discrete.R`: `cv`, `select_terms`, `calc_varIC`) was read for the criteria.

- The initial Q is fixed: ctmle updates Q between clever covariates, p2a does not (WP29-9).
- The initial Q is a parametric glm / lm on all candidates, not SuperLearner.
- The folds are deterministic (`i mod V`), not random.
- g is fitted on the training folds; ctmle fits g on all rows (for stability) and epsilon on the
  training rows. p2a fits both on the training rows.
- A continuous outcome is fitted with the linear fluctuation (`Linear`) or, under the logistic
  Q, rescaled once to [0, 1].
- The propensity bounds `gbound` are symmetric truncation bounds (default (0.025, 0.975)).

## R reference

Case `ctmle_cv` in `validation/r/cases/causal_stdreg.R` writes the algorithm out with
`stats::glm` (epsilon 1e-14) on `validation/datasets/generated/ctmle_sim.csv` (n = 300; w1
confounder, w2 instrument, w3 precision variable, w4 noise), V = 5, gbound (0.025, 0.975).

| Run | criterion path | moved | selected step (mask) | ATE, SE |
|---|---|---|---|---|
| `logit_rss` | 53.1424, 53.0899, 53.0949, 53.1003, 53.1293 | 0, 3, 2, 1 | 1 (1, 0, 0, 0) | 0.22194283711310114, 0.053305920357218366 |
| `logit_varic` | 0.71199, 0.71725, 0.75155, 0.94620, 1.86630 | 3, 2, 0, 1 | 0 | 0.22396697298544035, 0.04775037014637593 |
| `logit_penrss` (penalty 2) | 53.1424, 64.4974, 75.9100, 87.3230, 98.7595 | 0, 3, 2, 1 | 0 | 0.22396697298544035, 0.04775037014637593 |
| `linear_rss` | 322.910, 322.620, 322.633, 322.514, 321.874 | 1, 3, 0, 2 | 4 (1, 1, 1, 1) | 1.93014013777743, 0.1559891534625552 |
| `linear_varic` | 4.4276, 4.4879, 4.5590, 5.8412, 18.8754 | 3, 2, 0, 1 | 0 | 2.047586907800023, 0.1178942301568494 |
| `logit_backward` | 53.1293, 53.1003, 53.0949, 53.0899, 53.1424 | 1, 2, 3, 0 (removed) | 3 (1, 0, 0, 0) | 0.22194283711310114, 0.053305920357218366 |
| `logit_pre31` (order [2, 0]) | 53.1424, 53.1421, 53.0967 | 2, 0 | 2 (1, 0, 1, 0) | 0.2230074723486674, 0.053085208142047806 |

The case also records, per run, the criterion SE, CV RSS and CV varIC of every step, the
number of covariates, the ATE and SE of every step, epsilon, the CI, the p-value and the
one-SE selection (`logit_rss` selects step 0 under one-SE, `logit_backward` step 4), and the
fold-level RSS, varIC and epsilon of g = {w1, w3} (`cv13_*`).

Before the fix the criterion used the full-data Q and refitted epsilon on the validation
fold: the logistic path decreased monotonically (10.0453, 9.9904, 9.9849, 9.9709, 9.9667),
selected all four covariates and gave ATE 0.24745603630730373; VarIC and PenRSS were printed
but every selection used the RSS.

## Tests and tolerances

R-pinned (`load_case("ctmle_cv")`): `test_validate_ctmle_cv_logit_rss`,
`test_validate_ctmle_cv_linear_rss`, `test_validate_ctmle_cv_backward_and_prespecified`,
`test_validate_ctmle_criteria_varic`, `test_validate_ctmle_criteria_penrss`,
`test_validate_ctmle_cv_risk_uses_training_fluctuation` (fold level; flipping fold 0's
outcomes leaves its fitted epsilon bit-identical) and
`test_validate_ctmle_prespecified_order_in_arrays_path` (array and dataset entry points agree).

| Quantity | Tolerance |
|---|---|
| criteria, SEs, CV RSS and varIC, ATE, SE, epsilon, CI, per-step ATE and SE, fold values | `Tol::iterative(1e-8)` (glm epsilon 1e-14 on both sides; the smallest selection gap is 2.4e-5) |
| moved covariates, selected step and covariates, one-SE selection | exact |
| p-value | `Tol::p_value(300)` |

Property tests on the same data (`ctmle_sim.csv` read directly): the order is validated
(`test_ctmle_prespecified_order_is_validated`), `max_covariates` caps every order
(`test_ctmle_max_covariates_caps_every_order`), `n_folds` is checked on both entry points
(`test_ctmle_n_folds_checked_in_core`), a non-0/1 treatment is refused
(`test_ctmle_refuses_non_binary_treatment`), the logistic Q on a continuous outcome equals the
run on the rescaled outcome (`test_ctmle_logistic_q_rescales_continuous_outcome`), the
penalty and stop factor are validated, the option strings parse, and Display reports the
algorithm and criterion (`test_ctmle_display_reports_algorithm_and_criterion`). DGP tests:
`test_ctmle_ate_recovers_dgp`, `test_ctmle_selection_path_properties`,
`test_ctmle_se_is_ic_variance`, `test_ctmle_propensity_bounds`.

## Options

| Option | Strings |
|---|---|
| `cv_criterion` | `rss` (`cv_rss`, `likelihood`), `varic` (`var_ic`, `cv_varic`, `variance_ic`), `penrss` (`pen_rss`, `cv_penrss`, `penalized_rss`, `penalised_rss`) |
| `q_model` | `logistic` (`logit`, `binomial`, `binary`), `linear` (`gaussian`, `continuous`, `ols`) |
| `order` | `forward` (`fwd`), `backward` (`bwd`, `backward_elimination`), `prespecified` (`preorder`, `listed`, `pre_ordered`; the listed order) |
| `stopping_rule` | `cv_minimum` (`cvmin`, `cv_min`, `minimum`, `min`), `one_se` (`onese`, `1se`, `one_standard_error`), `max_covariates` (`maxcovariates`, `max`; needs `max_covariates`) |

Case and `-`/space are ignored.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::ctmle
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

R: ctmle is not installed here; not verified.

Tests: `test_validate_na_ctmle_columns` (case `causal_mediation_na`, registry
`validation/methods/causal_na.json`, id `weighting_na_refusal`).

## References

- van der Laan, M.J. & Gruber, S. (2010). "Collaborative Double Robust Targeted Maximum Likelihood Estimation." *The International Journal of Biostatistics*, 6(1).
- Ju, C., Schwab, J., & van der Laan, M.J. (2019). "On adaptive propensity score truncation in causal inference." *Statistical Methods in Medical Research*, 28(6), 1741-1760.
