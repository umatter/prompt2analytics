# Validation: glmnet (elastic net, lasso, ridge)

## Overview

`p2a_core::regression::glmnet` fits penalised linear and logistic regression along a lambda
path by coordinate descent:

| Function | Computes | R equivalent |
|---|---|---|
| `glmnet`, `run_glmnet` | the path of fits for a `GlmnetConfig` (gaussian or binomial) | `glmnet::glmnet` |
| `ridge`, `lasso`, `run_ridge`, `run_lasso` | `glmnet` with alpha 0 / 1 | `glmnet(alpha = 0 / 1)` |
| `cv_glmnet`, `run_cv_glmnet` | K-fold cross-validation with random folds (seed 42 by default) | `cv.glmnet` |
| `cv_glmnet_foldid` | cross-validation with given fold labels | `cv.glmnet(foldid = ...)` |
| `glmnet_predict`, `glmnet_coef` | predictions / coefficients at any lambda | `predict.glmnet(s, exact = FALSE)`, `coef.glmnet(s)` |

The MCP tools `regression_glmnet`, `regression_lasso`, `regression_ridge` and
`regression_cv_glmnet` call these functions and return their `Display` text (every fit with
its nonzero coefficients; for CV the coefficients at `lambda.min` and `lambda.1se`).

## Reference

glmnet 5.1 on R 4.3.3. One case, `glmnet_paths`, in
[`validation/r/cases/reg_glmnet.R`](../r/cases/reg_glmnet.R), with the data
`validation/datasets/generated/glmnet_sim.csv` (150 rows: `y`, binary `yb`, `X1..X8` with X2
correlated with X1 and X5 shifted and scaled, fold labels `foldid` (5 equal folds) and
`foldid7` (7 folds of 21 or 22 rows), and an AR(0.85) block `K1..K30` with a sparse response
`yk`).

```bash
Rscript validation/r/run.R --write glmnet_paths   # rewrite the JSON and the dataset
Rscript validation/r/run.R --check glmnet_paths   # recompute and compare
```

glmnet's coordinate descent stops when `max_j xv_j delta_j^2 < thresh` (standardised units),
so even at `thresh = 1e-14` its coefficients are up to 1.0e-7 (absolute) away from the
minimiser of its own objective on these data. The case therefore also writes `*_tight`
values: the exact minimiser, solved from the KKT conditions on glmnet's active set (one linear
solve per active set for gaussian, Newton with fixed signs for binomial), with an active-set
refinement loop and `stopifnot` checks that every KKT condition holds to 1e-12. Measured gaps,
glmnet output vs tight (max absolute): gaussian coefficients 4.5e-8 to 1.012e-7, binomial
8.7e-9 to 6.4e-8, dev.ratio at most 2.0e-9; `cvm` at most 6.7e-9 and `cvsd` at most 3.04e-8
relative. The Rust tests run at `thresh = 1e-24` and are pinned to the tight values; glmnet's
own output is a looser cross-check.

## Conventions reproduced

**Objective** (gaussian, in the standardised x units `z`):
`(1/2n) ||y - b0 - Z b||^2 + lambda [alpha ||b||_1 + (1 - alpha) / (2 ys) ||b||^2]`.
glmnet solves it on `yt = (y - ym) / ys`, so the L1 weight is `lambda alpha / ys` and the L2
weight `lambda (1 - alpha) / ys` in those units. A textbook ridge with penalty
`(lambda'/2) ||beta||^2` is glmnet's `lambda = lambda' * ys`.
Binomial: `-(1/n) sum(y eta - log(1 + e^eta)) + lambda [alpha ||b||_1 + (1 - alpha)/2 ||b||^2]`,
y not standardised.

**Standardisation** (glmnetpp `standard1`, unit weights):

| standardize | intercept | centre of x | scale of x | gaussian ym | gaussian ys |
|---|---|---|---|---|---|
| TRUE | TRUE | column mean | population sd | mean(y) | population sd of y |
| TRUE | FALSE | 0 | population sd of the centred column | 0 | sqrt(mean(y^2)) |
| FALSE | TRUE | column mean | 1 | mean(y) | population sd of y |
| FALSE | FALSE | 0 | 1 | 0 | sqrt(mean(y^2)) |

A constant column is excluded (its coefficient is 0 and the others are unchanged, glmnet's
`ju`); every column constant is an error, as is a constant gaussian y.

**lambda_max**: `max_j |z_j'(y - c)| / n / max(alpha, 1e-3)` with `c = ym` (gaussian) or the
null-fit probability `p0 = mean(y)` (binomial; 0.5 without an intercept). The ridge path
therefore starts at 1882.47 on the case data. **lambda_min_ratio**: 0.01 when n < p, else 1e-4.
The generated path is `lambda_max * ratio^(k / (nlambda - 1))`; `nlambda = 1` is `[lambda_max]`.

**Path rules** (generated paths, from the 5th fit, glmnet's `mnlam`): stop after the first fit
with df > `dfmax`, with a dev.ratio gain below `fdev * dev.ratio` (gaussian) or below `fdev`
(binomial), or with dev.ratio above `deviance_thresh` (devmax 0.999); the fit that triggers
is kept. User lambdas are sorted decreasingly and always fitted in full.

**Convergence**: a lambda has converged when a full pass over every usable variable moves no
coefficient by more than `max_j v_j delta_j^2 < thresh` (between full passes, passes over the
active set), so no excluded variable violates its KKT condition. `max_iter` counts coordinate
passes over the whole path (glmnet's cumulative `maxit`); when it runs out the path is cut with
a warning.

**Binomial**: IRLS (at most 100 outer iterations per lambda, probabilities bounded at 1e-9)
with an inner weighted coordinate descent in which the intercept is an unpenalised coordinate,
so `sum(y - p) = 0` at convergence. The response is coded by its two levels (the larger is 1,
as `factor()`); more than two classes, or a class with at most one observation, is an error,
a class with fewer than 8 observations a warning.

**Cross-validation**: the full-data path is the grid; each fold is fitted on its own path and
its held-out rows predicted at every grid lambda by linear interpolation, clamped at the
fold path's ends. Errors: squared residual, or the binomial deviance with probabilities
bounded at 1e-5 (`cv.lognet`). `cvm` and `cvsd` are cv.glmnet's `cvstats` (grouped): fold
means weighted by fold size, `cvsd = sqrt(sum_k w_k (m_k - cvm)^2 / sum w / (K - 1))`.
`lambda.min` is the largest lambda with the smallest `cvm`, `lambda.1se` the largest with
`cvm <= cvm_min + cvsd_min`.

**predict / coef**: `glmnet:::lambda.interp`: linear interpolation in lambda between the two
bracketing fits, the first / last fit outside the path.

## Test cases

All in `crates/p2a-core/src/regression/glmnet.rs`.

| Test | Keys | Tolerance | Finding |
|---|---|---|---|
| `test_validate_glmnet_gaussian_lasso_std` | `a10_std[_noint]_coef[_tight]`, `_dev_ratio[_tight]` | EXACT, R_GLMNET, R_DEV | REG-7 |
| `test_validate_glmnet_gaussian_lasso_raw` | `a10_raw[_noint]_*` | as above | REG-7 |
| `test_validate_glmnet_gaussian_enet_std` | `a5_std[_noint]_*` | as above | REG-40 |
| `test_validate_glmnet_gaussian_enet_raw` | `a5_raw[_noint]_*` | as above | REG-7, REG-40 |
| `test_validate_glmnet_gaussian_ridge_std` | `a0_std[_noint]_*` | as above | REG-40 |
| `test_validate_glmnet_gaussian_ridge_raw` | `a0_raw[_noint]_*` | as above | REG-7, REG-40 |
| `test_validate_glmnet_kkt_excluded_variable` | `kkt_lambda`, `kkt_df`, `kkt_coef_tight`; KKT of every excluded variable | exact; 1e-9 rel + 1e-10 abs | REG-41 |
| `test_validate_glmnet_convergence_is_scale_free` | y x 1e-6 and x 1e6: same passes and rescaled coefficients | 1e-6 rel | REG-41 |
| `test_validate_glmnet_maxit_truncates_path_with_warning` | cumulative `max_iter`, truncation warning, `ConvergenceFailure` | exact | REG-41 |
| `test_validate_glmnet_constant_column_is_excluded` | `a10_std_coef_tight`, `a10_raw_coef_tight` with a constant column added | EXACT | REG-7 |
| `test_validate_glmnet_default_path` | `path_lambda`, `path_df`, `path_coef_tight`, `path_dev_ratio[_tight]`, `ridge_path_lambda_max`, `ridge_path_length`, `enet_path_lambda_max` | `Tol::coef(150)`, exact, EXACT, R_DEV | REG-40 |
| `test_validate_glmnet_path_rules` | `wide_path_lambda`, `wide_path_df`, `dfmax2_path_df`, `fdev0_path_length`; user lambdas | `Tol::coef(20)`, exact | REG-40 |
| `test_validate_glmnet_nlambda_one_is_lambda_max` | `nlambda1_coef`, `nlambda1_dev_ratio`; CV with one lambda | EXACT, exact | REG-42 |
| `test_validate_glmnet_binomial` | `binom`, `binom_raw`, `binom_noint`, `binom_a5`: `_coef[_tight]`, `_dev_ratio[_tight]`; `sum(y - p) = 0` | EXACT, R_GLMNET, R_DEV | REG-6 |
| `test_validate_glmnet_binomial_path` | `binom_path_*`, `binom_noint_sub_lambda_max` | `Tol::coef(150)`, exact, EXACT, R_DEV | REG-6 |
| `test_validate_glmnet_binomial_two_level_coding` | {0, 1} and {-1, 1} give the same fit; small-class warning | exact | REG-6 |
| `test_validate_cv_glmnet_foldid` | `cv_*`, `cvb_*` (cvm, cvsd, indices, fold path lengths, lambda.min / 1se, coefficients) | 1e-9 rel (tight), 1e-7 / 1e-6 rel (glmnet), exact | REG-42 |
| `test_validate_glmnet_predict_interpolates` | `interp_coef[_tight]`, `interp_pred`, `binom_interp_prob[_tight]`; invalid lambda | EXACT, R_GLMNET, 1e-6 | REG-42, X-6 |
| `test_validate_glmnet_display_prints_coefficients` | the `Display` text of a path and a CV fit | - | REG-8 |

Property tests that load no case: `test_cv_glmnet_exact_fit_has_finite_cvsd` (REG-42: no NaN
`cvsd` when folds stop early), `test_glmnet_rejects_empty_lambda` (REG-42),
`test_glmnet_rejects_invalid_inputs` (every invalid parameter is an `InvalidSpecification`
naming it), `test_glmnetfamily_parses_aliases_and_rejects_unknown`.

## Known differences from R

1. `alpha` outside [0, 1] is an error (R clamps a value above 1 to 1 with a warning).
2. An empty user `lambda` is an error (R returns an empty model with a warning).
3. `nlambda = 1` reports `lambda_max` with a note (R reports `9.9e35 * sd(y)` for the same
   all-zero fit), and `nlambda = 2` reports `lambda_max` as its first value (R: the same large
   number).
4. `nfolds = 2` is accepted (R requires at least 3).
5. Non-convergence at the first lambda is a `ConvergenceFailure` (R returns an empty model).
6. The number of coordinate passes (`npasses`) differs from R's (no covariance updates or
   strong rules), so the lambda at which a small `max_iter` truncates the path differs, and R
   is faster on large n with p < 500.
7. Not offered: weights, offset, `penalty.factor`, `exclude`, lower / upper limits, `pmax`,
   `relax`, the binomial saturation stop (devmax stops separable paths), other families.

The soft-threshold treats a gradient within 1e-13 (relative) of the threshold as at it, so a
fit at a `lambda_max` computed in other arithmetic is exactly zero (as R's first fit); the
effect on any coefficient is below `1e-13 * lambda / v_j`.

## Tolerances

- `EXACT = Tol::new(1e-9, 1e-11)`: Rust at `thresh = 1e-24` against the tight values (an R
  port of the algorithm lands within 1e-12; the KKT path uses abs 1e-10, its largest gap in the
  port being 2.35e-11).
- `R_GLMNET = Tol::new(1e-6, 2e-7)`: against glmnet's own coefficients (up to 1.0e-7 from the
  minimiser).
- `R_DEV = Tol::new(1e-8, 5e-9)`: against glmnet's dev.ratio (up to 2.0e-9 off).
- Lambdas: `Tol::coef(n)` (1e-8 relative at n = 150, 1e-6 at n = 20); df, indices and path
  lengths exact; tight dev.ratio, `cvm`, `cvsd`: `Tol::iterative(1e-9)`; glmnet's `cvm` 1e-7 and
  `cvsd` 1e-6 relative.
