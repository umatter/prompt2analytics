# Model-based boosting (mboost)

## Method

`p2a_core::ml::mboost` is component-wise gradient boosting. The linear learner is mboost's `bols(xj)` with an
intercept, fitted to the negative gradient; the step of the selected learner is scaled by `nu`. Families, as in
mboost 2.9 `R/family.R` (the response `y` coded 0/1 for the two binomial families):

| `MboostFamily` | mboost | Negative gradient, with `yt = 2 y - 1` | Offset |
|---|---|---|---|
| `Gaussian` | `Gaussian()` | `y - f` | `mean(y)` |
| `Binomial` | `Binomial()` | `2 yt exp(-2 yt f) / (ln 2 (1 + exp(-2 yt f)))` | `log(p / (1 - p)) / 2` |
| `BinomialGlm` | `Binomial(type = "glm")` | `dlogis(f) (y - p) / (p (1 - p))` | `qlogis(mean((y + 0.5) / 2))` |
| `Poisson` | `Poisson()` | `y - exp(f)` | `log(mean(y))` |

`Binomial` coefficients are on mboost's scale, half the log-odds, so they converge to half of glm's logit
coefficients. `BinomialGlm` converges to glm's coefficients.

The tree learner fits a least-squares tree of depth `tree_depth` (1-30; 1 = stump) to one variable per iteration
and selects the variable with the smallest RSS. It is p2a's own learner, not mboost's ctree-based `btree`.

Fast mboost (`ml::mboost_fast`) stops early on a seeded validation split when `early_stopping_rounds > 0`
(default 0) and predicts with the training means.

`center = false` caveat: the slope is then `sum x (u - ubar) / sum x^2`, which is neither `bols` with nor without an
intercept, so the pins below hold for `center = true` (the default) only.

## R references

`validation/r/cases/ml_boosting_t3.R`, data `generated/mboost_sim.csv` (n = 200):

- `ml_mboost_pkg`: the installed package, `mboost::mboost(y ~ bols(x1) + bols(x2) + bols(x3), family,
  boost_control(mstop = 100, nu = 0.1))` (2.9-14): offset, coefficients, intercept, selections and fitted
  responses for all four families.
- `ml_mboost_bols`: the same family definitions written out in R (agreeing with the package to 4e-15), for the first
  step and the 2000-step paths too, and the converged limits from `stats::lm` and `stats::glm`
  (`Binomial()` at glm / 2). Its tree-learner values are p2a's learner written out in R: a self-consistency check, not
  R-pinned.

## Decisive values

| Quantity | Value |
|---|---|
| Gaussian coefficients after 100 steps | 1.9820924, 0.4086268, 0.0145259 |
| `Binomial()` offset | 0.0300090 |
| `Binomial(type = "glm")` offset | 0.0300023 |
| Selections after 100 steps, `Binomial()` | 38, 37, 25 |
| Selections after 100 steps, `Binomial(type = "glm")` | 47, 53, 0 |
| Tree learner, depth 2, selections after 30 steps | 23, 7, 0 |

## Tests

Registered as `ml_mboost` (`ml/mboost.rs`): `test_validate_mboost_package_paths`,
`test_validate_mboost_gaussian_path`, `test_validate_mboost_binomial_path`,
`test_validate_mboost_binomial_glm_path`, `test_validate_mboost_poisson_path`,
`test_validate_mboost_converged_limits`, `test_validate_mboost_tree_depth`.

Fast mboost's early stopping, seed and training means have property tests in `ml/mboost_fast.rs`.
`crates/p2a-core/tests/validation_boosting_methods.rs` holds DGP-recovery checks, not R comparisons.

## Tolerances

100-step paths and tree fits: `Tol::coef(200)` (1e-8 relative for n = 200); 2000-step paths and limits: 1e-6 relative
(iterative); selections exact.

## Wall-clock

`release`, `ci` features, against `fe13b889`: one fit per process, nine rounds, base and this tree
interleaved per row with the order alternating, one P core (`taskset`), `RAYON_NUM_THREADS=1`, 2026-11-08,
load average 1.5-4.2 (falling from a finished build; the fits run on their own core). Data: n x 5 uniform
design, y = 2 x1 + 0.5 x2 + noise. Cells are medians; "ratio" is this tree over the base, "min ratio" the
ratio of the minima.

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| mboost, Gaussian, `bols` (default) | 100 | 0.148 | 0.130 | 0.88 | 1.02 |
|  | 1,000 | 1.06 | 1.05 | 0.99 | 1.00 |
|  | 10,000 | 10.52 | 10.49 | 1.00 | 1.01 |
| mboost, binomial | 1,000 | 3.42 | 3.38 | 0.99 | 0.99 |
| mboost, tree learner, depth 1 | 1,000 | 2.67 | 2.75 | 1.03 | 1.03 |
|  | 10,000 | 27.75 | 28.57 | 1.03 | 1.03 |
| fast mboost, defaults | 100 | 1.54 | 1.89 | 1.23 | 0.99 |
|  | 1,000 | 3.22 | 3.20 | 0.99 | 0.98 |
|  | 10,000 | 15.03 | 15.04 | 1.00 | 0.99 |

Within 1.1x. Fast mboost at n = 100: single fits fall into two modes on both sides, about 0.65 ms and 1.9 ms;
the minima agree (0.99) and the median of the per-round ratios is 1.02, while the median of the medians (1.23)
counts more slow-mode rounds on this tree's side. The binomial family's definition changed (half log-odds), with
the same work per step.

## Not pinned

The tree learner against mboost (`btree` is a different learner), `mboost_cv`'s folds. For gbm, see
`tree_importance.md`.
