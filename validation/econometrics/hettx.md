# hettx: effect summary and regression imputation

**Status:** R-pinned (registry entry `hettx_effect_summary_and_imputation` in
`validation/methods/numeric_hygiene.json`).

R's `hettx` package is not installed in the validation environment, so the method as a whole is not
pinned to it. Two of its building blocks are pinned to base R instead: the summary of the estimated
individual effects (and the IQR statistic of the permutation test) to `stats::quantile(type = 7)` and
`stats::IQR`, and the regression imputation of the outcome model on one arm to `stats::lm` and
`predict.lm`. Cases: `validation/r/cases/numeric_hettx.R`.

## Effect summary and IQR statistic (`qtl_hettx_summary`)

p2a used to read the quantiles of the effect vector at a sorted index (`s[n / 4]`, `s[n / 2]`, ...),
which is none of R's nine quantile types. The summary and the IQR statistic now use
`stats::robust::quantile_sorted` with type 7, R's default.

R call: `quantile(tau, c(0.1, 0.25, 0.5, 0.75, 0.9)); IQR(tau)` on 38 effects with ties, chosen so that
`(n - 1) p` is fractional at 0.25, 0.5 and 0.75.

| Quantity | R (type 7) | old index rule |
|---|---|---|
| p10 | -1.745 | -2.27 |
| p25 | -0.765 | -0.81 |
| median | 0.675 | 0.76 |
| p75 | 2.1875 | 2.25 |
| p90 | 4.11 | 4.53 |
| IQR | 2.9525 | 3.06 |

Min (-3.14), max (6.96) and sd (2.3656879283013965) are pinned too.
Test: `test_validate_hettx_effect_summary_type7`.

## Outcome model on an aliased arm (`inv_hettx_outcome_model`)

Regression imputation fits `y ~ x` on one arm and predicts every row. When the arm's design is
rank-deficient, p2a used to solve the normal equations with a generalized inverse, which splits a
coefficient between collinear columns and gives predictions that differ from R's. It now solves with
`linalg::solve::least_squares` (the normal equations first, the pivoted QR when they are singular), sets
an aliased coefficient to NA as `lm` does, predicts with it at 0 as `predict.lm` does on a rank-deficient
fit, and records R's warning "Coefficients not defined because of singularities: x2".

R call: `lm(y ~ x1 + x2, subset = treat == 1); predict(fit, newdata = d)` with `x2 = 2 x1 + 1` among the
treated. Coefficients: intercept 1.5604117287238273, x1 -0.11117833449336709, x2 NA; all 40 predictions
and the R² are pinned. Test: `test_validate_hettx_outcome_model_aliased_subset`.

Tolerances: `validation/README.md` for n = 20-40.
