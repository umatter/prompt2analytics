# Sensitivity to unmeasured confounding (`regression::sensemakr`)

## Method

Cinelli & Hazlett (2020): partial R^2 of the treatment with the outcome `t^2 / (t^2 + dof)`, the
robustness value `RV_q = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2` with `f_q = q |t| / sqrt(dof)`, and
benchmark bounds from the partial R^2 of a covariate with the treatment and the outcome.
`sensemakr()` works from an `OlsResult`; `run_sensemakr()` from a dataset, with the exact benchmark
partial R^2 of the treatment on each benchmark covariate (auxiliary regressions by pivoted QR; a
benchmark collinear with the other covariates is an error naming it).

## R reference

`sensemakr` 0.1.6, case `sensemakr_darfur`: `sensemakr(lm(peacefactor ~ directlyharmed + age +
farmer_dar + herder_dar + pastvoted + hhsize_darfur + female, data = darfur), treatment =
"directlyharmed", benchmark_covariates = "female", kd = 1)` (no village effects; n = 1276, dof 1268).

## Values

estimate 0.04888549597934379, se 0.01839185800844235, t 2.6579965959341383, dof 1268, `r2yd_x`
0.005540851802641748, `rv_q` 0.07191008791292285, `rv_q05` 0.036632022814959704, `rv_qa`
0.019338593626010935; benchmark female: `female_r2dxj` 0.002574930988431467, `female_r2yxj`
0.11546024184356445.

## Tolerances

`Tol::coef(n)` for the estimate, `Tol::se(n)`, `Tol::stat(n)` for t, partial R^2 and RV, dof exact;
`partial_r2(t, dof)` to 1e-14.

## Tests

`test_validate_sensemakr_darfur` (the formulas on the classical OLS t),
`test_validate_sensemakr_darfur_benchmark` (X-16: the exact benchmark partial R^2 of the treatment on
female; a revert to `0.5 * R^2_{Y~X_j}` fails it). Registry entry `sensemakr`. The old R test
asserted RV in 0.10-0.20 at a dof of 1276, which is not the
dataset's, and is deleted. `test_sensemakr_benchmark_collinear_is_an_error` checks the collinearity
error.

## Deviations

- REG-28: `rv_qa` (R uses `qt(alpha/2, dof - 1) / sqrt(dof - 1)` and the extreme-RV branch; p2a gives
  0.019359950325335992). `run_sensemakr`'s se, t, partial R^2, RV and `r2_yd_benchmark` use the HC1
  t (R: classical; `female_r2yxj` p2a 0.1204729846).
- REG-27: `bound_r2dz_x`, `bound_r2yz_dx`, `bound_adjusted_*` (bound formulas, adjusted SE and CI).
