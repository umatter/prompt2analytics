# Sensitivity to unmeasured confounding (`regression::sensemakr`)

## Method

Cinelli & Hazlett (2020): partial R^2 of the treatment with the outcome `t^2 / (t^2 + dof)`, the
robustness value `RV_q = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2` with `f_q = q |t| / sqrt(dof)`, and
benchmark bounds from the partial R^2 of a covariate with the treatment and the outcome.
`sensemakr()` works from an `OlsResult`; `run_sensemakr()` from a dataset, with the exact benchmark
partial R^2 of the treatment on each benchmark covariate (auxiliary regressions by pivoted QR; a
benchmark collinear with the other covariates is an error naming it). Both use the classical OLS fit
(`CovarianceType::Standard`; another covariance type is refused). `RV_{q,alpha}` is sensemakr's
`robustness_value(t, dof, q, alpha)`: `f_c = |qt(alpha/2, dof - 1)| / sqrt(dof - 1)`,
`f_qa = f_q - f_c`, 0 for `f_qa <= 0`, the extreme robustness value `(f_q^2 - f_c^2) / (1 + f_q^2)`
when `f_q > 1 / f_c`, and `2 / (1 + sqrt(1 + 4 / f_qa^2))` otherwise. Benchmark bounds follow
`ovb_partial_r2_bound`: `r2dz.x = kd r2dxj / (1 - r2dxj)` and the implied `r2yz.dx` (an impossible
kd is an error; an `r2yz.dx` above 1 is set to 1 with a warning), the adjusted SE is
`sqrt((1 - r2yz.dx) / (1 - r2dz.x)) se sqrt(dof / (dof - 1))`, the adjusted t is taken against
`h0 = (1 - q) estimate`, and the interval uses `qt(alpha/2, dof)`, as the bounds table of
`sensemakr()`.

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

`test_validate_sensemakr_darfur` (the formulas on the classical OLS t, `rv_qa` included),
`test_validate_sensemakr_darfur_benchmark` (`run_sensemakr`'s se, t, partial R^2, RV and RV_alpha,
and X-16: the exact benchmark partial R^2 of the treatment on female, a revert to
`0.5 * R^2_{Y~X_j}` fails it; `female_r2yxj`), `test_validate_sensemakr_rv_alpha` (case
`sensemakr_rv_alpha`: six inputs of `robustness_value(t, dof, q, alpha)`, row 1 in the extreme
branch; 1e-10 relative, because p2a's and R's t quantiles differ by about 5e-13 and `f_q - f_c`
amplifies that). `test_sensemakr_refuses_robust_ols` checks the refusal of an HC1 fit. The darfur benchmark test
also pins the female bound (`bound_*`). Registry entry `sensemakr`. The old R test
asserted RV in 0.10-0.20 at a dof of 1276, which is not the
dataset's, and is deleted. `test_sensemakr_benchmark_collinear_is_an_error` checks the collinearity
error.

## Deviations

None.

## Timing

The benchmark partial R² takes OLS's solve decision (normal equations on a full-rank,
well-conditioned X_{-j}, the pivoted QR otherwise) for both auxiliary regressions; new / base
against `ff8e7559` and the tests are in `gls.md` ("Normal equations at a given rho").
