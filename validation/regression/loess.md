# Validation: LOESS (Local Polynomial Regression)

## Method

`regression::loess` fits, at each point x0, a weighted polynomial of degree 0, 1 or 2 on the
q nearest neighbours:

- **Neighbourhood.** q = `min(n, floor(n span + 1e-5))` (R `lowesd`), all n for span >= 1.
  The q nearest points by distance are taken (ties to the left), and every point at the
  q-th distance is added, so a tie group is never split. The window is two-sided; it used
  to slide only right (REG-14) and to use `ceil(span n)` points (REG-45).
- **Radius.** The q-th distance; for span > 1 it is multiplied by sqrt(span). R's
  documentation says span^(1/p); its code (Cleveland's dloess, `ehg127`) multiplies the
  squared radius by span, which is sqrt(span) for one predictor. p2a follows the code
  (verified for spans 1.2, 1.5, 2 and 3; the old factor span was 0.0207 off at span 1.2).
- **Weights.** Tricube in `u = (x - x0) / radius`; points at the q-th distance have weight 0.
- **Local fit.** Weighted least squares on `[1, u, u^2][..=degree]` with lm.fit's relative
  rank rule (`LM_QR_TOL` on the weighted columns), solved through the moment matrix by an
  LDL' factorisation: no absolute cutoff, so the fit does not depend on the scale of x
  (REG-46). A degree that drops for lack of distinct x values is reported in `warnings`.
- **Zero-width neighbourhoods.** When at least q observations share x0 the radius is 0.
  The fit there is the weighted mean of the tied responses, R `lowess`'s rule, with a
  warning naming the smallest span that avoids it. R's `loess` returns 0 there and warns
  "zero-width neighborhood. make span bigger" (decision P2: a documented deviation from a
  degenerate R output). A constant x is an error.
- **Exact fits.** p2a evaluates every local fit, as R's `loess(surface = "direct")`; R's
  default kd-tree interpolation differs from the exact fits by up to about 1e-3 relative on
  these data, so the pins use `surface = "direct"`.
- `enp` is tr(L), R's `trace.hat`; R's "Equivalent Number of Parameters" is
  `one.delta + 2 tr(L) - n` (REG-60, open, with the residual SE and the robust iterations).

## R reference

Cases in `validation/r/cases/regression_smoothers_t3.R`:

- `loess_x10_direct`: x = 1:10, y = (2.5, 3.8, 6.1, 7.9, 10.2, 12.0, 14.3, 16.1, 18.0, 20.2)
  (span n = 7.5, so q = 7):
  ```r
  dir <- loess.control(surface = "direct")
  loess(y ~ x, span = 0.75, degree = 2, control = dir)   # also degree 1; span 0.5 degree 0; span 1.2 degree 2
  predict(f2, data.frame(x = c(2.5, 5.5, 9.5)))
  ```
- `loess_skewed_direct`: lognormal x (`set.seed(20261128)`, n = 100), span 0.75 degree 2 and
  span 0.3 degree 1, `surface = "direct"`.
- `smoothers_gcv_periodic_loess`: `ym ~ xm` with 60 of 100 x values at 0, span 0.5,
  degree 1, `surface = "direct"`; at the ties the expected value is
  `lowess(xm, ym, f = 0.5, iter = 0)$y[1]`.

## Values

| Key | R |
|---|---|
| `fit_s075_d2` | 2.390395787309991, 4.077224371939557, 5.946666803764424, 8.054688091794901, 10.0453119082051, 12.154688091794899, 14.176249526564078, 16.124002238733567, 18.119470241934515, 20.142475836606128 |
| `trace_s075_d2` | 5.47107540343813 (R `enp` 4.950126136174497, not compared: REG-60) |
| `fit_s075_d1` | 2.234268487117532, 4.147675425582901, 6.068700753943249, ... |
| `trace_s075_d1` | 3.4699695365013867 |
| `fit_s05_d0` | 4.166718441080087, 4.5407187084643565, 5.956844741235391, ... |
| `fit_s12_d2` | 2.271706205788957, 4.141896015429392, 6.060655156609892, ... |
| `predict_s075_d2` (2.5, 5.5, 9.5) | 4.989202822729835, 11.100000000000005, 19.126055662623422 |
| `loess_skewed_direct` `fit_s075_d2` | 0.9941107518754562, 0.7014705352666484, 0.5225800999281611, ...; `trace_s075_d2` 6.294163401455551 |
| `loess_skewed_direct` `fit_s03_d1` | 0.9921777665983564, 0.724530082228946, 0.4561462230839579, ...; `trace_s03_d1` 8.282627772986432 |
| `loess_masspoint_expected` | 0.9731694359533333 at the 60 zeros (the tie mean; R's direct and interpolated fits are 0 there), then 1.255397451326043, 1.3096848725151853, 2.840648807854402, ... (R direct) |

The old rule (ceil(span n) points, a right-sliding window, radius times span) was 0.0617 off
at `fit_s075_d2` and up to 0.473 off on the lognormal x.

## Tolerances

`validation/README.md` coefficient row for fitted values and predictions (n = 10: 1e-6;
n = 100: 1e-8), the statistics row for the trace. The fits agree with R to about 2e-15.

## Tests

Registered (`loess_direct` in `validation/methods/regression_smoothers_t3.json`):

- `test_validate_loess_x10_direct`: degrees 0-2, span 0.75, 0.5 and 1.2, the trace and
  predictions; no warnings.
- `test_validate_loess_skewed_direct`: the two-sided neighbourhood on lognormal x.
- `test_validate_loess_mass_point`: the 60% mass point, R's direct fit off the ties and
  lowess's rule at them; the zero-width warning ("60 of 100", q = 50, span >= 0.61) and its
  line in Display.
- `test_validate_loess_scale_invariance`: x times 1e-6 and 1e6 give the R fit (1e-9
  relative), with no warning.

Property tests (not registered): `test_loess_mass_point_no_panic` (tie groups larger than
q at degrees 0-2 and in prediction, a Likert-style mass point, a constant x refused),
`test_loess_rejects_invalid_controls` (span, degree, robust_iterations, tolerance,
non-finite x), `test_loess_local_solver_matches_qr` (the moment solver equals pivoted-QR
least squares on sqrt(w) [1, u, u^2] at 1e-10 relative, with the same effective degree,
for degrees 0-2 and spans 0.1, 0.5 and 1.2).

## Deviations

- REG-60: R's `enp`, residual SE (`one.delta`) and robust iterations (Tier 4).
- Zero-width neighbourhoods: lowess's tie mean instead of R loess's 0 (decision P2).

## Performance

The fit, its complexity and the timings against the pre-Tier 3 engine are in
[Timing (WP11-13)](#timing-wp11-13) below; the older comparison with R's CLOESS described an engine
that no longer exists and was removed.

## Column Types and Missing Values (run_loess and the quantreg, smooth.spline and glmnet wrappers)

The dataset wrappers read their columns through `p2a_core::data::columns`: integer and
boolean columns are numbers, a text column is `NonNumericColumn`, an absent one
`ColumnNotFound`, and missing values (null or NaN) follow the R function's default.

| Function | Missing x or y | R behaviour followed |
|---|---|---|
| `run_loess` | row dropped; `LoessResult::missing`, x then y, and a last `Missing values:` line | R's `loess(y ~ x)` uses `na.omit` (incomplete rows dropped) |
| `quantreg` / `run_quantreg` | row dropped; `QuantRegResult::missing`, y then the regressors, and a `Missing values:` line | `quantreg::rq` (`na.omit`) |
| `run_smooth_spline` | `NullValues` naming the first column (x, then y) | `smooth.spline`: "missing or infinite values in inputs are not allowed" |
| `run_glmnet`, `run_cv_glmnet`, `run_ridge`, `run_lasso` | `NullValues` naming the column | `glmnet`: "missing value where TRUE/FALSE needed" |
| MCP `regression_gls` (y) | `NullValues` naming y | `nlme::gls` (`na.fail`) |

Before this, the four core wrappers read with `.f64()` without a cast (glmnet with a
non-strict cast), so an Int64 column was rejected as non-numeric, a null entered the fit as
0, and a text y became zeros in glmnet. `p2a reg quantreg` and `reg loess` print
`Note: <k> of <n> rows dropped ...` and add a `missing` object to `-F json`.

R: case `reg_smoothers_na` of `validation/r/cases/regression_na.R`
([source](../r/cases/regression_na.R)), 60 rows in
`validation/datasets/generated/smoothers_na.csv` with 2 missing x and 2 missing y; R's
count of complete rows is `n_complete` = 56.

Registered tests (registry entry `regression_wrappers_columns` in
`validation/methods/regression_na.json`): the row deletion of `run_loess` and `quantreg`
is R-pinned, since the number of rows each fit uses is compared with R's `n_complete`
exactly:

- `test_validate_na_loess_columns`, `test_validate_na_quantreg_columns`: `n_complete`; the
  fit on the data with missing cells equals the fit on the 56 complete rows, with the
  reported counts; besides, Int64 equals the same values as Float64 exactly (mtcars
  `hp`), text and absent columns are refused, and with no complete row the error is
  `InsufficientData` naming the method;
- `test_validate_mcp_regression_loess_missing_and_integer`: `n_complete` through MCP
  `regression_loess`, the `missing` object, and an Int64 x;
- `test_validate_cli_reg_loess_quantreg_missing_note`: `n_complete` through `p2a reg
  loess` and `reg quantreg` with `-F json`, the `Note:` line and the `missing` object.

Property tests, not registered as R-pinned (no value is compared with R; they load
`validation/datasets/generated/mtcars.csv`, whose `hp` is Int64, and `smoothers_na.csv`
without a case):

- `test_na_smooth_spline_columns`, `test_na_glmnet_columns`: Int64 equals Float64; a
  missing x or y is `NullValues` naming the column;
- `test_mcp_regression_gls_missing_y_is_error`,
  `test_mcp_regression_smooth_spline_missing_is_error`: the MCP GLS y and the smoothing
  spline refuse missing values;
- `test_cli_reg_integer_columns`, `test_cli_reg_smooth_spline_missing_is_error`: Int64
  columns in `reg quantreg`, `loess`, `smooth-spline`, `line` and `supsmu`, and `reg
  smooth-spline` refusing missing values.

The fits themselves are not compared here: the loess fit is pinned by the `loess_direct`
entry above (REG-14 and REG-45 are fixed), the quantreg SEs by their own entry, and the
smooth.spline fit differs from R (REG-29, REG-30, Tier 4), which no registered test covers.

## References

- Cleveland, W. S. (1979). "Robust locally weighted regression and smoothing scatterplots." *Journal of the American Statistical Association*, 74(368), 829-836.
- Cleveland, W. S., & Devlin, S. J. (1988). "Locally weighted regression: an approach to regression analysis by local fitting." *Journal of the American Statistical Association*, 83(403), 596-610.
- Cleveland, W. S., Grosse, E., & Shyu, W. M. (1992). "Local regression models." Chapter 8 of *Statistical Models in S*, eds J.M. Chambers and T.J. Hastie.
- R Documentation: `stats::loess()` - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loess.html

## Timing (WP11-13)

Release build, CPU 0-3, `RAYON_NUM_THREADS=1`, base (pre-package engine, 7ae513e) and head interleaved,
medians; synthetic noisy data. Head/base: degree 1 span 0.75 (unified): 1.13 / 1.06 / 1.07 at n = 100 /
1,000 / 10,000; degree 2 span 0.5 (LOESS_Gaussian): 1.10 / 0.98 / 0.99; robust (LOESS_Robust): 0.99 /
0.90 / 0.89.
Re-measured on the merged tree against `ff8e7559` (one P core, three interleaved runs each): degree 1
span 0.75 (unified) 0.99 / 1.05 / 1.06 at n = 100 / 1,000 / 10,000; LOESS_Gaussian 1.06 / 0.99 / 0.97;
LOESS_Robust 0.93 / 0.86 / 0.87 (n = 100 / 1,000 / 5,000).
