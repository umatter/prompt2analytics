# Smoothing spline (`regression::smooth_spline`)

## Method

R's `smooth.spline` algorithm, written in Rust: x scaled to [0, 1] by `(x - min) / range`;
x values within `1e-6 * IQR(x)` of each other merged (weights summed, `ybar` the weighted mean);
weights rescaled to sum to the number of positive weights; knots at every distinct x up to 49
distinct values, else `.nknots.smspl`'s count at R's positions (`all_knots` keeps every distinct
x); cubic B-splines; the penalty `Omega = integral B_i'' B_j''` with `sgram.f`'s 0.333 for 1/3;
`lambda = ratio * 256^(3 spar - 1)` with `ratio = tr(X'WX) / tr(Omega)` over the inner
coefficients; a banded Cholesky solve of `(X'WX + lambda Omega) c = X'Wy` with the exact
leverages `w_i b_i' (X'WX + lambda Omega)^-1 b_i` from the band of the inverse; and `sbart.c`'s
Brent search on spar in [-1.5, 1.5] (tolerance 1e-4, eps 2e-8, 500 steps) for GCV, ordinary CV
or a df target (`3 + (df - target)^2`). As R, the search reports the best spar with the lambda,
df, fit and leverages of its last evaluation. O(n) time and memory.

Precedence: `df` > `spar` > `lambda` > the criterion (GCV by default, as R's `cv = FALSE`; CV
with `criterion = Cv` or R's `cv = true`). A given lambda reports the spar it corresponds to.
`cv_crit` is R's on the observations: CV when CV selected spar, else GCV. A df target the search
misses by more than `DF_TARGET_TOL` (1e-2) is a warning. `smooth_spline_predict` evaluates the
stored B-spline coefficients (`fit`), linearly beyond the data range with the end slopes, as R's
`predict.smooth.spline.fit`.

Inputs: df in (1, number of distinct x], spar and lambda finite (lambda non-negative), weights
finite and non-negative, not all zero; at least 4 distinct x.

### Criterion semantics (REG-53)

| Request | R | p2a |
|---|---|---|
| nothing given | `smooth.spline(x, y)` = `cv = FALSE`: GCV | `SmoothSplineConfig::default()`: GCV |
| GCV | `cv = FALSE` | `criterion = SplineCriterion::Gcv` |
| ordinary leave-one-out CV | `cv = TRUE` | `criterion = SplineCriterion::Cv`, or `cv = true` (R's meaning; overrides `criterion`) |

`with_cv` is deprecated (`with_criterion`): its argument has the opposite sense of R's `cv`.

## R reference

- Case `smooth_spline_t4` (`validation/r/cases/regression_smoothers_t4.R`): data A (n = 120, 69
  knots), B (150 rows, weights, 81 distinct x), C (x = 1:30). Keys `a_*`, `b_*`, `c_*` are R's
  algorithm written out in R (`sbart`/`sslvrg`/`sgram` with 0.333 and the last evaluation);
  `r_*` keys are `smooth.spline` itself. The two agree to about 1.4e-9.
- Case `smoothers_gcv_periodic_loess` (`regression_smoothers_t3.R`): `smooth.spline(cv = FALSE)`,
  `cv = TRUE`, `df = 6` on x = sorted `runif(100)`.

## Tolerances

A given spar or lambda: the coefficient row of `validation/README.md` (`Tol::coef(120)`).
Searched fits: `Tol::iterative(1e-6)` (Brent stops at 1e-4 in spar; both sides run the same
search, so the pins hold to rounding).

## Tests

- `test_validate_smooth_spline_spar_fixed`: knots, lambda, df, fit, leverages, GCV at spar 0.5;
  lambda 1e-4; all knots; predictions inside and outside the range.
- `test_validate_smooth_spline_searched`: df 6, GCV, CV on A; GCV and df 5 with weights and ties
  on B (merged x, residuals); df 6 on C.
- `test_validate_smooth_spline_gcv_cv_t3`: GCV, CV and df 6 of the t3 case.
- Properties: the B-spline basis against de Boor's recursion, the penalty against Gauss-Legendre
  quadrature (exact 1/3), the band inverse against a dense inverse, Brent on a quadratic,
  `.nknots.smspl` against R's values, scale invariance, ties merged at the IQR tolerance,
  cancellation, linear memory at large n, the criterion option table and the input checks.

## Deviations

None. Release note: every smooth.spline number changed (the fit, df, lambda on R's scale,
`n_knots`, `cv_crit`, predictions).

## NA handling

`run_smooth_spline` refuses a missing x or y (`NullValues`), as R's smooth.spline; see
[loess.md](loess.md), "Column Types and Missing Values".
