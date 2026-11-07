# Autoregressive models

`ar`, `run_ar` and `run_ar_with_order` in `crates/p2a-core/src/forecasting/ar.rs`.
R reference case: `ts_filter_ar_cpt` (`validation/r/cases/ts_utils.R`), series `x` (n = 60) and `s` (n = 24) of
`validation/datasets/generated/ts_utils.csv`.

## OLS

`ArMethod::Ols` is `ar(method = "ols", intercept = FALSE)` (`stats:::ar.ols`). R's rescaling by the standard deviation
cancels in the coefficients, the rank decision and the relative AIC, so p2a omits it.

| Step | Rule |
|---|---|
| Orders | 0..=order_max with AIC, otherwise the fixed order or order_max |
| Stop | at the first order m with fewer rows than lags (n - m < m), or whose `qr(X'X)` is rank-deficient at R's tolerance 1e-7; the warning is R's "singularities ... only valid up to model order m - 1" |
| Prediction variance | SSE / (n - m) |
| AIC | n ln(var) + 2m, relative to its minimum; `+inf` for orders not fitted |
| Unestimable fixed order | `InsufficientData` (rows < lags) or `SingularMatrix`, both naming "AR(m) by OLS" |

**Factorisation.** One lag design and one X'X per order. The normal equations solve it when `normal_inverse_from_gram`
certifies the conditioning; otherwise a pivoted QR of X does, and always for a saturated order. X'X also takes the rank
test with R's rule.

**Saturated order.** With rows = lags, as for m = 12 on `s`, the fit is exact. The SSE is exactly 0, so that order's AIC
is -inf, the relative AIC is 0 for it and `+inf` for the others, and it is selected as in R, with a second warning. R reports
var.pred 3.365475531705031e-30 and finite relative AICs (1611.6, ...), which are the rounding of `solve(X'X)` and not
comparable across implementations. So `ar_ols_var` and `ar_ols_aic` are recorded, not compared: the test pins the order,
the coefficients and the infinite orders, and `test_validate_ar_ols_aic_by_order` pins the AIC differences on the
call with `order.max = 11`.

**Convention kept as p2a's.** p2a fits no intercept. R's default `ar.ols(intercept = demean)` fits a constant after
demeaning, so R's default `ar(x, method = "ols")` differs; the pins call R with `intercept = FALSE`.

## Yule-Walker

`ArMethod::YuleWalker` is `ar.yw`: Durbin-Levinson on the sample autocovariances. var.pred is the Durbin-Levinson variance
times n / (n - (order + 1)). An order with n <= order + 1 is `InsufficientData` (R returns an infinite or negative value).

## Burg

`ArMethod::Burg` is `ar.burg` with `var.method = 1`. Each order's reflection coefficient and errors use the previous order's
forward and backward errors, as R's `C_Burg`. The variance is `var[p - 1] (1 - k^2)`, with no n / (n - p - 1) factor.

## Budget

Every route checks the budget once per order (`ar_ols_and_burg_honour_cancellation`).

## Pins

| Test | R call | Pinned at |
|---|---|---|
| `test_validate_ar_ols_short_series` | `ar(s, method = "ols", demean = TRUE, intercept = FALSE)` | order 12, exact; coefficients `Tol::coef(24)`; order 13 infinite |
| `test_validate_ar_ols_aic_by_order` | the same, differences over orders 0..11, p2a with `order_max = 11` | `Tol::stat(24)` |
| `test_validate_ar_ols_n60` | `ar(x, method = "ols", demean = TRUE, intercept = FALSE)` | order exact; coefficient, mean `Tol::coef(60)`; AIC, var `Tol::stat(60)` |
| `test_validate_ar_ols_fixed_order` | `ar(s, method = "ols", aic = FALSE, order.max = 2 | 3)` | coefficients `Tol::coef(24)`, var `Tol::stat(24)` |
| `test_validate_ar_yule_walker_var_pred` | `ar(x, method = "yule-walker")`, and fixed order 3 | order exact; coefficients `Tol::coef(60)`, var `Tol::stat(60)` |
| `test_validate_ar_burg` | `ar(x, method = "burg")`, and fixed order 3 | order exact; coefficients and partial ACF `Tol::coef(60)`; var and AIC `Tol::stat(60)` |

`test_ar_ols_unestimable_fixed_order_is_an_error` and `test_ar_ols_rank_stop_keeps_the_estimable_orders` hold the refusal and
the stop; they compare no R number.

The `test_ar_*_on_ar2_dgp` tests and `test_ar2_coefficients_recover_dgp` are properties on a known process, not R pins.
