# GARCH(p, q)

`garch(data, Some(GarchConfig))`, `run_garch(data, p, q, include_mean)` and `garch_forecast(&result, h)`
in `crates/p2a-core/src/forecasting/garch.rs`. MCP `ts_garch_fit`; CLI `p2a ts garch`.

## Model

r_t = mu + e_t, where mu = 0 when `include_mean` is false. The variance follows
h_t = omega + sum_{i=1..p} alpha_i e_{t-i}^2 + sum_{j=1..q} beta_j h_{t-j}, with e_t = sqrt(h_t) z_t.
z_t is N(0, 1) (`distribution = norm`) or fGarch's standardised Student-t with shape nu > 2
(`distribution = std`). p is the ARCH order (>= 1) and q the GARCH order (>= 0); both are at most 20,
and n must exceed the parameter count.

## Estimator

Conditional maximum likelihood with fGarch's "mci" initialisation: h_t = mean(e^2) for t <= max(p, q),
and the sum runs over all n observations.

| Step | Rule |
|---|---|
| Parameters | mu = xbar + s theta; omega = s^2 exp(theta); (alpha, beta) by a positive softmax, so alpha + beta < 1; shape = 2 + 198 logistic(theta) (at most 200) |
| Gradient | analytic, from the recursions for dh_t / dparameter |
| Optimiser | BFGS with a strong-Wolfe line search (Nocedal & Wright 2006, Alg. 3.5 and 6.1). B0 is the inverse finite-difference Hessian at the start. |
| Starts | (sum alpha, sum beta) = (0.1, 0.8), (0.05, 0.05) and (0.3, 0.6), with omega = s^2 (1 - sum) and shape 8. The highest log-likelihood wins. |
| Polish | up to 3 Newton steps in the natural parameters, on the winning start only. A step is kept only if it is feasible, logLik does not fall and the criterion falls. |
| Convergence | max_i \|d(loglik/n)/d theta_i\| <= `tolerance` (default 1e-8). The criterion is scale free: data times 1e-6 or 1e6 converge in the same number of iterations. Stopping at `max_iter` (500 BFGS iterations per start) gives a warning. |
| SEs | sqrt(diag(I^-1)), where I is the negative Hessian of the log-likelihood in the natural parameters, from central differences of the analytic gradient. If I is not positive definite, the SEs are NaN with a warning. |
| Bounds | alpha_i or beta_j at or below 1e-4 of alpha + beta, or a shape above 199.8, is on its bound. It is held fixed for the SEs (its own SE is NaN) and gets a warning. |
| Tests | z = estimate / SE (the `t_stats` field holds these z values); p = 2 Phi(-\|z\|), as in fGarch and tseries |
| Forecasts | sigma2_{T+k} = omega + sum alpha_i E[e^2_{T+k-i}] + sum beta_j sigma2_{T+k-j} |

Errors:
- `InsufficientData`: fewer than 20 observations, or n <= parameters.
- `InvalidSpecification`: p = 0, an order above 20, `max_iter` = 0, a tolerance that is not finite and
  positive, or a constant series.
- `NullValues` / `NonFiniteValues`: NaN or infinite data.

## R reference

fGarch and rugarch are not installed, so the likelihood is written out in `validation/r/cases/ts_garch.R`
(case `garch_normal_t`). The data are `validation/datasets/generated/garch_returns.csv`: three simulated
series with n = 1500 and seed 20261105.
- `r_norm`: GARCH(1,1) with omega 2e-6, alpha 0.08, beta 0.90 and mu 5e-4, normal errors.
- `r_t`: the same process with t(6) errors.
- `r_g21`: GARCH(2,1) with alpha (0.05, 0.06) and beta 0.85.
- `r_norm_dm`: `r_norm` demeaned.

The pinned values are Newton iterations on the analytic gradient, with numDeriv::jacobian (Richardson)
Hessians. The SEs are those of the inverse observed information. The design's optim fits (Appendix
A.31) are kept as cross-checks: they stop 1.7e-6 short of the optimum, and their second-difference
Hessian puts SE(mu) 1.3 % off. tseries::garch checks the likelihood. Its objective equals the
written-out one at its own estimates, to 1e-9. From its default start it stops early ("FALSE
CONVERGENCE", logLik 4843.03 against 4854.14), so it is not a reference for the estimates.
R 4.3.3, tseries 0.10.63, numDeriv 2016.8.1.1.

| Fit | mu | omega | alpha | beta | shape | logLik |
|---|---|---|---|---|---|---|
| (1,1) normal, `r_norm` | 2.796397001e-4 | 1.509863051e-6 | 0.07288109558 | 0.9126905023 | | 4854.142437443824 |
| (1,1) std, `r_t` | 4.028463510e-4 | 3.245957954e-6 | 0.08758740782 | 0.8779744457 | 5.027807762 | 5063.6496551278715 |
| (1,1) normal, `r_t` | 3.940055487e-4 | 2.954109948e-6 | 0.09200725367 | 0.8747706628 | | 5015.932743004753 |
| (1,2) normal, `r_norm` | 2.833395217e-4 | 1.723879698e-6 | 0.08318114407 | 0.7506650534, 0.1495858555 | | 4854.4259677210475 |
| (2,1) normal, `r_g21` | 4.144647784e-4 | 1.745405302e-6 | 0.03579720401, 0.07742795168 | 0.8450000267 | | 5523.02468968211 |
| (2,1) normal, `r_norm` (alpha2 = 0 bound) | 2.814289984e-4 | 1.512973839e-6 | 0.07293642429, 0 | 0.9125663990 | | 4854.249990521646 |
| ARCH(1), `r_norm` | 3.813700055e-4 | 8.355443633e-5 | 0.1780716712 | | | 4788.42616629871 |
| (1,1) zero mean, `r_norm_dm` | | 1.510057871e-6 | 0.07289143048 | 0.9126786949 | | 4854.141886446746 |

The (1,1) normal fit:
- SEs: 2.246232993e-4, 5.885878152e-7, 0.01321600012, 0.01548639430.
- z: 1.2449, 2.5652, 5.5146 and 58.935.
- p: 0.2132, 0.01031, 3.4955e-8 and 0.
- corr(omega, beta1) = -0.7115.
- Persistence 0.9855715979, unconditional variance 1.046452019e-4, half-life 47.69 periods.
- Forecasts sigma2_{T+1..T+3}: 8.593447744e-5, 8.620444329e-5, 8.647051397e-5.

## Tests and tolerances

| Test (`forecasting::garch::tests::`) | Pins | Tolerance |
|---|---|---|
| `test_validate_garch_normal` | (1,1) parameters, logLik, AIC, BIC, h_1..h_3, h_T, unconditional variance; Appendix optim values | 1e-8 relative; `Tol::stat(1500)`; Appendix 1e-5 |
| `test_validate_garch_student_t`, `test_validate_garch_student_t_shape` | t fit with shape and SEs; normal fit of the t series; the two differ | 1e-8 / 1e-6 relative |
| `test_validate_garch_se`, `test_validate_garch_vcov` | SEs, z, normal p-values, covariance matrix | 1e-6 relative; p 1e-5 + 1e-12 |
| `test_validate_garch_orders` | (1,2), (2,1), ARCH(1), zero mean: parameters, SEs, logLik, AIC, BIC, h_1..h_3 | 1e-8 / 1e-6 |
| `test_validate_garch_bound` | alpha2 = 0 bound: free parameters and SEs, flag, warning | 1e-6 relative |
| `test_validate_garch_forecast` | forecasts of six fits; GARCH(1,1) closed form | 1e-8 relative |
| `test_validate_garch_scale_invariance` | x 1e-6 and x 1e6 converge in the same number of iterations (within 2) to the scaled optimum | 1e-8 |
| `test_validate_garch_display_rows` | table rows per order and `include_mean`, with their own SEs | print precision |
| `test_garch_forecast_recursion_hand`, `test_garch_forecast_needs_the_fit_path`, `test_garch_hessian_not_pd_gives_nan_se`, `test_garch_bound_rule`, `test_garch_reports_non_convergence`, `test_garch_distribution_parses_aliases_and_rejects_unknown`, `test_garch_rejects_n_le_params`, `test_garch_rejects_degenerate_inputs`, `test_run_garch_delegates`, `garch_fit_stops_when_the_budget_is_cancelled`, `garch_likelihood_and_inference_passes_honour_cancellation`, `garch_bfgs_checkpoints_each_iteration`, `garch_forecast_refuses_absurd_horizon` | behaviour (hand values, budgets, no R values; not registered) | exact / 1e-12 |

The tolerances are looser than `validation/README.md`'s n > 1000 row (1e-10 / 1e-8), because this is an
iterative MLE with a finite-difference Hessian. In the R emulation of the Rust algorithm
(`compare.R`, planning) the parameters agree to <= 5e-14 and the SEs to <= 2e-8. Registry:
`validation/methods/ts_garch.json` (id `ts_garch`), which lists the `test_validate_*` tests above.

## p2a conventions

The written-out reference has no bounds of its own; these are p2a's and leave every pinned
quantity unchanged (the case's shape is 5.03 and the bound fit is pinned at R's restricted maximum):

- `BOUND_REL` = 1e-4: a coefficient at or below 1e-4 of alpha + beta is on its bound 0.
- `GARCH_SHAPE_MAX` = 200 bounds the Student-t shape (rugarch uses 100). Above it the standardised
  t is indistinguishable from the normal, and lgamma((nu+1)/2) - lgamma(nu/2) loses digits.
- Where the observed information is not positive definite (checked on the diagonally equilibrated
  matrix, so the check is scale free), the SEs, z values and p-values are NaN with a warning; no
  substitute SE is reported.

## Not implemented

- ARMA mean equations, asymmetric models (GJR, EGARCH, APARCH) and skewed distributions (fGarch's
  "sstd", "ged").
- Robust (QML sandwich) SEs, which fGarch reports alongside the Hessian SEs.

## References

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of
  Econometrics* 31(3), 307-327.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of
  United Kingdom inflation. *Econometrica* 50(4), 987-1007.
- Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., ch. 3 and 6. Springer.
- Würtz, D., Chalabi, Y. and Luksan, L. (2006). Parameter estimation of ARMA models with GARCH/APARCH
  errors: an R and SPlus software implementation. *Journal of Statistical Software* (fGarch).
