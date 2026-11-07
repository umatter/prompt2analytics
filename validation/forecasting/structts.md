# Structural time series (StructTS)

Status: **R-pinned** to `stats::StructTS` (`validation/methods/ts_statespace.json`, method
`forecasting_structts`, case `structts_nile_ukgas`, `validation/r/cases/ts_statespace.R`). Recompute with
`Rscript validation/r/run.R --check structts_nile_ukgas`; rewrite with
`Rscript validation/r/run.R --write structts_nile_ukgas`.

`p2a_core::forecasting::structts::struct_ts` fits the local level (`Level`), local linear trend (`Trend`) and
basic structural model (`BSM`) by Gaussian maximum likelihood through the Kalman filter
(`validation/forecasting/kalman.md`).

## What p2a computes, step by step

| Step | p2a | R |
|---|---|---|
| Scale | `vx = var(y)/100` over the non-missing values | `vx <- var(x, na.rm = TRUE)/100` |
| Prior | `a_0 = (y_1, 0, ...)`; `P_0 = 1e6 vx` in **every** entry, a matrix of rank one | `Z$a[1] <- x[1]`, `Z$P[] <- 1e+06 * vx` |
| State layout | level, [slope], [seasonal `s-1` states]; `m = 1 + slope + (s-1)` | `makeLevel`, `makeTrend`, `makeBSM` |
| Objective | `f = 0.5 (Σ v²/F + Σ ln F) / n_obs`, over the observed values, with every observation in it | `getLike` through `C_KalmanLike` |
| Parameters | variances in units of `vx`, starting at 1; at least 0 | `init[mask] / vx`, `lower = 0` |
| `fixed` | absolute variances in R's order: (level, epsilon), (level, slope, epsilon), (level, slope, seas, epsilon) | the same |
| All variances 0 | `f = 1000`, gradient 0 | `getLike`: `if(all(p == 0)) return(1000)` |
| Optimiser | projected Newton on `p >= 0` with the exact gradient (the differentiated filter) | L-BFGS-B with a finite-difference gradient |
| Log-likelihood | `-n_obs f - 0.5 n_obs ln 2π`, with `n_obs` the observed values | `-length(y) f - 0.5 n_obs ln 2π`: see "Series with gaps" |
| `filtered_level`, `filtered_slope`, `filtered_seasonal` | filtered states | `fitted(fit)` |
| `level`, `slope`, `seasonal` | smoothed states | `tsSmooth(fit)` |
| `standardized_residuals` | `v_t / sqrt(F_t)`, NaN at a missing value | `residuals(fit)` |

AIC and BIC count the free variances; BIC uses the observed values. R reports neither.

**Every observation stays in the likelihood.** The written-out recursion with no threshold reproduces
`C_KalmanLike` exactly for Nile and to `3.6e-11` for the raw UKgas BSM. A rule that drops an observation by the
size of its gain gives NaN on Nile and `0.96048893` where C gives `0.95159552` on UKgas, so p2a has no such rule.
The first observation therefore adds a large `ln F` and a small `v²/F`; its standardised residual is 0.

**Why the tight optimum, not R's stopping point.** R's L-BFGS-B stops on a relative change in `f` of about
`2e-9` with a finite-difference gradient, short of the optimum of its own objective. Its stopping point cannot be
reproduced by another optimiser and is not a property of the model. The reference is the optimum of the same
objective, found in the R case by Newton's method on the exact gradient, written out and checked against
`stats::KalmanLike`, with the Karush-Kuhn-Tucker sign checked at every bound (`*_tight` keys); R's default fit stays
pinned as a cross-check at its measured gap.

**Why the optimiser is what it is.** Nelder-Mead on log variances could not reach a variance of 0, which is
where the Nile trend slope and the UKgas level sit. The projected Newton search holds a bound with
`A = {i : x_i = 0, g_i >= 0}`, takes the Hessian of the rest from forward differences of the exact gradient, shifts it
when it is not positive definite, and accepts a projected step on the Armijo condition.

## Guards

- `n >= 10`; for BSM, `period >= 2` and `n >= 2 period` (at most half the series).
- `max_iter >= 1`; `tolerance` finite and positive.
- `fixed` of R's length, each value finite and at least 0, not all of them set.
- The first value must not be missing: R stops there too (`the first value of the time series must not be
  missing`), so its branches for a leading NA are dead code. An infinite value is refused; NaN is a missing value.
- A constant series is `InvalidSpecification` (there is no variance to estimate).

## Pinned values

`test_validate_structts_nile`, `test_validate_structts_nile_trend`, `test_validate_structts_ukgas`,
`test_validate_structts_fixed`, `test_validate_structts_missing`, `test_validate_structts_fixed_changes_the_fit` and
`test_validate_structts_scale_invariant` load the case.

| Fit | Variances at the tight optimum | Log-likelihood | Gap of R's default fit (relative) |
|---|---|---|---|
| Nile, level | level 1469.17100, epsilon 15098.52464 | -643.20099 | 1.66e-5 |
| Nile, trend | level 1426.84359, slope 0, epsilon 15047.40747 | -645.80743 | 7.55e-5 |
| UKgas (`log10`), BSM period 4 | level 0, slope 1.73040e-5, seas 7.13574e-4, epsilon 3.68206e-4 | 153.99930 | 1.50e-3 |
| Nile, level fixed at 1500 | level 1500, epsilon 15052.64834 | -643.20127 | - |
| Nile with gaps (rows 21-25, 61) | level 523.90653, epsilon 16798.60704 | -603.99846 (observed-data) | - |

Tolerances: tight variances `Tol::iterative(1e-6)`; log-likelihoods `Tol::stat(n)`; states and standardised
residuals, which cross zero, relative `1e-6` with absolute `1e-8`; R's default variances `1e-4` for Nile and `2e-3`
for UKgas, its measured gaps. Variances on their bound are exactly 0, as in R.

### Series with gaps

R's `loglik` is `-length(y) f - 0.5 n_obs ln 2π`, which scales the data term by `length(y)/n_obs` when values are
missing. p2a reports the Gaussian log-likelihood of the observed values, `-n_obs f - 0.5 n_obs ln 2π`, pinned to
`nile_na_loglik_obs_tight`, computed in the R case from the same objective. This is the registry's deviation
for `nile_na_loglik` and `nile_na_loglik_tight`. The estimates, the objective and every series without gaps are
unaffected: with no missing value the two formulas agree.

## Convergence

The stopping criterion is the scaled projected gradient `max_{i free} |g_i| max(x_i, 1) <= tolerance` (default
`1e-8`). `x` is in units of `vx`, so a rescaled series follows the same path and `f` only shifts by a constant
(`test_validate_structts_scale_invariant`, factors `1e-6` and `1e6`). The search also stops as converged when a step changes
`f` by at most `1e-14` and `x` by at most `1e-12`, or when no step is accepted and the Newton step predicts a reduction
of at most `1e-14 max(|f|, 1)`: no further reduction is possible in double precision.

A stop at `max_iter` is `converged = false`, recorded in `convergence` and in `warnings` as "StructTS did not
converge after N iterations (...)" (`test_structts_reports_iteration_limit`). A line search that cannot reduce the
objective adds "StructTS: the line search could not reduce the objective further; the estimates may not be at the
optimum". A stopped budget or an allocation refusal stops the search at once; a failed filter pass in a trial
only halves the step (`minimize_nonnegative_treats_only_a_failed_filter_as_a_failed_trial`).

## Performance

Regenerate with `cargo bench -p p2a-core --bench forecasting_benchmarks -- StructTS`.

## References

- Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge.
- Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford.
- R Core Team. `stats::StructTS`, `src/library/stats/R/StructTS.R` and `src/library/ts/src/arima.c`.
