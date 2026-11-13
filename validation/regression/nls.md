# Nonlinear least squares (`regression::nls`)

## Method

`nls()` (one regressor), `nls_multi()` and `run_nls()` minimise `RSS(theta) = sum w_i (y_i -
f(x_i, theta))^2` (unit weights by default) by Levenberg-Marquardt (default) or Gauss-Newton.
Standard errors are `sigma^2 (J_w'J_w)^-1` with `J_w = diag(sqrt w) J`, a central-difference
Jacobian at the solution, `sigma^2 = RSS / (n_w - p)` with `n_w` the observations of positive
weight, t with `n_w - p` degrees of freedom, as R's `nls(weights =)`. Residuals are `y - f`
(unweighted), the RSS weighted, as R's `resid`/`deviance`.

Weights must be one per observation, finite and non-negative; bounds one per parameter, either
side alone (`-inf`/`+inf` for an unbounded parameter), lower <= upper; `nls` and `nls_multi`
check both alike (REG-48). With bounds, a parameter at a bound whose step points outward is held
there while the others move (an active set), as `algorithm = "port"` reaches a solution on the
bound. Without weights and bounds the arithmetic is the unweighted fit's, unchanged.

## R reference

`stats::nls` (Gauss-Newton, default control), cases `nls_expdecay` and `nls_mm`
(`validation/r/cases/regression_smoothers_t3.R`), on the data of the old tests:

- `nls(y ~ a * exp(-b * x) + c, start = list(a = 8, b = 0.3, c = 1))`, n = 11;
- `nls(V ~ Vmax * S / (Km + S), start = list(Vmax = 150, Km = 0.05))`, n = 8.

## Values

| case | coefficients | SE | RSS | sigma | df |
|---|---|---|---|---|---|
| `nls_expdecay` | a 9.706801233316728, b 0.512999792647369, c 2.13177718560619 | 0.08301198918772204, 0.012367990386702051, 0.08529685580528593 | 0.03602895019482792 | 0.06710900665598836 | 8 |
| `nls_mm` | Vmax 200.7335463021054, Km 0.10356566295276147 | 1.1639656778633676, 0.0026812382258010452 | 18.91258376330853 | 1.775414682231944 | 6 |

The values the old tests quoted (a = 9.776, RSS 0.198; Vmax = 200.2, Km = 0.102) are not R's.

## Tolerances

Iterative: R stops at a relative offset of 1e-7, p2a's Levenberg-Marquardt at a 1e-8 relative
change in RSS. Coefficients `Tol::iterative(1e-6)`, SEs and t `1e-5` (central-difference
Jacobian), p-values `1e-3`, RSS and sigma `1e-8`, df exact.

## Tests

`test_validate_nls_expdecay`, `test_validate_nls_michaelis_menten` (registry entry `nls` in
`validation/methods/regression_smoothers_t3.json`). Both pass at `NlsConfig::default()`.

Case `nls_weighted_t4` (`validation/r/cases/regression_smoothers_t4.R`; n = 40, variance rising
with x, `w = 1 / (1 + x / 2)^2`): `test_validate_nls_weighted` pins the weighted coefficients,
SEs, t, p, RSS, sigma, df, fitted values and residuals; `test_validate_nls_one_sided_bounds`
pins `nls(algorithm = "port")` with a lower bound alone on c (weighted) and an upper bound alone
on b (unweighted), both optima on the bound (registry entry `nls_weighted`,
`regression_smoothers_t4.json`; same tolerances, RSS change tolerance 1e-12).
`test_nls_multi_rejects_short_weights_and_bounds` covers the input checks. Release note: weighted
fits changed (before, the weights entered only the RSS that accepts a step, not the step or the
covariance), and a single bound is now applied.

## Deviations

None in the pinned quantities.

## Timing (T4)

Release builds, `--no-default-features --features ci`, base f80b0131 against this tree, one
process per measurement pinned to one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), nine
interleaved processes per side, median of the per-process medians; machine load 24-34 on 12
cores (lane timing, preliminary). Synthetic data: x evenly spaced on [0, 10] with a small
jitter, y = 3 sin(0.7 x) + noise.

`y = 10 exp(-0.5 x) + 2 + noise` on x in [0, 8], start (8, 0.3, 1).

| Fit | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|
| unweighted (the same arithmetic) | 125 -> 133 us (1.06x) | 1.05 -> 1.13 ms (1.08x) | 12.0 -> 12.1 ms (1.01x) |
| weighted | 232 -> 144 us (0.62x) | 1.49 -> 1.22 ms (0.82x) | 13.6 -> 13.2 ms (0.97x) |

The weighted fit now takes weighted steps and converges in fewer iterations.
