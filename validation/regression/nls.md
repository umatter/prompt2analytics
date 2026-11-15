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

Quiet re-time, 2026-11-15 (this tree at `2d0bb8b6`): release builds, `--no-default-features --features
ci`; one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), cwd `/dev/shm`; base f80b0131 and this tree
interleaved per row with the order rotated per round, each process the median of its calls after a
warm-up (calls calibrated to about 0.8 s per process); cells are the median over five rounds (nine or ten
for rows within 1.05-1.15x), the ratio of the minima in brackets; 1-minute load average 1.0-3.0.
Synthetic data: x evenly spaced on [0, 10] with a small
jitter, y = 3 sin(0.7 x) + noise.

`y = 10 exp(-0.5 x) + 2 + noise` on x in [0, 8], start (8, 0.3, 1).

| Fit | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|
| unweighted (the same arithmetic) | 129 µs -> 129 µs (1.01x [1.01]) | 1.11 ms -> 1.11 ms (1.00x [1.00]) | 12.2 ms -> 12.1 ms (0.99x [0.99]) |
| weighted | 206 µs -> 170 µs (0.83x [0.82]) | 1.53 ms -> 1.15 ms (0.75x [0.74]) | 13.9 ms -> 12.8 ms (0.92x [0.92]) |

The weighted fit now takes weighted steps and converges in fewer iterations.
