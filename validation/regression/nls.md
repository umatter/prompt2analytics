# Nonlinear least squares (`regression::nls`)

## Method

`nls()` (one regressor), `nls_multi()` and `run_nls()` minimise `RSS(theta) = sum (y_i - f(x_i,
theta))^2` by Levenberg-Marquardt (default) or Gauss-Newton. Standard errors are
`sigma^2 (J'J)^-1` with a forward-difference Jacobian (step 1e-7) at the solution, `sigma^2 = RSS /
(n - p)`, t with `n - p` degrees of freedom.

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
change in RSS. Coefficients `Tol::iterative(1e-6)`, SEs and t `1e-5` (forward-difference
Jacobian), p-values `1e-3`, RSS and sigma `1e-8`, df exact.

## Tests

`test_validate_nls_expdecay`, `test_validate_nls_michaelis_menten` (registry entry `nls` in
`validation/methods/regression_smoothers_t3.json`). Both pass at `NlsConfig::default()`: nls's
unweighted fit has no defect these pins found.

## Deviations

None in the pinned quantities.
