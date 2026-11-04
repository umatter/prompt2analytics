# Validation: VAR impulse responses (IRF)

Registry: [`validation/methods/ts_var.json`](../../methods/ts_var.json) (method `ts_var_irf`).
R case: `var_canada` in [`validation/r/cases/ts_var.R`](../../r/cases/ts_var.R). The VAR
itself: [var.md](var.md).

## Method

`compute_irf(var_result, steps, orthogonalize)` computes the moving-average matrices of a
fitted VAR(p),

```
Phi_0 = I,   Phi_h = sum_{j=1}^{min(h, p)} A_j Phi_{h-j},
```

and, for `orthogonalize = true`, `Psi_h = Phi_h P` with `P` the lower Cholesky factor of
`sigma_u` (divisor T - Kp - 1), as `vars:::Psi.varest`; for `orthogonalize = false` the
responses are `Phi_h` (unit shocks, `vars:::Phi.varest`). `run_var_irf` fits the VAR and
returns the orthogonalised responses.

**Steps and layout.** `steps` is the number of horizons including h = 0, i.e. vars'
`n.ahead + 1`. `irf[h][response][impulse]`.

**Errors.** `steps == 0` and a `VarResult` whose vectors do not match `n_vars` and `lags`
are `InvalidSpecification` (the message starts "VarResult is inconsistent"); a `sigma_u`
that is not positive definite is `SingularMatrix` for `orthogonalize = true`.

## Reference case

`var_canada`: `irf(VAR(Canada, p = 2, type = "const"), n.ahead = 12, ortho = TRUE | FALSE,
boot = FALSE)` for every impulse (13 horizons x 4 responses x 4 impulses, both kinds).

| Orthogonalised, impulse e | e | prod | rw | U |
|---|---|---|---|---|
| h = 0 | 0.36281501944370154 | -0.020585540581001906 | -0.11603351918236395 | -0.19042004797535295 |
| h = 1 | 0.5475337468458898 | -0.0012009465234944505 | -0.20208313974589998 | -0.32912415302801407 |
| h = 2 | 0.6179181392574051 | 0.01480843588626763 | -0.1802773351180763 | -0.369053587402071 |
| h = 12 | -0.1532799203358315 | -0.3244506552085916 | 0.5529087204934391 | 0.16340467454342417 |

The plain response of e to e at h = 1 is `A_1[e, e]` = 1.6378206022871895; the
orthogonalised response of U to its own impulse at h = 0 is `P[U, U]` = 0.20376704574855273.

## Tests and tolerances

- `test_validate_var_irf_canada`: every orthogonalised and plain response of the case, and
  `run_var_irf` equal to `compute_irf(.., true)`.
- `test_validate_var_irf_display_prints_every_horizon`: the text `ts_var_irf` returns has one
  table per impulse with all 13 horizons, pinned at its 6 printed decimals.
- Not R-pinned: `test_irf_rejects_inconsistent_var_result` (shape check),
  `test_var_irf_shapes_and_decay`, and the resource tests `test_irf_rejects_unallocatable_horizon`
  and `test_irf_stops_when_cancelled`.

Tolerance (validation/README.md, n = 82): 1e-5 relative, 1e-14 absolute; the structural zeros
at h = 0 are exact in both implementations.

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::timeseries::var
```

## History

Before Tier 3 the impact matrix was the Cholesky factor of `U'U / T`, so every orthogonalised
response was too small by `sqrt((T - Kp - 1) / T)` (h = 0 of e: 0.34232590195739032), and the
Display printed horizons 0-4 followed by "... (n more steps)".

## Known differences

No bootstrap intervals (`boot = TRUE`), no cumulative responses, no generalised (Pesaran-Shin)
or structural (SVAR) impulse responses.
