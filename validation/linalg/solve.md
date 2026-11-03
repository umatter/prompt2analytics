# Least squares, symmetric inverses and the shared GLM (`linalg::solve`, `econometrics::glm`)

## Purpose

One numerical path for the Tier 3 estimators (design rules R9 and R12, decision D1):

- `linalg::solve::{Qr, least_squares, weighted_least_squares}`: rank-revealing least squares by
  pivoted QR, with R's `lm.fit` / `lm.wfit` conventions for aliased columns and zero weights;
- `linalg::solve::sym_inverse` and `linalg::matrix_ops::safe_inverse`: symmetric inverses that
  report their numerical rank and conditioning instead of silently returning a pseudoinverse;
- `econometrics::glm::{fit_glm, fit_propensity, fit_logistic_fluctuation}`: generalized linear
  models by IRLS, the GLM every Tier 3 estimator shares.

All three are checked against R (4.3.3, base R only) on the case `linalg_qr_glm`, at the solution.
They are not ports for exactness: R is the test oracle and the default convention.

## Least squares (dqrdc2, lm.fit, lm.wfit)

- **Pivoting rule.** `Qr::decompose(x, tol)` is a Householder QR with LINPACK `dqrdc2`'s limited
  column pivoting: before step `l`, a column whose remaining norm is below `tol` times its original
  norm is moved to the end, and the rank is the number of columns that were never moved (capped at
  `n`). So the *later* of two collinear columns is the aliased one, as in R; a standard
  rank-revealing QR (LAPACK `dgeqp3`) picks the largest column first and would alias a different
  column. The cost is that of an unpivoted Householder QR.
- **Numerics.** The overflow-safe `dnrm2`, the column-norm downdate with Bates' 1999 recomputation
  when more than 1 - 1e-6 of a norm has been removed, and the stable Householder sign are kept.
  The factor is stored column by column (the working copy is `X'`), so every norm, dot product and
  update runs on contiguous memory. LINPACK's exact operation order is not required: there is no
  bit-level agreement with `base::qr`, and none is tested.
- **`lm.fit` conventions.** `least_squares(x, y)` uses `tol = 1e-7` (`LM_QR_TOL`); coefficients of
  aliased columns are `NaN` (R's `NA`), and so are their rows and columns of `unscaled_cov`
  (`chol2inv(R11)` unpivoted, R's `vcov(lm)` before scaling); residuals are `Q [0; (Q'y)[rank..]]`,
  `fitted = y - residuals`, `df_residual = n - rank`. With `p = 0` the fit has rank 0 and the
  residuals are `y`.
- **`lm.wfit` conventions.** `weighted_least_squares(x, y, w)` fits `sqrt(w) x` on `sqrt(w) y` over
  the rows with `w > 0`; residuals are returned on the original scale; rows with weight 0 get
  `fitted = x0 b` (aliased coefficients count as 0) and `residual = y0 - x0 b`; `df_residual` counts
  the positive weights. `unscaled_cov` is `(X'WX)^{-1}`.
- **Warning text.** `aliasing_message(aliased, names)` gives R's "Coefficients not defined because of
  singularities: x2, x5" (default names `x{j+1}`).

## Symmetric inverses (sym_inverse, safe_inverse)

- **Symmetry rule.** A matrix is symmetric when `max |m_ij - m_ji| <= 1e-10 * max |m_ij|`
  (`SYMMETRY_TOL`; scale-free, so a matrix with tiny entries cannot be 1 % asymmetric and still
  count as symmetric). `sym_inverse` refuses anything else, and a non-finite or non-square matrix.
  The lower triangle defines the matrix.
- **Equilibration and rank.** `E = S m S` with `S = diag(1 / sqrt|m_ii|)`, and scale 1 for a zero
  diagonal entry (a positive semi-definite matrix with a zero diagonal has a zero row anyway, and a
  symmetric indefinite KKT system `[[H, A'], [A, 0]]` stays invertible). Eigenvalues of `E` with
  `|lambda| <= SYM_RANK_TOL * dim * lambda_max` (`SYM_RANK_TOL = 1e-14`) count as zero.
- **Full rank.** Positive-definite matrices are inverted by Cholesky of `m` (the code `safe_inverse`
  always ran, so every full-rank SPD caller gets bit-identical output); negative-definite ones by
  the Cholesky factor of `-m` (a negated Hessian passed un-negated); indefinite ones by the verified
  partial-pivot LU of `matrix_ops` (RF-31), with the exact 2-norm condition of `m` from its
  eigenvalues as the residual allowance, and the eigen route if that fails.
- **Certificate.** When Cholesky succeeds, `kappa_1(E) = ||E||_1 ||E^-1||_1` is computed in
  O(dim^2) from the inverse. For a symmetric matrix `kappa_2 <= kappa_1`, so
  `kappa_1 <= min(1e10, 0.1 / (SYM_RANK_TOL * dim))` proves full rank and "no warning" without any
  eigenvalue. `reciprocal_condition` is then `1 / kappa_1(E)`, a lower bound within a factor `dim`
  of the exact value and always above 1e-10. Otherwise it is the exact `min|lambda| / max|lambda|`
  of `E` (0 for the zero matrix), so thresholds at or below 1e-10 decide exactly.
- **Rank-deficient.** `SingularPolicy::Error` gives `SingularMatrix { context: "<what>: rank r of
  d" }`; `SingularPolicy::Generalized` gives `S E_r^+ S`, the Moore-Penrose inverse of `E` on its
  identified eigenvalues mapped back. It is symmetric and reflexive (`m G m = m`, `G m G = G`),
  equals `MASS::ginv(m)` when `diag(m)` is constant, reproduces every estimable quantity, and is
  robust to badly scaled columns. `SymInverse::warning` words it: "X'X is singular (rank 3 of 4); a
  generalized inverse was used".
- **`safe_inverse`** (signature unchanged) returns, by input class:

  | Input | Result |
  |---|---|
  | non-square / non-finite | `LinalgError::NotSquare` / `LinalgError::NonFinite` |
  | symmetric, full rank, equilibrated condition <= 1e10 | the inverse (SPD: Cholesky, bit for bit as before), `None` |
  | symmetric, full rank, equilibrated condition > 1e10 | the inverse, `Some(1 / reciprocal_condition)` |
  | symmetric, rank-deficient | `S E_r^+ S`, `Some(1 / reciprocal_condition)` (possibly infinite) |
  | not symmetric | verified partial-pivot LU, `None`; SVD condition > 1e10: the pseudoinverse, `Some(condition)` |

  Non-symmetric input never reaches Cholesky (REG-59: faer's `llt` reads only the lower triangle, so
  `[[2, 1], [0, 3]]` was inverted as `diag(2, 3)`).
- **Worked numbers** (equilibrated condition): Longley's X'X 1.87e9 (no flag); a raw quadratic trend
  in calendar years 1960-2020, 3.67e9 (no flag); the raw cubic 2.71e14, flagged and classified as
  rank 3 of 4 (equilibrated `rcond` 3.68e-15 < `SYM_RANK_TOL * 4`), while `lm` keeps rank 4. The
  cross-product test is stricter than QR on X by up to a factor of `dim`, which is why linear-model
  coefficients use `least_squares` (R9).

## Case linalg_qr_glm

`validation/r/cases/linalg_glm.R`, dataset `validation/datasets/generated/linalg_glm.csv`
(`set.seed(20261210)`, n = 120, 12 columns): `x1`, `x2`, `x4` standard normal, `x3 = x1 - 2 x2`
(exactly collinear), `off = log(runif(0.5, 2))`, a binary `yb`, a Poisson count `yp` with offset
`off`, a fractional `yq` with weights `wq`, `ysep = 1{x4 > 0.1}` (separated by `x4`), a continuous
`yc` and weights `w0` whose first five entries are 0. Decisive values:

```
qr_rank = 4    qr_pivot = 1, 2, 3, 5, 4            (cbind(1, x1, x3, x2, x4): x2 aliased)
qr_r_diag_rank = -10.954451150103322, -10.433367291245888, -21.877583025446434, -10.364941823858397
qr2_rank = 4   qr2_pivot = 1, 2, 4, 6, 3, 5         (cbind(1, x1, 2 x1, x2, 0, x4))
qr3_rank = 3   qr3_pivot = 1, 2, 3, 4, 5            (first three rows)
lmfit_coef = 1.031666376059943, 1.602838770035396, 0.4920805087863006, NA, 0.04729853413856718
lmwfit_coef = 1.0354818647115143, 2.078732004897483, -1.0228147680226813   lmwfit_df = 112
logit_coef = -0.1283444910844334, 0.8579385967497148, -0.25180889064483325
probit_coef = -0.0722049136445435, 0.5118247636828549, -0.1551746951197294
poisson_deviance = 143.81961600971843   null 154.50134008807606 (glm() refit: offset + intercept)
quasibin_dispersion = 0.04133511582570598   gaussian_aic = 313.5857933039046
aliased_coef = -0.1283444910844299, 0.7320341514272675, 0.12590444532240996, NA
fluct_coef = -0.22961934884322305   fluct_int_coef = -0.22403035045381348
trap_rank = 3   trap_x1_coef = 2.065140641939357   trap_rss = 204.11400660276115
```

The GLM fits use `glm.control(epsilon = 1e-14, maxit = 100)` in R and `epsilon = 1e-14` in p2a:
both sides are converged, so the solutions are compared at the README's tolerances. The propensity
and fluctuation fits run at p2a's default `epsilon` (1e-10) against R at 1e-14:
`Tol::iterative(1e-8)` (the fluctuation fit at 1e-10 differs from the 1e-14 fit by 5.8e-12
relative).

| Quantity | Tolerance (n = 120) |
|---|---|
| coefficients, `|diag(R)|`, fitted values, residuals | `Tol::coef`: 1e-8 relative |
| SEs, deviances, dispersion, AIC, logLik, RSS | `Tol::se` / `Tol::stat`: 1e-6 relative |
| rank, pivot, df | exact |
| default-epsilon fits against R at 1e-14 | `Tol::iterative(1e-8)` |

**Emitted, not pinned.** The case also emits LINPACK's compact storage (`qr_qraux`,
`qr_qraux_rank`, `qr3_qraux`) and the signed `qr_r_diag` (only `|diag(R)|` is pinned: its sign is a
convention), the IRLS iteration counts at R's default control (`*_iter_default`,
`logit_start0_iter`, `logit_x1e6_iter`, `logit_x1em6_iter`, `gauss_ye6_iter`, `gauss_yem6_iter`)
and the separated fit's `sep_fitted_extreme`, `sep_converged_r` and `sep_iter`. They describe R's
path, not the solution; p2a tests convergence, separation and scale equivariance as properties.

## Tests

Registered in `validation/methods/linalg_glm.json` (they load `linalg_qr_glm`):

- `linalg_qr_least_squares`: `linalg::solve::tests::test_validate_qr_dqrdc2_aliasing`,
  `linalg::solve::tests::test_validate_qr_edge_shapes`,
  `linalg::solve::tests::test_validate_lm_wfit_zero_weights`,
  `linalg::solve::tests::test_qr_factorization_properties` (QR identities: `Q [R_j; 0]` reproduces
  each identified column, aliased columns lie in the span of the identified ones to `tol`, `Q` is
  orthogonal; properties, not R-pinned).

Other tests (quoted R values or properties; not R-pinned):

- `linalg::solve::tests::test_least_squares_rejects_invalid_input`: length, finiteness, tolerance and
  weight checks.

## How to regenerate

```bash
Rscript validation/r/run.R --write linalg_qr_glm
Rscript validation/r/run.R --check linalg_qr_glm
```

Never edit `validation/expected/linalg_qr_glm.json` by hand.
