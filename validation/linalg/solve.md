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
  the positive weights. `unscaled_cov` is `(X'WX)^{-1}`. A non-finite `x` is refused in every row,
  zero-weight rows included, as `lm.wfit` does.
- **Errors, not panics.** `Qr::qty` and `Qr::coefficients` refuse a `y` of the wrong length with
  `InvalidSpecification`; `Qr::unscaled_covariance` allocates its `rank^2` and `p^2` buffers
  through the allocation ceiling (`p` can exceed the `n x p` workspace `decompose` checked).
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
  partial-pivot LU of `matrix_ops` (RF-31), accepted when `max |m X - I|` is within
  `100 * dim * eps * min(kappa_1, cap)`: `kappa_1 = ||m||_1 ||X||_1` from the candidate (within a
  factor `dim` of `kappa_2(m)` for a correct one), capped by `||m||_1 max s_i^2 / min |lambda(E)|`,
  an upper bound on `kappa_2(m)` that a wrong candidate cannot raise; the eigen route if that
  fails. On badly scaled indefinite input the cap can exceed `kappa_2(m)` by up to
  `max s_i^2 / min s_i^2`, which widens the allowance for a candidate by the same factor. (No second eigendecomposition for the exact condition of `m`.)
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
  `[[2, 1], [0, 3]]` was inverted as `diag(2, 3)`). Only a failed eigendecomposition sends
  symmetric input on to the general route.
- **Budgets.** `sym_inverse` calls `resource::checkpoint()` and checks the `dim^2` allocation
  against the ceiling. `safe_inverse` does neither, as before: it runs `sym_inverse`'s unguarded
  variant, so its callers (which already hold the matrix and checkpoint their own loops) see a
  cancellation as `Cancelled` at their next checkpoint, not as a failed inverse, and no refusal can
  reroute symmetric input to the SVD/LU route.
- **Worked numbers** (equilibrated condition): Longley's X'X 1.87e9 (no flag); a raw quadratic trend
  in calendar years 1960-2020, 3.67e9 (no flag); the raw cubic 2.71e14, flagged and classified as
  rank 3 of 4 (equilibrated `rcond` 3.68e-15 < `SYM_RANK_TOL * 4`), while `lm` keeps rank 4. The
  cross-product test is stricter than QR on X by up to a factor of `dim`, which is why linear-model
  coefficients use `least_squares` (R9).

## Generalized linear models (glm.fit)

`fit_glm(x, y, &GlmSpec)` fits by Fisher scoring (IRLS) as `stats::glm.fit`; `x` carries its own
intercept column.

| Family | Links | Variance | Dispersion |
|---|---|---|---|
| `Gaussian` | `Identity`, `Log` | 1 | estimated |
| `Binomial` | `Logit`, `Probit`, `Cloglog` | `mu (1 - mu)` | 1 |
| `QuasiBinomial` | `Logit`, `Probit`, `Cloglog` | `mu (1 - mu)` | estimated |
| `Poisson` | `Log`, `Identity` | `mu` | 1 |
| `QuasiPoisson` | `Log`, `Identity` | `mu` | estimated |

- **Family and link functions** are R's: the logit inverse link clamps `|eta| > 30`, the probit
  clamps `eta` to `-qnorm(.Machine$double.eps) = 8.125890664701906`, the cloglog and log inverse
  links keep `mu` within `[eps, 1 - eps]` and `>= eps`, and every `mu.eta` is at least `eps`. Binomial
  deviance residuals use `y log(y / mu)` with `0 log 0 = 0`.
- **Initialisation** (`family$initialize`): binomial `y` must be in [0, 1] (rows with weight 0 get
  `y = 0`), `mustart = (w y + 0.5) / (w + 1)`, and "non-integer #successes in a binomial glm!" when
  some `w y` is not an integer; Poisson `y >= 0`, `mustart = y + 0.1`; Gaussian `mustart = y`, and
  the log link without `start` needs `y > 0` ("cannot find valid starting values: please specify
  some").
- **IRLS.** Each step solves the weighted working regression on the rows with `w > 0` and
  `mu.eta != 0` by `Qr` with tolerance 1e-11 (R's `min(1e-7, epsilon / 1000)` at the default
  `glm.control`); aliased columns get coefficient 0 in the iteration and `NaN` in the result. A
  non-finite deviance or invalid fitted values halve the step towards the previous coefficients
  ("step size truncated due to divergence" / "step size truncated: out of bounds", `boundary`).
  After the loop: "glm.fit: algorithm did not converge", "glm.fit: algorithm stopped at boundary
  value", "glm.fit: fitted probabilities numerically 0 or 1 occurred" (sets `separation`), "glm.fit:
  fitted rates numerically 0 occurred".
- **Convergence.** `|dev - devold| / (|dev| + 0.1 * D0 / n_ok) < epsilon`; see "Deviations from R".
- **Results.** SEs and dispersion as `summary.glm` (Pearson dispersion over the rows with positive
  working weight; "observations with zero weight not used for calculating dispersion"), `vcov` from
  the QR of the last step; the null deviance as `glm()`: `linkinv(offset)` without an intercept, the
  weighted mean with one, and a refit of the intercept-only model when there is an offset too
  (skipped when `x` is the single column of ones: the null model is then the fitted model). The
  model has an intercept when some column of `x` is identically 1. AIC is `family$aic + 2 rank`
  (binomial: `round(w y)` successes of `round(w)` trials, R's round-half-even; Poisson: `+Inf` and
  R's "non-integer x" warning for a non-integer count), and `logLik = p - AIC / 2` with
  `p = rank + 1` for the Gaussian family and `rank` otherwise (`logLik.glm`). Quasi families have
  `NaN` AIC and log-likelihood.
- **Defaults** differ from `glm.control`: `epsilon` 1e-10 (R 1e-8) and `max_iter` 100 (R 25), about
  one more IRLS iteration and closer to the MLE.
- **Helpers.** `fit_propensity(x, treat, link)` is the binomial MLE of a 0/1 treatment with the
  logit, probit or cloglog link; `fit_logistic_fluctuation(y, offset, covariates, weights)` is TMLE's
  quasi-binomial logit fluctuation (`glm(y ~ -1 + H, offset)` or `glm(y ~ 1, offset, weights)`);
  `qlogis`, `plogis`, `bound`.

### Deviations from R

- **Convergence criterion.** `glm.fit` stops when `|dev - devold| / (|dev| + 0.1) < epsilon`; the
  absolute 0.1 makes the rule depend on the scale of `y` (Gaussian, quasi-Poisson) and of the prior
  weights. For the Gaussian family `mustart = y` gives `devold = 0`, so a small-scale response
  "converges" after the first step; with the log link that step is not the MLE. Checked in R 4.3.3:
  `glm(I(1e-6 * (yp + 0.5)) ~ x1, family = gaussian(link = "log"))` on this case's data stops after 1
  iteration with slope 0.15911284076242163, against 0.2132268406899209 for the unscaled response (5
  iterations). p2a uses the floor `0.1 * D0 / n_ok`, where `D0` is the deviance of the weighted mean
  of `y` over the `n_ok` rows with positive weight (R's 0.1 when that is 0 or not finite); it is
  invariant to rescaling `y` or `w`, and about 0.14 for unweighted 0/1 data, so no result at the
  solution changes.
- **Gaussian AIC with zero weights** is computed over the rows with positive weight,
  `n_ok (log(2 pi dev / n_ok) + 1) + 2 - sum log w + 2 rank` (`logLik.lm`'s convention); `glm` gives
  `+Inf`.
- **Aliasing tolerance** fixed at 1e-11 (R's value at the default `glm.control`) instead of
  `min(1e-7, epsilon / 1000)`: R's coupling gives 1e-17 at `epsilon = 1e-14`, which detects no
  aliasing at all.
- **Errors instead of undefined fits:** non-finite IRLS coefficients are `DivergenceError` (R warns
  "non-finite coefficients at iteration k" and returns an undefined `coef`); no positive weight is
  `InsufficientData`.
- **Flags:** `separation` and its warning also cover `QuasiBinomial`, and "fitted rates numerically
  0" also `QuasiPoisson` (R warns for `binomial` and `poisson` only).
- **`weighted_least_squares`** refuses an all-zero weight vector (`InsufficientData`); `lm.wfit`
  returns an all-`NA` fit.

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
and fluctuation fits run at p2a's default `epsilon` (1e-10) against R at 1e-14. Where the
deviance-change rule converges quadratically (logit coefficients and fitted values, fluctuation
coefficients and deviances: the fluctuation coefficient at 1e-10 differs from the 1e-14 fit by
5.8e-12 relative) they are compared at `Tol::iterative(1e-8)`. Where it converges only linearly
(the probit, a non-canonical link; the fluctuation SE and dispersion, which `summary.glm` takes from
the working weights of the last IRLS step) the gap at 1e-10 is up to 2.1e-6 relative, and R's own
`glm` at `epsilon = 1e-10` shows the same gap (probit intercept -0.0722050664731, fluctuation SE
0.039090634610887): these are compared at `Tol::new(1e-5, 1e-12)` (`DEFAULT_EPS_TOL`).

| Quantity | Tolerance (n = 120) |
|---|---|
| coefficients, `|diag(R)|`, fitted values, residuals | `Tol::coef`: 1e-8 relative |
| SEs, deviances, dispersion, AIC, logLik, RSS | `Tol::se` / `Tol::stat`: 1e-6 relative |
| rank, pivot, df | exact |
| default-epsilon fits against R at 1e-14 | `Tol::iterative(1e-8)`; linearly converging quantities `Tol::new(1e-5, 1e-12)` |

**Emitted, not pinned.** The case also emits LINPACK's compact storage (`qr_qraux`,
`qr_qraux_rank`, `qr3_qraux`) and the signed `qr_r_diag` (only `|diag(R)|` is pinned: its sign is a
convention), the IRLS iteration counts at R's default control (`*_iter_default`,
`logit_start0_iter`, `logit_x1e6_iter`, `logit_x1em6_iter`, `gauss_ye6_iter`, `gauss_yem6_iter`)
and the separated fit's `sep_fitted_extreme`, `sep_converged_r` and `sep_iter`. They describe R's
path, not the solution; p2a tests convergence, separation and scale equivariance as properties.

## Performance

Rule-5 timing check of T3-WP01 (2026-11-03): release builds, `--no-default-features --features ci`,
`RAYON_NUM_THREADS` fixed, head and base runs interleaved on a shared 12-core machine that other
builds were loading. "Base" is the head tree with `safe_inverse`'s body replaced by the one at
5da3cbe9 (the only hot-path change; `least_squares` and `fit_glm` are new and have no base).
Each figure is the lowest over the interleaved runs; times in microseconds, median / p75.

End to end (`unified_benchmarks --only OLS,OLS_Bootstrap,Logit,Probit` on the paper's CSVs,
two runs each, cores 10-11, 2 threads; `forecasting_benchmarks KalmanSmoother`, Criterion, two
runs each):

| Benchmark | n | base | head | head / base |
|---|---|---|---|---|
| OLS standard | 100 / 1000 / 10000 | 25.4 / 145.5 / 1310 | 25.6 / 132.2 / 962 | 1.01 / 0.91 / 0.73 |
| OLS HC1 | 100 / 1000 / 10000 | 57.7 / 311 / 2757 | 52.5 / 249 / 2146 | 0.91 / 0.80 / 0.78 |
| OLS_Bootstrap pairs | 100 / 1000 | 1570 / 10425 | 1821 / 10973 | 1.16 / 1.05 |
| Logit MLE | 100 / 1000 / 10000 | 114 / 1143 / 10763 | 86 / 1003 / 8470 | 0.75 / 0.88 / 0.79 |
| Probit MLE | 100 / 1000 / 10000 | 166 / 1450 / 14597 | 125 / 1228 / 14816 | 0.75 / 0.85 / 1.02 |
| KalmanSmoother | 100 / 500 / 1000 / 5000 | 126 / 587 / 1189 / 6631 | 133 / 704 / 1388 / 6475 | 1.06 / 1.20 / 1.17 / 0.98 |

p75 is within 5 % of the median in every row except base OLS HC1 at n = 10000 (3995) and the
Kalman rows (head p75 143 / 724 / 1425 / 6864). The OLS ratios below 1 are load noise (the run
spread is as large); Logit and Probit invert a negative-definite Hessian, which now takes the
negated Cholesky instead of the SVD condition and LU, so part of their gain is real. Outputs agree to 2.3e-16
relative (Logit and Probit at n = 100 differ in the last bit, from the negated-Cholesky route).
Nothing end to end exceeds 1.5x. (Measured before the review fixes below, which only remove
work from `safe_inverse`.)

`safe_inverse` alone (scratch micro-benchmark, not committed: the head function against the
5da3cbe9 body copied into an example, release, one core, `RAYON_NUM_THREADS=1`, six rounds of
five interleaved batches; ns, median / p75; at d = 1 the "indefinite" input is a negative
scalar). First measurement of the package, then after the review fixes (one pass for
finiteness and asymmetry, no repeated validation on the `safe_inverse` path, the certificate on
the column-major faer copy in four-lane sums, and no second eigendecomposition for indefinite
input):

| Input | d = 1 | d = 3 | d = 10 | d = 50 |
|---|---|---|---|---|
| SPD, first head / base | 0.95 | 1.20 | **1.73** | **1.69** |
| SPD, base / head | 251 / 338 (1.35) | 374 / 495 (1.32) | 1699 / 2035 (1.20) | 36749 / 42060 (1.14) |
| negative definite, base / head | 985 / 514 (0.52) | 3265 / 638 (0.20) | 19562 / 2537 (0.13) | 336080 / 45753 (0.14) |
| indefinite, first head / base | 0.43 | 1.11 | **1.68** | **1.73** |
| indefinite, base / head | 1039 / 546 (0.53) | 3106 / 2696 (0.87) | 21045 / 16549 (0.79) | 378750 / 321503 (0.85) |

(p75 within 5 % of the median except base indefinite at d = 1 (1152) and head (607); the inverses agree with base to 1e-15 relative, SPD
bit for bit.) No row exceeds 1.5x. What remains on SPD input is O(d^2) work base did not do, at
a small constant: the finiteness and symmetry scan (about 2 us at d = 50, two contiguous-friendly
passes), the scale vector, and the `kappa_1` certificate that proves full rank without
eigenvalues (about 2.4 us at d = 50); the faer copy and the copy back are faster than base's.
At d = 1 and 3 the difference is about 100 ns of fixed cost (one scratch allocation and the
certificate). Indefinite input now takes one symmetric eigendecomposition (rank and conditioning
of the equilibrated matrix) and the LU, whose residual allowance uses `kappa_1(m)` from the LU
candidate, capped by `||m||_1 max s_i^2 / min |lambda(E)|`, an upper bound on `kappa_2(m)` that
does not depend on the candidate; for a correct candidate the allowance is at most `dim` times
the one the exact 2-norm condition gave.

Informational, for the D1 gate of WP09 (same machine, one core, six rounds of five batches; us,
median / p75). `least_squares` (pivoted QR; it also returns residuals, fitted values, the
unscaled covariance and the rank) against `xtx` + `xty` + a Cholesky solve, on a design with an
intercept and uniform noise columns:

| k | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|
| 6: head / xtx + Cholesky | 12.4/12.9 vs 3.0/3.2 (4.1) | 126/127 vs 28.2/28.5 (4.5) | 1719/1748 vs 309/318 (5.6) |
| 10 | 26.2/26.4 vs 7.6/7.8 (3.4) | 252/254 vs 63.5/64.9 (4.0) | 3365/3756 vs 808/869 (4.2) |
| 50 | 300/320 vs 77.6/84.1 (3.9) | 2616/2707 vs 535/554 (4.9) | 48901/54754 vs 11347/14424 (4.3) |

`fit_glm(logit)` on a matrix against `run_logit` on a dataset (k = 3: intercept and two
covariates, simulated logit outcome): n = 100: 47.1/53.5 vs 72.0/77.4 (0.65); n = 1,000:
428/453 vs 617/699 (0.69); n = 10,000: 5198/7216 vs 7689/9947 (0.68).

## Tests

Registered in `validation/methods/linalg_glm.json` (they load `linalg_qr_glm`):

- `linalg_qr_least_squares`: `linalg::solve::tests::test_validate_qr_dqrdc2_aliasing`,
  `linalg::solve::tests::test_validate_qr_edge_shapes`,
  `linalg::solve::tests::test_validate_lm_wfit_zero_weights`,
  `linalg::solve::tests::test_validate_qr_factorization_properties` (QR identities: `Q [R_j; 0]` reproduces
  each identified column, aliased columns lie in the span of the identified ones to `tol`, `Q` is
  orthogonal; properties, not R-pinned).
- `linalg_sym_inverse`: `linalg::solve::tests::test_validate_sym_inverse_reports_rank` (the dummy trap
  `[1, x1, ysep, 1 - ysep]`: rank 3 of 4, the identified slope and the RSS through the generalized
  inverse, the generalized-inverse identities, `SingularPolicy::Error`, and `safe_inverse`'s flag).
- `econ_glm`: `econometrics::glm::tests::test_validate_glm_families` (nine family/link fits, the
  aliased logit, `fit_propensity`; plus, as properties, the score equations at the solution,
  convergence at the default spec and `start = 0`),
  `econometrics::glm::tests::test_validate_glm_fluctuation`,
  `econometrics::glm::tests::test_validate_glm_binomial_noninteger`,
  `econometrics::glm::tests::test_validate_glm_separation_flag` (property: flag, warning, deviance below
  `2 log 2`, every observation classified correctly),
  `econometrics::glm::tests::test_validate_glm_scale_equivariance` (property),
  `econometrics::glm::tests::test_validate_glm_zero_weights_match_subset` (property),
  `econometrics::glm::tests::test_validate_glm_fit_serde_round_trip` (JSON round trip with `NaN` as `null`).

Other tests (quoted R values or properties; not R-pinned):

- `linalg::solve::tests::test_least_squares_rejects_invalid_input`: length, finiteness, tolerance and
  weight checks; a non-finite `x` in a zero-weight row; `Qr::qty` and `Qr::coefficients` on a short
  `y`; the `p x p` covariance of a 2 x 64 `x` under a 4 KiB allocation ceiling.
- `linalg::solve::tests::test_sym_inverse_rejects_nonsymmetric_and_nonfinite`: asymmetry, non-finite
  and non-square input, rounding-level asymmetry accepted, the zero and empty matrices.
- `linalg::solve::tests::test_sym_inverse_negative_definite_and_indefinite`:
  `solve(matrix(c(-4, 1, 1, -3), 2))` and its equilibrated reciprocal condition 0.5519815245204085;
  the KKT matrix `[[2, 0, 1], [0, 2, 1], [1, 1, 0]]` (rank 3, reciprocal condition
  0.38196601125010504).
- `linalg::matrix_ops::tests::test_safe_inverse_rejects_lower_triangle_trick`:
  `solve(matrix(c(2, 0, 1, 3), 2))` (REG-59).
- `linalg::matrix_ops::tests::test_safe_inverse_warns_on_near_singular_pd`: `[[1, a], [a, 1]]` with
  `a = 1 - 2^-40`, equilibrated condition `2^41 - 1` (REG-4).
- `linalg::matrix_ops::tests::test_safe_inverse_unchanged_on_well_conditioned_spd`: the Cholesky
  inverse bit for bit at condition 2.1e6, and `solve()` of the KKT matrix.
- `linalg::matrix_ops::tests::test_safe_inverse_symmetric_ignores_budget_and_ceiling`: SPD,
  indefinite and rank-deficient 20 x 20 input gives the same result under a cancelled budget and a
  1 KiB allocation ceiling; `sym_inverse` itself returns `Cancelled` and `ResourceLimit`.
- `linalg::matrix_ops::tests::test_safe_inverse_ill_conditioned_symmetric_flags_condition` and
  `linalg::matrix_ops::tests::test_safe_inverse_ill_conditioned_nonsymmetric_keeps_pseudoinverse`.
- T1's RF-31 tests of `matrix_ops.rs` (`test_safe_inverse_lu_fallback_negative_diagonal`,
  `test_safe_inverse_lu_fallback_tridiagonal`, `test_safe_inverse_lu_fallback_sparse_sweep`,
  `test_safe_inverse_spd_keeps_cholesky_path`, `test_linalg_rejects_non_finite`,
  `test_verified_inverse_rejects_bad_candidates`).
- `econometrics::glm::tests::test_glm_rejects_invalid_input` and
  `econometrics::glm::tests::test_logistic_helpers` (`qlogis(0.25)`, `plogis(0.3)`,
  `-qnorm(.Machine$double.eps)`).

## How to regenerate

```bash
Rscript validation/r/run.R --write linalg_qr_glm
Rscript validation/r/run.R --check linalg_qr_glm
```

Never edit `validation/expected/linalg_qr_glm.json` by hand.
